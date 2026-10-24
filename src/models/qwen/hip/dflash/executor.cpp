#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/hip/dev_timing.hpp"
#include "src/models/qwen/hip/dev_sync.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace gufo::hip {
namespace {

constexpr std::array<std::uint8_t, 8> kDFlashGpuPersistentMagic = {
    'G', 'D', 'F', 'G', 'P', 'U', '0', '1'};
constexpr std::uint32_t kDFlashGpuPersistentVersion = 2;
constexpr std::size_t kDFlashGpuPersistentHeaderBytes = 64;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("DFlash GPU persistent header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
[[nodiscard]] T GetLittleEndian(std::span<const std::uint8_t> source,
                                std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("DFlash GPU persistent header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

[[nodiscard]] std::size_t CheckedPersistentAdd(std::size_t left,
                                               std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

/// Widest proposal block; BF16 context injection also uses sixteen rows.
constexpr std::size_t kDFlashMaxSharedBatch = 8;

void NormalizeAndRotateQK(float* query, float* key, const float* query_weight,
                          const float* key_weight,
                          const core::ModelConfig& config,
                          std::uint32_t query_heads, std::size_t rows,
                          std::uint32_t position, hipStream_t stream) {
  // The target's fused kernel has the same reduction and RoPE arithmetic.
  // DFlash2 keeps its own ring cache, so only the in-place Q/K outputs are
  // used.
  if (config.head_dim <= 256) {
    LaunchBatchedFusedQKNormRoPEKvWrite(
        query, key, key, query_weight, key_weight, query, key, nullptr, nullptr,
        nullptr, nullptr, 0, position, rows, 0, query_heads,
        config.num_key_value_heads, config.head_dim, config.rotary_dim,
        config.rope_theta, 1e-6F, stream);
    return;
  }
  if (query_heads != 0) {
    LaunchBatchedPerHeadRMSNorm(query, query_weight, query, rows, query_heads,
                                config.head_dim, 1e-6F, stream);
  }
  LaunchBatchedPerHeadRMSNorm(key, key_weight, key, rows,
                              config.num_key_value_heads, config.head_dim,
                              1e-6F, stream);
  LaunchBatchedRoPE(query, key, rows, query_heads, config.num_key_value_heads,
                    config.head_dim, config.rotary_dim, position,
                    config.rope_theta, stream);
}

void TraceTensor(const DFlashTrace& trace, std::string_view name,
                 const float* device, std::size_t count, hipStream_t stream) {
  if (!trace)
    return;
  std::vector<float> host(count);
  HIP_CHECK(hipMemcpyAsync(host.data(), device, count * sizeof(float),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  trace(name, host);
}

}  // namespace

QwenDFlashGpuSnapshot::~QwenDFlashGpuSnapshot() {
  if (d_k_ != nullptr) {
    (void)hipFree(d_k_);
  }
  if (d_v_ != nullptr) {
    (void)hipFree(d_v_);
  }
}

std::size_t QwenDFlashGpuSnapshot::PersistentPayloadBytes() const {
  return CheckedPersistentAdd(kDFlashGpuPersistentHeaderBytes, payload_bytes_);
}

std::size_t QwenDFlashGpuSnapshot::SerializePersistent(
    std::span<std::uint8_t> destination) const {
  const std::size_t expected_bytes = PersistentPayloadBytes();
  if (destination.size() != expected_bytes) {
    throw std::invalid_argument(
        "DFlash GPU persistent destination size is invalid");
  }
  if (history_capacity_ == 0 || history_capacity_ > max_context_ ||
      valid_context_ > max_context_ ||
      (kv_width_ != 0 &&
       valid_context_ > std::numeric_limits<std::size_t>::max() / kv_width_) ||
      elements_per_layer_ != static_cast<std::size_t>(
                                 std::min(valid_context_, history_capacity_)) *
                                 kv_width_ ||
      (elements_per_layer_ != 0 &&
       num_layers_ >
           std::numeric_limits<std::size_t>::max() / elements_per_layer_)) {
    throw std::invalid_argument("DFlash GPU snapshot metadata is malformed");
  }
  const std::size_t total_elements =
      static_cast<std::size_t>(num_layers_) * elements_per_layer_;
  if (total_elements >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  const std::size_t bytes_per_kind = total_elements * sizeof(float);
  if (bytes_per_kind >
          std::numeric_limits<std::size_t>::max() - bytes_per_kind ||
      payload_bytes_ != bytes_per_kind + bytes_per_kind ||
      (bytes_per_kind != 0 && (d_k_ == nullptr || d_v_ == nullptr))) {
    throw std::invalid_argument("DFlash GPU snapshot payload is malformed");
  }

  // K/V copies overwrite the payload; do not clear those pages on the CPU.
  std::fill_n(destination.begin(), kDFlashGpuPersistentHeaderBytes,
              std::uint8_t{0});
  std::copy(kDFlashGpuPersistentMagic.begin(), kDFlashGpuPersistentMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8, kDFlashGpuPersistentVersion);
  PutLittleEndian<std::uint32_t>(
      destination, 12,
      static_cast<std::uint32_t>(kDFlashGpuPersistentHeaderBytes));
  PutLittleEndian<std::uint32_t>(destination, 16, num_layers_);
  PutLittleEndian<std::uint32_t>(destination, 20, kv_width_);
  PutLittleEndian<std::uint32_t>(destination, 24, max_context_);
  PutLittleEndian<std::uint32_t>(destination, 28, valid_context_);
  PutLittleEndian<std::uint64_t>(
      destination, 32, static_cast<std::uint64_t>(elements_per_layer_));
  PutLittleEndian<std::uint64_t>(destination, 40,
                                 static_cast<std::uint64_t>(payload_bytes_));
  PutLittleEndian<std::uint64_t>(destination, 48,
                                 static_cast<std::uint64_t>(expected_bytes));

  PutLittleEndian<std::uint64_t>(destination, 56, history_capacity_);

  if (bytes_per_kind != 0) {
    SnapshotTransfer transfer;
    transfer.Copy(destination.data() + kDFlashGpuPersistentHeaderBytes, d_k_,
                  bytes_per_kind);
    transfer.Copy(
        destination.data() + kDFlashGpuPersistentHeaderBytes + bytes_per_kind,
        d_v_, bytes_per_kind);
  }
  return destination.size();
}

QwenDFlashGpuExecutor::QwenDFlashGpuExecutor(
    std::shared_ptr<const QwenDFlashGpuModel> model, std::uint32_t max_context)
    : model_(std::move(model)), max_context_(max_context) {
  try {
    Allocate();
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenDFlashGpuExecutor::~QwenDFlashGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenDFlashGpuExecutor> QwenDFlashGpuExecutor::Create(
    std::shared_ptr<const QwenDFlashGpuModel> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GPU executor context is invalid";
    }
    return nullptr;
  }
  // The last gate before `new`. Allocate() runs from the constructor
  // (executor.cpp:202) and sizes its convolutions from conv_group_size, which a
  // DSpark artifact leaves at 0. A refusal placed after the `new` would already
  // have sized a drafter that cannot draft, so it goes here.
  {
    const std::string refusal = speculative::DFlashGpuExecutorSupportRefusal(
        model->GetWeights());
    if (!refusal.empty()) {
      if (error_msg != nullptr) {
        *error_msg = refusal;
      }
      return nullptr;
    }
  }
  try {
    return std::unique_ptr<QwenDFlashGpuExecutor>(
        new QwenDFlashGpuExecutor(std::move(model), max_context));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenDFlashGpuExecutor::Allocate() {
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t block_size = df_cfg.block_size;
  const std::size_t num_layers = df_cfg.num_layers;
  const std::size_t enc_in_dim = df_cfg.target_layer_ids.size() * hidden_size;
  const std::size_t dynamic_size = speculative::DFlashDynamicCoefficientCount(
      hidden_size, df_cfg.conv_kernel_size, df_cfg.conv_group_size);
  const bool markov_head = speculative::IsDSparkDraft(model_->GetWeights());

  const auto error = hipStreamCreate(&stream_);
  if (error != hipSuccess) {
    throw std::runtime_error("DFlash hipStreamCreate failed");
  }
  if (block_size > kDFlashMaxSharedBatch) {
    HIPBLAS_CHECK(hipblasCreate(&hipblas_handle_));
    HIPBLAS_CHECK(hipblasSetStream(hipblas_handle_, stream_));
    hipblaslt_gemm_ = std::make_unique<HipblasLtGemm>();
  }

  // Attention only reads the sliding window. Preserve FP32 values and global
  // RoPE positions, but reuse physical slots as history advances. A layer the
  // sliding_window_pattern leaves unbounded needs the whole context, so the
  // capacity is the widest layer window (0 == unbounded == everything).
  history_capacity_ = std::min(max_context_, speculative::MaxSlidingWindow(df_cfg));
  injection_capacity_ = std::min(max_context_, std::uint32_t{256});
  // Allocate KV caches per layer
  d_injected_k_.resize(num_layers, nullptr);
  d_injected_v_.resize(num_layers, nullptr);
  for (std::size_t i = 0; i < num_layers; ++i) {
    AllocateBuffer(d_injected_k_[i], history_capacity_ * kv_dim);
    AllocateBuffer(d_injected_v_[i], history_capacity_ * kv_dim);
  }

  const auto allocate_scratch = [this](auto*& pointer, std::size_t elements) {
    // A DSpark draft sizes its selector scratch at zero. hipMalloc(0) is not a
    // portable success, so those rows simply do not exist.
    if (elements == 0) {
      pointer = nullptr;
      return;
    }
    AllocateBuffer(pointer, elements);
    scratch_bytes_ += elements * sizeof(*pointer);
  };
  // Scratch buffers for prompt injection & block drafting
  allocate_scratch(d_target_features_, injection_capacity_ * enc_in_dim);
  // Wide enough for every injection GEMM input (fc reads enc_in_dim
  // columns, attn_k/attn_v read hidden_size).
  if (GufoDevDFlashInjectWmma() != 0 && injection_capacity_ > 0) {
    inject_q8_bytes_ = QuantizedActivationBytes(
        injection_capacity_, std::max<std::size_t>(enc_in_dim, hidden_size));
    d_inject_q8_ = detail::AllocateDevice(inject_q8_bytes_);
    scratch_bytes_ += inject_q8_bytes_;
  }
  // GUFO_DEV_HOSTOPS_FEAT staging. One block of rows is the most a
  // single round can leave pending, and the event is created disabled-timing
  // because it only orders the two streams.
  if (GufoDevHostopsFeatures() && enc_in_dim != 0) {
    staging_capacity_ =
        static_cast<std::uint32_t>(std::min<std::size_t>(block_size, max_context_));
    allocate_scratch(d_staged_features_, staging_capacity_ * enc_in_dim);
    if (hipEventCreateWithFlags(&feature_ready_, hipEventDisableTiming) !=
        hipSuccess) {
      feature_ready_ = nullptr;
      staging_capacity_ = 0;
    }
  }
  allocate_scratch(
      d_fused_features_,
      std::max<std::size_t>(injection_capacity_, block_size) * hidden_size);
  allocate_scratch(
      d_block_normed_,
      std::max<std::size_t>(injection_capacity_, block_size) * hidden_size);
  allocate_scratch(
      d_k_block_,
      std::max<std::size_t>(injection_capacity_, block_size) * kv_dim);
  allocate_scratch(
      d_v_block_,
      std::max<std::size_t>(injection_capacity_, block_size) * kv_dim);

  // Scratch buffers for block drafting (size = block_size)
  allocate_scratch(d_block_hidden_, block_size * hidden_size);
  allocate_scratch(d_conv_hidden_, block_size * hidden_size);
  allocate_scratch(d_dynamic_coefficients_, block_size * dynamic_size);
  allocate_scratch(d_q_, block_size * q_dim);
  allocate_scratch(d_attn_out_, block_size * q_dim);
  const auto selector_scratch_elements =
      kernels::DFlashSelectorScratchElements(cfg.vocab_size);
  const auto ffn_scratch_width =
      std::max(intermediate_size, selector_scratch_elements);
  allocate_scratch(d_ffn_gate_, block_size * ffn_scratch_width);
  allocate_scratch(d_ffn_up_, block_size * ffn_scratch_width);
  allocate_scratch(d_ffn_down_, block_size * hidden_size);
  allocate_scratch(d_logits_, block_size * cfg.vocab_size);
  const auto& draft_vocab_cut = model_->GetDraftVocabCut();
  allocate_scratch(d_draft_vocab_logits_, block_size * draft_vocab_cut.size);
  allocate_scratch(d_selector_hidden_, block_size * df_cfg.selector_rank);
  allocate_scratch(d_selector_candidate_ids_,
                   block_size * df_cfg.selector_top_k);
  allocate_scratch(d_selector_candidate_probabilities_,
                   block_size * df_cfg.selector_top_k);
  allocate_scratch(d_selector_uniforms_, block_size);
  allocate_scratch(d_confidences_, block_size);
  allocate_scratch(d_out_token_, block_size);
  if (markov_head) {
    // Markov head scratch: one bias row, one markov_w1 row, one running
    // predecessor id, and the emitted token row. The emitted row cannot share
    // d_out_token_, because step zero reads the anchor out of that buffer.
    allocate_scratch(d_markov_bias_, cfg.vocab_size);
    allocate_scratch(d_markov_w1_prev_, df_cfg.markov_rank);
    allocate_scratch(d_markov_prev_, 1);
    allocate_scratch(d_draft_tokens_, block_size);
  }
  if (block_size > kDFlashMaxSharedBatch)
    allocate_scratch(d_bf16_input_, block_size * intermediate_size);

  // The estimate sizes a runner before this constructor runs, so an executor
  // that needs more than the estimate promised would fault on the first write
  // past a buffer instead of reporting.
  {
    const std::size_t estimated = speculative::DFlashDraftScratchElements(
        cfg, df_cfg, max_context_,
        markov_head ? 0U
                    : kernels::DFlashSelectorScratchElements(cfg.vocab_size),
        markov_head);
    if (scratch_bytes_ < estimated * sizeof(float)) {
      throw std::runtime_error(
          "DFlash executor scratch is smaller than its own memory estimate");
    }
  }

  PrewarmBlockGemms();
  // RAM ledger 10-13: the sidecar mapping was 99% resident at serve time even
  // after Create()'s drop; drop again once setup is done and log residency.
  model_->DropFilePages("after_prewarm");
  // GUFO_DEV_DRAFT_VOCAB: the cut head only ever scatters its own ids, so the
  // rest of every row is -inf once, here, after the prewarm's last write.
  if (draft_vocab_cut.size != 0) {
    HIP_CHECK(hipMemsetD32Async(d_logits_, static_cast<int>(0xFF800000U),
                                block_size * cfg.vocab_size, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
  }
}

void QwenDFlashGpuExecutor::PrewarmBlockGemms() {
  if (hipblaslt_gemm_ == nullptr) {
    return;
  }

  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  if (weights.layers.empty() || df_cfg.block_size < 2) {
    return;
  }

  const auto& layer = weights.layers.front();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const std::size_t dynamic_size = speculative::DFlashDynamicCoefficientCount(
      hidden_size, df_cfg.conv_kernel_size, df_cfg.conv_group_size);

  HIP_CHECK(hipMemsetAsync(d_block_normed_, 0,
                           df_cfg.block_size * hidden_size * sizeof(float),
                           stream_));
  HIP_CHECK(hipMemsetAsync(d_conv_hidden_, 0,
                           df_cfg.block_size * hidden_size * sizeof(float),
                           stream_));
  HIP_CHECK(hipMemsetAsync(d_attn_out_, 0,
                           df_cfg.block_size * q_dim * sizeof(float), stream_));
  HIP_CHECK(hipMemsetAsync(
      d_ffn_gate_, 0, df_cfg.block_size * intermediate_size * sizeof(float),
      stream_));

  const auto prewarm = [&](const models::QwenTensorRef& weight,
                           const float* input, float* output,
                           std::size_t batch_size, std::size_t output_size,
                           std::size_t input_size) {
    if (weight.type == core::GgmlType::kBF16 &&
        batch_size > kDFlashMaxSharedBatch) {
      RunBlockGemm(weight, input, output, batch_size, output_size, input_size);
    }
  };

  for (std::size_t block_count = 2; block_count <= df_cfg.block_size;
       ++block_count) {
    prewarm(layer.attention_conv_projection, d_block_normed_,
            d_dynamic_coefficients_, block_count, dynamic_size, hidden_size);
    prewarm(layer.transformer.attn_q, d_conv_hidden_, d_q_, block_count, q_dim,
            hidden_size);
    prewarm(layer.transformer.attn_k, d_conv_hidden_, d_k_block_, block_count,
            kv_dim, hidden_size);
    prewarm(layer.transformer.attn_output, d_attn_out_, d_fused_features_,
            block_count, hidden_size, q_dim);
    prewarm(layer.transformer.ffn_gate, d_conv_hidden_, d_ffn_gate_,
            block_count, intermediate_size, hidden_size);
    prewarm(layer.transformer.ffn_down, d_ffn_gate_, d_ffn_down_, block_count,
            hidden_size, intermediate_size);
  }

  for (std::size_t draft_count = 1; draft_count < df_cfg.block_size;
       ++draft_count) {
    prewarm(weights.selector_hidden, d_block_normed_, d_selector_hidden_,
            draft_count, df_cfg.selector_rank, hidden_size);
    prewarm(weights.output, d_block_normed_, d_logits_, draft_count,
            cfg.vocab_size, hidden_size);
  }

  HIP_CHECK(hipStreamSynchronize(stream_));
}

void QwenDFlashGpuExecutor::RunBlockGemm(const models::QwenTensorRef& weight,
                                         const float* input, float* output,
                                         std::size_t batch_size,
                                         std::size_t output_size,
                                         std::size_t input_size) {
  // opt-q4kxl: the Q4_K_M DFlash-2 companion keeps its K-quant weights packed
  // (PackBlockMatrix), so they must take a quant-aware route. The dense
  // LaunchBatchedGEMM below reads its operand as F32 or BF16 and would walk far
  // past the end of a 0.56 byte-per-element Q4_K row.
  if (weight.type == core::GgmlType::kQ8_0 ||
      detail::IsNativeWmmaQuant(weight.type)) {
    // The FP32-activation route streams each quantized weight row once for the
    // whole block and needs no BF16 activation round trip. It also beat the
    // matrix-core W8A8 route once the shared kernel read activations as float4
    // (11.3 vs 12.2 ms per draft block), so there is one route here.
    if (batch_size <= kDFlashMaxSharedBatch) {
      LaunchBatchedQuantGEMMFp32(weight.type, weight.data, input, output,
                                 batch_size, output_size, input_size, stream_);
      return;
    }
    LaunchFloatToBfloat16(input, d_bf16_input_, batch_size * input_size,
                          stream_);
    LaunchBatchedQuantGEMMBf16(weight.type, weight.data, d_bf16_input_, output,
                               batch_size, output_size, input_size, stream_);
    return;
  }

  if (weight.type != core::GgmlType::kBF16) {
    LaunchBatchedGEMM(weight.data, weight.type == core::GgmlType::kBF16, input,
                      output, batch_size, output_size, input_size, stream_);
    return;
  }

  if (batch_size <= kDFlashMaxSharedBatch) {
    LaunchExactBf16GEMMFp32SmallBatch(weight.data, input, output, batch_size,
                                      output_size, input_size, stream_);
    return;
  }

  LaunchFloatToBfloat16(input, d_bf16_input_, batch_size * input_size, stream_);
  if (hipblaslt_gemm_ != nullptr &&
      hipblaslt_gemm_->RunBf16(weight.data, d_bf16_input_, output, batch_size,
                               output_size, input_size, stream_)) {
    return;
  }
  LaunchHipblasGEMMBF16(hipblas_handle_, weight.data, d_bf16_input_, output,
                        batch_size, output_size, input_size, stream_);
}

void QwenDFlashGpuExecutor::RunInjectGemm(const models::QwenTensorRef& weight,
                                          const float* input, float* output,
                                          std::size_t num_tokens,
                                          std::size_t output_size,
                                          std::size_t input_size,
                                          bool raw_input) {
  if (weight.data == nullptr || num_tokens == 0) {
    return;
  }
  // Share weight reads across tokens without rounding the FP32 features.
  // Sixteen rows amortize the large BF16 feature projection best; quantized
  // projections and tails use the existing kernels for up to eight rows.
  const bool is_bf16 = weight.type == core::GgmlType::kBF16;
  const bool is_packed_quant = weight.type == core::GgmlType::kQ8_0 ||
                               detail::IsNativeWmmaQuant(weight.type);
  if (!is_bf16 && !is_packed_quant) {
    LaunchBatchedGEMM(weight.data, is_bf16, input, output, num_tokens,
                      output_size, input_size, stream_);
    return;
  }

  // GUFO_DEV_DFLASH_INJECT_WMMA: a prompt-injection chunk of > 64 rows
  // goes through the int8 WMMA prefill GEMM in one launch instead of one per
  // 8 rows (measured: 9630 launches, 728 ms of GPU per 8.5k-token prompt).
  // Activations are rounded to int8 as in the target's own prefill, so drafts
  // can change; the verified output cannot. Batched decode stages at most 8
  // sessions x 8 rows = 64, so decode keeps the exact route. Raw (un-normed)
  // inputs only at level 2.
  if (is_packed_quant && d_inject_q8_ != nullptr && num_tokens > 64 &&
      (!raw_input || GufoDevDFlashInjectWmma() >= 2) &&
      QuantizedActivationBytes(num_tokens, input_size) <= inject_q8_bytes_) {
    LaunchQuantizeActivationQ8_1FromFp32(input, d_inject_q8_, num_tokens,
                                         input_size, stream_);
    LaunchBatchedQuantGEMMPreQuantized(weight.type, weight.data, d_inject_q8_,
                                       output, num_tokens, output_size,
                                       input_size, stream_);
    return;
  }

  for (std::size_t offset = 0; offset < num_tokens;) {
    const std::size_t chunk =
        is_bf16 && num_tokens - offset >= 16
            ? 16
            : std::min(kDFlashMaxSharedBatch, num_tokens - offset);
    const float* chunk_input = input + (offset * input_size);
    float* chunk_output = output + (offset * output_size);
    if (is_bf16) {
      LaunchExactBf16GEMMFp32SmallBatch(weight.data, chunk_input, chunk_output,
                                        chunk, output_size, input_size,
                                        stream_);
    } else {
      LaunchBatchedQuantGEMMFp32(weight.type, weight.data, chunk_input,
                                 chunk_output, chunk, output_size, input_size,
                                 stream_);
    }
    offset += chunk;
  }
}

void QwenDFlashGpuExecutor::Free() noexcept {
  for (auto*& ptr : d_injected_k_) {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
      ptr = nullptr;
    }
  }
  for (auto*& ptr : d_injected_v_) {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
      ptr = nullptr;
    }
  }
  if (d_target_features_ != nullptr)
    (void)hipFree(d_target_features_);
  if (d_staged_features_ != nullptr) {
    (void)hipFree(d_staged_features_);
    d_staged_features_ = nullptr;
  }
  if (d_inject_q8_ != nullptr) {
    (void)hipFree(d_inject_q8_);
    d_inject_q8_ = nullptr;
  }
  if (feature_ready_ != nullptr) {
    (void)hipEventDestroy(feature_ready_);
    feature_ready_ = nullptr;
  }
  if (d_fused_features_ != nullptr)
    (void)hipFree(d_fused_features_);
  if (d_block_hidden_ != nullptr)
    (void)hipFree(d_block_hidden_);
  if (d_block_normed_ != nullptr)
    (void)hipFree(d_block_normed_);
  if (d_conv_hidden_ != nullptr)
    (void)hipFree(d_conv_hidden_);
  if (d_dynamic_coefficients_ != nullptr) {
    (void)hipFree(d_dynamic_coefficients_);
  }
  if (d_q_ != nullptr)
    (void)hipFree(d_q_);
  if (d_k_block_ != nullptr)
    (void)hipFree(d_k_block_);
  if (d_v_block_ != nullptr)
    (void)hipFree(d_v_block_);
  if (d_attn_out_ != nullptr)
    (void)hipFree(d_attn_out_);
  if (d_ffn_gate_ != nullptr)
    (void)hipFree(d_ffn_gate_);
  if (d_ffn_up_ != nullptr)
    (void)hipFree(d_ffn_up_);
  if (d_ffn_down_ != nullptr)
    (void)hipFree(d_ffn_down_);
  if (d_logits_ != nullptr)
    (void)hipFree(d_logits_);
  if (d_draft_vocab_logits_ != nullptr)
    (void)hipFree(d_draft_vocab_logits_);
  if (d_selector_hidden_ != nullptr)
    (void)hipFree(d_selector_hidden_);
  if (d_selector_candidate_ids_ != nullptr) {
    (void)hipFree(d_selector_candidate_ids_);
  }
  if (d_selector_candidate_probabilities_ != nullptr) {
    (void)hipFree(d_selector_candidate_probabilities_);
  }
  if (d_selector_uniforms_ != nullptr) {
    (void)hipFree(d_selector_uniforms_);
  }
  if (d_confidences_ != nullptr)
    (void)hipFree(d_confidences_);
  if (d_out_token_ != nullptr)
    (void)hipFree(d_out_token_);
  if (d_markov_bias_ != nullptr)
    (void)hipFree(d_markov_bias_);
  if (d_markov_w1_prev_ != nullptr)
    (void)hipFree(d_markov_w1_prev_);
  if (d_markov_prev_ != nullptr)
    (void)hipFree(d_markov_prev_);
  if (d_draft_tokens_ != nullptr)
    (void)hipFree(d_draft_tokens_);
  if (d_bf16_input_ != nullptr)
    (void)hipFree(d_bf16_input_);
  if (d_batch_scratch_ != nullptr)
    (void)hipFree(d_batch_scratch_);
  hipblaslt_gemm_.reset();
  if (hipblas_handle_ != nullptr) {
    (void)hipblasDestroy(hipblas_handle_);
    hipblas_handle_ = nullptr;
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

std::size_t QwenDFlashGpuExecutor::StateBytes() const noexcept {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  return 2U * model_->GetDFlashConfig().num_layers * history_capacity_ *
         kv_width * sizeof(float);
}

QwenGpuMemoryUsage QwenDFlashGpuExecutor::GetMemoryUsage() const noexcept {
  return {StateBytes(),
          scratch_bytes_ + batch_scratch_bytes_ + embedding_stage_.Bytes()};
}

QwenGpuMemoryUsage QwenDFlashGpuExecutor::EstimateMemoryUsage(
    const QwenDFlashGpuModel& model, std::uint32_t max_context,
    std::size_t max_batch_width) {
  if (max_context == 0 || max_context > model.GetConfig().context_length ||
      max_batch_width == 0 || max_batch_width > 8)
    throw std::invalid_argument("DFlash memory estimate has invalid limits");
  {
    // Same predicate as the constructor gate, reached earlier on the serve
    // route: ResourceClaim() calls this while sizing a runner, before any
    // executor exists. It sizes the convolutions from conv_group_size, so a
    // refusal placed after the sizing would already have divided by zero.
    const std::string refusal = speculative::DFlashGpuExecutorSupportRefusal(
        model.GetWeights());
    if (!refusal.empty()) {
      throw std::invalid_argument(refusal);
    }
  }
  const auto& config = model.GetConfig();
  const auto& draft = model.GetDFlashConfig();
  const std::size_t hidden = config.hidden_size;
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t block = draft.block_size;
  // 0 for DSpark: no convolutions, no selector. The Markov head's bias row,
  // markov_w1 row, predecessor id and emitted token row are inside this.
  const std::size_t selector_scratch_elements =
      speculative::IsDSparkDraft(model.GetWeights())
          ? 0U
          : kernels::DFlashSelectorScratchElements(config.vocab_size);
  const std::size_t scratch_elements = speculative::DFlashDraftScratchElements(
      config, draft, max_context, selector_scratch_elements,
      speculative::IsDSparkDraft(model.GetWeights()));
  QwenGpuMemoryUsage usage{
      .request_state_bytes = 2U * draft.num_layers *
                             static_cast<std::size_t>(
                                 std::min(max_context,
                                          speculative::MaxSlidingWindow(
                                              draft))) *
                             kv * sizeof(float),
      .temporary_scratch_bytes = scratch_elements * sizeof(float),
  };
  if (block > kDFlashMaxSharedBatch)
    usage.temporary_scratch_bytes +=
        block * config.intermediate_size * sizeof(hip_bfloat16);
  if (max_batch_width > 1 && config.head_dim <= 256)
    usage.temporary_scratch_bytes += BatchScratchBytes(
        model, max_batch_width * std::min<std::size_t>(block, max_context));
  return usage;
}

std::size_t QwenDFlashGpuExecutor::SnapshotPayloadBytes() const noexcept {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  return 2U * model_->GetDFlashConfig().num_layers *
         std::min(injected_context_len_, history_capacity_) * kv_width *
         sizeof(float);
}

std::unique_ptr<QwenDFlashGpuSnapshot> QwenDFlashGpuExecutor::SaveSnapshot()
    const {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  const std::size_t elements_per_layer =
      static_cast<std::size_t>(
          std::min(injected_context_len_, history_capacity_)) *
      kv_width;
  const std::size_t total_elements =
      model_->GetDFlashConfig().num_layers * elements_per_layer;
  const std::size_t bytes = total_elements * sizeof(float);

  auto snapshot =
      std::unique_ptr<QwenDFlashGpuSnapshot>(new QwenDFlashGpuSnapshot());
  snapshot->elements_per_layer_ = elements_per_layer;
  snapshot->num_layers_ = model_->GetDFlashConfig().num_layers;
  snapshot->kv_width_ = static_cast<std::uint32_t>(kv_width);
  snapshot->max_context_ = max_context_;
  snapshot->valid_context_ = injected_context_len_;
  snapshot->history_capacity_ = history_capacity_;
  snapshot->payload_bytes_ = SnapshotPayloadBytes();

  if (bytes == 0) {
    return snapshot;
  }
  HIP_CHECK(hipMalloc(&snapshot->d_k_, bytes));
  HIP_CHECK(hipMalloc(&snapshot->d_v_, bytes));
  auto* snapshot_k = static_cast<float*>(snapshot->d_k_);
  auto* snapshot_v = static_cast<float*>(snapshot->d_v_);
  for (std::size_t layer = 0; layer < snapshot->num_layers_; ++layer) {
    HIP_CHECK(hipMemcpyAsync(
        snapshot_k + layer * elements_per_layer, d_injected_k_[layer],
        elements_per_layer * sizeof(float), hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(
        snapshot_v + layer * elements_per_layer, d_injected_v_[layer],
        elements_per_layer * sizeof(float), hipMemcpyDeviceToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  return snapshot;
}

void QwenDFlashGpuExecutor::RestoreSnapshot(
    const QwenDFlashGpuSnapshot& snapshot) {
  const std::size_t kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  const std::size_t expected_elements_per_layer =
      static_cast<std::size_t>(
          std::min(snapshot.valid_context_, history_capacity_)) *
      kv_width;
  if (snapshot.num_layers_ != model_->GetDFlashConfig().num_layers ||
      snapshot.kv_width_ != kv_width || snapshot.max_context_ != max_context_ ||
      snapshot.valid_context_ > max_context_ ||
      snapshot.history_capacity_ != history_capacity_ ||
      snapshot.elements_per_layer_ != expected_elements_per_layer) {
    throw std::invalid_argument(
        "DFlash snapshot is incompatible with the executor");
  }
  const std::size_t bytes =
      snapshot.elements_per_layer_ * snapshot.num_layers_ * sizeof(float);
  if (bytes != 0 && (snapshot.d_k_ == nullptr || snapshot.d_v_ == nullptr)) {
    throw std::invalid_argument("DFlash snapshot payload is incomplete");
  }
  const auto* snapshot_k = static_cast<const float*>(snapshot.d_k_);
  const auto* snapshot_v = static_cast<const float*>(snapshot.d_v_);
  for (std::size_t layer = 0; layer < snapshot.num_layers_; ++layer) {
    if (snapshot.elements_per_layer_ == 0) {
      continue;
    }
    HIP_CHECK(hipMemcpyAsync(d_injected_k_[layer],
                             snapshot_k + layer * snapshot.elements_per_layer_,
                             snapshot.elements_per_layer_ * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_injected_v_[layer],
                             snapshot_v + layer * snapshot.elements_per_layer_,
                             snapshot.elements_per_layer_ * sizeof(float),
                             hipMemcpyDeviceToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  injected_context_len_ = snapshot.valid_context_;
}

void QwenDFlashGpuExecutor::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kDFlashGpuPersistentHeaderBytes ||
      !std::equal(kDFlashGpuPersistentMagic.begin(),
                  kDFlashGpuPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kDFlashGpuPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kDFlashGpuPersistentHeaderBytes ||
      GetLittleEndian<std::uint64_t>(payload, 56) != history_capacity_) {
    throw std::invalid_argument("DFlash GPU persistent header is invalid");
  }
  const std::uint32_t num_layers = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::uint32_t kv_width = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 24);
  const std::uint32_t valid_context =
      GetLittleEndian<std::uint32_t>(payload, 28);
  const std::size_t elements_per_layer =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));

  const std::size_t expected_kv_width =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  if (expected_kv_width > std::numeric_limits<std::uint32_t>::max() ||
      num_layers != model_->GetDFlashConfig().num_layers ||
      kv_width != expected_kv_width || max_context != max_context_ ||
      valid_context > max_context_ ||
      (kv_width != 0 &&
       valid_context > std::numeric_limits<std::size_t>::max() / kv_width) ||
      elements_per_layer !=
          static_cast<std::size_t>(std::min(valid_context, history_capacity_)) *
              kv_width ||
      (elements_per_layer != 0 &&
       num_layers >
           std::numeric_limits<std::size_t>::max() / elements_per_layer)) {
    throw std::invalid_argument(
        "DFlash GPU persistent metadata is incompatible");
  }
  const std::size_t total_elements =
      static_cast<std::size_t>(num_layers) * elements_per_layer;
  if (total_elements >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("DFlash GPU persistent size overflows");
  }
  const std::size_t bytes_per_kind = total_elements * sizeof(float);
  if (bytes_per_kind >
          std::numeric_limits<std::size_t>::max() - bytes_per_kind ||
      payload_bytes != bytes_per_kind + bytes_per_kind ||
      total_bytes != payload.size() ||
      CheckedPersistentAdd(kDFlashGpuPersistentHeaderBytes, payload_bytes) !=
          payload.size()) {
    throw std::invalid_argument(
        "DFlash GPU persistent payload size is invalid");
  }

  for (std::size_t layer = 0; layer < num_layers; ++layer) {
    if (elements_per_layer == 0) {
      continue;
    }
    const std::size_t layer_bytes = elements_per_layer * sizeof(float);
    const std::size_t layer_offset = layer * layer_bytes;
    HIP_CHECK(hipMemcpyAsync(
        d_injected_k_[layer],
        payload.data() + kDFlashGpuPersistentHeaderBytes + layer_offset,
        layer_bytes, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_injected_v_[layer],
                             payload.data() + kDFlashGpuPersistentHeaderBytes +
                                 bytes_per_kind + layer_offset,
                             layer_bytes, hipMemcpyHostToDevice, stream_));
  }
  HIP_CHECK(hipStreamSynchronize(stream_));
  injected_context_len_ = valid_context;
}

void QwenDFlashGpuExecutor::Reset() noexcept {
  // Attention only reads the valid prefix; new rows overwrite stale KV.
  injected_context_len_ = 0;
}

bool QwenDFlashGpuExecutor::InjectTargetContext(
    std::span<const float> target_features, std::uint32_t position,
    std::uint32_t num_tokens, const DFlashTrace& trace,
    bool allow_aligned_prefix_skip) {
  const std::size_t width = GetTargetFeaturesSize();
  const bool aligned_prefix_skip =
      allow_aligned_prefix_skip && injection_capacity_ > 0 &&
      position > injected_context_len_ &&
      (position - injected_context_len_) % injection_capacity_ == 0;
  if ((position != injected_context_len_ && !aligned_prefix_skip) ||
      history_capacity_ == 0 || injection_capacity_ == 0 ||
      position > max_context_ || num_tokens > max_context_ - position ||
      target_features.size() != static_cast<std::size_t>(num_tokens) * width) {
    return false;
  }
  // Only the final attention window survives this injection. Skip complete
  // scratch chunks that it overwrites, keeping the original batch boundaries
  // and absolute RoPE positions for every surviving row.
  const auto expired =
      num_tokens > history_capacity_ ? num_tokens - history_capacity_ : 0U;
  const auto first_chunk =
      (expired / injection_capacity_) * injection_capacity_;
  for (std::uint32_t offset = first_chunk; offset < num_tokens;) {
    const auto count = std::min(injection_capacity_, num_tokens - offset);
    if (!InjectTargetContextChunk(
            target_features.subspan(static_cast<std::size_t>(offset) * width,
                                    static_cast<std::size_t>(count) * width),
            position + offset, count, trace)) {
      return false;
    }
    offset += count;
  }
  return true;
}

bool QwenDFlashGpuExecutor::StageTargetFeaturesDevice(
    const float* device_rows, std::size_t row_stride, hipStream_t src_stream,
    std::uint32_t staged_rows, std::uint32_t num_rows) {
  const std::size_t width = GetTargetFeaturesSize();
  if (d_staged_features_ == nullptr || feature_ready_ == nullptr ||
      device_rows == nullptr || row_stride != width || width == 0 ||
      num_rows == 0 || staged_rows > staging_capacity_ ||
      num_rows > staging_capacity_ - staged_rows) {
    return false;
  }
  // The copy runs on the TARGET's stream, not this one. That is what makes it
  // safe without a host wait: the source rows were written on that stream, and
  // the next verification batch that overwrites them is also on that stream,
  // so plain stream order brackets the read on both sides. Only the draft side
  // needs telling, and one event does that. Issuing the copy on the draft
  // stream instead would leave it racing the next batch's writes, since the
  // target's arena stream is non-blocking and shares no implicit order.
  if (hipMemcpyAsync(d_staged_features_ + (staged_rows * width), device_rows,
                     static_cast<std::size_t>(num_rows) * width * sizeof(float),
                     hipMemcpyDeviceToDevice, src_stream) != hipSuccess) {
    return false;
  }
  if (src_stream != stream_) {
    if (hipEventRecord(feature_ready_, src_stream) != hipSuccess ||
        hipStreamWaitEvent(stream_, feature_ready_, 0) != hipSuccess) {
      return false;
    }
  }
  return true;
}

bool QwenDFlashGpuExecutor::InjectStagedTargetFeatures(
    std::uint32_t position, std::uint32_t num_tokens) {
  const std::size_t width = GetTargetFeaturesSize();
  if (d_staged_features_ == nullptr || num_tokens == 0 ||
      num_tokens > staging_capacity_ || position != injected_context_len_ ||
      history_capacity_ == 0 || injection_capacity_ == 0 ||
      position > max_context_ || num_tokens > max_context_ - position) {
    return false;
  }
  // One round stages at most one block, which is below injection_capacity_, so
  // the host entry's chunk loop and its history_capacity_ expiry arithmetic
  // cannot be reached from here: a single chunk always covers the rows.
  if (width == 0 || num_tokens > injection_capacity_ ||
      num_tokens > history_capacity_) {
    return false;
  }
  return InjectChunkFromDeviceFeatures(d_staged_features_, position,
                                       num_tokens, {});
}

bool QwenDFlashGpuExecutor::ReadStagedTargetFeatures(
    std::uint32_t num_rows, std::vector<float>& out) const {
  const std::size_t width = GetTargetFeaturesSize();
  if (d_staged_features_ == nullptr || num_rows > staging_capacity_ ||
      width == 0) {
    return false;
  }
  out.resize(static_cast<std::size_t>(num_rows) * width);
  if (out.empty())
    return true;
  // The staging copy ran on the target's stream with this stream waiting on
  // it, so draining this stream drains the copy too. The target's arena stream
  // is non-blocking, so a null-stream hipMemcpy alone would not wait for it.
  if (hipStreamSynchronize(stream_) != hipSuccess)
    return false;
  return hipMemcpy(out.data(), d_staged_features_, out.size() * sizeof(float),
                   hipMemcpyDeviceToHost) == hipSuccess;
}

bool QwenDFlashGpuExecutor::InjectTargetContextChunk(
    std::span<const float> target_features, std::uint32_t position,
    std::uint32_t num_tokens, const DFlashTrace& trace) {
  const std::size_t enc_in_dim = GetTargetFeaturesSize();
  if (target_features.size() != num_tokens * enc_in_dim ||
      position + num_tokens > max_context_) {
    return false;
  }
  // Upload target features, then run the shared chain on them.
  const std::size_t bytes = target_features.size() * sizeof(float);
  const auto cpy_err =
      hipMemcpyAsync(d_target_features_, target_features.data(), bytes,
                     hipMemcpyHostToDevice, stream_);
  if (cpy_err != hipSuccess) {
    return false;
  }
  return InjectChunkFromDeviceFeatures(d_target_features_, position, num_tokens,
                                       trace);
}

bool QwenDFlashGpuExecutor::InjectChunkFromDeviceFeatures(
    const float* device_features, std::uint32_t position,
    std::uint32_t num_tokens, const DFlashTrace& trace) {
  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t enc_in_dim = GetTargetFeaturesSize();
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;

  if (device_features == nullptr || num_tokens == 0 ||
      position + num_tokens > max_context_) {
    return false;
  }

  ::gufo::core::hip::dev_timing::EventScope inject_timer(
      ::gufo::core::hip::dev_timing::Phase::kInject, stream_);

  // FC Projection & Norm
  RunInjectGemm(weights.fc_projection, device_features, d_fused_features_,
                num_tokens, hidden_size, enc_in_dim, /*raw_input=*/true);

  LaunchBatchedRMSNorm(
      d_fused_features_, static_cast<const float*>(weights.fc_norm.data),
      d_block_normed_, nullptr, num_tokens, hidden_size, 1e-6F, stream_);
  const auto prefix =
      trace ? "context." + std::to_string(position) + "." : std::string{};
  if (trace)
    TraceTensor(trace, prefix + "normalized", d_block_normed_,
                num_tokens * hidden_size, stream_);

  // Project and store K / V for each draft layer
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const auto& layer = weights.layers[i].transformer;
    if (layer.attn_k.data != nullptr && layer.attn_v.data != nullptr) {
      RunInjectGemm(layer.attn_k, d_block_normed_, d_k_block_, num_tokens,
                    kv_dim, hidden_size);
      RunInjectGemm(layer.attn_v, d_block_normed_, d_v_block_, num_tokens,
                    kv_dim, hidden_size);
    }

    if (layer.attn_k_norm.data != nullptr) {
      NormalizeAndRotateQK(nullptr, d_k_block_, nullptr,
                           static_cast<const float*>(layer.attn_k_norm.data),
                           cfg, 0, num_tokens, position, stream_);
    } else {
      LaunchBatchedRoPE(nullptr, d_k_block_, num_tokens, 0,
                        cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim,
                        position, cfg.rope_theta, stream_);
    }

    if (trace) {
      const auto layer_prefix = prefix + std::to_string(i) + ".";
      TraceTensor(trace, layer_prefix + "k", d_k_block_, num_tokens * kv_dim,
                  stream_);
      TraceTensor(trace, layer_prefix + "v", d_v_block_, num_tokens * kv_dim,
                  stream_);
    }

    // Split a write that crosses the physical end of the ring.
    for (std::uint32_t offset = 0; offset < num_tokens;) {
      const auto slot = (position + offset) % history_capacity_;
      const auto count =
          std::min(num_tokens - offset, history_capacity_ - slot);
      const std::size_t cache_bytes = count * kv_dim * sizeof(float);
      HIP_CHECK(hipMemcpyAsync(d_injected_k_[i] + slot * kv_dim,
                               d_k_block_ + offset * kv_dim, cache_bytes,
                               hipMemcpyDeviceToDevice, stream_));
      HIP_CHECK(hipMemcpyAsync(d_injected_v_[i] + slot * kv_dim,
                               d_v_block_ + offset * kv_dim, cache_bytes,
                               hipMemcpyDeviceToDevice, stream_));
      offset += count;
    }
  }

  injected_context_len_ =
      std::max(injected_context_len_, position + num_tokens);
  inject_timer.Stop();
  if (!::gufo::hip::GufoDevSkipExtraSync())
    (void)hipStreamSynchronize(stream_);
  return true;
}

std::vector<tokenization::TokenId> QwenDFlashGpuExecutor::ForwardBlock(
    tokenization::TokenId anchor_token, std::uint32_t current_pos,
    std::uint32_t draft_count, float temperature,
    std::span<const float> sample_uniforms, std::vector<float>* out_confidences,
    std::vector<tokenization::TokenId>* out_candidate_ids,
    std::vector<float>* out_candidate_probabilities, const DFlashTrace& trace) {
  last_draft_logits_rows_ = 0;
  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t num_q_heads = cfg.num_attention_heads;
  const std::size_t num_kv_heads = cfg.num_key_value_heads;
  const std::size_t head_dim = cfg.head_dim;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t kv_dim = static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t vocab_size = cfg.vocab_size;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  const std::size_t dynamic_size = speculative::DFlashDynamicCoefficientCount(
      hidden_size, df_cfg.conv_kernel_size, df_cfg.conv_group_size);

  draft_count = std::min(draft_count,
                         df_cfg.block_size > 0 ? df_cfg.block_size - 1U : 0U);
  if (anchor_token >= vocab_size || df_cfg.mask_token_id >= vocab_size ||
      current_pos != injected_context_len_ || current_pos >= max_context_) {
    throw std::invalid_argument(
        "DFlash block token or committed position is invalid");
  }
  draft_count = std::min(draft_count, max_context_ - current_pos - 1U);
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument(
        "DFlash sampling temperature must be finite and nonnegative");
  }
  if (temperature > 0.0F && sample_uniforms.size() < draft_count) {
    throw std::invalid_argument(
        "DFlash sampled drafting requires one random value per proposal");
  }
  if (draft_count == 0) {
    if (out_confidences != nullptr) {
      out_confidences->clear();
    }
    if (out_candidate_ids != nullptr) {
      out_candidate_ids->clear();
    }
    if (out_candidate_probabilities != nullptr) {
      out_candidate_probabilities->clear();
    }
    return {};
  }
  const std::uint32_t block_count = draft_count + 1U;
  ::gufo::core::hip::dev_timing::EventScope draft_forward_timer(
      ::gufo::core::hip::dev_timing::Phase::kDraftForward, stream_);

  // Slot zero stays the selector's predecessor until its first step. Proposal
  // slots start as masks and are overwritten only after the embedding lookup.
  // Keep the upload source alive through the final stream synchronization.
  std::vector<tokenization::TokenId> block_tokens(block_count,
                                                  df_cfg.mask_token_id);
  block_tokens.front() = anchor_token;
  HIP_CHECK(hipMemcpyAsync(d_out_token_, block_tokens.data(),
                           block_tokens.size() * sizeof(block_tokens.front()),
                           hipMemcpyHostToDevice, stream_));
  if (weights.token_embedding.data != nullptr) {
    embedding_stage_.LaunchBatch(
        model_->GetEmbeddingRows(), weights.token_embedding, block_tokens,
        d_out_token_, d_block_hidden_, hidden_size, stream_);
  }
  TraceTensor(trace, "embedding", d_block_hidden_, block_count * hidden_size,
              stream_);

  // Five DFlash-2 decoder blocks.
  for (std::size_t i = 0; i < df_cfg.num_layers; ++i) {
    const auto& dflash_layer = weights.layers[i];
    const auto& layer = dflash_layer.transformer;
    const auto emit = [&](std::string_view stage, const float* data,
                          std::size_t width) {
      if (trace)
        TraceTensor(trace,
                    "layer." + std::to_string(i) + "." + std::string(stage),
                    data, block_count * width, stream_);
    };

    LaunchBatchedRMSNorm(
        d_block_hidden_, static_cast<const float*>(layer.attn_norm.data),
        d_block_normed_, nullptr, block_count, hidden_size, 1e-6F, stream_);
    emit("attn_norm", d_block_normed_, hidden_size);

    // Absent convolutions (DSpark): Q/K/V read the normed hidden directly and
    // no conv residual add runs at all, which is what the reference does when
    // dflash_attn_conv_proj is null (src/models/dflash.cpp:709-712, :742-743).
    const bool has_conv = !dflash_layer.attention_conv_base.empty();
    float* const attention_input = has_conv ? d_conv_hidden_ : d_block_normed_;
    if (has_conv) {
      RunBlockGemm(dflash_layer.attention_conv_projection, d_block_normed_,
                   d_dynamic_coefficients_, block_count, dynamic_size,
                   hidden_size);
      kernels::LaunchDFlashGroupedDynamicConv(
          d_block_normed_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.attention_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 0, stream_);
      emit("attn_conv_in", d_conv_hidden_, hidden_size);
    }

    RunBlockGemm(layer.attn_q, attention_input, d_q_, block_count, q_dim,
                 hidden_size);
    RunBlockGemm(layer.attn_k, attention_input, d_k_block_, block_count, kv_dim,
                 hidden_size);
    RunBlockGemm(layer.attn_v, attention_input, d_v_block_, block_count, kv_dim,
                 hidden_size);

    NormalizeAndRotateQK(d_q_, d_k_block_,
                         static_cast<const float*>(layer.attn_q_norm.data),
                         static_cast<const float*>(layer.attn_k_norm.data), cfg,
                         num_q_heads, block_count, current_pos, stream_);
    emit("q", d_q_, q_dim);
    emit("k", d_k_block_, kv_dim);
    emit("v", d_v_block_, kv_dim);

    kernels::LaunchDFlashNonCausalAttention(
        d_q_, d_injected_k_[i], d_injected_v_[i], d_k_block_, d_v_block_,
        d_attn_out_, current_pos, std::min(current_pos, injected_context_len_),
        block_count, speculative::LayerSlidingWindow(df_cfg, i),
        static_cast<std::uint32_t>(num_q_heads),
        static_cast<std::uint32_t>(num_kv_heads),
        static_cast<std::uint32_t>(head_dim), scale, stream_,
        history_capacity_);
    emit("attention", d_attn_out_, q_dim);

    RunBlockGemm(layer.attn_output, d_attn_out_, d_fused_features_, block_count,
                 hidden_size, q_dim);
    emit("attn_output", d_fused_features_, hidden_size);
    if (has_conv) {
      kernels::LaunchDFlashGroupedDynamicConv(
          d_fused_features_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.attention_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 1, stream_);
      emit("attn_conv_out", d_conv_hidden_, hidden_size);
      LaunchBatchedResidualAdd(d_block_hidden_, d_conv_hidden_, d_block_hidden_,
                               block_count, hidden_size, stream_);
    } else {
      LaunchBatchedResidualAdd(d_block_hidden_, d_fused_features_,
                               d_block_hidden_, block_count, hidden_size,
                               stream_);
    }
    emit("attn_residual", d_block_hidden_, hidden_size);

    LaunchBatchedRMSNorm(
        d_block_hidden_, static_cast<const float*>(layer.ffn_norm.data),
        d_block_normed_, nullptr, block_count, hidden_size, 1e-6F, stream_);
    emit("ffn_norm", d_block_normed_, hidden_size);

    float* const ffn_input = has_conv ? d_conv_hidden_ : d_block_normed_;
    if (has_conv) {
      RunBlockGemm(dflash_layer.ffn_conv_projection, d_block_normed_,
                   d_dynamic_coefficients_, block_count, dynamic_size,
                   hidden_size);
      kernels::LaunchDFlashGroupedDynamicConv(
          d_block_normed_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.ffn_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 0, stream_);
      emit("ffn_conv_in", d_conv_hidden_, hidden_size);
    }

    RunBlockGemm(layer.ffn_gate, ffn_input, d_ffn_gate_, block_count,
                 intermediate_size, hidden_size);
    RunBlockGemm(layer.ffn_up, ffn_input, d_ffn_up_, block_count,
                 intermediate_size, hidden_size);
    kernels::LaunchDFlashSiLUMul(d_ffn_gate_, d_ffn_up_,
                                 block_count * intermediate_size, stream_);
    RunBlockGemm(layer.ffn_down, d_ffn_gate_, d_ffn_down_, block_count,
                 hidden_size, intermediate_size);
    emit("ffn_down", d_ffn_down_, hidden_size);

    if (has_conv) {
      kernels::LaunchDFlashGroupedDynamicConv(
          d_ffn_down_, d_dynamic_coefficients_,
          static_cast<const float*>(dflash_layer.ffn_conv_base.data),
          d_conv_hidden_, block_count, hidden_size, df_cfg.conv_kernel_size,
          df_cfg.conv_group_size, 1, stream_);
      emit("ffn_conv_out", d_conv_hidden_, hidden_size);
      LaunchBatchedResidualAdd(d_block_hidden_, d_conv_hidden_, d_block_hidden_,
                               block_count, hidden_size, stream_);
    } else {
      LaunchBatchedResidualAdd(d_block_hidden_, d_ffn_down_, d_block_hidden_,
                               block_count, hidden_size, stream_);
    }
    emit("output", d_block_hidden_, hidden_size);
  }

  LaunchBatchedRMSNorm(
      d_block_hidden_, static_cast<const float*>(weights.output_norm.data),
      d_block_normed_, nullptr, block_count, hidden_size, 1e-6F, stream_);
  TraceTensor(trace, "normalized", d_block_normed_, block_count * hidden_size,
              stream_);
  draft_forward_timer.Stop();

  std::vector<tokenization::TokenId> tokens(draft_count, 0);

  // DSpark: base output GEMM, then the Markov bias and the confidence head,
  // chained inside the block because the bias of position i reads the token
  // drafted at position i-1 (src/models/dflash.cpp:384 `prev = argmax(col)`).
  //
  // `dflash.sample_from_anchor` decides the chain START, exactly as the
  // reference does at :316-318: true drafts the anchor slot itself and starts
  // at row 0, false treats slot 0 as a bonus anchor and starts at row 1. It
  // does not decide how MANY tokens the caller receives: the block is
  // draft_count + 1 rows wide, and the verifier contract is draft_count, so the
  // range is capped there (DSparkChainRangeFor).
  if (speculative::IsDSparkDraft(weights)) {
    ::gufo::core::hip::dev_timing::EventScope markov_timer(
        ::gufo::core::hip::dev_timing::Phase::kMarkovConfidence, stream_);
    const auto chain =
        speculative::DSparkChainRangeFor(block_count, draft_count,
                                         df_cfg.sample_from_anchor);
    const std::size_t rows = chain.rows;
    const std::size_t first_row = chain.first_row;
    RunBlockGemm(weights.output,
                 d_block_normed_ + first_row * hidden_size, d_logits_, rows,
                 vocab_size, hidden_size);
    TraceTensor(trace, "logits", d_logits_, rows * vocab_size, stream_);

    if (weights.markov_w1.type != core::GgmlType::kBF16 ||
        weights.markov_w2.type != core::GgmlType::kBF16 ||
        weights.conf_projection.type != core::GgmlType::kF32 ||
        weights.conf_bias.type != core::GgmlType::kF32) {
      throw std::runtime_error(
          "DFlash DSpark Markov weights are not in their packed BF16/F32 form");
    }
    if (weights.conf_projection.num_elements !=
        hidden_size + df_cfg.markov_rank) {
      throw std::runtime_error(
          "DFlash DSpark confidence projection has the wrong width");
    }
    HIP_CHECK(hipMemcpyAsync(d_markov_prev_, d_out_token_,
                             sizeof(std::uint32_t), hipMemcpyDeviceToDevice,
                             stream_));
    for (std::size_t step = 0; step < rows; ++step) {
      float* const row_logits = d_logits_ + step * vocab_size;
      const float* const row_hidden =
          d_block_normed_ + (first_row + step) * hidden_size;
      // STEP 0 ONLY, BEFORE the Markov loop: the BASE logit rows, as the output
      // GEMM just wrote them. The post-Markov dump further down holds
      // base + bias, and a HOST reference can only compute the BIAS
      // (ComputeMarkovBiasRow), so the difference of the two dumps is the one
      // quantity both sides can produce. Dumping only the biased rows would make
      // the comparison impossible rather than merely hard.
      if (const char* const dump = std::getenv("GUFO_DEV_DRAFT_DUMP");
          step == 0 && bias_dump_written_ == false && dump != nullptr &&
          dump[0] != '\0') {
        {
          const std::string base_path = std::string(dump) + ".base.bin";
          if (std::FILE* const out = std::fopen(base_path.c_str(), "wb")) {
            const std::uint32_t header[2] = {static_cast<std::uint32_t>(rows),
                                             static_cast<std::uint32_t>(vocab_size)};
            std::fwrite(header, sizeof(header), 1, out);
            std::vector<float> host(rows * vocab_size, 0.0F);
            if (hipMemcpy(host.data(), d_logits_,
                          host.size() * sizeof(float),
                          hipMemcpyDeviceToHost) == hipSuccess) {
              std::fwrite(host.data(), sizeof(float), host.size(), out);
            }
            std::fclose(out);
          }
        }
      }
      kernels::LaunchDFlashMarkovStep(
          weights.markov_w1.data, weights.markov_w2.data, d_markov_prev_,
          d_markov_bias_, d_markov_w1_prev_, row_logits, row_hidden,
          weights.conf_projection.data, weights.conf_bias.data,
          d_confidences_ + step, d_draft_tokens_ + step, vocab_size,
          df_cfg.markov_rank, hidden_size, temperature,
          temperature > 0.0F
              ? static_cast<const float*>(d_selector_uniforms_) + step
              : nullptr,
          stream_);
      // The chain always continues on the argmax of the biased column, even
      // when this position was sampled (src/models/dflash.cpp:381-384).
      HIP_CHECK(hipMemcpyAsync(d_markov_prev_, d_draft_tokens_ + step,
                               sizeof(std::uint32_t),
                               hipMemcpyDeviceToDevice, stream_));
    }
    std::vector<tokenization::TokenId> draft_tokens(rows, 0);
    const auto draft_copy =
        hipMemcpyAsync(draft_tokens.data(), d_draft_tokens_,
                       rows * sizeof(tokenization::TokenId),
                       hipMemcpyDeviceToHost, stream_);
    if (draft_copy != hipSuccess) {
      throw std::runtime_error("DFlash GPU Markov draft download failed");
    }
    if (out_confidences != nullptr) {
      out_confidences->resize(rows);
      const auto confidence_copy = hipMemcpyAsync(
          out_confidences->data(), d_confidences_, rows * sizeof(float),
          hipMemcpyDeviceToHost, stream_);
      if (confidence_copy != hipSuccess) {
        throw std::runtime_error("DFlash GPU confidence download failed");
      }
    }
    markov_timer.Stop();
    if (hipStreamSynchronize(stream_) != hipSuccess) {
      throw std::runtime_error("DFlash GPU block synchronization failed");
    }
    last_draft_logits_rows_ = rows;
    // Draft-bias dump, step 0 ONLY: the post-Markov logit row of each proposed
    // position, so a HOST reference can recompute the Markov bias from the
    // GGUF and compare. d_logits_ holds what LaunchDFlashMarkovStep wrote — the
    // BIASED column, which is the thing the reference has to reproduce.
    //
    // FIRST CALL ONLY, decided by the file not existing rather than by a counter
    // member: a member would need resetting between runs and would silently
    // write the wrong step after a resume. Placed AFTER the stream sync above,
    // because before it the rows are not guaranteed to be the Markov output.
    //
    // Header: two uint32 (rows, vocab), then rows * vocab f32, row-major. The
    // header is what makes the file self-describing — an unlabelled float blob
    // is unreadable a month later.
    if (const char* const dump = std::getenv("GUFO_DEV_DRAFT_DUMP");
        dump != nullptr && dump[0] != '\0' && bias_dump_written_ == false) {
      const std::string bias_path = std::string(dump) + ".bias.bin";
      if (std::FILE* const probe = std::fopen(bias_path.c_str(), "rb")) {
        std::fclose(probe);  // already written by an earlier call
      } else if (std::FILE* const out = std::fopen(bias_path.c_str(), "wb")) {
        const std::uint32_t header[2] = {static_cast<std::uint32_t>(rows),
                                         static_cast<std::uint32_t>(vocab_size)};
        std::fwrite(header, sizeof(header), 1, out);
        std::vector<float> host(rows * vocab_size, 0.0F);
        const hipError_t copy = hipMemcpy(
            host.data(), d_logits_, host.size() * sizeof(float),
            hipMemcpyDeviceToHost);
        if (copy == hipSuccess) {
          std::fwrite(host.data(), sizeof(float), host.size(), out);
        }
        std::fclose(out);
      }
      bias_dump_written_ = true;
    }
    // A DSpark proposal carries no candidate lattice; the selector tensors it
    // would read do not exist. The tokens the caller receives are exactly the
    // rows the chain produced.
    return draft_tokens;
  }

  ::gufo::core::hip::dev_timing::EventScope markov_timer(
      ::gufo::core::hip::dev_timing::Phase::kMarkovConfidence, stream_);
  RunBlockGemm(weights.selector_hidden, d_block_normed_ + hidden_size,
               d_selector_hidden_, draft_count, df_cfg.selector_rank,
               hidden_size);
  TraceTensor(trace, "selector_hidden", d_selector_hidden_,
              draft_count * df_cfg.selector_rank, stream_);

  if (const auto& cut = model_->GetDraftVocabCut(); cut.size != 0) {
    RunBlockGemm(cut.head, d_block_normed_ + hidden_size,
                 d_draft_vocab_logits_, draft_count, cut.size, hidden_size);
    kernels::LaunchDFlashScatterDraftLogits(
        d_draft_vocab_logits_, cut.d_ids, d_logits_, draft_count, cut.size,
        static_cast<std::uint32_t>(vocab_size), stream_);
  } else {
    RunBlockGemm(weights.output, d_block_normed_ + hidden_size, d_logits_,
                 draft_count, vocab_size, hidden_size);
  }
  TraceTensor(trace, "logits", d_logits_, draft_count * vocab_size, stream_);

  if (temperature > 0.0F) {
    const auto uniform_copy = hipMemcpyAsync(
        d_selector_uniforms_, sample_uniforms.data(),
        draft_count * sizeof(float), hipMemcpyHostToDevice, stream_);
    if (uniform_copy != hipSuccess) {
      throw std::runtime_error("DFlash GPU sampling upload failed");
    }
  }
  // The FFN is finished. Its buffers hold all independent top-k partials;
  // one selector launch then walks this request's predecessor chain.
  static const bool draft_conf = [] {
    const char* const env = std::getenv("GUFO_DEV_DRAFT_CONF");
    return env != nullptr && env[0] == '1';
  }();
  const std::array sequences{kernels::DFlashSelectorSequence{
      draft_count, temperature, draft_conf ? 1U : 0U}};
  kernels::LaunchDFlashSelectorBatch(
      d_logits_, d_selector_hidden_, weights.selector_predecessor.data,
      weights.selector_successor.data, d_out_token_, d_confidences_,
      d_ffn_gate_, reinterpret_cast<std::uint32_t*>(d_ffn_up_),
      temperature > 0.0F ? d_selector_uniforms_ : nullptr,
      out_candidate_ids != nullptr ? d_selector_candidate_ids_ : nullptr,
      out_candidate_probabilities != nullptr
          ? d_selector_candidate_probabilities_
          : nullptr,
      sequences, vocab_size, df_cfg.selector_rank, df_cfg.selector_top_k,
      stream_,
      kernels::DFlashMarkovRerank{weights.rerank_markov_w1.data,
                                  weights.rerank_markov_w2.data,
                                  weights.rerank_markov_rank,
                                  weights.rerank_markov_scale});

  const auto token_copy =
      hipMemcpyAsync(tokens.data(), d_out_token_ + 1U,
                     draft_count * sizeof(tokenization::TokenId),
                     hipMemcpyDeviceToHost, stream_);
  if (token_copy != hipSuccess) {
    throw std::runtime_error("DFlash GPU selector download failed");
  }

  if (out_confidences != nullptr) {
    out_confidences->resize(draft_count);
    const auto confidence_copy = hipMemcpyAsync(
        out_confidences->data(), d_confidences_, draft_count * sizeof(float),
        hipMemcpyDeviceToHost, stream_);
    if (confidence_copy != hipSuccess) {
      throw std::runtime_error("DFlash GPU confidence download failed");
    }
  }
  const std::size_t candidate_count =
      static_cast<std::size_t>(draft_count) * df_cfg.selector_top_k;
  if (out_candidate_ids != nullptr) {
    out_candidate_ids->resize(candidate_count);
    const auto candidate_copy =
        hipMemcpyAsync(out_candidate_ids->data(), d_selector_candidate_ids_,
                       candidate_count * sizeof(tokenization::TokenId),
                       hipMemcpyDeviceToHost, stream_);
    if (candidate_copy != hipSuccess) {
      throw std::runtime_error("DFlash GPU candidate download failed");
    }
  }
  if (out_candidate_probabilities != nullptr) {
    out_candidate_probabilities->resize(candidate_count);
    const auto probability_copy = hipMemcpyAsync(
        out_candidate_probabilities->data(),
        d_selector_candidate_probabilities_, candidate_count * sizeof(float),
        hipMemcpyDeviceToHost, stream_);
    if (probability_copy != hipSuccess) {
      throw std::runtime_error(
          "DFlash GPU candidate probability download failed");
    }
  }
  markov_timer.Stop();
  if (hipStreamSynchronize(stream_) != hipSuccess) {
    throw std::runtime_error("DFlash GPU block synchronization failed");
  }
  last_draft_logits_rows_ = draft_count;

  return tokens;
}

bool QwenDFlashGpuExecutor::TryGetDraftRankStats(
    std::size_t draft_row, tokenization::TokenId target_token,
    std::uint32_t& rank, float& top1_probability) const noexcept {
  try {
    const std::size_t vocab_size = model_->GetConfig().vocab_size;
    if (d_logits_ == nullptr || draft_row >= last_draft_logits_rows_ ||
        target_token >= vocab_size || vocab_size == 0) {
      return false;
    }
    std::vector<float> host_logits(vocab_size);
    const float* const device_row = d_logits_ + draft_row * vocab_size;
    if (hipMemcpy(host_logits.data(), device_row, vocab_size * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return false;
    }

    const double negative_infinity = -std::numeric_limits<double>::infinity();
    double logit_max = negative_infinity;
    // One row copy serves both the rank and top-1 probability; no extra GPU
    // launch or transfer is needed for p1.
    double exp_sum = 0.0;
    std::uint32_t tokens_ahead = 0;
    const double target_logit = host_logits[target_token];
    for (std::size_t token = 0; token < vocab_size; ++token) {
      const double value = host_logits[token];
      if (std::isnan(value) ||
          value == std::numeric_limits<double>::infinity()) {
        return false;
      }
      if (value > target_logit ||
          (value == target_logit && token < target_token)) {
        ++tokens_ahead;
      }
      if (value > logit_max) {
        exp_sum = logit_max == negative_infinity
                      ? 1.0
                      : exp_sum * std::exp(logit_max - value) + 1.0;
        logit_max = value;
      } else if (value != negative_infinity) {
        exp_sum += std::exp(value - logit_max);
      }
    }
    if (!std::isfinite(logit_max) || !std::isfinite(exp_sum) ||
        exp_sum <= 0.0) {
      return false;
    }
    constexpr std::uint32_t kLoggedRankLimit = 32;
    rank = tokens_ahead < kLoggedRankLimit ? tokens_ahead + 1U : 0U;
    top1_probability = static_cast<float>(1.0 / exp_sum);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

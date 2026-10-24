#if defined(ENGINE_ENABLE_HIP)
#include <cstdlib>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/mapped_prefetch.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/dflash/draft_vocab.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kPackChunkBytes = 8ULL << 20;

/// GUFO_DEV_DFLASH_KEEP_QUANT=1. ONE definition, two readers: the fc/K/V
/// `pack_dense` choice in PackDFlashWeights and PackBlockMatrix's widened
/// keep-list. The predicate is `value[0] == '1'` exactly as the original inline
/// read had it, so the switch means the same thing it has always meant.
[[nodiscard]] bool DFlashKeepQuantEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_DFLASH_KEEP_QUANT");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

/// Dev switch GUFO_DEV_DFLASH_DROP_PAGECACHE=1: after the pack, hand
/// back the sidecar's prefaulted file pages. See DropMappedRangeFromPageCache
/// for why the order of the two syscalls is the whole point.
[[nodiscard]] bool DFlashDropPageCacheEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_DFLASH_DROP_PAGECACHE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

[[nodiscard]] std::uint16_t FloatToBfloat16Bits(float value) noexcept {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
  bits += rounding_bias;
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] const void* QuantizedRow(const models::QwenTensorRef& tensor,
                                       std::size_t row,
                                       std::size_t columns) noexcept {
  const std::size_t row_bytes = quant::QuantizedRowBytes(tensor.type, columns);
  if (row_bytes == 0) {
    return nullptr;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + (row * row_bytes);
}

void ReadMatrixRow(const models::QwenTensorRef& tensor, std::size_t row,
                   std::size_t columns, float* output) {
  if (tensor.type == core::GgmlType::kF32) {
    const auto* source =
        static_cast<const float*>(tensor.data) + (row * columns);
    std::copy_n(source, columns, output);
    return;
  }
  if (tensor.type == core::GgmlType::kBF16) {
    const std::size_t offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto bits =
          static_cast<const std::uint16_t*>(tensor.data)[offset + column];
      output[column] =
          std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    return;
  }

  const void* source = QuantizedRow(tensor, row, columns);
  if (source == nullptr) {
    throw std::runtime_error("unsupported DFlash matrix quantization");
  }
  switch (tensor.type) {
    case core::GgmlType::kQ8_0:
      quant::DequantizeQ8_0(source, output, columns);
      return;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(source, output, columns);
      return;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(source, output, columns);
      return;
    case core::GgmlType::kQ2_K:
      quant::DequantizeQ2_K(source, output, columns);
      return;
    case core::GgmlType::kQ3_K:
      quant::DequantizeQ3_K(source, output, columns);
      return;
    case core::GgmlType::kQ5_K:
      quant::DequantizeQ5_K(source, output, columns);
      return;
    case core::GgmlType::kIQ4_XS:
      quant::DequantizeIQ4_XS(source, output, columns);
      return;
    default:
      throw std::runtime_error("unsupported DFlash tensor format for packing");
  }
}

void CopyToDevice(void* destination, const void* source, std::size_t bytes) {
  const auto check = [](hipError_t error) {
    if (error != hipSuccess) {
      throw std::runtime_error(std::string("DFlash hipMemcpy failed: ") +
                               hipGetErrorString(error));
    }
  };
  // Measured RAM ledger: after both page drops the sidecar
  // mapping was still 1001 MiB Rss and 1.06 GB in the page cache. A pageable
  // hipMemcpy of a large span pins the SOURCE pages (model_loader.cpp says the
  // same of the target weights), and pinned file pages can be neither dropped
  // nor zapped without an MMU-notifier eviction. Every copy therefore goes
  // through a pinned bounce buffer (small ones too, found in review: small
  // raw GGUF spans at the callers below), so the mapping is only read by memcpy.
  // ponytail: one synchronous buffer per call, up to 64 MiB; double-buffer like
  // CopyRegionToDevice if sidecar load time ever matters.
  if (bytes == 0) return;
  const std::size_t bounce_bytes = std::min<std::size_t>(bytes, 64ULL << 20);
  void* bounce = nullptr;
  check(hipHostMalloc(&bounce, bounce_bytes, hipHostMallocDefault));
  const std::unique_ptr<void, decltype(&hipHostFree)> owned(bounce,
                                                            &hipHostFree);
  for (std::size_t offset = 0; offset < bytes; offset += bounce_bytes) {
    const std::size_t chunk = std::min(bounce_bytes, bytes - offset);
    std::memcpy(bounce, static_cast<const std::uint8_t*>(source) + offset,
                chunk);
    check(hipMemcpy(static_cast<std::uint8_t*>(destination) + offset, bounce,
                    chunk, hipMemcpyHostToDevice));
  }
}

models::QwenTensorRef CopyVectorF32(const models::QwenTensorRef& source,
                                    std::size_t elements,
                                    std::vector<void*>& allocations,
                                    std::size_t& packed_bytes) {
  if (source.empty() || elements == 0) {
    return {};
  }
  std::vector<float> host(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    host[index] = source.Get(index);
  }
  const std::size_t bytes = elements * sizeof(float);
  void* device = detail::AllocateDevice(bytes);
  allocations.push_back(device);
  CopyToDevice(device, host.data(), bytes);
  packed_bytes += bytes;
  return {
      .data = device,
      .type = core::GgmlType::kF32,
      .num_elements = elements,
  };
}

models::QwenTensorRef PackMatrixBf16(const models::QwenTensorRef& source,
                                     std::size_t rows, std::size_t columns,
                                     std::vector<void*>& allocations,
                                     std::size_t& packed_bytes) {
  if (source.empty() || rows == 0 || columns == 0) {
    return {};
  }
  const std::size_t total_elements = rows * columns;
  const std::size_t total_bytes = total_elements * sizeof(std::uint16_t);
  void* device = detail::AllocateDevice(total_bytes);
  allocations.push_back(device);

  if (source.type == core::GgmlType::kBF16) {
    CopyToDevice(device, source.data, total_bytes);
    packed_bytes += total_bytes;
    return {.data = device,
            .type = core::GgmlType::kBF16,
            .num_elements = total_elements};
  }

  const std::size_t chunk_rows = std::min(
      rows, std::max<std::size_t>(
                1, kPackChunkBytes / (columns * sizeof(std::uint16_t))));
  std::vector<float> float_rows(chunk_rows * columns);
  std::vector<std::uint16_t> bf16_rows(chunk_rows * columns);
  for (std::size_t start = 0; start < rows; start += chunk_rows) {
    const std::size_t count = std::min(chunk_rows, rows - start);
    for (std::size_t local_row = 0; local_row < count; ++local_row) {
      ReadMatrixRow(source, start + local_row, columns,
                    float_rows.data() + (local_row * columns));
    }
    const std::size_t chunk_elements = count * columns;
    for (std::size_t index = 0; index < chunk_elements; ++index) {
      bf16_rows[index] = FloatToBfloat16Bits(float_rows[index]);
    }
    CopyToDevice(static_cast<std::uint8_t*>(device) +
                     (start * columns * sizeof(std::uint16_t)),
                 bf16_rows.data(), chunk_elements * sizeof(std::uint16_t));
  }

  packed_bytes += total_bytes;
  return {
      .data = device,
      .type = core::GgmlType::kBF16,
      .num_elements = total_elements,
  };
}

models::QwenTensorRef PackBlockMatrix(const models::QwenTensorRef& source,
                                      std::size_t rows, std::size_t columns,
                                      std::vector<void*>& allocations,
                                      std::size_t& packed_bytes) {
  // Keep the quantized formats accepted by the DFlash loader packed: draft
  // matrices are read on every speculation step.
  const bool native_block = source.type == core::GgmlType::kQ8_0 ||
                            source.type == core::GgmlType::kQ4_K ||
                            source.type == core::GgmlType::kQ6_K;
  // THIS LIST WAS NARROWER THAN ITS OWN CONSUMERS.
  //
  // Both draft GEMM routes accept `kQ8_0 || detail::IsNativeWmmaQuant(type)` --
  // RunBlockGemm (dflash/executor.cpp:452) and RunInjectGemm (:504) -- which is
  // also Q5_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ3_S, IQ2_XXS/XS/S, IQ3_XXS and
  // IQ1_M. The three names above are a subset of that, so a sidecar in any of
  // the other formats was expanded to BF16 on the device although the kernels
  // could have read it packed: ~3.85 GB instead of ~1.0 GB at the 1.9B draft
  // geometry, on the box where free RAM is the stall trigger.
  //
  // Widened under GUFO_DEV_DFLASH_KEEP_QUANT, beside the fc/K/V choice that
  // switch already makes, so the default path stays byte-for-byte what it was.
  //
  // AND ONLY WHEN THE ROW IS ADDRESSABLE. QuantizedRowBytes returns 0 when
  // `columns` is not a whole number of blocks. The legacy branch below treats
  // that as a hard refusal and returns an EMPTY ref -- a silently null weight.
  // That is pre-existing behaviour for the three legacy types and is left
  // alone; for a newly admitted type the same case falls back to BF16 instead,
  // because admitting a format must not turn a working load into a null
  // matrix.
  const bool widened_block =
      !native_block && DFlashKeepQuantEnabled() &&
      detail::IsNativeWmmaQuant(source.type) &&
      quant::QuantizedRowBytes(source.type, columns) != 0;
  if (!native_block && !widened_block) {
    return PackMatrixBf16(source, rows, columns, allocations, packed_bytes);
  }

  const std::size_t row_bytes = quant::QuantizedRowBytes(source.type, columns);
  if (source.empty() || rows == 0 || columns == 0 || row_bytes == 0) {
    return {};
  }
  const std::size_t total_bytes = rows * row_bytes;
  void* device = detail::AllocateDevice(total_bytes);
  allocations.push_back(device);
  CopyToDevice(device, source.data, total_bytes);
  packed_bytes += total_bytes;
  return {
      .data = device,
      .type = source.type,
      .num_elements = rows * columns,
  };
}

/// Builds the draft-private Q8_0 LM head. Halves the largest read in the draft
/// graph; cannot affect an emitted token because the verifier keeps the BF16
/// head.
[[nodiscard]] models::QwenTensorRef QuantizeDraftHead(
    const models::QwenTensorRef& source, std::vector<void*>& allocations,
    std::size_t& packed_bytes) {
  const std::size_t block_elements =
      quant::QuantizedBlockElements(core::GgmlType::kQ8_0);
  const std::size_t blocks = source.num_elements / block_elements;
  const std::size_t total_bytes =
      blocks * quant::QuantizedRowBytes(core::GgmlType::kQ8_0, block_elements);
  void* device = detail::AllocateDevice(total_bytes);
  allocations.push_back(device);
  kernels::LaunchDFlashQuantizeBf16ToQ8_0(source.data, device,
                                          source.num_elements, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  packed_bytes += total_bytes;
  return {
      .data = device,
      .type = core::GgmlType::kQ8_0,
      .num_elements = source.num_elements,
  };
}

void PackDFlashWeights(speculative::QwenDFlashWeights& weights,
                       std::vector<void*>& allocations,
                       std::size_t& packed_bytes) {
  const auto& cfg = weights.config;
  const auto& df_cfg = weights.dflash_config;
  const std::size_t hidden = cfg.hidden_size;
  const std::size_t intermediate = cfg.intermediate_size;
  const std::size_t attention = cfg.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const std::size_t enc_in_dim = df_cfg.target_layer_ids.size() * hidden;

  // Feature Fusion Encoder
  weights.fc_norm =
      CopyVectorF32(weights.fc_norm, hidden, allocations, packed_bytes);
  // Dev switch GUFO_DEV_DFLASH_KEEP_QUANT=1: keep fc and K/V in their
  // GGUF K-quant (RunBlockGemm/RunInjectGemm take the quant route) instead of
  // expanding them to BF16: -249 MiB device for the Q4_K_M companion.
  const auto pack_dense =
      DFlashKeepQuantEnabled() ? PackBlockMatrix : PackMatrixBf16;
  weights.fc_projection = pack_dense(weights.fc_projection, hidden, enc_in_dim,
                                     allocations, packed_bytes);

  weights.output_norm =
      CopyVectorF32(weights.output_norm, hidden, allocations, packed_bytes);

  // Candidate selector (DFlash-2 only; DSpark has none). Tested on the tensors
  // rather than on `is_dspark`, matching the executor's discriminator.
  if (!weights.selector_predecessor.empty()) {
    weights.selector_predecessor =
        PackMatrixBf16(weights.selector_predecessor, cfg.vocab_size,
                       df_cfg.selector_rank, allocations, packed_bytes);
    weights.selector_successor =
        PackMatrixBf16(weights.selector_successor, cfg.vocab_size,
                       df_cfg.selector_rank, allocations, packed_bytes);
    weights.selector_hidden =
        PackBlockMatrix(weights.selector_hidden, df_cfg.selector_rank, hidden,
                        allocations, packed_bytes);
  }

  // DSpark Markov head. Both matrices are stored [markov_rank, vocab] in GGUF
  // and read as one contiguous row of `rank` per token id, which is exactly
  // PackMatrixBf16's (rows = vocab, columns = rank) layout. Packed BF16 rather
  // than kept Q4_K: the bias is a rank-length product per drafted token, so
  // bandwidth is not the cost, and the BF16 row read is the form the kernel
  // and the CPU oracle can both address.
  if (!weights.markov_w1.empty()) {
    weights.markov_w1 = PackMatrixBf16(weights.markov_w1, cfg.vocab_size,
                                       df_cfg.markov_rank, allocations,
                                       packed_bytes);
    weights.markov_w2 = PackMatrixBf16(weights.markov_w2, cfg.vocab_size,
                                       df_cfg.markov_rank, allocations,
                                       packed_bytes);
  }
  // Confidence head: sigmoid(conf_proj . [hidden ; markov_w1[prev]] + b) over
  // hidden_size + markov_rank.
  if (!weights.conf_projection.empty()) {
    weights.conf_projection =
        CopyVectorF32(weights.conf_projection, hidden + df_cfg.markov_rank,
                      allocations, packed_bytes);
    weights.conf_bias =
        CopyVectorF32(weights.conf_bias, 1, allocations, packed_bytes);
  }

  // Draft Transformer Layers
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    auto& layer = weights.layers[i];
    auto& transformer = layer.transformer;
    transformer.attn_norm =
        CopyVectorF32(transformer.attn_norm, hidden, allocations, packed_bytes);
    transformer.attn_q = PackBlockMatrix(transformer.attn_q, attention, hidden,
                                         allocations, packed_bytes);
    // Prompt target-feature injection projects a full context through K/V
    // with the dense batched GEMM path, so these two matrices retain BF16
    // storage even when the draft artifact is Q8_0.
    transformer.attn_k = pack_dense(transformer.attn_k, kv, hidden,
                                    allocations, packed_bytes);
    transformer.attn_v = pack_dense(transformer.attn_v, kv, hidden,
                                    allocations, packed_bytes);
    transformer.attn_output = PackBlockMatrix(
        transformer.attn_output, hidden, attention, allocations, packed_bytes);
    transformer.attn_q_norm = CopyVectorF32(
        transformer.attn_q_norm, cfg.head_dim, allocations, packed_bytes);
    transformer.attn_k_norm = CopyVectorF32(
        transformer.attn_k_norm, cfg.head_dim, allocations, packed_bytes);
    transformer.ffn_norm =
        CopyVectorF32(transformer.ffn_norm, hidden, allocations, packed_bytes);
    transformer.ffn_gate = PackBlockMatrix(transformer.ffn_gate, intermediate,
                                           hidden, allocations, packed_bytes);
    transformer.ffn_up = PackBlockMatrix(transformer.ffn_up, intermediate,
                                         hidden, allocations, packed_bytes);
    transformer.ffn_down = PackBlockMatrix(
        transformer.ffn_down, hidden, intermediate, allocations, packed_bytes);

    // DSpark layers are plain transformer blocks: no dynamic convolution, so
    // conv_group_size is zero and must not be divided by.
    if (!layer.attention_conv_base.empty()) {
      layer.attention_conv_base = CopyVectorF32(
          layer.attention_conv_base, hidden * df_cfg.conv_kernel_size * 2U,
          allocations, packed_bytes);
      layer.ffn_conv_base = CopyVectorF32(layer.ffn_conv_base,
                                          hidden * df_cfg.conv_kernel_size * 2U,
                                          allocations, packed_bytes);
      const std::size_t dynamic_size =
          speculative::DFlashDynamicCoefficientCount(
              hidden, df_cfg.conv_kernel_size, df_cfg.conv_group_size);
      layer.attention_conv_projection =
          PackBlockMatrix(layer.attention_conv_projection, dynamic_size, hidden,
                          allocations, packed_bytes);
      layer.ffn_conv_projection =
          PackBlockMatrix(layer.ffn_conv_projection, dynamic_size, hidden,
                          allocations, packed_bytes);
    }
  }
}

/// GUFO_DEV_DRAFT_VOCAB: gathers the kept LM-head rows into one packed device
/// matrix of the same format, so RunBlockGemm takes its usual route at
/// output_size = cut.size. The target's head is a row-major GGUF matrix on the
/// device (or the draft-private Q8_0 copy above), so a row is a contiguous
/// byte range and a run of ids is one device-to-device copy.
QwenDFlashGpuModel::DraftVocabCut BuildDraftVocabCut(
    const models::QwenTensorRef& head, std::size_t vocab, std::size_t hidden,
    std::string_view spec, std::vector<void*>& allocations,
    std::size_t& packed_bytes) {
  std::string error;
  const auto ids = dflash::ParseDraftVocab(spec, vocab, error);
  if (ids.empty()) throw std::runtime_error(error);
  const std::size_t row_bytes =
      head.type == core::GgmlType::kBF16  ? hidden * sizeof(std::uint16_t)
      : head.type == core::GgmlType::kF32 ? hidden * sizeof(float)
                                          : quant::QuantizedRowBytes(head.type, hidden);
  if (head.empty() || row_bytes == 0 || head.num_elements != vocab * hidden)
    throw std::runtime_error("GUFO_DEV_DRAFT_VOCAB: LM head rows not addressable");
  void* rows = detail::AllocateDevice(ids.size() * row_bytes);
  allocations.push_back(rows);
  dflash::ForEachDraftVocabRun(
      ids, [&](std::size_t source_row, std::size_t destination_row,
               std::size_t count) {
        HIP_CHECK(hipMemcpy(
            static_cast<std::uint8_t*>(rows) + destination_row * row_bytes,
            static_cast<const std::uint8_t*>(head.data) + source_row * row_bytes,
            count * row_bytes, hipMemcpyDefault));
      });
  void* device_ids = detail::AllocateDevice(ids.size() * sizeof(std::uint32_t));
  allocations.push_back(device_ids);
  CopyToDevice(device_ids, ids.data(), ids.size() * sizeof(std::uint32_t));
  packed_bytes += ids.size() * (row_bytes + sizeof(std::uint32_t));
  std::fprintf(stderr, "GUFO_DEV_DRAFT_VOCAB rows=%zu of %zu head_bytes=%zu\n",
               ids.size(), vocab, ids.size() * row_bytes);
  return {
      .head = {.data = rows, .type = head.type,
               .num_elements = ids.size() * hidden},
      .d_ids = static_cast<const std::uint32_t*>(device_ids),
      .size = static_cast<std::uint32_t>(ids.size()),
  };
}

/// Drafter Markov bias, GUFO_DEV_DFLASH_MARKOV=<gguf>: bind ONLY markov_w1/markov_w2
/// from that file (today: the RedHat DSpark GGUF, or a count-built one from
/// tools/qwen27b/markov_from_counts.py) and pack them as the DFlash-2 selector's
/// rerank bias. GUFO_DEV_DFLASH_MARKOV_SCALE (default 1) multiplies it. Throws
/// on any mismatch: a wrong-vocabulary bias would silently rerank by garbage.
void LoadRerankMarkov(speculative::QwenDFlashWeights& weights,
                      std::vector<void*>& allocations,
                      std::size_t& packed_bytes) {
  const char* const path = std::getenv("GUFO_DEV_DFLASH_MARKOV");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  if (weights.dflash_config.is_dspark) {
    throw std::runtime_error(
        "GUFO_DEV_DFLASH_MARKOV applies to DFlash-2 drafts only; this draft "
        "already carries its own Markov head");
  }
  std::string error;
  // ponytail: the reader (and its page cache) is released when this returns;
  // only the two packed device copies stay.
  const auto reader = core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    throw std::runtime_error("GUFO_DEV_DFLASH_MARKOV: cannot open " +
                             std::string(path) + ": " + error);
  }
  const std::size_t vocab = weights.config.vocab_size;
  const auto bind = [&](std::string_view name) {
    const auto* tensor = reader->FindTensor(name);
    if (tensor == nullptr || tensor->data == nullptr ||
        tensor->dimensions.size() != 2 || tensor->dimensions[0] == 0 ||
        tensor->dimensions[1] != vocab) {
      throw std::runtime_error(
          "GUFO_DEV_DFLASH_MARKOV: " + std::string(name) +
          " must be a 2-D [markov_rank, vocab] tensor with vocab = " +
          std::to_string(vocab) + " (the draft's vocabulary)");
    }
    models::QwenTensorRef ref{.data = tensor->data,
                              .type = tensor->type,
                              .num_elements = tensor->ElementCount(),
                              .available_bytes = tensor->size_bytes};
    if (!ref.FitsAvailableStorage()) {
      throw std::runtime_error("GUFO_DEV_DFLASH_MARKOV: " + std::string(name) +
                               " is truncated");
    }
    return std::pair{ref, tensor->dimensions[0]};
  };
  const auto [w1, rank] = bind("markov_w1.weight");
  const auto [w2, rank2] = bind("markov_w2.weight");
  if (rank != rank2 || !std::in_range<std::uint32_t>(rank)) {
    throw std::runtime_error(
        "GUFO_DEV_DFLASH_MARKOV: markov_w2 rank differs from markov_w1");
  }
  float scale = 1.0F;
  if (const char* const value = std::getenv("GUFO_DEV_DFLASH_MARKOV_SCALE");
      value != nullptr && value[0] != '\0') {
    char* end = nullptr;
    scale = std::strtof(value, &end);
    // Refuse at load: "abc" would silently mean 0 (bias off), "inf" would pass
    // here and throw at the first draft, mid-serve.
    if (end == value || *end != '\0' || !std::isfinite(scale)) {
      throw std::runtime_error(
          "GUFO_DEV_DFLASH_MARKOV_SCALE is not a finite number");
    }
  }
  weights.rerank_markov_w1 =
      PackMatrixBf16(w1, vocab, rank, allocations, packed_bytes);
  weights.rerank_markov_w2 =
      PackMatrixBf16(w2, vocab, rank, allocations, packed_bytes);
  weights.rerank_markov_rank = static_cast<std::uint32_t>(rank);
  weights.rerank_markov_scale = scale;
  std::fprintf(stderr,
               "GUFO_DEV_DFLASH_MARKOV rank=%zu scale=%g device_bytes=%zu\n",
               static_cast<std::size_t>(rank), static_cast<double>(scale),
               2 * vocab * static_cast<std::size_t>(rank) *
                   sizeof(std::uint16_t));
  std::fflush(stderr);
}

}  // namespace

QwenDFlashGpuModel::QwenDFlashGpuModel(
    std::shared_ptr<const core::GgufReader> dflash_reader,
    std::shared_ptr<const QwenGpuModel> target_model,
    speculative::QwenDFlashWeights weights, std::vector<void*> allocations,
    std::size_t packed_weight_bytes, DraftVocabCut draft_vocab_cut)
    : dflash_reader_(std::move(dflash_reader)),
      target_model_(std::move(target_model)),
      weights_(std::move(weights)),
      allocations_(std::move(allocations)),
      packed_weight_bytes_(packed_weight_bytes),
      draft_vocab_cut_(draft_vocab_cut) {}

void QwenDFlashGpuModel::DropFilePages(const char* where) const noexcept {
  if (!DFlashDropPageCacheEnabled() || dflash_reader_ == nullptr)
    return;
  std::size_t before = 0, dropped = 0, after = 0;
  for (const auto& region : dflash_reader_->GetMappedRegions()) {
    before += core::ResidentMappedBytes(region.data, region.size);
    dropped += core::DropMappedRangeFromPageCache(region.data, region.size,
                                                  region.file_descriptor);
    after += core::ResidentMappedBytes(region.data, region.size);
  }
  // resident_* = page-cache residency (mincore), the measurement; dropped = the
  // span asked for, a receipt. A memory A/B needs all three apart.
  std::fprintf(stderr,
               "dflash page cache %s resident_before=%zu dropped=%zu "
               "resident_after=%zu packed=%zu bytes\n",
               where, before, dropped, after, packed_weight_bytes_);
  std::fflush(stderr);
}

QwenDFlashGpuModel::~QwenDFlashGpuModel() {
  for (void* pointer : allocations_) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
    }
  }
}

std::shared_ptr<const QwenDFlashGpuModel> QwenDFlashGpuModel::Create(
    std::shared_ptr<const core::GgufReader> dflash_reader,
    std::shared_ptr<const QwenGpuModel> target_model, std::string* error_msg) {
  if (dflash_reader == nullptr || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash reader and target GPU model are required";
    }
    return nullptr;
  }

  auto raw_weights =
      speculative::QwenDFlashWeights::LoadFromGguf(*dflash_reader, error_msg);
  if (!raw_weights.has_value()) {
    return nullptr;
  }

  // Before any weight packing and before any executor exists. The executor
  // divides hidden_size by conv_group_size (hip/dflash/executor.cpp Allocate
  // and ForwardBlock), which a DSpark artifact leaves at 0, so an artifact the
  // executor cannot size must be refused here. This is the funnel every CLI
  // route goes through (bench.cpp, prompt.cpp, inference_backend.cpp), so one
  // refusal here covers all of them and the message exists in exactly one
  // place: src/models/qwen/dflash_weights.cpp.
  {
    const std::string refusal =
        speculative::DFlashGpuExecutorSupportRefusal(*raw_weights);
    if (!refusal.empty()) {
      if (error_msg != nullptr) {
        *error_msg = refusal;
      }
      return nullptr;
    }
  }

  const auto& target_config = target_model->GetConfig();
  const auto& draft_config = raw_weights->config;
  if (target_config.hidden_size != draft_config.hidden_size ||
      target_config.vocab_size != draft_config.vocab_size) {
    if (error_msg != nullptr) {
      *error_msg =
          "DFlash-2 target hidden/vocabulary dimensions do not match the "
          "draft";
    }
    return nullptr;
  }
  if (raw_weights->dflash_config.target_layer_ids.empty() ||
      raw_weights->dflash_config.target_layer_ids.back() >=
          target_config.num_layers) {
    if (error_msg != nullptr) {
      *error_msg =
          "DFlash-2 target layer taps exceed the target model layer count";
    }
    return nullptr;
  }

  std::vector<void*> allocations;
  std::size_t packed_bytes = 0;
  try {
    speculative::QwenDFlashWeights weights = *raw_weights;
    for (const auto& region : dflash_reader->GetMappedRegions())
      core::PrefaultMappedRange(region.data, region.size);

    // Pack all private DFlash matrices and vectors into GPU device memory
    PackDFlashWeights(weights, allocations, packed_bytes);
    LoadRerankMarkov(weights, allocations, packed_bytes);

    // Unconditionally bind GPU device pointers from the target model for tied
    // weights
    weights.token_embedding = target_model->GetWeights().token_embd;
    weights.output = target_model->GetWeights().output;
    if (weights.output.type == core::GgmlType::kBF16 &&
        weights.output.num_elements %
                quant::QuantizedBlockElements(core::GgmlType::kQ8_0) ==
            0) {
      weights.output =
          QuantizeDraftHead(weights.output, allocations, packed_bytes);
    }
    DraftVocabCut draft_vocab_cut{};
    if (const char* const spec = std::getenv("GUFO_DEV_DRAFT_VOCAB");
        spec != nullptr && spec[0] != '\0') {
      // DSpark's Markov chain adds a full-vocabulary bias row before its
      // argmax; a cut head would leave that path scoring stale rows.
      if (speculative::IsDSparkDraft(weights))
        throw std::runtime_error("GUFO_DEV_DRAFT_VOCAB is DFlash-2 only");
      draft_vocab_cut = BuildDraftVocabCut(
          weights.output, target_config.vocab_size, target_config.hidden_size,
          spec, allocations, packed_bytes);
    }

    // EVERY tensor ref in `weights` now points at device memory: the
    // private matrices were rebound by PackDFlashWeights above, and the two
    // tied ones were just rebound to the target's own device pointers. So the
    // sidecar's ~1 GB of prefaulted file pages is dead weight from here on, and
    // on this box it is dead weight parked next to the target's file-backed
    // weight pages.
    //
    // The reader stays alive and the mapping stays mapped -- this only drops
    // the resident pages and the clean cache behind them. A ref this audit
    // missed would therefore re-fault from the file and still read the right
    // bytes; the cost of being wrong here is latency, never a wrong weight.
    // The executor drops again after its prewarm (DropFilePages "after_prewarm").
    std::shared_ptr<const QwenDFlashGpuModel> model(new QwenDFlashGpuModel(
        std::move(dflash_reader), std::move(target_model), std::move(weights),
        std::move(allocations), packed_bytes, draft_vocab_cut));
    model->DropFilePages("after_pack");
    return model;
  } catch (const std::exception& ex) {
    for (void* ptr : allocations) {
      if (ptr != nullptr)
        (void)hipFree(ptr);
    }
    if (error_msg != nullptr) {
      *error_msg = ex.what();
    }
    return nullptr;
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/hip/dev_timing.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/dev_sync.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kMaxDecodeBatch = 8;

// Dev switch: GUFO_DEV_TURBO4_VERIFY_WMMA=1 routes the multi-row Turbo4
// verify reader through the WMMA tile kernel (not bit-identical; default off).
[[nodiscard]] bool Turbo4VerifyWmmaEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_TURBO4_VERIFY_WMMA");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

// Dev switch: GUFO_DEV_BATCHED_VERIFY_ROPE=1 rotates a
// Turbo4 verify sequence's rows in one LaunchBatchedRoPE instead of one
// LaunchRoPE per row (16 layers x width launches -> 16). Both kernels compute
// the same expressions in the same order, and the rows sit at start_pos + row
// (host_positions is filled that way), so the output is bitwise identical.
[[nodiscard]] bool BatchedVerifyRopeEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_BATCHED_VERIFY_ROPE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

[[nodiscard]] constexpr bool SupportsExactSharedProjection(
    core::GgmlType type) noexcept {
  // These formats preserve scalar FP32 arithmetic while sharing weight loads.
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K ||
         detail::IsNativeWmmaQuant(type);
}

void LaunchProjection(const models::QwenTensorRef& weight, const float* input,
                      float* output, std::size_t batch_size,
                      std::size_t output_size, std::size_t input_size,
                      hipStream_t stream) {
  if (weight.layout != models::QwenWeightLayout::kNative)
    throw std::invalid_argument("tagged weight reached native projection");
  if (SupportsExactSharedProjection(weight.type)) {
    LaunchBatchedQuantGEMMFp32(weight.type, weight.data, input, output,
                               batch_size, output_size, input_size, stream);
    return;
  }
  if (weight.type == core::GgmlType::kBF16) {
    LaunchExactBf16GEMMFp32SmallBatch(weight.data, input, output, batch_size,
                                      output_size, input_size, stream);
    return;
  }

  // F32 ssm_alpha lands here; one row-grid launch instead of
  // batch_size LaunchGEMV calls (bitwise equal, see LaunchGEMVRows).
  LaunchGEMVRows(weight.data, weight.type, input, output, batch_size,
                 output_size, input_size, stream);
}

void LaunchFfnActivation(const models::QwenLayerWeights& layer,
                         const QwenGpuScratchView& scratch,
                         std::size_t batch_size, std::size_t intermediate_size,
                         std::size_t hidden_size, hipStream_t stream) {
  const auto& gate = layer.ffn_gate;
  const auto& up = layer.ffn_up;
  if (gate.layout != models::QwenWeightLayout::kNative ||
      up.layout != models::QwenWeightLayout::kNative) {
    if (batch_size < 1 || batch_size > 8 || intermediate_size != 17408 ||
        hidden_size != 5120)
      throw std::invalid_argument(
          "tagged IQ3 FFN width/profile not integrated");
    // The model-wide executor fence remains until the other readers land.
    // This branch never attempts a native adjacency/raw-pointer fallback.
    if (GufoDevIq3Packed()) {
      const auto bytes = 2 * batch_size * intermediate_size * sizeof(float);
      if (scratch.decode.weight_bf16.size_bytes() < bytes)
        throw std::runtime_error(
            "tagged IQ3 packed activation scratch too small");
      auto* const packed =
          reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
      LaunchSnIq3Pair(gate, up, scratch.decode.normed.data(), nullptr, nullptr,
                      packed, batch_size, stream);
      LaunchPackedSwiGLUActivation(packed, scratch.ffn.activation.data(),
                                   batch_size, intermediate_size, stream);
    } else {
      LaunchSnIq3Pair(gate, up, scratch.decode.normed.data(),
                      scratch.ffn.gate.data(), scratch.ffn.up.data(), nullptr,
                      batch_size, stream);
      LaunchBatchedSwiGLUActivation(scratch.ffn.gate.data(),
                                    scratch.ffn.up.data(),
                                    scratch.ffn.activation.data(), nullptr,
                                    batch_size * intermediate_size, stream);
    }
    return;
  }
  const bool packed_format = gate.type == core::GgmlType::kQ4_K ||
                             gate.type == core::GgmlType::kQ5_K ||
                             gate.type == core::GgmlType::kQ6_K ||
                             gate.type == core::GgmlType::kIQ4_XS ||
                             gate.type == core::GgmlType::kQ8_0 ||
                             (GufoDevIq3Packed() &&
                              (gate.type == core::GgmlType::kIQ3_S ||
                               gate.type == core::GgmlType::kIQ3_XXS));
  if (batch_size >= 2 && intermediate_size == 17408 && hidden_size == 5120 &&
      packed_format && gate.type == up.type &&
      gate.num_elements == intermediate_size * hidden_size &&
      up.num_elements == gate.num_elements) {
    const std::size_t matrix_bytes = gate.EncodedSizeBytes();
    const std::size_t packed_elements = 2 * batch_size * intermediate_size;
    if (gate.available_bytes >= 2 * matrix_bytes &&
        static_cast<const std::byte*>(gate.data) + matrix_bytes == up.data) {
      if (TryLaunchPackedQuantSwiGLUFp32(
              gate.type, gate.data, scratch.decode.normed.data(),
              scratch.ffn.activation.data(), batch_size, intermediate_size,
              hidden_size, stream))
        return;
      if (scratch.decode.weight_bf16.size_bytes() >=
          packed_elements * sizeof(float)) {
        // Adjacent GGUF tensors form one matrix with twice as many output rows.
        // Exact projections do not need dequantization scratch; reuse it until
        // SwiGLU has consumed the packed gate/up rows on this stream.
        auto* const packed =
            reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
        LaunchProjection(gate, scratch.decode.normed.data(), packed, batch_size,
                         2 * intermediate_size, hidden_size, stream);
        LaunchPackedSwiGLUActivation(packed, scratch.ffn.activation.data(),
                                     batch_size, intermediate_size, stream);
        return;
      }
    }
  }
  LaunchProjection(gate, scratch.decode.normed.data(), scratch.ffn.gate.data(),
                   batch_size, intermediate_size, hidden_size, stream);
  LaunchProjection(up, scratch.decode.normed.data(), scratch.ffn.up.data(),
                   batch_size, intermediate_size, hidden_size, stream);
  LaunchBatchedSwiGLUActivation(scratch.ffn.gate.data(), scratch.ffn.up.data(),
                                scratch.ffn.activation.data(), nullptr,
                                batch_size * intermediate_size, stream);
}

struct SsmControls {
  const float* alpha;
  const float* beta;
  std::size_t row_stride;
};

SsmControls LaunchSsmControls(const models::QwenLayerWeights& layer,
                              const QwenGpuScratchView& scratch,
                              std::size_t batch_size,
                              std::size_t time_step_rank,
                              std::size_t hidden_size, hipStream_t stream) {
  const auto& alpha = layer.ssm_alpha;
  const auto& beta = layer.ssm_beta;
  if (time_step_rank == 48 && hidden_size == 5120 &&
      alpha.type == core::GgmlType::kQ8_0 && beta.type == alpha.type &&
      alpha.num_elements == time_step_rank * hidden_size &&
      beta.num_elements == alpha.num_elements) {
    const std::size_t matrix_bytes = alpha.EncodedSizeBytes();
    const std::size_t row_stride = 2 * time_step_rank;
    if (alpha.available_bytes >= 2 * matrix_bytes &&
        static_cast<const std::byte*>(alpha.data) + matrix_bytes == beta.data &&
        scratch.decode.weight_bf16.size_bytes() >=
            batch_size * row_stride * sizeof(float)) {
      // Adjacent control matrices share one launch. Recurrence already accepts
      // a row stride, so it can consume the interleaved output directly.
      auto* const packed =
          reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
      LaunchProjection(alpha, scratch.decode.normed.data(), packed, batch_size,
                       row_stride, hidden_size, stream);
      return {packed, packed + time_step_rank, row_stride};
    }
  }
  LaunchProjection(alpha, scratch.decode.normed.data(),
                   scratch.ssm.alpha.data(), batch_size, time_step_rank,
                   hidden_size, stream);
  LaunchProjection(beta, scratch.decode.normed.data(), scratch.ssm.beta.data(),
                   batch_size, time_step_rank, hidden_size, stream);
  return {scratch.ssm.alpha.data(), scratch.ssm.beta.data(), time_step_rank};
}

}  // namespace

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardTokenBatch(
    std::span<const QwenGpuBatchItem> items) {
  // NOT kDecode, and this is a safety decision, not an oversight (C1-R1).
  // ForwardToken is kDecode because ExecuteDecodeStep carries the block-packed
  // route (LaunchTurbo4KvWrite then LaunchTurbo4AttentionScalar). The batched
  // route below has NO such branch: it writes K/V through LaunchBatched*
  // kernels into scratch.attention.k/v and never addresses d_kv_turbo4_*.
  // Annotating it would let a Turbo4 policy run batched attention against the
  // absent f16 plane members -- the confident garbage the refusal exists to
  // prevent.
  //
  // E6.0: kBatchedToken is named EXPLICITLY and that is the fix. This entry
  // used to take the default, and the default used to refuse. Unit E gave
  // kPrefill its own GUFO_DEV_TURBO4_PREFILL admission arm, which silently
  // turned this entry into an admitted one: with the switch set, as every T arm
  // sets it, this guard returned without throwing and the f16 launches below
  // ran against d_kv_cache and d_attention_kv_f16, both nullptr on a Turbo4
  // session. The refusal it was written to provide had been reopened three
  // units away. Naming the phase keeps the grep honest: this entry is now
  // findable, and a future phase that does earn a Turbo4 route is a visible
  // edit here.
  for (const auto& item : items)
    if (item.executor)
      item.executor->CheckReset(QwenKvExecutionPhase::kBatchedToken);
  if (items.size() < 2 || items.size() > kMaxDecodeBatch) {
    throw std::invalid_argument(
        "Qwen GPU decode batch requires two to eight sessions");
  }

  QwenGpuExecutor* coordinator = items.front().executor;
  if (coordinator == nullptr) {
    throw std::invalid_argument(
        "Qwen GPU decode batch contains a null session");
  }
  const auto* shared_model = coordinator->model_.get();
  const std::uint64_t shared_policy = coordinator->policy_.Fingerprint();
  for (std::size_t index = 0; index < items.size(); ++index) {
    QwenGpuExecutor* executor = items[index].executor;
    if (executor == nullptr || executor->model_.get() != shared_model ||
        executor->policy_.Fingerprint() != shared_policy) {
      throw std::invalid_argument(
          "Qwen GPU decode batch sessions are not execution-compatible");
    }
    if (items[index].position >= executor->arena_.GetMaxContext()) {
      throw std::length_error(
          "token position exceeds a batched Qwen session context");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (items[previous].executor == executor) {
        throw std::invalid_argument(
            "Qwen GPU decode batch contains a duplicate session");
      }
    }
  }
  if (items.size() > coordinator->arena_.GetMaxBatch()) {
    throw std::length_error(
        "Qwen GPU decode width exceeds the coordinator arena capacity");
  }

  bool capture_replay = false;
  for (const auto& item : items) {
    auto& executor = *item.executor;
    if (executor.replaying_ssm_state_) {
      executor.replaying_ssm_state_ = false;
      executor.arena_.DisableSsmReplayCapture();
    }
    // Replay and capture-flag updates may still be queued on a session's
    // stream. The coordinator takes ownership only after they finish.
    if (&executor != coordinator) {
      HIP_CHECK(hipStreamSynchronize(executor.arena_.stream));
    }
    capture_replay |= executor.arena_.IsSsmReplayCaptureActive();
    executor.arena_.MarkSsmReplayPosition(item.position);
    executor.last_hidden_offset_ = 0;
  }

  const std::size_t batch_size = items.size();
  const auto& weights = coordinator->weights_;
  const auto& config = weights.config;
  auto& arena = coordinator->arena_;
  const auto scratch = arena.GetScratchView(batch_size);
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;

  std::array<std::uint32_t, kMaxDecodeBatch> host_tokens{};
  std::array<std::uint32_t, kMaxDecodeBatch> host_positions{};
  std::array<SsmSequenceState, kMaxDecodeBatch> ssm_sequences{};
  for (std::size_t row = 0; row < batch_size; ++row) {
    host_tokens[row] = items[row].token_id;
    host_positions[row] = items[row].position;
    auto& state_arena = items[row].executor->arena_;
    auto replay = state_arena.GetSsmReplayCapture();
    replay.position = scratch.decode.prompt_tokens.data() + row;
    ssm_sequences[row] = {state_arena.BeginRecurrentWrite(), replay,
                          static_cast<std::uint32_t>(row), 1};
  }
  HIP_CHECK(hipMemcpyAsync(
      scratch.decode.prompt_tokens.data(), host_tokens.data(),
      batch_size * sizeof(std::uint32_t), hipMemcpyHostToDevice, arena.stream));
  coordinator->batch_embedding_.LaunchBatch(
      coordinator->model_->GetEmbeddingRows(), weights.token_embd,
      std::span(host_tokens).first(batch_size),
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      hidden_size, arena.stream);
  for (std::size_t row = 0; row < items.size(); ++row) {
    items[row].executor->vision_input_.Inject(
        scratch.decode.hidden.data() + row * hidden_size, items[row].position,
        1, hidden_size, 1, arena.stream);
  }
  if (capture_replay) {
    // Embedding has consumed the token IDs. Reuse the buffer for the actual
    // per-session positions until every recurrent layer has captured its row.
    HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                             host_positions.data(),
                             batch_size * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, arena.stream));
  }

  EmitDecodeRouteTelemetry(weights, coordinator->policy_, "batched_token");
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.attn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena.stream);

    if (layer.is_full_attention) {
      LaunchProjection(layer.attn_q, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, q_projection_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.attn_k, scratch.decode.normed.data(),
                       scratch.attention.k.data(), batch_size, kv_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.attn_v, scratch.decode.normed.data(),
                       scratch.attention.v.data(), batch_size, kv_size,
                       hidden_size, arena.stream);
      LaunchBatchedUnpackQG(scratch.ssm.qkv.data(), scratch.attention.q.data(),
                            scratch.ssm.gate.data(), batch_size,
                            config.num_attention_heads, config.head_dim,
                            arena.stream);

      if (!layer.attn_q_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            scratch.attention.q.data(),
            static_cast<const float*>(layer.attn_q_norm.data),
            scratch.attention.q.data(), batch_size, config.num_attention_heads,
            config.head_dim, 1e-6F, arena.stream);
      }
      if (!layer.attn_k_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            scratch.attention.k.data(),
            static_cast<const float*>(layer.attn_k_norm.data),
            scratch.attention.k.data(), batch_size, config.num_key_value_heads,
            config.head_dim, 1e-6F, arena.stream);
      }

      const std::uint32_t attention_layer =
          layer_index / config.full_attention_interval;
      for (std::size_t row = 0; row < batch_size; ++row) {
        auto& state_arena = items[row].executor->arena_;
        const std::size_t total_k = state_arena.GetAttentionKvPlaneElements();
        float* query = scratch.attention.q.data() + (row * attention_size);
        float* key = scratch.attention.k.data() + (row * kv_size);
        float* value = scratch.attention.v.data() + (row * kv_size);
        float* gate = scratch.ssm.gate.data() + (row * attention_size);
        float* context = scratch.ssm.out.data() + (row * attention_size);
        const std::uint32_t position = items[row].position;

        LaunchRoPE(query, key, config.num_attention_heads,
                   config.num_key_value_heads, config.head_dim,
                   config.rotary_dim, position, config.rope_theta, arena.stream,
                   items[row].executor->vision_input_.rope());
        const bool use_split_k = detail::IsSplitKDecodeAttentionSupported(
            static_cast<std::size_t>(position) + 1, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim);
        LaunchAttention(
            query, key, value, gate, state_arena.d_kv_cache,
            OffsetIfPresent(state_arena.d_kv_cache, total_k),
            state_arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                total_k),
            context, attention_layer, position, state_arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream,
            use_split_k ? arena.d_split_k_attention : nullptr);
      }
      LaunchProjection(layer.attn_output, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       attention_size, arena.stream);
    } else {
      LaunchProjection(layer.attn_qkv, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, ssm_qkv_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.attn_gate, scratch.decode.normed.data(),
                       scratch.ssm.gate.data(), batch_size, ssm_inner_size,
                       hidden_size, arena.stream);
      const auto controls =
          LaunchSsmControls(layer, scratch, batch_size, time_step_rank,
                            hidden_size, arena.stream);

      LaunchSSMConvRecurrenceBatch(
          scratch.ssm.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data),
          scratch.ssm.conv_out.data(), controls.alpha, controls.beta,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          scratch.ssm.gate.data(), scratch.ssm.out.data(),
          std::span(ssm_sequences).first(batch_size),
          config.SsmLayerIndex(layer_index), ssm_qkv_size,
          config.ssm_group_count, config.ssm_time_step_rank,
          config.ssm_state_size, config.SsmValueSize(), controls.row_stride,
          ssm_inner_size, arena.stream, arena.GetRecurrentStateStorage());

      LaunchProjection(layer.ssm_out, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       ssm_inner_size, arena.stream);
    }

    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena.stream);
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.ffn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena.stream);

    LaunchFfnActivation(layer, scratch, batch_size, intermediate_size,
                        hidden_size, arena.stream);
    LaunchProjection(layer.ffn_down, scratch.ffn.activation.data(),
                     scratch.ffn.out.data(), batch_size, hidden_size,
                     intermediate_size, arena.stream);
    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.ffn.out.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena.stream);

    for (std::size_t row = 0; row < batch_size; ++row) {
      auto& state_arena = items[row].executor->arena_;
      if (const auto tap = state_arena.GetTargetLayerCaptureIndex(layer_index);
          tap.has_value()) {
        HIP_CHECK(hipMemcpyAsync(
            state_arena.d_target_layer_features + (*tap * hidden_size),
            scratch.decode.hidden.data() + (row * hidden_size),
            hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
            arena.stream));
      }
    }
  }

  LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                       static_cast<const float*>(weights.output_norm.data),
                       scratch.decode.normed.data(), nullptr, batch_size,
                       hidden_size, 1e-6F, arena.stream);
  coordinator->EnsureVerificationLogits(batch_size);
  LaunchProjection(weights.output, scratch.decode.normed.data(),
                   coordinator->d_verification_logits_, batch_size, vocab_size,
                   hidden_size, arena.stream);
  LaunchBatchedGPUArgmax(coordinator->d_verification_logits_,
                         scratch.decode.prompt_tokens.data(), batch_size,
                         vocab_size, scratch.ffn.out, arena.stream);

  std::array<std::uint32_t, kMaxDecodeBatch> host_frontiers{};
  HIP_CHECK(hipMemcpyAsync(
      host_frontiers.data(), scratch.decode.prompt_tokens.data(),
      batch_size * sizeof(std::uint32_t), hipMemcpyDeviceToHost, arena.stream));
  for (std::size_t row = 0; row < batch_size; ++row) {
    auto& state_arena = items[row].executor->arena_;
    if (&state_arena != &arena || row != 0) {
      HIP_CHECK(hipMemcpyAsync(
          state_arena.d_hidden,
          scratch.decode.hidden.data() + (row * hidden_size),
          hidden_size * sizeof(float), hipMemcpyDeviceToDevice, arena.stream));
    }
    HIP_CHECK(hipMemcpyAsync(
        state_arena.d_logits,
        coordinator->d_verification_logits_ + (row * vocab_size),
        vocab_size * sizeof(float), hipMemcpyDeviceToDevice, arena.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  for (std::size_t row = 0; row < batch_size; ++row)
    (void)CheckedSampleToken(host_frontiers[row], vocab_size);
  return {host_frontiers.begin(),
          host_frontiers.begin() + static_cast<std::ptrdiff_t>(batch_size)};
}

std::vector<tokenization::TokenId>
QwenGpuExecutor::ForwardDecodeEquivalentVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  if (candidate_tokens.empty()) {
    h_verification_hidden_.clear();
    h_verification_logits_.clear();
    last_verification_rows_ = 0;
    last_verification_features_ = nullptr;
    last_verification_feature_width_ = 0;
    return {};
  }
  const QwenGpuVerificationItem item{this, candidate_tokens, start_pos,
                                     capture_logits};
  auto predictions = ForwardVerificationBatch(std::span(&item, 1));
  return std::move(predictions.front());
}

std::vector<std::vector<tokenization::TokenId>>
QwenGpuExecutor::ForwardVerificationBatch(
    std::span<const QwenGpuVerificationItem> items) {
  // E6.0 gave verification its own phase so PREFILL can never admit this
  // batched entry. E6.2 admits only this single-session verify route under the
  // DECODE switch; ForwardTokenBatch remains refused on Turbo4.
  for (const auto& item : items)
    if (item.executor)
      item.executor->CheckReset(QwenKvExecutionPhase::kVerify);
  if (items.empty() || items.size() > kMaxDecodeBatch ||
      items.front().executor == nullptr) {
    throw std::invalid_argument(
        "Qwen verification requires one to eight sessions");
  }
  auto& coordinator = *items.front().executor;
  auto& arena_ = coordinator.arena_;
  const auto& weights_ = coordinator.weights_;
  const auto& config = weights_.config;
  const bool turbo4_verify =
      QwenKvIsBlockAddressed(coordinator.policy_.kv_cache_storage);
  if (turbo4_verify && items.size() != 1U) {
    throw std::invalid_argument(
        "Qwen Turbo4 verification requires one session per verify chunk");
  }
  if (turbo4_verify &&
      (coordinator.arena_.d_kv_turbo4_write_status == nullptr ||
       coordinator.arena_.d_kv_turbo4_read_status == nullptr)) {
    throw std::logic_error(
        "Qwen Turbo4 verification: status words were not allocated");
  }
  const bool capture_prompt_hidden_ = coordinator.capture_prompt_hidden_;
  const auto policy = coordinator.policy_.Fingerprint();
  constexpr std::size_t kMaxRows = kMaxDecodeBatch * kMaxDecodeBatch;
  std::array<std::size_t, kMaxDecodeBatch> offsets{};
  std::size_t batch_size = 0;
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    const auto* executor = item.executor;
    if (executor == nullptr || executor->model_ != coordinator.model_ ||
        executor->policy_.Fingerprint() != policy ||
        executor->capture_prompt_hidden_ != capture_prompt_hidden_ ||
        !std::ranges::equal(executor->arena_.GetTargetLayerCapture(),
                            arena_.GetTargetLayerCapture()) ||
        item.tokens.empty() || item.tokens.size() > kMaxDecodeBatch) {
      throw std::invalid_argument(
          "Qwen verification has incompatible sessions or token counts");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (items[previous].executor == executor)
        throw std::invalid_argument("Qwen verification repeats a session");
    }
    if (item.position > executor->arena_.GetMaxContext() ||
        item.tokens.size() > executor->arena_.GetMaxContext() - item.position) {
      throw std::length_error("Qwen verification exceeds a session context");
    }
    offsets[index] = batch_size;
    batch_size += item.tokens.size();
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error(
        "Qwen verification exceeds coordinator scratch capacity");
  }
  for (const auto& item : items) {
    auto& executor = *item.executor;
    if (&executor != &coordinator)
      HIP_CHECK(hipStreamSynchronize(executor.arena_.stream));
    executor.replaying_ssm_state_ = false;
    executor.last_verification_rows_ = item.tokens.size();
    executor.h_verification_hidden_.clear();
    executor.h_verification_logits_.clear();
    executor.last_verification_features_ = nullptr;
    executor.last_verification_feature_width_ = 0;
  }
  const auto scratch = arena_.GetScratchView(batch_size);
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const std::size_t captured_layer_count =
      arena_.GetTargetLayerCapture().size();
  const std::size_t feature_width = captured_layer_count * hidden_size;
  const std::size_t feature_elements =
      capture_prompt_hidden_ ? batch_size * feature_width : 0;
  const std::size_t feature_rows =
      feature_elements / vocab_size + (feature_elements % vocab_size != 0);
  // The last FFN releases this scratch before the vocabulary projection.
  // Reuse it for the combined rows; each session retains only its own logits.
  const auto full_gate = arena_.GetScratchView(arena_.GetMaxBatch()).ffn.gate;
  const bool reuse_gate =
      items.size() > 1 && full_gate.size() >= batch_size * vocab_size;
  coordinator.EnsureVerificationLogits(std::max(
      reuse_gate ? items.front().tokens.size() : batch_size, feature_rows));
  // GUFO_DEV_HOSTOPS_FEAT: the drafter consumes the captured feature
  // rows AFTER the host has seen the predictions, by which time the output
  // projection has overwritten the shared d_verification_logits_. The handoff
  // route therefore captures into a dedicated buffer that outlives that GEMM.
  // One-shot: the request is consumed here, so a later path that captures
  // hidden states (prompt priming, prefix extension) cannot inherit it and
  // find a device pointer where it expects host rows.
  // All or nothing across the batch. A mixed batch would hand device rows to a
  // session whose step expects host rows (the target-only route reads them on
  // the host), and that session would fail on an empty span.
  bool all_handoff = true;
  for (const auto& item : items)
    all_handoff = all_handoff && item.executor->device_feature_handoff_;
  const bool device_features = GufoDevHostopsFeatures() &&
                               capture_prompt_hidden_ &&
                               captured_layer_count > 0 && all_handoff;
  for (const auto& item : items)
    item.executor->device_feature_handoff_ = false;
  if (device_features)
    coordinator.EnsureVerificationFeatures(batch_size * feature_width);
  // The per-round byte counts this change claims to remove, printed
  // once so the gain is read off a run rather than taken from an estimate.
  if (GufoDevHostops() || GufoDevHostopsFeatures() ||
      GufoDevHostopsFrontier()) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(stderr,
                   "GUFO_DEV_HOSTOPS rows=%zu feature_width=%zu vocab=%zu "
                   "feature_bytes=%zu frontier_bytes=%zu device_features=%d\n",
                   batch_size, feature_width, vocab_size,
                   batch_size * feature_width * sizeof(float),
                   vocab_size * sizeof(float), device_features ? 1 : 0);
    }
  }
  auto* const feature_buffer = device_features
                                   ? coordinator.d_verification_features_
                                   : coordinator.d_verification_logits_;
  auto* const logits_buffer =
      reuse_gate ? full_gate.data() : coordinator.d_verification_logits_;
  for (const auto& item : items) {
    if (item.executor != &coordinator)
      item.executor->EnsureVerificationLogits(item.tokens.size());
    if (capture_prompt_hidden_ && !device_features)
      item.executor->h_verification_hidden_.resize(
          item.tokens.size() * (feature_width ? feature_width : hidden_size));
  }
  std::array<std::uint32_t, kMaxRows> host_tokens{};
  std::array<std::uint32_t, kMaxRows> host_positions{};
  std::array<SsmSequenceState, kMaxDecodeBatch> ssm_sequences{};
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& state_arena = item.executor->arena_;
    auto replay = state_arena.GetSsmReplayCapture();
    replay.position = scratch.decode.prompt_tokens.data() + offsets[index];
    ssm_sequences[index] = {state_arena.BeginRecurrentWrite(), replay,
                            static_cast<std::uint32_t>(offsets[index]),
                            static_cast<std::uint32_t>(item.tokens.size())};
    for (std::size_t row = 0; row < item.tokens.size(); ++row) {
      const auto position = item.position + static_cast<std::uint32_t>(row);
      host_tokens[offsets[index] + row] = item.tokens[row];
      host_positions[offsets[index] + row] = position;
      item.executor->arena_.MarkSsmReplayPosition(position);
    }
  }
  ::gufo::core::hip::dev_timing::EventScope verify_timer(
      ::gufo::core::hip::dev_timing::Phase::kVerify, arena_.stream);
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  coordinator.batch_embedding_.LaunchBatch(
      coordinator.model_->GetEmbeddingRows(), weights_.token_embd,
      std::span(host_tokens).first(batch_size),
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      hidden_size, arena_.stream);
  for (std::size_t index = 0; index < items.size(); ++index) {
    items[index].executor->vision_input_.Inject(
        scratch.decode.hidden.data() + offsets[index] * hidden_size,
        items[index].position, items[index].tokens.size(), hidden_size, 1,
        arena_.stream);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_positions.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  EmitDecodeRouteTelemetry(weights_, coordinator.policy_, "verify");
  // Dev switch GUFO_DEV_FUSE_ADDNORM=1 (shared with single-token
  // decode): each residual add launches with the RMSNorm that reads it.
  static const bool kFuseAddNormEnv = [] {
    const char* v = std::getenv("GUFO_DEV_FUSE_ADDNORM");
    const bool on = v != nullptr && v[0] == '1' && v[1] == '\0';
    if (on)
      std::fprintf(stderr, "GUFO_DEV_FUSE_ADDNORM verify=on\n");
    return on;
  }();
  const bool fuse_add_norm =
      kFuseAddNormEnv && hidden_size == 5120 && batch_size <= 64;
  const auto add_norm = [&](const float* update, const void* weight) {
    (void)TryLaunchBatchedAddRMSNorm(scratch.decode.hidden.data(), update,
                                     static_cast<const float*>(weight),
                                     scratch.decode.normed.data(), batch_size,
                                     hidden_size, 1e-6F, arena_.stream);
  };
  // Each session's Turbo4 start-position word holds item.position
  // for the whole round (only the KV writer reads it, nothing in the round
  // writes it, sessions are distinct), so stage it once per round, not once
  // per attention layer. Precedent: prefill_chunk.cpp turbo4_first_pos_staged.
  std::array<bool, kMaxDecodeBatch> turbo4_first_pos_staged{};
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights_.layers[layer_index];
    const auto route_plan =
        ResolveQwenLayerRoute(coordinator.policy_, QwenExecutionMode::kDecode,
                              layer.is_full_attention);

    // Fused: the previous layer's last residual add already wrote normed.
    if (!(fuse_add_norm && layer_index > 0))
      LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                           static_cast<const float*>(layer.attn_norm.data),
                           scratch.decode.normed.data(), nullptr, batch_size,
                           hidden_size, 1e-6F, arena_.stream);

    if (layer.is_full_attention) {
      LaunchProjection(layer.attn_q, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, q_projection_size,
                       hidden_size, arena_.stream);
      LaunchProjection(layer.attn_k, scratch.decode.normed.data(),
                       scratch.attention.k.data(), batch_size, kv_size,
                       hidden_size, arena_.stream);
      LaunchProjection(layer.attn_v, scratch.decode.normed.data(),
                       scratch.attention.v.data(), batch_size, kv_size,
                       hidden_size, arena_.stream);
      LaunchBatchedUnpackQG(scratch.ssm.qkv.data(), scratch.attention.q.data(),
                            scratch.ssm.gate.data(), batch_size,
                            config.num_attention_heads, config.head_dim,
                            arena_.stream);

      const std::uint32_t attention_layer =
          layer_index / config.full_attention_interval;
      const bool fused_qknorm_rope_kv =
          !turbo4_verify && route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (!fused_qknorm_rope_kv) {
        if (!layer.attn_q_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              scratch.attention.q.data(),
              static_cast<const float*>(layer.attn_q_norm.data),
              scratch.attention.q.data(), batch_size,
              config.num_attention_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              scratch.attention.k.data(),
              static_cast<const float*>(layer.attn_k_norm.data),
              scratch.attention.k.data(), batch_size,
              config.num_key_value_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }
      }

      for (std::size_t index = 0; index < items.size(); ++index) {
        const auto& item = items[index];
        auto& state_arena = item.executor->arena_;
        const std::size_t offset = offsets[index];
        const std::size_t sequence_size = item.tokens.size();
        const std::uint32_t start_pos = item.position;
        const std::size_t total_k = state_arena.GetAttentionKvPlaneElements();
        if (turbo4_verify) {
          const std::size_t arena_bytes = state_arena.Turbo4KvBytesRecorded();
          Turbo4LayerSlices slices{};
          // Block width (66/82/98 B) follows the storage.
          const turbo4::Bits kv_bits =
              QwenKvTurboBits(coordinator.policy_.kv_cache_storage);
          const auto kv_formats =
              QwenKvTurboLayerFormats(coordinator.policy_, attention_layer);
          if (!Turbo4LayerPlaneSlices(
                  arena_bytes, /*base_bytes=*/0, config.head_dim,
                  config.num_key_value_heads, state_arena.GetMaxContext(),
                  coordinator.policy_.kv_layers.layer_count, attention_layer,
                  &slices, kv_bits)) {
            throw std::logic_error(
                "Qwen Turbo4 verification: layer slice does not fit the arena");
          }
          const std::size_t partial_bytes =
              state_arena.Turbo4DecodePartialBytesRecorded();
          const std::size_t workspace_bytes =
              state_arena.Turbo4DecodeWorkspaceBytesRecorded();
          if (state_arena.d_turbo4_decode_workspace == nullptr ||
              partial_bytes == 0U || partial_bytes > workspace_bytes) {
            throw std::logic_error(
                "Qwen Turbo4 verification: split-K workspace was not "
                "allocated");
          }
          auto* workspace =
              static_cast<float*>(state_arena.d_turbo4_decode_workspace);
          const std::size_t vector_values = attention_size;
          float* const q_rot = workspace + partial_bytes / sizeof(float);
          float* const out_rot = q_rot + vector_values;
          const std::size_t q_bytes = vector_values * sizeof(float);
          const std::size_t gate_bytes = q_bytes;
          const std::size_t kv_source_bytes =
              sequence_size * kv_size * sizeof(float);
          if (state_arena.d_kv_turbo4 == nullptr ||
              state_arena.d_turbo4_first_pos == nullptr ||
              state_arena.d_kv_turbo4_write_status == nullptr ||
              state_arena.d_kv_turbo4_read_status == nullptr ||
              sequence_size > std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error(
                "Qwen Turbo4 verification: KV arena metadata is incomplete");
          }
          if (!turbo4_first_pos_staged[index]) {
            HIP_CHECK(hipMemsetD32Async(state_arena.d_turbo4_first_pos,
                                        start_pos, 1, arena_.stream));
            turbo4_first_pos_staged[index] = true;
          }
          if (BatchedVerifyRopeEnabled() &&
              attention_size == static_cast<std::size_t>(
                                    config.num_attention_heads) *
                                    config.head_dim) {
            LaunchBatchedRoPE(
                scratch.attention.q.data() + offset * attention_size,
                scratch.attention.k.data() + offset * kv_size, sequence_size,
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, config.rotary_dim, start_pos,
                config.rope_theta, arena_.stream,
                item.executor->vision_input_.rope());
          } else {
            for (std::size_t row = 0; row < sequence_size; ++row) {
              float* const query =
                  (scratch.attention.q.data() + offset * attention_size) +
                  row * attention_size;
              float* const key =
                  (scratch.attention.k.data() + offset * kv_size) + row * kv_size;
              LaunchRoPE(query, key, config.num_attention_heads,
                         config.num_key_value_heads, config.head_dim,
                         config.rotary_dim, host_positions[offset + row],
                         config.rope_theta, arena_.stream,
                         item.executor->vision_input_.rope());
            }
          }
          const auto write_status = LaunchTurbo4KvWriteBatched(
              scratch.attention.k.data() + offset * kv_size,
              scratch.attention.v.data() + offset * kv_size,
              static_cast<std::uint8_t*>(state_arena.d_kv_turbo4) +
                  slices.k_offset,
              static_cast<std::uint8_t*>(state_arena.d_kv_turbo4) +
                  slices.v_offset,
              state_arena.d_turbo4_first_pos,
              static_cast<std::uint32_t>(sequence_size), slices.plane_bytes,
              kv_source_bytes, state_arena.GetMaxContext(),
              config.num_key_value_heads, config.head_dim,
              static_cast<Turbo4KernelStatus*>(
                  state_arena.d_kv_turbo4_write_status),
              arena_.stream, kv_formats);
          if (write_status != Turbo4LaunchStatus::kLaunched) {
            throw std::logic_error(
                "Qwen Turbo4 verification: batched KV writer refused to "
                "launch");
          }
          // One launch per stage for all rows. Exact verification
          // projections do not use weight dequantization scratch, so (as the
          // fp16 path below does) it holds the per-row partials and rotated
          // vectors. Too small, or a single row: per-row loop on the arena
          // workspace.
          const std::size_t batched_bytes = Turbo4BatchedDecodeWorkspaceBytes(
              start_pos + 1U, sequence_size, config.num_attention_heads,
              config.head_dim);
          const bool batched_reader =
              sequence_size > 1U && batched_bytes != 0U &&
              scratch.decode.weight_bf16.size_bytes() >= batched_bytes;
          if (batched_reader) {
            const std::size_t rows_bytes = sequence_size * q_bytes;
            const std::size_t row_partial_bytes =
                Turbo4BatchedDecodePartialBytes(start_pos + 1U, sequence_size,
                                                config.num_attention_heads,
                                                config.head_dim);
            auto* const batched_partials =
                reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
            float* const batched_q_rot =
                batched_partials +
                sequence_size * row_partial_bytes / sizeof(float);
            float* const batched_out_rot =
                batched_q_rot + sequence_size * vector_values;
            // The WMMA kernel serves up to 8 rows (48 query vectors).
            const bool wmma_requested =
                Turbo4VerifyWmmaEnabled() && sequence_size <= kMaxDecodeBatch;
            // Turbo5/6 (qh / padded slots) are served by the same
            // WMMA reader, which decodes each K/V block once per (kv head,
            // split) for all rows. That path is dev-gated; off, Turbo5/6 keeps
            // the format-aware split reader (one whole plane read per row).
            const bool wmma_capable =
                kv_formats.IsTurbo4() ||
                (GufoDevVerifyWmmaWide() && kv_formats.Valid());
            if (wmma_requested && !wmma_capable) {
              static const bool reported = [] {
                std::fprintf(stderr,
                             "gufo: Turbo5/6 WMMA verify refused "
                             "(set GUFO_DEV_VERIFY_WMMA_WIDE=1); using "
                             "format-aware split reader\n");
                return true;
              }();
              (void)reported;
            }
            const auto launch_reader = wmma_requested && wmma_capable
                                           ? &LaunchTurbo4AttentionVerifyWmma
                                           : &LaunchTurbo4AttentionSplitBatched;
            const auto reader_status = launch_reader(
                scratch.attention.q.data() + offset * attention_size,
                static_cast<const std::uint8_t*>(state_arena.d_kv_turbo4) +
                    slices.k_offset,
                static_cast<const std::uint8_t*>(state_arena.d_kv_turbo4) +
                    slices.v_offset,
                scratch.ssm.out.data() + offset * attention_size, batched_q_rot,
                batched_out_rot, batched_partials, slices.plane_bytes,
                rows_bytes, rows_bytes, rows_bytes, rows_bytes,
                sequence_size * row_partial_bytes,
                scratch.ssm.gate.data() + offset * attention_size,
                sequence_size * gate_bytes, config.num_attention_heads,
                config.num_key_value_heads, config.head_dim, start_pos + 1U,
                static_cast<std::uint32_t>(sequence_size),
                state_arena.GetMaxContext(),
                1.0F / std::sqrt(static_cast<float>(config.head_dim)),
                static_cast<const Turbo4KernelStatus*>(
                    state_arena.d_kv_turbo4_write_status),
                static_cast<Turbo4KernelStatus*>(
                    state_arena.d_kv_turbo4_read_status),
                arena_.stream, kv_formats);
            if (reader_status != Turbo4LaunchStatus::kLaunched) {
              throw std::logic_error(
                  "Qwen Turbo4 verification: batched split reader refused to "
                  "launch");
            }
          }
          for (std::size_t row = 0; !batched_reader && row < sequence_size;
               ++row) {
            const auto reader_status = LaunchTurbo4AttentionSplit(
                scratch.attention.q.data() + offset * attention_size +
                    row * attention_size,
                static_cast<const std::uint8_t*>(state_arena.d_kv_turbo4) +
                    slices.k_offset,
                static_cast<const std::uint8_t*>(state_arena.d_kv_turbo4) +
                    slices.v_offset,
                scratch.ssm.out.data() + offset * attention_size +
                    row * attention_size,
                q_rot, out_rot, workspace, slices.plane_bytes, q_bytes, q_bytes,
                q_bytes, q_bytes, partial_bytes,
                scratch.ssm.gate.data() + offset * attention_size +
                    row * attention_size,
                gate_bytes, config.num_attention_heads,
                config.num_key_value_heads, config.head_dim,
                start_pos + static_cast<std::uint32_t>(row) + 1U,
                state_arena.GetMaxContext(),
                1.0F / std::sqrt(static_cast<float>(config.head_dim)),
                static_cast<const Turbo4KernelStatus*>(
                    state_arena.d_kv_turbo4_write_status),
                static_cast<Turbo4KernelStatus*>(
                    state_arena.d_kv_turbo4_read_status),
                arena_.stream, /*query_is_rotated=*/false, kv_formats);
            if (reader_status != Turbo4LaunchStatus::kLaunched) {
              throw std::logic_error(
                  "Qwen Turbo4 verification: split block reader refused to "
                  "launch");
            }
          }
        } else if (fused_qknorm_rope_kv) {
          LaunchBatchedFusedQKNormRoPEKvWrite(
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.attention.k.data() + offset * kv_size),
              (scratch.attention.v.data() + offset * kv_size),
              static_cast<const float*>(layer.attn_q_norm.data),
              static_cast<const float*>(layer.attn_k_norm.data),
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.attention.k.data() + offset * kv_size),
              state_arena.d_kv_cache,
              OffsetIfPresent(state_arena.d_kv_cache, total_k),
              state_arena.d_attention_kv_f16,
              OffsetIfPresent(
                  static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                  total_k),
              attention_layer, start_pos, sequence_size,
              state_arena.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, config.rotary_dim,
              config.rope_theta, 1e-6F, arena_.stream,
              item.executor->vision_input_.rope());
          // Exact verification projections do not use weight dequantization
          // scratch. Reuse it for independent split-K rows without allocating
          // another buffer or changing the scalar decode workspace.
          auto attention_scratch = scratch.attention.split_k;
          if (scratch.decode.weight_bf16.size_bytes() >=
              sequence_size * attention_scratch.size_bytes()) {
            attention_scratch = {
                reinterpret_cast<float*>(scratch.decode.weight_bf16.data()),
                sequence_size * attention_scratch.size()};
          }
          LaunchCausalDecodeAttention(
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.ssm.gate.data() + offset * attention_size),
              state_arena.d_kv_cache,
              OffsetIfPresent(state_arena.d_kv_cache, total_k),
              state_arena.d_attention_kv_f16,
              OffsetIfPresent(
                  static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                  total_k),
              (scratch.ssm.out.data() + offset * attention_size),
              attention_layer, start_pos, sequence_size,
              state_arena.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, arena_.stream,
              attention_scratch);
        } else if (!turbo4_verify) {
          for (std::size_t row = 0; row < sequence_size; ++row) {
            float* const query =
                (scratch.attention.q.data() + offset * attention_size) +
                (row * attention_size);
            float* const key = (scratch.attention.k.data() + offset * kv_size) +
                               (row * kv_size);
            const std::uint32_t position = host_positions[offset + row];
            LaunchRoPE(query, key, config.num_attention_heads,
                       config.num_key_value_heads, config.head_dim,
                       config.rotary_dim, position, config.rope_theta,
                       arena_.stream, item.executor->vision_input_.rope());
            LaunchAttention(
                query, key,
                (scratch.attention.v.data() + offset * kv_size) + row * kv_size,
                (scratch.ssm.gate.data() + offset * attention_size) +
                    row * attention_size,
                state_arena.d_kv_cache,
                OffsetIfPresent(state_arena.d_kv_cache, total_k),
                state_arena.d_attention_kv_f16,
                OffsetIfPresent(
                    static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                    total_k),
                (scratch.ssm.out.data() + offset * attention_size) +
                    row * attention_size,
                attention_layer, position, state_arena.GetMaxContext(),
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, arena_.stream,
                state_arena.d_split_k_attention);
          }
        }
      }
      LaunchProjection(layer.attn_output, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       attention_size, arena_.stream);
    } else {
      LaunchProjection(layer.attn_qkv, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, ssm_qkv_size,
                       hidden_size, arena_.stream);
      LaunchProjection(layer.attn_gate, scratch.decode.normed.data(),
                       scratch.ssm.gate.data(), batch_size, ssm_inner_size,
                       hidden_size, arena_.stream);
      const auto controls =
          LaunchSsmControls(layer, scratch, batch_size, time_step_rank,
                            hidden_size, arena_.stream);
      LaunchSSMConvRecurrenceBatch(
          scratch.ssm.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data),
          scratch.ssm.conv_out.data(), controls.alpha, controls.beta,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          scratch.ssm.gate.data(), scratch.ssm.out.data(),
          std::span(ssm_sequences).first(items.size()),
          config.SsmLayerIndex(layer_index), ssm_qkv_size,
          config.ssm_group_count, config.ssm_time_step_rank,
          config.ssm_state_size, config.SsmValueSize(), controls.row_stride,
          ssm_inner_size, arena_.stream, arena_.GetRecurrentStateStorage());

      LaunchProjection(layer.ssm_out, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       ssm_inner_size, arena_.stream);
    }

    if (fuse_add_norm) {
      add_norm(scratch.attention.output.data(), layer.ffn_norm.data);
    } else {
      LaunchBatchedResidualAdd(
          scratch.decode.hidden.data(), scratch.attention.output.data(),
          scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
      LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                           static_cast<const float*>(layer.ffn_norm.data),
                           scratch.decode.normed.data(), nullptr, batch_size,
                           hidden_size, 1e-6F, arena_.stream);
    }

    LaunchFfnActivation(layer, scratch, batch_size, intermediate_size,
                        hidden_size, arena_.stream);
    LaunchProjection(layer.ffn_down, scratch.ffn.activation.data(),
                     scratch.ffn.out.data(), batch_size, hidden_size,
                     intermediate_size, arena_.stream);
    // Fused: with the next layer's attn norm, or the output norm (verify
    // always computes logits below).
    if (fuse_add_norm)
      add_norm(scratch.ffn.out.data(),
               layer_index + 1 < config.num_layers
                   ? weights_.layers[layer_index + 1].attn_norm.data
                   : weights_.output_norm.data);
    else
      LaunchBatchedResidualAdd(
          scratch.decode.hidden.data(), scratch.ffn.out.data(),
          scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
    if (capture_prompt_hidden_) {
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(layer_index);
          tap.has_value()) {
        float* const destination = feature_buffer + (*tap * hidden_size);
        HIP_CHECK(hipMemcpy2DAsync(destination, feature_width * sizeof(float),
                                   scratch.decode.hidden.data(),
                                   hidden_size * sizeof(float),
                                   hidden_size * sizeof(float), batch_size,
                                   hipMemcpyDeviceToDevice, arena_.stream));
      }
    }
  }

  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& executor = *item.executor;
    const std::size_t offset = offsets[index];
    if (capture_prompt_hidden_ && captured_layer_count > 0) {
      if (device_features) {
        // The rows stay here; the draft backend copies them device to
        // device after the host decides how many were committed.
        executor.last_verification_features_ =
            feature_buffer + offset * feature_width;
        executor.last_verification_feature_width_ = feature_width;
      } else {
        HIP_CHECK(
            hipMemcpyAsync(executor.h_verification_hidden_.data(),
                           feature_buffer + offset * feature_width,
                           item.tokens.size() * feature_width * sizeof(float),
                           hipMemcpyDeviceToHost, arena_.stream));
      }
      HIP_CHECK(hipMemcpyAsync(
          executor.arena_.d_target_layer_features,
          feature_buffer + (offset + item.tokens.size() - 1) * feature_width,
          feature_width * sizeof(float), hipMemcpyDeviceToDevice,
          arena_.stream));
    } else if (capture_prompt_hidden_) {
      HIP_CHECK(
          hipMemcpyAsync(executor.h_verification_hidden_.data(),
                         scratch.decode.hidden.data() + offset * hidden_size,
                         item.tokens.size() * hidden_size * sizeof(float),
                         hipMemcpyDeviceToHost, arena_.stream));
    }
    if (&executor == &coordinator) {
      executor.last_hidden_offset_ = (item.tokens.size() - 1) * hidden_size;
    } else {
      HIP_CHECK(hipMemcpyAsync(
          executor.arena_.d_hidden,
          scratch.decode.hidden.data() +
              (offset + item.tokens.size() - 1) * hidden_size,
          hidden_size * sizeof(float), hipMemcpyDeviceToDevice, arena_.stream));
      executor.last_hidden_offset_ = 0;
    }
  }
  if (!fuse_add_norm)
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(weights_.output_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena_.stream);
  LaunchProjection(weights_.output, scratch.decode.normed.data(), logits_buffer,
                   batch_size, vocab_size, hidden_size, arena_.stream);
  auto* const d_out_tokens = scratch.decode.prompt_tokens.data();
  LaunchBatchedGPUArgmax(logits_buffer, d_out_tokens, batch_size, vocab_size,
                         scratch.ffn.out, arena_.stream);
  std::array<tokenization::TokenId, kMaxRows> host_predictions{};
  HIP_CHECK(hipMemcpyAsync(host_predictions.data(), d_out_tokens,
                           batch_size * sizeof(tokenization::TokenId),
                           hipMemcpyDeviceToHost, arena_.stream));
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& executor = *item.executor;
    const std::size_t count = item.tokens.size() * vocab_size;
    const auto* logits = logits_buffer + offsets[index] * vocab_size;
    if (logits != executor.d_verification_logits_) {
      HIP_CHECK(hipMemcpyAsync(executor.d_verification_logits_, logits,
                               count * sizeof(float), hipMemcpyDeviceToDevice,
                               arena_.stream));
    }
    if (item.capture_logits) {
      executor.h_verification_logits_.resize(count);
      HIP_CHECK(hipMemcpyAsync(executor.h_verification_logits_.data(), logits,
                               count * sizeof(float), hipMemcpyDeviceToHost,
                               arena_.stream));
    }
  }
  std::array<Turbo4KernelStatus, 2> turbo4_status_words{};
  if (turbo4_verify) {
    HIP_CHECK(hipMemcpyAsync(
        &turbo4_status_words[0], coordinator.arena_.d_kv_turbo4_write_status,
        sizeof(Turbo4KernelStatus), hipMemcpyDeviceToHost, arena_.stream));
    HIP_CHECK(hipMemcpyAsync(
        &turbo4_status_words[1], coordinator.arena_.d_kv_turbo4_read_status,
        sizeof(Turbo4KernelStatus), hipMemcpyDeviceToHost, arena_.stream));
  }
  verify_timer.Stop();
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  if (turbo4_verify && Turbo4StepStatusFromWords(
                           static_cast<std::uint32_t>(turbo4_status_words[0]),
                           static_cast<std::uint32_t>(turbo4_status_words[1]))
                           .Refuses()) {
    throw std::logic_error(
        "Qwen Turbo4 verification: KV write or attention read status refused");
  }
  // A successful host verdict acknowledges this epoch; failures remain latched.
  if (turbo4_verify) {
    Turbo4ResetStepStatusWords(static_cast<Turbo4KernelStatus*>(
                                   coordinator.arena_.d_kv_turbo4_write_status),
                               static_cast<Turbo4KernelStatus*>(
                                   coordinator.arena_.d_kv_turbo4_read_status),
                               arena_.stream);
  }
  for (std::size_t row = 0; row < batch_size; ++row)
    (void)CheckedSampleToken(host_predictions[row], vocab_size);
  std::vector<std::vector<tokenization::TokenId>> predictions;
  predictions.reserve(items.size());
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    predictions.emplace_back(
        host_predictions.begin() + offsets[index],
        host_predictions.begin() + offsets[index] + item.tokens.size());
  }
  return predictions;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

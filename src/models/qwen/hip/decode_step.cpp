#include <algorithm>
#include <cmath>
#include <cstdlib>
#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"
#include "src/models/qwen/hip/turbo4_plane_index.hpp"
#include "src/models/qwen/modules/modules.hpp"

namespace gufo::hip {

void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy,
                              std::string_view mode) {
  if (!detail::DispatchTelemetryEnabled()) {
    return;
  }
  for (std::uint32_t layer_index = 0; layer_index < weights.config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    const auto resolution = ResolveQwenLayerRouteWithReasons(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    detail::EmitQwenRouteResolution(
        mode, layer_index, layer.is_full_attention ? "attention" : "ssm",
        resolution.plan.Fingerprint(),
        static_cast<std::uint32_t>(resolution.rejected));
  }
}

void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits,
                       QwenRecurrentStateIo recurrent_state,
                       models::qwen::vision::DeviceInput* image_input,
                       const void* embedding_row) {
  const auto* rope = image_input != nullptr ? image_input->rope() : nullptr;
  (void)token_id;

  // NO RESET HERE. It USED to be at the top of the step, which was wrong one
  // level above the layer case: `if (!compute_logits) return 0;` skips the
  // readback, so a refusal raised during a NON-logits step survived no readback
  // at all — and this reset then erased it before anyone looked.
  //
  // The reset belongs AFTER a PASSING verdict in decode.cpp, not before the
  // first layer here. Then the words hold whatever the last few steps
  // accumulated until something actually reads them.
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  auto scratch = arena.GetScratchView();
  auto& decode_scratch = scratch.decode;
  auto& attention_scratch = scratch.attention;
  auto& ssm_scratch = scratch.ssm;
  auto& ffn_scratch = scratch.ffn;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  // GPU parameter buffers
  const std::uint32_t* d_in_token = decode_scratch.prompt_tokens.data();
  const std::uint32_t* d_in_pos = decode_scratch.prompt_tokens.data() + 1;
  auto* d_out_token = decode_scratch.sampled_token.data();

  // 1. Embedding lookup
  if (embedding_row) {
    LaunchEmbeddingLookup(embedding_row, weights.token_embd.type, 0U,
                          decode_scratch.hidden.data(), hidden_size,
                          arena.stream);
  } else {
    LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                          d_in_token, decode_scratch.hidden.data(), hidden_size,
                          arena.stream);
  }

  if (image_input != nullptr)
    image_input->Inject(decode_scratch.hidden.data(), pos, 1, hidden_size, 1,
                        arena.stream);

  // 2. Layer stack
  // Dev switch GUFO_DEV_FUSE_ADDNORM=1: each residual add launches
  // together with the RMSNorm that reads it (bit-exact, -128 launches/token).
  static const bool kFuseAddNormEnv = [] {
    const char* v = std::getenv("GUFO_DEV_FUSE_ADDNORM");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  const bool fuse_add_norm = kFuseAddNormEnv && hidden_size == 5120;
  const auto add_norm = [&](std::span<const float> update, const void* w) {
    (void)TryLaunchAddRMSNorm(decode_scratch.hidden.data(), update.data(),
                              static_cast<const float*>(w),
                              decode_scratch.normed.data(), hidden_size, 1e-6F,
                              arena.stream);
  };

  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    const auto route_plan = ResolveQwenLayerRoute(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    const gufo::models::qwen::HipModuleContext module_ctx(
        static_cast<void*>(arena.stream), l, pos);

    // Fused: the previous layer's last residual add already wrote d_normed.
    if (!(fuse_add_norm && l > 0)) {
      // Pre-RMSNorm, routed through the norm module (HIP backend). Same
      // kernel, same args, same arena slices (d_hidden in, d_normed out);
      // behavior identical to the former inline `LaunchRMSNorm` call.
      const auto norm_view =
          gufo::models::qwen::MakeAttnNormView(layer, config);
      gufo::models::qwen::NormForward(
          module_ctx, norm_view, decode_scratch.hidden, decode_scratch.normed);
    }

    if (layer.is_full_attention) {
      // Full attention path
      const std::size_t total_k = arena.GetAttentionKvPlaneElements();
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      LaunchFusedQKVProjections(
          layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
          layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
          decode_scratch.normed.data(), ssm_scratch.qkv.data(),
          attention_scratch.k.data(), attention_scratch.v.data(),
          q_projection_size, kv_size, hidden_size, arena.stream);
      // De-interleave Q and Gate from attn_q projection
      LaunchUnpackQG(ssm_scratch.qkv.data(), attention_scratch.q.data(),
                     ssm_scratch.gate.data(), config.num_attention_heads,
                     config.head_dim, arena.stream);

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the
      // policy toggle as the independent reference.
      //
      // RESOLVED, NOT INFERRED. `ResolveQwenLayerKvPlan` decides
      // fused-vs-unfused per layer from the format table, and it OVERRIDES the
      // policy toggle for a block-packed layer: opt-c010 writes the cache at an
      // element pitch and has no 66-byte-block variant, so running it against a
      // Turbo4 plane writes fp16-shaped bytes into a block plane. That is the
      // silent corruption the override exists to prevent, and it is why this
      // reads `kv_plan` rather than `route_plan.fuse_qk_norm_rope_kv`.
      const gufo::hip::QwenLayerKvPlan kv_plan =
          gufo::hip::ResolveQwenLayerKvPlan(
              policy, attn_layer_idx,
              detail::IsFusedQkNormSupported(config.head_dim));
      const bool fused_qknorm_rope_kv = kv_plan.fuse_qk_norm_rope_kv;
      const bool turbo4_layer =
          kv_plan.reader == gufo::hip::QwenKvReaderKind::kBlockPacked;
      const std::size_t q_bytes = attention_scratch.q.size_bytes();
      const std::size_t out_bytes = ssm_scratch.out.size_bytes();
      const std::size_t gate_bytes = ssm_scratch.gate.size_bytes();
      const std::size_t kv_source_bytes = std::min(
          attention_scratch.k.size_bytes(), attention_scratch.v.size_bytes());
      const std::size_t q_rot_bytes =
          static_cast<std::size_t>(config.num_attention_heads) *
          config.head_dim * sizeof(float);
      std::size_t partial_bytes = 0U;
      float* decode_workspace = nullptr;
      float* q_rot = nullptr;
      float* out_rot = nullptr;
      gufo::hip::Turbo4LayerSlices slices{};
      auto* turbo4_base = static_cast<std::uint8_t*>(arena.d_kv_turbo4);
      // Block width (66/82/98 B) follows the storage the arena holds.
      const gufo::hip::turbo4::Bits kv_bits =
          gufo::hip::QwenKvTurboBits(policy.kv_cache_storage);

      gufo::hip::turbo4::PlaneFormats kv_formats{kv_bits};
      if (turbo4_layer) {
        kv_formats = gufo::hip::QwenKvTurboLayerFormats(policy, attn_layer_idx);
        const std::size_t turbo4_arena_bytes = arena.Turbo4KvBytesRecorded();
        if (!gufo::hip::Turbo4LayerPlaneSlices(
                turbo4_arena_bytes, /*base_bytes=*/0, config.head_dim,
                config.num_key_value_heads, arena.GetMaxContext(),
                policy.kv_layers.layer_count, attn_layer_idx, &slices,
                kv_bits)) {
          throw std::logic_error(
              "Qwen Turbo4 decode: layer slice does not fit the arena");
        }
        partial_bytes = arena.Turbo4DecodePartialBytesRecorded();
        if (arena.d_turbo4_decode_workspace == nullptr || partial_bytes == 0U ||
            partial_bytes > arena.Turbo4DecodeWorkspaceBytesRecorded()) {
          throw std::logic_error(
              "Qwen Turbo4 decode: split-K workspace was not allocated");
        }
        decode_workspace = static_cast<float*>(arena.d_turbo4_decode_workspace);
        q_rot = decode_workspace + partial_bytes / sizeof(float);
        out_rot = q_rot + static_cast<std::size_t>(config.num_attention_heads) *
                              config.head_dim;
        const auto* q_norm_weight =
            layer.attn_q_norm.empty()
                ? nullptr
                : static_cast<const float*>(layer.attn_q_norm.data);
        const auto* k_norm_weight =
            layer.attn_k_norm.empty()
                ? nullptr
                : static_cast<const float*>(layer.attn_k_norm.data);
        const gufo::hip::Turbo4LaunchStatus ps = LaunchTurbo4DecodePrepare(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), q_norm_weight, k_norm_weight, q_rot,
            turbo4_base + slices.k_offset, turbo4_base + slices.v_offset,
            d_in_pos, q_bytes, kv_source_bytes, q_rot_bytes, slices.plane_bytes,
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, arena.GetMaxContext(),
            config.rope_theta, 1e-6F,
            static_cast<gufo::hip::Turbo4KernelStatus*>(
                arena.d_kv_turbo4_write_status),
            static_cast<gufo::hip::Turbo4KernelStatus*>(
                arena.d_kv_turbo4_read_status),
            rope, arena.stream, kv_formats);
        if (ps != gufo::hip::Turbo4LaunchStatus::kLaunched) {
          throw std::logic_error(
              "Qwen Turbo4 decode: fused QKV pre-step refused to launch");
        }
      } else if (fused_qknorm_rope_kv) {
        LaunchFusedQKNormRoPEKvWrite(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(),
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data),
            attention_scratch.q.data(), attention_scratch.k.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            attn_layer_idx, d_in_pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, 1e-6F,
            arena.stream, rope);
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.q.data(),
              static_cast<const float*>(layer.attn_q_norm.data),
              attention_scratch.q.data(), config.num_attention_heads,
              config.head_dim, 1e-6F, arena.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.k.data(),
              static_cast<const float*>(layer.attn_k_norm.data),
              attention_scratch.k.data(), config.num_key_value_heads,
              config.head_dim, 1e-6F, arena.stream);
        }

        // RoPE (using device pos pointer for graph capture invariance)
        LaunchRoPE(attention_scratch.q.data(), attention_scratch.k.data(),
                   config.num_attention_heads, config.num_key_value_heads,
                   config.head_dim, config.rotary_dim, d_in_pos,
                   config.rope_theta, arena.stream, rope);
      }

      // Softmax Attention + Gating
      if (turbo4_layer) {
        // BLOCK-PACKED ROUTE. The fused Q/K/V pre-step above writes the K/V
        // blocks and q_rot; this reader consumes both on the same stream.
        // The status words are not reset here: the kernels latch through them
        // until decode.cpp reads them beside the sampled token.
        //
        // REACHABLE ONLY UNDER THE DEV DECODE OVERRIDE. The runtime policy must
        // select Turbo4 storage and GUFO_DEV_TURBO4_DECODE=1 must admit this
        // decode phase. The default policy remains fp16; E5/C2 exercised this
        // block-packed reader on device.
        //
        // `valid_context` is the HOST's pos + 1, not a device read. That is
        // exactly why kv_plan.graph_eligible is false for this layer: a
        // captured graph would replay with the context frozen at capture time.
        const gufo::hip::Turbo4LaunchStatus rs = LaunchTurbo4AttentionSplit(
            q_rot, turbo4_base + slices.k_offset, turbo4_base + slices.v_offset,
            ssm_scratch.out.data(), q_rot, out_rot, decode_workspace,
            slices.plane_bytes, q_bytes, out_bytes, q_rot_bytes, q_rot_bytes,
            partial_bytes, ssm_scratch.gate.data(), gate_bytes,
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, pos + 1, arena.GetMaxContext(),
            1.0F / std::sqrt(static_cast<float>(config.head_dim)),
            static_cast<const gufo::hip::Turbo4KernelStatus*>(
                arena.d_kv_turbo4_write_status),
            static_cast<gufo::hip::Turbo4KernelStatus*>(
                arena.d_kv_turbo4_read_status),
            arena.stream, /*query_is_rotated=*/true, kv_formats);
        if (rs != gufo::hip::Turbo4LaunchStatus::kLaunched) {
          throw std::logic_error(
              "Qwen Turbo4 decode: the split block reader refused to launch");
        }
        // NOTHING HERE TOUCHES THE STATUS WORDS. An earlier version latched
        // them from HOST code, which was the defect review
        // caught: `arena.d_kv_turbo4_*_status` are hipMalloc DEVICE pointers,
        // so a host `*status = value` is a host read-modify-write of device
        // memory — unordered against the kernels already queued on the stream,
        // and exactly what review refused at the launcher.
        //
        // It was also dead: the two throws above already fire on anything but
        // kLaunched, so the value being latched was always kOk.
        //
        // The words are latched by the KERNELS on the device, reset once per
        // step by the caller, and read back once by decode.cpp in the same sync
        // as the sampled token.
      } else if (use_split_k_decode) {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            ssm_scratch.out.data(), attn_layer_idx, pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream, attention_scratch.split_k.data(),
            fused_qknorm_rope_kv);
      } else {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            ssm_scratch.out.data(), attn_layer_idx, d_in_pos,
            arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena.stream,
            fused_qknorm_rope_kv);
      }

      // Output projection, routed through the quant_gemm module (HIP
      // backend). Same kernel, same args, same arena slices (d_ssm_out in,
      // d_attn_out out); behavior-identical to the former inline `LaunchGEMV`.
      gufo::models::qwen::QuantGemm(
          module_ctx, layer.attn_output, ssm_scratch.out.first(attention_size),
          hidden_size, attention_size, attention_scratch.output);
    } else {
      // SSM path

      LaunchFusedSSMInputProjections(
          layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
          layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
          layer.ssm_beta.data, layer.ssm_beta.type,
          decode_scratch.normed.data(), ssm_scratch.qkv.data(),
          ssm_scratch.gate.data(), ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), hidden_size, ssm_qkv_size, ssm_inner_size,
          time_step_rank, arena.stream);
      LaunchSSMConvRecurrence(
          ssm_scratch.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data), recurrent_state,
          ssm_scratch.conv_out.data(), ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          ssm_scratch.gate.data(), ssm_scratch.out.data(),
          config.SsmLayerIndex(l), ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena.stream, arena.GetSsmReplayCapture(),
          arena.GetRecurrentStateStorage());

      LaunchGEMV(layer.ssm_out.data, layer.ssm_out.type, ssm_scratch.out.data(),
                 attention_scratch.output.data(), hidden_size, ssm_inner_size,
                 arena.stream);
    }

    if (fuse_add_norm) {
      add_norm(attention_scratch.output, layer.ffn_norm.data);
    } else {
      gufo::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                      attention_scratch.output);

      {
        // FFN Pre-RMSNorm
        LaunchRMSNorm(decode_scratch.hidden.data(),
                      static_cast<const float*>(layer.ffn_norm.data),
                      decode_scratch.normed.data(), hidden_size, 1e-6F,
                      arena.stream);
      }
    }

    // Fused SwiGLU FFN
    {
      // Non-fused FFN, routed through the module. Same fused SwiGLU kernel +
      // same down GEMV as the former inline calls; behavior identical. The
      // module reads the arena device spans (x = d_normed, act_scratch =
      // d_ffn_act, out = d_ffn_out) directly.
      const auto ffn_view = gufo::models::qwen::MakeFfnView(layer, config);
      gufo::models::qwen::FfnForward(
          module_ctx, ffn_view, decode_scratch.normed, ffn_scratch.gate,
          ffn_scratch.up, ffn_scratch.activation, ffn_scratch.out);
    }

    // Residual Add (fused: with the next layer's attn norm, or the output norm)
    const bool last = l + 1 == config.num_layers;
    if (fuse_add_norm && (!last || compute_logits))
      add_norm(ffn_scratch.out, last ? weights.output_norm.data
                                     : weights.layers[l + 1].attn_norm.data);
    else
      gufo::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                      ffn_scratch.out);

    if (const auto tap_index = arena.GetTargetLayerCaptureIndex(l);
        tap_index.has_value()) {
      HIP_CHECK(hipMemcpyAsync(
          arena.d_target_layer_features + (*tap_index * hidden_size),
          decode_scratch.hidden.data(), hidden_size * sizeof(float),
          hipMemcpyDeviceToDevice, arena.stream));
    }
  }

  if (compute_logits) {
    // 3. Final Output Norm (fused into the last residual add when enabled)
    if (!fuse_add_norm)
      LaunchRMSNorm(decode_scratch.hidden.data(),
                    static_cast<const float*>(weights.output_norm.data),
                    decode_scratch.normed.data(), hidden_size, 1e-6F,
                    arena.stream);

    // 4. LM Head Logits GEMV on final token
    LaunchGEMV(weights.output.data, weights.output.type,
               decode_scratch.normed.data(), decode_scratch.logits.data(),
               vocab_size, hidden_size, arena.stream);

    // 5. Parallel GPU Argmax
    LaunchBatchedGPUArgmax(decode_scratch.logits.data(), d_out_token, 1,
                           vocab_size, ffn_scratch.out, arena.stream);
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

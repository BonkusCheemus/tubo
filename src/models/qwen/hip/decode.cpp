#if defined(ENGINE_ENABLE_HIP)
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/dev_step_dump.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace gufo::hip {

tokenization::TokenId QwenGpuExecutor::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  // Classified DECODE: this is the Turbo4 decode dev-switch path.
  CheckReset(QwenKvExecutionPhase::kDecode);
  if (pos >= arena_.GetMaxContext()) {
    throw std::length_error("token position exceeds the GPU context length");
  }

  if (replaying_ssm_state_ && !compute_logits) {
    if (arena_.CanReplaySsmPosition(pos)) {
      ReplaySsmState(pos);
      return 0;
    }
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  } else if (replaying_ssm_state_) {
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  }

  arena_.MarkSsmReplayPosition(pos);
  // One recurrent pass per step. Its buffer pair is part of the graph key below
  // because the captured launches bake the state pointers.
  const auto recurrent_state = arena_.BeginRecurrentWrite();
  last_hidden_offset_ = 0;
  const auto& config = weights_.config;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  auto scratch = arena_.GetScratchView();
  auto* d_out_token = scratch.decode.sampled_token.data();

  // Refresh outside capture, including every replay; the device address is
  // fixed.
  if (auto* store = model_->GetEmbeddingRows())
    decode_embedding_.Prepare(*store, {&token_id, 1}, arena_.stream, false);

  // Copy token_id and pos to GPU device memory
  const std::uint32_t in_params[2] = {token_id, pos};
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(), in_params,
                           sizeof(in_params), hipMemcpyHostToDevice,
                           arena_.stream));

  EmitDecodeRouteTelemetry(weights_, policy_, "decode");
  auto graph_key = graph_key_;
  const auto target_layers = arena_.GetTargetLayerCapture();
  graph_key.workload_identity = ExtendQwenGraphWorkloadIdentity(
      graph_key.workload_identity, target_layers.size());
  for (const auto layer : target_layers) {
    graph_key.workload_identity =
        ExtendQwenGraphWorkloadIdentity(graph_key.workload_identity, layer);
  }
  graph_key.workload_identity = ExtendQwenGraphWorkloadIdentity(
      graph_key.workload_identity, arena_.RecurrentIoIdentity());
  // B-D2: the policy-aware resolver, which ALSO refuses capture when any layer
  // is block-packed. The plain one below knew nothing of block layers, so the
  // only field recording that danger was set and never read — and a captured
  // graph would have frozen the block reader's host `pos + 1`, silently reading a
  // shorter context than the plane holds.
  const QwenGraphRejection graph_rejections =
      ResolveQwenGraphRejectionsForPolicy(
      policy_, compute_logits, use_split_k_decode,
      graph_executor_.IsEnabled() &&
          pos >= vision_input_.layout().PrefixLength());
  detail::EmitQwenGraphEligibility(
      graph_key.execution_identity, graph_key.workload_identity,
      static_cast<std::uint32_t>(graph_rejections));

  const auto launch_captured_graph = [&]() {
    if (!graph_executor_.Launch(arena_.stream, graph_key)) {
      graph_executor_.Reset();
      throw std::runtime_error(
          "Qwen HIP graph launch failed; captured graph was invalidated");
    }
  };

  if (graph_rejections == QwenGraphRejection::kNone) {
    if (graph_executor_.IsCapturedFor(graph_key)) {
      launch_captured_graph();
    } else {
      // Target feature taps change the captured copy operations.
      if (graph_executor_.IsCaptured())
        graph_executor_.Reset();
      const bool ok =
          graph_executor_.TryCapture(arena_.stream, graph_key, [&]() {
            ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos,
                              compute_logits, recurrent_state, &vision_input_,
                              decode_embedding_.Data());
          });
      if (ok) {
        launch_captured_graph();
      } else {
        ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos,
                          compute_logits, recurrent_state, &vision_input_,
                          decode_embedding_.Data());
      }
    }
  } else {
    ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos, compute_logits,
                      recurrent_state, &vision_input_, decode_embedding_.Data());
  }

  if (!compute_logits) {
    return 0;
  }

  // (b) THE STATUS READBACK JOINS THIS SYNC. Both status words are copied in the
  // SAME stream-ordered D2H pair as the sampled token, and there is still exactly
  // ONE stream synchronize below. Adding a second sync here would serialise the
  // host twice per token for no information.
  //
  // When the dev decode override admits Turbo4, this joins both status words
  // into the existing token readback; with the default fp16 policy no Turbo4
  // plane was allocated and these reads are skipped.
  std::uint32_t write_word = 0;
  std::uint32_t read_word = 0;
  const bool turbo4_step =
      arena_.d_kv_turbo4_write_status != nullptr &&
      arena_.d_kv_turbo4_read_status != nullptr;
  if (turbo4_step) {
    HIP_CHECK(hipMemcpyAsync(&write_word, arena_.d_kv_turbo4_write_status,
                             sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                             arena_.stream));
    HIP_CHECK(hipMemcpyAsync(&read_word, arena_.d_kv_turbo4_read_status,
                             sizeof(std::uint32_t), hipMemcpyDeviceToHost,
                             arena_.stream));
  }

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));

  // C1b: the logits row is copied ONLY when the dump is switched on, and it
  // joins the SAME stream-ordered copy pair above, so it still costs exactly one
  // host synchronisation per step. Unset, this is a single false test and the
  // stream is untouched.
  const DevStepDumpConfig step_dump = ReadDevStepDumpConfig();
  std::vector<float> dump_logits;
  if (step_dump.enabled) {
    dump_logits.resize(config.vocab_size);
    HIP_CHECK(hipMemcpyAsync(dump_logits.data(), scratch.decode.logits.data(),
                             config.vocab_size * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  // ...and the verdict is taken BEFORE the token is published. Refusing after
  // publishing would hand back a token computed from a plane nobody finished
  // writing, which is the silent stale success this design has refused since the
  // start.
  if (turbo4_step &&
      Turbo4StepStatusFromWords(write_word, read_word).Refuses()) {
    throw std::logic_error(
        "Qwen decode: a Turbo4 KV plane reported a failure and the sampled "
        "token is not published");
  }

  // THE RESET FOLLOWS THE VERDICT, IT DOES NOT PRECEDE THE STEP.
  // Resetting at the top of the step meant a step that
  // returned before the readback (`if (!compute_logits) return 0;`) had its
  // failure erased by the next step, with nothing ever having read it. Here the
  // words accumulate until a verdict is actually taken.
  //
  // TWO PROPERTIES FALL OUT OF THE ORDER, and both are why it is here:
  //   - a step that fails leaves its evidence IN the words, because the reset is
  //     below the throw and never runs;
  //   - a failure raised in a non-logits step is not lost, because the next
  //     readback still sees it.
  //
  // Queued on the stream AFTER the synchronize above, so it is ordered against
  // the next step's kernels rather than racing them.
  if (turbo4_step) {
    Turbo4ResetStepStatusWords(
        static_cast<Turbo4KernelStatus*>(arena_.d_kv_turbo4_write_status),
        static_cast<Turbo4KernelStatus*>(arena_.d_kv_turbo4_read_status),
        arena_.stream);
  }

  // C1b: emitted AFTER the verdict above, so a step whose Turbo4 status refuses
  // is never recorded as a normal observation -- it throws first and leaves no
  // line. A dump that logged refused steps as ordinary rows would let the very
  // failure the measurement exists to catch pass as data.
  // `next_token_id` is the id this step emitted, already on the host from the
  // d_out_token copy above -- a DIFFERENT device buffer from the logits row, and
  // written by the GPU argmax kernel rather than by a host loop. Passing it in
  // rather than re-reading keeps R3 free: no extra copy, no second sync, and the
  // emitted id is never recomputed from the row that produced the host argmax.
  AppendDevStepDump(step_dump, pos, dump_logits, config.vocab_size, write_word,
                    read_word, next_token_id);

  return CheckedSampleToken(next_token_id, config.vocab_size);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

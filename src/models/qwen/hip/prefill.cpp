#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
void QwenGpuExecutor::CheckPrefillCancellation() const {
  if (cancellation_check_ && cancellation_check_())
    throw std::runtime_error("Qwen prefill cancelled");
}

void QwenGpuExecutor::PrefillLayerCheckpoint(std::uint32_t layer) {
  CheckPrefillCancellation();
  // Keep one four-layer group ahead of the GPU. Without backpressure, AR
  // can enqueue the whole prompt before the client disconnects; checking
  // cancellation on the host then cannot stop seconds of abandoned work.
  if (!cancellation_check_ || layer == 0 || layer % 4 != 0)
    return;
  for (auto& event : prefill_events_) {
    if (!event)
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  }
  const auto slot = (layer / 4 - 1) % prefill_events_.size();
  HIP_CHECK(hipEventRecord(prefill_events_[slot], arena_.stream));
  if (layer < 8)
    return;
  hipError_t status;
  while ((status = hipEventQuery(prefill_events_[1 - slot])) ==
         hipErrorNotReady) {
    CheckPrefillCancellation();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  HIP_CHECK(status);
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  CheckPrefillCancellation();
  if (prompt_tokens.empty()) {
    return 0;
  }
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  if (capture_prompt_hidden_) {
    h_prompt_hidden_.clear();
    const std::size_t row_offset =
        prompt_hidden_capture_start_position_ > start_pos
            ? std::min<std::size_t>(
                  prompt_hidden_capture_start_position_ - start_pos,
                  prompt_tokens.size())
            : 0;
    const std::size_t captured_tokens = prompt_tokens.size() - row_offset;
    const std::size_t captured_layers =
        std::max<std::size_t>(arena_.GetTargetLayerCapture().size(), 1U);
    h_prompt_hidden_.reserve(captured_tokens * captured_layers *
                             weights_.config.hidden_size);
  }

  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }
  // Size the Turbo4 f16 scratch ONCE, from the prefix this whole batch
  // will reach, before the chunk loop starts. Sizing it per chunk instead would
  // grow it ceil(prompt / max_batch) times -- 22 large reallocations for a 90k
  // prompt, during the phase where free RAM is tightest. No-op unless
  // GUFO_DEV_SCRATCH_TILED is set.
  arena_.EnsureTurbo4PrefillScratch(end_pos);

  tokenization::TokenId next_token = 0;
  for (std::size_t offset = 0; offset < prompt_tokens.size();
       offset += arena_.GetMaxBatch()) {
    CheckPrefillCancellation();
    const std::size_t chunk_size = std::min<std::size_t>(
        arena_.GetMaxBatch(), prompt_tokens.size() - offset);
    const bool is_last = offset + chunk_size == prompt_tokens.size();
    next_token =
        ForwardPromptChunk(prompt_tokens.subspan(offset, chunk_size),
                           start_pos + static_cast<std::uint32_t>(offset),
                           compute_logits && is_last);
  }
  return next_token;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

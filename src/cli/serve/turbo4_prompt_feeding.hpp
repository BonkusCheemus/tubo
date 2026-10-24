// Per-token prompt fallback for Turbo4 when the Turbo4 prefill path is off.
//
// WHY THIS EXISTS. With GUFO_DEV_TURBO4_PREFILL unset, a Turbo4 KV policy is
// still refused by prefill entry points. This driver keeps that C1d fallback;
// E3 selects ForwardPromptBatch only when its separate prefill switch is on.
//
// SCOPE. Dev only. f16 keeps ForwardPromptBatch, unchanged and byte-identical.
// The selector keeps the storage check and the E3 switch in one place.
//
// WHY IT CALLS ForwardToken AND NOT TextRunner::Advance. Advance begins
//     if (!qwen.frontier().has_value())
//       throw std::logic_error("Qwen decode state has no retained frontier");
// and the frontier is first published BY prefill. So Advance cannot be driven
// for the FIRST prompt token -- there is nothing yet. This driver calls the
// engine entry Advance itself calls (QwenGpuExecutor::ForwardToken), from
// inside Prefill, where that precondition does not apply. Same mechanism and
// the same C1 guard: ForwardToken is annotated kDecode, so nothing here widens
// a phase.
//
// WHY A TEMPLATE. The driver is a pure function over two engine callbacks, so a
// CPU test can pass recording lambdas and observe exactly WHICH entry was
// called, HOW MANY times, with WHICH token, position and compute_logits flag.
// That is the only way to test "f16 still batches" without a device, and it is
// what makes a silent fallback to the batched path detectable rather than
// invisible.
//
// LOGIT SEMANTICS MATCH ForwardPromptBatch: only the LAST token of the chunk
// computes logits, so a prompt costs one logits step and N-1 non-logits steps.
// That is the shape the C brief describes.

#ifndef GUFO_CLI_SERVE_TURBO4_PROMPT_FEEDING_HPP_
#define GUFO_CLI_SERVE_TURBO4_PROMPT_FEEDING_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"

namespace gufo::server {

enum class QwenPromptFeedingMode {
  /// The ordinary path: one ForwardPromptBatch call for the whole chunk.
  kBatched,
  /// DEV fallback: one ForwardToken call per token when block prefill is off.
  kTokenByToken,
};

/// Which mode a resolved policy requires. Turbo4 is batched only when its
/// separately guarded prefill route is enabled; otherwise C1d remains the
/// token-by-token fallback.
[[nodiscard]] inline QwenPromptFeedingMode SelectQwenPromptFeedingMode(
    hip::QwenKvCacheStorage storage) noexcept {
  if (hip::QwenKvIsBlockAddressed(storage) &&
      !hip::QwenKvDevTurbo4PrefillEnabled())
    return QwenPromptFeedingMode::kTokenByToken;
  return QwenPromptFeedingMode::kBatched;
}

/// Feeds one prompt chunk and reports what it consumed.
///
///   forward_batch(span_of_chunk, start_position, compute_logits) -> frontier
///   token id forward_token(token, position, compute_logits) -> frontier token
///   id publish_frontier(frontier_token_id)      // batched or last token only
///
/// The driver chooses between them and does nothing else: it does not inspect
/// the policy, the context or the environment.
template<typename ForwardBatch, typename ForwardToken, typename PublishFrontier>
TextPrefillStep FeedQwenPromptChunk(QwenPromptFeedingMode mode,
                                    std::span<const TextRunnerToken> prompt,
                                    std::size_t offset, std::size_t consumed,
                                    bool decode_ready,
                                    ForwardBatch&& forward_batch,
                                    ForwardToken&& forward_token,
                                    PublishFrontier&& publish_frontier) {
  if (consumed == 0)
    return {.consumed_tokens = 0, .decode_ready = false};

  if (mode == QwenPromptFeedingMode::kBatched) {
    // The frontier is what the batched prefill RETURNED. Publishing
    // prompt[offset] here (the chunk's first token, '<|im_start|>' for a
    // think-on Qwen prompt) broke every fp16 AR request (seen in testing).
    const TextRunnerToken frontier =
        forward_batch(prompt.subspan(offset, consumed),
                      static_cast<std::uint32_t>(offset), decode_ready);
    if (decode_ready)
      publish_frontier(frontier);
    return {.consumed_tokens = consumed, .decode_ready = decode_ready};
  }

  // Per-token. `decode_ready` is a statement about the whole prompt, so the
  // logits flag belongs to the last token of the CHUNK the scheduler handed us,
  // exactly as the batched call passes it once for the chunk.
  for (std::size_t i = 0; i < consumed; ++i) {
    const std::size_t position = offset + i;
    const bool last_of_chunk = (i + 1 == consumed);
    const std::uint32_t frontier =
        forward_token(prompt[position], static_cast<std::uint32_t>(position),
                      decode_ready && last_of_chunk);
    if (decode_ready && last_of_chunk)
      publish_frontier(frontier);
  }
  return {.consumed_tokens = consumed, .decode_ready = decode_ready};
}

}  // namespace gufo::server

#endif  // GUFO_CLI_SERVE_TURBO4_PROMPT_FEEDING_HPP_

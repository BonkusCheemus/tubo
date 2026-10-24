#ifndef GUFO_CORE_SPECULATIVE_DRAFT_BACKEND_HPP_
#define GUFO_CORE_SPECULATIVE_DRAFT_BACKEND_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::speculative {

struct PromptCaptureWindow {
  std::uint32_t first_position{0};  // Absolute target position of the first row.
  std::size_t row_offset{0};       // Row offset within the requested segment.
};

struct PromptCaptureRows {
  std::size_t first_row{0};
  std::size_t row_count{0};
};

[[nodiscard]] constexpr PromptCaptureWindow MakeAlignedPromptCaptureWindow(
    std::uint32_t segment_start, std::size_t token_count,
    std::size_t history_rows, std::size_t injection_chunk) noexcept {
  if (history_rows == 0 || injection_chunk == 0) {
    return {.first_position = segment_start, .row_offset = 0};
  }
  const std::size_t expired =
      token_count > history_rows ? token_count - history_rows : 0;
  const std::size_t row_offset =
      (expired / injection_chunk) * injection_chunk;
  if (row_offset >
      std::numeric_limits<std::uint32_t>::max() - segment_start) {
    return {.first_position = segment_start, .row_offset = 0};
  }
  return {.first_position =
              segment_start + static_cast<std::uint32_t>(row_offset),
          .row_offset = row_offset};
}

[[nodiscard]] constexpr PromptCaptureRows PromptCaptureRowsForChunk(
    std::uint32_t chunk_start, std::size_t chunk_rows,
    std::uint32_t capture_start) noexcept {
  if (capture_start <= chunk_start) {
    return {.first_row = 0, .row_count = chunk_rows};
  }
  const auto first_row = static_cast<std::size_t>(capture_start - chunk_start);
  if (first_row >= chunk_rows) {
    return {.first_row = chunk_rows, .row_count = 0};
  }
  return {.first_row = first_row, .row_count = chunk_rows - first_row};
}

/// Represents a speculative draft proposal block
struct DraftProposal {
  std::vector<tokenization::TokenId> tokens;
  std::vector<tokenization::TokenId> candidate_ids;
  std::vector<float> candidate_probabilities;
  std::size_t candidates_per_token{0};
  std::uint32_t start_pos{0};
  /// The token the drafter was GIVEN as the anchor — the last
  /// committed token, `ForwardBlock`'s first argument. It is NOT
  /// `tokens.front()`: a DSpark block is [anchor, predicted...], so token 0 is
  /// a PREDICTION made after the anchor, and the two usually differ. This is the
  /// same value that seeds the drafter's Markov prev-chain (dflash/executor.cpp
  /// :1192), so it is what a bias reference must be keyed on.
  tokenization::TokenId anchor_token{0};
  /// DSpark confidence head, one float per PROPOSED token, same order and
  /// length as `tokens`. Empty for every backend that has no confidence head.
  ///
  /// Why it is here: the verifier is the only place the proposal AND
  /// the accepted count are both in scope, so a dump written there cannot get
  /// the confidences unless they travel with the proposal. `ForwardBlock`
  /// already produces them (`out_confidences`) — the DSpark backend was passing
  /// nullptr and dropping them.
  std::vector<float> confidences;
};

struct DraftTargetContext {
  std::span<const tokenization::TokenId> prompt_tokens;
  std::span<const float> prompt_hidden_states;
  std::size_t hidden_size{0};
  /// Absolute target position of prompt_tokens.front().
  std::uint32_t first_position{0};
  tokenization::TokenId first_token{0};
};

class IDraftBackend;

/// Independent requests that may share a draft-model forward pass. Each RNG
/// belongs to its request; batching must preserve its proposal distribution.
struct DraftProposalRequest {
  IDraftBackend* backend{nullptr};
  std::span<const tokenization::TokenId> tokens;
  std::uint32_t position{0};
  std::uint32_t max_tokens{0};
  float temperature{0.0F};
  std::uint64_t* rng_state{nullptr};
};

class IDraftBackendSnapshot {
public:
  IDraftBackendSnapshot() = default;
  virtual ~IDraftBackendSnapshot() = default;

  IDraftBackendSnapshot(const IDraftBackendSnapshot&) = delete;
  IDraftBackendSnapshot& operator=(const IDraftBackendSnapshot&) = delete;
  IDraftBackendSnapshot(IDraftBackendSnapshot&&) = delete;
  IDraftBackendSnapshot& operator=(IDraftBackendSnapshot&&) = delete;

  [[nodiscard]] virtual std::size_t PayloadBytes() const noexcept = 0;

  /// Stable model-owned byte payload used by persistent continuation caches.
  [[nodiscard]] virtual std::size_t PersistentPayloadBytes() const {
    throw std::logic_error(
        "draft backend snapshot does not support persistent sizing");
  }

  /// Serializes the persistent payload into an exactly sized destination.
  [[nodiscard]] virtual std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const {
    (void)destination;
    throw std::logic_error(
        "draft backend snapshot does not support persistence");
  }
};

/// Provider-neutral interface for draft token generators (MTP heads, small
/// model, heuristic)
class IDraftBackend {
public:
  virtual ~IDraftBackend() = default;

  /// Returns the name / type of the draft backend
  [[nodiscard]] virtual std::string_view Name() const noexcept = 0;

  /// Proposes up to max_tokens draft tokens given the current sequence
  [[nodiscard]] virtual DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) = 0;

  /// Samples proposals from the draft distribution and returns each sparse
  /// proposal row needed by lossless speculative rejection sampling. A row
  /// must describe the distribution actually sampled, conditional on earlier
  /// proposals. Do not discard a sampled token based on its own probability:
  /// that conditions the proposal without updating its reported distribution.
  [[nodiscard]] virtual DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) {
    (void)prompt_tokens;
    (void)current_pos;
    (void)max_tokens;
    (void)temperature;
    (void)rng_state;
    throw std::logic_error(
        "draft backend does not support lossless sampled proposals");
  }

  /// Returns true when ProposeSampled returns the exact proposal
  /// probabilities required by lossless speculative rejection sampling.
  [[nodiscard]] virtual bool SupportsSampledProposals() const noexcept {
    return false;
  }

  /// Providers can share projections while retaining separate request state.
  /// The default preserves the ordinary per-request proposal operations.
  [[nodiscard]] virtual std::vector<DraftProposal> ProposeBatch(
      std::span<const DraftProposalRequest> requests) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
      if (requests[index].backend == nullptr)
        throw std::invalid_argument("draft batch contains a null backend");
      for (std::size_t previous = 0; previous < index; ++previous) {
        if (requests[index].backend == requests[previous].backend)
          throw std::invalid_argument("draft batch repeats a session");
      }
    }
    std::vector<DraftProposal> proposals;
    proposals.reserve(requests.size());
    for (const auto& request : requests) {
      proposals.push_back(
          request.temperature > 0.0F
              ? request.backend->ProposeSampled(
                    request.tokens, request.position, request.max_tokens,
                    request.temperature, request.rng_state)
              : request.backend->Propose(request.tokens, request.position,
                                         request.max_tokens));
    }
    return proposals;
  }

  /// Returns true when the backend consumes target-model hidden states.
  [[nodiscard]] virtual bool RequiresTargetHiddenStates() const noexcept {
    return false;
  }

  /// Exact zero-based target layer outputs required by the draft backend.
  /// An empty span means the final target layer only.
  [[nodiscard]] virtual std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept {
    return {};
  }

  /// Selects the prompt feature suffix to capture. Default backends need all rows.
  [[nodiscard]] virtual PromptCaptureWindow TargetPromptCaptureWindow(
      std::uint32_t segment_start, std::size_t token_count) const noexcept {
    (void)token_count;
    return {.first_position = segment_start, .row_offset = 0};
  }

  /// Primes provider-specific state after target prompt prefill.
  [[nodiscard]] virtual bool PrimeTargetContext(
      const DraftTargetContext& context) {
    (void)context;
    return true;
  }

  /// Appends externally supplied tokens and their target features to an
  /// already primed draft. Position is the first new target input position.
  [[nodiscard]] virtual bool AppendTargetContext(
      const DraftTargetContext& context, std::uint32_t position) {
    (void)position;
    if (context.hidden_size == 0 ||
        context.prompt_hidden_states.size() !=
            context.prompt_tokens.size() * context.hidden_size) {
      return false;
    }
    for (std::size_t row = 0; row < context.prompt_tokens.size(); ++row) {
      UpdateTargetHidden(context.prompt_hidden_states.subspan(
          row * context.hidden_size, context.hidden_size));
    }
    return true;
  }

  /// Supplies the target hidden state paired with the next correction token.
  virtual void UpdateTargetHidden(std::span<const float> hidden) {
    (void)hidden;
  }

  /// Notifies the draft backend of which tokens were accepted and the target
  /// correction token
  virtual void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                              tokenization::TokenId correction_token) {
    (void)accepted;
    (void)correction_token;
  }

  /// Exact payload bytes that Snapshot() will allocate at the current
  /// committed boundary.
  [[nodiscard]] virtual std::size_t SnapshotPayloadBytes() const {
    throw std::logic_error("draft backend does not support snapshot sizing");
  }

  [[nodiscard]] virtual std::unique_ptr<IDraftBackendSnapshot> Snapshot()
      const {
    throw std::logic_error("draft backend does not support snapshots");
  }

  virtual void RestoreSnapshot(const IDraftBackendSnapshot&) {
    throw std::logic_error("draft backend does not support snapshot restore");
  }

  /// Restores a stable model-owned payload into this backend.
  virtual void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) {
    (void)payload;
    throw std::logic_error(
        "draft backend does not support persistent snapshot restore");
  }

  /// Resets internal draft generator state
  virtual void Reset() noexcept {}

  /// Starts a new generation over retained model state. Request-local proposal
  /// policies must reset so cache reuse does not change seeded generation.
  virtual void BeginRequest() noexcept {}
};

/// Mock / test draft backend for deterministic verification testing
class MockDraftBackend : public IDraftBackend {
public:
  explicit MockDraftBackend(std::vector<tokenization::TokenId> candidate_pool)
      : candidate_pool_(std::move(candidate_pool)) {}

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "MockDraftBackend";
  }

  [[nodiscard]] DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override {
    (void)prompt_tokens;
    DraftProposal proposal;
    proposal.start_pos = current_pos;
    const std::size_t count =
        std::min<std::size_t>(max_tokens, candidate_pool_.size());
    for (std::size_t i = 0; i < count; ++i) {
      proposal.tokens.push_back(candidate_pool_[i]);
    }
    return proposal;
  }

  void SetCandidates(std::vector<tokenization::TokenId> candidates) {
    candidate_pool_ = std::move(candidates);
  }

private:
  std::vector<tokenization::TokenId> candidate_pool_;
};

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_DRAFT_BACKEND_HPP_

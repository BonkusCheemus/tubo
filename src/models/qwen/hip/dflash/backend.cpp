#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/dflash/lookup.hpp"

namespace gufo::hip {
namespace {

constexpr std::array<std::uint8_t, 8> kDFlashDraftPersistentMagic = {
    'G', 'D', 'F', 'D', 'R', 'F', '0', '1'};
constexpr std::uint32_t kDFlashDraftPersistentVersion = 3;
constexpr std::size_t kDFlashDraftPersistentHeaderBytes = 80;
constexpr std::uint32_t kDFlashDraftPrimedFlag = 1U << 0U;

[[nodiscard]] bool LookupDraftingEnabled() noexcept {
  const char* const value = std::getenv("GUFO_DEV_LOOKUP");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// Lookup drafting v2 (upstream PR #295 + #570 cooldown). GUFO_DEV_LOOKUP_V2=1: after
/// the drafter's rows, fill the verifier's remaining width with the tokens
/// that followed the longest earlier match of history + draft. Sampled rows
/// are point-mass proposals, so rejection sampling keeps the target
/// distribution exact. GUFO_DEV_LOOKUP_PURE=1: lookup rows only, no drafter.
/// GUFO_DEV_LOOKUP_MIN: minimum match (default 12, as #295).
/// Single-session only (dev): the ProposeBatch cohort path ignores these, and
/// multi-item batched verify refuses items wider than 8 rows.
[[nodiscard]] bool LookupPureEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_LOOKUP_PURE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

[[nodiscard]] bool LookupAppendEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_LOOKUP_V2");
    return LookupPureEnabled() ||
           (value != nullptr && value[0] == '1' && value[1] == '\0');
  }();
  return enabled;
}

[[nodiscard]] std::size_t LookupMinMatch() noexcept {
  static const std::size_t value = [] {
    const char* const env = std::getenv("GUFO_DEV_LOOKUP_MIN");
    const long parsed = env != nullptr ? std::strtol(env, nullptr, 10) : 0;
    return parsed >= 2 ? static_cast<std::size_t>(parsed) : std::size_t{12};
  }();
  return value;
}

/// Lookup router (WhiFlash idea, arXiv 2606.07710, entropy policy replaced by
/// match length; no training). GUFO_DEV_LOOKUP_ROUTE=<n>: a round whose
/// committed history ends in a span that occurred earlier, at least n tokens
/// long, drafts from that lookup alone and skips the drafter forward; every
/// other round runs the drafter as configured. Unset or 0 = off; 1..7 is
/// raised to 8 (2-token repeats are everywhere in prose, the drafter would
/// almost never run). Single-session path only: ProposeBatch ignores it, as it
/// does V2/PURE. The drafter's context still catches up each round:
/// InjectPendingFeatures runs at the top of ProposeImpl whether or not the
/// forward runs.
[[nodiscard]] std::size_t LookupRouteMinMatch() noexcept {
  static const std::size_t value = [] {
    const char* const env = std::getenv("GUFO_DEV_LOOKUP_ROUTE");
    const long parsed = env != nullptr ? std::strtol(env, nullptr, 10) : 0;
    return parsed <= 0 ? std::size_t{0}
                       : static_cast<std::size_t>(std::max(parsed, 8L));
  }();
  return value;
}

// PR #570: after a round whose lookup rows were reached but mostly rejected,
// demand this match length for the next kLookupCooldownSpan positions.
constexpr std::size_t kLookupCooldownMatch = 48;
constexpr std::uint32_t kLookupCooldownSpan = 32;

/// GUFO_DEV_DRAFT_CONF_LOG=1 (with GUFO_DEV_DRAFT_LEN_LOG=1): append the
/// drafter/lookup row split and each row's confidence to the length log line,
/// so a stop rule can be fitted offline from served rounds.
[[nodiscard]] bool DraftConfLoggingEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_DRAFT_CONF_LOG");
    return value != nullptr && value[0] == '1';
  }();
  return enabled;
}

[[nodiscard]] bool DraftLengthLoggingEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_DRAFT_LEN_LOG");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

[[nodiscard]] bool DraftRankLoggingEnabled() noexcept {
  const char* const value = std::getenv("GUFO_DEV_DRAFT_RANK_LOG");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

/// Latched once: a decode policy that could change mid-process would
/// make two halves of one request incomparable, which is how a dev switch
/// outlives its dev window.
[[nodiscard]] bool DraftBanditEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_DRAFT_BANDIT");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

void ApplyLookupDrafting(std::span<const tokenization::TokenId> history,
                         std::vector<tokenization::TokenId>& draft,
                         std::vector<float>& confidences) {
  const auto match = dflash::FindSuffixLookupMatch(history);
  const std::size_t overridden =
      match ? dflash::FuseSuffixLookupHead(
                  std::span<tokenization::TokenId>{draft}, history, *match)
            : 0;
  if (overridden > 0) {
    if (confidences.size() < draft.size()) {
      confidences.resize(draft.size(), 0.0F);
    }
    std::fill_n(confidences.begin(), overridden, 1.0F);
  }
  std::fprintf(
      stderr, "GUFO_DEV_LOOKUP match_len=%zu overridden=%zu history=%zu\n",
      match ? match->match_len : std::size_t{0}, overridden, history.size());
}

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("DFlash draft persistent header is truncated");
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
    throw std::invalid_argument("DFlash draft persistent header is truncated");
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
    throw std::overflow_error("DFlash draft persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("DFlash draft persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

class QwenDFlashDraftSnapshot final
    : public speculative::IDraftBackendSnapshot {
public:
  QwenDFlashDraftSnapshot(std::unique_ptr<QwenDFlashGpuSnapshot> gpu_snapshot,
                          std::vector<float> pending_target_features,
                          std::size_t target_feature_width,
                          QwenDFlashGpuDraftConfig config,
                          float controller_state, std::uint32_t last_full_width,
                          bool primed)
      : gpu_snapshot(std::move(gpu_snapshot)),
        pending_target_features(std::move(pending_target_features)),
        target_feature_width(target_feature_width),
        config(config),
        controller_state(controller_state),
        last_full_width(last_full_width),
        primed(primed) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (gpu_snapshot != nullptr ? gpu_snapshot->PayloadBytes() : 0) +
           pending_target_features.size() * sizeof(float) +
           sizeof(controller_state) + sizeof(last_full_width);
  }

  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    if (gpu_snapshot == nullptr ||
        pending_target_features.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::invalid_argument(
          "DFlash draft snapshot persistent metadata is invalid");
    }
    return CheckedPersistentAdd(
        CheckedPersistentAdd(kDFlashDraftPersistentHeaderBytes,
                             gpu_snapshot->PersistentPayloadBytes()),
        pending_target_features.size() * sizeof(std::uint32_t));
  }

  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    const std::size_t expected_bytes = PersistentPayloadBytes();
    if (destination.size() != expected_bytes || target_feature_width == 0 ||
        target_feature_width > std::numeric_limits<std::uint32_t>::max() ||
        pending_target_features.size() % target_feature_width != 0) {
      throw std::invalid_argument(
          "DFlash draft persistent destination is invalid");
    }
    const std::size_t gpu_payload_bytes =
        gpu_snapshot->PersistentPayloadBytes();
    const std::size_t pending_bytes =
        pending_target_features.size() * sizeof(std::uint32_t);
    // The nested serializer and pending features overwrite the entire payload.
    std::fill_n(destination.begin(), kDFlashDraftPersistentHeaderBytes,
                std::uint8_t{0});
    std::copy(kDFlashDraftPersistentMagic.begin(),
              kDFlashDraftPersistentMagic.end(), destination.begin());
    PutLittleEndian<std::uint32_t>(destination, 8,
                                   kDFlashDraftPersistentVersion);
    PutLittleEndian<std::uint32_t>(
        destination, 12,
        static_cast<std::uint32_t>(kDFlashDraftPersistentHeaderBytes));
    PutLittleEndian<std::uint32_t>(destination, 16,
                                   primed ? kDFlashDraftPrimedFlag : 0U);
    PutLittleEndian<std::uint32_t>(
        destination, 20, static_cast<std::uint32_t>(target_feature_width));
    PutLittleEndian<std::uint64_t>(
        destination, 24,
        static_cast<std::uint64_t>(pending_target_features.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 32, static_cast<std::uint64_t>(gpu_payload_bytes));
    PutLittleEndian<std::uint64_t>(destination, 40,
                                   static_cast<std::uint64_t>(expected_bytes));
    PutLittleEndian<std::uint32_t>(destination, 48,
                                   gpu_snapshot->ValidContext());
    PutLittleEndian<std::uint32_t>(destination, 52, config.max_context);
    PutLittleEndian<std::uint32_t>(
        destination, 56, std::bit_cast<std::uint32_t>(controller_state));
    PutLittleEndian<std::uint32_t>(destination, 60,
                                   static_cast<std::uint32_t>(config.policy));
    PutLittleEndian<std::uint32_t>(destination, 64, config.max_draft_tokens);
    PutLittleEndian<std::uint32_t>(destination, 68, last_full_width);

    const std::size_t gpu_offset = kDFlashDraftPersistentHeaderBytes;
    const std::size_t pending_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t written = gpu_snapshot->SerializePersistent(
        destination.subspan(gpu_offset, gpu_payload_bytes));
    if (written != gpu_payload_bytes) {
      throw std::runtime_error(
          "DFlash GPU persistent serializer returned the wrong byte count");
    }
    for (std::size_t index = 0; index < pending_target_features.size();
         ++index) {
      PutLittleEndian<std::uint32_t>(
          destination, pending_offset + index * sizeof(std::uint32_t),
          std::bit_cast<std::uint32_t>(pending_target_features[index]));
    }
    if (CheckedPersistentAdd(pending_offset, pending_bytes) !=
        destination.size()) {
      throw std::logic_error(
          "DFlash draft persistent serializer size mismatch");
    }
    return destination.size();
  }

  std::unique_ptr<QwenDFlashGpuSnapshot> gpu_snapshot;
  std::vector<float> pending_target_features;
  std::size_t target_feature_width{0};
  QwenDFlashGpuDraftConfig config;
  float controller_state{0.0F};
  std::uint32_t last_full_width{0};
  bool primed{false};
};

}  // namespace

QwenDFlashGpuDraftBackend::QwenDFlashGpuDraftBackend(
    std::unique_ptr<QwenDFlashGpuExecutor> executor,
    QwenDFlashGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      controller_(
          config.policy, config.max_draft_tokens,
          // The tied output tensor distinguishes the qualified Q8 target
          // (Q8_0 head) from Q4 (Q6_K head) for verification cost selection.
          executor_->GetModel().GetWeights().output.type ==
              core::GgmlType::kQ8_0,
          DraftBanditEnabled()) {}

std::unique_ptr<QwenDFlashGpuDraftBackend> QwenDFlashGpuDraftBackend::Create(
    std::shared_ptr<const QwenDFlashGpuModel> model,
    QwenDFlashGpuDraftConfig config, std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0 ||
      model->GetDFlashConfig().block_size < 2) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GPU draft configuration is invalid";
    }
    return nullptr;
  }
  config.max_draft_tokens = std::min(
      {config.max_draft_tokens, model->GetDFlashConfig().block_size - 1U,
       speculative::DFlashLengthController::kMaxDraftTokens});
  auto executor = QwenDFlashGpuExecutor::Create(std::move(model),
                                                config.max_context, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<QwenDFlashGpuDraftBackend>(
      new QwenDFlashGpuDraftBackend(std::move(executor), config));
}

std::unique_ptr<QwenDFlashGpuDraftBackend>
QwenDFlashGpuDraftBackend::CreateFromGguf(
    std::string_view dflash_model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenDFlashGpuDraftConfig config, std::string* error_msg) {
  if (dflash_model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  auto reader_owner =
      core::GgufReader::OpenFile(std::string(dflash_model_path), error_msg);
  if (reader_owner == nullptr) {
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto model = QwenDFlashGpuModel::Create(std::move(reader),
                                          std::move(target_model), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenDFlashGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty()) {
    return false;
  }

  const std::size_t num_tokens = context.prompt_tokens.size();
  const std::size_t enc_in_dim = executor_->GetTargetFeaturesSize();

  if (context.hidden_size != enc_in_dim ||
      context.prompt_hidden_states.size() != num_tokens * enc_in_dim) {
    return false;
  }

  try {
    const auto ok = executor_->InjectTargetContext(
        context.prompt_hidden_states, context.first_position,
        static_cast<std::uint32_t>(num_tokens), {}, true);
    if (!ok) {
      return false;
    }
    primed_ = true;
    return true;
  } catch (const std::exception&) {
    Reset();
    return false;
  }
}

speculative::DraftProposal QwenDFlashGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  return ProposeImpl(prompt_tokens, current_pos, max_tokens, 0.0F, nullptr);
}

speculative::DraftProposal QwenDFlashGpuDraftBackend::ProposeSampled(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (!std::isfinite(temperature) || temperature <= 0.0F) {
    throw std::invalid_argument(
        "DFlash sampled proposal temperature must be finite and positive");
  }
  if (rng_state == nullptr) {
    throw std::invalid_argument("DFlash sampled proposal requires RNG state");
  }
  return ProposeImpl(prompt_tokens, current_pos, max_tokens, temperature,
                     rng_state);
}

std::vector<speculative::DraftProposal> QwenDFlashGpuDraftBackend::ProposeBatch(
    std::span<const speculative::DraftProposalRequest> requests) {
  if (requests.size() < 2 || requests.size() > 8)
    return IDraftBackend::ProposeBatch(requests);
  std::array<QwenDFlashGpuDraftBackend*, 8> backends{};
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto* backend =
        dynamic_cast<QwenDFlashGpuDraftBackend*>(requests[index].backend);
    if (backend == nullptr ||
        &backend->executor_->GetModel() != &executor_->GetModel())
      return IDraftBackend::ProposeBatch(requests);
    backends[index] = backend;
  }
  // Validate every request before changing caches or drawing random values.
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const auto& backend = *backends[index];
    if (!backend.primed_ || request.tokens.empty() || request.position == 0 ||
        backend.proposal_active_ || !std::isfinite(request.temperature) ||
        request.temperature < 0.0F ||
        (request.temperature > 0.0F && request.rng_state == nullptr))
      throw std::invalid_argument(
          "DFlash2 proposal batch has an invalid request");
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (backends[previous] == backends[index])
        throw std::invalid_argument("DFlash2 proposal batch repeats a session");
    }
  }
  std::vector<QwenDFlashContextRequest> contexts;
  contexts.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto& backend = *backends[index];
    // A backend whose rows are staged on the device cannot join this
    // shared host-feature injection. Inject it on its own, which takes the
    // staged route; adding it to `contexts` would pass the empty host vector
    // and lose the rows without any error.
    if (backend.pending_device_rows_ != 0) {
      backend.InjectPendingFeatures(requests[index].position);
      continue;
    }
    if (backend.PendingFeatureCount(requests[index].position) > 0) {
      contexts.push_back({backend.executor_.get(),
                          backend.pending_target_features_,
                          backend.executor_->GetInjectedContextLength(),
                          {}});
    }
  }
  if (!contexts.empty()) {
    QwenDFlashGpuExecutor::InjectTargetContextBatch(contexts);
    for (auto* backend : std::span(backends).first(requests.size()))
      backend->pending_target_features_.clear();
  }
  std::vector<speculative::DraftProposal> proposals(requests.size());
  std::array<
      std::array<float, speculative::DFlashLengthController::kMaxDraftTokens>,
      8>
      uniforms{};
  std::vector<QwenDFlashBlockRequest> blocks;
  std::vector<std::size_t> indices;
  blocks.reserve(requests.size());
  indices.reserve(requests.size());
  // Sampled and mixed cohorts retain private block choices and RNG replay.
  const auto greedy_batch_size =
      std::ranges::all_of(
          requests,
          [](const auto& request) { return request.temperature == 0.0F; })
          ? requests.size()
          : 1U;
  std::array<const speculative::DFlashLengthController*, 8> controllers{};
  std::array<std::uint32_t, 8> budgets{}, positions{};
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const auto& backend = *backends[index];
    const auto budget =
        request.position < backend.config_.max_context
            ? backend.config_.max_context - request.position - 1U
            : 0U;
    controllers[index] = &backend.controller_;
    budgets[index] = std::min(request.max_tokens, budget);
    positions[index] = request.position;
  }
  const auto shared_count =
      greedy_batch_size > 1
          ? speculative::DFlashLengthController::ChooseGreedyBatch(
                std::span(controllers).first(requests.size()),
                std::span(budgets).first(requests.size()),
                std::span(positions).first(requests.size()))
          : std::nullopt;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    auto& backend = *backends[index];
    proposals[index].start_pos = request.position;
    backend.draft_len_log_pending_ = false;
    backend.draft_rank_log_pending_ = false;
    backend.draft_len_log_confidences_.clear();
    const bool log_draft_len =
        backend.config_.policy == speculative::DFlashDraftPolicy::kAdaptive &&
        DraftLengthLoggingEnabled();
    const float mean_before =
        log_draft_len ? backend.controller_.State() : 0.0F;
    // A round verified alongside peers costs what the COHORT's row
    // count costs, not what this request's width costs alone. Feeding that
    // time to a per-request width would teach the bandit a number from a
    // different experiment, so the round is timed but not learned from.
    backend.round_cost_is_comparable_ = requests.size() == 1;
    const auto count =
        shared_count
            ? *shared_count
            : backend.controller_.Choose(budgets[index], request.position,
                                         greedy_batch_size,
                                         request.temperature == 0.0F);
    if (count == 0)
      continue;
    if (log_draft_len) {
      backend.draft_len_log_position_ = request.position;
      backend.draft_len_log_mean_before_ = mean_before;
      backend.draft_len_log_chosen_ = count;
      backend.draft_len_log_pending_ = true;
    }
    backend.proposed_tokens_.clear();
    if (request.temperature > 0.0F) {
      for (std::uint32_t row = 0; row < count; ++row)
        uniforms[index][row] =
            static_cast<float>(sampling::Uniform(request.rng_state));
    }
    indices.push_back(index);
    blocks.push_back({backend.executor_.get(),
                      request.tokens.back(),
                      request.position,
                      count,
                      request.temperature,
                      std::span(uniforms[index]).first(count),
                      {}});
  }
  if (blocks.empty())
    return proposals;
  const auto& first_model = backends[indices.front()]->executor_->GetModel();
  const bool rank_rows_remain_per_executor =
      blocks.size() == 1 ||
      speculative::IsDSparkDraft(first_model.GetWeights()) ||
      first_model.GetConfig().head_dim > 256;
  if (DraftRankLoggingEnabled() && rank_rows_remain_per_executor) {
    for (const auto index : indices) {
      auto& backend = *backends[index];
      backend.draft_rank_log_position_ = requests[index].position;
      backend.draft_rank_log_pending_ = true;
    }
  }
  auto generated = QwenDFlashGpuExecutor::ForwardBlockBatch(blocks);
  for (std::size_t block = 0; block < generated.size(); ++block) {
    const auto index = indices[block];
    auto& backend = *backends[index];
    backend.proposed_tokens_ = generated[block].tokens;
    backend.proposal_active_ = !backend.proposed_tokens_.empty();
    backend.draft_len_log_pending_ &= backend.proposal_active_;
    backend.draft_rank_log_pending_ &= backend.proposal_active_;
    proposals[index] = std::move(generated[block]);
  }
  return proposals;
}

std::uint32_t QwenDFlashGpuDraftBackend::PendingFeatureCount(
    std::uint32_t current_pos) const {
  if (current_pos < executor_->GetInjectedContextLength()) {
    throw std::logic_error(
        "DFlash committed position precedes injected history");
  }
  // Inject newly committed target tokens into DFlash draft KV cache
  if (current_pos > executor_->GetInjectedContextLength()) {
    const std::uint32_t start_p = executor_->GetInjectedContextLength();
    const std::uint32_t count = current_pos - start_p;
    // The staged-device route counts rows instead of floats. The row
    // count is still checked against the committed position, so a dropped or
    // doubled row is caught exactly as it was on the host route.
    if (pending_device_rows_ != 0) {
      if (!pending_target_features_.empty() || pending_device_rows_ != count) {
        throw std::logic_error(
            "DFlash committed target feature history is incomplete");
      }
      return count;
    }
    const std::size_t expected =
        static_cast<std::size_t>(count) * executor_->GetTargetFeaturesSize();
    if (pending_target_features_.size() != expected) {
      throw std::logic_error(
          "DFlash committed target feature history is incomplete");
    }
    return count;
  } else if (!pending_target_features_.empty() || pending_device_rows_ != 0) {
    throw std::logic_error(
        "DFlash has target features without a matching committed position");
  }
  return 0;
}

void QwenDFlashGpuDraftBackend::InjectPendingFeatures(
    std::uint32_t current_pos) {
  const auto count = PendingFeatureCount(current_pos);
  if (count == 0)
    return;
  if (pending_device_rows_ != 0) {
    if (!executor_->InjectStagedTargetFeatures(
            executor_->GetInjectedContextLength(), count)) {
      throw std::runtime_error(
          "DFlash staged target feature injection failed");
    }
    pending_device_rows_ = 0;
    return;
  }
  if (!executor_->InjectTargetContext(pending_target_features_,
                                      executor_->GetInjectedContextLength(),
                                      count)) {
    throw std::runtime_error(
        "DFlash committed target feature injection failed");
  }
  pending_target_features_.clear();
}

bool QwenDFlashGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || proposal_active_ || context.first_position < position ||
      context.hidden_size != executor_->GetTargetFeaturesSize() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    return false;
  }
  InjectPendingFeatures(position);
  return executor_->InjectTargetContext(
      context.prompt_hidden_states, context.first_position,
      static_cast<std::uint32_t>(context.prompt_tokens.size()), {}, true);
}

speculative::DraftProposal QwenDFlashGpuDraftBackend::ProposeImpl(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("DFlash GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("DFlash GPU proposal feedback is pending");
  }

  InjectPendingFeatures(current_pos);

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  // The anchor is the last committed token, and it is the value the Markov
  // chain starts from (dflash/executor.cpp:1192). Record it here, beside the
  // ForwardBlock call that consumes it, so a dump cannot report a prediction as
  // if it were the anchor.
  const tokenization::TokenId anchor_token = prompt_tokens.back();
  proposal.anchor_token = anchor_token;
  const std::uint32_t context_budget =
      current_pos < config_.max_context ? config_.max_context - current_pos - 1U
                                        : 0;
  draft_len_log_pending_ = false;
  draft_rank_log_pending_ = false;
  draft_len_log_confidences_.clear();
  // This request drafted and verified on its own, so its round time is a
  // sample of this width's cost. See the cohort note in ProposeBatch.
  round_cost_is_comparable_ = true;
  const bool log_draft_rank = DraftRankLoggingEnabled();
  const bool log_draft_len =
      config_.policy == speculative::DFlashDraftPolicy::kAdaptive &&
      DraftLengthLoggingEnabled();
  const float mean_before = log_draft_len ? controller_.State() : 0.0F;
  // The drafter's own staging holds block_size - 1 drafted rows plus the
  // correction, so the verified width never exceeds the block.
  const std::size_t lookup_width = std::min<std::size_t>(
      {max_tokens, context_budget,
       executor_->GetModel().GetDFlashConfig().block_size - 1U});
  std::vector<tokenization::TokenId> routed_tail;
  if (const std::size_t route_min = LookupRouteMinMatch();
      route_min > 0 && !LookupPureEnabled()) {
    const std::size_t nmin = dflash::LookupMinMatchAt(
        route_min, current_pos, lookup_cooldown_until_, kLookupCooldownMatch);
    routed_tail = dflash::LookupContinuation<tokenization::TokenId>(
        prompt_tokens, lookup_width, nmin,
        std::max(nmin, kLookupCooldownMatch + 16));
  }
  const bool lookup_pure = LookupPureEnabled() || !routed_tail.empty();
  const std::uint32_t count =
      lookup_pure ? 0U
                  : controller_.Choose(std::min(max_tokens, context_budget),
                                       current_pos, 1, temperature == 0.0F);
  drafter_count_ = 0;
  lookup_count_ = 0;
  if (count == 0 && !lookup_pure) {
    return proposal;
  }
  if (log_draft_len && count > 0) {
    draft_len_log_position_ = current_pos;
    draft_len_log_mean_before_ = mean_before;
    draft_len_log_chosen_ = count;
    draft_len_log_pending_ = true;
  }
  if (log_draft_rank && count > 0) {
    draft_rank_log_position_ = current_pos;
    draft_rank_log_pending_ = true;
  }

  proposed_tokens_.clear();

  std::vector<float> sample_uniforms;
  if (temperature > 0.0F) {
    sample_uniforms.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
      sample_uniforms.push_back(
          static_cast<float>(sampling::Uniform(rng_state)));
    }
    proposal.candidates_per_token =
        executor_->GetModel().GetDFlashConfig().selector_top_k;
  }
  if (count > 0)
    proposed_tokens_ = executor_->ForwardBlock(
        anchor_token, current_pos, count, temperature, sample_uniforms,
        &proposal.confidences,
        temperature > 0.0F ? &proposal.candidate_ids : nullptr,
        temperature > 0.0F ? &proposal.candidate_probabilities : nullptr);

  // DEV: DSpark confidence cut. Stop the draft at the first position
  // after the first whose confidence-head score is below
  // GUFO_DEV_DSPARK_CONF_MIN, so verify does not pay for rows that are likely
  // rejected. Position 0 is always kept (min-draft-tokens 1). DFlash-2 greedy
  // confidences are 1.0 unless GUFO_DEV_DRAFT_CONF=1; sampled ones are the
  // drafter's probability of the token it sampled. Sampled rounds are cut too
  // (sampling at t >= 0.6 is a common setting), but there the low row is KEPT and the draft
  // stops after it: dropping a row because of the value sampled in it would
  // emit tokens the drafter underrates less often than the target does (found
  // in review). Stopping after row i depends only on x_1..x_i, so
  // verify stays lossless. The per-row candidate arrays shrink with the rows.
  // DEV: GUFO_DEV_DSPARK_CONF_CUM=tau also stops once the product of
  // confidences would fall below tau (greedy: ~P(whole prefix accepted); sampled:
  // how peaked the drafter was, not an acceptance probability)
  // (SparseSpec-style; log replay: cum 0.1 gave +0.1..+1.3% in all six cells).
  // Same stopping-time structure as the per-row cut, so verify stays lossless.
  const char* const min_env = std::getenv("GUFO_DEV_DSPARK_CONF_MIN");
  const char* const cum_env = std::getenv("GUFO_DEV_DSPARK_CONF_CUM");
  const bool has_min = min_env != nullptr && min_env[0] != '\0';
  const bool has_cum = cum_env != nullptr && cum_env[0] != '\0';
  if (proposal.confidences.size() == proposed_tokens_.size() &&
      !proposed_tokens_.empty()) {
    if (has_min || has_cum) {
      const float min_conf = has_min ? std::strtof(min_env, nullptr) : 0.0F;
      const float cum_min = has_cum ? std::strtof(cum_env, nullptr) : 0.0F;
      float prefix = proposal.confidences[0];
      std::size_t keep = 1;
      while (keep < proposed_tokens_.size() &&
             proposal.confidences[keep] >= min_conf &&
             (!has_cum || prefix * proposal.confidences[keep] >= cum_min)) {
        prefix *= proposal.confidences[keep];
        ++keep;
      }
      if (temperature > 0.0F) ++keep;
      proposed_tokens_.resize(std::min(keep, proposed_tokens_.size()));
      proposal.confidences.resize(proposed_tokens_.size());
      if (temperature > 0.0F) {
        const std::size_t cells =
            proposed_tokens_.size() * proposal.candidates_per_token;
        if (proposal.candidate_ids.size() > cells)
          proposal.candidate_ids.resize(cells);
        if (proposal.candidate_probabilities.size() > cells)
          proposal.candidate_probabilities.resize(cells);
      }
    }
  }
  if (temperature == 0.0F && LookupDraftingEnabled()) {
    // The verifier supplies the full committed sequence as prompt_tokens.
    ApplyLookupDrafting(prompt_tokens, proposed_tokens_, proposal.confidences);
  }
  drafter_count_ = proposed_tokens_.size();
  if (!routed_tail.empty() || LookupAppendEnabled()) {
    const std::size_t width = lookup_width;
    if (width > drafter_count_) {
      // A routed round already holds its tail (drafter_count_ == 0 there).
      std::vector<tokenization::TokenId> tail = std::move(routed_tail);
      if (tail.empty()) {
        std::vector<tokenization::TokenId> seq(prompt_tokens.begin(),
                                               prompt_tokens.end());
        seq.insert(seq.end(), proposed_tokens_.begin(),
                   proposed_tokens_.end());
        const std::size_t nmin =
            dflash::LookupMinMatchAt(LookupMinMatch(), current_pos,
                                     lookup_cooldown_until_,
                                     kLookupCooldownMatch);
        tail = dflash::LookupContinuation<tokenization::TokenId>(
            seq, width - drafter_count_, nmin,
            std::max(nmin, kLookupCooldownMatch + 16));
      }
      if (!tail.empty() && temperature > 0.0F) {
        // Point mass on the copied token; the zero-probability fillers only
        // satisfy the verifier's distinct-ids-per-row check.
        const std::size_t k = std::max<std::size_t>(
            1, executor_->GetModel().GetDFlashConfig().selector_top_k);
        proposal.candidates_per_token = k;
        for (const auto token : tail) {
          proposal.candidate_ids.push_back(token);
          proposal.candidate_probabilities.push_back(1.0F);
          for (tokenization::TokenId filler = 0, added = 1; added < k;
               ++filler) {
            if (filler == token) continue;
            proposal.candidate_ids.push_back(filler);
            proposal.candidate_probabilities.push_back(0.0F);
            ++added;
          }
        }
      }
      if (!tail.empty()) {
        if (proposal.confidences.size() == drafter_count_)
          proposal.confidences.insert(proposal.confidences.end(), tail.size(),
                                      1.0F);
        proposed_tokens_.insert(proposed_tokens_.end(), tail.begin(),
                                tail.end());
        lookup_count_ = tail.size();
        lookup_position_ = current_pos;
      }
    }
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = !proposal.tokens.empty();
  if (draft_len_log_pending_ && DraftConfLoggingEnabled())
    draft_len_log_confidences_ = proposal.confidences;
  draft_len_log_pending_ &= proposal_active_;
  draft_rank_log_pending_ &= proposal_active_;
  return proposal;
}

void QwenDFlashGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("DFlash GPU proposal feedback is invalid");
  }

  if (draft_rank_log_pending_) {
    const std::size_t rejected_row = accepted.size();
    // Lookup rows have no drafter logits to rank against.
    if (rejected_row < proposed_tokens_.size() &&
        (lookup_count_ == 0 || rejected_row < drafter_count_)) {
      std::uint32_t rank = 0;
      float top1_probability = 0.0F;
      const std::uint32_t rejected_position =
          draft_rank_log_position_ + static_cast<std::uint32_t>(rejected_row);
      if (executor_->TryGetDraftRankStats(rejected_row, correction_token, rank,
                                          top1_probability)) {
        std::fprintf(stderr,
                     "GUFO_DEV_DRAFT_RANK_LOG position=%u drafted=%zu "
                     "accepted=%zu target=%u rank=%u p1=%.9g\n",
                     rejected_position, proposed_tokens_.size(),
                     accepted.size(), static_cast<unsigned>(correction_token),
                     rank, top1_probability);
      } else {
        std::fprintf(stderr,
                     "GUFO_DEV_DRAFT_RANK_LOG position=%u drafted=%zu "
                     "accepted=%zu target=%u rank=NA p1=NA "
                     "status=logits_unavailable\n",
                     rejected_position, proposed_tokens_.size(),
                     accepted.size(), static_cast<unsigned>(correction_token));
      }
    }
    draft_rank_log_pending_ = false;
  }
  // One round's wall time, selector to selector -- the same span the
  // offline cost tables were measured over. Taken here because the verifier
  // has already copied its predictions back to the host to compare tokens, so
  // this adds no sync of its own; it is one steady_clock read per round.
  double round_seconds = 0.0;
  if (DraftBanditEnabled()) {
    const auto now = std::chrono::steady_clock::now();
    if (last_round_end_.time_since_epoch().count() != 0) {
      round_seconds = std::chrono::duration<double>(now - last_round_end_)
                          .count();
    }
    last_round_end_ = now;
  }
  if (draft_len_log_pending_) {
    std::fprintf(stderr,
                 "GUFO_DEV_DRAFT_LEN_LOG position=%u mean_before=%.6f "
                 "chosen_length=%u drafted=%zu accepted=%zu",
                 draft_len_log_position_, draft_len_log_mean_before_,
                 draft_len_log_chosen_, proposed_tokens_.size(),
                 accepted.size());
    // Appended only with the bandit on, so a baseline log line stays
    // byte-identical for the scripts that already parse this.
    if (DraftBanditEnabled())
      std::fprintf(stderr, " round_ms=%.3f", round_seconds * 1000.0);
    if (DraftConfLoggingEnabled() &&
        draft_len_log_confidences_.size() == proposed_tokens_.size()) {
      std::fprintf(stderr, " drafter=%zu lookup=%zu conf=", drafter_count_,
                   lookup_count_);
      for (std::size_t row = 0; row < draft_len_log_confidences_.size(); ++row)
        std::fprintf(stderr, row == 0 ? "%.4f" : ",%.4f",
                     draft_len_log_confidences_[row]);
    }
    draft_len_log_confidences_.clear();
    std::fputc('\n', stderr);
    draft_len_log_pending_ = false;
  }
  if (lookup_count_ > 0) {
    // Judge lookup only on rounds where its rows were reached.
    if (accepted.size() >= drafter_count_ &&
        2 * (accepted.size() - drafter_count_) < lookup_count_)
      lookup_cooldown_until_ = lookup_position_ +
                               static_cast<std::uint32_t>(drafter_count_) +
                               kLookupCooldownSpan;
    if (DraftLengthLoggingEnabled())
      std::fprintf(stderr,
                   "GUFO_DEV_LOOKUP_V2 position=%u drafter=%zu lookup=%zu "
                   "accepted=%zu\n",
                   lookup_position_, drafter_count_, lookup_count_,
                   accepted.size());
    // One line per routed round: lookup rounds = this count, DFlash2 rounds =
    // draft_rounds minus it. ponytail: a stderr count, not a request metric;
    // plumb a draft_rounds split through the scheduler if the bench needs it
    // per request rather than per server log.
    if (drafter_count_ == 0 && LookupRouteMinMatch() > 0 &&
        !LookupPureEnabled())
      std::fprintf(stderr,
                   "GUFO_DEV_LOOKUP_ROUTE position=%u lookup=%zu "
                   "accepted=%zu\n",
                   lookup_position_, lookup_count_, accepted.size());
  }
  // The length controller models the drafter alone: lookup rows neither
  // count as its proposals nor make the round's width comparable.
  // Without lookup rows this is exactly the old Observe/ObserveRound (the
  // cohort path in ProposeBatch never sets drafter_count_).
  const std::size_t drafted =
      lookup_count_ > 0 ? drafter_count_ : proposed_tokens_.size();
  if (drafted > 0) {
    controller_.Observe(std::min(accepted.size(), drafted), drafted);
    if (round_cost_is_comparable_ && lookup_count_ == 0)
      controller_.ObserveRound(drafted, round_seconds);
  }
  drafter_count_ = 0;
  lookup_count_ = 0;
  proposal_active_ = false;
  proposed_tokens_.clear();
}

void QwenDFlashGpuDraftBackend::UpdateTargetHidden(
    std::span<const float> hidden) {
  const std::size_t enc_in_dim = executor_->GetTargetFeaturesSize();
  if (hidden.size() != enc_in_dim) {
    throw std::invalid_argument(
        "DFlash committed target feature width is invalid");
  }
  if (pending_device_rows_ != 0) {
    throw std::logic_error(
        "DFlash target features are already staged on the device");
  }
  pending_target_features_.insert(pending_target_features_.end(),
                                  hidden.begin(), hidden.end());
}

bool QwenDFlashGpuDraftBackend::UpdateTargetHiddenDevice(
    const float* device_rows, std::size_t row_stride, hipStream_t src_stream,
    std::uint32_t num_rows) {
  // Refuse rather than mix the two routes: a round that has already appended
  // host rows must finish on the host route, or the pending row count and the
  // pending float count would describe different histories.
  if (!pending_target_features_.empty())
    return false;
  if (!executor_->StageTargetFeaturesDevice(device_rows, row_stride, src_stream,
                                            pending_device_rows_, num_rows)) {
    return false;
  }
  pending_device_rows_ += num_rows;
  return true;
}

void QwenDFlashGpuDraftBackend::DiscardPendingTargetContext(
    std::uint32_t position) {
  if (proposal_active_ || executor_->GetInjectedContextLength() != position)
    throw std::logic_error("DFlash cancellation frontier is not retained");
  pending_target_features_.clear();
  pending_device_rows_ = 0;
}

std::size_t QwenDFlashGpuDraftBackend::SnapshotPayloadBytes() const {
  const std::size_t gpu_bytes = CheckedPersistentAdd(
      executor_->SnapshotPayloadBytes(), sizeof(float) + sizeof(std::uint32_t));
  // Staged device rows serialize exactly as host rows do; only the
  // place they are read from differs, so the payload size is the same.
  const std::size_t pending_floats =
      pending_device_rows_ != 0
          ? static_cast<std::size_t>(pending_device_rows_) *
                executor_->GetTargetFeaturesSize()
          : pending_target_features_.size();
  if (pending_floats >
      (std::numeric_limits<std::size_t>::max() - gpu_bytes) / sizeof(float)) {
    throw std::overflow_error("DFlash snapshot size overflows");
  }
  return gpu_bytes + pending_floats * sizeof(float);
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenDFlashGpuDraftBackend::Snapshot() const {
  if (proposal_active_) {
    throw std::logic_error(
        "DFlash snapshot requires a committed proposal boundary");
  }
  // Staged rows live on the device, so read them back here. A
  // snapshot is a request-boundary event, not a per-round one, so this
  // transfer does not sit in the decode loop.
  std::vector<float> pending = pending_target_features_;
  if (pending_device_rows_ != 0 &&
      !executor_->ReadStagedTargetFeatures(pending_device_rows_, pending)) {
    throw std::runtime_error("DFlash staged target feature readback failed");
  }
  return std::make_unique<QwenDFlashDraftSnapshot>(
      executor_->SaveSnapshot(), std::move(pending),
      executor_->GetTargetFeaturesSize(), config_, controller_.State(),
      controller_.LastFullWidth(), primed_);
}

void QwenDFlashGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  ResetLookupState();  // an absolute cooldown position must not survive a rewind
  const auto* dflash_snapshot =
      dynamic_cast<const QwenDFlashDraftSnapshot*>(&snapshot);
  if (dflash_snapshot == nullptr || dflash_snapshot->gpu_snapshot == nullptr) {
    throw std::invalid_argument(
        "DFlash draft snapshot is incompatible with the backend");
  }
  const std::size_t feature_width = executor_->GetTargetFeaturesSize();
  if (feature_width == 0 ||
      dflash_snapshot->config.max_context != config_.max_context ||
      dflash_snapshot->config.max_draft_tokens != config_.max_draft_tokens ||
      dflash_snapshot->config.policy != config_.policy ||
      dflash_snapshot->pending_target_features.size() % feature_width != 0) {
    throw std::invalid_argument(
        "DFlash draft snapshot has malformed pending target features");
  }
  auto controller = controller_;
  controller.Restore(dflash_snapshot->controller_state,
                     dflash_snapshot->last_full_width);
  executor_->RestoreSnapshot(*dflash_snapshot->gpu_snapshot);
  controller_ = controller;
  pending_target_features_ = dflash_snapshot->pending_target_features;
  pending_device_rows_ = 0;
  proposed_tokens_.clear();
  primed_ = dflash_snapshot->primed;
  proposal_active_ = false;
}

void QwenDFlashGpuDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  ResetLookupState();
  if (payload.size() < kDFlashDraftPersistentHeaderBytes ||
      !std::equal(kDFlashDraftPersistentMagic.begin(),
                  kDFlashDraftPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kDFlashDraftPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kDFlashDraftPersistentHeaderBytes ||
      GetLittleEndian<std::uint64_t>(payload, 72) != 0) {
    throw std::invalid_argument("DFlash draft persistent header is invalid");
  }
  const std::uint32_t flags = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::size_t feature_width = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::size_t pending_count =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 24));
  const std::size_t gpu_payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::uint32_t injected_context =
      GetLittleEndian<std::uint32_t>(payload, 48);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 52);
  const float controller_state =
      std::bit_cast<float>(GetLittleEndian<std::uint32_t>(payload, 56));
  const auto policy = GetLittleEndian<std::uint32_t>(payload, 60);
  const auto max_draft_tokens = GetLittleEndian<std::uint32_t>(payload, 64);
  const auto last_full_width = GetLittleEndian<std::uint32_t>(payload, 68);
  const std::size_t expected_feature_width = executor_->GetTargetFeaturesSize();

  if ((flags & ~kDFlashDraftPrimedFlag) != 0 || feature_width == 0 ||
      feature_width != expected_feature_width ||
      pending_count >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      pending_count % feature_width != 0 ||
      max_context != config_.max_context ||
      policy != static_cast<std::uint32_t>(config_.policy) ||
      max_draft_tokens != config_.max_draft_tokens ||
      injected_context > config_.max_context ||
      pending_count / feature_width >
          static_cast<std::size_t>(config_.max_context - injected_context) ||
      total_bytes != payload.size()) {
    throw std::invalid_argument(
        "DFlash draft persistent metadata is incompatible");
  }
  auto controller = controller_;
  controller.Restore(controller_state, last_full_width);
  const bool primed = (flags & kDFlashDraftPrimedFlag) != 0;
  if (!primed && (injected_context != 0 || pending_count != 0)) {
    throw std::invalid_argument("DFlash draft unprimed payload contains state");
  }
  const std::size_t gpu_offset = kDFlashDraftPersistentHeaderBytes;
  const std::size_t pending_offset =
      CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
  const std::size_t pending_bytes = pending_count * sizeof(std::uint32_t);
  if (CheckedPersistentAdd(pending_offset, pending_bytes) != payload.size()) {
    throw std::invalid_argument(
        "DFlash draft persistent payload size is invalid");
  }

  std::vector<float> pending_target_features;
  pending_target_features.reserve(pending_count);
  for (std::size_t index = 0; index < pending_count; ++index) {
    pending_target_features.push_back(
        std::bit_cast<float>(GetLittleEndian<std::uint32_t>(
            payload, pending_offset + index * sizeof(std::uint32_t))));
  }

  executor_->RestorePersistentSnapshot(
      payload.subspan(gpu_offset, gpu_payload_bytes));
  if (executor_->GetInjectedContextLength() != injected_context) {
    throw std::invalid_argument(
        "DFlash draft persistent context length is inconsistent");
  }
  pending_target_features_ = std::move(pending_target_features);
  pending_device_rows_ = 0;
  controller_ = controller;
  proposed_tokens_.clear();
  primed_ = primed;
  proposal_active_ = false;
  draft_rank_log_pending_ = false;
}

void QwenDFlashGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  controller_.Reset();
  pending_target_features_.clear();
  pending_device_rows_ = 0;
  proposed_tokens_.clear();
  primed_ = false;
  proposal_active_ = false;
  draft_rank_log_pending_ = false;
  ResetLookupState();
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

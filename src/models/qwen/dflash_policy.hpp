#ifndef GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_
#define GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

namespace gufo::speculative {

enum class DFlashDraftPolicy : std::uint32_t { kFixed, kAdaptive };

[[nodiscard]] inline DFlashDraftPolicy ParseDFlashDraftPolicy(
    std::string_view value) {
  if (value == "fixed")
    return DFlashDraftPolicy::kFixed;
  if (value.empty() || value == "adaptive")
    return DFlashDraftPolicy::kAdaptive;
  throw std::invalid_argument("DFlash2 draft policy must be fixed or adaptive");
}

[[nodiscard]] inline std::string_view DFlashDraftPolicyName(
    DFlashDraftPolicy policy) {
  return policy == DFlashDraftPolicy::kFixed ? "fixed" : "adaptive";
}

// Select the whole block before drawing any proposals. Sampled decisions use
// private acceptance history and position, preserving RNG consumption on
// replay. Greedy cohorts use their measured wider-verification cost.
// Full acceptance is censored: probe upward instead of treating the block limit
// as the true stopping point.
class DFlashLengthController {
public:
  static constexpr std::uint32_t kMaxDraftTokens = 7;

  DFlashLengthController(DFlashDraftPolicy policy, std::uint32_t limit,
                         bool q8_target = false, bool bandit = false)
      : policy_(policy),
        limit_(std::clamp(limit, 1U, kMaxDraftTokens)),
        q8_target_(q8_target),
        bandit_(bandit) {
    Reset();
  }

  /// `greedy` must be false for any request drawing random numbers. The block
  /// length decides how many uniforms the drafter consumes
  /// (dflash/backend.cpp ProposeImpl) and how many acceptance draws the
  /// verifier makes, so a length that depends on wall-clock time would make a
  /// sampled request irreproducible and break snapshot/continuation replay.
  /// Greedy requests draw nothing, so timing may enter their decision.
  ///
  /// Not const: the bandit records the modelled cost of the width it picked so
  /// ObserveRound can calibrate the table against the clock.
  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget,
                                     std::uint32_t position = 0,
                                     std::size_t greedy_batch_size = 1,
                                     bool greedy = false) noexcept {
    const auto cap = std::min(budget, limit_);
    if (policy_ == DFlashDraftPolicy::kFixed || cap == 0)
      return cap;
    // Preserve the full-acceptance probe at C5-C8. Extrapolating attention
    // cost beyond the measured depths must not shorten a saturated block.
    if (greedy_batch_size >= 5 && greedy_batch_size <= 8 &&
        mean_ == static_cast<float>(limit_))
      return cap;

    // Weight reads impose a substantial cost even on a short verification.
    // gfx1151 profiles estimate draft-plus-verification cost at each width.
    // Use that offline estimate, never request timings, and maximize expected
    // emitted tokens (including the target correction) per unit of work.
    // At the cap, full acceptance is still censored: the actual accepted run
    // can be longer than our limit. Probe the wider profitable block instead
    // of imposing an artificial rejection probability (1/8 at limit seven).
    const float probability =
        mean_ == static_cast<float>(limit_) ? 1.0F : mean_ / (mean_ + 1.0F);
    float survival = 1.0F;
    float expected_tokens = 1.0F;
    float best_score = 0.0F;
    std::uint32_t best_length = 1;
    const float context_scale =
        static_cast<float>(position > 2048 ? position - 2048 : 0) / 30720.0F;
    const auto* relative_costs = &kQ4RelativeCost;
    float context_multiplier = 1.0F;
    switch (greedy_batch_size) {
      case 2:
        relative_costs = &kQ4PairedRelativeCost;
        context_multiplier = 1.92F;
        break;
      // A finishing request does not make the remaining weight reads cheap.
      // Use the next measured capacity for odd cohorts too.
      case 3:
      case 4:
        relative_costs = &kQ4FourRequestRelativeCost;
        context_multiplier = 3.20F;
        break;
      case 5:
      case 6:
        relative_costs = &kQ4SixRequestRelativeCost;
        context_multiplier = 3.50F;
        break;
      case 7:
      case 8:
        relative_costs = &kQ4EightRequestRelativeCost;
        context_multiplier = 4.42F;
        break;
    }
    if (q8_target_ && greedy_batch_size == 4) {
      relative_costs = &kQ8FourRequestRelativeCost;
      context_multiplier = 4.0F * 100.0F / 170.66F;
    }
    const float attention_cost_scale = context_scale * context_multiplier;
    // The offline tables were measured at one cohort, one quant, one clock and
    // 2K-32K context. Every one of those drifts in service. When the bandit is
    // on and this request draws no random numbers, prefer the round time this
    // machine actually spent at each width and keep the table only as the
    // prior for widths that have never been timed.
    const bool use_measured = bandit_ && greedy && scale_ > 0.0F;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      survival *= probability;
      expected_tokens += survival;
      const float modelled =
          q8_target_ && greedy_batch_size != 4
              ? 1.0F + 0.02F * static_cast<float>(length)
              : (*relative_costs)[length - 1] +
                    attention_cost_scale * kQ4ContextCost[length - 1];
      if (bandit_)
        modelled_at_play_[length - 1] = modelled;
      const float cost =
          use_measured ? MeasuredCost(length, modelled) : modelled;
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best_length = length;
      }
    }
    if (use_measured) {
      if (const auto probe = ProbeWidth(best_length, cap); probe != 0)
        return probe;
    }
    return best_length;
  }

  // All-adaptive Q8 cohorts use a shared width. Their
  // target cost depends on the total row count: mixing individually cheap
  // widths can require two expensive projection launches. Keep each history
  // private, sum its expected tokens and choose one physical batch shape.
  [[nodiscard]] static std::optional<std::uint32_t> ChooseGreedyBatch(
      std::span<const DFlashLengthController* const> controllers,
      std::span<const std::uint32_t> budgets,
      std::span<const std::uint32_t> positions) noexcept {
    const auto users = controllers.size();
    if (budgets.size() != users || positions.size() != users)
      return std::nullopt;
    const auto* relative_costs = &kQ8FourRequestRelativeCost;
    double base_ms = 170.66;
    std::uint32_t probe_width = 3;
    // Odd cohorts use the next measured capacity's costs. A finishing request
    // must not send its remaining peers back to C1's cheap-wide-block estimate.
    switch (users) {
      case 2:
        relative_costs = &kQ8PairedRelativeCost;
        base_ms = 158.26;
        break;
      case 3:
      case 4:
        break;
      case 5:
      case 6:
        relative_costs = &kQ8SixRequestRelativeCost;
        base_ms = 189.32;
        probe_width = 1;
        break;
      case 7:
      case 8:
        relative_costs = &kQ8EightRequestRelativeCost;
        base_ms = 209.20;
        probe_width = 1;
        break;
      default:
        return std::nullopt;
    }
    std::uint32_t cap = kMaxDraftTokens;
    std::array<double, 8> probabilities{};
    double attention_cost_scale = 0.0;
    bool full_tile = true;
    bool neutral_or_full = users % 2 != 0;
    std::size_t full_histories = 0;
    for (std::size_t index = 0; index < controllers.size(); ++index) {
      const auto* controller = controllers[index];
      if (controller == nullptr || !controller->q8_target_ ||
          controller->policy_ != DFlashDraftPolicy::kAdaptive)
        return std::nullopt;
      cap = std::min(cap, std::min(budgets[index], controller->limit_));
      const double mean = controller->mean_;
      probabilities[index] =
          mean == controller->limit_ ? 1.0 : mean / (mean + 1.0);
      attention_cost_scale +=
          static_cast<double>(positions[index] > 2048 ? positions[index] - 2048
                                                      : 0) /
          30720.0 * 100.0 / base_ms;
      full_tile &= controller->last_full_width_ >= probe_width;
      const bool full_history = controller->last_full_width_ >= 3;
      // A single accepted short block is too weak to carry new peers into a
      // full probe. Two such blocks raise a neutral estimate from 3 to 4.5.
      full_histories += full_history && mean >= 4.5;
      neutral_or_full &=
          full_history ||
          (controller->last_full_width_ == 0 &&
           mean == std::min(3.0F, static_cast<float>(controller->limit_)));
    }
    if (cap == 0)
      return std::nullopt;
    // All requests fully accepted a short tile. Probe wider together; one lucky
    // request must not force its peers into a costly ragged verification.
    // Two established full blocks can carry a growing odd cohort through the
    // probe when its other estimates remain neutral. Other histories use the
    // ordinary shared cost decision.
    if (full_tile || (neutral_or_full && full_histories >= 2))
      return cap;
    std::array<double, 8> survival{};
    survival.fill(1.0);
    double expected_tokens = static_cast<double>(users), best_score = 0.0;
    std::uint32_t best_length = 1;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      for (std::size_t index = 0; index < controllers.size(); ++index) {
        survival[index] *= probabilities[index];
        expected_tokens += survival[index];
      }
      const double cost = (*relative_costs)[length - 1] +
                          attention_cost_scale * kQ4ContextCost[length - 1];
      const double score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best_length = length;
      }
    }
    return best_length;
  }

  void Observe(std::size_t accepted, std::size_t drafted) noexcept {
    if (policy_ == DFlashDraftPolicy::kFixed || drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    last_full_width_ =
        accepted == drafted ? static_cast<std::uint32_t>(drafted) : 0U;
    // For a geometric accepted run with mean m, censoring at any block width
    // preserves E[accepted - m * rejected] == 0. Weight a completed block by
    // its accepted tokens too; a fixed increment biases short blocks upward
    // and long blocks downward.
    mean_ = accepted == drafted
                ? std::min(mean_ + 0.25F * static_cast<float>(accepted),
                           static_cast<float>(limit_))
                : 0.75F * mean_ + 0.25F * static_cast<float>(accepted);
  }

  /// One completed round: `drafted` tokens were proposed and verified in
  /// `seconds` of host wall time, measured selector-to-selector exactly as the
  /// offline tables above were. Call once per round, after Observe.
  ///
  /// This is the bandit's only reward signal. The numerator of the score
  /// (expected emitted tokens) is shared across widths because one acceptance
  /// observation informs every width through `mean_`, so the per-arm unknown is
  /// cost alone and only cost needs exploring. That is why this is not a UCB
  /// over seven independent arms: independent arms would throw away the
  /// structure that makes one observation worth seven.
  void ObserveRound(std::size_t drafted, double seconds) noexcept {
    if (!bandit_ || drafted == 0 || drafted > limit_ ||
        !std::isfinite(seconds) || seconds <= 0.0)
      return;
    // A round that took longer than this was not a width cost. The measured
    // cohort cycles above are 104-209 ms, so a whole second is a 5-10x
    // outlier: a thermal throttle, a KFD stall, or a gap where some other
    // round did not reach AcceptFeedback. Admitting one would pin an arm at a
    // fictional cost for the rest of the request.
    constexpr double kStallSeconds = 1.0;
    if (seconds > kStallSeconds)
      return;
    const auto arm = drafted - 1;
    const auto sample = static_cast<float>(seconds);
    measured_[arm] = measured_[arm] == 0.0F
                         ? sample
                         : 0.75F * measured_[arm] + 0.25F * sample;
    last_play_[arm] = ++round_;
    if (modelled_at_play_[arm] > 0.0F) {
      // Rounds of measured cost per unit of modelled cost. Converts the table
      // into this machine's units so an unmeasured width stays comparable.
      const float ratio = measured_[arm] / modelled_at_play_[arm];
      scale_ = scale_ == 0.0F ? ratio : 0.75F * scale_ + 0.25F * ratio;
    }
  }

  void Reset() noexcept {
    mean_ = std::min(3.0F, static_cast<float>(limit_));
    last_full_width_ = 0;
    // Measured round costs are a property of this GPU at this clock, not of
    // the request, so they deliberately survive BeginRequest. Only a new
    // controller starts blind.
  }
  [[nodiscard]] float State() const noexcept { return mean_; }
  [[nodiscard]] std::uint32_t LastFullWidth() const noexcept {
    return last_full_width_;
  }
  void Restore(float mean, std::uint32_t last_full_width = 0) {
    if (!std::isfinite(mean) || mean < 0.0F ||
        mean > static_cast<float>(limit_) || last_full_width > limit_)
      throw std::invalid_argument("DFlash2 controller state is invalid");
    mean_ = mean;
    last_full_width_ = last_full_width;
  }

private:
  /// This width's cost: its own round-time EMA once it has ever been timed,
  /// otherwise the table converted into measured units by `scale_`.
  ///
  /// A width that has been timed never reverts to the table, however old the
  /// measurement is. Two earlier drafts got this wrong and both failed the
  /// same way: `scale_` is calibrated from whichever width is played most, so
  /// a table-derived prior for the OTHER widths carries that one width's
  /// ratio times the table's SHAPE -- and the shape is the thing we stopped
  /// trusting. With a profile the table has backwards it ranked the narrow
  /// widths cheapest and spent a third of all rounds off the best width.
  /// Drift is tracked by re-measuring through the probe schedule and the EMA,
  /// not by forgetting.
  [[nodiscard]] float MeasuredCost(std::uint32_t length,
                                   float modelled) const noexcept {
    const float measured = measured_[length - 1];
    return measured == 0.0F ? scale_ * modelled : measured;
  }

  /// The width to probe this round instead of the argmax, or zero for none.
  ///
  /// Round-robin over the argmax's two neighbours, once every kProbeInterval
  /// rounds. Neighbours only, because expected-tokens-per-round is near
  /// unimodal in width -- the cliffs are at row-tile boundaries, which a
  /// one-step walk crosses -- so tracking a drifting optimum never needs a
  /// distant arm, and a neighbour costs a few percent more than the best
  /// rather than the 2x a far arm can cost. That bounds the price of
  /// exploration at roughly (1/kProbeInterval) x (neighbour penalty), well
  /// under a tenth of a percent of decode time.
  ///
  /// Driven by the deterministic round counter, never a random draw, so a
  /// replayed decode makes the same decisions.
  [[nodiscard]] std::uint32_t ProbeWidth(std::uint32_t best,
                                         std::uint32_t cap) const noexcept {
    constexpr std::uint64_t kProbeInterval = 32;
    if (cap < 2 || round_ == 0 || round_ % kProbeInterval != 0)
      return 0;
    const std::uint32_t lower = best > 1 ? best - 1 : 0;
    const std::uint32_t upper = best < cap ? best + 1 : 0;
    if (lower == 0)
      return upper;
    if (upper == 0)
      return lower;
    // Whichever neighbour has waited longer.
    return last_play_[lower - 1] <= last_play_[upper - 1] ? lower : upper;
  }

  // Q8 C2 complete cycles cost 158/161/161/172/178/211/187 ms.
  // The fourteen-row verification at six drafts is slower than sixteen rows.
  static constexpr std::array<float, kMaxDraftTokens> kQ8PairedRelativeCost{
      1.0F, 1.015F, 1.020F, 1.089F, 1.127F, 1.333F, 1.182F};
  // Q8_K_XL C4 complete cycles cost 171/181/197/327/325/357/316 ms.
  // Four through six drafts split verification into separate launches;
  // seven fills two adjacent sixteen-row groups and reuses their weights.
  // The same attention geometry uses the measured absolute per-request
  // context delta below, normalized to this cohort's one-draft cycle.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ8FourRequestRelativeCost{1.0F,   1.06F,  1.153F, 1.914F,
                                 1.907F, 2.089F, 1.853F};
  // C6 complete cycles cost 189/333/334/405/445/474/455 ms.
  // A short twelve-row tile avoids the wider target projection launches.
  static constexpr std::array<float, kMaxDraftTokens> kQ8SixRequestRelativeCost{
      1.0F, 1.760F, 1.763F, 2.141F, 2.352F, 2.505F, 2.404F};
  // C8 complete cycles cost 209/337/326/467/457/596/589 ms for 1–7 drafts.
  // Ragged 24/40/56-row projections cost more than their adjacent full tiles.
  // These selector-to-selector spans include target verification, context
  // injection, the next draft and host acceptance work.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ8EightRequestRelativeCost{1.0F,   1.610F, 1.561F, 2.231F,
                                  2.183F, 2.850F, 2.813F};
  static constexpr std::array<float, kMaxDraftTokens> kQ4RelativeCost{
      1.0F, 1.02F, 1.045F, 1.08F, 1.14F, 1.20F, 1.29F};
  // Two greedy requests verify 4–16 rows together. Crossing eight rows has a
  // large projection cost that the C1 estimate misses. Complete-cycle release
  // controls normalize to 104 ms at one draft; the paired context multiplier
  // is 2 * 100 / 104, using the existing per-request attention calibration.
  static constexpr std::array<float, kMaxDraftTokens> kQ4PairedRelativeCost{
      1.0F, 1.04F, 1.10F, 1.48F, 1.45F, 1.50F, 1.50F};
  // Four-request cycles cross another projection cliff above sixteen rows.
  // One fixed-width profile per point gives 125/163/167/246/265/286/289 ms.
  // The context multiplier is 4 * 100 / 125.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ4FourRequestRelativeCost{1.0F,  1.30F, 1.33F, 1.97F,
                                 2.12F, 2.29F, 2.32F};
  // Six requests verify 12–48 rows. Measured complete cycles cost
  // 171/247/273/323/378/416/423 ms; the context multiplier is 6 * 100 / 171.
  static constexpr std::array<float, kMaxDraftTokens> kQ4SixRequestRelativeCost{
      1.0F, 1.44F, 1.59F, 1.89F, 2.21F, 2.43F, 2.47F};
  // Eight requests verify 16–64 rows. Measured complete cycles cost
  // 181/276/304/400/430/523/558 ms; the context multiplier is 8 * 100 / 181.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ4EightRequestRelativeCost{1.0F,  1.52F, 1.68F, 2.21F,
                                  2.38F, 2.89F, 3.08F};
  // Extra cost from 2K to 32K for the target's 16 full-attention layers,
  // relative to the roughly 100 ms shallow draft/verification cycle. These
  // cold-KV measurements include the anchor plus 1–7 proposals and preserve
  // the actual row-tile boundaries. Position is deterministic on replay;
  // request timings and sampled proposals never enter the decision.
  static constexpr std::array<float, kMaxDraftTokens> kQ4ContextCost{
      0.11F, 0.14F, 0.21F, 0.23F, 0.25F, 0.31F, 0.35F};
  DFlashDraftPolicy policy_;
  std::uint32_t limit_;
  bool q8_target_;
  bool bandit_{false};
  float mean_{0.0F};
  std::uint32_t last_full_width_{0};
  // GUFO_DEV_DRAFT_BANDIT state. Not snapshotted: these are machine timings,
  // not request state, and the persistent format would have to change version
  // to carry them. A restored session relearns them within ~20 rounds.
  std::array<float, kMaxDraftTokens> measured_{};
  std::array<float, kMaxDraftTokens> modelled_at_play_{};
  std::array<std::uint64_t, kMaxDraftTokens> last_play_{};
  float scale_{0.0F};
  std::uint64_t round_{0};
};

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_

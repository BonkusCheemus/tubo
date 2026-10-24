#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace gufo::hip::dflash {

inline constexpr std::size_t kLookupNMin = 3;
inline constexpr std::size_t kLookupNMax = 32;
inline constexpr std::size_t kLookupNStrong = 8;
inline constexpr std::size_t kLookupAgreeMin = 2;

struct SuffixLookupMatch {
  std::size_t match_len{0};
  std::size_t match_end{0};
};

/// Find the longest suffix that occurred earlier, breaking ties by the latest
/// earlier end position. The earlier match may overlap the suffix; its
/// available continuation repeats to fill the drafter's head.
template<typename Token>
[[nodiscard]] std::optional<SuffixLookupMatch> FindSuffixLookupMatch(
    std::span<const Token> history, std::size_t nmin = kLookupNMin,
    std::size_t nmax = kLookupNMax) {
  if (nmin == 0 || nmax < nmin || history.size() <= nmin) {
    return std::nullopt;
  }

  // ponytail: O(history * 32) reverse scan; add an n-gram index only if
  // profiling shows the fixed NMAX scan is material.
  const std::size_t max_match = std::min(nmax, history.size() - 1);
  SuffixLookupMatch best;
  for (std::size_t end = history.size() - 2;; --end) {
    std::size_t matched = 0;
    while (matched < max_match && matched <= end &&
           history[end - matched] == history[history.size() - 1 - matched]) {
      ++matched;
    }
    if (matched >= nmin && matched > best.match_len) {
      best = {matched, end};
      if (matched == max_match) {
        return best;
      }
    }
    if (end == nmin - 1) {
      break;
    }
  }
  if (best.match_len < nmin) {
    return std::nullopt;
  }
  return best;
}

/// Replace the draft head only for a strong match or when a shorter match is
/// corroborated by agree_min drafter tokens. Repeat the available continuation
/// as needed to fill the head.
template<typename Token>
[[nodiscard]] std::size_t FuseSuffixLookupHead(
    std::span<Token> draft, std::span<const Token> history,
    SuffixLookupMatch match, std::size_t nmin = kLookupNMin,
    std::size_t nstrong = kLookupNStrong,
    std::size_t agree_min = kLookupAgreeMin) {
  if (nmin == 0 || match.match_len < nmin || history.size() < 2 ||
      match.match_end >= history.size() - 1) {
    return 0;
  }

  const std::size_t source = match.match_end + 1;
  const std::size_t period = history.size() - source;
  if (period == 0) {
    return 0;
  }
  if (match.match_len < nstrong) {
    if (agree_min > draft.size()) {
      return 0;
    }
    for (std::size_t index = 0; index < agree_min; ++index) {
      if (draft[index] != history[source + index % period]) {
        return 0;
      }
    }
  }

  for (std::size_t index = 0; index < draft.size(); ++index) {
    draft[index] = history[source + index % period];
  }
  return draft.size();
}

/// Lookup drafting v2 (upstream gufo PR #295 shape): the `count` tokens that followed
/// the longest earlier occurrence of `seq`'s suffix (at least nmin long), with
/// a short period repeating as in FuseSuffixLookupHead. Empty when no match.
/// `seq` is the committed history plus the drafter's tokens, so the result
/// continues the draft rather than replacing it.
template<typename Token>
[[nodiscard]] std::vector<Token> LookupContinuation(std::span<const Token> seq,
                                                    std::size_t count,
                                                    std::size_t nmin,
                                                    std::size_t nmax) {
  std::vector<Token> tail;
  const auto match = FindSuffixLookupMatch(seq, nmin, nmax);
  if (count == 0 || !match || match->match_end + 1 >= seq.size()) return tail;
  const std::size_t source = match->match_end + 1;
  const std::size_t period = seq.size() - source;
  tail.reserve(count);
  for (std::size_t index = 0; index < count; ++index)
    tail.push_back(seq[source + index % period]);
  return tail;
}

/// The minimum match a lookup draft needs at `position`: `base`, raised to
/// `cooldown_match` until `cooldown_until` after a mostly rejected lookup
/// round (PR #570). Shared by the appended tail and the per-round router.
[[nodiscard]] inline std::size_t LookupMinMatchAt(std::size_t base,
                                                  std::uint32_t position,
                                                  std::uint32_t cooldown_until,
                                                  std::size_t cooldown_match) {
  return position < cooldown_until ? std::max(base, cooldown_match) : base;
}

}  // namespace gufo::hip::dflash

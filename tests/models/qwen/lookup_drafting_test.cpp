#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

#include "src/models/qwen/hip/dflash/lookup.hpp"

namespace {

using Token = std::uint32_t;
namespace lookup = gufo::hip::dflash;

int failures = 0;

void Check(bool ok, const char* what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) {
    ++failures;
  }
}

void OverlapUsesItsPeriod() {
  const std::array<Token, 6> history{1, 2, 1, 2, 1, 2};
  const auto match =
      lookup::FindSuffixLookupMatch<Token>(std::span<const Token>{history});
  Check(match.has_value() && match->match_len == 4 && match->match_end == 3,
        "overlap: repeated period matches the suffix");
  if (!match) {
    return;
  }

  std::array<Token, 4> agreeing_draft{1, 2, 92, 93};
  const std::size_t copied = lookup::FuseSuffixLookupHead<Token>(
      std::span<Token>{agreeing_draft}, std::span<const Token>{history},
      *match);
  const std::array<Token, 4> expected{1, 2, 1, 2};
  Check(copied == 4 && agreeing_draft == expected,
        "overlap: agreement permits periodic continuation to fill the head");

  std::array<Token, 4> disagreeing_draft{90, 91, 92, 93};
  const std::size_t rejected = lookup::FuseSuffixLookupHead<Token>(
      std::span<Token>{disagreeing_draft}, std::span<const Token>{history},
      *match);
  const std::array<Token, 4> unchanged{90, 91, 92, 93};
  Check(rejected == 0 && disagreeing_draft == unchanged,
        "overlap: short match is rejected without drafter agreement");
}

void NoMatchReturnsEmpty() {
  const std::array<Token, 8> history{0, 1, 2, 3, 4, 5, 6, 7};
  const auto match =
      lookup::FindSuffixLookupMatch<Token>(std::span<const Token>{history});
  Check(!match.has_value(), "no repeated suffix of length NMIN");
}

void MostRecentTieWins() {
  const std::array<Token, 15> history{0, 1, 2, 3, 4, 9, 1, 2,
                                      3, 4, 8, 1, 2, 3, 4};
  const auto match =
      lookup::FindSuffixLookupMatch<Token>(std::span<const Token>{history});
  Check(match.has_value() && match->match_len == 4 && match->match_end == 9,
        "equal-length matches choose the latest earlier occurrence");
  Check(match.has_value() && history[match->match_end + 1] == 8,
        "latest occurrence supplies its own continuation");
}

void FusionRuleTable() {
  const std::array<Token, 12> history{0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  struct Case {
    const char* name;
    lookup::SuffixLookupMatch match;
    std::array<Token, 4> draft;
    std::array<Token, 4> expected;
    std::size_t expected_copied;
  };
  const std::array<Case, 5> cases{{
      {"strong match overrides without drafter agreement",
       {8, 2},
       {90, 91, 92, 93},
       {4, 5, 6, 7},
       4},
      {"short match overrides when first two tokens agree",
       {5, 2},
       {4, 5, 90, 91},
       {4, 5, 6, 7},
       4},
      {"short match is rejected when drafter disagrees",
       {5, 2},
       {0, 1, 90, 91},
       {0, 1, 90, 91},
       0},
      {"match below NMIN is rejected",
       {2, 2},
       {0, 1, 90, 91},
       {0, 1, 90, 91},
       0},
      {"short match repeats a one-token continuation",
       {5, 10},
       {12, 12, 14, 15},
       {12, 12, 12, 12},
       4},
  }};

  for (const auto& test : cases) {
    auto draft = test.draft;
    const std::size_t copied = lookup::FuseSuffixLookupHead<Token>(
        std::span<Token>{draft}, std::span<const Token>{history}, test.match);
    Check(copied == test.expected_copied && draft == test.expected, test.name);
  }
}

// The continuation of the history + draft suffix, appended after
// the drafter's rows, never a rewrite of them.
void ContinuationAppendsAfterDraft() {
  // "1 2 3 4 5 6" appeared before; the drafter's 5 6 extends the suffix to
  // a length-6 match, so lookup continues with what followed it: 7 8 9.
  const std::vector<Token> seq = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6};
  const auto tail = lookup::LookupContinuation<Token>(
      std::span<const Token>{seq}, 3, 4, 64);
  Check(tail == std::vector<Token>({7, 8, 9}), "continuation after draft");
  const auto none = lookup::LookupContinuation<Token>(
      std::span<const Token>{seq}, 3, 12, 64);
  Check(none.empty(), "min match 12 rejects a 6-token match");
  const std::vector<Token> loop = {4, 4, 4, 4};
  const auto repeat = lookup::LookupContinuation<Token>(
      std::span<const Token>{loop}, 5, 2, 64);
  Check(repeat == std::vector<Token>({4, 4, 4, 4, 4}), "period repeats");
}

// Lookup router (GUFO_DEV_LOOKUP_ROUTE): a round drafts from lookup alone
// only when the history suffix matches at least the route minimum, and the
// #570 cooldown raises that minimum after a rejected lookup round.
void RouterThresholdAndCooldown() {
  // A 16-token span repeats once: the suffix matches 16 tokens, and what
  // followed the earlier copy (100) is the routed draft.
  std::vector<Token> seq;
  for (Token t = 1; t <= 16; ++t) seq.push_back(t);
  seq.push_back(100);
  for (Token t = 1; t <= 16; ++t) seq.push_back(t);
  const std::span<const Token> history{seq};
  Check(lookup::LookupContinuation<Token>(history, 4, 16, 64).size() == 4 &&
            lookup::LookupContinuation<Token>(history, 4, 16, 64)[0] == 100,
        "route: a 16-token match routes at minimum 16");
  Check(lookup::LookupContinuation<Token>(history, 4, 17, 64).empty(),
        "route: a 16-token match stays on the drafter at minimum 17");

  Check(lookup::LookupMinMatchAt(16, 10, 0, 48) == 16,
        "route: no cooldown keeps the base minimum");
  Check(lookup::LookupMinMatchAt(16, 10, 42, 48) == 48,
        "route: inside the cooldown the minimum rises to 48");
  Check(lookup::LookupMinMatchAt(16, 42, 42, 48) == 16,
        "route: the cooldown ends at its frontier");
  Check(lookup::LookupMinMatchAt(64, 10, 42, 48) == 64,
        "route: cooldown never lowers a larger base");
  const std::size_t cooled = lookup::LookupMinMatchAt(16, 10, 42, 48);
  Check(lookup::LookupContinuation<Token>(history, 4, cooled, 64).empty(),
        "route: the cooldown keeps a 16-token match on the drafter");
}

}  // namespace

int main() {
  OverlapUsesItsPeriod();
  NoMatchReturnsEmpty();
  MostRecentTieWins();
  FusionRuleTable();
  ContinuationAppendsAfterDraft();
  RouterThresholdAndCooldown();
  return failures == 0 ? 0 : 1;
}

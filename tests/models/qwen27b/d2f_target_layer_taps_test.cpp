/// The target-layer tap allocation must be sized by the CONSTANT.
///
/// Found in review: arena.cpp sized d_target_layer_features with a LITERAL 5, the
/// old DFlash2 tap cap. An earlier change raised kMaxTargetLayerTaps to 8 and moved the
/// memory ESTIMATE onto the constant, but not the allocation, so a DSpark draft
/// capturing 8 taps copied 8 * hidden floats into room for 5. The copy returned
/// hipErrorInvalidValue — an API copy is bounds-checked against the known
/// allocation, where a kernel overrun faults instead — and because the size is
/// fixed per call it reproduced at every prompt length, which is what made it
/// look unrelated to the tap count.
///
/// WHY A SOURCE CHECK AND NOT AN ALLOCATION ASSERT: the failing copy happens on
/// a GPU path that needs a loaded 17 GiB model and a live draft backend. What
/// must not come back is the LITERAL — the defect is two sites holding one
/// number, and the cheap oracle is that they hold the same name. A test that
/// loads the model cannot run in CI; this one can, and it fails the moment
/// someone reintroduces a literal.
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::string RepoRoot() {
  const std::string self = __FILE__;
  const std::string tail = "tests/models/qwen27b/d2f_target_layer_taps_test.cpp";
  const auto pos = self.rfind(tail);
  if (pos == std::string::npos) return {};
  return self.substr(0, pos);
}

std::string Slurp(const std::string& path) {
  std::ifstream in(path);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

/// Source with EVERY whitespace byte removed, so a structural search cannot be
/// defeated by a line wrap. The first version of this test searched the raw
/// text for "d_target_layer_features, kMaxTargetLayerTaps" and failed on a
/// correctly formatted two-line call — a test that fails on correct code is
/// worse than no test, because the next reader trusts it.
std::string StripWhitespace(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    if (!std::isspace(static_cast<unsigned char>(c))) out += c;
  }
  return out;
}

/// The allocation and the constant must live in the SAME file, or a literal can
/// drift back in without this test noticing the constant moved.
void AllocationUsesTheConstant() {
  const std::string root = RepoRoot();
  Check(!root.empty(), "located the repository root from __FILE__");
  if (root.empty()) return;

  const std::string arena =
      Slurp(root + "src/models/qwen/hip/arena.cpp");
  Check(!arena.empty(), "read arena.cpp");
  if (arena.empty()) return;
  const std::string flat = StripWhitespace(arena);

  const auto at = flat.find("d_target_layer_features,kMaxTargetLayerTaps");
  Check(at != std::string::npos,
        "d_target_layer_features is sized by kMaxTargetLayerTaps");

  // The literal must be gone. This is the assertion that would have caught the
  // original defect: a number here is a second source of truth.
  const auto literal = flat.find("d_target_layer_features,5*");
  Check(literal == std::string::npos,
        "no literal tap count survives at the d_target_layer_features allocation");

  // Every hipMalloc site of this buffer uses the same expression. The window
  // spans BOTH directions: hipMalloc( precedes the name and the size expression
  // follows it, so a forward-only or backward-only window misses half of it.
  std::size_t sites = 0;
  std::size_t bad = 0;
  for (std::size_t pos = flat.find("&d_target_layer_features");
       pos != std::string::npos;
       pos = flat.find("&d_target_layer_features", pos + 1)) {
    const std::size_t from = pos > 60 ? pos - 60 : 0;
    const std::string around = flat.substr(from, 200);
    if (around.find("hipMalloc") == std::string::npos) continue;
    ++sites;
    if (around.find("kMaxTargetLayerTaps") == std::string::npos) ++bad;
  }
  Check(sites == 1,
        "exactly one hipMalloc site for this buffer (found " +
            std::to_string(sites) + ")");
  Check(bad == 0,
        "every allocation site sizes by kMaxTargetLayerTaps (" +
            std::to_string(bad) + " do not)");
}

/// The constant must still be 8, and the estimate must still read it — the pair
/// that was already correct, kept honest so the next raise cannot half-apply.
void ConstantAndEstimateAgree() {
  const std::string root = RepoRoot();
  if (root.empty()) return;
  const std::string arena = Slurp(root + "src/models/qwen/hip/arena.cpp");
  if (arena.empty()) return;

  Check(arena.find("constexpr std::size_t kMaxTargetLayerTaps = 8;") !=
            std::string::npos,
        "kMaxTargetLayerTaps is still 8");
  const auto estimate = arena.find("CheckedMultiply(kMaxTargetLayerTaps, hidden)");
  Check(estimate != std::string::npos,
        "the memory ESTIMATE still reads the constant (the half that WAS updated)");

  // D1 M2 raised the cap to 8; anything that still assumes 5 taps anywhere in
  // the tree is the same defect wearing a different hat.
  const std::string executor =
      Slurp(root + "src/models/qwen/hip/batched_decode.cpp");
  Check(executor.find("5 * hidden_size") == std::string::npos,
        "batched_decode.cpp has no 5-tap assumption either");
}

}  // namespace

int main() {
  AllocationUsesTheConstant();
  ConstantAndEstimateAgree();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
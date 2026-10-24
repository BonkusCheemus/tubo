// The two BATCHED entries must not inherit the prefill admission.
//
// WHY THIS TEST EXISTS AND WHY IT IS NOT THE EXISTING ONE.
// qwen_turbo4_dev_decode_switch_test already enumerates every CheckReset() call
// site, but it asserts an ANNOTATION, not a behaviour: a site is kDecode if and
// only if it is in a fixed table. ForwardTokenBatch is deliberately absent from
// that table (it has no Turbo4 route in any callee), so the test asserted it
// "uses the refusing default" — and that sentence was TRUE when it was written.
// Unit E then gave kPrefill an admission arm (kv_allocation.hpp, the
// GUFO_DEV_TURBO4_PREFILL switch), which made the default stop refusing. The
// annotation test could not see that, because the annotation did not change.
// Only the behaviour changed, so only a behavioural test can catch it.
//
// WHAT IS ASSERTED. For each of the two batched entries, read the phase it
// ACTUALLY passes to CheckReset (no argument means the kPrefill default), then
// require the guard to REFUSE a Turbo4 policy with that phase while the prefill
// switch is set — the exact state every T arm runs in. Before E6.0 both entries
// resolve to kPrefill and the guard admits, so this fails; after, each names its
// own phase and the guard refuses.
//
// Reading the phase out of the source rather than hardcoding it is what makes
// this a test of the CALL SITE. A hardcoded kBatchedToken here would keep
// passing if someone deleted the annotation, which is the failure mode that
// produced E3 reopening C1-R1.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "src/models/qwen/hip/kv_allocation.hpp"

namespace fs = std::filesystem;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvExecutionPhase;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

/// The message the guard throws for `storage`, or "" when it does not throw.
std::string RefusalMessage(QwenKvCacheStorage storage,
                           QwenKvExecutionPhase phase) {
  try {
    gufo::hip::RequireExecutableKvStorage(storage, phase);
  } catch (const std::exception &e) {
    return e.what();
  }
  return {};
}

std::string RepoRoot() {
  const std::string self = __FILE__;
  const std::string tail = "tests/models/qwen/e6_batched_phase_refusal_test.cpp";
  const auto pos = self.rfind(tail);
  if (pos == std::string::npos) return {};
  return self.substr(0, pos);
}

/// Resolves a line to the entry point it sits in. A definition starts in column
/// 0 and names the executor method; mirrors the existing switch test so the two
/// classify the tree the same way.
std::string EnclosingEntry(const std::string &line) {
  if (line.empty() || line[0] == ' ' || line[0] == '\t') return {};
  const std::string marker = "QwenGpuExecutor::";
  const auto at = line.find(marker);
  if (at == std::string::npos) return {};
  const auto open = line.find('(', at);
  if (open == std::string::npos) return {};
  return line.substr(at + marker.size(), open - at - marker.size());
}

/// The phase named on `entry`'s CheckReset line, or "" when it passes none.
/// kPrefill is spelled "DEFAULTED" because that is the fact under test: an
/// unannotated entry rides the kPrefill default, and after unit E that default
/// is an admission rather than a refusal.
std::string PhaseOfEntry(const fs::path &batched_decode, const std::string &entry) {
  std::ifstream in(batched_decode);
  if (!in) return {};
  std::string line;
  std::string current;
  while (std::getline(in, line)) {
    const std::string owner = EnclosingEntry(line);
    if (!owner.empty()) current = owner;
    if (current != entry) continue;
    if (line.find("CheckReset(") == std::string::npos) continue;
    const auto after = line.find("CheckReset(");
    const auto close = line.find(')', after);
    if (close == std::string::npos) return {};
    const std::string args = line.substr(after + 11, close - after - 11);
    static const std::string kPrefix = "QwenKvExecutionPhase::";
    if (args.find(kPrefix) == std::string::npos) return "DEFAULTED";
    // Take the prefix's own length rather than a hand-counted constant: at 23
    // this silently ate the leading "k" and every phase then failed to resolve.
    std::string phase = args.substr(args.find(kPrefix) + kPrefix.size());
    while (!phase.empty() && (phase.back() == ' ' || phase.back() == ',')) {
      phase.pop_back();
    }
    return phase;
  }
  return {};
}

/// Resolve the phase token found in the source to the enum value it names.
/// Before E6.0 the two entries resolved to kPrefill and this fallback was the
/// whole point of the test. After it, an unrecognised token is a NEW phase
/// nobody has taught this test about, and treating that as kPrefill would let a
/// future phase be written and pass unseen. So an unknown token now fails
/// loudly instead of silently inheriting prefill's meaning.
QwenKvExecutionPhase PhaseFromToken(const std::string &token, bool &known) {
  known = true;
  if (token == "kDecode") return QwenKvExecutionPhase::kDecode;
  if (token == "kNoKvAccess") return QwenKvExecutionPhase::kNoKvAccess;
  if (token == "kBatchedToken") return QwenKvExecutionPhase::kBatchedToken;
  if (token == "kVerify") return QwenKvExecutionPhase::kVerify;
  // kPrefill and the unannotated default are the same entry as far as this test
  // is concerned: both mean "whatever kPrefill currently means", and the whole
  // question is whether that is a refusal.
  if (token == "kPrefill" || token == "DEFAULTED") {
    return QwenKvExecutionPhase::kPrefill;
  }
  known = false;
  return QwenKvExecutionPhase::kPrefill;
}

/// ---------------------------------------------------------------- the defect
///
/// One body, both entries. The prefill switch is ON, exactly as every T arm
/// runs it; the entry must still refuse.
void BatchedEntriesRefuseWhenPrefillSwitchIsOn() {
  const std::string root = RepoRoot();
  Check(!root.empty(), "call site: located the repository root from __FILE__");
  if (root.empty()) return;

  const fs::path batched_decode =
      fs::path(root) / "src/models/qwen/hip/batched_decode.cpp";
  Check(fs::is_regular_file(batched_decode), "call site: batched_decode.cpp exists");
  if (!fs::is_regular_file(batched_decode)) return;

  ::setenv("GUFO_DEV_TURBO4_PREFILL", "1", 1);
  Check(::getenv("GUFO_DEV_TURBO4_PREFILL") != nullptr,
        "prefill switch: set for this case");

  for (const char *entry : {"ForwardTokenBatch", "ForwardVerificationBatch"}) {
    const std::string token = PhaseOfEntry(batched_decode, entry);
    Check(!token.empty(),
          std::string("call site: ") + entry + " has a CheckReset guard");
    if (token.empty()) continue;

    const std::string expected_phase =
        std::string(entry) == "ForwardVerificationBatch" ? "kVerify"
                                                          : "kBatchedToken";
    Check(token == expected_phase,
          std::string("call site: ") + entry + " explicitly names " +
              expected_phase + ", got: " + token);

    bool known = false;
    const QwenKvExecutionPhase phase = PhaseFromToken(token, known);
    Check(known,
          std::string("call site: ") + entry + " names a phase this test knows (" +
              token + ") -- teach the test before adding a phase");

    const std::string message = RefusalMessage(QwenKvCacheStorage::kTurbo4, phase);
    Check(!message.empty(),
          std::string("REFUSED: ") + entry + " (phase " + token +
              ") refuses a Turbo4 policy while GUFO_DEV_TURBO4_PREFILL=1");

    // The hole was silent, so the refusal must also be traceable: an operator
    // reading the thrown text has to be able to tell WHICH entry was refused.
    Check(message.find(entry) != std::string::npos,
          std::string("names the entry: the ") + entry +
              " refusal names " + entry + " itself, got: " + message);
  }

  ::unsetenv("GUFO_DEV_TURBO4_PREFILL");
}

/// ------------------------------------------------------- the switch is not a hole
///
/// The control for the case above: with the switch OFF, kPrefill must still
/// refuse, so E6.0 cannot be passed by simply widening the switch.
void VerifyUsesDecodeSwitchOnly() {
  ::unsetenv("GUFO_DEV_TURBO4_PREFILL");
  ::unsetenv("GUFO_DEV_TURBO4_DECODE");
  const auto storage = QwenKvCacheStorage::kTurbo4;
  const auto phase = QwenKvExecutionPhase::kVerify;
  Check(!RefusalMessage(storage, phase).empty(),
        "verify: refuses Turbo4 when decode switch is unset");
  ::setenv("GUFO_DEV_TURBO4_PREFILL", "1", 1);
  Check(!RefusalMessage(storage, phase).empty(),
        "verify: prefill switch alone cannot admit verification");
  ::setenv("GUFO_DEV_TURBO4_DECODE", "1", 1);
  Check(RefusalMessage(storage, phase).empty(),
        "verify: decode switch admits verification");
  ::unsetenv("GUFO_DEV_TURBO4_PREFILL");
  ::unsetenv("GUFO_DEV_TURBO4_DECODE");
}

void PrefillStillRefusesWithSwitchOff() {
  ::unsetenv("GUFO_DEV_TURBO4_PREFILL");
  const std::string off = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                         QwenKvExecutionPhase::kPrefill);
  Check(!off.empty(),
        "control: kPrefill still refuses a Turbo4 policy with the switch unset");

  ::setenv("GUFO_DEV_TURBO4_PREFILL", "1", 1);
  const std::string on = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                        QwenKvExecutionPhase::kPrefill);
  Check(on.empty(),
        "control: kPrefill IS admitted with the switch on, which is why the "
        "batched entries cannot ride the default");
  ::unsetenv("GUFO_DEV_TURBO4_PREFILL");
}

/// ------------------------------------------------------------ no f16 regression
///
/// E6.0 must not change anything for an f16 or fp32 session: those are
/// executable on every phase and must stay executable on any new one.
void ExecutableStorageIsUnaffected() {
  ::unsetenv("GUFO_DEV_TURBO4_PREFILL");
  ::unsetenv("GUFO_DEV_TURBO4_DECODE");
  Check(RefusalMessage(QwenKvCacheStorage::kFp16, QwenKvExecutionPhase::kPrefill).empty(),
        "f16: still executable on kPrefill, unchanged by E6.0");
  Check(RefusalMessage(QwenKvCacheStorage::kFp32, QwenKvExecutionPhase::kPrefill).empty(),
        "fp32: still executable on kPrefill, unchanged by E6.0");
  Check(RefusalMessage(QwenKvCacheStorage::kFp16, QwenKvExecutionPhase::kDecode).empty(),
        "f16: still executable on kDecode, unchanged by E6.0");
}

}  // namespace

int main() {
  BatchedEntriesRefuseWhenPrefillSwitchIsOn();
  VerifyUsesDecodeSwitchOnly();
  PrefillStillRefusesWithSwitchOff();
  ExecutableStorageIsUnaffected();

  if (g_failures != 0) {
    std::printf("\n%d CHECK(S) FAILED\n", g_failures);
    return 1;
  }
  std::printf("\nALL CHECKS PASSED\n");
  return 0;
}
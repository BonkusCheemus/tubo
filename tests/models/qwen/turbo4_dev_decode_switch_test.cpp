// The GUFO_DEV_TURBO4_DECODE dev switch.
//
// Three things are asserted here, and they are not the same kind of claim:
//
//   1. The switch is OFF by default. With the variable unset, a Turbo4 policy is
//      refused on BOTH phases.
//   2. With the switch ON, the PREFILL phase still refuses, and refuses with the
//      BYTE-IDENTICAL message it produces with the switch off. The oracle is the
//      switch-off message itself, not a copy of the string pasted into this
//      file: a pasted copy only proves the copy is still true.
//   3. THE CALL SITE. Part 1 and 2 call the guard directly, which is the HELPER.
//      Three review rounds were lost to tests that proved a helper and
//      left the defect in the caller, so part 3 is the one that matters: it
//      enumerates every CheckReset() invocation in the real executor sources,
//      resolves each to its enclosing entry point, and asserts the phase
//      annotation matches a fixed table.
//
// WHY PART 3 IS STATIC RATHER THAN EXECUTED. ForwardToken needs a hipStream, an
// arena and loaded weights; it cannot run on CPU while the GPU is busy. Grepping
// the shipped source is the strongest available oracle here, and it is
// BIDIRECTIONAL: annotating a prefill entry with kDecode fails, AND dropping
// kDecode from ForwardToken fails. A one-way grep would only catch the mistake
// that was already made.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

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

void SetSwitch(const char *value) {
  if (value == nullptr) {
    ::unsetenv("GUFO_DEV_TURBO4_DECODE");
  } else {
    ::setenv("GUFO_DEV_TURBO4_DECODE", value, 1);
  }
}

/// ---------------------------------------------------------------- part 1 + 2

void SwitchIsOffByDefault() {
  SetSwitch(nullptr);
  Check(::getenv("GUFO_DEV_TURBO4_DECODE") == nullptr,
        "switch: variable is unset in the default state under test");

  const std::string decode = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                            QwenKvExecutionPhase::kDecode);
  Check(decode.empty() == false,
        "switch OFF: Turbo4 decode is refused even on the decode phase");

  const std::string prefill = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                             QwenKvExecutionPhase::kPrefill);
  Check(prefill.empty() == false,
        "switch OFF: Turbo4 prefill is refused");
  // C1e CHANGED THIS ASSERTION, and the change is deliberate. It used to
  // require prefill and decode to be byte-identical, which is exactly the
  // property that made the refusal useless: the same sentence was served for
  // every phase, so a reader could not tell which entry had been refused. The
  // switch-off behaviour that still must hold byte-for-byte is the one in
  // PrefillStillRefusesWithSwitchOn — the same phase says the same thing in
  // both switch states. What changed is that each phase now NAMES itself.
  Check(prefill.find("the prefill phase") != std::string::npos,
        "switch OFF: the prefill refusal names the prefill phase");
  Check(decode.find("the decode phase") != std::string::npos,
        "switch OFF: the decode refusal names the decode phase and the switch");
  Check(decode.find("GUFO_DEV_TURBO4_DECODE unset") != std::string::npos,
        "switch OFF: the decode refusal says WHY it refused (switch unset)");
  Check(prefill != decode,
        "switch OFF: the two phases no longer produce the identical refusal");

  const std::string no_kv = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                          QwenKvExecutionPhase::kNoKvAccess);
  Check(no_kv.empty() == false,
        "switch OFF: a no-KV entry still refuses a Turbo4 policy");
  Check(no_kv.find("no-KV-access entry") != std::string::npos,
        "switch OFF: the no-KV refusal names the no-KV phase");
  Check(no_kv != prefill && no_kv != decode,
        "switch OFF: all three phases refuse distinctly");

  Check(decode.find("block pitch") != std::string::npos,
        "switch OFF: refusal names the 66-byte block pitch (production text)");
}

void NoKvAccessNeedsTheSameSwitch() {
  SetSwitch(nullptr);
  const std::string off_no_kv = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                               QwenKvExecutionPhase::kNoKvAccess);
  Check(off_no_kv.empty() == false,
        "no-KV: refused while GUFO_DEV_TURBO4_DECODE is unset");

  SetSwitch("1");
  const std::string on_no_kv = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                              QwenKvExecutionPhase::kNoKvAccess);
  Check(on_no_kv.empty(),
        "no-KV: accepted for KNOWN block-addressed storage with the switch on");

  // The new phase is not a wider door: an unknown precision keeps its own
  // refusal, exactly as before.
  const std::string unknown_no_kv = RefusalMessage(
      static_cast<QwenKvCacheStorage>(200), QwenKvExecutionPhase::kNoKvAccess);
  Check(unknown_no_kv.empty() == false,
        "no-KV: an unknown precision still refuses with the switch on");
  Check(unknown_no_kv.find("not a known precision") != std::string::npos,
        "no-KV: an unknown precision keeps its distinct refusal text");

  // fp16/fp32 are executable and must stay accepted on the new phase too.
  Check(RefusalMessage(QwenKvCacheStorage::kFp16,
                       QwenKvExecutionPhase::kNoKvAccess)
            .empty(),
        "no-KV: fp16 passes with the switch on, as on every other phase");

  // Switch-off bytes are stable for the new phase, mirroring prefill.
  SetSwitch(nullptr);
  Check(RefusalMessage(QwenKvCacheStorage::kTurbo4,
                       QwenKvExecutionPhase::kNoKvAccess) == off_no_kv,
        "no-KV: the refusal is byte-identical in both switch states");
  SetSwitch(nullptr);
}

void PrefillStillRefusesWithSwitchOn() {
  SetSwitch(nullptr);
  const std::string off_prefill = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                                 QwenKvExecutionPhase::kPrefill);

  SetSwitch("1");
  Check(::getenv("GUFO_DEV_TURBO4_DECODE") != nullptr,
        "switch ON: variable is set for this case");

  const std::string on_prefill = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                                QwenKvExecutionPhase::kPrefill);
  Check(on_prefill.empty() == false,
        "switch ON: PREFILL still refuses a Turbo4 policy");
  Check(on_prefill == off_prefill,
        "switch ON: the prefill refusal is byte-identical to switch OFF");

  const std::string on_decode = RefusalMessage(QwenKvCacheStorage::kTurbo4,
                                               QwenKvExecutionPhase::kDecode);
  Check(on_decode.empty(),
        "switch ON: the decode phase accepts a Turbo4 policy");

  // The switch must not become a blanket override: an unknown precision is not
  // a Turbo4 policy, so it keeps its own distinct refusal.
  SetSwitch(nullptr);
  const std::string unknown_off = RefusalMessage(
      static_cast<QwenKvCacheStorage>(200), QwenKvExecutionPhase::kPrefill);
  SetSwitch("1");
  const std::string unknown_on = RefusalMessage(
      static_cast<QwenKvCacheStorage>(200), QwenKvExecutionPhase::kPrefill);
  Check(unknown_on.empty() == false && unknown_on == unknown_off,
        "switch ON: an unknown precision still refuses, unchanged");

  // fp16 and fp32 are unaffected in both states.
  SetSwitch("1");
  Check(RefusalMessage(QwenKvCacheStorage::kFp16,
                       QwenKvExecutionPhase::kDecode)
             .empty() &&
            RefusalMessage(QwenKvCacheStorage::kFp32,
                           QwenKvExecutionPhase::kPrefill)
                .empty(),
        "switch ON: fp16 and fp32 pass on both phases, as before");

  SetSwitch(nullptr);
}

/// ---------------------------------------------------------------- part 3
///
/// THE RULE (C1-R1). `kDecode` is legal at an executor entry point only when
/// one of these two is true, and the reason is recorded so it can be checked:
///
///   kTurbo4Route  - the entry executes attention against the KV plane, and a
///                   block-packed read/write path exists in its callees.
///   decodeReadback- the entry does NOT execute attention; it only reads back a
///                   buffer that a kTurbo4Route entry has already produced.
///
/// Why the second kind is sound rather than a loophole: these entries launch no
/// attention and never address the KV plane, so they cannot produce "attention
/// against null f16 planes". What they read IS written on the Turbo4 route --
/// decode_step.cpp writes decode_scratch.logits after the layer loop, on the
/// shared path -- so refusing them would make a Turbo4 decode step unobservable
/// while protecting nothing.
///
/// The ForwardTokenBatch case is the negative example that made this rule
/// necessary: it looked like decode by name, and had no Turbo4 path in any
/// callee. A table written from NAMES hides exactly that, so each entry below
/// names the source symbol that justifies it and the test re-reads that symbol
/// from the tree on every run.

enum class Justification { kTurbo4Route, kDecodeReadback };

struct DecodeEntry {
  const char* entry;
  Justification why;
  const char* evidence_file;   // must contain evidence_symbol
  const char* evidence_symbol;
};

const std::vector<DecodeEntry>& ExpectedDecodeEntries() {
  static const std::vector<DecodeEntry> kEntries = {
      // The justification is unchanged by E7: ForwardToken still carries a
      // block-packed read/write route. The SYMBOL moved -- unit E replaced the
      // per-position LaunchTurbo4KvWrite with the fused LaunchTurbo4DecodePrepare
      // (the write and the decode prepare in one launcher) -- so this pointer is
      // updated to the symbol that now carries the ruling. Recording the rename
      // matters: this entry exists precisely so a table cannot outlive the thing
      // that justified it, and it did fail here for exactly that reason while
      // E7 was in flight.
      {"ForwardToken", Justification::kTurbo4Route,
       "src/models/qwen/hip/decode_step.cpp", "LaunchTurbo4DecodePrepare"},
      {"CopyLastLogits", Justification::kDecodeReadback,
       "src/models/qwen/hip/decode_step.cpp", "LaunchGEMV"},
      {"SampleLastLogits", Justification::kDecodeReadback,
       "src/models/qwen/hip/executor.cpp", "LaunchGPUSampling"},
      {"SampleCachedLogits", Justification::kDecodeReadback,
       "src/models/qwen/hip/executor.cpp", "LaunchGPUSampling"},
      {"CopyLastHidden", Justification::kDecodeReadback,
       "src/models/qwen/hip/executor.cpp", "hipMemcpyAsync"},
  };
  return kEntries;
}

/// Entries that must NEVER be annotated, with the reason. Present so a
/// re-annotation is a named, deliberate act rather than a silent widening.
const std::vector<std::pair<std::string, std::string>>& ForbiddenEntries() {
  static const std::vector<std::pair<std::string, std::string>> kForbidden = {
      {"ForwardTokenBatch",
       "LaunchBatched* kernels write scratch.attention.k/v; no Turbo4 callee"},
      {"ForwardPromptBatch", "prefill"},
      {"ForwardPromptChunk", "prefill"},
      {"GenerateFromPrefix", "runs ForwardPromptBatch internally"},
  };
  return kForbidden;
}

std::string RepoRoot() {
  const std::string self = __FILE__;
  const std::string tail = "tests/models/qwen/turbo4_dev_decode_switch_test.cpp";
  const auto pos = self.rfind(tail);
  if (pos == std::string::npos) return {};
  return self.substr(0, pos);
}

/// Resolves a line to the entry point it sits in. Mirrors the classification
/// used when the table above was derived: a definition starts in column 0 and
/// names the executor method.
std::string EnclosingEntry(const std::string &line) {
  if (line.empty() || line[0] == ' ' || line[0] == '\t') return {};
  const std::string marker = "QwenGpuExecutor::";
  const auto at = line.find(marker);
  if (at == std::string::npos) return {};
  const auto open = line.find('(', at);
  if (open == std::string::npos) return {};
  return line.substr(at + marker.size(), open - at - marker.size());
}

std::string Slurp(const fs::path &path) {
  std::ifstream in(path);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void CallSitesAreClassified() {
  const std::string root = RepoRoot();
  Check(!root.empty(),
        "call site: located the repository root from __FILE__");
  if (root.empty()) return;

  const fs::path hip = fs::path(root) / "src/models/qwen/hip";
  Check(fs::is_directory(hip), "call site: the executor source directory exists");
  if (!fs::is_directory(hip)) return;

  const std::string batch_source =
      Slurp(hip / "batched_decode.cpp");
  Check(batch_source.find(
            "QwenGpuExecutor::ForwardTokenBatch") != std::string::npos &&
            batch_source.find("QwenKvExecutionPhase::kBatchedToken") !=
                std::string::npos,
        "batched token entry remains explicitly kBatchedToken and refused");
  Check(batch_source.find(
            "QwenGpuExecutor::ForwardVerificationBatch") != std::string::npos &&
            batch_source.find("QwenKvExecutionPhase::kVerify") !=
                std::string::npos,
        "verification entry has its separate kVerify phase");

  // Rule check 1: every justification must still point at a symbol that exists.
  // Without this, deleting the Turbo4 route would leave a table that still says
  // "safe" -- the table would outlive the thing that justified it.
  for (const auto &want : ExpectedDecodeEntries()) {
    const std::string body = Slurp(fs::path(root) / want.evidence_file);
    Check(body.find(want.evidence_symbol) != std::string::npos,
          std::string("rule: ") + want.entry + " is justified by " +
              want.evidence_symbol + " in " + want.evidence_file);
  }

  int sites = 0;
  int decode_sites = 0;
  std::set<std::string> seen_decode_entries;

  for (const auto &entry : fs::directory_iterator(hip)) {
    if (entry.path().extension() != ".cpp") continue;
    std::ifstream in(entry.path());
    if (!in) continue;
    std::string line;
    std::string current;
    while (std::getline(in, line)) {
      const std::string owner = EnclosingEntry(line);
      if (!owner.empty()) current = owner;
      if (line.find("CheckReset(") == std::string::npos) continue;

      ++sites;
      const bool annotated_decode =
          line.find("kDecode", line.find("CheckReset(")) != std::string::npos;
      const std::string file = entry.path().filename().string();
      const std::string where = file + ":" + current;

      bool should_be_decode = false;
      for (const auto &want : ExpectedDecodeEntries()) {
        if (current == want.entry) should_be_decode = true;
      }

      if (annotated_decode) ++decode_sites;
      Check(annotated_decode == should_be_decode,
            "call site: " + where +
                (annotated_decode ? " is annotated kDecode" :
                                   " uses the refusing default"));
      if (annotated_decode) seen_decode_entries.insert(current);

      // Rule check 2: a forbidden entry that has been annotated fails loudly
      // and says why, instead of just disagreeing with a table.
      if (annotated_decode) {
        for (const auto &bad : ForbiddenEntries()) {
          if (current == bad.first) {
            Check(false, "rule: " + where + " must NOT be kDecode -- " +
                             bad.second);
          }
        }
      }
    }
  }

  Check(sites >= 20, "call site: found the executor guard sites (" +
                         std::to_string(sites) + ")");
  Check(decode_sites ==
            static_cast<int>(ExpectedDecodeEntries().size()),
        "call site: annotated count matches the justified table (" +
            std::to_string(decode_sites) + " annotated, " +
            std::to_string(ExpectedDecodeEntries().size()) + " justified)");
  for (const auto &want : ExpectedDecodeEntries()) {
    Check(seen_decode_entries.count(want.entry) == 1,
          std::string("call site: justified entry is still annotated: ") +
              want.entry);
  }
}

void ConfigureVisionIsClassifiedByPrompt() {
  // C1e. The whole ruling rests on ONE line of source, so the line is the test.
  // A no-KV claim that quietly becomes unconditional is exactly the silent
  // wrong-output this phase was introduced to prevent, and no runtime check on
  // this box can see it: the interesting case needs a loaded 17 GiB model.
  const std::string root = RepoRoot();
  Check(!root.empty(), "ConfigureVision: located the repository root");
  if (root.empty()) return;

  const std::string executor = Slurp(fs::path(root) /
                                     "src/models/qwen/hip/executor.cpp");
  Check(!executor.empty(), "ConfigureVision: read executor.cpp");

  const auto at = executor.find("QwenGpuExecutor::ConfigureVision");
  Check(at != std::string::npos, "ConfigureVision: found the definition");
  if (at == std::string::npos) return;

  // Scope the search to this function's own body, up to the next definition.
  const auto next = executor.find("\nvoid QwenGpuExecutor::", at + 1);
  const std::string body = executor.substr(
      at, next == std::string::npos ? std::string::npos : next - at);

  Check(body.find("CheckReset(") != std::string::npos,
        "ConfigureVision: still guards through CheckReset");
  Check(body.find("QwenKvExecutionPhase::kNoKvAccess") != std::string::npos,
        "ConfigureVision: names kNoKvAccess (C1e)");
  Check(body.find("QwenKvExecutionPhase::kPrefill") != std::string::npos,
        "ConfigureVision: a real image prompt still names kPrefill");
  Check(body.find("prompt ?") != std::string::npos,
        "ConfigureVision: the phase is chosen BY the prompt, not hardcoded");
  Check(body.find("prompt->rope.Validate") != std::string::npos,
        "ConfigureVision: the vision path itself is unchanged");

  // The switch is the only admission, so the entry must not claim decode.
  Check(body.find("kDecode") == std::string::npos,
        "ConfigureVision: must NOT claim kDecode (C1d feeds vision via decode)");
}

}  // namespace

int main() {
  SwitchIsOffByDefault();
  NoKvAccessNeedsTheSameSwitch();
  PrefillStillRefusesWithSwitchOn();
  ConfigureVisionIsClassifiedByPrompt();
  CallSitesAreClassified();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
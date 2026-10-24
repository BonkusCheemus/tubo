// Feed the prompt one token at a time when prefill must refuse.
//
// WHY. A Turbo4 KV policy is refused by the prefill entry points by design:
// RequireExecutableKvStorage refuses every phase except decode. Without a
// per-token alternative the T arm cannot be run at all. This is the driver that
// provides it, and it is DEV ONLY: f16 must keep taking the batched path, byte
// for byte.
//
// THE DEFECT CLASS IS SILENT, so the assertions are about which engine entry
// actually gets called. A driver that quietly fell back to ForwardPromptBatch,
// or that engaged while the policy was f16, would look perfectly healthy and
// prove nothing. These checks name the call sites in the shipped source rather
// than inferring intent.
//
// SCOPE NOTE, because it is a real deviation. The obvious implementation is to
// loop over TextRunner::Advance. That CANNOT work: Advance is
//   if (!qwen.frontier().has_value()) throw "Qwen decode state has no retained frontier";
// and the frontier is first published BY prefill, so the first prompt token has
// none. The driver therefore calls the same engine entry Advance calls --
// QwenGpuExecutor::ForwardToken -- from inside Prefill, where no frontier is
// required. Same mechanism, same C1 guard (ForwardToken is annotated kDecode,
// so no guard is widened), minus a precondition that cannot be satisfied yet.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "src/cli/serve/turbo4_prompt_feeding.hpp"

namespace fs = std::filesystem;
using gufo::server::FeedQwenPromptChunk;
using gufo::server::QwenPromptFeedingMode;
using gufo::server::SelectQwenPromptFeedingMode;
using gufo::server::TextRunnerToken;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::string Slurp(const fs::path& p) {
  std::ifstream in(p);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

/// Removes comments, keeping string literals. The checks below are about CODE,
/// and this header's doc comments legitimately NAME ForwardPromptBatch and
/// RequireExecutableKvStorage in order to explain why neither is used there. A
/// verifier that matches prose fails on a correct artefact, which is worse than
/// no verifier because it invites re-running until green.
std::string StripComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  bool line = false, block = false, str = false;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
    if (line) {
      if (c == '\n') { line = false; out += c; }
      continue;
    }
    if (block) {
      if (c == '*' && n == '/') { block = false; ++i; }
      continue;
    }
    if (str) {
      out += c;
      if (c == '\\' && n != '\0') { out += n; ++i; continue; }
      if (c == '"') str = false;
      continue;
    }
    if (c == '/' && n == '/') { line = true; ++i; continue; }
    if (c == '/' && n == '*') { block = true; ++i; continue; }
    if (c == '"') { str = true; out += c; continue; }
    out += c;
  }
  return out;
}

std::string RepoRoot() {
  const std::string self = __FILE__;
  const std::string tail = "tests/models/qwen/c1d_turbo4_prompt_feeding_test.cpp";
  const auto pos = self.rfind(tail);
  if (pos == std::string::npos) return {};
  return self.substr(0, pos);
}

/// The shipped prefill must consult the selector and delegate to the driver.
void PrefillConsultsTheDriver() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "call site: located the repository root");
    return;
  }
  const std::string backend =
      Slurp(fs::path(root) / "src/cli/serve/inference_backend.cpp");
  Check(!backend.empty(), "call site: inference_backend.cpp is readable");
  if (backend.empty()) return;

  Check(backend.find("SelectQwenPromptFeedingMode") != std::string::npos,
        "call site: prefill consults the feeding-mode selector");
  Check(backend.find("FeedQwenPromptChunk") != std::string::npos,
        "call site: prefill delegates to the driver");
  Check(backend.find("SelectQwenPromptFeedingMode(") != std::string::npos &&
            backend.find("Prefill(") != std::string::npos,
        "call site: the selector is consulted from the Qwen prefill path");
}

/// The driver itself must exist as a header, or the call site check above is
/// checking a symbol that nothing defines.
void DriverHeaderExists() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "driver: located the repository root");
    return;
  }
  const fs::path header =
      fs::path(root) / "src/cli/serve/turbo4_prompt_feeding.hpp";
  Check(fs::exists(header), "driver: turbo4_prompt_feeding.hpp exists");
  if (!fs::exists(header)) return;
  const std::string body = StripComments(Slurp(header));
  Check(body.find("SelectQwenPromptFeedingMode") != std::string::npos,
        "driver: the header defines the mode selector");
  Check(body.find("FeedQwenPromptChunk") != std::string::npos,
        "driver: the header defines the feeding driver");
  Check(body.find("ForwardPromptBatch") == std::string::npos,
        "driver: the driver code does not name the batched engine entry, so "
        "the choice is made by mode and not by the driver hard-coding prefill");
}

/// The C1 guard must stay decode-only. If feeding the prompt needed prefill to
/// be relaxed, the annotation would be wrong somewhere -- so assert the driver
/// adds no prefill allowance and never touches the guard.
void NoGuardWasWidened() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "guard: located the repository root");
    return;
  }
  const std::string header =
      StripComments(Slurp(fs::path(root) / "src/cli/serve/turbo4_prompt_feeding.hpp"));
  Check(!header.empty(), "guard: the driver header is readable");
  if (header.empty()) return;
  Check(header.find("RequireExecutableKvStorage") == std::string::npos,
        "guard: the driver never calls the KV guard");
  Check(header.find("QwenExecutionPhase") == std::string::npos,
        "guard: the driver never widens an execution phase -- feeding the "
        "prompt goes through decode, which is already permitted");
}

/// The behavioural half, and the part that matters: WHICH engine entry is
/// called. Recording callbacks stand in for the executor, so this is observable
/// with no device and no model.
///
/// A driver that quietly fell back to the batched path, or that engaged while
/// the policy was f16, would pass every source check above and still be wrong.
struct Calls {
  int batch_calls = 0;
  int token_calls = 0;
  int published = 0;
  std::size_t batch_start = 0;
  bool batch_logits = false;
  std::vector<std::pair<TextRunnerToken, std::pair<std::size_t, bool>>> tokens;
  TextRunnerToken published_value = 0;
};

Calls RunFeed(QwenPromptFeedingMode mode,
              const std::vector<TextRunnerToken>& prompt, std::size_t offset,
              std::size_t consumed, bool decode_ready) {
  Calls calls;
  FeedQwenPromptChunk(
      mode, std::span<const TextRunnerToken>(prompt.data(), prompt.size()),
      offset, consumed, decode_ready,
      [&](std::span<const TextRunnerToken> chunk, std::uint32_t start,
          bool want_logits) {
        ++calls.batch_calls;
        calls.batch_start = start;
        calls.batch_logits = want_logits;
        return static_cast<TextRunnerToken>(chunk.back() + 2000U);
      },
      [&](TextRunnerToken token, std::uint32_t position, bool want_logits) {
        ++calls.token_calls;
        calls.tokens.emplace_back(token,
                                  std::pair<std::size_t, bool>{position,
                                                              want_logits});
        return static_cast<TextRunnerToken>(token + 1000U);
      },
      [&](TextRunnerToken value) {
        ++calls.published;
        calls.published_value = value;
      });
  return calls;
}

void ModeFollowsStorageNotTheEnvironment() {
  using gufo::hip::QwenKvCacheStorage;
  unsetenv("GUFO_DEV_TURBO4_PREFILL");
  Check(SelectQwenPromptFeedingMode(QwenKvCacheStorage::kTurbo4) ==
            QwenPromptFeedingMode::kTokenByToken,
        "mode: a Turbo4 policy feeds per token");
  Check(SelectQwenPromptFeedingMode(QwenKvCacheStorage::kFp16) ==
            QwenPromptFeedingMode::kBatched,
        "mode: an f16 policy stays batched");
  Check(SelectQwenPromptFeedingMode(QwenKvCacheStorage::kFp32) ==
            QwenPromptFeedingMode::kBatched,
        "mode: an fp32 policy stays batched");
  Check(SelectQwenPromptFeedingMode(static_cast<QwenKvCacheStorage>(99)) ==
            QwenPromptFeedingMode::kBatched,
        "mode: an unknown storage does NOT enable the dev route");
}

void PrefillSwitchSelectsTurbo4Batching() {
  using gufo::hip::QwenKvCacheStorage;
  unsetenv("GUFO_DEV_TURBO4_PREFILL");
  Check(SelectQwenPromptFeedingMode(QwenKvCacheStorage::kTurbo4) ==
            QwenPromptFeedingMode::kTokenByToken,
        "Turbo4 prefill unset: keep token-by-token fallback");
  setenv("GUFO_DEV_TURBO4_PREFILL", "1", 1);
  Check(SelectQwenPromptFeedingMode(QwenKvCacheStorage::kTurbo4) ==
            QwenPromptFeedingMode::kBatched,
        "Turbo4 prefill enabled: use batched prompt feeding");
  setenv("GUFO_DEV_TURBO4_PREFILL", "true", 1);
  Check(SelectQwenPromptFeedingMode(QwenKvCacheStorage::kTurbo4) ==
            QwenPromptFeedingMode::kTokenByToken,
        "Turbo4 prefill accepts only the exact switch value 1");
  unsetenv("GUFO_DEV_TURBO4_PREFILL");
}

void F16TakesTheBatchedPathUnchanged() {
  using gufo::hip::QwenKvCacheStorage;
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13};
  const Calls calls = RunFeed(
      SelectQwenPromptFeedingMode(QwenKvCacheStorage::kFp16), prompt, 0, 4,
      /*decode_ready=*/true);

  Check(calls.batch_calls == 1, "f16: exactly ONE batched call (" +
                                     std::to_string(calls.batch_calls) + ")");
  Check(calls.token_calls == 0,
        "f16: ZERO per-token calls (" + std::to_string(calls.token_calls) + ")");
  Check(calls.batch_start == 0 && calls.batch_logits,
        "f16: the batched call carries the chunk start and the logits flag");
  Check(calls.published == 1 && calls.published_value == prompt.back() + 2000U,
        "f16: the frontier published is the batched call's returned value, "
        "not the chunk's first token");
}

void Turbo4FeedsEveryTokenWithLogitsOnlyOnTheLast() {
  using gufo::hip::QwenKvCacheStorage;
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13};
  const Calls calls = RunFeed(
      SelectQwenPromptFeedingMode(QwenKvCacheStorage::kTurbo4), prompt, 0, 4,
      /*decode_ready=*/true);

  Check(calls.batch_calls == 0,
        "turbo4: ZERO batched calls (" + std::to_string(calls.batch_calls) +
            ") -- a batched call is the silent fallback");
  Check(calls.token_calls == 4,
        "turbo4: one call per token (" + std::to_string(calls.token_calls) +
            ")");
  bool ordered = true;
  bool logits_only_last = true;
  for (std::size_t i = 0; i < calls.tokens.size(); ++i) {
    if (calls.tokens[i].first != prompt[i]) ordered = false;
    if (calls.tokens[i].second.first != i) ordered = false;
    const bool expect = (i + 1 == prompt.size());
    if (calls.tokens[i].second.second != expect) logits_only_last = false;
  }
  Check(ordered, "turbo4: tokens arrive in order at their own positions");
  Check(logits_only_last,
        "turbo4: compute_logits is true only for the LAST token, matching "
        "ForwardPromptBatch's one-logits-step-per-chunk semantics");
  Check(calls.published == 1 && calls.published_value == prompt.back() + 1000U,
        "turbo4: the frontier published is the LAST token's returned value, "
        "not the token itself");
}

void ChunkedPrefillRespectsTheOffset() {
  using gufo::hip::QwenKvCacheStorage;
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13};

  // Mid-prompt chunk: offset 2, two tokens, not yet decode_ready.
  const Calls mid = RunFeed(
      SelectQwenPromptFeedingMode(QwenKvCacheStorage::kTurbo4), prompt, 2, 2,
      /*decode_ready=*/false);
  Check(mid.token_calls == 2 && mid.published == 0,
        "chunk: a non-final chunk publishes no frontier");
  bool positions_absolute = mid.tokens.size() == 2 &&
                            mid.tokens[0].second.first == 2 &&
                            mid.tokens[1].second.first == 3;
  Check(positions_absolute,
        "chunk: token positions are absolute, continuing from the offset");
  bool no_logits = true;
  for (const auto& t : mid.tokens) {
    if (t.second.second) no_logits = false;
  }
  Check(no_logits,
        "chunk: a non-final chunk computes no logits at all");

  const Calls f16_mid = RunFeed(
      SelectQwenPromptFeedingMode(QwenKvCacheStorage::kFp16), prompt, 2, 2,
      /*decode_ready=*/false);
  Check(f16_mid.batch_calls == 1 && f16_mid.batch_start == 2 &&
            !f16_mid.batch_logits,
        "chunk: f16 still batches, from the same offset, without logits");
}

void EmptyChunkIsANoOp() {
  const std::vector<TextRunnerToken> prompt{};
  const Calls calls =
      RunFeed(QwenPromptFeedingMode::kTokenByToken, prompt, 0, 0, false);
  Check(calls.batch_calls == 0 && calls.token_calls == 0 && calls.published == 0,
        "empty: a zero-length chunk calls nothing");
}

}  // namespace

int main() {
  PrefillConsultsTheDriver();
  DriverHeaderExists();
  NoGuardWasWidened();
  ModeFollowsStorageNotTheEnvironment();
  PrefillSwitchSelectsTurbo4Batching();
  F16TakesTheBatchedPathUnchanged();
  Turbo4FeedsEveryTokenWithLogitsOnlyOnTheLast();
  ChunkedPrefillRespectsTheOffset();
  EmptyChunkIsANoOp();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

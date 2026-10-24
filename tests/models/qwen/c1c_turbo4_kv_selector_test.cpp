// GUFO_DEV_TURBO4_KV selects a Turbo4 KV policy.
//
// WHY THIS EXISTS. The C1 dev switch GUFO_DEV_TURBO4_DECODE only PERMITS a
// Turbo4 policy past RequireExecutableKvStorage. Nothing ever REQUESTED one:
// `kv_cache_storage = kTurbo4` appeared zero times in src/, Production() is a
// constexpr `{}` defaulting to kFp16, and so a "Turbo4 arm" ran a second f16 arm.
// Its dump came out byte-identical to the f16 arm. That is the failure this test
// exists to make impossible to repeat silently: env set, storage still f16.
//
// WHAT IS ASSERTED.
//   1. unset  -> the resolved policy is f16, and is FIELD-FOR-FIELD Production().
//   2. set    -> kv_cache_storage is kTurbo4 and nothing else differs.
//   3. only the exact string "1" turns it on. A typo fails CLOSED to f16 rather
//      than silently selecting a format the caller did not ask for.
//   4. Production() itself is untouched and still constexpr. It must not become
//      an env reader: it is the compile-time default for every declaration.
//   5. NO CONSTRUCTION PATH CAN REACH BARE Production(). This is the check that
//      matters, and it is a source check on purpose. QwenGpuExecutor has three
//      factory/constructor entry points whose policy parameter DEFAULT to
//      Production(), and the CLI and bench callers pass no policy at all -- so a
//      selector placed at the serve call site alone would leave those paths on
//      f16 with the env set and no error anywhere. An invariant test cannot see
//      that; only reading the shipped construction sites can.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"

namespace fs = std::filesystem;
using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenKvCacheStorage;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

const char* Name(QwenKvCacheStorage s) {
  switch (s) {
    case QwenKvCacheStorage::kFp32: return "kFp32";
    case QwenKvCacheStorage::kFp16: return "kFp16";
    case QwenKvCacheStorage::kTurbo4: return "kTurbo4";
  }
  return "?";
}

std::string Slurp(const fs::path& p) {
  std::ifstream in(p);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

std::string StripComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  bool line = false, str = false;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
    if (line) {
      if (c == '\n') { line = false; out += c; }
      continue;
    }
    if (str) { if (c == '"') str = false; continue; }
    if (c == '/' && n == '/') { line = true; ++i; continue; }
    if (c == '"') { str = true; continue; }
    out += c;
  }
  return out;
}

/// Removes comments but KEEPS string literals. Needed to count env reads: the
/// variable name lives inside a literal, so a stripper that also drops strings
/// counts zero and reports a perfect single-read site as none at all.
std::string StripCommentsOnly(const std::string& src) {
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
  const std::string tail = "tests/models/qwen/c1c_turbo4_kv_selector_test.cpp";
  const auto pos = self.rfind(tail);
  if (pos == std::string::npos) return {};
  return self.substr(0, pos);
}

// --- behaviour ------------------------------------------------------------

void UnsetIsFp16AndIdenticalToProduction() {
  ::unsetenv("GUFO_DEV_TURBO4_KV");
  const QwenExecutionPolicy resolved = gufo::hip::ResolveRuntimeQwenExecutionPolicy();
  Check(resolved.kv_cache_storage == QwenKvCacheStorage::kFp16,
        std::string("unset: resolved storage is f16 (") +
            Name(resolved.kv_cache_storage) + ")");

  const QwenExecutionPolicy production = QwenExecutionPolicy::Production();
  Check(resolved.fuse_qk_norm_rope_kv == production.fuse_qk_norm_rope_kv &&
            resolved.kv_cache_storage == production.kv_cache_storage &&
            resolved.recurrent_state_storage == production.recurrent_state_storage &&
            resolved.kv_layers.layer_count == production.kv_layers.layer_count,
        "unset: the resolved policy is field-for-field Production()");
}

void SetIsTurbo4AndChangesNothingElse() {
  const QwenExecutionPolicy production = QwenExecutionPolicy::Production();

  ::setenv("GUFO_DEV_TURBO4_KV", "1", 1);
  const QwenExecutionPolicy on = gufo::hip::ResolveRuntimeQwenExecutionPolicy();
  Check(on.kv_cache_storage == QwenKvCacheStorage::kTurbo4,
        std::string("set: resolved storage is turbo4 (") +
            Name(on.kv_cache_storage) + ")");
  Check(on.fuse_qk_norm_rope_kv == production.fuse_qk_norm_rope_kv &&
            on.recurrent_state_storage == production.recurrent_state_storage &&
            on.kv_layers.layer_count == production.kv_layers.layer_count,
        "set: ONLY kv_cache_storage differs from Production()");
  // The layer table stays empty, which is what makes kv_cache_storage describe
  // the WHOLE cache. Populating it would be a mixed-cache product decision.
  Check(on.kv_layers.layer_count == 0U,
        "set: the per-layer table stays empty, so the whole cache is Turbo4");

  // It must be readable, not merely allocatable: this is the property that
  // distinguishes a Turbo4 plane from an f16 plane at every offset.
  Check(!gufo::hip::QwenKvIsBlockAddressed(QwenKvCacheStorage::kFp16) &&
            gufo::hip::QwenKvIsBlockAddressed(QwenKvCacheStorage::kTurbo4),
        "set: the selected storage is the block-addressed one");
}

void OnlyExactlyOneTurnsItOn() {
  const QwenExecutionPolicy production = QwenExecutionPolicy::Production();
  for (const char* value : {"", "0", "true", "yes", "2", "01", " 1", "1 "}) {
    ::setenv("GUFO_DEV_TURBO4_KV", value, 1);
    const QwenExecutionPolicy p = gufo::hip::ResolveRuntimeQwenExecutionPolicy();
    Check(p.kv_cache_storage == production.kv_cache_storage,
          std::string("fails closed: value \"") + value +
              "\" leaves the production storage (" + Name(p.kv_cache_storage) +
              ")");
  }
  ::unsetenv("GUFO_DEV_TURBO4_KV");
}

// --- the construction sites ----------------------------------------------

/// Every policy default in the executor must resolve through the env-aware
/// helper. A bare Production() default is exactly the silent-f16 path.
void NoConstructionPathFallsBackToBareProduction() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "call site: located the repository root");
    return;
  }
  const std::string executor =
      StripComments(Slurp(fs::path(root) / "src/models/qwen/hip/executor.hpp"));
  Check(!executor.empty(), "call site: executor.hpp is readable");
  if (executor.empty()) return;

  // Count the policy parameters that still default to bare Production().
  std::size_t bare_defaults = 0;
  std::size_t resolved_defaults = 0;
  std::size_t at = 0;
  while ((at = executor.find("QwenExecutionPolicy policy =", at)) !=
         std::string::npos) {
    const std::size_t end = executor.find(';', at);
    const std::string line = executor.substr(at, end - at);
    if (line.find("ResolveRuntimeQwenExecutionPolicy()") != std::string::npos) {
      ++resolved_defaults;
    } else if (line.find("QwenExecutionPolicy::Production()") != std::string::npos) {
      ++bare_defaults;
      if (bare_defaults <= 3) {
        std::printf("       bare default: %s\n", line.c_str());
      }
    }
    at = end == std::string::npos ? at + 1 : end + 1;
  }
  Check(bare_defaults == 0,
        "call site: no executor policy parameter still defaults to bare "
        "Production() (" + std::to_string(bare_defaults) + " bare, " +
            std::to_string(resolved_defaults) + " resolved)");
  Check(resolved_defaults >= 6,
        "call site: every executor policy default resolves through the helper (" +
            std::to_string(resolved_defaults) + ")");

  // The serve runner builds its policy explicitly, so its own construction is a
  // separate site from the defaults above.
  const std::string backend = StripComments(Slurp(
      fs::path(root) / "src/cli/serve/inference_backend.cpp"));
  Check(!backend.empty(), "call site: inference_backend.cpp is readable");
  if (!backend.empty()) {
    Check(backend.find("execution_policy_(hip::QwenExecutionPolicy::Production())") ==
              std::string::npos,
          "call site: the serve runner no longer constructs its policy from bare "
          "Production()");
    Check(backend.find("execution_policy_(") != std::string::npos &&
              backend.find("ResolveRuntimeQwenExecutionPolicy()") !=
                  std::string::npos,
          "call site: the serve runner resolves through the env-aware helper");
  }
}

/// Production() must stay constexpr and must not read the environment. If it
/// grew a getenv, every default in the tree would silently depend on process
/// state, including the ones this test cannot see.
void ProductionStaysPureAndConstexpr() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "purity: located the repository root");
    return;
  }
  const std::string policy =
      Slurp(fs::path(root) / "src/models/qwen/hip/execution_policy.hpp");
  Check(!policy.empty(), "purity: execution_policy.hpp is readable");
  if (policy.empty()) return;

  const auto at = policy.find("static constexpr QwenExecutionPolicy Production()");
  Check(at != std::string::npos,
        "purity: Production() is still declared static constexpr");
  // A window, not "up to the next }": `return {};` itself contains a brace, so
  // slicing to the first one truncates the body mid-statement and makes the
  // next check fail against a correct file.
  const std::string window = policy.substr(at, 200);
  Check(window.find("getenv") == std::string::npos,
        "purity: Production() does not read the environment");
  Check(window.find("return {};") != std::string::npos,
        "purity: Production() still returns the plain default");

  // The env read must live in exactly one place, so there is one thing to audit.
  // Counted over CODE ONLY: the resolver's own doc comment names the variable,
  // and a prose mention is not a second read.
  const std::string code = StripCommentsOnly(policy);
  std::size_t reads = 0;
  std::size_t at2 = 0;
  while ((at2 = code.find("GUFO_DEV_TURBO4_KV", at2)) != std::string::npos) {
    ++reads;
    at2 += 1;
  }
  Check(reads == 1,
        "purity: exactly one place in the policy header reads the variable (" +
            std::to_string(reads) + ")");
}

// --- the selection receipt ----------------------------------------------

/// Proof 1 and proof 2 in one host-callable string: the run names its storage
/// and states the bytes it allocated. f16 and Turbo4 differ by ~7.76x at the
/// same element count, so a T arm that prints the f16 number is visibly not
/// Turbo4 no matter what the operator believes.
void ReceiptNamesStorageAndBytes() {
  const std::size_t elements = 1000U * 4096U;  // a ctx-4096 sized plane
  QwenExecutionPolicy policy = QwenExecutionPolicy::Production();

  const std::string f16 = gufo::hip::DescribeQwenKvSelection(policy, elements);
  Check(f16.find("kv storage=fp16") != std::string::npos,
        "receipt: the f16 run names fp16");
  Check(f16.find("bits_per_value=16") != std::string::npos,
        "receipt: the f16 run reports 16 bits per value");

  policy.kv_cache_storage = QwenKvCacheStorage::kTurbo4;
  const std::string turbo4 =
      gufo::hip::DescribeQwenKvSelection(policy, elements);
  Check(turbo4.find("kv storage=turbo4") != std::string::npos,
        "receipt: the Turbo4 run names turbo4");
  Check(turbo4.find("bits_per_value=4.125") != std::string::npos,
        "receipt: the Turbo4 run reports 4.125 bits per value");

  // The two numbers must be genuinely different, or the receipt proves nothing.
  const std::size_t f16_bytes = elements * 2U;
  const std::size_t turbo4_bytes = gufo::hip::QwenKvBytesForElements(
      QwenKvCacheStorage::kTurbo4, elements);
  Check(f16.find("bytes=" + std::to_string(f16_bytes)) != std::string::npos,
        "receipt: the f16 byte count is in the line");
  Check(turbo4.find("bytes=" + std::to_string(turbo4_bytes)) !=
            std::string::npos,
        "receipt: the Turbo4 byte count is in the line");
  Check(turbo4_bytes != f16_bytes,
        "receipt: the two arms report DIFFERENT byte counts (" +
            std::to_string(f16_bytes) + " vs " + std::to_string(turbo4_bytes) +
            ")");

  // An unknown storage must not be reported as a confident number.
  policy.kv_cache_storage = static_cast<QwenKvCacheStorage>(99);
  const std::string bogus =
      gufo::hip::DescribeQwenKvSelection(policy, elements);
  Check(bogus.find("storage=unknown") != std::string::npos &&
            bogus.find("bytes=0") != std::string::npos,
        "receipt: an unknown storage is named, not silently sized");
}

/// ------------------------------------------------------------------ C1f
///
/// Found in review: the selector once left kv_layers EMPTY believing that meant "the whole
/// cache". `QwenKvLayerIsBlockAddressed` returns false when layer_count == 0, so
/// the empty table routed EVERY layer to the element-wise f16 reader. Under
/// Turbo4 the f16 planes are null, so the engine ran the fused f16 KV write and
/// then handed the f16 attention kernel a nil pointer (the measured nil-address
/// fault). The empty table ALSO fed graph rejection - the loop at
/// execution_policy.hpp:486 sets no bit when layer_count is 0 - so Turbo4 stayed
/// graph-eligible while reading null planes. Two of these assertions are about
/// graph eligibility because that was the second, quieter half of the same bug.
///
/// 16 attention layers is this model's geometry (interval 4 over 64 blocks), and
/// 34603008 is the accepted Turbo4 arena size for that geometry.

using gufo::hip::QwenKvExecutionPhase;
using gufo::hip::QwenKvLayerFormat;
using gufo::hip::QwenKvReaderKind;
using gufo::hip::ResolveQwenGraphRejectionsForPolicy;
using gufo::hip::ResolveQwenKvLayerTableForStorage;
using gufo::hip::ResolveQwenLayerKvPlan;

constexpr std::uint32_t kAttentionLayers = 16U;

QwenExecutionPolicy BlockAddressedEmptyTable() {
  QwenExecutionPolicy policy = QwenExecutionPolicy::Production();
  policy.kv_cache_storage = QwenKvCacheStorage::kTurbo4;
  policy.kv_layers.layer_count = 0U;
  return policy;
}

void FillMakesEveryLayerBlockPacked() {
  const QwenExecutionPolicy filled = ResolveQwenKvLayerTableForStorage(
      BlockAddressedEmptyTable(), kAttentionLayers);
  Check(filled.kv_layers.layer_count == kAttentionLayers,
        "C1f: the table is filled to the model's attention-layer count (" +
            std::to_string(filled.kv_layers.layer_count) + ")");
  Check(filled.kv_layers.IsUniform(QwenKvLayerFormat::kTurbo4),
        "C1f: every filled layer is K=V=Turbo4 (uniform, not mixed)");
  Check(QwenKvCacheStorageIsBlockAddressed(filled.kv_cache_storage),
        "C1f: the whole-cache storage is still Turbo4 after the fill");

  // The point of the ruling, per layer.
  for (std::uint32_t i = 0; i < kAttentionLayers; ++i) {
    const auto plan = ResolveQwenLayerKvPlan(filled, i, /*head_dim_fusion=*/true);
    const std::string at = "layer " + std::to_string(i);
    Check(plan.reader == QwenKvReaderKind::kBlockPacked, at + ": reader is block-packed");
    Check(plan.fuse_qk_norm_rope_kv == false,
          at + ": the fused f16 KV kernel is refused (it has no 66-byte-block variant)");
    Check(plan.graph_eligible == false, at + ": a block layer is graph-ineligible");
  }

  // Graph rejection must be non-empty now; the empty table left it empty, which
  // is how Turbo4 stayed capturable while reading null planes.
  const auto rejected = ResolveQwenGraphRejectionsForPolicy(
      filled, /*compute_logits=*/true, /*split_k_attention=*/false,
      /*graph_enabled=*/true);
  Check(rejected != gufo::hip::QwenGraphRejection::kNone,
        "C1f: graph rejection is non-empty once the table is filled");
}

void EmptyTableWithBlockStorageThrows() {
  // The invariant, stated as a refusal: no attention layers to fill from.
  bool threw = false;
  std::string what;
  try {
    (void)ResolveQwenKvLayerTableForStorage(BlockAddressedEmptyTable(), 0U);
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  Check(threw, "C1f invariant: block-addressed storage with no fillable layer count THROWS");
  Check(what.find("attention layers") != std::string::npos,
        "C1f invariant: the refusal says how many attention layers were seen");

  // Over the table cap is the same class of mistake and must also refuse.
  threw = false;
  try {
    (void)ResolveQwenKvLayerTableForStorage(
        BlockAddressedEmptyTable(),
        gufo::hip::QwenKvFormatTable::kMaxLayers + 1U);
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw, "C1f invariant: a layer count past the table cap THROWS, never truncates");
}

void Fp16AndFp32EmptyTableUnchanged() {
  for (const QwenKvCacheStorage storage :
       {QwenKvCacheStorage::kFp16, QwenKvCacheStorage::kFp32}) {
    QwenExecutionPolicy policy = QwenExecutionPolicy::Production();
    policy.kv_cache_storage = storage;
    policy.kv_layers.layer_count = 0U;
    const QwenExecutionPolicy out =
        ResolveQwenKvLayerTableForStorage(policy, kAttentionLayers);
    const std::string label =
        storage == QwenKvCacheStorage::kFp16 ? "fp16" : "fp32";
    Check(out.kv_layers.layer_count == 0U,
          "C1f: " + label + " with an empty table stays empty (unchanged)");
    // ...and it must not have become block-addressed behind our back.
    const auto plan = ResolveQwenLayerKvPlan(out, 0U, true);
    Check(plan.reader == QwenKvReaderKind::kElementWise,
          "C1f: " + label + " still routes element-wise");
  }
}

void Turbo4ArenaReceiptUnchanged() {
  // 34603008 is the accepted Turbo4 arena receipt. C1f fills a TABLE, which
  // must not move a single allocated byte.
  //
  // The element count is DERIVED from the model's real geometry, not invented:
  // executor.hpp:382-386 GetAttentionKvPlaneElements is
  // FullAttentionLayerCount * num_key_value_heads * max_context * head_dim,
  // which for Qwen3.8-27B at ctx 4096 is 16 * 4 * 4096 * 256 = 67,108,864, and
  // 67,108,864 values at 4.125 bits is EXACTLY 34,603,008 bytes. An earlier
  // version of this test guessed the count and got 33792, which is how a
  // receipt assertion starts testing the test.
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kHeadDim = 256U;
  constexpr std::size_t kMaxContext = 4096U;
  const std::size_t elements =
      static_cast<std::size_t>(kAttentionLayers) * kKvHeads * kMaxContext *
      kHeadDim;
  Check(elements == 67108864U,
        "C1f: the plane element count matches the executor formula (" +
            std::to_string(elements) + ")");
  const std::size_t bytes =
      gufo::hip::QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4, elements);
  Check(bytes == 34603008U,
        "C1f: the Turbo4 plane byte count is still 34603008 (got " +
            std::to_string(bytes) + ")");

  // And the invariant that actually matters: the fill moves no bytes. Compare
  // the receipt for the SAME storage with and without a populated table.
  const std::string before = gufo::hip::DescribeQwenKvSelection(
      BlockAddressedEmptyTable(), elements);
  const std::string after = gufo::hip::DescribeQwenKvSelection(
      ResolveQwenKvLayerTableForStorage(BlockAddressedEmptyTable(),
                                        kAttentionLayers),
      elements);
  Check(before == after,
        "C1f: filling the table leaves the selection receipt byte-identical");
  Check(after.find("storage=turbo4") != std::string::npos &&
            after.find("34603008") != std::string::npos,
        "C1f: the receipt still reads storage=turbo4 bytes=34603008");
}

void SwitchUnsetIsByteIdenticalToProduction() {
  ::unsetenv("GUFO_DEV_TURBO4_KV");
  const QwenExecutionPolicy resolved =
      gufo::hip::ResolveRuntimeQwenExecutionPolicy();
  const QwenExecutionPolicy production = QwenExecutionPolicy::Production();
  Check(resolved.kv_cache_storage == production.kv_cache_storage,
        "C1f: switch unset -> storage is Production's");
  Check(resolved.kv_layers.layer_count == production.kv_layers.layer_count &&
            std::memcmp(resolved.kv_layers.cells, production.kv_layers.cells,
                        sizeof(resolved.kv_layers.cells)) == 0,
        "C1f: switch unset -> the whole per-layer table is byte-identical to Production()");
  Check(resolved.Fingerprint() == production.Fingerprint(),
        "C1f: switch unset -> the policy fingerprint is Production's");

  // And the fill is a no-op on the production policy, so the ctor cannot change
  // an f16 run by accident.
  const QwenExecutionPolicy filled = ResolveQwenKvLayerTableForStorage(
      production, kAttentionLayers);
  Check(filled.kv_layers.layer_count == 0U &&
            filled.Fingerprint() == production.Fingerprint(),
        "C1f: the fill is a no-op for non-block-addressed storage");
}

}  // namespace

int main() {
  for (const auto* name : {"GUFO_DEV_TURBO4_KV", "GUFO_DEV_TURBO_KV_BITS",
                           "GUFO_DEV_TURBO_KV_K_BITS", "GUFO_DEV_TURBO_KV_V_BITS"})
    ::unsetenv(name);
  UnsetIsFp16AndIdenticalToProduction();
  SetIsTurbo4AndChangesNothingElse();
  OnlyExactlyOneTurnsItOn();
  NoConstructionPathFallsBackToBareProduction();
  ProductionStaysPureAndConstexpr();
  ReceiptNamesStorageAndBytes();
  FillMakesEveryLayerBlockPacked();
  EmptyTableWithBlockStorageThrows();
  Fp16AndFp32EmptyTableUnchanged();
  Turbo4ArenaReceiptUnchanged();
  SwitchUnsetIsByteIdenticalToProduction();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
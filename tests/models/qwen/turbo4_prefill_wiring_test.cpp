// The switch-unset path is byte-identical, and a
// block layer stays graph-ineligible.
//
// RULE 5 is "switch unset -> f16 path byte-identical". What has to be true for
// that, precisely:
//
//   * `turbo4_layer` must be FALSE on an f16 run, so every ternary added to the
//     three attention launches takes its f16 arm and no f16 byte changes;
//   * the fused QK-Norm+RoPE+KV decision must be UNCHANGED, so the override I
//     added (`!turbo4_layer && ...`) does not quietly disable the fused kernel
//     for everybody;
//   * no new allocation happens on the f16 path, because the device word that
//     E1's writer needs is created only on the first block layer.
//
// A full byte-identical run needs the model on a GPU and belongs to E4. What is
// checked HERE is every DECISION those bytes depend on, plus a source pin that
// the f16 pair can no longer reach a block-layer launch except through the
// E3 helper. That is the part which is cheap to get wrong and expensive to
// discover on a device.
//
// RULE 6 is "graph stays ineligible for block layers". A captured graph replays
// with its scalar arguments frozen, and this route's are not all frozen values:
// the dequant range is start_pos + batch_size and the position itself is staged
// on the device. Capturing it would replay a stale range.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace fs = std::filesystem;
using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvLayerFormat;
using gufo::hip::QwenKvReaderKind;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
  if (!ok) ++g_failures;
}

std::string Slurp(const fs::path& p) {
  std::ifstream in(p);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

std::string StripComments(std::string src) {
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
    if (str) {
      if (c == '"') str = false;
      continue;
    }
    if (c == '/' && n == '/') { line = true; continue; }
    if (c == '/' && n == '*') {
      // Skip to the closing */ and RESUME, whatever the comment looked like.
      // The first version of this function fell through to `line = true` for a
      // single-star /*, which silently ate the rest of the line -- and on
      // `/*skip_kv_write=*/turbo4_layer ? kTurbo4SkipKvWrite` that hid a real
      // call site and made the count 2 instead of 3. A comment stripper that
      // deletes code is worse than no comment stripper.
      i += 2;
      while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/')) ++i;
      i += 1;
      continue;
    }
    if (c == '"') str = true;
    out += c;
  }
  return out;
}

// Whitespace is FLATTENED, not preserved. A pin that matches only when the
// expression happens to sit on one line fails on a reformat and passes on a
// behaviour change, which is the wrong way round.
std::string Flatten(std::string s) {
  std::string out;
  for (const char c : s)
    if (c != ' ' && c != '\n' && c != '\t' && c != '\r') out += c;
  return out;
}

std::string PrefillSource() {
  return Flatten(StripComments(Slurp("src/models/qwen/hip/prefill_chunk.cpp")));
}

std::size_t Count(const std::string& hay, const std::string& needle) {
  std::size_t n = 0;
  for (std::size_t at = hay.find(needle); at != std::string::npos;
       at = hay.find(needle, at + needle.size())) {
    ++n;
  }
  return n;
}

// RULE 5, decision half. The switch-unset default is Production(); c1c's
// selector test already proves that resolves to f16, so what matters here is
// what the PREFILL kv plan does with that policy.
void SwitchUnsetTakesEveryF16Arm() {
  const auto production = QwenExecutionPolicy::Production();
  Check(production.kv_cache_storage == QwenKvCacheStorage::kFp16,
        "with the switch unset the policy is f16 -- so the block arm below is "
        "unreachable on this path");

  const auto plan = gufo::hip::ResolveQwenLayerKvPlan(production, 0, true);
  Check(plan.reader == QwenKvReaderKind::kElementWise,
        "an f16 policy resolves to the ELEMENT-WISE reader, so turbo4_layer is "
        "false in prefill_chunk.cpp and every added ternary takes its f16 arm");
  Check(plan.graph_eligible,
        "an f16 layer stays graph-eligible: the block route's ineligibility must "
        "not leak into the default path");
  Check(plan.fuse_qk_norm_rope_kv == production.fuse_qk_norm_rope_kv,
        "the fused QK-Norm+RoPE+KV decision is UNCHANGED when the policy is f16 "
        "-- the `!turbo4_layer &&` override must not disable the fused kernel "
        "for everybody");
}

void BlockLayerOverridesFusionAndGraphs() {
  // The TABLE, not just the whole-cache enumerator: QwenKvLayerIsBlockAddressed
  // reads kv_layers, so setting kv_cache_storage alone leaves every layer
  // element-wise and the block arm is never entered. Setting only the
  // enumerator here made this check pass for the wrong reason once already.
  auto policy = QwenExecutionPolicy::Production();
  policy.kv_cache_storage = QwenKvCacheStorage::kTurbo4;
  policy.kv_layers.layer_count = 16;
  for (std::uint32_t layer = 0; layer < policy.kv_layers.layer_count; ++layer) {
    policy.kv_layers.Set(layer, QwenKvLayerFormat::kTurbo4,
                         QwenKvLayerFormat::kTurbo4);
  }
  policy.fuse_qk_norm_rope_kv = true;  // asked for, must still be refused
  const auto plan = gufo::hip::ResolveQwenLayerKvPlan(policy, 0, true);
  Check(plan.reader == QwenKvReaderKind::kBlockPacked,
        "a Turbo4 policy resolves to the BLOCK reader, which is what makes "
        "turbo4_layer true");
  Check(!plan.fuse_qk_norm_rope_kv,
        "the fused kernel is REFUSED for a block layer even when the policy asks "
        "for it: it writes the cache at an element pitch and has no 66-byte "
        "block variant");
  Check(!plan.graph_eligible,
        "a block layer is graph-INELIGIBLE (rule 6): the dequant range and the "
        "staged position would be frozen into a captured graph");
}

// RULE 5, wiring half. The f16 pair must reach an attention launch ONLY through
// the E3 helper on a block layer -- otherwise one of the three sites keeps the
// old all-layers pointer and the scratch work buys nothing.
void EveryLaunchGoesThroughThePair() {
  // The source is WHITESPACE-FLATTENED, so every needle must be flattened too.
  // A pin written with spaces matches nothing against a flattened haystack and
  // then fails for a formatting reason, which is a test that lies.
  const std::string src = PrefillSource();
  if (src.empty()) {
    Check(false, "prefill_chunk.cpp was readable from the project root");
    return;
  }
  Check(Count(src, Flatten("turbo4_layer ? turbo4_pair.k : "
                           "arena_.d_attention_kv_f16")) == 3,
        "all THREE attention launches take the f16 K through the pair, never "
        "unconditionally from the arena");
  Check(Count(src, Flatten("turbo4_layer ? turbo4_pair.v")) == 3,
        "all THREE take the f16 V through the pair too -- one layer on, not the "
        "all-layers total_k");
  Check(Count(src, Flatten("turbo4_layer ? nullptr : arena_.d_kv_cache")) == 3,
        "all THREE pass a null f32 pair on a block layer, so "
        "attention_batched.hip:36-62 cannot store f32 over the blocks");
  Check(Count(src, Flatten("turbo4_layer ? kTurbo4AttentionLayerIdx : "
                           "attn_layer_idx")) == 3,
        "all THREE pass layer index 0 on a block layer and the real index "
        "otherwise");
  Check(Count(src, "kTurbo4SkipKvWrite") >= 3,
        "all THREE set skip_kv_write on a block layer");
  Check(src.find("PrepareTurbo4PrefillLayer") != std::string::npos,
        "prefill calls the one preparation helper (write, then dequant)");
  // THE SILENT-CORRUPTION GUARD. The fused QK-Norm+RoPE+KV kernel writes the
  // cache at an ELEMENT pitch and has no 66-byte-block variant, so running it
  // for a block layer writes f16-shaped bytes into a block plane with no
  // refusal anywhere. The resolver already refuses it; this line is prefill's
  // OWN use of the resolver, and without the override a block layer would ask
  // `route_plan` -- the un-overridden policy toggle -- and get yes.
  Check(src.find(Flatten("!turbo4_layer && route_plan.fuse_qk_norm_rope_kv")) !=
            std::string::npos,
        "prefill OVERRIDES the fused-kernel toggle for a block layer rather "
        "than consulting it -- element-pitch writes over 66-byte blocks are "
        "silent corruption");
  Check(src.find("ScopedDeviceWord") == std::string::npos,
        "prefill does not hipMalloc/hipMemcpy/hipFree a temporary start word");
  Check(Count(src, Flatten("hipMemsetD32Async(arena_.d_turbo4_first_pos, "
                           "start_pos, 1, arena_.stream)")) == 1,
        "the arena-owned position word is staged asynchronously on its stream");
  Check(src.find(Flatten(
            "kv_source_bytes, arena_.d_turbo4_first_pos, visible_context")) !=
            std::string::npos,
        "the batched writer receives the arena-owned position word");
}

void TheOrderIsWriteThenDequant() {
  const std::string src = PrefillSource();
  // Two anchors, and they are NOT the same thing: the helper DEFINITION holds
  // the two launches, the CALL site is later in the file. Searching after the
  // call for the launches finds nothing, which is what happened the first time
  // this check was written.
  const std::string def_anchor = Flatten("bool PrepareTurbo4PrefillLayer(");
  const std::size_t definition = src.find(def_anchor);
  const std::size_t call = src.find(Flatten("PrepareTurbo4PrefillLayer(arena_"));
  const std::size_t wmma = src.find("LaunchQwenWmmaAttention(");
  const std::size_t tile = src.find("LaunchBatchedAttentionTile(");
  const std::size_t batched = src.find("LaunchBatchedAttention(");

  Check(definition != std::string::npos,
        "the preparation helper is defined in prefill_chunk.cpp");
  if (definition == std::string::npos) return;
  Check(call != std::string::npos && call < wmma && call < tile &&
            call < batched,
        "the write+dequant happens BEFORE any attention launch -- the other "
        "order reads a scratch the dequant has not filled yet");

  const std::string body = src.substr(definition);
  const std::size_t w = body.find("LaunchTurbo4KvWriteBatched(");
  const std::size_t d = body.find("LaunchTurbo4DequantLayer(");
  Check(w != std::string::npos && d != std::string::npos && w < d,
        "inside the helper the E1 batched WRITE precedes the dequant");
  // The ARGUMENT, not the identifier. The loose needle "visible_context)" also
  // matches the helper's own parameter declaration, so it kept passing when the
  // argument had been changed to batch_size -- the exact defect rule 3 exists
  // to prevent. The cast pins the call site.
  Check(body.find(Flatten(
            "static_cast<std::uint32_t>(visible_context), "
            "arena.GetMaxContext()")) != std::string::npos,
        "the dequant ARGUMENT is visible_context = start_pos + batch_size, so "
        "the PREFIX written by earlier chunks is included (rule 3)");
  Check(body.find(Flatten(
            "static_cast<std::uint32_t>(batch_size), "
            "arena.GetMaxContext()")) == std::string::npos,
        "and it is NOT batch_size alone: dequantising only the new chunk would "
        "leave the prefix as whatever the scratch happened to hold");
}

}  // namespace

int main() {
  SwitchUnsetTakesEveryF16Arm();
  BlockLayerOverridesFusionAndGraphs();
  EveryLaunchGoesThroughThePair();
  TheOrderIsWriteThenDequant();
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

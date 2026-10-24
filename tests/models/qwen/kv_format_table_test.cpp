// The per-layer, per-K/V FORMAT TABLE, on CPU.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/kv_format_table_test.cpp
//       -o /tmp/kvft
//
// WHY THIS FILE EXISTS. Unit A made the Turbo4 reader and writer testable. What
// it could not express is the thing the mixed-precision KV experiment exists to test: giving the LAST N
// KV-BEARING layers more precision than the rest. That is not a value of
// `QwenKvCacheStorage` — it is a table — so unit B introduces one and this file
// is its oracle.
//
// THE TRAP THIS FILE IS BUILT AROUND, named because it has already cost us once.
// This model has 65 blocks of which 16 carry a KV cache. "The last 4 layers"
// resolved over BLOCK indices selects blocks 61-64, none of which have a KV
// cache at all; resolved over KV-BEARING layers it selects the last 4 of the
// 16. The identical bug hit the same experiment on another engine, where a positional
// rule reached only 3 KV layers of 8. Every test below pins the difference
// between those two answers, because a table that agrees with the wrong one is
// not a slow experiment, it is a wrong one reported as a clean one.
//
// WHAT IS NOT COVERED HERE: nothing is launched, no device memory is touched,
// and no arena is allocated. This is pure policy arithmetic — sizes, refusals,
// and graph identity — which is exactly the part that can be decided on CPU.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"

namespace p = gufo::hip;

namespace {

int g_failures = 0;

/// Arm-C shape (K Turbo4, V f16), built once for the route checks.
p::QwenKvFormatTable c_arm{};

void Check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

/// THE REAL GEOMETRY, TRANSCRIBED FROM THE MODEL HEADER, not from any
/// production constant. 65 blocks, every 4th is full attention, minus the one
/// MTP block, gives 16 KV-bearing layers.
constexpr std::uint32_t kBlocks = 65;
constexpr std::uint32_t kAttentionLayers = 16;

/// full_attention[il] for the real geometry: block 4, 8, 12, ... and block 0 is
/// the MTP layer, which is why the count is (65-1)/4 rather than 65/4.
using AttentionMask = std::array<bool, kBlocks>;

/// std::array, not std::vector<bool>: the latter is a bit-packed specialisation
/// with no contiguous storage, so it cannot be handed to a `const bool*` API at
/// all.
AttentionMask RealFullAttention() {
  AttentionMask fa{};
  fa.fill(false);
  for (std::uint32_t il = 1; il < kBlocks; il += 4) fa[il] = true;
  return fa;
}

p::QwenKvFormatTable Build(std::uint32_t tail_n,
                           p::QwenKvLayerFormat tail_format,
                           p::QwenKvLayerFormat base_format) {
  const AttentionMask fa = RealFullAttention();
  p::QwenKvFormatTable t;
  const bool ok = p::MakeQwenKvTailFormatTable(
      fa.data(), kBlocks, tail_n, tail_format, tail_format, base_format,
      base_format, &t);
  if (!ok) t = p::QwenKvFormatTable{};
  return t;
}

// ---------------------------------------------------------------------------

void CountsKvBearingLayersNotBlocks() {
  // THE HEADLINE CHECK. A tail of 4 must produce a table of 16 layers whose
  // last 4 are the tail — not a table of 65 whose last 4 are blocks 61-64.
  const auto t = Build(4, p::QwenKvLayerFormat::kFp16,
                       p::QwenKvLayerFormat::kTurbo4);
  Check(t.layer_count == kAttentionLayers,
        "B1: the table is sized by KV-BEARING layers (16), not by blocks (65) — "
        "the step-1 trap, asserted as a number");

  std::uint32_t tail = 0;
  for (std::uint32_t i = 0; i < t.layer_count; ++i)
    if (t.K(i) == p::QwenKvLayerFormat::kFp16) ++tail;
  Check(tail == 4,
        "B1: a tail of 4 makes EXACTLY 4 layers f16, not 7 and not 4 blocks — "
        "the ordinal predicate is pinned");

  // ...and they are the LAST four, contiguously. A table that put the tail in
  // the middle would have the same count and a completely different meaning.
  Check(t.K(t.layer_count - 1) == p::QwenKvLayerFormat::kFp16 &&
            t.K(t.layer_count - 2) == p::QwenKvLayerFormat::kFp16 &&
            t.K(t.layer_count - 3) == p::QwenKvLayerFormat::kFp16 &&
            t.K(t.layer_count - 4) == p::QwenKvLayerFormat::kFp16 &&
            t.K(t.layer_count - 5) == p::QwenKvLayerFormat::kTurbo4,
        "B1: the f16 tail is the LAST 4 layers contiguously, and layer 11 is "
        "still Turbo4");

  // THE WRONG READING IS NOT A TOTAL MISS, WHICH IS WHY IT IS DANGEROUS.
  // Taking "the last 4" over BLOCK indices instead of KV-bearing layers lands on
  // blocks 61-64, of which exactly ONE (61) carries a KV cache. So the buggy
  // rule does not crash and does not produce an obviously empty table — it
  // produces a plausible one that has quietly promoted ONE layer instead of
  // four, and an experiment run on it reports a number for the wrong arm.
  std::uint32_t kv_blocks_in_last_4 = 0;
  for (std::uint32_t il = kBlocks - 4; il < kBlocks; ++il)
    if (RealFullAttention()[il]) ++kv_blocks_in_last_4;
  Check(kv_blocks_in_last_4 == 1,
        "B1: the WRONG reading (last 4 BLOCKS) selects exactly 1 KV layer on "
        "this geometry, not 4 — a near miss that looks like a working table, "
        "which is why it is pinned as a number rather than left to review");

  // A TAIL OF EVERY KV LAYER IS THE WHOLE CACHE, and must say so.
  const auto all = Build(kAttentionLayers, p::QwenKvLayerFormat::kFp16,
                         p::QwenKvLayerFormat::kTurbo4);
  Check(all.IsUniform(p::QwenKvLayerFormat::kFp16),
        "B1: a tail covering every KV layer yields a UNIFORM f16 table, which is "
        "the arm-D reference (acceptance with no KV quant)");
}

void RefusesRatherThanClamping() {
  const AttentionMask fa = RealFullAttention();
  p::QwenKvFormatTable t;

  // ONE MORE THAN EXISTS. Clamping is how "the last 8" silently becomes "the
  // last 3 that exist" and the arm is still reported as run.
  Check(!p::MakeQwenKvTailFormatTable(fa.data(), kBlocks, 17,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kTurbo4,
                                      p::QwenKvLayerFormat::kTurbo4, &t),
        "B1: a tail LARGER than the number of KV layers refuses rather than "
        "clamping to all of them");

  Check(p::MakeQwenKvTailFormatTable(fa.data(), kBlocks, 16,
                                     p::QwenKvLayerFormat::kFp16,
                                     p::QwenKvLayerFormat::kFp16,
                                     p::QwenKvLayerFormat::kTurbo4,
                                     p::QwenKvLayerFormat::kTurbo4, &t),
        "B1: a tail exactly equal to the KV-layer count is accepted");

  Check(!p::MakeQwenKvTailFormatTable(fa.data(), 0, 0,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kTurbo4,
                                      p::QwenKvLayerFormat::kTurbo4, &t),
        "B1: zero blocks refuses");

  // A MODEL WITH NO FULL-ATTENTION LAYERS has no KV cache to describe. An
  // empty table would read as "no table, use the enumerator" and quietly
  // resurrect the whole-cache path.
  const AttentionMask none{};
  Check(!p::MakeQwenKvTailFormatTable(none.data(), kBlocks, 4,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kTurbo4,
                                      p::QwenKvLayerFormat::kTurbo4, &t),
        "B1: a model with no KV-bearing layer refuses rather than producing an "
        "empty table that would read as 'no table'");

  Check(!p::MakeQwenKvTailFormatTable(nullptr, kBlocks, 4,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kTurbo4,
                                      p::QwenKvLayerFormat::kTurbo4, &t),
        "B1: a null attention mask refuses");

  Check(!p::MakeQwenKvTailFormatTable(fa.data(), kBlocks, 4,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kTurbo4,
                                      p::QwenKvLayerFormat::kTurbo4, nullptr),
        "B1: a null output table refuses");

  // BEYOND THE CAP IS A REFUSAL, not a truncation that drops the tail.
  std::array<bool, p::QwenKvFormatTable::kMaxLayers + 1> huge{};
  huge.fill(true);
  Check(!p::MakeQwenKvTailFormatTable(huge.data(), huge.size(), 1,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kFp16,
                                      p::QwenKvLayerFormat::kTurbo4,
                                      p::QwenKvLayerFormat::kTurbo4, &t),
        "B1: more blocks than the table cap refuses — truncating would silently "
        "drop the accurate end of the model");

  // A TAIL OF ZERO IS NOT AN ERROR, it is the uniform base case (arm A).
  const auto zero = Build(0, p::QwenKvLayerFormat::kFp16,
                          p::QwenKvLayerFormat::kTurbo4);
  Check(zero.layer_count == kAttentionLayers &&
            zero.IsUniform(p::QwenKvLayerFormat::kTurbo4),
        "B1: a tail of 0 is the UNIFORM Turbo4 table — arm A, production");
}

void UniformityAndCountsAreDistinctQuestions() {
  // The fast path is allowed only when the table is uniform. A table that is
  // uniform in K but not V is NOT uniform, and must not take it.
  p::QwenKvFormatTable t;
  t.layer_count = 4;
  for (std::uint32_t i = 0; i < 4; ++i)
    t.Set(i, p::QwenKvLayerFormat::kTurbo4, p::QwenKvLayerFormat::kFp16);
  Check(!t.IsUniform(p::QwenKvLayerFormat::kTurbo4) &&
            !t.IsUniform(p::QwenKvLayerFormat::kFp16),
        "B1: a table whose K and V differ is uniform in NEITHER, so no "
        "whole-cache fast path may claim it");
  Check(t.HasMixedPlanes(),
        "B1: mixed planes are reported as mixed rather than inferred");
  Check(t.BlockAddressedCount(/*for_k=*/true) == 4 &&
            t.BlockAddressedCount(/*for_k=*/false) == 0,
        "B1: K and V are counted SEPARATELY — the allocation is sized per plane, "
        "so a count that used one enumerator for both would size it wrong");

  // THE K>V ARM. Arm C of the mixed-precision KV experiment is K=Turbo4, V=f16: the literature's biggest
  // lever, and the shape that only a per-K/V table can express at all.
  p::QwenKvFormatTable& c = c_arm;
  const AttentionMask fa = RealFullAttention();
  Check(p::MakeQwenKvTailFormatTable(
            fa.data(), kBlocks, kAttentionLayers, p::QwenKvLayerFormat::kTurbo4,
            p::QwenKvLayerFormat::kFp16, p::QwenKvLayerFormat::kTurbo4,
            p::QwenKvLayerFormat::kTurbo4, &c),
        "B1: an ASYMMETRIC table builds — K and V are named separately");
  Check(c.layer_count == kAttentionLayers &&
            c.BlockAddressedCount(/*for_k=*/true) == kAttentionLayers &&
            c.BlockAddressedCount(/*for_k=*/false) == 0,
        "B1: a K-only tail is expressible: every layer K is block-addressed, no "
        "layer V is — the arm-C shape, which one enumerator cannot state");
  Check(c.HasMixedPlanes() && !c.IsUniform(p::QwenKvLayerFormat::kTurbo4),
        "B1: the arm-C table is mixed and takes no whole-cache fast path");
}

void GraphIdentityTracksTheTable() {
  // A GRAPH CAPTURED UNDER ONE TABLE MUST NOT REPLAY UNDER ANOTHER. This is
  // the defect the single-enumerator fingerprint was written to prevent, now
  // arriving through the table instead: two policies differing only in WHICH
  // layers are f16 would share a fingerprint, and the route guard would let the
  // captured graph run against a cache whose layout no longer matched.
  const auto a = Build(4, p::QwenKvLayerFormat::kFp16,
                       p::QwenKvLayerFormat::kTurbo4);
  const auto b = Build(8, p::QwenKvLayerFormat::kFp16,
                       p::QwenKvLayerFormat::kTurbo4);
  const auto c = Build(4, p::QwenKvLayerFormat::kFp16,
                       p::QwenKvLayerFormat::kTurbo4);
  Check(a.Fingerprint() != b.Fingerprint(),
        "B1: tail 4 and tail 8 have DIFFERENT table fingerprints — a graph "
        "cannot be replayed across them");
  Check(a.Fingerprint() == c.Fingerprint(),
        "B1: two tables built the same way have the SAME fingerprint");
  Check(a.Fingerprint() != 0ULL, "B1: a populated table contributes a non-zero "
                                 "fingerprint");

  // ...AND THE WHOLE POLICY. Same enumerator, different table, different graph
  // identity. This is the check the policy-level Fingerprint has to make.
  p::QwenExecutionPolicy pa = p::QwenExecutionPolicy::Production();
  p::QwenExecutionPolicy pb = p::QwenExecutionPolicy::Production();
  const std::uint64_t bare = pa.Fingerprint();
  pa.kv_layers = a;
  pb.kv_layers = b;
  Check(pa.Fingerprint() != pb.Fingerprint(),
        "B1: two POLICIES differing only in the table have different "
        "fingerprints");
  Check(pa.Fingerprint() != bare,
        "B1: adding a table CHANGES the policy fingerprint, so the layout is "
        "part of graph identity");

  // THE COMPATIBILITY HALF, and it matters more than the half above. Every
  // fingerprint an existing build could have written must be UNCHANGED, or its
  // graph-cache entries become unreachable for no reason but a new field.
  p::QwenExecutionPolicy prod = p::QwenExecutionPolicy::Production();
  p::QwenExecutionPolicy f32 = p::QwenExecutionPolicy::Production();
  f32.kv_cache_storage = p::QwenKvCacheStorage::kFp32;
  p::QwenExecutionPolicy bf = p::QwenExecutionPolicy::Production();
  bf.recurrent_state_storage = p::QwenRecurrentStateStorage::kBf16;
  p::QwenExecutionPolicy t4 = p::QwenExecutionPolicy::Production();
  t4.kv_cache_storage = p::QwenKvCacheStorage::kTurbo4;
  Check(prod.Fingerprint() == (1ULL | (1ULL << 8U)),
        "B1: production (fused, fp16, fp32 state) keeps fingerprint "
        "1 | (1<<8) exactly — the value every existing build wrote");
  Check(f32.Fingerprint() == 1ULL,
        "B1: fp32 KV keeps fingerprint 1 — bit 8 clear, as before");
  Check(bf.Fingerprint() == (1ULL | (1ULL << 8U) | (1ULL << 9U)),
        "B1: bf16 state still lights bit 9 and nothing else");
  // Turbo4 is enumerator 2: bit 16 set, bit 8 clear. The aliasing the comment
  // above warns about (fp32 and turbo4 colliding) must still be broken.
  Check(t4.Fingerprint() == (1ULL | (1ULL << 16U)),
        "B1: Turbo4 KV keeps fingerprint 1 | (1<<16) — distinct from fp32");
  Check(t4.Fingerprint() != f32.Fingerprint(),
        "B1: Turbo4 and fp32 do NOT alias, which is the reason the kv bits are "
        "split across 8 and 16");
}

void BlockAddressedPredicateIsHonest() {
  Check(p::QwenKvLayerFormatIsBlockAddressed(p::QwenKvLayerFormat::kTurbo4),
        "B1: only Turbo4 is block-addressed");
  Check(!p::QwenKvLayerFormatIsBlockAddressed(p::QwenKvLayerFormat::kFp16),
        "B1: f16 is not block-addressed");
}

// ---------------------------------------------------------------------------
// The per-layer ROUTE, which is the part decode_step now consults instead of
// re-deriving at the call site.
// ---------------------------------------------------------------------------

void RouteIsResolvedNotInferred() {
  p::QwenExecutionPolicy policy = p::QwenExecutionPolicy::Production();

  // NO TABLE = the whole cache is the enumerator = the existing behaviour.
  const auto base = p::ResolveQwenLayerKvPlan(policy, 0, /*fusion_ok=*/true);
  Check(base.reader == p::QwenKvReaderKind::kElementWise &&
            base.fuse_qk_norm_rope_kv && base.graph_eligible,
        "B4: with no table the route is exactly what it was — element-wise "
        "reader, fused kernel, graph-eligible");

  // A TABLE OF ALL-Turbo4 TURNS THE WHOLE CACHE TO THE BLOCK ROUTE, and flips
  // both other fields with it.
  const auto t4 = Build(0, p::QwenKvLayerFormat::kFp16,
                        p::QwenKvLayerFormat::kTurbo4);
  policy.kv_layers = t4;
  const auto blk = p::ResolveQwenLayerKvPlan(policy, 0, /*fusion_ok=*/true);
  Check(blk.reader == p::QwenKvReaderKind::kBlockPacked,
        "B4: a Turbo4 layer selects the block reader");
  Check(!blk.fuse_qk_norm_rope_kv,
        "B4: a Turbo4 layer REFUSES the fused QK-norm/RoPE/KV-write kernel even "
        "when the policy asks for it — opt-c010 has no 66-byte-block variant, "
        "so honouring the toggle would write fp16-shaped bytes into a block "
        "plane");
  Check(!blk.graph_eligible,
        "B4: a Turbo4 layer is graph-INELIGIBLE, because the reader takes the "
        "host pos+1 and a captured graph would freeze the valid context");

  // THE FUSION FLAG IS NOT SIMPLY INVERTED — an unsupported head_dim also
  // disables it, so a block layer and a fused layer are distinguishable.
  const auto nofusion = p::ResolveQwenLayerKvPlan(policy, 0, /*fusion_ok=*/false);
  Check(!nofusion.fuse_qk_norm_rope_kv && nofusion.reader == blk.reader,
        "B4: an unsupported head_dim disables the fused kernel independently of "
        "the block decision");

  // THE MIXED TABLE: only the tail layers change route.
  const auto mixed = Build(4, p::QwenKvLayerFormat::kFp16,
                           p::QwenKvLayerFormat::kTurbo4);
  p::QwenExecutionPolicy mp = p::QwenExecutionPolicy::Production();
  mp.kv_layers = mixed;
  Check(p::ResolveQwenLayerKvPlan(mp, 11, true).reader ==
            p::QwenKvReaderKind::kBlockPacked &&
            p::ResolveQwenLayerKvPlan(mp, 12, true).reader ==
                p::QwenKvReaderKind::kElementWise,
        "B4: in a tail-4 table, layer 11 is block-packed and layer 12 is not — "
        "the route follows the TABLE, not the whole-cache enumerator");

  // AN OUT-OF-RANGE LAYER IS NOT SILENTLY TREATED AS THE BASE FORMAT. It falls
  // back to the element-wise reader, which is the safe direction only because
  // the executor refuses block storage before reaching here.
  Check(!p::QwenKvLayerIsBlockAddressed(policy, 999),
        "B4: a layer index past the table is not block-addressed rather than "
        "wrapping to some layer's format");

  // K>bits THAN V: block K with fp16 V still needs the BLOCK reader for K.
  p::QwenExecutionPolicy asy = p::QwenExecutionPolicy::Production();
  asy.kv_layers = c_arm;
  Check(p::QwenKvLayerIsBlockAddressed(asy, 0),
        "B4: a layer with Turbo4 K and fp16 V is still block-addressed — "
        "routing on V would read 66-byte blocks as fp16 pairs");
  p::QwenKvFormatTable rev{};
  rev.layer_count = 4;
  for (std::uint32_t i = 0; i < 4; ++i)
    rev.Set(i, p::QwenKvLayerFormat::kFp16, p::QwenKvLayerFormat::kTurbo4);
  p::QwenExecutionPolicy revp = p::QwenExecutionPolicy::Production();
  revp.kv_layers = rev;
  Check(p::QwenKvLayerIsBlockAddressed(revp, 0),
        "B4: and the same holds with the planes swapped — the predicate is on "
        "EITHER plane");
}

// ---------------------------------------------------------------------------
// B-D1/D2/D3 — the three fields that were set and never READ.
// ---------------------------------------------------------------------------

void FieldsAreActuallyConsumed() {
  p::QwenExecutionPolicy policy = p::QwenExecutionPolicy::Production();
  policy.kv_layers = c_arm;  // all 16 layers K=Turbo4 V=f16 -> block packed

  // ---- B-D2: graph_eligible is now CONSUMED by the capture guard. ----
  const auto plain = p::ResolveQwenGraphRejections(true, false, true);
  Check(!p::HasQwenGraphRejection(plain,
                                  p::QwenGraphRejection::kBlockKvGraphIneligible),
        "B-D2 CONTROL: the OLD plain resolver knows nothing about block layers — "
        "this is the gap that let capture through");
  const auto aware =
      p::ResolveQwenGraphRejectionsForPolicy(policy, true, false, true);
  Check(p::HasQwenGraphRejection(aware,
                                 p::QwenGraphRejection::kBlockKvGraphIneligible),
        "B-D2: a Turbo4 policy REFUSES graph capture");
  Check(p::HasQwenGraphRejection(aware,
                                 p::QwenGraphRejection::kNone) == false,
        "B-D2: and the refusal is a real rejection, not kNone");

  // An f16-only table must NOT trip it — the guard keys on block packing, not on
  // having a table at all.
  p::QwenExecutionPolicy fp16 = p::QwenExecutionPolicy::Production();
  fp16.kv_layers = Build(0, p::QwenKvLayerFormat::kFp16,
                         p::QwenKvLayerFormat::kFp16);
  Check(!p::HasQwenGraphRejection(
            p::ResolveQwenGraphRejectionsForPolicy(fp16, true, false, true),
            p::QwenGraphRejection::kBlockKvGraphIneligible),
        "B-D2: an all-f16 table does NOT refuse capture — the guard is about "
        "block packing, not about tables existing");

  // And it composes with the pre-existing reasons rather than replacing them.
  const auto both = p::ResolveQwenGraphRejectionsForPolicy(policy, false, false,
                                                            true);
  Check(p::HasQwenGraphRejection(both,
                                 p::QwenGraphRejection::kBlockKvGraphIneligible) &&
            p::HasQwenGraphRejection(both,
                                      p::QwenGraphRejection::kLogitsNotRequested),
        "B-D2: the block refusal COMPOSES with the pre-existing rejections");

  // ---- B-D3: a table needing an unallocated plane is refused. ----
  Check(!p::QwenKvTableIsServableByStorage(
            Build(4, p::QwenKvLayerFormat::kFp16, p::QwenKvLayerFormat::kTurbo4),
            p::QwenKvCacheStorage::kTurbo4),
        "B-D3: an f16 tail under Turbo4 storage is NOT servable — it would "
        "route those layers to the element-wise reader with a NULL f16 pointer");
  Check(!p::QwenKvTableIsServableByStorage(c_arm,
                                           p::QwenKvCacheStorage::kTurbo4),
        "B-D3: and a K-Turbo4/V-f16 table is not servable either — V is the "
        "plane that was never allocated");
  Check(p::QwenKvTableIsServableByStorage(Build(0, p::QwenKvLayerFormat::kFp16,
                                                p::QwenKvLayerFormat::kTurbo4),
                                          p::QwenKvCacheStorage::kTurbo4),
        "B-D3: an ALL-Turbo4 table IS servable under Turbo4 storage");
  Check(p::QwenKvTableIsServableByStorage(p::QwenKvFormatTable{},
                                          p::QwenKvCacheStorage::kTurbo4),
        "B-D3: an EMPTY table is servable — no table means the enumerator rules");
  Check(p::QwenKvTableFirstUnservableLayer(
            Build(4, p::QwenKvLayerFormat::kFp16, p::QwenKvLayerFormat::kTurbo4),
            p::QwenKvCacheStorage::kTurbo4) == 12,
        "B-D3: the refusal NAMES the first offending layer (12), so the message "
        "can point at it instead of saying invalid");
  Check(p::QwenKvTableFirstUnservableLayer(
            Build(0, p::QwenKvLayerFormat::kFp16, p::QwenKvLayerFormat::kTurbo4),
            p::QwenKvCacheStorage::kTurbo4) == 16,
        "B-D3: a servable table reports layer_count as the sentinel");
}

// ---------------------------------------------------------------------------
// The status latch across layers: a later layer must not erase an earlier failure.
// ---------------------------------------------------------------------------

void StatusLatchSurvivesLaterLayers() {
  using S = p::Turbo4KernelStatus;
  constexpr std::uint32_t kAttentionLayers = 16;

  // THE HOST READS THE WORD ONCE, AT THE END OF THE STEP, alongside the sampled
  // token. That is the whole of B-D1: the per-call reset did not stop the
  // FAILING LAYER from seeing its own refusal, it stopped the HOST from seeing
  // it, because by then layer 15 had reset the word to kOk. So the check below
  // is made after the loop, not inside it — reading inside would pass in both
  // worlds and prove nothing.

  // THE OLD BEHAVIOUR, reproduced.
  {
    S word = S::kOk;
    for (std::uint32_t layer = 0; layer < kAttentionLayers; ++layer) {
      word = S::kOk;  // THE PER-CALL RESET THAT B-D1 REMOVED
      if (layer == 3) p::Turbo4LatchFailure(&word, S::kQuantizeRefused);
    }
    // End of step. The host reads here.
    Check(!p::Turbo4StepStatusFromWords(
               static_cast<std::uint32_t>(word), 0U)
               .Refuses(),
          "B-D1 CONTROL: with a per-call reset, layer 3 refuses, layers 4-15 "
          "each reset the word, and the HOST reads kOk — the defect, "
          "reproduced at the point it actually bites");
  }

  // THE FIX: reset once per step, latch only. Same failure, same layers.
  {
    S word = S::kOk;  // reset ONCE, before the loop
    for (std::uint32_t layer = 0; layer < kAttentionLayers; ++layer)
      if (layer == 3) p::Turbo4LatchFailure(&word, S::kQuantizeRefused);
    Check(p::Turbo4StepStatusFromWords(static_cast<std::uint32_t>(word), 0U)
              .Refuses(),
          "B-D1: with the reset moved to the step boundary, a layer-3 refusal "
          "SURVIVES layers 4..15 and the host sees it");
  }

  // ...and a later kOk cannot lower it, across the remaining layers.
  {
    S word = S::kOk;
    p::Turbo4LatchFailure(&word, S::kPoisonedNorm);
    for (int i = 0; i < 15; ++i) p::Turbo4LatchFailure(&word, S::kOk);
    Check(word == S::kPoisonedNorm,
          "B-D1: fifteen later kOk results do not erase the earlier refusal");
  }

  // BOTH WORDS ARE READ, TOGETHER. One predicate, so there is no path that
  // consults one word and ignores the other.
  Check(!p::Turbo4StepStatusFromWords(0U, 0U).Refuses(),
        "B-D1: both words kOk does not refuse");
  for (std::uint32_t bad : {1U, 2U, 3U, 4U, 5U, 6U}) {
    if (!p::Turbo4StepStatusFromWords(bad, 0U).Refuses())
      Check(false, "B-D1: a write-word failure must refuse");
    if (!p::Turbo4StepStatusFromWords(0U, bad).Refuses())
      Check(false, "B-D1: a read-word failure must refuse");
  }
  Check(true, "B-D1: EVERY failure code in EITHER word refuses — neither plane "
              "is checked alone");

  // A word holding a value this build does not know is a DEVICE/HOST contract
  // disagreement. It must refuse, not coerce into a valid enumerator.
  Check(p::Turbo4StepStatusFromWords(0U, 99U).Refuses() &&
            p::Turbo4StepStatusFromWords(4294967295U, 0U).Refuses(),
        "B-D1: an UNKNOWN word value refuses rather than being treated as fine");
}

}  // namespace

int main() {
  CountsKvBearingLayersNotBlocks();
  RefusesRatherThanClamping();
  UniformityAndCountsAreDistinctQuestions();
  GraphIdentityTracksTheTable();
  BlockAddressedPredicateIsHonest();
  RouteIsResolvedNotInferred();
  FieldsAreActuallyConsumed();
  StatusLatchSurvivesLaterLayers();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
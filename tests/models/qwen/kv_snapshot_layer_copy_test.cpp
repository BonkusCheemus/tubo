// The per-layer SNAPSHOT copy plan, on CPU.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/kv_snapshot_layer_copy_test.cpp
//       -o /tmp/kvslc
//
// THE QUESTION THIS FILE ANSWERS. The co-plan asked what the snapshot transfer
// does today, and warned: "each layer's valid prefix is contiguous, but the
// WHOLE buffer's prefix is not." Both halves are true and they have opposite
// consequences, so this file settles them with numbers rather than prose:
//
//   1. The existing 2D copy is ALREADY per-layer for a uniform Turbo4 cache.
//      Its `rows` is the layer count and its pitch is the layer stride, so no
//      rewrite is warranted — and `UniformStrideIsPerLayer` PROVES that rather
//      than assuming it. "They look the same" is how a snapshot restores layer
//      7's tokens into layer 8's bytes.
//
//   2. The bytes moved per layer are the SNAPSHOT plane size, not the live one.
//      The live arena is sized for max_context and the snapshot for the context
//      it captured; copying the live stride would read a megabyte per layer and
//      write it past a buffer holding kilobytes.
//
// A MIXED per-layer table is a different layout and is NOT covered by the uniform
// fast path; that is stated here rather than implied.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#include "src/models/qwen/hip/turbo4_codec.hpp"
#include "src/models/qwen/hip/turbo4_plane_index.hpp"

namespace t4 = gufo::hip::turbo4;
namespace t4i = gufo::hip;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kLayers = 16;
constexpr std::size_t kSnapCtx = 3;
constexpr std::size_t kHeader = 80;

const std::size_t kTokenPitch =
    t4i::Turbo4BlocksPerToken(kHeadDim, kKvHeads) * t4::kBlockBytes;  // 528

// ---------------------------------------------------------------------------

void UniformStrideIsPerLayer() {
  Check(kTokenPitch == 528,
        "B3: one token is 528 B across all 4 kv heads (8 blocks x 66 B)");

  // THE PROOF the whole decision rests on: for EVERY context in range, one
  // layer's plane is exactly `context * token_pitch`. That identity is what makes
  // a 2D copy with rows=layers address layer i at the same offset this file
  // computes layer by layer. Checking only the manifest context would leave
  // every other context unproven.
  Check(t4i::Turbo4LayerPrefixesAgreeWithUniformStride(kHeadDim, kKvHeads,
                                                       kLayers, kTokenPitch),
        "B3: layer plane == context x 528 for every context 1..4096, so the "
        "existing rows=layers 2D copy IS per-layer and needs no rewrite");

  // And the same claim at the live context, spelled out.
  const std::size_t live_plane =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, t4i::kTurbo4ManifestContext);
  Check(live_plane == t4i::kTurbo4ManifestContext * kTokenPitch,
        "B3: the live plane at the manifest context is 4096 x 528 = 2162688 B");
}

void PlanMovesTheSnapshotPrefixNotTheLiveStride() {
  const std::size_t live_plane =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, t4i::kTurbo4ManifestContext);
  const std::size_t snap_plane =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kSnapCtx);
  std::array<t4i::Turbo4LayerPrefix, kLayers> plan{};

  Check(t4i::Turbo4LayerPrefixPlan(live_plane * kLayers * 2,
                                   t4i::kTurbo4ManifestContext,
                                   snap_plane * kLayers * 2 + kHeader, kSnapCtx,
                                   0, kHeader, kHeadDim, kKvHeads, kLayers,
                                   plan.data(), plan.size()),
        "B3: the plan builds for the real geometry");

  Check(plan[0].bytes == snap_plane,
        "B3: bytes moved per layer is the SNAPSHOT plane (1584 B), not the live "
        "one (2162688 B) — copying the live stride would overrun the snapshot");
  for (const auto& p : plan)
    if (p.bytes != snap_plane) {
      Check(false, "B3: every layer moves the same prefix size");
      return;
    }
  Check(true, "B3: every layer moves the same prefix size");

  // LAYER 0: live K at 0, and the snapshot K offset by exactly the header.
  Check(plan[0].k_src == 0 && plan[0].k_dst == kHeader,
        "B3: layer 0 K is at 0 live and 80 in the compact snapshot");
  // V REGION: after ALL 16 K layers in both buffers.
  Check(plan[0].v_src == live_plane * kLayers,
        "B3: layer 0 V follows all 16 K layers in the LIVE buffer");
  Check(plan[0].v_dst == kHeader + snap_plane * kLayers,
        "B3: ...and likewise in the snapshot");

  // THE LAST LAYER. Two DIFFERENT ends, and conflating them is the mistake this
  // check was written after making:
  //
  //   - the live REGION ends at the arena end, and it is the live plane wide,
  //     because the arena was sized for max_context;
  //   - the copied PREFIX is only valid_context wide, so it necessarily stops
  //     SHORT of the arena end. It is not supposed to reach it.
  //
  // Asserting the copied bytes reach the arena end would be asserting that the
  // live buffer holds nothing beyond the valid context, which is false and is
  // exactly what a snapshot is for.
  const std::size_t live_bytes = live_plane * kLayers * 2;
  const std::size_t snap_bytes = snap_plane * kLayers * 2 + kHeader;
  Check(plan[kLayers - 1].v_src + live_plane == live_bytes,
        "B3: the last layer's live V REGION ends exactly at the end of the "
        "arena (live_plane wide, sized for max_context)");
  Check(plan[kLayers - 1].bytes < live_plane &&
            plan[kLayers - 1].v_src + plan[kLayers - 1].bytes <= live_bytes,
        "B3: the copied PREFIX is narrower than the live region and fits "
        "inside it — the live arena legitimately holds more than the snapshot");
  Check(plan[kLayers - 1].v_dst + plan[kLayers - 1].bytes == snap_bytes,
        "B3: the last layer's snapshot V ends EXACTLY at the end of the "
        "snapshot, because the snapshot is sized for the context it captured");

  // LAYER STRIDE IS THE LIVE PLANE SIZE on the source side and the SNAPSHOT
  // plane size on the destination side. Getting these the wrong way round is the
  // bug: it shifts every layer after the first.
  Check(plan[1].k_src - plan[0].k_src == live_plane,
        "B3: the SOURCE stride is the live plane size (max_context)");
  Check(plan[1].k_dst - plan[0].k_dst == snap_plane,
        "B3: the DESTINATION stride is the snapshot plane size (valid_context)");

  // CONTIGUOUS, so a bulk copy would work — but only per layer, never for the
  // whole buffer at once.
  Check(plan[2].k_src == plan[1].k_src + plan[1].k_src - plan[0].k_src,
        "B3: live K offsets are contiguous layer to layer");
}

void PlanRefusesRatherThanGuessing() {
  const std::size_t live_plane =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, t4i::kTurbo4ManifestContext);
  const std::size_t snap_plane =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kSnapCtx);
  std::array<t4i::Turbo4LayerPrefix, kLayers> plan{};

  // A SNAPSHOT BIGGER THAN THE ARENA IS IMPOSSIBLE, and taking it would be a
  // restore reading a buffer the arena never held.
  Check(!t4i::Turbo4LayerPrefixPlan(live_plane * kLayers * 2,
                                    t4i::kTurbo4ManifestContext,
                                    snap_plane * kLayers * 2 + kHeader, 4096, 0,
                                    kHeader, kHeadDim, kKvHeads, kLayers,
                                    plan.data(), plan.size()),
        "B3: a snapshot context LARGER than the arena refuses");

  Check(!t4i::Turbo4LayerPrefixPlan(live_plane * kLayers * 2,
                                    t4i::kTurbo4ManifestContext,
                                    snap_plane * kLayers * 2 + kHeader, 0, 0,
                                    kHeader, kHeadDim, kKvHeads, kLayers,
                                    plan.data(), plan.size()),
        "B3: a zero snapshot context refuses");

  Check(!t4i::Turbo4LayerPrefixPlan(live_plane * kLayers * 2,
                                    t4i::kTurbo4ManifestContext,
                                    snap_plane * kLayers * 2 + kHeader, kSnapCtx,
                                    0, kHeader, kHeadDim, kKvHeads, kLayers,
                                    plan.data(), 4),
        "B3: an output array smaller than the layer count refuses rather than "
        "silently planning only the first few");

  Check(!t4i::Turbo4LayerPrefixPlan(live_plane * kLayers * 2,
                                    t4i::kTurbo4ManifestContext,
                                    snap_plane * kLayers * 2 + kHeader, kSnapCtx,
                                    0, kHeader, kHeadDim, kKvHeads, kLayers,
                                    nullptr, plan.size()),
        "B3: a null output refuses");

  Check(!t4i::Turbo4LayerPrefixPlan(live_plane * kLayers * 2,
                                    t4i::kTurbo4ManifestContext,
                                    snap_plane * kLayers * 2, kSnapCtx, 0,
                                    kHeader, kHeadDim, kKvHeads, kLayers,
                                    plan.data(), plan.size()),
        "B3: a snapshot buffer too small to hold its own header refuses");

  Check(!t4i::Turbo4LayerPrefixPlan(live_plane * kLayers * 2,
                                    t4i::kTurbo4ManifestContext,
                                    snap_plane * kLayers * 2 + kHeader, kSnapCtx,
                                    0, kHeader, kHeadDim, kKvHeads, 0,
                                    plan.data(), plan.size()),
        "B3: zero layers refuses");

  // ---- B-D4: the RECORDED length is what the decode step now passes. A plane
  // that was never allocated records 0, and 0 must mean "refuse", never
  // "unbounded". This is the CPU-observable half of the fix; the other half is
  // that the arena records the length at hipMalloc instead of a formula being
  // recomputed at the call site, which no CPU test can reach.
  t4i::Turbo4LayerSlices slices{};
  Check(!t4i::Turbo4LayerPlaneSlices(/*arena_bytes=*/0, 0, kHeadDim, kKvHeads,
                                     t4i::kTurbo4ManifestContext, kLayers, 0,
                                     &slices),
        "B-D4: a recorded arena length of 0 REFUSES every layer — a missing "
        "allocation is a refusal, not an unlimited plane");
  Check(!t4i::Turbo4LayerPrefixPlan(/*arena_bytes=*/0,
                                    t4i::kTurbo4ManifestContext,
                                    snap_plane * kLayers * 2 + kHeader, kSnapCtx,
                                    0, kHeader, kHeadDim, kKvHeads, kLayers,
                                    plan.data(), plan.size()),
        "B-D4: and the prefix plan refuses on a recorded length of 0 too");
}

}  // namespace

int main() {
  UniformStrideIsPerLayer();
  PlanMovesTheSnapshotPrefixNotTheLiveStride();
  PlanRefusesRatherThanGuessing();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
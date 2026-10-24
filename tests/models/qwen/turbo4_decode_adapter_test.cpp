// The Turbo4 DECODE ADAPTER, on CPU.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/turbo4_decode_adapter_test.cpp
//       -o /tmp/t4a
//
// WHAT THIS FILE IS FOR. Unit A adds three things to the Turbo4 path: the
// sigmoid gate, the writer-failure word, and a layer-slice function. Each of
// them is a place where a plausible-looking constant produces a confident
// wrong answer rather than a crash. So this file drives the ACTUAL bodies —
// `Turbo4WriteHeadBody`, `Turbo4AttentionQueryBody`, `Turbo4LayerPlaneSlices` —
// through a real write-then-read over a real arena, and compares against a
// dense float reference written here.
//
// WHY THE EXISTING ORACLE WAS NOT ENOUGH. `turbo4_kernel_ref_test.cpp` checks
// the reader's refusals with a pre-filled plane and one query head at a time.
// Nothing there ever WROTE a plane and then read it back, so a gate that read
// the wrong row, a slice that addressed layer 0 when asked for layer 1, or a
// writer-failure check that ran after the arithmetic would all have stayed
// green. This file writes and reads.
//
// WHAT THIS FILE IS NOT. It is not codec coverage — `turbo4_codec_test.cpp`
// owns that. The reference below reads the stored plane back through
// `DequantizeBlock` to isolate the ATTENTION arithmetic (GQA map, slicing,
// softmax, gate) from quantisation error, and reference 2 then measures that
// error separately. It is not silicon coverage either: grid geometry and launch
// configuration remain compile-only, which is why the runner's .hip syntax
// check exists alongside this.
//
// TWO DELIBERATE NON-INDEPENDENCIES, stated rather than hidden:
//   1. The reference locates blocks with `Turbo4BlockByteOffset`. The offset
//      authority has its own independent spec transcription in
//      `turbo4_kernel_ref_test.cpp`; re-transcribing it here would test the
//      same arithmetic twice, not cover a second thing.
//   2. The reference attends in token order, as the body does. That IS part of
//      the contract — stable softmax forbids reordering — so the reference
//      encodes the same requirement rather than inventing a looser one.
//
// GEOMETRY: UD-Q4_K_XL. 24 query heads, 4 kv heads, head_dim 256, and 16
// full-attention layers (block_count 65 includes one MTP layer, so
// (65 - 1) / interval 4 = 16).
//
// A LABELING RULE THIS FILE FOLLOWS, LEARNED THE HARD WAY. The runner tallies a
// suite with an UNANCHORED `grep -c 'FAIL'` over the persisted log, so a check
// whose LABEL contains the substring "FAIL" is counted as a failure even when
// the check passed. A first draft of this file labelled its writer-status checks
// "WRITER-FAIL:" and turned an entirely green 45-check suite red in the tally.
// Check labels here use "ok" and "FAIL" only as the runner's own leading column,
// never inside prose.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/turbo4_codec.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_plane_index.hpp"

namespace t4 = gufo::hip::turbo4;
namespace t4i = gufo::hip;
using S = gufo::hip::Turbo4KernelStatus;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string &what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kQHeads = 24;
constexpr std::uint32_t kLayers = 16;
/// Three DISTINCT tokens. One token would pass a body that ignores `pos`, and
/// two would pass one that reads only token 0; three is the smallest count that
/// fails both.
constexpr std::uint32_t kCtx = 3;

constexpr std::size_t kWriterSourceBytes = kKvHeads * kHeadDim * sizeof(float);
constexpr float kScale = 0.0625f;  // 1/sqrt(256)

/// Deterministic values. A fixed generator, not <random>: this file's failures
/// have to be reproducible from the run directory alone.
struct Lcg {
  std::uint64_t state;
  float Next() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const std::int64_t v =
        static_cast<std::int64_t>(state >> 11) % 2000001;  // [-1, 1]
    return static_cast<float>(v) / 1000000.0f;
  }
};

/// [ctx][kv_head][head_dim] source for one layer, same value at every (token,
/// kv head) index plus a per-index offset so no two tokens are interchangeable.
using LayerSource = std::vector<float>;

std::size_t SourceIndex(std::uint32_t tok, std::uint32_t head) {
  return (static_cast<std::size_t>(tok) * kKvHeads + head) * kHeadDim;
}

LayerSource MakeSource(std::uint64_t seed) {
  LayerSource s(static_cast<std::size_t>(kCtx) * kWriterSourceBytes /
                sizeof(float));
  Lcg r{seed};
  for (std::size_t i = 0; i < s.size(); ++i) s[i] = r.Next();
  return s;
}

/// Read one kv head's K and V for every token back OUT of the stored plane.
///
/// This is the "K/V that went through codec round-trip" of reference 1. It uses
/// the production dequantiser on purpose: the question here is whether attention
/// over the STORED bytes is correct, and using a different dequantiser would
/// answer a different question.
void DequantizeStored(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                      std::uint32_t kv_head, LayerSource* k, LayerSource* v) {
  for (std::uint32_t t = 0; t < kCtx; ++t) {
    std::size_t base = 0;
    if (!t4i::Turbo4BlockByteOffset(t, kv_head, kHeadDim, kKvHeads, kCtx,
                                    &base))
      return;
    for (std::size_t b = 0; b < kHeadDim / t4::kBlockValues; ++b) {
      float ks[t4::kBlockValues];
      float vs[t4::kBlockValues];
      t4::DequantizeBlock(k_plane + base + b * t4::kBlockBytes, ks);
      t4::DequantizeBlock(v_plane + base + b * t4::kBlockBytes, vs);
      t4::RotateInverse(ks);
      t4::RotateInverse(vs);
      const std::size_t off = SourceIndex(t, kv_head);
      for (std::size_t i = 0; i < t4::kBlockValues; ++i) {
        (*k)[off + b * t4::kBlockValues + i] = ks[i];
        (*v)[off + b * t4::kBlockValues + i] = vs[i];
      }
    }
  }
}

/// DENSE FLOAT ATTENTION, written here, over float K/V for ONE NAMED kv head.
///
/// The kv head is passed in by the caller, which derives it from the map
/// sentence rather than calling `Turbo4KvHeadForQueryHead` — so a wrong map in
/// the production helper does not silently agree with this reference. The map
/// itself is pinned by `GroupedQueryMapBelow`, which is a different kind of check
/// that no single re-implementation can satisfy.
void DenseAttentionForKvHead(const LayerSource& k, const LayerSource& v,
                             const float* q_head, const float* gate,
                             std::uint32_t kv_head, float* out) {
  std::vector<float> sc(kCtx, 0.0f);
  float run_max = -3.402823466e+38f;
  for (std::uint32_t t = 0; t < kCtx; ++t) {
    const std::size_t off = SourceIndex(t, kv_head);
    float s = 0.0f;
    for (std::size_t i = 0; i < kHeadDim; ++i)
      s += q_head[i] * k[off + i];
    s *= kScale;
    sc[t] = s;
    if (s > run_max) run_max = s;
  }
  float denom = 0.0f;
  for (std::uint32_t t = 0; t < kCtx; ++t) {
    sc[t] = std::exp(sc[t] - run_max);
    denom += sc[t];
  }
  const float inv = 1.0f / denom;
  for (std::size_t i = 0; i < kHeadDim; ++i) out[i] = 0.0f;
  for (std::uint32_t t = 0; t < kCtx; ++t) {
    const std::size_t off = SourceIndex(t, kv_head);
    for (std::size_t i = 0; i < kHeadDim; ++i)
      out[i] += sc[t] * v[off + i];
  }
  for (std::size_t i = 0; i < kHeadDim; ++i) {
    out[i] *= inv;
    // TRANSCRIBED FROM attention_decode.hip:165-168, NOT FROM THE HELPER.
    //
    //     const float sig = 1.0F / (1.0F + expf(-g));
    //     ctx *= sig;
    //
    // This line read `gate[i] / (1.0f + std::exp(-gate[i]))` — g*sigmoid(g),
    // SiLU — and the helper it was checking read the SAME thing. REF1 passed at
    // 0.000000 relative against a gate that was wrong. Deleting the gate line
    // made the suite red and proved nothing about the formula. A reference
    // copied from the code under test cannot disagree with it by construction,
    // so it is written from the source, and Turbo4SigmoidGatePinnedValues below
    // pins the arithmetic to literals neither side can move.
    if (gate != nullptr) out[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
  }
}

/// THE GATE, PINNED TO LITERALS.
///
/// These three numbers are what a logistic sigmoid must produce, computed by
/// hand and NOT by calling `Turbo4SigmoidGate` — a pinned value that is
/// computed from the code it is pinning proves nothing. They are what makes the
/// D1 class of bug impossible to hide again: the helper and this reference can
/// no longer be wrong in the same direction without these going off first.
///
///   gate(0)  = 1/(1+e^0)     = 0.5
///   gate(2)  = 1/(1+e^-2)    = 0.8807970779778823
///   gate(-2) = 1/(1+e^2)     = 0.11920292202211755
struct PinnedGate {
  float in;
  float want;
};
constexpr PinnedGate kPinnedGate[] = {
    {0.0f, 0.5f},
    {2.0f, 0.8807970779778823f},
    {-2.0f, 0.11920292202211755f},
};

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.0f;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

/// WORST ERROR RELATIVE TO THE SCALE OF THE WHOLE VECTOR, not per element.
///
/// A per-element relative error is meaningless here: an attention output that
/// crosses zero has a relative error of 1 whatever the codec did, because the
/// denominator is the value itself. The codec's error is a property of the
/// vector, so it is measured against the vector's largest magnitude. A genuine
/// defect — an uninitialised plane, a wrong head — moves that scale, not just
/// one element.
float ScaledError(const std::vector<float>& a, const std::vector<float>& b) {
  float scale = 0.0f;
  for (float x : a)
    if (std::fabs(x) > scale) scale = std::fabs(x);
  if (scale == 0.0f) return MaxAbsDiff(a, b);
  return MaxAbsDiff(a, b) / scale;
}

float MaxRelDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float worst = 0.0f;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const float denom = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
    if (denom <= 0.0f) continue;
    const float r = std::fabs(a[i] - b[i]) / denom;
    if (r > worst) worst = r;
  }
  return worst;
}

// ---------------------------------------------------------------------------
// A3 — the slice function, and its agreement with what the arena actually does.
// ---------------------------------------------------------------------------

void SliceGeometry() {
  const std::size_t plane =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kCtx);
  Check(plane == 1584U,
        "SLICE: one layer's plane at the real geometry is 1584 B (8 blocks x 3 "
        "tokens x 66 B) — the constant the rest of this file assumes");

  t4i::Turbo4LayerSlices sl{};
  Check(t4i::Turbo4LayerPlaneSlices(2 * plane * kLayers, 0, kHeadDim, kKvHeads,
                                    kCtx, kLayers, 0, &sl),
        "SLICE: layer 0 resolves in a live-sized arena");
  Check(sl.k_offset == 0U && sl.v_offset == plane * kLayers,
        "SLICE: layer 0 K is at 0 and V is after ALL 16 K layers, which is the "
        "region-major shape arena.cpp:673-677 copies");

  t4i::Turbo4LayerSlices last{};
  const std::size_t arena_bytes = 2 * plane * kLayers;
  Check(t4i::Turbo4LayerPlaneSlices(arena_bytes, 0, kHeadDim, kKvHeads, kCtx,
                                    kLayers, kLayers - 1, &last),
        "SLICE: the LAST layer resolves in an arena sized exactly for it");
  Check(last.v_offset + last.plane_bytes == arena_bytes,
        "SLICE: the last V slice ends EXACTLY at the end of the arena — an "
        "exact fit is a fit, not an off-by-one refusal");
  Check(last.k_offset == last.v_offset - plane * kLayers,
        "SLICE: every layer's K sits the same distance before its V");

  // ONE PAST THE LAST LAYER IS A REFUSAL, and it must be refused by the layer
  // bound rather than by an offset that happens to land somewhere legal.
  t4i::Turbo4LayerSlices past{};
  Check(!t4i::Turbo4LayerPlaneSlices(arena_bytes, 0, kHeadDim, kKvHeads, kCtx,
                                     kLayers, kLayers, &past),
        "SLICE: layer 16 of 16 refuses");
  Check(!t4i::Turbo4LayerPlaneSlices(arena_bytes, 0, kHeadDim, kKvHeads, kCtx,
                                     0, 0, &past),
        "SLICE: a zero-layer arena refuses rather than dividing by it");

  // ONE BYTE SHORT OF THE ARENA. The function requires the arena to hold BOTH
  // regions in full, not merely this layer's slice — a layer that resolves while
  // its V region is missing is a slice that will be read once the arena is
  // actually allocated, and the allocation is exactly `plane * layers * 2`.
  Check(!t4i::Turbo4LayerPlaneSlices(arena_bytes - 1, 0, kHeadDim, kKvHeads,
                                     kCtx, kLayers, kLayers - 1, &past),
        "SLICE: an arena one byte short refuses the last layer");
  Check(!t4i::Turbo4LayerPlaneSlices(arena_bytes - 1, 0, kHeadDim, kKvHeads,
                                     kCtx, kLayers, 0, &past),
        "SLICE: ...and it refuses EVERY layer, not just the last — the bound is "
        "the whole arena holding both regions, so no layer is ever handed an "
        "offset into a buffer that cannot back it");

  // THE COMPACT SNAPSHOT DIFFERS BY EXACTLY THE HEADER. This is the one number
  // that makes the two layouts interchangeable: if it were not exactly 80, a
  // slice computed for one layout would read the other one's header as a norm.
  const std::size_t header = 80;
  t4i::Turbo4LayerSlices compact{};
  Check(t4i::Turbo4LayerPlaneSlices(arena_bytes + header, header, kHeadDim,
                                    kKvHeads, kCtx, kLayers, 3, &compact),
        "SLICE: the compact layout resolves at a base of 80");
  t4i::Turbo4LayerSlices live3{};
  Check(t4i::Turbo4LayerPlaneSlices(arena_bytes, 0, kHeadDim, kKvHeads, kCtx,
                                    kLayers, 3, &live3),
        "SLICE: the live layout resolves at a base of 0");
  Check(compact.k_offset - live3.k_offset == header &&
            compact.v_offset - live3.v_offset == header,
        "SLICE: live and compact differ by EXACTLY the 80-byte compact header, "
        "for both K and V — no second difference hides here");

  // THE REAL ALLOCATION FORMULA AGREES. executor.hpp sizes the arena with
  // `QwenKvBytesForElements(kTurbo4, total_kv) * 2` over ALL layers; the slice
  // function builds the same span 16 planes at a time. If these ever diverge,
  // the last layer's slice silently runs past the allocation.
  const std::size_t total_kv =
      static_cast<std::size_t>(kLayers) * kCtx * kKvHeads * kHeadDim;
  const std::size_t exec_bytes =
      t4i::QwenKvBytesForElements(t4i::QwenKvCacheStorage::kTurbo4, total_kv) *
      2;
  Check(exec_bytes == arena_bytes,
        "SLICE: the slice span equals QwenKvBytesForElements(kTurbo4, all "
        "layers) * 2 — the sizing arena.cpp:1014 actually allocates ("
        + std::to_string(exec_bytes) + " vs " + std::to_string(arena_bytes) +
            ")");
}

// ---------------------------------------------------------------------------
// A1/A2/A3 — write two layers, read every query head back.
// ---------------------------------------------------------------------------

struct Fixture {
  std::vector<std::uint8_t> arena;
  LayerSource k[2];
  LayerSource v[2];
  std::vector<float> q;
  std::vector<float> gate;
  std::size_t plane{0};
  std::size_t arena_bytes{0};
};

/// Write `layer`'s K and V for all 3 tokens into the arena through the ACTUAL
/// writer body. Returns the number of heads that failed.
int WriteLayer(Fixture& f, std::uint32_t layer, std::vector<std::uint8_t>* kp,
               std::vector<std::uint8_t>* vp) {
  t4i::Turbo4LayerSlices sl{};
  if (!t4i::Turbo4LayerPlaneSlices(f.arena_bytes, 0, kHeadDim, kKvHeads, kCtx,
                                   kLayers, layer, &sl))
    return -1;
  kp->assign(f.arena.begin() + static_cast<std::ptrdiff_t>(sl.k_offset),
             f.arena.begin() + static_cast<std::ptrdiff_t>(sl.k_offset +
                                                          sl.plane_bytes));
  vp->assign(f.arena.begin() + static_cast<std::ptrdiff_t>(sl.v_offset),
             f.arena.begin() + static_cast<std::ptrdiff_t>(sl.v_offset +
                                                          sl.plane_bytes));
  int failed = 0;
  for (std::uint32_t tok = 0; tok < kCtx; ++tok) {
    // THE WRITER DERIVES THE HEAD OFFSET ITSELF — `head_base = head * head_dim`
    // inside the body. So the pointer handed in is the base of the whole
    // [num_kv_heads, head_dim] source for ONE TOKEN, not the head's own row.
    // Passing a per-head pointer double-counts the head, and on the LAST token
    // that reads past the end of the allocation: the failures landed on
    // (token 2, heads 2 and 3) of a 3-token buffer, which is exactly where
    // 2*4*256 + 2*256 lands. ASan catches it; without it, it is a quiet
    // quantize refusal that looks like a codec problem.
    const std::size_t tok_base = static_cast<std::size_t>(tok) * kKvHeads * kHeadDim;
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      const S st = t4i::Turbo4WriteHeadBody(
          f.k[layer].data() + tok_base, f.v[layer].data() + tok_base, kp->data(),
          vp->data(), sl.plane_bytes, kWriterSourceBytes, h, tok, kCtx, kKvHeads,
          kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
      if (st != S::kOk) ++failed;
    }
  }
  return failed;
}

t4i::Turbo4BufferSizes ReaderSizes(const Fixture& f, std::size_t gate_bytes) {
  const std::size_t rows =
      static_cast<std::size_t>(kQHeads) * kHeadDim * sizeof(float);
  return t4i::Turbo4BufferSizes{f.plane, rows, rows,
                                 static_cast<std::size_t>(kCtx) * sizeof(float),
                                 gate_bytes};
}

void WriteThenReadThroughTheRealBodies() {
  Fixture f;
  f.plane = t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kCtx);
  f.arena_bytes = 2 * f.plane * kLayers;
  f.arena.assign(f.arena_bytes, 0);
  for (std::uint32_t l = 0; l < 2; ++l) {
    f.k[l] = MakeSource(0x9E3779B97F4A7C15ULL + l * 7919ULL);
    f.v[l] = MakeSource(0xD1B54A32D192ED03ULL + l * 6271ULL);
  }
  {
    Lcg r{0x243F6A8885A308D3ULL};
    f.q.resize(static_cast<std::size_t>(kQHeads) * kHeadDim);
    for (float& x : f.q) x = r.Next();
    f.gate.resize(static_cast<std::size_t>(kQHeads) * kHeadDim);
    for (float& x : f.gate) x = r.Next() * 3.0f;
  }

  std::vector<std::uint8_t> k0, v0, k1, v1;
  Check(WriteLayer(f, 0, &k0, &v0) == 0,
        "E2E: layer 0 writes through Turbo4WriteHeadBody with no refusal");
  // LAYER ISOLATION. Layer 0's bytes are captured BEFORE layer 1 is written and
  // compared after. A slice function that addressed layer 0 while writing layer
  // 1 would pass every other check in this file: the values written would still
  // be the ones layer 1's reference expects, just at the wrong place.
  const std::vector<std::uint8_t> k0_before = k0;
  const std::vector<std::uint8_t> v0_before = v0;
  Check(WriteLayer(f, 1, &k1, &v1) == 0,
        "E2E: layer 1 writes through Turbo4WriteHeadBody with no refusal");
  Check(k0 == k0_before && v0 == v0_before,
        "E2E: writing layer 1 leaves layer 0's K and V bytes IDENTICAL "
        "(memcmp, not a re-read)");

  std::vector<std::uint8_t>* planes[2][2] = {{&k0, &v0}, {&k1, &v1}};
  LayerSource dk[2], dv[2];
  // ASSIGN ONCE PER LAYER, then dequantise into it head by head. This file
  // first cleared inside the head loop, which left dk holding ONLY the last kv
  // head — and the symptom was a reader that was correct disagreeing with a
  // reference that had quietly lost three quarters of its data.
  for (std::uint32_t l = 0; l < 2; ++l) {
    dk[l].assign(kCtx * kKvHeads * kHeadDim, 0.0f);
    dv[l].assign(kCtx * kKvHeads * kHeadDim, 0.0f);
    for (std::uint32_t h = 0; h < kKvHeads; ++h)
      DequantizeStored(planes[l][0]->data(), planes[l][1]->data(), h, &dk[l],
                       &dv[l]);
  }

  std::vector<float> out(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
  std::vector<float> scores(kCtx, 0.0f);
  const std::size_t gate_bytes = f.gate.size() * sizeof(float);

  // REFERENCE 1 (tight). Every query head, both layers, against dense float
  // attention over the ROUND-TRIPPED K/V plus the same sigmoid gate. This is
  // the check that a deleted gate line fails, and that a wrong GQA map fails.
  float worst_ref1 = 0.0f;
  int refusals = 0;
  for (std::uint32_t l = 0; l < 2; ++l) {
    for (std::uint32_t qh = 0; qh < kQHeads; ++qh) {
      const std::size_t off = static_cast<std::size_t>(qh) * kHeadDim;
      const S st = t4i::Turbo4AttentionQueryBody(
          f.q.data(), planes[l][0]->data(), planes[l][1]->data(), out.data(),
          ReaderSizes(f, gate_bytes), S::kOk, f.gate.data(), qh, kQHeads,
          kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
      if (st != S::kOk) ++refusals;
      // The GQA map, transcribed: group = 24/4 = 6, kv head = qh / 6.
      const std::uint32_t kv_head = qh / (kQHeads / kKvHeads);
      std::vector<float> ref(kHeadDim, 0.0f);
      DenseAttentionForKvHead(dk[l], dv[l], f.q.data() + off,
                               f.gate.data() + off, kv_head, ref.data());
      std::vector<float> got(out.begin() +
                                 static_cast<std::ptrdiff_t>(off),
                             out.begin() +
                                 static_cast<std::ptrdiff_t>(off + kHeadDim));
      const float d = MaxRelDiff(ref, got);
      if (d > worst_ref1) worst_ref1 = d;
    }
  }
  Check(refusals == 0, "E2E: all 48 read-backs (24 query heads x 2 layers) "
                       "return kOk");
  Check(worst_ref1 <= 1e-5f,
        "REF1 (tight): every gated query head matches dense float attention over "
        "the round-tripped plane to within 1e-5 relative (worst " +
            std::to_string(worst_ref1) + ")");

  // REFERENCE 2 (loose). The same attention over the ORIGINAL floats, with NO
  // gate, so what it measures is quantisation error and not the gate. It is a
  // bound, not a parity claim: 66 B per 128 values is a lossy format and the
  // threshold only catches an absurd result such as an uninitialised plane.
  float worst_ref2 = 0.0f;
  for (std::uint32_t l = 0; l < 2; ++l) {
    for (std::uint32_t qh = 0; qh < kQHeads; ++qh) {
      const std::size_t off = static_cast<std::size_t>(qh) * kHeadDim;
      const S st = t4i::Turbo4AttentionQueryBody(
          f.q.data(), planes[l][0]->data(), planes[l][1]->data(), out.data(),
          ReaderSizes(f, 0), S::kOk, nullptr, qh, kQHeads, kKvHeads, kHeadDim,
          kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
      if (st != S::kOk) continue;
      const std::uint32_t kv_head = qh / (kQHeads / kKvHeads);
      std::vector<float> ref(kHeadDim, 0.0f);
      DenseAttentionForKvHead(f.k[l], f.v[l], f.q.data() + off, nullptr,
                               kv_head, ref.data());
      std::vector<float> got(out.begin() +
                                 static_cast<std::ptrdiff_t>(off),
                             out.begin() +
                                 static_cast<std::ptrdiff_t>(off + kHeadDim));
      const float d = ScaledError(ref, got);
      if (d > worst_ref2) worst_ref2 = d;
    }
  }
  // THE BOUND HAS TO DISCRIMINATE, so it carries its own control. The same
  // measure is computed against a reference that is WRONG on purpose — dense
  // attention over the neighbouring kv head — and that number must be far
  // larger than the codec's. Without this, "under 0.1" is a number chosen to
  // pass, and a reader addressing the wrong head would sit under whatever
  // threshold the codec happened to need.
  float wrong_head_error = 0.0f;
  {
    // SIZED FOR ALL 24 HEADS, not one. The body writes `out` at
    // [query_head, head_dim], so a single-head buffer fed query head 18 writes
    // 7 KiB past its end. ASan found exactly that here when this control was
    // first written — which is the argument for running this file under ASan
    // even though the arithmetic was already green.
    std::vector<float> out(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
    std::vector<float> ref(kHeadDim, 0.0f);
    for (std::uint32_t g = 0; g < kKvHeads; ++g) {
      const std::uint32_t qh = g * (kQHeads / kKvHeads);
      const std::size_t off = static_cast<std::size_t>(qh) * kHeadDim;
      t4i::Turbo4AttentionQueryBody(
          f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, 0), S::kOk,
          nullptr, qh, kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
      DenseAttentionForKvHead(f.k[1], f.v[1], f.q.data() + off, nullptr,
                               (g + 1U) % kKvHeads, ref.data());
      const std::vector<float> got(
          out.begin() + static_cast<std::ptrdiff_t>(off),
          out.begin() + static_cast<std::ptrdiff_t>(off + kHeadDim));
      const float e = ScaledError(ref, got);
      if (e > wrong_head_error) wrong_head_error = e;
    }
  }
  std::printf(
      "        codec round-trip worst error, scaled by max|reference|: "
      "%.6f\n",
      static_cast<double>(worst_ref2));
  std::printf(
      "        same measure against a DELIBERATELY WRONG kv head:      "
      "%.6f\n",
      static_cast<double>(wrong_head_error));
  Check(worst_ref2 <= 0.35f,
        "REF2 (loose): codec error stays under 0.35 of the vector's scale "
        "against the original float K/V (worst " + std::to_string(worst_ref2) +
            ") — a bound on the format, NOT a parity claim: 4 bits per value "
            "with one shared norm per 128 is lossy by construction");
  Check(wrong_head_error > 3.0f * worst_ref2,
        "REF2 CONTROL: reading the WRONG kv head is at least 3x worse ("
        + std::to_string(wrong_head_error) + " vs " +
            std::to_string(worst_ref2) +
            "), so the bound above is wide enough for the format and still "
            "narrow enough to catch a wrong answer");

  // THE GATE IS NOT A NO-OP. Run the identical call twice, gate absent and
  // gate present. The logistic gate 1/(1+exp(-g)) is < 1 for every g > 0 and
  // > 0 for every g, so it has no fixed point these two could share — and a
  // body that ignored the pointer would make them identical.
  // (The D1 form, g/(1+exp(-g)), was g*sigmoid(g): also never 1 for g > 0, so
  // this particular check would NOT have distinguished it. The pinned literals
  // in GateFormulaIsPinned are what do that.)
  std::vector<float> no_gate(kHeadDim, 0.0f), with_gate(kHeadDim, 0.0f);
  const std::size_t qh0 = 0;
  t4i::Turbo4AttentionQueryBody(f.q.data(), k1.data(), v1.data(), no_gate.data(),
                                 ReaderSizes(f, 0), S::kOk, nullptr,
                                 static_cast<std::uint32_t>(qh0), kQHeads,
                                 kKvHeads, kHeadDim, kCtx, kScale,
                                 scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  t4i::Turbo4AttentionQueryBody(
      f.q.data(), k1.data(), v1.data(), with_gate.data(),
      ReaderSizes(f, gate_bytes), S::kOk, f.gate.data(),
      static_cast<std::uint32_t>(qh0), kQHeads, kKvHeads, kHeadDim, kCtx,
      kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(MaxAbsDiff(no_gate, with_gate) > 1e-4f,
        "GATE: the same read with and without a gate produces DIFFERENT output, "
        "so the gate is reaching the arithmetic");

  // ...and gate-absent is not merely different, it is the ungated reference.
  std::vector<float> ref_nogate(kHeadDim, 0.0f);
  DenseAttentionForKvHead(dk[1], dv[1], f.q.data(), nullptr,
                           qh0 / (kQHeads / kKvHeads), ref_nogate.data());
  Check(MaxAbsDiff(ref_nogate, no_gate) <= 1e-5f,
        "GATE: with gate_bytes == 0 the output equals the UNGATED reference "
        "byte-for-byte within 1e-5 — existing callers are unaffected");
}

// ---------------------------------------------------------------------------
// The GQA map, pinned by a check a re-implementation cannot satisfy.
// ---------------------------------------------------------------------------

void GroupedQueryMapBelow() {
  Fixture f;
  f.plane = t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kCtx);
  f.arena_bytes = 2 * f.plane * kLayers;
  f.arena.assign(f.arena_bytes, 0);
  for (std::uint32_t l = 0; l < 2; ++l) {
    f.k[l] = MakeSource(11ULL + l * 31ULL);
    f.v[l] = MakeSource(97ULL + l * 17ULL);
  }
  // ONE q vector, shared by all 24 query heads. Then the only thing that can
  // make two heads' outputs differ is WHICH kv head each one read, so the
  // outputs fall into exactly 4 identical groups of 6 — and a map that is off by
  // one head, or that splits a group, shows up immediately as a wrong grouping.
  // ALL 24 ROWS GET THE SAME VECTOR, not just row 0. `q` is indexed as
  // [query_head, head_dim], so filling only the first row leaves heads 1-23
  // reading zeros — which breaks the grouping below for a reason that has
  // nothing to do with the GQA map.
  std::vector<float> one_head(kHeadDim, 0.0f);
  Lcg r{0xB7E151628AED2A6BULL};
  for (std::size_t i = 0; i < kHeadDim; ++i) one_head[i] = r.Next();
  f.q.assign(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
  for (std::uint32_t qh = 0; qh < kQHeads; ++qh)
    for (std::size_t i = 0; i < kHeadDim; ++i)
      f.q[static_cast<std::size_t>(qh) * kHeadDim + i] = one_head[i];

  std::vector<std::uint8_t> k1, v1;
  if (WriteLayer(f, 0, &k1, &v1) != 0) {
    Check(false, "GQA MAP: fixture write failed");
    return;
  }

  std::vector<float> out(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
  std::vector<float> scores(kCtx, 0.0f);
  int ok_count = 0;
  for (std::uint32_t qh = 0; qh < kQHeads; ++qh) {
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, 0), S::kOk,
        nullptr, qh, kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    if (st == S::kOk) ++ok_count;
  }
  Check(ok_count == static_cast<int>(kQHeads),
        "GQA MAP: all 24 query heads read successfully");

  // Exactly 4 distinct outputs, each shared by exactly 6 heads, contiguous.
  bool grouped = true;
  for (std::uint32_t g = 0; g < kKvHeads && grouped; ++g) {
    const std::size_t off = static_cast<std::size_t>(g * 6) * kHeadDim;
    const std::vector<float> rep(out.begin() +
                                     static_cast<std::ptrdiff_t>(off),
                                 out.begin() +
                                     static_cast<std::ptrdiff_t>(off + kHeadDim));
    // Every head in the group matches the group's first head...
    for (std::uint32_t m = 1; m < 6; ++m) {
      const std::size_t o = static_cast<std::size_t>(g * 6 + m) * kHeadDim;
      const std::vector<float> got(out.begin() +
                                       static_cast<std::ptrdiff_t>(o),
                                   out.begin() +
                                       static_cast<std::ptrdiff_t>(o + kHeadDim));
      if (MaxAbsDiff(rep, got) > 0.0f) grouped = false;
    }
    // ...and every head in the NEXT group differs, so this is 4 real groups
    // rather than 24 copies of one answer.
    if (g + 1 < kKvHeads) {
      const std::size_t o = static_cast<std::size_t>((g + 1) * 6) * kHeadDim;
      const std::vector<float> nxt(out.begin() +
                                       static_cast<std::ptrdiff_t>(o),
                                   out.begin() +
                                       static_cast<std::ptrdiff_t>(o + kHeadDim));
      if (MaxAbsDiff(rep, nxt) == 0.0f) grouped = false;
    }
  }
  Check(grouped,
        "GQA MAP: 24 query heads over 4 kv heads form exactly 4 groups of 6 "
        "IDENTICAL outputs, and the groups differ from each other — a wrong map "
        "cannot produce this");

  // And each group really is its kv head's own data.
  LayerSource dk, dv;
  dk.assign(kCtx * kKvHeads * kHeadDim, 0.0f);
  dv.assign(kCtx * kKvHeads * kHeadDim, 0.0f);
  bool matches = true;
  for (std::uint32_t g = 0; g < kKvHeads; ++g) {
    DequantizeStored(k1.data(), v1.data(), g, &dk, &dv);
    std::vector<float> ref(kHeadDim, 0.0f);
    DenseAttentionForKvHead(dk, dv, f.q.data(), nullptr, g, ref.data());
    const std::size_t off = static_cast<std::size_t>(g * 6) * kHeadDim;
    const std::vector<float> got(out.begin() +
                                     static_cast<std::ptrdiff_t>(off),
                                 out.begin() +
                                     static_cast<std::ptrdiff_t>(off + kHeadDim));
    if (MaxAbsDiff(ref, got) > 1e-5f) matches = false;
  }
  Check(matches,
        "GQA MAP: group g equals dense attention over kv head g's own stored "
        "plane, so the grouping above is the RIGHT grouping");
}

// ---------------------------------------------------------------------------
// A2 — the writer-failure word, and every refusal that must leave `out` alone.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// D1 — the gate FORMULA, pinned to hand-computed literals.
// ---------------------------------------------------------------------------

void GateFormulaIsPinned() {
  // The check that would have caught D1. It calls the production helper and
  // compares against numbers computed by hand from 1/(1+exp(-g)), so the two
  // sides cannot be wrong in the same direction: agreeing with this file used
  // to mean agreeing with a bug, because both sides came from the same memory.
  float worst = 0.0f;
  std::string detail;
  for (const PinnedGate& p : kPinnedGate) {
    const float got = t4i::Turbo4SigmoidGate(p.in);
    const float d = std::fabs(got - p.want);
    if (d > worst) worst = d;
    if (d > 1e-6f)
      detail += " gate(" + std::to_string(p.in) + ")=" +
                std::to_string(got) + " want " + std::to_string(p.want);
  }
  Check(worst <= 1e-6f,
        "GATE FORMULA: 1/(1+exp(-g)) at g=0, 2, -2 matches hand-computed "
        "literals to 1e-6" +
            detail);

  // THE SPECIFIC REGRESSION, named so a future diff cannot reintroduce it
  // quietly: g*sigmoid(g) is 0 at g=0 and 1.7616 at g=2, so the old wrong
  // helper fails the first pin alone.
  Check(std::fabs(t4i::Turbo4SigmoidGate(0.0f) - 0.0f) > 1e-6f,
        "GATE FORMULA: gate(0) is 0.5 and NOT 0 — the g*sigmoid(g) (SiLU) form "
        "that D1 caught is excluded by this single value");

  // A GATE IS STRICTLY BOUNDED IN (0,1) FOR EVERY INPUT. The wrong form is not:
  // it grows without limit and goes negative for g < 0, which would let a
  // negative gate flip the sign of an attention output rather than attenuate it.
  bool bounded = true;
  for (int i = -200; i <= 200 && bounded; ++i) {
    const float g = static_cast<float>(i) / 20.0f;
    const float v = t4i::Turbo4SigmoidGate(g);
    if (!(v > 0.0f && v < 1.0f)) bounded = false;
  }
  Check(bounded,
        "GATE FORMULA: over g in [-10, 10] the gate stays strictly inside (0, "
        "1) — so it can only attenuate an output, never invert or amplify it");

  // AND IT IS MONOTONE, which the wrong form is not across the whole range
  // (g*sigmoid(g) falls for large negative g).
  bool monotone = true;
  float prev = -1.0f;
  for (int i = -100; i <= 100; ++i) {
    const float g = static_cast<float>(i) / 10.0f;
    const float v = t4i::Turbo4SigmoidGate(g);
    if (v < prev) monotone = false;
    prev = v;
  }
  Check(monotone, "GATE FORMULA: the gate is monotone increasing in g over "
                  "[-10, 10]");
}

// ---------------------------------------------------------------------------
// D2 — a null writer word is a WRITER FAILURE, not an assumed success.
// ---------------------------------------------------------------------------

void NullWriterWordFailsClosed() {
  // The launcher refuses a null `d_write_status`, but the KERNEL is reachable
  // without it — so the decision lives in Turbo4ReadWriterStatus, host+device,
  // and is called here rather than grepped for. A source-text check could only
  // have confirmed a comment; this one actually runs the branch.
  Check(t4i::Turbo4ReadWriterStatus(nullptr) == S::kWriterFailed,
        "D2: a null writer word reads as kWriterFailed, NOT kOk — no evidence "
        "the writer ran is a writer failure");

  const S ok = S::kOk;
  Check(t4i::Turbo4ReadWriterStatus(&ok) == S::kOk,
        "D2: a real word with kOk in it reads as kOk — the reader still works");

  for (S ws : {S::kGeometryRefused, S::kPositionOutOfRange, S::kQuantizeRefused,
               S::kPoisonedNorm, S::kDegenerateSoftmax, S::kWriterFailed}) {
    Check(t4i::Turbo4ReadWriterStatus(&ws) == ws,
          "D2: writer status " + std::to_string(static_cast<unsigned>(ws)) +
              " passes through unchanged — the helper does not invent a code");
  }

  // AND THE CONSEQUENCE, end to end: a reader handed the value this helper
  // returns for a null word must refuse, not attend over a plane nobody wrote.
  {
    Fixture f;
    f.plane = t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kCtx);
    f.arena_bytes = 2 * f.plane * kLayers;
    f.arena.assign(f.arena_bytes, 0);
    f.k[0] = MakeSource(41ULL);
    f.v[0] = MakeSource(43ULL);
    std::vector<std::uint8_t> k1, v1;
    if (WriteLayer(f, 0, &k1, &v1) != 0) {
      Check(false, "D2: fixture write failed");
      return;
    }
    Lcg r{77ULL};
    f.q.assign(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
    for (float& x : f.q) x = r.Next();
    std::vector<float> scores(kCtx, 0.0f);
    const auto sentinel = std::vector<float>(kHeadDim, -4242.0f);
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, 0),
        t4i::Turbo4ReadWriterStatus(nullptr), nullptr, 0, kQHeads, kKvHeads,
        kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kWriterFailed && out == sentinel,
          "D2: a reader given the null-word result refuses with kWriterFailed "
          "and leaves `out` untouched");
  }
}

void WriterFailureBlocksTheReader() {
  Fixture f;
  f.plane = t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kCtx);
  f.arena_bytes = 2 * f.plane * kLayers;
  f.arena.assign(f.arena_bytes, 0);
  f.k[0] = MakeSource(5ULL);
  f.v[0] = MakeSource(9ULL);
  std::vector<std::uint8_t> k1, v1;
  if (WriteLayer(f, 0, &k1, &v1) != 0) {
    Check(false, "WRITER-STATUS: fixture write failed");
    return;
  }
  {
    Lcg r{3ULL};
    f.q.assign(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
    for (float& x : f.q) x = r.Next();
  }
  std::vector<float> scores(kCtx, 0.0f);

  const auto sentinel = std::vector<float>(kHeadDim, -12345.0f);
  for (S ws : {S::kGeometryRefused, S::kPositionOutOfRange, S::kQuantizeRefused,
               S::kPoisonedNorm, S::kDegenerateSoftmax}) {
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, 0), ws,
        nullptr, 0, kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kWriterFailed && out == sentinel,
          "WRITER-STATUS: writer status " +
              std::to_string(static_cast<unsigned>(ws)) +
              " reports kWriterFailed and leaves `out` BYTE-IDENTICAL");
  }

  // ORDER MATTERS. A caller whose plane is ALSO short should hear about the
  // writer: that is the reason the context is incoherent, and the short plane
  // is a downstream symptom. If this returned kGeometryRefused the body would
  // be reporting the wrong reason for a real condition.
  std::vector<float> out = sentinel;
  t4i::Turbo4BufferSizes short_plane = ReaderSizes(f, 0);
  short_plane.plane_bytes -= 1;
  const S both = t4i::Turbo4AttentionQueryBody(
      f.q.data(), k1.data(), v1.data(), out.data(), short_plane,
      S::kQuantizeRefused, nullptr, 0, kQHeads, kKvHeads, kHeadDim, kCtx,
      kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(both == S::kWriterFailed && out == sentinel,
        "WRITER-STATUS: the writer is checked BEFORE the capacity gate, so a "
        "failed writer on a short plane still reports kWriterFailed");

  // ...and the writer being OK leaves the capacity refusal to the capacity gate.
  std::vector<float> out2 = sentinel;
  const S only_short = t4i::Turbo4AttentionQueryBody(
      f.q.data(), k1.data(), v1.data(), out2.data(), short_plane, S::kOk,
      nullptr, 0, kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(only_short == S::kGeometryRefused && out2 == sentinel,
        "WRITER-STATUS: with the writer OK, the same short plane is refused by "
        "the capacity gate instead — the two checks are independent");

  // THE LATCH RANKS IT ABOVE THE OTHERS. kWriterFailed is the largest value, so
  // a maximum-latch reports it over any other code when both occur.
  S latched = S::kOk;
  t4i::Turbo4LatchFailure(&latched, S::kPoisonedNorm);
  t4i::Turbo4LatchFailure(&latched, S::kWriterFailed);
  t4i::Turbo4LatchFailure(&latched, S::kOk);
  Check(latched == S::kWriterFailed,
        "WRITER-STATUS: the max-latch keeps kWriterFailed when a kOk follows it, "
        "so a success cannot erase a writer failure");
}

// ---------------------------------------------------------------------------
// A1 — the gate's declaration rules and its refusals.
// ---------------------------------------------------------------------------

void GateDeclarationIsChecked() {
  Fixture f;
  f.plane = t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kCtx);
  f.arena_bytes = 2 * f.plane * kLayers;
  f.arena.assign(f.arena_bytes, 0);
  f.k[0] = MakeSource(13ULL);
  f.v[0] = MakeSource(21ULL);
  std::vector<std::uint8_t> k1, v1;
  if (WriteLayer(f, 0, &k1, &v1) != 0) {
    Check(false, "GATE RULES: fixture write failed");
    return;
  }
  Lcg r{0xC2B2AE3D27D4EB4FULL};
  f.q.assign(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
  f.gate.assign(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
  for (std::size_t i = 0; i < f.q.size(); ++i) {
    f.q[i] = r.Next();
    f.gate[i] = r.Next() * 2.0f;
  }
  std::vector<float> scores(kCtx, 0.0f);
  const std::size_t rows =
      static_cast<std::size_t>(kQHeads) * kHeadDim * sizeof(float);
  const auto sentinel = std::vector<float>(kHeadDim, -777.0f);

  // A POINTER WITH NO BYTES. The caller stated a gate and gave nothing to read.
  {
    std::vector<float> out = sentinel;
    t4i::Turbo4BufferSizes sz{f.plane, rows, rows, kCtx * sizeof(float), 0U};
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), sz, S::kOk, f.gate.data(),
        0, kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
          "GATE RULES: a gate pointer with gate_bytes == 0 refuses — the two "
          "halves of the declaration must agree");
  }
  // BYTES WITH NO POINTER.
  {
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, rows), S::kOk,
        nullptr, 0, kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
          "GATE RULES: a non-zero gate_bytes with a null gate pointer refuses "
          "for the same reason");
  }
  // A GATE ONE ROW SHORT. Non-null and non-zero, so the declaration agrees; the
  // buffer simply does not hold [24, 256].
  {
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(),
        ReaderSizes(f, rows - kHeadDim * sizeof(float)), S::kOk, f.gate.data(), 0,
        kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
          "GATE RULES: a gate one query head short of [num_heads, head_dim] "
          "refuses rather than reading past it");
  }
  // EXACTLY ENOUGH IS ENOUGH.
  {
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, rows), S::kOk,
        f.gate.data(), 0, kQHeads, kKvHeads, kHeadDim, kCtx, kScale,
        scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kOk && out != sentinel,
          "GATE RULES: a gate of EXACTLY num_heads * head_dim floats is "
          "accepted — the bound is tight, not blanket");
  }
  // A POISONED GATE. Checked above phase 4, so `out` is still untouched.
  {
    std::vector<float> poisoned = f.gate;
    poisoned[5] = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, rows), S::kOk,
        poisoned.data(), 0, kQHeads, kKvHeads, kHeadDim, kCtx, kScale,
        scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
          "GATE RULES: a NaN in the gate refuses with `out` untouched — the scan "
          "runs BEFORE the first write to out, or the refusal would clobber it");
  }
  // THE LAST QUERY HEAD'S GATE ROW IS THE TIGHT ONE: it ends at byte 24*1024.
  // A buffer sized for 23 rows passes every head except this one, so it is the
  // only row where an off-by-one in the gate's own indexing would be caught.
  {
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(),
        ReaderSizes(f, rows - kHeadDim * sizeof(float)), S::kOk, f.gate.data(),
        kQHeads - 1, kQHeads, kKvHeads, kHeadDim, kCtx, kScale,
        scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
        "GATE RULES: query head 23 refuses on a gate buffer sized for 23 rows "
        "— the tight row is gated too");
  }
  // THE GATE IS VALIDATED AS A WHOLE [num_heads, head_dim] TENSOR, exactly like
  // `q` and `out` — not as "enough for the head you happen to be reading". A
  // 23-row gate therefore refuses head 0 as well, even though head 0's own row
  // is fully inside it. That is the same whole-tensor contract the q/out
  // capacity checks use, and it is the safer one: a per-head bound would let a
  // caller pass a short gate and have the refusal depend on which head ran.
  {
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(),
        ReaderSizes(f, rows - kHeadDim * sizeof(float)), S::kOk, f.gate.data(), 0,
        kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
          "GATE RULES: ...and query head 0 refuses on that same 23-row gate too, "
          "because the gate is bounded as a whole [num_heads, head_dim] tensor, "
          "not per head");
  }
}

// ---------------------------------------------------------------------------
// The reader's own refusals, over a real written plane.
// ---------------------------------------------------------------------------

void ReaderRefusalsOverARealPlane() {
  Fixture f;
  f.plane = t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kCtx);
  f.arena_bytes = 2 * f.plane * kLayers;
  f.arena.assign(f.arena_bytes, 0);
  f.k[0] = MakeSource(31ULL);
  f.v[0] = MakeSource(37ULL);
  std::vector<std::uint8_t> k1, v1;
  if (WriteLayer(f, 0, &k1, &v1) != 0) {
    Check(false, "REFUSALS: fixture write failed");
    return;
  }
  Lcg r{0x6A09E667F3BCC909ULL};
  f.q.assign(static_cast<std::size_t>(kQHeads) * kHeadDim, 0.0f);
  for (float& x : f.q) x = r.Next();
  std::vector<float> scores(kCtx, 0.0f);
  const auto sentinel = std::vector<float>(kHeadDim, -31337.0f);

  // PLANE ONE BYTE SHORT. The whole-buffer requirement is
  // Turbo4LayerPlaneBytes(head_dim, kv_heads, valid_context), so shaving one byte
  // off must refuse — and must refuse WITHOUT having read the missing byte.
  {
    std::vector<float> out = sentinel;
    t4i::Turbo4BufferSizes sz = ReaderSizes(f, 0);
    sz.plane_bytes -= 1;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), sz, S::kOk, nullptr, 0,
        kQHeads, kKvHeads, kHeadDim, kCtx, kScale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
          "REFUSALS: a plane one byte short of 3 tokens refuses with `out` "
          "untouched");
  }
  // ONE TOKEN MORE THAN THE PLANE HOLDS. Every offset this addresses is
  // individually valid against the CONTEXT the body was handed, which is why
  // this had to be caught against the real allocation.
  {
    std::vector<float> out = sentinel;
    std::vector<float> s4(4, 0.0f);
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), k1.data(), v1.data(), out.data(), ReaderSizes(f, 0), S::kOk,
        nullptr, 0, kQHeads, kKvHeads, kHeadDim, 4U, kScale, s4.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kGeometryRefused && out == sentinel,
          "REFUSALS: claiming 4 tokens against a 3-token plane refuses even "
          "though every offset it computes is individually legal");
  }
  // A POISONED STORED NORM, in a plane that was genuinely written. The norm is
  // bytes 0-1 of a block; fp16 +infinity is 0x7C00, so byte 1 = 0x7C.
  {
    std::vector<std::uint8_t> poisoned = k1;
    poisoned[1] = 0x7C;
    std::vector<float> out = sentinel;
    const S st = t4i::Turbo4AttentionQueryBody(
        f.q.data(), poisoned.data(), v1.data(), out.data(), ReaderSizes(f, 0),
        S::kOk, nullptr, 0, kQHeads, kKvHeads, kHeadDim, kCtx, kScale,
        scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(st == S::kPoisonedNorm && out == sentinel,
          "REFUSALS: a poisoned stored norm in a real plane reports "
          "kPoisonedNorm and decodes nothing");
  }
  // A QUANTIZATION REFUSAL DOES NOT HALF-WRITE A HEAD. The writer stages every
  // block of K and V before committing any of them, so a NaN in one token leaves
  // the whole plane untouched. Written into a layer nothing else in this file
  // touches, and compared byte for byte.
  {
    t4i::Turbo4LayerSlices sl{};
    if (!t4i::Turbo4LayerPlaneSlices(f.arena_bytes, 0, kHeadDim, kKvHeads, kCtx,
                                     kLayers, 5, &sl)) {
      Check(false, "REFUSALS: could not resolve the staging layer");
      return;
    }
    std::vector<std::uint8_t> kslice(sl.plane_bytes, 0xA5);
    std::vector<std::uint8_t> vslice(sl.plane_bytes, 0x5A);
    LayerSource bad = f.k[0];
    bad[SourceIndex(1, 0)] = std::numeric_limits<float>::quiet_NaN();
    const std::size_t tok1 = static_cast<std::size_t>(1) * kKvHeads * kHeadDim;
    const S st = t4i::Turbo4WriteHeadBody(bad.data() + tok1,
                                          f.v[0].data() + tok1, kslice.data(),
                                          vslice.data(), sl.plane_bytes,
                                          kWriterSourceBytes, 0, 1, kCtx,
                                          kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    bool intact = true;
    for (std::size_t i = 0; i < sl.plane_bytes; ++i) {
      if (kslice[i] != 0xA5 || vslice[i] != 0x5A) intact = false;
    }
    Check(st == S::kQuantizeRefused && intact,
          "REFUSALS: a NaN in one token's K refuses the write AND leaves both "
          "planes byte-identical — the head is staged whole or not at all");
  }
}

}  // namespace

int main() {
  SliceGeometry();
  WriteThenReadThroughTheRealBodies();
  GroupedQueryMapBelow();
  GateFormulaIsPinned();
  NullWriterWordFailsClosed();
  WriterFailureBlocksTheReader();
  GateDeclarationIsChecked();
  ReaderRefusalsOverARealPlane();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
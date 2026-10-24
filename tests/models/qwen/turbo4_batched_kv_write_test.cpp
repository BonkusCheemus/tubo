// The BATCHED Turbo4 KV write, checked as BYTES.
//
// WHY BYTES AND NOT SHAPES (lesson from the snapshot-copy bug). A batched writer that writes
// the right NUMBER of blocks in the right ORDER can still be wrong in every way
// that matters: a block written from the wrong source offset produces a
// well-formed plane full of another position's values, and a shape check calls
// that a pass. So this compares the PLANE BYTES against the single-position
// writer the engine already ships, and requires them equal byte for byte.
//
// THE ORACLE IS NOT A RE-IMPLEMENTATION. `Turbo4WriteHeadBody` is the function
// the kernel itself calls (kernels/turbo4_kv_write.hip:50). Calling it N times
// is what the engine does today, so it is the definition of correct here. This
// file adds no independent maths: the only new thing is that the batched path
// reads a CHUNK-major source and must land identical bytes.
//
// THE POINT OF THE TEST IS THE SOURCE LAYOUT DIFFERENCE. The single-position
// API is handed one position: [num_kv_heads, head_dim]. A chunk is
// [count, num_kv_heads, head_dim]. Every indexing mistake this test can catch is
// a mistake in translating between those two.
//
// MID-BLOCK CHUNKS. pos0 = 0 is the friendly case. A chunk that starts at an
// arbitrary position has to land its first block at the plane offset that
// position really owns, so unaligned starts are included on purpose.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/turbo4_codec.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_plane_index.hpp"

namespace t4i = gufo::hip;
namespace t4 = gufo::hip::turbo4;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kBlocksPerHead = kHeadDim / t4::kBlockValues;  // 2
constexpr std::size_t kPerPosSrc = kKvHeads * kHeadDim;

/// Deterministic, finite, and NOT smooth: a smooth ramp compresses to a handful
/// of centroids and would hide an indexing bug behind a lucky match.
float Noise(std::size_t seed) {
  std::uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  const float unit = static_cast<float>(x >> 40) / 16777216.0f;  // [0,1)
  return (unit - 0.5f) * 6.0f;
}

std::vector<float> MakeChunk(std::size_t count) {
  std::vector<float> src(count * kPerPosSrc);
  for (std::size_t i = 0; i < src.size(); ++i)
    src[i] = Noise(i + 1);
  return src;
}

/// The plane bytes, from N single-position writes — the engine's current path.
void WriteOneAtATime(const std::vector<float>& k, const std::vector<float>& v,
                     std::size_t count, std::uint32_t first_pos,
                     std::uint32_t context, std::vector<std::uint8_t>* k_plane,
                     std::vector<std::uint8_t>* v_plane) {
  for (std::size_t p = 0; p < count; ++p) {
    const float* kp = k.data() + p * kPerPosSrc;
    const float* vp = v.data() + p * kPerPosSrc;
    const auto st = t4i::Turbo4WriteHeadBody(
        kp, vp, k_plane->data(), v_plane->data(), k_plane->size(),
        kPerPosSrc * sizeof(float), /*head=*/0U,
        static_cast<std::uint32_t>(first_pos + p), context,
        static_cast<std::uint32_t>(kKvHeads),
        static_cast<std::uint32_t>(kHeadDim), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    if (st != t4i::Turbo4KernelStatus::kOk) std::printf("  single write refused\n");
  }
}

void BatchedMatchesSingle(std::size_t count, std::uint32_t first_pos,
                         std::uint32_t context) {
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  const std::vector<float> k = MakeChunk(count);
  const std::vector<float> v = MakeChunk(count + 9777);  // distinct stream

  std::vector<std::uint8_t> k_ref(plane_bytes, 0xAB), v_ref(plane_bytes, 0xCD);
  std::vector<std::uint8_t> k_bat(plane_bytes, 0xAB), v_bat(plane_bytes, 0xCD);

  WriteOneAtATime(k, v, count, first_pos, context, &k_ref, &v_ref);

  const auto st = t4i::Turbo4WriteHeadRangeBody(
      k.data(), v.data(), k_bat.data(), v_bat.data(), plane_bytes,
      count * kPerPosSrc * sizeof(float), /*head=*/0U, first_pos,
      static_cast<std::uint32_t>(count), context,
      static_cast<std::uint32_t>(kKvHeads),
      static_cast<std::uint32_t>(kHeadDim), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));

  const std::string tag = "count=" + std::to_string(count) +
                          " first_pos=" + std::to_string(first_pos);
  if (st != t4i::Turbo4KernelStatus::kOk) {
    Check(false, tag + ": batched write returns kOk");
    return;
  }
  // WHOLE plane, not just the touched span: a batched writer that scribbled
  // past its range would otherwise pass.
  const bool k_same = std::memcmp(k_ref.data(), k_bat.data(), plane_bytes) == 0;
  const bool v_same = std::memcmp(v_ref.data(), v_bat.data(), plane_bytes) == 0;
  Check(k_same && v_same, tag + ": batched plane is BYTE-EQUAL to N single writes");
}

void RefusesWithoutWriting() {
  const std::uint32_t context = 64;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  const std::vector<float> k = MakeChunk(4);
  const std::vector<float> v = MakeChunk(4);
  std::vector<std::uint8_t> kp(plane_bytes, 0x11), vp(plane_bytes, 0x22);

  // A range that runs past the context must refuse, and must not half-write.
  const auto st = t4i::Turbo4WriteHeadRangeBody(
      k.data(), v.data(), kp.data(), vp.data(), plane_bytes,
      4 * kPerPosSrc * sizeof(float), 0U, /*first_pos=*/context - 2U,
      /*count=*/8U, context, static_cast<std::uint32_t>(kKvHeads),
      static_cast<std::uint32_t>(kHeadDim), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(st != t4i::Turbo4KernelStatus::kOk,
        "a range past max_context is refused, not clipped");
  bool untouched = true;
  for (std::size_t i = 0; i < plane_bytes; ++i)
    if (kp[i] != 0x11 || vp[i] != 0x22) untouched = false;
  Check(untouched, "a refused range leaves the plane untouched");
}

void RewritingOnePositionPreservesEveryNeighborByte() {
  const std::uint32_t context = 128;
  const std::uint32_t target = 65;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  auto k = MakeChunk(1);
  auto v = MakeChunk(1);
  std::vector<std::uint8_t> first_k(plane_bytes, 0xA5),
      first_v(plane_bytes, 0x5A);
  const auto initial = t4i::Turbo4WriteHeadRangeBody(
      k.data(), v.data(), first_k.data(), first_v.data(), plane_bytes,
      kPerPosSrc * sizeof(float), 0U, target, 1U, context,
      static_cast<std::uint32_t>(kKvHeads),
      static_cast<std::uint32_t>(kHeadDim), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(initial == t4i::Turbo4KernelStatus::kOk,
        "first target-position write succeeds");

  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = -k[i] * 2.0F + 0.25F;
    v[i] = v[i] * 3.0F - 0.5F;
  }
  auto rewritten_k = first_k;
  auto rewritten_v = first_v;
  const auto rewrite = t4i::Turbo4WriteHeadRangeBody(
      k.data(), v.data(), rewritten_k.data(), rewritten_v.data(), plane_bytes,
      kPerPosSrc * sizeof(float), 0U, target, 1U, context,
      static_cast<std::uint32_t>(kKvHeads),
      static_cast<std::uint32_t>(kHeadDim), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(rewrite == t4i::Turbo4KernelStatus::kOk,
        "rewriting the target position succeeds");
  std::size_t target_base = 0;
  Check(t4i::Turbo4BlockByteOffset(target, 0U, kHeadDim, kKvHeads, context,
                                    &target_base),
        "target position has a block offset");
  bool neighbors_equal = true;
  for (std::size_t offset = 0; offset < plane_bytes; ++offset) {
    if (offset >= target_base &&
        offset < target_base +
            t4i::Turbo4BlocksPerToken(kHeadDim, kKvHeads) * t4::kBlockBytes) {
      continue;
    }
    if (rewritten_k[offset] != first_k[offset] ||
        rewritten_v[offset] != first_v[offset]) {
      neighbors_equal = false;
      break;
    }
  }
  Check(neighbors_equal,
        "rewriting position p leaves every neighboring Turbo4 K/V byte unchanged");
}

void TouchesOnlyItsOwnPositions() {
  // Positions outside the range must keep their sentinel bytes.
  const std::uint32_t context = 128;
  const std::uint32_t first_pos = 5;
  const std::uint32_t count = 6;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  const std::vector<float> k = MakeChunk(count);
  const std::vector<float> v = MakeChunk(count);
  std::vector<std::uint8_t> kp(plane_bytes, 0), vp(plane_bytes, 0);

  const auto st = t4i::Turbo4WriteHeadRangeBody(
      k.data(), v.data(), kp.data(), vp.data(), plane_bytes,
      count * kPerPosSrc * sizeof(float), 0U, first_pos, count, context,
      static_cast<std::uint32_t>(kKvHeads),
      static_cast<std::uint32_t>(kHeadDim), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(st == t4i::Turbo4KernelStatus::kOk, "range write returns kOk");

  int touched_positions = 0;
  for (std::uint32_t pos = 0; pos < context; ++pos) {
    std::size_t base = 0;
    if (!t4i::Turbo4BlockByteOffset(pos, 0U, kHeadDim, kKvHeads, context, &base))
      continue;
    const bool in_range = pos >= first_pos && pos < first_pos + count;
    bool any = false;
    for (std::size_t b = 0; b < kBlocksPerHead * 2; ++b) {
      const std::size_t off = base + b * t4::kBlockBytes;
      if (off + t4::kBlockBytes > plane_bytes) break;
      for (std::size_t i = 0; i < t4::kBlockBytes; ++i)
        if (kp[off + i] != 0 || vp[off + i] != 0) any = true;
    }
    if (any) {
      ++touched_positions;
      if (!in_range) {
        Check(false, "position " + std::to_string(pos) + " outside the range was written");
        return;
      }
    }
  }
  Check(touched_positions == static_cast<int>(count),
        "exactly the range's positions were written (" +
            std::to_string(touched_positions) + " of " + std::to_string(count) + ")");
}

}  // namespace

int main() {
  RewritingOnePositionPreservesEveryNeighborByte();
  std::printf("E1 batched Turbo4 KV write: bytes vs N single-position writes\n");

  std::printf("geometry: %zu blocks/head, %zu bytes/block\n", kBlocksPerHead,
              t4::kBlockBytes);

  // Counts spanning one block, a full head, and several token pitches.
  for (std::size_t count : {1u, 2u, 3u, 5u, 8u, 17u, 64u})
    BatchedMatchesSingle(count, 0U, 512);

  // Unaligned and mid-block starts.
  for (std::uint32_t pos : {1u, 3u, 5u, 17u, 33u, 129u})
    BatchedMatchesSingle(7u, pos, 512);

  RefusesWithoutWriting();
  TouchesOnlyItsOwnPositions();

  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
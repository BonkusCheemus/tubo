// Dequantize a Turbo4 layer into an f16 scratch, checked
// BIT-EXACT against a host decode of the same blocks.
//
// WHAT IS AND IS NOT INDEPENDENT HERE. The reference below is the SAME decode
// the single-position READ body performs (turbo4_kernels.hpp:451-453 for K,
// :485-487 for V): DequantizeBlock, then RotateInverse, then to half. It is not
// new maths — inventing a second decode would create a second definition of
// what a block means, which is the failure this whole unit exists to avoid.
// What IS independent, and what the test actually checks, is the INDEXING:
// which block belongs to which (position, head), and where each element lands
// in the f16 scratch.
//
// LAYOUT IS TAKEN FROM THE CONSUMER. attention_wmma.hip:143 defaults
// kHeadMajor=false, so :253-255 makes the offset
// `pos * kKvWidth + kv_head * head_dim + i`. A scratch laid out any other way
// is the exact bug E2 exists to catch, so the test writes the destination with
// that expression spelled out rather than reusing the body's arithmetic.
//
// BIT-EXACT, not close. The output is half bits; the reference produces half
// bits; they are compared as integers, so a one-ULP drift is a failure.

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
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;
constexpr std::size_t kBlocksPerHead = kHeadDim / t4::kBlockValues;

float Noise(std::size_t seed) {
  std::uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  const float unit = static_cast<float>(x >> 40) / 16777216.0f;
  return (unit - 0.5f) * 6.0f;
}

/// Build a plane with a real write, so the blocks under test are ones the codec
/// actually produced rather than hand-made bytes.
std::vector<std::uint8_t> MakePlane(std::uint32_t context, std::uint32_t seed) {
  const std::size_t bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  std::vector<std::uint8_t> plane(bytes, 0);
  std::vector<float> src(kHeadDim);
  for (std::uint32_t t = 0; t < context; ++t) {
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      for (std::size_t i = 0; i < kHeadDim; ++i)
        src[i] = Noise(seed + t * 7919U + h * 104729U + i);
      std::size_t base = 0;
      if (!t4i::Turbo4BlockByteOffset(t, h, kHeadDim, kKvHeads, context, &base))
        continue;
      for (std::size_t b = 0; b < kBlocksPerHead; ++b)
        (void)t4::QuantizeBlockChecked(src.data() + b * t4::kBlockValues,
                                       plane.data() + base +
                                           b * t4::kBlockBytes);
    }
  }
  return plane;
}

void CompareCount(std::uint32_t context, std::uint32_t count) {
  const std::vector<std::uint8_t> k_plane = MakePlane(context, 11);
  const std::vector<std::uint8_t> v_plane = MakePlane(context, 90210);

  // One layer of f16, K and V each max_context * kKvWidth halfs.
  const std::size_t plane_halfs =
      static_cast<std::size_t>(context) * kKvWidth;
  std::vector<std::uint16_t> k_got(plane_halfs, 0xBEEF);
  std::vector<std::uint16_t> v_got(plane_halfs, 0xBEEF);

  const auto st = t4i::Turbo4DequantLayerBody(
      k_plane.data(), v_plane.data(), k_got.data(), v_got.data(), count,
      context, static_cast<std::uint32_t>(kKvHeads),
      static_cast<std::uint32_t>(kHeadDim), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  const std::string tag = "context=" + std::to_string(context) +
                          " count=" + std::to_string(count);
  if (st != t4i::Turbo4KernelStatus::kOk) {
    Check(false, tag + ": dequant body returns kOk");
    return;
  }

  // The reference, walking the scratch with the CONSUMER's expression written
  // out here rather than calling into the body's arithmetic.
  std::size_t k_bad = 0, v_bad = 0;
  float scratch[t4::kBlockValues];
  for (std::uint32_t t = 0; t < count; ++t) {
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      std::size_t base = 0;
      if (!t4i::Turbo4BlockByteOffset(t, h, kHeadDim, kKvHeads, context, &base))
        continue;
      const std::size_t dst = static_cast<std::size_t>(t) * kKvWidth +
                              static_cast<std::size_t>(h) * kHeadDim;
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        for (int plane = 0; plane < 2; ++plane) {
          const std::vector<std::uint8_t>& src = (plane == 0) ? k_plane : v_plane;
          t4::DequantizeBlock(src.data() + base + b * t4::kBlockBytes, scratch);
          t4::RotateInverse(scratch);
          const std::vector<std::uint16_t>& got =
              (plane == 0) ? k_got : v_got;
          for (std::size_t i = 0; i < t4::kBlockValues; ++i) {
            const std::uint16_t want = t4::FloatToHalfBits(scratch[i]);
            if (got[dst + b * t4::kBlockValues + i] != want) {
              if (plane == 0) ++k_bad; else ++v_bad;
            }
          }
        }
      }
    }
  }
  Check(k_bad == 0 && v_bad == 0,
        tag + ": f16 scratch is BIT-EXACT vs the host decode (k=" +
            std::to_string(k_bad) + " v=" + std::to_string(v_bad) + " bad)");

  // Nothing past the requested count may be written.
  std::size_t beyond = 0;
  for (std::size_t i = static_cast<std::size_t>(count) * kKvWidth;
       i < plane_halfs; ++i)
    if (k_got[i] != 0xBEEF || v_got[i] != 0xBEEF) ++beyond;
  Check(beyond == 0, tag + ": positions at or past count are untouched (" +
                         std::to_string(beyond) + " written)");
}

}  // namespace

int main() {
  std::printf("E2(a): Turbo4 -> f16 layer dequant, bit-exact vs host decode\n");
  CompareCount(64, 64);
  CompareCount(64, 17);
  CompareCount(64, 1);
  CompareCount(512, 33);
  CompareCount(512, 512);
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
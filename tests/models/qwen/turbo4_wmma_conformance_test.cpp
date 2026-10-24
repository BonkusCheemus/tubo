// WMMA prompt-attention dtype/layout conformance.
//
// Oracle from the route matrix: "dtype/layout conformance". The question this
// answers is narrow and useful: CAN the deployed Qwen WMMA prompt path consume a
// Turbo4 cache directly, or is a dequantizing staging buffer structurally
// required?
//
// The answer, from source, is that it cannot — and for a concrete arithmetic
// reason rather than a preference:
//
//   attention_wmma.hip:106-110  "A fragment is 16 contiguous halves = 32 bytes
//                               ... Every fragment base here is 16-byte aligned
//                               by construction."
//   attention_wmma.hip:167      static_assert(kWmmaKStride % 8 == 0 &&
//                               kVtStride % 8 == 0,
//                               "fragment rows must start on a 16-byte boundary")
//
// A Turbo4 block is 66 bytes. Successive block offsets therefore step by 66,
// which mod 16 advances by 2, so block starts cycle through every even
// alignment class and only every 8th block is 16-byte aligned. A packed Turbo4
// array cannot serve the kernel's aligned fragment loads, so a staging FP16
// buffer is not an optimization — it is a structural requirement.
//
// The test asserts all of that against the real kernel source, so if gufo's
// fragment contract changes the test fails rather than describing an old file.
//
//   g++ -std=c++20 -I. tests/models/qwen/turbo4_wmma_conformance_test.cpp

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace t4 = gufo::hip::turbo4;
namespace kv = gufo::hip;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("  FAIL  %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("  ok    %s\n", what.c_str());
  }
}

std::string ReadFile(const std::string& p) {
  std::ifstream f(p);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

int main() {
  std::printf("WMMA prompt dtype/layout conformance\n");

  const std::string wmma = ReadFile(
      "src/models/qwen/hip/kernels/attention_wmma.hip");
  Check(!wmma.empty(), "attention_wmma.hip is readable");

  // ---- 1. the deployed kernel's dtype contract ------------------------
  Check(wmma.find("__half* __restrict__ k_cache_f16") != std::string::npos,
        "the prompt path takes an __half K cache pointer");
  Check(wmma.find("__half* __restrict__ v_cache_f16") != std::string::npos,
        "the prompt path takes an __half V cache pointer");
  Check(wmma.find("PackAttentionHeads(const __half* keys, const __half* values") !=
            std::string::npos,
        "PackAttentionHeads is typed in halves end to end");
  Check(wmma.find("reinterpret_cast<const __half*>") != std::string::npos,
        "values are bit-cast to half rather than dequantized");

  // No Turbo4-shaped entry point exists.
  Check(wmma.find("turbo4") == std::string::npos &&
            wmma.find("TURBO4") == std::string::npos,
        "there is no turbo4 entry point in the WMMA prompt kernel");

  // ---- 2. the alignment contract --------------------------------------
  Check(wmma.find("16-byte aligned by construction") != std::string::npos,
        "the kernel documents 16-byte-aligned fragment bases");
  Check(wmma.find("fragment rows must start on a 16-byte boundary") !=
            std::string::npos,
        "the kernel static_asserts the 16-byte fragment-row boundary");

  // ---- 3. Turbo4 block stride violates it -----------------------------
  {
    constexpr std::size_t kBlockBytes = t4::kBlockBytes;  // 66
    int misaligned = 0;
    int classes[16] = {0};
    for (std::size_t k = 0; k < 16; ++k) {
      const std::size_t off = k * kBlockBytes;
      classes[k] = static_cast<int>(off % 16);
      if (off % 16 != 0) ++misaligned;
    }
    Check(misaligned == 14,
          "14 of 16 consecutive turbo4 blocks are NOT 16-byte aligned");
    bool distinct = true;
    for (std::size_t k = 1; k < 8; ++k) {
      if (classes[k] == classes[0]) distinct = false;
    }
    Check(distinct,
          "block alignment cycles through 8 distinct classes, so alignment "
          "must be repaired by staging rather than by an offset");
    Check((kBlockBytes * 8) % 16 == 0,
          "alignment repeats every 8 blocks (66*8 = 528 = 16*33)");
    Check(kBlockBytes % 16 != 0,
          "the 66-byte stride is not a multiple of 16, which is the root cause");
  }

  // ---- 4. Turbo4 bytes are not a conforming half sequence ------------
  {
    // 66 bytes is 33 halves, but only the first is meaningful; the remaining
    // 32 are nibble indices and the block's single shared norm applies to all
    // 128 values, not to each 2-byte pair.
    Check(t4::kBlockBytes / 2 == 33,
          "66 bytes would read as 33 halves, which is not 128 values");
    Check(t4::kBlockBytes / 2 != 64 && t4::kBlockBytes / 2 != 128,
          "the packed block does not correspond to a half value per element");
    Check(t4::kBlockValues == 128 && t4::kBlockBytes == 66,
          "128 values share ONE fp16 norm plus 64 nibble bytes");
  }

  // ---- 5. staging is bounded, not cache-sized -------------------------
  {
    // The staging buffer a prompt tile needs is set by the tile, not by the
    // context. A 16-row x 256-dim half fragment group is 8 KiB per plane.
    constexpr std::size_t kRowDim = 16, kHeadDim = 256, kBytesPerHalf = 2;
    const std::size_t tile_bytes =
        kRowDim * kHeadDim * kBytesPerHalf * 2;  // K and V
    Check(tile_bytes == 16384, "a 16x256 half K+V tile is 16384 B");

    constexpr std::size_t kLayers = 16, kWidth = 1024, kCtx = 4096;
    const std::size_t cache_bytes = kv::QwenKvBytesForElements(
        kv::QwenKvCacheStorage::kTurbo4, kLayers * kCtx * kWidth);
    Check(tile_bytes * 1000 < cache_bytes,
          "staging is bounded by the tile: a tile is <0.1% of the cache");
    Check(cache_bytes % t4::kBlockBytes == 0,
          "the cache is an exact multiple of the turbo4 block");

    // And the reverse: dequantizing a whole cache into halves would be the
    // second full copy the plan forbids. Quantify it.
    const std::size_t whole_cache_as_half = 2 * kv::QwenKvBytesForElements(
        kv::QwenKvCacheStorage::kFp16, kLayers * kCtx * kWidth);
    Check(whole_cache_as_half / cache_bytes > 3,
          "a full-cache half expansion would be >3x the packed cache, which "
          "is exactly the unaccounted copy the plan forbids");
  }

  // ---- 6. dtype conformance summary ----------------------------------
  {
    // The staging dtype must be the one the kernel already accepts, or route 6
    // also needs a kernel change rather than just a feeding change.
    Check(sizeof(float) == 4 && sizeof(std::uint16_t) == 2,
          "half is 2 bytes and float is 4 on this target");
    Check(t4::kBlockBytes ==
              sizeof(std::uint16_t) + t4::kBlockValues / 2,
          "the block is exactly an fp16 norm plus 64 nibble bytes");
    // Consequence, stated as a fact: feeding Turbo4 to this kernel requires a
    // dequantizing staging buffer, so route 6 is a feeding change, not a
    // kernel rewrite, PROVIDED the staging dtype is half.
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
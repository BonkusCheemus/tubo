// VERTICAL SLICE INTEGRATION: the gufo pieces composed in one path.
//
// Every other test in this directory checks ONE piece. This one runs the real
// gufo functions in the order the runtime would, on the UD-Q4_K_XL geometry:
//
//   values -> turbo4 codec -> kv_allocation sizing -> compact snapshot header
//          -> parse back -> policy/graph identity
//
// The property no unit test covers: THE BLOCKS WRITTEN MUST EXACTLY FILL THE
// ALLOCATION THE ALLOCATOR RESERVED. If the allocator's unit count and the
// codec's block count ever disagreed by one, every individual test would still
// pass — allocation would report one total, the codec would write another, and
// the first write would either overrun the buffer or leave a hole at the end.
// Both are silent in isolation and only appear when the parts are run together.
//
//   g++ -std=c++20 -I. tests/models/qwen/turbo4_vertical_slice_test.cpp

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// UD-Q4_K_XL header: block_count 65 with nextn 1, interval 4, 4 kv heads,
// key_length 256. So 16 full-attention layers and 1024 values per plane.
constexpr int kLayers = 16;
constexpr int kWidth = 1024;
constexpr std::size_t kContext = 4096;
constexpr std::size_t kHeaderBytes = 80;

std::size_t Elements() {
  return static_cast<std::size_t>(kLayers) * kContext * kWidth;
}

float Sample(int layer, int head, int token, int j) {
  const double s = 0.37 * j + 0.11 * head + 1.7 * layer + 0.01 * token;
  return static_cast<float>(std::sin(s) * (1.0 + 0.001 * token));
}

void PutU32(std::vector<std::uint8_t>& d, std::size_t o, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) d[o + i] = std::uint8_t(v >> (8 * i));
}
void PutU64(std::vector<std::uint8_t>& d, std::size_t o, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) d[o + i] = std::uint8_t(v >> (8 * i));
}
std::uint32_t GetU32(const std::vector<std::uint8_t>& d, std::size_t o) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= std::uint32_t(d[o + i]) << (8 * i);
  return v;
}
std::uint64_t GetU64(const std::vector<std::uint8_t>& d, std::size_t o) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= std::uint64_t(d[o + i]) << (8 * i);
  return v;
}

}  // namespace

int main() {
  std::printf("vertical slice: composed gufo path\n");

  constexpr auto kStorage = kv::QwenKvCacheStorage::kTurbo4;

  // ---- step 1: size the cache with the allocator ----------------------
  const std::size_t elements = Elements();
  const std::size_t units = kv::QwenKvAllocationUnits(kStorage, elements);
  const std::size_t plane_bytes = kv::QwenKvBytesForElements(kStorage, elements);
  const std::size_t blocks_per_plane = plane_bytes / t4::kBlockBytes;

  Check(units == blocks_per_plane,
        "allocator unit count equals the block count the codec will write");
  Check(units * t4::kBlockBytes == plane_bytes,
        "plane bytes are an exact whole number of codec blocks");
  Check(plane_bytes == 34603008, "plane bytes match the measured value");

  // ---- step 2: reserve it and write every block -----------------------
  std::vector<std::uint8_t> k_plane(plane_bytes, 0xEE);  // poison
  std::vector<std::uint8_t> v_plane(plane_bytes, 0xEE);
  std::vector<float> block(t4::kBlockValues);

  std::size_t written_k = 0, written_v = 0;
  const int kBlocksPerToken = kWidth / static_cast<int>(t4::kBlockValues);
  for (int layer = 0; layer < kLayers; ++layer) {
    for (int token = 0; token < static_cast<int>(kContext); ++token) {
      for (int h = 0; h < kBlocksPerToken; ++h) {
        for (int j = 0; j < static_cast<int>(t4::kBlockValues); ++j) {
          block[j] = Sample(layer, h, token, j);
        }
        const std::size_t idx = (written_k++) * t4::kBlockBytes;
        t4::QuantizeBlock(block.data(), k_plane.data() + idx);
        // V takes the negated values: K and V are different tensors, and a
        // shared buffer reused without reseeding would hide a transposition.
        for (int j2 = 0; j2 < static_cast<int>(t4::kBlockValues); ++j2) {
          block[j2] = -Sample(layer, h, token, j2);
        }
        t4::QuantizeBlock(block.data(), v_plane.data() + written_v * t4::kBlockBytes);
        ++written_v;
      }
    }
  }

  Check(written_k == units && written_v == units,
        "the number of blocks WRITTEN equals the number ALLOCATED, exactly");
  Check(written_k * t4::kBlockBytes == k_plane.size(),
        "the write consumed the reservation precisely: no overrun, no hole");

  // Prove the END of the buffer was actually written. A short fill would
  // leave a hole at the tail, which is where an off-by-one in the block count
  // would show up.
  //
  // NOTE: an earlier version of this check poisoned the buffer with 0xEE and
  // asserted no 0xEE survived. That was WRONG: 0xEE is a perfectly legal
  // quantized byte, so the scan could fail on correct data and pass on a
  // short fill that happened to contain no 0xEE. Content at the last block is
  // the sound check; a poison VALUE is not.
  {
    const int last_layer = kLayers - 1;
    const int last_token = static_cast<int>(kContext) - 1;
    const int last_h = kBlocksPerToken - 1;
    std::vector<float> expect(t4::kBlockValues);
    for (int j = 0; j < static_cast<int>(t4::kBlockValues); ++j) {
      expect[j] = Sample(last_layer, last_h, last_token, j);
    }
    std::uint8_t standalone[t4::kBlockBytes];
    t4::QuantizeBlock(expect.data(), standalone);
    Check(std::memcmp(k_plane.data() + (units - 1) * t4::kBlockBytes,
                      standalone, t4::kBlockBytes) == 0,
          "the LAST block of the plane holds exactly the values its index "
          "implies, so the tail was written and not just counted");
  }

  // ---- step 3: the written data is decodable and norm-preserving ------
  {
    std::vector<float> back(t4::kBlockValues);
    t4::DequantizeBlock(k_plane.data(), back.data());
    float in_l2 = 0.0f, out_l2 = 0.0f;
    for (int j = 0; j < static_cast<int>(t4::kBlockValues); ++j) {
      const float x = Sample(0, 0, 0, j);
      in_l2 += x * x;
      out_l2 += back[j] * back[j];
    }
    const float ratio = std::sqrt(out_l2 / in_l2);
    Check(std::fabs(ratio - 1.0f) < 5e-3f,
          "the first written block round-trips with its L2 norm preserved "
          "(ratio " + std::to_string(ratio) + ")");
  }

  // ---- step 4: build the compact snapshot and parse it back ----------
  const std::size_t total = kHeaderBytes + 2 * plane_bytes;
  std::vector<std::uint8_t> snap(total, 0);
  const char magic[8] = {'G', 'Q', 'K', 'V', 'S', 'N', 'P', '1'};
  for (int i = 0; i < 8; ++i) snap[i] = std::uint8_t(magic[i]);
  PutU32(snap, 8, 1);
  PutU32(snap, 12, std::uint32_t(kHeaderBytes));
  PutU32(snap, 16, std::uint32_t(kLayers));
  PutU32(snap, 20, std::uint32_t(kWidth));
  PutU32(snap, 24, std::uint32_t(kContext));
  PutU32(snap, 28, std::uint32_t(kContext));
  PutU32(snap, 32, std::uint32_t(kStorage));
  PutU32(snap, 36, std::uint32_t(kv::QwenRecurrentStateStorage::kFp32));
  PutU64(snap, 40, std::uint64_t(elements));
  PutU64(snap, 48, 0);
  PutU64(snap, 56, 0);
  PutU64(snap, 64, std::uint64_t(total));
  PutU32(snap, 72, 0);
  PutU32(snap, 76, 0);
  // The payload the header describes must be exactly what we wrote.
  std::memcpy(snap.data() + kHeaderBytes, k_plane.data(), plane_bytes);
  std::memcpy(snap.data() + kHeaderBytes + plane_bytes, v_plane.data(),
              plane_bytes);

  Check(GetU64(snap, 64) == total, "header total matches the real payload");
  Check(snap.size() == kHeaderBytes + 2 * k_plane.size(),
        "snapshot is header + exactly the two planes we wrote");

  // Parse back the way arena.cpp does: recompute bytes from storage+elements.
  {
    const auto parsed_storage =
        static_cast<kv::QwenKvCacheStorage>(GetU32(snap, 32));
    const std::size_t parsed_elements = static_cast<std::size_t>(GetU64(snap, 40));
    const std::size_t parsed_plane =
        kv::QwenKvBytesForElements(parsed_storage, parsed_elements);
    const std::size_t parsed_total = kHeaderBytes + 2 * parsed_plane;
    Check(parsed_storage == kStorage, "parsed storage round-trips");
    Check(parsed_elements == elements, "parsed element count round-trips");
    Check(parsed_total == total,
          "recomputing the byte count from (storage, elements) reproduces the "
          "payload exactly");
    Check(std::memcmp(snap.data() + kHeaderBytes, k_plane.data(), plane_bytes) == 0,
          "the K plane survives serialize/parse byte for byte");
  }

  // ---- step 5: route identity for this composed path ------------------
  {
    kv::QwenExecutionPolicy policy;
    policy.kv_cache_storage = kStorage;
    const std::uint64_t fp = policy.Fingerprint();
    kv::QwenExecutionPolicy fp16 = kv::QwenExecutionPolicy::Production();
    Check(fp != fp16.Fingerprint(),
          "the turbo4 route has a distinct graph identity from production fp16");
    Check(fp16.Fingerprint() == 0x101ULL,
          "production's identity is unchanged at 0x101");
    // And the composed route is stable across identical construction.
    kv::QwenExecutionPolicy again;
    again.kv_cache_storage = kStorage;
    Check(again.Fingerprint() == fp, "route identity is deterministic");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
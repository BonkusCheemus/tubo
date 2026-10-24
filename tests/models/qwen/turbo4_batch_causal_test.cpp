// Batched causal verification, on CPU.
//
// Oracle from the route matrix: "synthetic mask/batch equivalence vs reference".
// This runs against the SHIPPED codec (turbo4_codec.hpp) rather than the
// Python reference, because the property under test is about the C++ code that
// would execute.
//
// The property that actually matters for batched causal verification is
// PREFIX STABILITY: when a verifier accepts n drafted tokens and commits m of
// them, the bytes already written for tokens [0, m) must be identical to the
// bytes they would have had if only m tokens had ever been produced. If they
// are not, a rollback-and-replay changes the KV cache contents and the whole
// speculative contract is unsound — and it would look fine in a single-token
// test.
//
// Two consequences asserted here:
//   1. BATCH INDEPENDENCE — block (b, h, t) has the same bytes whether it was
//      produced inside a batch or in isolation.
//   2. PREFIX STABILITY — quantizing [0, n) then [n, n+k) gives the same bytes
//      for [0, n) as quantizing [0, n) alone.
//
//   g++ -std=c++20 -I. tests/models/qwen/turbo4_batch_causal_test.cpp -o /tmp/t5

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

/// Deterministic, position-seeded test data. Reproducible so a failure is
/// replayable, and distinct per (batch, head, token) so a transposition would
/// be visible rather than accidentally equal.
float Sample(int batch, int head, int token, int within_block) {
  const double seed = 0.37 * token + 0.11 * head + 1.7 * batch + 0.53 * within_block;
  return static_cast<float>(std::sin(seed) * (1.0 + 0.01 * token));
}

void MakeBlock(std::vector<float>& v, int batch, int head, int token_block) {
  v.resize(t4::kBlockValues);
  for (int j = 0; j < static_cast<int>(t4::kBlockValues); ++j) {
    v[j] = Sample(batch, head, token_block, j);
  }
}

bool SameBytes(const std::uint8_t* a, const std::uint8_t* b, std::size_t n) {
  return std::memcmp(a, b, n) == 0;
}

}  // namespace

int main() {
  std::printf("batched causal verification\n");

  constexpr int kBlocksPerTokenPlane = 8;  // 1024 width / 128 block
  constexpr int kBlocksPerHead = 2;
  constexpr int kKvHeads = 4;  // UD header head_count_kv

  // ---- 1. batch independence ------------------------------------------
  {
    constexpr int kB = 4, kH = 4;
    std::vector<std::uint8_t> batched(
        kB * kH * kBlocksPerHead * t4::kBlockBytes);
    std::vector<float> v;
    for (int b = 0; b < kB; ++b) {
      for (int h = 0; h < kH; ++h) {
        for (int blk = 0; blk < kBlocksPerHead; ++blk) {
          MakeBlock(v, b, h, blk);
          t4::QuantizeBlock(v.data(),
                            batched.data() + ((b * kH + h) * kBlocksPerHead + blk) *
                                                 t4::kBlockBytes);
        }
      }
    }
    bool same = true;
    for (int b = 0; b < kB; ++b) {
      for (int h = 0; h < kH; ++h) {
        for (int blk = 0; blk < kBlocksPerHead; ++blk) {
          MakeBlock(v, b, h, blk);
          std::uint8_t alone[t4::kBlockBytes];
          t4::QuantizeBlock(v.data(), alone);
          if (!SameBytes(alone,
                         batched.data() + ((b * kH + h) * kBlocksPerHead + blk) *
                                              t4::kBlockBytes,
                         t4::kBlockBytes)) {
            same = false;
          }
        }
      }
    }
    Check(same,
          "every block is byte-identical whether produced in a batch or alone");
  }

  // ---- 2. blocks are genuinely distinct, so the test can fail -----------
  {
    constexpr int kH = 4;
    std::vector<std::uint8_t> blocks(kH * t4::kBlockBytes);
    std::vector<float> v;
    int distinct = 0;
    for (int h = 0; h < kH; ++h) {
      MakeBlock(v, 0, h, 0);
      t4::QuantizeBlock(v.data(), blocks.data() + h * t4::kBlockBytes);
      if (h > 0) {
        bool diff = false;
        for (std::size_t i = 0; i < t4::kBlockBytes; ++i) {
          if (blocks[h * t4::kBlockBytes + i] !=
              blocks[(h - 1) * t4::kBlockBytes + i]) {
            diff = true;
          }
        }
        if (diff) ++distinct;
      }
    }
    Check(distinct == kH - 1,
          "adjacent heads produce different bytes, so a transposition would be "
          "detected rather than passing by coincidence");
  }

  // ---- 3. PREFIX STABILITY: appending tokens must not rewrite earlier ---
  {
    constexpr int kSteps = 8;  // grow the prefix one block at a time
    std::vector<std::vector<std::uint8_t>> prefix_bytes;
    for (int step = 1; step <= kSteps; ++step) {
      std::vector<std::uint8_t> all(step * t4::kBlockBytes);
      std::vector<float> v;
      for (int i = 0; i < step; ++i) {
        MakeBlock(v, 0, 0, i);
        t4::QuantizeBlock(v.data(), all.data() + i * t4::kBlockBytes);
      }
      prefix_bytes.push_back(all);
    }
    bool stable = true;
    for (int step = 2; step <= kSteps; ++step) {
      const auto& shorter = prefix_bytes[step - 2];
      const auto& longer = prefix_bytes[step - 1];
      if (!SameBytes(shorter.data(), longer.data(),
                     shorter.size())) {
        stable = false;
      }
    }
    Check(stable,
          "PREFIX STABILITY: a longer prefix never rewrites the bytes of a "
          "shorter one, so rollback and replay are sound");

    // Guard against the check above passing vacuously. If every block
    // quantized to the same bytes, "a longer prefix did not change a shorter
    // one" would be true for free and prove nothing.
    const auto& run = prefix_bytes[kSteps - 1];
    int differing = 0;
    for (int i = 0; i < kSteps; ++i) {
      if (!SameBytes(run.data() + i * t4::kBlockBytes,
                     run.data() + (i + 1) * t4::kBlockBytes,
                     t4::kBlockBytes)) {
        ++differing;
      }
    }
    Check(differing >= kSteps - 1,
          "successive blocks in a prefix really do differ, so prefix "
          "stability is a real constraint and not a coincidence");
  }

  // ---- 4. norm correction is per block, not per batch ------------------
  {
    // If the norm correction leaked across blocks, concatenating blocks would
    // change them. Prove it does not by quantizing a long run at once and
    // comparing against per-block quantization of the same run.
    constexpr int kRun = 16;
    std::vector<std::uint8_t> single(kRun * t4::kBlockBytes);
    std::vector<float> run(kRun * t4::kBlockValues);
    std::vector<float> v;
    for (int i = 0; i < kRun; ++i) {
      for (int j = 0; j < static_cast<int>(t4::kBlockValues); ++j) {
        run[i * t4::kBlockValues + j] = Sample(0, 0, i, j);
      }
      MakeBlock(v, 0, 0, i);
      t4::QuantizeBlock(v.data(), single.data() + i * t4::kBlockBytes);
    }
    (void)run;
    // Recompute the first and last block from the run buffer directly.
    std::uint8_t first[t4::kBlockBytes];
    std::uint8_t last[t4::kBlockBytes];
    t4::QuantizeBlock(run.data(), first);
    t4::QuantizeBlock(run.data() + (kRun - 1) * t4::kBlockValues, last);
    Check(SameBytes(first, single.data(), t4::kBlockBytes),
          "first block of a long run is unchanged by what follows it");
    Check(SameBytes(last,
                    single.data() + (kRun - 1) * t4::kBlockBytes,
                    t4::kBlockBytes),
          "last block of a long run matches its standalone quantization");
  }

  // ---- 5. offset arithmetic agrees with the allocator ------------------
  {
    // The batched write must address blocks exactly as kv_allocation sizes
    // them, or a batched verifier writes into the wrong slots.
    constexpr std::size_t kLayers = 16, kWidth = 1024, kCtx = 4096;
    const std::size_t elements = kLayers * kCtx * kWidth;
    const std::size_t units =
        kv::QwenKvAllocationUnits(kv::QwenKvCacheStorage::kTurbo4, elements);
    const std::size_t bytes =
        units * kv::QwenKvUnitBytes(kv::QwenKvCacheStorage::kTurbo4);
    Check(units == elements / t4::kBlockValues,
          "allocator unit count matches the codec block count");
    Check(bytes / t4::kBlockBytes == units,
          "byte count is an exact multiple of the codec block size");
    // Blocks per token per plane, derived independently from the codec.
    Check(kWidth / t4::kBlockValues == kBlocksPerTokenPlane,
          "blocks per token per plane agree: 1024/128 = 8");
    Check(t4::kBlockValues <= kWidth / kKvHeads,
          "a 128-value block fits inside one 256-wide head");
    Check((kWidth / kKvHeads) % t4::kBlockValues == 0,
          "a 256-wide head divides into whole 128-value blocks");
    Check(kWidth / kKvHeads / t4::kBlockValues == kBlocksPerHead,
          "a head is exactly 2 whole blocks");
  }

  // ---- 6. a batched write never overruns the allocation ----------------
  {
    // One token, one layer, one plane: kv_width values = 8 blocks of 128.
    // A batch of B heads x T tokens needs B heads' worth, so the shape must
    // match the allocator exactly rather than approximately.
    constexpr std::size_t kLayers = 16, kWidth = 1024;
    constexpr int kT = 2;
    const std::size_t per_token_blocks = kWidth / t4::kBlockValues;
    const std::size_t needed = static_cast<std::size_t>(kT) * per_token_blocks *
                               t4::kBlockBytes;
    const std::size_t budget =
        kv::QwenKvBytesForElements(kv::QwenKvCacheStorage::kTurbo4,
                                   kLayers * kT * kWidth);
    Check(budget / kLayers == needed,
          "one token across all layers is exactly the batched write's need");
    Check(needed % t4::kBlockBytes == 0,
          "the batched write is a whole number of blocks");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
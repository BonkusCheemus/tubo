// CPU test for the Gufo-native Turbo4 KV path.
//
// Compiles and runs with plain g++ (no HIP, no GPU): both headers it includes
// are constexpr-only and HIP-free, which is why the accounting could be tested
// before any build of the real engine was permitted.
//
//   g++ -std=c++20 -Isrc tests/models/qwen/kv_allocation_turbo4_test.cpp -o /tmp/t
//
// Geometry is the UD-Q4_K_XL header read earlier: block_count 65 with
// nextn_predict_layers 1, full_attention_interval 4, head_count_kv 4,
// key_length 256 -> 16 full-attention layers and kv_width 1024 per plane.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"

namespace {

using gufo::hip::CeilDiv;
using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenKvAllocationUnits;
using gufo::hip::QwenKvBytesForElements;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvHasElementBytes;
using gufo::hip::QwenKvHeadDimIsBlockAligned;
using gufo::hip::QwenKvHeadsAreBlockAligned;
using gufo::hip::QwenKvElementBytes;
using gufo::hip::QwenKvUnitBytes;
using gufo::hip::QwenKvElementBytes;
using gufo::hip::QwenKvIsBlockAddressed;
using gufo::hip::QwenKvIsExactlyBlockAligned;
using gufo::hip::QwenKvTailElements;
using gufo::hip::QwenKvUnitBytes;
using gufo::hip::kQwenTurbo4BitsPerValue;
using gufo::hip::kQwenTurbo4BlockBytes;
using gufo::hip::kQwenTurbo4BlockValues;

constexpr std::size_t kLayers = 16;   // 64 base layers / interval 4
constexpr std::size_t kKvWidth = 1024; // 4 kv heads x 256
constexpr std::size_t kContext = 4096;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::printf("  FAIL  %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("  ok    %s\n", what.c_str());
  }
}

std::size_t PlaneElements(std::size_t context) {
  return kLayers * context * kKvWidth;
}

}  // namespace

int main() {
  std::printf("Turbo4 KV slice\n");

  // ---- enumerator -------------------------------------------------------
  Check(static_cast<std::uint8_t>(QwenKvCacheStorage::kFp32) == 0,
        "kFp32 stays 0 so persisted snapshots keep their meaning");
  Check(static_cast<std::uint8_t>(QwenKvCacheStorage::kFp16) == 1,
        "kFp16 stays 1");
  Check(static_cast<std::uint8_t>(QwenKvCacheStorage::kTurbo4) == 2,
        "kTurbo4 appended, not inserted");

  // ---- block constants mirror the donor ---------------------------------
  Check(kQwenTurbo4BlockValues == 128, "block is 128 values");
  Check(kQwenTurbo4BlockBytes == 66, "block is 66 bytes");
  Check(kQwenTurbo4BitsPerValue > 4.1249 && kQwenTurbo4BitsPerValue < 4.1251,
        "4.125 bits per value");

  // ---- addressing mode --------------------------------------------------
  Check(!QwenKvIsBlockAddressed(QwenKvCacheStorage::kFp16),
        "fp16 is element addressed");
  Check(!QwenKvIsBlockAddressed(QwenKvCacheStorage::kFp32),
        "fp32 is element addressed");
  Check(QwenKvIsBlockAddressed(QwenKvCacheStorage::kTurbo4),
        "turbo4 is block addressed");

  // ---- fp16 path unchanged ---------------------------------------------
  const auto elements = PlaneElements(kContext);
  const std::size_t fp16_bytes =
      QwenKvBytesForElements(QwenKvCacheStorage::kFp16, elements);
  Check(fp16_bytes == elements * 2,
        "fp16 plane still elements x 2 (134217728)");
  Check(fp16_bytes == 134217728ULL, "fp16 plane is the historical value");
  const std::size_t fp32_bytes =
      QwenKvBytesForElements(QwenKvCacheStorage::kFp32, elements);
  Check(fp32_bytes == elements * 4, "fp32 plane still elements x 4");

  // ---- turbo4 path is block granular ------------------------------------
  const std::size_t t4_bytes =
      QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4, elements);
  Check(QwenKvAllocationUnits(QwenKvCacheStorage::kTurbo4, elements) ==
            elements / kQwenTurbo4BlockValues,
        "aligned count needs elements/128 whole blocks");
  Check(t4_bytes == (elements / kQwenTurbo4BlockValues) * 66,
        "turbo4 plane is blocks x 66 (34603008)");
  Check(t4_bytes == 34603008ULL, "turbo4 plane measured value");
  Check(t4_bytes < fp16_bytes, "turbo4 is smaller than fp16");
  Check(t4_bytes % 66 == 0, "turbo4 plane is a whole number of blocks");

  // ---- the truncation trap ---------------------------------------------
  {
    const std::size_t odd = 100;  // not a multiple of 128
    const std::size_t correct =
        QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4, odd);
    const std::size_t naive = static_cast<std::size_t>(
        static_cast<double>(odd) * kQwenTurbo4BitsPerValue / 8.0);
    Check(correct == 66, "100 elements still allocate one whole block");
    Check(naive == 51, "the naive elements x 4.125/8 truncates to 51");
    Check(naive < correct,
          "naive under-allocates by 15 bytes on a tail");
    Check(QwenKvTailElements(QwenKvCacheStorage::kTurbo4, odd) == 28,
          "tail is 28 padded values");
    Check(!QwenKvIsExactlyBlockAligned(QwenKvCacheStorage::kTurbo4, odd),
          "100 elements is not exactly block aligned");
  }
  Check(QwenKvTailElements(QwenKvCacheStorage::kFp16, elements) == 0,
        "element storage reports no tail");

  // ---- the UD geometry is aligned at every context ----------------------
  bool all_aligned = true;
  for (std::size_t ctx : {1UL, 2UL, 127UL, 4096UL, 32768UL, 262144UL}) {
    if (!QwenKvIsExactlyBlockAligned(QwenKvCacheStorage::kTurbo4,
                                     PlaneElements(ctx))) {
      all_aligned = false;
    }
  }
  Check(all_aligned,
        "16384 elements/token/plane is 128 whole blocks at every context");

  // ---- per-head nesting: no block crosses a head ------------------------
  {
    const std::size_t head_dim = 256;
    const std::size_t blocks = elements / (kContext * kKvWidth);
    bool nested = (head_dim % kQwenTurbo4BlockValues) == 0;
    for (std::size_t j = 0; j < blocks; ++j) {
      const std::size_t start = j * kQwenTurbo4BlockValues;
      const std::size_t head = start / head_dim;
      const std::size_t offset = start % head_dim;
      if (offset + kQwenTurbo4BlockValues > head_dim) {
        nested = false;
      }
      (void)head;
    }
    Check(nested, "no 128-value block straddles a 256-wide head");
  }

  // ---- unit bytes -------------------------------------------------------
  Check(QwenKvUnitBytes(QwenKvCacheStorage::kFp16) == 2, "fp16 unit is 2 B");
  Check(QwenKvUnitBytes(QwenKvCacheStorage::kTurbo4) == 66,
        "turbo4 unit is 66 B, not 0.515625");
  // Review: this line used to assert `== 4` while its own message
  // said the number was meaningless. The test enshrined the exact defect the
  // review named — a fabricated element size for a block-addressed format —
  // and the description contradicted the assertion beside it.
  Check(QwenKvElementBytes(QwenKvCacheStorage::kTurbo4) == 0,
        "element bytes refuse for turbo4 rather than fabricating a size");

  // ---- fingerprint: one route per executor -----------------------------
  {
    std::vector<std::uint64_t> prints;
    for (auto kv : {QwenKvCacheStorage::kFp32, QwenKvCacheStorage::kFp16,
                    QwenKvCacheStorage::kTurbo4}) {
      for (bool fuse : {false, true}) {
        for (auto rs : {gufo::hip::QwenRecurrentStateStorage::kFp32,
                        gufo::hip::QwenRecurrentStateStorage::kBf16}) {
          QwenExecutionPolicy p;
          p.fuse_qk_norm_rope_kv = fuse;
          p.kv_cache_storage = kv;
          p.recurrent_state_storage = rs;
          prints.push_back(p.Fingerprint());
        }
      }
    }
    std::vector<std::uint64_t> sorted = prints;
    std::sort(sorted.begin(), sorted.end());
    Check(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
          "all 12 policies have distinct fingerprints");
  }
  {
    QwenExecutionPolicy prod = QwenExecutionPolicy::Production();
    Check(prod.kv_cache_storage == QwenKvCacheStorage::kFp16,
          "production default is still fp16");
    Check(prod.Fingerprint() == 0x101ULL,
          "production fingerprint unchanged at 0x101: existing graph-cache "
          "entries stay reachable");

    QwenExecutionPolicy fp32;
    fp32.kv_cache_storage = QwenKvCacheStorage::kFp32;
    QwenExecutionPolicy t4;
    t4.kv_cache_storage = QwenKvCacheStorage::kTurbo4;
    Check(fp32.Fingerprint() != t4.Fingerprint(),
          "fp32 and turbo4 no longer collide");
    Check(fp32.Fingerprint() == 0x001ULL, "fp32 fingerprint unchanged");
    Check(t4.Fingerprint() == 0x10001ULL, "turbo4 fingerprint is 0x10001");
  }
  {
    // The bit-9 overlap that a whole-byte placement would have introduced.
    QwenExecutionPolicy t4_bf16;
    t4_bf16.kv_cache_storage = QwenKvCacheStorage::kTurbo4;
    t4_bf16.recurrent_state_storage =
        gufo::hip::QwenRecurrentStateStorage::kBf16;
    QwenExecutionPolicy fp32_bf16;
    fp32_bf16.kv_cache_storage = QwenKvCacheStorage::kFp32;
    fp32_bf16.recurrent_state_storage =
        gufo::hip::QwenRecurrentStateStorage::kBf16;
    Check(t4_bf16.Fingerprint() != fp32_bf16.Fingerprint(),
          "turbo4+bf16 state is distinguishable from fp32+bf16 state");
    Check((t4_bf16.Fingerprint() & (1ULL << 9)) != 0,
          "bit 9 still carries only the recurrent-state flag");
  }

  // ---- CeilDiv ----------------------------------------------------------
  Check(CeilDiv(0, 128) == 0, "CeilDiv(0,128) is 0");
  Check(CeilDiv(1, 128) == 1, "CeilDiv(1,128) is 1");
  Check(CeilDiv(128, 128) == 1, "CeilDiv(128,128) is 1");
  Check(CeilDiv(129, 128) == 2, "CeilDiv(129,128) is 2");

  // ---- CeilDiv does not wrap ---------------------------------------------
  // The old `(n + d - 1) / d` wraps to 0 here: SIZE_MAX + 128 - 1 == 126.
  // A wrapped ceiling returns a SMALLER number than the true answer, so the
  // bug hides behind an under-allocation that looks like a valid size.
  Check(CeilDiv(static_cast<std::size_t>(-1), 128) ==
            static_cast<std::size_t>(-1) / 128 + 1,
        "CeilDiv at SIZE_MAX does not wrap (old form returned 0)");
  Check(CeilDiv(static_cast<std::size_t>(-1), 128) != 0,
        "CeilDiv at SIZE_MAX is not silently 0");
  Check(CeilDiv(static_cast<std::size_t>(-1) - 100U, 128) ==
            (static_cast<std::size_t>(-1) - 100U) / 128 + 1,
        "CeilDiv just below SIZE_MAX does not wrap");
  Check(CeilDiv(5, 0) == 0, "CeilDiv by zero refuses instead of trapping");

  // ---- QwenKvElementBytes REFUSES for Turbo4 -----------------------------
  // This is the specific defect called out in review: the old ternary fell
  // through to sizeof(float) and answered 4 for Turbo4.
  Check(!QwenKvHasElementBytes(QwenKvCacheStorage::kTurbo4),
        "Turbo4 declares that it has no bytes-per-element");
  Check(QwenKvElementBytes(QwenKvCacheStorage::kTurbo4) == 0,
        "QwenKvElementBytes(Turbo4) refuses with 0, not a fabricated 4");
  Check(QwenKvElementBytes(QwenKvCacheStorage::kTurbo4) != 4,
        "the refused answer is specifically not the old wrong 4");
  Check(QwenKvHasElementBytes(QwenKvCacheStorage::kFp16),
        "fp16 does have bytes-per-element");
  Check(QwenKvElementBytes(QwenKvCacheStorage::kFp16) == 2,
        "fp16 element size unchanged at 2");
  Check(QwenKvHasElementBytes(QwenKvCacheStorage::kFp32),
        "fp32 does have bytes-per-element");
  Check(QwenKvElementBytes(QwenKvCacheStorage::kFp32) == 4,
        "fp32 element size unchanged at 4");
  {
    // The block size must still be reachable through the unit-size API, so
    // refusing the element view did not lose the real number.
    Check(QwenKvUnitBytes(QwenKvCacheStorage::kTurbo4) == 66,
          "Turbo4 block size is still 66 bytes via QwenKvUnitBytes");
  }

  // ---- Checked multiply ---------------------------------------------------
  Check(gufo::hip::QwenKvMultiplyOverflows(static_cast<std::size_t>(-1), 2),
        "SIZE_MAX x 2 is detected as overflow");
  Check(!gufo::hip::QwenKvMultiplyOverflows(66, 524288),
        "the real UD geometry multiply is not overflow");
  Check(!gufo::hip::QwenKvMultiplyOverflows(0, static_cast<std::size_t>(-1)),
        "0 x SIZE_MAX is not overflow");
  {
    std::size_t out = 12345U;
    Check(!gufo::hip::QwenKvMulChecked(static_cast<std::size_t>(-1), 2, &out),
          "QwenKvMulChecked reports overflow");
    Check(out == 12345U, "QwenKvMulChecked left the output untouched");
    Check(gufo::hip::QwenKvMulChecked(66, 524288, &out) && out == 34603008U,
          "QwenKvMulChecked computes the real 34,603,008 B plane");
  }

  // ---- Per-head 128 alignment --------------------------------------------
  // UD-Q4 geometry: kv_width 1024 per plane, head_count_kv 4 -> 256 per head.
  Check(QwenKvHeadDimIsBlockAligned(128), "128 values per head is aligned");
  Check(QwenKvHeadDimIsBlockAligned(256), "256 values per head is aligned");
  Check(!QwenKvHeadDimIsBlockAligned(127),
        "127 values per head straddles a block and is refused");
  Check(!QwenKvHeadDimIsBlockAligned(64),
        "64 values per head is a half block and is refused");
  Check(!QwenKvHeadDimIsBlockAligned(0),
        "a zero-width head is refused, not called aligned");
  // The UD TARGET: head_count_kv 4 x key_length 256 = kv_width 1024, so each
  // head spans TWO 128-value blocks. Not 8 x 128.
  Check(QwenKvHeadsAreBlockAligned(256, 4),
        "UD TARGET: 4 heads of 256 (two blocks each) passes");
  Check(QwenKvHeadsAreBlockAligned(128, 4),
        "128-wide x 4 is aligned too, it just is not THIS model's geometry");
  Check(QwenKvHeadsAreBlockAligned(128, 8),
        "128-wide x 8 heads is aligned, but is not this model's geometry");
  Check(!QwenKvHeadsAreBlockAligned(100, 4),
        "100-wide heads are refused even though 400 is not a multiple of 128");
  {
    // THE POINT OF THE CHECK: a flat element count that is a whole number of
    // blocks can still hide heads that straddle boundaries. 256 elements as a
    // single 256-wide head is fine; the same 256 values split as two 128-wide
    // heads is also fine; but a plane whose heads are 64 wide puts a block
    // boundary inside a head and must not be accepted on the flat count alone.
    Check(QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4, 256) == 132,
          "256 Turbo4 values still allocate 2 blocks = 132 B");
    Check(!QwenKvHeadsAreBlockAligned(64, 4),
          "4 heads of 64 is refused even though 256 values = 2 whole blocks");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
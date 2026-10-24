#pragma once
// NPU B-cache key for tagged IQ3 prefill weights. Host and device.
// Tagged gate/up are unpacked per layer into ONE arena scratch, so the scratch pointer
// names no tensor. The cache keys such a weight by its logical source instead: the
// tagged pair root (one per layer), the pair row of the packed slice (gate rows start
// at 0, up rows at full_rows / 2) and the layout. Its fingerprint is FNV-1a over the
// first 256 native bytes of that row, read from the root through Iq3ByteOffset: the
// same bytes the unpack writes and a native model's fingerprint reads, so a record's
// fingerprint does not depend on which form the weights were loaded in.
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "src/models/qwen/iq3_layout.hpp"

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define GUFO_NPU_KEY_HD __host__ __device__
#else
#define GUFO_NPU_KEY_HD
#endif

namespace gufo::hip::npu {

// Native rows at an aliased scratch pointer = rows [row_begin, ...) of this tagged pair.
struct LogicalWeight {
  const void* root = nullptr;  // tagged pair tensor
  models::QwenWeightLayout layout = models::QwenWeightLayout::kNative;
  std::size_t full_rows = 0, columns = 0, row_begin = 0;
};

// What a packed slice is: (tensor, first row, layout). Native: (w, row0, kNative).
struct WeightKey {
  const void* base;
  std::uint64_t row;
  std::uint8_t layout;
  bool operator==(const WeightKey&) const = default;
  bool operator<(const WeightKey& o) const {
    return std::tie(base, row, layout) < std::tie(o.base, o.row, o.layout);
  }
};

inline WeightKey KeyOf(const void* w, std::uint32_t row0, const LogicalWeight* src) {
  if (src == nullptr) return {w, row0, std::uint8_t(models::QwenWeightLayout::kNative)};
  return {src->root, src->row_begin + row0, std::uint8_t(src->layout)};
}

// FNV-1a of native bytes [0, 256) of logical row `row0` of `src`; byte(off) reads root byte off.
// Needs >= 256 native bytes per row (columns / 256 * block bytes; 5120 -> 2200 IQ3_S, 1960 XXS).
template <class ByteAt>
GUFO_NPU_KEY_HD std::uint64_t LogicalRowFnv(const LogicalWeight& src, std::size_t row0, ByteAt byte) {
  const std::size_t bb = models::Iq3BlockBytes(src.layout), blocks = src.columns / 256;
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < 256; ++i)
    h = (h ^ byte(models::Iq3ByteOffset(src.layout, src.full_rows, blocks, src.row_begin + row0,
                                        i / bb, i % bb))) * 1099511628211ULL;
  return h;
}

}  // namespace gufo::hip::npu
#undef GUFO_NPU_KEY_HD

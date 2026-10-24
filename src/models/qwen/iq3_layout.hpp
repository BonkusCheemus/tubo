#ifndef GUFO_MODELS_QWEN_IQ3_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_IQ3_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "src/models/qwen/rows32_layout.hpp"

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define GUFO_IQ3_HD __host__ __device__
#else
#define GUFO_IQ3_HD
#endif

namespace gufo::models {

GUFO_IQ3_HD constexpr bool IsIq3Layout(QwenWeightLayout layout) noexcept {
  return layout == QwenWeightLayout::kIq3SRows8 ||
         layout == QwenWeightLayout::kIq3XxsRows4;
}
GUFO_IQ3_HD constexpr std::size_t Iq3BlockBytes(
    QwenWeightLayout layout) noexcept {
  return layout == QwenWeightLayout::kIq3SRows8     ? 110
         : layout == QwenWeightLayout::kIq3XxsRows4 ? 98
                                                    : 0;
}
GUFO_IQ3_HD constexpr std::size_t Iq3GroupRows(
    QwenWeightLayout layout) noexcept {
  return layout == QwenWeightLayout::kIq3SRows8     ? 8
         : layout == QwenWeightLayout::kIq3XxsRows4 ? 4
                                                    : 0;
}

// Only full groups: no padding, no size growth. Also bound logical elements.
constexpr bool Iq3ShapeValid(QwenWeightLayout layout, std::size_t rows,
                             std::size_t columns) noexcept {
  return IsIq3Layout(layout) && rows && columns && columns % 256 == 0 &&
         rows % Iq3GroupRows(layout) == 0 &&
         rows <= std::numeric_limits<std::size_t>::max() / columns;
}
constexpr std::size_t Iq3PackedBytes(QwenWeightLayout layout, std::size_t rows,
                                     std::size_t columns) noexcept {
  return Iq3ShapeValid(layout, rows, columns)
             ? rows * (columns / 256) * Iq3BlockBytes(layout)
             : 0;
}

// Native block byte -> physical packed byte. Preconditions: validated full
// shape and row/block/byte bounds. full_rows is the PAIR's physical height,
// not a logical G/U height. Pure arithmetic, shared with a future GPU unpack.
GUFO_IQ3_HD constexpr std::size_t Iq3ByteOffset(
    QwenWeightLayout layout, std::size_t full_rows, std::size_t blocks,
    std::size_t row, std::size_t block, std::size_t byte) noexcept {
  if (layout == QwenWeightLayout::kIq3SRows8) {
    const auto r = row % 8;
    const auto record = (row / 8 * blocks + block) * 880;
    if (byte < 2)
      return record + 864 + r * 2 + byte;
    if (byte < 66) {
      const auto o = byte - 2;
      return record + o / 16 * 128 + r * 16 + o % 16;
    }
    if (byte < 74)
      return record + 768 + r * 8 + byte - 66;
    if (byte < 106) {
      const auto o = byte - 74;
      return record + 512 + o / 16 * 128 + r * 16 + o % 16;
    }
    return record + 832 + r * 4 + byte - 106;
  }
  const auto index = row / 4 * (4 * blocks) + block * 4 + row % 4;
  if (byte < 2)
    return full_rows * blocks * 96 + index * 2 + byte;
  if (byte < 66)
    return index * 64 + byte - 2;
  return full_rows * blocks * 64 + index * 32 + byte - 66;
}

// GPU unpack index map, the inverse of the copy plan through Iq3ByteOffset:
// native byte n of a logical view whose row 0 is physical row row_begin ->
// physical packed byte. Every IQ3 field boundary and packed field offset is
// even, so for even n the bytes [n, n+2) are contiguous in both layouts.
template<QwenWeightLayout Layout>
GUFO_IQ3_HD constexpr std::size_t Iq3UnpackSourceOffset(
    std::size_t full_rows, std::uint32_t blocks, std::size_t row_begin,
    std::uint32_t native) noexcept {
  constexpr auto bb = static_cast<std::uint32_t>(Iq3BlockBytes(Layout));
  const auto block = native / bb;
  return Iq3ByteOffset(Layout, full_rows, blocks, row_begin + block / blocks,
                       block % blocks, native % bb);
}

inline void ValidateIq3Range(QwenWeightLayout layout, std::size_t rows,
                             std::size_t columns, std::size_t begin,
                             std::size_t end) {
  if (!Iq3ShapeValid(layout, rows, columns) || begin > end || end > rows ||
      begin % Iq3GroupRows(layout) || end % Iq3GroupRows(layout))
    throw std::invalid_argument("invalid IQ3 full-group row range");
}

// Full tensor bases; disjoint source/destination. Writes only the specified
// rows, in all full-tensor planes. The inverse uses the EXACT same byte map.
inline void CopyIq3RowRange(const void* source, void* destination,
                            QwenWeightLayout layout, std::size_t rows,
                            std::size_t columns, std::size_t begin,
                            std::size_t end, bool unpack = false) {
  ValidateIq3Range(layout, rows, columns, begin, end);
  if (begin == end)
    return;
  if (!source || !destination)
    throw std::invalid_argument("null IQ3 copy span");
  const auto bytes = Iq3PackedBytes(layout, rows, columns);
  const auto s = reinterpret_cast<std::uintptr_t>(source);
  const auto d = reinterpret_cast<std::uintptr_t>(destination);
  if ((s <= d ? d - s : s - d) < bytes)
    throw std::invalid_argument("overlapping IQ3 copy spans");
  const auto* in = static_cast<const std::uint8_t*>(source);
  auto* out = static_cast<std::uint8_t*>(destination);
  const auto nb = columns / 256;
  const auto bb = Iq3BlockBytes(layout);
  for (auto r = begin; r < end; ++r)
    for (std::size_t b = 0; b < nb; ++b)
      for (std::size_t o = 0; o < bb; ++o) {
        const auto native = (r * nb + b) * bb + o;
        const auto packed = Iq3ByteOffset(layout, rows, nb, r, b, o);
        if (unpack)
          out[native] = in[packed];
        else
          out[packed] = in[native];
      }
}

// Production forward copier: constant-size field copies, full groups only.
// Keep CopyIq3RowRange above as the independent per-byte oracle/inverse.
inline void CopyIq3RowRangeFast(const void* source, void* destination,
                                QwenWeightLayout layout, std::size_t rows,
                                std::size_t columns, std::size_t begin,
                                std::size_t end) {
  ValidateIq3Range(layout, rows, columns, begin, end);
  if (begin == end)
    return;
  const auto bytes = Iq3PackedBytes(layout, rows, columns);
  const auto s = reinterpret_cast<std::uintptr_t>(source);
  const auto d = reinterpret_cast<std::uintptr_t>(destination);
  if (!source || !destination || (s <= d ? d - s : s - d) < bytes)
    throw std::invalid_argument("invalid/overlapping fast IQ3 copy spans");
  const auto* in = static_cast<const std::uint8_t*>(source);
  auto* out = static_cast<std::uint8_t*>(destination);
  const auto nb = columns / 256;
  const auto group_rows = Iq3GroupRows(layout);
  for (auto g = begin; g < end; g += group_rows)
    for (std::size_t b = 0; b < nb; ++b)
      for (std::size_t r = 0; r < group_rows; ++r) {
        const auto* block = in + ((g + r) * nb + b) * Iq3BlockBytes(layout);
        if (layout == QwenWeightLayout::kIq3SRows8) {
          auto* record = out + (g / 8 * nb + b) * 880;
          for (std::size_t c = 0; c < 4; ++c)
            std::memcpy(record + c * 128 + r * 16, block + 2 + c * 16, 16);
          for (std::size_t c = 0; c < 2; ++c)
            std::memcpy(record + 512 + c * 128 + r * 16, block + 74 + c * 16,
                        16);
          std::memcpy(record + 768 + r * 8, block + 66, 8);
          std::memcpy(record + 832 + r * 4, block + 106, 4);
          std::memcpy(record + 864 + r * 2, block, 2);
        } else {
          const auto i = g * nb + b * 4 + r;
          std::memcpy(out + i * 64, block + 2, 64);
          std::memcpy(out + rows * nb * 64 + i * 32, block + 66, 32);
          std::memcpy(out + rows * nb * 96 + i * 2, block, 2);
        }
      }
}

}  // namespace gufo::models
#undef GUFO_IQ3_HD
#endif

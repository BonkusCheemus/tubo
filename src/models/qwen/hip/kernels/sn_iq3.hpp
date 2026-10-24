#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_HPP_

// Strix-native IQ3_S layout, variant B: the host-side load-time repack of IQ3_S
// rows into the row-interleaved layout the snb kernels consume, plus its exact
// inverse and the byte map (also documented in sn_iq3.hip).
//
// Why: block_iq3_s is 110 bytes and only 2-byte aligned, so a wave whose lanes
// each own one output row would touch 32 cache lines per 16-byte load. The
// layout groups kSnIq3RowsPerGroup (8) rows and stores five planes per block,
// each plane chunk-major over the group, so the 8 lanes of a group cover one
// 128-byte line. The repack is a permutation: SnIq3Unpack inverts it byte for
// byte, and only a trailing partial group pays up to 15 pad bytes per record.

#include <cstddef>
#include <cstdint>
#include <cstring>

// The CPU layout test includes this header in a plain C++ TU, which knows
// neither __host__ nor __device__; the kernel build needs both.
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define GUFO_SN_IQ3_HD __host__ __device__
#else
#define GUFO_SN_IQ3_HD
#endif

namespace gufo::hip {

// Sizes of ggml's block_iq3_s fields; all must match core::quant::block_iq3_s.
inline constexpr std::size_t kSnIq3Elements = 256;
inline constexpr std::size_t kSnIq3BlockBytes = 110;
inline constexpr std::size_t kSnIq3RowsPerGroup = 8;
inline constexpr std::size_t kSnIq3QsBytes = 64;
inline constexpr std::size_t kSnIq3SignsBytes = 32;
inline constexpr std::size_t kSnIq3QhBytes = 8;
inline constexpr std::size_t kSnIq3ScalesBytes = 4;
inline constexpr std::size_t kSnIq3ScaleBytes = 2;

// Field offsets inside one block record, as multiples of the row count.
inline constexpr std::size_t kSnIq3SignsOffset = kSnIq3QsBytes;
inline constexpr std::size_t kSnIq3QhOffset =
    kSnIq3SignsOffset + kSnIq3SignsBytes;
inline constexpr std::size_t kSnIq3ScalesOffset =
    kSnIq3QhOffset + kSnIq3QhBytes;
inline constexpr std::size_t kSnIq3DOffset =
    kSnIq3ScalesOffset + kSnIq3ScalesBytes;

// Source byte offsets inside ggml's block_iq3_s (fp16 d, then the same order).
inline constexpr std::size_t kSnIq3SrcQs = kSnIq3ScaleBytes;
inline constexpr std::size_t kSnIq3SrcQh = kSnIq3SrcQs + kSnIq3QsBytes;
inline constexpr std::size_t kSnIq3SrcSigns = kSnIq3SrcQh + kSnIq3QhBytes;
inline constexpr std::size_t kSnIq3SrcScales =
    kSnIq3SrcSigns + kSnIq3SignsBytes;

[[nodiscard]] GUFO_SN_IQ3_HD inline constexpr std::size_t SnIq3RoundUp16(
    std::size_t bytes) noexcept {
  return (bytes + 15U) & ~static_cast<std::size_t>(15U);
}

// Block records per row: columns must be a multiple of kSnIq3Elements.
[[nodiscard]] GUFO_SN_IQ3_HD inline constexpr std::size_t SnIq3Blocks(
    std::size_t columns) noexcept {
  return columns / kSnIq3Elements;
}

// Bytes of one block record for `rows_in_group` rows: kSnIq3BlockBytes per row,
// rounded up to 16 so a phase's record base stays 16-byte aligned.
[[nodiscard]] GUFO_SN_IQ3_HD inline constexpr std::size_t SnIq3BlockStride(
    std::size_t rows_in_group) noexcept {
  return SnIq3RoundUp16(rows_in_group * kSnIq3BlockBytes);
}

// Bytes of one full group. Every group but the last one of a tensor has this
// size, so a group's base offset is group_index * SnIq3GroupStride(blocks).
[[nodiscard]] GUFO_SN_IQ3_HD inline constexpr std::size_t SnIq3GroupStride(
    std::size_t blocks) noexcept {
  return blocks * SnIq3BlockStride(kSnIq3RowsPerGroup);
}

// Rows in `group`, clamped to the 1..kSnIq3RowsPerGroup window.
[[nodiscard]] GUFO_SN_IQ3_HD inline constexpr std::size_t SnIq3GroupRows(
    std::size_t rows, std::size_t group) noexcept {
  const std::size_t base = group * kSnIq3RowsPerGroup;
  const std::size_t left = rows > base ? rows - base : 0U;
  return left < kSnIq3RowsPerGroup ? left : kSnIq3RowsPerGroup;
}

[[nodiscard]] GUFO_SN_IQ3_HD inline constexpr std::size_t SnIq3GroupBytes(
    std::size_t blocks, std::size_t rows_in_group) noexcept {
  return blocks * SnIq3BlockStride(rows_in_group);
}

// Total size of the repacked tensor: never over +2% of the original bytes.
[[nodiscard]] GUFO_SN_IQ3_HD inline constexpr std::size_t SnIq3PackedBytes(
    std::size_t rows, std::size_t columns) noexcept {
  const std::size_t blocks = SnIq3Blocks(columns);
  std::size_t total = 0;
  for (std::size_t group = 0; group * kSnIq3RowsPerGroup < rows; ++group)
    total += SnIq3GroupBytes(blocks, SnIq3GroupRows(rows, group));
  return total;
}

// Repack `rows` x `columns` IQ3_S weights from ggml's row-major blocks; `dst`
// must hold SnIq3PackedBytes(rows, columns) bytes and must not overlap `src`.
inline void SnIq3Repack(const void* src, std::uint8_t* dst, std::size_t rows,
                        std::size_t columns) noexcept {
  const std::size_t blocks = SnIq3Blocks(columns);
  const auto* const in = static_cast<const std::uint8_t*>(src);
  std::size_t group_base = 0;
  for (std::size_t group = 0; group * kSnIq3RowsPerGroup < rows; ++group) {
    const std::size_t n = SnIq3GroupRows(rows, group);
    const std::size_t chunks = 16U * n;
    const std::size_t signs = n * kSnIq3SignsOffset;
    const std::size_t qh = n * kSnIq3QhOffset;
    const std::size_t scales = n * kSnIq3ScalesOffset;
    const std::size_t d = n * kSnIq3DOffset;
    const std::size_t record = SnIq3BlockStride(n);
    std::uint8_t* const group_out = dst + group_base;
    for (std::size_t b = 0; b < blocks; ++b) {
      std::uint8_t* const block = group_out + (b * record);
      for (std::size_t r = 0; r < n; ++r) {
        const std::size_t src_index =
            (((group * kSnIq3RowsPerGroup) + r) * blocks) + b;
        const std::uint8_t* const row = in + (src_index * kSnIq3BlockBytes);
        for (std::size_t c = 0; c < 4U; ++c)
          std::memcpy(block + (c * chunks) + (r * 16U),
                      row + kSnIq3SrcQs + (16U * c), 16);
        for (std::size_t c = 0; c < 2U; ++c)
          std::memcpy(block + signs + (c * chunks) + (r * 16U),
                      row + kSnIq3SrcSigns + (16U * c), 16);
        std::memcpy(block + qh + (r * 8U), row + kSnIq3SrcQh, 8);
        std::memcpy(block + scales + (r * 4U), row + kSnIq3SrcScales, 4);
        std::memcpy(block + d + (r * 2U), row, 2);
      }
      for (std::size_t i = n * kSnIq3BlockBytes; i < record; ++i)
        block[i] = 0;
    }
    group_base += SnIq3GroupBytes(blocks, n);
  }
}

// Exact inverse of SnIq3Repack: restores ggml's row-major block order.
inline void SnIq3Unpack(const std::uint8_t* packed, void* dst,
                        std::size_t rows, std::size_t columns) noexcept {
  const std::size_t blocks = SnIq3Blocks(columns);
  auto* const out = static_cast<std::uint8_t*>(dst);
  std::size_t group_base = 0;
  for (std::size_t group = 0; group * kSnIq3RowsPerGroup < rows; ++group) {
    const std::size_t n = SnIq3GroupRows(rows, group);
    const std::size_t chunks = 16U * n;
    const std::size_t signs = n * kSnIq3SignsOffset;
    const std::size_t qh = n * kSnIq3QhOffset;
    const std::size_t scales = n * kSnIq3ScalesOffset;
    const std::size_t d = n * kSnIq3DOffset;
    const std::size_t record = SnIq3BlockStride(n);
    const std::uint8_t* const group_in = packed + group_base;
    for (std::size_t b = 0; b < blocks; ++b) {
      const std::uint8_t* const block = group_in + (b * record);
      for (std::size_t r = 0; r < n; ++r) {
        const std::size_t dst_index =
            (((group * kSnIq3RowsPerGroup) + r) * blocks) + b;
        std::uint8_t* const row = out + (dst_index * kSnIq3BlockBytes);
        for (std::size_t c = 0; c < 4U; ++c)
          std::memcpy(row + kSnIq3SrcQs + (16U * c),
                      block + (c * chunks) + (r * 16U), 16);
        for (std::size_t c = 0; c < 2U; ++c)
          std::memcpy(row + kSnIq3SrcSigns + (16U * c),
                      block + signs + (c * chunks) + (r * 16U), 16);
        std::memcpy(row + kSnIq3SrcQh, block + qh + (r * 8U), 8);
        std::memcpy(row + kSnIq3SrcScales, block + scales + (r * 4U), 4);
        std::memcpy(row, block + d + (r * 2U), 2);
      }
    }
    group_base += SnIq3GroupBytes(blocks, n);
  }
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_HPP_

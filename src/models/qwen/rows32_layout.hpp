#ifndef GUFO_MODELS_QWEN_ROWS32_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_ROWS32_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gufo::models {

enum class QwenWeightLayout : std::uint8_t {
  kNative,
  kRows32,
  kIq3SRows8,
  kIq3XxsRows4
};

// Native block byte -> Rows32 byte. rows/blocks describe the FULL tensor,
// including when copying only a row range. Q6 d / IQ4 heads are global planes.
constexpr std::size_t Rows32ByteOffset(std::size_t rows, std::size_t blocks,
                                       std::size_t row, std::size_t block,
                                       std::size_t byte,
                                       std::size_t block_bytes) noexcept {
  const bool iq4 = block_bytes == 136;
  const std::size_t meta = iq4 ? 8 : (block_bytes == 210 ? 2 : 0);
  const std::size_t payload = block_bytes - meta;
  const std::size_t tile = row / 32 * blocks + block;
  if ((iq4 && byte < meta) || (!iq4 && byte >= payload))
    return rows * blocks * payload + (tile * 32 + row % 32) * meta +
           (iq4 ? byte : byte - payload);
  const auto offset = byte - (iq4 ? meta : 0);
  return tile * 32 * payload + offset / 16 * 512 + row % 32 * 16 + offset % 16;
}

// Source and destination are non-overlapping full tensor bases. Row ranges
// must cover whole tiles. No tensor-sized scratch; metadata retains full R.
inline void InterleaveKQuantRows32Range(const void* src, void* dst,
                                        std::size_t rows, std::size_t k,
                                        std::size_t block_bytes,
                                        std::size_t begin, std::size_t end) {
  const bool iq4 = block_bytes == 136;
  const std::size_t meta = iq4 ? 8 : (block_bytes == 210 ? 2 : 0);
  const std::size_t payload = block_bytes - meta;
  if (rows % 32 || k % 256 || !block_bytes || payload % 16 || begin % 32 ||
      end % 32 || begin > end || end > rows ||
      (k / 256 && rows > std::numeric_limits<std::size_t>::max() / (k / 256)) ||
      (rows * (k / 256) >
       std::numeric_limits<std::size_t>::max() / block_bytes))
    throw std::invalid_argument("invalid Rows32 tensor or full-tile row range");
  if (begin == end || !k)
    return;
  if (!src || !dst)
    throw std::invalid_argument("null Rows32 source/destination");
  const std::size_t bytes = rows * (k / 256) * block_bytes;
  const auto saddr = reinterpret_cast<std::uintptr_t>(src);
  const auto daddr = reinterpret_cast<std::uintptr_t>(dst);
  if ((saddr <= daddr ? daddr - saddr : saddr - daddr) < bytes)
    throw std::invalid_argument("Rows32 source/destination overlap");
  const auto* s = static_cast<const std::uint8_t*>(src);
  auto* d = static_cast<std::uint8_t*>(dst);
  const std::size_t blocks = k / 256;
  for (std::size_t row = begin; row < end; ++row) {
    for (std::size_t b = 0; b < blocks; ++b) {
      const auto* native = s + (row * blocks + b) * block_bytes;
      for (std::size_t c = 0; c < payload / 16; ++c)
        std::memcpy(d + Rows32ByteOffset(rows, blocks, row, b,
                                         (iq4 ? 8 : 0) + c * 16, block_bytes),
                    native + (iq4 ? 8 : 0) + c * 16, 16);
      if (meta)
        std::memcpy(d + Rows32ByteOffset(rows, blocks, row, b,
                                         iq4 ? 0 : payload, block_bytes),
                    native + (iq4 ? 0 : payload), meta);
    }
  }
}

inline void InterleaveKQuantRows32(const void* src, void* dst, std::size_t rows,
                                   std::size_t k, std::size_t block_bytes) {
  InterleaveKQuantRows32Range(src, dst, rows, k, block_bytes, 0, rows);
}

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_ROWS32_LAYOUT_HPP_

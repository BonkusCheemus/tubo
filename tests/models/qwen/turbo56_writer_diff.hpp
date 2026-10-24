#ifndef GUFO_TESTS_QWEN_TURBO56_WRITER_DIFF_HPP_
#define GUFO_TESTS_QWEN_TURBO56_WRITER_DIFF_HPP_

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace gufo::test {

// Planner-approved numerical contract, not proof of an exact centroid tie.
inline bool Turbo56WriterByteDiffAllowed(const std::uint8_t* gpu,
                                         const std::uint8_t* host,
                                         std::size_t byte,
                                         hip::turbo4::Bits bits) {
  namespace t4 = hip::turbo4;
  if (byte >= t4::BlockBytes(bits))
    return false;  // Padding is never tolerant.
  if (byte < t4::kNormBytes) {
    const unsigned g = gpu[0] | (static_cast<unsigned>(gpu[1]) << 8U);
    const unsigned h = host[0] | (static_cast<unsigned>(host[1]) << 8U);
    // Nonnegative finite half encodings are ordered in one-half-ULP steps.
    return t4::GroupNormIsStorable(t4::HalfBitsToFloat(g)) &&
           t4::GroupNormIsStorable(t4::HalfBitsToFloat(h)) &&
           (g > h ? g - h : h - g) <= 2U;
  }
  const bool qs = byte < t4::kQhOffset;
  const std::size_t count = qs ? 2U : bits == t4::Bits::k5 ? 8U : 4U;
  const std::size_t first = qs ? (byte - t4::kNormBytes) * 2U
                              : (byte - t4::kQhOffset) * count;
  for (std::size_t j = first; j < first + count; ++j) {
    const auto g = t4::CodeAt(bits, gpu, j);
    const auto h = t4::CodeAt(bits, host, j);
    if ((g > h ? g - h : h - g) > 1U)
      return false;
  }
  return true;
}

inline bool Turbo56WriterDiffCountAllowed(std::size_t diff,
                                          std::size_t total_bytes) {
  return diff <= total_bytes / 10000U;  // Exact inclusive 1e-4 byte cap.
}

}  // namespace gufo::test

#endif  // GUFO_TESTS_QWEN_TURBO56_WRITER_DIFF_HPP_

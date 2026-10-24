// Shared pseudo-random block generator for the iron quant tests
// (CPU: tests/core/iquant_decode_test.cpp, GPU:
// tests/models/qwen/hip/quant/iquant_decode_gpu_test.hip).
//
// Random bytes cover every packed field (grid index, sign index, scale nibble).
// Only the fp16 scale fields are conditioned: a random half is often inf, NaN
// or denormal, which would turn a bit-exact comparison into noise. The scales
// span 2^-3 .. 2^5 and both signs.
#ifndef GUFO_TESTS_CORE_IQUANT_TEST_BLOCKS_HPP_
#define GUFO_TESTS_CORE_IQUANT_TEST_BLOCKS_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::test {

struct IQuantCase {
  core::GgmlType type;
  const char* name;
};

inline constexpr IQuantCase kIQuantCases[] = {
    {core::GgmlType::kQ2_K, "Q2_K"},
    {core::GgmlType::kIQ2_XXS, "IQ2_XXS"},
    {core::GgmlType::kIQ2_XS, "IQ2_XS"},
    {core::GgmlType::kIQ2_S, "IQ2_S"},
    {core::GgmlType::kIQ3_XXS, "IQ3_XXS"},
    {core::GgmlType::kIQ1_M, "IQ1_M"},
};

inline std::uint32_t NextRandom(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

inline std::uint16_t RandomHalfBits(std::uint32_t& state) {
  const std::uint32_t r = NextRandom(state);
  const std::uint32_t exponent = 12 + (r % 9);  // 2^-3 .. 2^5
  return static_cast<std::uint16_t>(((r >> 16) & 0x8000U) | (exponent << 10) |
                                    ((r >> 4) & 0x3FFU));
}

inline void PutU16(std::uint8_t* at, std::uint16_t v) {
  std::memcpy(at, &v, sizeof(v));
}

/// `blocks` super-blocks of `type`, as one contiguous row.
inline std::vector<std::uint8_t> MakeIQuantBlocks(core::GgmlType type,
                                                  std::size_t blocks,
                                                  std::uint32_t seed) {
  const std::size_t block_bytes = quant::QuantizedRowBytes(type, 256);
  std::vector<std::uint8_t> bytes(blocks * block_bytes);
  std::uint32_t state = seed | 1U;
  for (auto& byte : bytes) {
    byte = static_cast<std::uint8_t>(NextRandom(state) >> 11);
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* block = bytes.data() + (b * block_bytes);
    switch (type) {
      case core::GgmlType::kQ2_K:
        PutU16(block + 80, RandomHalfBits(state));  // d
        PutU16(block + 82, RandomHalfBits(state));  // dmin
        break;
      case core::GgmlType::kIQ1_M: {
        // fp16 super-scale spread over the top nibbles of 4 uint16 words.
        const std::uint16_t bits = RandomHalfBits(state);
        for (int word = 0; word < 4; ++word) {
          std::uint8_t* at = block + 48 + (2 * word);
          at[1] = static_cast<std::uint8_t>(
              (at[1] & 0x0FU) | (((bits >> (4 * word)) & 0x0FU) << 4U));
        }
        break;
      }
      default:
        PutU16(block, RandomHalfBits(state));  // d is the first field
        break;
    }
  }
  return bytes;
}

}  // namespace gufo::test

#endif  // GUFO_TESTS_CORE_IQUANT_TEST_BLOCKS_HPP_

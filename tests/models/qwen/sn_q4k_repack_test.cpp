// clang-format off
// CPU-only checks for the repacked layout in
// src/models/qwen/hip/kernels/sn_q4k.hpp.
//
// Random and edge blocks (all bits set/clear, extreme six-bit scales and
// mins, signed zero and subnormal fp16 d/dmin, infinities) must satisfy
// unpack(repack(x)) == x byte for byte; the scalar dequant read from the
// repacked layout must equal the ggml dequant bit for bit; and the repacked
// group words must match get_scale_min_k4 directly. Row counts cover both
// multiples and non-multiples of every plausible interleave, although the SN
// layout keeps rows independent so all counts are legal.
// clang-format on

#define GUFO_SN_Q4K_HOST_ONLY 1
#define GUFO_SN_Q4K_K4_HOST_ONLY 1

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/qwen/hip/kernels/sn_q4k.hpp"
#include "src/models/qwen/hip/kernels/sn_q4k_k4.hpp"

namespace {

using gufo::hip::sn_q4k::GetScaleMinK4;
using gufo::hip::sn_q4k::kBlockBytes;
using gufo::hip::sn_q4k::kGgmlBlockBytes;
using gufo::hip::sn_q4k::kQk;
using gufo::hip::sn_q4k::RepackBlock;
using gufo::hip::sn_q4k::RepackedRowBytes;
using gufo::hip::sn_q4k::RepackTensor;
using gufo::hip::sn_q4k::UnpackTensor;

void Fail(const char* what) {
  std::fprintf(stderr, "FAIL: %s\n", what);
  std::exit(1);
}

void Check(bool condition, const char* what) {
  if (!condition) {
    Fail(what);
  }
}

std::uint16_t ReadU16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) |
         static_cast<std::uint16_t>(p[1] << 8);
}

std::uint32_t ReadU32(const std::uint8_t* p) {
  std::uint32_t value;
  std::memcpy(&value, p, sizeof(value));
  return value;
}

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
  const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10U) & 0x1FU;
  const std::uint32_t man = static_cast<std::uint32_t>(h) & 0x3FFU;
  std::uint32_t bits = 0;
  if (exp == 0U) {
    if (man == 0U) {
      bits = sign;
    } else {
      int e = -1;
      std::uint32_t m = man;
      do {
        ++e;
        m <<= 1U;
      } while ((m & 0x400U) == 0U);
      bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23U) |
             ((m & 0x3FFU) << 13U);
    }
  } else if (exp == 31U) {
    bits = sign | 0x7F800000U | (man << 13U);
  } else {
    bits = sign | ((exp - 15U + 127U) << 23U) | (man << 13U);
  }
  return __builtin_bit_cast(float, bits);
}

// get_scale_min_k4 against a hand-computed vector, independent of the repack.
void CheckReferencePacking() {
  const std::uint8_t scales[12] = {
      0xC1, 0x82, 0x43, 0x04, 0xC5, 0x86, 0x47, 0x08, 0x19, 0x2A, 0x3B, 0x0C};
  const std::uint32_t expected_sc[8] = {1, 2, 3, 4, 57, 42, 27, 12};
  const std::uint32_t expected_mn[8] = {5, 6, 7, 8, 49, 34, 19, 0};
  for (std::uint32_t j = 0; j < 8U; ++j) {
    std::uint32_t sc = 0;
    std::uint32_t mn = 0;
    GetScaleMinK4(j, scales, &sc, &mn);
    Check(sc == expected_sc[j], "get_scale_min_k4 reference scale");
    Check(mn == expected_mn[j], "get_scale_min_k4 reference min");
  }
}

// ggml dequantize_row_q4_K's arithmetic, read from the original block.
void DequantGgml(const std::uint8_t* block, float* out) {
  const float d = HalfToFloat(ReadU16(block));
  const float dmin = HalfToFloat(ReadU16(block + 2U));
  const std::uint8_t* const scales = block + 4U;
  const std::uint8_t* const qs = block + 16U;
  for (std::uint32_t j = 0; j < 8U; ++j) {
    std::uint32_t sc = 0;
    std::uint32_t mn = 0;
    GetScaleMinK4(j, scales, &sc, &mn);
    const float ds = d * static_cast<float>(sc);
    const float ms = dmin * static_cast<float>(mn);
    for (std::uint32_t l = 0; l < 32U; ++l) {
      const std::uint32_t q =
          (qs[((j >> 1U) * 32U) + l] >> (4U * (j & 1U))) & 0x0FU;
      out[(j * 32U) + l] = ds * static_cast<float>(q) - ms;
    }
  }
}

// Forward declarations: the known block is checked in both layouts below.
void DequantSn(const std::uint8_t* block, float* out);
bool SameBits(float a, float b);

// Hand-built block with known elements, independent of FillBlock and the
// repack: low nibbles are elements 0-31 in byte order, high nibbles 32-63.
void CheckKnownBlock() {
  // d = 1.0, dmin = 1.0; sub-blocks 0 and 1 both have sc = 2 and mn = 1
  // (get_scale_min_k4: sc = scales[0]/scales[1], mn = scales[4]/scales[5]).
  std::uint8_t block[144] = {};
  block[0] = 0x00;
  block[1] = 0x3C;
  block[2] = 0x00;
  block[3] = 0x3C;
  block[4] = 2;  // sc of sub-block 0
  block[5] = 2;  // sc of sub-block 1
  block[8] = 1;  // mn of sub-block 0
  block[9] = 1;  // mn of sub-block 1
  // Byte 0: low = element 0 (5), high = 32 (0); byte 1: high = element 33 (3).
  block[16] = 0x05;
  block[17] = 0x30;
  float values[256] = {};
  DequantGgml(block, values);
  Check(values[0] == 9.0F, "known element 0");
  Check(values[1] == -1.0F, "known element 1");
  Check(values[32] == -1.0F, "known element 32");
  Check(values[33] == 5.0F, "known element 33");

  // The same block through the repacked layout directly, so sub-block 1's
  // group word (scale and min) is checked without relying on the random loop.
  std::uint8_t sn[144] = {};
  RepackBlock(block, sn);
  float sn_values[256] = {};
  DequantSn(sn, sn_values);
  for (std::uint32_t i = 0; i < kQk; ++i) {
    Check(SameBits(values[i], sn_values[i]), "known SN dequant");
  }
}

// The same arithmetic, but the pair is read straight from the SN group words.
void DequantSn(const std::uint8_t* block, float* out) {
  const float d = HalfToFloat(ReadU16(block + 140U));
  const float dmin = HalfToFloat(ReadU16(block + 142U));
  const std::uint8_t* const qs = block;
  const std::uint8_t* const groups = block + 128U;
  for (std::uint32_t j = 0; j < 8U; ++j) {
    const std::uint32_t base = (j >> 1U) * 3U;
    const std::uint32_t word =
        static_cast<std::uint32_t>(groups[base]) |
        (static_cast<std::uint32_t>(groups[base + 1U]) << 8U) |
        (static_cast<std::uint32_t>(groups[base + 2U]) << 16U);
    const std::uint32_t shift = (j & 1U) != 0U ? 12U : 0U;
    const std::uint32_t sc = (word >> shift) & 63U;
    const std::uint32_t mn = (word >> (shift + 6U)) & 63U;
    const float ds = d * static_cast<float>(sc);
    const float ms = dmin * static_cast<float>(mn);
    for (std::uint32_t l = 0; l < 32U; ++l) {
      const std::uint32_t q =
          (qs[((j >> 1U) * 32U) + l] >> (4U * (j & 1U))) & 0x0FU;
      out[(j * 32U) + l] = ds * static_cast<float>(q) - ms;
    }
  }
}

// The SN group words against get_scale_min_k4 on the original scales.
void CheckScaleGroups(const std::uint8_t* ggml, const std::uint8_t* sn) {
  for (std::uint32_t j = 0; j < 8U; ++j) {
    std::uint32_t sc = 0;
    std::uint32_t mn = 0;
    GetScaleMinK4(j, ggml + 4U, &sc, &mn);
    const std::uint32_t base = 128U + ((j >> 1U) * 3U);
    const std::uint32_t word =
        static_cast<std::uint32_t>(sn[base]) |
        (static_cast<std::uint32_t>(sn[base + 1U]) << 8U) |
        (static_cast<std::uint32_t>(sn[base + 2U]) << 16U);
    const std::uint32_t shift = (j & 1U) != 0U ? 12U : 0U;
    Check(((word >> shift) & 63U) == sc, "SN scale field");
    Check(((word >> (shift + 6U)) & 63U) == mn, "SN min field");
  }
}

bool SameBits(float a, float b) {
  const std::uint32_t ua = __builtin_bit_cast(std::uint32_t, a);
  const std::uint32_t ub = __builtin_bit_cast(std::uint32_t, b);
  return ua == ub || (std::isnan(a) && std::isnan(b));
}

// Modes: 0 random, 1 clear, 2 set, 3 extreme, 4 +-0, 5 subnormal, 6 inf.
void FillBlock(std::uint8_t* block, std::mt19937* rng, std::uint32_t mode) {
  for (std::uint32_t i = 0; i < kGgmlBlockBytes; ++i) {
    block[i] = static_cast<std::uint8_t>((*rng)() & 0xFFU);
  }
  switch (mode) {
    case 1U:
      std::memset(block, 0x00, kGgmlBlockBytes);
      break;
    case 2U:
      std::memset(block, 0xFF, kGgmlBlockBytes);
      break;
    case 3U:  // every six-bit field 63, qs zero, d/dmin +-65504
      std::memset(block + 16U, 0x00, 128U);
      std::memset(block + 4U, 0xFF, 12U);
      block[0] = 0xFF;
      block[1] = 0x7B;
      block[2] = 0xFF;
      block[3] = 0xFB;
      break;
    case 4U:  // d = -0, dmin = +0
      block[0] = 0x00;
      block[1] = 0x80;
      block[2] = 0x00;
      block[3] = 0x00;
      break;
    case 5U:  // d = 2^-24, dmin = -largest subnormal
      block[0] = 0x01;
      block[1] = 0x00;
      block[2] = 0xFF;
      block[3] = 0x83;
      break;
    case 6U:  // +inf and -inf
      block[0] = 0x00;
      block[1] = 0x7C;
      block[2] = 0x00;
      block[3] = 0xFC;
      break;
    default:
      break;
  }
}

// Check the actual host helper against an independent group/block/chunk/lane
// inverse, for both Q4_K's 9 chunks and Q5_K's 11 chunks.
// CPU model of the shared decode fields and interleaved reads versus ggml's
// scalar natural-order dequant. GPU Iq4NlLookup4 is reused unchanged.
void CheckIq4XsDot() {
  constexpr std::size_t rows = 64U, blocks = 3U, copies = 2U;
  constexpr std::size_t bytes = rows * blocks * 136U;
  constexpr int table[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1,    13,   25,  38,  53,  69,  89,  113};
  constexpr std::uint16_t ds[] = {0x3C00, 0xBC00, 0x1400, 0x7BFF, 0x0001,
                                  0x8001, 0x3555, 0xB555, 0x0000, 0x8000};
  std::mt19937 rng(20261019U);
  std::vector<std::uint8_t> original(copies * bytes), packed(copies * bytes);
  for (auto& v : original)
    v = static_cast<std::uint8_t>(rng());
  for (std::size_t c = 0; c < copies; ++c) {
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t b = 0; b < blocks; ++b) {
        auto* block = original.data() + c * bytes + (r * blocks + b) * 136U;
        const std::size_t tag = (c * rows + r) * blocks + b;
        std::memcpy(block, &ds[tag % 10U], 2U);
        if (tag < 64U) {
          const auto hi = static_cast<std::uint16_t>((tag >> 4U) * 0x5555U);
          std::memcpy(block + 2U, &hi, 2U);
          std::memset(block + 4U, static_cast<int>((tag & 15U) * 17U), 4U);
        }
      }
    gufo::hip::InterleaveKQuantRows32(original.data() + c * bytes,
                                      packed.data() + c * bytes, rows,
                                      blocks * 256U, 136U);
  }
  std::size_t cases = 0;
  for (std::size_t c = 0; c < copies; ++c)
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t b = 0; b < blocks; ++b) {
        const auto* ref = original.data() + c * bytes + (r * blocks + b) * 136U;
        const std::size_t tiled = (r / 32U * blocks + b) * 32U + r % 32U;
        const auto* meta =
            packed.data() + c * bytes + rows * blocks * 128U + tiled * 8U;
        const auto hx = ReadU32(meta), hy = ReadU32(meta + 4U);
        const float d = HalfToFloat(ReadU16(meta));
        float reference[256];
        for (std::uint32_t j = 0; j < 8U; ++j) {
          const int sc =
              static_cast<int>(((ref[4U + j / 2U] >> (4U * (j & 1U))) & 15U) |
                               (((ReadU16(ref + 2U) >> (2U * j)) & 3U) << 4U)) -
              32;
          const float scale =
              HalfToFloat(ReadU16(ref)) * static_cast<float>(sc);
          for (std::uint32_t l = 0; l < 16U; ++l) {
            const auto q = ref[8U + 16U * j + l];
            reference[32U * j + l] = scale * static_cast<float>(table[q & 15U]);
            reference[32U * j + 16U + l] =
                scale * static_cast<float>(table[q >> 4U]);
          }
        }
        for (std::size_t t = 0; t < 8U; ++t) {
          std::int8_t x[256];
          for (auto& v : x)
            v = static_cast<std::int8_t>(
                t == 0U   ? 127
                : t == 1U ? -127
                          : static_cast<int>(rng() % 255U) - 127);
          int isum = 0;
          for (std::uint32_t gp = 0; gp < 4U; ++gp)
            for (std::uint32_t sub = 0; sub < 4U; ++sub) {
              const auto sub16 = gp * 4U + sub;
              const auto* q = packed.data() + c * bytes +
                              (r / 32U * blocks + b) * 32U * 128U +
                              (2U * gp + sub / 2U) * 512U + (r % 32U) * 16U;
              int dot = 0;
              for (std::uint32_t i = 0; i < 4U; ++i) {
                const auto indices = gufo::hip::sn_q4k_k4_detail::Iq4XsIndices4(
                    ReadU32(q + 4U * i), sub16);
                for (std::uint32_t j = 0; j < 4U; ++j) {
                  const auto n = (indices >> (8U * j)) & 15U;
                  const std::uint64_t lut =
                      n < 8U ? 0xF6EADDCFBFAD9881ULL : 0x7159453526190D01ULL;
                  const auto code =
                      static_cast<std::int8_t>((lut >> (8U * (n & 7U))) & 255U);
                  dot += static_cast<int>(code) * x[sub16 * 16U + i * 4U + j];
                }
              }
              isum +=
                  gufo::hip::sn_q4k_k4_detail::Iq4XsScale(hx, hy, sub16) * dot;
            }
          Check(std::abs(isum) <= 132128768, "IQ4_XS integer bound");
          double expected = 0;
          for (std::uint32_t i = 0; i < 256U; ++i)
            expected += static_cast<double>(reference[i]) * x[i];
          Check(static_cast<double>(d) * isum == expected,
                "IQ4_XS integer decode dot differs from scalar ggml dequant");
          ++cases;
        }
      }
  std::printf("PASS: IQ4_XS scalar dequant dot (%zu block/token cases)\n",
              cases);
}

void CheckChunkInterleave() {
  std::mt19937 rng(20261019U);
  const std::size_t row_counts[] = {32U, 64U};
  const std::size_t ks[] = {256U, 512U, 6144U, 17408U};
  const std::size_t block_sizes[] = {144U, 176U, 210U, 136U};
  for (const std::size_t block_bytes : block_sizes) {
    for (const std::size_t rows : row_counts) {
      for (const std::size_t k : ks) {
        const std::size_t blocks = k / 256U;
        const bool iq4 = block_bytes == 136U;
        const std::size_t meta_bytes =
            iq4 ? 8U : (block_bytes == 210U ? 2U : 0U);
        const std::size_t chunk_bytes = block_bytes - meta_bytes;
        const std::size_t payload_offset = iq4 ? 8U : 0U;
        const std::size_t chunks = chunk_bytes / 16U;
        const std::size_t bytes = rows * blocks * block_bytes;
        std::vector<std::uint8_t> original(2U * bytes), packed(2U * bytes),
            restored(2U * bytes);
        for (auto& value : original)
          value = static_cast<std::uint8_t>(rng());
        for (std::size_t copy = 0; copy < 2U; ++copy) {
          // Distinct metadata tags catch plane/copy mixing deterministically.
          if (meta_bytes)
            for (std::size_t r = 0; r < rows; ++r)
              for (std::size_t b = 0; b < blocks; ++b) {
                const auto tag = static_cast<std::uint64_t>(
                    ((copy * rows + r) * blocks) + b);
                std::memcpy(original.data() + copy * bytes +
                                (r * blocks + b) * block_bytes +
                                (iq4 ? 0U : chunk_bytes),
                            &tag, meta_bytes);
              }
          gufo::hip::InterleaveKQuantRows32(original.data() + copy * bytes,
                                            packed.data() + copy * bytes, rows,
                                            k, block_bytes);
          std::vector<std::uint8_t> visited(rows * blocks * chunks, 0U);
          for (std::size_t group = 0; group < rows / 32U; ++group) {
            for (std::size_t b = 0; b < blocks; ++b) {
              for (std::size_t c = 0; c < chunks; ++c)
                for (std::size_t lane = 0; lane < 32U; ++lane) {
                  const std::size_t r = group * 32U + lane;
                  const std::size_t src_offset =
                      (r * blocks + b) * block_bytes + payload_offset + c * 16U;
                  const std::size_t dst_chunk =
                      ((group * blocks + b) * chunks + c) * 32U + lane;
                  Check(++visited[dst_chunk] == 1U,
                        "interleave repeats a chunk");
                  std::memcpy(restored.data() + copy * bytes + src_offset,
                              packed.data() + copy * bytes + dst_chunk * 16U,
                              16U);
                }
              if (meta_bytes)
                for (std::size_t lane = 0; lane < 32U; ++lane) {
                  const std::size_t r = group * 32U + lane;
                  const std::size_t d_offset =
                      rows * blocks * chunk_bytes +
                      ((group * blocks + b) * 32U + lane) * meta_bytes;
                  std::memcpy(restored.data() + copy * bytes +
                                  (r * blocks + b) * block_bytes +
                                  (iq4 ? 0U : chunk_bytes),
                              packed.data() + copy * bytes + d_offset,
                              meta_bytes);
                }
            }
          }
          for (const auto count : visited)
            Check(count == 1U, "interleave misses a chunk");
        }
        Check(original == restored, "K-quant chunk interleave round trip");
      }
    }
  }
  const auto rejects = [](std::size_t rows, std::size_t k, std::size_t bytes) {
    try {
      gufo::hip::InterleaveKQuantRows32(nullptr, nullptr, rows, k, bytes);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  Check(rejects(32U, 256U, 0U), "zero block size rejected");
  Check(rejects(32U, 256U, 175U), "unaligned block size rejected");
  Check(rejects(31U, 256U, 144U), "partial row group rejected");
  Check(rejects(32U, 257U, 176U), "partial K block rejected");
}

}  // namespace

int main() {
  CheckChunkInterleave();
  CheckIq4XsDot();
  CheckReferencePacking();
  CheckKnownBlock();

  std::mt19937 rng(20261017U);
  constexpr std::uint32_t kModes = 7U;
  const std::size_t row_counts[] = {
      1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 15U, 16U, 17U, 33U};
  const std::size_t ks[] = {256U, 512U, 2048U, 5120U};

  Check(RepackedRowBytes(256U) == kBlockBytes, "one SN block is 144 bytes");
  for (const std::size_t k : ks) {
    const std::size_t blocks = k / kQk;
    const std::size_t ggml_row = blocks * kGgmlBlockBytes;
    const std::size_t sn_row = RepackedRowBytes(k);
    Check(sn_row == blocks * kBlockBytes, "SN row size");
    Check(sn_row * 100U <= ggml_row * 102U, "repack stays within +2% bytes");
    for (const std::size_t rows : row_counts) {
      std::vector<std::uint8_t> original(rows * ggml_row, 0U);
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < blocks; ++b) {
          const std::size_t at = (r * ggml_row) + (b * kGgmlBlockBytes);
          const std::uint32_t mode =
              static_cast<std::uint32_t>((r + b) % kModes);
          FillBlock(original.data() + at, &rng, mode);
        }
      }
      std::vector<std::uint8_t> packed(original.size(), 0U);
      std::vector<std::uint8_t> restored(original.size(), 0U);
      RepackTensor(original.data(), packed.data(), rows, k);
      UnpackTensor(packed.data(), restored.data(), rows, k);
      const std::size_t bytes = original.size();
      const bool same =
          std::memcmp(original.data(), restored.data(), bytes) == 0;
      Check(same, "unpack(repack(x)) == x");

      std::vector<float> expected(kQk, 0.0F);
      std::vector<float> actual(kQk, 0.0F);
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < blocks; ++b) {
          const std::uint8_t* const ggml_block =
              original.data() + (r * ggml_row) + (b * kGgmlBlockBytes);
          const std::uint8_t* const sn_block =
              packed.data() + (r * sn_row) + (b * kBlockBytes);
          DequantGgml(ggml_block, expected.data());
          DequantSn(sn_block, actual.data());
          for (std::size_t i = 0; i < kQk; ++i) {
            Check(SameBits(expected[i], actual[i]),
                  "scalar dequant differs between ggml and SN layouts");
          }
          CheckScaleGroups(ggml_block, sn_block);
        }
      }
    }
  }

  std::printf(
      "PASS: Q4_K/Q5_K/Q6_K/IQ4_XS interleave; SN Q4_K repack and dequant\n");
  return 0;
}

#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SN_Q4K_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SN_Q4K_HPP_

// clang-format off
// Strix-native (SN) Q4_K: Q4_K for gfx1151, written from zero against
// the GGUF block, with a host-side repack of the load-time weights.
//
// ggml block_q4_K is 144 B per 256 weights: {half d; half dmin; uint8
// scales[12]; uint8 qs[128]}. A stored weight is
//
//   w(i) = (d * sc_j) * q_i - (dmin * mn_j),   j = i / 32,
//
// where q_i is a nibble of qs in natural element order and (sc_j, mn_j) are
// the six-bit fields recovered by get_scale_min_k4(j, scales): for j < 4
// (scales[j] & 63, scales[j+4] & 63), and for j >= 4 the low nibbles of
// scales[j+4] spliced with the top two bits of scales[j-4] (sc) and scales[j]
// (mn). Decoding one sub-block therefore costs a branch and five to seven
// VALU ops. The 128 qs bytes already pair sub-blocks per 32-byte chunk: a lane
// that owns a 64-weight pair reads two aligned qwords and gets both nibbles
// of both sub-blocks.
//
// The SN layout keeps qs byte for byte and rewrites only the 16-byte header.
// The 12 scale bytes become four 24-bit groups, one per 64-weight pair:
//
//   group g = sc_{2g} | mn_{2g} << 6 | sc_{2g+1} << 12 | mn_{2g+1} << 18,
//
// followed by d and dmin verbatim, so a lane extracts each of its four values
// with one shift and one AND. The block stays exactly 144 B per 256 weights
// (+0.0%), and UnpackBlock reconstructs the ggml scale bytes bit for bit.
//
// No row interleave: 144-byte rows are already 16-byte aligned and a lane's
// RowsPerLane adjacent rows are adjacent in memory, so an interleave would
// only add a tail case. Every row count round-trips.
// clang-format on

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace gufo::hip::sn_q4k {

// Elements and stored bytes per Q4_K super-block, in both layouts.
inline constexpr std::uint32_t kQk = 256;
inline constexpr std::uint32_t kGgmlBlockBytes = 144;
inline constexpr std::uint32_t kBlockBytes = 144;
inline constexpr std::uint32_t kQsBytes = 128;
inline constexpr std::uint32_t kScalesBytes = 12;
inline constexpr std::uint32_t kScalesGroups = 4;  // 64-weight pairs

// Repacked Q4_K super-block: ggml qs, group-packed scales, d and dmin bits.
struct alignas(16) SnQ4KBlock {
  std::uint8_t qs[kQsBytes];
  std::uint8_t scales[kScalesBytes];
  std::uint16_t d;
  std::uint16_t dmin;
};
static_assert(sizeof(SnQ4KBlock) == kBlockBytes, "SN block must be 144B");
static_assert(offsetof(SnQ4KBlock, qs) == 0, "qs");
static_assert(offsetof(SnQ4KBlock, scales) == kQsBytes, "scales");
static_assert(offsetof(SnQ4KBlock, d) == kQsBytes + kScalesBytes, "d");
static_assert(offsetof(SnQ4KBlock, dmin) == kQsBytes + kScalesBytes + 2,
              "dmin");

// ggml get_scale_min_k4: the six-bit (scale, min) pair of sub-block j.
inline void GetScaleMinK4(std::uint32_t j, const std::uint8_t* scales,
                          std::uint32_t* sc, std::uint32_t* mn) {
  if (j < 4U) {
    *sc = scales[j] & 63U;
    *mn = scales[j + 4U] & 63U;
  } else {
    *sc = (scales[j + 4U] & 0x0FU) | ((scales[j - 4U] >> 6U) << 4U);
    *mn = (scales[j + 4U] >> 4U) | ((scales[j] >> 6U) << 4U);
  }
}

// One ggml block -> one SN block; dst is the SnQ4KBlock layout above.
inline void RepackBlock(const std::uint8_t* src, std::uint8_t* dst) {
  const std::uint8_t* const scales = src + 4U;
  std::uint32_t group[kScalesGroups] = {};
  for (std::uint32_t j = 0; j < 8U; ++j) {
    std::uint32_t sc = 0;
    std::uint32_t mn = 0;
    GetScaleMinK4(j, scales, &sc, &mn);
    const std::uint32_t shift = (j & 1U) != 0U ? 12U : 0U;
    group[j >> 1U] |= (sc << shift) | (mn << (shift + 6U));
  }
  std::uint8_t* const header = dst + kQsBytes;
  for (std::uint32_t g = 0; g < kScalesGroups; ++g) {
    std::uint8_t* const out = header + (3U * g);
    out[0] = static_cast<std::uint8_t>(group[g] & 0xFFU);
    out[1] = static_cast<std::uint8_t>((group[g] >> 8U) & 0xFFU);
    out[2] = static_cast<std::uint8_t>((group[g] >> 16U) & 0xFFU);
  }
  std::memcpy(header + kScalesBytes, src, 4);  // d and dmin, verbatim
  std::memcpy(dst, src + 16U, kQsBytes);       // qs, unchanged
}

// One SN block -> one ggml block, the exact inverse of RepackBlock.
inline void UnpackBlock(const std::uint8_t* src, std::uint8_t* dst) {
  const std::uint8_t* const header = src + kQsBytes;
  std::uint32_t group[kScalesGroups];
  for (std::uint32_t g = 0; g < kScalesGroups; ++g) {
    group[g] = static_cast<std::uint32_t>(header[3U * g]) |
               (static_cast<std::uint32_t>(header[(3U * g) + 1U]) << 8U) |
               (static_cast<std::uint32_t>(header[(3U * g) + 2U]) << 16U);
  }
  std::uint32_t sc[8];
  std::uint32_t mn[8];
  for (std::uint32_t g = 0; g < kScalesGroups; ++g) {
    sc[2U * g] = group[g] & 63U;
    mn[2U * g] = (group[g] >> 6U) & 63U;
    sc[(2U * g) + 1U] = (group[g] >> 12U) & 63U;
    mn[(2U * g) + 1U] = (group[g] >> 18U) & 63U;
  }
  std::uint8_t scales[kScalesBytes] = {};
  for (std::uint32_t j = 0; j < 4U; ++j) {
    scales[j] = static_cast<std::uint8_t>(sc[j]);
    scales[j + 4U] = static_cast<std::uint8_t>(mn[j]);
  }
  for (std::uint32_t j = 4U; j < 8U; ++j) {
    const std::uint32_t sc_old = scales[j - 4U];
    const std::uint32_t sc_high = sc[j] >> 4U;
    const std::uint32_t mn_high = mn[j] >> 4U;
    scales[j - 4U] = static_cast<std::uint8_t>(sc_old | (sc_high << 6U));
    scales[j] = static_cast<std::uint8_t>(scales[j] | (mn_high << 6U));
    const std::uint32_t packed = (sc[j] & 15U) | ((mn[j] & 15U) << 4U);
    scales[j + 4U] = static_cast<std::uint8_t>(packed);
  }
  std::memcpy(dst, header + kScalesBytes, 4);  // d and dmin at src + 140
  std::memcpy(dst + 4U, scales, kScalesBytes);
  std::memcpy(dst + 16U, src, kQsBytes);
}

// Bytes of one repacked row; the SN layout is exactly the 144 B/256 size.
// Precondition: k is a multiple of 256. Q4_K has no partial super-block; the
// alternative is silently dropping k % 256 weights from the row size.
inline std::size_t RepackedRowBytes(std::size_t k) {
  assert((k % kQk) == 0U);
  return (k / kQk) * kBlockBytes;
}

inline void RepackRow(const std::uint8_t* src, std::uint8_t* dst,
                      std::size_t k) {
  const std::size_t blocks = k / kQk;
  for (std::size_t b = 0; b < blocks; ++b) {
    RepackBlock(src + (b * kGgmlBlockBytes), dst + (b * kBlockBytes));
  }
}

inline void UnpackRow(const std::uint8_t* src, std::uint8_t* dst,
                      std::size_t k) {
  const std::size_t blocks = k / kQk;
  for (std::size_t b = 0; b < blocks; ++b) {
    UnpackBlock(src + (b * kBlockBytes), dst + (b * kGgmlBlockBytes));
  }
}

// Repacks `rows` rows of `k` Q4_K weights. src and dst must not overlap.
inline void RepackTensor(const void* src, void* dst, std::size_t rows,
                         std::size_t k) {
  const auto* const s = static_cast<const std::uint8_t*>(src);
  auto* const d = static_cast<std::uint8_t*>(dst);
  const std::size_t row_bytes = RepackedRowBytes(k);
  for (std::size_t r = 0; r < rows; ++r) {
    RepackRow(s + (r * row_bytes), d + (r * row_bytes), k);
  }
}

inline void UnpackTensor(const void* src, void* dst, std::size_t rows,
                         std::size_t k) {
  const auto* const s = static_cast<const std::uint8_t*>(src);
  auto* const d = static_cast<std::uint8_t*>(dst);
  const std::size_t row_bytes = RepackedRowBytes(k);
  for (std::size_t r = 0; r < rows; ++r) {
    UnpackRow(s + (r * row_bytes), d + (r * row_bytes), k);
  }
}

}  // namespace gufo::hip::sn_q4k

// GUFO_SN_Q4K_HOST_ONLY excludes the HIP half from CPU translation units.
#if defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_HOST_ONLY)

#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"

namespace gufo::hip {

// 24-bit scale/min group g of the SN header (d | dmin << 16).
__device__ __forceinline__ std::uint32_t SnQ4KGroupWord(
    const uint4& meta, std::uint32_t group) {
  if (group == 0U)
    return meta.x;
  if (group == 1U)
    return (meta.x >> 24U) | (meta.y << 8U);
  if (group == 2U)
    return (meta.y >> 16U) | (meta.z << 16U);
  return meta.z >> 8U;
}

// clang-format off
// Q4_K GEMV / skinny GEMM on the SN layout, Batch 1..8, q8_K activations
// (QuantizeActivationsQ8KDot4Kernel's xq/xds).
//
// A wave owns RowsPerWave output rows; its 32/RowsPerWave lanes stride one
// row's super-blocks, so a lane visits every kLanesPerRow-th super-block and
// walks the four 64-weight pairs of each. A pair's two 16-byte qs chunks
// decode once into low- and high-nibble qwords, which are dotted against
// every token and every one of the RowsPerLane adjacent rows, so decode and
// activation reads amortize over Batch * RowsPerLane dots.
//
// Instructions per 32 weights per token, one lane, steady state:
//   Batch 8: 8 v_dot4_i32_iu8 (the dot itself)
//            + 2 int (isum += sc_j * dot_j, one __mul24+add per sub-block)
//            + ~2 decode/scale (96 decode + 32 scale VALU per 256 weights,
//              amortized over the 8 tokens)
//            + ~3.9 min/flush (31 VALU per (row, token, super-block) over
//              the 8 sub-blocks: A_j add, mn_j*A_j, one float flush)
//            = ~15.9 VALU, against ~23 for the V3 ActK half-block decode,
//              which pays get_scale_min_k4 per 16 weights and an isum update
//              per 16 weights.
//   Batch 1: the same body is ~30 VALU per 32 weights per token, of which 8
//            are the dots; B1 is DRAM-bound and has 4x headroom.
//
// Integer bounds per 256-weight super-block and one token: |dot_j| <=
// 32 * 15 * 127 = 60960 < 2^16, so |sc_j * dot_j| < 2^22 and |isum| < 2^25;
// |A_j| <= 32 * 127 = 4064, so |mn_j * A_j| < 2^18 and |imin| < 2^21. Both
// accumulators stay in int32 and every __mul24 operand fits 24 signed bits.
// The float flush xd * (d * isum - dmin * imin) cannot overflow fp32 for any
// fp16 d/dmin and those integer bounds.
//
// No LDS and no barriers: qx/xds come from the read-only caches, and
// RowsPerLane=2 shares each activation read across two rows. RowsPerLane
// stops at 2 because sums and acc already hold 2*Rows*Batch int/float
// registers before decode state.
// clang-format on
template<std::size_t Batch, std::size_t Rows, std::uint32_t RowsPerWave,
         std::uint32_t Waves>
__launch_bounds__(Waves * 32) __global__
    void SnQ4KSmallBatchKernel(const void* __restrict__ w,
                               const std::int8_t* __restrict__ xq,
                               const float2* __restrict__ xds,
                               float* __restrict__ y, std::uint32_t m,
                               std::uint32_t k) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(Rows == 1 || Rows == 2);
  static_assert(RowsPerWave >= 2 && RowsPerWave <= 16 &&
                (32U % RowsPerWave) == 0U);
  constexpr std::uint32_t kLanesPerRow = 32U / RowsPerWave;
  constexpr std::uint32_t kNibbleMask = 0x0F0F0F0FU;

  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t unit = lane % kLanesPerRow;
  const std::uint32_t wave = (blockIdx.x * Waves) + (threadIdx.x >> 5U);
  const std::uint32_t row_in_wave = lane / kLanesPerRow;
  const std::uint32_t row0 =
      ((wave * RowsPerWave) + row_in_wave) * static_cast<std::uint32_t>(Rows);
  if (row0 >= m)
    return;

  const std::uint32_t num_sb = k >> 8U;
  const std::size_t row_bytes =
      static_cast<std::size_t>(num_sb) * sn_q4k::kBlockBytes;
  const std::uint8_t* rowq[Rows];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    const std::uint32_t row = row0 + r;
    const std::size_t safe = static_cast<std::size_t>(row < m ? row : m - 1U);
    rowq[r] = static_cast<const std::uint8_t*>(w) + (safe * row_bytes);
  }

  float sums[Rows][Batch] = {};
  int acc[Rows][Batch] = {};

  for (std::uint32_t sb = unit; sb < num_sb; sb += kLanesPerRow) {
    const std::size_t block_off =
        static_cast<std::size_t>(sb) * sn_q4k::kBlockBytes;
    uint4 meta[Rows];
    float d[Rows];
    float dm[Rows];
    int mn8[Rows][8];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      const std::uint8_t* const bp = rowq[r] + block_off;
      __builtin_memcpy(&meta[r], bp + sn_q4k::kQsBytes, 16);
      const std::uint16_t d_bits = static_cast<std::uint16_t>(meta[r].w);
      const std::uint16_t m_bits =
          static_cast<std::uint16_t>(meta[r].w >> 16U);
      d[r] = __half2float(__ushort_as_half(d_bits));
      dm[r] = __half2float(__ushort_as_half(m_bits));
#pragma unroll
      for (std::uint32_t j = 0; j < 8U; ++j) {
        const std::uint32_t shift = (j & 1U) != 0U ? 18U : 6U;
        const std::uint32_t word = SnQ4KGroupWord(meta[r], j >> 1U);
        mn8[r][j] = static_cast<int>((word >> shift) & 63U);
      }
    }
#pragma unroll
    for (std::uint32_t gp = 0; gp < sn_q4k::kScalesGroups; ++gp) {
#pragma unroll
      for (std::uint32_t jj = 0; jj < 2U; ++jj) {
        int sc[Rows];
#pragma unroll
        for (std::uint32_t r = 0; r < Rows; ++r) {
          const std::uint32_t shift = jj == 0U ? 0U : 12U;
          const std::uint32_t word = SnQ4KGroupWord(meta[r], gp);
          sc[r] = static_cast<int>((word >> shift) & 63U);
        }
        std::uint32_t p0[Rows][4];
        std::uint32_t p1[Rows][4];
#pragma unroll
        for (std::uint32_t r = 0; r < Rows; ++r) {
          uint4 c0;
          uint4 c1;
          __builtin_memcpy(&c0, rowq[r] + block_off + (32U * gp), 16);
          __builtin_memcpy(&c1, rowq[r] + block_off + (32U * gp) + 16U, 16);
          const std::uint32_t w0[4] = {c0.x, c0.y, c0.z, c0.w};
          const std::uint32_t w1[4] = {c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (std::uint32_t i = 0; i < 4U; ++i) {
            if (jj == 0U) {
              p0[r][i] = w0[i] & kNibbleMask;
              p1[r][i] = w1[i] & kNibbleMask;
            } else {
              p0[r][i] = (w0[i] >> 4U) & kNibbleMask;
              p1[r][i] = (w1[i] >> 4U) & kNibbleMask;
            }
          }
        }
        // Sub-block 2gp: low nibbles -> slots 4gp/4gp+1; odd: high -> +2/+3.
        const std::uint32_t slot = (sb * 16U) + (4U * gp) + (2U * jj);
#pragma unroll
        for (std::uint32_t t = 0; t < Batch; ++t) {
          const std::int8_t* const xrow =
              xq + (static_cast<std::size_t>(t) * k);
          uint4 qa;
          uint4 qb;
          __builtin_memcpy(&qa, xrow + (16U * slot), 16);
          __builtin_memcpy(&qb, xrow + (16U * slot) + 16U, 16);
#pragma unroll
          for (std::uint32_t r = 0; r < Rows; ++r) {
            int dot = 0;
            dot = __builtin_amdgcn_sudot4(true, p0[r][0], true, qa.x, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, p0[r][1], true, qa.y, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, p0[r][2], true, qa.z, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, p0[r][3], true, qa.w, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, p1[r][0], true, qb.x, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, p1[r][1], true, qb.y, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, p1[r][2], true, qb.z, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, p1[r][3], true, qb.w, dot,
                                          false);
            acc[r][t] += __mul24(sc[r], dot);
          }
        }
      }
    }
    // Per super-block: min term from the activation sums, then one flush.
#pragma unroll
    for (std::uint32_t t = 0; t < Batch; ++t) {
      const std::size_t xbase = static_cast<std::size_t>(t) * (k >> 4U);
      const std::size_t xoff = xbase + (sb * 16U);
      const auto* const pairs = reinterpret_cast<const uint4*>(xds + xoff);
      int a[8];
#pragma unroll
      for (std::uint32_t j = 0; j < 8U; ++j) {
        const uint4 pair = pairs[j];
        a[j] = static_cast<int>(pair.y) + static_cast<int>(pair.w);
      }
      const float xd = xds[xoff].x;
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        int imin = 0;
#pragma unroll
        for (std::uint32_t j = 0; j < 8U; ++j) {
          imin += __mul24(mn8[r][j], a[j]);
        }
        const float dot_f = static_cast<float>(acc[r][t]);
        const float min_f = static_cast<float>(imin);
        sums[r][t] += xd * ((d[r] * dot_f) - (dm[r] * min_f));
        acc[r][t] = 0;
      }
    }
  }

  // Every lane of the row group holds the full total after the butterfly.
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
#pragma unroll
    for (std::uint32_t t = 0; t < Batch; ++t) {
#pragma unroll
      for (std::uint32_t off = kLanesPerRow >> 1U; off > 0U; off >>= 1U) {
        sums[r][t] += XorLane(sums[r][t], static_cast<int>(off));
      }
    }
  }
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    if (row0 + r >= m)
      continue;
#pragma unroll
    for (std::uint32_t t = 0; t < Batch; ++t) {
      if ((t % kLanesPerRow) == unit) {
        y[(static_cast<std::size_t>(t) * m) + row0 + r] = sums[r][t];
      }
    }
  }
}

// Quantize activations to q8_K, then launch; k must be a multiple of 256.
template<std::size_t Batch, std::size_t Rows, std::uint32_t RowsPerWave,
         std::uint32_t Waves>
void LaunchSnQ4K(const void* w, const float* x, float* y, std::int8_t* xq,
                 float2* xds, std::uint32_t m, std::uint32_t k,
                 hipStream_t stream) {
  const std::uint32_t subs = static_cast<std::uint32_t>(Batch) * (k / 16U);
  QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
      x, xq, xds, subs);
  constexpr std::uint32_t kRowsPerBlock =
      Waves * RowsPerWave * static_cast<std::uint32_t>(Rows);
  SnQ4KSmallBatchKernel<Batch, Rows, RowsPerWave, Waves>
      <<<(m + kRowsPerBlock - 1U) / kRowsPerBlock, Waves * 32, 0, stream>>>(
          w, xq, xds, y, m, k);
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_HOST_ONLY)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SN_Q4K_HPP_

#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_HIP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_HIP_

// Names containing "Opus" (kernels, bench arms) are historical: that width-8
// variant was first written by an AI model (Claude Opus) during development.
//
// Strix-native IQ3_S layout, variant B: IQ3_S GEMV/skinny-GEMM on the
// row-interleaved layout of sn_iq3.hpp.
//
// BYTE MAP (n = rows in the group, B = k/256 block records; each block record
// is round_up(110*n, 16) bytes so every phase base stays 16-byte aligned):
//
//   offset      bytes  contents
//   0           64n    qs:     4 chunks; chunk c holds rows 0..n-1 x 16 bytes
//   64n         32n    signs:  2 chunks; chunk c holds rows 0..n-1 x 16 bytes
//   96n          8n    qh:     row r x 8 bytes (byte s = qh of sub-block s)
//   104n         4n    scales: row r x 4 bytes (nibbles 2s, 2s+1 = the scale of
//                              sub-blocks 2s, 2s+1)
//   108n         2n    d:      row r x fp16
//   round16(110n)     next block of the same group (B block records; the pad
//                     after 110n is never read)
//
// Lane r of the group reads, inside block record b (stride round16(110n)):
//   qs chunk c    at  b*stride + c*(16n) + r*16         (16-byte aligned)
//   signs chunk c at  b*stride + 64n + c*(16n) + r*16   (16-byte aligned)
//   qh            at  b*stride + 96n + r*8              (8-byte aligned)
//   scales        at  b*stride + 104n + r*4             (4-byte aligned)
//   d             at  b*stride + 108n + r*2             (2-byte aligned)
//
// GEOMETRY - one lane owns one output row, a wave reads whole cache lines.
// A wave of 32 lanes covers 32 consecutive output rows (Split == 1), i.e. four
// interleave groups. Lane r of a group reads its own row's 16-byte chunk from
// each plane, so the 8 lanes of a group cover exactly 128 bytes = one line, and
// the four groups of a wave cover four lines per load instruction. The old
// V3 IQ3_S path reads a 16-byte chunk per lane from 32 *different* lines for
// the same instruction, because block_iq3_s rows are 110 bytes apart and only
// 2-byte aligned. Everything a lane loads is 16/8/4/2-byte aligned by
// construction (sn_iq3.hpp), so there is no memcpy-shaped load assembly left.
//
// Weight decode is hoisted out of the token loop: one row's 32-weight
// sub-block is decoded once and dotted against all Batch activation rows.
//
// SIGN-FOLDED LDS CODEBOOK (default arm, Folded == true).
// w = d * (1 + 2*scale_nibble) * grid[idx9][j] * sign[j], idx9 = qs byte |
// (qh bit << 8), grid values 1..15 (odd), j = 0..3 inside one grid word and
// sign[j] = bit j of the group's sign nibble. The 512-entry grid x 16 sign
// nibbles = 8192 u32 = 32 KiB is staged into LDS, so ONE `ds_read_b32` (4 B)
// returns four *signed* int8 weights and the whole sign path disappears: the
// index is built as
//   idx = qs | ((qhbit | (sign_nibble << 1)) << 8)
// because grid_index + 512*sign = qs + 256*(qhbit + 2*sign_nibble).
// The comparison arm (Folded == false) stages the plain 512-entry grid and
// applies NegateSelectedBytes (6 VALU) to every word instead.
//
// ACTIVATIONS - q8_K style (QuantizeActivationsQ8KDot4Kernel): int8 values
// with ONE scale per 256 elements. Every lane of a wave needs the same 32
// activation bytes per (token, sub-block), so the load is a uniform-address
// broadcast (one cache line) and no LDS staging or barrier is needed. With one
// activation scale per 256 weights the whole super-block can be accumulated in
// int32 and flushed once:
//   isum[t] += (1 + 2*scale_nibble_s) * dot32(s, t)   (int32, s = 0..7)
//   sum[t] += d * xd[t] * (float)isum[t]              (once per 256 weights)
//
// INT32 BOUND (same style as the IQ4_XS ActK comment in small_batch_dot4.hpp):
// |grid| <= 15 (every iq3s grid byte is odd, 1..15), |int8 activation| <= 127,
// so |one v_dot4| <= 4*15*127 = 7620, |dot32 of 8 words| <= 60960 < 2^16,
// |1 + 2*nibble| <= 31, and over the 8 sub-blocks of one super-block
// |isum| <= 8*31*60960 = 15,118,080 < 2^24. Both v_mad_i32_i24 operands
// (31 and 60960 < 2^23) fit 24 bits, and isum is flushed to float every
// super-block, so nothing is ever accumulated towards int32 overflow.
//
// INSTRUCTION COUNT per 32 weights (one sub-block = 8 grid words) per row, from
// the ISA (rdna35_instruction_set_architecture.md):
//   decode, once per row (amortized over the Batch tokens):
//     8 x { V_BFE_U32 qs byte, V_BFE_U32 qh bit, V_BFE_U32 sign nibble,
//           V_LSHL_OR_B32 tag = qhbit | sign<<1,
//           V_LSHL_OR_B32 idx = qs | tag<<8,
//           V_LSHL_ADD_U32 vaddr = idx*4 }                   = 48 VALU
//     8 x ds_read_b32 (4 signed int8 weights each)           = 8 LDS
//     + 9/8 global loads, + ~1.5 address/loop, + 1 scale BFE
//   per token:
//     8 x V_DOT4_I32_IU8 (two independent 4-deep chains)      = 8 VALU
//     V_ADD_U32 chain join, V_MAD_I32_I24 scale fold          = 2 VALU
//     2 x global_load_dwordx4 (32 activation bytes, broadcast) = 2 LDG
//   => 10 VALU + 2 LDG per 32 weights per token.
// Totals per 32 weights (13.75 weight bytes): ~61 VALU + 8 LDS + ~3 LDG at
// Batch 1, ~131 VALU + 8 LDS + ~17 LDG at Batch 8. The V3 IQ3_S helper costs
// u=124 + d=14 per 32 weights per token (the CEILING_compute.md count), so this
// is ~1.9x fewer issues at B1 and ~2.4x at B8; at 1 issue/cycle/SIMD that is
// about 830 GB/s instead of 400 at Batch 1 and 384 instead of 237 at Batch 8,
// i.e. the DRAM roof (256 GB/s) stays the binding limit at every Batch.
//
// Split (geometry arm, not a different layout): 32/Split lanes own one row
// each, the other lanes of a wave take the remaining blocks of K (block b
// belongs to phase b % Split). That multiplies the number of waves - and hence
// the number of busy SIMDs - without changing the layout or the inner loop:
// the 32 KiB table caps the block at one per CU, so with 16-wave blocks the
// brief's Split 1 geometry needs m >= 20480 rows to fill 40 CUs. The per-phase
// partial sums are combined with XorLane.

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "src/models/qwen/hip/dev_sync.hpp"
#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3_cut.hpp"
#include "src/models/qwen/hip/kernels/xor_lane.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

namespace snb_detail {

// Stage the IQ3_S codebook into LDS: 512 u32 (plain grid) or 8192 u32
// (grid x 16 sign nibbles). Every entry of a group is 1..15, so
// NegateSelectedBytes' 0x01010100 - v trick is valid for all of them.
template<bool Folded>
__device__ __forceinline__ void SnbStageCodebook(std::uint32_t* table,
                                                 std::uint32_t tid,
                                                 std::uint32_t threads) {
  if constexpr (Folded) {
    for (std::uint32_t i = tid; i < 512U; i += threads) {
      const std::uint32_t grid = kDeviceIq3sGrid[i];
#pragma unroll
      for (std::uint32_t sign = 0; sign < 16U; ++sign)
        table[(sign << 9U) | i] = NegateSelectedBytes(grid, sign);
    }
  } else {
    for (std::uint32_t i = tid; i < 512U; i += threads)
      table[i] = kDeviceIq3sGrid[i];
  }
}

// b8cut codebook: the same 8192 signed grid words as SnbStageCodebook<true>,
// at the slot cut::TableIndex<Idx> assigns (Idx 2: a 16384-word table, half of
// it unused). Idx 0 is SnbStageCodebook<true> exactly.
template<std::uint32_t Idx>
__device__ __forceinline__ void SnbStageCutCodebook(std::uint32_t* table,
                                                    std::uint32_t tid,
                                                    std::uint32_t threads) {
  for (std::uint32_t i = tid; i < 512U; i += threads) {
    const std::uint32_t grid = kDeviceIq3sGrid[i];
#pragma unroll
    for (std::uint32_t sign = 0; sign < 16U; ++sign)
      table[cut::TableIndex<Idx>(i & 255U, i >> 8U, sign)] =
          NegateSelectedBytes(grid, sign);
  }
}

}  // namespace snb_detail

// y[token][row] = sum_k W[row][k] * x[token][k], with W in the sn_iq3.hpp
// layout and x already quantized to int8 with one scale per 256 (xds).
// One lane owns one output row; the 32/Split lanes of a wave that own the same
// row take different blocks of K.
template<std::size_t Batch, std::uint32_t Waves, std::uint32_t Split,
         bool Folded>
__launch_bounds__(Waves * 32U) __global__
    void SnbIq3sGemmKernel(const std::uint8_t* __restrict__ w,
                           const std::int8_t* __restrict__ xq,
                           const float2* __restrict__ xds,
                           float* __restrict__ y, std::uint32_t m,
                           std::uint32_t k) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(Split == 1U || Split == 2U || Split == 4U);
  static_assert(Waves >= 1U && Waves <= 32U);
  __shared__ std::uint32_t codebook[Folded ? 8192 : 512];
  snb_detail::SnbStageCodebook<Folded>(codebook, threadIdx.x, Waves * 32U);
  __syncthreads();

  constexpr std::uint32_t kRowsPerWave = 32U / Split;
  constexpr std::uint32_t kRowsPerBlock = Waves * kRowsPerWave;
  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
  const std::uint32_t split = lane / kRowsPerWave;
  // Rows past m-1 are clamped so their loads stay inside the last group's
  // record; their outputs are simply not stored.
  const std::uint32_t row0 = (blockIdx.x * kRowsPerBlock) +
                             (wave * kRowsPerWave) + (lane % kRowsPerWave);
  const std::uint32_t row = row0 < m ? row0 : m - 1U;
  const std::uint32_t group = row / kSnIq3RowsPerGroup;
  const std::uint32_t in_group = row % kSnIq3RowsPerGroup;
  const std::uint32_t rows_left = m - (group * kSnIq3RowsPerGroup);
  const std::uint32_t n =
      rows_left < kSnIq3RowsPerGroup ? rows_left : kSnIq3RowsPerGroup;
  const std::size_t record = SnIq3BlockStride(n);
  const std::size_t group_base =
      (static_cast<std::size_t>(group) * SnIq3GroupStride(blocks)) +
      (static_cast<std::size_t>(split) * record);

  const std::uint8_t* qs_cur = w + group_base + (in_group * 16U);
  const std::uint8_t* sg_cur =
      w + group_base + (n * kSnIq3SignsOffset) + (in_group * 16U);
  const std::uint8_t* qh_cur =
      w + group_base + (n * kSnIq3QhOffset) + (in_group * 8U);
  const std::uint8_t* sc_cur =
      w + group_base + (n * kSnIq3ScalesOffset) + (in_group * 4U);
  const std::uint8_t* d_cur =
      w + group_base + (n * kSnIq3DOffset) + (in_group * 2U);
  const std::uint32_t chunk_bytes = 16U * n;
  const std::uint32_t step = Split * static_cast<std::uint32_t>(record);
  const std::uint32_t num_sub = k / 16U;

  float sums[Batch] = {};
  for (std::uint32_t b = split; b < blocks; b += Split) {
    const uint4 q0 = *reinterpret_cast<const uint4*>(qs_cur);
    const uint4 q1 = *reinterpret_cast<const uint4*>(qs_cur + chunk_bytes);
    const uint4 q2 = *reinterpret_cast<const uint4*>(qs_cur + 2U * chunk_bytes);
    const uint4 q3 = *reinterpret_cast<const uint4*>(qs_cur + 3U * chunk_bytes);
    qs_cur += step;
    const uint4 s0 = *reinterpret_cast<const uint4*>(sg_cur);
    const uint4 s1 = *reinterpret_cast<const uint4*>(sg_cur + chunk_bytes);
    sg_cur += step;
    std::uint64_t qh;
    __builtin_memcpy(&qh, qh_cur, 8);
    qh_cur += step;
    std::uint32_t sc;
    __builtin_memcpy(&sc, sc_cur, 4);
    sc_cur += step;
    std::uint16_t d_bits;
    __builtin_memcpy(&d_bits, d_cur, 2);
    d_cur += step;
    const float d = __half2float(__ushort_as_half(d_bits));

    // 16 qs words = 8 sub-blocks x 8 index bytes, 8 sign words = 8 sub-blocks
    // x 8 sign nibbles. Both flatten to registers: every index below is a
    // compile-time constant once the loops are unrolled.
    const std::uint32_t qs_words[16] = {
        q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w,
        q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
    const std::uint32_t sign_words[8] = {s0.x, s0.y, s0.z, s0.w,
                                         s1.x, s1.y, s1.z, s1.w};
    int isum[Batch] = {};
#pragma unroll
    for (std::uint32_t s = 0; s < 8U; ++s) {
      const std::uint32_t qh_byte =
          static_cast<std::uint32_t>(qh >> (8U * s)) & 0xFFU;
      const std::uint32_t signs = sign_words[s];
      // scale_nibble_s = nibble s of the row's 4 packed scale bytes.
      const std::uint32_t scale_int = 1U + (2U * ((sc >> (4U * s)) & 0x0FU));
      std::uint32_t word[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8U; ++i) {
        const std::uint32_t qs_byte =
            (qs_words[(2U * s) + (i >> 2U)] >> (8U * (i & 3U))) & 0xFFU;
        const std::uint32_t qh_bit = (qh_byte >> i) & 1U;
        const std::uint32_t sign_nibble = (signs >> (4U * i)) & 0x0FU;
        if constexpr (Folded) {
          word[i] = codebook[qs_byte | ((qh_bit | (sign_nibble << 1U)) << 8U)];
        } else {
          word[i] = NegateSelectedBytes(codebook[qs_byte | (qh_bit << 8U)],
                                        sign_nibble);
        }
      }
      const std::uint32_t act_base =
          (b * static_cast<std::uint32_t>(kSnIq3Elements)) + (s * 32U);
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        const std::int8_t* const act = xq + (t * k) + act_base;
        const uint4 a0 = *reinterpret_cast<const uint4*>(act);
        const uint4 a1 = *reinterpret_cast<const uint4*>(act + 16U);
        int lo = 0;
        int hi = 0;
        lo = __builtin_amdgcn_sudot4(true, word[0], true, a0.x, lo, false);
        lo = __builtin_amdgcn_sudot4(true, word[1], true, a0.y, lo, false);
        lo = __builtin_amdgcn_sudot4(true, word[2], true, a0.z, lo, false);
        lo = __builtin_amdgcn_sudot4(true, word[3], true, a0.w, lo, false);
        hi = __builtin_amdgcn_sudot4(true, word[4], true, a1.x, hi, false);
        hi = __builtin_amdgcn_sudot4(true, word[5], true, a1.y, hi, false);
        hi = __builtin_amdgcn_sudot4(true, word[6], true, a1.z, hi, false);
        hi = __builtin_amdgcn_sudot4(true, word[7], true, a1.w, hi, false);
        isum[t] += __mul24(static_cast<int>(scale_int), lo + hi);
      }
    }
    // One float multiply per (row, token, 256 weights); xd is the q8_K
    // activation scale of this super-block.
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float xd = xds[(t * num_sub) + (b * 16U)].x;
      sums[t] += (d * xd) * static_cast<float>(isum[t]);
    }
  }

  if constexpr (Split > 1U) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
      for (std::uint32_t mask = kRowsPerWave; mask < 32U; mask <<= 1U)
        sums[t] += XorLane(sums[t], static_cast<int>(mask));
  }
  if (split == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
}

// Run the GEMM on activations already quantized by LaunchSnbIq3s (q8_K style:
// one scale per 256). xq holds Batch*k int8, xds Batch*k/16 float2, both
// 16-byte aligned; k must be a multiple of 256.
template<std::size_t Batch, std::uint32_t Waves, std::uint32_t Split,
         bool Folded>
void LaunchSnbIq3sTyped(const void* w, float* y,
                        std::int8_t* xq, float2* xds, std::uint32_t m,
                        std::uint32_t k, hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = Waves * (32U / Split);
  SnbIq3sGemmKernel<Batch, Waves, Split, Folded>
      <<<(m + kRowsPerBlock - 1U) / kRowsPerBlock, Waves * 32U, 0, stream>>>(
          static_cast<const std::uint8_t*>(w), xq, xds, y, m, k);
}

// opus_b8_k4act: QuantizeActivationsQ8KDot4Kernel with the tile
// written block-major / token-interleaved. Thread i = (token t = i / (k/16),
// sub-block sb = i % (k/16)); values, scale and sum are the same as there, only
// the destination moves: int8 to ((blk*Batch + t)*256 + (sb%16)*16) and float2
// {d, sum} to blk*Batch + t, with blk = sb / 16 and Batch = subs / (k/16).
//
// When xh != nullptr the same 16 int8 values are ALSO written
// as 16 f16 (exact: |q| <= 127) to xh[slot*256 + (sb%16)*16 ..], i.e. the same
// block-major layout, 2 bytes per value. The v_dual_dot2acc arms read only xh
// and xds; k4act reads only xq and xds.
static __global__ void QuantizeActivationsQ8KInterleavedKernel(
    const float* __restrict__ x, std::int8_t* __restrict__ xq,
    float2* __restrict__ xds, std::uint32_t subs, std::uint32_t k,
    std::uint16_t* __restrict__ xh = nullptr) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool live = i < subs;  // every lane reaches the shuffles
  float v[16] = {};
  if (live) {
    const float4* src = reinterpret_cast<const float4*>(x) + (i * 4U);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4 f = src[j];
      v[(4 * j) + 0] = f.x;
      v[(4 * j) + 1] = f.y;
      v[(4 * j) + 2] = f.z;
      v[(4 * j) + 3] = f.w;
    }
  }
  float amax = 0.0F;
  int nan = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    amax = fmaxf(amax, fabsf(v[j]));
    nan |= v[j] != v[j] ? 1 : 0;
  }
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    nan |= __shfl_xor(nan, o, 32);
  }
  if (!live)
    return;
  const float d = nan != 0 ? __builtin_nanf("") : amax / 127.0F;
  const float id = amax > 0.0F ? 127.0F / amax : 0.0F;
  std::uint32_t packed[4] = {};
  int sum = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int q =
        static_cast<int>(rintf(fminf(fmaxf(v[j] * id, -127.0F), 127.0F)));
    sum += q;
    packed[j / 4] |= (static_cast<std::uint32_t>(q) & 0xFFU) << (8 * (j % 4));
  }
  const std::uint32_t per_token = k / 16U;  // sub-blocks per token
  const std::uint32_t batch = subs / per_token;
  const std::uint32_t t = i / per_token;
  const std::uint32_t sb = i % per_token;
  const std::uint32_t slot = ((sb / 16U) * batch) + t;  // blk * Batch + t
  reinterpret_cast<uint4*>(xq)[(slot * 16U) + (sb % 16U)] = {
      packed[0], packed[1], packed[2], packed[3]};
  xds[slot] = {d, __int_as_float(sum)};
  if (xh != nullptr) {
    std::uint32_t h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const auto lo = static_cast<std::int8_t>((packed[j / 2] >> (16 * (j % 2))) & 0xFFU);
      const auto hi = static_cast<std::int8_t>((packed[j / 2] >> ((16 * (j % 2)) + 8)) & 0xFFU);
      h[j] = static_cast<std::uint32_t>(__builtin_bit_cast(
                 std::uint16_t, static_cast<_Float16>(static_cast<float>(lo)))) |
             (static_cast<std::uint32_t>(__builtin_bit_cast(
                  std::uint16_t, static_cast<_Float16>(static_cast<float>(hi))))
              << 16U);
    }
    uint4* const dst = reinterpret_cast<uint4*>(xh) + (((slot * 16U) + (sb % 16U)) * 2U);
    dst[0] = {h[0], h[1], h[2], h[3]};
    dst[1] = {h[4], h[5], h[6], h[7]};
  }
}

// B8 "opus" arms: same layout, same sign-folded codebook, same int8
// activations (one q8_K scale per 256) and the same int32 sums as
// SnbIq3sGemmKernel<..., Folded = true>; only the instruction schedule differs.
//
// WHY (gfx1151 ISA of SnbIq3sGemmKernel<8,16,1,true>, -O3): the activations are
// already scalar loads (s_load_b512, the address is wave-uniform), but the
// fully unrolled 8 x 8 (sub-block x token) body wants 512 SGPRs of activations
// at once: 121 SGPR spills into VGPR lanes, 106 v_writelane + 126 v_readlane
// per 256 weights and an lgkmcnt(0) after each s_load. Loop body: 1296 VALU per
// 256 weights per row, of which 512 are v_dot4.
//
// Changes, all exact:
//  1. Activations stay SGPR operands of v_dot4 (no VGPR, no LDS, no VMEM), but
//     each sub-block's 8 tokens x 32 bytes (64 SGPRs) are loaded right before
//     their sub-block and a sched_barrier stops the scheduler from hoisting the
//     next sub-block's loads: no SGPR spill. SMEM returns out of order, so a
//     wave cannot prefetch its own scalar loads (only lgkmcnt(0) is legal);
//     other waves on the SIMD hide that latency.
//  2. Codebook address with V_PERM_B32: per sub-block build two "tag" words
//     whose byte i = qh_bit_i | sign_nibble_i << 1 (12 VALU for 8 words), then
//     each LDS address is perm(tag, qs) << 2 (2 VALU) instead of ~6.
//  3. One 8-deep dot4 chain per token, folded with one v_mad_i32_i24 (the 8
//     token chains are independent, so no ILP is lost).
//  4. The next K-block's 110 weight bytes are loaded one iteration ahead
//     (VMEM returns in order, so this prefetch does work in-wave).
// KPhases > 1 splits K across the waves of a block (wave w takes blocks
// w / RowWaves + j*KPhases) and adds the partials through LDS in a fixed order
// after the main loop (the codebook LDS is reused): more, smaller blocks, so
// m = 6144 fills all 20 WGPs and m = 34816 balances to 272 blocks.
// ISA (gfx1151, -O3, Batch 8, per 256 weights per row = one loop iteration):
// 887 VALU (512 v_dot4, 80 v_perm, 64 mad/mul24, 26 v_dual_mov for the
// prefetch copy), 64 ds_load_b32, 32 s_load_b512 + 8 s_load_b32, 9 VMEM,
// 17 s_waitcnt, 0 SGPR spills; 95 VGPRs, 102 SGPRs, 16 waves/SIMD.
// amdgpu_waves_per_eu(16) holds the VGPRs at <= 96 (97 without it = 12 waves).
// Probe == true (bench arm opus_b8_alu, never served): no weight bytes are
// read. The weight registers are seeded from the row and bumped by one v_add
// per dword each iteration (in place of the prefetch copy), so decode, LDS,
// activation SMEM, dot4 and the y store all still run: it times this kernel's
// compute + activation ceiling. Its outputs are meaningless.
// Probe arms, all on this kernel (k4 = Pad8 = Slope = Clock = false,
// whose code is unchanged):
//  Pad8 (opus_b8_k4pad): n is the constant kSnIq3RowsPerGroup, so record and
//    every plane offset are immediates. Needs m % 8 == 0 (every group whole);
//    the launcher falls back to k4 otherwise. Same math, same order.
//  Slope (opus_b8_k4slope): 128 chained v_and_b32 per K block on a chain
//    seeded from row (asm volatile, so they stay). The chain reaches the output
//    as an int 0 (zero_gate = m >> 31 is 0 for every m < 2^31): output = k4's.
//  Clock (opus_b8_k4clk): per workgroup, clk[4*b+0..3] = steady (100 MHz) counter
//    at entry, same after the y store, shader-cycle counter at entry, same after
//    the store (thread 0 only). gfx1151 has no s_memrealtime/s_memtime: the
//    builtins lower to s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME and
//    s_getreg HW_REG_SHADER_CYCLES (20 bits, wraps: the bench masks the delta).
//    clk must hold 4 * ceil(m / 128) u64.
//  ActI (opus_b8_k4act): the activation tile is block-major / token-interleaved
//    (QuantizeActivationsQ8KInterleavedKernel): xq[(blk*Batch + t)*256 + i%256],
//    scale xds[blk*Batch + t].x. One base per K block plus immediates covers all
//    tokens; same int8 values, scales and K order, so output = k4's bit for bit.
// Cross-WG combine of split-K partials, no second launch. Every
// split WG parks its row-group partials (phase 0's sums, already the in-WG
// ((p0+p1)+p2)+p3) in scratch[(split * Batch + t) * m + row]; the LAST WG of the
// row group to arrive (counter[ticket]) adds the S partials in split
// order 0..S-1 and the caller stores y; the ticket is reset for the next launch.
// Agent-scope fences make the partials visible across CUs; the loads are
// agent-scope atomics so they cannot hit a stale L0 line. `flag` is one LDS word
// outside the reduction scratch. Returns true in the WG that must store y.
template<std::size_t Batch>
__device__ __forceinline__ bool SnbSplitKCombine(
    float (&sums)[Batch], std::uint32_t* flag, float* scratch,
    std::uint32_t* counter, std::uint32_t ticket, std::uint32_t m,
    std::uint32_t row0,
    std::uint32_t split, std::uint32_t splits, bool phase0) {
  if (phase0 && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      scratch[(((split * Batch) + t) * m) + row0] = sums[t];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0U)
    *flag = atomicAdd(&counter[ticket], 1U) == splits - 1U;
  __syncthreads();
  const bool last = *flag != 0U;  // uniform across the workgroup
  if (last) {
    __threadfence();
    if (phase0 && row0 < m) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        float acc = 0.0f;
        for (std::uint32_t s = 0; s < splits; ++s) {
          const float part =
              s == split ? sums[t]
                         : __hip_atomic_load(
                               &scratch[(((s * Batch) + t) * m) + row0],
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          acc = s == 0U ? part : acc + part;
        }
        sums[t] = acc;
      }
    }
    if (threadIdx.x == 0U) counter[ticket] = 0U;
  }
  return last;
}

// End of one work item (row group x K split) of a PERSISTENT
// workgroup. The codebook stays in LDS for the whole launch, so the cross-phase
// reduction ((p0+p1)+p2)+p3 uses its own LDS array `lds` (KPhases-1 phase
// partials + one flag word). Then the split-K combine (S > 1) or a plain store,
// then sums restart at 0 for the next item. Two barriers here keep item j+1's
// partial writes behind item j's reads.
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool GuardRows = false>
__device__ __forceinline__ void SnbPersistEpilogue(
    float (&sums)[Batch], float* lds, float* y, float* scratch,
    std::uint32_t* counter, std::uint32_t m, std::uint32_t group,
    std::uint32_t row0, std::uint32_t row_wave, std::uint32_t lane,
    std::uint32_t phase, std::uint32_t split, std::uint32_t splits) {
  constexpr std::uint32_t kPartial = (KPhases - 1U) * RowWaves * Batch * 32U;
  const auto slot = [&](std::uint32_t p, std::size_t t) {
    return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
  };
  if (phase > 0U) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) lds[slot(phase, t)] = sums[t];
  }
  __syncthreads();
  if (phase == 0U) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
      for (std::uint32_t p = 1U; p < KPhases; ++p) sums[t] += lds[slot(p, t)];
  }
  __syncthreads();
  bool store_y = true;
  if (splits > 1U)
    store_y = SnbSplitKCombine<Batch>(sums, reinterpret_cast<std::uint32_t*>(&lds[kPartial]),
                                      scratch, counter, group, m, row0, split,
                                      splits, phase == 0U);
  if (store_y && phase == 0U && (!GuardRows || row0 < m)) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) y[(t * m) + row0] = sums[t];
  }
#pragma unroll
  for (std::size_t t = 0; t < Batch; ++t) sums[t] = 0.0f;
}

// s_setprio around a wave's next-block load issue (b8prio): Prio 1 -> 3, 2 -> 2.
// Memory clobber keeps the loads inside the pair; the priority only decides who
// the instruction arbiter picks among the waves of a SIMD.
#define GUFO_SETPRIO_ON(P)                                  \
  do {                                                      \
    if constexpr ((P) == 1U)                                \
      asm volatile("s_setprio 3" ::: "memory");             \
    else if constexpr ((P) == 2U)                           \
      asm volatile("s_setprio 2" ::: "memory");             \
  } while (0)
#define GUFO_SETPRIO_OFF(P)                                 \
  do {                                                      \
    if constexpr ((P) != 0U)                                \
      asm volatile("s_setprio 0" ::: "memory");             \
  } while (0)

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool Probe = false, bool Pad8 = false, bool Slope = false,
         bool Clock = false, bool ActI = false, std::uint32_t Idx = 0,
         bool Quad3 = false, bool ActFix = false, bool NoDot = false,
         bool Early = false, std::uint32_t Prio = 0, bool PrioW = false,
         bool SplitK = false, bool Persist = false, bool Balanced = false>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(16))) void OpusIq3sB8Kernel(
        const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k, std::uint64_t* __restrict__ clk, float* scratch,
        std::uint32_t* counter, std::uint32_t ps_splits) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(!Balanced || (Persist && SplitK && Pad8 && RowWaves == 7U &&
                              KPhases == 4U && (Batch == 7 || Batch == 8)));
  static_assert(!Persist || (SplitK && Pad8 && !Probe && !Early && !PrioW &&
                             !Clock && !Slope));
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  // Phase partials for the LDS reduction must fit in the reused codebook.
  static_assert((KPhases - 1U) * RowWaves * Batch * 32U <= 8192U);
  __shared__ std::uint32_t codebook[8192];
  if constexpr (Clock) {
    if (threadIdx.x == 0U) {
      clk[(4U * blockIdx.x)] =
          static_cast<std::uint64_t>(__builtin_readsteadycounter());
      clk[(4U * blockIdx.x) + 2U] =
          static_cast<std::uint64_t>(__builtin_readcyclecounter());
    }
  }
  // Persist: the table is staged ONCE, after the first block's loads are issued
  // (below), so the DRAM round trip overlaps the staging.
  if constexpr (!Persist) {
    if constexpr (Idx == 0U)
      snb_detail::SnbStageCodebook<true>(codebook, threadIdx.x, kThreads);
    else
      snb_detail::SnbStageCutCodebook<Idx>(codebook, threadIdx.x, kThreads);
    __syncthreads();
  }

  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  // SplitK (b8sk): blockIdx.y = split, each split owns `span` contiguous K
  // blocks [b_begin, b_end); the launcher guarantees gridDim.y divides blocks.
  // Without SplitK these fold to 0 / blocks and the code is k4's.
  // Persist (b8ps): gridDim.x resident WGs, WG w always does split w % S of row
  // groups g0 + j * gstep (g0 = w / S, gstep = gridDim.x / S; S divides gridDim.x).
  const std::uint32_t splits = Persist ? ps_splits : (SplitK ? gridDim.y : 1U);
  const std::uint32_t split =
      Persist ? blockIdx.x % splits : (SplitK ? blockIdx.y : 0U);
  const std::uint32_t g0 = Persist ? blockIdx.x / splits : blockIdx.x;
  const std::uint32_t b_begin = SplitK ? split * (blocks / splits) : 0U;
  const std::uint32_t b_end = SplitK ? b_begin + (blocks / splits) : blocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  // readfirstlane: the compiler must see the wave index as uniform, or the
  // activation address (which depends on the phase) stops being an s_load.
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t row0 = (((g0 * RowWaves) + row_wave) * 32U) + lane;
  const std::uint32_t row = row0 < m ? row0 : m - 1U;
  const std::uint32_t group = row / kSnIq3RowsPerGroup;
  const std::uint32_t in_group = row % kSnIq3RowsPerGroup;
  const std::uint32_t rows_left = m - (group * kSnIq3RowsPerGroup);
  // Pad8: m % 8 == 0 (checked by the launcher), so every group is whole.
  const std::uint32_t n =
      Pad8 ? static_cast<std::uint32_t>(kSnIq3RowsPerGroup)
           : (rows_left < kSnIq3RowsPerGroup ? rows_left : kSnIq3RowsPerGroup);
  const std::uint32_t record = static_cast<std::uint32_t>(SnIq3BlockStride(n));
  const std::uint32_t chunk_bytes = 16U * n;
  const std::uint8_t* base =
      w + (static_cast<std::size_t>(group) * SnIq3GroupStride(blocks)) +
      (static_cast<std::size_t>(b_begin + phase) * record);
  const std::uint32_t qs_off = in_group * 16U;
  const std::uint32_t sg_off = (n * kSnIq3SignsOffset) + (in_group * 16U);
  const std::uint32_t qh_off = (n * kSnIq3QhOffset) + (in_group * 8U);
  const std::uint32_t sc_off = (n * kSnIq3ScalesOffset) + (in_group * 4U);
  const std::uint32_t d_off = (n * kSnIq3DOffset) + (in_group * 2U);
  const std::uint32_t step = KPhases * record;
  const std::uint32_t num_sub = k / 16U;

  struct Weights {
    uint4 q[4];
    uint4 s[2];
    uint2 qh;
    std::uint32_t sc;
    std::uint16_t d;
  };
  const auto load = [&](const std::uint8_t* p) {
    Weights v;
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      v.q[c] = *reinterpret_cast<const uint4*>(p + qs_off + (c * chunk_bytes));
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      v.s[c] = *reinterpret_cast<const uint4*>(p + sg_off + (c * chunk_bytes));
    v.qh = *reinterpret_cast<const uint2*>(p + qh_off);
    v.sc = *reinterpret_cast<const std::uint32_t*>(p + sc_off);
    v.d = *reinterpret_cast<const std::uint16_t*>(p + d_off);
    return v;
  };

  // Persist state: this wave's blocks per item (nb), total blocks over all its
  // items (ps_total), the pointer jump from an item's last block to the next
  // item's first block, and the item stride in rows.
  [[maybe_unused]] std::uint32_t ps_total = 0U;
  [[maybe_unused]] std::uint32_t ps_nb = 0U;
  [[maybe_unused]] std::uint32_t ps_gstep = 0U;
  [[maybe_unused]] std::size_t ps_jump = 0U;
  if constexpr (Persist) {
    const std::uint32_t groups_total =
        Balanced ? (m + RowWaves * 32U - 1U) / (RowWaves * 32U)
                 : m / (RowWaves * 32U);
    ps_gstep = gridDim.x / splits;
    ps_nb = ((blocks / splits) - phase + KPhases - 1U) / KPhases;
    ps_total = ((groups_total - g0 + ps_gstep - 1U) / ps_gstep) * ps_nb;
    ps_jump = (static_cast<std::size_t>(ps_gstep) * RowWaves * 4U *
               SnIq3GroupStride(blocks)) -
              (static_cast<std::size_t>(ps_nb - 1U) * step);
  }
  float sums[Batch] = {};
  Weights next{};
  const auto bump = [](uint4& v, std::uint32_t c) {
    v.x += c;
    v.y += c;
    v.z += c;
    v.w += c;
  };
  if constexpr (Probe) {
    const std::uint32_t seed = row * 0x9E3779B9U;
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      next.q[c] = make_uint4(seed, seed ^ 0x5BD1E995U, seed + c, ~seed);
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      next.s[c] = make_uint4(~seed, seed + c, seed ^ 0x27D4EB2FU, seed >> 3U);
    next.qh = make_uint2(seed ^ 0x165667B1U, seed >> 7U);
    next.sc = seed ^ 0xC2B2AE35U;
    next.d = 0x3C00U;  // fp16 1.0: keeps every output finite
  } else if (b_begin + phase < b_end) {
    next = load(base);
  }
  [[maybe_unused]] float* ps_lds = nullptr;
  if constexpr (Persist) {
    __shared__ float ps_buf[((KPhases - 1U) * RowWaves * Batch * 32U) + 1U];
    ps_lds = ps_buf;
    if constexpr (Idx == 0U)
      snb_detail::SnbStageCodebook<true>(codebook, threadIdx.x, kThreads);
    else
      snb_detail::SnbStageCutCodebook<Idx>(codebook, threadIdx.x, kThreads);
    __syncthreads();
  }
  [[maybe_unused]] std::uint32_t ps_n = 0U;
  [[maybe_unused]] std::uint32_t ps_j = 0U;
  [[maybe_unused]] std::uint32_t chain = row;
  [[maybe_unused]] const std::uint32_t slope_seed = row | 0x80000000U;
  for (std::uint32_t b = b_begin + phase; Persist ? ps_n < ps_total : b < b_end;
       b += Persist ? 0U : KPhases) {
    const Weights cur = next;
    if constexpr (Slope) {
#pragma unroll
      for (std::uint32_t i = 0; i < 128U; ++i)
        asm volatile("v_and_b32 %0, %1, %0" : "+v"(chain) : "v"(slope_seed));
    }
    if constexpr (Probe) {
      // Data-dependent per iteration, so nothing is hoisted out of the loop.
#pragma unroll
      for (std::uint32_t c = 0; c < 4U; ++c)
        bump(next.q[c], 0x9E3779B9U);
#pragma unroll
      for (std::uint32_t c = 0; c < 2U; ++c)
        bump(next.s[c], 0x7F4A7C15U);
      next.qh.x += 0x85EBCA6BU;
      next.qh.y += 0xC2B2AE35U;
      next.sc += 0x27D4EB2FU;
    } else if constexpr (Early) {
      // b8ep: the next block's loads come first, unconditionally (the last
      // block re-reads itself, a cache hit), so no branch / copy sits before
      // them; sched_barrier keeps them ahead of the decode.
      base += (b + KPhases < b_end) ? step : 0U;
      GUFO_SETPRIO_ON(Prio);
      next = load(base);
      GUFO_SETPRIO_OFF(Prio);
      __builtin_amdgcn_sched_barrier(0);
    } else {
      // b8prioW: priority up before the pointer adds and held to the last load.
      if constexpr (PrioW) {
        asm volatile("s_setprio 3" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      }
      // Persist: the last block of an item is followed by the next item's
      // first block (a row group ps_gstep groups on), so the prefetch runs
      // across the item boundary and only the very last block has no next.
      if constexpr (Balanced) {
        if (ps_n + 1U < ps_total) {
          if (b + KPhases < b_end) {
            base += step;
          } else {
            // A 224-row tile leaves 128 padded rows at m=34816. Keep the
            // lane's field offsets by repeating the final whole row-wave.
            const std::uint32_t next_row =
                row0 + ((ps_j + 1U) * ps_gstep * RowWaves * 32U);
            const std::uint32_t safe_row =
                next_row < m ? next_row : m - 32U + lane;
            base = w +
                   (static_cast<std::size_t>(safe_row / kSnIq3RowsPerGroup) *
                    SnIq3GroupStride(blocks)) +
                   (static_cast<std::size_t>(b_begin + phase) * record);
          }
        }
      } else if constexpr (Persist) {
        base += (b + KPhases < b_end) ? step : ps_jump;
      } else {
        base += step;
      }
      if (Persist ? (ps_n + 1U < ps_total) : (b + KPhases < b_end)) {
        // b8prio: raise this wave's arbiter priority only across the load group.
        GUFO_SETPRIO_ON(Prio);
        next = load(base);
        GUFO_SETPRIO_OFF(Prio);
      }
      if constexpr (PrioW) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 0" ::: "memory");
      }
    }
    const float d = __half2float(__ushort_as_half(cur.d));
    const std::uint32_t qs_words[16] = {
        cur.q[0].x, cur.q[0].y, cur.q[0].z, cur.q[0].w, cur.q[1].x, cur.q[1].y,
        cur.q[1].z, cur.q[1].w, cur.q[2].x, cur.q[2].y, cur.q[2].z, cur.q[2].w,
        cur.q[3].x, cur.q[3].y, cur.q[3].z, cur.q[3].w};
    const std::uint32_t sign_words[8] = {cur.s[0].x, cur.s[0].y, cur.s[0].z,
                                         cur.s[0].w, cur.s[1].x, cur.s[1].y,
                                         cur.s[1].z, cur.s[1].w};
    // ActFix (ceiling probe opus_b8_p_actfix, outputs wrong): the activation
    // block index is an opaque scalar 0, so every K block re-reads block 0's
    // tiles and scales (scalar-cache hits) and the s_loads cannot be hoisted.
    std::uint32_t ab = b;
    if constexpr (ActFix)
      asm volatile("s_mov_b32 %0, 0" : "=s"(ab));
    const std::int8_t* const act_block =
        xq + (static_cast<std::size_t>(ab) * kSnIq3Elements *
              (ActI ? Batch : 1U));
    int isum[Batch] = {};
#pragma unroll
    for (std::uint32_t pair = 0; pair < 4U; ++pair) {
      // Decode sub-blocks 2*pair and 2*pair+1 once (16 signed grid words).
      std::uint32_t word[2][8];
      int scale_int[2];
#pragma unroll
      for (std::uint32_t h = 0; h < 2U; ++h) {
        const std::uint32_t s = (2U * pair) + h;
        // Tag byte i (word i) = qh_bit_i | sign_nibble_i << 1, i.e. bits 8..12
        // of the codebook index qs | (qhbit | sign << 1) << 8.
        const std::uint32_t signs = sign_words[s];
        const std::uint32_t qh_word = s < 4U ? cur.qh.x : cur.qh.y;
        const std::uint32_t qh_shift = 8U * (s & 3U);
        if constexpr (Idx != 0U) {
          // b8cut address fold (sn_iq3_cut.hpp): the perm is the LDS byte address.
          std::uint32_t tag[2];
          std::uint32_t addr[8];
          cut::Tags<Idx>(signs, qh_word, qh_shift, tag);
          const std::uint32_t qs2[2] = {qs_words[2U * s], qs_words[(2U * s) + 1U]};
          cut::WordAddrs<Idx>(tag, qs2, addr);
          const char* const table = reinterpret_cast<const char*>(codebook);
#pragma unroll
          for (std::uint32_t i = 0; i < 8U; ++i)
            word[h][i] = *reinterpret_cast<const std::uint32_t*>(table + addr[i]);
        } else {
        const std::uint32_t even = (signs << 1U) & 0x1E1E1E1EU;  // n0 n2 n4 n6
        const std::uint32_t odd = (signs >> 3U) & 0x1E1E1E1EU;   // n1 n3 n5 n7
        // Spread 4 qh bits to bit 0 of 4 bytes: bit j of q lands at 8j (the
        // four shifted copies 0/7/14/21 never overlap, so no carries).
        const std::uint32_t qh_lo =
            ((qh_word >> qh_shift) & 0x0FU) * 0x00204081U;
        const std::uint32_t qh_hi =
            ((qh_word >> (qh_shift + 4U)) & 0x0FU) * 0x00204081U;
        const std::uint32_t tag[2] = {
            (qh_lo & 0x01010101U) |
                __builtin_amdgcn_perm(odd, even, 0x05010400U),
            (qh_hi & 0x01010101U) |
                __builtin_amdgcn_perm(odd, even, 0x07030602U)};
#pragma unroll
        for (std::uint32_t i = 0; i < 8U; ++i) {
          // byte 0 = qs byte i, byte 1 = tag byte i, bytes 2..3 = 0.
          const std::uint32_t sel =
              0x0C0C0000U | ((4U + (i & 3U)) << 8U) | (i & 3U);
          word[h][i] = codebook[__builtin_amdgcn_perm(
              tag[i >> 2U], qs_words[(2U * s) + (i >> 2U)], sel)];
        }
        }
        scale_int[h] =
            static_cast<int>(1U + (2U * ((cur.sc >> (4U * s)) & 0x0FU)));
      }
      // Tokens in quads: one quad = 4 tokens x 64 bytes (both sub-blocks) =
      // 4 s_load_b512 = 64 SGPRs, the most that fits beside the rest.
      // Quad3 (b8cut): 3 tokens = 48 SGPRs at Batch 7, 8.
      constexpr std::size_t kQuad = (Quad3 && Batch >= 7) ? 3U : 4U;
#pragma unroll
      for (std::size_t t0 = 0; t0 < Batch; t0 += kQuad) {
        uint4 a[kQuad][4];
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4* const p = reinterpret_cast<const uint4*>(
              act_block + ((t0 + j) * (ActI ? kSnIq3Elements : k)) +
              (pair * 64U));
#pragma unroll
          for (std::uint32_t c = 0; c < 4U; ++c)
            a[j][c] = p[c];
        }
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
#pragma unroll
          for (std::uint32_t h = 0; h < 2U; ++h) {
            const uint4 lo = a[j][2U * h];
            const uint4 hi = a[j][(2U * h) + 1U];
            const std::uint32_t* const wd = word[h];
            // NoDot (ceiling probe opus_b8_p_nodot, outputs wrong): each v_dot4
            // becomes one v_add3_u32 on the same three registers, same chain.
            const auto Dot4 = [](std::uint32_t wv, std::uint32_t av, int acc) {
              if constexpr (NoDot) {
                int r;
                asm("v_add3_u32 %0, %1, %2, %3"
                    : "=v"(r)
                    : "v"(wv), "s"(av), "v"(acc));
                return r;
              } else {
                return __builtin_amdgcn_sudot4(true, wv, true, av, acc, false);
              }
            };
            int dot = 0;
            dot = Dot4(wd[0], lo.x, dot);
            dot = Dot4(wd[1], lo.y, dot);
            dot = Dot4(wd[2], lo.z, dot);
            dot = Dot4(wd[3], lo.w, dot);
            dot = Dot4(wd[4], hi.x, dot);
            dot = Dot4(wd[5], hi.y, dot);
            dot = Dot4(wd[6], hi.z, dot);
            dot = Dot4(wd[7], hi.w, dot);
            isum[t0 + j] += __mul24(scale_int[h], dot);
          }
        }
        // Keep the next quad's 64 SGPRs of activations out of this one.
        __builtin_amdgcn_sched_barrier(0);
      }
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float xd =
          ActI ? xds[(b * Batch) + t].x : xds[(t * num_sub) + (ab * 16U)].x;
      sums[t] += (d * xd) * static_cast<float>(isum[t]);
    }
    if constexpr (Persist) {
      ++ps_n;
      if (b + KPhases < b_end) {
        b += KPhases;
      } else {
        // Item done: phase reduce, split-K combine / store, restart at 0.
        SnbPersistEpilogue<Batch, RowWaves, KPhases, Balanced>(
            sums, ps_lds, y, scratch, counter, m, g0 + (ps_j * ps_gstep),
            row0 + (ps_j * ps_gstep * RowWaves * 32U), row_wave, lane, phase,
            split, splits);
        ++ps_j;
        b = b_begin + phase;
      }
    }
  }

  if constexpr (Slope) {
    // m >> 31 is 0 for every m < 2^31, so this adds +0 (exact, sums >= +0).
    const std::uint32_t zero_gate = m >> 31U;
    sums[0] += static_cast<float>(static_cast<int>(chain & zero_gate));
  }
  if constexpr (!Persist) {
  if constexpr (KPhases > 1U) {
    // Fixed-order cross-wave add: phase 0 + 1 + ... + KPhases-1.
    __syncthreads();  // every wave is done with the codebook
    float* const partial = reinterpret_cast<float*>(codebook);
    const auto slot = [&](std::uint32_t p, std::size_t t) {
      return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
    };
    if (phase > 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
        partial[slot(phase, t)] = sums[t];
    }
    __syncthreads();
    if (phase == 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
        for (std::uint32_t p = 1U; p < KPhases; ++p)
          sums[t] += partial[slot(p, t)];
    }
  }
  [[maybe_unused]] bool store_y = true;
  if constexpr (SplitK)
    store_y = SnbSplitKCombine<Batch>(sums, &codebook[8191], scratch, counter,
                                      blockIdx.x, m, row0, split, splits,
                                      phase == 0U);
  if (store_y && phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
  }
  if constexpr (Clock) {
    if (threadIdx.x == 0U) {
      clk[(4U * blockIdx.x) + 1U] =
          static_cast<std::uint64_t>(__builtin_readsteadycounter());
      clk[(4U * blockIdx.x) + 3U] =
          static_cast<std::uint64_t>(__builtin_readcyclecounter());
    }
  }
}

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool Probe = false, bool Pad8 = false, bool Slope = false,
         bool Clock = false, bool ActI = false, std::uint32_t Idx = 0,
         bool Quad3 = false, bool ActFix = false, bool NoDot = false,
         bool Early = false, std::uint32_t Prio = 0, bool PrioW = false,
         bool SplitK = false>
void LaunchOpusIq3sB8Typed(const void* w, float* y,
                           std::int8_t* xq, float2* xds, std::uint32_t m,
                           std::uint32_t k, hipStream_t stream,
                           std::uint64_t* clk = nullptr,
                           float* scratch = nullptr,
                           std::uint32_t* counter = nullptr,
                           std::uint32_t splits = 1U) {
  constexpr std::uint32_t kRowsPerBlock = RowWaves * 32U;
  OpusIq3sB8Kernel<Batch, RowWaves, KPhases, Probe, Pad8, Slope, Clock, ActI,
                   Idx, Quad3, ActFix, NoDot, Early, Prio, PrioW, SplitK>
      <<<dim3((m + kRowsPerBlock - 1U) / kRowsPerBlock, SplitK ? splits : 1U),
         RowWaves * KPhases * 32U, 0, stream>>>(
          static_cast<const std::uint8_t*>(w), xq, xds, y, m, k, clk, scratch,
          counter, 1U);
}

// How many K splits (WGs per row group) for the k4 grid.
// k4 = 512 threads, 32 KiB LDS -> 4 WGs per WGP x 20 WGPs = 80 resident WGs, so
// G row-group WGs fill ceil(G / 80) rounds. Splitting K into S pieces makes
// G*S WGs; pick the S (a divisor of the K blocks, each split keeping at least
// one block per K phase) with the least idle share ceil(G*S/80)*80/(G*S).
// S = 1 unless the idle share at S = 1 is above 1.2 (so a grid that is nearly
// full, e.g. m34816's 272 WGs = 3.4 rounds, stays the k4prio kernel). Ties -> smaller S.
// scratch_floats caps S so S * Batch * m floats fit the caller's scratch.
// forced > 0 (arms k4sk2 / k4sk4 / cvsk2 / cvsk4) takes that S when it is legal
// (divides blocks, one block per phase, scratch fits), else 1.
inline std::uint32_t SnbPickSplitK(std::uint32_t m, std::uint32_t blocks,
                                   std::uint32_t k_phases, std::uint32_t batch,
                                   std::size_t scratch_floats,
                                   std::uint32_t forced = 0U) noexcept {
  if (forced != 0U)
    return (blocks % forced == 0U && forced * k_phases <= blocks &&
            static_cast<std::size_t>(forced) * batch * m <= scratch_floats)
               ? forced
               : 1U;
  constexpr std::uint32_t kResidentWgs = 80U;  // gfx1151: 20 WGPs x 4 WGs
  const std::uint32_t groups = (m + 127U) / 128U;
  const auto idle_share = [&](std::uint32_t s) {
    const std::uint32_t wgs = groups * s;
    return static_cast<double>(((wgs + kResidentWgs - 1U) / kResidentWgs) *
                               kResidentWgs) /
           static_cast<double>(wgs);
  };
  std::uint32_t best = 1U;
  double best_share = idle_share(1U);
  if (best_share <= 1.2) return 1U;
  for (std::uint32_t s = 2U; s * k_phases <= blocks; ++s) {
    if (blocks % s != 0U) continue;
    if (static_cast<std::size_t>(s) * batch * m > scratch_floats) continue;
    const double share = idle_share(s);
    if (share < best_share) {
      best = s;
      best_share = share;
    }
  }
  return best;
}

// k4cut: OpusIq3sB8Kernel<..., Pad8 = true> with fewer
// non-dot4 VALU per K block. Same layout, same activations, same int32 sums,
// same per-row output order: bit-identical to k4 by construction (the decode
// reads the same signed grid word for every (qs byte, qh bit, sign nibble); the
// CPU test tests/models/qwen/sn_iq3_cut_test.cpp checks that for all three Idx).
// Needs m % 8 == 0 (n = 8 is a compile-time constant, as k4pad); the launcher
// falls back to k4 otherwise.
//
// WHY (k4pad ISA, B8, per 256 weights per row): 879 VALU = 512 v_dot4 + 367.
// The loop-carried `next = load()` / `cur = next` pair cost 27 v_dual_mov +
// v_mov + v_mov_b16 = 29 issue slots (2 x 14 dual pairs: one set before the
// conditional load, one after it). This kernel keeps the big weight tiles in
// place: pair p's qs tile (q[p]) is dead as soon as its codebook addresses are
// built, so the NEXT block's q[p] is loaded into the same registers right then
// (and s[p / 2] after the second pair that uses it). The only prefetch copy
// left is the 4 scalars (qh, scales, d). Idx 1 / 2 additionally remove the 64
// `<< 2` address shifts (sn_iq3_cut.hpp).
// Idx 0 = k4's index, copies removed only (attribution arm).
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         std::uint32_t Idx>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(Idx == 2U ? 8 : 16))) void OpusIq3sB8CutKernel(
        const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(Idx <= 2U);
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  constexpr std::uint32_t kTableWords = cut::kTableWords<Idx>;
  // Phase partials for the LDS reduction must fit in the reused codebook.
  static_assert((KPhases - 1U) * RowWaves * Batch * 32U <= kTableWords);
  __shared__ std::uint32_t codebook[kTableWords];
  snb_detail::SnbStageCutCodebook<Idx>(codebook, threadIdx.x, kThreads);
  __syncthreads();

  constexpr std::uint32_t kN = static_cast<std::uint32_t>(kSnIq3RowsPerGroup);
  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t row0 =
      (((blockIdx.x * RowWaves) + row_wave) * 32U) + lane;
  const std::uint32_t row = row0 < m ? row0 : m - 1U;
  const std::uint32_t group = row / kN;
  const std::uint32_t in_group = row % kN;
  constexpr std::uint32_t record = static_cast<std::uint32_t>(SnIq3BlockStride(kN));
  constexpr std::uint32_t chunk_bytes = 16U * kN;
  // 32-bit lane offsets from one uniform base: the 5 planes share the pointer
  // arithmetic. The launcher guarantees the tensor is < 4 GiB. (The compiler
  // still emits one 64-bit v_add_co/ci pair per plane per block; the SADDR
  // form that would need none was not reachable.)
  const std::uint8_t* wbase = w + (static_cast<std::size_t>(phase) * record);
  const std::uint32_t goff =
      group * static_cast<std::uint32_t>(SnIq3GroupStride(blocks));
  const std::uint32_t qs_off = goff + (in_group * 16U);
  const std::uint32_t sg_off =
      goff + static_cast<std::uint32_t>(kN * kSnIq3SignsOffset) + (in_group * 16U);
  const std::uint32_t qh_off =
      goff + static_cast<std::uint32_t>(kN * kSnIq3QhOffset) + (in_group * 8U);
  const std::uint32_t sc_off =
      goff + static_cast<std::uint32_t>(kN * kSnIq3ScalesOffset) + (in_group * 4U);
  const std::uint32_t d_off =
      goff + static_cast<std::uint32_t>(kN * kSnIq3DOffset) + (in_group * 2U);
  constexpr std::uint32_t step = KPhases * record;
  const std::uint32_t num_sub = k / 16U;

  struct Small {
    uint2 qh;
    std::uint32_t sc;
    std::uint16_t d;
  };
  const auto load_q = [&](const std::uint8_t* p, std::uint32_t c) {
    return *reinterpret_cast<const uint4*>(p + static_cast<std::size_t>(qs_off) +
                                           (c * chunk_bytes));
  };
  const auto load_s = [&](const std::uint8_t* p, std::uint32_t c) {
    return *reinterpret_cast<const uint4*>(p + static_cast<std::size_t>(sg_off) +
                                           (c * chunk_bytes));
  };
  const auto load_small = [&](const std::uint8_t* p) {
    Small v;
    v.qh = *reinterpret_cast<const uint2*>(p + static_cast<std::size_t>(qh_off));
    v.sc = *reinterpret_cast<const std::uint32_t*>(p + static_cast<std::size_t>(sc_off));
    v.d = *reinterpret_cast<const std::uint16_t*>(p + static_cast<std::size_t>(d_off));
    return v;
  };

  float sums[Batch] = {};
  uint4 q[4] = {};
  uint4 s[2] = {};
  Small small{};
  if (phase < blocks) {
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      q[c] = load_q(wbase, c);
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      s[c] = load_s(wbase, c);
    small = load_small(wbase);
  }
  const char* const table = reinterpret_cast<const char*>(codebook);
  for (std::uint32_t b = phase; b < blocks; b += KPhases) {
    // Wave-uniform. The last block re-reads itself (cache hit): no branch
    // around the loads, so nothing is copied to make the join of two paths.
    wbase += (b + KPhases < blocks) ? step : 0U;
    const Small cur = small;
    small = load_small(wbase);
    const float d = __half2float(__ushort_as_half(cur.d));
    const std::int8_t* const act_block =
        xq + (static_cast<std::size_t>(b) * kSnIq3Elements);
    int isum[Batch] = {};
    // Tokens per activation tile: 4 tokens x 64 B = 64 SGPRs. At Batch 7, 8
    // that spills SGPRs (k4pad needed no spill at 103 SGPRs; this kernel's
    // loop state is a little larger), so 3 tokens = 48 SGPRs there.
    constexpr std::size_t kQuad = Batch >= 7 ? 3U : 4U;
#pragma unroll
    for (std::uint32_t pair = 0; pair < 4U; ++pair) {
      // Decode sub-blocks 2*pair and 2*pair+1 once (16 signed grid words).
      const std::uint32_t qw[4] = {q[pair].x, q[pair].y, q[pair].z, q[pair].w};
      const uint4 sg = s[pair >> 1U];
      const std::uint32_t sign_words[2] = {(pair & 1U) != 0U ? sg.z : sg.x,
                                           (pair & 1U) != 0U ? sg.w : sg.y};
      std::uint32_t word[2][8];
      int scale_int[2];
#pragma unroll
      for (std::uint32_t h = 0; h < 2U; ++h) {
        const std::uint32_t sub = (2U * pair) + h;
        const std::uint32_t qh_word = sub < 4U ? cur.qh.x : cur.qh.y;
        std::uint32_t tag[2];
        std::uint32_t addr[8];
        cut::Tags<Idx>(sign_words[h], qh_word, 8U * (sub & 3U), tag);
        const std::uint32_t qs2[2] = {qw[2U * h], qw[(2U * h) + 1U]};
        cut::WordAddrs<Idx>(tag, qs2, addr);
#pragma unroll
        for (std::uint32_t i = 0; i < 8U; ++i)
          word[h][i] = *reinterpret_cast<const std::uint32_t*>(table + addr[i]);
        scale_int[h] =
            static_cast<int>(1U + (2U * ((cur.sc >> (4U * sub)) & 0x0FU)));
      }
      // The qs tile (and, after the second pair, the signs tile) is dead:
      // refill it in place with the next block's.
      q[pair] = load_q(wbase, pair);
      if ((pair & 1U) != 0U)
        s[pair >> 1U] = load_s(wbase, pair >> 1U);
#pragma unroll
      for (std::size_t t0 = 0; t0 < Batch; t0 += kQuad) {
        uint4 a[kQuad][4];
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4* const p = reinterpret_cast<const uint4*>(
              act_block + ((t0 + j) * k) + (pair * 64U));
#pragma unroll
          for (std::uint32_t c = 0; c < 4U; ++c)
            a[j][c] = p[c];
        }
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
#pragma unroll
          for (std::uint32_t h = 0; h < 2U; ++h) {
            const uint4 lo = a[j][2U * h];
            const uint4 hi = a[j][(2U * h) + 1U];
            const std::uint32_t* const wd = word[h];
            int dot = 0;
            dot = __builtin_amdgcn_sudot4(true, wd[0], true, lo.x, dot, false);
            dot = __builtin_amdgcn_sudot4(true, wd[1], true, lo.y, dot, false);
            dot = __builtin_amdgcn_sudot4(true, wd[2], true, lo.z, dot, false);
            dot = __builtin_amdgcn_sudot4(true, wd[3], true, lo.w, dot, false);
            dot = __builtin_amdgcn_sudot4(true, wd[4], true, hi.x, dot, false);
            dot = __builtin_amdgcn_sudot4(true, wd[5], true, hi.y, dot, false);
            dot = __builtin_amdgcn_sudot4(true, wd[6], true, hi.z, dot, false);
            dot = __builtin_amdgcn_sudot4(true, wd[7], true, hi.w, dot, false);
            isum[t0 + j] += __mul24(scale_int[h], dot);
          }
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float xd = xds[(t * num_sub) + (b * 16U)].x;
      sums[t] += (d * xd) * static_cast<float>(isum[t]);
    }
  }

  if constexpr (KPhases > 1U) {
    // Fixed-order cross-wave add: phase 0 + 1 + ... + KPhases-1 (as k4).
    __syncthreads();  // every wave is done with the codebook
    float* const partial = reinterpret_cast<float*>(codebook);
    const auto slot = [&](std::uint32_t p, std::size_t t) {
      return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
    };
    if (phase > 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
        partial[slot(phase, t)] = sums[t];
    }
    __syncthreads();
    if (phase == 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
        for (std::uint32_t p = 1U; p < KPhases; ++p)
          sums[t] += partial[slot(p, t)];
    }
  }
  if (phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
}

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         std::uint32_t Idx>
void LaunchOpusIq3sB8Cut(const void* w, float* y, std::int8_t* xq, float2* xds,
                         std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = RowWaves * 32U;
  OpusIq3sB8CutKernel<Batch, RowWaves, KPhases, Idx>
      <<<(m + kRowsPerBlock - 1U) / kRowsPerBlock, RowWaves * KPhases * 32U, 0,
         stream>>>(static_cast<const std::uint8_t*>(w), xq, xds, y, m, k);
}

// The cut kernel needs whole groups (m % 8 == 0, as Pad8) and 32-bit byte
// offsets into the tensor (< 4 GiB); any other shape runs k4, the same math.
template<std::size_t Batch, std::uint32_t Idx>
void LaunchOpusIq3sB8CutOrK4(const void* w, float* y, std::int8_t* xq,
                             float2* xds, std::uint32_t m, std::uint32_t k,
                             hipStream_t stream) {
  const std::size_t groups = (m + kSnIq3RowsPerGroup - 1U) / kSnIq3RowsPerGroup;
  const bool fits = groups * SnIq3GroupStride(k / kSnIq3Elements) <
                    (static_cast<std::size_t>(1) << 32U);
  if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) == 0U && fits)
    LaunchOpusIq3sB8Cut<Batch, 4, 4, Idx>(w, y, xq, xds, m, k, stream);
  else
    LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
}

// The MAC engine on VOPD. Same layout, same K4 geometry (4
// row-waves x 4 K-phases, 512 threads), same tag / index build, same prefetch
// of the next K block's 110 weight bytes, same per-row output order as
// OpusIq3sB8Kernel; what changes is how the 2048 MACs per 256 weights x 8 tokens
// are issued: 512 `v_dual_dot2acc_f32_f16` (2 x dot2acc = 4 MACs) instead of 512
// `v_dot4_i32_iu8`. Measured with an issue-rate microbench: one such pair costs the same issue
// slot as one single dot2acc / fmac.
//
// EXACT: weights w*s (|w| <= 15 odd, s = 2n+1 <= 31, |w*s| <= 465) and int8
// activations are exact in f16; dot2acc multiplies and accumulates in f32, every
// product (<= 59055) and partial sum (<= 8*32*465*127 = 15.1M < 2^24) is an
// exact integer, so the f32 sum equals k4's int32 isum bit for bit, and the
// per-block fold `sums += (d*xd) * isum` is k4's.
//
// ACCUMULATORS: per token two (lo = weights 4i+0,1 of every word, hi = 4i+2,3),
// pinned v(2t) (even, VOPD X) and v(2t+1) (odd, VOPD Y); lo+hi after the block.
// ACTIVATIONS: f16 pairs (xh, block-major, QuantizeActivationsQ8KInterleavedKernel
// with xh) as SGPR src0. 32 weights x 4 tokens = 4 x s_load_b512 = 64 SGPR a
// quad (k4: 64 as well, but int8); the quad's loads are fenced by sched_barrier
// as in k4. One M asm covers 16 weights x a quad (4*NT VOPD).
// WEIGHTS: one half sub-block (4 words = 8 f16 pairs) is made by a "D" asm
// directly into pinned v16..v23 (half 0) / v24..v31 (half 1): word i = pair A
// v(w0+2i) (X vsrc1) + pair B v(w0+2i+1) (Y vsrc1), banks differ (VOPD rule).
// The "M" asm of a quad then issues 4 words x NT tokens = 4*NT VOPD.
//  CV == false (opus_b8_vopd_t16): LDS table of 8 B entries (2 signed f16 pairs,
//    8192 entries = 64 KiB = 8 waves/SIMD); D = 8 v_pk_mul_f16 by (s, s).
//  CV == true  (opus_b8_vopd_cv): k4's 32 KiB table with bytes biased by +128
//    (xor 0x80); per word v_perm_b32(0x64646464, word, sel) gives f16 pairs
//    0x64bb = 1152 + w; D = 8 v_pk_fma_f16(pair, s, -1152*s) = w*s exactly
//    (one rounding, the exact result 465 is representable).
// GENERATED asm macros: .isa/gen_vopd_asm.py (do not edit by hand).
//GENERATED_BEGIN
#define SNB_VOPD_M_Q0_N1_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v23\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q0_N2_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_0], v16 :: v_dual_dot2acc_f32_f16 v3, %[a1_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_2], v18 :: v_dual_dot2acc_f32_f16 v3, %[a1_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_4], v20 :: v_dual_dot2acc_f32_f16 v3, %[a1_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_6], v22 :: v_dual_dot2acc_f32_f16 v3, %[a1_7], v23\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]), "+{v2}"(acc[2]), "+{v3}"(acc[3]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), [a1_0] "s"(a[1][0]), [a1_1] "s"(a[1][1]), [a1_2] "s"(a[1][2]), [a1_3] "s"(a[1][3]), [a1_4] "s"(a[1][4]), [a1_5] "s"(a[1][5]), [a1_6] "s"(a[1][6]), [a1_7] "s"(a[1][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q0_N3_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_0], v16 :: v_dual_dot2acc_f32_f16 v3, %[a1_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_0], v16 :: v_dual_dot2acc_f32_f16 v5, %[a2_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_2], v18 :: v_dual_dot2acc_f32_f16 v3, %[a1_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_2], v18 :: v_dual_dot2acc_f32_f16 v5, %[a2_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_4], v20 :: v_dual_dot2acc_f32_f16 v3, %[a1_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_4], v20 :: v_dual_dot2acc_f32_f16 v5, %[a2_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_6], v22 :: v_dual_dot2acc_f32_f16 v3, %[a1_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_6], v22 :: v_dual_dot2acc_f32_f16 v5, %[a2_7], v23\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]), "+{v2}"(acc[2]), "+{v3}"(acc[3]), "+{v4}"(acc[4]), "+{v5}"(acc[5]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), [a1_0] "s"(a[1][0]), [a1_1] "s"(a[1][1]), [a1_2] "s"(a[1][2]), [a1_3] "s"(a[1][3]), [a1_4] "s"(a[1][4]), [a1_5] "s"(a[1][5]), [a1_6] "s"(a[1][6]), [a1_7] "s"(a[1][7]), [a2_0] "s"(a[2][0]), [a2_1] "s"(a[2][1]), [a2_2] "s"(a[2][2]), [a2_3] "s"(a[2][3]), [a2_4] "s"(a[2][4]), [a2_5] "s"(a[2][5]), [a2_6] "s"(a[2][6]), [a2_7] "s"(a[2][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q0_N4_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_0], v16 :: v_dual_dot2acc_f32_f16 v3, %[a1_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_0], v16 :: v_dual_dot2acc_f32_f16 v5, %[a2_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_0], v16 :: v_dual_dot2acc_f32_f16 v7, %[a3_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_2], v18 :: v_dual_dot2acc_f32_f16 v3, %[a1_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_2], v18 :: v_dual_dot2acc_f32_f16 v5, %[a2_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_2], v18 :: v_dual_dot2acc_f32_f16 v7, %[a3_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_4], v20 :: v_dual_dot2acc_f32_f16 v3, %[a1_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_4], v20 :: v_dual_dot2acc_f32_f16 v5, %[a2_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_4], v20 :: v_dual_dot2acc_f32_f16 v7, %[a3_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_6], v22 :: v_dual_dot2acc_f32_f16 v3, %[a1_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_6], v22 :: v_dual_dot2acc_f32_f16 v5, %[a2_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_6], v22 :: v_dual_dot2acc_f32_f16 v7, %[a3_7], v23\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]), "+{v2}"(acc[2]), "+{v3}"(acc[3]), "+{v4}"(acc[4]), "+{v5}"(acc[5]), "+{v6}"(acc[6]), "+{v7}"(acc[7]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), [a1_0] "s"(a[1][0]), [a1_1] "s"(a[1][1]), [a1_2] "s"(a[1][2]), [a1_3] "s"(a[1][3]), [a1_4] "s"(a[1][4]), [a1_5] "s"(a[1][5]), [a1_6] "s"(a[1][6]), [a1_7] "s"(a[1][7]), [a2_0] "s"(a[2][0]), [a2_1] "s"(a[2][1]), [a2_2] "s"(a[2][2]), [a2_3] "s"(a[2][3]), [a2_4] "s"(a[2][4]), [a2_5] "s"(a[2][5]), [a2_6] "s"(a[2][6]), [a2_7] "s"(a[2][7]), [a3_0] "s"(a[3][0]), [a3_1] "s"(a[3][1]), [a3_2] "s"(a[3][2]), [a3_3] "s"(a[3][3]), [a3_4] "s"(a[3][4]), [a3_5] "s"(a[3][5]), [a3_6] "s"(a[3][6]), [a3_7] "s"(a[3][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q1_N1_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v23\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q1_N2_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_0], v16 :: v_dual_dot2acc_f32_f16 v11, %[a1_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_2], v18 :: v_dual_dot2acc_f32_f16 v11, %[a1_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_4], v20 :: v_dual_dot2acc_f32_f16 v11, %[a1_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_6], v22 :: v_dual_dot2acc_f32_f16 v11, %[a1_7], v23\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]), "+{v10}"(acc[10]), "+{v11}"(acc[11]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), [a1_0] "s"(a[1][0]), [a1_1] "s"(a[1][1]), [a1_2] "s"(a[1][2]), [a1_3] "s"(a[1][3]), [a1_4] "s"(a[1][4]), [a1_5] "s"(a[1][5]), [a1_6] "s"(a[1][6]), [a1_7] "s"(a[1][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q1_N3_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_0], v16 :: v_dual_dot2acc_f32_f16 v11, %[a1_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_0], v16 :: v_dual_dot2acc_f32_f16 v13, %[a2_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_2], v18 :: v_dual_dot2acc_f32_f16 v11, %[a1_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_2], v18 :: v_dual_dot2acc_f32_f16 v13, %[a2_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_4], v20 :: v_dual_dot2acc_f32_f16 v11, %[a1_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_4], v20 :: v_dual_dot2acc_f32_f16 v13, %[a2_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_6], v22 :: v_dual_dot2acc_f32_f16 v11, %[a1_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_6], v22 :: v_dual_dot2acc_f32_f16 v13, %[a2_7], v23\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]), "+{v10}"(acc[10]), "+{v11}"(acc[11]), "+{v12}"(acc[12]), "+{v13}"(acc[13]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), [a1_0] "s"(a[1][0]), [a1_1] "s"(a[1][1]), [a1_2] "s"(a[1][2]), [a1_3] "s"(a[1][3]), [a1_4] "s"(a[1][4]), [a1_5] "s"(a[1][5]), [a1_6] "s"(a[1][6]), [a1_7] "s"(a[1][7]), [a2_0] "s"(a[2][0]), [a2_1] "s"(a[2][1]), [a2_2] "s"(a[2][2]), [a2_3] "s"(a[2][3]), [a2_4] "s"(a[2][4]), [a2_5] "s"(a[2][5]), [a2_6] "s"(a[2][6]), [a2_7] "s"(a[2][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q1_N4_H0() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v16 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_0], v16 :: v_dual_dot2acc_f32_f16 v11, %[a1_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_0], v16 :: v_dual_dot2acc_f32_f16 v13, %[a2_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_0], v16 :: v_dual_dot2acc_f32_f16 v15, %[a3_1], v17\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v18 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_2], v18 :: v_dual_dot2acc_f32_f16 v11, %[a1_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_2], v18 :: v_dual_dot2acc_f32_f16 v13, %[a2_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_2], v18 :: v_dual_dot2acc_f32_f16 v15, %[a3_3], v19\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v20 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_4], v20 :: v_dual_dot2acc_f32_f16 v11, %[a1_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_4], v20 :: v_dual_dot2acc_f32_f16 v13, %[a2_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_4], v20 :: v_dual_dot2acc_f32_f16 v15, %[a3_5], v21\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v22 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_6], v22 :: v_dual_dot2acc_f32_f16 v11, %[a1_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_6], v22 :: v_dual_dot2acc_f32_f16 v13, %[a2_7], v23\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_6], v22 :: v_dual_dot2acc_f32_f16 v15, %[a3_7], v23\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]), "+{v10}"(acc[10]), "+{v11}"(acc[11]), "+{v12}"(acc[12]), "+{v13}"(acc[13]), "+{v14}"(acc[14]), "+{v15}"(acc[15]) \
      : [a0_0] "s"(a[0][0]), [a0_1] "s"(a[0][1]), [a0_2] "s"(a[0][2]), [a0_3] "s"(a[0][3]), [a0_4] "s"(a[0][4]), [a0_5] "s"(a[0][5]), [a0_6] "s"(a[0][6]), [a0_7] "s"(a[0][7]), [a1_0] "s"(a[1][0]), [a1_1] "s"(a[1][1]), [a1_2] "s"(a[1][2]), [a1_3] "s"(a[1][3]), [a1_4] "s"(a[1][4]), [a1_5] "s"(a[1][5]), [a1_6] "s"(a[1][6]), [a1_7] "s"(a[1][7]), [a2_0] "s"(a[2][0]), [a2_1] "s"(a[2][1]), [a2_2] "s"(a[2][2]), [a2_3] "s"(a[2][3]), [a2_4] "s"(a[2][4]), [a2_5] "s"(a[2][5]), [a2_6] "s"(a[2][6]), [a2_7] "s"(a[2][7]), [a3_0] "s"(a[3][0]), [a3_1] "s"(a[3][1]), [a3_2] "s"(a[3][2]), [a3_3] "s"(a[3][3]), [a3_4] "s"(a[3][4]), [a3_5] "s"(a[3][5]), [a3_6] "s"(a[3][6]), [a3_7] "s"(a[3][7]), "{v16}"(wv[0]), "{v17}"(wv[1]), "{v18}"(wv[2]), "{v19}"(wv[3]), "{v20}"(wv[4]), "{v21}"(wv[5]), "{v22}"(wv[6]), "{v23}"(wv[7]))

#define SNB_VOPD_M_Q0_N1_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v31\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_M_Q0_N2_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_0], v24 :: v_dual_dot2acc_f32_f16 v3, %[a1_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_2], v26 :: v_dual_dot2acc_f32_f16 v3, %[a1_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_4], v28 :: v_dual_dot2acc_f32_f16 v3, %[a1_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_6], v30 :: v_dual_dot2acc_f32_f16 v3, %[a1_7], v31\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]), "+{v2}"(acc[2]), "+{v3}"(acc[3]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), [a1_0] "s"(a[1][8]), [a1_1] "s"(a[1][9]), [a1_2] "s"(a[1][10]), [a1_3] "s"(a[1][11]), [a1_4] "s"(a[1][12]), [a1_5] "s"(a[1][13]), [a1_6] "s"(a[1][14]), [a1_7] "s"(a[1][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_M_Q0_N3_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_0], v24 :: v_dual_dot2acc_f32_f16 v3, %[a1_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_0], v24 :: v_dual_dot2acc_f32_f16 v5, %[a2_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_2], v26 :: v_dual_dot2acc_f32_f16 v3, %[a1_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_2], v26 :: v_dual_dot2acc_f32_f16 v5, %[a2_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_4], v28 :: v_dual_dot2acc_f32_f16 v3, %[a1_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_4], v28 :: v_dual_dot2acc_f32_f16 v5, %[a2_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_6], v30 :: v_dual_dot2acc_f32_f16 v3, %[a1_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_6], v30 :: v_dual_dot2acc_f32_f16 v5, %[a2_7], v31\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]), "+{v2}"(acc[2]), "+{v3}"(acc[3]), "+{v4}"(acc[4]), "+{v5}"(acc[5]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), [a1_0] "s"(a[1][8]), [a1_1] "s"(a[1][9]), [a1_2] "s"(a[1][10]), [a1_3] "s"(a[1][11]), [a1_4] "s"(a[1][12]), [a1_5] "s"(a[1][13]), [a1_6] "s"(a[1][14]), [a1_7] "s"(a[1][15]), [a2_0] "s"(a[2][8]), [a2_1] "s"(a[2][9]), [a2_2] "s"(a[2][10]), [a2_3] "s"(a[2][11]), [a2_4] "s"(a[2][12]), [a2_5] "s"(a[2][13]), [a2_6] "s"(a[2][14]), [a2_7] "s"(a[2][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_M_Q0_N4_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v0, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v1, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_0], v24 :: v_dual_dot2acc_f32_f16 v3, %[a1_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_0], v24 :: v_dual_dot2acc_f32_f16 v5, %[a2_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_0], v24 :: v_dual_dot2acc_f32_f16 v7, %[a3_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v1, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_2], v26 :: v_dual_dot2acc_f32_f16 v3, %[a1_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_2], v26 :: v_dual_dot2acc_f32_f16 v5, %[a2_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_2], v26 :: v_dual_dot2acc_f32_f16 v7, %[a3_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v1, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_4], v28 :: v_dual_dot2acc_f32_f16 v3, %[a1_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_4], v28 :: v_dual_dot2acc_f32_f16 v5, %[a2_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_4], v28 :: v_dual_dot2acc_f32_f16 v7, %[a3_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v0, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v1, %[a0_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v2, %[a1_6], v30 :: v_dual_dot2acc_f32_f16 v3, %[a1_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v4, %[a2_6], v30 :: v_dual_dot2acc_f32_f16 v5, %[a2_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v6, %[a3_6], v30 :: v_dual_dot2acc_f32_f16 v7, %[a3_7], v31\n" \
      : "+{v0}"(acc[0]), "+{v1}"(acc[1]), "+{v2}"(acc[2]), "+{v3}"(acc[3]), "+{v4}"(acc[4]), "+{v5}"(acc[5]), "+{v6}"(acc[6]), "+{v7}"(acc[7]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), [a1_0] "s"(a[1][8]), [a1_1] "s"(a[1][9]), [a1_2] "s"(a[1][10]), [a1_3] "s"(a[1][11]), [a1_4] "s"(a[1][12]), [a1_5] "s"(a[1][13]), [a1_6] "s"(a[1][14]), [a1_7] "s"(a[1][15]), [a2_0] "s"(a[2][8]), [a2_1] "s"(a[2][9]), [a2_2] "s"(a[2][10]), [a2_3] "s"(a[2][11]), [a2_4] "s"(a[2][12]), [a2_5] "s"(a[2][13]), [a2_6] "s"(a[2][14]), [a2_7] "s"(a[2][15]), [a3_0] "s"(a[3][8]), [a3_1] "s"(a[3][9]), [a3_2] "s"(a[3][10]), [a3_3] "s"(a[3][11]), [a3_4] "s"(a[3][12]), [a3_5] "s"(a[3][13]), [a3_6] "s"(a[3][14]), [a3_7] "s"(a[3][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_M_Q1_N1_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v31\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_M_Q1_N2_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_0], v24 :: v_dual_dot2acc_f32_f16 v11, %[a1_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_2], v26 :: v_dual_dot2acc_f32_f16 v11, %[a1_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_4], v28 :: v_dual_dot2acc_f32_f16 v11, %[a1_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_6], v30 :: v_dual_dot2acc_f32_f16 v11, %[a1_7], v31\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]), "+{v10}"(acc[10]), "+{v11}"(acc[11]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), [a1_0] "s"(a[1][8]), [a1_1] "s"(a[1][9]), [a1_2] "s"(a[1][10]), [a1_3] "s"(a[1][11]), [a1_4] "s"(a[1][12]), [a1_5] "s"(a[1][13]), [a1_6] "s"(a[1][14]), [a1_7] "s"(a[1][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_M_Q1_N3_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_0], v24 :: v_dual_dot2acc_f32_f16 v11, %[a1_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_0], v24 :: v_dual_dot2acc_f32_f16 v13, %[a2_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_2], v26 :: v_dual_dot2acc_f32_f16 v11, %[a1_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_2], v26 :: v_dual_dot2acc_f32_f16 v13, %[a2_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_4], v28 :: v_dual_dot2acc_f32_f16 v11, %[a1_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_4], v28 :: v_dual_dot2acc_f32_f16 v13, %[a2_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_6], v30 :: v_dual_dot2acc_f32_f16 v11, %[a1_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_6], v30 :: v_dual_dot2acc_f32_f16 v13, %[a2_7], v31\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]), "+{v10}"(acc[10]), "+{v11}"(acc[11]), "+{v12}"(acc[12]), "+{v13}"(acc[13]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), [a1_0] "s"(a[1][8]), [a1_1] "s"(a[1][9]), [a1_2] "s"(a[1][10]), [a1_3] "s"(a[1][11]), [a1_4] "s"(a[1][12]), [a1_5] "s"(a[1][13]), [a1_6] "s"(a[1][14]), [a1_7] "s"(a[1][15]), [a2_0] "s"(a[2][8]), [a2_1] "s"(a[2][9]), [a2_2] "s"(a[2][10]), [a2_3] "s"(a[2][11]), [a2_4] "s"(a[2][12]), [a2_5] "s"(a[2][13]), [a2_6] "s"(a[2][14]), [a2_7] "s"(a[2][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_M_Q1_N4_H1() \
  asm volatile( \
      "v_dual_dot2acc_f32_f16 v8, %[a0_0], v24 :: v_dual_dot2acc_f32_f16 v9, %[a0_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_0], v24 :: v_dual_dot2acc_f32_f16 v11, %[a1_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_0], v24 :: v_dual_dot2acc_f32_f16 v13, %[a2_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_0], v24 :: v_dual_dot2acc_f32_f16 v15, %[a3_1], v25\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_2], v26 :: v_dual_dot2acc_f32_f16 v9, %[a0_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_2], v26 :: v_dual_dot2acc_f32_f16 v11, %[a1_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_2], v26 :: v_dual_dot2acc_f32_f16 v13, %[a2_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_2], v26 :: v_dual_dot2acc_f32_f16 v15, %[a3_3], v27\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_4], v28 :: v_dual_dot2acc_f32_f16 v9, %[a0_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_4], v28 :: v_dual_dot2acc_f32_f16 v11, %[a1_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_4], v28 :: v_dual_dot2acc_f32_f16 v13, %[a2_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_4], v28 :: v_dual_dot2acc_f32_f16 v15, %[a3_5], v29\n" \
      "v_dual_dot2acc_f32_f16 v8, %[a0_6], v30 :: v_dual_dot2acc_f32_f16 v9, %[a0_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v10, %[a1_6], v30 :: v_dual_dot2acc_f32_f16 v11, %[a1_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v12, %[a2_6], v30 :: v_dual_dot2acc_f32_f16 v13, %[a2_7], v31\n" \
      "v_dual_dot2acc_f32_f16 v14, %[a3_6], v30 :: v_dual_dot2acc_f32_f16 v15, %[a3_7], v31\n" \
      : "+{v8}"(acc[8]), "+{v9}"(acc[9]), "+{v10}"(acc[10]), "+{v11}"(acc[11]), "+{v12}"(acc[12]), "+{v13}"(acc[13]), "+{v14}"(acc[14]), "+{v15}"(acc[15]) \
      : [a0_0] "s"(a[0][8]), [a0_1] "s"(a[0][9]), [a0_2] "s"(a[0][10]), [a0_3] "s"(a[0][11]), [a0_4] "s"(a[0][12]), [a0_5] "s"(a[0][13]), [a0_6] "s"(a[0][14]), [a0_7] "s"(a[0][15]), [a1_0] "s"(a[1][8]), [a1_1] "s"(a[1][9]), [a1_2] "s"(a[1][10]), [a1_3] "s"(a[1][11]), [a1_4] "s"(a[1][12]), [a1_5] "s"(a[1][13]), [a1_6] "s"(a[1][14]), [a1_7] "s"(a[1][15]), [a2_0] "s"(a[2][8]), [a2_1] "s"(a[2][9]), [a2_2] "s"(a[2][10]), [a2_3] "s"(a[2][11]), [a2_4] "s"(a[2][12]), [a2_5] "s"(a[2][13]), [a2_6] "s"(a[2][14]), [a2_7] "s"(a[2][15]), [a3_0] "s"(a[3][8]), [a3_1] "s"(a[3][9]), [a3_2] "s"(a[3][10]), [a3_3] "s"(a[3][11]), [a3_4] "s"(a[3][12]), [a3_5] "s"(a[3][13]), [a3_6] "s"(a[3][14]), [a3_7] "s"(a[3][15]), "{v24}"(wv[8]), "{v25}"(wv[9]), "{v26}"(wv[10]), "{v27}"(wv[11]), "{v28}"(wv[12]), "{v29}"(wv[13]), "{v30}"(wv[14]), "{v31}"(wv[15]))

#define SNB_VOPD_D_T16_H0() \
  asm volatile( \
      "v_pk_mul_f16 v16, %[r0], %[sc]\n" \
      "v_pk_mul_f16 v17, %[r1], %[sc]\n" \
      "v_pk_mul_f16 v18, %[r2], %[sc]\n" \
      "v_pk_mul_f16 v19, %[r3], %[sc]\n" \
      "v_pk_mul_f16 v20, %[r4], %[sc]\n" \
      "v_pk_mul_f16 v21, %[r5], %[sc]\n" \
      "v_pk_mul_f16 v22, %[r6], %[sc]\n" \
      "v_pk_mul_f16 v23, %[r7], %[sc]\n" \
      : "=&{v16}"(wv[0]), "=&{v17}"(wv[1]), "=&{v18}"(wv[2]), "=&{v19}"(wv[3]), "=&{v20}"(wv[4]), "=&{v21}"(wv[5]), "=&{v22}"(wv[6]), "=&{v23}"(wv[7]) \
      : [r0] "v"(r[0]), [r1] "v"(r[1]), [r2] "v"(r[2]), [r3] "v"(r[3]), [r4] "v"(r[4]), [r5] "v"(r[5]), [r6] "v"(r[6]), [r7] "v"(r[7]), [sc] "v"(sc))

#define SNB_VOPD_D_T16_H1() \
  asm volatile( \
      "v_pk_mul_f16 v24, %[r0], %[sc]\n" \
      "v_pk_mul_f16 v25, %[r1], %[sc]\n" \
      "v_pk_mul_f16 v26, %[r2], %[sc]\n" \
      "v_pk_mul_f16 v27, %[r3], %[sc]\n" \
      "v_pk_mul_f16 v28, %[r4], %[sc]\n" \
      "v_pk_mul_f16 v29, %[r5], %[sc]\n" \
      "v_pk_mul_f16 v30, %[r6], %[sc]\n" \
      "v_pk_mul_f16 v31, %[r7], %[sc]\n" \
      : "=&{v24}"(wv[8]), "=&{v25}"(wv[9]), "=&{v26}"(wv[10]), "=&{v27}"(wv[11]), "=&{v28}"(wv[12]), "=&{v29}"(wv[13]), "=&{v30}"(wv[14]), "=&{v31}"(wv[15]) \
      : [r0] "v"(r[0]), [r1] "v"(r[1]), [r2] "v"(r[2]), [r3] "v"(r[3]), [r4] "v"(r[4]), [r5] "v"(r[5]), [r6] "v"(r[6]), [r7] "v"(r[7]), [sc] "v"(sc))

#define SNB_VOPD_D_CV_H0() \
  asm volatile( \
      "v_pk_fma_f16 v16, %[r0], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v17, %[r1], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v18, %[r2], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v19, %[r3], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v20, %[r4], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v21, %[r5], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v22, %[r6], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v23, %[r7], %[sc], %[cc]\n" \
      : "=&{v16}"(wv[0]), "=&{v17}"(wv[1]), "=&{v18}"(wv[2]), "=&{v19}"(wv[3]), "=&{v20}"(wv[4]), "=&{v21}"(wv[5]), "=&{v22}"(wv[6]), "=&{v23}"(wv[7]) \
      : [r0] "v"(r[0]), [r1] "v"(r[1]), [r2] "v"(r[2]), [r3] "v"(r[3]), [r4] "v"(r[4]), [r5] "v"(r[5]), [r6] "v"(r[6]), [r7] "v"(r[7]), [sc] "v"(sc), [cc] "v"(cc))

#define SNB_VOPD_D_CV_H1() \
  asm volatile( \
      "v_pk_fma_f16 v24, %[r0], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v25, %[r1], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v26, %[r2], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v27, %[r3], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v28, %[r4], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v29, %[r5], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v30, %[r6], %[sc], %[cc]\n" \
      "v_pk_fma_f16 v31, %[r7], %[sc], %[cc]\n" \
      : "=&{v24}"(wv[8]), "=&{v25}"(wv[9]), "=&{v26}"(wv[10]), "=&{v27}"(wv[11]), "=&{v28}"(wv[12]), "=&{v29}"(wv[13]), "=&{v30}"(wv[14]), "=&{v31}"(wv[15]) \
      : [r0] "v"(r[0]), [r1] "v"(r[1]), [r2] "v"(r[2]), [r3] "v"(r[3]), [r4] "v"(r[4]), [r5] "v"(r[5]), [r6] "v"(r[6]), [r7] "v"(r[7]), [sc] "v"(sc), [cc] "v"(cc))
//GENERATED_END

namespace snb_detail {

__device__ __forceinline__ std::uint32_t SnbF16Bits(int v) {
  return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(static_cast<float>(v)));
}

// One 8-byte table entry: the 4 signed int8 weights of `word` as two f16 pairs.
__device__ __forceinline__ uint2 SnbF16Entry(std::uint32_t word) {
  const auto b = [&](std::uint32_t i) {
    return SnbF16Bits(static_cast<std::int8_t>((word >> (8U * i)) & 0xFFU));
  };
  return make_uint2(b(0) | (b(1) << 16U), b(2) | (b(3) << 16U));
}

template<bool CV>
__device__ __forceinline__ void SnbStageCodebookVopd(std::uint32_t* table,
                                                     std::uint32_t tid,
                                                     std::uint32_t threads) {
  for (std::uint32_t i = tid; i < 512U; i += threads) {
    const std::uint32_t grid = kDeviceIq3sGrid[i];
#pragma unroll
    for (std::uint32_t sign = 0; sign < 16U; ++sign) {
      const std::uint32_t signed_word = NegateSelectedBytes(grid, sign);
      if constexpr (CV)
        table[(sign << 9U) | i] = signed_word ^ 0x80808080U;  // byte = w + 128
      else
        reinterpret_cast<uint2*>(table)[(sign << 9U) | i] =
            SnbF16Entry(signed_word);
    }
  }
}

}  // namespace snb_detail

typedef unsigned VopdU16 __attribute__((ext_vector_type(16)));

// Half H (words 4H..4H+3 = weights 16H..16H+15) of sub-block sb: table reads
// (two f16 pairs a word; CV: biased bytes -> two f16 pairs 1152 + w by two
// v_perm), then the D asm into the pinned weights. A sched_barrier after it
// keeps the other half's reads from being hoisted (VGPR budget).
#define SNB_VOPD_DO_D(H)                                                   \
  do {                                                                     \
    std::uint32_t r[8];                                                    \
    _Pragma("unroll") for (std::uint32_t ii = 0; ii < 4U; ++ii) {          \
      const std::uint32_t i = (4U * (H)) + ii;                             \
      const std::uint32_t sel =                                            \
          0x0C0C0000U | ((4U + (i & 3U)) << 8U) | (i & 3U);                \
      const std::uint32_t idx = __builtin_amdgcn_perm(                     \
          tag[i >> 2U], qs_words[(2U * sb) + (i >> 2U)], sel);             \
      if constexpr (CV) {                                                  \
        const std::uint32_t e = codebook[idx];                             \
        r[(2U * ii) + 0U] =                                                \
            __builtin_amdgcn_perm(0x64646464U, e, 0x05010400U);            \
        r[(2U * ii) + 1U] =                                                \
            __builtin_amdgcn_perm(0x64646464U, e, 0x05030402U);            \
      } else {                                                             \
        const uint2 e = reinterpret_cast<const uint2*>(codebook)[idx];     \
        r[(2U * ii) + 0U] = e.x;                                           \
        r[(2U * ii) + 1U] = e.y;                                           \
      }                                                                    \
    }                                                                      \
    if constexpr (CV) {                                                    \
      SNB_VOPD_D_CV_H##H();                                                \
    } else {                                                               \
      SNB_VOPD_D_T16_H##H();                                               \
    }                                                                      \
    __builtin_amdgcn_sched_barrier(0);                                     \
  } while (0)

// Quad Q (tokens 4Q..4Q+3 that exist) of sub-block sb: 4 x s_load_b512 (32
// weights x 4 tokens as f16 = 64 SGPR), then the half-0 and half-1 M asm.
#define SNB_VOPD_DO_Q(Q)                                                   \
  if constexpr (Batch > 4U * (Q)) {                                        \
    constexpr std::size_t kNt = (Batch - (4U * (Q))) < 4U                  \
                                    ? (Batch - (4U * (Q)))                 \
                                    : 4U;                                  \
    VopdU16 a[4];                                                          \
    _Pragma("unroll") for (std::size_t jj = 0; jj < kNt; ++jj)             \
        a[jj] = *reinterpret_cast<const VopdU16*>(                         \
            act_blk + (((4U * (Q)) + jj) * 256U) + (sb * 32U));            \
    if constexpr (kNt == 1U) {                                             \
      SNB_VOPD_M_Q##Q##_N1_H0();                                           \
      SNB_VOPD_M_Q##Q##_N1_H1();                                           \
    } else if constexpr (kNt == 2U) {                                      \
      SNB_VOPD_M_Q##Q##_N2_H0();                                           \
      SNB_VOPD_M_Q##Q##_N2_H1();                                           \
    } else if constexpr (kNt == 3U) {                                      \
      SNB_VOPD_M_Q##Q##_N3_H0();                                           \
      SNB_VOPD_M_Q##Q##_N3_H1();                                           \
    } else {                                                               \
      SNB_VOPD_M_Q##Q##_N4_H0();                                           \
      SNB_VOPD_M_Q##Q##_N4_H1();                                           \
    }                                                                      \
    __builtin_amdgcn_sched_barrier(0);                                     \
  }

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool CV, bool Pad8 = false, bool Prio = false, bool SplitK = false,
         bool Persist = false>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(CV ? 16 : 8))) void OpusIq3sB8VopdKernel(
        const std::uint8_t* __restrict__ w, const std::uint16_t* __restrict__ xh,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k, float* scratch, std::uint32_t* counter,
        std::uint32_t ps_splits) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(!Persist || (SplitK && Pad8));
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  constexpr std::uint32_t kTableWords = CV ? 8192U : 16384U;
  static_assert((KPhases - 1U) * RowWaves * Batch * 32U <= kTableWords);
  __shared__ __attribute__((aligned(16))) std::uint32_t codebook[kTableWords];
  // Persist: staged once, after the first block's loads are issued (below).
  if constexpr (!Persist) {
    snb_detail::SnbStageCodebookVopd<CV>(codebook, threadIdx.x, kThreads);
    __syncthreads();
  }

  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  // SplitK (b8sk): blockIdx.y = split over contiguous K blocks [b_begin, b_end)
  // (the launcher guarantees gridDim.y divides blocks); else 0 / blocks.
  // Persist (b8ps): gridDim.x resident WGs, WG w does split w % S of row groups
  // g0 + j * gstep (g0 = w / S, gstep = gridDim.x / S; S divides gridDim.x).
  const std::uint32_t splits = Persist ? ps_splits : (SplitK ? gridDim.y : 1U);
  const std::uint32_t split =
      Persist ? blockIdx.x % splits : (SplitK ? blockIdx.y : 0U);
  const std::uint32_t g0 = Persist ? blockIdx.x / splits : blockIdx.x;
  const std::uint32_t b_begin = SplitK ? split * (blocks / splits) : 0U;
  const std::uint32_t b_end = SplitK ? b_begin + (blocks / splits) : blocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t row0 = (((g0 * RowWaves) + row_wave) * 32U) + lane;
  const std::uint32_t row = row0 < m ? row0 : m - 1U;
  const std::uint32_t group = row / kSnIq3RowsPerGroup;
  const std::uint32_t in_group = row % kSnIq3RowsPerGroup;
  const std::uint32_t rows_left = m - (group * kSnIq3RowsPerGroup);
  // Pad8 (as OpusIq3sB8Kernel): m % 8 == 0, every group whole, n is a
  // constant and the plane offsets become immediates (fewer live VGPRs).
  const std::uint32_t n =
      Pad8 ? static_cast<std::uint32_t>(kSnIq3RowsPerGroup)
           : (rows_left < kSnIq3RowsPerGroup ? rows_left
                                             : kSnIq3RowsPerGroup);
  const std::uint32_t record = static_cast<std::uint32_t>(SnIq3BlockStride(n));
  const std::uint32_t chunk_bytes = 16U * n;
  const std::uint8_t* base =
      w + (static_cast<std::size_t>(group) * SnIq3GroupStride(blocks)) +
      (static_cast<std::size_t>(b_begin + phase) * record);
  const std::uint32_t qs_off = in_group * 16U;
  const std::uint32_t sg_off = (n * kSnIq3SignsOffset) + (in_group * 16U);
  const std::uint32_t qh_off = (n * kSnIq3QhOffset) + (in_group * 8U);
  const std::uint32_t sc_off = (n * kSnIq3ScalesOffset) + (in_group * 4U);
  const std::uint32_t d_off = (n * kSnIq3DOffset) + (in_group * 2U);
  const std::uint32_t step = KPhases * record;

  struct Weights {
    uint4 q[4];
    uint4 s[2];
    uint2 qh;
    std::uint32_t sc;
    std::uint16_t d;
  };
  const auto load = [&](const std::uint8_t* p) {
    Weights v;
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      v.q[c] = *reinterpret_cast<const uint4*>(p + qs_off + (c * chunk_bytes));
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      v.s[c] = *reinterpret_cast<const uint4*>(p + sg_off + (c * chunk_bytes));
    v.qh = *reinterpret_cast<const uint2*>(p + qh_off);
    v.sc = *reinterpret_cast<const std::uint32_t*>(p + sc_off);
    v.d = *reinterpret_cast<const std::uint16_t*>(p + d_off);
    return v;
  };

  [[maybe_unused]] std::uint32_t ps_total = 0U;
  [[maybe_unused]] std::uint32_t ps_gstep = 0U;
  [[maybe_unused]] std::size_t ps_jump = 0U;
  if constexpr (Persist) {
    const std::uint32_t groups_total = m / (RowWaves * 32U);
    const std::uint32_t nb =
        ((blocks / splits) - phase + KPhases - 1U) / KPhases;
    ps_gstep = gridDim.x / splits;
    ps_total = ((groups_total - g0 + ps_gstep - 1U) / ps_gstep) * nb;
    ps_jump = (static_cast<std::size_t>(ps_gstep) * RowWaves * 4U *
               SnIq3GroupStride(blocks)) -
              (static_cast<std::size_t>(nb - 1U) * step);
  }
  float sums[Batch] = {};
  Weights next{};
  if (b_begin + phase < b_end)
    next = load(base);
  [[maybe_unused]] float* ps_lds = nullptr;
  if constexpr (Persist) {
    __shared__ float ps_buf[((KPhases - 1U) * RowWaves * Batch * 32U) + 1U];
    ps_lds = ps_buf;
    snb_detail::SnbStageCodebookVopd<CV>(codebook, threadIdx.x, kThreads);
    __syncthreads();
  }
  [[maybe_unused]] std::uint32_t ps_n = 0U;
  [[maybe_unused]] std::uint32_t ps_j = 0U;
  for (std::uint32_t b = b_begin + phase; Persist ? ps_n < ps_total : b < b_end;
       b += Persist ? 0U : KPhases) {
    const Weights cur = next;
    // The next block's 110 bytes are requested halfway through this block
    // (sub-block 4), not at its start: cur's qs/sign words of sub-blocks 0..3
    // are dead by then, so cur + next never both hold everything (VGPR budget of
    // the 16-waves/SIMD arm). A DRAM round trip (~1-2 us) is far below one
    // block's wall time per wave (~20 us with 16 waves sharing a SIMD). The
    // last block re-reads its own record instead of branching.
    // Persist: the last block of an item is followed by the next item's first
    // block (ps_jump on), so the prefetch crosses the item boundary; the very
    // last block of the launch re-reads its own record.
    const bool more_in_item = b + KPhases < b_end;
    const std::uint8_t* const next_base =
        Persist ? ((ps_n + 1U < ps_total)
                       ? base + (more_in_item ? step : ps_jump)
                       : base)
                : (more_in_item ? base + step : base);
    if constexpr (Persist)
      base += more_in_item ? step : ps_jump;
    else
      base += step;
    const float d = __half2float(__ushort_as_half(cur.d));
    const std::uint32_t qs_words[16] = {
        cur.q[0].x, cur.q[0].y, cur.q[0].z, cur.q[0].w, cur.q[1].x, cur.q[1].y,
        cur.q[1].z, cur.q[1].w, cur.q[2].x, cur.q[2].y, cur.q[2].z, cur.q[2].w,
        cur.q[3].x, cur.q[3].y, cur.q[3].z, cur.q[3].w};
    const std::uint32_t sign_words[8] = {cur.s[0].x, cur.s[0].y, cur.s[0].z,
                                         cur.s[0].w, cur.s[1].x, cur.s[1].y,
                                         cur.s[1].z, cur.s[1].w};
    // readfirstlane hides the induction variable from SCEV: otherwise the
    // pointer is strength-reduced across iterations and every s_load gets a
    // 64-bit s_add/s_addc of a negative constant instead of an immediate offset.
    const std::uint16_t* const act_blk =
        xh + (static_cast<std::size_t>(__builtin_amdgcn_readfirstlane(b)) *
              Batch * kSnIq3Elements);
    float acc[16];
#pragma unroll
    for (std::uint32_t i = 0; i < 2U * Batch; ++i)
      acc[i] = 0.0F;
    std::uint32_t wv[16];  // pinned v16..v31, written only by the D asm
#pragma unroll
    for (std::uint32_t sb = 0; sb < 8U; ++sb) {
      if (sb == 4U) {
        if constexpr (Prio)  // b8prio: raised priority across the load group
          asm volatile("s_setprio 3" ::: "memory");
        next = load(next_base);
        if constexpr (Prio)
          asm volatile("s_setprio 0" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
      }
      // Tag byte i (word i) = qh_bit_i | sign_nibble_i << 1: k4's build.
      const std::uint32_t signs = sign_words[sb];
      const std::uint32_t even = (signs << 1U) & 0x1E1E1E1EU;
      const std::uint32_t odd = (signs >> 3U) & 0x1E1E1E1EU;
      const std::uint32_t qh_word = sb < 4U ? cur.qh.x : cur.qh.y;
      const std::uint32_t qh_shift = 8U * (sb & 3U);
      const std::uint32_t qh_lo =
          ((qh_word >> qh_shift) & 0x0FU) * 0x00204081U;
      const std::uint32_t qh_hi =
          ((qh_word >> (qh_shift + 4U)) & 0x0FU) * 0x00204081U;
      const std::uint32_t tag[2] = {
          (qh_lo & 0x01010101U) |
              __builtin_amdgcn_perm(odd, even, 0x05010400U),
          (qh_hi & 0x01010101U) |
              __builtin_amdgcn_perm(odd, even, 0x07030602U)};
      // (s, s) and (-1152 s, -1152 s) as f16 pairs; s = 1 + 2*scale_nibble.
      const int scale_int =
          static_cast<int>(1U + (2U * ((cur.sc >> (4U * sb)) & 0x0FU)));
      const std::uint32_t sbits = snb_detail::SnbF16Bits(scale_int);
      [[maybe_unused]] const std::uint32_t sc = sbits | (sbits << 16U);
      [[maybe_unused]] std::uint32_t cc = 0U;
      if constexpr (CV) {
        const std::uint32_t cbits = snb_detail::SnbF16Bits(-1152 * scale_int);
        cc = cbits | (cbits << 16U);
      }
      SNB_VOPD_DO_D(0);
      SNB_VOPD_DO_D(1);
      SNB_VOPD_DO_Q(0);
      SNB_VOPD_DO_Q(1);
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float xd = xds[(b * Batch) + t].x;
      sums[t] += (d * xd) * (acc[2U * t] + acc[(2U * t) + 1U]);
    }
    if constexpr (Persist) {
      ++ps_n;
      if (more_in_item) {
        b += KPhases;
      } else {
        SnbPersistEpilogue<Batch, RowWaves, KPhases>(
            sums, ps_lds, y, scratch, counter, m, g0 + (ps_j * ps_gstep),
            row0 + (ps_j * ps_gstep * RowWaves * 32U), row_wave, lane, phase,
            split, splits);
        ++ps_j;
        b = b_begin + phase;
      }
    }
  }

  if constexpr (!Persist) {
  if constexpr (KPhases > 1U) {
    __syncthreads();  // every wave is done with the codebook
    float* const partial = reinterpret_cast<float*>(codebook);
    const auto slot = [&](std::uint32_t p, std::size_t t) {
      return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
    };
    if (phase > 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
        partial[slot(phase, t)] = sums[t];
    }
    __syncthreads();
    if (phase == 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
        for (std::uint32_t p = 1U; p < KPhases; ++p)
          sums[t] += partial[slot(p, t)];
    }
  }
  [[maybe_unused]] bool store_y = true;
  if constexpr (SplitK)
    store_y = SnbSplitKCombine<Batch>(sums, &codebook[kTableWords - 1U], scratch,
                                      counter, blockIdx.x, m, row0, split,
                                      splits, phase == 0U);
  if (store_y && phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
  }
}

template<std::size_t Batch, bool CV, bool Prio = false>
void LaunchOpusIq3sB8Vopd(const void* w, float* y, std::uint16_t* xh,
                          float2* xds, std::uint32_t m, std::uint32_t k,
                          hipStream_t stream, float* scratch = nullptr,
                          std::uint32_t* counter = nullptr,
                          std::uint32_t splits = 1U) {
  constexpr std::uint32_t kRowWaves = 4U;
  constexpr std::uint32_t kKPhases = 4U;
  constexpr std::uint32_t kRowsPerBlock = kRowWaves * 32U;
  // The 16-waves/SIMD arm needs the Pad8 immediates to stay inside 96 VGPRs
  // (without: 13 spilled VGPRs at B8); every real matrix has m % 8 == 0.
  if constexpr (CV) {
    if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) == 0U) {
      if (splits > 1U) {  // b8sk: caller guarantees splits | blocks, scratch, counter
        OpusIq3sB8VopdKernel<Batch, kRowWaves, kKPhases, CV, true, Prio, true>
            <<<dim3((m + kRowsPerBlock - 1U) / kRowsPerBlock, splits),
               kRowWaves * kKPhases * 32U, 0, stream>>>(
                static_cast<const std::uint8_t*>(w), xh, xds, y, m, k, scratch,
                counter, 1U);
        return;
      }
      OpusIq3sB8VopdKernel<Batch, kRowWaves, kKPhases, CV, true, Prio>
          <<<(m + kRowsPerBlock - 1U) / kRowsPerBlock,
             kRowWaves * kKPhases * 32U, 0, stream>>>(
              static_cast<const std::uint8_t*>(w), xh, xds, y, m, k, nullptr,
              nullptr, 1U);
      return;
    }
  }
  OpusIq3sB8VopdKernel<Batch, kRowWaves, kKPhases, CV, false, Prio>
      <<<(m + kRowsPerBlock - 1U) / kRowsPerBlock, kRowWaves * kKPhases * 32U,
         0, stream>>>(static_cast<const std::uint8_t*>(w), xh, xds, y, m, k,
                      nullptr, nullptr, 1U);
}

// PERSISTENT stream-K launch for the k4prio (CV = false) and
// vopd_cv (CV = true) bodies. kPsR WGs of 1024 threads (8 row-waves x 4
// K-phases = 256 rows) are launched once and loop over work items (row group g,
// K split s); WG w always takes split w % S and row groups w / S + j * (R / S).
// Why 1024-thread WGs and R = 40: the 32 KiB table stays in LDS for the whole
// launch, so the phase reduction needs its own LDS (24 KiB at B8), and
// 4 x (32 + 24) KiB does not fit a WGP's 128 KiB; 2 x 56 KiB does. Two WGs of
// 32 waves per WGP are still 16 waves per SIMD (<= 96 VGPR), so occupancy is
// k4's and R = 2 WGs x 20 WGPs = 40. cv has the same table size and VGPR cap,
// so the same R.
constexpr std::uint32_t kPsResidentWgs = 40U;

// S in divisors of the K blocks (each item keeps >= one block per K phase) that
// also divide R (the schedule needs R % S == 0), minimising the idle share
// ceil(items / R) * R / items of the R workgroups; ties -> smaller S. S > 1 needs
// S * Batch * m floats of scratch. GUFO_PS_SPLITS=<S> forces a legal S (bench sweep).
inline std::uint32_t SnbPickPersistSplit(std::uint32_t m, std::uint32_t blocks,
                                         std::uint32_t k_phases,
                                         std::uint32_t batch,
                                         std::size_t scratch_floats) noexcept {
  static const std::uint32_t forced = [] {
    const char* const v = std::getenv("GUFO_PS_SPLITS");
    return v != nullptr ? static_cast<std::uint32_t>(std::atoi(v)) : 0U;
  }();
  const std::uint32_t groups = m / 256U;
  const auto legal = [&](std::uint32_t s) {
    return s >= 1U && blocks % s == 0U && s * k_phases <= blocks &&
           kPsResidentWgs % s == 0U &&
           (s == 1U ||
            static_cast<std::size_t>(s) * batch * m <= scratch_floats);
  };
  if (forced != 0U && legal(forced)) return forced;
  std::uint32_t best = 1U;
  double best_share = 1e30;
  for (std::uint32_t s = 1U; s * k_phases <= blocks; ++s) {
    if (!legal(s)) continue;
    const std::uint32_t items = groups * s;
    const double share =
        static_cast<double>(((items + kPsResidentWgs - 1U) / kPsResidentWgs) *
                            kPsResidentWgs) /
        static_cast<double>(items);
    if (share < best_share) {
      best = s;
      best_share = share;
    }
  }
  return best;
}

// Returns false (nothing launched) when the shape is not persistent-capable
// (m % 256 != 0); the caller then runs its plain kernel.
template<std::size_t Batch, bool CV>
bool LaunchOpusIq3sB8Persist(const void* w, float* y, const void* act,
                             float2* xds, std::uint32_t m, std::uint32_t k,
                             hipStream_t stream, float* scratch,
                             std::uint32_t* counter,
                             std::size_t scratch_floats) {
  if (m % 256U != 0U) return false;
  if constexpr (!CV && (Batch == 7 || Batch == 8)) {
    if (m == 34816U && k == 5120U && GufoDevIq3X()) {
      // 156 tiles / 40 WGs, instead of 136 / 40. Four K phases and S=1
      // retain the served per-column reduction order; no scratch is used.
      OpusIq3sB8Kernel<Batch, 7, 4, false, true, false, false, false, 0U, false,
                       false, false, false, 1U, false, true, true, true>
          <<<40, 896, 0, stream>>>(static_cast<const std::uint8_t*>(w),
                                   static_cast<const std::int8_t*>(act), xds, y,
                                   m, k, nullptr, nullptr, nullptr, 1U);
      return true;
    }
  }
  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  const std::uint32_t splits = SnbPickPersistSplit(
      m, blocks, 4U, static_cast<std::uint32_t>(Batch),
      (scratch != nullptr && counter != nullptr) ? scratch_floats : 0U);
  const std::uint32_t items = (m / 256U) * splits;
  const std::uint32_t wgs = items < kPsResidentWgs ? items : kPsResidentWgs;
  if constexpr (CV)
    OpusIq3sB8VopdKernel<Batch, 8, 4, true, true, false, true, true>
        <<<wgs, 1024, 0, stream>>>(static_cast<const std::uint8_t*>(w),
                                   static_cast<const std::uint16_t*>(act), xds,
                                   y, m, k, scratch, counter, splits);
  else
    OpusIq3sB8Kernel<Batch, 8, 4, false, true, false, false, false, 0U, false,
                     false, false, false, 1U, false, true, true>
        <<<wgs, 1024, 0, stream>>>(static_cast<const std::uint8_t*>(w),
                                   static_cast<const std::int8_t*>(act), xds, y,
                                   m, k, nullptr, scratch, counter, splits);
  return true;
}

// OpusIq3sB8StreamKernel. Same layout, codebook, int math and
// per-row accumulation order as OpusIq3sB8Kernel (isum per 256 in int32, one
// float fold per K block, phases added 0, 1, ..., KPhases-1). Three changes:
//  1. Rows (1 or 2) output rows per lane, rows lane and lane + 32 of a 64-row
//     wave tile: every activation s_load feeds Rows x as many v_dot4.
//  2. Prefetch one pair ahead, no copy: each pair loads the next pair's 16 qs
//     + 8 sign bytes per row (pair 3: pair 0 of block b + phases), so only
//     ~10 weight VGPRs a row are live instead of 28 (+28 for a full-block
//     copy). VMEM returns in order, so the waits stay partial. Cover: one
//     pair of one wave is ~1/4 iteration (~220 VALU a row), x the 8-16 waves
//     sharing the SIMD = ~2000-3500 cycles of wall time, about one DRAM
//     latency under load. 32-bit offsets from w (the tensor is < 4 GiB).
//  3. ksplit > 1 (runtime) also splits K across ksplit blocks per row tile:
//     block (tile, split) owns K phases split*KPhases .. +KPhases-1 of
//     phases = ksplit*KPhases. Each block reduces its phases in LDS (as
//     before), writes its partial to `partial`, and the last block of the tile
//     (integer ticket, no float atomics) adds the partials in split order
//     0..ksplit-1 and writes y, then resets the ticket. Deterministic: the sum
//     order never depends on which block finishes last.
//  4. A sched_barrier between a quad's activation s_loads and its dot4s: the
//     4 s_load_b512 (and the pair's ds_loads) share one lgkmcnt(0).
// ISA (gfx1151, -O3, Batch 8, one K iteration; OpusIq3sB8Kernel<8,4,4>: 886
// VALU, 18 lgkmcnt(0), 96 VGPR):
//   Rows 1: 857 VALU (512 dot4), 64 ds_load, 32 s_load_b512, 12 VMEM,
//           10 lgkmcnt(0), 72 VGPR, 0 spill, 16 waves/SIMD.
//   Rows 2: 1700 VALU (1024 dot4), 128 ds_load, 32 s_load_b512, 24 VMEM,
//           10 lgkmcnt(0), 120 VGPR + 40 B scratch (5 scratch loads an
//           iteration), 12 waves/SIMD (3 x 16-wave blocks a WGP).
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         std::uint32_t Rows>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(Rows == 1U ? 16U : 12U))) void
    OpusIq3sB8StreamKernel(const std::uint8_t* __restrict__ w,
                           const std::int8_t* __restrict__ xq,
                           const float2* __restrict__ xds,
                           float* __restrict__ y, std::uint32_t m,
                           std::uint32_t k, std::uint32_t ksplit,
                           float* partial, std::uint32_t* ticket) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(Rows == 1U || Rows == 2U);
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  static_assert((KPhases - 1U) * RowWaves * Rows * Batch * 32U <= 8192U);
  // Exactly 32 KiB: one more LDS word would cap a WGP at 3 blocks, not 4.
  __shared__ std::uint32_t codebook[8192];

  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t tile = blockIdx.x / ksplit;
  const std::uint32_t split = blockIdx.x - (tile * ksplit);
  const std::uint32_t phases = ksplit * KPhases;
  const std::uint32_t first = (split * KPhases) + phase;
  const std::uint32_t group_stride =
      static_cast<std::uint32_t>(SnIq3GroupStride(blocks));
  const std::uint32_t num_sub = k / 16U;

  std::uint32_t row0[Rows], off[Rows], step[Rows], chunk[Rows], sg_off[Rows],
      qh_off[Rows], sc_off[Rows], d_off[Rows];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    row0[r] = (((((tile * RowWaves) + row_wave) * Rows) + r) * 32U) + lane;
    const std::uint32_t row = row0[r] < m ? row0[r] : m - 1U;
    const std::uint32_t group = row / kSnIq3RowsPerGroup;
    const std::uint32_t in_group = row % kSnIq3RowsPerGroup;
    const std::uint32_t rows_left = m - (group * kSnIq3RowsPerGroup);
    const std::uint32_t n =
        rows_left < kSnIq3RowsPerGroup ? rows_left : kSnIq3RowsPerGroup;
    const std::uint32_t record =
        static_cast<std::uint32_t>(SnIq3BlockStride(n));
    // off = this row's qs chunk 0 inside block record `first`; the other
    // planes are fixed offsets from it.
    off[r] = (group * group_stride) + (first * record) + (in_group * 16U);
    step[r] = phases * record;
    chunk[r] = 16U * n;
    sg_off[r] = n * kSnIq3SignsOffset;
    qh_off[r] = (n * kSnIq3QhOffset) - (in_group * 8U);
    sc_off[r] = (n * kSnIq3ScalesOffset) - (in_group * 12U);
    d_off[r] = (n * kSnIq3DOffset) - (in_group * 14U);
  }
  const auto ld = [&]<typename T>(std::uint32_t o) {
    return *reinterpret_cast<const T*>(w + o);  // o: 32-bit byte offset
  };
  // Weight registers per row, each loaded one pair ahead of its reader:
  //   qn  = qs chunk of the next pair (16 B), sn = its 2 sign words (8 B),
  //   qhx/qhy = qh bytes 0-3 / 4-7, sc = 4 scale bytes, dv = fp16 d.
  // Pair p (after its decode) loads pair p+1's qn/sn from block b, pair 3
  // loads pair 0's from block b + phases, plus that block's qhx and sc; pair
  // 1 loads qhy of block b; d of b + phases follows the fold.
  uint4 qn[Rows] = {};
  uint2 sn[Rows] = {};
  std::uint32_t qhx[Rows] = {}, qhy[Rows] = {}, sc[Rows] = {};
  std::uint16_t dv[Rows] = {};
  if (first < blocks) {
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      qn[r] = ld.template operator()<uint4>(off[r]);
      sn[r] = ld.template operator()<uint2>(off[r] + sg_off[r]);
      qhx[r] = ld.template operator()<std::uint32_t>(off[r] + qh_off[r]);
      sc[r] = ld.template operator()<std::uint32_t>(off[r] + sc_off[r]);
      dv[r] = ld.template operator()<std::uint16_t>(off[r] + d_off[r]);
    }
  }
  // Stage the codebook while the first weights are in flight (a kN wave may
  // own a single K block, so this latency is not hidden by the loop).
  snb_detail::SnbStageCodebook<true>(codebook, threadIdx.x, kThreads);
  __syncthreads();

  float sums[Rows][Batch] = {};
  for (std::uint32_t b = first; b < blocks; b += phases) {
    // Loads for block b + phases; the last iteration re-reads block b
    // instead (in bounds, never used).
    const bool more = b + phases < blocks;
    std::uint32_t nxt[Rows];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r)
      nxt[r] = more ? off[r] + step[r] : off[r];
    const std::int8_t* const act_block =
        xq + (static_cast<std::size_t>(b) * kSnIq3Elements);
    int isum[Rows][Batch] = {};
#pragma unroll
    for (std::uint32_t pair = 0; pair < 4U; ++pair) {
      std::uint32_t word[Rows][2][8];
      int scale_int[Rows][2];
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        const uint4 qc = qn[r];
        const uint2 sw = sn[r];
        const std::uint32_t qh_word = pair < 2U ? qhx[r] : qhy[r];
#pragma unroll
        for (std::uint32_t h = 0; h < 2U; ++h) {
          const std::uint32_t s = (2U * pair) + h;
          const std::uint32_t signs = h == 0U ? sw.x : sw.y;
          const std::uint32_t even = (signs << 1U) & 0x1E1E1E1EU;
          const std::uint32_t odd = (signs >> 3U) & 0x1E1E1E1EU;
          const std::uint32_t qh_shift = 8U * (s & 3U);
          const std::uint32_t qh_lo =
              ((qh_word >> qh_shift) & 0x0FU) * 0x00204081U;
          const std::uint32_t qh_hi =
              ((qh_word >> (qh_shift + 4U)) & 0x0FU) * 0x00204081U;
          const std::uint32_t tag[2] = {
              (qh_lo & 0x01010101U) |
                  __builtin_amdgcn_perm(odd, even, 0x05010400U),
              (qh_hi & 0x01010101U) |
                  __builtin_amdgcn_perm(odd, even, 0x07030602U)};
          // qs words 2s, 2s+1 = words 2h, 2h+1 of this pair's chunk.
          const std::uint32_t qs_w[2] = {h == 0U ? qc.x : qc.z,
                                         h == 0U ? qc.y : qc.w};
#pragma unroll
          for (std::uint32_t i = 0; i < 8U; ++i) {
            const std::uint32_t sel =
                0x0C0C0000U | ((4U + (i & 3U)) << 8U) | (i & 3U);
            word[r][h][i] = codebook[__builtin_amdgcn_perm(
                tag[i >> 2U], qs_w[i >> 2U], sel)];
          }
          scale_int[r][h] =
              static_cast<int>(1U + (2U * ((sc[r] >> (4U * s)) & 0x0FU)));
        }
        // Next pair's registers (pair 3: pair 0 of block b + phases).
        const std::uint32_t np = (pair + 1U) & 3U;
        const std::uint32_t src = pair == 3U ? nxt[r] : off[r];
        qn[r] = ld.template operator()<uint4>(src + (np * chunk[r]));
        sn[r] = ld.template operator()<uint2>(src + sg_off[r] +
                                              ((np >> 1U) * chunk[r]) +
                                              ((np & 1U) * 8U));
        if (pair == 1U)
          qhy[r] =
              ld.template operator()<std::uint32_t>(off[r] + qh_off[r] + 4U);
        if (pair == 3U) {
          qhx[r] = ld.template operator()<std::uint32_t>(nxt[r] + qh_off[r]);
          sc[r] = ld.template operator()<std::uint32_t>(nxt[r] + sc_off[r]);
        }
      }
#pragma unroll
      for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
        constexpr std::size_t kQuad = 4U;
        uint4 a[kQuad][4];
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4* const p = reinterpret_cast<const uint4*>(
              act_block + ((t0 + j) * k) + (pair * 64U));
#pragma unroll
          for (std::uint32_t c = 0; c < 4U; ++c)
            a[j][c] = p[c];
        }
        // All of the quad's s_loads (and the pair's ds_loads) issue before the
        // first dot4, so they share one lgkmcnt(0) instead of one each: SMEM
        // returns out of order, so a partial wait is never legal for it.
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
#pragma unroll
          for (std::uint32_t r = 0; r < Rows; ++r) {
#pragma unroll
            for (std::uint32_t h = 0; h < 2U; ++h) {
              const uint4 lo = a[j][2U * h];
              const uint4 hi = a[j][(2U * h) + 1U];
              const std::uint32_t* const wd = word[r][h];
              int dot = 0;
              dot = __builtin_amdgcn_sudot4(true, wd[0], true, lo.x, dot, false);
              dot = __builtin_amdgcn_sudot4(true, wd[1], true, lo.y, dot, false);
              dot = __builtin_amdgcn_sudot4(true, wd[2], true, lo.z, dot, false);
              dot = __builtin_amdgcn_sudot4(true, wd[3], true, lo.w, dot, false);
              dot = __builtin_amdgcn_sudot4(true, wd[4], true, hi.x, dot, false);
              dot = __builtin_amdgcn_sudot4(true, wd[5], true, hi.y, dot, false);
              dot = __builtin_amdgcn_sudot4(true, wd[6], true, hi.z, dot, false);
              dot = __builtin_amdgcn_sudot4(true, wd[7], true, hi.w, dot, false);
              isum[r][t0 + j] += __mul24(scale_int[r][h], dot);
            }
          }
        }
        // Keep the next quad's 64 SGPRs of activations out of this one.
        __builtin_amdgcn_sched_barrier(0);
      }
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float xd = xds[(t * num_sub) + (b * 16U)].x;
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        const float d = __half2float(__ushort_as_half(dv[r]));
        sums[r][t] += (d * xd) * static_cast<float>(isum[r][t]);
      }
    }
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      dv[r] = ld.template operator()<std::uint16_t>(nxt[r] + d_off[r]);
      off[r] = nxt[r];
    }
  }

  if constexpr (KPhases > 1U) {
    // Fixed-order cross-wave add: phase 0 + 1 + ... + KPhases-1.
    __syncthreads();  // every wave is done with the codebook
    float* const part = reinterpret_cast<float*>(codebook);
    const auto slot = [&](std::uint32_t p, std::uint32_t r, std::size_t t) {
      return (((((((p - 1U) * RowWaves) + row_wave) * Rows) + r) * Batch + t) *
              32U) +
             lane;
    };
    if (phase > 0U) {
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r)
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t)
          part[slot(phase, r, t)] = sums[r][t];
    }
    __syncthreads();
    if (phase == 0U) {
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r)
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
          for (std::uint32_t p = 1U; p < KPhases; ++p)
            sums[r][t] += part[slot(p, r, t)];
    }
  }
  if (ksplit == 1U) {
    if (phase == 0U) {
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r)
        if (row0[r] < m)
#pragma unroll
          for (std::size_t t = 0; t < Batch; ++t)
            y[(t * m) + row0[r]] = sums[r][t];
    }
    return;
  }
  // Cross-block split: partial[(split * Batch + t) * m + row].
  if (phase == 0U) {
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r)
      if (row0[r] < m)
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t)
          partial[(((split * Batch) + t) * m) + row0[r]] = sums[r][t];
  }
  __threadfence();  // release this block's partials before its ticket
  __syncthreads();
  // The barrier above also ends every LDS read, so codebook[0] is free.
  if (threadIdx.x == 0U)
    codebook[0] = __hip_atomic_fetch_add(&ticket[tile], 1U, __ATOMIC_ACQ_REL,
                                         __HIP_MEMORY_SCOPE_AGENT) ==
                  ksplit - 1U;
  __syncthreads();
  if (codebook[0] == 0U)
    return;
  __threadfence();  // acquire: the other blocks' partials, not stale L0 lines
  if (phase == 0U) {
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r)
      if (row0[r] < m)
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t) {
          float v = partial[(t * m) + row0[r]];
          for (std::uint32_t sp = 1U; sp < ksplit; ++sp)
            v += partial[(((sp * Batch) + t) * m) + row0[r]];
          y[(t * m) + row0[r]] = v;
        }
  }
  if (threadIdx.x == 0U)
    __hip_atomic_store(&ticket[tile], 0U, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
}

// K split for the opus_b8_kN arm: the smallest ksplit whose total phase count
// ksplit * KPhases divides the K blocks (every wave gets the same number of
// blocks) and gives >= kOpusTargetBlocks blocks; if none reaches it, the
// largest such ksplit. kOpusTargetBlocks = 3 full rounds of the resident
// block slots: 20 WGPs x 4 blocks (the 32 KiB codebook caps a 128 KiB WGP at
// 4 blocks; 16-wave blocks at <= 96 VGPRs fit 4 x 16 = 64 waves).
// ponytail: hard-coded for the 40-CU gfx1151; query the CU count if ported.
inline constexpr std::uint32_t kOpusTargetBlocks = 240U;
[[nodiscard]] inline std::uint32_t OpusPickKSplit(std::uint32_t tiles,
                                                  std::uint32_t kblocks,
                                                  std::uint32_t kphases) {
  std::uint32_t best = 1U;
  for (std::uint32_t d = 1U; d * kphases <= kblocks; ++d) {
    if (kblocks % (d * kphases) != 0U)
      continue;
    best = d;
    if (tiles * d >= kOpusTargetBlocks)
      break;
  }
  return best;
}

// Partials + tickets for ksplit > 1. Tickets start at 0 and every last block
// resets its own, so they are zeroed once per allocation.
// ponytail: one grow-only process-wide buffer, never freed; not safe under
// stream capture or concurrent launches on two streams. Bench arm only; a
// served path needs a caller-owned workspace.
[[nodiscard]] inline bool OpusSplitScratch(std::size_t floats,
                                           std::size_t tickets,
                                           float** partial,
                                           std::uint32_t** ticket) {
  static float* p = nullptr;
  static std::uint32_t* t = nullptr;
  static std::size_t p_size = 0;
  static std::size_t t_size = 0;
  if (floats > p_size) {
    (void)hipFree(p);
    p = nullptr;
    p_size = 0;
    if (hipMalloc(&p, floats * sizeof(float)) != hipSuccess)
      return false;
    p_size = floats;
  }
  if (tickets > t_size) {
    (void)hipFree(t);
    t = nullptr;
    t_size = 0;
    if (hipMalloc(&t, tickets * sizeof(std::uint32_t)) != hipSuccess ||
        hipMemset(t, 0, tickets * sizeof(std::uint32_t)) != hipSuccess)
      return false;
    t_size = tickets;
  }
  *partial = p;
  *ticket = t;
  return true;
}

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         std::uint32_t Rows, bool SplitK>
void LaunchOpusIq3sB8Stream(const void* w, float* y,
                            std::int8_t* xq, float2* xds, std::uint32_t m,
                            std::uint32_t k, hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = RowWaves * Rows * 32U;
  const std::uint32_t tiles = (m + kRowsPerBlock - 1U) / kRowsPerBlock;
  std::uint32_t ksplit = 1U;
  float* partial = nullptr;
  std::uint32_t* ticket = nullptr;
  if constexpr (SplitK) {
    ksplit = OpusPickKSplit(tiles, k / static_cast<std::uint32_t>(kSnIq3Elements),
                            KPhases);
    if (ksplit > 1U &&
        !OpusSplitScratch(static_cast<std::size_t>(ksplit) * Batch * m, tiles,
                          &partial, &ticket))
      ksplit = 1U;  // no scratch: same result, no K split across blocks
  }
  OpusIq3sB8StreamKernel<Batch, RowWaves, KPhases, Rows>
      <<<tiles * ksplit, RowWaves * KPhases * 32U, 0, stream>>>(
          static_cast<const std::uint8_t*>(w), xq, xds, y, m, k, ksplit,
          partial, ticket);
}

// Waves/SIMD the ring kernel is compiled for (VGPR budget 1536 / waves), from
// its gfx1151 ISA with no cap: Depth 1 needs 71-91 VGPR (16 waves), Depth 2
// 94/96 at Batch 1/2 (16 waves) and 98-114 at Batch 3-8 (12 waves, 3 blocks a
// WGP). A tighter cap spills into scratch, and a scratch reload is a VMEM op:
// its vmcnt wait drains the whole ring.
[[nodiscard]] constexpr std::uint32_t OpusRingWaves(std::size_t batch,
                                                    std::uint32_t depth) {
  return depth == 1U || batch <= 2U ? 16U : 12U;
}

// OpusIq3sB8RingKernel ("opus_b8_stream"). OpusIq3sB8Kernel's
// math, decode, activation s_loads, sums and phase reduction, unchanged (same
// per-row add order, so the output is bit-identical to opus_b8_sgpr_k4 at the
// same RowWaves x KPhases). Only the weight-load path changes.
//
// WHY (measured): B1 and B8 of OpusIq3sB8Kernel run the SAME 9 loads
// (same addresses, depth-1 prefetch, 16 waves/SIMD, same grid), yet B1 streams
// 221 GB/s and B8 198 (387 us vs 346 us at m34816). In the B8 ISA the
// prefetch of block b+1 is issued at the top of block b and drained by an
// s_waitcnt vmcnt(0) before b's fold (cur.d and next.d share one VGPR as
// d16 halves), then copied (26 v_dual_mov). With ~1350 SIMD cycles of compute
// per block the prefetch lands long before it is needed, so (the hypothesis
// this arm tests) a B8 wave holds 0 bytes in flight for most of its block and
// DRAM sees far fewer queued requests than at B1, where compute is ~1/4 of a
// block's time and the next block is in flight almost all the time.
//
// Depth 2 (opus_b8_stream): a 2-block register ring, no copy. Sets X and Y
// each hold one block (28 VGPR). The loop is unrolled by two blocks so the
// sets keep static registers; while block b is decoded from set S, each piece
// of S is refilled with block b + 2*KPhases right after its last reader: qs
// chunk c after sub-block 2c+1, sign chunk c after sub-block 4c+3, qh/sc
// after sub-block 7, d after the fold. A wave keeps ~1-2 blocks in flight
// (up to 7 KB vs k4's <= 3.5 KB) and every wait in the loop is a partial
// vmcnt. Cost: 101-114 VGPR at Batch 3-8 = 12 waves/SIMD (OpusRingWaves).
// Depth 1 (opus_b8_stream_d1, control arm): k4's one-block prefetch (whole
// next block issued at the top, 16 waves/SIMD) with the same body, so
// stream - stream_d1 isolates the ring depth (and its occupancy cost).
// Both: d is an aligned u32 (no shared d16 VGPR) and the body decodes one
// sub-block at a time behind one lgkmcnt(0) each (9-10 drains a block, k4 18).
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         std::uint32_t Depth>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(OpusRingWaves(Batch, Depth)))) void
    OpusIq3sB8RingKernel(
        const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(Depth == 1U || Depth == 2U);
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  static_assert((KPhases - 1U) * RowWaves * Batch * 32U <= 8192U);
  __shared__ std::uint32_t codebook[8192];

  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t row0 =
      (((blockIdx.x * RowWaves) + row_wave) * 32U) + lane;
  const std::uint32_t row = row0 < m ? row0 : m - 1U;
  const std::uint32_t group = row / kSnIq3RowsPerGroup;
  const std::uint32_t in_group = row % kSnIq3RowsPerGroup;
  // Full 8-row groups only (the launcher checks m % 8 == 0): record, chunk
  // and plane offsets are then constants, the block offset b * kRec is
  // wave-uniform (SGPR base), and each lane needs just 4 constant VGPR
  // offsets; every weight load is global_load vaddr, saddr, offset:imm.
  constexpr std::uint32_t kN = kSnIq3RowsPerGroup;
  constexpr std::uint32_t kRec = static_cast<std::uint32_t>(SnIq3BlockStride(kN));
  constexpr std::uint32_t kChunk = 16U * kN;
  const std::uint32_t gbase =
      group * static_cast<std::uint32_t>(SnIq3GroupStride(blocks));
  const std::uint32_t lq = gbase + (in_group * 16U);  // qs / signs
  const std::uint32_t lh = gbase + (in_group * 8U);   // qh
  const std::uint32_t ls = gbase + (in_group * 4U);   // scales
  // d: the 4-byte-aligned word of rows (in_group & ~1, in_group | 1); a u32
  // load, not u16, so two sets' d never share one VGPR as d16 halves (a false
  // dependency that forced a vmcnt(0) at every fold in k4's ISA).
  const std::uint32_t lv = gbase + ((in_group & ~1U) * 2U);
  const std::uint32_t num_sub = k / 16U;

  struct Weights {
    uint4 q[4];
    uint4 s[2];
    uint2 qh;
    std::uint32_t sc;
    std::uint32_t d;  // fp16 bits of rows in_group & ~1 (low), | 1 (high)
  };
  // Plane piece `off` (a constant) of block b for this lane.
  const auto ld = [&]<typename T>(std::uint32_t b, std::uint32_t off,
                                  std::uint32_t lane_off) {
    const std::uint8_t* const wb = w + (static_cast<std::size_t>(b) * kRec);
    return *reinterpret_cast<const T*>(wb + lane_off + off);
  };
  constexpr std::uint32_t kSg = kN * kSnIq3SignsOffset;
  constexpr std::uint32_t kQh = kN * kSnIq3QhOffset;
  constexpr std::uint32_t kSc = kN * kSnIq3ScalesOffset;
  constexpr std::uint32_t kD = kN * kSnIq3DOffset;
  const auto load = [&](std::uint32_t b, Weights& v) {
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      v.q[c] = ld.template operator()<uint4>(b, c * kChunk, lq);
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      v.s[c] = ld.template operator()<uint4>(b, kSg + (c * kChunk), lq);
    v.qh = ld.template operator()<uint2>(b, kQh, lh);
    v.sc = ld.template operator()<std::uint32_t>(b, kSc, ls);
    v.d = ld.template operator()<std::uint32_t>(b, kD, lv);
  };

  Weights x_set{}, y_set{};
  // First Depth blocks go out before the codebook staging (as kN does).
  if (phase < blocks)
    load(phase, x_set);
  if (Depth == 2U && phase + KPhases < blocks)
    load(phase + KPhases, y_set);
  snb_detail::SnbStageCodebook<true>(codebook, threadIdx.x, kThreads);
  __syncthreads();

  float sums[Batch] = {};
  // One K block from set `cur`; Refill: also reload `cur` with block
  // b + Depth*KPhases piece by piece. Refill is a template flag, not a runtime
  // branch: a branch around each refill load split the block and the
  // allocator spilled (ISA: 668 B scratch at B8).
  const auto block = [&]<bool Refill>(Weights& cur, std::uint32_t b)
                         __attribute__((always_inline)) {
    const std::uint32_t b2 = b + (Depth * KPhases);
    // `now` = this block's pieces. Depth 1: a snapshot, and the whole next
    // block is issued at once into cur (k4's schedule: tried refilling each
    // piece in place at its first reader, but the loop-carried registers then
    // need a copy at the latch, i.e. a vmcnt(0) at the end of every block,
    // and the last piece had 1/4 block of lookahead). Depth 2: plain names;
    // each piece of cur is refilled after its last reader below.
    const Weights now = cur;
    if constexpr (Refill && Depth == 1U)
      load(b2, cur);
    const uint2 qh = now.qh;
    const std::uint32_t sc = now.sc;
    const std::uint32_t d_bits = now.d;
    const std::int8_t* const act_block =
        xq + (static_cast<std::size_t>(b) * kSnIq3Elements);
    int isum[Batch] = {};
    // One sub-block per step (8 words live, not k4's 16: the ring has no VGPR
    // to spare). Its Batch tokens x 32 activation bytes are Batch s_load_b256
    // (<= 64 SGPRs) behind one lgkmcnt(0): 8 drains a block (k4: 18). `s` is
    // a template constant so cur.q[s >> 1] is a constant index when SROA runs
    // (an unrolled loop index left both sets as one alloca vector: 187 VGPR +
    // spills). Same int adds per token in the same sub-block order.
    const auto sub_step = [&]<std::uint32_t s>() __attribute__((always_inline)) {
      const uint4 qc = now.q[s >> 1U];
      const uint4 sgc = now.s[s >> 2U];
      const std::uint32_t sw = (s & 3U) == 0U   ? sgc.x
                               : (s & 3U) == 1U ? sgc.y
                               : (s & 3U) == 2U ? sgc.z
                                                : sgc.w;
      const std::uint32_t even = (sw << 1U) & 0x1E1E1E1EU;
      const std::uint32_t odd = (sw >> 3U) & 0x1E1E1E1EU;
      const std::uint32_t qh_word = s < 4U ? qh.x : qh.y;
      constexpr std::uint32_t kQhShift = 8U * (s & 3U);
      const std::uint32_t qh_lo = ((qh_word >> kQhShift) & 0x0FU) * 0x00204081U;
      const std::uint32_t qh_hi =
          ((qh_word >> (kQhShift + 4U)) & 0x0FU) * 0x00204081U;
      const std::uint32_t tag[2] = {
          (qh_lo & 0x01010101U) | __builtin_amdgcn_perm(odd, even, 0x05010400U),
          (qh_hi & 0x01010101U) |
              __builtin_amdgcn_perm(odd, even, 0x07030602U)};
      // qs words 2s, 2s+1 = words 2(s&1), 2(s&1)+1 of qs chunk s >> 1.
      const std::uint32_t qs_w[2] = {(s & 1U) == 0U ? qc.x : qc.z,
                                     (s & 1U) == 0U ? qc.y : qc.w};
      std::uint32_t word[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8U; ++i) {
        const std::uint32_t sel =
            0x0C0C0000U | ((4U + (i & 3U)) << 8U) | (i & 3U);
        word[i] = codebook[__builtin_amdgcn_perm(tag[i >> 2U], qs_w[i >> 2U],
                                                 sel)];
      }
      const int scale_int =
          static_cast<int>(1U + (2U * ((sc >> (4U * s)) & 0x0FU)));
      if constexpr (Refill && Depth == 2U) {  // last reader
        if constexpr ((s & 1U) == 1U)
          cur.q[s >> 1U] =
              ld.template operator()<uint4>(b2, (s >> 1U) * kChunk, lq);
        if constexpr ((s & 3U) == 3U)
          cur.s[s >> 2U] = ld.template operator()<uint4>(
              b2, kSg + ((s >> 2U) * kChunk), lq);
        if constexpr (s == 7U) {
          cur.qh = ld.template operator()<uint2>(b2, kQh, lh);
          cur.sc = ld.template operator()<std::uint32_t>(b2, kSc, ls);
        }
      }
      // The empty asm makes the sub-block offset opaque (an SGPR the compiler
      // cannot see through), so the load/store optimizer cannot merge two
      // sub-blocks' 32 B loads into one 64 B load: that hoists the next
      // sub-block's 64 SGPRs into this one (ISA: 3000+ SGPR lane spills).
      std::uint32_t sub_off = s * 32U;
      asm volatile("" : "+s"(sub_off));
      const std::int8_t* const act_sub = act_block + sub_off;
      uint4 a[Batch][2];
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        const uint4* const p = reinterpret_cast<const uint4*>(act_sub + (t * k));
        a[t][0] = p[0];
        a[t][1] = p[1];
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        const uint4 lo = a[t][0];
        const uint4 hi = a[t][1];
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, word[0], true, lo.x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, word[1], true, lo.y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, word[2], true, lo.z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, word[3], true, lo.w, dot, false);
        dot = __builtin_amdgcn_sudot4(true, word[4], true, hi.x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, word[5], true, hi.y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, word[6], true, hi.z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, word[7], true, hi.w, dot, false);
        isum[t] += __mul24(scale_int, dot);
      }
      // Keep the next sub-block's activation SGPRs out of this one.
      __builtin_amdgcn_sched_barrier(0);
    };
    sub_step.template operator()<0U>();
    sub_step.template operator()<1U>();
    sub_step.template operator()<2U>();
    sub_step.template operator()<3U>();
    sub_step.template operator()<4U>();
    sub_step.template operator()<5U>();
    sub_step.template operator()<6U>();
    sub_step.template operator()<7U>();
    const float d = __half2float(__ushort_as_half(static_cast<std::uint16_t>(
        d_bits >> ((__lane_id() & 1U) * 16U))));
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float xd = xds[(t * num_sub) + (b * 16U)].x;
      sums[t] += (d * xd) * static_cast<float>(isum[t]);
    }
    if constexpr (Refill && Depth == 2U)
      cur.d = ld.template operator()<std::uint32_t>(b2, kD, lv);
  };

  // This wave's blocks: phase, phase + KPhases, ... (rem of them).
  std::uint32_t b = phase;
  std::uint32_t rem =
      phase < blocks ? ((blocks - phase) + KPhases - 1U) / KPhases : 0U;
  if constexpr (Depth == 1U) {
    // One set: snapshot, then the whole next block is issued into it.
    for (; rem >= 2U; --rem, b += KPhases)
      block.template operator()<true>(x_set, b);
    if (rem == 1U)
      block.template operator()<false>(x_set, b);
    rem = 0U;
  }
  // Depth 2: the main loop runs while both refills exist (rem >= 4); the 0-3
  // left over drain the ring without refills (once per wave).
  for (; Depth == 2U && rem >= 4U; rem -= 2U, b += 2U * KPhases) {
    block.template operator()<true>(x_set, b);
    block.template operator()<true>(y_set, b + KPhases);
  }
  bool y_next = false;
  if (rem == 3U) {  // X's refill (block b + 2*KPhases) is the wave's last
    block.template operator()<true>(x_set, b);
    b += KPhases;
    rem = 2U;
    y_next = true;
  }
  for (std::uint32_t i = 0; i < rem; ++i, b += KPhases) {
    Weights cur = (y_next != ((i & 1U) != 0U)) ? y_set : x_set;
    block.template operator()<false>(cur, b);
  }

  if constexpr (KPhases > 1U) {
    // Fixed-order cross-wave add: phase 0 + 1 + ... + KPhases-1.
    __syncthreads();  // every wave is done with the codebook
    float* const partial = reinterpret_cast<float*>(codebook);
    const auto slot = [&](std::uint32_t p, std::size_t t) {
      return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
    };
    if (phase > 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
        partial[slot(phase, t)] = sums[t];
    }
    __syncthreads();
    if (phase == 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
        for (std::uint32_t p = 1U; p < KPhases; ++p)
          sums[t] += partial[slot(p, t)];
    }
  }
  if (phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
}

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         std::uint32_t Depth>
void LaunchOpusIq3sB8Ring(const void* w, float* y, std::int8_t* xq,
                          float2* xds, std::uint32_t m, std::uint32_t k,
                          hipStream_t stream) {
  // The ring kernel needs full 8-row groups and 32-bit lane offsets; any
  // other shape runs OpusIq3sB8Kernel, the same math (bit-identical sums).
  if (m % kSnIq3RowsPerGroup != 0U ||
      SnIq3PackedBytes(m, k) > 0xFFFFFFFFULL) {
    LaunchOpusIq3sB8Typed<Batch, RowWaves, KPhases>(w, y, xq, xds, m, k,
                                                    stream);
    return;
  }
  constexpr std::uint32_t kRowsPerBlock = RowWaves * 32U;
  OpusIq3sB8RingKernel<Batch, RowWaves, KPhases, Depth>
      <<<(m + kRowsPerBlock - 1U) / kRowsPerBlock, RowWaves * KPhases * 32U, 0,
         stream>>>(static_cast<const std::uint8_t*>(w), xq, xds, y, m, k);
}

// Floor probes (bench arms only, never served; their outputs
// are meaningless and rel_error is not a quality signal). Same grid, block and
// LDS as opus_b8_sgpr_k4 (OpusIq3sB8Kernel<Batch, 4, 4>: ceil(m/128) blocks of
// 16 waves, 32 KiB static LDS, the same KPhases LDS reduction and y store), so
// each times one part of that kernel's floor:
//   kNull      no codebook staging; one dword of weights per wave (uniform
//              address: one DRAM round trip). Time = the fixed cost of one
//              launch of this grid: dispatch, wave launch, one DRAM latency,
//              reduction, y store, kernel boundary.
//   kNullStage kNull + k4's codebook staging and barrier. Minus kNull = the
//              staging cost (paid once per block: 272 times at m34816).
//   kDot4      kNullStage + k4's v_dot4 count per K block (64 x Batch:
//              8 sub-blocks x Batch tokens x 8 words), registers only (VGPR
//              weights, SGPR activations, no VMEM / LDS / SMEM in the loop).
//              (kDot4 - kNullStage) x 1.7 GHz / (dot4s on the busiest SIMD) =
//              cycles per v_dot4_i32_iu8: 1 = full rate, 2 = half. Decides the
//              B8 floor (measured): k4's 1663 cycles per wave-block hold
//              512 dot4 and 374 other VALU.
//
// Round 5 rate arms (same prologue/epilogue as kNullStage, so subtract
// opus_b8_null_cb): the floor probes measured kDot4 at 2.26-2.35 cycles per v_dot4 at
// B1/B4/B8 (1, 4, 8 accumulators per wave), while WMMA iu8 measured 94% of one
// dot4-equivalent per clock (WMMA rate microbench). Each arm below changes ONE thing against
// kRateDot4S and issues exactly one instruction kind (inline asm, so the
// compiler can neither fold, pair (VOPD) nor reorder them): 512 instructions
// per K block per wave, round-robin over 8 independent accumulators (each
// dependent pair is 8 instructions apart, more than the 5-cycle VALU
// dependency of ISA 5.7's s_delay_alu example). The WMMA arms issue 32 WMMA
// per K block (B8 padded to 16 = 32 rows x 256 K x 16 tokens) over 4
// accumulators. Cycles per instruction = (ms - null_cb ms) x 1.7e6 /
// (wave-blocks on the busiest SIMD x 512, or x 32 for WMMA).
//   kRateDot4S    v_dot4_i32_iu8 v, v, s   (= kDot4 in asm: harness check)
//   kRateDot4V    v_dot4_i32_iu8 v, v, v   (suspect: SGPR operand)
//   kRateDot4U8   v_dot4_u32_u8 v, v, s    (suspect: signed / neg_lo form)
//   kRateDot8     v_dot8_i32_iu4 v, v, s   (another int DOT op)
//   kRateDot2     v_dot2_f32_f16 v, v, v   (float DOT op, VOP3P, 8 B)
//   kRateDot2Acc  v_dot2acc_f32_f16 v, v   (same op, VOP2, 4 B)
//   kRateFma      v_fma_f32 v, v, v        (non-DOT VOP3, 8 B)
//   kRateFmac     v_fmac_f32 v, v          (non-DOT VOP2, 4 B: 1/clk baseline)
//   kRateWmma8    v_wmma_i32_16x16x16_iu8  (calculator: 32 cycles)
//   kRateWmma4    v_wmma_i32_16x16x16_iu4  (calculator: 16 cycles)
//   kRateDualVV   v_dual_dot2acc_f32_f16 x2, all VGPR (VOPD: 2 ops per instr)
//   kRateDualSV   same, SGPR src0 (activations as k4's s_load operands)
// The dual arms issue 512 VOPD per K block = 1024 dot2acc over 16
// accumulators (each pair 8 VOPD apart). Registers are pinned (v60-v79) so
// the VOPD bank rules hold: X dst even / Y dst odd, X/Y src0 and vsrc1 in
// different banks (reg % 4); the assembler rejects an illegal pair. One slot
// per pair -> time = r_dot2acc's; two slots -> 2x r_dot2acc (b8gap a1/a10).
enum class OpusFloorProbe {
  kNull, kNullStage, kDot4,
  kRateDot4S, kRateDot4V, kRateDot4U8, kRateDot8, kRateDot2, kRateDot2Acc,
  kRateFma, kRateFmac, kRateWmma8, kRateWmma4, kRateDualVV, kRateDualSV,
};

namespace snb_detail {

// One instruction of a rate arm: acc = op(x, s, acc). x is a VGPR, s an SGPR
// (the VGPR-only arms use y, a second VGPR, instead).
template<OpusFloorProbe Op>
__device__ __forceinline__ void RateStep(std::uint32_t& acc, std::uint32_t x,
                                         std::uint32_t y, std::uint32_t s) {
  if constexpr (Op == OpusFloorProbe::kRateDot4S)
    asm volatile("v_dot4_i32_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]"
                 : "+v"(acc) : "v"(x), "s"(s));
  else if constexpr (Op == OpusFloorProbe::kRateDot4V)
    asm volatile("v_dot4_i32_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]"
                 : "+v"(acc) : "v"(x), "v"(y));
  else if constexpr (Op == OpusFloorProbe::kRateDot4U8)
    asm volatile("v_dot4_u32_u8 %0, %1, %2, %0" : "+v"(acc) : "v"(x), "s"(s));
  else if constexpr (Op == OpusFloorProbe::kRateDot8)
    asm volatile("v_dot8_i32_iu4 %0, %1, %2, %0 neg_lo:[1,1,0]"
                 : "+v"(acc) : "v"(x), "s"(s));
  else if constexpr (Op == OpusFloorProbe::kRateDot2)
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(x), "v"(y));
  else if constexpr (Op == OpusFloorProbe::kRateDot2Acc)
    asm volatile("v_dot2acc_f32_f16_e32 %0, %1, %2"
                 : "+v"(acc) : "v"(x), "v"(y));
  else if constexpr (Op == OpusFloorProbe::kRateFma)
    asm volatile("v_fma_f32 %0, %1, %2, %0" : "+v"(acc) : "v"(x), "v"(y));
  else if constexpr (Op == OpusFloorProbe::kRateFmac)
    asm volatile("v_fmac_f32_e32 %0, %1, %2" : "+v"(acc) : "v"(x), "v"(y));
}

}  // namespace snb_detail

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         OpusFloorProbe Mode>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(16))) void OpusIq3sB8FloorProbeKernel(
        const std::uint8_t* __restrict__ w, float* __restrict__ y,
        std::uint32_t m, std::uint32_t k) {
  static_assert(Batch >= 1 && Batch <= 8);
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert((KPhases - 1U) * RowWaves * Batch * 32U <= 8192U);
  __shared__ std::uint32_t codebook[8192];
  if constexpr (Mode != OpusFloorProbe::kNull) {
    snb_detail::SnbStageCodebook<true>(codebook, threadIdx.x, kThreads);
    __syncthreads();
  }
  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t wave_row = ((blockIdx.x * RowWaves) + row_wave) * 32U;
  const std::uint32_t row0 = wave_row + lane;
  // The wave's first group, its phase's first block record: k4's first load.
  const std::uint32_t first = wave_row < m ? wave_row : m - 1U;
  const std::uint32_t group = first / kSnIq3RowsPerGroup;
  const std::uint32_t rows_left = m - (group * kSnIq3RowsPerGroup);
  const std::uint32_t n =
      rows_left < kSnIq3RowsPerGroup ? rows_left : kSnIq3RowsPerGroup;
  std::uint32_t v = 0;
  if (phase < blocks)
    v = *reinterpret_cast<const std::uint32_t*>(
        w + (static_cast<std::size_t>(group) * SnIq3GroupStride(blocks)) +
        (static_cast<std::size_t>(phase) * SnIq3BlockStride(n)));

  float sums[Batch];
  if constexpr (Mode == OpusFloorProbe::kDot4) {
    // 8 weight words per lane (VGPR), 8 activation words per wave (SGPR, as
    // k4's s_load operands). One chain per token, carried across sub-blocks
    // and K blocks, so no dot4 can be hoisted or merged; Batch independent
    // chains per wave, 16 waves per SIMD to hide the dependent-issue latency.
    std::uint32_t wd[8];
    std::uint32_t a[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8U; ++i) {
      wd[i] = (row0 * 0x9E3779B9U) ^ (v + (i * 0x01010101U));
      a[i] = __builtin_amdgcn_readfirstlane((v * (i + 3U)) + k);
    }
    int acc[Batch] = {};
    for (std::uint32_t b = phase; b < blocks; b += KPhases) {
#pragma unroll
      for (std::uint32_t s = 0; s < 8U; ++s)
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
          for (std::uint32_t i = 0; i < 8U; ++i)
            acc[t] = __builtin_amdgcn_sudot4(
                true, wd[i], true, a[(i + t + s) & 7U], acc[t], false);
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      sums[t] = static_cast<float>(acc[t]);
  } else if constexpr (Mode == OpusFloorProbe::kRateWmma8 ||
                       Mode == OpusFloorProbe::kRateWmma4) {
    using V8 = int __attribute__((ext_vector_type(8)));
    using V4 = int __attribute__((ext_vector_type(4)));
    using V2 = int __attribute__((ext_vector_type(2)));
    const int seed = static_cast<int>((row0 * 0x9E3779B9U) ^ v);
    V8 acc[4] = {};
    for (std::uint32_t b = phase; b < blocks; b += KPhases) {
#pragma unroll
      for (std::uint32_t r = 0; r < 8U; ++r)
#pragma unroll
        for (std::uint32_t c = 0; c < 4U; ++c) {
          const int q = seed + static_cast<int>((r & 1U) + c);
          if constexpr (Mode == OpusFloorProbe::kRateWmma8)
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, V4{q, q + 1, q + 2, q + 3}, true,
                V4{q ^ 5, q ^ 6, q ^ 7, q ^ 9}, acc[c], false);
          else
            acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, V2{q, q + 1}, true, V2{q ^ 5, q ^ 6}, acc[c], false);
        }
    }
    // Every accumulator feeds every output, so no WMMA chain is dead at
    // Batch < 4.
    int tot = 0;
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      tot += acc[c][0] + acc[c][7];
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      sums[t] = static_cast<float>(tot + static_cast<int>(t));
  } else if constexpr (Mode == OpusFloorProbe::kRateDualVV ||
                       Mode == OpusFloorProbe::kRateDualSV) {
    float a0 = row0, a1 = row0 + 1U, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0,
          a7 = 0, a8 = 0, a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0,
          a15 = 0;
    const std::uint32_t x0 = (row0 * 0x9E3779B9U) ^ v, x1 = x0 ^ 0x3C003C00U,
                        w0 = x0 + 0x01010101U, w1 = w0 ^ 0x3C003C00U;
    const std::uint32_t s0 = __builtin_amdgcn_readfirstlane(v + k),
                        s1 = __builtin_amdgcn_readfirstlane((v * 3U) + k);
    for (std::uint32_t b = phase; b < blocks; b += KPhases) {
#pragma unroll
      for (std::uint32_t r = 0; r < 64U; ++r) {
#define SNB_DUAL8(S0, S1)                                                    \
  "v_dual_dot2acc_f32_f16 v60, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v61, " S1 \
  ", v79\n"                                                                  \
  "v_dual_dot2acc_f32_f16 v62, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v63, " S1 \
  ", v79\n"                                                                  \
  "v_dual_dot2acc_f32_f16 v64, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v65, " S1 \
  ", v79\n"                                                                  \
  "v_dual_dot2acc_f32_f16 v66, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v67, " S1 \
  ", v79\n"                                                                  \
  "v_dual_dot2acc_f32_f16 v68, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v69, " S1 \
  ", v79\n"                                                                  \
  "v_dual_dot2acc_f32_f16 v70, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v71, " S1 \
  ", v79\n"                                                                  \
  "v_dual_dot2acc_f32_f16 v72, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v73, " S1 \
  ", v79\n"                                                                  \
  "v_dual_dot2acc_f32_f16 v74, " S0 ", v78 :: v_dual_dot2acc_f32_f16 v75, " S1 \
  ", v79\n"
#define SNB_DUAL_ACCS                                                        \
  "+{v60}"(a0), "+{v61}"(a1), "+{v62}"(a2), "+{v63}"(a3), "+{v64}"(a4),     \
      "+{v65}"(a5), "+{v66}"(a6), "+{v67}"(a7), "+{v68}"(a8), "+{v69}"(a9), \
      "+{v70}"(a10), "+{v71}"(a11), "+{v72}"(a12), "+{v73}"(a13),           \
      "+{v74}"(a14), "+{v75}"(a15)
        if constexpr (Mode == OpusFloorProbe::kRateDualVV)
          asm volatile(SNB_DUAL8("v76", "v77")
                       : SNB_DUAL_ACCS
                       : "{v76}"(x0), "{v77}"(x1), "{v78}"(w0), "{v79}"(w1));
        else
          asm volatile(SNB_DUAL8("%16", "%17")
                       : SNB_DUAL_ACCS
                       : "s"(s0), "s"(s1), "{v78}"(w0), "{v79}"(w1));
#undef SNB_DUAL_ACCS
#undef SNB_DUAL8
      }
    }
    const float tot = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 +
                      a11 + a12 + a13 + a14 + a15;
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      sums[t] = tot + static_cast<float>(t);
  } else if constexpr (Mode >= OpusFloorProbe::kRateDot4S) {
    std::uint32_t acc[8];
    std::uint32_t x[8];
    std::uint32_t yv[8];
    std::uint32_t sv[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8U; ++i) {
      acc[i] = row0 + i;
      x[i] = (row0 * 0x9E3779B9U) ^ (v + (i * 0x01010101U));
      yv[i] = x[i] ^ 0x3C003C00U;
      sv[i] = __builtin_amdgcn_readfirstlane((v * (i + 3U)) + k);
    }
    for (std::uint32_t b = phase; b < blocks; b += KPhases) {
#pragma unroll
      for (std::uint32_t r = 0; r < 64U; ++r)
#pragma unroll
        for (std::uint32_t c = 0; c < 8U; ++c)
          snb_detail::RateStep<Mode>(acc[c], x[(r + c) & 7U], yv[(r + c) & 7U],
                                     sv[(r + c) & 7U]);
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      sums[t] = static_cast<float>(acc[t & 7U]);
  } else {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      sums[t] = static_cast<float>(v) * static_cast<float>(t + 1U);
  }

  // k4's epilogue, unchanged: fixed-order LDS phase reduction, then y.
  if constexpr (KPhases > 1U) {
    __syncthreads();
    float* const partial = reinterpret_cast<float*>(codebook);
    const auto slot = [&](std::uint32_t p, std::size_t t) {
      return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
    };
    if (phase > 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
        partial[slot(phase, t)] = sums[t];
    }
    __syncthreads();
    if (phase == 0U) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
        for (std::uint32_t p = 1U; p < KPhases; ++p)
          sums[t] += partial[slot(p, t)];
    }
  }
  if (phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
}

template<std::size_t Batch, OpusFloorProbe Mode>
void LaunchOpusIq3sB8FloorProbe(const void* w, float* y, std::uint32_t m,
                                std::uint32_t k, hipStream_t stream) {
  constexpr std::uint32_t kRowWaves = 4U;
  constexpr std::uint32_t kKPhases = 4U;
  constexpr std::uint32_t kRowsPerBlock = kRowWaves * 32U;
  OpusIq3sB8FloorProbeKernel<Batch, kRowWaves, kKPhases, Mode>
      <<<(m + kRowsPerBlock - 1U) / kRowsPerBlock, kRowWaves * kKPhases * 32U,
         0, stream>>>(static_cast<const std::uint8_t*>(w), y, m, k);
}

// Bench arms (all share the one sn_iq3.hpp layout; only the kernel geometry and
// the codebook arm differ).
//
// The 32 KiB sign-folded table caps the block at one per CU, so the number of
// *blocks* decides how many CUs are busy: blocks = m / (Waves * 32 / Split).
// The arms cover that: Split 1 is the brief's geometry (one lane per row, a
// wave per 32 rows) and needs m >= 20480 rows to keep 40 CUs busy, the same
// geometry with 4-wave blocks spreads over 40 CUs from m = 5120, and Split 4
// multiplies the waves per row so 5120 rows fill all 80 SIMDs. The plain-grid
// arm is the codebook comparison at the brief's geometry.
enum class SnbIq3Arm {
  kFoldR8S1W16,   // sign-folded codebook, 1 lane/row, 16 waves/block
  kFoldR8S1W4,    // same mapping, 4 waves/block (finer CU spread)
  kFoldR8S4W16,   // sign-folded, 4 K-phases, 16 waves/block
  kGridR8S1W16,   // plain 512-entry grid + NegateSelectedBytes (comparison)
  kOpusB8Sgpr,    // OpusIq3sB8Kernel, 16 row-waves, whole K per wave
  kOpusB8SgprK4,  // OpusIq3sB8Kernel, 4 row-waves x 4 K-phases per block
  kOpusB8Alu,     // kOpusB8SgprK4 with no weight loads: ceiling probe only
  kOpusB8K4TwoRow,  // OpusIq3sB8StreamKernel, 2 rows/lane, 4 x 4 per block
  kOpusB8KN,      // OpusIq3sB8StreamKernel, 1 row/lane, 4 x 4, K split by m
  kOpusB8Stream,    // OpusIq3sB8RingKernel, 2-block ring, 4 x 4 per block
  kOpusB8StreamD1,  // same kernel, 1-block in-place refill (depth control)
  kOpusB8Null,      // floor probe: k4 grid, no staging, 1 dword a wave, y
  kOpusB8NullStage,  // floor probe: kOpusB8Null + codebook staging
  kOpusB8Dot4,      // floor probe: kOpusB8NullStage + k4's dot4s, regs only
  kOpusB8RDot4S,  // rate arm OpusFloorProbe::kRateDot4S
  kOpusB8RDot4V,  // rate arm OpusFloorProbe::kRateDot4V
  kOpusB8RDot4U8,  // rate arm OpusFloorProbe::kRateDot4U8
  kOpusB8RDot8,  // rate arm OpusFloorProbe::kRateDot8
  kOpusB8RDot2,  // rate arm OpusFloorProbe::kRateDot2
  kOpusB8RDot2Acc,  // rate arm OpusFloorProbe::kRateDot2Acc
  kOpusB8RFma,  // rate arm OpusFloorProbe::kRateFma
  kOpusB8RFmac,  // rate arm OpusFloorProbe::kRateFmac
  kOpusB8RWmma8,  // rate arm OpusFloorProbe::kRateWmma8
  kOpusB8RWmma4,  // rate arm OpusFloorProbe::kRateWmma4
  kOpusB8RDualVV,  // rate arm OpusFloorProbe::kRateDualVV
  kOpusB8RDualSV,  // rate arm OpusFloorProbe::kRateDualSV
  kOpusB8K4Pad,    // kOpusB8SgprK4 with n = 8 compile-time (m % 8 == 0, else k4)
  kOpusB8K4Slope,  // kOpusB8SgprK4 + 128 chained v_and_b32 per K block (probe)
  kOpusB8K4Clk,    // kOpusB8SgprK4 + s_memrealtime/s_memtime per workgroup
  kOpusB8K4Act,    // kOpusB8SgprK4 on the block-major token-interleaved tile
  kOpusB8K4CutR,   // OpusIq3sB8CutKernel Idx 0: k4pad, prefetch copies removed
  kOpusB8K4Cut,    // OpusIq3sB8CutKernel Idx 1: + `<< 2` folded into the tag/qs perm
  kOpusB8K4Fold,   // k4pad + Idx 1 address fold only (attribution)
  kOpusB8K4Pad3,   // k4pad + 3-token activation tile at Batch 7, 8 (attribution)
  kOpusB8PActFix,  // ceiling probe: k4pad, activation addresses stuck on block 0
  kOpusB8PNoDot,   // ceiling probe: k4pad, each v_dot4 -> one v_add3_u32
  kOpusB8K4Early,  // k4pad, next-block loads first, unconditional (Little's-law test)
  kOpusB8K4Prio,   // k4pad + s_setprio 3/0 around the next-block loads
  kOpusB8K4Ep,     // both
  kOpusB8K4Pf,     // k4prio + Idx 1 address fold (stack test)
  kOpusB8K4Prio2,  // k4prio with s_setprio 2
  kOpusB8K4PrioW,  // priority 3 over pointer adds + loads
  kOpusB8VopdCvp,  // vopd_cv + s_setprio 3/0 around its next-block loads
  kOpusB8K4Sk,     // k4prio + split-K across WGs (grid.y), last-arriver combine
  kOpusB8K4Sk2,    // same, S forced to 2 (S = 1 if illegal)
  kOpusB8K4Sk4,    // same, S forced to 4
  kOpusB8CvSk,     // vopd_cv + split-K, auto S
  kOpusB8CvSk2,    // vopd_cv + split-K, S = 2
  kOpusB8CvSk4,    // vopd_cv + split-K, S = 4
  kOpusB8CvpSk,    // vopd_cvp (setprio pair) + split-K, auto S
  kOpusB8K4Ps,     // persistent stream-K, k4prio body (40 WGs x 1024 threads)
  kOpusB8CvPs,     // persistent stream-K, vopd_cv body
  kOpusB8K4Cut64,  // OpusIq3sB8CutKernel Idx 2: + 64 KiB table, no extra decode op
  kOpusB8VopdT16,  // OpusIq3sB8VopdKernel<..., false>: 64 KiB f16-pair table
  kOpusB8VopdCv,   // OpusIq3sB8VopdKernel<..., true>: biased bytes + perm/pk_fma
};

// Arms that read the f16 activation tile (xh) from the interleaved quantize.
inline bool SnbArmWantsHalfActs(SnbIq3Arm arm) noexcept {
  return arm == SnbIq3Arm::kOpusB8VopdT16 || arm == SnbIq3Arm::kOpusB8VopdCv ||
         arm == SnbIq3Arm::kOpusB8VopdCvp || arm == SnbIq3Arm::kOpusB8CvSk ||
         arm == SnbIq3Arm::kOpusB8CvSk2 || arm == SnbIq3Arm::kOpusB8CvSk4 ||
         arm == SnbIq3Arm::kOpusB8CvpSk || arm == SnbIq3Arm::kOpusB8CvPs;
}

inline const char* SnbIq3ArmName(SnbIq3Arm arm) noexcept {
  switch (arm) {
    case SnbIq3Arm::kFoldR8S1W16:
      return "snb_r8_fold";
    case SnbIq3Arm::kFoldR8S1W4:
      return "snb_r8_fold_w4";
    case SnbIq3Arm::kFoldR8S4W16:
      return "snb_r8_fold_s4";
    case SnbIq3Arm::kGridR8S1W16:
      return "snb_r8_grid";
    case SnbIq3Arm::kOpusB8Sgpr:
      return "opus_b8_sgpr";
    case SnbIq3Arm::kOpusB8SgprK4:
      return "opus_b8_sgpr_k4";
    case SnbIq3Arm::kOpusB8Alu:
      return "opus_b8_alu";
    case SnbIq3Arm::kOpusB8K4TwoRow:
      return "opus_b8_k4_2row";
    case SnbIq3Arm::kOpusB8KN:
      return "opus_b8_kN";
    case SnbIq3Arm::kOpusB8Stream:
      return "opus_b8_stream";
    case SnbIq3Arm::kOpusB8StreamD1:
      return "opus_b8_stream_d1";
    case SnbIq3Arm::kOpusB8Null:
      return "opus_b8_null";
    case SnbIq3Arm::kOpusB8NullStage:
      return "opus_b8_null_cb";
    case SnbIq3Arm::kOpusB8Dot4:
      return "opus_b8_dot4";
    case SnbIq3Arm::kOpusB8RDot4S:
      return "opus_b8_r_dot4s";
    case SnbIq3Arm::kOpusB8RDot4V:
      return "opus_b8_r_dot4v";
    case SnbIq3Arm::kOpusB8RDot4U8:
      return "opus_b8_r_dot4u8";
    case SnbIq3Arm::kOpusB8RDot8:
      return "opus_b8_r_dot8";
    case SnbIq3Arm::kOpusB8RDot2:
      return "opus_b8_r_dot2";
    case SnbIq3Arm::kOpusB8RDot2Acc:
      return "opus_b8_r_dot2acc";
    case SnbIq3Arm::kOpusB8RFma:
      return "opus_b8_r_fma";
    case SnbIq3Arm::kOpusB8RFmac:
      return "opus_b8_r_fmac";
    case SnbIq3Arm::kOpusB8RWmma8:
      return "opus_b8_r_wmma8";
    case SnbIq3Arm::kOpusB8RWmma4:
      return "opus_b8_r_wmma4";
    case SnbIq3Arm::kOpusB8RDualVV:
      return "opus_b8_r_dual_vv";
    case SnbIq3Arm::kOpusB8RDualSV:
      return "opus_b8_r_dual_sv";
    case SnbIq3Arm::kOpusB8K4Pad:
      return "opus_b8_k4pad";
    case SnbIq3Arm::kOpusB8K4Slope:
      return "opus_b8_k4slope";
    case SnbIq3Arm::kOpusB8K4Clk:
      return "opus_b8_k4clk";
    case SnbIq3Arm::kOpusB8K4Act:
      return "opus_b8_k4act";
    case SnbIq3Arm::kOpusB8K4CutR:
      return "opus_b8_k4cut_r";
    case SnbIq3Arm::kOpusB8K4Cut:
      return "opus_b8_k4cut";
    case SnbIq3Arm::kOpusB8K4Cut64:
      return "opus_b8_k4cut64";
    case SnbIq3Arm::kOpusB8PActFix:
      return "opus_b8_p_actfix";
    case SnbIq3Arm::kOpusB8PNoDot:
      return "opus_b8_p_nodot";
    case SnbIq3Arm::kOpusB8K4Early:
      return "opus_b8_k4early";
    case SnbIq3Arm::kOpusB8K4Prio:
      return "opus_b8_k4prio";
    case SnbIq3Arm::kOpusB8K4Ep:
      return "opus_b8_k4ep";
    case SnbIq3Arm::kOpusB8K4Pf:
      return "opus_b8_k4pf";
    case SnbIq3Arm::kOpusB8K4Prio2:
      return "opus_b8_k4prio2";
    case SnbIq3Arm::kOpusB8K4PrioW:
      return "opus_b8_k4prioW";
    case SnbIq3Arm::kOpusB8VopdCvp:
      return "opus_b8_vopd_cvp";
    case SnbIq3Arm::kOpusB8K4Sk:
      return "opus_b8_k4sk";
    case SnbIq3Arm::kOpusB8K4Sk2:
      return "opus_b8_k4sk2";
    case SnbIq3Arm::kOpusB8K4Sk4:
      return "opus_b8_k4sk4";
    case SnbIq3Arm::kOpusB8CvSk:
      return "opus_b8_cvsk";
    case SnbIq3Arm::kOpusB8CvSk2:
      return "opus_b8_cvsk2";
    case SnbIq3Arm::kOpusB8CvSk4:
      return "opus_b8_cvsk4";
    case SnbIq3Arm::kOpusB8CvpSk:
      return "opus_b8_cvpsk";
    case SnbIq3Arm::kOpusB8K4Ps:
      return "opus_b8_k4ps";
    case SnbIq3Arm::kOpusB8CvPs:
      return "opus_b8_cvps";
    case SnbIq3Arm::kOpusB8K4Fold:
      return "opus_b8_k4fold";
    case SnbIq3Arm::kOpusB8K4Pad3:
      return "opus_b8_k4pad3";
    case SnbIq3Arm::kOpusB8VopdT16:
      return "opus_b8_vopd_t16";
    case SnbIq3Arm::kOpusB8VopdCv:
      return "opus_b8_vopd_cv";
  }
  return "snb_r8_fold";
}

// quantize == false skips the activation quantize: the caller has already
// filled xq/xds from x with QuantizeActivationsQ8KDot4Kernel (bench
// GUFO_DFLASH_PREQUANT=1, the served path's one quantize per input).
template<std::size_t Batch>
void LaunchSnbIq3s(SnbIq3Arm arm, const void* w, const float* x, float* y,
                   std::int8_t* xq, float2* xds, std::uint32_t m,
                   std::uint32_t k, hipStream_t stream, bool quantize = true,
                   std::uint64_t* clk = nullptr, std::uint16_t* xh = nullptr,
                   float* sk_scratch = nullptr, std::uint32_t* sk_counter = nullptr,
                   std::size_t sk_scratch_floats = 0U) {
  static_assert(Batch >= 1 && Batch <= 8);
  if (quantize) {
    const std::uint32_t subs = static_cast<std::uint32_t>(Batch) * (k / 16U);
    if (arm == SnbIq3Arm::kOpusB8K4Act)
      QuantizeActivationsQ8KInterleavedKernel<<<(subs + 255U) / 256U, 256, 0,
                                                stream>>>(x, xq, xds, subs, k);
    else if (SnbArmWantsHalfActs(arm))
      QuantizeActivationsQ8KInterleavedKernel<<<(subs + 255U) / 256U, 256, 0,
                                                stream>>>(x, xq, xds, subs, k,
                                                          xh);
    else
      QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256, 0,
                                         stream>>>(x, xq, xds, subs);
  }
  switch (arm) {
    case SnbIq3Arm::kFoldR8S1W16:
      LaunchSnbIq3sTyped<Batch, 16, 1, true>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kFoldR8S1W4:
      LaunchSnbIq3sTyped<Batch, 4, 1, true>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kFoldR8S4W16:
      LaunchSnbIq3sTyped<Batch, 16, 4, true>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kGridR8S1W16:
      LaunchSnbIq3sTyped<Batch, 16, 1, false>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8Sgpr:
      LaunchOpusIq3sB8Typed<Batch, 16, 1>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8SgprK4:
      LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8Alu:
      LaunchOpusIq3sB8Typed<Batch, 4, 4, true>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4TwoRow:
      LaunchOpusIq3sB8Stream<Batch, 4, 4, 2, false>(w, y, xq, xds, m, k,
                                                    stream);
      break;
    case SnbIq3Arm::kOpusB8KN:
      LaunchOpusIq3sB8Stream<Batch, 4, 4, 1, true>(w, y, xq, xds, m, k,
                                                   stream);
      break;
    case SnbIq3Arm::kOpusB8Stream:
      LaunchOpusIq3sB8Ring<Batch, 4, 4, 2>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8StreamD1:
      LaunchOpusIq3sB8Ring<Batch, 4, 4, 1>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8Null:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kNull>(w, y, m, k,
                                                               stream);
      break;
    case SnbIq3Arm::kOpusB8NullStage:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kNullStage>(w, y, m, k,
                                                                    stream);
      break;
    case SnbIq3Arm::kOpusB8Dot4:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kDot4>(w, y, m, k,
                                                               stream);
      break;
    case SnbIq3Arm::kOpusB8RDot4S:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDot4S>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RDot4V:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDot4V>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RDot4U8:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDot4U8>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RDot8:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDot8>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RDot2:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDot2>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RDot2Acc:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDot2Acc>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RFma:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateFma>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RFmac:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateFmac>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RWmma8:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateWmma8>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RWmma4:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateWmma4>(w, y, m, k,
                                                                stream);
      break;
    case SnbIq3Arm::kOpusB8RDualVV:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDualVV>(w, y, m, k,
                                                                 stream);
      break;
    case SnbIq3Arm::kOpusB8RDualSV:
      LaunchOpusIq3sB8FloorProbe<Batch, OpusFloorProbe::kRateDualSV>(w, y, m, k,
                                                                 stream);
      break;
    case SnbIq3Arm::kOpusB8K4Pad:
      // Pad8 needs whole groups; otherwise the plain k4 kernel (same math).
      if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) == 0U)
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true>(w, y, xq, xds, m, k,
                                                        stream);
      else
        LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Slope:
      LaunchOpusIq3sB8Typed<Batch, 4, 4, false, false, true>(w, y, xq, xds, m,
                                                             k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Clk:
      LaunchOpusIq3sB8Typed<Batch, 4, 4, false, false, false, true>(
          w, y, xq, xds, m, k, stream, clk);
      break;
    case SnbIq3Arm::kOpusB8K4Act:
      LaunchOpusIq3sB8Typed<Batch, 4, 4, false, false, false, false, true>(
          w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4CutR:
      LaunchOpusIq3sB8CutOrK4<Batch, 0>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Cut:
      LaunchOpusIq3sB8CutOrK4<Batch, 1>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Cut64:
      LaunchOpusIq3sB8CutOrK4<Batch, 2>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8PActFix:
    case SnbIq3Arm::kOpusB8PNoDot:
      // Ceiling probes: outputs are wrong by design (read only the ms column).
      if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) != 0U)
        LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
      else if (arm == SnbIq3Arm::kOpusB8PActFix)
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, true, false>(w, y, xq, xds, m, k, stream);
      else
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, true>(w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Early:
    case SnbIq3Arm::kOpusB8K4Prio:
    case SnbIq3Arm::kOpusB8K4Ep:
      if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) != 0U)
        LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
      else if (arm == SnbIq3Arm::kOpusB8K4Early)
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, false, true, false>(w, y, xq, xds, m,
                                                                k, stream);
      else if (arm == SnbIq3Arm::kOpusB8K4Prio)
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, false, false, true>(w, y, xq, xds, m,
                                                                k, stream);
      else
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, false, true, true>(w, y, xq, xds, m,
                                                               k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Pf:
    case SnbIq3Arm::kOpusB8K4Prio2:
    case SnbIq3Arm::kOpusB8K4PrioW:
      if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) != 0U)
        LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
      else if (arm == SnbIq3Arm::kOpusB8K4Pf)
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 1U,
                              false, false, false, false, 1U, false>(
            w, y, xq, xds, m, k, stream);
      else if (arm == SnbIq3Arm::kOpusB8K4Prio2)
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, false, false, 2U, false>(
            w, y, xq, xds, m, k, stream);
      else
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, false, false, 0U, true>(
            w, y, xq, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8VopdCvp:
      LaunchOpusIq3sB8Vopd<Batch, true, true>(w, y, xh, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Sk:
    case SnbIq3Arm::kOpusB8K4Sk2:
    case SnbIq3Arm::kOpusB8K4Sk4:
    case SnbIq3Arm::kOpusB8CvSk:
    case SnbIq3Arm::kOpusB8CvSk2:
    case SnbIq3Arm::kOpusB8CvSk4:
    case SnbIq3Arm::kOpusB8CvpSk: {
      // k4prio / vopd_cv with the K range split across WGs when the row-group
      // grid leaves the 80 resident WG slots idle (SnbPickSplitK; the *2 / *4
      // arms force S). S = 1, no scratch / counter, or m % 8 != 0 -> the plain
      // k4prio / k4 / cv launch. Needs counter[] zeroed once (kernel resets it).
      const bool cv = SnbArmWantsHalfActs(arm);
      const std::uint32_t forced =
          (arm == SnbIq3Arm::kOpusB8K4Sk2 || arm == SnbIq3Arm::kOpusB8CvSk2)   ? 2U
          : (arm == SnbIq3Arm::kOpusB8K4Sk4 || arm == SnbIq3Arm::kOpusB8CvSk4) ? 4U
                                                                               : 0U;
      const bool pad = m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) == 0U;
      const std::uint32_t splits =
          (pad && sk_scratch != nullptr && sk_counter != nullptr)
              ? SnbPickSplitK(m, k / static_cast<std::uint32_t>(kSnIq3Elements),
                              4U, static_cast<std::uint32_t>(Batch),
                              sk_scratch_floats, forced)
              : 1U;
      if (cv) {
        if (arm == SnbIq3Arm::kOpusB8CvpSk)
          LaunchOpusIq3sB8Vopd<Batch, true, true>(w, y, xh, xds, m, k, stream,
                                                  sk_scratch, sk_counter, splits);
        else
          LaunchOpusIq3sB8Vopd<Batch, true>(w, y, xh, xds, m, k, stream,
                                            sk_scratch, sk_counter, splits);
      } else if (!pad) {
        LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
      } else if (splits == 1U) {
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, false, false, 1U, false>(
            w, y, xq, xds, m, k, stream);
      } else {
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              false, false, false, false, 1U, false, true>(
            w, y, xq, xds, m, k, stream, nullptr, sk_scratch, sk_counter,
            splits);
      }
      break;
    }
    case SnbIq3Arm::kOpusB8K4Ps:
      // Persistent stream-K; shapes with m % 256 != 0 run plain k4prio / k4.
      if (!LaunchOpusIq3sB8Persist<Batch, false>(w, y, xq, xds, m, k, stream,
                                                 sk_scratch, sk_counter,
                                                 sk_scratch_floats)) {
        if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) != 0U)
          LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
        else
          LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false,
                                0U, false, false, false, false, 1U, false>(
              w, y, xq, xds, m, k, stream);
      }
      break;
    case SnbIq3Arm::kOpusB8CvPs:
      if (!LaunchOpusIq3sB8Persist<Batch, true>(w, y, xh, xds, m, k, stream,
                                                sk_scratch, sk_counter,
                                                sk_scratch_floats))
        LaunchOpusIq3sB8Vopd<Batch, true>(w, y, xh, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8K4Fold:
    case SnbIq3Arm::kOpusB8K4Pad3: {
      // k4pad (+ one change); other shapes run plain k4, the same math.
      if (m % static_cast<std::uint32_t>(kSnIq3RowsPerGroup) != 0U)
        LaunchOpusIq3sB8Typed<Batch, 4, 4>(w, y, xq, xds, m, k, stream);
      else if (arm == SnbIq3Arm::kOpusB8K4Fold)
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 1U,
                              false>(w, y, xq, xds, m, k, stream);
      else
        LaunchOpusIq3sB8Typed<Batch, 4, 4, false, true, false, false, false, 0U,
                              true>(w, y, xq, xds, m, k, stream);
      break;
    }
    case SnbIq3Arm::kOpusB8VopdT16:
      LaunchOpusIq3sB8Vopd<Batch, false>(w, y, xh, xds, m, k, stream);
      break;
    case SnbIq3Arm::kOpusB8VopdCv:
      LaunchOpusIq3sB8Vopd<Batch, true>(w, y, xh, xds, m, k, stream);
      break;
  }
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_HIP_

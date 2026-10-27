#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_DOT4_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_DOT4_HPP_

// Speculative-verify GEMM for 2-8 tokens
// with int8 activations. The FP32 path does 16 FMAs per weight per token and
// stages 4-byte activations, so at width 8 it is ALU/LDS bound (Q4_K 150 GB/s
// against 220 at width 1). Here one weight sub-block is decoded once, then each
// token costs 4 v_dot4_i32_iu8 instead of 16 FMAs, and activations are 1 byte.
//
// Prior art: llama.cpp mmvq.cu (vendored for the MoE model in
// src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hip.cpp, Q4_K/Q5_K/Q8_0
// only): quantize the activations once per GEMM, dot4 against the weights,
// every weight row read once for all tokens. The two earlier gufo DOT4 tries
// broke exactly those two rules (one re-quantized the activations inside every
// CTA; the other launched one CTA per token, so weights were read B times).
//
// Decode reuses DecodeQuantSub16, so every format it knows works here.
// NOT bit-exact with the FP32 kernels: activations are rounded to int8 per
// 16-element sub-block (llama.cpp rounds per 32, so this is finer).

#include <cstdint>

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/kernels/xor_lane.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

// One thread per (token, 16-element sub-block). x is [batch][k] FP32; xq gets
// the int8 values in the same layout, xds[i] = {d, d * sum(q)} per sub-block,
// so Sum(w*x) ~= d * Sum(w*q) and Sum(x) ~= d * sum(q) for the affine formats.
// static: a non-template __global__ in a header would be a duplicate symbol
// once a second translation unit includes it.
static __global__ void QuantizeActivationsDot4Kernel(const float* __restrict__ x,
                                              std::int8_t* __restrict__ xq,
                                              float2* __restrict__ xds,
                                              std::uint32_t subs) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= subs)
    return;
  const float4* src = reinterpret_cast<const float4*>(x) + (i * 4U);
  float v[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float4 f = src[j];
    v[(4 * j) + 0] = f.x;
    v[(4 * j) + 1] = f.y;
    v[(4 * j) + 2] = f.z;
    v[(4 * j) + 3] = f.w;
  }
  float amax = 0.0F;
  bool nan = false;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    amax = fmaxf(amax, fabsf(v[j]));
    nan |= v[j] != v[j];
  }
  // fmaxf drops NaN, so without this a NaN activation would vanish into a
  // finite output. A NaN scale makes every dot over this sub-block NaN.
  const float d = nan ? __builtin_nanf("") : amax / 127.0F;
  const float id = amax > 0.0F ? 127.0F / amax : 0.0F;
  std::uint32_t packed[4] = {};
  int sum = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    // Clamp: with a denormal amax, id is inf and rintf(inf) would wrap.
    const int q =
        static_cast<int>(rintf(fminf(fmaxf(v[j] * id, -127.0F), 127.0F)));
    sum += q;
    packed[j / 4] |= (static_cast<std::uint32_t>(q) & 0xFFU) << (8 * (j % 4));
  }
  reinterpret_cast<uint4*>(xq)[i] = {packed[0], packed[1], packed[2],
                                     packed[3]};
  xds[i] = {d, d * static_cast<float>(sum)};
}

// ActK: activations with ONE scale per 256 values per token
// (ggml's CPU K-quant path, q8_K: one d per super-block + per-16 sums). Same
// layout as above, so the same scratch serves both: xq = int8 [batch][k],
// xds[i] = {d of i's super-block, bit-cast int sum(q) of the 16-sub-block}.
// The 16 threads of a super-block are 16 consecutive i (k % 256 == 0), an
// aligned 16-lane group of the wave, so a 4-step XOR shuffle finds its amax.
static __global__ void QuantizeActivationsQ8KDot4Kernel(const float* __restrict__ x,
                                                 std::int8_t* __restrict__ xq,
                                                 float2* __restrict__ xds,
                                                 std::uint32_t subs) {
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
  reinterpret_cast<uint4*>(xq)[i] = {packed[0], packed[1], packed[2],
                                     packed[3]};
  xds[i] = {d, __int_as_float(sum)};
}

// Each wave owns RowsPerWave output rows; lane L walks sub-blocks L, L+32, ...
// For each sub-block it decodes every row once and dots it against all Batch
// tokens, so a weight byte is read from DRAM exactly once per launch.
template<core::GgmlType WType, std::size_t Batch, std::size_t RowsPerWave,
         std::uint32_t Waves, bool FastIq = false>
__launch_bounds__(Waves * 32) __global__
    void SmallBatchDot4GEMMKernel(const void* __restrict__ w,
                                  const std::int8_t* __restrict__ xq,
                                  const float2* __restrict__ xds,
                                  float* __restrict__ y, std::uint32_t m,
                                  std::uint32_t k) {
  static_assert(Batch >= 1 && Batch <= 8);
  constexpr bool kHasOffset = WType == core::GgmlType::kQ4_K ||
                              WType == core::GgmlType::kQ5_K ||
                              WType == core::GgmlType::kQ2_K;
  __shared__ std::uint64_t fast_iq_grid[FastIq ? kIqFastGridWords<WType> : 1];
  __shared__ std::uint8_t fast_iq_signs[FastIq ? kIqFastSignBytes<WType> : 1];
  if constexpr (FastIq) {
    StageIqFastTables<WType>(fast_iq_grid, fast_iq_signs, threadIdx.x,
                             blockDim.x);
    __syncthreads();
  }
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t row_base =
      ((blockIdx.x * Waves) + (threadIdx.x >> 5U)) * RowsPerWave;
  if (row_base >= m)
    return;
  const std::uint32_t num_sub = k / 16U;
  const std::uint32_t row_bytes =
      static_cast<std::uint32_t>(QuantRowBytes(WType, k));
  const auto* const wb = static_cast<const std::uint8_t*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (std::uint32_t sub = lane; sub < num_sub; sub += 32U) {
    uint4 qx[Batch];
    float2 ds[Batch];
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      qx[t] = reinterpret_cast<const uint4*>(xq + (t * k))[sub];
      ds[t] = xds[(t * num_sub) + sub];
    }
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::uint32_t row = row_base + r < m ? row_base + r : m - 1;
      QuantSub16 dec;
      DecodeQuantSub16<true, FastIq>(WType, wb + (row * row_bytes), sub, dec,
                                     fast_iq_grid, fast_iq_signs);
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, dec.w[0], true, qx[t].x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, dec.w[1], true, qx[t].y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, dec.w[2], true, qx[t].z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, dec.w[3], true, qx[t].w, dot, false);
        float c = dec.scale * (ds[t].x * static_cast<float>(dot));
        if constexpr (kHasOffset)
          c -= dec.offset * ds[t].y;
        sums[r][t] += c;
      }
    }
  }

  // After the xor butterfly every lane holds every total; lane t stores token t.
#pragma unroll
  for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float total = ReduceKQuantWave<16>(sums[r][t]);
      if (lane == t && row_base + r < m)
        y[(t * m) + row_base + r] = total;
    }
  }
}

// v2 (review: v1 re-reads 192 B of activations per lane per sub-block
// from L1/L2, 3-7x the weight bytes). The block stages SubsPerLane*32
// sub-blocks of every token's int8 activations and scales in LDS once and all
// waves read them there; double-buffered, one barrier per tile. The final
// reduction hands each token to 4 lanes (ReduceKQuantColumns, 9 swizzles per
// row instead of 40), the same store pattern as the exact kernel's
// DistributedOutput path.
// k95small: body with an explicit workgroup index (see the V3 body).
template<core::GgmlType WType, std::size_t Batch, std::size_t RowsPerWave,
         std::uint32_t Waves, std::uint32_t SubsPerLane = 1,
         bool FastIq = false>
__device__ __forceinline__ void SmallBatchDot4LdsGEMMBody(
    const void* __restrict__ w, const std::int8_t* __restrict__ xq,
    const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
    std::uint32_t k, std::uint32_t block_index) {
  static_assert(Batch >= 1 && Batch <= 8);
  constexpr bool kHasOffset = WType == core::GgmlType::kQ4_K ||
                              WType == core::GgmlType::kQ5_K ||
                              WType == core::GgmlType::kQ2_K;
  constexpr std::uint32_t kTileSubs = 32U * SubsPerLane;
  constexpr std::uint32_t kThreads = Waves * 32U;
  __shared__ uint4 stage_q[2][Batch * kTileSubs];
  __shared__ float2 stage_ds[2][Batch * kTileSubs];
  __shared__ std::uint64_t fast_iq_grid[FastIq ? kIqFastGridWords<WType> : 1];
  __shared__ std::uint8_t fast_iq_signs[FastIq ? kIqFastSignBytes<WType> : 1];
  if constexpr (FastIq) {
    StageIqFastTables<WType>(fast_iq_grid, fast_iq_signs, threadIdx.x,
                             blockDim.x);
  }
  const std::uint32_t num_sub = k / 16U;
  // Every wave of the block stages, even one whose rows are past m, so the
  // barriers stay uniform; such a wave just skips the math and the store.
  const auto stage = [&](std::uint32_t buffer, std::uint32_t tile_base) {
    for (std::uint32_t i = threadIdx.x; i < Batch * kTileSubs; i += kThreads) {
      const std::uint32_t t = i / kTileSubs;
      const std::uint32_t sub = tile_base + (i % kTileSubs);
      if (sub < num_sub) {
        stage_q[buffer][i] =
            reinterpret_cast<const uint4*>(xq + (t * k))[sub];
        stage_ds[buffer][i] = xds[(t * num_sub) + sub];
      } else {
        stage_q[buffer][i] = {0U, 0U, 0U, 0U};
        stage_ds[buffer][i] = {0.0F, 0.0F};
      }
    }
  };
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t row_base =
      ((block_index * Waves) + (threadIdx.x >> 5U)) * RowsPerWave;
  const bool active = row_base < m;
  const std::uint32_t row_bytes =
      static_cast<std::uint32_t>(QuantRowBytes(WType, k));
  const auto* const wb = static_cast<const std::uint8_t*>(w);
  float sums[RowsPerWave][Batch] = {};

  stage(0, 0);
  __syncthreads();
  std::uint32_t buffer = 0;
  for (std::uint32_t tile_base = 0; tile_base < num_sub;
       tile_base += kTileSubs) {
    // The other buffer was last read before the previous barrier.
    if (tile_base + kTileSubs < num_sub)
      stage(buffer ^ 1U, tile_base + kTileSubs);
    if (active) {
#pragma unroll
      for (std::uint32_t s = 0; s < SubsPerLane; ++s) {
        const std::uint32_t slot = (s * 32U) + lane;
        const std::uint32_t sub = tile_base + slot;
        if (sub >= num_sub)
          break;
        uint4 qx[Batch];
        float2 ds[Batch];
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t) {
          qx[t] = stage_q[buffer][(t * kTileSubs) + slot];
          ds[t] = stage_ds[buffer][(t * kTileSubs) + slot];
        }
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          const std::uint32_t row = row_base + r < m ? row_base + r : m - 1;
          QuantSub16 dec;
          DecodeQuantSub16<true, FastIq>(WType, wb + (row * row_bytes), sub,
                                         dec, fast_iq_grid, fast_iq_signs);
#pragma unroll
          for (std::size_t t = 0; t < Batch; ++t) {
            int dot = 0;
            dot = __builtin_amdgcn_sudot4(true, dec.w[0], true, qx[t].x, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, dec.w[1], true, qx[t].y, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, dec.w[2], true, qx[t].z, dot,
                                          false);
            dot = __builtin_amdgcn_sudot4(true, dec.w[3], true, qx[t].w, dot,
                                          false);
            float c = dec.scale * (ds[t].x * static_cast<float>(dot));
            if constexpr (kHasOffset)
              c -= dec.offset * ds[t].y;
            sums[r][t] += c;
          }
        }
      }
    }
    __syncthreads();
    buffer ^= 1U;
  }

  if (!active)
    return;
#pragma unroll
  for (std::size_t r = 0; r < RowsPerWave; ++r) {
    float values[8] = {};
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      values[t] = sums[r][t];
    // Lanes 4t..4t+3 end up holding token t's total.
    const float total = ReduceKQuantColumns<8, 16>(values);
    const std::uint32_t t = lane / 4U;
    if (lane % 4U == 0U && t < Batch && row_base + r < m)
      y[(t * m) + row_base + r] = total;
  }
}

template<core::GgmlType WType, std::size_t Batch, std::size_t RowsPerWave,
         std::uint32_t Waves, std::uint32_t SubsPerLane = 1,
         bool FastIq = false>
__launch_bounds__(Waves * 32) __global__
    void SmallBatchDot4LdsGEMMKernel(const void* __restrict__ w,
                                     const std::int8_t* __restrict__ xq,
                                     const float2* __restrict__ xds,
                                     float* __restrict__ y, std::uint32_t m,
                                     std::uint32_t k) {
  SmallBatchDot4LdsGEMMBody<WType, Batch, RowsPerWave, Waves, SubsPerLane,
                            FastIq>(w, xq, xds, y, m, k, blockIdx.x);
}

template<core::GgmlType WType, std::size_t Batch, std::size_t RowsPerWave,
         std::uint32_t Waves, std::uint32_t SubsPerLane = 1,
         bool FastIq = false>
void LaunchSmallBatchDot4Lds(const void* w, const float* x, float* y,
                             std::int8_t* xq, float2* xds, std::uint32_t m,
                             std::uint32_t k, hipStream_t stream) {
  const std::uint32_t subs = static_cast<std::uint32_t>(Batch) * (k / 16U);
  QuantizeActivationsDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
      x, xq, xds, subs);
  constexpr std::uint32_t kRowsPerBlock = Waves * RowsPerWave;
  SmallBatchDot4LdsGEMMKernel<WType, Batch, RowsPerWave, Waves, SubsPerLane,
                              FastIq>
      <<<(m + kRowsPerBlock - 1) / kRowsPerBlock, Waves * 32, 0, stream>>>(
          w, xq, xds, y, m, k);
}

// v3 Q6_K: one lane = half a super-block (elements
// 128*half .. +127), the ggml dequantize_row_q6_K split. block_q6_K = ql[128]
// (low nibbles), qh[64] (2 high bits), int8 scales[16] (one per 16 elements),
// fp16 d at byte 208. A half reads ql 64 B + qh 32 B + 8 scales + d = 106 B for
// 105 B of weight. 210-byte blocks are only 2-byte aligned, so the 16-byte
// loads go through memcpy (unaligned global access; gfx9+ HSA allows it).
// Sub-block s (16 elements, natural order within the half): ql chunk
// 2*((s>>1)&1) + (s&1), nibble s>>2, qh chunk s&1, qh bits 2*(s>>1), scale s.
// Value = d * sc * (q - 32) with q in 0..63, so per token
// d*sc*(ds.x*dot(q, xq) - 32*ds.y) (ds.y = d_x * sum(xq)).
// ActK: q8_K activations (one d per 256, so one per half). The
// sub-block q term becomes integer isum += sc_j*dot_j; the -32 offset is Q4_K's
// min term without a separate dmin -- value = xd*d*(isum - 32*ioff), ioff =
// sum_j sc_j*sumq_j. |dot| <= 16*63*127 < 2^17, |sc| <= 128 (int8), |sumq| <=
// 16*127; 8 sub-blocks: |isum| < 2^27, |32*ioff| < 2^26, |isum - 32*ioff| <
// 2^28, fits int32, and every __mul24 operand fits 24 bits. ONE float multiply
// per (row, token, half).
// Rows: that many adjacent rows per lane share each LDS activation read;
// per-(row, token) order is the Rows == 1 order.
//
// Group (headq6): the lane loads and consumes its Rows rows Group at a
// time, so only Group rows' ql/qh are live at once. Group == Rows (the
// default) is the old body with the outer loop run once and g0 == 0.
// BIT-EXACT at any Group that divides Rows: a row's accumulator is private to
// it, and every (row, token) sees the same s order, the same four-dot4 chain
// and the same epilogue expression -- no value crosses rows, so grouping can
// only change WHEN a row is computed, never its value.
// Why: the served output head is Rows == 4, whose ql[4][4] + qh[4][2] = 96
// VGPRs of weight stay live across the whole s loop; it compiles to 252 VGPR
// (B7) and 256 VGPR + 72 B/lane of scratch (B8) = 5 waves/SIMD. Group 2 gives
// 174/170 VGPR, no scratch and 8 waves/SIMD; Group 1 gives 116/122 and 10/8.
// Cost: an LDS activation read serves Group rows instead of Rows, so stage_q
// is read Rows/Group times per tile (B8 tile loop 3437 -> 3648 instructions).
// The slice must stay inside this function on the whole-array references:
// hoisting it into a wrapper that passes `sums + g0` makes the accumulator
// escape SROA and costs 540-1316 B/lane of scratch (measured).
template<std::size_t Batch, std::uint32_t Rows, bool ActK = false,
         std::uint32_t Group = Rows>
__device__ __forceinline__ void Q6KHalfBlockDot(
    const std::uint8_t* const (&bps)[Rows], std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots,
    float (&sums)[Rows][Batch]) {
  static_assert(Group >= 1U && Group <= Rows && (Rows % Group) == 0U);
#pragma unroll
  for (std::uint32_t g0 = 0; g0 < Rows; g0 += Group) {
  uint4 ql[Group][4];
  uint4 qh[Group][2];
  std::uint32_t sc_words[Group][2];
  float d[Group];
  int sc_int[Group];
#pragma unroll
  for (std::uint32_t r = 0; r < Group; ++r) {
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      __builtin_memcpy(&ql[r][c], bps[g0 + r] + (64U * half) + (16U * c), 16);
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      __builtin_memcpy(&qh[r][c], bps[g0 + r] + 128U + (32U * half) + (16U * c), 16);
    __builtin_memcpy(sc_words[r], bps[g0 + r] + 192U + (8U * half), 8);
    unsigned short d_bits;
    __builtin_memcpy(&d_bits, bps[g0 + r] + 208U, 2);
    d[r] = __half2float(__ushort_as_half(d_bits));
  }
  int isum[Group][ActK ? Batch : 1] = {};  // ActK: one half-block's int sum
#pragma unroll
  for (std::uint32_t s = 0; s < 8U; ++s) {
    const std::uint32_t nib = 4U * (s >> 2U);
    const std::uint32_t hb = 2U * (s >> 1U);
    std::uint32_t q[Group][4];
    float scale[Group];
#pragma unroll
    for (std::uint32_t r = 0; r < Group; ++r) {
      const uint4 lo = ql[r][(2U * ((s >> 1U) & 1U)) + (s & 1U)];
      const uint4 hi = qh[r][s & 1U];
      const std::uint32_t lw[4] = {lo.x, lo.y, lo.z, lo.w};
      const std::uint32_t hw[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        q[r][i] = ((lw[i] >> nib) & 0x0F0F0F0FU) |
                  (((hw[i] >> hb) & 0x03030303U) << 4U);
      const int sc = static_cast<std::int8_t>(
          (sc_words[r][s >> 2U] >> (8U * (s & 3U))) & 0xFFU);
      scale[r] = d[r] * static_cast<float>(sc);
      sc_int[r] = sc;
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
      if constexpr (ActK) {
        // Integer only: isum += sc_j * dot_j per sub-block; the -32 offset is
        // applied once per half below from the per-sub-block activation sums.
#pragma unroll
        for (std::uint32_t r = 0; r < Group; ++r) {
          int dot = 0;
          dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
          dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
          dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
          dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
          isum[r][t] += __mul24(sc_int[r], dot);
        }
      } else {
        const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
#pragma unroll
        for (std::uint32_t r = 0; r < Group; ++r) {
          int dot = 0;
          dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
          dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
          dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
          dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
          sums[g0 + r][t] +=
              scale[r] * ((ds.x * static_cast<float>(dot)) - (32.0F * ds.y));
        }
      }
    }
  }
  if constexpr (ActK) {
    // Once per half: the -32 offset from the 8 sub-block activation sums, then
    // xd * d * (isum - 32*ioff) per (row, token) -- one float multiply. Each
    // token's sums are read from LDS once for all Rows rows.
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float2* dst = &stage_ds[(t * tile_slots) + slot_base];
      int bs[8];
#pragma unroll
      for (std::uint32_t j = 0; j < 8U; ++j)
        bs[j] = __float_as_int(dst[j].y);  // sum(q) of sub-block j, per-256 d
      const float xd = dst[0].x;
#pragma unroll
      for (std::uint32_t r = 0; r < Group; ++r) {
        int ioff = 0;
#pragma unroll
        for (std::uint32_t j = 0; j < 8U; ++j) {
          const int scj = static_cast<std::int8_t>(
              (sc_words[r][j >> 2U] >> (8U * (j & 3U))) & 0xFFU);
          ioff += __mul24(scj, bs[j]);
        }
        sums[g0 + r][t] +=
            (d[r] * xd) * static_cast<float>(isum[r][t] - (32 * ioff));
      }
    }
  }
  // Keep the next group's weight loads below this group's dots: hoisting
  // them is what would put all Rows rows live again (W23 sched_barrier).
  if constexpr (Group < Rows) __builtin_amdgcn_sched_barrier(0);
  }
}

// v3 IQ4_XS: same half-super-block split. block_iq4_xs =
// fp16 d, u16 scales_h, u8 scales_l[4], qs[128]; 136-byte blocks are only
// 8-byte aligned, so the qs loads go through memcpy like Q6_K's. Sub-block
// j (32 elements) = qs[16j .. 16j+15], low nibbles = elements 0-15, high
// nibbles = 16-31; scale ls - 32 with ls = scales_l nibble j | (scales_h bits
// 2j, 2j+1) << 4 (ggml dequantize_row_iq4_xs). A half reads the 8-byte header
// + 64 B of qs for 68 B of weight. Value = d * (ls - 32) * kvalues_iq4nl[q]:
// symmetric, so no offset term. Sub-block s of the half (16 elements) is
// sub-block j = 4*half + s/2, nibble s%2 -- DecodeQuantSub16's order.
// Rows: that many adjacent rows share
// each LDS activation read; per-(row, token) order is the Rows == 1 order.
// ActK: q8_K activations (one scale per 256, so one per
// half), integer isum += (ls-32)*dot per sub-block, ONE float multiply per
// (row, token, half) instead of cvt+mul+fma per sub-block. No min term in
// IQ4_XS. |dot| <= 16*127*127 < 2^18, |ls-32| <= 32, 8 sub-blocks: < 2^27.
template<std::size_t Batch, std::uint32_t Rows, bool ActK = false>
__device__ __forceinline__ void Iq4XsHalfBlockDot(
    const std::uint8_t* const (&bps)[Rows], std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots,
    float (&sums)[Rows][Batch]) {
  uint2 head[Rows];
  uint4 qs[Rows][4];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    __builtin_memcpy(&head[r], bps[r], 8);
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      __builtin_memcpy(&qs[r][c], bps[r] + 8U + (64U * half) + (16U * c), 16);
  }
  int isum[Rows][ActK ? Batch : 1] = {};
#pragma unroll
  for (std::uint32_t s = 0; s < 8U; ++s) {
    const std::uint32_t j = (4U * half) + (s >> 1U);
    const std::uint32_t shift = 4U * (s & 1U);
    float scale[Rows];
    int ls_int[Rows];
    std::uint32_t q[Rows][4];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      const float d = __half2float(
          __ushort_as_half(static_cast<unsigned short>(head[r].x & 0xFFFFU)));
      const std::uint32_t scales_h = head[r].x >> 16U;
      const std::uint32_t scales_l = head[r].y;
      const std::uint32_t ls = ((scales_l >> (4U * j)) & 15U) |
                               (((scales_h >> (2U * j)) & 3U) << 4U);
      ls_int[r] = static_cast<int>(ls) - 32;
      scale[r] = d * static_cast<float>(ls_int[r]);
      const uint4 word = qs[r][s >> 1U];
      const std::uint32_t words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        q[r][i] = Iq4NlLookup4((words[i] >> shift) & 0x0F0F0F0FU);
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
        if constexpr (ActK) {
          isum[r][t] += __mul24(ls_int[r], dot);
        } else {
          const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
          sums[r][t] += scale[r] * (ds.x * static_cast<float>(dot));
        }
      }
    }
  }
  if constexpr (ActK) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float xd = stage_ds[(t * tile_slots) + slot_base].x;  // per-256 scale
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        const float d = __half2float(
            __ushort_as_half(static_cast<unsigned short>(head[r].x & 0xFFFFU)));
        sums[r][t] += (d * xd) * static_cast<float>(isum[r][t]);
      }
    }
  }
}

// v3 Q3_K: block_q3_K = hmask[32], qs[64], scales[12]
// (16 packed 6-bit), fp16 d at byte 108 = 110 B, 2-byte aligned -> memcpy loads.
// ggml dequantize_row_q3_K: half n (elements 128n..+127) uses qs[32n..32n+31]
// and hmask bits 4n..4n+3 (m keeps shifting across halves); 16-element
// sub-block s of the half: qs/hmask chunk p = s&1 (bytes 16p..16p+15), 2-bit
// field j = s>>1 (shift 2j), hmask bit 4n+j, scale index 8n+s. Value =
// d*(sc-32)*(q - 4) with q = low2 | hbit<<2 in 0..7, so per token
// d*(sc-32)*(ds.x*dot(q, xq) - 4*ds.y) (ds.y = d_x * sum(xq), as Q6_K).
// Rows: adjacent rows share each LDS read; order as Rows == 1.
template<std::size_t Batch, std::uint32_t Rows>
__device__ __forceinline__ void Q3KHalfBlockDot(
    const std::uint8_t* const (&bps)[Rows], std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots,
    float (&sums)[Rows][Batch]) {
  float d[Rows];
  uint4 hm[Rows][2];
  uint4 qs[Rows][2];
  std::uint32_t sc_words[Rows][2];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c) {
      __builtin_memcpy(&hm[r][c], bps[r] + (16U * c), 16);
      __builtin_memcpy(&qs[r][c], bps[r] + 32U + (32U * half) + (16U * c), 16);
    }
    std::uint32_t aux[3];
    __builtin_memcpy(aux, bps[r] + 96U, 12);
    unsigned short d_bits;
    __builtin_memcpy(&d_bits, bps[r] + 108U, 2);
    d[r] = __half2float(__ushort_as_half(d_bits));
    // ggml's aux unpack; half 0 needs scale words 0,1 and half 1 words 2,3.
    const std::uint32_t tmp = aux[2];
    if (half == 0U) {
      sc_words[r][0] = (aux[0] & 0x0F0F0F0FU) | (((tmp >> 0U) & 0x03030303U) << 4U);
      sc_words[r][1] = (aux[1] & 0x0F0F0F0FU) | (((tmp >> 2U) & 0x03030303U) << 4U);
    } else {
      sc_words[r][0] = ((aux[0] >> 4U) & 0x0F0F0F0FU) | (((tmp >> 4U) & 0x03030303U) << 4U);
      sc_words[r][1] = ((aux[1] >> 4U) & 0x0F0F0F0FU) | (((tmp >> 6U) & 0x03030303U) << 4U);
    }
  }
#pragma unroll
  for (std::uint32_t s = 0; s < 8U; ++s) {
    const std::uint32_t shift = 2U * (s >> 1U);
    const std::uint32_t hb = (4U * half) + (s >> 1U);
    std::uint32_t q[Rows][4];
    float scale[Rows];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      const uint4 lo = qs[r][s & 1U];
      const uint4 hi = hm[r][s & 1U];
      const std::uint32_t lw[4] = {lo.x, lo.y, lo.z, lo.w};
      const std::uint32_t hw[4] = {hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        q[r][i] = ((lw[i] >> shift) & 0x03030303U) |
                  (((hw[i] >> hb) & 0x01010101U) << 2U);
      const int sb = (sc_words[r][s >> 2U] >> (8U * (s & 3U))) & 0xFFU;
      scale[r] = d[r] * static_cast<float>(sb - 32);
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
      const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
        const float dotf = static_cast<float>(dot);
        sums[r][t] += scale[r] * ((ds.x * dotf) - (4.0F * ds.y));
      }
    }
  }
}

// v3 IQ3_S (written to speed up the lower-impact matrix
// types of the 2-bit EfficientThink model, IQ3_S x172 tensors). block_iq3_s
// = fp16 d@0, qs[64]@2, qh[8]@66, signs[32]@74, scales[4]@106 (110 B, 2-byte
// aligned: all loads memcpy). Half h owns sb32 4h..4h+3 = qs bytes 32h.., qh
// bytes 4h.., signs bytes 16h.., scales bytes 2h... For sub-block s of the half
// (c = s>>1, p = s&1) and grid word w, DecodeQuantSub16's indices reduce to:
// qs byte 4s+w (= byte w of qs word s), qh bit 4p+w of qh byte c, sign nibble
// w of the 16-bit sign word s, scale nibble c. Grid from LDS (512 x u32).
template<std::size_t Batch>
__device__ __forceinline__ void Iq3sHalfBlockDot(
    const std::uint8_t* __restrict__ bp, std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots, float (&sums)[Batch],
    const std::uint32_t* __restrict__ grid) {
  std::uint32_t qs[8];
  __builtin_memcpy(qs, bp + 2U + (32U * half), 32);
  std::uint32_t qh;
  __builtin_memcpy(&qh, bp + 66U + (4U * half), 4);
  std::uint32_t sg[4];
  __builtin_memcpy(sg, bp + 74U + (16U * half), 16);
  unsigned short sc;
  __builtin_memcpy(&sc, bp + 106U + (2U * half), 2);
  unsigned short d_bits;
  __builtin_memcpy(&d_bits, bp, 2);
  const float d = __half2float(__ushort_as_half(d_bits));
#pragma unroll
  for (std::uint32_t s = 0; s < 8U; ++s) {
    const std::uint32_t c = s >> 1U;
    const std::uint32_t hbits = qh >> ((8U * c) + (4U * (s & 1U)));
    const std::uint32_t sbits = sg[c] >> (16U * (s & 1U));
    std::uint32_t q[4];
#pragma unroll
    for (std::uint32_t w = 0; w < 4U; ++w)
      q[w] = NegateSelectedBytes(
          grid[((qs[s] >> (8U * w)) & 0xFFU) | (((hbits >> w) & 1U) << 8U)],
          sbits >> (4U * w));
    const float scale =
        d * static_cast<float>(
                1U + (2U * ((static_cast<std::uint32_t>(sc) >> (4U * c)) & 0x0FU)));
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
      const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
      int dot = 0;
      dot = __builtin_amdgcn_sudot4(true, q[0], true, qx.x, dot, false);
      dot = __builtin_amdgcn_sudot4(true, q[1], true, qx.y, dot, false);
      dot = __builtin_amdgcn_sudot4(true, q[2], true, qx.z, dot, false);
      dot = __builtin_amdgcn_sudot4(true, q[3], true, qx.w, dot, false);
      sums[t] += scale * (ds.x * static_cast<float>(dot));
    }
  }
}

// IQ3_XXS (98 B: fp16 d@0, grid indices qs[64]@2, then one u32 per sb32 @66:
// four 7-bit sign indices + 4-bit scale in the top nibble). Same half split as
// IQ3_S: qs byte 4s+w = byte w of qs word s; sign index 2p+(w>>1) of aux word
// c through the 128-B ksigns table (LDS), nibble w&1; scale d*(0.5+ls)*0.5.
template<std::size_t Batch>
__device__ __forceinline__ void Iq3xxsHalfBlockDot(
    const std::uint8_t* __restrict__ bp, std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots, float (&sums)[Batch],
    const std::uint32_t* __restrict__ grid,
    const std::uint8_t* __restrict__ ksigns) {
  std::uint32_t qs[8];
  __builtin_memcpy(qs, bp + 2U + (32U * half), 32);
  std::uint32_t ax[4];
  __builtin_memcpy(ax, bp + 66U + (16U * half), 16);
  unsigned short d_bits;
  __builtin_memcpy(&d_bits, bp, 2);
  const float d = __half2float(__ushort_as_half(d_bits));
  // Sub-block s from its qs word and aux word (p = s & 1 picks the sign half).
  const auto sub = [&](std::uint32_t s, std::uint32_t qsw, std::uint32_t axw) {
    std::uint32_t q[4];
#pragma unroll
    for (std::uint32_t w = 0; w < 4U; ++w) {
      const std::uint32_t sg =
          ksigns[(axw >> (7U * ((2U * (s & 1U)) + (w >> 1U)))) & 127U];
      q[w] = NegateSelectedBytes(grid[(qsw >> (8U * w)) & 0xFFU],
                                        sg >> (4U * (w & 1U)));
    }
    const float scale = d * (0.5F + static_cast<float>(axw >> 28U)) * 0.5F;
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
      const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
      int dot = 0;
      dot = __builtin_amdgcn_sudot4(true, q[0], true, qx.x, dot, false);
      dot = __builtin_amdgcn_sudot4(true, q[1], true, qx.y, dot, false);
      dot = __builtin_amdgcn_sudot4(true, q[2], true, qx.z, dot, false);
      dot = __builtin_amdgcn_sudot4(true, q[3], true, qx.w, dot, false);
      sums[t] += scale * (ds.x * static_cast<float>(dot));
    }
  };
  // A runtime index into qs/ax put both arrays in scratch
  // (48 B, a store + reloads in the hot loop; IQ3_S's full unroll has none).
  if constexpr (Batch <= 3) {
    // Few accumulators: full unroll fits (as Iq3sHalfBlockDot).
#pragma unroll
    for (std::uint32_t s = 0; s < 8U; ++s) sub(s, qs[s], ax[s >> 1U]);
  } else {
    // Full unroll hoists all 32 grid gathers: VGPR 248 -> 135 at B8 (ISA).
    // Keep two sub-blocks per pass, always read slots 0/1, then
    // shift the queues down, so every index is a compile-time constant.
#pragma unroll 1
    for (std::uint32_t c = 0; c < 4U; ++c) {
      sub(2U * c, qs[0], ax[0]);
      sub((2U * c) + 1U, qs[1], ax[0]);
#pragma unroll
      for (std::uint32_t i = 0; i < 6U; ++i) qs[i] = qs[i + 2U];
#pragma unroll
      for (std::uint32_t i = 0; i < 3U; ++i) ax[i] = ax[i + 1U];
    }
  }
}

// Two or four IQ3 output rows share each activation LDS read. Grid/sign
// decode remains outside the token loop, and each row keeps the R1 K16 update
// sequence. Prior art: strix-llama vec_dot_iq3_s_q8_1_pair; not its Q8_1
// layout. The R1 helpers above serve B1-3 (measured: XXS full unroll there).
// Rows is 2 or 4 (4 only at Batch >= 6, same kernel static_assert): four rows
// share each LDS read. Every per-row loop is a constexpr-trip unroll, so the
// queue slots stay compile-time constants (verified: no scratch). VGPR estimate
// at Rows 4, B8 (a FLOOR: the compiler may allocate more): 32 (sums_all) +
// 32 (qs) + 16 (aux) + 16 (decoded q per sub-block) + ~8 scales/d + ~20
// row/loop state ~= 124 hot VGPRs in the helper; the full kernel sits in the
// 152-176 band measured for B8 x2, likely nearer the top at x4.
template<bool IsIQ3S, std::size_t Batch, std::uint32_t Rows>
__device__ __forceinline__ void Iq3MultiRowHalfBlockDot(
    const std::uint8_t* const (&bp)[Rows], std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots,
    float (&sums)[Rows][Batch], const std::uint32_t* __restrict__ grid,
    const std::uint8_t* __restrict__ ksigns) {
  static_assert(Rows == 2U || (Rows == 4U && Batch >= 6));
  static_assert(Batch >= 4 && Batch <= 8);
  std::uint32_t qs[Rows][8];
  std::uint32_t aux[Rows][4];
  std::uint32_t qh[Rows];
  unsigned short sc[Rows];
  float d[Rows];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    __builtin_memcpy(qs[r], bp[r] + 2U + (32U * half), 32);
    unsigned short d_bits;
    __builtin_memcpy(&d_bits, bp[r], 2);
    d[r] = __half2float(__ushort_as_half(d_bits));
    if constexpr (IsIQ3S) {
      __builtin_memcpy(&qh[r], bp[r] + 66U + (4U * half), 4);
      __builtin_memcpy(aux[r], bp[r] + 74U + (16U * half), 16);
      __builtin_memcpy(&sc[r], bp[r] + 106U + (2U * half), 2);
    } else {
      __builtin_memcpy(aux[r], bp[r] + 66U + (16U * half), 16);
    }
  }
  // Sub-block s = 2c + p from qs word qsw[r] and aux word axw[r].
  const auto sub = [&](std::uint32_t s, const std::uint32_t (&qsw)[Rows],
                       const std::uint32_t (&axw)[Rows])
      __attribute__((always_inline)) {
    const std::uint32_t c = s >> 1U;
    std::uint32_t q[Rows][4];
    float scale[Rows];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      if constexpr (IsIQ3S) {
        const std::uint32_t hbits = qh[r] >> ((8U * c) + (4U * (s & 1U)));
        const std::uint32_t sbits = axw[r] >> (16U * (s & 1U));
#pragma unroll
        for (std::uint32_t w = 0; w < 4U; ++w)
          q[r][w] =
              NegateSelectedBytes(grid[((qsw[r] >> (8U * w)) & 0xFFU) |
                                              (((hbits >> w) & 1U) << 8U)],
                                         sbits >> (4U * w));
        scale[r] =
            d[r] *
            static_cast<float>(
                1U + (2U * ((static_cast<std::uint32_t>(sc[r]) >> (4U * c)) &
                            0x0FU)));
      } else {
#pragma unroll
        for (std::uint32_t w = 0; w < 4U; ++w) {
          const std::uint32_t sg =
              ksigns[(axw[r] >> (7U * ((2U * (s & 1U)) + (w >> 1U)))) & 127U];
          q[r][w] =
              NegateSelectedBytes(grid[(qsw[r] >> (8U * w)) & 0xFFU],
                                         sg >> (4U * (w & 1U)));
        }
        scale[r] = d[r] * (0.5F + static_cast<float>(axw[r] >> 28U)) * 0.5F;
      }
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
      const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
        sums[r][t] += scale[r] * (ds.x * static_cast<float>(dot));
      }
    }
  };
  // Full unroll hoists IQ grid gathers and raises VGPR pressure at B8 (measured).
  // Two sub-blocks per pass; read queue slots 0/1 and shift, so
  // no runtime index puts qs/aux in scratch (was 112 B, reloads in the loop).
#pragma unroll 1
  for (std::uint32_t c = 0; c < 4U; ++c) {
    std::uint32_t q0[Rows];
    std::uint32_t q1[Rows];
    std::uint32_t a0[Rows];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      q0[r] = qs[r][0];
      q1[r] = qs[r][1];
      a0[r] = aux[r][0];
    }
    sub(2U * c, q0, a0);
    sub((2U * c) + 1U, q1, a0);
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
#pragma unroll
      for (std::uint32_t i = 0; i < 6U; ++i) qs[r][i] = qs[r][i + 2U];
#pragma unroll
      for (std::uint32_t i = 0; i < 3U; ++i) aux[r][i] = aux[r][i + 1U];
    }
  }
}

// IQ2_S (82 B: fp16 d@0, grid indices qs[32]@2, sign bytes qs[32]@34, qh[8]@66,
// scales[8]@74). Half h owns sb32 4h..4h+3: index/sign bytes 16h.., qh and
// scale bytes 4h... Sub-block s (c = s>>1, p = s&1), grid entry g: l = 2p+g,
// index = byte l of index word c | (qh byte c << (8-2l)) & 0x300 (10 bits, a
// u64 of 8 magnitudes), sign byte l of sign word c; scale d*(0.5+nib)*0.25,
// nibble p of scale byte c. Grid 1024 x u64 (8 KB) in LDS.
// Rows: that many adjacent rows share each LDS
// activation read; per-(row, token) order is the Rows == 1 order.
template<std::size_t Batch, std::uint32_t Rows>
__device__ __forceinline__ void Iq2sHalfBlockDot(
    const std::uint8_t* const (&bps)[Rows], std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots,
    float (&sums)[Rows][Batch], const std::uint64_t* __restrict__ grid) {
  std::uint32_t ix[Rows][4];
  std::uint32_t sw[Rows][4];
  std::uint32_t qh[Rows];
  std::uint32_t sc[Rows];
  float d[Rows];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
    __builtin_memcpy(ix[r], bps[r] + 2U + (16U * half), 16);
    __builtin_memcpy(sw[r], bps[r] + 34U + (16U * half), 16);
    __builtin_memcpy(&qh[r], bps[r] + 66U + (4U * half), 4);
    __builtin_memcpy(&sc[r], bps[r] + 74U + (4U * half), 4);
    unsigned short d_bits;
    __builtin_memcpy(&d_bits, bps[r], 2);
    d[r] = __half2float(__ushort_as_half(d_bits));
  }
#pragma unroll  // unroll 2 here puts ix/sw in scratch (32 B private, ISA)
  for (std::uint32_t s = 0; s < 8U; ++s) {
    const std::uint32_t c = s >> 1U;
    std::uint32_t q[Rows][4];
    float scale[Rows];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      const std::uint32_t qhb = (qh[r] >> (8U * c)) & 0xFFU;
#pragma unroll
      for (std::uint32_t g = 0; g < 2U; ++g) {
        const std::uint32_t l = (2U * (s & 1U)) + g;
        const std::uint64_t e =
            grid[((ix[r][c] >> (8U * l)) & 0xFFU) |
                 ((qhb << (8U - (2U * l))) & 0x300U)];
        const std::uint32_t sg = (sw[r][c] >> (8U * l)) & 0xFFU;
        q[r][2U * g] = NegateSelectedBytes(static_cast<std::uint32_t>(e), sg);
        q[r][(2U * g) + 1U] =
            NegateSelectedBytes(static_cast<std::uint32_t>(e >> 32U), sg >> 4U);
      }
      scale[r] =
          d[r] * (0.5F + static_cast<float>((sc[r] >> ((8U * c) + (4U * (s & 1U)))) & 0x0FU)) *
          0.25F;
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
      const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
        sums[r][t] += scale[r] * (ds.x * static_cast<float>(dot));
      }
    }
  }
}

// v3 IQ4_NL: block_iq4_nl = fp16 d + qs[16] (18 B, so
// blocks are only 2-byte aligned: every load goes through memcpy). The kernel
// treats 8 consecutive blocks (256 elements, 144 B) as one virtual super-block,
// so a row is k/256 of them and a half = 4 blocks. Inside a block the nibble
// order is IQ4_XS's sub-block order (low = elements 0-15, high = 16-31) and the
// values come from the same kvalues_iq4nl table; only the scale differs: d of
// the block (ggml dequantize_row_iq4_nl), no 6-bit ls. Slot order = Iq4Xs's.
// Rows: that many adjacent rows share each LDS activation read (4 at Batch >= 6);
// per-(row, token) order is the Rows == 1 order.
template<std::size_t Batch, std::uint32_t Rows>
__device__ __forceinline__ void Iq4NlHalfBlockDot(
    const std::uint8_t* const (&bps)[Rows], std::uint32_t half,
    const uint4* __restrict__ stage_q, const float2* __restrict__ stage_ds,
    std::uint32_t slot_base, std::uint32_t tile_slots,
    float (&sums)[Rows][Batch]) {
  float d[Rows][4];
  uint4 qs[Rows][4];
#pragma unroll
  for (std::uint32_t r = 0; r < Rows; ++r) {
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c) {
      const std::uint8_t* const b = bps[r] + (18U * ((4U * half) + c));
      unsigned short h;
      __builtin_memcpy(&h, b, 2);
      d[r][c] = __half2float(__ushort_as_half(h));
      __builtin_memcpy(&qs[r][c], b + 2U, 16);
    }
  }
#pragma unroll
  for (std::uint32_t s = 0; s < 8U; ++s) {
    const std::uint32_t shift = 4U * (s & 1U);
    std::uint32_t q[Rows][4];
#pragma unroll
    for (std::uint32_t r = 0; r < Rows; ++r) {
      const uint4 word = qs[r][s >> 1U];
      const std::uint32_t words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        q[r][i] = Iq4NlLookup4((words[i] >> shift) & 0x0F0F0F0FU);
    }
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const uint4 qx = stage_q[(t * tile_slots) + slot_base + s];
      const float2 ds = stage_ds[(t * tile_slots) + slot_base + s];
#pragma unroll
      for (std::uint32_t r = 0; r < Rows; ++r) {
        int dot = 0;
        dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
        dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
        sums[r][t] += d[r][s >> 1U] * (ds.x * static_cast<float>(dot));
      }
    }
  }
}

// v3 (written to make Q5_K read faster): Q4_K / Q5_K at the
// DRAM roof. v1/v2 give each lane ONE 16-element sub-block through
// DecodeQuantSub16: 16 B of qs of which only one nibble is used (the other is
// a sibling sub-block's), 16 B of qh for one bit per byte, the scale header
// again -- ~48 B touched per 11 B of weight, and a wave holds 2 super-blocks
// (352 B) in flight. Here a lane owns HALF a super-block (128 elements, the
// same split llama.cpp's vec_dot_q5_K_q8_1 uses): one 16-B header load, one
// 16-B qh load (Q5_K only), four 16-B qs loads, every byte used for all eight
// of its sub-blocks -> 96 B loaded per 88 B of weight (Q5_K), and a wave
// keeps 16 super-blocks (2.8 KB) in flight.
//
// Lanes: RowsPerWave rows per wave, kLanesPerRow = 32 / RowsPerWave lanes per
// row, one half-block each per tile, so a tile is kLanesPerRow / 2 super-
// blocks of K. All waves of the block share the tile's int8 activations in
// LDS (double-buffered like v2); lanes of different rows read the same LDS
// address (broadcast). Every lane reaches the final XorLane reduction, so no
// lane returns early. Q4_K/Q5_K need 16-byte aligned weight rows: a row is
// (k/256) * 144 or 176 bytes, a multiple of 16, so only the base pointer is
// checked (TryLaunchKQuantV3). Q6_K rows (210-byte blocks) are NOT 16-byte
// aligned; Q6KHalfBlockDot reads them with memcpy, never a uint4 dereference.
// Math = DecodeQuantSub16's Q4_K/Q5_K case; Q6_K = ggml dequantize_row_q6_K.
//
// Measured: at Batch 8 the kernel loses half its waves
// (VGPR 64-80 -> 152-176, LDS up to 27.6 KB per block; 128 KB per WGP). One
// stage buffer instead of two (half the LDS, one extra barrier per tile) is
// +5.5% for Q4_K at B8 but -4..-5% for Q5_K at B7-8, and capping VGPRs
// (amdgpu_waves_per_eu 12 -> 96 VGPR, no spill) alone changed nothing. So only
// Q4_K at B8 single-buffers.
//
// RowsPerLane: a lane computes that many ADJACENT rows
// from one LDS read of each activation sub-block (Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL;
// Q6_K at 2 for any Batch and 4 for Batch >= 6, like IQ4_NL/IQ4_XS; and
// IQ3_S/IQ3_XXS at 2 for Batch >= 4 and 4 for Batch >= 6). B8 issues
// 8x the LDS reads of B1 for 3x the VALU (measured); two rows halve the reads per
// weight byte. Same per-(row, token) accumulation order -> bit-exact with 1.
// Prior art: llama.cpp mmvq RDNA3.5 rows-per-block 2-4 at ncols 2-8.
//
// Four rows (Batch >= 6, like IQ3 x4): the per-row register arrays double
// again (head + qh 16 each, qs 64, sums_all 32; isum 32 only x4k):
// arithmetic minimum ~184 hot VGPRs non-ActK / ~216 ActK at B8, and the x2
// kernel measured ~30-50 above its own minimum (152-176), so expect
// ~215-240 -- Q5_K x4k at B8 may cross the ~230 scratch-risk line; bench
// decides per geometry.
// WaveSize: the wave size this instantiation is compiled for.
// It only names the symbol apart, so a -mwavefrontsize64 TU's copy is not
// merged with the wave32 one at link time; the kernel traps if it disagrees.
// k95small: the body takes its workgroup index as `block`, so a fused
// launch (k95_small.hpp) can run it on a sub-range of its grid. The kernel
// below passes blockIdx.x; per-row arithmetic does not depend on `block`.
// RowGroup (headq6, Q6_K only): 0 or RowsPerLane = this body verbatim;
// 1 or 2 = Q6KHalfBlockDot's Group (see there), bit-exact.
template<core::GgmlType WType, std::size_t Batch, std::uint32_t RowsPerWave,
         std::uint32_t Waves, std::uint32_t RowsPerLane = 1, bool ActK = false,
         std::uint32_t WaveSize = 32, std::uint32_t RowGroup = 0>
__device__ __forceinline__ void SmallBatchDot4KQuantV3Body(
    const void* __restrict__ w, const std::int8_t* __restrict__ xq,
    const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
    std::uint32_t k, std::uint32_t block_index) {
  static_assert(WType == core::GgmlType::kQ4_K ||
                WType == core::GgmlType::kQ5_K ||
                WType == core::GgmlType::kQ6_K ||
                WType == core::GgmlType::kIQ4_XS ||
                WType == core::GgmlType::kIQ4_NL ||
                WType == core::GgmlType::kQ3_K ||
                WType == core::GgmlType::kIQ3_S ||
                WType == core::GgmlType::kIQ3_XXS ||
                WType == core::GgmlType::kIQ2_S);
  static_assert(Batch >= 1 && Batch <= 8);
  if (__builtin_amdgcn_wavefrontsize() != WaveSize) __builtin_trap();
  // RowsPerWave 1 would stage 16 super-blocks per token (110 KB of LDS at B8).
  static_assert(RowsPerWave >= 2 && RowsPerWave <= 16 &&
                (32U % RowsPerWave) == 0U);
  static_assert(
      RowsPerLane == 1U ||
      ((RowsPerLane == 2U) &&
       (WType == core::GgmlType::kQ4_K || WType == core::GgmlType::kQ5_K ||
        WType == core::GgmlType::kQ6_K || WType == core::GgmlType::kIQ4_XS ||
        WType == core::GgmlType::kIQ4_NL || WType == core::GgmlType::kQ3_K ||
        ((WType == core::GgmlType::kIQ3_S ||
          WType == core::GgmlType::kIQ3_XXS ||
          WType == core::GgmlType::kIQ2_S) &&
         Batch >= 4))) ||
      ((RowsPerLane == 4U) && Batch >= 6 &&
       (WType == core::GgmlType::kQ4_K || WType == core::GgmlType::kQ5_K ||
        WType == core::GgmlType::kQ6_K ||
        WType == core::GgmlType::kIQ4_XS ||
        WType == core::GgmlType::kIQ4_NL || WType == core::GgmlType::kQ3_K ||
        WType == core::GgmlType::kIQ3_S || WType == core::GgmlType::kIQ3_XXS ||
        WType == core::GgmlType::kIQ2_S)));
  // ActK: activations from QuantizeActivationsQ8KDot4Kernel
  // (Q4_K/Q5_K/Q6_K/IQ4_XS).
  static_assert(!ActK || WType == core::GgmlType::kQ4_K ||
                WType == core::GgmlType::kQ5_K ||
                WType == core::GgmlType::kQ6_K ||
                WType == core::GgmlType::kIQ4_XS);
  constexpr bool kQh = WType == core::GgmlType::kQ5_K;
  constexpr bool kQ6 = WType == core::GgmlType::kQ6_K;
  constexpr bool kIq4 = WType == core::GgmlType::kIQ4_XS;
  constexpr bool kIq4Nl = WType == core::GgmlType::kIQ4_NL;
  constexpr bool kQ3 = WType == core::GgmlType::kQ3_K;
  constexpr bool kIq3s = WType == core::GgmlType::kIQ3_S;
  constexpr bool kIq3xxs = WType == core::GgmlType::kIQ3_XXS;
  constexpr bool kIq2s = WType == core::GgmlType::kIQ2_S;
  constexpr bool kIqGrid = kIq3s || kIq3xxs || kIq2s;  // grid (+ksigns) in LDS
  // IQ4_NL: 8 blocks of 18 B = one 144-B virtual super-block (Iq4NlHalfBlockDot).
  constexpr std::uint32_t kBlockBytes =
      (kQ3 || kIq3s) ? 110U : kIq3xxs ? 98U : kIq2s ? 82U : (kIq4 ? 136U : (kQ6 ? 210U : (kQh ? 176U : 144U)));
  constexpr std::uint32_t kQsOffset = kQh ? 48U : 16U;
  constexpr std::uint32_t kLanesPerRow = 32U / RowsPerWave;
  constexpr std::uint32_t kTileBlocks = kLanesPerRow / 2U;
  constexpr std::uint32_t kTileSubs = kTileBlocks * 16U;
  // +2 entries per super-block: lanes of one row read sub-blocks 16 apart, so
  // without padding every pair of them hits the same LDS banks.
  constexpr std::uint32_t kBlockStride = 18U;
  constexpr std::uint32_t kTileSlots = kTileBlocks * kBlockStride;
  constexpr std::uint32_t kThreads = Waves * 32U;
  constexpr std::uint32_t kBuffers =
      (WType == core::GgmlType::kQ4_K && Batch >= 8U) ? 1U : 2U;
  __shared__ uint4 stage_q[kBuffers][Batch * kTileSlots];
  __shared__ float2 stage_ds[kBuffers][Batch * kTileSlots];
  __shared__ std::uint64_t fast_iq_grid[kIqGrid ? kIqFastGridWords<WType> : 1];
  __shared__ std::uint8_t fast_iq_signs[kIqGrid ? kIqFastSignBytes<WType> : 1];
  const std::uint32_t num_sub = k / 16U;
  const std::uint32_t num_blocks = k / 256U;
  const auto stage = [&](std::uint32_t buffer, std::uint32_t tile_block) {
    for (std::uint32_t i = threadIdx.x; i < Batch * kTileSubs; i += kThreads) {
      const std::uint32_t t = i / kTileSubs;
      const std::uint32_t local = i % kTileSubs;
      const std::uint32_t slot = (t * kTileSlots) +
                                 ((local / 16U) * kBlockStride) + (local % 16U);
      const std::uint32_t sub = (tile_block * 16U) + local;
      if (sub < num_sub) {
        stage_q[buffer][slot] =
            reinterpret_cast<const uint4*>(xq + (t * k))[sub];
        stage_ds[buffer][slot] = xds[(t * num_sub) + sub];
      } else {
        stage_q[buffer][slot] = {0U, 0U, 0U, 0U};
        stage_ds[buffer][slot] = {0.0F, 0.0F};
      }
    }
  };
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t unit = lane % kLanesPerRow;
  const std::uint32_t row0 =
      ((((block_index * Waves) + (threadIdx.x >> 5U)) * RowsPerWave) +
       (lane / kLanesPerRow)) *
      RowsPerLane;
  const std::uint32_t tile_block_of_lane = unit / 2U;
  const std::uint32_t half = unit % 2U;
  // Rows past m re-read row m-1 (cached) and skip the store.
  const std::uint8_t* row_bytes[RowsPerLane];
#pragma unroll
  for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
    const std::uint32_t row = row0 + r;
    row_bytes[r] = static_cast<const std::uint8_t*>(w) +
                   (static_cast<std::size_t>(row < m ? row : m - 1U) *
                    num_blocks * kBlockBytes);
  }
  float sums_all[RowsPerLane][Batch] = {};
  int isum[RowsPerLane][ActK ? Batch : 1] = {};  // ActK: one half-block's int sum
  float (&sums)[Batch] = sums_all[0];

  if constexpr (kIqGrid)  // ends with its own __syncthreads; no lane returns early
    StageIqFastTables<WType>(fast_iq_grid, fast_iq_signs, threadIdx.x, kThreads);
  stage(0, 0);
  __syncthreads();
  std::uint32_t buffer = 0;
  for (std::uint32_t tile_block = 0; tile_block < num_blocks;
       tile_block += kTileBlocks) {
    const bool more = tile_block + kTileBlocks < num_blocks;  // block-uniform
    if constexpr (kBuffers == 2U) {
      if (more)
        stage(buffer ^ 1U, tile_block + kTileBlocks);
    }
    const std::uint32_t block = tile_block + tile_block_of_lane;
    if constexpr (kIq4Nl) {
      if (block < num_blocks) {
        const std::uint8_t* bps[RowsPerLane];
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerLane; ++r)
          bps[r] = row_bytes[r] + (static_cast<std::size_t>(block) * kBlockBytes);
        Iq4NlHalfBlockDot<Batch, RowsPerLane>(
            bps, half, stage_q[buffer], stage_ds[buffer],
            (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
            sums_all);
      }
    } else if constexpr (kIq4) {
      if (block < num_blocks) {
        const std::uint8_t* bps[RowsPerLane];
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerLane; ++r)
          bps[r] = row_bytes[r] + (static_cast<std::size_t>(block) * kBlockBytes);
        Iq4XsHalfBlockDot<Batch, RowsPerLane, ActK>(
            bps, half, stage_q[buffer], stage_ds[buffer],
            (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
            sums_all);
      }
    } else if constexpr (kQ3) {
      if (block < num_blocks) {
        const std::uint8_t* bps[RowsPerLane];
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerLane; ++r)
          bps[r] = row_bytes[r] + (static_cast<std::size_t>(block) * kBlockBytes);
        Q3KHalfBlockDot<Batch, RowsPerLane>(
            bps, half, stage_q[buffer], stage_ds[buffer],
            (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
            sums_all);
      }
    } else if constexpr (kIq3s || kIq3xxs) {
      if (block < num_blocks) {
        if constexpr (RowsPerLane == 2U || RowsPerLane == 4U) {
          const std::uint8_t* bps[RowsPerLane];
#pragma unroll
          for (std::uint32_t r = 0; r < RowsPerLane; ++r)
            bps[r] =
                row_bytes[r] + (static_cast<std::size_t>(block) * kBlockBytes);
          Iq3MultiRowHalfBlockDot<kIq3s, Batch, RowsPerLane>(
              bps, half, stage_q[buffer], stage_ds[buffer],
              (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
              sums_all, reinterpret_cast<const std::uint32_t*>(fast_iq_grid),
              fast_iq_signs);
        } else if constexpr (kIq3s) {
          Iq3sHalfBlockDot<Batch>(
              row_bytes[0] + (static_cast<std::size_t>(block) * kBlockBytes),
              half, stage_q[buffer], stage_ds[buffer],
              (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
              sums, reinterpret_cast<const std::uint32_t*>(fast_iq_grid));
        } else {
          Iq3xxsHalfBlockDot<Batch>(
              row_bytes[0] + (static_cast<std::size_t>(block) * kBlockBytes),
              half, stage_q[buffer], stage_ds[buffer],
              (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
              sums, reinterpret_cast<const std::uint32_t*>(fast_iq_grid),
              fast_iq_signs);
        }
      }
    } else if constexpr (kIq2s) {
      if (block < num_blocks) {
        const std::uint8_t* bps[RowsPerLane];
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerLane; ++r)
          bps[r] = row_bytes[r] + (static_cast<std::size_t>(block) * kBlockBytes);
        Iq2sHalfBlockDot<Batch, RowsPerLane>(
            bps, half, stage_q[buffer], stage_ds[buffer],
            (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
            sums_all, fast_iq_grid);
      }
    } else if constexpr (kQ6) {
      if (block < num_blocks) {
        const std::uint8_t* bps[RowsPerLane];
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerLane; ++r)
          bps[r] = row_bytes[r] + (static_cast<std::size_t>(block) * kBlockBytes);
        Q6KHalfBlockDot<Batch, RowsPerLane, ActK,
                        RowGroup == 0U ? RowsPerLane : RowGroup>(
            bps, half, stage_q[buffer], stage_ds[buffer],
            (tile_block_of_lane * kBlockStride) + (8U * half), kTileSlots,
            sums_all);
      }
    } else if (block < num_blocks) {
      uint4 head[RowsPerLane];
      uint4 qh[RowsPerLane];
      uint4 qs[RowsPerLane][4];
#pragma unroll
      for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
        const std::uint8_t* const bp =
            row_bytes[r] + (static_cast<std::size_t>(block) * kBlockBytes);
        head[r] = *reinterpret_cast<const uint4*>(bp);
        qh[r] = {0U, 0U, 0U, 0U};
        if constexpr (kQh)
          qh[r] = *reinterpret_cast<const uint4*>(bp + 16U + (16U * half));
#pragma unroll
        for (std::uint32_t g = 0; g < 4U; ++g)
          qs[r][g] = *reinterpret_cast<const uint4*>(bp + kQsOffset +
                                                     (32U * g) + (16U * half));
      }
      const std::uint32_t slot_base = (tile_block_of_lane * kBlockStride) + half;
#pragma unroll
      for (std::uint32_t g = 0; g < 4U; ++g) {
#pragma unroll
        for (std::uint32_t n = 0; n < 2U; ++n) {
          // get_scale_min_k4(j): scales bytes 0-3 = head.y, 4-7 = head.z,
          // 8-11 = head.w (GetQKScaleMin, ggml).
          const std::uint32_t j = (2U * g) + n;
          const std::uint32_t b = 8U * (j & 3U);
          float scale[RowsPerLane];
          float offset[RowsPerLane];
          std::uint32_t sc_int[RowsPerLane];
          std::uint32_t q[RowsPerLane][4];
#pragma unroll
          for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
            const uint4 hd = head[r];
            const float d = __half2float(
                __ushort_as_half(static_cast<unsigned short>(hd.x & 0xFFFFU)));
            const float dmin = __half2float(
                __ushort_as_half(static_cast<unsigned short>(hd.x >> 16U)));
            std::uint32_t sc;
            std::uint32_t mn;
            if (j < 4U) {
              sc = (hd.y >> b) & 63U;
              mn = (hd.z >> b) & 63U;
            } else {
              sc = ((hd.w >> b) & 15U) | (((hd.y >> b) >> 6U & 3U) << 4U);
              mn = ((hd.w >> b) >> 4U & 15U) |
                   (((hd.z >> b) >> 6U & 3U) << 4U);
            }
            scale[r] = d * static_cast<float>(sc);
            offset[r] = dmin * static_cast<float>(mn);
            sc_int[r] = sc;
            const std::uint32_t words[4] = {qs[r][g].x, qs[r][g].y, qs[r][g].z,
                                            qs[r][g].w};
            const std::uint32_t highs[4] = {qh[r].x, qh[r].y, qh[r].z, qh[r].w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              q[r][i] = (words[i] >> (4U * n)) & 0x0F0F0F0FU;
              if constexpr (kQh)
                q[r][i] |= ((highs[i] >> j) & 0x01010101U) << 4U;
            }
          }
          // Sub-block (2j + half) of this super-block.
          const std::uint32_t slot = slot_base + (4U * g) + (2U * n);
#pragma unroll
          for (std::size_t t = 0; t < Batch; ++t) {
            const uint4 qx = stage_q[buffer][(t * kTileSlots) + slot];
            if constexpr (ActK) {
              // Integer only: isum += sc_j * dot_j. |dot| <= 16*31*127, sc <= 63,
              // 8 sub-blocks: |isum| < 2^25. __mul24 (v_mul/mad_i32_i24): both
              // factors fit 24 bits, and a plain int * compiled to v_mul_lo_u32
              // (227 per B8 x2 loop in the ISA dump); RDNA3.5 ISA on V_MUL_LO_U32: "for
              // integers with small magnitudes consider V_MUL_U32_U24".
#pragma unroll
              for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
                int dot = 0;
                dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
                dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
                dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
                dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
                isum[r][t] += __mul24(static_cast<int>(sc_int[r]), dot);
              }
            } else {
              const float2 ds = stage_ds[buffer][(t * kTileSlots) + slot];
#pragma unroll
              for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
                int dot = 0;
                dot = __builtin_amdgcn_sudot4(true, q[r][0], true, qx.x, dot, false);
                dot = __builtin_amdgcn_sudot4(true, q[r][1], true, qx.y, dot, false);
                dot = __builtin_amdgcn_sudot4(true, q[r][2], true, qx.z, dot, false);
                dot = __builtin_amdgcn_sudot4(true, q[r][3], true, qx.w, dot, false);
                sums_all[r][t] += (scale[r] * (ds.x * static_cast<float>(dot))) -
                                  (offset[r] * ds.y);
              }
            }
          }
        }
      }
      if constexpr (ActK) {
        // Once per half-block: the min term from the 8 sub-block sums, then
        // x_d * (d * isum - dmin * imin) per (row, token). Each token's sums are
        // read from LDS once for all RowsPerLane rows.
        float dd[RowsPerLane];
        float dm[RowsPerLane];
        std::uint32_t mn8[RowsPerLane][8];
#pragma unroll
        for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
          const uint4 hd = head[r];
          dd[r] = __half2float(
              __ushort_as_half(static_cast<unsigned short>(hd.x & 0xFFFFU)));
          dm[r] = __half2float(
              __ushort_as_half(static_cast<unsigned short>(hd.x >> 16U)));
#pragma unroll
          for (std::uint32_t j = 0; j < 8U; ++j) {
            const std::uint32_t b = 8U * (j & 3U);
            mn8[r][j] = j < 4U ? ((hd.z >> b) & 63U)
                               : (((hd.w >> b) >> 4U & 15U) |
                                  (((hd.z >> b) >> 6U & 3U) << 4U));
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t) {
          const float2* dst = &stage_ds[buffer][(t * kTileSlots) + slot_base];
          int bs[8];
#pragma unroll
          for (std::uint32_t j = 0; j < 8U; ++j)
            bs[j] = __float_as_int(dst[2U * j].y);
          const float xd = dst[0].x;
#pragma unroll
          for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
            int imin = 0;
#pragma unroll
            for (std::uint32_t j = 0; j < 8U; ++j)
              imin += __mul24(static_cast<int>(mn8[r][j]), bs[j]);  // |bs| <= 16*127
            sums_all[r][t] += xd * ((dd[r] * static_cast<float>(isum[r][t])) -
                                    (dm[r] * static_cast<float>(imin)));
            isum[r][t] = 0;
          }
        }
      }
    }
    __syncthreads();
    if constexpr (kBuffers == 2U) {
      buffer ^= 1U;
    } else if (more) {
      // One buffer: every wave has finished reading it (barrier above).
      stage(0, tile_block + kTileBlocks);
      __syncthreads();
    }
  }

  // Sum the row's kLanesPerRow lanes (an aligned power-of-two group).
#pragma unroll
  for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
#pragma unroll
      for (std::uint32_t offset = kLanesPerRow / 2U; offset > 0U; offset >>= 1U)
        sums_all[r][t] += XorLane(sums_all[r][t], static_cast<int>(offset));
    }
  }
#pragma unroll
  for (std::uint32_t r = 0; r < RowsPerLane; ++r) {
    const std::uint32_t row = row0 + r;
    if (row >= m)
      continue;
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      if (t % kLanesPerRow == unit)
        y[(t * m) + row] = sums_all[r][t];
    }
  }
}

template<core::GgmlType WType, std::size_t Batch, std::uint32_t RowsPerWave,
         std::uint32_t Waves, std::uint32_t RowsPerLane = 1, bool ActK = false,
         std::uint32_t WaveSize = 32, std::uint32_t RowGroup = 0>
__launch_bounds__(Waves * 32) __global__
    void SmallBatchDot4KQuantV3Kernel(const void* __restrict__ w,
                                      const std::int8_t* __restrict__ xq,
                                      const float2* __restrict__ xds,
                                      float* __restrict__ y, std::uint32_t m,
                                      std::uint32_t k) {
  SmallBatchDot4KQuantV3Body<WType, Batch, RowsPerWave, Waves, RowsPerLane,
                             ActK, WaveSize, RowGroup>(w, xq, xds, y, m, k,
                                                       blockIdx.x);
}

// quantize = false (k95small): the caller already wrote xq/xds from the same
// x with the same quantizer kind (ActK or not) on this stream.
template<core::GgmlType WType, std::size_t Batch, std::uint32_t RowsPerWave,
         std::uint32_t Waves, std::uint32_t RowsPerLane = 1, bool ActK = false,
         std::uint32_t WaveSize = 32, std::uint32_t RowGroup = 0>
void LaunchSmallBatchDot4V3(const void* w, const float* x, float* y,
                            std::int8_t* xq, float2* xds, std::uint32_t m,
                            std::uint32_t k, hipStream_t stream,
                            bool quantize = true) {
  const std::uint32_t subs = static_cast<std::uint32_t>(Batch) * (k / 16U);
  if (quantize) {
    if constexpr (ActK)
      QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256, 0,
                                         stream>>>(x, xq, xds, subs);
    else
      QuantizeActivationsDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
          x, xq, xds, subs);
  }
  constexpr std::uint32_t kRowsPerBlock = Waves * RowsPerWave * RowsPerLane;
  SmallBatchDot4KQuantV3Kernel<WType, Batch, RowsPerWave, Waves, RowsPerLane, ActK,
                               WaveSize, RowGroup>
      <<<(m + kRowsPerBlock - 1) / kRowsPerBlock, Waves * 32, 0, stream>>>(
          w, xq, xds, y, m, k);
}

// Quantize, then GEMM, on one stream. xq needs batch*k bytes, xds batch*k/16
// float2; both must stay untouched until the GEMM has run. x, xq, xds must be
// 16-byte aligned and k a multiple of 256.
template<core::GgmlType WType, std::size_t Batch, std::size_t RowsPerWave,
         std::uint32_t Waves, bool FastIq = false>
void LaunchSmallBatchDot4(const void* w, const float* x, float* y,
                          std::int8_t* xq, float2* xds, std::uint32_t m,
                          std::uint32_t k, hipStream_t stream) {
  const std::uint32_t subs = static_cast<std::uint32_t>(Batch) * (k / 16U);
  QuantizeActivationsDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
      x, xq, xds, subs);
  constexpr std::uint32_t kRowsPerBlock = Waves * RowsPerWave;
  SmallBatchDot4GEMMKernel<WType, Batch, RowsPerWave, Waves, FastIq>
      <<<(m + kRowsPerBlock - 1) / kRowsPerBlock, Waves * 32, 0, stream>>>(
          w, xq, xds, y, m, k);
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_DOT4_HPP_

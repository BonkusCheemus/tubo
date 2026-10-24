#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SN_Q4K_K4_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SN_Q4K_K4_HPP_

// Q4_K/Q5_K GEMV/skinny-GEMM for gfx1151 on
// native ggml blocks (144/176 B) or a 32-row chunk interleave, in the k4
// structure of OpusIq3sB8Kernel (sn_iq3.hip, arm opus_b8_k4prio):
//
//   * RowWaves row-waves x KPhases K-phases per workgroup, 512 threads at
//     4 x 4; one lane owns one output row for the whole of K, the KPhases
//     waves split K and their partials are added through LDS in fixed phase
//     order (phase 0 + 1 + ... + KPhases-1);
//   * the q8_K int8 activations (QuantizeActivationsQ8KDot4Kernel: one scale
//     per 256) are SGPR operands of v_dot4, loaded wave-uniform right before
//     the sub-block pair that consumes them, so a whole 256-weight block
//     accumulates in int32 and folds to float once;
//   * weights are read IN BODY (WeightPace::kInBody,
//   ref/xxs_k4ns_sn_iq3xxs.hip):
//     the block's 144/176 bytes are loaded in the loop iteration that consumes
//     them, never held one block ahead (no extra prefetched block at B8);
//   * optional split-K S (1, 2, 4): blockIdx.y = split, each split owns a
//     contiguous span of K blocks and the partials of the S splits of a row
//     group are added by the LAST-arriving workgroup in fixed split order
//     (SnQ4KSplitKCombine, the deterministic b8sk combine).
//
// ggml block_q4_K: {half d; half dmin; uint8 scales[12]; uint8 qs[128]}.
// A stored weight is w(i) = (d * sc_j) * q_i - (dmin * mn_j), j = i / 32; q_i
// is a nibble of qs and (sc_j, mn_j) are the six-bit fields of
// get_scale_min_k4(j, scales). The 128 qs bytes are 4 groups of 32; group gp
// holds the low nibbles of sub-block 2gp and the high nibbles of 2gp+1, in
// natural element order (byte l of the group is weight l and weight l+32 of
// the group's 64 weights). Q5_K inserts 32 qh bytes before qs: bit j of
// qh[l] is the fifth bit of byte l in sub-block j. The head is unchanged.
//
// The min term uses the activation block sums (already in xds), exactly as
// dot4v3's ActK path: with x_i ~= xd * xq_i,
//   y = sum_i w_i x_i = xd * (d * sum_j sc_j dot_j - dmin * sum_j mn_j A_j).
// dot4v3 splits each 32-weight sub-block across two lanes (RowsPerWave/2) and
// flushes once per 128-weight half-block; here one lane owns the whole
// sub-block and flushes once per 256-weight block, and the KPhases cross-wave
// add is an fp32 fp add through LDS. The integer sub-block dot_j and sum A_j
// are bit-identical; only the fp32 summation order differs.
//
// INT32 BOUND (Q5_K, also covers Q4_K): q <= 31, |xq| <= 127, so
// |one v_dot4| <= 4*31*127 = 15,748 and |dot_j| over 32 weights <= 125,984
// < 2^17; sc <= 63, so |sc_j * dot_j| <= 7,936,992 and |isum| over 8
// sub-blocks <= 63,495,936 < 2^26. |A_j| <= 32*127 = 4064 and
// |mn_j * A_j| <= 256,032, so |imin| <= 2,048,256 < 2^21. Both __mul24
// operands (sc/mn <= 63, dot/A <= 125,984) fit 24 signed bits, and both
// accumulators are flushed every 256 weights.
// Q6_K unsigned dot16 <= 128016; signed scale magnitude <= 128. Unsigned
// |isum| <= 262176768 < 2^28; |ioff| <= 4161536 < 2^22. The corrected
// isum - 32*ioff fits int32 (tight bound 133169152 < 2^27), and every
// __mul24 operand fits signed 24 bits. Q6's packed record keeps each A16.
// IQ4_XS: signed table values |q|<=127, |xq|<=127, |ls-32|<=32.
// |dot16|<=258064<2^18; |isum256|<=132128768<2^27; mul24 operands fit.
// Symmetric codebook: no min-term or packed activation sums are needed.
//
// Q4_K INSTRUCTION COUNT per 32 weights per row per token (steady state, B8):
//   8 v_dot4 (the 32-weight dot), and per 256 weights: 64 nibble-mask VALU,
//   ~46 scale/min decode VALU (once per row, amortized over Batch tokens),
//   8 __mul24 (sub-block scale fold, per token), min fold ~19 VALU per token
//   per 256, 9 weight loads (1 uint4 head + 8 uint4 qs), ~4 s_load_b512
//   activation loads per 64 weights per 4 tokens.
// Matches dot4v3 ActK's per-token dot4 count; the non-dot VALU is decode +
// min fold, and there is no LDS table and no sign path (Q4_K has no codebook).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "src/models/qwen/rows32_layout.hpp"

namespace gufo::hip {

// Keep the existing bench API; the CPU loader uses the same permutation.
using models::InterleaveKQuantRows32;

namespace sn_q4k_k4_detail {

// Shared with the CPU contract test; only the existing codebook lookup is
// target-specific. sub16 follows natural activation order, 0..15.
#if defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_K4_HOST_ONLY)
__attribute__((host, device))
#endif
constexpr int Iq4XsScale(std::uint32_t head_x, std::uint32_t head_y,
                         std::uint32_t sub16) noexcept {
  const std::uint32_t j = sub16 >> 1U;
  return static_cast<int>(((head_y >> (4U * j)) & 15U) |
                          (((head_x >> (16U + 2U * j)) & 3U) << 4U)) -
         32;
}

#if defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_K4_HOST_ONLY)
__attribute__((host, device))
#endif
constexpr std::uint32_t Iq4XsIndices4(std::uint32_t word,
                                      std::uint32_t sub16) noexcept {
  return (word >> (4U * (sub16 & 1U))) & 0x0F0F0F0FU;
}

}  // namespace sn_q4k_k4_detail

}  // namespace gufo::hip

#if defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_K4_HOST_ONLY)

#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"

namespace gufo::hip {

namespace sn_q4k_k4_detail {

// ggml get_scale_min_k4(j, scales) on the block head uint4: head.x holds
// {d, dmin} as two fp16, head.y = scales[0..3], head.z = scales[4..7],
// head.w = scales[8..11].
__device__ __forceinline__ void Q4KScaleMin(const uint4& head, std::uint32_t j,
                                            std::uint32_t* sc,
                                            std::uint32_t* mn) {
  const std::uint32_t b = 8U * (j & 3U);
  if (j < 4U) {
    *sc = (head.y >> b) & 63U;
    *mn = (head.z >> b) & 63U;
  } else {
    *sc = ((head.w >> b) & 15U) | ((((head.y >> b) >> 6U) & 3U) << 4U);
    *mn = (((head.w >> b) >> 4U) & 15U) | ((((head.z >> b) >> 6U) & 3U) << 4U);
  }
}

// Cross-WG combine of split-K partials, a copy of SnbSplitKCombine
// (sn_iq3.hip): every split parks its row-group partials in
// scratch[(split*Batch + t)*m + row]; the LAST workgroup of the row group to
// arrive (counter[ticket]) adds the S partials in split order 0..S-1 and the
// caller stores y; the ticket is reset for the next launch. Agent-scope fences
// make the partials visible across CUs; the loads are agent-scope atomics so
// they cannot hit a stale L0 line. Returns true in the WG that must store y.
template<std::size_t Batch>
__device__ __forceinline__ bool SnQ4KSplitKCombine(
    float (&sums)[Batch], std::uint32_t* flag, float* scratch,
    std::uint32_t* counter, std::uint32_t ticket, std::uint32_t m,
    std::uint32_t row0, std::uint32_t split, std::uint32_t splits,
    bool phase0) {
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

}  // namespace sn_q4k_k4_detail

// opt76-240a packed min-term sums: per (K block b, token t) the 8 sub-block
// activation sums A_j = xds[2j].y + xds[2j+1].y (|A_j| <= 32*127 = 4064, fits
// int16) as 4 dwords {A_2k | A_2k+1 << 16}, then per block the Batch xd
// floats: record b = 5*Batch dwords, [4t..4t+3] = token t's pairs,
// [4*Batch + t] = token t's xd. One wave-uniform record per block, so the
// k4 kernel reads its min-term inputs with one scalar wait per block instead
// of one per token. Same integers, same floats: bit-identical y. This is a
// quantize-side step (fusable into QuantizeActivationsQ8KDot4Kernel).
// Q6 keeps all sixteen A16 sums instead: eight packed dwords per token,
// followed by Batch xd words (9*Batch dwords/block). |A16| <= 2032.
template<std::size_t Batch, bool Q6 = false>
__global__ void SnQ4KPackSumsKernel(const float2* __restrict__ xds,
                                    std::int32_t* __restrict__ xsum,
                                    std::uint32_t blocks) {
  const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= blocks * Batch)
    return;
  const std::uint32_t b = i / Batch;
  const std::uint32_t t = i % Batch;
  const float2* const src = xds + (t * (blocks * 16U)) + (b * 16U);
  constexpr std::uint32_t kPairs = Q6 ? 8U : 4U;
  std::int32_t* const rec = xsum + (b * (kPairs + 1U) * Batch);
#pragma unroll
  for (std::uint32_t q = 0; q < kPairs; ++q) {
    const int lo =
        Q6 ? __float_as_int(src[2U * q].y)
           : __float_as_int(src[4U * q].y) + __float_as_int(src[4U * q + 1U].y);
    const int hi = Q6 ? __float_as_int(src[2U * q + 1U].y)
                      : __float_as_int(src[4U * q + 2U].y) +
                            __float_as_int(src[4U * q + 3U].y);
    rec[(kPairs * t) + q] =
        static_cast<std::int32_t>((static_cast<std::uint32_t>(lo) & 0xFFFFU) |
                                  (static_cast<std::uint32_t>(hi) << 16U));
  }
  rec[(kPairs * Batch) + t] = __float_as_int(src[0].x);
}

// y[token][row] = sum_k W[row][k] * x[token][k], W in ggml Q4_K/Q5_K layout
// or its chunk interleave; x is int8 with one scale per 256 (xq/xds).
// One lane owns one output row; the KPhases waves of a workgroup split K.
// `scratch`/`counter` are only read when SplitK (the split-K combine).
//
// Interleaved: the weights are InterleaveKQuantRows32's
// layout, so the 32 lanes of a wave read one chunk of 32 rows as 512
// contiguous bytes instead of 32 strided 16-byte pieces. Same bytes, same math.
//
// Hoist (opt76-240a): load the block's 8 qs chunks together with head/qh at
// the top of the block iteration. Without it the sched_barrier(0) fences of
// the token quads keep each group's qs loads inside its group, so a wave
// waits on DRAM 5 times per block (ISA: vmcnt(0) after 5, 2, 1, 1, 2 loads).
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool SplitK, bool Interleaved = false, bool Q5 = false,
         bool Hoist = false, bool PackedSums = false, int Probe = 0,
         bool Dynamic = false, bool Q6 = false, bool IQ4 = false>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(16))) void SnQ4KK4Kernel(
        const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k, float* __restrict__ scratch,
        std::uint32_t* __restrict__ counter,
        const std::int32_t* __restrict__ xsum = nullptr) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(!IQ4 || (Interleaved && Hoist && !PackedSums && !Q5 && !Q6 &&
                         !SplitK && Probe == 0 && !Dynamic),
                "IQ4_XS uses hoisted split metadata, no min term/experiments");
  static_assert(!Q6 || (Interleaved && !Q5 && Probe == 0 && !Dynamic),
                "Q6 uses the split-d interleave, not Probe/Dynamic");
  static_assert(KPhases >= 1U && RowWaves >= 1U);
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  constexpr std::uint32_t kPartialFloats =
      (KPhases - 1U) * RowWaves * Batch * 32U;
  __shared__ float lds[kPartialFloats + 4U];
  __shared__ std::uint32_t sk_flag;
  // Dynamic (opt76-240a): the KPhases waves of one row-wave take their first
  // block statically (b_begin + phase) and every later one from this LDS
  // counter, so a wave that DRAM serves early takes more blocks and the
  // siblings reach the LDS reduce together (trace: static waits p90 22 us).
  // fp32 sums then depend on which wave took which block: not bitwise
  // run-to-run deterministic (rel error unchanged).
  // 32 slots per row-wave, one per lane: a per-lane (divergent) address keeps
  // LLVM's atomic optimizer from rewriting the claim into a lane scan, which
  // costs ~500 SGPR spills in this loop. Only lane 0's slot counts.
  __shared__ std::uint32_t dyn_next[Dynamic ? RowWaves * 32U : 1];
  if constexpr (!SplitK) {
    (void)scratch;
    (void)counter;
  }

  const std::uint32_t blocks = k >> 8U;
  // SplitK: blockIdx.y = split; each split owns `blocks/splits` contiguous K
  // blocks. Without SplitK these fold to 0 / blocks and the code is the plain
  // k4 path.
  const std::uint32_t splits = SplitK ? gridDim.y : 1U;
  const std::uint32_t split = SplitK ? blockIdx.y : 0U;
  const std::uint32_t b_begin = SplitK ? split * (blocks / splits) : 0U;
  const std::uint32_t b_end = SplitK ? b_begin + (blocks / splits) : blocks;

  const std::uint32_t lane = threadIdx.x & 31U;
  // readfirstlane: the compiler must see the wave index as uniform, or the
  // activation address (which depends on the phase) stops being an s_load.
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  const std::uint32_t row0 = (((blockIdx.x * RowWaves) + row_wave) * 32U) + lane;
  const std::uint32_t row = row0 < m ? row0 : m - 1U;
  const std::uint32_t num_sub = k >> 4U;
  // Probe 3/4 trace (opt76-240a): lane 0 of each wave writes 8 u64 wall-clock
  // stamps (100 MHz) to scratch: start, end of block iterations 0..3, after
  // the LDS reduce, end, hw id. 3 = full kernel, 4 = loads only.
  constexpr bool kTrace = Probe >= 3;
  constexpr bool kLoadOnly = Probe == 2 || Probe == 4;
  static_assert(!(kTrace && SplitK), "trace stamps live in the split scratch");
  std::uint64_t* const ts =
      kTrace ? reinterpret_cast<std::uint64_t*>(scratch) +
                   (8U * ((static_cast<std::size_t>(blockIdx.x) *
                           (RowWaves * KPhases)) + wave))
             : nullptr;
  std::uint32_t it = 0;
  if constexpr (kTrace) {
    if (lane == 0U) ts[0] = wall_clock64();
  }

  // Rows past m are clamped so their loads stay inside the last row; their
  // outputs are simply not stored.
  // Chunk c of block b, row r: (r/32 * blocks + b) * (32*block_bytes)
  // + c*512 + (r%32)*16. Q4_K has 9 chunks (4608 B), Q5_K 11 (5632 B).
  constexpr std::uint32_t kBlockBytes =
      IQ4 ? 136U : (Q6 ? 210U : (Q5 ? 176U : 144U));
  constexpr std::uint32_t kChunkBytes = IQ4 ? 128U : (Q6 ? 208U : kBlockBytes);
  constexpr std::uint32_t kBlockStride =
      Interleaved ? 32U * kChunkBytes : kBlockBytes;
  constexpr std::uint32_t kChunk = Interleaved ? 512U : 16U;
  constexpr std::uint32_t kQsChunk = Q5 ? 3U : 1U;
  const std::uint8_t* const row_base =
      static_cast<const std::uint8_t*>(w) +
      (Interleaved
           ? (static_cast<std::size_t>(row >> 5U) * blocks * kBlockStride) +
                 ((row & 31U) * 16U)
           : static_cast<std::size_t>(row) * (blocks * kBlockBytes));

  float sums[Batch] = {};
  if constexpr (Dynamic) {
    if (phase == 0U) dyn_next[(row_wave * 32U) + lane] = KPhases;
    __syncthreads();
  }
  const auto next_block = [&](std::uint32_t b) -> std::uint32_t {
    if constexpr (Dynamic) {
      // Branch-free: every lane adds (lane == 0) to its own slot; lane 0's
      // old value is the claimed block.
      const std::uint32_t v =
          atomicAdd(&dyn_next[(row_wave * 32U) + lane], lane == 0U ? 1U : 0U);
      return b_begin + __builtin_amdgcn_readfirstlane(v);
    } else {
      return b + KPhases;
    }
  };
  for (std::uint32_t b = b_begin + phase; b < b_end; b = next_block(b)) {
    const std::uint8_t* const bp =
        row_base + (static_cast<std::size_t>(b) * kBlockStride);

    if constexpr (IQ4) {
      const auto* const meta_plane =
          w + static_cast<std::size_t>(m) * blocks * 128U;
      const std::size_t meta_index =
          ((static_cast<std::size_t>(row / 32U) * blocks + b) * 32U) +
          row % 32U;
      const uint2 head =
          *reinterpret_cast<const uint2*>(meta_plane + meta_index * 8U);
      uint4 payload[8];
#pragma unroll
      for (std::uint32_t c = 0; c < 8U; ++c)
        payload[c] = *reinterpret_cast<const uint4*>(bp + c * kChunk);
      __builtin_amdgcn_sched_barrier(0);
      const float d = __half2float(
          __ushort_as_half(static_cast<unsigned short>(head.x & 0xFFFFU)));
      int isum[Batch] = {};
#pragma unroll
      for (std::uint32_t gp = 0; gp < 4U; ++gp) {
        std::uint32_t q[4][4];
        int scale[4];
#pragma unroll
        for (std::uint32_t sub = 0; sub < 4U; ++sub) {
          const std::uint32_t sub16 = 4U * gp + sub;
          const uint4 packed = payload[2U * gp + (sub >> 1U)];
          const std::uint32_t words[4] = {packed.x, packed.y, packed.z,
                                          packed.w};
          scale[sub] = sn_q4k_k4_detail::Iq4XsScale(head.x, head.y, sub16);
#pragma unroll
          for (std::uint32_t i = 0; i < 4U; ++i)
            q[sub][i] =
                Iq4NlLookup4(sn_q4k_k4_detail::Iq4XsIndices4(words[i], sub16));
        }
#pragma unroll
        for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
          uint4 a[4][4];
#pragma unroll
          for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
            const auto* const p = reinterpret_cast<const uint4*>(
                xq + (t0 + j) * k + b * 256U + gp * 64U);
#pragma unroll
            for (std::uint32_t sub = 0; sub < 4U; ++sub)
              a[j][sub] = p[sub];
          }
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
#pragma unroll
            for (std::uint32_t sub = 0; sub < 4U; ++sub) {
              const uint4 ax = a[j][sub];
              int dot = 0;
              dot = __builtin_amdgcn_sudot4(true, q[sub][0], true, ax.x, dot,
                                            false);
              dot = __builtin_amdgcn_sudot4(true, q[sub][1], true, ax.y, dot,
                                            false);
              dot = __builtin_amdgcn_sudot4(true, q[sub][2], true, ax.z, dot,
                                            false);
              dot = __builtin_amdgcn_sudot4(true, q[sub][3], true, ax.w, dot,
                                            false);
              isum[t0 + j] += __mul24(scale[sub], dot);
            }
          }
          __builtin_amdgcn_sched_barrier(0);
        }
      }
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        const float xd = xds[t * num_sub + b * 16U].x;
        sums[t] += (d * xd) * static_cast<float>(isum[t]);
      }
      continue;
    }
    if constexpr (Q6) {
      if constexpr (Hoist || PackedSums) {
        const std::uint8_t* const d_plane =
            static_cast<const std::uint8_t*>(w) +
            (static_cast<std::size_t>(m) * blocks * kChunkBytes);
        const std::size_t d_index =
            ((static_cast<std::size_t>(row / 32U) * blocks + b) * 32U) +
            (row % 32U);
        const unsigned short d_bits =
            *reinterpret_cast<const unsigned short*>(d_plane + (d_index * 2U));
        uint4 payload[13];
#pragma unroll
        for (std::uint32_t c = 0; c < 13U; ++c)
          payload[c] = *reinterpret_cast<const uint4*>(bp + (c * kChunk));
        __builtin_amdgcn_sched_barrier(0);
        const float d = __half2float(__ushort_as_half(d_bits));
        const uint4 scales = payload[12];
        const std::uint32_t sc_words[4] = {scales.x, scales.y, scales.z,
                                           scales.w};
        int isum[Batch] = {};
#pragma unroll
        for (std::uint32_t gp = 0; gp < 4U; ++gp) {
          const std::uint32_t h = gp >> 1U;
          std::uint32_t q[4][4];
#pragma unroll
          for (std::uint32_t sub = 0; sub < 4U; ++sub) {
            const uint4 ql = payload[4U * h + sub];
            const uint4 qh = payload[8U + 2U * h + (sub & 1U)];
            const std::uint32_t lw[4] = {ql.x, ql.y, ql.z, ql.w};
            const std::uint32_t hw[4] = {qh.x, qh.y, qh.z, qh.w};
#pragma unroll
            for (std::uint32_t i = 0; i < 4U; ++i)
              q[sub][i] = ((lw[i] >> (4U * (gp & 1U))) & 0x0F0F0F0FU) |
                          (((hw[i] >> (4U * (gp & 1U) + 2U * (sub >> 1U))) &
                            0x03030303U)
                           << 4U);
          }
          // A quad is four tokens x 64 bytes, as in the Q5 winner. The two
          // barriers prevent later quads/sub-blocks from retaining hundreds
          // of activation SGPRs across this quad.
#pragma unroll
          for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
            uint4 a[4][4];
#pragma unroll
            for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
              const uint4* const p = reinterpret_cast<const uint4*>(
                  xq + ((t0 + j) * k) + (b * 256U) + (gp * 64U));
#pragma unroll
              for (std::uint32_t sub = 0; sub < 4U; ++sub)
                a[j][sub] = p[sub];
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (std::size_t j = 0; j < 4U && t0 + j < Batch; ++j) {
#pragma unroll
              for (std::uint32_t sub = 0; sub < 4U; ++sub) {
                const uint4 ax = a[j][sub];
                int dot = 0;
                dot = __builtin_amdgcn_sudot4(true, q[sub][0], true, ax.x, dot,
                                              false);
                dot = __builtin_amdgcn_sudot4(true, q[sub][1], true, ax.y, dot,
                                              false);
                dot = __builtin_amdgcn_sudot4(true, q[sub][2], true, ax.z, dot,
                                              false);
                dot = __builtin_amdgcn_sudot4(true, q[sub][3], true, ax.w, dot,
                                              false);
                const std::uint32_t s = 4U * gp + sub;
                const int sc = static_cast<std::int8_t>(
                    (sc_words[s >> 2U] >> (8U * (s & 3U))) & 0xFFU);
                isum[t0 + j] += __mul24(sc, dot);
              }
            }
            __builtin_amdgcn_sched_barrier(0);
          }
        }
        if constexpr (PackedSums) {
          const std::int32_t* const rec =
              xsum + (static_cast<std::size_t>(b) * 9U * Batch);
          std::int32_t pk[9U * Batch];
#pragma unroll
          for (std::uint32_t i = 0; i < 9U * Batch; ++i)
            pk[i] = rec[i];
#pragma unroll
          for (std::size_t t = 0; t < Batch; ++t) {
            int ioff = 0;
#pragma unroll
            for (std::uint32_t sub = 0; sub < 16U; ++sub) {
              const std::int32_t pair = pk[8U * t + (sub >> 1U)];
              const int a_sum =
                  (sub & 1U) ? pair >> 16
                             : static_cast<int>(static_cast<short>(pair));
              const int sc = static_cast<std::int8_t>(
                  (sc_words[sub >> 2U] >> (8U * (sub & 3U))) & 0xFFU);
              ioff += __mul24(sc, a_sum);
            }
            const float xd = __int_as_float(pk[8U * Batch + t]);
            sums[t] += (d * xd) * static_cast<float>(isum[t] - 32 * ioff);
          }
        } else {
#pragma unroll
          for (std::size_t t = 0; t < Batch; ++t) {
            int ioff = 0;
#pragma unroll
            for (std::uint32_t sub = 0; sub < 16U; ++sub) {
              const int sc = static_cast<std::int8_t>(
                  (sc_words[sub >> 2U] >> (8U * (sub & 3U))) & 0xFFU);
              ioff += __mul24(
                  sc, __float_as_int(xds[t * num_sub + b * 16U + sub].y));
            }
            sums[t] += (d * xds[t * num_sub + b * 16U].x) *
                       static_cast<float>(isum[t] - 32 * ioff);
          }
        }
      } else {
        const std::uint8_t* const d_plane =
            static_cast<const std::uint8_t*>(w) +
            (static_cast<std::size_t>(m) * blocks * kChunkBytes);
        const std::size_t d_index =
            ((static_cast<std::size_t>(row / 32U) * blocks + b) * 32U) +
            (row % 32U);
        const unsigned short d_bits =
            *reinterpret_cast<const unsigned short*>(d_plane + (d_index * 2U));
        const float d = __half2float(__ushort_as_half(d_bits));
        const uint4 scales =
            *reinterpret_cast<const uint4*>(bp + (12U * kChunk));
        const std::uint32_t sc_words[4] = {scales.x, scales.y, scales.z,
                                           scales.w};
        int isum[Batch] = {};
        // In-body loads only. Keep just two ql chunks and one qh chunk live
        // per (128-weight half, 16-byte piece), not all 13 chunks at once.
#pragma unroll
        for (std::uint32_t h = 0; h < 2U; ++h) {
#pragma unroll
          for (std::uint32_t piece = 0; piece < 2U; ++piece) {
            const uint4 ql0 = *reinterpret_cast<const uint4*>(
                bp + ((4U * h + piece) * kChunk));
            const uint4 ql1 = *reinterpret_cast<const uint4*>(
                bp + ((4U * h + 2U + piece) * kChunk));
            const uint4 qh = *reinterpret_cast<const uint4*>(
                bp + ((8U + 2U * h + piece) * kChunk));
            const std::uint32_t hw[4] = {qh.x, qh.y, qh.z, qh.w};
#pragma unroll
            for (std::uint32_t g = 0; g < 4U; ++g) {
              const std::uint32_t sub = 8U * h + 2U * g + piece;
              const uint4 ql = (g & 1U) != 0U ? ql1 : ql0;
              const std::uint32_t lw[4] = {ql.x, ql.y, ql.z, ql.w};
              std::uint32_t q[4];
#pragma unroll
              for (std::uint32_t i = 0; i < 4U; ++i)
                q[i] = ((lw[i] >> (4U * (g >> 1U))) & 0x0F0F0F0FU) |
                       (((hw[i] >> (2U * g)) & 0x03030303U) << 4U);
              const int sc = static_cast<std::int8_t>(
                  (sc_words[sub >> 2U] >> (8U * (sub & 3U))) & 0xFFU);
#pragma unroll
              for (std::size_t t = 0; t < Batch; ++t) {
                const uint4 ax = *reinterpret_cast<const uint4*>(
                    xq + (static_cast<std::size_t>(t) * k) + (b * 256U) +
                    (sub * 16U));
                int dot = 0;
                dot =
                    __builtin_amdgcn_sudot4(true, q[0], true, ax.x, dot, false);
                dot =
                    __builtin_amdgcn_sudot4(true, q[1], true, ax.y, dot, false);
                dot =
                    __builtin_amdgcn_sudot4(true, q[2], true, ax.z, dot, false);
                dot =
                    __builtin_amdgcn_sudot4(true, q[3], true, ax.w, dot, false);
                const int a_sum =
                    __float_as_int(xds[(t * num_sub) + (b * 16U) + sub].y);
                isum[t] += __mul24(sc, dot - (32 * a_sum));
              }
            }
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < Batch; ++t) {
          const float xd = xds[(t * num_sub) + (b * 16U)].x;
          sums[t] += (d * xd) * static_cast<float>(isum[t]);
        }
      }
      continue;
    }
    // Shared head: {d, dmin} + 12 scale bytes. Q5_K has two qh chunks
    // before the eight qs chunks; Q4_K has no qh plane.
    const uint4 head = *reinterpret_cast<const uint4*>(bp);
    uint4 qh0{};
    uint4 qh1{};
    if constexpr (Q5) {
      qh0 = *reinterpret_cast<const uint4*>(bp + kChunk);
      qh1 = *reinterpret_cast<const uint4*>(bp + (2U * kChunk));
    }
    const std::uint32_t qhw[8] = {qh0.x, qh0.y, qh0.z, qh0.w,
                                  qh1.x, qh1.y, qh1.z, qh1.w};
    const float d = __half2float(
        __ushort_as_half(static_cast<unsigned short>(head.x & 0xFFFFU)));
    const float dmin = __half2float(
        __ushort_as_half(static_cast<unsigned short>(head.x >> 16U)));
    int sc[8];
    int mn[8];
#pragma unroll
    for (std::uint32_t j = 0; j < 8U; ++j) {
      std::uint32_t s = 0;
      std::uint32_t m8 = 0;
      sn_q4k_k4_detail::Q4KScaleMin(head, j, &s, &m8);
      sc[j] = static_cast<int>(s);
      mn[j] = static_cast<int>(m8);
    }
    // Probe (timing only, WRONG y): 1 = every block reads block 0's
    // activations and sums (always scalar-cache hot); 2 = loads only.
    const std::uint32_t ab = Probe == 1 ? 0U : b;
    const std::int8_t* const act_block =
        xq + (static_cast<std::size_t>(ab) * 256U);
    uint4 qs[Hoist ? 8 : 1];
    if constexpr (Hoist) {
#pragma unroll
      for (std::uint32_t c = 0; c < 8U; ++c)
        qs[c] = *reinterpret_cast<const uint4*>(bp + (kChunk * (kQsChunk + c)));
      // All 11 (Q5) / 9 (Q4) loads of the block issue before any use.
      __builtin_amdgcn_sched_barrier(0);
    }
    if constexpr (kLoadOnly) {
      std::uint32_t x = head.x ^ head.y ^ head.z ^ head.w ^ qh0.x ^ qh1.w;
#pragma unroll
      for (std::uint32_t c = 0; c < (Hoist ? 8U : 1U); ++c)
        x ^= qs[c].x ^ qs[c].y ^ qs[c].z ^ qs[c].w;
      sums[0] += static_cast<float>(x & 1U);
      if constexpr (kTrace) {
        __builtin_amdgcn_sched_barrier(0);
        ts[1U + (it & 3U)] = wall_clock64();  // all lanes, one address: no branch
        ++it;
      }
      continue;
    }
    int isum[Batch] = {};
#pragma unroll
    for (std::uint32_t gp = 0; gp < 4U; ++gp) {
      // Group gp's 32 qs bytes: low nibbles = sub-block 2gp, high = 2gp+1.
      uint4 c0;
      uint4 c1;
      if constexpr (Hoist) {
        c0 = qs[2U * gp];
        c1 = qs[(2U * gp) + 1U];
      } else {
        c0 = *reinterpret_cast<const uint4*>(
            bp + (kChunk * (kQsChunk + (2U * gp))));
        c1 = *reinterpret_cast<const uint4*>(
            bp + (kChunk * (kQsChunk + 1U + (2U * gp))));
      }
      const std::uint32_t gq[8] = {c0.x, c0.y, c0.z, c0.w,
                                   c1.x, c1.y, c1.z, c1.w};
      std::uint32_t lo[8];
      std::uint32_t hi[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8U; ++i) {
        lo[i] = gq[i] & 0x0F0F0F0FU;
        hi[i] = (gq[i] >> 4U) & 0x0F0F0F0FU;
        if constexpr (Q5) {
          lo[i] |= ((qhw[i] >> (2U * gp)) & 0x01010101U) << 4U;
          hi[i] |= ((qhw[i] >> ((2U * gp) + 1U)) & 0x01010101U) << 4U;
        }
      }
      const int sc_lo = sc[2U * gp];
      const int sc_hi = sc[(2U * gp) + 1U];
      // Tokens in quads: a quad = 4 tokens x 64 bytes (both 32-weight
      // sub-blocks of the group) = 4 s_load_b512 = 64 SGPRs, the most that
      // fits beside the rest.
#pragma unroll
      for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
        constexpr std::size_t kQuad = 4U;
        uint4 a[kQuad][4];
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4* const p = reinterpret_cast<const uint4*>(
              act_block + ((t0 + j) * k) + (gp * 64U));
#pragma unroll
          for (std::uint32_t c = 0; c < 4U; ++c)
            a[j][c] = p[c];
        }
        // Keep the next quad's 64 SGPRs of activations out of this one.
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4 alo = a[j][0];
          const uint4 ami = a[j][1];
          const uint4 blo = a[j][2];
          const uint4 bmi = a[j][3];
          int dot_lo = 0;
          int dot_hi = 0;
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[0], true, alo.x, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[1], true, alo.y, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[2], true, alo.z, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[3], true, alo.w, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[4], true, ami.x, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[5], true, ami.y, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[6], true, ami.z, dot_lo, false);
          dot_lo =
              __builtin_amdgcn_sudot4(true, lo[7], true, ami.w, dot_lo, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[0], true, blo.x, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[1], true, blo.y, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[2], true, blo.z, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[3], true, blo.w, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[4], true, bmi.x, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[5], true, bmi.y, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[6], true, bmi.z, dot_hi, false);
          dot_hi =
              __builtin_amdgcn_sudot4(true, hi[7], true, bmi.w, dot_hi, false);
          isum[t0 + j] +=
              __mul24(sc_lo, dot_lo) + __mul24(sc_hi, dot_hi);
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
    // Per 256-weight block: the min term from the activation sums, then one
    // float fold per (row, token).
    if constexpr (PackedSums) {
      const std::int32_t* const rec =
          xsum + (static_cast<std::size_t>(ab) * 5U * Batch);
      std::int32_t pk[5U * Batch];
#pragma unroll
      for (std::uint32_t i = 0; i < 5U * Batch; ++i)
        pk[i] = rec[i];
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t) {
        int imin = 0;
#pragma unroll
        for (std::uint32_t q = 0; q < 4U; ++q) {
          const std::int32_t v = pk[(4U * t) + q];
          imin += __mul24(mn[2U * q], static_cast<int>(static_cast<short>(v)));
          imin += __mul24(mn[(2U * q) + 1U], v >> 16);
        }
        const float xd = __int_as_float(pk[(4U * Batch) + t]);
        sums[t] += xd * ((d * static_cast<float>(isum[t])) -
                         (dmin * static_cast<float>(imin)));
      }
    } else {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t) {
      const float2* const base = xds + ((t * num_sub) + (b * 16U));
      const uint4* const pairs = reinterpret_cast<const uint4*>(base);
      int imin = 0;
#pragma unroll
      for (std::uint32_t j = 0; j < 8U; ++j) {
        const uint4 pr = pairs[j];
        const int a_sum = static_cast<int>(pr.y) + static_cast<int>(pr.w);
        imin += __mul24(mn[j], a_sum);
      }
      const float xd = base[0].x;
      sums[t] += xd * ((d * static_cast<float>(isum[t])) -
                       (dmin * static_cast<float>(imin)));
    }
    }
    if constexpr (kTrace) {
      __builtin_amdgcn_sched_barrier(0);
      ts[1U + (it & 3U)] = wall_clock64();  // all lanes, one address: no branch
      ++it;
    }
  }

  if constexpr (KPhases > 1U) {
    // Fixed-order cross-wave add: phase 0 + 1 + ... + KPhases-1.
    __syncthreads();  // every wave is done reading its weights
    float* const partial = lds;
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
  if constexpr (kTrace) {
    if (lane == 0U) ts[5] = wall_clock64();
  }
  bool store_y = true;
  if constexpr (SplitK)
    store_y = sn_q4k_k4_detail::SnQ4KSplitKCombine<Batch>(
        sums, &sk_flag, scratch, counter, blockIdx.x, m, row0, split, splits,
        phase == 0U);
  if (store_y && phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
  if constexpr (kTrace) {
    __builtin_amdgcn_sched_barrier(0);
    if (lane == 0U) {
      ts[6] = wall_clock64();
      ts[7] = __smid();
    }
  }
}

// SnQ4KPackSumsKernel over k/256 blocks x Batch tokens; xsum needs
// k/256 * (Q6 ? 9 : 5) * Batch int32.
template<std::size_t Batch, bool Q6 = false>
void LaunchSnQ4KPackSums(const float2* xds, std::int32_t* xsum, std::uint32_t k,
                         hipStream_t stream) {
  const std::uint32_t n = (k / 256U) * static_cast<std::uint32_t>(Batch);
  SnQ4KPackSumsKernel<Batch, Q6>
      <<<(n + 255U) / 256U, 256, 0, stream>>>(xds, xsum, k / 256U);
}

// Quantize the activations to q8_K (one int8 scale per 256), then run the k4
// kernel. k must be a multiple of 256 and `w` 16-byte aligned. quantize=false
// skips the quantize launch (GUFO_DFLASH_PREQUANT=1). For SplitK, `scratch`
// needs splits*Batch*m floats and `counter` one uint32 per row group of
// RowWaves*32 rows (blockIdx.x is the ticket), zeroed; `splits` must divide
// k/256.
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool SplitK = false, bool Interleaved = false, bool Q5 = false,
         bool Hoist = false, bool PackedSums = false, int Probe = 0,
         bool Dynamic = false, bool Q6 = false, bool IQ4 = false>
void LaunchSnQ4KK4(const void* w, const float* x, float* y, std::int8_t* xq,
                   float2* xds, std::uint32_t m, std::uint32_t k,
                   hipStream_t stream, std::uint32_t splits = 1U,
                   float* scratch = nullptr, std::uint32_t* counter = nullptr,
                   bool quantize = true, std::int32_t* xsum = nullptr) {
  static_assert(Batch >= 1 && Batch <= 8);
  if (PackedSums && xsum == nullptr)
    throw std::invalid_argument("PackedSums needs a packed-sum buffer");
  if (quantize) {
    const std::uint32_t subs = static_cast<std::uint32_t>(Batch) * (k / 16U);
    QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
        x, xq, xds, subs);
    if constexpr (PackedSums)
      LaunchSnQ4KPackSums<Batch, Q6>(xds, xsum, k, stream);
  }
  if (Interleaved && m % 32U != 0U)
    throw std::invalid_argument("K-quant interleave needs m % 32 == 0");
  constexpr std::uint32_t kRowsPerBlock = RowWaves * 32U;
  SnQ4KK4Kernel<Batch, RowWaves, KPhases, SplitK, Interleaved, Q5, Hoist,
                PackedSums, Probe, Dynamic, Q6, IQ4>
      <<<dim3((m + kRowsPerBlock - 1U) / kRowsPerBlock, SplitK ? splits : 1U),
         RowWaves * KPhases * 32U, 0, stream>>>(
          static_cast<const std::uint8_t*>(w), xq, xds, y, m, k, scratch,
          counter, xsum);
}

// Legal split count for the k4 grid: the forced S when it is legal (divides
// the K blocks), else 1. Mirrors SnbPickSplitK's legality test.
inline std::uint32_t SnQ4KPickSplit(std::uint32_t blocks,
                                    std::uint32_t forced) noexcept {
  if (forced <= 1U)
    return 1U;
  return blocks % forced == 0U ? forced : 1U;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP) && !defined(GUFO_SN_Q4K_K4_HOST_ONLY)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SN_Q4K_K4_HPP_

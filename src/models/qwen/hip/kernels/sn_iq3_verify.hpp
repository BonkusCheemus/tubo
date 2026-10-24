#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_VERIFY_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3_VERIFY_HPP_

// Selected opus_b8_k4ps (S, S=1) and snx_k4ns_pq (XXS,
// in-body/nonpersistent). Imported from the bench kernels
// unchanged. No bench arm selector or alternate geometry is exported.
// S half views advance by complete 8-row records. XXS keeps the physical
// pair plane sizes and adds logical row_begin independently of output stride.
#include <algorithm>

#include "src/models/qwen/hip/dev_sync.hpp"
#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3xxs.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"

namespace gufo::hip {
namespace snb_detail {
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

}  // namespace snb_detail

template<std::size_t Batch>
__device__ __forceinline__ bool SnbSplitKCombine(
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
    if (threadIdx.x == 0U)
      counter[ticket] = 0U;
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
    for (std::size_t t = 0; t < Batch; ++t)
      lds[slot(phase, t)] = sums[t];
  }
  __syncthreads();
  if (phase == 0U) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
      for (std::uint32_t p = 1U; p < KPhases; ++p)
        sums[t] += lds[slot(p, t)];
  }
  __syncthreads();
  bool store_y = true;
  if (splits > 1U)
    store_y = SnbSplitKCombine<Batch>(
        sums, reinterpret_cast<std::uint32_t*>(&lds[kPartial]), scratch,
        counter, group, m, row0, split, splits, phase == 0U);
  if (store_y && phase == 0U && (!GuardRows || row0 < m)) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
#pragma unroll
  for (std::size_t t = 0; t < Batch; ++t)
    sums[t] = 0.0f;
}

// s_setprio around a wave's next-block load issue (b8prio): Prio 1 -> 3, 2
// -> 2. Memory clobber keeps the loads inside the pair; the priority only
// decides who the instruction arbiter picks among the waves of a SIMD.
#define GUFO_SETPRIO_ON(P)                      \
  do {                                          \
    if constexpr ((P) == 1U)                    \
      asm volatile("s_setprio 3" ::: "memory"); \
    else if constexpr ((P) == 2U)               \
      asm volatile("s_setprio 2" ::: "memory"); \
  } while (0)
#define GUFO_SETPRIO_OFF(P)                     \
  do {                                          \
    if constexpr ((P) != 0U)                    \
      asm volatile("s_setprio 0" ::: "memory"); \
  } while (0)

template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool Probe = false, bool Pad8 = false, bool Slope = false,
         bool Clock = false, bool ActI = false, std::uint32_t Idx = 0,
         bool Quad3 = false, bool ActFix = false, bool NoDot = false,
         bool Early = false, std::uint32_t Prio = 0, bool PrioW = false,
         bool SplitK = false, bool Persist = false, bool Balanced = false>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(16))) void SnIq3SVerifyKernel(
        const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k, std::uint64_t* __restrict__ clk, float* scratch,
        std::uint32_t* counter, std::uint32_t ps_splits) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(!Balanced || (Persist && SplitK && Pad8 && RowWaves == 7U &&
                              KPhases == 4U && (Batch == 7 || Batch == 8)));
  static_assert(!ActFix && !NoDot && !Probe && Idx == 0U,
                "serving IQ3 forbids wrong-output probes and alternate tables");
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
    snb_detail::SnbStageCodebook<true>(codebook, threadIdx.x, kThreads);
    __syncthreads();
  }

  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3Elements);
  // SplitK (b8sk): blockIdx.y = split, each split owns `span` contiguous K
  // blocks [b_begin, b_end); the launcher guarantees gridDim.y divides blocks.
  // Without SplitK these fold to 0 / blocks and the code is k4's.
  // Persist (b8ps): gridDim.x resident WGs, WG w always does split w % S of row
  // groups g0 + j * gstep (g0 = w / S, gstep = gridDim.x / S; S divides
  // gridDim.x).
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
    snb_detail::SnbStageCodebook<true>(codebook, threadIdx.x, kThreads);
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
        // b8prio: raise this wave's arbiter priority only across the load
        // group.
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
        xq +
        (static_cast<std::size_t>(ab) * kSnIq3Elements * (ActI ? Batch : 1U));
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

#undef GUFO_SETPRIO_ON
#undef GUFO_SETPRIO_OFF

namespace sn {
constexpr std::size_t kSnCodebookEntries = 256 * 16;
__device__ __forceinline__ void SnIq3xxsK4Tables(std::uint32_t* codebook,
                                                 std::uint32_t* spread,
                                                 std::uint32_t tid,
                                                 std::uint32_t threads) {
  for (std::uint32_t i = tid; i < kSnCodebookEntries; i += threads)
    codebook[i] = NegateSelectedBytes(kDeviceIq3xxsGrid[i & 255U], i >> 8U);
  if (tid < 128U) {
    const std::uint32_t k = kDeviceKsignsIq2xs[tid];
    spread[tid] = (k & 0x0FU) | ((k >> 4U) << 8U);
  }
}

// One lane's half-super-block (128 weights, 8 sub-blocks) for Rows rows. `qs`,

enum class WeightPace { kPrefetch, kInBody, kQsOnly };
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases>
__device__ __forceinline__ void SnxPersistEpilogue(
    float (&sums)[Batch], float* lds, float* y, std::uint32_t m,
    std::uint32_t row0, std::uint32_t row_wave, std::uint32_t lane,
    std::uint32_t phase) {
  const auto slot = [&](std::uint32_t p, std::size_t t) {
    return (((((p - 1U) * RowWaves) + row_wave) * Batch + t) * 32U) + lane;
  };
  if (phase > 0U) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      lds[slot(phase, t)] = sums[t];
  }
  __syncthreads();
  if (phase == 0U) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
#pragma unroll
      for (std::uint32_t p = 1U; p < KPhases; ++p)
        sums[t] += lds[slot(p, t)];
  }
  __syncthreads();
  if (phase == 0U && row0 < m) {
#pragma unroll
    for (std::size_t t = 0; t < Batch; ++t)
      y[(t * m) + row0] = sums[t];
  }
#pragma unroll
  for (std::size_t t = 0; t < Batch; ++t)
    sums[t] = 0.0F;
}

// K4 arm for IQ3_XXS (bench geometry "snx_k4"). Same structure as the
//
// Persist (default off, so the plain arms' ISA is untouched) ports the IQ3_S
// b8ps schedule: gridDim.x resident workgroups, workgroup w taking the row
// groups g0 = w, w + gridDim.x, ... (S = 1: no split-K, no scratch, no atomic
// ticket), one ITEM = RowWaves*32 output rows, and the codebook staged ONCE for
// the whole launch (after the first block's loads are issued, so its DRAM round
// trip overlaps the staging), so the pipeline never drains between row groups.
// With kPrefetch / kQsOnly the next-block prefetch runs across the item
// boundary as well; kInBody has no prefetch to keep running and just carries
// the item loop, which is the pacing the persistent arms are built for. One
// item is exactly the row group one block of the plain arm does, and the blocks
// of an item are visited in the same order, so the phase reduce is the same
// ((p0+p1)+p2)+p3 and the per-row accumulation order is unchanged: the
// persistent arm is bit-identical to the plain one, item for item. The phase
// partials need their OWN LDS array (ps_buf), since the codebook stays live
// across items, hence SnxPersistEpilogue above.
template<std::size_t Batch, std::uint32_t RowWaves, std::uint32_t KPhases,
         bool Prio = false, WeightPace Pace = WeightPace::kPrefetch,
         bool Persist = false, bool Balanced = false>
__launch_bounds__(RowWaves * KPhases * 32U) __global__
    __attribute__((amdgpu_waves_per_eu(16))) void SnIq3XxsVerifyKernel(
        const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
        const float2* __restrict__ xds, float* __restrict__ y, std::uint32_t m,
        std::uint32_t k, std::uint32_t weight_rows, std::uint32_t row_begin) {
  static_assert(Batch >= 1 && Batch <= 8);
  static_assert(!Balanced ||
                (Persist && Pace == WeightPace::kInBody && RowWaves == 7U &&
                 KPhases == 4U && (Batch == 7 || Batch == 8)));
  static_assert(!Balanced || 34816U % kSnIq3xxsRowInterleave == 0U);
  static_assert(KPhases >= 1U && RowWaves >= 1U);
  // Review: kQsOnly issues its own load groups and never
  // takes the s_setprio bracket, so Prio would be silently ignored.
  static_assert(!(Prio && Pace == WeightPace::kQsOnly),
                "Prio is not implemented for WeightPace::kQsOnly");
  constexpr std::uint32_t kThreads = RowWaves * KPhases * 32U;
  static_assert(kThreads <= 1024U);
  // Phase partials for the LDS reduction must fit in the reused codebook (the
  // plain arm's tail aliases it; Persist uses ps_buf instead).
  static_assert(Persist ||
                (KPhases - 1U) * RowWaves * Batch * 32U <= kSnCodebookEntries);
  // Persist: the partials get an array of their own (the codebook stays live
  // for the whole launch), so the LDS budget is the sum of both. gfx1151 gives
  // 64 KiB per workgroup, the same figure the b8ps schedule (40 WGs x 1024
  // threads, 16 waves/SIMD, 2 WGs per WGP) relies on.
  static_assert(!Persist || (((KPhases - 1U) * RowWaves * Batch * 32U * 4U) +
                                 (kSnCodebookEntries * 4U) + (128U * 4U) <=
                             64U * 1024U),
                "SN IQ3_XXS persistent arm exceeds the 64 KiB LDS budget");
  __shared__ std::uint32_t codebook[kSnCodebookEntries];
  __shared__ std::uint32_t sign_spread[128];
  // Persist stages the tables once, after the first block's loads are issued.
  if constexpr (!Persist) {
    SnIq3xxsK4Tables(codebook, sign_spread, threadIdx.x, kThreads);
    __syncthreads();
  }

  const std::uint32_t blocks = k / static_cast<std::uint32_t>(kSnIq3xxsBlockK);
  const std::uint32_t lane = threadIdx.x & 31U;
  // readfirstlane: the compiler must see the wave index as uniform, or the
  // activation address (which depends on the phase) stops being an s_load.
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t row_wave = wave % RowWaves;
  const std::uint32_t phase = wave / RowWaves;
  // Persist: workgroup g0 takes the row groups g0 + j*gridDim.x (S = 1), so
  // row0 below is this wave's row of the FIRST item and item j adds j*ps_row.
  const std::uint32_t g0 = blockIdx.x;
  const std::uint32_t row0 = (((g0 * RowWaves) + row_wave) * 32U) + lane;
  const std::uint32_t row = row_begin + (row0 < m ? row0 : m - 1U);

  const std::size_t nb = blocks;
  const std::uint8_t* const qs_plane = static_cast<const std::uint8_t*>(w);
  const std::uint8_t* const aux_plane =
      qs_plane +
      (static_cast<std::size_t>(weight_rows) * nb * kSnIq3xxsQsBytes);
  const std::uint8_t* const d_plane =
      aux_plane +
      (static_cast<std::size_t>(weight_rows) * nb * kSnIq3xxsAuxBytes);
  const std::size_t num_full = weight_rows / kSnIq3xxsRowInterleave;
  const std::size_t rem = weight_rows % kSnIq3xxsRowInterleave;
  const std::size_t group = row / kSnIq3xxsRowInterleave;
  const std::size_t lr = row % kSnIq3xxsRowInterleave;
  const bool in_full_group = row < num_full * kSnIq3xxsRowInterleave;
  const std::size_t g = in_full_group ? kSnIq3xxsRowInterleave : rem;
  const std::size_t base_rows = in_full_group
                                    ? group * kSnIq3xxsRowInterleave
                                    : num_full * kSnIq3xxsRowInterleave;
  const std::uint8_t* qs = qs_plane + (base_rows * nb * kSnIq3xxsQsBytes) +
                           (lr * kSnIq3xxsQsBytes) +
                           (phase * g * kSnIq3xxsQsBytes);
  const std::uint8_t* aux = aux_plane + (base_rows * nb * kSnIq3xxsAuxBytes) +
                            (lr * kSnIq3xxsAuxBytes) +
                            (phase * g * kSnIq3xxsAuxBytes);
  const std::uint8_t* dp =
      d_plane + (base_rows * nb * 2) + (lr * 2) + (phase * g * 2);
  const std::size_t qs_step = KPhases * g * kSnIq3xxsQsBytes;
  const std::size_t aux_step = KPhases * g * kSnIq3xxsAuxBytes;
  const std::size_t d_step = KPhases * g * 2;
  const std::uint32_t num_sub = k / 16U;

  struct Weights {
    uint4 q[4];  // qs plane: 64 bytes, 16 grid words
    uint4 a[2];  // aux plane: 32 bytes, 8 aux words
    std::uint16_t d;
  };
  const auto load = [&]() {
    Weights v;
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      v.q[c] = *reinterpret_cast<const uint4*>(qs + (c * 16U));
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      v.a[c] = *reinterpret_cast<const uint4*>(aux + (c * 16U));
    v.d = *reinterpret_cast<const std::uint16_t*>(dp);
    return v;
  };
  // The same load group with the wave's priority raised around it (Prio only).
  const auto load_prio = [&]() {
    if constexpr (Prio) {
      asm volatile("s_setprio 3" ::: "memory");
      const Weights v = load();
      asm volatile("s_setprio 0" ::: "memory");
      return v;
    } else {
      return load();
    }
  };

  // Persist state: this workgroup's items (ps_items, all its row groups, g0,
  // g0+gstep, ...), this wave's K blocks per item (ps_nb, the same for every
  // wave because the launcher requires blocks % KPhases == 0), the row stride
  // between items (ps_row) and the byte jump from an item's last block to the
  // next item's first block in each plane. Item j's first block sits ps_row
  // rows (hence ps_row * nb records) on, minus what this item's blocks already
  // advanced the pointer by.
  [[maybe_unused]] std::uint32_t ps_items = 0U;
  [[maybe_unused]] std::uint32_t ps_n = 0U;  // blocks done in this item
  [[maybe_unused]] std::uint32_t ps_j = 0U;  // items done
  [[maybe_unused]] std::uint32_t ps_row = 0U;
  [[maybe_unused]] std::size_t ps_qs_jump = 0U;
  [[maybe_unused]] std::size_t ps_aux_jump = 0U;
  [[maybe_unused]] std::size_t ps_d_jump = 0U;
  if constexpr (Persist) {
    const std::uint32_t ps_nb = blocks / KPhases;
    const std::uint32_t groups_total =
        Balanced ? (m + RowWaves * 32U - 1U) / (RowWaves * 32U)
                 : m / (RowWaves * 32U);
    const std::uint32_t ps_gstep = gridDim.x;
    ps_row = ps_gstep * RowWaves * 32U;
    ps_items = ((groups_total - g0 + ps_gstep - 1U) / ps_gstep) * ps_nb;
    ps_qs_jump = (static_cast<std::size_t>(ps_row) * nb * kSnIq3xxsQsBytes) -
                 (static_cast<std::size_t>(ps_nb - 1U) * qs_step);
    ps_aux_jump = (static_cast<std::size_t>(ps_row) * nb * kSnIq3xxsAuxBytes) -
                  (static_cast<std::size_t>(ps_nb - 1U) * aux_step);
    ps_d_jump = (static_cast<std::size_t>(ps_row) * nb * 2U) -
                (static_cast<std::size_t>(ps_nb - 1U) * d_step);
  }
  float sums[Batch] = {};
  Weights next{};
  // kQsOnly: prefetch the 64-byte qs plane a full body ahead (it is the bulk
  // of the bytes and its first use is furthest down the body) and load the
  // 32-byte aux plane plus d in the body that consumes them.
  uint4 nq[4];
  const auto load_q = [&]() {
#pragma unroll
    for (std::uint32_t c = 0; c < 4U; ++c)
      nq[c] = *reinterpret_cast<const uint4*>(qs + (c * 16U));
  };
  const auto load_ad = [&]() {
    Weights v;
#pragma unroll
    for (std::uint32_t c = 0; c < 2U; ++c)
      v.a[c] = *reinterpret_cast<const uint4*>(aux + (c * 16U));
    v.d = *reinterpret_cast<const std::uint16_t*>(dp);
    return v;
  };
  if constexpr (Pace == WeightPace::kQsOnly) {
    if (phase < blocks)
      load_q();
  } else if constexpr (Pace == WeightPace::kPrefetch) {
    if (phase < blocks)
      next = load();
  }
  // Persist: the item's phase partials live in their own LDS array (the
  // codebook is staged once and stays live), and the tables are staged here,
  // after the first block's loads above are issued.
  [[maybe_unused]] float* ps_lds = nullptr;
  if constexpr (Persist) {
    __shared__ float ps_buf[((KPhases - 1U) * RowWaves * Batch * 32U) + 1U];
    ps_lds = ps_buf;
    SnIq3xxsK4Tables(codebook, sign_spread, threadIdx.x, kThreads);
    __syncthreads();
  }
  for (std::uint32_t b = phase; Persist ? ps_n < ps_items : b < blocks;
       b += Persist ? 0U : KPhases) {
    // One more block of THIS item, or the next item's first block (Persist, the
    // prefetch runs across the item boundary), or nothing at all (the very last
    // block of the launch). At an item's end the pointers jump to the next
    // item's first block instead of one K block on, which the following body
    // consumes as `cur`; every plane jumps by the same rule.
    const bool more = b + KPhases < blocks;
    const bool has_next = more || (Persist && ps_n + 1U < ps_items);
    const std::size_t qs_adv = (Persist && !more) ? ps_qs_jump : qs_step;
    const std::size_t aux_adv = (Persist && !more) ? ps_aux_jump : aux_step;
    const std::size_t d_adv = (Persist && !more) ? ps_d_jump : d_step;
    Weights cur{};
    if constexpr (Pace == WeightPace::kQsOnly) {
#pragma unroll
      for (std::uint32_t c = 0; c < 4U; ++c)
        cur.q[c] = nq[c];
      // aux + d of THIS block, before the pointers advance.
      const Weights ad = load_ad();
      qs += qs_adv;
      aux += aux_adv;
      dp += d_adv;
      if (has_next)
        load_q();
#pragma unroll
      for (std::uint32_t c = 0; c < 2U; ++c)
        cur.a[c] = ad.a[c];
      cur.d = ad.d;
    } else if constexpr (Pace == WeightPace::kInBody) {
      cur = load_prio();
      if constexpr (Balanced) {
        if (has_next) {
          if (more) {
            qs += qs_step;
            aux += aux_step;
            dp += d_step;
          } else {
            const std::uint32_t next_row = row0 + ((ps_j + 1U) * ps_row);
            const std::size_t safe_row =
                row_begin + (next_row < m ? next_row : m - 32U + lane);
            const std::size_t next_lr = safe_row % kSnIq3xxsRowInterleave;
            const std::size_t next_base = safe_row - next_lr;
            // Rebuild physical plane addresses, including the pair view's
            // row_begin. Padded waves repeat valid rows and never store.
            qs = qs_plane + (next_base * nb * kSnIq3xxsQsBytes) +
                 (next_lr * kSnIq3xxsQsBytes) + (phase * g * kSnIq3xxsQsBytes);
            aux = aux_plane + (next_base * nb * kSnIq3xxsAuxBytes) +
                  (next_lr * kSnIq3xxsAuxBytes) +
                  (phase * g * kSnIq3xxsAuxBytes);
            dp = d_plane + (next_base * nb * 2U) + (next_lr * 2U) +
                 (phase * g * 2U);
          }
        }
      } else {
        qs += qs_adv;
        aux += aux_adv;
        dp += d_adv;
      }
    } else {
      cur = next;
      qs += qs_adv;
      aux += aux_adv;
      dp += d_adv;
      if (has_next)
        next = load_prio();
    }
    const float d = __half2float(__ushort_as_half(cur.d)) * 0.25F;
    const std::uint32_t qs_words[16] = {
        cur.q[0].x, cur.q[0].y, cur.q[0].z, cur.q[0].w, cur.q[1].x, cur.q[1].y,
        cur.q[1].z, cur.q[1].w, cur.q[2].x, cur.q[2].y, cur.q[2].z, cur.q[2].w,
        cur.q[3].x, cur.q[3].y, cur.q[3].z, cur.q[3].w};
    const std::uint32_t aux_words[8] = {cur.a[0].x, cur.a[0].y, cur.a[0].z,
                                        cur.a[0].w, cur.a[1].x, cur.a[1].y,
                                        cur.a[1].z, cur.a[1].w};
    const std::int8_t* const act_block =
        xq + (static_cast<std::size_t>(b) * kSnIq3xxsBlockK);
    int isum[Batch] = {};
#pragma unroll
    for (std::uint32_t quad = 0; quad < 4U; ++quad) {
      // Quad = 64 weights = sub-blocks 4*quad .. 4*quad+3 = aux words 2*quad,
      // 2*quad+1; the 64 activation bytes per token are one s_load_b512.
      std::uint32_t word[4][4];
      int scale_int[2];
#pragma unroll
      for (std::uint32_t h = 0; h < 2U; ++h) {
        const std::uint32_t a = aux_words[(2U * quad) + h];
        // l = 2p + g: sub-block 2h+p takes the two 7-bit sign indices at
        // bit 14p and 14p+7, each covering grid words 2p and 2p+1.
        const std::uint32_t ja[2] = {(a >> 0U) & 127U, (a >> 14U) & 127U};
        const std::uint32_t jb[2] = {(a >> 7U) & 127U, (a >> 21U) & 127U};
        scale_int[h] =
            1 + (2 * static_cast<int>(static_cast<std::uint32_t>(a) >> 28U));
#pragma unroll
        for (std::uint32_t p = 0; p < 2U; ++p) {
          const std::uint32_t ta = sign_spread[ja[p]];
          const std::uint32_t tb = sign_spread[jb[p]];
          const std::uint32_t qw = qs_words[(4U * quad) + (2U * h) + p];
          std::uint32_t* const wd = word[(2U * h) + p];
          wd[0] = codebook[__builtin_amdgcn_perm(ta, qw, 0x07060400U)];
          wd[1] = codebook[__builtin_amdgcn_perm(ta, qw, 0x07060501U)];
          wd[2] = codebook[__builtin_amdgcn_perm(tb, qw, 0x07060402U)];
          wd[3] = codebook[__builtin_amdgcn_perm(tb, qw, 0x07060503U)];
        }
      }
#pragma unroll
      for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
        constexpr std::size_t kQuad = 4U;
        uint4 av[kQuad][4];
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
          const uint4* const p = reinterpret_cast<const uint4*>(
              act_block + ((t0 + j) * k) + (quad * 64U));
#pragma unroll
          for (std::uint32_t c = 0; c < 4U; ++c)
            av[j][c] = p[c];
        }
        // Keep the next quad's 64 SGPRs of activations out of this one.
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
#pragma unroll
          for (std::uint32_t h = 0; h < 2U; ++h) {
            const std::uint32_t* const w0 = word[2U * h];
            const std::uint32_t* const w1 = word[(2U * h) + 1U];
            const uint4 lo = av[j][2U * h];
            const uint4 hi = av[j][(2U * h) + 1U];
            int dot = 0;
            dot = __builtin_amdgcn_sudot4(true, w0[0], true, lo.x, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w0[1], true, lo.y, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w0[2], true, lo.z, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w0[3], true, lo.w, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w1[0], true, hi.x, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w1[1], true, hi.y, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w1[2], true, hi.z, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w1[3], true, hi.w, dot, false);
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
    if constexpr (Persist) {
      ++ps_n;
      if (more) {
        b += KPhases;
      } else {
        // Item done: phase reduce, store, sums restart at 0 for the next item.
        SnxPersistEpilogue<Batch, RowWaves, KPhases>(
            sums, ps_lds, y, m, row0 + (ps_j * ps_row), row_wave, lane, phase);
        ++ps_j;
        b = phase;
      }
    }
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
    if (phase == 0U && row0 < m) {
#pragma unroll
      for (std::size_t t = 0; t < Batch; ++t)
        y[(t * m) + row0] = sums[t];
    }
  }  // !Persist
}

}  // namespace sn

// xq/xds belong to the caller and must remain live through BOTH consumers.
// No activation quantization, allocation, split scratch, or hidden state here.
template<std::size_t Batch>
void LaunchSnIq3Prepared(const models::QwenTensorRef& weight,
                         const std::int8_t* xq, const float2* xds, float* y,
                         hipStream_t stream) {
  if (!models::IsSnIq3VerifyView(weight))
    throw std::invalid_argument("unsupported IQ3 prepared view/geometry");
  const auto m =
      static_cast<std::uint32_t>(weight.num_elements / weight.layout_columns);
  const auto k = static_cast<std::uint32_t>(weight.layout_columns);
  if (weight.layout == models::QwenWeightLayout::kIq3SRows8) {
    const auto* w = static_cast<const std::uint8_t*>(weight.data) +
                    weight.layout_row_begin * (k / 256U) * 110U;
    if constexpr (Batch == 7 || Batch == 8) {
      if (m == 34816U && k == 5120U && GufoDevIq3X()) {
        SnIq3SVerifyKernel<Batch, 7, 4, false, true, false, false, false, 0U,
                           false, false, false, false, 1U, false, true, true,
                           true><<<40, 896, 0, stream>>>(
            w, xq, xds, y, m, k, nullptr, nullptr, nullptr, 1U);
        return;
      }
    }
    // Fixed S=1: no scratch, ticket, or GUFO_PS_SPLITS override in serving.
    const auto wgs = std::min(m / 256U, 40U);
    SnIq3SVerifyKernel<Batch, 8, 4, false, true, false, false, false, 0U, false,
                       false, false, false, 1U, false, true, true>
        <<<wgs, 1024, 0, stream>>>(w, xq, xds, y, m, k, nullptr, nullptr,
                                   nullptr, 1U);
  } else {
    if constexpr (Batch == 7 || Batch == 8) {
      if (m == 34816U && k == 5120U && weight.layout_rows == m &&
          weight.layout_row_begin == 0U && GufoDevIq3X()) {
        sn::SnIq3XxsVerifyKernel<Batch, 7, 4, false, sn::WeightPace::kInBody,
                                 true, true><<<40, 896, 0, stream>>>(
            static_cast<const std::uint8_t*>(weight.data), xq, xds, y, m, k,
            static_cast<std::uint32_t>(weight.layout_rows),
            static_cast<std::uint32_t>(weight.layout_row_begin));
        return;
      }
    }
    sn::SnIq3XxsVerifyKernel<Batch, 4, 4, false, sn::WeightPace::kInBody>
        <<<(m + 127U) / 128U, 512, 0, stream>>>(
            static_cast<const std::uint8_t*>(weight.data), xq, xds, y, m, k,
            static_cast<std::uint32_t>(weight.layout_rows),
            static_cast<std::uint32_t>(weight.layout_row_begin));
  }
}

}  // namespace gufo::hip
#endif

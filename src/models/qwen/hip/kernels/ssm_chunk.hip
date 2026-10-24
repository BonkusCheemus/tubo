// gdn4, GUFO_DEV_GDN_CHUNK=1 (default off; NOT bit-exact): the prefill
// DeltaNet recurrence in chunks of 16 tokens on the f16 matrix cores, a drop-in
// for the token-serial row-split scan. Same inputs (the conv output, the kq
// scales and alpha/beta of the prep kernels, the raw alpha projection for the
// log decay), same raw output and FP32 carried state; the post-norm-gate
// epilogue after it is unchanged.
//
// Per value head h and chunk (tokens r, s = 0..15 of it; S0 the state before
// it, [value][key]; k^, q^ the normalized key and query, q^ carrying the
// 1/sqrt(128) of q_scale):
//   g_r = log alpha_r = softplus(alpha_in + dt) * a,   G_r = g_0 + ... + g_r
//   A[r][s] = beta_r (k^_r . k^_s) e^(G_r - G_s)   s < r
//   P[r][s] = (q^_r . k^_s) e^(G_r - G_s)          s <= r
//   T' = (I + A)^-1 diag(beta)
//   U  = T' (V - diag(e^G) K^ S0^T)
//   O  = diag(e^G) Q^ S0^T + P U
//   S1 = e^G_15 S0 + U^T diag(e^(G_15 - G)) K^
// That is the token recurrence S_t = alpha_t S + beta_t (v_t - alpha_t S k^_t)
// k^_t^T, o_t = S_t q^_t regrouped (Gated DeltaNet, Yang et al. 2024; the WY /
// UT form of FLA's chunk_gated_delta_rule; strix-llama #91 and Strata's
// gdn_wmma_scan_kernel run the same algebra on gfx11).
//
// One block per value head, eight waves; wave w owns values 16w .. 16w + 15,
// its slice of the state as eight 16 x 16 FP32 accumulators (rows = keys).
// Rounded to f16 as WMMA operands: k^, q^, P, T', the state, V - e^G K^ S0^T,
// U. Accumulation, the decay factors and the carried state stay FP32. The
// state's own decay is applied in FP32 before each update, and g comes from
// the raw projection, not log(alpha): near alpha = 1 the FP32 alpha keeps
// only a few bits of 1 - alpha.
//
// Per chunk: stage k^ q^ (and k^ transposed) in LDS from the prefetched FP32
// rows, three LDS-only barriers (a __syncthreads would wait for the next
// chunk's prefetch too), K K^T / Q K^T as eight partial tiles, A and P one
// element a thread, T' by back substitution in each wave (lane c owns row c,
// the A-fragment layout, so no transpose), then the five products. The
// unrolled loops load one step ahead behind a sched_barrier: left alone the
// scheduler hoists every LDS fragment load and the 64 state registers spill.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::hip {
namespace {

using ChunkV16h = _Float16 __attribute__((ext_vector_type(16)));
using ChunkV8f = float __attribute__((ext_vector_type(8)));
using ChunkV2h = _Float16 __attribute__((ext_vector_type(2)));

constexpr int kChunk = 16;               // tokens a chunk: one WMMA tile
constexpr int kDim = 128;                // key_dim == val_dim
constexpr int kRowStride = kDim + 8;     // k^ / q^ rows (halves), 16-byte rows
constexpr int kTileStride = kChunk + 8;  // k^T / P rows (halves)

__device__ __forceinline__ ChunkV8f Wmma(ChunkV16h a, ChunkV16h b, ChunkV8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// 16 contiguous halves, 16-byte aligned: lane (l & 15)'s row of an A operand
// or column of a B operand.
__device__ __forceinline__ ChunkV16h LoadFrag(const _Float16* p) {
  union {
    ChunkV16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// An accumulator tile (lane l: rows 2e + (l >> 4), column l & 15) as the B
// operand of the next product (lane l: column l & 15, rows 0..15). Each lane
// packs its own two rows of a pair, takes the other half's across
// permlanex16 and interleaves them by byte permute; the selectors depend on
// the lane's half.
__device__ __forceinline__ ChunkV16h AccToB(ChunkV8f acc, unsigned sel_even,
                                            unsigned sel_odd) {
  unsigned w[8];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const ChunkV2h pair = {static_cast<_Float16>(acc[2 * m]),
                           static_cast<_Float16>(acc[(2 * m) + 1])};
    const unsigned own = __builtin_bit_cast(unsigned, pair);
    const unsigned other = __builtin_amdgcn_permlanex16(
        own, own, 0x76543210U, 0xfedcba98U, false, false);
    w[2 * m] = __builtin_amdgcn_perm(other, own, sel_even);
    w[(2 * m) + 1] = __builtin_amdgcn_perm(other, own, sel_odd);
  }
  return __builtin_bit_cast(ChunkV16h, w);
}

// Only LDS is shared between the waves inside the chunk loop; the global
// loads in flight are the next chunk's prefetch and must stay in flight.
__device__ __forceinline__ void LdsBarrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ void SchedFence() {
  __builtin_amdgcn_sched_barrier(0);
}

// One thread's share of a chunk's k / q staging: token (tid >> 4), keys
// 8 (tid & 15) .. + 7 with that token's scales, and the gate inputs of token
// (tid & 15) (wave 0 consumes those).
struct ChunkRaw {
  float4 k[2];
  float4 q[2];
  float inv_k, q_scale, alpha_in, beta;
};

template<bool kDiverge>
__launch_bounds__(256) __global__ void DeltaNetChunkScanKernel(
    const float* __restrict__ conv_out, const float* __restrict__ kq_scales,
    const float* __restrict__ alpha_beta, const float* __restrict__ alpha_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    float* __restrict__ state, const float* __restrict__ state_in,
    float* __restrict__ raw_out, std::uint32_t layer_idx, int batch_size,
    int qkv_size, int num_key_heads, int num_heads) {
  __shared__ __align__(16) _Float16 ks_lds[2][kChunk * kRowStride];
  __shared__ __align__(16) _Float16 qs_lds[2][kChunk * kRowStride];
  __shared__ __align__(16) _Float16 kt_lds[2][kDim * kTileStride];
  __shared__ __align__(16) _Float16 p_lds[kChunk * kTileStride];
  __shared__ __align__(16) float a_lds[kChunk][kChunk];
  __shared__ float part_lds[8][kChunk][kChunk + 1];
  __shared__ float gate_lds[2][4][kChunk];  // G, e^G, e^(G_15 - G), beta

  const int h = static_cast<int>(blockIdx.x);
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 31U;
  const int wave = __builtin_amdgcn_readfirstlane(static_cast<int>(tid >> 5));
  const unsigned cl = lane & 15U;
  const unsigned rh = lane >> 4U;
  const unsigned sel_even = rh != 0U ? 0x01000504U : 0x05040100U;
  const unsigned sel_odd = rh != 0U ? 0x03020706U : 0x07060302U;
  const int kh = h % num_key_heads;
  const unsigned col = (static_cast<unsigned>(wave) * 16U) + cl;
  const std::size_t inner = static_cast<std::size_t>(num_heads) * kDim;

  const std::size_t head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_heads) + h) * kDim * kDim;
  const float* s_src = (kDiverge ? state_in : state) + head_offset;
  ChunkV8f st[8];  // tile it: rows = keys 16 it + 2e + rh, column = value col
#pragma unroll
  for (int it = 0; it < 8; ++it) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      st[it][e] = s_src[(col * kDim) + (it * 16) + (2 * e) + rh];
    }
  }

  const unsigned sr = tid >> 4U;
  const unsigned kb = (tid & 15U) * 8U;
  const unsigned q_off = (static_cast<unsigned>(kh) * kDim) + kb;
  const unsigned k_off =
      (static_cast<unsigned>(num_key_heads + kh) * kDim) + kb;
  const unsigned v_off =
      (static_cast<unsigned>((2 * num_key_heads) + h) * kDim) + col;
  const float dt = ssm_dt != nullptr ? ssm_dt[h] : 0.0F;
  const float a_val = ssm_a != nullptr ? ssm_a[h] : -0.05F;

  // A partial chunk's missing tokens load the last token's data, never
  // zeros: their beta = 0 and g = 0 (set where the gate is built) zero their
  // rows of T', so U, the state update and the stored outputs never see them,
  // and A and P only pair a token with earlier ones. A select on the loaded
  // value would also make the compiler wait for the prefetch to land there.
  // Addresses are a block-uniform chunk base plus a 32-bit lane byte offset
  // (the scalar-base load form; no 64-bit lane arithmetic). A chunk spans at
  // most 16 tokens, so every offset stays far below 2^32.
  const unsigned qkv_u = static_cast<unsigned>(qkv_size);
  const unsigned nkh_u = static_cast<unsigned>(num_key_heads);
  const unsigned nh_u = static_cast<unsigned>(num_heads);
  const auto fetch_kq = [&](int c, ChunkRaw& x) {
    const int c0 = c * kChunk;
    const unsigned last = static_cast<unsigned>(batch_size - 1 - c0);
    const unsigned ti = min(sr, last);
    const unsigned tg = min(cl, last);
    const char* tok = reinterpret_cast<const char*>(
        conv_out + (static_cast<std::size_t>(c0) * qkv_size));
    const unsigned tok_b = ti * qkv_u * 4U;
    x.k[0] = *reinterpret_cast<const float4*>(tok + (tok_b + (k_off * 4U)));
    x.k[1] =
        *reinterpret_cast<const float4*>(tok + (tok_b + (k_off * 4U) + 16U));
    x.q[0] = *reinterpret_cast<const float4*>(tok + (tok_b + (q_off * 4U)));
    x.q[1] =
        *reinterpret_cast<const float4*>(tok + (tok_b + (q_off * 4U) + 16U));
    const char* kq = reinterpret_cast<const char*>(
        kq_scales + (static_cast<std::size_t>(c0) * num_key_heads * 3));
    const unsigned kq_b = ((ti * nkh_u) + static_cast<unsigned>(kh)) * 12U;
    x.inv_k = *reinterpret_cast<const float*>(kq + kq_b);
    x.q_scale = *reinterpret_cast<const float*>(kq + (kq_b + 4U));
    const unsigned gi = (tg * nh_u) + static_cast<unsigned>(h);
    x.alpha_in =
        alpha_buf != nullptr
            ? *reinterpret_cast<const float*>(
                  reinterpret_cast<const char*>(
                      alpha_buf + (static_cast<std::size_t>(c0) * num_heads)) +
                  (gi * 4U))
            : 0.0F;
    x.beta = *reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(
            alpha_beta + (static_cast<std::size_t>(c0) * num_heads * 2)) +
        (gi * 8U) + 4U);
  };
  // v of this lane's accumulator elements: tokens 2e + rh, value col.
  const auto fetch_v = [&](int c, float (&v)[8]) {
    const int c0 = c * kChunk;
    const unsigned last = static_cast<unsigned>(batch_size - 1 - c0);
    const char* tok = reinterpret_cast<const char*>(
        conv_out + (static_cast<std::size_t>(c0) * qkv_size));
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned tv = min((2U * e) + rh, last);
      v[e] =
          *reinterpret_cast<const float*>(tok + (((tv * qkv_u) + v_off) * 4U));
    }
  };

  const int nch = (batch_size + kChunk - 1) / kChunk;
  ChunkRaw raw;
  float vbuf[8];
  if (nch > 0) {
    fetch_kq(0, raw);
    fetch_v(0, vbuf);
  }
  for (int c = 0; c < nch; ++c) {
    const int b = c & 1;
    const int cn = c + 1 < nch ? c + 1 : nch - 1;
    _Float16* const ks = ks_lds[b];
    _Float16* const qs = qs_lds[b];
    _Float16* const kt = kt_lds[b];
    float (&gate)[4][kChunk] = gate_lds[b];

    // ---- stage chunk c from the registers fetched during the last one
    {
      union {
        _Float16 h[8];
        uint4 u;
      } kh8, qh8;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        kh8.h[(4 * j) + 0] = static_cast<_Float16>(raw.k[j].x * raw.inv_k);
        kh8.h[(4 * j) + 1] = static_cast<_Float16>(raw.k[j].y * raw.inv_k);
        kh8.h[(4 * j) + 2] = static_cast<_Float16>(raw.k[j].z * raw.inv_k);
        kh8.h[(4 * j) + 3] = static_cast<_Float16>(raw.k[j].w * raw.inv_k);
        qh8.h[(4 * j) + 0] = static_cast<_Float16>(raw.q[j].x * raw.q_scale);
        qh8.h[(4 * j) + 1] = static_cast<_Float16>(raw.q[j].y * raw.q_scale);
        qh8.h[(4 * j) + 2] = static_cast<_Float16>(raw.q[j].z * raw.q_scale);
        qh8.h[(4 * j) + 3] = static_cast<_Float16>(raw.q[j].w * raw.q_scale);
      }
      *reinterpret_cast<uint4*>(ks + (sr * kRowStride) + kb) = kh8.u;
      *reinterpret_cast<uint4*>(qs + (sr * kRowStride) + kb) = qh8.u;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        kt[((kb + i) * kTileStride) + sr] = kh8.h[i];
      }
      if (wave == 0) {
        const int t = (c * kChunk) + static_cast<int>(cl);
        const bool live = t < batch_size;
        const float biased = raw.alpha_in + dt;
        const float softplus = biased > 20.0F ? biased : log1pf(expf(biased));
        float g = live ? softplus * a_val : 0.0F;
        // Inclusive prefix sum over the 16 tokens (each half of the wave).
#pragma unroll
        for (int off = 1; off < kChunk; off <<= 1) {
          const float y = __shfl_up(g, static_cast<unsigned>(off), kChunk);
          if (cl >= static_cast<unsigned>(off))
            g += y;
        }
        const float g_last = __shfl(g, kChunk - 1, kChunk);
        if (rh == 0U) {
          gate[0][cl] = g;
          gate[1][cl] = expf(g);
          gate[2][cl] = expf(g_last - g);
          gate[3][cl] = live ? raw.beta : 0.0F;
        }
      }
    }
    fetch_kq(cn, raw);
    LdsBarrier();

    // ---- K K^T (waves 0-3) and Q K^T (waves 4-7), two 16-key steps a wave
    {
      const _Float16* arow = (wave < 4 ? ks : qs) + (cl * kRowStride);
      const _Float16* brow = ks + (cl * kRowStride);
      const int k0 = (wave & 3) * 32;
      ChunkV8f part = {};
      part = Wmma(LoadFrag(arow + k0), LoadFrag(brow + k0), part);
      part = Wmma(LoadFrag(arow + k0 + 16), LoadFrag(brow + k0 + 16), part);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        part_lds[wave][(2 * e) + rh][cl] = part[e];
      }
    }
    SchedFence();
    // ---- K^ S0^T and Q^ S0^T for this wave's 16 values
    ChunkV8f ks_acc = {};
    ChunkV8f qs_acc = {};
    {
      const _Float16* krow = ks + (cl * kRowStride);
      const _Float16* qrow = qs + (cl * kRowStride);
      ChunkV16h ka = LoadFrag(krow);
      ChunkV16h qa = LoadFrag(qrow);
#pragma unroll
      for (int it = 0; it < 8; ++it) {
        const ChunkV16h ka_cur = ka;
        const ChunkV16h qa_cur = qa;
        if (it + 1 < 8) {
          ka = LoadFrag(krow + ((it + 1) * 16));
          qa = LoadFrag(qrow + ((it + 1) * 16));
        }
        const ChunkV16h sb = AccToB(st[it], sel_even, sel_odd);
        ks_acc = Wmma(ka_cur, sb, ks_acc);
        qs_acc = Wmma(qa_cur, sb, qs_acc);
        SchedFence();
      }
    }
    // Y = V - diag(e^G) K^ S0^T, then the next chunk's v: it has the rest of
    // this chunk to land.
    ChunkV8f y;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      y[e] = fmaf(-gate[1][(2 * e) + rh], ks_acc[e], vbuf[e]);
    }
    fetch_v(cn, vbuf);
    // Finish the products here: otherwise machine sinking moves the 16 WMMAs
    // to their first use past the next barrier, and their 24 fragment
    // registers each stay live across it.
    asm volatile("" : "+v"(y), "+v"(qs_acc));
    LdsBarrier();

    // ---- A (FP32, row-major) and P (f16 rows), one element a thread
    {
      const unsigned r = tid >> 4U;
      const unsigned s = tid & 15U;
      const float kk =
          ((part_lds[0][r][s] + part_lds[1][r][s]) + part_lds[2][r][s]) +
          part_lds[3][r][s];
      const float qk =
          ((part_lds[4][r][s] + part_lds[5][r][s]) + part_lds[6][r][s]) +
          part_lds[7][r][s];
      // Computed for every element and selected (G falls, so s > r may be
      // inf; it is dropped): a branch around expf costs more than the lanes.
      const float decay_all = expf(gate[0][r] - gate[0][s]);
      const float decay = s <= r ? decay_all : 0.0F;
      a_lds[r][s] = s < r ? gate[3][r] * kk * decay : 0.0F;
      p_lds[(r * kTileStride) + s] = static_cast<_Float16>(qk * decay);
    }
    LdsBarrier();

    // ---- T' = (I + A)^-1 diag(beta): lane c computes row c of T. From
    // T (I + A) = I, T[c][s] = [s == c] - sum_{r > s} T[c][r] A[r][s]; walking
    // r down and pushing each finished T[c][r] into every s < r keeps the
    // dependent chain 15 deep. Row r of A has r live entries, read (as a
    // broadcast) two rows ahead of its use.
    ChunkV16h t_frag;
    {
      float x[kChunk];
#pragma unroll
      for (int s = 0; s < kChunk; ++s) {
        x[s] = static_cast<unsigned>(s) == cl ? 1.0F : 0.0F;
      }
      // Fixed trip counts with predicates, so every loop unrolls fully and
      // the ring and x stay in registers.
      float4 ring[3][4];
      const auto load_row = [&](int r) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (4 * j < r) {
            ring[r % 3][j] = *reinterpret_cast<const float4*>(&a_lds[r][4 * j]);
          }
        }
      };
      load_row(kChunk - 1);
      load_row(kChunk - 2);
#pragma unroll
      for (int i = 0; i < kChunk - 1; ++i) {
        const int r = kChunk - 1 - i;
        if (r > 2)
          load_row(r - 2);
#pragma unroll
        for (int s = 0; s < kChunk - 1; ++s) {
          if (s < r) {
            const float4 a4 = ring[r % 3][s >> 2];
            const float a = (s & 3) == 0   ? a4.x
                            : (s & 3) == 1 ? a4.y
                            : (s & 3) == 2 ? a4.z
                                           : a4.w;
            x[s] = fmaf(-a, x[r], x[s]);
          }
        }
        SchedFence();
      }
#pragma unroll
      for (int s = 0; s < kChunk; ++s) {
        t_frag[s] = static_cast<_Float16>(x[s] * gate[3][s]);
      }
    }

    // ---- U = T' Y
    const ChunkV8f u = Wmma(t_frag, AccToB(y, sel_even, sel_odd), ChunkV8f{});
    const ChunkV16h ub = AccToB(u, sel_even, sel_odd);

    // ---- O = diag(e^G) Q^ S0^T + P U
    ChunkV8f o;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      o[e] = gate[1][(2 * e) + rh] * qs_acc[e];
    }
    o = Wmma(LoadFrag(p_lds + (cl * kTileStride)), ub, o);
    {
      char* obase = reinterpret_cast<char*>(
          raw_out + (static_cast<std::size_t>(c) * kChunk * inner));
      const unsigned inner_u = static_cast<unsigned>(inner);
      const unsigned o_b =
          ((rh * inner_u) + (static_cast<unsigned>(h) * kDim) + col) * 4U;
      // Uniform test: only the last chunk masks its stores.
      if ((c + 1) * kChunk <= batch_size) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          *reinterpret_cast<float*>(obase + (o_b + (e * 8U * inner_u))) = o[e];
        }
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if ((c * kChunk) + (2 * e) + static_cast<int>(rh) < batch_size) {
            *reinterpret_cast<float*>(obase + (o_b + (e * 8U * inner_u))) =
                o[e];
          }
        }
      }
    }

    // ---- S1 = e^G_15 S0 + K^T diag(e^(G_15 - G)) U
    ChunkV8f u2;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      u2[e] = u[e] * gate[2][(2 * e) + rh];
    }
    const ChunkV16h ub2 = AccToB(u2, sel_even, sel_odd);
    const float decay_chunk = gate[1][kChunk - 1];
    {
      const _Float16* ktrow = kt + (cl * kTileStride);
      ChunkV16h kf = LoadFrag(ktrow);
#pragma unroll
      for (int it = 0; it < 8; ++it) {
        const ChunkV16h kf_cur = kf;
        if (it + 1 < 8) {
          kf = LoadFrag(ktrow + ((it + 1) * 16 * kTileStride));
        }
        ChunkV8f acc = st[it];
#pragma unroll
        for (int e = 0; e < 8; ++e)
          acc[e] *= decay_chunk;
        st[it] = Wmma(kf_cur, ub2, acc);
        SchedFence();
      }
    }
  }

  float* s_dst = state + head_offset;
#pragma unroll
  for (int it = 0; it < 8; ++it) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      s_dst[(col * kDim) + (it * 16) + (2 * e) + rh] = st[it][e];
    }
  }
}

}  // namespace

void LaunchDeltaNetChunkScan(const float* conv_out, const float* kq_scales,
                             const float* alpha_beta, const float* alpha_buf,
                             const float* ssm_a, const float* ssm_dt,
                             QwenRecurrentStateIo state, float* raw_out,
                             std::uint32_t layer_idx, std::size_t batch_size,
                             std::size_t qkv_size, std::uint32_t num_key_heads,
                             std::uint32_t num_heads, hipStream_t stream) {
  const auto run = [&]<bool kDiverge>() {
    DeltaNetChunkScanKernel<kDiverge><<<num_heads, 256, 0, stream>>>(
        conv_out, kq_scales, alpha_beta, alpha_buf, ssm_a, ssm_dt,
        static_cast<float*>(state.recurrent_out),
        static_cast<const float*>(state.recurrent_in), raw_out, layer_idx,
        static_cast<int>(batch_size), static_cast<int>(qkv_size),
        static_cast<int>(num_key_heads), static_cast<int>(num_heads));
  };
  if (state.Diverges())
    run.template operator()<true>();
  else
    run.template operator()<false>();
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

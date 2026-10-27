#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/kernels/k95_small.hpp"
#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3_verify.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"

namespace gufo::hip {

template<bool StoreActivationSum, typename T>
__global__ void QuantizeActivationToQ8_1Kernel(const T* __restrict__ x,
                                               void* __restrict__ y,
                                               std::size_t batch,
                                               std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float val = static_cast<float>(x[elem_idx]);
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const int8_t q = static_cast<int8_t>(roundf(val * id));

  StoreQ8ActLane<StoreActivationSum>(y, batch, num_blocks, tok, blk, lane_id, d,
                                     q);
}

// opt-c173-norm-quant: optional residual add, RMSNorm, and the tiled Q8_1
// quantization in one pass.
//
// Every projection reading the SSM pre-norm and the FFN norm is Q8_0, so the
// FP32 normed row and the BF16 staging copy the old chain produced were both
// dead: RMSNorm wrote 42 MB + 21 MB per layer at batch 2048 and the quantizer
// read the 21 MB straight back. Staging the row in LDS removes the rest of the
// traffic -- the reduction and the normalization share one global read, and
// with `residual` the preceding ResidualAdd pass disappears as well. Per norm
// at batch 2048 that is 221 MB of round trips down to 137 MB.
//
// The sum-of-squares reduction is the same loop and the same tree as
// BatchedRMSNormKernel, and folding the residual in registers gives the same
// FP32 value the separate add would have stored, so `inv_rms` and every normed
// value are bit-identical to the unfused chain. Only the quantization input
// changes, from BF16 to the FP32 value it was rounded from, which strictly
// narrows the quantization error -- the same trade `opt-c164-swiglu-quant` and
// `opt-c172-fp32-quant` made.
template<bool StoreActivationSum>
__global__ void BatchedFusedRMSNormQuantizeQ8_1Kernel(
    const float* __restrict__ x, const float* __restrict__ residual,
    const float* __restrict__ weight, float* __restrict__ sum_out,
    void* __restrict__ q8_out, std::size_t batch_size, std::size_t dim,
    float eps) {
  extern __shared__ float s_row[];
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;
  const float* in_row = x + (b * dim);
  const float* res_row = (residual != nullptr) ? residual + (b * dim) : nullptr;
  float* sum_row = (sum_out != nullptr) ? sum_out + (b * dim) : nullptr;

  // Single pass over global memory: the row is staged in LDS so neither the
  // reduction nor the normalization re-reads it.
  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += block_size) {
    float val = in_row[i];
    if (res_row != nullptr) {
      val += res_row[i];
      if (sum_row != nullptr) {
        sum_row[i] = val;
      }
    }
    s_row[i] = val;
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < dim; i += block_size) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    s_row[i] = (s_row[i] * inv_rms) * w;
  }
  __syncthreads();

  // One wave per 32-element quantization block, matching
  // QuantizeActivationToQ8_1Kernel's per-block scale and rounding exactly.
  const std::size_t num_blocks = dim / 32;
  const std::size_t lane_id = tid & 31u;
  const std::size_t wave_id = tid >> 5u;
  const std::size_t waves = block_size >> 5u;
  for (std::size_t blk = wave_id; blk < num_blocks; blk += waves) {
    const float val = s_row[(blk * 32) + lane_id];
    float max_abs = fabsf(val);
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    const float d = max_abs / 127.0F;
    const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
    const std::int8_t q = static_cast<std::int8_t>(roundf(val * id));
    StoreQ8ActLane<StoreActivationSum>(q8_out, batch_size, num_blocks, b, blk,
                                       lane_id, d, q);
  }
}

// opt-c174-ssm-epilogue-quant: the SSM per-head post-RMSNorm + SiLU gate
// writing the tiled Q8_1 activation directly.
//
// The gated output's only consumer is the Q8_0 `ssm_out` projection, so the
// FP32 row was written and read straight back by the quantizer: 214 MB of round
// trips per layer at batch 2048 against 114 MB for one pass. The head's value
// dimension is a multiple of 32, so a block that owns one (head, token) already
// owns whole quantization blocks and needs no extra communication.
//
// The RMS reduction is the same warp butterfly, the same four-way cross-warp
// sum in the same order, and the same epsilon as BatchedSSMPostNormGateKernel,
// so every gated value is bit-identical to that kernel; only the quantization
// input changes from a re-read FP32 store to the register it came from, which
// is the same value.
template<bool StoreActivationSum>
__global__ void BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel(
    const float* __restrict__ raw_out, const float* __restrict__ ssm_norm,
    const float* __restrict__ gate, void* __restrict__ q8_out,
    std::size_t batch_size, std::size_t inner_size, std::uint32_t num_heads,
    std::uint32_t val_dim) {
  const std::size_t h = blockIdx.x;
  const std::size_t t = blockIdx.y;
  if (h >= num_heads || t >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  const float* in_h = raw_out + (t * inner_size) + (h * val_dim);

  const float in_val = (tid < val_dim) ? in_h[tid] : 0.0F;
  float o_sq = in_val * in_val;
  for (int offset = 16; offset > 0; offset >>= 1) {
    o_sq += __shfl_xor(o_sq, offset);
  }

  __shared__ float s_warp_rms[4];
  if (lane_id == 0) {
    s_warp_rms[warp_id] = o_sq;
  }
  __syncthreads();

  __shared__ float s_inv_rms;
  if (tid == 0) {
    const float total_o_sq =
        s_warp_rms[0] + s_warp_rms[1] + s_warp_rms[2] + s_warp_rms[3];
    const float mean_sq = total_o_sq / static_cast<float>(val_dim);
    s_inv_rms = rsqrtf(mean_sq + 1e-6F);
  }
  __syncthreads();

  if (tid >= val_dim) {
    return;
  }
  const float normed = in_val * s_inv_rms;
  const float w = (ssm_norm != nullptr) ? ssm_norm[tid] : 1.0F;
  float val = normed * w;
  if (gate != nullptr) {
    const float g = (gate + (t * inner_size))[(h * val_dim) + tid];
    const float sig = 1.0F / (1.0F + expf(-g));
    val *= (g * sig);
  }

  // Each wave of the block owns one whole 32-element quantization block.
  const std::size_t num_blocks = inner_size / 32;
  const std::size_t blk = ((h * val_dim) + tid) / 32;
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const std::int8_t q = static_cast<std::int8_t>(roundf(val * id));
  StoreQ8ActLane<StoreActivationSum>(q8_out, batch_size, num_blocks, t, blk,
                                     lane_id, d, q);
}

/// Zeroes the tail of the last token tile so a batch that is not a multiple of
/// 16 cannot feed uninitialized scales into the GEMM epilogue.
template<bool StoreActivationSum>
__global__ void ZeroQ8ActTailKernel(void* __restrict__ y, std::size_t batch,
                                    std::size_t num_blocks) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t tt = batch / kQ8ActTileTokens;
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t lanes = kQ8ActTileTokens - tail;
  if (idx >= num_blocks * lanes) {
    return;
  }
  const std::size_t kb = idx / lanes;
  const std::size_t tl = tail + (idx % lanes);
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, kb);
  reinterpret_cast<int32x4_t*>(tile + (tl * 16))[0] = {};
  reinterpret_cast<int32x4_t*>(tile + 256 + (tl * 16))[0] = {};
  *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
      0.0F;
  if constexpr (StoreActivationSum) {
    float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
    sums[(((tt * num_blocks) + kb) * kQ8ActTileTokens) + tl] = 0.0F;
  }
}

template<bool StoreActivationSum>
__global__ void BatchedFusedSwiGLUQuantizeQ8_1Kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    void* __restrict__ y, std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float g = gate[elem_idx];
  const float u = up[elem_idx];
  const float silu_g = g / (1.0F + expf(-g));
  const float val = silu_g * u;

  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const int8_t q = static_cast<int8_t>(roundf(val * id));

  StoreQ8ActLane<StoreActivationSum>(y, batch, num_blocks, tok, blk, lane_id, d,
                                     q);
}

// opt-c192-swiglu-epilogue: SwiGLU + tiled Q8_1 quantization straight out of a
// blocked GEMM's accumulator.
//
// The FFN up projection's only consumer is SwiGLU, whose only consumer is the
// Q8_1 activation ffn_down reads. Writing the FP32 intermediate therefore costs
// 143 MB of stores per layer at batch 2048 and another 143 MB of loads in a
// separate bandwidth-bound pass, to reproduce a value the GEMM already had in a
// register.
//
// Two shape facts make the fused form possible without any cross-wave traffic:
//
//   * A wave's two row subtiles are rows [32*wave_row, 32*wave_row+32) of the
//     macro tile, which is exactly one 32-element quantization block of the
//     *next* GEMM's K dimension. So one wave owns every value a block scale
//     depends on.
//   * Staging both subtiles through the wave's LDS slot puts one token's
//     sixteen rows in sixteen consecutive lanes, so the gate load is one
//     64-byte segment, the code stores are one 16-byte segment, and the max
//     reduction is four `__shfl_xor` inside a half-wave.
//
// Bit-identical to LaunchBatchedFusedSwiGLUQuantizeQ8_1 over the same inputs:
// `acc` is the float the FP32 store would have round-tripped exactly, `gate`
// comes from the same buffer, and `fmaxf` and integer addition are
// order-independent so the changed reduction shape cannot move a result.
template<int TokTiles, bool StoreActivationSum>
__device__ __forceinline__ void BlockedSwiGluQuantEpilogue(
    const float (*acc_lo)[8], const float (*acc_hi)[8], float* scratch_lo,
    float* scratch_hi, const float* __restrict__ gate,
    void* __restrict__ q8_out, std::size_t batch, std::size_t m,
    std::size_t r_base, std::size_t t_base, int lane_id) {
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const std::size_t num_blocks = m / 32;
  const std::size_t blk = r_base / 32;
#pragma unroll
  for (int j = 0; j < TokTiles; ++j) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      scratch_lo[(sub_lane * 16) + (2 * l) + half_id] = acc_lo[j][l];
      scratch_hi[(sub_lane * 16) + (2 * l) + half_id] = acc_hi[j][l];
    }
    __builtin_amdgcn_wave_barrier();
    const std::size_t t0 = t_base + static_cast<std::size_t>(j * 16);
    std::int8_t* tile =
        Q8ActTile(q8_out, num_blocks, t0 / kQ8ActTileTokens, blk);
    // Every gate load for this tile is issued before any of the dependent math.
    // This epilogue runs after the K loop, so unlike the GEMM body there is no
    // other work left to hide memory latency behind: one load per step would
    // serialize sixteen round trips per tile, which measured as most of the
    // epilogue's cost. Sixteen floats per lane is sixteen registers, and the
    // block's whole gate panel (64 KiB) is far too large to stage instead.
    // A token past the batch contributes nothing and must not read `gate` out
    // of bounds; its tile slot is zeroed by ZeroQ8ActTailKernel, exactly as on
    // the separate-pass route.
    float g_lo[8];
    float g_hi[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int flat = (s * 32) + lane_id;
      const std::size_t tok = t0 + static_cast<std::size_t>(flat >> 4);
      const std::size_t row_base =
          (tok * m) + r_base + static_cast<std::size_t>(flat & 15);
      const bool live = tok < batch;
      g_lo[s] = live ? gate[row_base] : 0.0F;
      g_hi[s] = live ? gate[row_base + 16] : 0.0F;
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int flat = (s * 32) + lane_id;
      const int tl = flat >> 4;
      const int row = flat & 15;
      const std::size_t tok = t0 + static_cast<std::size_t>(tl);
      const bool live = tok < batch;
      float v_lo = 0.0F;
      float v_hi = 0.0F;
      if (live) {
        v_lo = (g_lo[s] / (1.0F + expf(-g_lo[s]))) * scratch_lo[flat];
        v_hi = (g_hi[s] / (1.0F + expf(-g_hi[s]))) * scratch_hi[flat];
      }
      float max_abs = fmaxf(fabsf(v_lo), fabsf(v_hi));
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off, 16));
      }
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q_lo = static_cast<std::int8_t>(roundf(v_lo * id));
      const auto q_hi = static_cast<std::int8_t>(roundf(v_hi * id));
      if (live) {
        tile[(tl * 16) + row] = q_lo;
        tile[256 + (tl * 16) + row] = q_hi;
        if (row == 0) {
          *reinterpret_cast<float*>(
              tile + kQ8ActScaleOffset +
              (static_cast<std::size_t>(tl) * sizeof(float))) = d;
        }
      }
      if constexpr (StoreActivationSum) {
        int qsum = static_cast<int>(q_lo) + static_cast<int>(q_hi);
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          qsum += __shfl_xor(qsum, off, 16);
        }
        if (live && row == 0) {
          float* sums = Q8ActSumSidecar(q8_out, batch, m);
          sums[(((t0 / kQ8ActTileTokens * num_blocks) + blk) *
                kQ8ActTileTokens) +
               static_cast<std::size_t>(tl)] = d * static_cast<float>(qsum);
        }
      }
    }
    __builtin_amdgcn_wave_barrier();
  }
}

template<bool SubOpsel>
__device__ __forceinline__ int32x8_t Wmma_16x16x16_iu8(int32x4_t a, int32x4_t b,
                                                       int32x8_t c) {
#if defined(__has_builtin) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c,
                                                    SubOpsel);
#else
  return c;
#endif
}

// Two-dimensionally blocked W8A8 WMMA GEMM (opt-c163-blocked-w8a8).
//
// Traffic per multiply-accumulate is 34/(32*BN) bytes of weight and
// 36/(32*BM) bytes of activation. The previous 16-row macro tile therefore
// re-read the whole activation panel m/16 times (1088x, ~12.8 GB for the FFN
// shape at batch 2048), which dominated everything else. Blocking both
// dimensions at 128 cuts that to m/128 and lifts the kernel from 35% to 58% of
// the measured 55.07 TOPS WMMA int8 ceiling.
//
//   block = BM rows x BN tokens, BK 32-element K blocks per LDS stage
//   waves = WM row groups x WN token groups (WM*WN == 8)
//   wave  = (BM/WM) rows x (BN/WN) tokens
//
// Weights are read from the production row-major block_q8_0 layout, so no
// second copy of the model is needed; activations arrive in the tiled layout
// the quantize kernel emits. Accumulation order over K blocks is unchanged from
// the 16-row kernel, so results are bit-identical.
template<int BM, int BN, int BK, int WM, int WN, bool FusedSwiGluQuant = false,
         bool StoreActivationSum = false>
__launch_bounds__(256) __global__ void W8A8BlockedWmmaGEMMKernel(
    const void* __restrict__ w, const void* __restrict__ x_blocks,
    float* __restrict__ y, std::size_t batch, std::size_t m, std::size_t k,
    const float* __restrict__ swiglu_gate = nullptr,
    void* __restrict__ q8_out = nullptr) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage. Fetching one stage ahead lets the
  // latency of the scattered 34-byte block_q8_0 reads overlap the WMMA work
  // instead of gating the LDS fill.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  // opt-c179-gemm-addressing. An ISA dump of this kernel put only 64 of its
  // 2752 instructions in WMMA: ~309 were exec-mask manipulation
  // (`s_and_saveexec_b32` / `s_or_b32` / `s_and_b32`) from the per-thread
  // bounds guards below, and ~265 were 64-bit address arithmetic
  // (`v_mad_u64_u32`, `v_add_co_u32`) from recomputing `(r * num_blocks) + kb`
  // per stage. Both go away here:
  //
  //   * Every 64-bit base is hoisted out of the K loop, so a stage advance is a
  //     32-bit add.
  //   * Out-of-range rows and K blocks are CLAMPED rather than branched, and
  //     their *scale* is zeroed with a multiply. The epilogue then multiplies
  //     by dw = 0, so an out-of-range row contributes exactly zero with no
  //     divergence -- bit-identical to zero-filling the operands, because
  //     0 * dx * float(c) is 0 for any finite c.
  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = reinterpret_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

    // Unroll the K blocks of a stage only if there are two. In a fully
    // unrolled BK=4 stage, the scheduler can move the LDS operand reads of all
    // blocks above the WMMAs of the first block. Clang 23 does this. Then the
    // kernel needs more than 256 VGPRs and spills.
#pragma unroll(BK <= 2 ? BK : 1)
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // The token-tile operands are read one tile at a time, inside the j loop,
      // rather than all kWaveTokTiles of them up front. Hoisting them held
      // 4 x (2 int32x4 + 1 float) = 36 extra VGPRs live across the whole WMMA
      // block, which is what pushed a BK=2 stage into 500 bytes/lane of
      // scratch.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = Wmma_16x16x16_iu8<true>(a0[i], b0, c);
          c = Wmma_16x16x16_iu8<true>(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
      // Keep the LDS operand reads of the next K block below the WMMAs of this
      // K block. This prevents the same spill in an unrolled BK=2 stage. The
      // arithmetic order does not change.
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }

  // Transpose each 16x16 result tile through LDS so the global stores cover 16
  // consecutive rows of one token instead of 16 separate cache lines.
  __syncthreads();
  if constexpr (FusedSwiGluQuant) {
    // The fused epilogue owns two 256-float slots per wave and never touches
    // `y`, so it takes the whole of both staging arrays. The conditions are
    // written against the flag because clang checks a static_assert in a
    // discarded `if constexpr` branch once the enclosing template is
    // instantiated, and the narrow tiles do not satisfy them.
    static_assert(!FusedSwiGluQuant || kWaveRowTiles == 2,
                  "the fused SwiGLU epilogue needs a wave to own one whole "
                  "32-element quantization block");
    static_assert(
        !FusedSwiGluQuant || (sizeof(s_a) >= WM * WN * 256 * sizeof(float) &&
                              sizeof(s_b) >= WM * WN * 256 * sizeof(float)),
        "the fused epilogue stages two output tiles per wave");
    BlockedSwiGluQuantEpilogue<kWaveTokTiles, StoreActivationSum>(
        acc[0], acc[1],
        reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256),
        reinterpret_cast<float*>(&s_b[0][0][0]) + (wave_id * 256), swiglu_gate,
        q8_out, batch, m, r_block + static_cast<std::size_t>(wave_row * 32),
        t_block + static_cast<std::size_t>(wave_tok * kWaveTokTiles * 16),
        lane_id);
    return;
  }
  float* tile_scratch =
      reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256);
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // One uniform branch per 16x16 tile instead of eight per-lane predicates,
      // which is where most of the kernel's exec-mask traffic came from.
      if (t0 + 16 <= batch && r0 + 16 <= m) {
        const std::size_t out_base = (t0 * m) + r0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          y[out_base + (static_cast<std::size_t>(flat >> 4) * m) +
            static_cast<std::size_t>(flat & 15)] = tile_scratch[flat];
        }
      } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          const std::size_t tok = t0 + static_cast<std::size_t>(flat >> 4);
          const std::size_t r = r0 + static_cast<std::size_t>(flat & 15);
          if (tok < batch && r < m) {
            y[(tok * m) + r] = tile_scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

template<std::uint32_t WavesPerBlock, std::uint32_t TokensPerWave>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void TiledBatchedQ8_0GEMMKernel(const void* __restrict__ w,
                                    const hip_bfloat16* __restrict__ x,
                                    float* __restrict__ y, std::size_t batch,
                                    std::size_t m, std::size_t k) {
  constexpr std::size_t kRowsPerWave = 4;
  static_assert(TokensPerWave == 2 || TokensPerWave == 4 || TokensPerWave == 8);
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  const std::size_t row_base =
      (blockIdx.x * WavesPerBlock + warp_id) * kRowsPerWave;
  const std::size_t tok_base = blockIdx.y * TokensPerWave;
  if (row_base >= m || tok_base >= batch) {
    return;
  }

  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* w_blocks = static_cast<const Q8_0Block*>(w);

  const auto* row0 = w_blocks + ((row_base + 0) * num_blocks);
  const auto* row1 =
      (row_base + 1 < m) ? (w_blocks + ((row_base + 1) * num_blocks)) : nullptr;
  const auto* row2 =
      (row_base + 2 < m) ? (w_blocks + ((row_base + 2) * num_blocks)) : nullptr;
  const auto* row3 =
      (row_base + 3 < m) ? (w_blocks + ((row_base + 3) * num_blocks)) : nullptr;

  const std::size_t group_id = lane_id >> 3u;       // 0..3
  const std::size_t lane_in_group = lane_id & 7u;   // 0..7
  const std::size_t elem_off = lane_in_group * 4u;  // 0, 4, 8..28

  float acc[4][TokensPerWave] = {};

  const std::size_t num_steps = (num_blocks + 3) / 4;
  for (std::size_t step = 0; step < num_steps; ++step) {
    const std::size_t b = (step * 4) + group_id;

    if (b < num_blocks) {
      // Load four bfloat16 activation values for each token column.
      hip_bfloat16 xv[TokensPerWave][4];
#pragma unroll
      for (std::size_t t = 0; t < TokensPerWave; ++t) {
        if (tok_base + t < batch) {
          *reinterpret_cast<uint64_t*>(&xv[t][0]) =
              *reinterpret_cast<const uint64_t*>(
                  &x[((tok_base + t) * k) + (b * kQ8_0BlockSize) + elem_off]);
        }
      }

      // Process row 0
      {
        const Q8_0Block& b0 = row0[b];
        const float d0 = __half2float(b0.d);
        const auto* qs0 = b0.qs + elem_off;
        float dot0[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d0 * static_cast<float>(qs0[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot0[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[0][t] += dot0[t];
        }
      }

      // Process row 1
      if (row1 != nullptr) {
        const Q8_0Block& b1 = row1[b];
        const float d1 = __half2float(b1.d);
        const auto* qs1 = b1.qs + elem_off;
        float dot1[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d1 * static_cast<float>(qs1[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot1[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[1][t] += dot1[t];
        }
      }

      // Process row 2
      if (row2 != nullptr) {
        const Q8_0Block& b2 = row2[b];
        const float d2 = __half2float(b2.d);
        const auto* qs2 = b2.qs + elem_off;
        float dot2[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d2 * static_cast<float>(qs2[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot2[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[2][t] += dot2[t];
        }
      }

      // Process row 3
      if (row3 != nullptr) {
        const Q8_0Block& b3 = row3[b];
        const float d3 = __half2float(b3.d);
        const auto* qs3 = b3.qs + elem_off;
        float dot3[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d3 * static_cast<float>(qs3[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot3[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[3][t] += dot3[t];
        }
      }
    }
  }

  // Reduce within group (8 lanes per block) and across groups (4 groups in
  // wave)
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (std::size_t r = 0; r < 4; ++r) {
#pragma unroll
      for (std::size_t t = 0; t < TokensPerWave; ++t) {
        acc[r][t] += __shfl_xor(acc[r][t], off);
      }
    }
  }

  if (lane_id == 0) {
#pragma unroll
    for (std::size_t r = 0; r < 4; ++r) {
      const std::size_t row = row_base + r;
      if (row < m) {
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          const std::size_t tok = tok_base + t;
          if (tok < batch) {
            y[(tok * m) + row] = acc[r][t];
          }
        }
      }
    }
  }
}

template<typename Input>
__global__ void BatchedQuantGEMVKernel(core::GgmlType type,
                                       const void* __restrict__ w,
                                       const Input* __restrict__ x,
                                       float* __restrict__ y, std::size_t m,
                                       std::size_t k) {
  constexpr std::size_t kWarp = 32;
  const std::size_t warp_id = threadIdx.x / kWarp;
  const std::size_t lane = threadIdx.x % kWarp;
  const std::size_t warps_per_block = blockDim.x / kWarp;

  const std::size_t b = blockIdx.y;
  const std::size_t row = blockIdx.x * warps_per_block + warp_id;
  if (row >= m) {
    return;
  }

  const float dot =
      QuantWarpBlockDot<Input>(type, w, row, x + (b * k), k, lane, 0, 1);
  if (lane == 0) {
    y[(b * m) + row] = dot;
  }
}

template<std::uint32_t WavesPerBlock, std::size_t TokensPerWave>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void SmallBatchQ8KFp32GEMMKernel(const void* __restrict__ w,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, std::size_t batch,
                                     std::size_t m, std::size_t k) {
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row = (blockIdx.x * WavesPerBlock) + warp_id;
  if (row >= m) {
    return;
  }

  const std::size_t num_blocks = k / kQ8KBlockSize;
  const auto* row_blocks = static_cast<const Q8KBlock*>(w) + (row * num_blocks);
  float sums[TokensPerWave] = {};

  for (std::size_t block = 0; block < num_blocks; ++block) {
    const Q8KBlock& weights = row_blocks[block];
    float weight_values[8];
#pragma unroll
    for (std::size_t element = 0; element < 8; ++element) {
      const std::size_t index = lane + (32 * element);
      weight_values[element] = static_cast<float>(weights.qs[index]);
    }

#pragma unroll
    for (std::size_t token = 0; token < TokensPerWave; ++token) {
      float dot = 0.0F;
      if (token < batch) {
        const float* input = x + (token * k) + (block * kQ8KBlockSize);
#pragma unroll
        for (std::size_t element = 0; element < 8; ++element) {
          const std::size_t index = lane + (32 * element);
          dot += weight_values[element] * input[index];
        }
      }
      sums[token] += weights.d * dot;
    }
  }

#pragma unroll
  for (std::size_t token = 0; token < TokensPerWave; ++token) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[token] += __shfl_xor(sums[token], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (std::size_t token = 0; token < TokensPerWave; ++token) {
      if (token < batch) {
        y[(token * m) + row] = sums[token];
      }
    }
  }
}

namespace {

template<bool StoreActivationSum>
void LaunchZeroQ8ActTail(void* q8_1_out, std::size_t batch, std::size_t k,
                         hipStream_t stream) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t work = num_blocks * (kQ8ActTileTokens - tail);
  ZeroQ8ActTailKernel<StoreActivationSum>
      <<<static_cast<unsigned int>((work + 255) / 256), 256, 0, stream>>>(
          q8_1_out, batch, num_blocks);
}

}  // namespace

void LaunchQuantizeActivationQ8_1(const void* bf16_x, void* q8_1_out,
                                  std::size_t batch, std::size_t k,
                                  hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  QuantizeActivationToQ8_1Kernel<true>
      <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
          reinterpret_cast<const hip_bfloat16*>(bf16_x), q8_1_out, batch, k);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
}

bool IsFusedSSMEpilogueQuantizeQ8_1Supported(std::uint32_t val_dim,
                                             std::size_t inner_size) noexcept {
  // One block owns one (head, token) and quantizes its own whole blocks, so the
  // value dimension has to be a multiple of 32 and fit the 128-thread block the
  // RMS reduction is written for.
  return val_dim != 0 && val_dim <= 128 && (val_dim % 32) == 0 &&
         (inner_size % 32) == 0;
}

void LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1(
    const float* raw_out, const float* ssm_norm, const float* gate,
    void* q8_1_out, std::size_t batch_size, std::size_t inner_size,
    std::uint32_t num_heads, std::uint32_t val_dim, hipStream_t stream) {
  if (batch_size == 0 || num_heads == 0 || val_dim == 0) {
    return;
  }
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel<true>
      <<<grid, 128, 0, stream>>>(raw_out, ssm_norm, gate, q8_1_out, batch_size,
                                 inner_size, num_heads, val_dim);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch_size, inner_size, stream);
}

bool IsFusedRMSNormQuantizeQ8_1Supported(std::size_t dim) noexcept {
  // The normed row is staged in LDS, so the row has to fit the 32 KiB a block
  // can hold alongside the reduction scratch, and the quantizer needs whole
  // 32-element blocks.
  return dim != 0 && (dim % 32) == 0 && (dim * sizeof(float)) <= 32768;
}

void LaunchBatchedFusedRMSNormQuantizeQ8_1(
    const float* x, const float* residual, const float* weight, float* sum_out,
    void* q8_1_out, std::size_t batch_size, std::size_t dim, float eps,
    hipStream_t stream) {
  if (batch_size == 0 || dim == 0) {
    return;
  }
  const unsigned int lds_bytes = static_cast<unsigned int>(dim * sizeof(float));
  BatchedFusedRMSNormQuantizeQ8_1Kernel<true>
      <<<static_cast<unsigned int>(batch_size), 256, lds_bytes, stream>>>(
          x, residual, weight, sum_out, q8_1_out, batch_size, dim, eps);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch_size, dim, stream);
}

void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  QuantizeActivationToQ8_1Kernel<true>
      <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
          fp32_x, q8_1_out, batch, k);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
}

void LaunchBatchedFusedSwiGLUQuantizeQ8_1(const float* gate, const float* up,
                                          void* q8_1_out, std::size_t batch,
                                          std::size_t k, hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  BatchedFusedSwiGLUQuantizeQ8_1Kernel<true>
      <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
          gate, up, q8_1_out, batch, k);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
}

std::size_t QuantizedActivationBytes(std::size_t batch, std::size_t k) {
  return Q8ActTiledBytes(batch, k) + Q8ActSumBytes(batch, k);
}

// opt-q4kxl: two-level dispatch for SmallBatchKQuantExactFp32GEMMKernel --
// exact batch width (a padded width would stage and multiply activations no
// output consumes) crossed with weight format.
// Sixteen warps per block, not eight: the staged activation tile is shared by
// every warp in the block, so doubling the warps halves the staging cost per
// output row. This is the same shape opt-c18x-q8-vec-smallbatch settled on for
// the Q8_0 vec kernel.
constexpr std::size_t kKQuantSmallBatchWarps = 16;
constexpr std::size_t kKQuantSmallBatchRowsPerWave = 2;

static_assert(FitsSmallBatch32BitIndices(8, 17408, 5120, 3520));
static_assert(FitsSmallBatch32BitIndices(8, 248320, 5120, 4200));
static_assert(FitsSmallBatch32BitIndices(2, 17408, 5120, 3520));
static_assert(FitsSmallBatch32BitIndices(2, 248320, 5120, 4200));
static_assert(!FitsSmallBatch32BitIndices(2, 1, 1ULL << 31, 1));
static_assert(!FitsSmallBatch32BitIndices(2, 1ULL << 31, 1, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 1, 1ULL << 29, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 1ULL << 29, 1, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 65536, 131072, 90112));
static_assert(!FitsSmallBatch32BitIndices(8, 1, 1, 0));

template<std::size_t RowsPerWave, core::GgmlType WType>
void LaunchKQuantSmallBatchTyped(const void* w, const float* fp32_x, float* y,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k, dim3 grid, dim3 block,
                                 hipStream_t stream) {
  const auto launch = [&]<std::size_t Batch>() {
    // Adjacent gate/up tensors use the same kernel with twice as many rows.
    const bool ffn_up = (m == 17408 || (Batch >= 2 && m == 34816)) && k == 5120;
    if constexpr (Batch >= 3 && (WType == core::GgmlType::kIQ4_NL ||
                                 WType == core::GgmlType::kIQ3_S ||
                                 WType == core::GgmlType::kQ6_K ||
                                 WType == core::GgmlType::kIQ4_XS)) {
      const bool ffn_down = m == 5120 && k == 17408;
      // The remaining mixed formats benefit from four-row activation reuse
      // at these shapes and widths. Keep narrower batches where reuse loses.
      const bool reuse_four_rows =
          (WType == core::GgmlType::kIQ4_NL &&
           ((m == 17408 && k == 5120) || (Batch >= 5 && ffn_down))) ||
          (WType == core::GgmlType::kIQ3_S && (Batch == 3 || Batch >= 5) &&
           ffn_down) ||
          (WType == core::GgmlType::kQ6_K &&
           ((Batch >= 6 && m == 5120 && k == 6144) ||
            (Batch == 8 && m == 10240 && k == 5120))) ||
          (WType == core::GgmlType::kIQ4_XS &&
           ((m == 5120 && k == 6144) ||
            (Batch == 8 && k == 5120 && (m == 10240 || m == 12288))));
      if (reuse_four_rows) {
        SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, WType, 1, 4, 1, true>
            <<<(m + 63) / 64, 512, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
    }
    if constexpr (WType == core::GgmlType::kIQ4_XS &&
                  (Batch == 2 || Batch == 3 || Batch == 4 || Batch == 5 ||
                   Batch == 7)) {
      const bool up = ffn_up;
      const bool down = m == 5120 && k == 17408;
      if (up || (Batch != 4 && down)) {
        const auto launch_rows = [&]<std::size_t Tiles>() {
          SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, WType, Tiles, 4, 1,
                                              true>
              <<<(m + 63) / 64, 512, 0, stream>>>(w, fp32_x, y, m, k);
        };
        if constexpr (Batch == 3) {
          if (up) {
            launch_rows.template operator()<1>();
          } else {
            launch_rows.template operator()<2>();
          }
        } else {
          launch_rows.template operator()<2>();
        }
        return;
      }
    }
    if constexpr (WType == core::GgmlType::kIQ4_XS && Batch == 8) {
      if ((ffn_up) || (m == 5120 && k == 17408)) {
        SmallBatchKQuantExactFp32GEMMKernel<8, Batch, 4, WType, 1, 4, 1, true>
            <<<(m + 31) / 32, 256, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
    }
    if constexpr (Batch >= 2 && (WType == core::GgmlType::kQ4_K ||
                                 WType == core::GgmlType::kQ5_K)) {
      if (FitsSmallBatch32BitIndices(Batch, m, k,
                                     quant::QuantizedRowBytes(WType, k))) {
        if constexpr (WType == core::GgmlType::kQ4_K) {
          const bool up = ffn_up;
          const bool down = m == 5120 && k == 17408;
          // Four rows share two staged tiles. Widths five and six benefit
          // only for the longer down projection.
          if (down || (up && Batch != 5 && Batch != 6)) {
            SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, WType, 2, 4, 1,
                                                true>
                <<<(m + 63) / 64, 512, 0, stream>>>(w, fp32_x, y, m, k);
            return;
          }
        }
        if constexpr (WType == core::GgmlType::kQ5_K && Batch <= 6) {
          const auto launch_rows = [&]<std::size_t Tiles,
                                       std::size_t TokensPerStep = Batch,
                                       std::uint32_t Waves = 16>() {
            SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, 4, WType, Tiles,
                                                4, 1, true, TokensPerStep>
                <<<(m + Waves * 4 - 1) / (Waves * 4), Waves * 32, 0, stream>>>(
                    w, fp32_x, y, m, k);
          };
          // Short adaptive blocks frequently use these medium projections.
          // Reuse four rows only at widths/shapes with measured gains.
          if constexpr (Batch == 5) {
            if (m == 6144 && k == 5120) {
              launch_rows.template operator()<1, Batch, 8>();
              return;
            }
            if (m == 5120 && k == 6144) {
              launch_rows.template operator()<1>();
              return;
            }
          }
          if constexpr (Batch == 5 || Batch == 6) {
            if ((m == 5120 && k == 6144) ||
                (k == 5120 && (m == 6144 || m == 10240 || m == 12288))) {
              launch_rows.template operator()<2>();
              return;
            }
          }
          // Reuse four output rows for the FFNs and selected smaller
          // projections, including the shortest speculative blocks.
          if constexpr (Batch == 2) {
            if ((m == 5120 && k == 6144) || (m == 10240 && k == 5120)) {
              launch_rows.template operator()<2>();
              return;
            }
          }
          if constexpr (Batch == 4) {
            if (m == 5120 && k == 6144) {
              launch_rows.template operator()<1, 2>();
              return;
            }
            if (m == 6144 && k == 5120) {
              launch_rows.template operator()<1>();
              return;
            }
          }
          if (m == 5120 && k == 17408) {
            if constexpr (Batch == 4) {
              launch_rows.template operator()<1, 2>();
            } else if constexpr (Batch == 5) {
              launch_rows.template operator()<1>();
            } else {
              launch_rows.template operator()<2>();
            }
            return;
          }
          if constexpr (Batch == 2 || Batch == 3 || Batch == 5) {
            if (ffn_up) {
              launch_rows.template operator()<2>();
              return;
            }
          }
        }
        if constexpr (WType == core::GgmlType::kQ5_K && Batch >= 7) {
          const auto launch_rows = [&]<std::uint32_t Waves,
                                       std::size_t TokensPerStep = Batch>() {
            SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, 4, WType, 1, 4, 1,
                                                true, TokensPerStep>
                <<<(m + Waves * 4 - 1) / (Waves * 4), Waves * 32, 0, stream>>>(
                    w, fp32_x, y, m, k);
          };
          // These batch-8 projections finish one token's dots at a time.
          // The long FFN down projection shares staging across more waves.
          if constexpr (Batch == 8) {
            if (m == 5120 && k == 17408) {
              launch_rows.template operator()<16, 1>();
              return;
            }
            if ((m == 5120 && k == 6144) ||
                (k == 5120 && (m == 1024 || m == 6144 || m == 10240 ||
                               m == 12288 || m == 17408 || m == 34816))) {
              launch_rows.template operator()<8, 1>();
              return;
            }
          }
          launch_rows.template operator()<8>();
          return;
        } else {
          constexpr bool kDoubleStage =
              (WType == core::GgmlType::kQ4_K && Batch >= 5) ||
              (WType == core::GgmlType::kQ5_K && Batch >= 5 && Batch <= 6);
          if constexpr (kDoubleStage) {
            if ((ffn_up) || (m == 5120 && k == 17408)) {
              SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                                  RowsPerWave, WType, 2, 12, 1,
                                                  true>
                  <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
              return;
            }
          }
          SmallBatchKQuantExactFp32GEMMKernel<
              kKQuantSmallBatchWarps, Batch, RowsPerWave, WType, 1, 12, 1, true>
              <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        }
      }
    }
    // At the large FFN shapes, two tiles halve staging barriers. Select only
    // the widths/formats that benefit: notably Q4 batch 7 also avoids spills.
    constexpr bool kDoubleStage =
        (WType == core::GgmlType::kQ4_K && Batch >= 5) ||
        (WType == core::GgmlType::kQ5_K && (Batch == 6 || Batch == 7)) ||
        (WType == core::GgmlType::kIQ4_XS &&
         (Batch == 5 || Batch == 6 || Batch == 8));
    if constexpr (kDoubleStage) {
      if ((ffn_up) || (m == 5120 && k == 17408)) {
        SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                            RowsPerWave, WType, 2>
            <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
    }
    SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                        RowsPerWave, WType>
        <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
  };
#define GUFO_KQUANT_SMALL_BATCH(N)   \
  case N:                            \
    launch.template operator()<N>(); \
    return;
  switch (batch) {
    GUFO_KQUANT_SMALL_BATCH(1)
    GUFO_KQUANT_SMALL_BATCH(2)
    GUFO_KQUANT_SMALL_BATCH(3)
    GUFO_KQUANT_SMALL_BATCH(4)
    GUFO_KQUANT_SMALL_BATCH(5)
    GUFO_KQUANT_SMALL_BATCH(6)
    GUFO_KQUANT_SMALL_BATCH(7)
    GUFO_KQUANT_SMALL_BATCH(8)
    default:
      return;
  }
#undef GUFO_KQUANT_SMALL_BATCH
}

// Dev switch GUFO_DEV_SMALLBATCH_TUNED=1 (read once): route the
// (format, width, shape) cells in kTunedSmallBatch below to the geometries
// tools/qwen27b/dflash_gemm_bench.hip measured faster. Same kernel template
// with the bench's exact arguments, built in this wave32 unit like the bench's
// own kernels; results stay bit-exact. Default off = production dispatch.
// The first version covered Q4_K B5/B8 and Q5_K B5; this one adds the
// per-format cells and turns the if-chain into the static table.
[[nodiscard]] static bool GufoDevSmallBatchTuned() noexcept {
  static const bool enabled = [] {
    const char* v = std::getenv("GUFO_DEV_SMALLBATCH_TUNED");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return enabled;
}

// The measured geometries live in ONE static table, read once, instead
// of a growing if-chain. Modelled on apollosenvy/kernel-anvil (MIT, RDNA3),
// which auto-tunes llama.cpp's mmvq per shape and feeds the result in as a
// config rather than as code; llama.cpp issues/28090 shows why the key has to
// be per shape and not per format, since on one model the MMQ/MMVQ crossover
// moved from width 6 to "never" across four projections of the same net.
//
// Every geometry name here is the geometry name the bench prints, so a row can
// be copied straight out of bench_types3.log with nothing to translate.
enum class TunedGeometry : std::uint8_t {
  kNarrowTile1,
  kNarrowTile2,
  kNarrowRows3Tile2,
  kNarrowRows2Tile1,
  kRows4Tile1,
  kRows4Tile2,
  kNarrowRows4,
  kTokenMajorRows4,
  kWaves32Rows4,
  kNarrowGroups4,  // width 8 only: two token groups of four, one launch
  kSplit2Plus3,    // width 5 only: rows 0-1 then rows 2-4, two launches
};

struct TunedEntry {
  core::GgmlType type;
  std::uint32_t batch;
  std::uint32_t m;
  std::uint32_t k;
  TunedGeometry geometry;
};

// Shapes are written out, never as a predicate: these are the exact (m, k)
// pairs the bench measured, and a predicate would silently also claim shapes
// nobody timed.
//
// FILLED FROM THE CHAIN HARNESS (chained dflash_gemm_bench run,
// fails=0, 768 cells, GUFO_DFLASH_CHAIN=1 across 6 shapes incl. 34816x5120).
//
// A row is here only because its geometry beat the PRODUCTION row for that
// exact cell by more than the pass1/pass2 spread. "Production row" means
// production_wave64 where the native wave64 small-batch path accepts the cell,
// otherwise the transcribed wave32 shape-gated pick for that format. The nine
// cells wave64 accepts are all Q6_K (B5/B7/B8 on 5120x6144, 5120x17408,
// 17408x5120, 34816x5120); it beat every sweep geometry in six of them, so
// those six are deliberately absent. The three it lost are wired.
//
// The first version's rows are NOT carried over, and nothing replaces them:
//
// Measured GUFO_DEV_SMALLBATCH_TUNED on real decode: prose tg -4 to -7% in
// every clean cell. Re-ranking bench.log -- the same log those winners were
// chosen from -- against the row production actually runs accounts for it:
//   type B       M      K   production_wave64   b5_split2+3 / narrow_groups4
//   q4   5   17408   5120     0.2468 ms           0.4472 ms   -81%
//   q4   5    5120  17408     0.2789              0.4657      -67%
//   q4   5    6144   5120     0.1194              0.1939      -62%
//   q4   5   10240   5120     0.1663              0.2689      -62%
//   q4   5   12288   5120     0.2019              0.3341      -66%
//   q4   8   17408   5120     0.3338              0.4790      -44%
//   q4   8    5120  17408     0.3124              0.5082      -63%
//   q4   8    6144   5120     0.1515              0.1846      -22%
//   q4   8   10240   5120     0.2269              0.2974      -31%
//   q4   8   12288   5120     0.2808              0.3345      -19%
//   q5   5   17408   5120     0.2919              0.5498      -88%
// Those geometries were the fastest in the WAVE32 family, and that is what was
// ranked. But LaunchKQuantSmallBatchByType tries the tuned table BEFORE
// TryLaunchKQuantSmallBatchWave64, and wave64 accepts every one of these
// (format, width, shape) cells -- so the switch moved the model's most
// expensive projections off a fast wave64 kernel onto a slower wave32 one.
// "Fastest in the sweep" is not "faster than production".
//
// Separately, m == 34816 (the packed gate/up: the largest GEMM per layer, and
// one of three gated projections in each of the 64 layers) was NEVER BENCHED at
// any width. The old `up` predicate matched it on the strength of a measurement
// at m == 17408. Shapes are rows here precisely so a predicate can no longer
// claim a shape nobody timed. The chained sweep above covers it.
//
// SETTLED by a second chained sweep (fails=0, 288 cells): the
// chained sweep of q4/q5/iq4 over the same 6 shapes and B1-8. The native wave64
// small-batch kernel runs in 50 of those cells (Q4_K and Q5_K at B4-8, IQ4_XS at
// B4/B6/B7/B8, all 6 shapes) and it BEAT every sweep geometry in 49 of them.
// Exactly one cell went the other way, and it is the only Q4_K row below:
//   q4 B4 5120x17408  rows4_tile2  +6.2% vs production_wave64 (spread 0.65%)
// It was +8.6% in the old pipelined data, so chaining cost it margin but not the
// win. The other two first-version candidates sat at m == 12288, which is not
// among the six shapes that sweep ran, so they are untested rather than refuted -- and on
// this evidence the prior is that wave64 wins there too.
//
// That 49-of-50 is the clearest statement of the regression's root cause: for
// Q4_K/Q5_K/IQ4_XS there was never a wave32 geometry to find, so every old
// row was measuring the wrong contest. It also means these three formats take
// essentially nothing from this table, and the gain lives in the low-bit
// formats wave64 refuses (IQ2/IQ3/Q2_K/Q3_K) plus Q6_K.
//
// B1-B3 for q4/q5/iq4 stay unwired on purpose: wave64 does not run there, and
// the wave32 shape-gated tree for those formats is still not transcribed, so
// those cells have no baseline the rule can use. The analyzer skips them loudly
// rather than comparing against a plausible-looking wrong row -- which is
// precisely the mistake that produced the decode regression.
//
// Gains below are per cell against that cell's production row; the geometry
// that wins is not the same one across formats or widths, which is the
// result itself. Today's generic fallback, tile1, wins almost nowhere.
inline constexpr TunedEntry kTunedSmallBatchRows[] = {
    // ---- kQ4_K ---- (one cell; wave64 takes the other 49 it runs in)
    {core::GgmlType::kQ4_K, 4, 5120, 17408, TunedGeometry::kRows4Tile2},  // +6.2% vs production_wave64
    // ---- kQ6_K ----
    {core::GgmlType::kQ6_K, 3, 5120, 6144, TunedGeometry::kNarrowTile1},  // +1.5% vs tile1
    {core::GgmlType::kQ6_K, 4, 5120, 6144, TunedGeometry::kNarrowRows3Tile2},  // +3.7% vs tile1
    {core::GgmlType::kQ6_K, 6, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +7.3% vs rows4_tile1
    {core::GgmlType::kQ6_K, 8, 5120, 6144, TunedGeometry::kRows4Tile2},  // +5.4% vs production_wave64
    {core::GgmlType::kQ6_K, 2, 5120, 17408, TunedGeometry::kRows4Tile2},  // +7.9% vs tile1
    {core::GgmlType::kQ6_K, 3, 5120, 17408, TunedGeometry::kRows4Tile2},  // +1.7% vs tile1
    {core::GgmlType::kQ6_K, 4, 5120, 17408, TunedGeometry::kNarrowTile1},  // +4.6% vs tile1
    {core::GgmlType::kQ6_K, 6, 5120, 17408, TunedGeometry::kRows4Tile2},  // +5.8% vs narrow_rows3_tile2
    {core::GgmlType::kQ6_K, 7, 5120, 17408, TunedGeometry::kRows4Tile2},  // +3.5% vs production_wave64
    {core::GgmlType::kQ6_K, 8, 5120, 17408, TunedGeometry::kRows4Tile2},  // +7.9% vs production_wave64
    {core::GgmlType::kQ6_K, 2, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +6.5% vs tile1
    {core::GgmlType::kQ6_K, 3, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +9.8% vs tile1
    {core::GgmlType::kQ6_K, 4, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +14.8% vs tile1
    {core::GgmlType::kQ6_K, 5, 6144, 5120, TunedGeometry::kNarrowRows4},  // +7.6% vs tile1
    {core::GgmlType::kQ6_K, 6, 6144, 5120, TunedGeometry::kNarrowRows4},  // +37.3% vs tile1
    {core::GgmlType::kQ6_K, 7, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +29.9% vs tile1
    {core::GgmlType::kQ6_K, 8, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +28.1% vs tile1
    {core::GgmlType::kQ6_K, 2, 10240, 5120, TunedGeometry::kNarrowRows3Tile2},  // +1.3% vs tile1
    {core::GgmlType::kQ6_K, 4, 10240, 5120, TunedGeometry::kNarrowTile1},  // +3.1% vs tile1
    {core::GgmlType::kQ6_K, 5, 10240, 5120, TunedGeometry::kNarrowRows3Tile2},  // +1.6% vs tile1
    {core::GgmlType::kQ6_K, 6, 10240, 5120, TunedGeometry::kRows4Tile2},  // +31.8% vs tile1
    {core::GgmlType::kQ6_K, 7, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +29.1% vs tile1
    {core::GgmlType::kQ6_K, 8, 10240, 5120, TunedGeometry::kRows4Tile2},  // +3.7% vs rows4_tile1
    {core::GgmlType::kQ6_K, 3, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +3.2% vs tile1
    {core::GgmlType::kQ6_K, 4, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +4.5% vs tile1
    {core::GgmlType::kQ6_K, 3, 34816, 5120, TunedGeometry::kNarrowRows4},  // +1.6% vs tile1
    {core::GgmlType::kQ6_K, 4, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +2.3% vs tile1
    {core::GgmlType::kQ6_K, 5, 34816, 5120, TunedGeometry::kNarrowTile2},  // +1.0% vs tile1
    {core::GgmlType::kQ6_K, 6, 34816, 5120, TunedGeometry::kNarrowRows3Tile2},  // +35.3% vs tile1
    // ---- kQ3_K ----
    {core::GgmlType::kQ3_K, 2, 5120, 6144, TunedGeometry::kNarrowRows4},  // +15.8% vs tile1
    {core::GgmlType::kQ3_K, 3, 5120, 6144, TunedGeometry::kNarrowTile2},  // +5.0% vs tile1
    {core::GgmlType::kQ3_K, 4, 5120, 6144, TunedGeometry::kNarrowTile2},  // +1.8% vs tile1
    {core::GgmlType::kQ3_K, 5, 5120, 6144, TunedGeometry::kNarrowTile2},  // +6.5% vs tile1
    {core::GgmlType::kQ3_K, 6, 5120, 6144, TunedGeometry::kNarrowRows3Tile2},  // +11.3% vs tile1
    {core::GgmlType::kQ3_K, 7, 5120, 6144, TunedGeometry::kNarrowTile2},  // +8.5% vs tile1
    {core::GgmlType::kQ3_K, 8, 5120, 6144, TunedGeometry::kNarrowTile2},  // +10.2% vs tile1
    {core::GgmlType::kQ3_K, 2, 5120, 17408, TunedGeometry::kNarrowRows4},  // +18.4% vs tile1
    {core::GgmlType::kQ3_K, 4, 5120, 17408, TunedGeometry::kNarrowTile2},  // +2.4% vs tile1
    {core::GgmlType::kQ3_K, 5, 5120, 17408, TunedGeometry::kNarrowRows3Tile2},  // +9.2% vs tile1
    {core::GgmlType::kQ3_K, 6, 5120, 17408, TunedGeometry::kNarrowTile2},  // +14.3% vs tile1
    {core::GgmlType::kQ3_K, 7, 5120, 17408, TunedGeometry::kRows4Tile2},  // +12.9% vs tile1
    {core::GgmlType::kQ3_K, 8, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +13.5% vs tile1
    {core::GgmlType::kQ3_K, 2, 6144, 5120, TunedGeometry::kNarrowRows2Tile1},  // +7.5% vs tile1
    {core::GgmlType::kQ3_K, 3, 6144, 5120, TunedGeometry::kNarrowTile1},  // +7.6% vs tile1
    {core::GgmlType::kQ3_K, 5, 6144, 5120, TunedGeometry::kNarrowRows3Tile2},  // +6.9% vs tile1
    {core::GgmlType::kQ3_K, 6, 6144, 5120, TunedGeometry::kNarrowRows3Tile2},  // +4.8% vs tile1
    {core::GgmlType::kQ3_K, 7, 6144, 5120, TunedGeometry::kNarrowTile2},  // +7.8% vs tile1
    {core::GgmlType::kQ3_K, 8, 6144, 5120, TunedGeometry::kNarrowTile2},  // +8.4% vs tile1
    {core::GgmlType::kQ3_K, 2, 10240, 5120, TunedGeometry::kNarrowRows4},  // +9.6% vs tile1
    {core::GgmlType::kQ3_K, 3, 10240, 5120, TunedGeometry::kNarrowTile1},  // +7.5% vs tile1
    {core::GgmlType::kQ3_K, 4, 10240, 5120, TunedGeometry::kNarrowTile2},  // +2.6% vs tile1
    {core::GgmlType::kQ3_K, 5, 10240, 5120, TunedGeometry::kNarrowTile2},  // +8.8% vs tile1
    {core::GgmlType::kQ3_K, 6, 10240, 5120, TunedGeometry::kNarrowRows3Tile2},  // +9.6% vs tile1
    {core::GgmlType::kQ3_K, 7, 10240, 5120, TunedGeometry::kNarrowTile1},  // +9.0% vs tile1
    {core::GgmlType::kQ3_K, 8, 10240, 5120, TunedGeometry::kNarrowTile2},  // +3.4% vs tile1
    {core::GgmlType::kQ3_K, 2, 17408, 5120, TunedGeometry::kNarrowRows4},  // +14.1% vs tile1
    {core::GgmlType::kQ3_K, 3, 17408, 5120, TunedGeometry::kNarrowTile1},  // +9.3% vs tile1
    {core::GgmlType::kQ3_K, 4, 17408, 5120, TunedGeometry::kNarrowTile2},  // +2.8% vs tile1
    {core::GgmlType::kQ3_K, 5, 17408, 5120, TunedGeometry::kNarrowTile2},  // +9.8% vs tile1
    {core::GgmlType::kQ3_K, 6, 17408, 5120, TunedGeometry::kNarrowRows3Tile2},  // +10.2% vs tile1
    {core::GgmlType::kQ3_K, 7, 17408, 5120, TunedGeometry::kNarrowTile2},  // +10.3% vs tile1
    {core::GgmlType::kQ3_K, 8, 17408, 5120, TunedGeometry::kNarrowTile2},  // +10.5% vs tile1
    {core::GgmlType::kQ3_K, 2, 34816, 5120, TunedGeometry::kNarrowRows4},  // +13.2% vs tile1
    {core::GgmlType::kQ3_K, 3, 34816, 5120, TunedGeometry::kNarrowTile1},  // +8.5% vs tile1
    {core::GgmlType::kQ3_K, 4, 34816, 5120, TunedGeometry::kNarrowTile2},  // +2.4% vs tile1
    {core::GgmlType::kQ3_K, 5, 34816, 5120, TunedGeometry::kNarrowTile2},  // +10.9% vs tile1
    {core::GgmlType::kQ3_K, 6, 34816, 5120, TunedGeometry::kNarrowRows3Tile2},  // +11.6% vs tile1
    {core::GgmlType::kQ3_K, 7, 34816, 5120, TunedGeometry::kNarrowTile2},  // +12.7% vs tile1
    {core::GgmlType::kQ3_K, 8, 34816, 5120, TunedGeometry::kNarrowTile2},  // +10.8% vs tile1
    // ---- kIQ4_NL ----
    {core::GgmlType::kIQ4_NL, 2, 5120, 6144, TunedGeometry::kNarrowTile2},  // +2.8% vs tile1
    {core::GgmlType::kIQ4_NL, 3, 5120, 6144, TunedGeometry::kRows4Tile2},  // +6.6% vs tile1
    {core::GgmlType::kIQ4_NL, 4, 5120, 6144, TunedGeometry::kRows4Tile2},  // +6.1% vs tile1
    {core::GgmlType::kIQ4_NL, 5, 5120, 6144, TunedGeometry::kRows4Tile2},  // +13.3% vs tile1
    {core::GgmlType::kIQ4_NL, 6, 5120, 6144, TunedGeometry::kRows4Tile2},  // +12.0% vs tile1
    {core::GgmlType::kIQ4_NL, 7, 5120, 6144, TunedGeometry::kRows4Tile2},  // +16.5% vs tile1
    {core::GgmlType::kIQ4_NL, 8, 5120, 6144, TunedGeometry::kNarrowRows4},  // +27.0% vs tile1
    {core::GgmlType::kIQ4_NL, 2, 5120, 17408, TunedGeometry::kRows4Tile2},  // +7.7% vs tile1
    {core::GgmlType::kIQ4_NL, 3, 5120, 17408, TunedGeometry::kRows4Tile2},  // +9.7% vs tile1
    {core::GgmlType::kIQ4_NL, 4, 5120, 17408, TunedGeometry::kRows4Tile2},  // +14.6% vs tile1
    {core::GgmlType::kIQ4_NL, 5, 5120, 17408, TunedGeometry::kRows4Tile2},  // +9.1% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 6, 5120, 17408, TunedGeometry::kRows4Tile2},  // +4.1% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 7, 5120, 17408, TunedGeometry::kRows4Tile2},  // +6.3% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 8, 5120, 17408, TunedGeometry::kNarrowRows4},  // +5.4% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 2, 6144, 5120, TunedGeometry::kNarrowRows4},  // +12.6% vs tile1
    {core::GgmlType::kIQ4_NL, 3, 6144, 5120, TunedGeometry::kNarrowRows4},  // +13.1% vs tile1
    {core::GgmlType::kIQ4_NL, 4, 6144, 5120, TunedGeometry::kNarrowRows4},  // +11.7% vs tile1
    {core::GgmlType::kIQ4_NL, 5, 6144, 5120, TunedGeometry::kNarrowRows4},  // +18.7% vs tile1
    {core::GgmlType::kIQ4_NL, 6, 6144, 5120, TunedGeometry::kNarrowRows4},  // +20.0% vs tile1
    {core::GgmlType::kIQ4_NL, 7, 6144, 5120, TunedGeometry::kNarrowRows4},  // +21.2% vs tile1
    {core::GgmlType::kIQ4_NL, 8, 6144, 5120, TunedGeometry::kNarrowRows4},  // +20.4% vs tile1
    {core::GgmlType::kIQ4_NL, 2, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +3.9% vs tile1
    {core::GgmlType::kIQ4_NL, 3, 10240, 5120, TunedGeometry::kRows4Tile2},  // +7.7% vs tile1
    {core::GgmlType::kIQ4_NL, 4, 10240, 5120, TunedGeometry::kNarrowRows4},  // +8.3% vs tile1
    {core::GgmlType::kIQ4_NL, 5, 10240, 5120, TunedGeometry::kRows4Tile2},  // +12.2% vs tile1
    {core::GgmlType::kIQ4_NL, 6, 10240, 5120, TunedGeometry::kNarrowRows4},  // +9.8% vs tile1
    {core::GgmlType::kIQ4_NL, 7, 10240, 5120, TunedGeometry::kRows4Tile2},  // +14.2% vs tile1
    {core::GgmlType::kIQ4_NL, 8, 10240, 5120, TunedGeometry::kNarrowRows4},  // +23.6% vs tile1
    {core::GgmlType::kIQ4_NL, 2, 17408, 5120, TunedGeometry::kNarrowRows4},  // +7.4% vs tile1
    {core::GgmlType::kIQ4_NL, 3, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +4.1% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 5, 17408, 5120, TunedGeometry::kRows4Tile2},  // +11.5% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 6, 17408, 5120, TunedGeometry::kNarrowRows4},  // +7.6% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 7, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +9.4% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 8, 17408, 5120, TunedGeometry::kNarrowRows4},  // +6.8% vs rows4_tile1
    {core::GgmlType::kIQ4_NL, 2, 34816, 5120, TunedGeometry::kRows4Tile2},  // +3.9% vs tile1
    {core::GgmlType::kIQ4_NL, 3, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +6.4% vs tile1
    {core::GgmlType::kIQ4_NL, 4, 34816, 5120, TunedGeometry::kRows4Tile2},  // +11.1% vs tile1
    {core::GgmlType::kIQ4_NL, 5, 34816, 5120, TunedGeometry::kRows4Tile2},  // +13.7% vs tile1
    {core::GgmlType::kIQ4_NL, 6, 34816, 5120, TunedGeometry::kNarrowRows4},  // +8.2% vs tile1
    {core::GgmlType::kIQ4_NL, 7, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +11.1% vs tile1
    {core::GgmlType::kIQ4_NL, 8, 34816, 5120, TunedGeometry::kNarrowRows4},  // +20.6% vs tile1
    // ---- kIQ3_S ----
    {core::GgmlType::kIQ3_S, 2, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +8.1% vs tile1
    {core::GgmlType::kIQ3_S, 3, 5120, 6144, TunedGeometry::kNarrowRows4},  // +5.4% vs tile1
    {core::GgmlType::kIQ3_S, 5, 5120, 6144, TunedGeometry::kRows4Tile2},  // +10.2% vs tile1
    {core::GgmlType::kIQ3_S, 6, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +13.0% vs tile1
    {core::GgmlType::kIQ3_S, 7, 5120, 6144, TunedGeometry::kNarrowRows4},  // +17.1% vs tile1
    {core::GgmlType::kIQ3_S, 8, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +16.7% vs tile1
    {core::GgmlType::kIQ3_S, 2, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +11.5% vs tile1
    {core::GgmlType::kIQ3_S, 3, 5120, 17408, TunedGeometry::kNarrowRows4},  // +4.0% vs rows4_tile1
    {core::GgmlType::kIQ3_S, 4, 5120, 17408, TunedGeometry::kRows4Tile2},  // +3.2% vs tile1
    {core::GgmlType::kIQ3_S, 5, 5120, 17408, TunedGeometry::kRows4Tile2},  // +5.6% vs rows4_tile1
    {core::GgmlType::kIQ3_S, 6, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +5.7% vs rows4_tile1
    {core::GgmlType::kIQ3_S, 7, 5120, 17408, TunedGeometry::kNarrowRows4},  // +6.7% vs rows4_tile1
    {core::GgmlType::kIQ3_S, 8, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +6.0% vs rows4_tile1
    {core::GgmlType::kIQ3_S, 2, 6144, 5120, TunedGeometry::kNarrowRows3Tile2},  // +3.8% vs tile1
    {core::GgmlType::kIQ3_S, 3, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +4.4% vs tile1
    {core::GgmlType::kIQ3_S, 5, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +6.9% vs tile1
    {core::GgmlType::kIQ3_S, 6, 6144, 5120, TunedGeometry::kNarrowRows3Tile2},  // +7.4% vs tile1
    {core::GgmlType::kIQ3_S, 7, 6144, 5120, TunedGeometry::kNarrowRows4},  // +9.5% vs tile1
    {core::GgmlType::kIQ3_S, 8, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +8.8% vs tile1
    {core::GgmlType::kIQ3_S, 2, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +3.6% vs tile1
    {core::GgmlType::kIQ3_S, 5, 10240, 5120, TunedGeometry::kRows4Tile2},  // +8.5% vs tile1
    {core::GgmlType::kIQ3_S, 6, 10240, 5120, TunedGeometry::kRows4Tile2},  // +10.0% vs tile1
    {core::GgmlType::kIQ3_S, 7, 10240, 5120, TunedGeometry::kNarrowRows4},  // +16.4% vs tile1
    {core::GgmlType::kIQ3_S, 8, 10240, 5120, TunedGeometry::kNarrowRows4},  // +13.2% vs tile1
    {core::GgmlType::kIQ3_S, 2, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +5.0% vs tile1
    {core::GgmlType::kIQ3_S, 3, 17408, 5120, TunedGeometry::kNarrowRows4},  // +4.5% vs tile1
    {core::GgmlType::kIQ3_S, 5, 17408, 5120, TunedGeometry::kNarrowTile2},  // +5.4% vs tile1
    {core::GgmlType::kIQ3_S, 6, 17408, 5120, TunedGeometry::kNarrowRows3Tile2},  // +8.0% vs tile1
    {core::GgmlType::kIQ3_S, 7, 17408, 5120, TunedGeometry::kNarrowRows4},  // +16.0% vs tile1
    {core::GgmlType::kIQ3_S, 8, 17408, 5120, TunedGeometry::kNarrowRows4},  // +12.0% vs tile1
    {core::GgmlType::kIQ3_S, 2, 34816, 5120, TunedGeometry::kNarrowRows4},  // +5.9% vs tile1
    {core::GgmlType::kIQ3_S, 3, 34816, 5120, TunedGeometry::kNarrowRows4},  // +2.8% vs tile1
    {core::GgmlType::kIQ3_S, 5, 34816, 5120, TunedGeometry::kNarrowTile2},  // +5.4% vs tile1
    {core::GgmlType::kIQ3_S, 6, 34816, 5120, TunedGeometry::kNarrowRows3Tile2},  // +8.2% vs tile1
    {core::GgmlType::kIQ3_S, 7, 34816, 5120, TunedGeometry::kNarrowRows4},  // +15.3% vs tile1
    {core::GgmlType::kIQ3_S, 8, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +11.4% vs tile1
    // ---- kIQ3_XXS ----
    {core::GgmlType::kIQ3_XXS, 3, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +6.0% vs tile1
    {core::GgmlType::kIQ3_XXS, 4, 5120, 6144, TunedGeometry::kRows4Tile2},  // +10.6% vs tile1
    {core::GgmlType::kIQ3_XXS, 5, 5120, 6144, TunedGeometry::kRows4Tile2},  // +11.8% vs tile1
    {core::GgmlType::kIQ3_XXS, 6, 5120, 6144, TunedGeometry::kRows4Tile1},  // +7.4% vs tile1
    {core::GgmlType::kIQ3_XXS, 7, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +10.2% vs tile1
    {core::GgmlType::kIQ3_XXS, 8, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +35.5% vs tile1
    {core::GgmlType::kIQ3_XXS, 2, 5120, 17408, TunedGeometry::kRows4Tile2},  // +4.5% vs tile1
    {core::GgmlType::kIQ3_XXS, 3, 5120, 17408, TunedGeometry::kRows4Tile2},  // +5.9% vs tile1
    {core::GgmlType::kIQ3_XXS, 4, 5120, 17408, TunedGeometry::kRows4Tile2},  // +14.7% vs tile1
    {core::GgmlType::kIQ3_XXS, 5, 5120, 17408, TunedGeometry::kRows4Tile2},  // +10.7% vs tile1
    {core::GgmlType::kIQ3_XXS, 6, 5120, 17408, TunedGeometry::kRows4Tile1},  // +7.9% vs tile1
    {core::GgmlType::kIQ3_XXS, 7, 5120, 17408, TunedGeometry::kNarrowRows4},  // +14.1% vs tile1
    {core::GgmlType::kIQ3_XXS, 8, 5120, 17408, TunedGeometry::kNarrowRows4},  // +43.7% vs tile1
    {core::GgmlType::kIQ3_XXS, 2, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +6.5% vs tile1
    {core::GgmlType::kIQ3_XXS, 3, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +4.0% vs tile1
    {core::GgmlType::kIQ3_XXS, 4, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +5.1% vs tile1
    {core::GgmlType::kIQ3_XXS, 5, 6144, 5120, TunedGeometry::kNarrowTile2},  // +6.4% vs tile1
    {core::GgmlType::kIQ3_XXS, 6, 6144, 5120, TunedGeometry::kNarrowRows4},  // +7.9% vs tile1
    {core::GgmlType::kIQ3_XXS, 8, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +18.0% vs tile1
    {core::GgmlType::kIQ3_XXS, 3, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +4.2% vs tile1
    {core::GgmlType::kIQ3_XXS, 4, 10240, 5120, TunedGeometry::kRows4Tile2},  // +8.6% vs tile1
    {core::GgmlType::kIQ3_XXS, 5, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +7.0% vs tile1
    {core::GgmlType::kIQ3_XXS, 6, 10240, 5120, TunedGeometry::kNarrowRows4},  // +5.3% vs tile1
    {core::GgmlType::kIQ3_XXS, 7, 10240, 5120, TunedGeometry::kNarrowRows4},  // +5.3% vs tile1
    {core::GgmlType::kIQ3_XXS, 8, 10240, 5120, TunedGeometry::kNarrowRows4},  // +20.7% vs tile1
    {core::GgmlType::kIQ3_XXS, 2, 17408, 5120, TunedGeometry::kNarrowRows4},  // +7.2% vs tile1
    {core::GgmlType::kIQ3_XXS, 3, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +9.1% vs tile1
    {core::GgmlType::kIQ3_XXS, 4, 17408, 5120, TunedGeometry::kRows4Tile2},  // +10.5% vs tile1
    {core::GgmlType::kIQ3_XXS, 5, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +9.7% vs tile1
    {core::GgmlType::kIQ3_XXS, 6, 17408, 5120, TunedGeometry::kNarrowRows3Tile2},  // +4.7% vs tile1
    {core::GgmlType::kIQ3_XXS, 7, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +3.6% vs tile1
    {core::GgmlType::kIQ3_XXS, 8, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +22.5% vs tile1
    {core::GgmlType::kIQ3_XXS, 2, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +6.0% vs tile1
    {core::GgmlType::kIQ3_XXS, 3, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +7.3% vs tile1
    {core::GgmlType::kIQ3_XXS, 4, 34816, 5120, TunedGeometry::kRows4Tile2},  // +11.5% vs tile1
    {core::GgmlType::kIQ3_XXS, 5, 34816, 5120, TunedGeometry::kRows4Tile2},  // +8.7% vs tile1
    {core::GgmlType::kIQ3_XXS, 6, 34816, 5120, TunedGeometry::kNarrowRows3Tile2},  // +3.8% vs tile1
    {core::GgmlType::kIQ3_XXS, 7, 34816, 5120, TunedGeometry::kRows4Tile2},  // +1.0% vs tile1
    {core::GgmlType::kIQ3_XXS, 8, 34816, 5120, TunedGeometry::kNarrowRows4},  // +22.4% vs tile1
    // ---- kIQ2_S ----
    {core::GgmlType::kIQ2_S, 2, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +7.7% vs tile1
    {core::GgmlType::kIQ2_S, 3, 5120, 6144, TunedGeometry::kNarrowRows4},  // +9.7% vs tile1
    {core::GgmlType::kIQ2_S, 4, 5120, 6144, TunedGeometry::kNarrowRows4},  // +10.4% vs tile1
    {core::GgmlType::kIQ2_S, 5, 5120, 6144, TunedGeometry::kRows4Tile2},  // +10.8% vs tile1
    {core::GgmlType::kIQ2_S, 6, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +11.2% vs tile1
    {core::GgmlType::kIQ2_S, 7, 5120, 6144, TunedGeometry::kNarrowRows4},  // +29.9% vs tile1
    {core::GgmlType::kIQ2_S, 8, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +28.1% vs tile1
    {core::GgmlType::kIQ2_S, 2, 5120, 17408, TunedGeometry::kNarrowRows3Tile2},  // +8.3% vs tile1
    {core::GgmlType::kIQ2_S, 3, 5120, 17408, TunedGeometry::kNarrowRows4},  // +7.4% vs tile1
    {core::GgmlType::kIQ2_S, 4, 5120, 17408, TunedGeometry::kNarrowRows4},  // +6.9% vs tile1
    {core::GgmlType::kIQ2_S, 5, 5120, 17408, TunedGeometry::kRows4Tile2},  // +15.0% vs tile1
    {core::GgmlType::kIQ2_S, 6, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +11.9% vs tile1
    {core::GgmlType::kIQ2_S, 7, 5120, 17408, TunedGeometry::kNarrowRows4},  // +34.7% vs tile1
    {core::GgmlType::kIQ2_S, 8, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +32.4% vs tile1
    {core::GgmlType::kIQ2_S, 3, 6144, 5120, TunedGeometry::kNarrowRows4},  // +7.2% vs tile1
    {core::GgmlType::kIQ2_S, 4, 6144, 5120, TunedGeometry::kNarrowRows4},  // +5.4% vs tile1
    {core::GgmlType::kIQ2_S, 5, 6144, 5120, TunedGeometry::kNarrowRows4},  // +12.3% vs tile1
    {core::GgmlType::kIQ2_S, 6, 6144, 5120, TunedGeometry::kNarrowRows4},  // +8.5% vs tile1
    {core::GgmlType::kIQ2_S, 7, 6144, 5120, TunedGeometry::kNarrowTile1},  // +24.9% vs tile1
    {core::GgmlType::kIQ2_S, 8, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +22.3% vs tile1
    {core::GgmlType::kIQ2_S, 2, 10240, 5120, TunedGeometry::kNarrowRows4},  // +9.1% vs tile1
    {core::GgmlType::kIQ2_S, 3, 10240, 5120, TunedGeometry::kNarrowRows4},  // +6.7% vs tile1
    {core::GgmlType::kIQ2_S, 5, 10240, 5120, TunedGeometry::kNarrowRows4},  // +8.2% vs tile1
    {core::GgmlType::kIQ2_S, 6, 10240, 5120, TunedGeometry::kNarrowRows4},  // +5.4% vs tile1
    {core::GgmlType::kIQ2_S, 7, 10240, 5120, TunedGeometry::kRows4Tile2},  // +27.8% vs tile1
    {core::GgmlType::kIQ2_S, 8, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +23.6% vs tile1
    {core::GgmlType::kIQ2_S, 2, 17408, 5120, TunedGeometry::kNarrowRows4},  // +13.6% vs tile1
    {core::GgmlType::kIQ2_S, 3, 17408, 5120, TunedGeometry::kNarrowRows4},  // +8.8% vs tile1
    {core::GgmlType::kIQ2_S, 4, 17408, 5120, TunedGeometry::kNarrowRows4},  // +9.7% vs tile1
    {core::GgmlType::kIQ2_S, 5, 17408, 5120, TunedGeometry::kNarrowRows4},  // +7.9% vs tile1
    {core::GgmlType::kIQ2_S, 6, 17408, 5120, TunedGeometry::kNarrowRows4},  // +4.8% vs tile1
    {core::GgmlType::kIQ2_S, 7, 17408, 5120, TunedGeometry::kNarrowRows4},  // +29.4% vs tile1
    {core::GgmlType::kIQ2_S, 8, 17408, 5120, TunedGeometry::kTokenMajorRows4},  // +26.4% vs tile1
    {core::GgmlType::kIQ2_S, 2, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +11.1% vs tile1
    {core::GgmlType::kIQ2_S, 3, 34816, 5120, TunedGeometry::kNarrowRows4},  // +10.6% vs tile1
    {core::GgmlType::kIQ2_S, 4, 34816, 5120, TunedGeometry::kNarrowRows4},  // +8.8% vs tile1
    {core::GgmlType::kIQ2_S, 5, 34816, 5120, TunedGeometry::kNarrowRows4},  // +8.2% vs tile1
    {core::GgmlType::kIQ2_S, 6, 34816, 5120, TunedGeometry::kNarrowTile2},  // +5.1% vs tile1
    {core::GgmlType::kIQ2_S, 7, 34816, 5120, TunedGeometry::kNarrowRows4},  // +28.5% vs tile1
    {core::GgmlType::kIQ2_S, 8, 34816, 5120, TunedGeometry::kTokenMajorRows4},  // +26.7% vs tile1
    // ---- kIQ2_XS ----
    {core::GgmlType::kIQ2_XS, 2, 5120, 6144, TunedGeometry::kRows4Tile1},  // +9.4% vs tile1
    {core::GgmlType::kIQ2_XS, 3, 5120, 6144, TunedGeometry::kRows4Tile1},  // +6.1% vs tile1
    {core::GgmlType::kIQ2_XS, 4, 5120, 6144, TunedGeometry::kRows4Tile2},  // +10.4% vs tile1
    {core::GgmlType::kIQ2_XS, 5, 5120, 6144, TunedGeometry::kRows4Tile1},  // +10.0% vs tile1
    {core::GgmlType::kIQ2_XS, 6, 5120, 6144, TunedGeometry::kRows4Tile2},  // +9.2% vs tile1
    {core::GgmlType::kIQ2_XS, 7, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +7.2% vs tile1
    {core::GgmlType::kIQ2_XS, 8, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +12.3% vs tile1
    {core::GgmlType::kIQ2_XS, 2, 5120, 17408, TunedGeometry::kRows4Tile1},  // +10.0% vs tile1
    {core::GgmlType::kIQ2_XS, 3, 5120, 17408, TunedGeometry::kRows4Tile2},  // +5.1% vs tile1
    {core::GgmlType::kIQ2_XS, 4, 5120, 17408, TunedGeometry::kRows4Tile2},  // +12.8% vs tile1
    {core::GgmlType::kIQ2_XS, 5, 5120, 17408, TunedGeometry::kRows4Tile1},  // +8.7% vs tile1
    {core::GgmlType::kIQ2_XS, 6, 5120, 17408, TunedGeometry::kRows4Tile2},  // +11.1% vs tile1
    {core::GgmlType::kIQ2_XS, 7, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +8.3% vs tile1
    {core::GgmlType::kIQ2_XS, 8, 5120, 17408, TunedGeometry::kTokenMajorRows4},  // +17.8% vs tile1
    {core::GgmlType::kIQ2_XS, 2, 6144, 5120, TunedGeometry::kNarrowTile2},  // +12.6% vs tile1
    {core::GgmlType::kIQ2_XS, 3, 6144, 5120, TunedGeometry::kNarrowRows4},  // +7.2% vs tile1
    {core::GgmlType::kIQ2_XS, 4, 6144, 5120, TunedGeometry::kNarrowRows4},  // +8.0% vs tile1
    {core::GgmlType::kIQ2_XS, 5, 6144, 5120, TunedGeometry::kNarrowTile2},  // +10.3% vs tile1
    {core::GgmlType::kIQ2_XS, 6, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +10.8% vs tile1
    {core::GgmlType::kIQ2_XS, 8, 6144, 5120, TunedGeometry::kNarrowTile2},  // +2.4% vs tile1
    {core::GgmlType::kIQ2_XS, 2, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +12.0% vs tile1
    {core::GgmlType::kIQ2_XS, 3, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +7.2% vs tile1
    {core::GgmlType::kIQ2_XS, 4, 10240, 5120, TunedGeometry::kNarrowRows4},  // +7.1% vs tile1
    {core::GgmlType::kIQ2_XS, 5, 10240, 5120, TunedGeometry::kNarrowRows4},  // +6.3% vs tile1
    {core::GgmlType::kIQ2_XS, 6, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +7.3% vs tile1
    {core::GgmlType::kIQ2_XS, 7, 10240, 5120, TunedGeometry::kNarrowRows4},  // +2.4% vs tile1
    {core::GgmlType::kIQ2_XS, 8, 10240, 5120, TunedGeometry::kNarrowTile1},  // +1.3% vs tile1
    {core::GgmlType::kIQ2_XS, 2, 17408, 5120, TunedGeometry::kNarrowRows4},  // +12.6% vs tile1
    {core::GgmlType::kIQ2_XS, 3, 17408, 5120, TunedGeometry::kNarrowRows4},  // +8.5% vs tile1
    {core::GgmlType::kIQ2_XS, 4, 17408, 5120, TunedGeometry::kNarrowRows4},  // +10.3% vs tile1
    {core::GgmlType::kIQ2_XS, 5, 17408, 5120, TunedGeometry::kNarrowRows4},  // +5.5% vs tile1
    {core::GgmlType::kIQ2_XS, 6, 17408, 5120, TunedGeometry::kNarrowTile2},  // +7.7% vs tile1
    {core::GgmlType::kIQ2_XS, 8, 17408, 5120, TunedGeometry::kNarrowTile1},  // +2.6% vs tile1
    {core::GgmlType::kIQ2_XS, 2, 34816, 5120, TunedGeometry::kNarrowRows4},  // +11.7% vs tile1
    {core::GgmlType::kIQ2_XS, 3, 34816, 5120, TunedGeometry::kNarrowRows4},  // +6.0% vs tile1
    {core::GgmlType::kIQ2_XS, 4, 34816, 5120, TunedGeometry::kRows4Tile2},  // +10.4% vs tile1
    {core::GgmlType::kIQ2_XS, 5, 34816, 5120, TunedGeometry::kNarrowRows4},  // +6.9% vs tile1
    {core::GgmlType::kIQ2_XS, 6, 34816, 5120, TunedGeometry::kNarrowRows4},  // +5.9% vs tile1
    {core::GgmlType::kIQ2_XS, 8, 34816, 5120, TunedGeometry::kNarrowTile1},  // +2.8% vs tile1
    // ---- kQ2_K ----
    {core::GgmlType::kQ2_K, 4, 5120, 6144, TunedGeometry::kRows4Tile2},  // +9.6% vs tile1
    {core::GgmlType::kQ2_K, 5, 5120, 6144, TunedGeometry::kRows4Tile2},  // +11.1% vs tile1
    {core::GgmlType::kQ2_K, 6, 5120, 6144, TunedGeometry::kTokenMajorRows4},  // +6.8% vs tile1
    {core::GgmlType::kQ2_K, 7, 5120, 6144, TunedGeometry::kRows4Tile2},  // +11.3% vs tile1
    {core::GgmlType::kQ2_K, 8, 5120, 6144, TunedGeometry::kRows4Tile2},  // +25.8% vs tile1
    {core::GgmlType::kQ2_K, 2, 5120, 17408, TunedGeometry::kRows4Tile2},  // +2.0% vs tile1
    {core::GgmlType::kQ2_K, 3, 5120, 17408, TunedGeometry::kRows4Tile2},  // +5.3% vs tile1
    {core::GgmlType::kQ2_K, 4, 5120, 17408, TunedGeometry::kRows4Tile2},  // +11.8% vs tile1
    {core::GgmlType::kQ2_K, 5, 5120, 17408, TunedGeometry::kRows4Tile2},  // +14.7% vs tile1
    {core::GgmlType::kQ2_K, 6, 5120, 17408, TunedGeometry::kRows4Tile2},  // +9.8% vs tile1
    {core::GgmlType::kQ2_K, 7, 5120, 17408, TunedGeometry::kRows4Tile2},  // +13.2% vs tile1
    {core::GgmlType::kQ2_K, 8, 5120, 17408, TunedGeometry::kRows4Tile2},  // +33.7% vs tile1
    {core::GgmlType::kQ2_K, 2, 6144, 5120, TunedGeometry::kRows4Tile1},  // +8.5% vs tile1
    {core::GgmlType::kQ2_K, 3, 6144, 5120, TunedGeometry::kNarrowRows4},  // +13.9% vs tile1
    {core::GgmlType::kQ2_K, 4, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +11.3% vs tile1
    {core::GgmlType::kQ2_K, 5, 6144, 5120, TunedGeometry::kTokenMajorRows4},  // +9.3% vs tile1
    {core::GgmlType::kQ2_K, 6, 6144, 5120, TunedGeometry::kNarrowRows4},  // +11.0% vs tile1
    {core::GgmlType::kQ2_K, 7, 6144, 5120, TunedGeometry::kNarrowRows4},  // +14.7% vs tile1
    {core::GgmlType::kQ2_K, 8, 6144, 5120, TunedGeometry::kNarrowRows4},  // +7.2% vs tile1
    {core::GgmlType::kQ2_K, 2, 10240, 5120, TunedGeometry::kRows4Tile1},  // +7.3% vs tile1
    {core::GgmlType::kQ2_K, 3, 10240, 5120, TunedGeometry::kNarrowRows4},  // +4.1% vs tile1
    {core::GgmlType::kQ2_K, 4, 10240, 5120, TunedGeometry::kRows4Tile2},  // +7.0% vs tile1
    {core::GgmlType::kQ2_K, 5, 10240, 5120, TunedGeometry::kNarrowRows4},  // +7.7% vs tile1
    {core::GgmlType::kQ2_K, 6, 10240, 5120, TunedGeometry::kTokenMajorRows4},  // +3.6% vs tile1
    {core::GgmlType::kQ2_K, 7, 10240, 5120, TunedGeometry::kRows4Tile2},  // +7.6% vs tile1
    {core::GgmlType::kQ2_K, 8, 10240, 5120, TunedGeometry::kRows4Tile2},  // +13.8% vs tile1
    {core::GgmlType::kQ2_K, 2, 17408, 5120, TunedGeometry::kRows4Tile1},  // +11.3% vs tile1
    {core::GgmlType::kQ2_K, 3, 17408, 5120, TunedGeometry::kNarrowRows4},  // +6.0% vs tile1
    {core::GgmlType::kQ2_K, 4, 17408, 5120, TunedGeometry::kRows4Tile2},  // +9.5% vs tile1
    {core::GgmlType::kQ2_K, 5, 17408, 5120, TunedGeometry::kRows4Tile2},  // +10.0% vs tile1
    {core::GgmlType::kQ2_K, 6, 17408, 5120, TunedGeometry::kNarrowTile2},  // +2.0% vs tile1
    {core::GgmlType::kQ2_K, 7, 17408, 5120, TunedGeometry::kNarrowRows4},  // +6.6% vs tile1
    {core::GgmlType::kQ2_K, 8, 17408, 5120, TunedGeometry::kRows4Tile2},  // +10.5% vs tile1
    {core::GgmlType::kQ2_K, 2, 34816, 5120, TunedGeometry::kRows4Tile1},  // +13.3% vs tile1
    {core::GgmlType::kQ2_K, 4, 34816, 5120, TunedGeometry::kRows4Tile2},  // +9.0% vs tile1
    {core::GgmlType::kQ2_K, 6, 34816, 5120, TunedGeometry::kNarrowTile2},  // +1.9% vs tile1
    {core::GgmlType::kQ2_K, 7, 34816, 5120, TunedGeometry::kNarrowRows4},  // +4.9% vs tile1
    {core::GgmlType::kQ2_K, 8, 34816, 5120, TunedGeometry::kRows4Tile2},  // +10.1% vs tile1
};
// std::to_array so the row count cannot go stale against the rows.
inline constexpr auto kTunedSmallBatch = std::to_array(kTunedSmallBatchRows);

// kSplit2Plus3 and kNarrowGroups4 exist at one width only. A row that puts one
// at another width would make LaunchTunedGeometry fall through having launched
// nothing, and the caller would read a stale output buffer as a result. That is
// silent wrong numbers, so it has to fail the build, not the bench.
//
// Takes the rows rather than reading the table, so the guard can be proven on a
// fixture below. With an empty table a guard that only reads the table is
// vacuously true, which is the same as not having one.
[[nodiscard]] constexpr bool TunedRowsAreSane(const TunedEntry* rows,
                                              std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    const auto& entry = rows[i];
    if (entry.geometry == TunedGeometry::kSplit2Plus3 && entry.batch != 5)
      return false;
    if (entry.geometry == TunedGeometry::kNarrowGroups4 && entry.batch != 8)
      return false;
    if (entry.batch < 1 || entry.batch > 8) return false;
  }
  return true;
}
static_assert(TunedRowsAreSane(kTunedSmallBatch.data(),
                               kTunedSmallBatch.size()),
              "a tuned row uses a geometry its width cannot launch");

// Fixtures: the guard has to actually reject, and actually accept, whatever the
// live table currently holds.
namespace tuned_table_test {
inline constexpr TunedEntry kWidthOk[] = {
    {core::GgmlType::kQ4_K, 5, 17408, 5120, TunedGeometry::kSplit2Plus3},
    {core::GgmlType::kQ4_K, 8, 17408, 5120, TunedGeometry::kNarrowGroups4},
};
inline constexpr TunedEntry kSplitAtWrongWidth[] = {
    {core::GgmlType::kQ4_K, 6, 17408, 5120, TunedGeometry::kSplit2Plus3},
};
inline constexpr TunedEntry kGroupsAtWrongWidth[] = {
    {core::GgmlType::kQ4_K, 5, 17408, 5120, TunedGeometry::kNarrowGroups4},
};
static_assert(TunedRowsAreSane(kWidthOk, 2));
static_assert(!TunedRowsAreSane(kSplitAtWrongWidth, 1));
static_assert(!TunedRowsAreSane(kGroupsAtWrongWidth, 1));
}  // namespace tuned_table_test

[[nodiscard]] constexpr const TunedEntry* FindTunedRow(
    const TunedEntry* rows, std::size_t count, core::GgmlType type,
    std::uint32_t batch, std::uint64_t m, std::uint64_t k) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    const auto& entry = rows[i];
    if (entry.type == type && entry.batch == batch && entry.m == m &&
        entry.k == k)
      return &entry;
  }
  return nullptr;
}

[[nodiscard]] constexpr const TunedEntry* FindTunedEntry(
    core::GgmlType type, std::uint32_t batch, std::uint64_t m,
    std::uint64_t k) noexcept {
  return FindTunedRow(kTunedSmallBatch.data(), kTunedSmallBatch.size(), type,
                      batch, m, k);
}

// The table drives instantiation as well as lookup, so there is no second list
// to keep in step with it. A (format, width) pair the table never names costs
// nothing in this translation unit, and neither does a geometry it never picks.
template<core::GgmlType Type, std::size_t Batch>
[[nodiscard]] constexpr bool TunedTableNames() noexcept {
  for (const auto& entry : kTunedSmallBatch) {
    if (entry.type == Type && entry.batch == Batch) return true;
  }
  return false;
}

template<core::GgmlType Type, std::size_t Batch, TunedGeometry Geometry>
[[nodiscard]] constexpr bool TunedTablePicks() noexcept {
  for (const auto& entry : kTunedSmallBatch) {
    if (entry.type == Type && entry.batch == Batch &&
        entry.geometry == Geometry)
      return true;
  }
  return false;
}

// What the live table must and must not contain. Q4_K has exactly ONE cell
// (B4 5120x17408, the one place in 50 chained cells where a sweep geometry beat
// the native wave64 kernel), and Q5_K and IQ4_XS have none: the sweep measured all
// three and wave64 won 49 of 50. Every first-version cell is asserted absent below, so
// re-adding one without a chained measurement fails the build -- which is the
// whole point: the decode regression came from wiring an unmeasured baseline.
static_assert(FindTunedEntry(core::GgmlType::kQ4_K, 4, 5120, 17408) !=
              nullptr);
static_assert(FindTunedEntry(core::GgmlType::kQ4_K, 4, 5120, 17408)->geometry ==
              TunedGeometry::kRows4Tile2);
// The first-version cells, all of which lost to wave64. These must stay misses.
static_assert(FindTunedEntry(core::GgmlType::kQ4_K, 5, 17408, 5120) ==
              nullptr);
static_assert(FindTunedEntry(core::GgmlType::kQ4_K, 5, 5120, 17408) ==
              nullptr);
static_assert(FindTunedEntry(core::GgmlType::kQ4_K, 8, 17408, 5120) ==
              nullptr);
static_assert(FindTunedEntry(core::GgmlType::kQ4_K, 8, 34816, 5120) ==
              nullptr);
static_assert(!TunedTableNames<core::GgmlType::kQ4_K, 5>());
static_assert(!TunedTableNames<core::GgmlType::kQ4_K, 8>());
static_assert(!TunedTableNames<core::GgmlType::kQ5_K, 5>());
static_assert(FindTunedEntry(core::GgmlType::kIQ4_XS, 8, 17408, 5120) ==
              nullptr);
// B1-B3 for these three formats have no transcribed baseline, so nothing may be
// wired there however fast a sweep geometry looked.
static_assert(!TunedTableNames<core::GgmlType::kQ4_K, 1>());
static_assert(!TunedTableNames<core::GgmlType::kQ4_K, 2>());
static_assert(!TunedTableNames<core::GgmlType::kQ4_K, 3>());
// IQ2_XXS is measure-only by design: benched, never wired.
static_assert(!TunedTableNames<core::GgmlType::kIQ2_XXS, 8>());

// Rows that ARE here, spot-checked so a bad splice cannot pass the build: the
// three Q6_K cells that beat the native wave64 kernel, and one of the six it
// won, which must therefore miss.
static_assert(FindTunedEntry(core::GgmlType::kQ6_K, 8, 5120, 6144) != nullptr);
static_assert(FindTunedEntry(core::GgmlType::kQ6_K, 8, 5120, 6144)->geometry ==
              TunedGeometry::kRows4Tile2);
static_assert(FindTunedEntry(core::GgmlType::kQ6_K, 7, 5120, 17408)->geometry ==
              TunedGeometry::kRows4Tile2);
static_assert(FindTunedEntry(core::GgmlType::kQ6_K, 8, 17408, 5120) ==
              nullptr);
// A shape no run covered still has to miss rather than borrow a nearby row.
static_assert(FindTunedEntry(core::GgmlType::kQ6_K, 8, 12288, 5120) ==
              nullptr);
static_assert(TunedTableNames<core::GgmlType::kIQ2_S, 8>());
static_assert(
    TunedTablePicks<core::GgmlType::kIQ2_S, 8,
                    TunedGeometry::kTokenMajorRows4>());
static_assert(
    !TunedTablePicks<core::GgmlType::kIQ2_S, 8,
                     TunedGeometry::kNarrowGroups4>());

// The lookup itself is also checked on a fixture, independent of the live rows.
namespace tuned_table_test {
static_assert(FindTunedRow(kWidthOk, 2, core::GgmlType::kQ4_K, 5, 17408,
                           5120) != nullptr);
static_assert(FindTunedRow(kWidthOk, 2, core::GgmlType::kQ4_K, 5, 17408, 5120)
                  ->geometry == TunedGeometry::kSplit2Plus3);
static_assert(FindTunedRow(kWidthOk, 2, core::GgmlType::kQ4_K, 8, 17408, 5120)
                  ->geometry == TunedGeometry::kNarrowGroups4);
// Right format and width, shape nobody timed: must miss, not fall back.
static_assert(FindTunedRow(kWidthOk, 2, core::GgmlType::kQ4_K, 5, 34816,
                           5120) == nullptr);
// Right shape, wrong width and wrong format.
static_assert(FindTunedRow(kWidthOk, 2, core::GgmlType::kQ4_K, 2, 17408,
                           5120) == nullptr);
static_assert(FindTunedRow(kWidthOk, 2, core::GgmlType::kQ5_K, 5, 17408,
                           5120) == nullptr);
}  // namespace tuned_table_test

// One launch per geometry, with the bench's exact template arguments and grid,
// so a geometry that was bit-exact in the bench is bit-exact here.
//
// A case whose TunedTablePicks is false launches nothing and returns false, so
// the caller falls through to the next family instead of reading a stale output
// buffer. The width-only geometries (kNarrowGroups4, kSplit2Plus3) return false
// at every other width for the same reason, which is also why batch-invariant
// mode cannot pin them. TunedTableWidthsAreSane keeps the table honest: do not
// remove that static_assert.
template<core::GgmlType Type, std::size_t Batch>
bool LaunchTunedGeometry(TunedGeometry geometry, const void* w, const float* x,
                         float* y, std::size_t m, std::size_t k,
                         hipStream_t stream) {
  const auto uniform = [&]<std::uint32_t Waves, std::size_t Rows,
                           std::size_t Tiles, std::uint32_t MinWaves,
                           std::size_t Step = Batch>() {
    SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, Rows, Type, Tiles,
                                        MinWaves, 1, true, Step>
        <<<(m + Waves * Rows - 1) / (Waves * Rows), Waves * 32, 0, stream>>>(
            w, x, y, m, k);
  };
  switch (geometry) {
    case TunedGeometry::kNarrowTile1:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kNarrowTile1>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kNarrowTile1>()) {
        uniform.template operator()<16, 3, 1, 12>();
        return true;
      }
      return false;
    case TunedGeometry::kNarrowTile2:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kNarrowTile2>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kNarrowTile2>()) {
        uniform.template operator()<16, 3, 2, 12>();
        return true;
      }
      return false;
    case TunedGeometry::kNarrowRows3Tile2:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kNarrowRows3Tile2>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kNarrowRows3Tile2>()) {
        uniform.template operator()<16, 3, 2, 1>();
        return true;
      }
      return false;
    case TunedGeometry::kNarrowRows2Tile1:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kNarrowRows2Tile1>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kNarrowRows2Tile1>()) {
        uniform.template operator()<16, 2, 1, 12>();
        return true;
      }
      return false;
    case TunedGeometry::kRows4Tile1:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kRows4Tile1>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kRows4Tile1>()) {
        uniform.template operator()<16, 4, 1, 4>();
        return true;
      }
      return false;
    case TunedGeometry::kRows4Tile2:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kRows4Tile2>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kRows4Tile2>()) {
        uniform.template operator()<16, 4, 2, 4>();
        return true;
      }
      return false;
    case TunedGeometry::kNarrowRows4:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kNarrowRows4>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kNarrowRows4>()) {
        uniform.template operator()<8, 4, 1, 4>();
        return true;
      }
      return false;
    case TunedGeometry::kTokenMajorRows4:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kTokenMajorRows4>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kTokenMajorRows4>()) {
        uniform.template operator()<8, 4, 1, 4, 1>();
        return true;
      }
      return false;
    case TunedGeometry::kWaves32Rows4:
      if constexpr (TunedTablePicks<Type, Batch,
                                    TunedGeometry::kWaves32Rows4>() ||
                    TunedTablePicks<Type, kBatchInvariantWidth,
                                    TunedGeometry::kWaves32Rows4>()) {
        uniform.template operator()<32, 4, 1, 1>();
        return true;
      }
      return false;
    case TunedGeometry::kNarrowGroups4:
      // Half the width per launch, two token groups; the grid carries both.
      if constexpr (Batch == 8 &&
                    TunedTablePicks<Type, Batch,
                                    TunedGeometry::kNarrowGroups4>()) {
        SmallBatchKQuantExactFp32GEMMKernel<16, 4, 3, Type, 1, 12, 2, true>
            <<<((m + 47) / 48) * 2, 512, 0, stream>>>(w, x, y, m, k);
        return true;
      }
      // Two token groups of 4: structurally a width-8 geometry. Nothing to pin
      // it to at widths 1-7, so batch-invariant mode declines the cell at every
      // width, width 8 included (CAP, see BATCH_INVARIANT notes).
      return false;
    case TunedGeometry::kSplit2Plus3:
      if constexpr (Batch == 5 &&
                    TunedTablePicks<Type, Batch,
                                    TunedGeometry::kSplit2Plus3>()) {
        const auto split = [&]<std::size_t Width>(std::size_t offset) {
          SmallBatchKQuantExactFp32GEMMKernel<16, Width, 4, Type, 2, 4, 1, true>
              <<<(m + 63) / 64, 512, 0, stream>>>(w, x + offset * k,
                                                  y + offset * m, m, k);
        };
        split.template operator()<2>(0);
        split.template operator()<3>(2);
        return true;
      }
      // 2+3 split: structurally a width-5 geometry, same CAP as kNarrowGroups4.
      return false;
  }
  return false;
}

// Only shapes the table names leave the production route, so what is routed
// away is exactly what was timed. Every tuned geometry uses 32-bit offsets, so
// a shape that does not fit them stays on production even if the table names
// it.
[[nodiscard]] static bool TryLaunchKQuantSmallBatchTuned(
    core::GgmlType type, const void* w, const float* x, float* y,
    std::size_t batch, std::size_t m, std::size_t k, hipStream_t stream) {
  // Batch-invariant mode resolves the tuned entry at width 8; the geometry is
  // then the same for every width, so a row's reduction order is too. The
  // table is sparse per width, so the instantiation guard has to admit the
  // width-8 names as well or widths with no table of their own would fall
  // through to another family.
  const auto* entry = FindTunedEntry(
      type, static_cast<std::uint32_t>(BatchInvariantSelect(batch)), m, k);
  if (entry == nullptr) return false;
  // The bound is monotone in the width, so resolving it at the selection width
  // is the conservative choice and keeps width 1 (which fails batch >= 2) on
  // the same route as the rest.
  if (!FitsSmallBatch32BitIndices(BatchInvariantSelect(batch), m, k,
                                  quant::QuantizedRowBytes(type, k)))
    return false;
  // CAP: these two geometries ARE a width (4+4 token groups, 2+3 split). There
  // is no way to run them at another width, so batch-invariant mode declines
  // the cell at every width -- width 8 included, or width 8 would keep a kernel
  // no other width can match. Those cells leave the tuned route and take the
  // next family; that is a speed change at width 8, never a wrong result.
  if (GufoDevBatchInvariant() &&
      (entry->geometry == TunedGeometry::kNarrowGroups4 ||
       entry->geometry == TunedGeometry::kSplit2Plus3))
    return false;
  const auto by_batch = [&]<core::GgmlType Type>() {
    const auto launch = [&]<std::size_t Batch>() {
      if constexpr (TunedTableNames<Type, Batch>() ||
                    TunedTableNames<Type, kBatchInvariantWidth>()) {
        return LaunchTunedGeometry<Type, Batch>(entry->geometry, w, x, y, m, k,
                                                stream);
      } else {
        return false;
      }
    };
    switch (batch) {
      case 1:
        return launch.template operator()<1>();
      case 2:
        return launch.template operator()<2>();
      case 3:
        return launch.template operator()<3>();
      case 4:
        return launch.template operator()<4>();
      case 5:
        return launch.template operator()<5>();
      case 6:
        return launch.template operator()<6>();
      case 7:
        return launch.template operator()<7>();
      case 8:
        return launch.template operator()<8>();
      default:
        return false;
    }
  };
  switch (type) {
    case core::GgmlType::kQ4_K:
      return by_batch.template operator()<core::GgmlType::kQ4_K>();
    case core::GgmlType::kQ5_K:
      return by_batch.template operator()<core::GgmlType::kQ5_K>();
    case core::GgmlType::kQ6_K:
      return by_batch.template operator()<core::GgmlType::kQ6_K>();
    case core::GgmlType::kQ3_K:
      return by_batch.template operator()<core::GgmlType::kQ3_K>();
    case core::GgmlType::kIQ4_NL:
      return by_batch.template operator()<core::GgmlType::kIQ4_NL>();
    case core::GgmlType::kIQ4_XS:
      return by_batch.template operator()<core::GgmlType::kIQ4_XS>();
    case core::GgmlType::kIQ3_S:
      return by_batch.template operator()<core::GgmlType::kIQ3_S>();
    case core::GgmlType::kIQ3_XXS:
      return by_batch.template operator()<core::GgmlType::kIQ3_XXS>();
    case core::GgmlType::kIQ2_S:
      return by_batch.template operator()<core::GgmlType::kIQ2_S>();
    case core::GgmlType::kIQ2_XS:
      return by_batch.template operator()<core::GgmlType::kIQ2_XS>();
    case core::GgmlType::kQ2_K:
      return by_batch.template operator()<core::GgmlType::kQ2_K>();
    default:
      // IQ2_XXS is measured only (the models under test do not use it),
      // so it is deliberately absent: no instantiation, no routing.
      return false;
  }
}

template<core::GgmlType Type>
[[nodiscard]] static bool TryLaunchKQuantSmallBatchIqFast(
    const void* w, const float* x, float* y, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream) {
  const auto launch = [&]<std::size_t Batch>() {
    SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 3, Type, 1, 12, 1, false,
                                        Batch, 32, 32, false, false, true>
        <<<((m + 47) / 48), 512, 0, stream>>>(w, x, y, m, k);
    return true;
  };
  switch (batch) {
    case 1: return launch.template operator()<1>();
    case 2: return launch.template operator()<2>();
    case 3: return launch.template operator()<3>();
    case 4: return launch.template operator()<4>();
    case 5: return launch.template operator()<5>();
    case 6: return launch.template operator()<6>();
    case 7: return launch.template operator()<7>();
    case 8: return launch.template operator()<8>();
    default: return false;
  }
}

// Dev switch GUFO_DEV_DOT4 (read once): int8-activation verify GEMM
// (small_batch_dot4.hpp) on the cells where a chained bench measured it faster than
// every exact geometry. NOT bit-exact (rel err ~0.35%). DEFAULT ON: served A/B
// code +10-15% / JSON t0.6 +10%, prose t0 +1.1% (prose holds). =0 turns it off.
[[nodiscard]] static bool GufoDevDot4() noexcept {
  static const bool enabled = [] {
    const char* v = std::getenv("GUFO_DEV_DOT4");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

// Per-stream activation scratch, sized for the largest call seen. Streams can
// overlap, so one shared buffer would race.
// ponytail: grows, never frees (<= 8 * 17408 * 1.5 B per stream).
struct Dot4Scratch {
  std::int8_t* q = nullptr;
  float2* ds = nullptr;
  std::size_t elems = 0;
};

[[nodiscard]] static Dot4Scratch* GetDot4Scratch(hipStream_t stream,
                                                 std::size_t elems) {
  static std::mutex mutex;
  static std::unordered_map<hipStream_t, Dot4Scratch> by_stream;
  const std::lock_guard<std::mutex> lock(mutex);
  Dot4Scratch& s = by_stream[stream];
  if (s.elems >= elems) return &s;
  // Growth/free/synchronization is forbidden inside a graph capture, and so is
  // a silent fallback to another kernel: refuse loudly. Reserve first
  // (ReserveDot4Scratch); a failed reserve leaves the old allocation usable.
  hipStreamCaptureStatus capture;
  if (hipStreamIsCapturing(stream, &capture) != hipSuccess ||
      capture != hipStreamCaptureStatusNone)
    throw std::runtime_error(
        "Dot4 scratch growth during graph capture; reserve before capture");
  if (s.q && hipStreamSynchronize(stream) != hipSuccess)
    return nullptr;
  Dot4Scratch next;
  if (hipMalloc(&next.q, elems) != hipSuccess)
    return nullptr;
  if (hipMalloc(&next.ds, (elems / 16) * sizeof(float2)) != hipSuccess) {
    (void)hipFree(next.q);
    return nullptr;
  }
  // The old buffers are leaked on purpose: a graph captured earlier may still
  // reference them, so freeing them here would dangle on its next replay.
  s = {next.q, next.ds, elems};
  return &s;
}

void ReserveDot4Scratch(std::size_t elems, hipStream_t stream) {
  if (!GetDot4Scratch(stream, elems))
    throw std::runtime_error("cannot reserve Dot4 scratch before capture");
}

// ---- NormQ (GUFO_DEV_NORMQ=1; added 2026-10-25) ----
// 203 of the ~470 activation-quantize launches of a served verify round read a
// norm output at k=5120: 0.710 ms of kernel plus 0.221 ms of launch gap in a
// 123 ms round (measured). The fused norm in prefill_norm.hip writes
// what they would have produced; this is its per-stream buffer pair and mark.
// See ops/gemm.hpp for why the Dot4 scratch cannot serve.
namespace {

struct NormQState {
  NormQTarget target;
  std::size_t elems = 0;
  // Mark: the fused norm wrote exactly these [batch][k] activations into
  // `target`, in both kinds, and nothing has written `x` since.
  const float* x = nullptr;
  std::size_t batch = 0;
  std::size_t k = 0;
};

std::mutex& NormQMutex() {
  static std::mutex mutex;
  return mutex;
}

// Caller holds NormQMutex().
NormQState& NormQStateFor(hipStream_t stream) {
  static std::unordered_map<hipStream_t, NormQState> by_stream;
  return by_stream[stream];
}

}  // namespace

bool GufoDevNormQ() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_NORMQ");
    const bool on = value != nullptr && value[0] == '1' && value[1] == '\0';
    if (on)
      std::fprintf(stderr, "GUFO_DEV_NORMQ on\n");
    return on;
  }();
  return enabled;
}

bool NormQAcquire(hipStream_t stream, std::size_t elems, NormQTarget* target) {
  const std::lock_guard<std::mutex> lock(NormQMutex());
  NormQState& s = NormQStateFor(stream);
  if (s.elems >= elems) {
    *target = s.target;
    return true;
  }
  // Growth is forbidden inside a capture; refuse so the caller runs the
  // unfused norm (correct, just unfused) instead of throwing mid-round.
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(stream, &capture) != hipSuccess ||
      capture != hipStreamCaptureStatusNone) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr,
                   "GUFO_DEV_NORMQ: scratch not reserved before capture; "
                   "norm+quantize stays unfused\n");
    }
    return false;
  }
  NormQTarget next;
  bool ok = true;
  for (int kind = 0; kind < 2 && ok; ++kind) {
    ok = hipMalloc(&next.q[kind], elems) == hipSuccess &&
         hipMalloc(&next.ds[kind], (elems / 16) * sizeof(float2)) == hipSuccess;
  }
  if (!ok) {
    for (int kind = 0; kind < 2; ++kind) {
      if (next.q[kind]) (void)hipFree(next.q[kind]);
      if (next.ds[kind]) (void)hipFree(next.ds[kind]);
    }
    return false;
  }
  // The old buffers leak on purpose, exactly as Dot4Scratch's do: a graph
  // captured earlier may still reference them.
  s.target = next;
  s.elems = elems;
  s.x = nullptr;  // the mark described the old buffers
  *target = next;
  return true;
}

void ReserveNormQScratch(std::size_t elems, hipStream_t stream) {
  if (!GufoDevNormQ())
    return;
  NormQTarget target;
  if (!NormQAcquire(stream, elems, &target))
    throw std::runtime_error("cannot reserve NormQ scratch before capture");
}

void NormQPublish(hipStream_t stream, const float* x, std::size_t batch,
                  std::size_t k) {
  const std::lock_guard<std::mutex> lock(NormQMutex());
  NormQState& s = NormQStateFor(stream);
  s.x = x;
  s.batch = batch;
  s.k = k;
}

void NormQClear(hipStream_t stream, const float* x) {
  if (!GufoDevNormQ())
    return;
  const std::lock_guard<std::mutex> lock(NormQMutex());
  NormQState& s = NormQStateFor(stream);
  if (s.x == x)
    s.x = nullptr;
}

bool NormQLookup(hipStream_t stream, const float* x, std::size_t batch,
                 std::size_t k, bool act_k, std::int8_t** q, float2** ds) {
  if (!GufoDevNormQ())
    return false;
  const std::lock_guard<std::mutex> lock(NormQMutex());
  const NormQState& s = NormQStateFor(stream);
  if (s.x == nullptr || s.x != x || s.batch != batch || s.k != k)
    return false;
  const int kind = act_k ? 1 : 0;
  if (s.target.q[kind] == nullptr)
    return false;
  *q = s.target.q[kind];
  *ds = s.target.ds[kind];
  return true;
}

void LaunchSnIq3Pair(const models::QwenTensorRef& gate,
                     const models::QwenTensorRef& up, const float* input,
                     float* gate_output, float* up_output, float* packed_output,
                     std::size_t batch, hipStream_t stream) {
  if (batch < 1 || batch > 8 || !models::IsSnIq3VerifyPair(gate, up) ||
      !input || (!packed_output && (!gate_output || !up_output ||
                                   gate_output == up_output)))
    throw std::invalid_argument("invalid tagged IQ3 verify pair/output");
  auto* const s = GetDot4Scratch(stream, 8U * 5120U);
  if (!s)
    throw std::runtime_error("IQ3 ActK scratch unavailable; reserve before capture");
  const auto subs = static_cast<std::uint32_t>(batch * 5120U / 16U);
  // NormQ: the ffn norm already wrote these ActK activations.
  std::int8_t* xq = s->q;
  float2* xds = s->ds;
  if (!NormQLookup(stream, input, batch, 5120U, true, &xq, &xds))
    QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
        input, xq, xds, subs);
  const auto launch = [&]<std::size_t Batch>() {
    if (packed_output) {
      auto pair = gate;
      pair.num_elements += up.num_elements;
      LaunchSnIq3Prepared<Batch>(pair, xq, xds, packed_output, stream);
    } else {
      LaunchSnIq3Prepared<Batch>(gate, xq, xds, gate_output, stream);
      LaunchSnIq3Prepared<Batch>(up, xq, xds, up_output, stream);
    }
  };
  switch (batch) {
    case 1: launch.template operator()<1>(); break;
    case 2: launch.template operator()<2>(); break;
    case 3: launch.template operator()<3>(); break;
    case 4: launch.template operator()<4>(); break;
    case 5: launch.template operator()<5>(); break;
    case 6: launch.template operator()<6>(); break;
    case 7: launch.template operator()<7>(); break;
    case 8: launch.template operator()<8>(); break;
  }
}

// One thread per native u16, a straight copy: consecutive lanes write
// consecutive bytes (coalesced stores); the gather reads 16/64-byte field runs
// of the same 880-byte S8 record or XXS plane rows, so a group's neighbours hit
// L2. Address math is Iq3ByteOffset itself, tested by the CPU oracle.
template<models::QwenWeightLayout Layout>
__global__ void UnpackIq3ViewKernel(const std::uint16_t* __restrict__ packed,
                                    std::uint16_t* __restrict__ native,
                                    std::size_t full_rows, std::uint32_t blocks,
                                    std::size_t row_begin,
                                    std::uint32_t units) {
  const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u < units)
    native[u] = packed[models::Iq3UnpackSourceOffset<Layout>(
                           full_rows, blocks, row_begin, 2 * u) /
                       2];
}

void LaunchUnpackIq3View(const models::QwenTensorRef& view, void* native,
                         std::size_t capacity, hipStream_t stream) {
  models::ValidateIq3Unpack(view, native, capacity);
  const auto bytes = view.EncodedSizeBytes();
  if (bytes > UINT32_MAX || reinterpret_cast<std::uintptr_t>(native) % 2)
    throw std::invalid_argument("IQ3 unpack exceeds 32-bit index/alignment");
  const auto units = static_cast<std::uint32_t>(bytes / 2);
  const auto launch = [&]<models::QwenWeightLayout Layout>() {
    UnpackIq3ViewKernel<Layout><<<(units + 255U) / 256U, 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(view.data),
        static_cast<std::uint16_t*>(native), view.layout_rows,
        static_cast<std::uint32_t>(view.layout_columns / 256),
        view.layout_row_begin, units);
  };
  if (view.layout == models::QwenWeightLayout::kIq3SRows8)
    launch.template operator()<models::QwenWeightLayout::kIq3SRows8>();
  else
    launch.template operator()<models::QwenWeightLayout::kIq3XxsRows4>();
  HIP_CHECK(hipGetLastError());
}

// Chained, GPU clock pinned at 1700 MHz: Q6_K cells where a launchable dot4
// geometry beat the BEST exact geometry (or, at B8, the current r4w8s2) by
// >= 3%. 0 = dot4_r2w8, 1 = dot4s_r4w8s2, 2 = dot4s_r2w8. v3 cells (B3-6 on
// the k-wide shapes) live in FindKQuantV3Cell and are tried first. -1 = none.
// (k95small: moved out of TryLaunchDot4SmallBatch so the fused K/V route can
// check that V's served geometry is the one it reproduces.)
[[nodiscard]] static int FindQ6Dot4Geometry(std::size_t m, std::size_t k,
                                            std::size_t batch) {
  struct Q6Cell { std::uint32_t m, k, batch, geometry; };
  static constexpr Q6Cell kQ6Cells[] = {
    {1024, 5120, 4, 0}, {1024, 5120, 5, 0}, {1024, 5120, 6, 1},
    {1024, 5120, 7, 2}, {1024, 5120, 8, 2},
    {5120, 6144, 6, 1}, {5120, 6144, 7, 0}, {5120, 6144, 8, 1},
    {5120, 17408, 7, 0},
    {10240, 5120, 4, 0}, {10240, 5120, 5, 0}, {10240, 5120, 6, 1},
    {10240, 5120, 7, 0},
    {17408, 5120, 4, 0}, {17408, 5120, 5, 0}, {17408, 5120, 6, 1},
    {17408, 5120, 7, 0},
    {248320, 5120, 4, 0}, {248320, 5120, 5, 0}, {248320, 5120, 6, 0},
    {248320, 5120, 7, 0},
  };
  int geometry = -1;
  for (const auto& c : kQ6Cells)
    if (c.m == m && c.k == k && c.batch == batch)
      geometry = static_cast<int>(c.geometry);
  return geometry;
}

// Chained ABBA dflash_gemm_bench: v2 dot4s_r2w8 wins B8 for Q4_K,
// Q5_K, IQ4_XS, IQ2_S; dot4s_r4w8s2 for Q6_K, IQ3_S, IQ3_XXS. At B4-7 only the
// IQ formats and one Q4_K shape win. Losing cells (Q5_K at k=6144 B8,
// IQ4_XS down B4 until a re-bench: dot4 now 1.10x there) stay on the exact path. Q6_K: per-cell table kQ6Cells
// below (GPU clock pinned at 1700 MHz) overrides this rule, incl. 5120x6144 B6-8.
[[nodiscard]] static bool TryLaunchDot4SmallBatch(core::GgmlType type,
                                                  const void* w, const float* x,
                                                  float* y, std::size_t batch,
                                                  std::size_t m, std::size_t k,
                                                  hipStream_t stream) {
  using T = core::GgmlType;
  // Batch-invariant mode resolves acceptance and geometry at width 8, so a
  // width below the usual 4-row floor takes the same family instead of falling
  // through to a different one.
  const std::size_t sel = BatchInvariantSelect(batch);
  if (batch < 1 || batch > 8 || sel < 4 || k % 256 != 0 || m == 0) return false;
  const bool iq = type == T::kIQ3_S || type == T::kIQ3_XXS || type == T::kIQ2_S;
  const bool k6144 = m == 5120 && k == 6144;
  // k95small extracted the Q6_K dot4 table into FindQ6Dot4Geometry (it
  // needs the same lookup); binv pins the lookup width, so this passes `sel`,
  // not `batch`. Default mode sel == batch, so the pick is unchanged.
  const int q6_geometry =
      type == T::kQ6_K ? FindQ6Dot4Geometry(m, k, sel) : -1;
  const bool use = q6_geometry >= 0 ||
      (sel == 8
           ? iq || type == T::kQ4_K || type == T::kIQ4_XS ||
                 ((type == T::kQ5_K || type == T::kQ6_K) && !k6144)
           : iq || type == T::kIQ4_XS ||
                 (type == T::kQ4_K && k == 5120 && m == 6144));
  if (!use) return false;
  // Pinned geometry for widths 1-7 = what width 8 picks below.
  const bool pin8 = sel != batch;
  Dot4Scratch* s = GetDot4Scratch(stream, batch * k);
  if (s == nullptr) return false;
  const auto mm = static_cast<std::uint32_t>(m);
  const auto kk = static_cast<std::uint32_t>(k);
  const auto launch = [&]<T Type, std::size_t Batch>() {
    constexpr bool kFast = Type == T::kIQ2_S || Type == T::kIQ3_S ||
                           Type == T::kIQ3_XXS;
    constexpr bool kWide = Type == T::kQ6_K || Type == T::kIQ3_S ||
                           Type == T::kIQ3_XXS;
    if constexpr (Type == T::kQ6_K) {
      if (q6_geometry == 0) {
        LaunchSmallBatchDot4<Type, Batch, 2, 8, kFast>(w, x, y, s->q, s->ds,
                                                       mm, kk, stream);
        return true;
      }
      if (q6_geometry == 2) {
        LaunchSmallBatchDot4Lds<Type, Batch, 2, 8, 1, kFast>(
            w, x, y, s->q, s->ds, mm, kk, stream);
        return true;
      }
      if (q6_geometry == 1) {
        LaunchSmallBatchDot4Lds<Type, Batch, 4, 8, 2, kFast>(
            w, x, y, s->q, s->ds, mm, kk, stream);
        return true;
      }
    }
    if constexpr (kWide) {
      if (pin8 || Batch >= 7) {
        LaunchSmallBatchDot4Lds<Type, Batch, 4, 8, 2, kFast>(
            w, x, y, s->q, s->ds, mm, kk, stream);
        return true;
      }
    } else if (pin8) {
      LaunchSmallBatchDot4Lds<Type, Batch, 2, 8, 1, kFast>(w, x, y, s->q, s->ds,
                                                           mm, kk, stream);
      return true;
    }
    if constexpr (Batch == 8)
      LaunchSmallBatchDot4Lds<Type, Batch, 2, 8, 1, kFast>(w, x, y, s->q, s->ds,
                                                           mm, kk, stream);
    else
      LaunchSmallBatchDot4<Type, Batch, 2, 8, kFast>(w, x, y, s->q, s->ds, mm,
                                                     kk, stream);
    return true;
  };
  const auto by_batch = [&]<T Type>() {
    switch (batch) {
      case 1: return launch.template operator()<Type, 1>();
      case 2: return launch.template operator()<Type, 2>();
      case 3: return launch.template operator()<Type, 3>();
      case 4: return launch.template operator()<Type, 4>();
      case 5: return launch.template operator()<Type, 5>();
      case 6: return launch.template operator()<Type, 6>();
      case 7: return launch.template operator()<Type, 7>();
      default: return launch.template operator()<Type, 8>();
    }
  };
  switch (type) {
    case T::kQ4_K: return by_batch.template operator()<T::kQ4_K>();
    case T::kQ5_K: return by_batch.template operator()<T::kQ5_K>();
    case T::kQ6_K: return by_batch.template operator()<T::kQ6_K>();
    case T::kIQ4_XS: return by_batch.template operator()<T::kIQ4_XS>();
    case T::kIQ3_S: return by_batch.template operator()<T::kIQ3_S>();
    case T::kIQ3_XXS: return by_batch.template operator()<T::kIQ3_XXS>();
    case T::kIQ2_S: return by_batch.template operator()<T::kIQ2_S>();
    default: return false;
  }
}

// GUFO_DEV_KQUANT_V3=1 (read once, default OFF): Q4_K/Q5_K verify GEMM on the
// v3 half-super-block kernel (small_batch_dot4.hpp) for the cells in the table
// below only; every other cell keeps the exact kernel. int8 activations like
// dot4 (rel err ~0.003-0.005, not bit-exact). Table = microbench (pinned
// 2000 MHz, 27B shapes, widths 1-8): a cell is in it when its best v3 geometry
// beat the best exact geometry by >= 3%. Regenerate:
//   python3 tools/qwen27b/v3_table_from_bench.py <bench_dir>
[[nodiscard]] static bool GufoDevKQuantV3() noexcept {
  static const bool enabled = [] {
    const char* v = std::getenv("GUFO_DEV_KQUANT_V3");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return enabled;
}

// GUFO_DEV_HEADQ6=<RowGroup> (read once, default 0 = OFF): the value IS the
// RowGroup the Q6_K output-head cell runs with, 1 or 2; anything else is OFF.
// Head only ({Q6_K, 248320, 5120, 7|8, 4, 8, 4, 1} = r4w8x4k ActK, the drafter
// and verify logits passes, 1,042,944,000 B read twice per round). RowGroup
// changes how many of a lane's 4 rows are live at once, never any arithmetic:
// each (row, token) keeps its own accumulator and the same s order, so the
// result is bitwise identical at any RowGroup (Q6KHalfBlockDot).
// Measured compile-time effect (isa_arms, gfx1151 wave32):
//   B7/B8  RowGroup 0 (served) 252/256 VGPR, 0/72 B scratch, occ 5/5
//          RowGroup 2          174/170 VGPR, 0 B scratch,    occ 8/8
//          RowGroup 1          116/122 VGPR, 0 B scratch,    occ 10/8
[[nodiscard]] static std::uint32_t GufoDevHeadQ6() noexcept {
  static const std::uint32_t group = [] {
    const char* v = std::getenv("GUFO_DEV_HEADQ6");
    if (v == nullptr || v[1] != '\0') return 0U;
    return v[0] == '1' ? 1U : v[0] == '2' ? 2U : 0U;
  }();
  return group;
}

namespace detail {
// small_batch_dot4_v3_wave64.hip (-mwavefrontsize64). false = no
// wave64 instantiation for this (type, batch, geometry).
bool TryLaunchDot4V3Wave64(core::GgmlType type, std::size_t batch,
                           std::uint32_t rows, std::uint32_t waves,
                           std::uint32_t per_lane, bool act_k, const void* w,
                           const float* x, float* y, std::int8_t* xq,
                           float2* xds, std::uint32_t m, std::uint32_t k,
                           hipStream_t stream);
}  // namespace detail

struct KQuantV3Cell {
  core::GgmlType type;
  std::uint32_t m, k, batch, rows_per_wave, waves;
  std::uint32_t rows_per_lane = 1;  // 1, 2 or 4 = RowsPerLane (2: K types + IQ4_NL, IQ3 at B>=4; 4: K types + IQ3 at B>=6)
  std::uint32_t act_k = 0;  // 1 = ActK: q8_K activations (Q4_K/Q5_K/IQ4_XS)
  std::uint32_t wave64 = 0;  // 1 = the same V3 kernel built -mwavefrontsize64 (IQ3_S/IQ3_XXS/IQ4_XS)
};

// ponytail: linear scan of ~60 rows per verify GEMM (host side, ~0.1 us);
// a hash map if the table grows past a few hundred cells.
[[nodiscard]] static const KQuantV3Cell* FindKQuantV3Cell(
    core::GgmlType type, std::size_t batch, std::size_t m, std::size_t k) {
  using T = core::GgmlType;
  static constexpr KQuantV3Cell kTable[] = {
    // GPU clock pinned at 1700 MHz: Q4_K/Q5_K, 20 target + DFlash2 drafter shapes x B1-8.
    // v3 (x1, x2 = RowsPerLane 2, 8th field 1 = ActK q8_K activations) must beat the best of EVERY
    // non-v3 geometry by >= 3%. ActK rel_error med 3.8e-3 vs v3 3.4e-3 (both int8 activations), same
    // top-1 floor. Replaces the earlier Q4_K/Q5_K rows (all shapes re-benched).
    {T::kQ4_K, 1024, 5120, 1, 4, 2},  // 1.28x vs typed
    {T::kQ4_K, 1024, 5120, 2, 4, 4, 1, 1},  // 1.17x vs rows4_tile2
    {T::kQ4_K, 1024, 5120, 3, 4, 2},  // 1.21x vs rows4_tile2
    {T::kQ4_K, 1024, 5120, 5, 4, 4, 2},  // 1.16x vs dot4_r2w8
    {T::kQ4_K, 1024, 5120, 7, 4, 4, 2},  // 1.09x vs dot4_r4w4
    {T::kQ4_K, 1280, 5120, 1, 4, 2},  // 1.25x vs typed
    {T::kQ4_K, 1280, 5120, 2, 4, 4, 1, 1},  // 1.19x vs rows4_tile2
    {T::kQ4_K, 1280, 5120, 3, 4, 2},  // 1.09x vs rows4_tile2
    {T::kQ4_K, 1280, 5120, 7, 4, 4},  // 1.08x vs dot4_r2w8
    {T::kQ4_K, 4096, 5120, 1, 2, 4, 1, 1},  // 1.37x vs typed
    {T::kQ4_K, 4096, 5120, 2, 4, 8},  // 1.23x vs narrow_tile1
    {T::kQ4_K, 4096, 5120, 3, 4, 2},  // 1.30x vs rows4_tile2
    {T::kQ4_K, 4096, 5120, 4, 8, 4, 1, 1},  // 1.10x vs dot4s_r2w8
    {T::kQ4_K, 4096, 5120, 5, 4, 8, 1, 1},  // 1.09x vs dot4_r2w8
    {T::kQ4_K, 4096, 5120, 6, 8, 4, 1, 1},  // 1.14x vs dot4_r2w8
    {T::kQ4_K, 4096, 5120, 7, 4, 8, 2},  // 1.22x vs dot4_r2w8
    {T::kQ4_K, 4096, 5120, 8, 4, 8, 1, 1},  // 1.10x vs dot4s_r2w8
    {T::kQ4_K, 5120, 4096, 1, 2, 4, 1, 1},  // 1.34x vs typed
    {T::kQ4_K, 5120, 4096, 2, 4, 4},  // 1.09x vs rows4_tile2
    {T::kQ4_K, 5120, 4096, 3, 8, 4, 2, 1},  // 1.18x vs narrow_tile1
    {T::kQ4_K, 5120, 4096, 4, 8, 4, 1, 1},  // 1.12x vs dot4_r2w8
    {T::kQ4_K, 5120, 4096, 6, 8, 4},  // 1.27x vs dot4_r2w8
    {T::kQ4_K, 5120, 4096, 7, 8, 4, 1, 1},  // 1.36x vs dot4_r4w8
    {T::kQ4_K, 5120, 4096, 8, 4, 8, 1, 1},  // 1.04x vs dot4s_r2w8
    {T::kQ4_K, 5120, 6144, 1, 2, 4},  // 1.28x vs typed
    {T::kQ4_K, 5120, 6144, 2, 2, 4},  // 1.14x vs narrow_tile1
    {T::kQ4_K, 5120, 6144, 3, 2, 4, 1, 1},  // 1.15x vs rows4_tile2
    {T::kQ4_K, 5120, 6144, 4, 4, 8, 1, 1},  // 1.25x vs dot4_r2w8
    {T::kQ4_K, 5120, 6144, 5, 8, 4},  // 1.04x vs dot4_r2w8
    {T::kQ4_K, 5120, 6144, 6, 4, 8, 2},  // 1.17x vs dot4_r2w8
    {T::kQ4_K, 5120, 6144, 7, 4, 8, 2},  // 1.38x vs dot4_r4w8
    {T::kQ4_K, 5120, 6144, 8, 4, 4, 2},  // 1.10x vs dot4s_r2w8
    {T::kQ4_K, 5120, 17408, 1, 4, 4, 1, 1},  // 1.23x vs typed
    {T::kQ4_K, 5120, 17408, 2, 4, 4, 1, 1},  // 1.16x vs rows4_tile2
    {T::kQ4_K, 5120, 17408, 3, 4, 8, 1, 1},  // 1.15x vs rows4_tile2
    {T::kQ4_K, 5120, 17408, 4, 4, 4, 2},  // 1.18x vs dot4_r2w8
    {T::kQ4_K, 5120, 17408, 5, 8, 4},  // 1.10x vs dot4_r2w8
    {T::kQ4_K, 5120, 17408, 6, 8, 4, 2},  // 1.14x vs dot4_r2w8
    {T::kQ4_K, 5120, 17408, 7, 4, 8, 2},  // 1.48x vs production_wave64
    {T::kQ4_K, 5120, 17408, 8, 4, 8, 2},  // 1.21x vs dot4s_r2w8
    {T::kQ4_K, 5120, 25600, 1, 2, 4},  // 1.16x vs typed
    {T::kQ4_K, 5120, 25600, 2, 4, 4, 1, 1},  // 1.08x vs rows4_tile2
    {T::kQ4_K, 5120, 25600, 3, 4, 2},  // 1.17x vs rows4_tile2
    {T::kQ4_K, 5120, 25600, 4, 4, 8},  // 1.22x vs dot4_r2w8
    {T::kQ4_K, 5120, 25600, 5, 8, 4, 2, 1},  // 1.17x vs dot4_r2w8
    {T::kQ4_K, 5120, 25600, 6, 8, 4, 2},  // 1.19x vs dot4_r2w8
    {T::kQ4_K, 5120, 25600, 7, 4, 8, 2, 1},  // 1.60x vs rows4_tile2
    {T::kQ4_K, 5120, 25600, 8, 4, 8, 2},  // 1.23x vs dot4s_r2w8
    {T::kQ4_K, 6144, 5120, 1, 2, 4, 1, 1},  // 1.37x vs typed
    {T::kQ4_K, 6144, 5120, 2, 2, 4},  // 1.27x vs narrow_tile2
    {T::kQ4_K, 6144, 5120, 3, 2, 4, 2, 1},  // 1.39x vs narrow_tile2
    {T::kQ4_K, 6144, 5120, 4, 4, 8, 2, 1},  // 1.14x vs dot4s_r2w8
    {T::kQ4_K, 6144, 5120, 5, 8, 4},  // 1.09x vs dot4_r2w8
    {T::kQ4_K, 6144, 5120, 6, 4, 8, 2, 1},  // warm-up round dropped: 176.6 -> 184.6 GB/s (was r8w4)
    {T::kQ4_K, 6144, 5120, 7, 8, 4, 1, 1},  // 1.30x vs dot4_r4w4
    {T::kQ4_K, 6144, 5120, 8, 4, 8, 2, 1},  // warm-up round dropped: 163.3 -> 173.1 GB/s (was r4w8x2)
    {T::kQ4_K, 10240, 5120, 1, 2, 4, 1, 1},  // 1.30x vs typed
    {T::kQ4_K, 10240, 5120, 2, 2, 4, 2, 1},  // warm-up round dropped: 215.9 -> 223.7 GB/s (was r4w4)
    {T::kQ4_K, 10240, 5120, 3, 2, 4, 2},  // warm-up round dropped: +2.9% after dropping the warm-up round, row reverted
    {T::kQ4_K, 10240, 5120, 4, 4, 4, 1, 1},  // 1.10x vs dot4s_r2w8
    {T::kQ4_K, 10240, 5120, 6, 8, 4, 2, 1},  // 1.05x vs dot4_r2w8
    {T::kQ4_K, 10240, 5120, 7, 4, 8, 2, 1},  // warm-up round dropped: 184.5 -> 192.4 GB/s (was r4w4x2)
    {T::kQ4_K, 10240, 5120, 8, 4, 4, 2, 1},  // warm-up round dropped: 186.4 -> 194.9 GB/s (was r8w4x2)
    {T::kQ4_K, 12288, 5120, 1, 4, 4},  // 1.21x vs typed
    {T::kQ4_K, 12288, 5120, 2, 2, 4, 1, 1},  // 1.07x vs rows4_tile2
    {T::kQ4_K, 12288, 5120, 3, 2, 4, 2, 1},  // warm-up round dropped: 205.7 -> 219.6 GB/s (was r2w4)
    {T::kQ4_K, 12288, 5120, 4, 4, 4, 1, 1},  // 1.10x vs dot4s_r2w8
    {T::kQ4_K, 12288, 5120, 5, 4, 4, 2, 1},  // 1.04x vs dot4_r2w8
    {T::kQ4_K, 12288, 5120, 6, 8, 4, 2, 1},  // warm-up round dropped: 194.0 -> 207.9 GB/s (was r4w4x2)
    {T::kQ4_K, 12288, 5120, 7, 4, 8, 2},  // 1.40x vs dot4_r4w4
    {T::kQ4_K, 12288, 5120, 8, 4, 8, 2},  // 1.07x vs dot4s_r2w8
    {T::kQ4_K, 17408, 5120, 1, 4, 4, 1, 1},  // 1.29x vs typed
    {T::kQ4_K, 17408, 5120, 2, 2, 4, 1, 1},  // 1.10x vs rows4_tile2
    {T::kQ4_K, 17408, 5120, 3, 2, 4, 2, 1},  // 1.17x vs rows4_tile2
    {T::kQ4_K, 17408, 5120, 4, 4, 4, 1, 1},  // 1.07x vs dot4s_r2w8
    {T::kQ4_K, 17408, 5120, 5, 4, 8, 1, 1},  // 1.05x vs dot4_r2w8
    {T::kQ4_K, 17408, 5120, 6, 4, 8, 1, 1},  // 1.08x vs dot4_r2w8
    {T::kQ4_K, 17408, 5120, 7, 4, 4, 2},  // 1.35x vs dot4_r2w8
    {T::kQ4_K, 17408, 5120, 8, 4, 4, 2},  // 1.07x vs dot4s_r2w8
    {T::kQ4_K, 34816, 5120, 1, 2, 4, 1, 1},  // 1.28x vs typed
    {T::kQ4_K, 34816, 5120, 2, 2, 4, 2},  // 1.06x vs narrow_tile1
    {T::kQ4_K, 34816, 5120, 3, 4, 4, 1, 1},  // 1.19x vs narrow_tile1
    {T::kQ4_K, 34816, 5120, 4, 4, 4, 1, 1},  // 1.07x vs dot4s_r2w8
    {T::kQ4_K, 34816, 5120, 5, 4, 8},  // 1.08x vs dot4_r2w8
    {T::kQ4_K, 34816, 5120, 6, 4, 8, 1, 1},  // 1.16x vs dot4_r2w8
    {T::kQ4_K, 34816, 5120, 7, 4, 8, 2},  // 1.45x vs dot4_r4w4
    {T::kQ4_K, 34816, 5120, 8, 4, 4, 2},  // 1.13x vs dot4s_r2w8
    {T::kQ5_K, 1024, 5120, 1, 4, 4, 1, 1},  // 1.28x vs typed
    {T::kQ5_K, 1024, 5120, 2, 4, 8},  // 1.10x vs rows4_tile2
    {T::kQ5_K, 1024, 5120, 3, 2, 4},  // warm-up round dropped: 105.1 -> 112.7 GB/s (was r4w8)
    {T::kQ5_K, 1024, 5120, 5, 4, 4},  // warm-up round dropped: 90.6 -> 94.0 GB/s (was r4w4k)
    {T::kQ5_K, 1024, 5120, 6, 4, 4, 2},  // 1.11x vs dot4_r2w8
    {T::kQ5_K, 1024, 5120, 8, 4, 4, 2},  // 1.10x vs dot4s_r2w8
    {T::kQ5_K, 5120, 6144, 1, 2, 4, 1, 1},  // 1.12x vs typed
    {T::kQ5_K, 5120, 6144, 2, 4, 8, 2, 1},  // 1.05x vs narrow_tile2
    {T::kQ5_K, 5120, 6144, 4, 4, 8},  // 1.17x vs narrow_tile1
    {T::kQ5_K, 5120, 6144, 5, 4, 8, 2},  // 1.12x vs rows4_tile2
    {T::kQ5_K, 5120, 6144, 6, 8, 4, 2, 1},  // 1.23x vs rows4_tile2
    {T::kQ5_K, 5120, 6144, 7, 4, 8, 2, 1},  // warm-up round dropped: 176.2 -> 183.9 GB/s (was r4w8x2)
    {T::kQ5_K, 5120, 6144, 8, 8, 4, 2, 1},  // warm-up round dropped: 170.0 -> 184.2 GB/s (was r8w4x2)
    {T::kQ5_K, 5120, 17408, 1, 8, 4, 1, 1},  // 1.11x vs typed
    {T::kQ5_K, 5120, 17408, 3, 8, 4, 1, 1},  // 1.06x vs rows4_tile2
    {T::kQ5_K, 5120, 17408, 4, 8, 4},  // 1.09x vs production_wave64
    {T::kQ5_K, 5120, 17408, 5, 8, 4, 2},  // 1.15x vs dot4s_r4w8s2
    {T::kQ5_K, 5120, 17408, 6, 8, 4, 2, 1},  // 1.23x vs production_wave64
    {T::kQ5_K, 5120, 17408, 7, 4, 8, 2, 1},  // 1.15x vs dot4s_r2w8
    {T::kQ5_K, 5120, 17408, 8, 4, 8, 2},  // 1.17x vs dot4s_r2w8
    {T::kQ5_K, 6144, 5120, 1, 4, 4, 1, 1},  // 1.19x vs typed
    {T::kQ5_K, 6144, 5120, 2, 4, 4, 2, 1},  // 1.10x vs narrow_tile2
    {T::kQ5_K, 6144, 5120, 3, 4, 4, 2, 1},  // 1.24x vs rows4_tile2
    {T::kQ5_K, 6144, 5120, 4, 4, 8, 2},  // 1.11x vs dot4s_r2w8
    {T::kQ5_K, 6144, 5120, 5, 8, 4, 1, 1},  // 1.25x vs dot4s_r4w8
    {T::kQ5_K, 6144, 5120, 6, 8, 4, 1, 1},  // 1.22x vs dot4s_r4w8s2
    {T::kQ5_K, 6144, 5120, 7, 4, 8, 1, 1},  // 1.14x vs dot4s_r2w8
    {T::kQ5_K, 6144, 5120, 8, 8, 4, 2},  // 1.09x vs dot4s_r2w8
    {T::kQ5_K, 10240, 5120, 1, 4, 4},  // 1.11x vs typed
    {T::kQ5_K, 10240, 5120, 3, 4, 4},  // 1.08x vs rows4_tile2
    {T::kQ5_K, 10240, 5120, 4, 8, 4, 1, 1},  // 1.07x vs dot4s_r2w8
    {T::kQ5_K, 10240, 5120, 5, 4, 8},  // 1.17x vs dot4s_r4w8
    {T::kQ5_K, 10240, 5120, 6, 8, 4, 2, 1},  // 1.18x vs dot4s_r4w8s2
    {T::kQ5_K, 10240, 5120, 7, 4, 4, 2},  // 1.08x vs dot4s_r2w8
    {T::kQ5_K, 10240, 5120, 8, 8, 4, 2},  // 1.08x vs dot4s_r2w8
    {T::kQ5_K, 12288, 5120, 1, 2, 4, 1, 1},  // 1.10x vs typed
    {T::kQ5_K, 12288, 5120, 2, 2, 4, 2, 1},  // 1.05x vs rows4_tile2
    {T::kQ5_K, 12288, 5120, 3, 4, 4, 1, 1},  // 1.11x vs narrow_tile1
    {T::kQ5_K, 12288, 5120, 4, 4, 8, 2},  // 1.07x vs dot4s_r2w8
    {T::kQ5_K, 12288, 5120, 5, 4, 8},  // 1.22x vs dot4s_r4w8s2
    {T::kQ5_K, 12288, 5120, 6, 4, 4, 2},  // 1.22x vs dot4s_r4w8s2
    {T::kQ5_K, 12288, 5120, 7, 4, 8, 1, 1},  // 1.12x vs dot4s_r2w8
    {T::kQ5_K, 12288, 5120, 8, 4, 4, 2},  // 1.04x vs dot4s_r2w8
    {T::kQ5_K, 17408, 5120, 1, 4, 8},  // 1.10x vs typed
    {T::kQ5_K, 17408, 5120, 2, 2, 4, 2, 1},  // 1.07x vs narrow_tile2
    {T::kQ5_K, 17408, 5120, 3, 4, 4, 1, 1},  // 1.06x vs rows4_tile2
    {T::kQ5_K, 17408, 5120, 4, 4, 4},  // 1.04x vs dot4_r2w8
    {T::kQ5_K, 17408, 5120, 5, 4, 8, 1, 1},  // 1.13x vs production_wave64
    {T::kQ5_K, 17408, 5120, 6, 4, 8, 1, 1},  // 1.20x vs dot4s_r4w8s2
    {T::kQ5_K, 17408, 5120, 7, 4, 8, 1, 1},  // 1.05x vs dot4s_r2w8
    {T::kQ5_K, 17408, 5120, 8, 4, 4, 2},  // 1.05x vs dot4s_r2w8
    {T::kQ5_K, 34816, 5120, 1, 4, 4, 1, 1},  // 1.13x vs typed
    {T::kQ5_K, 34816, 5120, 3, 4, 4},  // 1.03x vs rows4_tile2
    {T::kQ5_K, 34816, 5120, 4, 4, 4},  // 1.06x vs dot4_r2w8
    {T::kQ5_K, 34816, 5120, 5, 4, 8},  // 1.12x vs production_wave64
    {T::kQ5_K, 34816, 5120, 6, 4, 8},  // 1.17x vs dot4s_r4w8s2
    {T::kQ5_K, 34816, 5120, 7, 4, 8},  // 1.07x vs dot4s_r2w8
    {T::kQ5_K, 34816, 5120, 8, 4, 4, 2},  // 1.06x vs dot4s_r2w8
    // Pinned 1700 MHz: Q6_K v3 vs the best exact geometry.
    {T::kQ6_K, 1024, 5120, 3, 4, 2},   // warm-up round dropped: 111.4 -> 115.5 GB/s (was r2w4)
    {T::kQ6_K, 5120, 6144, 3, 2, 4},   // 1.12x
    {T::kQ6_K, 5120, 6144, 4, 4, 4, 2, 1},   // warm-up round dropped: 203.5 -> 210.7 GB/s (was r4w2)
    {T::kQ6_K, 5120, 6144, 5, 8, 4},   // 1.07x
    {T::kQ6_K, 5120, 6144, 6, 4, 8, 2, 1},   // warm-up round dropped: 192.8 -> 203.3 GB/s (new; was dot4s_r4w8s2)
    {T::kQ6_K, 5120, 6144, 7, 8, 4, 1, 1},   // warm-up round dropped: 189.2 -> 199.1 GB/s (new; was dot4_r2w8)
    {T::kQ6_K, 5120, 6144, 8, 4, 8, 2, 1},   // warm-up round dropped: 183.1 -> 188.7 GB/s (new; was dot4s_r4w8s2)
    {T::kQ6_K, 5120, 17408, 3, 4, 2},  // 1.08x
    {T::kQ6_K, 5120, 17408, 4, 8, 4},  // 1.12x
    {T::kQ6_K, 5120, 17408, 6, 8, 4},  // 1.07x
    // Re-bench, pinned 1700 MHz: Q6_K cells the dot4 arm still served got their
    // best_new v3 shape; receipt per row. Rows 7/8 field set = ActK (q8_K activations).
    {T::kQ6_K, 248320, 5120, 3, 2, 4},  // B3: dot4v3_r2w4 4.865 ms vs best exact 4.955 (+1.8%); served head
                                        // kernel 5.081 ms in a served trace (+4.4%). Drafter head at w4.
    {T::kQ6_K, 248320, 5120, 6, 4, 4, 4},  // warm-up round dropped: 211.5 -> 224.0 GB/s (new; was dot4_r2w8)
    {T::kQ6_K, 248320, 5120, 7, 4, 8, 4, 1},  // warm-up round dropped: 189.5 -> 222.1 GB/s (new; was dot4_r2w8)
    {T::kQ6_K, 248320, 5120, 8, 4, 8, 4, 1},  // warm-up round dropped: 219.4 -> 226.4 GB/s (new; was dot4s_r4w8s2)
    // Pinned 1700 MHz: IQ4_XS v3 incl. RowsPerLane=2 (7th field) vs the best
    // of EVERY other geometry (dot4 included). Replaces the earlier 16 x1 cells (B1-6);
    // x2 now wins B6-8 on most shapes (1.03-1.14x), where x1 lost.
    {T::kIQ4_XS, 5120, 6144, 1, 2, 4},  // 1.05x vs typed
    {T::kIQ4_XS, 5120, 6144, 2, 2, 4},  // 1.17x vs narrow_tile2
    {T::kIQ4_XS, 5120, 6144, 3, 4, 8},  // 1.15x vs rows4_tile2
    {T::kIQ4_XS, 5120, 6144, 4, 4, 4, 2},  // 1.07x vs dot4_r2w8
    {T::kIQ4_XS, 5120, 6144, 7, 4, 8, 2},  // 1.06x vs dot4_r2w8
    {T::kIQ4_XS, 5120, 17408, 2, 4, 4},  // 1.22x vs narrow_tile2
    {T::kIQ4_XS, 5120, 17408, 3, 4, 4, 2},  // 1.28x vs rows4_tile2
    {T::kIQ4_XS, 5120, 17408, 4, 4, 4, 2},  // 1.17x vs dot4_r2w8
    {T::kIQ4_XS, 5120, 17408, 6, 4, 8, 2},  // 1.16x vs dot4_r2w8
    {T::kIQ4_XS, 5120, 17408, 7, 4, 8, 2},  // 1.22x vs dot4_r2w8
    {T::kIQ4_XS, 5120, 17408, 5, 4, 8, 2, 0, 1},  // wave64: 203.1 GB/s vs best wave32 dot4_r4w8 193.1 (+5.2%)
    {T::kIQ4_XS, 5120, 17408, 8, 4, 8, 2},  // back to wave32 r4w8x2: kernel-only 209.0 GB/s vs served wave64 185.5; wave64 199.7 in the earlier bench was the old pick
    {T::kIQ4_XS, 6144, 5120, 2, 2, 4},  // 1.27x vs narrow_tile2
    {T::kIQ4_XS, 6144, 5120, 3, 2, 4, 2},  // 1.25x vs rows4_tile1
    {T::kIQ4_XS, 6144, 5120, 6, 4, 8, 2},  // 1.07x vs dot4_r2w8
    {T::kIQ4_XS, 6144, 5120, 7, 4, 8, 2},  // 1.08x vs dot4_r2w8
    {T::kIQ4_XS, 10240, 5120, 1, 2, 4},  // 1.04x vs typed
    {T::kIQ4_XS, 10240, 5120, 2, 2, 4},  // 1.19x vs narrow_tile2
    {T::kIQ4_XS, 10240, 5120, 3, 4, 4},  // 1.23x vs rows4_tile1
    {T::kIQ4_XS, 10240, 5120, 6, 8, 4, 2},  // 1.07x vs dot4_r2w8
    {T::kIQ4_XS, 10240, 5120, 7, 4, 4, 2},  // 1.10x vs dot4_r2w8
    {T::kIQ4_XS, 10240, 5120, 8, 8, 4, 2},  // 1.07x vs dot4s_r8w4
    {T::kIQ4_XS, 12288, 5120, 2, 2, 4},  // 1.17x vs narrow_tile2
    {T::kIQ4_XS, 12288, 5120, 3, 2, 4, 2},  // 1.23x vs rows4_tile2
    {T::kIQ4_XS, 12288, 5120, 6, 8, 4, 2},  // 1.05x vs dot4_r2w8
    {T::kIQ4_XS, 12288, 5120, 7, 8, 4, 2},  // 1.05x vs dot4_r2w8
    {T::kIQ4_XS, 34816, 5120, 1, 2, 4},  // 1.04x vs typed
    {T::kIQ4_XS, 34816, 5120, 2, 2, 4},  // 1.17x vs narrow_tile2
    {T::kIQ4_XS, 34816, 5120, 3, 4, 8},  // 1.21x vs rows4_tile1
    {T::kIQ4_XS, 34816, 5120, 6, 4, 8, 2},  // 1.03x vs dot4_r2w8
    {T::kIQ4_XS, 34816, 5120, 7, 4, 8, 2},  // 1.14x vs dot4_r2w8
    {T::kIQ4_XS, 34816, 5120, 8, 4, 8, 2},  // 1.10x vs dot4s_r8w4
    // Pinned 2000 MHz (other types here were measured at 1700): IQ4_NL v3 and Q3_K v3 vs the best of every
    // other geometry. rel_error 0.0035 = the int8-activation dot4 paths; top-1 1.000.
    {T::kIQ4_NL, 5120, 17408, 3, 4, 8},  // 1.04x vs rows4_tile2
    {T::kIQ4_NL, 5120, 17408, 4, 4, 4, 2},  // 1.09x vs rows4_tile2
    {T::kIQ4_NL, 5120, 17408, 5, 8, 4},  // 1.15x vs rows4_tile2
    {T::kIQ4_NL, 5120, 17408, 6, 8, 4},  // 1.18x vs rows4_tile2
    {T::kIQ4_NL, 5120, 17408, 7, 4, 4, 2},  // 1.31x vs narrow_rows4
    {T::kIQ4_NL, 5120, 17408, 8, 8, 4, 2},  // 1.29x vs rows4_tile2
    {T::kIQ4_NL, 10240, 5120, 3, 2, 4, 2},  // 1.08x vs narrow_rows4
    {T::kIQ4_NL, 10240, 5120, 4, 2, 4, 2},  // 1.10x vs rows4_tile2
    {T::kIQ4_NL, 10240, 5120, 5, 4, 4, 2},  // 1.22x vs rows4_tile2
    {T::kIQ4_NL, 10240, 5120, 6, 4, 4, 2},  // 1.28x vs narrow_rows4
    {T::kIQ4_NL, 10240, 5120, 7, 4, 4, 2},  // 1.43x vs token_major_rows4
    {T::kIQ4_NL, 10240, 5120, 8, 4, 8, 2},  // 1.44x vs rows4_tile2
    {T::kIQ4_NL, 17408, 5120, 1, 2, 4, 2},  // 1.12x vs generic
    {T::kIQ4_NL, 17408, 5120, 3, 4, 4},  // 1.07x vs token_major_rows4
    {T::kIQ4_NL, 17408, 5120, 4, 4, 4},  // 1.14x vs token_major_rows4
    {T::kIQ4_NL, 17408, 5120, 5, 4, 4, 2},  // 1.23x vs rows4_tile2
    {T::kIQ4_NL, 17408, 5120, 6, 4, 4, 2},  // 1.35x vs narrow_rows4
    {T::kIQ4_NL, 17408, 5120, 7, 4, 4, 2},  // 1.48x vs token_major_rows4
    {T::kIQ4_NL, 17408, 5120, 8, 8, 4, 2},  // 1.48x vs rows4_tile2
    {T::kQ3_K, 17408, 5120, 1, 2, 4},  // 1.81x vs typed
    {T::kQ3_K, 17408, 5120, 2, 4, 2},  // 1.92x vs narrow_rows4
    {T::kQ3_K, 17408, 5120, 3, 4, 4},  // 1.96x vs narrow_tile1
    {T::kQ3_K, 17408, 5120, 4, 4, 4},  // 1.43x vs dot4_r2w8
    {T::kQ3_K, 17408, 5120, 5, 4, 4},  // 1.34x vs dot4_r2w8
    {T::kQ3_K, 17408, 5120, 6, 8, 4},  // 1.22x vs dot4_r2w8
    {T::kQ3_K, 17408, 5120, 7, 4, 8},  // 1.11x vs dot4_r2w8
    {T::kQ3_K, 17408, 5120, 8, 8, 4},  // 1.03x vs dot4_r2w8
    // Pinned 1700 MHz, EF-apex IQ3 gate/up, packed m34816:
    // v3 IQ3 half-block dots without scratch vs best of every other geometry measured (incl.
    // bench-only sign-mask iq3sm_v3, so >= production).
    // Re-bench, pinned 1700 MHz: cells the dot4/dot4s arm still served got their
    // best_new v3 shape, receipt per row. These rows only fire under GUFO_DEV_IQ3_PACKED=1 (packing
    // makes m34816 reachable at batch >= 2, dev_sync.hpp:118) plus GUFO_DEV_KQUANT_V3=1.
    {T::kIQ3_S, 17408, 5120, 1, 2, 4},  // 1.09x vs iq3sm_v3_r4w4
    {T::kIQ3_S, 34816, 5120, 1, 2, 4},  // 1.04x vs iq3sm_v3_r4w4
    {T::kIQ3_S, 34816, 5120, 4, 4, 4, 2},  // 1.12x vs iq3sm_v3_r4w4x2
    {T::kIQ3_S, 34816, 5120, 5, 4, 8, 2},  // warm-up round dropped: 159.5 -> 186.3 GB/s (new; was dot4_r2w8)
    {T::kIQ3_S, 34816, 5120, 6, 4, 8, 4, 0, 1},  // wave64 r4w8x4: 180.9 vs best wave32 v3 173.1 (+4.5%); was r4w8x2
    {T::kIQ3_S, 34816, 5120, 7, 4, 8, 4},  // warm-up round dropped: 152.2 -> 158.8 GB/s (new; was dot4s_r4w8s2)
    {T::kIQ3_S, 34816, 5120, 8, 4, 8, 4, 0, 1},  // wave64: 153.9 vs wave32 150.6 (+2.2%); earlier wave32 row was 151.1
    {T::kIQ3_XXS, 17408, 5120, 1, 4, 2},  // 1.11x vs iq3sm_v3_r4w4
    {T::kIQ3_XXS, 34816, 5120, 1, 4, 2},  // 1.09x vs iq3sm_v3_r4w4
    {T::kIQ3_XXS, 34816, 5120, 7, 8, 4, 4},  // warm-up round dropped: 149.1 -> 162.8 GB/s (new; was dot4s_r4w8s2)
    {T::kIQ3_XXS, 34816, 5120, 8, 8, 4, 4},  // warm-up round dropped: 71.7 -> 154.2 GB/s (new; was dot4s_r4w8s2)
  };
  // A/B lever: GUFO_DEV_IQ4_B8_W64=1 restores the old wave64 pick for IQ4_XS 5120x17408 B8.
  static const bool kIq4B8W64 = [] { const char* v = std::getenv("GUFO_DEV_IQ4_B8_W64"); return v && v[0] == '1' && v[1] == 0; }();
  static constexpr KQuantV3Cell kIq4B8W64Cell = {T::kIQ4_XS, 5120, 17408, 8, 4, 8, 2, 0, 1};
  if (kIq4B8W64 && type == T::kIQ4_XS && batch == 8 && m == 5120 && k == 17408)
    return &kIq4B8W64Cell;
  for (const auto& cell : kTable)
    if (cell.type == type && cell.batch == batch && cell.m == m && cell.k == k)
      return &cell;
  return nullptr;
}

// quantize = false (k95small): xq/xds of this stream's scratch already hold x
// quantized by this cell's quantizer kind; the wave64 cells (other TU,
// quantize inside) refuse it.
[[nodiscard]] static bool TryLaunchKQuantV3(core::GgmlType type, const void* w,
                                            const float* x, float* y,
                                            std::size_t batch, std::size_t m,
                                            std::size_t k, hipStream_t stream,
                                            bool quantize = true) {
  // Q4_K/Q5_K v3 loads weights as aligned uint4: the row base must be 16-byte
  // aligned (rows are multiples of 144 / 176 B, so every later row is too).
  // Q6_K blocks are 210 B (2-byte aligned); its half-block loads go through
  // memcpy (unaligned global access), so no alignment check applies to it.
  // IQ4_XS blocks are 136 B; its loads go through memcpy too. The mask keeps
  // every block 8-byte aligned (rows are 136*n bytes) so the compiler's wide
  // loads of the memcpy stay naturally aligned for the header and qs.
  // IQ4_NL 18 B, IQ3_S 110, IQ3_XXS 98, IQ2_S 82 (2-byte aligned); all memcpy.
  const std::uintptr_t align_mask =
      type == core::GgmlType::kQ6_K || type == core::GgmlType::kQ3_K ? 0U
      : type == core::GgmlType::kIQ4_NL || type == core::GgmlType::kIQ3_S ||
                type == core::GgmlType::kIQ3_XXS || type == core::GgmlType::kIQ2_S
          ? 1U
      : type == core::GgmlType::kIQ4_XS ? 7U : 15U;
  if (batch < 1 || batch > 8 || k % 256 != 0 || m == 0 ||
      (reinterpret_cast<std::uintptr_t>(w) & align_mask) != 0U)
    return false;
  // Batch-invariant mode resolves the cell at width 8 so every width runs the
  // same (rows_per_wave, waves, rows_per_lane, act_k, wave64) geometry. Only
  // rows_per_wave changes a row's K-partition and reduction order, so pinning
  // the cell is what makes row r bitwise equal across widths.
  const KQuantV3Cell* cell =
      FindKQuantV3Cell(type, BatchInvariantSelect(batch), m, k);
  if (cell == nullptr) return false;
  Dot4Scratch* s = GetDot4Scratch(stream, batch * k);
  if (s == nullptr) return false;
  const auto mm = static_cast<std::uint32_t>(m);
  const auto kk = static_cast<std::uint32_t>(k);
  // k95small: the wave64 cells live in another TU and quantize inside, so they
  // cannot serve a caller that pre-quantized x into this stream's scratch.
  if (cell->wave64 != 0U && !quantize) return false;
  // NormQ: a fused norm already wrote x in this cell's quantizer kind, into
  // its own buffers. Applied whatever `quantize` says, so that a k95small
  // caller that skipped its own quantize for the same reason reads the same
  // bytes; the wave64 cells quantize inside and are excluded above.
  std::int8_t* xq = s->q;
  float2* xds = s->ds;
  if (cell->wave64 == 0U &&
      NormQLookup(stream, x, batch, k, cell->act_k != 0U, &xq, &xds))
    quantize = false;
  // binv: RowsPerLane 2 and 4 are only instantiated from Batch 4 / 6 up. A
  // row's K-partition and reduction order do not depend on RowsPerLane at all
  // (it only gives one lane more output rows, each with its own accumulator),
  // so a pinned geometry whose RowsPerLane the real width cannot instantiate
  // drops to 1 and stays bitwise equal. Default mode never reaches this: every
  // table row's own width already satisfies its RowsPerLane.
  std::uint32_t per_lane = cell->rows_per_lane;
  if (GufoDevBatchInvariant()) {
    const bool iq_grid = type == core::GgmlType::kIQ3_S ||
                         type == core::GgmlType::kIQ3_XXS ||
                         type == core::GgmlType::kIQ2_S;
    if (per_lane == 4U && batch < 6U) per_lane = 1U;
    if (per_lane == 2U && iq_grid && batch < 4U) per_lane = 1U;
  }
  if (cell->wave64 != 0U)
    return detail::TryLaunchDot4V3Wave64(type, batch, cell->rows_per_wave,
                                         cell->waves, per_lane,
                                         cell->act_k != 0U, w, x, y, s->q,
                                         s->ds, mm, kk, stream);
  const auto by_geometry = [&]<core::GgmlType Type, std::size_t Batch>() {
    const auto go = [&]<std::uint32_t Rows, std::uint32_t Waves,
                        std::uint32_t PerLane = 1>() {
      if constexpr (Type == core::GgmlType::kQ4_K ||
                    Type == core::GgmlType::kQ5_K ||
                    Type == core::GgmlType::kQ6_K ||
                    Type == core::GgmlType::kIQ4_XS) {
        if (cell->act_k != 0U) {
          LaunchSmallBatchDot4V3<Type, Batch, Rows, Waves, PerLane, true>(
              w, x, y, xq, xds, mm, kk, stream, quantize);
          return true;
        }
      }
      if (cell->act_k != 0U) return false;
      LaunchSmallBatchDot4V3<Type, Batch, Rows, Waves, PerLane>(
          w, x, y, xq, xds, mm, kk, stream, quantize);
      return true;
    };
    if constexpr (Type == core::GgmlType::kQ4_K ||
                  Type == core::GgmlType::kQ5_K ||
                  Type == core::GgmlType::kQ6_K ||
                  Type == core::GgmlType::kIQ4_XS ||
                  Type == core::GgmlType::kIQ4_NL ||
                  Type == core::GgmlType::kQ3_K ||
                  ((Type == core::GgmlType::kIQ3_S ||
                    Type == core::GgmlType::kIQ3_XXS ||
                    Type == core::GgmlType::kIQ2_S) &&
                   Batch >= 4)) {
      if (per_lane == 2U) {
        switch (cell->rows_per_wave * 16U + cell->waves) {
          case 2U * 16U + 4U: return go.template operator()<2, 4, 2>();
          case 4U * 16U + 4U: return go.template operator()<4, 4, 2>();
          case 4U * 16U + 8U: return go.template operator()<4, 8, 2>();
          case 8U * 16U + 4U: return go.template operator()<8, 4, 2>();
          default: return false;
        }
      }
    }
    // RowsPerLane 4: all nine types at Batch >= 6 (kernel static_assert), same
    // four (rows_per_wave, waves) cases as the x2 arm; go() carries act_k for
    // the four K types.
    if constexpr ((Type == core::GgmlType::kIQ3_S ||
                   Type == core::GgmlType::kIQ3_XXS ||
                   Type == core::GgmlType::kIQ2_S ||
                   Type == core::GgmlType::kQ4_K ||
                   Type == core::GgmlType::kQ5_K ||
                   Type == core::GgmlType::kQ6_K ||
                   Type == core::GgmlType::kIQ4_XS ||
                   Type == core::GgmlType::kIQ4_NL ||
                   Type == core::GgmlType::kQ3_K) &&
                  Batch >= 6) {
      // headq6: GUFO_DEV_HEADQ6 = the RowGroup, output head only
      // (Q6_K r4w8x4k ActK at B7/B8). See GufoDevHeadQ6 above; bit-exact.
      if constexpr (Type == core::GgmlType::kQ6_K && Batch >= 7) {
        const std::uint32_t head_group = GufoDevHeadQ6();
        if (head_group != 0U && per_lane == 4U && cell->act_k != 0U &&
            cell->rows_per_wave == 4U && cell->waves == 8U) {
          // xq/xds, never s->q/s->ds. The head IS a NormQ site
          // (output_norm -> head ActK), so with GUFO_DEV_NORMQ=1 the lookup
          // above has already set quantize = false and pointed xq/xds at the
          // fused norm's buffers. Reading s->q here would multiply against a
          // scratch nobody wrote this round.
          if (head_group == 2U)
            LaunchSmallBatchDot4V3<Type, Batch, 4, 8, 4, true, 32, 2>(
                w, x, y, xq, xds, mm, kk, stream, quantize);
          else
            LaunchSmallBatchDot4V3<Type, Batch, 4, 8, 4, true, 32, 1>(
                w, x, y, xq, xds, mm, kk, stream, quantize);
          return true;
        }
      }
      if (per_lane == 4U) {
        switch (cell->rows_per_wave * 16U + cell->waves) {
          case 2U * 16U + 4U: return go.template operator()<2, 4, 4>();
          case 4U * 16U + 4U: return go.template operator()<4, 4, 4>();
          case 4U * 16U + 8U: return go.template operator()<4, 8, 4>();
          case 8U * 16U + 4U: return go.template operator()<8, 4, 4>();
          default: return false;
        }
      }
    }
    if (per_lane != 1U) return false;
    switch (cell->rows_per_wave * 16U + cell->waves) {
      case 2U * 16U + 4U: return go.template operator()<2, 4>();
      case 4U * 16U + 2U: return go.template operator()<4, 2>();
      case 4U * 16U + 4U: return go.template operator()<4, 4>();
      case 4U * 16U + 8U: return go.template operator()<4, 8>();
      case 8U * 16U + 4U: return go.template operator()<8, 4>();
      default: return false;
    }
  };
  const auto by_batch = [&]<core::GgmlType Type>() {
    switch (batch) {
      case 1: return by_geometry.template operator()<Type, 1>();
      case 2: return by_geometry.template operator()<Type, 2>();
      case 3: return by_geometry.template operator()<Type, 3>();
      case 4: return by_geometry.template operator()<Type, 4>();
      case 5: return by_geometry.template operator()<Type, 5>();
      case 6: return by_geometry.template operator()<Type, 6>();
      case 7: return by_geometry.template operator()<Type, 7>();
      default: return by_geometry.template operator()<Type, 8>();
    }
  };
  if (type == core::GgmlType::kQ4_K)
    return by_batch.template operator()<core::GgmlType::kQ4_K>();
  if (type == core::GgmlType::kQ5_K)
    return by_batch.template operator()<core::GgmlType::kQ5_K>();
  if (type == core::GgmlType::kQ6_K)
    return by_batch.template operator()<core::GgmlType::kQ6_K>();
  if (type == core::GgmlType::kIQ4_XS)
    return by_batch.template operator()<core::GgmlType::kIQ4_XS>();
  if (type == core::GgmlType::kIQ4_NL)
    return by_batch.template operator()<core::GgmlType::kIQ4_NL>();
  if (type == core::GgmlType::kQ3_K)
    return by_batch.template operator()<core::GgmlType::kQ3_K>();
  if (type == core::GgmlType::kIQ3_S)
    return by_batch.template operator()<core::GgmlType::kIQ3_S>();
  if (type == core::GgmlType::kIQ3_XXS)
    return by_batch.template operator()<core::GgmlType::kIQ3_XXS>();
  if (type == core::GgmlType::kIQ2_S)
    return by_batch.template operator()<core::GgmlType::kIQ2_S>();
  return false;
}

void LaunchKQuantSmallBatchByType(core::GgmlType type, const void* w,
                                  const float* fp32_x, float* y,
                                  std::size_t batch, std::size_t m,
                                  std::size_t k, dim3 grid, dim3 block,
                                  hipStream_t stream) {
  if (GufoDevKQuantV3() &&
      TryLaunchKQuantV3(type, w, fp32_x, y, batch, m, k, stream))
    return;
  if (GufoDevDot4() &&
      TryLaunchDot4SmallBatch(type, w, fp32_x, y, batch, m, k, stream))
    return;
  if (GufoDevIqFastEnabled()) {
    switch (type) {
      case core::GgmlType::kIQ2_S:
        if (TryLaunchKQuantSmallBatchIqFast<core::GgmlType::kIQ2_S>(
                w, fp32_x, y, batch, m, k, stream)) return;
        break;
      case core::GgmlType::kIQ3_S:
        if (TryLaunchKQuantSmallBatchIqFast<core::GgmlType::kIQ3_S>(
                w, fp32_x, y, batch, m, k, stream)) return;
        break;
      case core::GgmlType::kIQ3_XXS:
        if (TryLaunchKQuantSmallBatchIqFast<core::GgmlType::kIQ3_XXS>(
                w, fp32_x, y, batch, m, k, stream)) return;
        break;
      default:
        break;
    }
  }
  if (GufoDevSmallBatchTuned() &&
      TryLaunchKQuantSmallBatchTuned(type, w, fp32_x, y, batch, m, k, stream)) {
    return;
  }
  // CAP: the wave64 K-quant family's acceptance IS a width table (Q5_K up at
  // 5,7,8 but not 6; Q6_K down at 5 and >= 7; nothing below 4) and its 64-lane
  // butterfly is a different reduction from the wave32 families. Nothing here
  // can be pinned, so batch-invariant mode declines it at every width. In the
  // served env (KQUANT_V3 + DOT4 + IQ_FAST on) it is not reached at width 8 for
  // any served shape, so width 8 keeps its kernel; confirm with the route dump.
  if (!GufoDevBatchInvariant() &&
      detail::TryLaunchKQuantSmallBatchWave64(type, w, fp32_x, y, batch, m, k,
                                              stream)) {
    return;
  }
  constexpr std::size_t RowsPerWave = 3;
  switch (type) {
    case core::GgmlType::kQ4_K:
      // CAP: Batch=4 with Groups=2 is structurally width 8 (two groups of 4).
      // Declined under batch-invariant mode; dot4 already owns Q4_K at width 8
      // in the served env, so this branch is not on the served route.
      if (!GufoDevBatchInvariant() && batch == 8 && k == 5120 && m == 1280) {
        SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, 4,
                                            RowsPerWave, core::GgmlType::kQ4_K,
                                            1, 12, 2, true>
            <<<((m + 47) / 48) * 2, block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ4_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kQ5_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ5_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kQ6_K: {
      // The large FFN projections reuse activations across three output rows.
      // Smaller batches keep two rows to avoid paying for unused capacity.
      if (BatchInvariantSelect(batch) >= 5 &&
          ((m == 17408 && k == 5120) || (m == 5120 && k == 17408))) {
        const auto launch_ffn = [&]<std::size_t Batch>() {
          SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch, 3,
                                              core::GgmlType::kQ6_K, 2, 1, 1,
                                              true>
              <<<(m + 47) / 48, block, 0, stream>>>(w, fp32_x, y, m, k);
        };
        switch (batch) {
          case 1:
            launch_ffn.template operator()<1>();
            return;
          case 2:
            launch_ffn.template operator()<2>();
            return;
          case 3:
            launch_ffn.template operator()<3>();
            return;
          case 4:
            launch_ffn.template operator()<4>();
            return;
          case 5:
            launch_ffn.template operator()<5>();
            return;
          case 6:
            launch_ffn.template operator()<6>();
            return;
          case 7:
            launch_ffn.template operator()<7>();
            return;
          case 8:
            launch_ffn.template operator()<8>();
            return;
          default:
            break;
        }
      }
      // The vocabulary projection reuses activations across three output
      // rows and two consecutive tiles, retaining the scalar sum order.
      // The Qwen27B head also benefits at shorter verification widths. m >= 65536
      // (was == 248320) so a drafter head cut by GUFO_DEV_DRAFT_VOCAB keeps it
      // as well; the only other k = 5120 Q6_K tensor is the 17408-row FFN.
      const std::size_t head_sel = BatchInvariantSelect(batch);
      if (((head_sel >= 2 && head_sel <= 6) && m >= 65536 && k == 5120) ||
          ((head_sel == 7 || head_sel == 8) && m >= 65536)) {
        constexpr std::size_t kHeadRows = kKQuantSmallBatchWarps * 3;
        const auto launch_head = [&]<std::size_t Batch>() {
          const auto launch_indices = [&]<bool Narrow>() {
            SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                                3, core::GgmlType::kQ6_K, 2, 1,
                                                1, Narrow>
                <<<(m + kHeadRows - 1) / kHeadRows, block, 0, stream>>>(
                    w, fp32_x, y, m, k);
          };
          // Widths 2-6 now take any m >= 65536, so every width checks.
          if (FitsSmallBatch32BitIndices(
                         BatchInvariantSelect(Batch), m, k,
                         quant::QuantizedRowBytes(core::GgmlType::kQ6_K, k))) {
            launch_indices.template operator()<true>();
          } else {
            launch_indices.template operator()<false>();
          }
        };
        switch (batch) {
          case 1:
            launch_head.template operator()<1>();
            break;
          case 2:
            launch_head.template operator()<2>();
            break;
          case 3:
            launch_head.template operator()<3>();
            break;
          case 4:
            launch_head.template operator()<4>();
            break;
          case 5:
            launch_head.template operator()<5>();
            break;
          case 6:
            launch_head.template operator()<6>();
            break;
          case 7:
            launch_head.template operator()<7>();
            break;
          case 8:
            launch_head.template operator()<8>();
            break;
          default:
            break;
        }
        return;
      }
      // Other shapes retain two rows to bound register pressure.
      constexpr std::size_t kRowsPerBlock =
          kKQuantSmallBatchWarps * kKQuantSmallBatchRowsPerWave;
      const dim3 q6_grid(
          static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
      LaunchKQuantSmallBatchTyped<kKQuantSmallBatchRowsPerWave,
                                  core::GgmlType::kQ6_K>(
          w, fp32_x, y, batch, m, k, q6_grid, block, stream);
    }
      return;
    case core::GgmlType::kQ3_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ3_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ4_NL:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ4_NL>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ4_XS:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ4_XS>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ3_S:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ3_S>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    // Iron's extra quants (Q2_K, IQ2_XXS/XS/S, IQ3_XXS, IQ1_M).
    case core::GgmlType::kQ2_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ2_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ2_XXS:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ2_XXS>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ2_XS:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ2_XS>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ2_S:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ2_S>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ3_XXS:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ3_XXS>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ1_M:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ1_M>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    default:
      return;
  }
}

// opt-q4kxl: tile-configuration dispatch for WKQuantA8BlockedWmmaGEMMKernel.
// The configurations mirror the Q8_0 launcher exactly so the two paths share
// the same occupancy tuning.
template<core::GgmlType WType>
void LaunchKQuantWmmaGEMMTyped(const void* w, const void* q8_1_x, float* y,
                               std::size_t batch, std::size_t m, std::size_t k,
                               hipStream_t stream) {
  if (batch <= 8) {
    constexpr int kBM = 128;
    const dim3 grid(1, static_cast<unsigned int>((m + kBM - 1) / kBM));
    if (m >= (3 * k)) {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 32, 4, 4, 2, WType>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    } else {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 16, 4, 8, 1, WType>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    }
    return;
  }
  if (batch >= 96) {
    constexpr int kBM = 128;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    // WM=4/WN=2 (2 row subtiles and 4 token subtiles per wave). The mirror
    // shape WM=2/WN=4 halves the per-token activation sums the offset term
    // needs, but measured far worse -- pp2048 464 -> 372 -- because four row
    // subtiles per wave hold twice the A-fragment and scale state.
    WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2, WType>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    return;
  }
  constexpr int kBM = 128;
  constexpr int kBN = 64;
  const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                  static_cast<unsigned int>((m + kBM - 1) / kBM));
  WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1, WType>
      <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
}

void LaunchKQuantWmmaGEMM(core::GgmlType type, const void* w,
                          const void* q8_1_x, float* y, std::size_t batch,
                          std::size_t m, std::size_t k, hipStream_t stream) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ4_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ5_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ5_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ6_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ6_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ3_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ3_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kIQ4_NL:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ4_NL>(w, q8_1_x, y, batch, m,
                                                         k, stream);
      return;
    case core::GgmlType::kIQ4_XS:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ4_XS>(w, q8_1_x, y, batch, m,
                                                         k, stream);
      return;
    case core::GgmlType::kIQ3_S:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ3_S>(w, q8_1_x, y, batch, m,
                                                        k, stream);
      return;
    case core::GgmlType::kQ2_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ2_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kIQ2_XXS:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ2_XXS>(w, q8_1_x, y, batch,
                                                          m, k, stream);
      return;
    case core::GgmlType::kIQ2_XS:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ2_XS>(w, q8_1_x, y, batch, m,
                                                         k, stream);
      return;
    case core::GgmlType::kIQ2_S:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ2_S>(w, q8_1_x, y, batch, m,
                                                        k, stream);
      return;
    case core::GgmlType::kIQ3_XXS:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ3_XXS>(w, q8_1_x, y, batch,
                                                          m, k, stream);
      return;
    case core::GgmlType::kIQ1_M:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ1_M>(w, q8_1_x, y, batch, m,
                                                        k, stream);
      return;
    default:
      return;
  }
}

void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (detail::TryLaunchQuantPrefillWave64(type, w, q8_1_x, y, batch, m, k,
                                          stream)) {
    return;
  }
  // opt-q4kxl: every K-quant/IQ format in the UD-Q4_K_XL shard goes through the
  // generalized WMMA kernel. Without this the only prefill route for them was
  // BatchedQuantGEMVKernel, one warp per (row, token), which re-reads the whole
  // weight matrix once per token and is unusable at prefill batch.
  if (type != core::GgmlType::kQ8_0) {
    if (::gufo::quant::QuantizedBlockElements(type) != 0) {
      LaunchKQuantWmmaGEMM(type, w, q8_1_x, y, batch, m, k, stream);
    }
    return;
  }
  if (batch <= 8) {
    constexpr int kBM = 128;
    const dim3 grid(1, static_cast<unsigned int>((m + kBM - 1) / kBM));
    if (m >= (3 * k)) {
      W8A8BlockedWmmaGEMMKernel<kBM, 32, 4, 4, 2>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    } else {
      W8A8BlockedWmmaGEMMKernel<kBM, 16, 4, 8, 1>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    }
    return;
  }
  // A 128-token macro tile is the throughput configuration; short prompt chunks
  // would leave most of it idle, so they take the 64-token variant.
  if (batch >= 96) {
    constexpr int kBM = 128;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    // Two K blocks per LDS stage, not four: the stage buffers drop from 36864
    // to 18432 bytes and occupancy rises from 6 to 8 waves per SIMD. This is
    // only reachable with the register pressure the addressing rewrite and the
    // one-token-tile-at-a-time compute loop freed up -- at BK=2 the previous
    // body needed 500 bytes/lane of scratch and ran 4.3x slower.
    W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
  } else {
    constexpr int kBM = 128;
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
  }
}

void LaunchBatchedDualQuantGEMMPreQuantized(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, float* y_up, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  // The fused 16-row dual kernel shared one activation stage between the two
  // weight matrices, which mattered when a block owned only 16 rows. With the
  // 128-row blocked macro tile the activation panel is already cheap, and two
  // blocked launches measured faster than a blocked dual (23.0 vs 24.4 ms for
  // the FFN shape at batch 2048) because the dual doubles LDS and halves
  // occupancy. See opt-c163-blocked-dual in
  // docs/models/qwen3.8-27b/BENCHMARKS.md.
  LaunchBatchedQuantGEMMPreQuantized(type, w_gate, q8_1_x, y_gate, batch, m, k,
                                     stream);
  LaunchBatchedQuantGEMMPreQuantized(type, w_up, q8_1_x, y_up, batch, m, k,
                                     stream);
}

namespace {

/// Dispatches the fused-epilogue instantiation of whichever blocked prefill
/// kernel `type` uses. The row-block base has to be a whole 32-element
/// quantization block for every block in the grid, which is what pins BM here.
template<int BN>
void LaunchFusedSwiGluTile(const void* w_up, const void* q8_1_x,
                           const float* gate, void* q8_1_out, std::size_t batch,
                           std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  const dim3 grid(static_cast<unsigned int>((batch + BN - 1) / BN),
                  static_cast<unsigned int>(m / kBM));
  W8A8BlockedWmmaGEMMKernel<kBM, BN, 2, 4, 2, true, true>
      <<<grid, 256, 0, stream>>>(w_up, q8_1_x, nullptr, batch, m, k, gate,
                                 q8_1_out);
}

}  // namespace

bool IsFusedSwiGluGemmEpilogueSupported(core::GgmlType type, std::size_t batch,
                                        std::size_t m,
                                        std::size_t q8_out_bytes) noexcept {
  return type == core::GgmlType::kQ8_0 && batch >= 96 && m != 0 &&
         m % 128 == 0 &&
         Q8ActTiledBytes(batch, m) + Q8ActSumBytes(batch, m) <= q8_out_bytes;
}

void LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, void* q8_1_out, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  LaunchBatchedQuantGEMMPreQuantized(type, w_gate, q8_1_x, y_gate, batch, m, k,
                                     stream);
  LaunchFusedSwiGluTile<128>(w_up, q8_1_x, y_gate, q8_1_out, batch, m, k,
                             stream);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, m, stream);
}

void LaunchBatchedQuantGEMM(core::GgmlType type, const void* w,
                            const void* bf16_x, float* y, std::size_t batch,
                            std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type == core::GgmlType::kBF16 || type == core::GgmlType::kF32) {
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    static thread_local void* s_act_q8_scratch = nullptr;
    static thread_local std::size_t s_act_q8_scratch_bytes = 0;

    const std::size_t req_bytes = Q8ActTiledBytes(batch, k);
    if (s_act_q8_scratch_bytes < req_bytes) {
      if (s_act_q8_scratch != nullptr) {
        (void)hipFree(s_act_q8_scratch);
      }
      HIP_CHECK(hipMalloc(&s_act_q8_scratch, req_bytes));
      s_act_q8_scratch_bytes = req_bytes;
    }

    LaunchQuantizeActivationQ8_1(bf16_x, s_act_q8_scratch, batch, k, stream);
    LaunchBatchedQuantGEMMPreQuantized(type, w, s_act_q8_scratch, y, batch, m,
                                       k, stream);
    return;
  }

  constexpr std::size_t kWarp = 32;
  const std::size_t warps_per_block = 8;
  const std::size_t rows_per_block = warps_per_block;
  const std::size_t num_blocks_x = (m + rows_per_block - 1) / rows_per_block;
  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * warps_per_block));
  BatchedQuantGEMVKernel<hip_bfloat16><<<grid, block, 0, stream>>>(
      type, w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, m, k);
}

void LaunchBatchedQuantGEMMBf16(core::GgmlType type, const void* w,
                                const void* bf16_x, float* y, std::size_t batch,
                                std::size_t m, std::size_t k,
                                hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || type == core::GgmlType::kBF16 ||
      type == core::GgmlType::kF32) {
    return;
  }

  constexpr std::size_t kWarp = 32;
  constexpr std::size_t kWarpsPerBlock = 8;
  if (type == core::GgmlType::kQ8_0 && batch <= 8) {
    constexpr std::size_t kRowsPerWave = 4;
    const std::size_t tokens_per_wave = batch <= 2 ? 2 : (batch <= 4 ? 4 : 8);
    const dim3 grid(
        static_cast<unsigned int>((m + (kWarpsPerBlock * kRowsPerWave) - 1) /
                                  (kWarpsPerBlock * kRowsPerWave)),
        1);
    const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
    if (tokens_per_wave == 2) {
      TiledBatchedQ8_0GEMMKernel<kWarpsPerBlock, 2><<<grid, block, 0, stream>>>(
          w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, batch, m, k);
    } else if (tokens_per_wave == 4) {
      TiledBatchedQ8_0GEMMKernel<kWarpsPerBlock, 4><<<grid, block, 0, stream>>>(
          w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, batch, m, k);
    } else {
      TiledBatchedQ8_0GEMMKernel<kWarpsPerBlock, 8><<<grid, block, 0, stream>>>(
          w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, batch, m, k);
    }
    return;
  }

  const std::size_t num_blocks_x = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
  BatchedQuantGEMVKernel<hip_bfloat16><<<grid, block, 0, stream>>>(
      type, w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, m, k);
}

void LaunchBatchedQuantGEMMFp32(core::GgmlType type, const void* w,
                                const float* fp32_x, float* y,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || type == core::GgmlType::kBF16 ||
      type == core::GgmlType::kF32) {
    return;
  }

  if (batch > 8) {
    // Paired seven-row blocks avoid a separate tail on these projections.
    const bool individual_up =
        m == 17408 && k == 5120 &&
        (type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kQ3_K ||
         type == core::GgmlType::kIQ4_NL);
    const bool iq4_groups = type == core::GgmlType::kIQ4_XS &&
                            ((batch == 28 && ((m == 34816 && k == 5120) ||
                                              (m == 5120 && k == 17408))) ||
                             (batch == 42 && m == 5120 && k == 17408));
    if ((type == core::GgmlType::kQ6_K || type == core::GgmlType::kQ3_K ||
         type == core::GgmlType::kIQ4_NL ||
         (type == core::GgmlType::kQ5_K && m == 5120 && k == 6144) ||
         iq4_groups || (individual_up && batch <= 42)) &&
        batch <= 56 && batch % 14 == 0 &&
        detail::TryLaunchKQuantSmallBatchWave64(type, w, fp32_x, y, 14, m, k,
                                                stream, batch / 14)) {
      return;
    }
    const bool ffn =
        ((m == 17408 || m == 34816) && k == 5120) || (m == 5120 && k == 17408);
    const bool q8_projection =
        type == core::GgmlType::kQ8_0 &&
        (ffn || (k == 5120 && (m == 6144 || m == 10240)) ||
         (m == 5120 && k == 6144));
    // Group ragged verification rows without rereading weights for the tail.
    // Full sixteen-row groups and unsupported shapes keep their own route.
    if (q8_projection && batch > 16 && batch <= 36 && batch != 32 &&
        detail::TryLaunchQ8SmallBatchWave64(w, fp32_x, y, batch, m, k,
                                            stream)) {
      return;
    }
    const auto launch_q8 = [&]<std::size_t Batch>(const float* input,
                                                  float* output) {
      if constexpr (Batch == 12 || Batch == 16) {
        if (detail::TryLaunchQ8SmallBatchWave64(w, input, output, Batch, m, k,
                                                stream))
          return;
      }
      constexpr std::size_t rows = Batch == 11 || Batch == 13 || Batch == 14 ? 4
                                   : Batch == 12                             ? 2
                                                 : 3;
      SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, rows, true, true>
          <<<(m + 16 * rows - 1) / (16 * rows), 512, 0, stream>>>(w, input,
                                                                  output, m, k);
    };
    // Cross-request verification can contain up to 64 rows. Reuse weights in
    // wider exact groups where measured, and keep the established small-batch
    // arithmetic for other projections and the final unpadded group.
    const bool exact_width = q8_projection || (m == 34816 && k == 5120) ||
                             (m == 5120 && k == 17408) ||
                             (type == core::GgmlType::kQ6_K &&
                              ((k == 5120 && (m == 248320 || m == 17408)) ||
                               (m == 5120 && k == 6144)));
    for (std::size_t offset = 0; offset < batch;) {
      const std::size_t remaining = batch - offset;
      const auto groups = std::min<std::size_t>(4, remaining / 16);
      if (groups > 1) {
        const bool launched =
            q8_projection ? detail::TryLaunchQ8SmallBatchWave64(
                                w, fp32_x + offset * k, y + offset * m, 16, m,
                                k, stream, groups)
                          : detail::TryLaunchKQuantSmallBatchWave64(
                                type, w, fp32_x + offset * k, y + offset * m,
                                16, m, k, stream, groups);
        if (launched) {
          offset += groups * 16;
          continue;
        }
      }
      std::size_t width =
          (exact_width || (individual_up && remaining == 14)) && remaining > 8
              ? std::min<std::size_t>(16, remaining)
          : remaining >= 16 ? 16
          : remaining >= 12 ? 12
                            : 0;
      const float* input = fp32_x + offset * k;
      float* output = y + offset * m;
      if (width != 0 && q8_projection) {
        switch (width) {
          case 9:
            launch_q8.template operator()<9>(input, output);
            break;
          case 10:
            launch_q8.template operator()<10>(input, output);
            break;
          case 11:
            launch_q8.template operator()<11>(input, output);
            break;
          case 12:
            launch_q8.template operator()<12>(input, output);
            break;
          case 13:
            launch_q8.template operator()<13>(input, output);
            break;
          case 14:
            launch_q8.template operator()<14>(input, output);
            break;
          case 15:
            launch_q8.template operator()<15>(input, output);
            break;
          case 16:
            launch_q8.template operator()<16>(input, output);
            break;
        }
      } else if (width != 0 &&
                 detail::TryLaunchKQuantSmallBatchWave64(type, w, input, output,
                                                         width, m, k, stream)) {
        // The native dispatcher owns the qualified K-quant shapes.
      } else {
        width = std::min<std::size_t>(8, remaining);
        LaunchBatchedQuantGEMMFp32(type, w, input, output, width, m, k, stream);
      }
      offset += width;
    }
    return;
  }

  constexpr std::size_t kWarp = 32;
  constexpr std::size_t kWarpsPerBlock = 8;
  const std::size_t num_blocks_x = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  // opt-q4kxl: K-quant/IQ formats share one weight stream across the whole
  // draft block instead of re-reading it per token.
  if (::gufo::hip::detail::IsNativeWmmaQuant(type) && batch <= 8) {
    // Exact width instantiations: a padded Batch would stage and multiply
    // activations that no output consumes.
    constexpr std::size_t kKWarpsPerBlock = kKQuantSmallBatchWarps;
    const dim3 block(static_cast<unsigned int>(kWarp * kKWarpsPerBlock));
    // Three rows with twelve waves is the retained geometry. Q6_K dispatches
    // two rows below because its extra scale state spills with three.
    constexpr std::size_t kRowsPerBlock = kKWarpsPerBlock * 3;
    const dim3 grid(
        static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
    LaunchKQuantSmallBatchByType(type, w, fp32_x, y, batch, m, k, grid, block,
                                 stream);
    return;
  }
  if ((type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K) &&
      batch >= 1 && batch <= 8) {
    const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
    // Large projections share each weight row across the batch. Narrow SSM
    // projections instead distribute token rows across more workgroups.
    if (type == core::GgmlType::kQ8_0) {
      // CAP: a width-6-only wave64 pick. Width 8 does not take it, so the
      // served round is unchanged; batch-invariant mode declines it.
      if (!GufoDevBatchInvariant() && batch == 6 &&
          detail::TryLaunchQ8SmallBatchWave64(w, fp32_x, y, batch, m, k,
                                              stream))
        return;
      // The 48-output SSM controls, or their joined 96-output pair, fit in
      // cache. Independent token rows expose more workgroups than the
      // large-matrix shared-weight geometry.
      if ((m == 48 || m == 96) && k == 5120) {
        SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 1, 1>
            <<<dim3((m + 7) / 8, static_cast<unsigned>(batch)), 256, 0,
               stream>>>(w, fp32_x, y, m, k);
        return;
      }
      constexpr std::size_t kVecWarpsPerBlock = 16;
      // Joined Qwen27B gate/up matrices benefit from three-row reuse at
      // these widths. Their bounded dimensions allow 32-bit device indexing;
      // each output keeps the scalar block-dot and reduction order.
      if (m == 34816 && k == 5120) {
        const auto launch_packed = [&]<std::size_t Batch>() {
          SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, 3, true>
              <<<(m + 47) / 48, 512, 0, stream>>>(w, fp32_x, y, m, k);
        };
        switch (batch) {
          case 2:
            launch_packed.template operator()<2>();
            return;
          case 4:
            launch_packed.template operator()<4>();
            return;
          case 8:
            launch_packed.template operator()<8>();
            return;
          default:
            break;
        }
      }
      // Batch 7 shares its 31.5 KiB activation stage across twice as many
      // waves. Batch 8 instead benefits from three rows per wave below.
      if (batch == 7 && m == 5120 && k == 17408) {
        SmallBatchQ8_0ExactFp32VecGEMMKernel<32, 7, 2>
            <<<(m + 63) / 64, 1024, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
      // Reuse the long FFN-down input across three output rows.
      if (batch == 8 && m == 5120 && k == 17408) {
        constexpr std::size_t kDownRows = kVecWarpsPerBlock * 3;
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 8, 3>
            <<<(m + kDownRows - 1) / kDownRows, kVecWarpsPerBlock * 32, 0,
               stream>>>(w, fp32_x, y, m, k);
        return;
      }
      constexpr std::size_t kVecRowsPerWave = 2;
      constexpr std::size_t kVecRowsPerBlock =
          kVecWarpsPerBlock * kVecRowsPerWave;
      const unsigned int vec_blocks = static_cast<unsigned int>(
          (m + kVecRowsPerBlock - 1) / kVecRowsPerBlock);
      const dim3 vec_block(
          static_cast<unsigned int>(kWarp * kVecWarpsPerBlock));
      switch (batch) {
        case 1:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 1,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 2:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 2,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 3:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 3,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 4:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 4,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 5:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 5,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 6:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 6,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 7:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 7,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        default:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 8,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
      }
    }
    // Only Q8_K reaches here: Q8_0 always takes the width-templated route
    // above.
    if (batch <= 1) {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 1>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    } else if (batch <= 2) {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 2>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    } else if (batch <= 4) {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 4>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    } else {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 8>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    }
    return;
  }

  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
  BatchedQuantGEMVKernel<float>
      <<<grid, block, 0, stream>>>(type, w, fp32_x, y, m, k);
}

// ---- k95small (GUFO_DEV_SMALL_K95=1; the caller reads the env) ----
// Each entry point either launches its whole group or launches nothing and
// returns false; the caller then runs the old per-projection calls. They
// only accept the cells whose served route they reproduce, and they are
// bitwise equal to that route (see k95_small.hpp).

namespace {

// A V3 cell this TU launches itself (not the wave64 TU, which quantizes
// inside), with the quantizer kind `act_k`. 16-byte weight alignment covers
// every V3 type's own requirement (TryLaunchKQuantV3).
[[nodiscard]] const KQuantV3Cell* K95OwnV3Cell(core::GgmlType type,
                                               const void* w, std::size_t batch,
                                               std::size_t m, std::size_t k,
                                               bool act_k) {
  const KQuantV3Cell* cell = FindKQuantV3Cell(type, batch, m, k);
  if (cell == nullptr || cell->wave64 != 0U || (cell->act_k != 0U) != act_k ||
      (reinterpret_cast<std::uintptr_t>(w) & 15U) != 0U || k % 256 != 0)
    return nullptr;
  return cell;
}

void K95Quantize(bool act_k, const float* x, Dot4Scratch* s, std::size_t batch,
                 std::size_t k, hipStream_t stream) {
  const auto subs = static_cast<std::uint32_t>(batch * (k / 16));
  if (act_k)
    QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
        x, s->q, s->ds, subs);
  else
    QuantizeActivationsDot4Kernel<<<(subs + 255U) / 256U, 256, 0, stream>>>(
        x, s->q, s->ds, subs);
}

void K95V3NoQuant(core::GgmlType type, const void* w, const float* x, float* y,
                  std::size_t batch, std::size_t m, std::size_t k,
                  hipStream_t stream) {
  // K95OwnV3Cell accepted the cell, so only an unlaunchable geometry fails
  // here; the scratch is already overwritten, so fail loudly, never silently.
  if (!TryLaunchKQuantV3(type, w, x, y, batch, m, k, stream, false))
    throw std::logic_error("k95small: V3 cell without a launchable geometry");
}

}  // namespace

bool TryLaunchK95AttentionQkv(core::GgmlType q_type, const void* wq,
                              core::GgmlType k_type, const void* wk,
                              core::GgmlType v_type, const void* wv,
                              const float* x, float* yq, float* yk, float* yv,
                              std::size_t batch, std::size_t q_m,
                              std::size_t kv_m, std::size_t k,
                              hipStream_t stream) {
  using T = core::GgmlType;
  // Served width-8 routes reproduced here: Q = any own non-ActK V3 cell;
  // K = V3 Q5_K, RowsPerWave 4, non-ActK; V = no V3 cell, then dot4 geometry
  // 2 = Dot4Lds<Q6_K, 8, 2, 8, 1>. All three quantize x with
  // QuantizeActivationsDot4Kernel, so one quantization serves all three.
  if (batch != 8 || kv_m == 0 || kv_m > UINT32_MAX || k > UINT32_MAX ||
      !GufoDevKQuantV3() || !GufoDevDot4() || k_type != T::kQ5_K ||
      v_type != T::kQ6_K)
    return false;
  if (K95OwnV3Cell(q_type, wq, batch, q_m, k, false) == nullptr)
    return false;
  const KQuantV3Cell* kc = K95OwnV3Cell(T::kQ5_K, wk, batch, kv_m, k, false);
  if (kc == nullptr || kc->rows_per_wave != 4U)
    return false;
  if (FindKQuantV3Cell(T::kQ6_K, batch, kv_m, k) != nullptr ||
      FindQ6Dot4Geometry(kv_m, k, batch) != 2)
    return false;
  Dot4Scratch* s = GetDot4Scratch(stream, batch * k);
  if (s == nullptr)
    return false;
  // NormQ: the attn norm already wrote these plain activations; K95V3NoQuant
  // below finds the same buffers through NormQLookup, so both agree.
  std::int8_t* xq = s->q;
  float2* xds = s->ds;
  if (!NormQLookup(stream, x, batch, k, false, &xq, &xds))
    K95Quantize(false, x, s, batch, k, stream);
  K95V3NoQuant(q_type, wq, x, yq, batch, q_m, k, stream);
  // ponytail: one geometry (16 waves, K 1 row/lane = 16 blocks, V 2 rows/wave
  // = 32 blocks); k95small_bench measures the alternatives.
  LaunchK95Kv<16, 1, 2>(wk, wv, xq, xds, yk, yv,
                        static_cast<std::uint32_t>(kv_m),
                        static_cast<std::uint32_t>(k), stream);
  return true;
}

bool TryLaunchK95SharedQuantPair(core::GgmlType a_type, const void* wa,
                                 float* ya, std::size_t a_m,
                                 core::GgmlType b_type, const void* wb,
                                 float* yb, std::size_t b_m, const float* x,
                                 std::size_t batch, std::size_t k,
                                 hipStream_t stream) {
  // Served route of each: V3 (KQUANT_V3 is tried first for batch 1-8). Same
  // x and same quantizer kind -> one quantization serves both.
  if (batch < 1 || batch > 8 || !GufoDevKQuantV3())
    return false;
  const KQuantV3Cell* ca = FindKQuantV3Cell(a_type, batch, a_m, k);
  if (ca == nullptr)
    return false;
  const bool act_k = ca->act_k != 0U;
  if (K95OwnV3Cell(a_type, wa, batch, a_m, k, act_k) == nullptr ||
      K95OwnV3Cell(b_type, wb, batch, b_m, k, act_k) == nullptr)
    return false;
  Dot4Scratch* s = GetDot4Scratch(stream, batch * k);
  if (s == nullptr)
    return false;
  // NormQ: as above. The buffers are not used here -- both K95V3NoQuant calls
  // below resolve them through NormQLookup themselves; only the quantize goes.
  std::int8_t* nq_q = nullptr;
  float2* nq_ds = nullptr;
  if (!NormQLookup(stream, x, batch, k, act_k, &nq_q, &nq_ds))
    K95Quantize(act_k, x, s, batch, k, stream);
  (void)nq_q;
  (void)nq_ds;
  K95V3NoQuant(a_type, wa, x, ya, batch, a_m, k, stream);
  K95V3NoQuant(b_type, wb, x, yb, batch, b_m, k, stream);
  return true;
}

bool TryLaunchK95SsmControls(core::GgmlType alpha_type, const void* alpha,
                             core::GgmlType beta_type, const void* beta,
                             const float* x, float* y_alpha, float* y_beta,
                             std::size_t batch, std::size_t m, std::size_t k,
                             hipStream_t stream) {
  // Served routes reproduced: beta Q8_0 [48, 5120] at width 1-8 except 6 ->
  // SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 1, 1> (LaunchBatchedQuantGEMMFp32);
  // alpha F32 -> LaunchGEMVRows -> FastGEMVBlockKernel, 256 threads.
  if (alpha_type != core::GgmlType::kF32 ||
      beta_type != core::GgmlType::kQ8_0 || batch < 1 || batch > 8 ||
      batch == 6 || m != 48 || k != 5120 ||
      models::qwen::ResolveQwenGemmRoute(
          {.type = alpha_type, .batch_size = 1, .m = m, .k = k,
           .mode = models::qwen::QwenGemmMode::kHipDecode})
              .route != models::qwen::QwenGemmRoute::kHipF32Baseline256)
    return false;
  // ponytail: the preloaded kernel is the served pick (predicted ~5 vs ~13
  // us); LaunchK95SsmControls (the served bodies, fused) stays as the bench's
  // second arm.
  LaunchK95SsmControlsPreload(static_cast<const float*>(alpha), beta, x,
                              y_alpha, y_beta,
                              static_cast<std::uint32_t>(batch),
                              static_cast<std::uint32_t>(m), stream);
  return true;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

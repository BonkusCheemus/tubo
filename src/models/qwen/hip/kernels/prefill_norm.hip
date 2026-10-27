#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

template<std::size_t CachedDim = 0>
__global__ void BatchedRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     hip_bfloat16* __restrict__ out_bf16,
                                     std::size_t batch_size, std::size_t dim,
                                     float eps) {
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;
  const float* in_row = x + (b * dim);
  float* out_row = (out != nullptr) ? out + (b * dim) : nullptr;
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (b * dim) : nullptr;

  static_assert(CachedDim % 256 == 0);
  constexpr std::size_t cached_items = CachedDim == 0 ? 1 : CachedDim / 256;
  float values[cached_items];
  float weights[cached_items];
  float sum_sq = 0.0F;
  if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      const std::size_t i = tid + item * 256;
      const float val = in_row[i];
      values[item] = val;
      weights[item] = weight != nullptr ? weight[i] : 1.0F;
      // Unrolling must retain the generic loop's fused accumulation.
      sum_sq = __fmaf_rn(val, val, sum_sq);
    }
  } else {
    for (std::size_t i = tid; i < dim; i += block_size) {
      const float val = in_row[i];
      sum_sq += val * val;
    }
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = block_size / 2;
       stride > (CachedDim != 0 ? 16U : 0U); stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }
  if constexpr (CachedDim != 0) {
    // Finish the same descending tree in its first wave.
    if (tid < 32) {
      float sum = s_sum[tid];
#pragma unroll
      for (unsigned offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
      }
      if (tid == 0)
        s_sum[0] = sum;
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      const std::size_t i = tid + item * 256;
      const float normed = (values[item] * inv_rms) * weights[item];
      if (out_row != nullptr)
        out_row[i] = normed;
      if (out_bf16_row != nullptr)
        out_bf16_row[i] = hip_bfloat16(normed);
    }
  } else {
    for (std::size_t i = tid; i < dim; i += block_size) {
      const float w = (weight != nullptr) ? weight[i] : 1.0F;
      const float normed = (in_row[i] * inv_rms) * w;
      if (out_row != nullptr)
        out_row[i] = normed;
      if (out_bf16_row != nullptr)
        out_bf16_row[i] = hip_bfloat16(normed);
    }
  }
}

// Verify path: BatchedResidualAdd + BatchedRMSNormKernel<5120> in one
// launch. Bit-exact with the pair: same float add, the cached-row FMA order and
// the same descending tree (LDS to 32, then the first wave's shuffles).
__global__ void BatchedAddRMSNorm5120Kernel(float* __restrict__ hidden,
                                            const float* __restrict__ update,
                                            const float* __restrict__ weight,
                                            float* __restrict__ out,
                                            std::size_t batch_size, float eps) {
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) return;
  constexpr std::size_t kDim = 5120;
  constexpr std::size_t kItems = kDim / 256;
  const std::size_t tid = threadIdx.x;
  float* const row = hidden + (b * kDim);
  const float* const add_row = update + (b * kDim);
  float values[kItems];
  float weights[kItems];
  float sum_sq = 0.0F;
#pragma unroll
  for (std::size_t item = 0; item < kItems; ++item) {
    const std::size_t i = tid + item * 256;
    const float val = row[i] + add_row[i];
    row[i] = val;
    values[item] = val;
    weights[item] = weight != nullptr ? weight[i] : 1.0F;
    sum_sq = __fmaf_rn(val, val, sum_sq);
  }
  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = 128; stride > 16U; stride /= 2) {
    if (tid < stride) s_sum[tid] += s_sum[tid + stride];
    __syncthreads();
  }
  if (tid < 32) {
    float sum = s_sum[tid];
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset /= 2)
      sum += __shfl_down(sum, offset);
    if (tid == 0) s_sum[0] = sum;
  }
  __syncthreads();
  const float inv_rms = rsqrtf(s_sum[0] / static_cast<float>(kDim) + eps);
  float* const out_row = out + (b * kDim);
#pragma unroll
  for (std::size_t item = 0; item < kItems; ++item)
    out_row[tid + item * 256] = (values[item] * inv_rms) * weights[item];
}

bool TryLaunchBatchedAddRMSNorm(float* hidden, const float* update,
                                const float* weight, float* out,
                                std::size_t batch_size, std::size_t dim,
                                float eps, hipStream_t stream) {
  if (dim != 5120 || batch_size == 0 || batch_size > 64) return false;
  // NormQ: this writes `out` without the quantize, so any mark on it is stale.
  // (Fusing the quantize in here too is the next step; GUFO_DEV_FUSE_ADDNORM
  // is not in the served env, so it buys nothing today.)
  NormQClear(stream, out);
  BatchedAddRMSNorm5120Kernel<<<batch_size, 256, 0, stream>>>(
      hidden, update, weight, out, batch_size, eps);
  return true;
}

// ---------------------------------------------------------------------------
// NormQ (GUFO_DEV_NORMQ=1, added 2026-10-25): BatchedRMSNormKernel<5120> with
// the activation quantize of its consumers folded in.
//
// WHY. 203 of the ~470 activation-quantize launches of a served verify round
// read a norm output at k=5120: 0.710 ms of kernel plus 0.221 ms of launch gap
// in a 123 ms round (measured). The norm already holds every value of
// the row in registers, so the quantize costs only its cross-lane reductions
// and its stores; the x re-read and the launch go away. It also subsumes the
// duplicate quantizes k95small dedups (attention Q/K/V, SSM qkv+gate), because
// one fused output serves every consumer of that norm.
//
// WHY BOTH KINDS. The plain (QuantizeActivationsDot4Kernel, one scale per 16)
// and ActK (QuantizeActivationsQ8KDot4Kernel, one scale per 256) quantizers
// live off the same norm in different layers -- attention Q/K/V plain, SSM
// qkv+gate / the FFN pair / the output head ActK -- and the consumer's cell,
// not the norm, knows which. Writing both costs ~60 KB of extra stores per
// norm and saves the caller a per-layer cell lookup at nine sites.
//
// WHY IT IS BIT-EXACT. Element i of row b lands on thread i % 256 at
// item i / 256, so a 16-element sub-block is an aligned 16-lane group at one
// item, and an ActK 256-element super-block is one item across all 256
// threads. The norm's own sum of squares keeps the base order exactly (same
// per-thread FMA chain, same descending LDS tree finished in the first wave).
// The quantizer's three reductions are order-free, so a cross-lane tree gives
// the base's chain bit for bit:
//   * amax is fmaxf seeded at 0.0F -- a true max over the non-negative |v|,
//     NaN dropped, so associative and commutative;
//   * the per-16 sum is int32 addition;
//   * the NaN flag is a logical OR.
// Everything after the reduction (d, id, rintf(fminf(fmaxf(...)))) is the base
// expression verbatim, and `normed` is the register the base kernel stores and
// the quantize kernel reads back -- one float32 round trip either way, no
// contraction opportunity in (values * inv_rms) * weights or in normed * id.
// Proved bit-exact against a CPU reference and on the device by
// normq_op_test.
static_assert(sizeof(float2) == 8);

namespace {

constexpr std::uint32_t kNormQDim = 5120;
constexpr std::uint32_t kNormQItems = kNormQDim / 256;  // 20
constexpr std::uint32_t kNormQWaves = 8;

}  // namespace

__launch_bounds__(256) __global__ void BatchedRMSNormQuant5120Kernel(
    const float* __restrict__ x, const float* __restrict__ weight,
    float* __restrict__ out, std::int8_t* __restrict__ xq_plain,
    float2* __restrict__ xds_plain, std::int8_t* __restrict__ xq_actk,
    float2* __restrict__ xds_actk, std::size_t batch_size, float eps) {
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) return;
  const std::uint32_t tid = threadIdx.x;
  const float* const in_row = x + (b * kNormQDim);

  float values[kNormQItems];
  float weights[kNormQItems];
  float sum_sq = 0.0F;
#pragma unroll
  for (std::uint32_t item = 0; item < kNormQItems; ++item) {
    const std::uint32_t i = tid + item * 256U;
    const float val = in_row[i];
    values[item] = val;
    weights[item] = weight != nullptr ? weight[i] : 1.0F;
    // Same fused accumulation as BatchedRMSNormKernel<5120>.
    sum_sq = __fmaf_rn(val, val, sum_sq);
  }
  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();
  for (std::uint32_t stride = 128U; stride > 16U; stride /= 2U) {
    if (tid < stride) s_sum[tid] += s_sum[tid + stride];
    __syncthreads();
  }
  if (tid < 32U) {
    float sum = s_sum[tid];
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset /= 2)
      sum += __shfl_down(sum, offset);
    if (tid == 0U) s_sum[0] = sum;
  }
  __syncthreads();
  const float mean_sq = s_sum[0] / static_cast<float>(kNormQDim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  // ActK scale group = one item across all 256 threads: reduce inside each
  // wave, then combine the 8 waves through LDS.
  __shared__ std::uint32_t s_amax[kNormQWaves * kNormQItems];
  __shared__ std::uint32_t s_nan[kNormQWaves * kNormQItems];
  const std::uint32_t wave = tid / 32U;
  const std::uint32_t lane = tid % 32U;
#pragma unroll
  for (std::uint32_t item = 0; item < kNormQItems; ++item) {
    const float normed = (values[item] * inv_rms) * weights[item];
    float amax = fmaxf(0.0F, fabsf(normed));
    int nan = normed != normed ? 1 : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
      nan |= __shfl_xor(nan, offset, 32);
    }
    if (lane == 0U) {
      s_amax[(wave * kNormQItems) + item] = __float_as_uint(amax);
      s_nan[(wave * kNormQItems) + item] = static_cast<std::uint32_t>(nan);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
  __syncthreads();
  if (tid < kNormQItems) {
    // Slot (0, tid) is index tid, which only this thread reads or writes.
    float amax = 0.0F;
    int nan = 0;
#pragma unroll
    for (std::uint32_t w = 0; w < kNormQWaves; ++w) {
      amax = fmaxf(amax, __uint_as_float(s_amax[(w * kNormQItems) + tid]));
      nan |= static_cast<int>(s_nan[(w * kNormQItems) + tid]);
    }
    s_amax[tid] = __float_as_uint(amax);
    s_nan[tid] = static_cast<std::uint32_t>(nan);
  }
  __syncthreads();

  float* const out_row = out + (b * kNormQDim);
  const std::size_t q_base = b * kNormQDim;
  const std::size_t ds_base = b * (kNormQDim / 16U);
  const bool ds_writer = (tid % 16U) == 0U;
#pragma unroll
  for (std::uint32_t item = 0; item < kNormQItems; ++item) {
    const float normed = (values[item] * inv_rms) * weights[item];
    out_row[tid + (item * 256U)] = normed;
    const std::size_t at = q_base + tid + (item * 256U);
    const std::size_t sub = ds_base + (item * 16U) + (tid / 16U);

    // Plain: one scale per 16 values = this aligned 16-lane group (offsets
    // 1, 2, 4, 8 only flip lane bits 0..3).
    float p_amax = fmaxf(0.0F, fabsf(normed));
    int p_nan = normed != normed ? 1 : 0;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      p_amax = fmaxf(p_amax, __shfl_xor(p_amax, offset, 32));
      p_nan |= __shfl_xor(p_nan, offset, 32);
    }
    // ActK: the super-block amax reduced above.
    const float k_amax = __uint_as_float(s_amax[item]);
    const int k_nan = static_cast<int>(s_nan[item]);

    // One sub-block's int8 value and its per-16 sum, for one quantizer kind.
    const auto emit = [&](float amax, int nan, std::int8_t* xq, float2* xds,
                          bool act_k) {
      // fmaxf drops NaN, so without the flag a NaN activation would vanish
      // into a finite output. A NaN scale makes every dot over it NaN.
      const float d = nan != 0 ? __builtin_nanf("") : amax / 127.0F;
      const float id = amax > 0.0F ? 127.0F / amax : 0.0F;
      // Clamp: with a denormal amax, id is inf and rintf(inf) would wrap.
      const int q = static_cast<int>(
          rintf(fminf(fmaxf(normed * id, -127.0F), 127.0F)));
      xq[at] = static_cast<std::int8_t>(q);
      int sum = q;
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
      if (ds_writer)
        xds[sub] = {d, act_k ? __int_as_float(sum)
                             : d * static_cast<float>(sum)};
    };
    emit(p_amax, p_nan, xq_plain, xds_plain, false);
    emit(k_amax, k_nan, xq_actk, xds_actk, true);
    __builtin_amdgcn_sched_barrier(0);
  }
}

void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps, hipStream_t stream) {
  // Verification and draft rows can retain the hidden vector across the
  // reduction. Large prefill batches keep the generic occupancy profile.
  if (dim == kNormQDim && batch_size <= 64) {
    // NormQ: the quantize of this norm's consumers, folded in. Every other
    // norm launcher drops the mark, so it is live exactly while the last write
    // to `out` was this kernel.
    if (out_bf16 == nullptr && batch_size >= 1 && batch_size <= 8 &&
        GufoDevNormQ()) {
      NormQTarget t;
      if (NormQAcquire(stream, batch_size * dim, &t)) {
        BatchedRMSNormQuant5120Kernel<<<batch_size, 256, 0, stream>>>(
            x, weight, out, t.q[0], t.ds[0], t.q[1], t.ds[1], batch_size, eps);
        NormQPublish(stream, out, batch_size, dim);
        return;
      }
    }
    NormQClear(stream, out);
    BatchedRMSNormKernel<5120><<<batch_size, 256, 0, stream>>>(
        x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
        eps);
    return;
  }
  NormQClear(stream, out);
  BatchedRMSNormKernel<><<<batch_size, 256, 0, stream>>>(
      x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
      eps);
}

__global__ void BatchedPerHeadRMSNormKernel(const float* __restrict__ x,
                                            const float* __restrict__ weight,
                                            float* __restrict__ out,
                                            std::size_t batch_size,
                                            std::uint32_t num_heads,
                                            std::uint32_t head_dim, float eps) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  if (b >= batch_size || h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t head_total = static_cast<std::size_t>(num_heads) * head_dim;
  const float* x_head =
      x + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);
  float* out_head =
      out + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(head_dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out_head[i] = (x_head[i] * inv_rms) * w;
  }
}

void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps,
                                 hipStream_t stream) {
  NormQClear(stream, out);
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  BatchedPerHeadRMSNormKernel<<<grid, 256, 0, stream>>>(
      x, weight, out, batch_size, num_heads, head_dim, eps);
}
}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

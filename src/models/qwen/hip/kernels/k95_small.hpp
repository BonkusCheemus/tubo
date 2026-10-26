#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_K95_SMALL_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_K95_SMALL_HPP_

// k95small (GUFO_DEV_SMALL_K95=1): the tiny verify
// projections run as segments of ONE launch instead of one launch each.
//
// Why: a lone 3.6-4.3 MB launch cannot reach the band. Measured with the
// pure-read probe (ceil.hip): a pure read in one launch tops out at 191.5 GB/s for 3.6 MB and
// 209.3 for 4.3 MB; the served B8 K (V3 Q5_K, 32 workgroups) runs 39 us = 91
// GB/s and V (Dot4Lds Q6_K, 64 workgroups) 32 us = 134 GB/s (decode
// trace). Each is a short chain of dependent tiles on too few
// waves to cover DRAM latency; together they keep twice the bytes in flight.
//
// Exactness: every segment runs the SAME body as the served kernel (the
// *Body functions the served kernels call), with only these free choices:
//  - Waves and RowsPerLane of the V3 body, and Waves and RowsPerWave of the
//    Dot4Lds body: they only say which wave owns which row. A row's lanes,
//    its per-lane accumulation order and its butterfly are fixed by
//    RowsPerWave (V3: lanes per row = 32 / RowsPerWave, kept at the served 4)
//    and by the 32-lane sub-block walk (Dot4Lds).
//  - The order of the segments in the grid.
// So each output float is computed by the same operations in the same order.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/kernels/fast_gemv_body.hpp"
#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"

namespace gufo::hip {

// Attention K (Q5_K) + V (Q6_K), width 8, both [m, k] with the same m. Served
// routes this reproduces: K = V3 <Q5_K, 8, RowsPerWave 4, non-ActK>, V =
// Dot4Lds <Q6_K, 8, SubsPerLane 1>; both read the non-ActK int8 activations
// (QuantizeActivationsDot4Kernel) of the same x. Grid = [K blocks | V blocks].
template<std::uint32_t Waves, std::uint32_t KRowsPerLane,
         std::size_t VRowsPerWave>
__launch_bounds__(Waves * 32) __global__
    void K95KvKernel(const void* __restrict__ wk, const void* __restrict__ wv,
                     const std::int8_t* __restrict__ xq,
                     const float2* __restrict__ xds, float* __restrict__ yk,
                     float* __restrict__ yv, std::uint32_t m, std::uint32_t k,
                     std::uint32_t k_blocks) {
  // Block-uniform branch: a workgroup runs exactly one body, so each body's
  // barriers see all of its waves.
  if (blockIdx.x < k_blocks)
    SmallBatchDot4KQuantV3Body<core::GgmlType::kQ5_K, 8, 4, Waves,
                               KRowsPerLane>(wk, xq, xds, yk, m, k,
                                             blockIdx.x);
  else
    SmallBatchDot4LdsGEMMBody<core::GgmlType::kQ6_K, 8, VRowsPerWave, Waves,
                              1>(wv, xq, xds, yv, m, k, blockIdx.x - k_blocks);
}

template<std::uint32_t Waves, std::uint32_t KRowsPerLane,
         std::size_t VRowsPerWave>
void LaunchK95Kv(const void* wk, const void* wv, const std::int8_t* xq,
                 const float2* xds, float* yk, float* yv, std::uint32_t m,
                 std::uint32_t k, hipStream_t stream) {
  constexpr std::uint32_t kKRows = Waves * 4U * KRowsPerLane;
  constexpr auto kVRows = static_cast<std::uint32_t>(Waves * VRowsPerWave);
  const std::uint32_t k_blocks = (m + kKRows - 1U) / kKRows;
  const std::uint32_t v_blocks = (m + kVRows - 1U) / kVRows;
  K95KvKernel<Waves, KRowsPerLane, VRowsPerWave>
      <<<k_blocks + v_blocks, Waves * 32, 0, stream>>>(wk, wv, xq, xds, yk, yv,
                                                       m, k, k_blocks);
}

// SSM controls: beta (Q8_0 [m, k]) + alpha (F32 [m, k]) for `batch` rows of x.
// Served routes: beta = SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 1, 1> on grid
// ((m + 7) / 8, batch); alpha = FastGEMVBlockKernel on grid (m, batch), both
// 256 threads (alpha's route is kHipF32Baseline256; its block reduction
// depends on blockDim, so 256 is part of the contract). Grid = [beta | alpha]:
// beta's five dependent tiles start first, alpha's short blocks fill in.
static __launch_bounds__(256) __global__ void K95SsmControlsKernel(
    const float* __restrict__ alpha, const void* __restrict__ beta,
    const float* __restrict__ x, float* __restrict__ y_alpha,
    float* __restrict__ y_beta, std::uint32_t m, std::uint32_t k,
    std::uint32_t beta_blocks) {
  const std::uint32_t beta_x = (m + 7U) / 8U;
  if (blockIdx.x < beta_blocks) {
    SmallBatchQ8_0ExactFp32VecGEMMBody<8, 1, 1>(beta, x, y_beta, m, k, 0,
                                                blockIdx.x % beta_x,
                                                blockIdx.x / beta_x);
  } else {
    const std::uint32_t b = blockIdx.x - beta_blocks;
    FastGEMVBlockBody(alpha, core::GgmlType::kF32, x, y_alpha, m, k, b % m,
                      b / m);
  }
}

inline void LaunchK95SsmControls(const float* alpha, const void* beta,
                                 const float* x, float* y_alpha, float* y_beta,
                                 std::uint32_t batch, std::uint32_t m,
                                 std::uint32_t k, hipStream_t stream) {
  const std::uint32_t beta_blocks = ((m + 7U) / 8U) * batch;
  K95SsmControlsKernel<<<beta_blocks + (m * batch), 256, 0, stream>>>(
      alpha, beta, x, y_alpha, y_beta, m, k, beta_blocks);
}

// Preloaded SSM controls, k = 5120 only. Same operations per output, in the
// same order, as the two served bodies above, written out with every weight
// load issued before the first FMA. Why: the served bodies wait on DRAM once
// per step -- beta's five 32-block tiles sit behind barriers, alpha's five
// loop iterations each load then FMA -- so 1.24 MB takes ~23 us (decode trace:
// beta 12.7 us = 21 GB/s). What is kept, op for op (verify on GPU: bench):
//  beta (Q8_0, served <8,1,1>): one wave per (row, token); lane L owns Q8_0
//   blocks L + 32j, j = 0..4 in order; per block 32 `bd += float(q) * x`
//   in element order from 0, then `sums += d * bd`; xor butterfly 16..1;
//   lane 0 stores. x comes from global instead of the LDS stage (same floats).
//  alpha (F32, served FastGEMV, 256 threads): thread t owns float4 t + 256j,
//   j = 0..4 in order, `dot += (w.x*x.x) + (w.y*x.y) + (w.z*x.z) + (w.w*x.w)`;
//   xor butterfly 16..1; warp sums -> warp 0, lanes >= 8 add 0, xor 4..1;
//   y = sum + 0.0F. One workgroup per row covers all Batch tokens, so each
//   weight row is read once (served: one workgroup per (row, token)).
template<std::uint32_t Batch>
static __launch_bounds__(256) __global__ void K95SsmControlsPreloadKernel(
    const float* __restrict__ alpha, const void* __restrict__ beta,
    const float* __restrict__ x, float* __restrict__ y_alpha,
    float* __restrict__ y_beta, std::uint32_t m) {
  constexpr std::uint32_t kK = 5120;
  constexpr std::uint32_t kIters = kK / 4 / 256;   // alpha float4 per thread
  constexpr std::uint32_t kTiles = kK / 32 / 32;   // beta Q8_0 blocks per lane
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t beta_blocks = (m * Batch + 7U) / 8U;
  if (blockIdx.x < beta_blocks) {
    // Block-uniform branch; this side has no barrier, so a spare wave of the
    // last workgroup may return.
    const std::uint32_t task = blockIdx.x * 8U + warp;
    if (task >= m * Batch)
      return;
    const std::uint32_t token = task / m;
    const std::uint32_t row = task % m;
    const auto* blocks =
        static_cast<const Q8_0Block*>(beta) + static_cast<std::size_t>(row) * (kK / 32);
    float scale[kTiles];
    std::uint32_t packed[kTiles][8];
#pragma unroll
    for (std::uint32_t j = 0; j < kTiles; ++j) {
      const Q8_0Block* b = blocks + (j * 32U) + lane;
      scale[j] = __half2float(b->d);
#pragma unroll
      for (std::uint32_t g = 0; g < 8U; ++g)
        __builtin_memcpy(&packed[j][g], b->qs + g * 4, sizeof(std::uint32_t));
    }
    const float* xt = x + static_cast<std::size_t>(token) * kK;
    float sums = 0.0F;
#pragma unroll
    for (std::uint32_t j = 0; j < kTiles; ++j) {
      float block_dots = 0.0F;
#pragma unroll
      for (std::uint32_t g = 0; g < 8U; ++g) {
        const float4 xv = *reinterpret_cast<const float4*>(
            xt + (((j * 32U) + lane) * 32U) + (g * 4U));
        const std::uint32_t p = packed[j][g];
        block_dots +=
            static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * xv.x;
        block_dots +=
            static_cast<float>(static_cast<std::int8_t>((p >> 8U) & 0xFFU)) *
            xv.y;
        block_dots +=
            static_cast<float>(static_cast<std::int8_t>((p >> 16U) & 0xFFU)) *
            xv.z;
        block_dots +=
            static_cast<float>(static_cast<std::int8_t>((p >> 24U) & 0xFFU)) *
            xv.w;
      }
      sums += scale[j] * block_dots;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
      sums += XorLane(sums, offset);
    if (lane == 0)
      y_beta[(token * m) + row] = sums;
    return;
  }
  const std::uint32_t row = blockIdx.x - beta_blocks;
  const auto* w4 = reinterpret_cast<const float4*>(
      alpha + static_cast<std::size_t>(row) * kK);
  float4 v[kIters];
#pragma unroll
  for (std::uint32_t i = 0; i < kIters; ++i)
    v[i] = w4[threadIdx.x + (256U * i)];
  float dot[Batch];
#pragma unroll
  for (std::uint32_t t = 0; t < Batch; ++t) {
    const auto* x4 = reinterpret_cast<const float4*>(x + (t * kK));
    dot[t] = 0.0F;
#pragma unroll
    for (std::uint32_t i = 0; i < kIters; ++i) {
      const float4 xv = x4[threadIdx.x + (256U * i)];
      dot[t] += (v[i].x * xv.x) + (v[i].y * xv.y) + (v[i].z * xv.z) +
                (v[i].w * xv.w);
    }
  }
  __shared__ float warp_sums[Batch][16];
#pragma unroll
  for (std::uint32_t t = 0; t < Batch; ++t) {
    for (int offset = 16; offset > 0; offset >>= 1)
      dot[t] += XorLane(dot[t], offset);
    if (lane == 0)
      warp_sums[t][warp] = dot[t];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (std::uint32_t t = 0; t < Batch; ++t) {
      float block_sum = lane < 8U ? warp_sums[t][lane] : 0.0F;
      for (int offset = 4; offset > 0; offset >>= 1)
        block_sum += XorLane(block_sum, offset);
      if (lane == 0) {
        const float r = 0.0F;
        y_alpha[(t * m) + row] = block_sum + r;
      }
    }
  }
}

inline void LaunchK95SsmControlsPreload(const float* alpha, const void* beta,
                                        const float* x, float* y_alpha,
                                        float* y_beta, std::uint32_t batch,
                                        std::uint32_t m, hipStream_t stream) {
  const std::uint32_t grid = ((m * batch + 7U) / 8U) + m;
  const auto go = [&]<std::uint32_t Batch>() {
    K95SsmControlsPreloadKernel<Batch>
        <<<grid, 256, 0, stream>>>(alpha, beta, x, y_alpha, y_beta, m);
  };
  switch (batch) {
    case 1: go.template operator()<1>(); break;
    case 2: go.template operator()<2>(); break;
    case 3: go.template operator()<3>(); break;
    case 4: go.template operator()<4>(); break;
    case 5: go.template operator()<5>(); break;
    case 6: go.template operator()<6>(); break;
    case 7: go.template operator()<7>(); break;
    default: go.template operator()<8>(); break;
  }
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_K95_SMALL_HPP_

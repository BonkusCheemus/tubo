#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_FAST_GEMV_BODY_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_FAST_GEMV_BODY_HPP_

// k95small (10-25): FastGEMVBlockKernel's body (moved verbatim from gemv.hip)
// with explicit workgroup indices, so the fused SSM-controls launch
// (k95_small.hpp) runs the same per-output arithmetic. block_x = output row,
// block_y = activation row; the reduction depends on blockDim.x only.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/kernels/xor_lane.hpp"

namespace gufo::hip {

__device__ __forceinline__ void FastGEMVBlockBody(
    const void* __restrict__ A, core::GgmlType a_type,
    const float* __restrict__ x, float* __restrict__ y, std::size_t M,
    std::size_t K, std::uint32_t block_x, std::uint32_t block_y) {
  const std::size_t m = block_x;
  if (m >= M)
    return;
  // block_y (blockIdx.y) selects an independent row of a row-major batch
  // (x row stride K, y row stride M). Single-row launches have gridDim.y == 1,
  // so the offset is 0 and the per-block math is unchanged.
  x += block_y * K;
  y += block_y * M;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (a_type == core::GgmlType::kBF16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < K; k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += XorLane(dot, static_cast<int>(offset));
  }

  __shared__ float s_warp_sums[16];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_sum += XorLane(block_sum, static_cast<int>(offset));
    }
    if (lane_id == 0) {
      const float r = 0.0F;
      y[m] = block_sum + r;
    }
  }
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_FAST_GEMV_BODY_HPP_

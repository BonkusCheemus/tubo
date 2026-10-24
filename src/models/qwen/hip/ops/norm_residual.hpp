#ifndef GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Computes RMSNorm on GPU: out = (x / sqrt(mean(x^2) + eps)) * weight
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps = 1e-6F,
                   hipStream_t stream = nullptr);

/// Computes per-head RMSNorm across num_heads
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps = 1e-6F, hipStream_t stream = nullptr);

/// hidden += update, then out = RMSNorm(hidden) * weight, in one launch.
/// Bit-exact with LaunchResidualAdd + LaunchRMSNorm. False (nothing launched)
/// when dim != 5120.
[[nodiscard]] bool TryLaunchAddRMSNorm(float* hidden, const float* update,
                                       const float* weight, float* out,
                                       std::size_t dim, float eps,
                                       hipStream_t stream);

/// Computes residual add: out = a + b
void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream = nullptr);

/// Batched RMSNorm across B tokens (optional BF16 output in single pass)
void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps = 1e-6F,
                          hipStream_t stream = nullptr);

/// Batched Per-Head RMSNorm across B tokens
void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps = 1e-6F,
                                 hipStream_t stream = nullptr);

/// Per row: hidden += update, then out = RMSNorm(hidden) * weight, one launch.
/// Bit-exact with LaunchBatchedResidualAdd + LaunchBatchedRMSNorm (dim 5120,
/// batch <= 64, no bf16 output); false and nothing launched otherwise.
[[nodiscard]] bool TryLaunchBatchedAddRMSNorm(float* hidden,
                                              const float* update,
                                              const float* weight, float* out,
                                              std::size_t batch_size,
                                              std::size_t dim, float eps,
                                              hipStream_t stream);

/// Batched Residual Add across B tokens
void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_

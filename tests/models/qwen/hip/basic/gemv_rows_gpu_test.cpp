// LaunchGEMVRows (one row-grid launch) must be BITWISE equal to the
// per-row LaunchGEMV loop it replaces in batched_decode.cpp LaunchProjection.
// Served case: F32 ssm_alpha, M=48 (time_step_rank), K=5120, 8 verify rows.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

// Deterministic values in [-1, 1); no zeros, so a skipped row cannot pass.
float NextValue(std::uint32_t& state) {
  state = state * 1664525U + 1013904223U;
  return static_cast<float>(state >> 8) / 8388608.0F - 1.0F + 1e-7F;
}

bool CheckShape(std::size_t rows, std::size_t M, std::size_t K) {
  std::uint32_t seed = 12345U + static_cast<std::uint32_t>(M * 31 + K);
  std::vector<float> h_A(M * K), h_x(rows * K);
  for (auto& v : h_A) v = NextValue(seed);
  for (auto& v : h_x) v = NextValue(seed);
  float *d_A = nullptr, *d_x = nullptr, *d_loop = nullptr, *d_rows = nullptr;
  HIP_CHECK(hipMalloc(&d_A, h_A.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_loop, rows * M * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_rows, rows * M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_loop, 0xFF, rows * M * sizeof(float)));
  HIP_CHECK(hipMemset(d_rows, 0xFF, rows * M * sizeof(float)));
  for (std::size_t row = 0; row < rows; ++row)
    gufo::hip::LaunchGEMV(d_A, gufo::core::GgmlType::kF32, d_x + row * K,
                          d_loop + row * M, M, K);
  gufo::hip::LaunchGEMVRows(d_A, gufo::core::GgmlType::kF32, d_x, d_rows, rows,
                            M, K);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> h_loop(rows * M), h_rows(rows * M);
  HIP_CHECK(hipMemcpy(h_loop.data(), d_loop, h_loop.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_rows.data(), d_rows, h_rows.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_loop));
  HIP_CHECK(hipFree(d_rows));
  const bool equal = std::memcmp(h_loop.data(), h_rows.data(),
                                 h_loop.size() * sizeof(float)) == 0;
  std::printf("gemv_rows rows=%zu M=%zu K=%zu bitwise=%s y[0]=%.9g y[last]=%.9g\n",
              rows, M, K, equal ? "EQUAL" : "DIFF", h_rows.front(),
              h_rows.back());
  return equal;
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "GEMV rows test");
  if (device_status != gufo::test::kHipTestSuccess) return device_status;
  bool ok = true;
  for (std::size_t rows = 1; rows <= 8; ++rows) ok = CheckShape(rows, 48, 5120) && ok;
  ok = CheckShape(8, 65, 1024) && ok;
  ok = CheckShape(8, 4096, 5120) && ok;
  std::puts(ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
#else
  return 77;
#endif
}

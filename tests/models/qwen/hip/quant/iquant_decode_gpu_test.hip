// Device decode of the extra iron quants (Q2_K, IQ2_XXS, IQ2_XS,
// IQ2_S, IQ3_XXS, IQ1_M) through DecodeQuantSub16, compared BIT-EXACT with the
// CPU decode in ggml_dequant.cpp (itself bit-exact with llama.cpp, see
// tests/core/iquant_decode_test.cpp).
//
// Three device readings of every element are checked:
//   1. QuantBlockElement: the helper the embedding lookup uses.
//   2. __fsub_rn(__fmul_rn(scale, q), offset): the explicit non-fused form the
//      fp16 prefill GEMM uses to build its weight tile.
//   3. fused scale*q - offset (what the GEMV paths compile to).
// All three are exact for these formats (every product has <= 22 significant
// bits), so any difference is a decode bug, not rounding.
//
// Needs a GPU; exits 77 (skip) when there is no HIP device.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
#include "tests/core/iquant_test_blocks.hpp"

namespace {

using gufo::core::GgmlType;

__global__ void DecodeKernel(GgmlType type, const std::uint8_t* row,
                             float* helper, float* explicit_rn, float* fused,
                             int* q_range_ok, std::size_t n_sub16) {
  const std::size_t sub = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) +
                          threadIdx.x;
  if (sub >= n_sub16) {
    return;
  }
  gufo::hip::QuantSub16 d;
  gufo::hip::DecodeQuantSub16(type, row, sub, d);
  for (int j = 0; j < 16; ++j) {
    const float q = static_cast<float>(d.q[j]);
    explicit_rn[(sub * 16) + j] = __fsub_rn(__fmul_rn(d.scale, q), d.offset);
    fused[(sub * 16) + j] = (d.scale * q) - d.offset;
    helper[(sub * 16) + j] =
        gufo::hip::QuantBlockElement(type, row, (sub * 16) + j);
  }
  // IQ1_M stores 8*g +- 1, so |q| <= 9; Q2_K stores the raw 2-bit quant.
  int ok = 1;
  for (int j = 0; j < 16; ++j) {
    if (type == GgmlType::kIQ1_M) {
      ok &= (d.q[j] >= -9 && d.q[j] <= 9) ? 1 : 0;
    } else if (type == GgmlType::kQ2_K) {
      ok &= (d.q[j] >= 0 && d.q[j] <= 3) ? 1 : 0;
    }
  }
  q_range_ok[sub] = ok;
}

bool Same(const float* a, const float* b, std::size_t n, const char* what,
          const char* name) {
  std::size_t bad = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t x;
    std::uint32_t y;
    std::memcpy(&x, a + i, 4);
    std::memcpy(&y, b + i, 4);
    if (x != y && bad++ < 5) {
      std::cerr << name << " " << what << " element " << i << " device=" << a[i]
                << " cpu=" << b[i] << '\n';
    }
  }
  std::cout << (bad == 0 ? "[ OK ] " : "[FAIL] ") << name << " " << what << ": "
            << n - bad << "/" << n << " bit-identical\n";
  return bad == 0;
}

void CpuDecode(GgmlType type, const void* bytes, float* out, std::size_t k) {
  switch (type) {
    case GgmlType::kQ2_K: gufo::quant::DequantizeQ2_K(bytes, out, k); return;
    case GgmlType::kIQ2_XXS: gufo::quant::DequantizeIQ2_XXS(bytes, out, k); return;
    case GgmlType::kIQ2_XS: gufo::quant::DequantizeIQ2_XS(bytes, out, k); return;
    case GgmlType::kIQ2_S: gufo::quant::DequantizeIQ2_S(bytes, out, k); return;
    case GgmlType::kIQ3_XXS: gufo::quant::DequantizeIQ3_XXS(bytes, out, k); return;
    case GgmlType::kIQ1_M: gufo::quant::DequantizeIQ1_M(bytes, out, k); return;
    default: std::abort();
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "no HIP device: skipping\n";
    return 77;
  }
  constexpr std::size_t kBlocks = 4096;
  constexpr std::size_t kElements = kBlocks * 256;
  constexpr std::size_t kSubs = kElements / 16;
  bool all_ok = true;
  for (const auto& c : gufo::test::kIQuantCases) {
    const auto bytes =
        gufo::test::MakeIQuantBlocks(c.type, kBlocks, 0xC0DE0001U);
    std::vector<float> cpu(kElements);
    CpuDecode(c.type, bytes.data(), cpu.data(), kElements);

    std::uint8_t* d_row = nullptr;
    float* d_out[3] = {};
    int* d_ok = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_row), bytes.size()));
    for (auto& p : d_out) {
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&p),
                          kElements * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ok), kSubs * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_row, bytes.data(), bytes.size(),
                        hipMemcpyHostToDevice));
    DecodeKernel<<<(kSubs + 127) / 128, 128>>>(c.type, d_row, d_out[0],
                                               d_out[1], d_out[2], d_ok, kSubs);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> got[3];
    for (int i = 0; i < 3; ++i) {
      got[i].resize(kElements);
      HIP_CHECK(hipMemcpy(got[i].data(), d_out[i], kElements * sizeof(float),
                          hipMemcpyDeviceToHost));
    }
    std::vector<int> ok(kSubs);
    HIP_CHECK(hipMemcpy(ok.data(), d_ok, kSubs * sizeof(int),
                        hipMemcpyDeviceToHost));
    all_ok &= Same(got[0].data(), cpu.data(), kElements, "QuantBlockElement", c.name);
    all_ok &= Same(got[1].data(), cpu.data(), kElements, "scale*q-offset (rn)", c.name);
    all_ok &= Same(got[2].data(), cpu.data(), kElements, "scale*q-offset (fused)", c.name);
    std::size_t bad_q = 0;
    for (const int v : ok) {
      bad_q += (v == 0);
    }
    std::cout << (bad_q == 0 ? "[ OK ] " : "[FAIL] ") << c.name
              << " int8 quant range\n";
    all_ok &= (bad_q == 0);
    HIP_CHECK(hipFree(d_row));
    for (auto& p : d_out) {
      HIP_CHECK(hipFree(p));
    }
    HIP_CHECK(hipFree(d_ok));
  }
  if (!all_ok) {
    return 1;
  }
  std::cout << "iquant_decode_gpu_test: all checks passed\n";
  return 0;
}

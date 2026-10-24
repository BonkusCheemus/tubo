// Device half of the IQ3 unpack oracle. Native IQ3 bytes ->
// SN pair (the loader's host copy plan) -> LaunchUnpackIq3View on the GPU for
// P, G and U -> byte-equal to the native source. Real 27B pair shape and a
// small odd one, IQ3_S and IQ3_XXS. The CPU half: qwen_sn_iq3_pair_plan_test.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"

using namespace gufo::models;

namespace {
int g_failures = 0;

void Case(QwenWeightLayout layout, std::size_t rows, std::size_t k) {
  const auto bytes = Iq3PackedBytes(layout, rows, k);
  const auto type = layout == QwenWeightLayout::kIq3SRows8
                        ? gufo::core::GgmlType::kIQ3_S
                        : gufo::core::GgmlType::kIQ3_XXS;
  std::vector<std::uint8_t> native(bytes), packed(bytes);
  std::uint64_t seed = 0x9e3779b97f4a7c15ULL ^ bytes;
  for (auto& x : native) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    x = static_cast<std::uint8_t>(seed);
  }
  CopyIq3RowRangeFast(native.data(), packed.data(), layout, rows, k, 0, rows);
  void* d_packed = nullptr;
  void* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_packed, bytes));
  HIP_CHECK(hipMalloc(&d_out, bytes + 64));
  HIP_CHECK(hipMemcpy(d_packed, packed.data(), bytes, hipMemcpyHostToDevice));
  const auto views = MakeIq3PairViews(d_packed, type, rows, k, bytes);
  std::vector<std::uint8_t> out(bytes);
  for (const auto& view : {views.pair, views.gate, views.up}) {
    const auto n = view.EncodedSizeBytes();
    HIP_CHECK(hipMemset(d_out, 0xcd, bytes + 64));
    gufo::hip::LaunchUnpackIq3View(view, d_out, bytes + 64, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out.data(), d_out, n, hipMemcpyDeviceToHost));
    std::uint8_t guard[64];
    HIP_CHECK(hipMemcpy(guard, static_cast<std::uint8_t*>(d_out) + n,
                        sizeof guard, hipMemcpyDeviceToHost));
    const auto* expect =
        native.data() + view.layout_row_begin * (k / 256) * Iq3BlockBytes(layout);
    const bool equal = std::memcmp(out.data(), expect, n) == 0;
    bool untouched = true;
    for (auto g : guard) untouched &= g == 0xcd;
    std::printf("%s layout=%s rows=%zu k=%zu row_begin=%zu bytes=%zu\n",
                equal && untouched ? "PASS" : "FAIL",
                layout == QwenWeightLayout::kIq3SRows8 ? "S8" : "XXS4", rows,
                k, view.layout_row_begin, n);
    g_failures += !(equal && untouched);
  }
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_packed));
}
}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  for (const auto layout :
       {QwenWeightLayout::kIq3SRows8, QwenWeightLayout::kIq3XxsRows4}) {
    Case(layout, 34816, 5120);
    Case(layout, 6 * Iq3GroupRows(layout), 768);
  }
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

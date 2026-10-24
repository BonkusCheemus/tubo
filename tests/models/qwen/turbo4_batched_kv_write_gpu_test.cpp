// The DEVICE half: the batched launch must produce the same
// plane BYTES as N single-position launches of the writer the engine ships.
//
// The CPU half (turbo4_batched_kv_write_test.cpp) proves the range BODY. This
// proves the KERNEL: the grid, the chunk-major source indexing on device, and
// the latched status. A body can be right and a kernel still drop the
// block-within-head term or the per-position stride, because the kernel indexes
// the SOURCE itself before handing a slice to the body.
//
//   /opt/rocm/bin/hipcc -std=c++20 -DENGINE_ENABLE_HIP -I. -x hip \
//       tests/models/qwen/turbo4_batched_kv_write_gpu_test.cpp -o /tmp/e1gpu \
//       -Lbuild-c1-turbo4-decode -lgufo_core

// hip_runtime.h FIRST. <cstring> declares a host-only ::memcpy that shadows
// the device one the Turbo4 headers' __host__ __device__ bodies call, so the
// order of these includes is load-bearing, not cosmetic.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"
#include "src/models/qwen/hip/turbo4_plane_index.hpp"

namespace t4i = gufo::hip;
namespace t4 = gufo::hip::turbo4;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok)
    ++g_failures;
}

#define HIP_OK(expr)                                                      \
  do {                                                                    \
    const hipError_t _e = (expr);                                         \
    if (_e != hipSuccess) {                                               \
      std::printf("[FAIL] hip error %d at %s:%d\n", static_cast<int>(_e), \
                  __FILE__, __LINE__);                                    \
      ++g_failures;                                                       \
      return;                                                             \
    }                                                                     \
  } while (0)

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kContext = 1024;

float Noise(std::size_t seed) {
  std::uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  const float unit = static_cast<float>(x >> 40) / 16777216.0f;
  return (unit - 0.5f) * 6.0f;
}

std::vector<std::uint8_t> ReadGoldenBlock(const std::string& scale) {
  const auto path =
      std::filesystem::path(__FILE__).parent_path() / "turbo4_golden.txt";
  std::ifstream input(path);
  const std::string prefix = "scale=" + scale + ":";
  std::string line;
  while (std::getline(input, line)) {
    if (line.compare(0, prefix.size(), prefix) != 0)
      continue;
    const std::string hex = line.substr(prefix.size());
    if (hex.size() != t4::kBlockBytes * 2)
      return {};
    std::vector<std::uint8_t> bytes(t4::kBlockBytes);
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::uint8_t>(
          std::stoul(hex.substr(2 * i, 2), nullptr, 16));
    return bytes;
  }
  return {};
}

void GoldenWriteOnDevice() {
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kContext);
  const std::size_t source_elems =
      static_cast<std::size_t>(kKvHeads) * kHeadDim;
  const std::size_t source_bytes = source_elems * sizeof(float);
  std::vector<float> hk(source_elems, 0.0F), hv(source_elems, 0.0F);
  const auto fill_block = [](float* dst, float scale) {
    for (std::size_t i = 0; i < t4::kBlockValues; ++i)
      dst[i] = scale * std::sin(static_cast<float>(i) * 0.37F) +
               0.01F * static_cast<float>(i);
  };
  fill_block(hk.data(), 0.01F);
  fill_block(hk.data() + t4::kBlockValues, 1.0F);
  fill_block(hv.data(), 37.0F);
  fill_block(hv.data() + t4::kBlockValues, 1000.0F);

  float *dk = nullptr, *dv = nullptr;
  std::uint8_t *k_plane = nullptr, *v_plane = nullptr;
  std::uint32_t* d_pos = nullptr;
  auto* d_status = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  HIP_OK(hipMalloc(&dk, source_bytes));
  HIP_OK(hipMalloc(&dv, source_bytes));
  HIP_OK(hipMalloc(&k_plane, plane_bytes));
  HIP_OK(hipMalloc(&v_plane, plane_bytes));
  HIP_OK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
  HIP_OK(hipMalloc(&d_status, sizeof(*d_status)));
  HIP_OK(hipMemcpy(dk, hk.data(), source_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dv, hv.data(), source_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemset(k_plane, 0xA5, plane_bytes));
  HIP_OK(hipMemset(v_plane, 0x5A, plane_bytes));
  const std::uint32_t pos = 0;
  HIP_OK(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice));
  HIP_OK(hipMemset(d_status, 0, sizeof(*d_status)));

  const auto status = t4i::LaunchTurbo4KvWrite(
      dk, dv, k_plane, v_plane, d_pos, plane_bytes, source_bytes, kContext,
      kKvHeads, kHeadDim, d_status, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(status == t4i::Turbo4LaunchStatus::kLaunched,
        "golden writer launch accepted");
  HIP_OK(hipDeviceSynchronize());

  std::vector<std::uint8_t> got_k(2 * t4::kBlockBytes);
  std::vector<std::uint8_t> got_v(2 * t4::kBlockBytes);
  HIP_OK(hipMemcpy(got_k.data(), k_plane, got_k.size(), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(got_v.data(), v_plane, got_v.size(), hipMemcpyDeviceToHost));
  const std::vector<std::string> scales{"0.01", "1.0", "37.0", "1000.0"};
  for (std::size_t block = 0; block < 2; ++block) {
    const auto expected_k = ReadGoldenBlock(scales[block]);
    const auto expected_v = ReadGoldenBlock(scales[block + 2]);
    const auto k_begin = got_k.begin() + block * t4::kBlockBytes;
    const auto v_begin = got_v.begin() + block * t4::kBlockBytes;
    Check(expected_k.size() == t4::kBlockBytes &&
              std::equal(expected_k.begin(), expected_k.end(), k_begin),
          "device K block matches Strix golden scale=" + scales[block]);
    Check(expected_v.size() == t4::kBlockBytes &&
              std::equal(expected_v.begin(), expected_v.end(), v_begin),
          "device V block matches Strix golden scale=" + scales[block + 2]);
  }
  (void)hipFree(dk);
  (void)hipFree(dv);
  (void)hipFree(k_plane);
  (void)hipFree(v_plane);
  (void)hipFree(d_pos);
  (void)hipFree(d_status);
}

void FusedPrepareMatchesReferenceOnDevice() {
  constexpr std::uint32_t kContext = 1024;
  constexpr std::uint32_t kPosition = 0;
  constexpr std::uint32_t kRotaryDim = 64;
  constexpr std::size_t q_values =
      static_cast<std::size_t>(t4i::detail::kTiledAttentionQueryHeads) *
      kHeadDim;
  constexpr std::size_t kv_values =
      static_cast<std::size_t>(kKvHeads) * kHeadDim;
  const std::size_t q_bytes = q_values * sizeof(float);
  const std::size_t kv_bytes = kv_values * sizeof(float);
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kContext);
  const std::size_t partial_bytes =
      t4i::Turbo4DecodePartialBytes(kContext, 24U, kKvHeads, kHeadDim);
  const std::size_t workspace_bytes =
      t4i::Turbo4DecodeWorkspaceBytes(kContext, 24U, kKvHeads, kHeadDim);
  std::vector<float> hq(q_values), hk(kv_values), hv(kv_values);
  std::vector<float> hq_weight(kHeadDim, 1.0F), hk_weight(kHeadDim, 1.0F);
  for (std::size_t i = 0; i < hq.size(); ++i)
    hq[i] = Noise(i + 1001);
  for (std::size_t i = 0; i < hk.size(); ++i)
    hk[i] = Noise(i + 2001);
  for (std::size_t i = 0; i < hv.size(); ++i)
    hv[i] = Noise(i + 3001);

  float *q_raw = nullptr, *k_raw = nullptr, *v = nullptr;
  float *q_weight = nullptr, *k_weight = nullptr;
  float *q_ref = nullptr, *k_ref = nullptr, *q_rot_fused = nullptr;
  float *out = nullptr, *workspace = nullptr;
  std::uint8_t *k_ref_plane = nullptr, *v_ref_plane = nullptr;
  std::uint8_t *k_fused_plane = nullptr, *v_fused_plane = nullptr;
  std::uint32_t* d_pos = nullptr;
  auto* d_write_ref = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  auto* d_read_ref = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  auto* d_write_fused = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  auto* d_read_fused = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  HIP_OK(hipMalloc(&q_raw, q_bytes));
  HIP_OK(hipMalloc(&k_raw, kv_bytes));
  HIP_OK(hipMalloc(&v, kv_bytes));
  HIP_OK(hipMalloc(&q_weight, kHeadDim * sizeof(float)));
  HIP_OK(hipMalloc(&k_weight, kHeadDim * sizeof(float)));
  HIP_OK(hipMalloc(&q_ref, q_bytes));
  HIP_OK(hipMalloc(&k_ref, kv_bytes));
  HIP_OK(hipMalloc(&q_rot_fused, q_bytes));
  HIP_OK(hipMalloc(&out, q_bytes));
  HIP_OK(hipMalloc(&workspace, workspace_bytes));
  HIP_OK(hipMalloc(&k_ref_plane, plane_bytes));
  HIP_OK(hipMalloc(&v_ref_plane, plane_bytes));
  HIP_OK(hipMalloc(&k_fused_plane, plane_bytes));
  HIP_OK(hipMalloc(&v_fused_plane, plane_bytes));
  HIP_OK(hipMalloc(&d_pos, sizeof(kPosition)));
  HIP_OK(hipMalloc(&d_write_ref, sizeof(*d_write_ref)));
  HIP_OK(hipMalloc(&d_read_ref, sizeof(*d_read_ref)));
  HIP_OK(hipMalloc(&d_write_fused, sizeof(*d_write_fused)));
  HIP_OK(hipMalloc(&d_read_fused, sizeof(*d_read_fused)));
  HIP_OK(hipMemcpy(q_raw, hq.data(), q_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(k_raw, hk.data(), kv_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(v, hv.data(), kv_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(q_weight, hq_weight.data(), kHeadDim * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(k_weight, hk_weight.data(), kHeadDim * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(
      hipMemcpy(d_pos, &kPosition, sizeof(kPosition), hipMemcpyHostToDevice));
  HIP_OK(hipMemset(k_ref_plane, 0xA5, plane_bytes));
  HIP_OK(hipMemset(v_ref_plane, 0x5A, plane_bytes));
  HIP_OK(hipMemset(k_fused_plane, 0xA5, plane_bytes));
  HIP_OK(hipMemset(v_fused_plane, 0x5A, plane_bytes));
  HIP_OK(hipMemset(d_write_ref, 0, sizeof(*d_write_ref)));
  HIP_OK(hipMemset(d_read_ref, 0, sizeof(*d_read_ref)));
  HIP_OK(hipMemset(d_write_fused, 0, sizeof(*d_write_fused)));
  HIP_OK(hipMemset(d_read_fused, 0, sizeof(*d_read_fused)));

  // Reference: current separate Q/K norm + RoPE, writer, and E5 rotate.
  HIP_OK(hipMemcpy(q_ref, q_raw, q_bytes, hipMemcpyDeviceToDevice));
  HIP_OK(hipMemcpy(k_ref, k_raw, kv_bytes, hipMemcpyDeviceToDevice));
  t4i::LaunchPerHeadRMSNorm(q_ref, q_weight, q_ref, 24U, kHeadDim, 1e-6F,
                            nullptr);
  t4i::LaunchPerHeadRMSNorm(k_ref, k_weight, k_ref, kKvHeads, kHeadDim, 1e-6F,
                            nullptr);
  t4i::LaunchRoPE(q_ref, k_ref, 24U, kKvHeads, kHeadDim, kRotaryDim, d_pos,
                  10000000.0F, nullptr, nullptr);
  const auto writer = t4i::LaunchTurbo4KvWrite(
      k_ref, v, k_ref_plane, v_ref_plane, d_pos, plane_bytes, kv_bytes,
      kContext, kKvHeads, kHeadDim, d_write_ref, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  auto* partials = workspace;
  auto* q_rot_ref = workspace + partial_bytes / sizeof(float);
  auto* out_rot = q_rot_ref + q_values;
  const auto reader = t4i::LaunchTurbo4AttentionSplit(
      q_ref, k_ref_plane, v_ref_plane, out, q_rot_ref, out_rot, partials,
      plane_bytes, q_bytes, q_bytes, q_bytes, q_bytes, partial_bytes, nullptr,
      0U, 24U, kKvHeads, kHeadDim, 1U, kContext,
      1.0F / std::sqrt(static_cast<float>(kHeadDim)), d_write_ref, d_read_ref,
      nullptr, false, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(writer == t4i::Turbo4LaunchStatus::kLaunched &&
            reader == t4i::Turbo4LaunchStatus::kLaunched,
        "fused-prep reference launches accepted");
  HIP_OK(hipDeviceSynchronize());

  const auto fused = t4i::LaunchTurbo4DecodePrepare(
      q_raw, k_raw, v, q_weight, k_weight, q_rot_fused, k_fused_plane,
      v_fused_plane, d_pos, q_bytes, kv_bytes, q_bytes, plane_bytes, 24U,
      kKvHeads, kHeadDim, kRotaryDim, kContext, 10000000.0F, 1e-6F,
      d_write_fused, d_read_fused, nullptr, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(fused == t4i::Turbo4LaunchStatus::kLaunched,
        "fused-prep launch accepted");
  HIP_OK(hipDeviceSynchronize());

  std::vector<float> q_ref_host(q_values), q_fused_host(q_values);
  std::vector<std::uint8_t> k_ref_host(plane_bytes), v_ref_host(plane_bytes);
  std::vector<std::uint8_t> k_fused_host(plane_bytes),
      v_fused_host(plane_bytes);
  t4i::Turbo4KernelStatus write_ref_status{}, read_ref_status{};
  t4i::Turbo4KernelStatus write_fused_status{}, read_fused_status{};
  HIP_OK(
      hipMemcpy(q_ref_host.data(), q_rot_ref, q_bytes, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(q_fused_host.data(), q_rot_fused, q_bytes,
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(k_ref_host.data(), k_ref_plane, plane_bytes,
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(v_ref_host.data(), v_ref_plane, plane_bytes,
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(k_fused_host.data(), k_fused_plane, plane_bytes,
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(v_fused_host.data(), v_fused_plane, plane_bytes,
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(&write_ref_status, d_write_ref, sizeof(write_ref_status),
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(&read_ref_status, d_read_ref, sizeof(read_ref_status),
                   hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(&write_fused_status, d_write_fused,
                   sizeof(write_fused_status), hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(&read_fused_status, d_read_fused, sizeof(read_fused_status),
                   hipMemcpyDeviceToHost));
  Check(write_ref_status == t4i::Turbo4KernelStatus::kOk &&
            read_ref_status == t4i::Turbo4KernelStatus::kOk &&
            write_fused_status == t4i::Turbo4KernelStatus::kOk &&
            read_fused_status == t4i::Turbo4KernelStatus::kOk,
        "fused-prep and reference status words remain kOk");
  Check(
      std::memcmp(q_ref_host.data(), q_fused_host.data(), q_bytes) == 0,
      "fused-prep q_rot is bit-equal to the separate norm/RoPE/E5 rotate path");
  Check(k_ref_host == k_fused_host && v_ref_host == v_fused_host,
        "fused-prep K/V plane bytes match the separate writer path");

  (void)hipFree(q_raw);
  (void)hipFree(k_raw);
  (void)hipFree(v);
  (void)hipFree(q_weight);
  (void)hipFree(k_weight);
  (void)hipFree(q_ref);
  (void)hipFree(k_ref);
  (void)hipFree(q_rot_fused);
  (void)hipFree(out);
  (void)hipFree(workspace);
  (void)hipFree(k_ref_plane);
  (void)hipFree(v_ref_plane);
  (void)hipFree(k_fused_plane);
  (void)hipFree(v_fused_plane);
  (void)hipFree(d_pos);
  (void)hipFree(d_write_ref);
  (void)hipFree(d_read_ref);
  (void)hipFree(d_write_fused);
  (void)hipFree(d_read_fused);
}

void CompareOnDevice(std::uint32_t count, std::uint32_t first_pos) {
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kContext);
  const std::size_t pos_stride = kKvHeads * kHeadDim;
  const std::size_t src_elems = static_cast<std::size_t>(count) * pos_stride;
  const std::size_t src_bytes = src_elems * sizeof(float);

  std::vector<float> hk(src_elems), hv(src_elems);
  for (std::size_t i = 0; i < src_elems; ++i)
    hk[i] = Noise(i + 1);
  for (std::size_t i = 0; i < src_elems; ++i)
    hv[i] = Noise(i + 555);

  float *dk = nullptr, *dv = nullptr;
  std::uint8_t *k_bat = nullptr, *v_bat = nullptr, *k_one = nullptr,
               *v_one = nullptr;
  std::uint32_t* d_pos = nullptr;
  std::uint8_t* d_status = nullptr;
  HIP_OK(hipMalloc(&dk, src_bytes));
  HIP_OK(hipMalloc(&dv, src_bytes));
  HIP_OK(hipMalloc(&k_bat, plane_bytes));
  HIP_OK(hipMalloc(&v_bat, plane_bytes));
  HIP_OK(hipMalloc(&k_one, plane_bytes));
  HIP_OK(hipMalloc(&v_one, plane_bytes));
  HIP_OK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
  HIP_OK(hipMalloc(&d_status, 16));
  HIP_OK(hipMemcpy(dk, hk.data(), src_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dv, hv.data(), src_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemset(k_bat, 0xAB, plane_bytes));
  HIP_OK(hipMemset(v_bat, 0xCD, plane_bytes));
  HIP_OK(hipMemset(k_one, 0xAB, plane_bytes));
  HIP_OK(hipMemset(v_one, 0xCD, plane_bytes));

  const std::string tag = "count=" + std::to_string(count) +
                          " first_pos=" + std::to_string(first_pos);

  // ---- ARM A: one batched launch ----
  HIP_OK(hipMemcpy(d_pos, &first_pos, sizeof(std::uint32_t),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemset(d_status, 0, 16));
  const auto st_bat = t4i::LaunchTurbo4KvWriteBatched(
      dk, dv, k_bat, v_bat, d_pos, count, plane_bytes, src_bytes, kContext,
      kKvHeads, kHeadDim, reinterpret_cast<t4i::Turbo4KernelStatus*>(d_status),
      nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  HIP_OK(hipDeviceSynchronize());

  // ---- ARM B: N single-position launches ----
  HIP_OK(hipMemset(d_status, 0, 16));
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t pos = first_pos + i;
    HIP_OK(
        hipMemcpy(d_pos, &pos, sizeof(std::uint32_t), hipMemcpyHostToDevice));
    const auto st = t4i::LaunchTurbo4KvWrite(
        dk + static_cast<std::size_t>(i) * pos_stride,
        dv + static_cast<std::size_t>(i) * pos_stride, k_one, v_one, d_pos,
        plane_bytes, pos_stride * sizeof(float), kContext, kKvHeads, kHeadDim,
        reinterpret_cast<t4i::Turbo4KernelStatus*>(d_status), nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    if (st != t4i::Turbo4LaunchStatus::kLaunched) {
      Check(false, tag + ": single launch refused");
      break;
    }
  }
  HIP_OK(hipDeviceSynchronize());

  std::vector<std::uint8_t> a(plane_bytes), b(plane_bytes);
  HIP_OK(hipMemcpy(a.data(), k_bat, plane_bytes, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(b.data(), k_one, plane_bytes, hipMemcpyDeviceToHost));
  const bool k_same = std::memcmp(a.data(), b.data(), plane_bytes) == 0;

  HIP_OK(hipMemcpy(a.data(), v_bat, plane_bytes, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(b.data(), v_one, plane_bytes, hipMemcpyDeviceToHost));
  const bool v_same = std::memcmp(a.data(), b.data(), plane_bytes) == 0;

  Check(st_bat == t4i::Turbo4LaunchStatus::kLaunched,
        tag + ": batched launch reports kLaunched");
  Check(k_same && v_same,
        tag + ": DEVICE batched plane is BYTE-EQUAL to N single launches");

  (void)hipFree(dk);
  (void)hipFree(dv);
  (void)hipFree(k_bat);
  (void)hipFree(v_bat);
  (void)hipFree(k_one);
  (void)hipFree(v_one);
  (void)hipFree(d_pos);
  (void)hipFree(d_status);
}

void MicrobenchSingleWriter() {
  constexpr std::uint32_t kLaunches = 1000;
  constexpr std::uint32_t kWarmup = 20;
  constexpr std::uint32_t kRepeats = 3;
  const std::size_t pos_stride = kKvHeads * kHeadDim;
  const std::size_t source_bytes = pos_stride * sizeof(float);
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kContext);

  std::vector<float> hk(pos_stride), hv(pos_stride);
  for (std::size_t i = 0; i < pos_stride; ++i) {
    hk[i] = Noise(i + 1);
    hv[i] = Noise(i + 555);
  }
  float *dk = nullptr, *dv = nullptr;
  std::uint8_t *k_plane = nullptr, *v_plane = nullptr;
  std::uint32_t* d_pos = nullptr;
  auto* d_status = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  HIP_OK(hipMalloc(&dk, source_bytes));
  HIP_OK(hipMalloc(&dv, source_bytes));
  HIP_OK(hipMalloc(&k_plane, plane_bytes));
  HIP_OK(hipMalloc(&v_plane, plane_bytes));
  HIP_OK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
  HIP_OK(hipMalloc(&d_status, sizeof(*d_status)));
  HIP_OK(hipMemcpy(dk, hk.data(), source_bytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dv, hv.data(), source_bytes, hipMemcpyHostToDevice));
  constexpr std::uint32_t kPosition = 8;
  HIP_OK(
      hipMemcpy(d_pos, &kPosition, sizeof(kPosition), hipMemcpyHostToDevice));
  HIP_OK(hipMemset(d_status, 0, sizeof(*d_status)));

  auto launch = [&]() {
    return t4i::LaunchTurbo4KvWrite(
        dk, dv, k_plane, v_plane, d_pos, plane_bytes, source_bytes, kContext,
        kKvHeads, kHeadDim, d_status, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  };
  for (std::uint32_t i = 0; i < kWarmup; ++i) {
    if (launch() != t4i::Turbo4LaunchStatus::kLaunched) {
      Check(false, "writer microbench warmup launch accepted");
      return;
    }
  }
  HIP_OK(hipDeviceSynchronize());

  hipEvent_t begin = nullptr, end = nullptr;
  HIP_OK(hipEventCreate(&begin));
  HIP_OK(hipEventCreate(&end));
  std::printf("E7 single Turbo4 writer: %u launches per repeat, %u repeats\n",
              kLaunches, kRepeats);
  for (std::uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
    HIP_OK(hipEventRecord(begin, nullptr));
    const auto host_begin = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < kLaunches; ++i) {
      if (launch() != t4i::Turbo4LaunchStatus::kLaunched) {
        Check(false, "writer microbench launch accepted");
        return;
      }
    }
    const auto host_end = std::chrono::steady_clock::now();
    HIP_OK(hipEventRecord(end, nullptr));
    HIP_OK(hipEventSynchronize(end));
    float elapsed_ms = 0.0F;
    HIP_OK(hipEventElapsedTime(&elapsed_ms, begin, end));
    const double event_us_per_launch =
        static_cast<double>(elapsed_ms) * 1000.0 / kLaunches;
    const double host_us_per_launch =
        std::chrono::duration<double, std::micro>(host_end - host_begin)
            .count() /
        kLaunches;
    std::printf(
        "repeat=%u event_us_per_launch=%.3f host_enqueue_us_per_launch=%.3f\n",
        repeat, event_us_per_launch, host_us_per_launch);
  }
  HIP_OK(hipEventDestroy(begin));
  HIP_OK(hipEventDestroy(end));
  (void)hipFree(dk);
  (void)hipFree(dv);
  (void)hipFree(k_plane);
  (void)hipFree(v_plane);
  (void)hipFree(d_pos);
  (void)hipFree(d_status);
}

struct LaunchTiming {
  double event_us_per_launch{0.0};
  double host_us_per_launch{0.0};
};

template<typename Launch>
LaunchTiming MeasureLaunches(const char* label, std::uint32_t context,
                             Launch&& launch) {
  constexpr std::uint32_t kLaunches = 1000;
  constexpr std::uint32_t kWarmup = 20;
  constexpr std::uint32_t kRepeats = 3;
  hipEvent_t begin = nullptr, end = nullptr;
  if (hipEventCreate(&begin) != hipSuccess ||
      hipEventCreate(&end) != hipSuccess) {
    Check(false, std::string(label) + ": HIP event creation");
    return {};
  }
  for (std::uint32_t i = 0; i < kWarmup; ++i) {
    if (!launch()) {
      Check(false, std::string(label) + ": warmup launch accepted");
      return {};
    }
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    Check(false, std::string(label) + ": warmup sync");
    return {};
  }

  LaunchTiming result{};
  for (std::uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
    if (hipEventRecord(begin, nullptr) != hipSuccess) {
      Check(false, std::string(label) + ": start event");
      return {};
    }
    const auto host_begin = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < kLaunches; ++i) {
      if (!launch()) {
        Check(false, std::string(label) + ": timed launch accepted");
        return {};
      }
    }
    const auto host_end = std::chrono::steady_clock::now();
    if (hipEventRecord(end, nullptr) != hipSuccess ||
        hipEventSynchronize(end) != hipSuccess) {
      Check(false, std::string(label) + ": end event sync");
      return {};
    }
    float elapsed_ms = 0.0F;
    if (hipEventElapsedTime(&elapsed_ms, begin, end) != hipSuccess) {
      Check(false, std::string(label) + ": event elapsed time");
      return {};
    }
    const double event_us =
        static_cast<double>(elapsed_ms) * 1000.0 / kLaunches;
    const double host_us =
        std::chrono::duration<double, std::micro>(host_end - host_begin)
            .count() /
        kLaunches;
    result.event_us_per_launch += event_us / kRepeats;
    result.host_us_per_launch += host_us / kRepeats;
    std::printf(
        "context=%u op=%s repeat=%u event_us=%.3f host_enqueue_us=%.3f\n",
        context, label, repeat, event_us, host_us);
  }
  (void)hipEventDestroy(begin);
  (void)hipEventDestroy(end);
  std::printf("context=%u op=%s mean_event_us=%.3f mean_host_enqueue_us=%.3f\n",
              context, label, result.event_us_per_launch,
              result.host_us_per_launch);
  return result;
}

void ProfileDecodeCosts() {
  constexpr std::uint32_t kNumHeads = 24;
  constexpr std::uint32_t kNumKvHeads = 4;
  constexpr std::uint32_t kHeadDim = 256;
  constexpr std::uint32_t kRotaryDim = 64;
  constexpr std::uint32_t kMaxContext = 8320;  // bench depth 8192 + tg128
  constexpr std::uint32_t kLayersPerToken = 16;
  constexpr std::size_t kVectorValues =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  constexpr std::size_t kKvValues =
      static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  constexpr std::size_t kVectorBytes = kVectorValues * sizeof(float);
  constexpr std::size_t kKvBytes = kKvValues * sizeof(float);
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kNumKvHeads, kMaxContext);
  const std::size_t partial_bytes = t4i::Turbo4DecodePartialBytes(
      kMaxContext, kNumHeads, kNumKvHeads, kHeadDim);
  const std::size_t workspace_bytes = t4i::Turbo4DecodeWorkspaceBytes(
      kMaxContext, kNumHeads, kNumKvHeads, kHeadDim);
  const std::size_t f_split_values =
      t4i::detail::DecodeAttentionScratchElements(kNumHeads, kHeadDim);
  const std::size_t f_cache_bytes =
      static_cast<std::size_t>(kMaxContext) * kKvValues * sizeof(std::uint16_t);
  if (plane_bytes == 0U || partial_bytes == 0U || workspace_bytes == 0U ||
      f_split_values == 0U) {
    Check(false, "profile workspace sizes are nonzero");
    return;
  }

  std::vector<float> hq(kVectorValues), hk(kKvValues), hv(kKvValues);
  std::vector<float> hq_weight(kHeadDim, 1.0F), hk_weight(kHeadDim, 1.0F);
  std::vector<float> hgate(kVectorValues, 0.5F);
  for (std::size_t i = 0; i < hq.size(); ++i)
    hq[i] = Noise(i + 17);
  for (std::size_t i = 0; i < hk.size(); ++i)
    hk[i] = Noise(i + 31);
  for (std::size_t i = 0; i < hv.size(); ++i)
    hv[i] = Noise(i + 47);

  float *q = nullptr, *k = nullptr, *v = nullptr, *gate = nullptr;
  float *q_out = nullptr, *k_out = nullptr, *out = nullptr;
  float *q_rot = nullptr, *out_rot = nullptr, *partials = nullptr;
  float* f_split = nullptr;
  float *q_weight = nullptr, *k_weight = nullptr;
  std::uint16_t *k_f16 = nullptr, *v_f16 = nullptr;
  std::uint8_t *k_turbo4 = nullptr, *v_turbo4 = nullptr;
  std::uint32_t* d_pos = nullptr;
  auto* d_write_status = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  auto* d_read_status = static_cast<t4i::Turbo4KernelStatus*>(nullptr);
  auto* workspace = static_cast<float*>(nullptr);
  HIP_OK(hipMalloc(&q, kVectorBytes));
  HIP_OK(hipMalloc(&k, kKvBytes));
  HIP_OK(hipMalloc(&v, kKvBytes));
  HIP_OK(hipMalloc(&gate, kVectorBytes));
  HIP_OK(hipMalloc(&q_out, kVectorBytes));
  HIP_OK(hipMalloc(&k_out, kKvBytes));
  HIP_OK(hipMalloc(&out, kVectorBytes));
  HIP_OK(hipMalloc(&f_split, f_split_values * sizeof(float)));
  HIP_OK(hipMalloc(&q_weight, kHeadDim * sizeof(float)));
  HIP_OK(hipMalloc(&k_weight, kHeadDim * sizeof(float)));
  HIP_OK(hipMalloc(&k_f16, f_cache_bytes));
  HIP_OK(hipMalloc(&v_f16, f_cache_bytes));
  HIP_OK(hipMalloc(&k_turbo4, plane_bytes));
  HIP_OK(hipMalloc(&v_turbo4, plane_bytes));
  HIP_OK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
  HIP_OK(hipMalloc(&d_write_status, sizeof(*d_write_status)));
  HIP_OK(hipMalloc(&d_read_status, sizeof(*d_read_status)));
  HIP_OK(hipMalloc(&workspace, workspace_bytes));
  HIP_OK(hipMemcpy(q, hq.data(), kVectorBytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(k, hk.data(), kKvBytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(v, hv.data(), kKvBytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(gate, hgate.data(), kVectorBytes, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(q_weight, hq_weight.data(), kHeadDim * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(k_weight, hk_weight.data(), kHeadDim * sizeof(float),
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemset(k_f16, 0, f_cache_bytes));
  HIP_OK(hipMemset(v_f16, 0, f_cache_bytes));
  HIP_OK(hipMemset(k_turbo4, 0, plane_bytes));
  HIP_OK(hipMemset(v_turbo4, 0, plane_bytes));
  HIP_OK(hipMemset(d_write_status, 0, sizeof(*d_write_status)));
  HIP_OK(hipMemset(d_read_status, 0, sizeof(*d_read_status)));
  const std::size_t q_rot_offset = partial_bytes / sizeof(float);
  q_rot = workspace + q_rot_offset;
  out_rot = q_rot + kVectorValues;
  partials = workspace;

  std::printf(
      "E7 fused-prestep component microbench: T and F; "
      "1000 launches x 3 repeats, %u full-attention layers/token\n",
      kLayersPerToken);
  for (const std::uint32_t context : {1U, 32U, 512U, 8192U}) {
    const std::uint32_t pos = context - 1U;
    HIP_OK(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_write_status, 0, sizeof(*d_write_status)));
    HIP_OK(hipMemset(d_read_status, 0, sizeof(*d_read_status)));

    const auto t_prepare = MeasureLaunches("T.fused_prepare", context, [&] {
      return t4i::LaunchTurbo4DecodePrepare(
                 q, k, v, q_weight, k_weight, q_rot, k_turbo4, v_turbo4, d_pos,
                 kVectorBytes, kKvBytes, kVectorBytes, plane_bytes, kNumHeads,
                 kNumKvHeads, kHeadDim, kRotaryDim, kMaxContext, 10000000.0F,
                 1e-6F, d_write_status, d_read_status, nullptr, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)) ==
             t4i::Turbo4LaunchStatus::kLaunched;
    });
    const auto t_reader = MeasureLaunches("T.split_reduce", context, [&] {
      return t4i::LaunchTurbo4AttentionSplit(
                 q_rot, k_turbo4, v_turbo4, out, q_rot, out_rot, partials,
                 plane_bytes, kVectorBytes, kVectorBytes, kVectorBytes,
                 kVectorBytes, partial_bytes, gate, kVectorBytes, kNumHeads,
                 kNumKvHeads, kHeadDim, context, kMaxContext,
                 1.0F / std::sqrt(static_cast<float>(kHeadDim)), d_write_status,
                 d_read_status, nullptr, true, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)) ==
             t4i::Turbo4LaunchStatus::kLaunched;
    });

    const auto f_fused = MeasureLaunches("F.fused_qk_rope_kv", context, [&] {
      t4i::LaunchFusedQKNormRoPEKvWrite(
          q, k, v, q_weight, k_weight, q_out, k_out, nullptr, nullptr, k_f16,
          v_f16, 0, d_pos, kMaxContext, kNumHeads, kNumKvHeads, kHeadDim,
          kRotaryDim, 10000000.0F, 1e-6F, nullptr, nullptr);
      return true;
    });
    const auto f_attention = MeasureLaunches("F.attention", context, [&] {
      t4i::LaunchAttention(q_out, k_out, v, gate, nullptr, nullptr, k_f16,
                           v_f16, out, 0, pos, kMaxContext, kNumHeads,
                           kNumKvHeads, kHeadDim, nullptr, f_split, true);
      return true;
    });

    const double t_layer_us =
        t_prepare.event_us_per_launch + t_reader.event_us_per_launch;
    const double f_layer_us =
        f_fused.event_us_per_launch + f_attention.event_us_per_launch;
    std::printf(
        "context=%u T_ms_per_token=%.3f F_ms_per_token=%.3f "
        "T_minus_F_ms_per_token=%.3f\n",
        context, t_layer_us * kLayersPerToken / 1000.0,
        f_layer_us * kLayersPerToken / 1000.0,
        (t_layer_us - f_layer_us) * kLayersPerToken / 1000.0);
  }
  (void)hipFree(q);
  (void)hipFree(k);
  (void)hipFree(v);
  (void)hipFree(gate);
  (void)hipFree(q_out);
  (void)hipFree(k_out);
  (void)hipFree(out);
  (void)hipFree(f_split);
  (void)hipFree(q_weight);
  (void)hipFree(k_weight);
  (void)hipFree(k_f16);
  (void)hipFree(v_f16);
  (void)hipFree(k_turbo4);
  (void)hipFree(v_turbo4);
  (void)hipFree(d_pos);
  (void)hipFree(d_write_status);
  (void)hipFree(d_read_status);
  (void)hipFree(workspace);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--bench-writer") == 0) {
    MicrobenchSingleWriter();
    return g_failures == 0 ? 0 : 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--profile-decode") == 0) {
    ProfileDecodeCosts();
    return g_failures == 0 ? 0 : 1;
  }
  std::printf("E1 device: batched Turbo4 KV launch vs N single launches\n");
  for (std::uint32_t count : {1U, 2U, 5U, 8U, 33U, 64U, 128U})
    CompareOnDevice(count, 0U);
  for (std::uint32_t pos : {1U, 7U, 33U, 257U})
    CompareOnDevice(9U, pos);
  GoldenWriteOnDevice();
  FusedPrepareMatchesReferenceOnDevice();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

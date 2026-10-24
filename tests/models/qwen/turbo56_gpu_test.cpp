// Turbo5 / Turbo6 on the device, with Turbo4 as the control.
//
// For each width (4, 5, 6) this checks the three kernels the width touches:
//   W  batched writer  vs the host range body (same planes, same source)
//   D  dequant layer   vs the host dequant body, on DEVICE-written planes
//   A  split-K reader and scalar reader vs a double host oracle that decodes
//      the SAME device-written plane through QwenKvReadHead (so quantisation error cancels and
//      the bound is the same 8e-4 the Turbo4 E5 test uses).
// Source-only fixture: rerun syntax/link/device checks in the gated window
// before using this revision as evidence of kernel correctness.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"
#include "tests/models/qwen/turbo56_writer_diff.hpp"

namespace hipq = gufo::hip;
namespace t4 = gufo::hip::turbo4;

namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kMaxContext = 1024;

int failures = 0;
void Check(bool ok, const std::string& message) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", message.c_str());
  if (!ok)
    ++failures;
}

#define HIP_CHECK(expr)                                                    \
  do {                                                                     \
    const hipError_t error = (expr);                                       \
    if (error != hipSuccess) {                                             \
      Check(false, std::string("HIP error: ") + hipGetErrorString(error)); \
      return false;                                                        \
    }                                                                      \
  } while (false)

float Noise(std::size_t n) {
  std::uint64_t x = n * 6364136223846793005ULL + 1442695040888963407ULL;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return static_cast<float>(x >> 40) / 16777216.0f - 0.5f;
}

bool PaddingIsZero(t4::PlaneFormats formats, const std::uint8_t* k,
                   const std::uint8_t* v, std::size_t blocks) {
  const auto slot = t4::BlockBytes(formats.stride);
  for (std::size_t b = 0; b < blocks; ++b) {
    for (std::size_t byte = t4::BlockBytes(formats.k); byte < slot; ++byte)
      if (k[b * slot + byte] != 0) return false;
    for (std::size_t byte = t4::BlockBytes(formats.v); byte < slot; ++byte)
      if (v[b * slot + byte] != 0) return false;
  }
  return true;
}

double RelativeL2(const std::vector<float>& a, const std::vector<float>& ref) {
  double e = 0.0, n = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - ref[i];
    e += d * d;
    n += static_cast<double>(ref[i]) * ref[i];
  }
  return std::sqrt(e) / std::max(1e-12, std::sqrt(n));
}

bool HostDoubleOracle(hipq::QwenKvCacheStorage storage, t4::PlaneFormats formats, const std::uint8_t* k,
                      const std::uint8_t* v, std::size_t plane_bytes,
                      std::uint32_t context, const std::vector<float>& q,
                      std::vector<float>* result) {
  const hipq::QwenKvPlaneGeometry geometry{kHeadDim, kKvHeads, kMaxContext};
  result->assign(kHeads * kHeadDim, 0.0f);
  std::vector<float> krow(kHeadDim), vrow(kHeadDim);
  for (std::uint32_t qh = 0; qh < kHeads; ++qh) {
    const std::size_t kvh = qh / (kHeads / kKvHeads);
    std::vector<double> scores(context);
    std::vector<std::vector<float>> values(context);
    for (std::uint32_t pos = 0; pos < context; ++pos) {
      if (!hipq::QwenKvReadHead(storage, k, plane_bytes, geometry, pos, kvh,
                                krow.data(), formats, true) ||
          !hipq::QwenKvReadHead(storage, v, plane_bytes, geometry, pos, kvh,
                                vrow.data(), formats, false))
        return false;
      double dot = 0.0;
      for (std::uint32_t d = 0; d < kHeadDim; ++d)
        dot += static_cast<double>(q[qh * kHeadDim + d]) * krow[d];
      scores[pos] = dot / std::sqrt(static_cast<double>(kHeadDim));
      values[pos] = vrow;
    }
    const double mx = *std::max_element(scores.begin(), scores.end());
    double den = 0.0;
    std::vector<double> num(kHeadDim, 0.0);
    for (std::uint32_t pos = 0; pos < context; ++pos) {
      const double w = std::exp(scores[pos] - mx);
      den += w;
      for (std::uint32_t d = 0; d < kHeadDim; ++d)
        num[d] += w * values[pos][d];
    }
    for (std::uint32_t d = 0; d < kHeadDim; ++d)
      (*result)[qh * kHeadDim + d] = static_cast<float>(num[d] / den);
  }
  return true;
}

bool RunWidth(t4::PlaneFormats formats, hipq::QwenKvCacheStorage storage) {
  const std::string tag =
      "K" + std::to_string(static_cast<int>(formats.k)) + "/V" +
      std::to_string(static_cast<int>(formats.v)) + "/slot" +
      std::to_string(static_cast<int>(formats.stride)) + " ";
  const std::size_t kvw = static_cast<std::size_t>(kKvHeads) * kHeadDim;
  const std::size_t plane_bytes =
      hipq::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kMaxContext, formats.stride);
  const std::size_t src_bytes = kMaxContext * kvw * sizeof(float);
  std::vector<float> ks(kMaxContext * kvw), vs(kMaxContext * kvw);
  for (std::size_t i = 0; i < ks.size(); ++i) {
    ks[i] = 0.7f + 0.25f * Noise(i + 10000);
    vs[i] = 0.6f + 0.30f * Noise(i + 40000);
  }

  // Host reference planes.
  std::vector<std::uint8_t> hk(plane_bytes, 0), hv(plane_bytes, 0);
  for (std::uint32_t head = 0; head < kKvHeads; ++head)
    if (hipq::Turbo4WriteHeadRangeBody(
            ks.data(), vs.data(), hk.data(), hv.data(), plane_bytes, src_bytes,
            head, 0, kMaxContext, kMaxContext, kKvHeads, kHeadDim, formats) != hipq::Turbo4KernelStatus::kOk) {
      Check(false, tag + "host range writer");
      return false;
    }

  float *dks = nullptr, *dvs = nullptr;
  std::uint8_t *dk = nullptr, *dv = nullptr;
  std::uint32_t* dpos = nullptr;
  hipq::Turbo4KernelStatus* dstat = nullptr;
  HIP_CHECK(hipMalloc(&dks, src_bytes));
  HIP_CHECK(hipMalloc(&dvs, src_bytes));
  HIP_CHECK(hipMalloc(&dk, plane_bytes));
  HIP_CHECK(hipMalloc(&dv, plane_bytes));
  HIP_CHECK(hipMalloc(&dpos, sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(&dstat, 16));
  HIP_CHECK(hipMemcpy(dks, ks.data(), src_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dvs, vs.data(), src_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dk, 0xA5, plane_bytes));
  HIP_CHECK(hipMemset(dv, 0xA5, plane_bytes));
  HIP_CHECK(hipMemset(dstat, 0, 16));
  const std::uint32_t first = 0;
  HIP_CHECK(hipMemcpy(dpos, &first, sizeof(first), hipMemcpyHostToDevice));

  // ---- W: batched writer ----
  const auto wst = hipq::LaunchTurbo4KvWriteBatched(
      dks, dvs, dk, dv, dpos, kMaxContext, plane_bytes, src_bytes, kMaxContext,
      kKvHeads, kHeadDim, dstat, nullptr, formats);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint8_t> gk(plane_bytes), gv(plane_bytes);
  hipq::Turbo4KernelStatus status = hipq::Turbo4KernelStatus::kWriterFailed;
  HIP_CHECK(hipMemcpy(gk.data(), dk, plane_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gv.data(), dv, plane_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&status, dstat, sizeof(status), hipMemcpyDeviceToHost));
  std::size_t diff = 0;
  std::size_t shown = 0;
  bool allowed_diffs = true;
  const auto slot = t4::BlockBytes(formats.stride);
  const auto blocks_per_head = hipq::Turbo4BlocksPerHead(kHeadDim);
  const auto blocks_per_token =
      hipq::Turbo4BlocksPerToken(kHeadDim, kKvHeads);
  auto show_diff = [&](const char* plane, t4::Bits bits,
                       const std::vector<std::uint8_t>& gpu,
                       const std::vector<std::uint8_t>& host, std::size_t i) {
    if (shown >= 16)
      return;
    ++shown;
    const auto block = i / slot;
    const auto byte = i % slot;
    const auto base = i - byte;
    const char* field = byte == 0 ? "norm0" : byte == 1 ? "norm1" :
        byte >= t4::BlockBytes(bits) ? "padding" :
        byte < t4::kQhOffset ? "qs" : "qh";
    const auto gpu_norm = static_cast<std::uint16_t>(
        gpu[base] | (static_cast<unsigned>(gpu[base + 1]) << 8U));
    const auto host_norm = static_cast<std::uint16_t>(
        host[base] | (static_cast<unsigned>(host[base + 1]) << 8U));
    std::printf(
        "[DIFF] %splane=%s row=%zu head=%zu block=%zu offset=%zu "
        "block_byte=%zu field=%s gpu=0x%02x host=0x%02x "
        "gpu_norm=%.9g(0x%04x) host_norm=%.9g(0x%04x)\n",
        tag.c_str(), plane, block / blocks_per_token,
        (block % blocks_per_token) / blocks_per_head,
        block % blocks_per_head, i, byte, field,
        static_cast<unsigned>(gpu[i]), static_cast<unsigned>(host[i]),
        static_cast<double>(t4::HalfBitsToFloat(gpu_norm)),
        static_cast<unsigned>(gpu_norm),
        static_cast<double>(t4::HalfBitsToFloat(host_norm)),
        static_cast<unsigned>(host_norm));
  };
  for (std::size_t i = 0; i < plane_bytes; ++i) {
    diff += (gk[i] != hk[i]) + (gv[i] != hv[i]);
    const auto base = i - i % slot;
    if (gk[i] != hk[i]) {
      allowed_diffs &= gufo::test::Turbo56WriterByteDiffAllowed(
          gk.data() + base, hk.data() + base, i % slot, formats.k);
      show_diff("K", formats.k, gk, hk, i);
    }
    if (gv[i] != hv[i]) {
      allowed_diffs &= gufo::test::Turbo56WriterByteDiffAllowed(
          gv.data() + base, hv.data() + base, i % slot, formats.v);
      show_diff("V", formats.v, gv, hv, i);
    }
  }
  // Measured on device, then fixed in review: bound every changed field AND the rate.
  // Padding, non-finite norms and non-neighbour codes remain failures.
  Check(wst == hipq::Turbo4LaunchStatus::kLaunched &&
            status == hipq::Turbo4KernelStatus::kOk && allowed_diffs &&
            gufo::test::Turbo56WriterDiffCountAllowed(diff, 2 * plane_bytes),
        tag + "batched writer bytes vs host body, differing=" +
            std::to_string(diff) + " of " + std::to_string(2 * plane_bytes));

  Check(PaddingIsZero(formats, gk.data(), gv.data(),
                      kMaxContext * hipq::Turbo4BlocksPerToken(kHeadDim, kKvHeads)),
        tag + "batched writer zeroes poisoned padding");

  // ---- D: dequant layer on the DEVICE-written planes ----
  const std::size_t scratch_halfs = static_cast<std::size_t>(kMaxContext) * kvw;
  void *dkd = nullptr, *dvd = nullptr;
  HIP_CHECK(hipMalloc(&dkd, scratch_halfs * 2));
  HIP_CHECK(hipMalloc(&dvd, scratch_halfs * 2));
  HIP_CHECK(hipMemset(dstat, 0, 16));
  const auto dst = hipq::LaunchTurbo4DequantLayer(
      dk, dv, dkd, dvd, kMaxContext, kMaxContext, kKvHeads, kHeadDim, dstat,
      nullptr, formats);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint16_t> gkd(scratch_halfs), gvd(scratch_halfs),
      hkd(scratch_halfs), hvd(scratch_halfs);
  HIP_CHECK(
      hipMemcpy(gkd.data(), dkd, scratch_halfs * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(gvd.data(), dvd, scratch_halfs * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&status, dstat, sizeof(status), hipMemcpyDeviceToHost));
  const auto hst = hipq::Turbo4DequantLayerBody(
      gk.data(), gv.data(), hkd.data(), hvd.data(), kMaxContext, kMaxContext,
      kKvHeads, kHeadDim, formats);
  double worst = 0.0;
  for (std::size_t i = 0; i < scratch_halfs; ++i) {
    if (!std::isfinite(t4::HalfBitsToFloat(gkd[i])) ||
        !std::isfinite(t4::HalfBitsToFloat(gvd[i])))
      worst = std::numeric_limits<double>::infinity();
    worst = std::max(
        worst, std::fabs(static_cast<double>(t4::HalfBitsToFloat(gkd[i])) -
                         t4::HalfBitsToFloat(hkd[i])));
    worst = std::max(
        worst, std::fabs(static_cast<double>(t4::HalfBitsToFloat(gvd[i])) -
                         t4::HalfBitsToFloat(hvd[i])));
  }
  Check(dst == hipq::Turbo4LaunchStatus::kLaunched &&
            hst == hipq::Turbo4KernelStatus::kOk &&
            status == hipq::Turbo4KernelStatus::kOk && worst < 2e-3,
        tag + "dequant layer vs host body, worst abs diff=" +
            std::to_string(worst));

  // ---- A: end-to-end readers on the DEVICE-written planes ----
  const std::size_t vector_bytes =
      static_cast<std::size_t>(kHeads) * kHeadDim * sizeof(float);
  const std::size_t partial_bytes =
      hipq::Turbo4DecodePartialBytes(kMaxContext, kHeads, kKvHeads, kHeadDim);
  std::vector<float> q(kHeads * kHeadDim);
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = 0.8f + 0.4f * Noise(i + 88000);
  float *dq = nullptr, *dout = nullptr, *dqr = nullptr, *dor = nullptr,
        *dpart = nullptr;
  hipq::Turbo4KernelStatus* dwst = nullptr;
  HIP_CHECK(hipMalloc(&dq, vector_bytes));
  HIP_CHECK(hipMalloc(&dout, vector_bytes));
  HIP_CHECK(hipMalloc(&dqr, vector_bytes));
  HIP_CHECK(hipMalloc(&dor, vector_bytes));
  HIP_CHECK(hipMalloc(&dpart, partial_bytes));
  HIP_CHECK(hipMalloc(&dwst, sizeof(*dwst)));
  HIP_CHECK(hipMemcpy(dq, q.data(), vector_bytes, hipMemcpyHostToDevice));
  const auto ok = hipq::Turbo4KernelStatus::kOk;
  HIP_CHECK(hipMemcpy(dwst, &ok, sizeof(ok), hipMemcpyHostToDevice));
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  for (const std::uint32_t context : {1U, 9U, 129U, 1000U, 1024U}) {
    std::vector<float> expected;
    if (!HostDoubleOracle(storage, formats, gk.data(), gv.data(), plane_bytes, context,
                          q, &expected)) {
      Check(false, tag + "host oracle at context " + std::to_string(context));
      continue;
    }
    for (int scalar = 0; scalar < 2; ++scalar) {
      if (scalar && context > 129U)
        continue;  // scalar kernel is slow; short contexts only
      HIP_CHECK(hipMemset(dstat, 0, 16));
      HIP_CHECK(hipMemset(dout, 0x6D, vector_bytes));
      const auto launched =
          scalar ? hipq::LaunchTurbo4AttentionScalar(
                       dq, dk, dv, dout, plane_bytes, vector_bytes,
                       vector_bytes, nullptr, 0, kHeads, kKvHeads, kHeadDim,
                       context, scale, dwst, dstat, nullptr, formats)
                 : hipq::LaunchTurbo4AttentionSplit(
                       dq, dk, dv, dout, dqr, dor, dpart, plane_bytes,
                       vector_bytes, vector_bytes, vector_bytes, vector_bytes,
                       partial_bytes, nullptr, 0, kHeads, kKvHeads, kHeadDim,
                       context, kMaxContext, scale, dwst, dstat, nullptr, false, formats);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> actual(kHeads * kHeadDim);
      HIP_CHECK(
          hipMemcpy(actual.data(), dout, vector_bytes, hipMemcpyDeviceToHost));
      HIP_CHECK(
          hipMemcpy(&status, dstat, sizeof(status), hipMemcpyDeviceToHost));
      const double err = RelativeL2(actual, expected);
      Check(launched == hipq::Turbo4LaunchStatus::kLaunched &&
                status == hipq::Turbo4KernelStatus::kOk && err < 8e-4,
            tag + (scalar ? "scalar" : "split") +
                " reader vs double oracle, context " + std::to_string(context) +
                " relL2=" + std::to_string(err));
    }
  }
  // F: fused prepare at zero AND nonzero position, including padded slots.
  // Unit-magnitude inputs and eps=0 give exact RMS=1. theta=1 makes every
  // RoPE pair use angle=pos: the independent host reference stays explicit.
  for (const std::uint32_t pos : {0U, 3U}) {
    const auto fused_tag = tag + "fused pos=" + std::to_string(pos) + " ";
    std::vector<float> fq(kHeads * kHeadDim), fk(kvw), fv(kvw);
    for (std::size_t i = 0; i < fq.size(); ++i) fq[i] = (i % 3 == 0) ? -1.0F : 1.0F;
    for (std::size_t i = 0; i < kvw; ++i) {
      fk[i] = (i % 5 == 0) ? -1.0F : 1.0F;
      fv[i] = 0.1F * Noise(i + 719);
    }
    auto rope = [pos](std::vector<float> values) {
      const float c = std::cos(static_cast<float>(pos));
      const float s = std::sin(static_cast<float>(pos));
      for (std::size_t base = 0; base < values.size(); base += kHeadDim)
        for (unsigned pair = 0; pair < kHeadDim / 2; ++pair) {
          const float x = values[base + pair];
          const float y = values[base + pair + kHeadDim / 2];
          values[base + pair] = x * c - y * s;
          values[base + pair + kHeadDim / 2] = x * s + y * c;
        }
      return values;
    };
    const auto rotated_k = rope(fk);
    auto expected_q = rope(fq);
    std::vector<std::uint8_t> expected_k(plane_bytes, 0xA5), expected_v(plane_bytes, 0xA5);
    for (std::uint32_t head = 0; head < kKvHeads; ++head)
      Check(hipq::Turbo4WriteHeadBody(rotated_k.data(), fv.data(), expected_k.data(),
          expected_v.data(), plane_bytes, kvw * sizeof(float), head, pos,
          kMaxContext, kKvHeads, kHeadDim, formats) == hipq::Turbo4KernelStatus::kOk,
          fused_tag + "host paired writer");
    for (std::size_t offset = 0; offset < expected_q.size(); offset += t4::kBlockValues)
      t4::RotateForward(expected_q.data() + offset);
    HIP_CHECK(hipMemcpy(dq, fq.data(), vector_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dks, fk.data(), kvw * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dvs, fv.data(), kvw * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dpos, &pos, sizeof(pos), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dk, 0xA5, plane_bytes));
    HIP_CHECK(hipMemset(dv, 0xA5, plane_bytes));
    HIP_CHECK(hipMemset(dstat, 0, 16));
    const auto prepared = hipq::LaunchTurbo4DecodePrepare(
        dq, dks, dvs, nullptr, nullptr, dqr, dk, dv, dpos, vector_bytes,
        kvw * sizeof(float), vector_bytes, plane_bytes, kHeads, kKvHeads,
        kHeadDim, kHeadDim, kMaxContext, 1.0F, 0.0F, dstat, dstat + 1,
        nullptr, nullptr, formats);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(gk.data(), dk, plane_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gv.data(), dv, plane_bytes, hipMemcpyDeviceToHost));
    std::vector<float> actual_q(fq.size());
    hipq::Turbo4KernelStatus fused_status[2]{};
    HIP_CHECK(hipMemcpy(actual_q.data(), dqr, vector_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(fused_status, dstat, sizeof(fused_status), hipMemcpyDeviceToHost));
    const auto blocks = hipq::Turbo4BlocksPerToken(kHeadDim, kKvHeads);
    const auto token_bytes = blocks * t4::BlockBytes(formats.stride);
    const auto offset = pos * token_bytes;
    Check(prepared == hipq::Turbo4LaunchStatus::kLaunched &&
          fused_status[0] == hipq::Turbo4KernelStatus::kOk &&
          fused_status[1] == hipq::Turbo4KernelStatus::kOk &&
          std::equal(gk.begin() + offset, gk.begin() + offset + token_bytes, expected_k.begin() + offset) &&
          std::equal(gv.begin() + offset, gv.begin() + offset + token_bytes, expected_v.begin() + offset),
          fused_tag + "paired bytes exactly match host");
    Check(RelativeL2(actual_q, expected_q) < 1e-6,
          fused_tag + "query rotation vs explicit norm/RoPE oracle");
    auto untouched = [offset, token_bytes](const auto& plane) {
      return std::all_of(plane.begin(), plane.begin() + offset, [](auto x) { return x == 0xA5; }) &&
             std::all_of(plane.begin() + offset + token_bytes, plane.end(), [](auto x) { return x == 0xA5; });
    };
    Check(untouched(gk) && untouched(gv), fused_tag + "other rows retain poison");
    Check(PaddingIsZero(formats, gk.data() + offset, gv.data() + offset, blocks),
          fused_tag + "paired writer zeroes poisoned padding");
  }

  for (void* p : {static_cast<void*>(dks), static_cast<void*>(dvs),
                  static_cast<void*>(dk), static_cast<void*>(dv),
                  static_cast<void*>(dpos), static_cast<void*>(dstat), dkd, dvd,
                  static_cast<void*>(dq), static_cast<void*>(dout),
                  static_cast<void*>(dqr), static_cast<void*>(dor),
                  static_cast<void*>(dpart), static_cast<void*>(dwst)})
    (void)hipFree(p);
  return true;
}

}  // namespace

int main() {
  std::printf("Turbo5/Turbo6 kernels (Turbo4 = control)\n");
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  bool ok = RunWidth(t4::PlaneFormats(t4::Bits::k4), hipq::QwenKvCacheStorage::kTurbo4);
  ok = RunWidth(t4::PlaneFormats(t4::Bits::k5), hipq::QwenKvCacheStorage::kTurbo5) && ok;
  ok = RunWidth(t4::PlaneFormats(t4::Bits::k6), hipq::QwenKvCacheStorage::kTurbo6) && ok;
  ok = RunWidth({t4::Bits::k5, t4::Bits::k5, t4::Bits::k4}, hipq::QwenKvCacheStorage::kTurbo5) && ok;
  ok = RunWidth({t4::Bits::k6, t4::Bits::k6, t4::Bits::k4}, hipq::QwenKvCacheStorage::kTurbo6) && ok;
  ok = RunWidth({t4::Bits::k6, t4::Bits::k5, t4::Bits::k4}, hipq::QwenKvCacheStorage::kTurbo6) && ok;
  ok = RunWidth({t4::Bits::k6, t4::Bits::k4, t4::Bits::k4}, hipq::QwenKvCacheStorage::kTurbo6) && ok;
  ok = RunWidth({t4::Bits::k6, t4::Bits::k4, t4::Bits::k6}, hipq::QwenKvCacheStorage::kTurbo6) && ok;
  ok = RunWidth({t4::Bits::k6, t4::Bits::k5, t4::Bits::k6}, hipq::QwenKvCacheStorage::kTurbo6) && ok;
  std::printf("%s %d failure(s)\n", ok && failures == 0 ? "PASS" : "FAIL",
              failures);
  return ok && failures == 0 ? 0 : 1;
}

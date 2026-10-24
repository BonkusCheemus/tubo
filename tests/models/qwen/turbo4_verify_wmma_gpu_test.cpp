// WMMA multi-row verify reader vs the split reader, for
// Turbo4 and the Turbo5/6 qh / padded-slot formats.
//
// NOT bit-identical (f16 Q/K/V/P operands, different summation order), so
// the gate is the per-row relative L2 error against the split reader
// (LaunchTurbo4AttentionSplitBatched, f32 FMA): must be < 2e-3. Max abs and
// max relative error are reported per case, and every case must be
// all-finite with both status words kOk. K/V are written through the
// production host writer (QwenKvWriteHead) for each plane's codec.
//
// Codecs checked: uniform k4, k5, k6, and one MIXED case (98-byte k6 slot,
// K k6, V k5) — the padded-slot case the plane code allows.
//
// VRD6 (uniform k6 only): the vrd6 reader at split caps 0 / 10 / 3 gets the
// same per-row L2 gate vs split, and an inf stored K norm in its range must
// latch kPoisonedNorm.
//
// With GUFO_TEST_TIME=1 the binary also times 50 launches of each reader at
// context 32768 / rows 8 (hipEvent, warm-up first) and prints one line per
// codec: `codec rows ctx split_ms wmma_ms ratio`.
//
// With GUFO_TEST_DUMP=<file> the binary appends, per case, the tag,
// the WMMA status word, the output rows and the partial records to <file>, with
// the partials pre-filled with 0x6D, so two runs (GUFO_DEV_VERIFY_T6_PIPE=0 vs
// unset) can be compared byte for byte with cmp.
//
// Needs a GPU; SKIPs without one.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace hipq = gufo::hip;
namespace t4 = gufo::hip::turbo4;

namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kMaxContext = 32768;
constexpr std::uint32_t kMaxRows = 8;
constexpr double kMaxRowL2 = 2e-3;
constexpr int kTimingIterations = 50;

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

// One codec (uniform or mixed) end to end: production-written planes, the split
// reader as reference, the WMMA reader as candidate, the per-row L2 gate, and
// (when `timing`) the 50-launch hipEvent comparison at 32k / rows 8.
bool RunCodec(t4::PlaneFormats formats, hipq::QwenKvCacheStorage storage,
              bool timing) {
  const std::string codec =
      "s" + std::to_string(static_cast<unsigned>(formats.stride)) + "k" +
      std::to_string(static_cast<unsigned>(formats.k)) + "v" +
      std::to_string(static_cast<unsigned>(formats.v));
  const std::string tag_prefix = " codec=" + codec;

  const std::size_t plane_bytes = hipq::Turbo4LayerPlaneBytes(
      kHeadDim, kKvHeads, kMaxContext, formats.stride);
  const std::size_t vector_floats = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t vector_bytes = vector_floats * sizeof(float);
  const std::size_t single_partial_bytes =
      hipq::Turbo4DecodePartialBytes(kMaxContext, kHeads, kKvHeads, kHeadDim);
  const std::size_t batched_capacity = kMaxRows * single_partial_bytes;
  if (plane_bytes == 0 || single_partial_bytes == 0) {
    Check(false, codec + " plane and partial sizing");
    return false;
  }

  // Random K/V through the production host writer. Zero-mean keys and a wide
  // query give a spread (not near-uniform) softmax, so a wrong mask or a wrong
  // online-softmax rescale cannot hide behind equal weights.
  std::vector<std::uint8_t> host_k(plane_bytes), host_v(plane_bytes);
  const hipq::QwenKvPlaneGeometry geometry{kHeadDim, kKvHeads, kMaxContext};
  std::vector<float> krow(kHeadDim), vrow(kHeadDim);
  for (std::uint32_t pos = 0; pos < kMaxContext; ++pos) {
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const std::size_t seed =
            static_cast<std::size_t>(pos) * kKvHeads * kHeadDim +
            head * kHeadDim + d;
        krow[d] = 3.0f * Noise(seed + 10000);
        vrow[d] = 0.25f + Noise(seed + 40000);
      }
      if (!hipq::QwenKvWriteHead(storage, host_k.data(), plane_bytes, geometry,
                                 pos, head, krow.data(), formats, true) ||
          !hipq::QwenKvWriteHead(storage, host_v.data(), plane_bytes, geometry,
                                 pos, head, vrow.data(), formats, false)) {
        Check(false, codec + " host writer produced every test row");
        return false;
      }
    }
  }

  std::vector<float> q(kMaxRows * vector_floats), gate(q.size());
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 4.0f * Noise(i + 88000);
    gate[i] = -0.5f + Noise(i + 99000);
  }

  std::uint8_t *device_k = nullptr, *device_v = nullptr;
  float *device_q = nullptr, *device_gate = nullptr;
  float *ref_out = nullptr, *ref_q_rot = nullptr, *ref_out_rot = nullptr,
        *ref_partials = nullptr;
  float *wmma_out = nullptr, *wmma_q_rot = nullptr;
  float *wmma_out_rot = nullptr, *wmma_partials = nullptr;
  hipq::Turbo4KernelStatus *device_writer_status = nullptr,
                           *device_status = nullptr;
  const std::size_t rows_bytes = kMaxRows * vector_bytes;
  HIP_CHECK(hipMalloc(&device_k, plane_bytes));
  HIP_CHECK(hipMalloc(&device_v, plane_bytes));
  HIP_CHECK(hipMalloc(&device_q, rows_bytes));
  HIP_CHECK(hipMalloc(&device_gate, rows_bytes));
  HIP_CHECK(hipMalloc(&ref_out, rows_bytes));
  HIP_CHECK(hipMalloc(&ref_q_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&ref_out_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&ref_partials, batched_capacity));
  HIP_CHECK(hipMalloc(&wmma_out, rows_bytes));
  HIP_CHECK(hipMalloc(&wmma_q_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&wmma_out_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&wmma_partials, batched_capacity));
  // ATTN6 arm (K = V = Turbo6 only).
  float *a6_out = nullptr, *a6_q_rot = nullptr, *a6_out_rot = nullptr,
        *a6_partials = nullptr;
  HIP_CHECK(hipMalloc(&a6_out, rows_bytes));
  HIP_CHECK(hipMalloc(&a6_q_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&a6_out_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&a6_partials, batched_capacity));
  const bool attn6_codec = formats.stride == t4::Bits::k6 &&
                           formats.k == t4::Bits::k6 &&
                           formats.v == t4::Bits::k6;
  HIP_CHECK(hipMalloc(&device_writer_status, sizeof(*device_writer_status)));
  HIP_CHECK(hipMalloc(&device_status, sizeof(*device_status)));
  HIP_CHECK(
      hipMemcpy(device_k, host_k.data(), plane_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_v, host_v.data(), plane_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_q, q.data(), rows_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_gate, gate.data(), rows_bytes, hipMemcpyHostToDevice));
  const auto ok = hipq::Turbo4KernelStatus::kOk;
  HIP_CHECK(
      hipMemcpy(device_writer_status, &ok, sizeof(ok), hipMemcpyHostToDevice));
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  bool passed = true;
  for (const std::uint32_t start_pos : {0U, 255U, 4095U, 32760U}) {
    for (const std::uint32_t rows : {1U, 2U, 5U, 8U}) {
      if (start_pos + rows > kMaxContext)
        continue;
      const std::string tag = tag_prefix + " start_pos=" +
                              std::to_string(start_pos) + " rows=" +
                              std::to_string(rows);
      const std::size_t row_partial = hipq::Turbo4BatchedDecodePartialBytes(
          start_pos + 1U, rows, kHeads, kHeadDim);

      // Reference: the split reader, every row in one batched launch.
      HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
      HIP_CHECK(hipMemset(ref_out, 0x6D, rows_bytes));
      const auto ref_launched = hipq::LaunchTurbo4AttentionSplitBatched(
          device_q, device_k, device_v, ref_out, ref_q_rot, ref_out_rot,
          ref_partials, plane_bytes, rows * vector_bytes, rows * vector_bytes,
          rows * vector_bytes, rows * vector_bytes, rows * row_partial,
          device_gate, rows * vector_bytes, kHeads, kKvHeads, kHeadDim,
          start_pos + 1U, rows, kMaxContext, scale, device_writer_status,
          device_status, nullptr, formats);
      if (ref_launched != hipq::Turbo4LaunchStatus::kLaunched) {
        Check(false, "split reference enqueues" + tag);
        return false;
      }
      HIP_CHECK(hipDeviceSynchronize());
      hipq::Turbo4KernelStatus ref_status =
          hipq::Turbo4KernelStatus::kWriterFailed;
      HIP_CHECK(hipMemcpy(&ref_status, device_status, sizeof(ref_status),
                          hipMemcpyDeviceToHost));
      Check(ref_status == hipq::Turbo4KernelStatus::kOk,
            "split reference latches no failure" + tag);
      std::vector<float> expected(rows * vector_floats);
      HIP_CHECK(hipMemcpy(expected.data(), ref_out, rows * vector_bytes,
                          hipMemcpyDeviceToHost));

      // Candidate: the WMMA reader.
      const char* const dump_path = std::getenv("GUFO_TEST_DUMP");
      HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
      HIP_CHECK(hipMemset(wmma_out, 0x6D, rows_bytes));
      if (dump_path != nullptr || attn6_codec)
        HIP_CHECK(hipMemset(wmma_partials, 0x6D, rows * row_partial));
      const auto wmma_launched = hipq::LaunchTurbo4AttentionVerifyWmma(
          device_q, device_k, device_v, wmma_out, wmma_q_rot, wmma_out_rot,
          wmma_partials, plane_bytes, rows * vector_bytes, rows * vector_bytes,
          rows * vector_bytes, rows * vector_bytes, rows * row_partial,
          device_gate, rows * vector_bytes, kHeads, kKvHeads, kHeadDim,
          start_pos + 1U, rows, kMaxContext, scale, device_writer_status,
          device_status, nullptr, formats);
      if (wmma_launched != hipq::Turbo4LaunchStatus::kLaunched) {
        Check(false, "WMMA reader enqueues" + tag);
        passed = false;
        continue;
      }
      HIP_CHECK(hipDeviceSynchronize());
      hipq::Turbo4KernelStatus status = hipq::Turbo4KernelStatus::kWriterFailed;
      HIP_CHECK(hipMemcpy(&status, device_status, sizeof(status),
                          hipMemcpyDeviceToHost));
      Check(status == hipq::Turbo4KernelStatus::kOk,
            "WMMA reader latches no failure" + tag);
      std::vector<float> actual(rows * vector_floats, 0.0f);
      HIP_CHECK(hipMemcpy(actual.data(), wmma_out, rows * vector_bytes,
                          hipMemcpyDeviceToHost));
      if (dump_path != nullptr) {
        std::vector<std::uint8_t> parts(rows * row_partial);
        HIP_CHECK(hipMemcpy(parts.data(), wmma_partials, parts.size(),
                            hipMemcpyDeviceToHost));
        std::FILE* f = std::fopen(dump_path, "ab");
        const bool wrote =
            f != nullptr &&
            std::fwrite(tag.data(), 1, tag.size(), f) == tag.size() &&
            std::fwrite(&status, sizeof(status), 1, f) == 1 &&
            std::fwrite(actual.data(), sizeof(float), actual.size(), f) ==
                actual.size() &&
            std::fwrite(parts.data(), 1, parts.size(), f) == parts.size();
        Check(f != nullptr && std::fclose(f) == 0 && wrote,
              "dump written" + tag);
      }

      double worst_l2 = 0.0, worst_abs = 0.0, worst_rel = 0.0;
      bool finite = true;
      for (std::uint32_t row = 0; row < rows; ++row) {
        const float* a = actual.data() + row * vector_floats;
        const float* e = expected.data() + row * vector_floats;
        double diff2 = 0.0, ref2 = 0.0, ref_max = 0.0;
        for (std::size_t i = 0; i < vector_floats; ++i)
          ref_max = std::max(ref_max, std::fabs(static_cast<double>(e[i])));
        for (std::size_t i = 0; i < vector_floats; ++i) {
          // Review: the split reference must be finite too; an inf/NaN
          // reference would otherwise make the row L2 NaN and std::max drop it.
          if (!std::isfinite(a[i]) || !std::isfinite(e[i])) {
            finite = false;
            continue;
          }
          const double d = static_cast<double>(a[i]) - e[i];
          diff2 += d * d;
          ref2 += static_cast<double>(e[i]) * e[i];
          worst_abs = std::max(worst_abs, std::fabs(d));
          // Relative to the element, floored at 1e-3 of the row's peak so a
          // near-zero reference element does not dominate.
          worst_rel = std::max(
              worst_rel,
              std::fabs(d) / std::max(std::fabs(static_cast<double>(e[i])),
                                      1e-3 * ref_max));
        }
        const double row_l2 = ref2 > 0.0 ? std::sqrt(diff2 / ref2) : 1.0;
        if (!std::isfinite(row_l2))
          finite = false;
        worst_l2 = std::max(worst_l2, row_l2);
      }
      std::printf("  %s max_abs=%.3e max_rel=%.3e max_row_l2=%.3e\n",
                  tag.c_str(), worst_abs, worst_rel, worst_l2);
      Check(finite, "WMMA output is finite" + tag);
      const bool close = finite && worst_l2 < kMaxRowL2;
      Check(close, "max per-row relative L2 error < 2e-3 vs split" + tag);
      passed = passed && finite && close &&
               status == hipq::Turbo4KernelStatus::kOk &&
               ref_status == hipq::Turbo4KernelStatus::kOk;

      // ATTN6: on the WMMA split plan (cap 0) the partials and the output
      // must equal the WMMA reader's bit for bit; with a split cap (10) the
      // merge order changes, so the gate is the same L2 bound vs split.
      if (attn6_codec) {
        std::vector<std::uint8_t> wmma_parts(rows * row_partial);
        HIP_CHECK(hipMemcpy(wmma_parts.data(), wmma_partials,
                            wmma_parts.size(), hipMemcpyDeviceToHost));
        for (const std::uint32_t cap : {0U, 10U}) {
          const std::string a6tag = tag + " attn6_cap=" + std::to_string(cap);
          HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
          HIP_CHECK(hipMemset(a6_out, 0x6D, rows_bytes));
          HIP_CHECK(hipMemset(a6_partials, 0x6D, rows * row_partial));
          const auto a6_launched = hipq::LaunchTurbo4AttentionVerifyAttn6(
              device_q, device_k, device_v, a6_out, a6_q_rot, a6_out_rot,
              a6_partials, plane_bytes, rows * vector_bytes,
              rows * vector_bytes, rows * vector_bytes, rows * vector_bytes,
              rows * row_partial, device_gate, rows * vector_bytes, kHeads,
              kKvHeads, kHeadDim, start_pos + 1U, rows, kMaxContext, scale,
              device_writer_status, device_status, nullptr, formats, cap);
          if (a6_launched != hipq::Turbo4LaunchStatus::kLaunched) {
            Check(false, "attn6 reader enqueues" + a6tag);
            passed = false;
            continue;
          }
          HIP_CHECK(hipDeviceSynchronize());
          hipq::Turbo4KernelStatus a6_status =
              hipq::Turbo4KernelStatus::kWriterFailed;
          HIP_CHECK(hipMemcpy(&a6_status, device_status, sizeof(a6_status),
                              hipMemcpyDeviceToHost));
          Check(a6_status == hipq::Turbo4KernelStatus::kOk,
                "attn6 reader latches no failure" + a6tag);
          std::vector<float> a6(rows * vector_floats, 0.0f);
          HIP_CHECK(hipMemcpy(a6.data(), a6_out, rows * vector_bytes,
                              hipMemcpyDeviceToHost));
          bool a6_ok = a6_status == hipq::Turbo4KernelStatus::kOk;
          if (cap == 0U) {
            std::vector<std::uint8_t> parts(rows * row_partial);
            HIP_CHECK(hipMemcpy(parts.data(), a6_partials, parts.size(),
                                hipMemcpyDeviceToHost));
            std::size_t diff_parts = 0, diff_out = 0;
            for (std::size_t i = 0; i < parts.size(); ++i)
              diff_parts += parts[i] != wmma_parts[i];
            for (std::size_t i = 0; i < a6.size(); ++i)
              diff_out += std::memcmp(&a6[i], &actual[i], sizeof(float)) != 0;
            std::printf("  %s differing partial bytes=%zu output floats=%zu\n",
                        a6tag.c_str(), diff_parts, diff_out);
            Check(diff_parts == 0 && diff_out == 0,
                  "attn6 partials and output bitwise == WMMA" + a6tag);
            a6_ok = a6_ok && diff_parts == 0 && diff_out == 0;
          } else {
            double a6_l2 = 0.0;
            bool a6_finite = true;
            for (std::uint32_t row = 0; row < rows; ++row) {
              double diff2 = 0.0, ref2 = 0.0;
              for (std::size_t i = 0; i < vector_floats; ++i) {
                const float a = a6[row * vector_floats + i];
                const float e = expected[row * vector_floats + i];
                if (!std::isfinite(a)) {
                  a6_finite = false;
                  continue;
                }
                diff2 += (static_cast<double>(a) - e) * (a - e);
                ref2 += static_cast<double>(e) * e;
              }
              a6_l2 = std::max(a6_l2,
                               ref2 > 0.0 ? std::sqrt(diff2 / ref2) : 1.0);
            }
            std::printf("  %s max_row_l2=%.3e\n", a6tag.c_str(), a6_l2);
            Check(a6_finite && a6_l2 < kMaxRowL2,
                  "attn6 max per-row relative L2 error < 2e-3 vs split" +
                      a6tag);
            a6_ok = a6_ok && a6_finite && a6_l2 < kMaxRowL2;
          }
          passed = passed && a6_ok;
        }
        // VRD6: never bitwise (folded norms, exp2, P rounding), so every
        // plan gets the same L2 bound vs split; cap 0 = the WMMA plan.
        for (const std::uint32_t cap : {0U, 10U, 3U}) {
          const std::string vtag = tag + " vrd6_cap=" + std::to_string(cap);
          HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
          HIP_CHECK(hipMemset(a6_out, 0x6D, rows_bytes));
          HIP_CHECK(hipMemset(a6_partials, 0x6D, rows * row_partial));
          const auto v_launched = hipq::LaunchTurbo4AttentionVerifyVrd6(
              device_q, device_k, device_v, a6_out, a6_q_rot, a6_out_rot,
              a6_partials, plane_bytes, rows * vector_bytes,
              rows * vector_bytes, rows * vector_bytes, rows * vector_bytes,
              rows * row_partial, device_gate, rows * vector_bytes, kHeads,
              kKvHeads, kHeadDim, start_pos + 1U, rows, kMaxContext, scale,
              device_writer_status, device_status, nullptr, formats, cap);
          if (v_launched != hipq::Turbo4LaunchStatus::kLaunched) {
            Check(false, "vrd6 reader enqueues" + vtag);
            passed = false;
            continue;
          }
          HIP_CHECK(hipDeviceSynchronize());
          hipq::Turbo4KernelStatus v_status =
              hipq::Turbo4KernelStatus::kWriterFailed;
          HIP_CHECK(hipMemcpy(&v_status, device_status, sizeof(v_status),
                              hipMemcpyDeviceToHost));
          Check(v_status == hipq::Turbo4KernelStatus::kOk,
                "vrd6 reader latches no failure" + vtag);
          std::vector<float> vr(rows * vector_floats, 0.0f);
          HIP_CHECK(hipMemcpy(vr.data(), a6_out, rows * vector_bytes,
                              hipMemcpyDeviceToHost));
          double v_l2 = 0.0, v_abs = 0.0;
          bool v_finite = true;
          for (std::uint32_t row = 0; row < rows; ++row) {
            double diff2 = 0.0, ref2 = 0.0;
            for (std::size_t i = 0; i < vector_floats; ++i) {
              const float a = vr[row * vector_floats + i];
              const float e = expected[row * vector_floats + i];
              if (!std::isfinite(a)) {
                v_finite = false;
                continue;
              }
              const double d = static_cast<double>(a) - e;
              diff2 += d * d;
              ref2 += static_cast<double>(e) * e;
              v_abs = std::max(v_abs, std::fabs(d));
            }
            v_l2 = std::max(v_l2, ref2 > 0.0 ? std::sqrt(diff2 / ref2) : 1.0);
          }
          std::printf("  %s max_abs=%.3e max_row_l2=%.3e\n", vtag.c_str(),
                      v_abs, v_l2);
          Check(v_finite && v_l2 < kMaxRowL2,
                "vrd6 max per-row relative L2 error < 2e-3 vs split" + vtag);
          passed = passed && v_status == hipq::Turbo4KernelStatus::kOk &&
                   v_finite && v_l2 < kMaxRowL2;
        }
      }
    }
  }

  // VRD6 poison rule: a non-finite stored K norm inside the read range
  // latches kPoisonedNorm (the plane is restored afterwards).
  if (attn6_codec) {
    const std::uint32_t rows = 2, first = 4095U;
    const std::size_t row_partial = hipq::Turbo4BatchedDecodePartialBytes(
        first, rows, kHeads, kHeadDim);
    std::size_t base = 0;
    if (!hipq::Turbo4BlockByteOffset(1234, 2, kHeadDim, kKvHeads, kMaxContext,
                                     &base, formats.stride)) {
      Check(false, "vrd6 poison offset");
      return false;
    }
    const std::size_t at = base + t4::BlockBytes(formats.stride);  // block 1
    const std::uint16_t inf_bits = 0x7C00U;
    HIP_CHECK(hipMemcpy(device_k + at, &inf_bits, 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
    const auto v_launched = hipq::LaunchTurbo4AttentionVerifyVrd6(
        device_q, device_k, device_v, a6_out, a6_q_rot, a6_out_rot,
        a6_partials, plane_bytes, rows * vector_bytes, rows * vector_bytes,
        rows * vector_bytes, rows * vector_bytes, rows * row_partial,
        device_gate, rows * vector_bytes, kHeads, kKvHeads, kHeadDim, first,
        rows, kMaxContext, scale, device_writer_status, device_status,
        nullptr, formats, 10U);
    HIP_CHECK(hipDeviceSynchronize());
    hipq::Turbo4KernelStatus v_status = hipq::Turbo4KernelStatus::kOk;
    HIP_CHECK(hipMemcpy(&v_status, device_status, sizeof(v_status),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(device_k + at, host_k.data() + at, 2,
                        hipMemcpyHostToDevice));
    Check(v_launched == hipq::Turbo4LaunchStatus::kLaunched &&
              v_status == hipq::Turbo4KernelStatus::kPoisonedNorm,
          codec + " vrd6 latches kPoisonedNorm on an inf K norm");
    passed = passed && v_status == hipq::Turbo4KernelStatus::kPoisonedNorm;
  }

  if (timing) {
    const std::uint32_t rows = kMaxRows;
    const std::uint32_t first = kMaxContext - rows + 1U;  // last row = 32768
    const std::size_t row_partial = hipq::Turbo4BatchedDecodePartialBytes(
        first, rows, kHeads, kHeadDim);
    hipEvent_t begin_ev = nullptr, end_ev = nullptr;
    HIP_CHECK(hipEventCreate(&begin_ev));
    HIP_CHECK(hipEventCreate(&end_ev));
    auto launch_split = [&]() {
      return hipq::LaunchTurbo4AttentionSplitBatched(
          device_q, device_k, device_v, ref_out, ref_q_rot, ref_out_rot,
          ref_partials, plane_bytes, rows * vector_bytes, rows * vector_bytes,
          rows * vector_bytes, rows * vector_bytes, rows * row_partial,
          device_gate, rows * vector_bytes, kHeads, kKvHeads, kHeadDim, first,
          rows, kMaxContext, scale, device_writer_status, device_status,
          nullptr, formats);
    };
    auto launch_wmma = [&]() {
      return hipq::LaunchTurbo4AttentionVerifyWmma(
          device_q, device_k, device_v, wmma_out, wmma_q_rot, wmma_out_rot,
          wmma_partials, plane_bytes, rows * vector_bytes, rows * vector_bytes,
          rows * vector_bytes, rows * vector_bytes, rows * row_partial,
          device_gate, rows * vector_bytes, kHeads, kKvHeads, kHeadDim, first,
          rows, kMaxContext, scale, device_writer_status, device_status,
          nullptr, formats);
    };
    // Warm-up (one untimed launch of each), then 50 timed launches per reader.
    // Review: every launch, sync and event call is checked, and the
    // device status is read after the timed loop; any error returns -1 and
    // FAILs, so a zero or garbage time never prints as a result. The first
    // checked launch is the warm-up.
    const auto measure = [&](auto&& launch, const char* label) -> double {
      const std::string what = codec + " " + label + " timing";
      bool ok = hipMemset(device_status, 0, sizeof(*device_status)) ==
                    hipSuccess &&
                launch() == hipq::Turbo4LaunchStatus::kLaunched &&
                hipDeviceSynchronize() == hipSuccess &&
                hipEventRecord(begin_ev) == hipSuccess;
      for (int i = 0; ok && i < kTimingIterations; ++i)
        ok = launch() == hipq::Turbo4LaunchStatus::kLaunched;
      ok = ok && hipEventRecord(end_ev) == hipSuccess &&
           hipEventSynchronize(end_ev) == hipSuccess;
      float ms = 0.0f;
      ok = ok && hipEventElapsedTime(&ms, begin_ev, end_ev) == hipSuccess;
      hipq::Turbo4KernelStatus st = hipq::Turbo4KernelStatus::kOk;
      ok = ok &&
           hipMemcpy(&st, device_status, sizeof(st), hipMemcpyDeviceToHost) ==
               hipSuccess &&
           st == hipq::Turbo4KernelStatus::kOk && std::isfinite(ms) &&
           ms > 0.0f;
      Check(ok, what + " launches, events and device status ok");
      return ok ? static_cast<double>(ms) / kTimingIterations : -1.0;
    };
    const double split_ms = measure(launch_split, "split");
    const double wmma_ms = measure(launch_wmma, "wmma");
    std::printf("%s rows=%u ctx=%u split_ms=%.4f wmma_ms=%.4f ratio=%.3f\n",
                codec.c_str(), rows, kMaxContext, split_ms, wmma_ms,
                wmma_ms > 0.0 ? split_ms / wmma_ms : 0.0);
    HIP_CHECK(hipEventDestroy(begin_ev));
    HIP_CHECK(hipEventDestroy(end_ev));
  }

  (void)hipFree(device_k);
  (void)hipFree(device_v);
  (void)hipFree(device_q);
  (void)hipFree(device_gate);
  (void)hipFree(ref_out);
  (void)hipFree(ref_q_rot);
  (void)hipFree(ref_out_rot);
  (void)hipFree(ref_partials);
  (void)hipFree(wmma_out);
  (void)hipFree(wmma_q_rot);
  (void)hipFree(wmma_out_rot);
  (void)hipFree(wmma_partials);
  (void)hipFree(a6_out);
  (void)hipFree(a6_q_rot);
  (void)hipFree(a6_out_rot);
  (void)hipFree(a6_partials);
  (void)hipFree(device_writer_status);
  (void)hipFree(device_status);
  return passed;
}

}  // namespace

int main() {
  std::printf(
      "WMMA verify reader vs split reader (Turbo4/5/6)\n");
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("SKIP: no HIP device\n");
    return 0;
  }
  const char* const time_env = std::getenv("GUFO_TEST_TIME");
  const bool timing = time_env != nullptr && std::string(time_env) == "1";
  bool ok = RunCodec(t4::PlaneFormats(t4::Bits::k4),
                     hipq::QwenKvCacheStorage::kTurbo4, timing);
  ok = RunCodec(t4::PlaneFormats(t4::Bits::k5),
                hipq::QwenKvCacheStorage::kTurbo5, timing) &&
       ok;
  ok = RunCodec(t4::PlaneFormats(t4::Bits::k6),
                hipq::QwenKvCacheStorage::kTurbo6, timing) &&
       ok;
  // Mixed: 98-byte k6 slots, K k6, V k5. Valid() allows k/v below the stride.
  ok = RunCodec(t4::PlaneFormats(t4::Bits::k6, t4::Bits::k6, t4::Bits::k5),
                hipq::QwenKvCacheStorage::kTurbo6, timing) &&
       ok;
  std::printf("%s %d failure(s)\n", ok && failures == 0 ? "PASS" : "FAIL",
              failures);
  return ok && failures == 0 ? 0 : 1;
}

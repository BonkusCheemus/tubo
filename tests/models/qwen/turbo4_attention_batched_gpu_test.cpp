// Batched multi-row Turbo4 verify reader vs the per-row split reader.
// Row r of the batch must equal LaunchTurbo4AttentionSplit at context
// start_pos + r + 1 BYTE FOR BYTE (same split plan, same merge order).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace hipq = gufo::hip;

namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kMaxContext = 32768;
constexpr std::uint32_t kMaxRows = 8;

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

bool Run() {
  const std::size_t plane_bytes =
      hipq::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kMaxContext);
  const std::size_t vector_floats = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t vector_bytes = vector_floats * sizeof(float);
  const std::size_t single_partial_bytes =
      hipq::Turbo4DecodePartialBytes(kMaxContext, kHeads, kKvHeads, kHeadDim);
  // Every row slice is at most the max_context-sized single-row partials.
  const std::size_t batched_capacity = kMaxRows * single_partial_bytes;

  // Random Turbo4 KV through the production host writer.
  std::vector<std::uint8_t> host_k(plane_bytes), host_v(plane_bytes);
  const hipq::QwenKvPlaneGeometry geometry{kHeadDim, kKvHeads, kMaxContext};
  std::vector<float> krow(kHeadDim), vrow(kHeadDim);
  for (std::uint32_t pos = 0; pos < kMaxContext; ++pos) {
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const std::size_t seed =
            static_cast<std::size_t>(pos) * kKvHeads * kHeadDim +
            head * kHeadDim + d;
        krow[d] = 0.7f + 0.25f * Noise(seed + 10000);
        vrow[d] = 0.6f + 0.30f * Noise(seed + 40000);
      }
      if (!hipq::QwenKvWriteHead(hipq::QwenKvCacheStorage::kTurbo4,
                                 host_k.data(), plane_bytes, geometry, pos,
                                 head, krow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(hipq::QwenKvCacheStorage::kTurbo4)), true) ||
          !hipq::QwenKvWriteHead(hipq::QwenKvCacheStorage::kTurbo4,
                                 host_v.data(), plane_bytes, geometry, pos,
                                 head, vrow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(hipq::QwenKvCacheStorage::kTurbo4)), true)) {
        Check(false, "host Turbo4 writer produced every test row");
        return false;
      }
    }
  }

  // Distinct query/gate per row so a row mix-up cannot hide.
  std::vector<float> q(kMaxRows * vector_floats), gate(q.size());
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.8f + 0.4f * Noise(i + 88000);
    gate[i] = -0.5f + Noise(i + 99000);
  }

  std::uint8_t *device_k = nullptr, *device_v = nullptr;
  float *device_q = nullptr, *device_gate = nullptr;
  float *single_out = nullptr, *single_q_rot = nullptr;
  float *single_out_rot = nullptr, *single_partials = nullptr;
  float *batch_out = nullptr, *batch_q_rot = nullptr;
  float *batch_out_rot = nullptr, *batch_partials = nullptr;
  hipq::Turbo4KernelStatus *device_writer_status = nullptr,
                           *device_status = nullptr;
  const std::size_t rows_bytes = kMaxRows * vector_bytes;
  HIP_CHECK(hipMalloc(&device_k, plane_bytes));
  HIP_CHECK(hipMalloc(&device_v, plane_bytes));
  HIP_CHECK(hipMalloc(&device_q, rows_bytes));
  HIP_CHECK(hipMalloc(&device_gate, rows_bytes));
  HIP_CHECK(hipMalloc(&single_out, vector_bytes));
  HIP_CHECK(hipMalloc(&single_q_rot, vector_bytes));
  HIP_CHECK(hipMalloc(&single_out_rot, vector_bytes));
  HIP_CHECK(hipMalloc(&single_partials, single_partial_bytes));
  HIP_CHECK(hipMalloc(&batch_out, rows_bytes));
  HIP_CHECK(hipMalloc(&batch_q_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&batch_out_rot, rows_bytes));
  HIP_CHECK(hipMalloc(&batch_partials, batched_capacity));
  HIP_CHECK(hipMalloc(&device_writer_status, sizeof(*device_writer_status)));
  HIP_CHECK(hipMalloc(&device_status, sizeof(*device_status)));
  HIP_CHECK(
      hipMemcpy(device_k, host_k.data(), plane_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_v, host_v.data(), plane_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_gate, gate.data(), rows_bytes, hipMemcpyHostToDevice));
  const auto ok = hipq::Turbo4KernelStatus::kOk;
  HIP_CHECK(
      hipMemcpy(device_writer_status, &ok, sizeof(ok), hipMemcpyHostToDevice));
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  bool passed = true;
  for (const std::uint32_t start_pos : {0U, 254U, 4095U, 32760U}) {
    for (const std::uint32_t rows : {1U, 2U, 3U, 8U}) {
      if (start_pos + rows > kMaxContext)
        continue;
      const std::string tag = " start_pos=" + std::to_string(start_pos) +
                              " rows=" + std::to_string(rows);
      const std::size_t row_partial = hipq::Turbo4BatchedDecodePartialBytes(
          start_pos + 1U, rows, kHeads, kHeadDim);
      HIP_CHECK(
          hipMemcpy(device_q, q.data(), rows_bytes, hipMemcpyHostToDevice));

      // Reference: the per-row launcher, one row at a time.
      HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
      std::vector<float> expected(rows * vector_floats);
      for (std::uint32_t row = 0; row < rows; ++row) {
        const auto launched = hipq::LaunchTurbo4AttentionSplit(
            device_q + row * vector_floats, device_k, device_v, single_out,
            single_q_rot, single_out_rot, single_partials, plane_bytes,
            vector_bytes, vector_bytes, vector_bytes, vector_bytes,
            single_partial_bytes, device_gate + row * vector_floats,
            vector_bytes, kHeads, kKvHeads, kHeadDim, start_pos + row + 1U,
            kMaxContext, scale, device_writer_status, device_status, nullptr,
            false, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
        if (launched != hipq::Turbo4LaunchStatus::kLaunched) {
          Check(false, "per-row reference enqueues" + tag);
          return false;
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(expected.data() + row * vector_floats, single_out,
                            vector_bytes, hipMemcpyDeviceToHost));
      }

      auto run_batched = [&](std::vector<float>* actual) -> bool {
        HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
        HIP_CHECK(hipMemset(batch_out, 0x6D, rows_bytes));
        const auto launched = hipq::LaunchTurbo4AttentionSplitBatched(
            device_q, device_k, device_v, batch_out, batch_q_rot, batch_out_rot,
            batch_partials, plane_bytes, rows * vector_bytes,
            rows * vector_bytes, rows * vector_bytes, rows * vector_bytes,
            rows * row_partial, device_gate, rows * vector_bytes, kHeads,
            kKvHeads, kHeadDim, start_pos + 1U, rows, kMaxContext, scale,
            device_writer_status, device_status, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
        if (launched != hipq::Turbo4LaunchStatus::kLaunched) {
          Check(false, "batched reader enqueues" + tag);
          return false;
        }
        HIP_CHECK(hipDeviceSynchronize());
        hipq::Turbo4KernelStatus status =
            hipq::Turbo4KernelStatus::kWriterFailed;
        HIP_CHECK(hipMemcpy(&status, device_status, sizeof(status),
                            hipMemcpyDeviceToHost));
        Check(status == hipq::Turbo4KernelStatus::kOk,
              "batched reader latches no failure" + tag);
        actual->assign(rows * vector_floats, 0.0f);
        HIP_CHECK(hipMemcpy(actual->data(), batch_out, rows * vector_bytes,
                            hipMemcpyDeviceToHost));
        return status == hipq::Turbo4KernelStatus::kOk;
      };

      std::vector<float> actual;
      if (!run_batched(&actual)) {
        passed = false;
        continue;
      }
      const bool same =
          std::memcmp(actual.data(), expected.data(), rows * vector_bytes) == 0;
      Check(same, "batched output is bit-identical to per-row" + tag);
      passed = passed && same;

      // Teeth: a perturbed query on the LAST row must change that row only.
      std::vector<float> perturbed = q;
      const std::size_t last = (rows - 1U) * vector_floats;
      for (std::size_t i = 0; i < vector_floats; ++i)
        perturbed[last + i] += 0.25f;
      HIP_CHECK(hipMemcpy(device_q, perturbed.data(), rows_bytes,
                          hipMemcpyHostToDevice));
      std::vector<float> changed;
      if (!run_batched(&changed)) {
        passed = false;
        continue;
      }
      // One visible key (start_pos 0, one row): softmax weight is 1 whatever
      // the query, so the output is that key's value and cannot change.
      const bool single_key = start_pos + rows == 1U;
      const bool last_differs =
          single_key || std::memcmp(changed.data() + last,
                                    expected.data() + last, vector_bytes) != 0;
      const bool earlier_same =
          last == 0U || std::memcmp(changed.data(), expected.data(),
                                    last * sizeof(float)) == 0;
      Check(last_differs,
            "teeth: perturbed last-row query changes its output" + tag);
      Check(earlier_same,
            "teeth: earlier rows are untouched by the perturbation" + tag);
      passed = passed && last_differs && earlier_same;
    }
  }

  (void)hipFree(device_k);
  (void)hipFree(device_v);
  (void)hipFree(device_q);
  (void)hipFree(device_gate);
  (void)hipFree(single_out);
  (void)hipFree(single_q_rot);
  (void)hipFree(single_out_rot);
  (void)hipFree(single_partials);
  (void)hipFree(batch_out);
  (void)hipFree(batch_q_rot);
  (void)hipFree(batch_out_rot);
  (void)hipFree(batch_partials);
  (void)hipFree(device_writer_status);
  (void)hipFree(device_status);
  return passed;
}

}  // namespace

int main() {
  std::printf("batched Turbo4 verify reader vs per-row split reader\n");
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("SKIP: no HIP device\n");
    return 0;
  }
  const bool ok = Run();
  std::printf("%s %d failure(s)\n", ok && failures == 0 ? "PASS" : "FAIL",
              failures);
  return ok && failures == 0 ? 0 : 1;
}

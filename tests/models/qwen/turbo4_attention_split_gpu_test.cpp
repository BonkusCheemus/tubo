// Split-K Turbo4 reader against the scalar GPU and double CPU
// oracle.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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
constexpr std::uint32_t kMaxContext = 8192;

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

bool HostDoubleOracle(const std::uint8_t* k, const std::uint8_t* v,
                      std::size_t plane_bytes, std::uint32_t context,
                      const std::vector<float>& q,
                      const std::vector<float>& gate,
                      std::vector<float>* result) {
  if (result == nullptr)
    return false;
  const hipq::QwenKvPlaneGeometry geometry{kHeadDim, kKvHeads, context};
  result->assign(kHeads * kHeadDim, 0.0f);
  std::vector<float> krow(kHeadDim), vrow(kHeadDim);
  for (std::uint32_t qh = 0; qh < kHeads; ++qh) {
    const std::size_t kvh = qh / (kHeads / kKvHeads);
    std::vector<double> scores(context);
    std::vector<std::vector<float>> values(context,
                                           std::vector<float>(kHeadDim));
    for (std::uint32_t pos = 0; pos < context; ++pos) {
      if (!hipq::QwenKvReadHead(hipq::QwenKvCacheStorage::kTurbo4, k,
                                plane_bytes, geometry, pos, kvh, krow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(hipq::QwenKvCacheStorage::kTurbo4)), true) ||
          !hipq::QwenKvReadHead(hipq::QwenKvCacheStorage::kTurbo4, v,
                                plane_bytes, geometry, pos, kvh, vrow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(hipq::QwenKvCacheStorage::kTurbo4)), true))
        return false;
      double dot = 0.0;
      for (std::uint32_t d = 0; d < kHeadDim; ++d)
        dot += static_cast<double>(q[qh * kHeadDim + d]) * krow[d];
      scores[pos] = dot / std::sqrt(static_cast<double>(kHeadDim));
      values[pos] = vrow;
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    std::vector<double> numerator(kHeadDim, 0.0);
    for (std::uint32_t pos = 0; pos < context; ++pos) {
      const double weight = std::exp(scores[pos] - maximum);
      denominator += weight;
      for (std::uint32_t d = 0; d < kHeadDim; ++d)
        numerator[d] += weight * values[pos][d];
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator))
      return false;
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      const double gate_value = gate[qh * kHeadDim + d];
      const double sigmoid = 1.0 / (1.0 + std::exp(-gate_value));
      (*result)[qh * kHeadDim + d] =
          static_cast<float>(numerator[d] / denominator * sigmoid);
    }
  }
  return true;
}

double RelativeL2(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
  double error = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double delta = static_cast<double>(actual[i]) - expected[i];
    error += delta * delta;
    norm += static_cast<double>(expected[i]) * expected[i];
  }
  return std::sqrt(error) / std::max(1e-12, std::sqrt(norm));
}

bool Run() {
  const std::size_t plane_bytes =
      hipq::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, kMaxContext);
  const std::size_t vector_bytes =
      static_cast<std::size_t>(kHeads) * kHeadDim * sizeof(float);
  const std::size_t partial_bytes =
      hipq::Turbo4DecodePartialBytes(kMaxContext, kHeads, kKvHeads, kHeadDim);
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

  std::vector<float> q(kHeads * kHeadDim), gate(kHeads * kHeadDim);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 0.8f + 0.4f * Noise(i + 88000);
    gate[i] = -0.5f + Noise(i + 99000);
  }

  std::uint8_t *device_k = nullptr, *device_v = nullptr;
  float *device_q = nullptr, *device_gate = nullptr, *device_out = nullptr;
  float *device_q_rot = nullptr, *device_out_rot = nullptr,
        *device_partials = nullptr;
  auto* device_writer_status = static_cast<hipq::Turbo4KernelStatus*>(nullptr);
  auto* device_status = static_cast<hipq::Turbo4KernelStatus*>(nullptr);
  HIP_CHECK(hipMalloc(&device_k, plane_bytes));
  HIP_CHECK(hipMalloc(&device_v, plane_bytes));
  HIP_CHECK(hipMalloc(&device_q, vector_bytes));
  HIP_CHECK(hipMalloc(&device_gate, vector_bytes));
  HIP_CHECK(hipMalloc(&device_out, vector_bytes));
  HIP_CHECK(hipMalloc(&device_q_rot, vector_bytes));
  HIP_CHECK(hipMalloc(&device_out_rot, vector_bytes));
  HIP_CHECK(hipMalloc(&device_partials, partial_bytes));
  HIP_CHECK(hipMalloc(&device_writer_status, sizeof(*device_writer_status)));
  HIP_CHECK(hipMalloc(&device_status, sizeof(*device_status)));
  HIP_CHECK(
      hipMemcpy(device_k, host_k.data(), plane_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_v, host_v.data(), plane_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_q, q.data(), vector_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(device_gate, gate.data(), vector_bytes, hipMemcpyHostToDevice));
  const auto ok = hipq::Turbo4KernelStatus::kOk;
  HIP_CHECK(
      hipMemcpy(device_writer_status, &ok, sizeof(ok), hipMemcpyHostToDevice));

  bool passed = true;
  for (const std::uint32_t context :
       {1U, 7U, 8U, 9U, 63U, 64U, 65U, 4095U, 4096U, 4097U, 8192U}) {
    HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
    HIP_CHECK(hipMemset(device_out, 0x6D, vector_bytes));
    const auto launched = hipq::LaunchTurbo4AttentionSplit(
        device_q, device_k, device_v, device_out, device_q_rot, device_out_rot,
        device_partials, plane_bytes, vector_bytes, vector_bytes, vector_bytes,
        vector_bytes, partial_bytes, device_gate, vector_bytes, kHeads,
        kKvHeads, kHeadDim, context, kMaxContext,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)), device_writer_status,
        device_status, nullptr, false, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    Check(launched == hipq::Turbo4LaunchStatus::kLaunched,
          "split reader enqueues at context " + std::to_string(context));
    if (launched != hipq::Turbo4LaunchStatus::kLaunched) {
      passed = false;
      continue;
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipq::Turbo4KernelStatus status = hipq::Turbo4KernelStatus::kWriterFailed;
    HIP_CHECK(hipMemcpy(&status, device_status, sizeof(status),
                        hipMemcpyDeviceToHost));
    std::vector<float> actual(kHeads * kHeadDim);
    HIP_CHECK(hipMemcpy(actual.data(), device_out, vector_bytes,
                        hipMemcpyDeviceToHost));
    std::vector<float> expected;
    const bool oracle = HostDoubleOracle(
        host_k.data(), host_v.data(), plane_bytes, context, q, gate, &expected);
    const double error = oracle ? RelativeL2(actual, expected) : INFINITY;
    const bool good =
        status == hipq::Turbo4KernelStatus::kOk && oracle && error < 8e-4;
    Check(good, "split reader agrees with double host oracle at context " +
                    std::to_string(context) +
                    " relL2=" + std::to_string(error));
    passed = passed && good;

    if (context <= 4096U) {
      HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
      const auto scalar = hipq::LaunchTurbo4AttentionScalar(
          device_q, device_k, device_v, device_out, plane_bytes, vector_bytes,
          vector_bytes, device_gate, vector_bytes, kHeads, kKvHeads, kHeadDim,
          context, 1.0f / std::sqrt(static_cast<float>(kHeadDim)),
          device_writer_status, device_status, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
      Check(scalar == hipq::Turbo4LaunchStatus::kLaunched,
            "scalar oracle enqueues through 4096");
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(&status, device_status, sizeof(status),
                          hipMemcpyDeviceToHost));
      std::vector<float> scalar_out(kHeads * kHeadDim);
      HIP_CHECK(hipMemcpy(scalar_out.data(), device_out, vector_bytes,
                          hipMemcpyDeviceToHost));
      const double scalar_error = RelativeL2(actual, scalar_out);
      const bool scalar_good =
          status == hipq::Turbo4KernelStatus::kOk && scalar_error < 8e-4;
      Check(scalar_good, "split reader matches scalar GPU at context " +
                             std::to_string(context) +
                             " relL2=" + std::to_string(scalar_error));
      passed = passed && scalar_good;
    }
  }

  std::vector<float> sentinel(kHeads * kHeadDim, 123.25f),
      untouched(vector_bytes / sizeof(float));
  HIP_CHECK(hipMemcpy(device_out, sentinel.data(), vector_bytes,
                      hipMemcpyHostToDevice));
  const auto beyond = hipq::LaunchTurbo4AttentionSplit(
      device_q, device_k, device_v, device_out, device_q_rot, device_out_rot,
      device_partials, plane_bytes, vector_bytes, vector_bytes, vector_bytes,
      vector_bytes, partial_bytes, device_gate, vector_bytes, kHeads, kKvHeads,
      kHeadDim, kMaxContext + 1, kMaxContext,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)), device_writer_status,
      device_status, nullptr, false, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  HIP_CHECK(hipMemcpy(untouched.data(), device_out, vector_bytes,
                      hipMemcpyDeviceToHost));
  Check(beyond == hipq::Turbo4LaunchStatus::kGeometryRefused &&
            untouched == sentinel,
        "context past max_context refuses before touching output");

  const auto writer_failed = hipq::Turbo4KernelStatus::kWriterFailed;
  HIP_CHECK(hipMemcpy(device_writer_status, &writer_failed,
                      sizeof(writer_failed), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(device_status, 0, sizeof(*device_status)));
  HIP_CHECK(hipMemcpy(device_out, sentinel.data(), vector_bytes,
                      hipMemcpyHostToDevice));
  const auto refused_writer = hipq::LaunchTurbo4AttentionSplit(
      device_q, device_k, device_v, device_out, device_q_rot, device_out_rot,
      device_partials, plane_bytes, vector_bytes, vector_bytes, vector_bytes,
      vector_bytes, partial_bytes, device_gate, vector_bytes, kHeads, kKvHeads,
      kHeadDim, 1, kMaxContext, 1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      device_writer_status, device_status, nullptr, false, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  HIP_CHECK(hipDeviceSynchronize());
  hipq::Turbo4KernelStatus writer_status = hipq::Turbo4KernelStatus::kOk;
  HIP_CHECK(hipMemcpy(&writer_status, device_status, sizeof(writer_status),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(untouched.data(), device_out, vector_bytes,
                      hipMemcpyDeviceToHost));
  Check(refused_writer == hipq::Turbo4LaunchStatus::kLaunched &&
            writer_status == hipq::Turbo4KernelStatus::kWriterFailed &&
            untouched == sentinel,
        "writer failure stays latched and leaves output untouched");

  (void)hipFree(device_k);
  (void)hipFree(device_v);
  (void)hipFree(device_q);
  (void)hipFree(device_gate);
  (void)hipFree(device_out);
  (void)hipFree(device_q_rot);
  (void)hipFree(device_out_rot);
  (void)hipFree(device_partials);
  (void)hipFree(device_writer_status);
  (void)hipFree(device_status);
  return passed;
}

}  // namespace

int main() {
  std::printf("E5 split-K Turbo4 reader vs scalar and double CPU oracles\n");
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

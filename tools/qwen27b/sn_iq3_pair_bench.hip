// Real IQ3 pair, same-ActK V3 oracle, per-slice gate, capture,
// refresh and per-launch timing, B1..8 ((c) stage 3 adds B1). No model
// executor fence is removed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3_verify.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"

namespace {
using gufo::core::GgmlType;
using gufo::models::QwenTensorRef;
constexpr std::size_t M = 34816, H = M / 2, K = 5120;
void Check(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t rc) {
  Check(rc == hipSuccess, hipGetErrorString(rc));
}
struct Device {
  void* p = nullptr;
  explicit Device(std::size_t n) { Hip(hipMalloc(&p, n)); }
  ~Device() { (void)hipFree(p); }
  template<class T>
  T* as() const {
    return static_cast<T*>(p);
  }
};
struct Stream {
  hipStream_t s;
  Stream() { Hip(hipStreamCreateWithFlags(&s, hipStreamNonBlocking)); }
  ~Stream() { (void)hipStreamDestroy(s); }
};
void Compare(const std::vector<float>& ref, const std::vector<float>& got,
             std::size_t batch, bool packed, const char* scope,
             const char* type) {
  Check(ref.size() == batch * M && got.size() == ref.size(), "output size");
  for (std::size_t slice = 0; slice < 2; ++slice) {
    double error = 0, scale = 0;
    for (std::size_t t = 0; t < batch; ++t)
      for (std::size_t r = 0; r < H; ++r) {
        const auto i = t * M + slice * H + r;
        const auto j = packed ? i : slice * batch * H + t * H + r;
        Check(std::isfinite(ref[i]) && std::isfinite(got[j]),
              "nonfinite/missing store");
        error = std::max(error, std::abs(double(ref[i]) - got[j]));
        scale = std::max(scale, std::abs(double(ref[i])));
      }
    const double rel = scale == 0 ? error : error / scale;
    std::printf(
        "PAIR type=%s B=%zu scope=%s slice=%zu max_abs=%.9g max_ref=%.9g "
        "rel=%.9g\n",
        type, batch, scope, slice, error, scale, rel);
    Check(rel <= 1e-3, "per-slice same-ActK gate failed");
  }
}

template<class F>
double Time(F run, hipStream_t stream) {
  hipEvent_t a, b;
  Hip(hipEventCreate(&a));
  Hip(hipEventCreate(&b));
  for (int i = 0; i < 4; ++i)
    run();
  Hip(hipStreamSynchronize(stream));
  std::vector<double> samples;
  for (int round = 0; round < 5; ++round) {
    Hip(hipEventRecord(a, stream));
    for (int i = 0; i < 16; ++i)
      run();
    Hip(hipEventRecord(b, stream));
    Hip(hipEventSynchronize(b));
    float ms = 0;
    Hip(hipEventElapsedTime(&ms, a, b));
    samples.push_back(ms * 1000.0 / 16);
    std::printf("RAW round=%d us=%.9g\n", round, samples.back());
  }
  Hip(hipEventDestroy(a));
  Hip(hipEventDestroy(b));
  std::sort(samples.begin(), samples.end());
  return samples[2];
}

template<GgmlType Type, std::size_t B>
void Run(const void* native_host, std::size_t bytes, const char* name) {
  using namespace gufo::hip;
  const auto layout = Type == GgmlType::kIQ3_S
                          ? gufo::models::QwenWeightLayout::kIq3SRows8
                          : gufo::models::QwenWeightLayout::kIq3XxsRows4;
  // Bench-only native oracle and SN bytes, not a serving weight mirror.
  std::vector<std::uint8_t> packed(bytes);
  gufo::models::CopyIq3RowRangeFast(native_host, packed.data(), layout, M, K, 0,
                                    M);
  Device native(bytes), sn(bytes), x(B * K * sizeof(float));
  Device q(B * K), ds(B * K / 16 * sizeof(float2));
  Device ref(B * M * sizeof(float)), out(B * M * sizeof(float));
  Device check(B * M * sizeof(float));
  Stream stream;
  Hip(hipMemcpy(native.p, native_host, bytes, hipMemcpyHostToDevice));
  Hip(hipMemcpy(sn.p, packed.data(), bytes, hipMemcpyHostToDevice));
  const auto views = gufo::models::MakeIq3PairViews(sn.p, Type, M, K, bytes);
  ReserveDot4Scratch(8U * K, stream.s);
  std::vector<float> hx(B * K), expected(B * M), actual(B * M);
  bool refreshed = false;
  std::vector<float> first;
  for (int input = 0; input < 3; ++input) {
    // Zero input is exact; mixed signs, tiny values and quantizer boundaries
    // in both later inputs expose reuse across fresh FFN invocations.
    for (std::size_t i = 0; i < hx.size(); ++i) {
      const int v = int((i * 37 + input * 13) % 255) - 127;
      hx[i] = input == 0
                  ? 0.0F
                  : v * (i % 17 == 0 ? 1e-7F : (input == 1 ? 0.003F : -0.007F));
    }
    Hip(hipMemcpyAsync(x.p, hx.data(), hx.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream.s));
    constexpr auto subs = B * K / 16;
    QuantizeActivationsQ8KDot4Kernel<<<(subs + 255) / 256, 256, 0, stream.s>>>(
        x.as<float>(), q.as<std::int8_t>(), ds.as<float2>(), subs);
    // V3 ActK=false intentionally consumes the PREPARED q8_K buffers. Its
    // per-16 scales are the q8_K producer's repeated per-256 scale, not a
    // second/per-16 activation quantization. Native encoded weights are read
    // by the unchanged V3 body, independent of the SN address/decode code.
    SmallBatchDot4KQuantV3Kernel<Type, B, 4, 4>
        <<<(M + 15) / 16, 128, 0, stream.s>>>(native.p, q.as<std::int8_t>(),
                                              ds.as<float2>(), ref.as<float>(),
                                              M, K);
    Hip(hipMemcpyAsync(expected.data(), ref.p, B * M * sizeof(float),
                       hipMemcpyDeviceToHost, stream.s));
    Hip(hipStreamSynchronize(stream.s));
    for (bool pair : {false, true}) {
      Hip(hipMemsetAsync(out.p, 0xff, B * M * sizeof(float), stream.s));
      LaunchSnIq3Pair(views.gate, views.up, x.as<float>(), out.as<float>(),
                      out.as<float>() + B * H, pair ? out.as<float>() : nullptr,
                      B, stream.s);
      Hip(hipMemcpyAsync(actual.data(), out.p, B * M * sizeof(float),
                         hipMemcpyDeviceToHost, stream.s));
      Hip(hipStreamSynchronize(stream.s));
      Compare(expected, actual, B, pair, pair ? "packed" : "separate", name);
      if (input == 1 && pair)
        first = actual;
      if (input == 2 && pair)
        refreshed = actual != first;
    }
  }
  Check(refreshed, "fresh input did not change outputs");
  // A capture contains precisely one activation producer and one/two weight
  // consumers. Allocations were pre-reserved; replay follows a changed input.
  for (bool pair : {false, true}) {
    hipGraph_t graph;
    hipGraphExec_t exec;
    Hip(hipStreamBeginCapture(stream.s, hipStreamCaptureModeThreadLocal));
    LaunchSnIq3Pair(views.gate, views.up, x.as<float>(), out.as<float>(),
                    out.as<float>() + B * H, pair ? out.as<float>() : nullptr,
                    B, stream.s);
    Hip(hipStreamEndCapture(stream.s, &graph));
    std::size_t nodes = 0;
    Hip(hipGraphGetNodes(graph, nullptr, &nodes));
    Check(nodes == (pair ? 2U : 3U), "capture quantizer/consumer count");
    Hip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    // Change input AFTER capture. A cached producer/first-call-only quantizer
    // would now replay stale activations and fail the native same-ActK oracle.
    for (auto& v : hx)
      v *= -0.33F;
    Hip(hipMemcpyAsync(x.p, hx.data(), hx.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream.s));
    constexpr auto subs = B * K / 16;
    QuantizeActivationsQ8KDot4Kernel<<<(subs + 255) / 256, 256, 0, stream.s>>>(
        x.as<float>(), q.as<std::int8_t>(), ds.as<float2>(), subs);
    SmallBatchDot4KQuantV3Kernel<Type, B, 4, 4>
        <<<(M + 15) / 16, 128, 0, stream.s>>>(native.p, q.as<std::int8_t>(),
                                              ds.as<float2>(), ref.as<float>(),
                                              M, K);
    Hip(hipMemcpyAsync(expected.data(), ref.p, B * M * sizeof(float),
                       hipMemcpyDeviceToHost, stream.s));
    Hip(hipMemsetAsync(out.p, 0xff, B * M * sizeof(float), stream.s));
    Hip(hipGraphLaunch(exec, stream.s));
    Hip(hipMemcpyAsync(actual.data(), out.p, B * M * sizeof(float),
                       hipMemcpyDeviceToHost, stream.s));
    Hip(hipStreamSynchronize(stream.s));
    Compare(expected, actual, B, pair, pair ? "graph_packed" : "graph_separate",
            name);
    Hip(hipGraphExecDestroy(exec));
    Hip(hipGraphDestroy(graph));
  }
  for (const auto* view : {&views.gate, &views.up, &views.pair}) {
    const auto rows = view->num_elements / K;
    std::printf(
        "TIMING_BEGIN type=%s B=%zu rows=%zu row0=%zu scope=weight_only n=80 "
        "rounds=5\n",
        name, B, rows, view->layout_row_begin);
    const auto us = Time(
        [&] {
          LaunchSnIq3Prepared<B>(*view, q.as<std::int8_t>(), ds.as<float2>(),
                                 check.as<float>(), stream.s);
        },
        stream.s);
    Hip(hipGetLastError());
    std::printf("RESULT type=%s B=%zu rows=%zu row0=%zu us=%.9g GBs=%.9g\n",
                name, B, rows, view->layout_row_begin, us,
                (rows * (K / 256) * (Type == GgmlType::kIQ3_S ? 110 : 98)) /
                    us / 1000.0);
  }
  Hip(hipGetLastError());
  std::printf("PASS type=%s B=%zu refresh=1 graphs=2 per_slice=1\n", name, B);
}
template<GgmlType Type>
void Batch(std::size_t b, const void* w, std::size_t bytes, const char* name) {
  switch (b) {
    case 1:  // single-token route; same ed6 gate as B2..8.
      Run<Type, 1>(w, bytes, name);
      break;
    case 2:
      Run<Type, 2>(w, bytes, name);
      break;
    case 3:
      Run<Type, 3>(w, bytes, name);
      break;
    case 4:
      Run<Type, 4>(w, bytes, name);
      break;
    case 5:
      Run<Type, 5>(w, bytes, name);
      break;
    case 6:
      Run<Type, 6>(w, bytes, name);
      break;
    case 7:
      Run<Type, 7>(w, bytes, name);
      break;
    case 8:
      Run<Type, 8>(w, bytes, name);
      break;
    default:
      throw std::invalid_argument("B must be 1..8");
  }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      std::vector<float> good(2 * M, 1), bad = good;
      Compare(good, good, 2, true, "checker_positive", "CPU");
      for (int mutation = 0; mutation < 3; ++mutation) {
        bad = good;
        bad[H] = mutation == 0   ? std::numeric_limits<float>::quiet_NaN()
                 : mutation == 1 ? 2.0F
                                 : std::numeric_limits<float>::infinity();
        bool rejected = false;
        try {
          Compare(good, bad, 2, true, "checker_negative", "CPU");
        } catch (const std::runtime_error&) {
          rejected = true;
        }
        Check(rejected, "checker accepted bad/missing up slice");
      }
      std::puts("PASS CPU checker rejects missing/error/nonfinite up slice");
      const auto views = gufo::models::MakeIq3PairViews(
          reinterpret_cast<const void*>(std::uintptr_t{0x1000}),
          GgmlType::kIQ3_S, M, K, M * (K / 256) * 110);
      for (int mutation = 0; mutation < 4; ++mutation) {
        auto bad_view = views.gate;
        if (mutation == 0)
          bad_view.layout = gufo::models::QwenWeightLayout::kNative;
        if (mutation == 1) {
          bad_view.layout_columns = 4864;  // 19 K blocks: unequal barriers.
          bad_view.num_elements = H * 4864;
        }
        if (mutation == 2)
          bad_view.layout_row_begin = 4;
        if (mutation == 3)
          bad_view.num_elements -= K;
        bool rejected = false;
        try {
          gufo::hip::LaunchSnIq3Prepared<2>(bad_view, nullptr, nullptr, nullptr,
                                            nullptr);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        Check(rejected, "prepared launcher accepted unsupported geometry");
      }
      std::puts(
          "PASS CPU actual prepared-launch guard: "
          "native/19-block/row0/row-count rejected before HIP");
      return 0;
    }
    Check(argc == 4, "usage: gufo_sn_iq3_pair_bench MODEL S|XXS B");
    const std::string type = argv[2];
    Check(type == "S" || type == "XXS", "type must be S|XXS");
    const auto b = std::stoul(argv[3]);
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Check(bool(reader), error.c_str());
    const auto layer = type == "S" ? 0 : 16;
    const auto* gate =
        reader->FindTensor("blk." + std::to_string(layer) + ".ffn_gate.weight");
    const auto* up =
        reader->FindTensor("blk." + std::to_string(layer) + ".ffn_up.weight");
    const auto t = type == "S" ? GgmlType::kIQ3_S : GgmlType::kIQ3_XXS;
    Check(gate && up && gate->type == t && up->type == t &&
              gate->dimensions == std::vector<std::uint64_t>({K, H}) &&
              up->dimensions == gate->dimensions &&
              up->size_bytes == gate->size_bytes &&
              static_cast<const std::byte*>(gate->data) + gate->size_bytes ==
                  up->data,
          "invalid real model pair");
    if (type == "S")
      Batch<GgmlType::kIQ3_S>(b, gate->data, 2 * gate->size_bytes, "S");
    else
      Batch<GgmlType::kIQ3_XXS>(b, gate->data, 2 * gate->size_bytes, "XXS");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}

// Is the existing int8 WMMA kernel
// (WKQuantA8BlockedWmmaGEMMKernel, reached via LaunchBatchedQuantGEMMPreQuantized;
// today prefill-only) faster than the production verify route
// (LaunchBatchedQuantGEMMFp32 -> v3 / dot4 / exact) at batch 2-8? Real weights:
// raw tensor bytes dumped from the GGUF (tools/bench/dump_tensor.py), random
// fp32 activations. The weight buffer is replicated past 256 MB so every timed
// launch reads DRAM, not the 32 MB MALL. Both routes quantize activations to
// int8, so neither is a bit-exact reference: rel_diff is max|a-b| / max|a|.
//
// usage: verify_wmma_bench <ggml_type_id> <m> <k> <raw_file> <batch> [iters=24]
//        verify_wmma_bench attention [iters=20] [splits_cap=10]
//        verify_wmma_bench vrd6 [iters=20] [splits_cap=10]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace {

void Check(hipError_t e) {
  if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}

// ATTN6 attention-reader arm (same CLI as the reader v2 arm): a real
// Turbo6 plane written through the production host writer, then the
// pipelined WMMA reader ("wmma"), the attn6 reader on the same split plan
// ("attn6", partials bitwise vs wmma) and the attn6 reader with
// splits_cap splits per KV head ("attn6_capN") timed over the same plane.
// GBps = compulsory K+V plane bytes of ctx tokens / per-launch time (rotate +
// reader + reduce, as the served launcher). ctx 8192 fits the 32 MB MALL
// (12.8 MB), so its GBps is not DRAM. Run with GUFO_DEV_ATTN6_V3 unset, so
// "wmma" is the old reader.
//
// Numerics at rows 8 of each ctx: out_rot (rotated basis, before the inverse
// WHT and gate) of each arm vs an fp64 host reference over the SAME stored
// codes (value = centroid * norm in double, q = the device's q_rot): max_abs,
// max_rel (element, floored at 1e-3 of the vector's peak) and max vector L2.
// "bitwise" = attn6 partials and out_rot equal to wmma's, bit for bit.
float BenchNoise(std::size_t n) {
  std::uint64_t x = n * 6364136223846793005ULL + 1442695040888963407ULL;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return static_cast<float>(x >> 40) / 16777216.0f - 0.5f;
}

int RunAttentionReaderBench(int iters, std::uint32_t splits_cap) {
  constexpr std::uint32_t kHeadDim = 256;
  constexpr std::uint32_t kKvHeads = 4;
  constexpr std::uint32_t kHeads = 24;
  constexpr std::uint32_t kMaxContext = 65536;
  constexpr std::uint32_t kMaxRows = 8;
  namespace t4 = gufo::hip::turbo4;
  using gufo::hip::Turbo4KernelStatus;
  using gufo::hip::Turbo4LaunchStatus;
  const t4::PlaneFormats formats(t4::Bits::k6);
  const std::size_t vector_floats = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t vector_bytes = vector_floats * sizeof(float);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  const std::size_t plane_bytes = gufo::hip::Turbo4LayerPlaneBytes(
      kHeadDim, kKvHeads, kMaxContext, formats.stride);
  const std::size_t single_partial = gufo::hip::Turbo4DecodePartialBytes(
      kMaxContext, kHeads, kKvHeads, kHeadDim);
  if (plane_bytes == 0 || single_partial == 0)
    throw std::runtime_error("attention bench geometry sizing");

  std::vector<std::uint8_t> host_k(plane_bytes), host_v(plane_bytes);
  const gufo::hip::QwenKvPlaneGeometry geometry{kHeadDim, kKvHeads,
                                                kMaxContext};
  std::vector<float> krow(kHeadDim), vrow(kHeadDim);
  for (std::uint32_t pos = 0; pos < kMaxContext; ++pos)
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const std::size_t seed =
            static_cast<std::size_t>(pos) * kKvHeads * kHeadDim +
            head * kHeadDim + d;
        krow[d] = 3.0f * BenchNoise(seed + 10000);
        vrow[d] = 0.25f + BenchNoise(seed + 40000);
      }
      if (!gufo::hip::QwenKvWriteHead(gufo::hip::QwenKvCacheStorage::kTurbo6,
                                      host_k.data(), plane_bytes, geometry,
                                      pos, head, krow.data(), formats, true) ||
          !gufo::hip::QwenKvWriteHead(gufo::hip::QwenKvCacheStorage::kTurbo6,
                                      host_v.data(), plane_bytes, geometry,
                                      pos, head, vrow.data(), formats, false))
        throw std::runtime_error("attention bench host writer refused");
    }

  std::uint8_t *device_k = nullptr, *device_v = nullptr;
  float *device_q = nullptr, *device_gate = nullptr;
  Turbo4KernelStatus *writer_status = nullptr, *device_status = nullptr;
  const std::size_t rows_bytes = kMaxRows * vector_bytes;
  const std::size_t partial_cap = kMaxRows * single_partial;
  std::vector<float> q(kMaxRows * vector_floats),
      gate(kMaxRows * vector_floats);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 4.0f * BenchNoise(i + 88000);
    gate[i] = -0.5f + BenchNoise(i + 99000);
  }
  // Per arm: out, q_rot, out_rot, partials.
  struct ArmBuffers {
    float *out = nullptr, *q_rot = nullptr, *out_rot = nullptr,
          *partials = nullptr;
  };
  ArmBuffers arm[3];
  Check(hipMalloc(&device_k, plane_bytes));
  Check(hipMalloc(&device_v, plane_bytes));
  Check(hipMalloc(&device_q, rows_bytes));
  Check(hipMalloc(&device_gate, rows_bytes));
  for (auto& a : arm) {
    Check(hipMalloc(&a.out, rows_bytes));
    Check(hipMalloc(&a.q_rot, rows_bytes));
    Check(hipMalloc(&a.out_rot, rows_bytes));
    Check(hipMalloc(&a.partials, partial_cap));
  }
  Check(hipMalloc(&writer_status, sizeof(*writer_status)));
  Check(hipMalloc(&device_status, sizeof(*device_status)));
  Check(hipMemcpy(device_k, host_k.data(), plane_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(device_v, host_v.data(), plane_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(device_q, q.data(), rows_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(device_gate, gate.data(), rows_bytes, hipMemcpyHostToDevice));
  const Turbo4KernelStatus ok = Turbo4KernelStatus::kOk;
  Check(hipMemcpy(writer_status, &ok, sizeof(ok), hipMemcpyHostToDevice));

  hipEvent_t begin_ev = nullptr, end_ev = nullptr;
  Check(hipEventCreate(&begin_ev));
  Check(hipEventCreate(&end_ev));
  const std::string cap_name = "attn6_cap" + std::to_string(splits_cap);
  const char* const names[3] = {"wmma", "attn6", cap_name.c_str()};
  std::printf("reader,codec,ctx,rows,ms,GBps\n");
  std::vector<std::string> numerics;

  // Host fp64 decode of one stored value (the same codes the GPU reads).
  const auto stored = [&](const std::vector<std::uint8_t>& plane,
                          std::uint32_t token, std::uint32_t head,
                          std::uint32_t d) {
    std::size_t base = 0;
    if (!gufo::hip::Turbo4BlockByteOffset(token, head, kHeadDim, kKvHeads,
                                          kMaxContext, &base,
                                          formats.stride))
      throw std::runtime_error("attention bench offset");
    const std::uint8_t* block =
        plane.data() + base + (d / 128U) * t4::BlockBytes(formats.stride);
    std::uint16_t bits = 0;
    std::memcpy(&bits, block, sizeof(bits));
    return static_cast<double>(t4::CentroidValue(
               t4::Bits::k6, t4::CodeAt(t4::Bits::k6, block, d % 128U))) *
           static_cast<double>(t4::HalfBitsToFloat(bits));
  };

  for (const std::uint32_t ctx : {8192U, 32768U, kMaxContext}) {
    for (const std::uint32_t rows : {1U, 4U, 8U}) {
      const std::uint32_t first = ctx - rows + 1U;
      const std::size_t row_partial =
          gufo::hip::Turbo4BatchedDecodePartialBytes(first, rows, kHeads,
                                                     kHeadDim);
      const double bytes = 2.0 * static_cast<double>(
                                     gufo::hip::Turbo4LayerPlaneBytes(
                                         kHeadDim, kKvHeads, ctx,
                                         formats.stride));
      const auto launch = [&](int which) {
        ArmBuffers& a = arm[which];
        if (which == 0)
          return gufo::hip::LaunchTurbo4AttentionVerifyWmma(
              device_q, device_k, device_v, a.out, a.q_rot, a.out_rot,
              a.partials, plane_bytes, rows * vector_bytes,
              rows * vector_bytes, rows * vector_bytes, rows * vector_bytes,
              rows * row_partial, device_gate, rows * vector_bytes, kHeads,
              kKvHeads, kHeadDim, first, rows, kMaxContext, scale,
              writer_status, device_status, nullptr, formats);
        return gufo::hip::LaunchTurbo4AttentionVerifyAttn6(
            device_q, device_k, device_v, a.out, a.q_rot, a.out_rot,
            a.partials, plane_bytes, rows * vector_bytes, rows * vector_bytes,
            rows * vector_bytes, rows * vector_bytes, rows * row_partial,
            device_gate, rows * vector_bytes, kHeads, kKvHeads, kHeadDim,
            first, rows, kMaxContext, scale, writer_status, device_status,
            nullptr, formats, which == 1 ? 0U : splits_cap);
      };
      for (int which = 0; which < 3; ++which) {
        // Same fill everywhere a reader does not write, so the bitwise
        // partial compare sees only what the readers wrote.
        Check(hipMemset(arm[which].partials, 0x6D, rows * row_partial));
        Check(hipMemset(device_status, 0, sizeof(*device_status)));
        if (launch(which) != Turbo4LaunchStatus::kLaunched)
          throw std::runtime_error("attention bench launch refused");
        Check(hipDeviceSynchronize());  // warm-up + the numerics sample
        Check(hipEventRecord(begin_ev));
        for (int i = 0; i < iters; ++i)
          launch(which);
        Check(hipEventRecord(end_ev));
        Check(hipEventSynchronize(end_ev));
        float ms = 0.0f;
        Check(hipEventElapsedTime(&ms, begin_ev, end_ev));
        Turbo4KernelStatus st = Turbo4KernelStatus::kWriterFailed;
        Check(hipMemcpy(&st, device_status, sizeof(st),
                        hipMemcpyDeviceToHost));
        if (st != Turbo4KernelStatus::kOk)
          throw std::runtime_error("attention bench status refused");
        const double per_ms = static_cast<double>(ms) / iters;
        std::printf("%s,k6,%u,%u,%.4f,%.1f\n", names[which], ctx, rows,
                    per_ms, bytes / (per_ms * 1e6));
      }

      // Bitwise: attn6 (same plan) vs wmma, partials and out_rot.
      std::vector<std::uint8_t> pa(rows * row_partial), pb(pa.size());
      Check(hipMemcpy(pa.data(), arm[0].partials, pa.size(),
                      hipMemcpyDeviceToHost));
      Check(hipMemcpy(pb.data(), arm[1].partials, pb.size(),
                      hipMemcpyDeviceToHost));
      std::vector<float> o[3];
      for (int which = 0; which < 3; ++which) {
        o[which].resize(rows * vector_floats);
        Check(hipMemcpy(o[which].data(), arm[which].out_rot,
                        rows * vector_bytes, hipMemcpyDeviceToHost));
      }
      const bool bitwise =
          pa == pb && std::memcmp(o[0].data(), o[1].data(),
                                  rows * vector_bytes) == 0;
      std::string line = "numerics ctx=" + std::to_string(ctx) +
                         " rows=" + std::to_string(rows) +
                         " attn6_bitwise_vs_wmma=" + (bitwise ? "yes" : "NO");
      if (rows == kMaxRows) {
        // fp64 reference of out_rot for every row and head.
        std::vector<float> q_rot(rows * vector_floats);
        Check(hipMemcpy(q_rot.data(), arm[0].q_rot, rows * vector_bytes,
                        hipMemcpyDeviceToHost));
        std::vector<double> ref(rows * vector_floats);
        std::vector<double> kd(static_cast<std::size_t>(ctx) * kHeadDim),
            vd(kd.size()), score(ctx);
        for (std::uint32_t kvh = 0; kvh < kKvHeads; ++kvh) {
          for (std::uint32_t t = 0; t < ctx; ++t)
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
              kd[static_cast<std::size_t>(t) * kHeadDim + d] =
                  stored(host_k, t, kvh, d);
              vd[static_cast<std::size_t>(t) * kHeadDim + d] =
                  stored(host_v, t, kvh, d);
            }
          for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t g = 0; g < 6; ++g) {
              const std::size_t qo =
                  r * vector_floats +
                  (static_cast<std::size_t>(kvh) * 6 + g) * kHeadDim;
              const std::uint32_t valid = first + r;
              double mx = -1e300;
              for (std::uint32_t t = 0; t < valid; ++t) {
                double acc = 0.0;
                for (std::uint32_t d = 0; d < kHeadDim; ++d)
                  acc += static_cast<double>(q_rot[qo + d]) *
                         kd[static_cast<std::size_t>(t) * kHeadDim + d];
                score[t] = acc * scale;
                mx = std::max(mx, score[t]);
              }
              double l = 0.0;
              for (std::uint32_t d = 0; d < kHeadDim; ++d)
                ref[qo + d] = 0.0;
              for (std::uint32_t t = 0; t < valid; ++t) {
                const double p = std::exp(score[t] - mx);
                l += p;
                for (std::uint32_t d = 0; d < kHeadDim; ++d)
                  ref[qo + d] +=
                      p * vd[static_cast<std::size_t>(t) * kHeadDim + d];
              }
              for (std::uint32_t d = 0; d < kHeadDim; ++d)
                ref[qo + d] /= l;
            }
        }
        for (int which = 0; which < 3; ++which) {
          double max_abs = 0.0, max_rel = 0.0, max_l2 = 0.0;
          for (std::size_t v = 0; v < rows * kHeads; ++v) {
            double peak = 0.0, d2 = 0.0, r2 = 0.0;
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
              peak = std::max(peak, std::fabs(ref[v * kHeadDim + d]));
            for (std::uint32_t d = 0; d < kHeadDim; ++d) {
              const double e = ref[v * kHeadDim + d];
              const double diff = o[which][v * kHeadDim + d] - e;
              max_abs = std::max(max_abs, std::fabs(diff));
              max_rel = std::max(max_rel, std::fabs(diff) /
                                              std::max(std::fabs(e),
                                                       1e-3 * peak));
              d2 += diff * diff;
              r2 += e * e;
            }
            max_l2 = std::max(max_l2, r2 > 0.0 ? std::sqrt(d2 / r2) : 1.0);
          }
          char buf[160];
          std::snprintf(buf, sizeof(buf),
                        " | %s vs fp64: max_abs=%.3e max_rel=%.3e "
                        "max_l2=%.3e",
                        names[which], max_abs, max_rel, max_l2);
          line += buf;
        }
      }
      numerics.push_back(line);
    }
  }
  for (const auto& line : numerics)
    std::printf("%s\n", line.c_str());
  Check(hipEventDestroy(begin_ev));
  Check(hipEventDestroy(end_ev));
  (void)hipFree(device_k);
  (void)hipFree(device_v);
  (void)hipFree(device_q);
  (void)hipFree(device_gate);
  for (auto& a : arm) {
    (void)hipFree(a.out);
    (void)hipFree(a.q_rot);
    (void)hipFree(a.out_rot);
    (void)hipFree(a.partials);
  }
  (void)hipFree(writer_status);
  (void)hipFree(device_status);
  return 0;
}

// VRD6 arm: `verify_wmma_bench vrd6 [iters=20] [splits_cap=10]`, run with
// GUFO_DEV_VRD6 and GUFO_DEV_ATTN6_V3 unset (so "wmma" is the old pipelined
// reader). Same production-written Turbo6 planes as the attention arm, max
// context 131072. Per ctx in {4096, 16384, 32768, 65536, 131072} and rows
// 1..8 (last row = ctx): times "wmma", "attn6" (cap 0, the served
// reader), "attn6_cap10" and "vrd6_capN"; GBps = compulsory K+V plane bytes /
// per-launch time (rotate + reader + reduce). Numerics: out_rot of every arm
// vs an fp64 host reference over the same stored codes for every (ctx, rows)
// (a rows-r launch reads query rows 8-r..7 so its rows match the reference's
// last r); gate: vrd6 max per-row relative L2 < 2e-3, every status kOk and
// every output finite. Last line: "VRD6 GATE PASS" or "VRD6 GATE FAIL".
int RunVrd6Bench(int iters, std::uint32_t splits_cap) {
  constexpr std::uint32_t kHeadDim = 256;
  constexpr std::uint32_t kKvHeads = 4;
  constexpr std::uint32_t kHeads = 24;
  constexpr std::uint32_t kMaxContext = 131072;
  constexpr std::uint32_t kMaxRows = 8;
  constexpr int kArms = 4;
  namespace t4 = gufo::hip::turbo4;
  using gufo::hip::Turbo4KernelStatus;
  using gufo::hip::Turbo4LaunchStatus;
  const t4::PlaneFormats formats(t4::Bits::k6);
  const std::size_t vector_floats = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t vector_bytes = vector_floats * sizeof(float);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  const std::size_t plane_bytes = gufo::hip::Turbo4LayerPlaneBytes(
      kHeadDim, kKvHeads, kMaxContext, formats.stride);
  const std::size_t single_partial = gufo::hip::Turbo4DecodePartialBytes(
      kMaxContext, kHeads, kKvHeads, kHeadDim);
  if (plane_bytes == 0 || single_partial == 0)
    throw std::runtime_error("vrd6 bench geometry sizing");

  std::vector<std::uint8_t> host_k(plane_bytes), host_v(plane_bytes);
  const gufo::hip::QwenKvPlaneGeometry geometry{kHeadDim, kKvHeads,
                                                kMaxContext};
  std::vector<float> krow(kHeadDim), vrow(kHeadDim);
  for (std::uint32_t pos = 0; pos < kMaxContext; ++pos)
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        const std::size_t seed =
            static_cast<std::size_t>(pos) * kKvHeads * kHeadDim +
            head * kHeadDim + d;
        krow[d] = 3.0f * BenchNoise(seed + 10000);
        vrow[d] = 0.25f + BenchNoise(seed + 40000);
      }
      if (!gufo::hip::QwenKvWriteHead(gufo::hip::QwenKvCacheStorage::kTurbo6,
                                      host_k.data(), plane_bytes, geometry,
                                      pos, head, krow.data(), formats, true) ||
          !gufo::hip::QwenKvWriteHead(gufo::hip::QwenKvCacheStorage::kTurbo6,
                                      host_v.data(), plane_bytes, geometry,
                                      pos, head, vrow.data(), formats, false))
        throw std::runtime_error("vrd6 bench host writer refused");
    }

  std::uint8_t *device_k = nullptr, *device_v = nullptr;
  float *device_q = nullptr, *device_gate = nullptr;
  Turbo4KernelStatus *writer_status = nullptr, *device_status = nullptr;
  const std::size_t rows_bytes = kMaxRows * vector_bytes;
  const std::size_t partial_cap = kMaxRows * single_partial;
  std::vector<float> q(kMaxRows * vector_floats),
      gate(kMaxRows * vector_floats);
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = 4.0f * BenchNoise(i + 88000);
    gate[i] = -0.5f + BenchNoise(i + 99000);
  }
  struct ArmBuffers {
    float *out = nullptr, *q_rot = nullptr, *out_rot = nullptr,
          *partials = nullptr;
  };
  ArmBuffers arm[kArms];
  Check(hipMalloc(&device_k, plane_bytes));
  Check(hipMalloc(&device_v, plane_bytes));
  Check(hipMalloc(&device_q, rows_bytes));
  Check(hipMalloc(&device_gate, rows_bytes));
  for (auto& a : arm) {
    Check(hipMalloc(&a.out, rows_bytes));
    Check(hipMalloc(&a.q_rot, rows_bytes));
    Check(hipMalloc(&a.out_rot, rows_bytes));
    Check(hipMalloc(&a.partials, partial_cap));
  }
  Check(hipMalloc(&writer_status, sizeof(*writer_status)));
  Check(hipMalloc(&device_status, sizeof(*device_status)));
  Check(hipMemcpy(device_k, host_k.data(), plane_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(device_v, host_v.data(), plane_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(device_q, q.data(), rows_bytes, hipMemcpyHostToDevice));
  Check(hipMemcpy(device_gate, gate.data(), rows_bytes, hipMemcpyHostToDevice));
  const Turbo4KernelStatus ok = Turbo4KernelStatus::kOk;
  Check(hipMemcpy(writer_status, &ok, sizeof(ok), hipMemcpyHostToDevice));

  hipEvent_t begin_ev = nullptr, end_ev = nullptr;
  Check(hipEventCreate(&begin_ev));
  Check(hipEventCreate(&end_ev));
  const std::string vrd6_name = "vrd6_cap" + std::to_string(splits_cap);
  const char* const names[kArms] = {"wmma", "attn6", "attn6_cap10",
                                    vrd6_name.c_str()};
  std::printf("reader,codec,ctx,rows,ms,GBps\n");
  std::vector<std::string> numerics;
  bool gate_ok = true;

  // fp64 decode of one stored value (the codes the GPU reads).
  const auto stored = [&](const std::vector<std::uint8_t>& plane,
                          std::uint32_t token, std::uint32_t head,
                          std::uint32_t d) {
    std::size_t base = 0;
    if (!gufo::hip::Turbo4BlockByteOffset(token, head, kHeadDim, kKvHeads,
                                          kMaxContext, &base,
                                          formats.stride))
      throw std::runtime_error("vrd6 bench offset");
    const std::uint8_t* block =
        plane.data() + base + (d / 128U) * t4::BlockBytes(formats.stride);
    std::uint16_t bits = 0;
    std::memcpy(&bits, block, sizeof(bits));
    return static_cast<double>(t4::CentroidValue(
               t4::Bits::k6, t4::CodeAt(t4::Bits::k6, block, d % 128U))) *
           static_cast<double>(t4::HalfBitsToFloat(bits));
  };

  for (const std::uint32_t ctx : {4096U, 16384U, 32768U, 65536U, kMaxContext}) {
    const double bytes = 2.0 * static_cast<double>(
                                   gufo::hip::Turbo4LayerPlaneBytes(
                                       kHeadDim, kKvHeads, ctx,
                                       formats.stride));
    // Outputs of every arm for every rows value, kept for the numerics.
    std::vector<std::vector<float>> outs(kArms * (kMaxRows + 1));
    std::vector<float> q_rot8(kMaxRows * vector_floats);
    for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows) {
      const std::uint32_t first = ctx - rows + 1U;
      const std::size_t row_partial =
          gufo::hip::Turbo4BatchedDecodePartialBytes(first, rows, kHeads,
                                                     kHeadDim);
      // Query rows 8 - rows .. 7, so row j attends over first + j keys with
      // the query the reference pairs with that context.
      const float* const q_rows =
          device_q + static_cast<std::size_t>(kMaxRows - rows) * vector_floats;
      const float* const g_rows =
          device_gate +
          static_cast<std::size_t>(kMaxRows - rows) * vector_floats;
      const auto launch = [&](int which) {
        ArmBuffers& a = arm[which];
        const std::size_t vb = rows * vector_bytes;
        if (which == 0)
          return gufo::hip::LaunchTurbo4AttentionVerifyWmma(
              q_rows, device_k, device_v, a.out, a.q_rot, a.out_rot,
              a.partials, plane_bytes, vb, vb, vb, vb, rows * row_partial,
              g_rows, vb, kHeads, kKvHeads, kHeadDim, first, rows, kMaxContext,
              scale, writer_status, device_status, nullptr, formats);
        if (which == 3)
          return gufo::hip::LaunchTurbo4AttentionVerifyVrd6(
              q_rows, device_k, device_v, a.out, a.q_rot, a.out_rot,
              a.partials, plane_bytes, vb, vb, vb, vb, rows * row_partial,
              g_rows, vb, kHeads, kKvHeads, kHeadDim, first, rows, kMaxContext,
              scale, writer_status, device_status, nullptr, formats,
              splits_cap);
        return gufo::hip::LaunchTurbo4AttentionVerifyAttn6(
            q_rows, device_k, device_v, a.out, a.q_rot, a.out_rot, a.partials,
            plane_bytes, vb, vb, vb, vb, rows * row_partial, g_rows, vb,
            kHeads, kKvHeads, kHeadDim, first, rows, kMaxContext, scale,
            writer_status, device_status, nullptr, formats,
            which == 1 ? 0U : 10U);
      };
      for (int which = 0; which < kArms; ++which) {
        Check(hipMemset(device_status, 0, sizeof(*device_status)));
        if (launch(which) != Turbo4LaunchStatus::kLaunched)
          throw std::runtime_error("vrd6 bench launch refused");
        Check(hipDeviceSynchronize());  // warm-up + the numerics sample
        Turbo4KernelStatus st = Turbo4KernelStatus::kWriterFailed;
        Check(hipMemcpy(&st, device_status, sizeof(st),
                        hipMemcpyDeviceToHost));
        std::vector<float>& o = outs[which * (kMaxRows + 1) + rows];
        o.resize(rows * vector_floats);
        Check(hipMemcpy(o.data(), arm[which].out_rot, rows * vector_bytes,
                        hipMemcpyDeviceToHost));
        if (rows == kMaxRows && which == 0)
          Check(hipMemcpy(q_rot8.data(), arm[0].q_rot, rows * vector_bytes,
                          hipMemcpyDeviceToHost));
        Check(hipEventRecord(begin_ev));
        for (int i = 0; i < iters; ++i)
          launch(which);
        Check(hipEventRecord(end_ev));
        Check(hipEventSynchronize(end_ev));
        float ms = 0.0f;
        Check(hipEventElapsedTime(&ms, begin_ev, end_ev));
        Turbo4KernelStatus st2 = Turbo4KernelStatus::kWriterFailed;
        Check(hipMemcpy(&st2, device_status, sizeof(st2),
                        hipMemcpyDeviceToHost));
        if (st != Turbo4KernelStatus::kOk || st2 != Turbo4KernelStatus::kOk) {
          std::printf("STATUS %s ctx=%u rows=%u st=%u\n", names[which], ctx,
                      rows, static_cast<unsigned>(st2));
          if (which == 3)
            gate_ok = false;
        }
        const double per_ms = static_cast<double>(ms) / iters;
        std::printf("%s,k6,%u,%u,%.4f,%.1f\n", names[which], ctx, rows,
                    per_ms, bytes / (per_ms * 1e6));
      }
    }

    // fp64 reference of out_rot for the 8 contexts ctx - 7 .. ctx: one KV
    // head's values at a time (2 x 268 MB at 128k), its 48 query vectors on
    // four threads.
    std::vector<double> ref(kMaxRows * vector_floats);
    std::vector<double> kd(static_cast<std::size_t>(ctx) * kHeadDim),
        vd(kd.size());
    for (std::uint32_t kvh = 0; kvh < kKvHeads; ++kvh) {
      for (std::uint32_t t = 0; t < ctx; ++t)
        for (std::uint32_t d = 0; d < kHeadDim; ++d) {
          kd[static_cast<std::size_t>(t) * kHeadDim + d] =
              stored(host_k, t, kvh, d);
          vd[static_cast<std::size_t>(t) * kHeadDim + d] =
              stored(host_v, t, kvh, d);
        }
      const auto ref_vectors = [&](std::uint32_t lane) {
        std::vector<double> score(ctx);
        for (std::uint32_t rg = lane; rg < kMaxRows * 6U; rg += 4U) {
          const std::uint32_t r = rg / 6U, g = rg % 6U;
          const std::size_t qo =
              r * vector_floats +
              (static_cast<std::size_t>(kvh) * 6 + g) * kHeadDim;
          const std::uint32_t valid = ctx - kMaxRows + 1U + r;
          double mx = -1e300;
          for (std::uint32_t t = 0; t < valid; ++t) {
            double acc = 0.0;
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
              acc += static_cast<double>(q_rot8[qo + d]) *
                     kd[static_cast<std::size_t>(t) * kHeadDim + d];
            score[t] = acc * scale;
            mx = std::max(mx, score[t]);
          }
          double l = 0.0;
          for (std::uint32_t d = 0; d < kHeadDim; ++d)
            ref[qo + d] = 0.0;
          for (std::uint32_t t = 0; t < valid; ++t) {
            const double p = std::exp(score[t] - mx);
            l += p;
            for (std::uint32_t d = 0; d < kHeadDim; ++d)
              ref[qo + d] +=
                  p * vd[static_cast<std::size_t>(t) * kHeadDim + d];
          }
          for (std::uint32_t d = 0; d < kHeadDim; ++d)
            ref[qo + d] /= l;
        }
      };
      std::vector<std::thread> pool;
      for (std::uint32_t lane = 0; lane < 4U; ++lane)
        pool.emplace_back(ref_vectors, lane);
      for (auto& th : pool)
        th.join();
    }
    for (std::uint32_t rows = 1; rows <= kMaxRows; ++rows) {
      std::string line = "numerics ctx=" + std::to_string(ctx) +
                         " rows=" + std::to_string(rows);
      for (int which = 0; which < kArms; ++which) {
        const std::vector<float>& o = outs[which * (kMaxRows + 1) + rows];
        double max_abs = 0.0, max_rel = 0.0, max_l2 = 0.0;
        bool finite = true;
        for (std::size_t v = 0; v < rows * kHeads; ++v) {
          // Launch row v / kHeads is reference row 8 - rows + v / kHeads.
          const double* e = ref.data() +
                            (kMaxRows - rows) * vector_floats + v * kHeadDim;
          double peak = 0.0, d2 = 0.0, r2 = 0.0;
          for (std::uint32_t d = 0; d < kHeadDim; ++d)
            peak = std::max(peak, std::fabs(e[d]));
          for (std::uint32_t d = 0; d < kHeadDim; ++d) {
            const float a = o[v * kHeadDim + d];
            if (!std::isfinite(a)) {
              finite = false;
              continue;
            }
            const double diff = a - e[d];
            max_abs = std::max(max_abs, std::fabs(diff));
            max_rel = std::max(max_rel, std::fabs(diff) /
                                            std::max(std::fabs(e[d]),
                                                     1e-3 * peak));
            d2 += diff * diff;
            r2 += e[d] * e[d];
          }
          max_l2 = std::max(max_l2, r2 > 0.0 ? std::sqrt(d2 / r2) : 1.0);
        }
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      " | %s vs fp64: max_abs=%.3e max_rel=%.3e max_l2=%.3e%s",
                      names[which], max_abs, max_rel, max_l2,
                      finite ? "" : " NONFINITE");
        line += buf;
        if (which == 3 && (!finite || !(max_l2 < 2e-3)))
          gate_ok = false;
      }
      numerics.push_back(line);
    }
  }
  for (const auto& line : numerics)
    std::printf("%s\n", line.c_str());
  std::printf("VRD6 GATE %s (vrd6 max per-row L2 vs fp64 < 2e-3, status kOk, "
              "finite)\n",
              gate_ok ? "PASS" : "FAIL");
  Check(hipEventDestroy(begin_ev));
  Check(hipEventDestroy(end_ev));
  (void)hipFree(device_k);
  (void)hipFree(device_v);
  (void)hipFree(device_q);
  (void)hipFree(device_gate);
  for (auto& a : arm) {
    (void)hipFree(a.out);
    (void)hipFree(a.q_rot);
    (void)hipFree(a.out_rot);
    (void)hipFree(a.partials);
  }
  (void)hipFree(writer_status);
  (void)hipFree(device_status);
  return gate_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "attention")
    return RunAttentionReaderBench(
        argc > 2 ? std::stoi(argv[2]) : 20,
        argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 10U);
  if (argc >= 2 && std::string(argv[1]) == "vrd6")
    return RunVrd6Bench(
        argc > 2 ? std::stoi(argv[2]) : 20,
        argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 10U);
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s type m k raw batch [iters]\n", argv[0]);
    return 2;
  }
  const auto type = static_cast<gufo::core::GgmlType>(std::stoi(argv[1]));
  const std::size_t m = std::stoull(argv[2]), k = std::stoull(argv[3]);
  const std::size_t batch = std::stoull(argv[5]);
  const int iters = argc > 6 ? std::stoi(argv[6]) : 24;
  std::ifstream f(argv[4], std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open raw file");
  const std::size_t bytes = static_cast<std::size_t>(f.tellg());
  if (bytes % m != 0) throw std::runtime_error("raw size is not m rows");
  std::vector<char> host(bytes);
  f.seekg(0);
  f.read(host.data(), static_cast<std::streamsize>(bytes));

  const std::size_t copies = std::max<std::size_t>(1, (256u << 20) / bytes + 1);
  char* w = nullptr;
  Check(hipMalloc(&w, bytes * copies));
  for (std::size_t c = 0; c < copies; ++c)
    Check(hipMemcpy(w + c * bytes, host.data(), bytes, hipMemcpyHostToDevice));

  std::vector<float> hx(batch * k);
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  for (float& v : hx) v = nd(rng);
  float *x = nullptr, *ya = nullptr, *yb = nullptr;
  void* q8 = nullptr;
  Check(hipMalloc(&x, hx.size() * sizeof(float)));
  Check(hipMalloc(&ya, batch * m * sizeof(float)));
  Check(hipMalloc(&yb, batch * m * sizeof(float)));
  Check(hipMalloc(&q8, gufo::hip::QuantizedActivationBytes(batch, k)));
  Check(hipMemcpy(x, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice));

  const auto verify = [&](const char* wp, float* y) {
    gufo::hip::LaunchBatchedQuantGEMMFp32(type, wp, x, y, batch, m, k, nullptr);
  };
  const auto wmma = [&](const char* wp, float* y) {
    gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(x, q8, batch, k, nullptr);
    gufo::hip::LaunchBatchedQuantGEMMPreQuantized(type, wp, q8, y, batch, m, k,
                                                  nullptr);
  };
  verify(w, ya);
  wmma(w, yb);
  Check(hipDeviceSynchronize());
  std::vector<float> a(batch * m), b(batch * m);
  Check(hipMemcpy(a.data(), ya, a.size() * sizeof(float), hipMemcpyDeviceToHost));
  Check(hipMemcpy(b.data(), yb, b.size() * sizeof(float), hipMemcpyDeviceToHost));
  float diff = 0.0F, ref = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]);
    diff = std::isfinite(d) ? std::max(diff, d) : INFINITY;
    ref = std::max(ref, std::fabs(a[i]));
  }

  hipEvent_t start, stop;
  Check(hipEventCreate(&start));
  Check(hipEventCreate(&stop));
  std::printf("route,type,m,k,batch,round,ms,weight_GBps,rel_diff\n");
  for (int round = 0; round < 4; ++round) {
    for (int r = 0; r < 2; ++r) {
      const bool use_wmma = (r ^ (round & 1)) != 0;  // alternate order
      Check(hipEventRecord(start));
      for (int i = 0; i < iters; ++i) {
        const char* wp = w + (static_cast<std::size_t>(i) % copies) * bytes;
        use_wmma ? wmma(wp, yb) : verify(wp, ya);
      }
      Check(hipEventRecord(stop));
      Check(hipEventSynchronize(stop));
      float ms = 0.0F;
      Check(hipEventElapsedTime(&ms, start, stop));
      ms /= static_cast<float>(iters);
      std::printf("%s,%d,%zu,%zu,%zu,%d,%.4f,%.1f,%.3g\n",
                  use_wmma ? "wmma_a8" : "verify", static_cast<int>(type), m,
                  k, batch, round, ms, bytes / (ms * 1e6), ref > 0 ? diff / ref : diff);
    }
  }
  return 0;
}

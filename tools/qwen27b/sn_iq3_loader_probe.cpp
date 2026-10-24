// Model-only IQ3-only load timing, complete selected-byte oracle,
// P1 timing control, one-copy/RSS budget. No executor or compute kernel.
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"
#include "src/models/qwen/sn_weight_plan.hpp"
namespace {
void Check(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t rc) {
  Check(rc == hipSuccess, hipGetErrorString(rc));
}
std::size_t ProcBytes(const char* path, const char* key) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line))
    if (line.starts_with(std::string(key) + ":")) {
      std::istringstream s(line.substr(line.find(':') + 1));
      std::size_t kb;
      Check(bool(s >> kb), "proc parse");
      return kb * 1024;
    }
  throw std::runtime_error("proc field absent");
}
void Drop(const gufo::core::GgufReader& reader, const void* data,
          std::size_t bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto page = std::size_t(sysconf(_SC_PAGESIZE));
  for (const auto& r : reader.GetMappedRegions()) {
    const auto base = reinterpret_cast<std::uintptr_t>(r.data);
    if (address < base || address - base >= r.size)
      continue;
    Check(bytes <= r.size - (address - base), "drop bounds");
    const auto prefix = address % page;
    Check(madvise(reinterpret_cast<void*>(address - prefix), bytes + prefix,
                  MADV_DONTNEED) == 0,
          "madvise");
    Check(posix_fadvise(r.file_descriptor, address - base, bytes,
                        POSIX_FADV_DONTNEED) == 0,
          "fadvise");
    return;
  }
  throw std::runtime_error("source span missing");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    using namespace gufo::models;
    using gufo::core::GgmlType;
    constexpr auto capacity = 64ULL << 20;
    constexpr auto budget = 24ULL << 30;
    Check(argc == 3, "usage: gufo_sn_iq3_loader_probe MODEL native|p1|iq3");
    const std::string mode = argv[2];
    Check(mode == "native" || mode == "p1" || mode == "iq3", "invalid mode");
    Check(std::getenv("GUFO_DEV_WEIGHTS_GTT") &&
              std::string(std::getenv("GUFO_DEV_WEIGHTS_GTT")) == "1",
          "served GTT env required");
    Check(!std::getenv("GUFO_DEV_EMBED_SSD"), "embedding SSD unsupported");
    Check(setenv("GUFO_SN_K4",
                 mode == "iq3"  ? "iq3"
                 : mode == "p1" ? "1"
                                : "0",
                 1) == 0,
          "setenv");
    Check(ProcBytes("/proc/meminfo", "MemAvailable") >= (1ULL << 29),
          "RAM emergency <0.5GiB");
    std::string error;
    auto unique = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Check(bool(unique), error.c_str());
    std::shared_ptr<const gufo::core::GgufReader> reader(std::move(unique));
    std::size_t source_bytes = 0;
    for (const auto& r : reader->GetMappedRegions())
      source_bytes += r.size;
    Check(
        source_bytes + 3 * capacity + ProcBytes("/proc/self/status", "VmRSS") <=
            budget,
        "preflight >24GiB");
    std::size_t free0, total, free1;
    Hip(hipMemGetInfo(&free0, &total));
    const auto t0 = std::chrono::steady_clock::now();
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    const auto t1 = std::chrono::steady_clock::now();
    Check(bool(model), error.c_str());
    Hip(hipDeviceSynchronize());
    Hip(hipMemGetInfo(&free1, &total));
    const auto rss = ProcBytes("/proc/self/status", "VmRSS");
    const auto load_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    rusage usage{};
    Check(getrusage(RUSAGE_SELF, &usage) == 0, "rusage");
    std::size_t tags = 0, s = 0, x = 0, compared = 0;
    const auto& weights = model->GetWeights();
    Check(weights.layers.size() == 64, "expected 64 main layers");
    std::vector<std::uint8_t> expected(capacity), actual(capacity);
    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
      const auto& layer = weights.layers[l];
      if (mode == "iq3") {
        Check(IsSnIq3VerifyPair(layer.ffn_gate, layer.ffn_up),
              "IQ3 pair tag/view");
        Check(layer.ssm_out.layout == QwenWeightLayout::kNative &&
                  layer.attn_output.layout == QwenWeightLayout::kNative &&
                  layer.ffn_down.layout == QwenWeightLayout::kNative,
              "IQ3 whitelist escape");
        tags += 2;
      } else {
        Check(layer.ffn_gate.layout == QwenWeightLayout::kNative &&
                  layer.ffn_up.layout == QwenWeightLayout::kNative,
              "baseline IQ3 tagged");
        tags += layer.ssm_out.layout != QwenWeightLayout::kNative;
        tags += layer.attn_output.layout != QwenWeightLayout::kNative;
      }
      if (mode != "iq3")
        continue;
      const auto* g =
          reader->FindTensor("blk." + std::to_string(l) + ".ffn_gate.weight");
      const auto* u =
          reader->FindTensor("blk." + std::to_string(l) + ".ffn_up.weight");
      Check(g && u && g->type == layer.ffn_gate.type && u->type == g->type &&
                g->dimensions == std::vector<std::uint64_t>({5120, 17408}) &&
                u->dimensions == g->dimensions &&
                g->size_bytes == u->size_bytes &&
                static_cast<const std::uint8_t*>(g->data) + g->size_bytes ==
                    u->data,
            "source pair descriptor");
      const auto bytes = 2 * g->size_bytes;
      const auto layout = layer.ffn_gate.layout;
      Check(bytes == Iq3PackedBytes(layout, 34816, 5120), "source size");
      const SnWeightSpan span{0, 34816, 5120, Iq3BlockBytes(layout), layout};
      const auto jobs = CoalesceIq3CopyJobs(
          PlanSnWeightCopies(bytes, std::span(&span, 1)), capacity);
      for (const auto& job : jobs) {
        // Independent per-byte CPU oracle remains unchanged. It packs only
        // this local tile; DMA descriptors relocate XXS into full-pair planes.
        const auto rows = job.row_end - job.row_begin;
        CopyIq3RowRange(
            static_cast<const std::uint8_t*>(g->data) + job.source_offset,
            expected.data(), layout, rows, 5120, 0, rows);
        const auto nb = 20ULL;
        const SnWeightStagedCopy parts[3] = {
            {0,
             layout == QwenWeightLayout::kIq3SRows8 ? job.source_offset
                                                    : job.row_begin * nb * 64,
             layout == QwenWeightLayout::kIq3SRows8 ? job.source_bytes
                                                    : rows * nb * 64},
            {rows * nb * 64, 34816 * nb * 64 + job.row_begin * nb * 32,
             layout == QwenWeightLayout::kIq3SRows8 ? 0 : rows * nb * 32},
            {rows * nb * 96, 34816 * nb * 96 + job.row_begin * nb * 2,
             layout == QwenWeightLayout::kIq3SRows8 ? 0 : rows * nb * 2}};
        for (const auto& p : parts)
          if (p.bytes) {
            Hip(hipMemcpy(
                actual.data(),
                static_cast<const std::uint8_t*>(layer.ffn_gate.data) +
                    p.destination_offset,
                p.bytes, hipMemcpyDeviceToHost));
            Check(std::memcmp(actual.data(), expected.data() + p.stage_offset,
                              p.bytes) == 0,
                  "IQ3 selected byte differs");
            compared += p.bytes;
          }
      }
      if (g->type == GgmlType::kIQ3_S)
        ++s;
      else
        ++x;
      Drop(*reader, g->data, bytes);
    }
    Check(tags == (mode == "iq3"  ? 128U
                   : mode == "p1" ? 63U
                                  : 0U),
          "tag count");
    Check(mode != "iq3" || (s == 32 && x == 32 && compared == 4634705920ULL),
          "IQ3 coverage");
    Check(weights.token_embd.layout == QwenWeightLayout::kNative &&
              weights.output.layout == QwenWeightLayout::kNative,
          "global tags");
    Check(getrusage(RUSAGE_SELF, &usage) == 0, "rusage");
    const auto peak = std::size_t(usage.ru_maxrss) * 1024;
    const auto device_delta = free0 > free1 ? free0 - free1 : 0;
    const auto all_in =
        std::max(model->GetResidentBytes(), device_delta) + peak;
    Check(all_in <= budget, "model-only all-in proxy >24GiB");
    std::printf(
        "{\"mode\":\"%s\",\"load_ms\":%.9g,\"tags\":%zu,\"s_pairs\":%zu,\"xxs_"
        "pairs\":%zu,\"compared_bytes\":%zu,\"after_load_rss\":%zu,\"peak_"
        "rss\":%zu,\"device_delta\":%zu,\"all_in_proxy\":%zu,\"resident_"
        "bytes\":%zu}\n",
        mode.c_str(), load_ms, tags, s, x, compared, rss, peak, device_delta,
        all_in, model->GetResidentBytes());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}

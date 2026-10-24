// Model-only GTT loader validation. No executor or model kernel launch.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {
using gufo::models::QwenTensorRef;
using gufo::models::QwenWeightLayout;
constexpr std::size_t kChunk = 64ULL << 20;   // Existing loader staging bound.
constexpr std::size_t kBudget = 24ULL << 30;  // Reviewer RAM gate.
void Check(bool ok, const char* what) {
  if (!ok)
    throw std::runtime_error(what);
}
void Hip(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
std::size_t ReadKB(const char* path, const char* key) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    const std::string prefix = std::string(key) + ':';
    if (line.starts_with(prefix)) {
      std::istringstream s(line.substr(prefix.size()));
      std::size_t kb = 0;
      Check(bool(s >> kb), "invalid proc memory value");
      return kb * 1024;
    }
  }
  throw std::runtime_error("missing proc memory value");
}
std::size_t PeakRSS() {
  rusage r{};
  Check(getrusage(RUSAGE_SELF, &r) == 0, "getrusage failed");
  return static_cast<std::size_t>(r.ru_maxrss) * 1024;
}
void MemorySnapshot(const char* phase) {
  std::fprintf(stderr, "MEMORY_PHASE %s peak_rss=%zu\n", phase, PeakRSS());
  for (const char* path : {"/proc/self/smaps_rollup", "/proc/self/smaps"}) {
    std::ifstream f(path);
    std::string line, mapping;
    while (std::getline(f, line)) {
      if (line.find('-') < line.find(' '))
        mapping = line;
      if (std::string_view(path).ends_with("smaps_rollup"))
        std::fprintf(stderr, "%s\n", line.c_str());
      else if (line.starts_with("Rss:"))
        std::fprintf(stderr, "MAPPING %s %s\n", mapping.c_str(), line.c_str());
    }
  }
}
struct Candidate {
  const gufo::core::GgufTensorInfo* info;
  const QwenTensorRef* ref;
  std::size_t region;
  std::size_t offset;
};
void DropSource(const gufo::core::GgufReader& reader, const void* data,
                std::size_t bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const long page = sysconf(_SC_PAGESIZE);
  Check(page > 0, "invalid page size");
  for (const auto& r : reader.GetMappedRegions()) {
    const auto base = reinterpret_cast<std::uintptr_t>(r.data);
    if (address < base || address - base >= r.size)
      continue;
    const auto offset = address - base;
    Check(bytes <= r.size - offset && r.file_descriptor >= 0,
          "source range is not file-backed");
    const auto prefix = address % static_cast<std::size_t>(page);
    Check(madvise(reinterpret_cast<void*>(address - prefix), bytes + prefix,
                  MADV_DONTNEED) == 0,
          "source madvise failed");
    Check(posix_fadvise(r.file_descriptor, static_cast<off_t>(offset),
                        static_cast<off_t>(bytes), POSIX_FADV_DONTNEED) == 0,
          "source fadvise failed");
    return;
  }
  throw std::runtime_error("source pointer not in reader region");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Check(argc == 3, "usage: gufo_sn_loader_probe MODEL native|sn");
    const std::string mode = argv[2];
    Check(mode == "native" || mode == "sn", "invalid mode");
    const bool sn = mode == "sn";
    Check(std::getenv("GUFO_DEV_WEIGHTS_GTT") &&
              std::string(std::getenv("GUFO_DEV_WEIGHTS_GTT")) == "1",
          "probe requires served GTT env");
    Check(setenv("GUFO_SN_K4", sn ? "1" : "0", 1) == 0, "setenv failed");
    Check(std::getenv("GUFO_DEV_EMBED_SSD") == nullptr,
          "probe does not support skipped embedding spans");
    Check(ReadKB("/proc/meminfo", "MemAvailable") >= (1ULL << 29),
          "RAM emergency: less than0.5GiB available before load");
    std::string error;
    auto unique_reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    Check(bool(unique_reader), error.c_str());
    std::shared_ptr<const gufo::core::GgufReader> reader(
        std::move(unique_reader));
    std::size_t file_bytes = 0;
    for (const auto& r : reader->GetMappedRegions())
      file_bytes += r.size;
    std::size_t free_before = 0, total = 0;
    Hip(hipMemGetInfo(&free_before, &total));
    // Conservative: staging and readback need not coexist.
    const auto preflight =
        file_bytes + 3 * kChunk + ReadKB("/proc/self/status", "VmRSS");
    Check(preflight <= kBudget,
          "RAM gate: weights/staging/readback/current RSS exceed24GiB");
    MemorySnapshot("before_model");
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Check(bool(model), error.c_str());
    MemorySnapshot("after_model");
    std::size_t free_after = 0;
    Hip(hipMemGetInfo(&free_after, &total));
    const auto device_delta =
        free_before > free_after ? free_before - free_after : 0;
    const auto regions = model->GetWeightRegions();
    const auto& w = model->GetWeights();
    Check(w.token_embd.layout == QwenWeightLayout::kNative &&
              w.output.layout == QwenWeightLayout::kNative &&
              w.output_norm.layout == QwenWeightLayout::kNative,
          "global tensor tagged");
    std::vector<Candidate> selected;
    std::size_t q5 = 0, q6 = 0, tags = 0;
    for (std::size_t l = 0; l < w.layers.size(); ++l) {
      const auto& x = w.layers[l];
      const QwenTensorRef* refs[] = {
          &x.attn_norm, &x.ffn_norm,    &x.attn_q,      &x.attn_k,
          &x.attn_v,    &x.attn_output, &x.attn_q_norm, &x.attn_k_norm,
          &x.attn_qkv,  &x.attn_gate,   &x.ssm_a,       &x.ssm_conv1d,
          &x.ssm_dt,    &x.ssm_alpha,   &x.ssm_beta,    &x.ssm_norm,
          &x.ssm_out,   &x.ffn_gate,    &x.ffn_up,      &x.ffn_down};
      for (const auto* ref : refs) {
        if (ref->layout != QwenWeightLayout::kNative) {
          ++tags;
          Check(sn && (ref == &x.ssm_out || ref == &x.attn_output),
                "unexpected non-native tensor");
        }
      }
      for (const auto& role :
           {std::string("ssm_out"), std::string("attn_output")}) {
        const auto& ref = role == "ssm_out" ? x.ssm_out : x.attn_output;
        const auto type = role == "ssm_out" ? gufo::core::GgmlType::kQ5_K
                                            : gufo::core::GgmlType::kQ6_K;
        if (ref.empty() || ref.type != type)
          continue;
        const auto* info = reader->FindTensor("blk." + std::to_string(l) + '.' +
                                              role + ".weight");
        Check(info &&
                  info->dimensions == std::vector<std::uint64_t>({6144, 5120}),
              "unexpected selected descriptor");
        Check(ref.layout ==
                  (sn ? QwenWeightLayout::kRows32 : QwenWeightLayout::kNative),
              "wrong selected tag");
        Check(ref.num_elements == 5120ULL * 6144,
              "wrong selected element count");
        Check(!sn || ref.layout_columns == 6144, "wrong tagged columns");
        bool found = false;
        const auto address = reinterpret_cast<std::uintptr_t>(ref.data);
        for (std::size_t i = 0; i < regions.size(); ++i) {
          const auto base =
              reinterpret_cast<std::uintptr_t>(regions[i].device_data);
          if (address < base || address - base >= regions[i].size)
            continue;
          const auto offset = address - base;
          Check(info->size_bytes <= regions[i].size - offset,
                "tensor escapes GPU region");
          Check(
              static_cast<const std::uint8_t*>(regions[i].host_data) + offset ==
                  info->data,
              "tensor/source remap offset differs");
          selected.push_back({info, &ref, i, offset});
          found = true;
          break;
        }
        Check(found, "selected tensor missing region");
        if (type == gufo::core::GgmlType::kQ5_K)
          ++q5;
        else
          ++q6;
      }
    }
    Check(q5 == 47 && q6 == 16 && tags == (sn ? 63U : 0U),
          "wrong tensor/tag counts");
    std::vector<std::uint8_t> buffer(kChunk);
    std::size_t selected_bytes = 0, unselected_bytes = 0;
    for (const auto& t : selected) {
      const auto bytes = t.info->size_bytes;
      const std::size_t bb =
          t.ref->type == gufo::core::GgmlType::kQ5_K ? 176 : 210;
      Check(t.info->type == t.ref->type && bytes == 5120 * 24 * bb,
            "selected descriptor type or size differs");
      Check(bytes <= buffer.size(), "tensor exceeds readback bound");
      Hip(hipMemcpy(buffer.data(), t.ref->data, bytes, hipMemcpyDeviceToHost));
      const auto* original = static_cast<const std::uint8_t*>(t.info->data);
      if (!sn) {
        Check(std::memcmp(buffer.data(), original, bytes) == 0,
              "native tensor differs");
      } else {
        const std::size_t payload = bb == 210 ? 208 : 176;
        for (std::size_t row = 0; row < 5120; ++row)
          for (std::size_t b = 0; b < 24; ++b) {
            const auto* block = original + (row * 24 + b) * bb;
            for (std::size_t o = 0; o < payload; o += 16)
              Check(std::memcmp(buffer.data() + gufo::models::Rows32ByteOffset(
                                                    5120, 24, row, b, o, bb),
                                block + o, 16) == 0,
                    "Rows32 payload byte mismatch");
            if (bb == 210)
              Check(std::memcmp(buffer.data() + gufo::models::Rows32ByteOffset(
                                                    5120, 24, row, b, 208, bb),
                                block + 208, 2) == 0,
                    "Rows32 d-plane byte mismatch");
          }
      }
      selected_bytes += bytes;
      DropSource(*reader, original, bytes);
    }
    MemorySnapshot("after_selected");
    std::vector<std::string> hashes;
    for (std::size_t i = 0; i < regions.size(); ++i) {
      const auto& region = regions[i];
      std::vector<Candidate> mask;
      for (const auto& t : selected)
        if (t.region == i)
          mask.push_back(t);
      std::sort(mask.begin(), mask.end(), [](const auto& a, const auto& b) {
        return a.offset < b.offset;
      });
      gufo::crypto::Sha256Hasher actual, expected;
      for (std::size_t at = 0; at < region.size; at += kChunk) {
        const auto bytes = std::min(kChunk, region.size - at);
        Hip(hipMemcpy(buffer.data(),
                      static_cast<const std::uint8_t*>(region.device_data) + at,
                      bytes, hipMemcpyDeviceToHost));
        const auto* original =
            static_cast<const std::uint8_t*>(region.host_data) + at;
        std::size_t cursor = 0;
        const auto gap = [&](std::size_t end) {
          if (end <= cursor)
            return;
          const auto n = end - cursor;
          Check(std::memcmp(buffer.data() + cursor, original + cursor, n) == 0,
                "non-selected region byte mismatch");
          actual.Update({buffer.data() + cursor, n});
          expected.Update({original + cursor, n});
          unselected_bytes += n;
          cursor = end;
        };
        for (const auto& m : mask) {
          const auto end = m.offset + m.info->size_bytes;
          if (end <= at || m.offset >= at + bytes)
            continue;
          gap(m.offset > at ? m.offset - at : 0);
          cursor = std::max(cursor, std::min(bytes, end - at));
        }
        gap(bytes);
        DropSource(*reader, original, bytes);
      }
      const auto digest = actual.FinishHex();
      Check(digest == expected.FinishHex(), "masked region hash mismatch");
      hashes.push_back(digest);
    }
    MemorySnapshot("after_nonselected");
    Check(selected_bytes + unselected_bytes == file_bytes,
          "validation byte coverage differs from mapped span");
    const auto rss = PeakRSS();
    const auto all_in = std::max(model->GetResidentBytes(), device_delta) + rss;
    Check(all_in <= kBudget,
          "measured weights/device-delta plus peak RSS exceeds24GiB");
    std::printf(
        "{\"ok\":true,\"mode\":\"%s\",\"q5\":%zu,\"q6\":%zu,\"tags\":%zu,"
        "\"selected_bytes\":%zu,\"nonselected_bytes\":%zu,\"resident_bytes\":%"
        "zu,\"device_delta_bytes\":%zu,\"peak_rss_bytes\":%zu,\"all_in_proxy_"
        "bytes\":%zu,\"masked_region_sha256\":[",
        mode.c_str(), q5, q6, tags, selected_bytes, unselected_bytes,
        model->GetResidentBytes(), device_delta, rss, all_in);
    for (std::size_t i = 0; i < hashes.size(); ++i)
      std::printf("%s\"%s\"", i ? "," : "", hashes[i].c_str());
    std::puts("]}");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}

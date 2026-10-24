// CPU only: the NPU B-cache key for tagged IQ3 gate/up.
// (1) Served geometry (64 layers, GPU rows gate/up 9984, down 3584, store 35%):
//     every packed slice of one prefill cycle gets its own key; the old
//     (pointer, row0) key collapses all tagged gate/up onto the one scratch.
// (2) LogicalRowFnv over the tagged root == FNV-1a of the first 256 bytes of
//     the same native row in the CPU unpack oracle (what the scratch holds and
//     what a native model's RowFingerprint hashes), both layouts, gate and up.
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "src/models/qwen/hip/npu_bcache_key.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"

using namespace gufo::models;
using gufo::hip::npu::KeyOf;
using gufo::hip::npu::LogicalRowFnv;
using gufo::hip::npu::LogicalWeight;
using gufo::hip::npu::WeightKey;

namespace {
int g_failures = 0;
void Expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  g_failures += !ok;
}
const void* Ptr(std::uintptr_t p) { return reinterpret_cast<const void*>(p); }

std::uint64_t Fnv(const std::uint8_t* p, std::size_t n) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
  return h;
}

void ServedKeys() {
  constexpr std::size_t kFfn = 17408, kRows = 2 * kFfn, kCols = 5120;
  constexpr std::uint32_t kGuRow0 = kFfn - 7424, kDownRow0 = 5120 - 1536;
  const void* scratch = Ptr(0x7000'0000'0000);  // one arena scratch for every layer
  const void* scratch_up = Ptr(0x7000'0000'0000 + kFfn * (kCols / 256) * 110);
  std::set<WeightKey> keys, old_keys;
  std::size_t slices = 0, jobs = 0;
  bool rows_inside = true;
  const auto add = [&](const WeightKey& k, const void* w, std::uint32_t row0) {
    keys.insert(k);
    old_keys.insert(KeyOf(w, row0, nullptr));
    ++slices;
  };
  for (std::size_t l = 0; l < 64; ++l) {
    const std::uintptr_t base = 0x1000'0000 + l * 0x1000'0000;  // distinct tensors per layer
    const auto layout = (l < 16 || l >= 48) ? QwenWeightLayout::kIq3SRows8 : QwenWeightLayout::kIq3XxsRows4;
    const LogicalWeight gate{Ptr(base), layout, kRows, kCols, 0};
    const LogicalWeight up{Ptr(base), layout, kRows, kCols, kFfn};
    if (l % 4 == 3) {  // full attention: attn_q m 12288 -> 4096 NPU rows
      add(KeyOf(Ptr(base + 0x100), 12288 - 4096, nullptr), Ptr(base + 0x100), 12288 - 4096);
      ++jobs;
    } else {  // GDN: qkv m 10240 -> 3584, gate m 6144 -> 2048
      add(KeyOf(Ptr(base + 0x200), 10240 - 3584, nullptr), Ptr(base + 0x200), 10240 - 3584);
      add(KeyOf(Ptr(base + 0x300), 6144 - 2048, nullptr), Ptr(base + 0x300), 6144 - 2048);
      jobs += 2;
    }
    add(KeyOf(scratch, kGuRow0, &gate), scratch, kGuRow0);
    add(KeyOf(scratch_up, kGuRow0, &up), scratch_up, kGuRow0);
    add(KeyOf(Ptr(base + 0x400), kDownRow0, nullptr), Ptr(base + 0x400), kDownRow0);
    jobs += 2;
    rows_inside &= gate.row_begin + kGuRow0 < kRows && up.row_begin + kGuRow0 < kRows;
  }
  std::printf("  %zu jobs, %zu slices, %zu unique keys, old key %zu unique\n", jobs, slices,
              keys.size(), old_keys.size());
  Expect(rows_inside, "served gate/up rows inside the pair");
  Expect(jobs == 240, "240 split jobs per cycle (as recorded on the served model)");
  Expect(keys.size() == slices, "every slice of the cycle has its own key");
  Expect(old_keys.size() == slices - 2 * 63, "control: old key folds 128 gate/up slices onto 2");
  // gate vs up of one layer, and a native tensor at the same row, never collide
  const LogicalWeight g{Ptr(0x5000), QwenWeightLayout::kIq3SRows8, kRows, kCols, 0};
  const LogicalWeight u{Ptr(0x5000), QwenWeightLayout::kIq3SRows8, kRows, kCols, kFfn};
  Expect(!(KeyOf(scratch, 0, &g) == KeyOf(scratch, 0, &u)), "gate row 0 != up row 0");
  Expect(!(KeyOf(scratch, kFfn, &g) == KeyOf(Ptr(0x5000), kFfn, nullptr)),
         "tagged view != native key on the same pointer/row");
}

void FingerprintMatchesUnpack() {
  std::mt19937_64 rng(105);
  for (const auto type : {gufo::core::GgmlType::kIQ3_S, gufo::core::GgmlType::kIQ3_XXS}) {
    constexpr std::size_t kRows = 32, kCols = 1024;  // 4 blocks: 440 / 392 native bytes per row
    const auto layout = type == gufo::core::GgmlType::kIQ3_S ? QwenWeightLayout::kIq3SRows8
                                                               : QwenWeightLayout::kIq3XxsRows4;
    const std::size_t bytes = Iq3PackedBytes(layout, kRows, kCols);
    const std::size_t row_bytes = kCols / 256 * Iq3BlockBytes(layout);
    std::set<std::uint64_t> fps;
    bool all_equal = true;
    for (int root_i = 0; root_i < 2; ++root_i) {  // two "layers"
      std::vector<std::uint8_t> root(bytes);
      for (auto& b : root) b = std::uint8_t(rng());
      const auto views = MakeIq3PairViews(root.data(), type, kRows, kCols, bytes);
      for (const auto* view : {&views.gate, &views.up}) {
        std::vector<std::uint8_t> native(view->EncodedSizeBytes());
        UnpackIq3View(*view, native.data(), native.size());
        const LogicalWeight src{root.data(), layout, kRows, kCols, view->layout_row_begin};
        for (std::size_t r = 0; r < kRows / 2; ++r) {
          const std::uint64_t got =
              LogicalRowFnv(src, r, [&](std::size_t off) { return root[off]; });
          all_equal &= got == Fnv(native.data() + r * row_bytes, 256);
          fps.insert(got);
        }
      }
    }
    Expect(all_equal, type == gufo::core::GgmlType::kIQ3_S ? "IQ3_S logical fp == unpacked-row fp"
                                                           : "IQ3_XXS logical fp == unpacked-row fp");
    Expect(fps.size() == 2 * kRows, "fps distinct across rows, gate/up, layers");
  }
}
}  // namespace

int main() {
  try {
    ServedKeys();
    FingerprintMatchesUnpack();
  } catch (const std::exception& e) {
    std::printf("FAIL exception %s\n", e.what());
    return 1;
  }
  std::printf("%s  %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
  return g_failures ? 1 : 0;
}

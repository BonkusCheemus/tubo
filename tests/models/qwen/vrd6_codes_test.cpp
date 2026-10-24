// VRD6 CPU check: the vrd6 verify reader's pair table (Vrd6PairOffsets +
// Vrd6LutEntry over the kernel's c16 table, the same functions the kernel
// calls) decodes every value of random Turbo6 blocks to
// f16(CentroidValue(CodeAt(...))), and Vrd6KeyColumn is the bijection the
// P / V^T / norm tiles assume.
//
//   g++ -std=c++20 -I. tests/models/qwen/vrd6_codes_test.cpp -o /tmp/vrd6
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

#include "src/models/qwen/hip/turbo4_kernels.hpp"

namespace t4 = gufo::hip::turbo4;

int main() {
  int failures = 0;
  // The kernel's 64-entry f16 centroid table (Turbo4VerifyVrd6Kernel).
  std::uint16_t c16[64];
  for (std::uint32_t code = 0; code < 64; ++code)
    c16[code] = t4::FloatToHalfBits(t4::TURBO_CENTROIDS_6BIT[code]);
  std::mt19937 rng(4321);
  std::uint8_t block[t4::kBlockBytes6];
  long checked = 0;
  for (int trial = 0; trial < 50000; ++trial) {
    for (auto& b : block)
      b = static_cast<std::uint8_t>(rng());
    // The reader's four 32-value slices (dims % 128 = 32 g ..): nibble bytes
    // at 2 + 16 g, qh bytes at kQhOffset + 8 g (Vrd6LoadSlice).
    for (std::uint32_t g = 0; g < 4; ++g) {
      std::uint32_t w[4], qh[2], off[16];
      std::memcpy(w, block + 2 + 16 * g, sizeof(w));
      std::memcpy(qh, block + t4::kQhOffset + 8 * g, sizeof(qh));
      gufo::hip::Vrd6PairOffsets(w, qh, off);
      for (std::uint32_t m = 0; m < 16; ++m) {
        const bool in_range =
            off[m] % 4 == 0 && off[m] / 4 < gufo::hip::kVrd6LutEntries;
        const std::uint32_t entry =
            in_range ? gufo::hip::Vrd6LutEntry(c16, off[m] / 4) : 0U;
        for (std::uint32_t e = 0; e < 2; ++e) {
          const std::uint32_t j = 32 * g + 2 * m + e;
          const std::uint16_t want = t4::FloatToHalfBits(
              t4::CentroidValue(t4::Bits::k6, t4::CodeAt(t4::Bits::k6, block, j)));
          const std::uint16_t got =
              static_cast<std::uint16_t>(entry >> (16 * e));
          ++checked;
          if ((!in_range || got != want) && failures++ < 10)
            std::printf("FAIL pair g=%u m=%u e=%u off=%u got=%04x want=%04x\n",
                        g, m, e, off[m], got, want);
        }
      }
    }
  }
  std::printf("pair values checked: %ld\n", checked);

  // Every table entry is reachable (index = nibble byte | qh nibble << 8).
  for (std::uint32_t idx = 0; idx < gufo::hip::kVrd6LutEntries; ++idx) {
    std::uint8_t b[t4::kBlockBytes6] = {};
    b[2] = static_cast<std::uint8_t>(idx & 0xFFU);  // values 0, 1
    b[t4::kQhOffset] = static_cast<std::uint8_t>((idx >> 8) & 0xFU);
    std::uint32_t w[4], qh[2], off[16];
    std::memcpy(w, b + 2, sizeof(w));
    std::memcpy(qh, b + t4::kQhOffset, sizeof(qh));
    gufo::hip::Vrd6PairOffsets(w, qh, off);
    if (off[0] != 4 * idx && failures++ < 20)
      std::printf("FAIL index idx=%u off=%u\n", idx, off[0]);
  }

  // Vrd6KeyColumn: bijection on 0..31; key 2i + h of half kb -> 16 kb + 8 h + i.
  bool seen[32] = {};
  for (std::uint32_t key = 0; key < 32; ++key) {
    const std::uint32_t col = gufo::hip::Vrd6KeyColumn(key);
    const std::uint32_t kb = key / 16, i = (key % 16) / 2, h = key % 2;
    if ((col >= 32 || seen[col] || col != 16 * kb + 8 * h + i) &&
        failures++ < 30)
      std::printf("FAIL column key=%u col=%u\n", key, col);
    if (col < 32)
      seen[col] = true;
  }
  std::printf("%s %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

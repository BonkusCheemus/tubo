// sn_iq3_cut.hpp decode test. For every index mode (Idx 0, 1, 2)
// checks that
//  1. TableIndex<Idx> is a bijection of (qs byte, qh bit, sign nibble) onto
//     the table, and every WordAddrs byte address is dword aligned;
//  2. for random qs / qh / sign words, the table word the kernel's address
//     arithmetic (Tags + WordAddrs) reads is the entry of the (qs byte, qh
//     bit, sign nibble) the IQ3_S definition assigns to that grid word:
//     word i of sub-block s uses qs byte i, qh bit i of byte s of the qh
//     dword, and sign nibble i of the sub-block's sign dword.
// The "table entry" is a hash of the triple, so a wrong slot cannot pass by
// holding the same weights; this is a bit-exactness check of the addressing,
// independent of the grid. Host only; no HIP, no GPU.
//
// Registered as the qwen_sn_iq3_cut_test CMake target.

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "src/models/qwen/hip/kernels/sn_iq3_cut.hpp"

namespace {

using gufo::hip::cut::kTableWords;
using gufo::hip::cut::TableIndex;
using gufo::hip::cut::Tags;
using gufo::hip::cut::WordAddrs;

int g_failures = 0;

void Check(bool ok, const char* what, std::uint32_t idx) {
  if (!ok) {
    std::printf("  FAIL  Idx %u: %s\n", idx, what);
    ++g_failures;
  }
}

std::uint32_t Entry(std::uint32_t qs8, std::uint32_t qh, std::uint32_t sign) {
  std::uint32_t h = (qs8 * 0x9E3779B1U) ^ (qh * 0x85EBCA6BU) ^ (sign * 0xC2B2AE35U);
  h ^= h >> 15;
  return (h * 0x27D4EB2FU) | 1U;  // never 0 = "unwritten"
}

template<std::uint32_t Idx>
void Run() {
  constexpr std::uint32_t kWords = kTableWords<Idx>;
  std::vector<std::uint32_t> table(kWords, 0U);
  for (std::uint32_t sign = 0; sign < 16U; ++sign)
    for (std::uint32_t qh = 0; qh < 2U; ++qh)
      for (std::uint32_t qs8 = 0; qs8 < 256U; ++qs8) {
        const std::uint32_t at = TableIndex<Idx>(qs8, qh, sign);
        Check(at < kWords, "table index in range", Idx);
        Check(table[at] == 0U, "table index unique", Idx);
        table[at] = Entry(qs8, qh, sign);
      }

  std::mt19937 rng(0xC07U + Idx);
  std::uint32_t bad = 0;
  for (std::uint32_t trial = 0; trial < 2000000U; ++trial) {
    const std::uint32_t signs = rng();
    const std::uint32_t qh_word = rng();
    const std::uint32_t s = rng() & 3U;  // sub-block's byte in qh_word
    const std::uint32_t qs[2] = {static_cast<std::uint32_t>(rng()),
                                 static_cast<std::uint32_t>(rng())};
    std::uint32_t tag[2];
    std::uint32_t addr[8];
    Tags<Idx>(signs, qh_word, 8U * s, tag);
    WordAddrs<Idx>(tag, qs, addr);
    for (std::uint32_t i = 0; i < 8U; ++i) {
      const std::uint32_t qs8 = (qs[i >> 2] >> (8U * (i & 3U))) & 0xFFU;
      const std::uint32_t qh = (qh_word >> ((8U * s) + i)) & 1U;
      const std::uint32_t sign = (signs >> (4U * i)) & 0xFU;
      const bool aligned = (addr[i] & 3U) == 0U;
      const bool in_range = (addr[i] >> 2U) < kWords;
      if (!aligned || !in_range || table[addr[i] >> 2U] != Entry(qs8, qh, sign))
        ++bad;
    }
  }
  Check(bad == 0U, "address arithmetic reads the defined entry", Idx);
  std::printf("  Idx %u: %s (%u mismatching words of 16000000)\n", Idx,
              bad == 0U ? "ok" : "FAIL", bad);
}

}  // namespace

int main() {
  Run<0>();
  Run<1>();
  Run<2>();
  if (g_failures != 0) {
    std::printf("sn_iq3_cut_test: %d FAILED\n", g_failures);
    return 1;
  }
  std::printf("sn_iq3_cut_test: ok\n");
  return 0;
}

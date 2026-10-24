// ATTN6 CPU check: Attn6PackCodes (the attn6 reader's packed Turbo6 code
// offsets) agrees with turbo4::CodeAt for every value of random blocks, and
// Attn6SplitSize keeps a plan the WMMA reader's partial buffer can hold.
//
//   g++ -std=c++20 -I. tests/models/qwen/attn6_codes_test.cpp -o /tmp/attn6
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

#include "src/models/qwen/hip/turbo4_kernels.hpp"

namespace t4 = gufo::hip::turbo4;

int main() {
  int failures = 0;
  std::mt19937 rng(1234);
  std::uint8_t block[t4::kBlockBytes6];
  long checked = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    for (auto& b : block)
      b = static_cast<std::uint8_t>(rng());
    // The reader's four 32-value slices (dgroup % 4): nibble bytes at
    // 2 + 16g, qh bytes at kQhOffset + 8g, as LoadTurbo4Slice<k6> reads them.
    for (std::uint32_t g = 0; g < 4; ++g) {
      std::uint32_t w[4], qh[2], words[2][4];
      std::memcpy(w, block + 2 + 16 * g, sizeof(w));
      std::memcpy(qh, block + t4::kQhOffset + 8 * g, sizeof(qh));
      gufo::hip::Attn6PackCodes(w, qh, words);
      for (std::uint32_t h = 0; h < 2; ++h)
        for (std::uint32_t r = 0; r < 4; ++r)
          for (std::uint32_t k = 0; k < 4; ++k) {
            const std::uint32_t j = 16 * h + 4 * k + r;
            const std::uint32_t got = (words[h][r] >> (8 * k)) & 0xFFU;
            const std::uint32_t want =
                4U * t4::CodeAt(t4::Bits::k6, block, 32 * g + j);
            ++checked;
            if (got != want && failures++ < 10)
              std::printf("FAIL code g=%u j=%u got=%u want=%u\n", g, j, got,
                          want);
          }
    }
  }
  std::printf("codes checked: %ld\n", checked);

  for (std::size_t ctx = 1; ctx <= 70000; ctx += (ctx < 2048 ? 1 : 97)) {
    for (std::size_t cap : {0, 1, 8, 15, 20, 40, 1000}) {
      const std::size_t size = gufo::hip::Attn6SplitSize(ctx, cap);
      const std::size_t count = (ctx + size - 1) / size;
      const bool ok =
          size % 32 == 0 && size > 0 &&
          count <= gufo::hip::Turbo4DecodeSplitCount(ctx) &&
          (cap == 0 || count <= cap ||
           count == gufo::hip::Turbo4DecodeSplitCount(ctx)) &&
          (cap != 0 || size == gufo::hip::Turbo4DecodeSplitSize(ctx));
      if (!ok && failures++ < 20)
        std::printf("FAIL split ctx=%zu cap=%zu size=%zu count=%zu\n", ctx,
                    cap, size, count);
    }
  }
  std::printf("%s %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}

// Bit-exact CPU decode tests for the quant types the iron (GSQ-RCO)
// GGUF needs beyond UD-Q4_K_XL: Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_M.
//
// Oracle: dequantize_row_* from llama.cpp ggml/src/ggml-quants.c (MIT), ported
// below line for line with the original variable names. The grid tables are
// shared with the code under test (iquant_grids.inc, generated from
// ggml-common.h), so table transcription errors are caught by the second
// oracle, tests/core/iquant_gguf_py_check.py, which compares the same blocks
// against gguf-py's independent tables. Set GUFO_IQUANT_DUMP_DIR to write
// <type>.blocks / <type>.f32 for that script.
//
// Exactness: every comparison is on float BIT PATTERNS, not a tolerance.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "tests/core/iquant_test_blocks.hpp"

namespace {

#define GUFO_IQ_TABLE(type, name, size) constexpr type k##name[size]
#include "src/core/quant/iquant_grids.inc"
#undef GUFO_IQ_TABLE

constexpr std::uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
constexpr float IQ1S_DELTA = 0.125F;
constexpr std::size_t QK_K = 256;

float H(std::uint16_t h) {
  return gufo::quant::Fp16ToFloat(h);
}

// ---- llama.cpp reference (ggml-quants.c), field names kept ----------------
void ref_q2_K(const gufo::quant::block_q2_K* x, float* y, std::size_t nb) {
  for (std::size_t i = 0; i < nb; i++) {
    const float d = H(x[i].d);
    const float min = H(x[i].dmin);
    const std::uint8_t* q = x[i].qs;
    int is = 0;
    float dl, ml;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        std::uint8_t sc = x[i].scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * ((std::int8_t)((q[l] >> shift) & 3)) - ml;
        sc = x[i].scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l)
          *y++ = dl * ((std::int8_t)((q[l + 16] >> shift) & 3)) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}

void ref_iq2_xxs(const gufo::quant::block_iq2_xxs* x, float* y,
                 std::size_t nb) {
  std::uint32_t aux32[2];
  const std::uint8_t* aux8 = (const std::uint8_t*)aux32;
  for (std::size_t i = 0; i < nb; i++) {
    const float d = H(x[i].d);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(std::uint32_t));
      const float db = d * (0.5F + (aux32[1] >> 28)) * 0.25F;
      for (int l = 0; l < 4; ++l) {
        const std::uint8_t* grid =
            (const std::uint8_t*)(kIq2xxsGrid + aux8[l]);
        const std::uint8_t signs = kKsignsIq2xs[(aux32[1] >> 7 * l) & 127];
        for (int j = 0; j < 8; ++j)
          y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.F : 1.F);
        y += 8;
      }
    }
  }
}

void ref_iq2_xs(const gufo::quant::block_iq2_xs* x, float* y,
                std::size_t nb) {
  float db[2];
  for (std::size_t i = 0; i < nb; i++) {
    const float d = H(x[i].d);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      db[0] = d * (0.5F + (x[i].scales[ib32] & 0xf)) * 0.25F;
      db[1] = d * (0.5F + (x[i].scales[ib32] >> 4)) * 0.25F;
      for (int l = 0; l < 4; ++l) {
        const std::uint8_t* grid =
            (const std::uint8_t*)(kIq2xsGrid + (x[i].qs[4 * ib32 + l] & 511));
        const std::uint8_t signs = kKsignsIq2xs[x[i].qs[4 * ib32 + l] >> 9];
        for (int j = 0; j < 8; ++j)
          y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.F : 1.F);
        y += 8;
      }
    }
  }
}

void ref_iq2_s(const gufo::quant::block_iq2_s* x, float* y, std::size_t nb) {
  float db[2];
  for (std::size_t i = 0; i < nb; i++) {
    const float d = H(x[i].d);
    const std::uint8_t* qs = x[i].qs;
    const std::uint8_t* qh = x[i].qh;
    const std::uint8_t* signs = qs + QK_K / 8;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      db[0] = d * (0.5F + (x[i].scales[ib32] & 0xf)) * 0.25F;
      db[1] = d * (0.5F + (x[i].scales[ib32] >> 4)) * 0.25F;
      for (int l = 0; l < 4; ++l) {
        const float dl = db[l / 2];
        const std::uint8_t* grid = (const std::uint8_t*)(kIq2sGrid +
            (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
        for (int j = 0; j < 8; ++j)
          y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.F : 1.F);
        y += 8;
      }
      qs += 4;
      signs += 4;
    }
  }
}

void ref_iq3_xxs(const gufo::quant::block_iq3_xxs* x, float* y,
                 std::size_t nb) {
  std::uint32_t aux32;
  for (std::size_t i = 0; i < nb; i++) {
    const float d = H(x[i].d);
    const std::uint8_t* qs = x[i].qs;
    const std::uint8_t* scales_and_signs = qs + QK_K / 4;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(std::uint32_t));
      const float db = d * (0.5F + (aux32 >> 28)) * 0.5F;
      for (int l = 0; l < 4; ++l) {
        const std::uint8_t signs = kKsignsIq2xs[(aux32 >> 7 * l) & 127];
        const std::uint8_t* grid1 =
            (const std::uint8_t*)(kIq3xxsGrid + qs[2 * l + 0]);
        const std::uint8_t* grid2 =
            (const std::uint8_t*)(kIq3xxsGrid + qs[2 * l + 1]);
        for (int j = 0; j < 4; ++j) {
          y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.F : 1.F);
          y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.F : 1.F);
        }
        y += 8;
      }
      qs += 8;
    }
  }
}

void ref_iq1_m(const gufo::quant::block_iq1_m* x, float* y, std::size_t nb) {
  float delta[4];
  std::uint16_t idx[4];
  union {
    std::uint16_t f16;
    std::uint16_t u16;
  } scale;
  for (std::size_t i = 0; i < nb; i++) {
    const std::uint16_t* sc = (const std::uint16_t*)x[i].scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = H(scale.f16);
    const std::uint8_t* qs = x[i].qs;
    const std::uint8_t* qh = x[i].qh;
    for (int ib = 0; ib < 8; ++ib) {
      const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
      const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
      idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
      idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
      idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
      idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
      delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
      delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
      delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
      delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
      for (int l = 0; l < 2; ++l) {
        const std::int8_t* grid = (const std::int8_t*)(kIq1sGrid + idx[l]);
        for (int j = 0; j < 8; ++j) y[j] = dl1 * (grid[j] + delta[l]);
        y += 8;
      }
      for (int l = 2; l < 4; ++l) {
        const std::int8_t* grid = (const std::int8_t*)(kIq1sGrid + idx[l]);
        for (int j = 0; j < 8; ++j) y[j] = dl2 * (grid[j] + delta[l]);
        y += 8;
      }
      qs += 4;
      qh += 2;
    }
  }
}

using gufo::core::GgmlType;

void Reference(GgmlType type, const std::uint8_t* bytes, float* out,
               std::size_t nb) {
  switch (type) {
    case GgmlType::kQ2_K:
      return ref_q2_K(
          reinterpret_cast<const gufo::quant::block_q2_K*>(bytes), out, nb);
    case GgmlType::kIQ2_XXS:
      return ref_iq2_xxs(
          reinterpret_cast<const gufo::quant::block_iq2_xxs*>(bytes), out, nb);
    case GgmlType::kIQ2_XS:
      return ref_iq2_xs(
          reinterpret_cast<const gufo::quant::block_iq2_xs*>(bytes), out, nb);
    case GgmlType::kIQ2_S:
      return ref_iq2_s(
          reinterpret_cast<const gufo::quant::block_iq2_s*>(bytes), out, nb);
    case GgmlType::kIQ3_XXS:
      return ref_iq3_xxs(
          reinterpret_cast<const gufo::quant::block_iq3_xxs*>(bytes), out, nb);
    case GgmlType::kIQ1_M:
      return ref_iq1_m(
          reinterpret_cast<const gufo::quant::block_iq1_m*>(bytes), out, nb);
    default:
      std::abort();
  }
}

void Gufo(GgmlType type, const std::uint8_t* bytes, float* out,
          std::size_t k) {
  switch (type) {
    case GgmlType::kQ2_K:
      return gufo::quant::DequantizeQ2_K(bytes, out, k);
    case GgmlType::kIQ2_XXS:
      return gufo::quant::DequantizeIQ2_XXS(bytes, out, k);
    case GgmlType::kIQ2_XS:
      return gufo::quant::DequantizeIQ2_XS(bytes, out, k);
    case GgmlType::kIQ2_S:
      return gufo::quant::DequantizeIQ2_S(bytes, out, k);
    case GgmlType::kIQ3_XXS:
      return gufo::quant::DequantizeIQ3_XXS(bytes, out, k);
    case GgmlType::kIQ1_M:
      return gufo::quant::DequantizeIQ1_M(bytes, out, k);
    default:
      std::abort();
  }
}

bool g_failed = false;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    g_failed = true;
  }
}

constexpr std::size_t kBlocks = 4096;  // 65k grid lookups per 1024-entry grid

void TestType(const gufo::test::IQuantCase& c) {
  const std::size_t block_bytes = gufo::quant::QuantizedRowBytes(c.type, 256);
  Check(gufo::quant::QuantizedBlockElements(c.type) == 256,
        std::string(c.name) + " block elements");
  Check(gufo::quant::EncodedSizeBytes(c.type, 256 * 5) == 5 * block_bytes,
        std::string(c.name) + " encoded size");
  Check(gufo::quant::EncodedSizeBytes(c.type, 255) == 0,
        std::string(c.name) + " misaligned size rejected");

  const auto bytes = gufo::test::MakeIQuantBlocks(c.type, kBlocks, 0xC0DE0001U);
  std::vector<float> expected(kBlocks * 256);
  std::vector<float> actual(kBlocks * 256, -1.0F);
  Reference(c.type, bytes.data(), expected.data(), kBlocks);
  Gufo(c.type, bytes.data(), actual.data(), kBlocks * 256);

  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    std::uint32_t a;
    std::uint32_t e;
    std::memcpy(&a, &actual[i], 4);
    std::memcpy(&e, &expected[i], 4);
    if (a != e && mismatches++ < 5) {
      std::cerr << c.name << " element " << i << " gufo=" << actual[i]
                << " reference=" << expected[i] << '\n';
    }
  }
  Check(mismatches == 0, std::string(c.name) + " bit-exact vs llama.cpp");
  std::cout << (mismatches == 0 ? "[ OK ] " : "[FAIL] ") << c.name << ": "
            << expected.size() - mismatches << "/" << expected.size()
            << " elements bit-identical (" << kBlocks << " blocks)\n";

  if (const char* dir = std::getenv("GUFO_IQUANT_DUMP_DIR"); dir != nullptr) {
    const std::string base = std::string(dir) + "/" + c.name;
    if (FILE* f = std::fopen((base + ".blocks").c_str(), "wb")) {
      std::fwrite(bytes.data(), 1, bytes.size(), f);
      std::fclose(f);
    }
    if (FILE* f = std::fopen((base + ".f32").c_str(), "wb")) {
      std::fwrite(actual.data(), 4, actual.size(), f);
      std::fclose(f);
    }
  }
}

// DecodeQuantSub16 (device) derives sign bytes from the 7-bit index instead of
// the 128-entry table; they must be the same function.
void TestSignFormula() {
  for (std::uint32_t i = 0; i < 128; ++i) {
    const std::uint32_t formula = i | ((__builtin_popcount(i) & 1U) << 7U);
    Check(formula == kKsignsIq2xs[i], "ksigns_iq2xs == i | parity << 7");
  }
}

// Every grid entry the device sign trick negates must be >= 1 per byte, or
// 0x01010100 - v would not borrow evenly across bytes (see NegateSelectedBytes).
void TestGridMagnitudes() {
  const auto check8 = [](const std::uint64_t* grid, std::size_t n,
                         const char* name) {
    for (std::size_t i = 0; i < n; ++i) {
      for (int j = 0; j < 8; ++j) {
        const auto v = static_cast<std::uint8_t>(grid[i] >> (8 * j));
        Check(v >= 1 && v < 128, std::string(name) + " magnitude in [1,127]");
      }
    }
  };
  check8(kIq2xxsGrid, 256, "iq2xxs_grid");
  check8(kIq2xsGrid, 512, "iq2xs_grid");
  check8(kIq2sGrid, 1024, "iq2s_grid");
  for (std::size_t i = 0; i < 256; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto v = static_cast<std::uint8_t>(kIq3xxsGrid[i] >> (8 * j));
      Check(v >= 1 && v < 128, "iq3xxs_grid magnitude in [1,127]");
    }
  }
  const std::uint32_t* iq3s = gufo::quant::Iq3sGrid();
  for (std::size_t i = 0; i < 512; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto v = static_cast<std::uint8_t>(iq3s[i] >> (8 * j));
      Check(v >= 1 && v < 128, "iq3s_grid magnitude in [1,127]");
    }
  }
  for (std::size_t i = 0; i < 2048; ++i) {
    for (int j = 0; j < 8; ++j) {
      const auto g = static_cast<std::int8_t>(
          static_cast<std::uint8_t>(kIq1sGrid[i] >> (8 * j)));
      Check(g >= -1 && g <= 1, "iq1s_grid entries in {-1,0,1}");
    }
  }
}

// GGUF reader accepts the tensor shape/size of every new type.
void TestGgufTypeIds() {
  Check(static_cast<int>(GgmlType::kIQ2_XS) == 17, "IQ2_XS id");
  Check(static_cast<int>(GgmlType::kIQ3_XXS) == 18, "IQ3_XXS id");
  Check(static_cast<int>(GgmlType::kIQ2_S) == 22, "IQ2_S id");
  Check(static_cast<int>(GgmlType::kIQ1_M) == 29, "IQ1_M id");
  Check(gufo::core::ToString(GgmlType::kIQ1_M) == "IQ1_M", "IQ1_M name");
  Check(gufo::core::ToString(GgmlType::kIQ3_XXS) == "IQ3_XXS", "IQ3_XXS name");
  // Bits per element sanity (ggufpeek: 2.625 2.0625 2.3125 2.5625 3.0625 1.75).
  const double bpw[] = {2.625, 2.0625, 2.3125, 2.5625, 3.0625, 1.75};
  std::size_t n = 0;
  for (const auto& c : gufo::test::kIQuantCases) {
    Check(gufo::quant::EncodedSizeBytes(c.type, 256 * 8) * 8.0 / (256 * 8) ==
              bpw[n++],
          std::string(c.name) + " bits per element");
  }
}

}  // namespace

int main() {
  TestSignFormula();
  TestGridMagnitudes();
  TestGgufTypeIds();
  for (const auto& c : gufo::test::kIQuantCases) {
    TestType(c);
  }
  if (g_failed) {
    return 1;
  }
  std::cout << "iquant_decode_test: all checks passed\n";
  return 0;
}

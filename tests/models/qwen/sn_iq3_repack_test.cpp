// sn_iq3.hpp layout test. Checks, for random and edge IQ3_S blocks
// and for row counts that are and are not multiples of the 8-row interleave,
// that SnIq3Unpack(SnIq3Repack(x)) is x byte for byte, that a scalar dequant
// read out of the repacked planes equals a scalar dequant of the original
// block, and that the repacked tensor never grows past +2%.
//
// Registered as the qwen_sn_iq3_repack_test CMake target; links gufo_core for
// Iq3sGrid / Fp16ToFloat / DequantizeIQ3_S.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3.hpp"

namespace {

using gufo::hip::kSnIq3BlockBytes;
using gufo::hip::kSnIq3Elements;
using gufo::hip::kSnIq3RowsPerGroup;
using gufo::hip::SnIq3BlockStride;
using gufo::hip::SnIq3Blocks;
using gufo::hip::SnIq3GroupBytes;
using gufo::hip::SnIq3GroupRows;
using gufo::hip::SnIq3PackedBytes;
using gufo::hip::SnIq3Repack;
using gufo::hip::SnIq3Unpack;

int g_failures = 0;

void Check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL  %s\n", what);
    ++g_failures;
  }
}

std::uint32_t NextRandom(std::uint32_t* state) {
  *state = (*state * 1664525U) + 1013904223U;
  return *state;
}

void FillRandom(std::vector<std::uint8_t>* bytes, std::uint32_t seed) {
  std::uint32_t state = seed;
  for (std::uint8_t& byte : *bytes)
    byte = static_cast<std::uint8_t>(NextRandom(&state) >> 24);
}

// One IQ3_S block with explicitly chosen fields; `index` picks which edge
// pattern to use and also rotates through all eight cases. d is left as the
// fp16 super-scale bits of the case.
void FillEdgeBlock(std::uint8_t* block, std::uint32_t index) {
  std::vector<std::uint8_t> filler(kSnIq3BlockBytes);
  FillRandom(&filler, 0x9E3779B9U + index);
  std::memcpy(block, filler.data(), kSnIq3BlockBytes);
  std::uint8_t* const qs = block + 2;
  std::uint8_t* const qh = block + 66;
  std::uint8_t* const signs = block + 74;
  std::uint8_t* const scales = block + 106;
  std::uint16_t d_bits = 0;
  switch (index % 8U) {
    case 0:  // every qh bit and every sign bit set, all qs indices zero
      d_bits = 0x3C00;  // 1.0
      std::fill(qs, qs + 64, 0x00);
      std::fill(qh, qh + 8, 0xFF);
      std::fill(signs, signs + 32, 0xFF);
      std::fill(scales, scales + 4, 0xFF);  // both nibbles 15
      break;
    case 1:  // all indices 0xFF, scales zero, tiny subnormal d
      d_bits = 0x0001;
      std::fill(qs, qs + 64, 0xFF);
      std::fill(qh, qh + 8, 0xFF);
      std::fill(signs, signs + 32, 0x00);
      std::fill(scales, scales + 4, 0x00);
      break;
    case 2:  // negative zero scale, empty high bits
      d_bits = 0x8000;
      std::fill(qs, qs + 64, 0x00);
      std::fill(qh, qh + 8, 0x00);
      std::fill(signs, signs + 32, 0x00);
      std::fill(scales, scales + 4, 0x00);
      break;
    case 3:  // max normal d, ramped indices, both nibble extremes
      d_bits = 0x7BFF;
      for (int i = 0; i < 64; ++i)
        qs[i] = static_cast<std::uint8_t>(i);
      std::fill(qh, qh + 8, 0xAA);
      std::fill(signs, signs + 32, 0x55);
      std::fill(scales, scales + 4, 0x0F);
      break;
    case 4:  // +inf d, all sign nibbles set
      d_bits = 0x7C00;
      for (int i = 0; i < 64; ++i)
        qs[i] = static_cast<std::uint8_t>(0xA5 + i);
      std::fill(qh, qh + 8, 0x55);
      std::fill(signs, signs + 32, 0xAA);
      std::fill(scales, scales + 4, 0xF0);
      break;
    case 5:  // -inf d, every qh bit set
      d_bits = 0xFC00;
      std::fill(qs, qs + 64, 0x80);
      std::fill(qh, qh + 8, 0xFF);
      std::fill(signs, signs + 32, 0x80);
      std::fill(scales, scales + 4, 0x5A);
      break;
    case 6:  // NaN d, single-bit indices and signs
      d_bits = 0x7E01;
      std::fill(qs, qs + 64, 0x01);
      std::fill(qh, qh + 8, 0x01);
      std::fill(signs, signs + 32, 0x01);
      std::fill(scales, scales + 4, 0x10);
      break;
    default:  // max subnormal d
      d_bits = 0x03FF;
      std::fill(qs, qs + 64, 0xFE);
      std::fill(qh, qh + 8, 0x80);
      std::fill(signs, signs + 32, 0xFE);
      std::fill(scales, scales + 4, 0xE1);
      break;
  }
  std::memcpy(block, &d_bits, 2);
}

void FillEdgeTensor(std::vector<std::uint8_t>* data, std::size_t rows,
                    std::size_t columns) {
  const std::size_t blocks = SnIq3Blocks(columns);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t b = 0; b < blocks; ++b)
      FillEdgeBlock(data->data() + (((row * blocks) + b) * kSnIq3BlockBytes),
                    static_cast<std::uint32_t>((row * 5U) + b));
}

// ggml's dequantize_row_iq3_s element, read from the original block layout.
float BlockValue(const std::uint8_t* block, std::size_t index) {
  const std::size_t ib32 = index / 32;
  const std::size_t within = index % 32;
  const std::size_t l = within / 8;
  const std::size_t j = within % 8;
  const std::size_t half = j / 4;
  const std::size_t jj = j % 4;
  const std::uint32_t grid_index =
      static_cast<std::uint32_t>(block[2 + (ib32 * 8) + (2 * l) + half]) |
      ((static_cast<std::uint32_t>(block[66 + ib32]) << (8 - (2 * l) - half)) &
       256U);
  const auto* const grid = reinterpret_cast<const std::uint8_t*>(
      &gufo::quant::Iq3sGrid()[grid_index]);
  const std::uint8_t sign_byte = block[74 + (ib32 * 4) + l];
  const std::uint8_t scale_byte = block[106 + (ib32 / 2)];
  const int nibble = (ib32 % 2) == 0 ? (scale_byte & 0x0F) : (scale_byte >> 4);
  std::uint16_t d_bits;
  std::memcpy(&d_bits, block, 2);
  const float db = gufo::quant::Fp16ToFloat(d_bits) *
                   static_cast<float>(1 + (2 * nibble));
  const float magnitude = static_cast<float>(grid[jj]);
  const bool negate = (sign_byte & (1U << j)) != 0U;
  return negate ? -(db * magnitude) : (db * magnitude);
}

// The same element, read through the repacked planes only.
float PackedValue(const std::uint8_t* packed, std::size_t rows,
                  std::size_t columns, std::size_t row, std::size_t index) {
  const std::size_t blocks = SnIq3Blocks(columns);
  const std::size_t block = index / kSnIq3Elements;
  const std::size_t within = index % kSnIq3Elements;
  const std::size_t group = row / kSnIq3RowsPerGroup;
  const std::size_t r = row % kSnIq3RowsPerGroup;
  const std::size_t n = SnIq3GroupRows(rows, group);
  std::size_t base = 0;
  for (std::size_t g = 0; g < group; ++g)
    base += SnIq3GroupBytes(blocks, SnIq3GroupRows(rows, g));
  const std::size_t record = base + (block * SnIq3BlockStride(n));
  const std::size_t ib32 = within / 32;
  const std::size_t sub = within % 32;
  const std::size_t l = sub / 8;
  const std::size_t j = sub % 8;
  const std::size_t half = j / 4;
  const std::size_t jj = j % 4;
  const std::size_t qs_byte = (ib32 * 8) + (2 * l) + half;
  const std::uint8_t qs = packed[record + ((qs_byte / 16) * 16 * n) +
                                 (r * 16) + (qs_byte % 16)];
  const std::uint8_t qh = packed[record + (96 * n) + (r * 8) + ib32];
  const std::size_t sign_byte = (ib32 * 4) + l;
  const std::uint8_t signs = packed[record + (64 * n) +
                                    ((sign_byte / 16) * 16 * n) + (r * 16) +
                                    (sign_byte % 16)];
  const std::uint8_t scale_byte =
      packed[record + (104 * n) + (r * 4) + (ib32 / 2)];
  std::uint16_t d_bits;
  std::memcpy(&d_bits, &packed[record + (108 * n) + (r * 2)], 2);
  const std::uint32_t grid_index =
      static_cast<std::uint32_t>(qs) |
      ((static_cast<std::uint32_t>(qh) << (8 - (2 * l) - half)) & 256U);
  const auto* const grid = reinterpret_cast<const std::uint8_t*>(
      &gufo::quant::Iq3sGrid()[grid_index]);
  const int nibble = (ib32 % 2) == 0 ? (scale_byte & 0x0F) : (scale_byte >> 4);
  const float db = gufo::quant::Fp16ToFloat(d_bits) *
                   static_cast<float>(1 + (2 * nibble));
  const float magnitude = static_cast<float>(grid[jj]);
  const bool negate = (signs & (1U << j)) != 0U;
  return negate ? -(db * magnitude) : (db * magnitude);
}

bool SameBits(float a, float b) {
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  std::memcpy(&x, &a, 4);
  std::memcpy(&y, &b, 4);
  if (x == y) return true;
  // NaN payloads are not part of what the layout has to preserve.
  const bool x_nan = (x & 0x7FFFFFFFU) > 0x7F800000U;
  const bool y_nan = (y & 0x7FFFFFFFU) > 0x7F800000U;
  return x_nan && y_nan;
}

bool SameFloats(const float* a, const float* b, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i)
    if (!SameBits(a[i], b[i])) return false;
  return true;
}

void CheckTensor(std::size_t rows, std::size_t columns,
                 const std::vector<std::uint8_t>& data, const char* label) {
  const std::size_t blocks = SnIq3Blocks(columns);
  const std::size_t original = rows * blocks * kSnIq3BlockBytes;
  const std::size_t packed_bytes = SnIq3PackedBytes(rows, columns);
  std::vector<std::uint8_t> packed(packed_bytes, 0x5A);
  std::vector<std::uint8_t> restored(original, 0x5A);
  SnIq3Repack(data.data(), packed.data(), rows, columns);
  SnIq3Unpack(packed.data(), restored.data(), rows, columns);

  if (std::memcmp(data.data(), restored.data(), original) != 0) {
    std::size_t first = original;
    for (std::size_t i = 0; i < original; ++i) {
      if (data[i] != restored[i]) {
        first = i;
        break;
      }
    }
    std::printf("  FAIL  %s %zux%zu round trip, first bad byte %zu\n", label,
                rows, columns, first);
    ++g_failures;
  }

  bool dequant_ok = true;
  for (std::size_t row = 0; row < rows && dequant_ok; ++row) {
    for (std::size_t index = 0; index < columns; ++index) {
      const std::uint8_t* const block =
          data.data() +
          (((row * blocks) + (index / kSnIq3Elements)) * kSnIq3BlockBytes);
      const float from_block = BlockValue(block, index % kSnIq3Elements);
      const float from_packed =
          PackedValue(packed.data(), rows, columns, row, index);
      if (!SameBits(from_block, from_packed)) {
        std::printf("  FAIL  %s %zux%zu dequant, row %zu element %zu\n", label,
                    rows, columns, row, index);
        dequant_ok = false;
        break;
      }
    }
  }
  if (!dequant_ok) ++g_failures;

  // The in-test dequant must agree with the shipped CPU dequantizer.
  std::vector<float> reference(columns);
  std::vector<float> mine(columns);
  bool shipped_ok = true;
  for (std::size_t row = 0; row < rows && shipped_ok; ++row) {
    const std::uint8_t* const row_in =
        data.data() + (row * blocks * kSnIq3BlockBytes);
    gufo::quant::DequantizeIQ3_S(row_in, reference.data(), columns);
    for (std::size_t index = 0; index < columns; ++index)
      mine[index] = BlockValue(
          row_in + ((index / kSnIq3Elements) * kSnIq3BlockBytes),
          index % kSnIq3Elements);
    if (!SameFloats(reference.data(), mine.data(), columns)) {
      std::printf("  FAIL  %s %zux%zu differs from DequantizeIQ3_S\n", label,
                  rows, columns);
      shipped_ok = false;
    }
  }
  if (!shipped_ok) ++g_failures;

  if (packed_bytes * 100U > original * 102U) {
    std::printf("  FAIL  %s %zux%zu grew from %zu to %zu bytes\n", label, rows,
                columns, original, packed_bytes);
    ++g_failures;
  }
}

void CheckSizes(std::size_t rows, std::size_t columns) {
  const std::size_t original = rows * SnIq3Blocks(columns) * kSnIq3BlockBytes;
  std::vector<std::uint8_t> random(original);
  FillRandom(&random, static_cast<std::uint32_t>((rows * 131U) + columns));
  CheckTensor(rows, columns, random, "random");
  std::vector<std::uint8_t> edge(original);
  FillEdgeTensor(&edge, rows, columns);
  CheckTensor(rows, columns, edge, "edge");
}

}  // namespace

int main() {
  std::printf("sn_iq3 layout test\n");
  for (std::size_t rows = 1; rows <= 17; ++rows)
    CheckSizes(rows, (rows % 4U) == 0U ? 1024U : 256U);
  for (const std::size_t rows : {24U, 32U, 33U, 64U})
    CheckSizes(rows, 512U);

  // A full-size shape: exact byte count for whole groups, plus round trip.
  const std::size_t rows = 1024;
  const std::size_t columns = 5120;
  const std::size_t original = rows * SnIq3Blocks(columns) * kSnIq3BlockBytes;
  std::vector<std::uint8_t> random(original);
  FillRandom(&random, 0xC0FFEEU);
  std::vector<std::uint8_t> packed(SnIq3PackedBytes(rows, columns));
  std::vector<std::uint8_t> restored(original);
  SnIq3Repack(random.data(), packed.data(), rows, columns);
  SnIq3Unpack(packed.data(), restored.data(), rows, columns);
  Check(std::memcmp(random.data(), restored.data(), original) == 0,
        "1024x5120 round trip");
  Check(SnIq3PackedBytes(rows, columns) == original,
        "whole groups keep the original byte count");
  Check(SnIq3PackedBytes(7, 256) == SnIq3BlockStride(7U),
        "a partial group pads the block record only");

  if (g_failures != 0) {
    std::printf("sn_iq3 layout test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("sn_iq3 layout test: ok\n");
  return 0;
}

// The Turbo4 PLANE INDEX authority, on CPU. This is the oracle the
// kernels are built against.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/turbo4_kernel_ref_test.cpp
//       -o /tmp/t4k
//
// WHY THIS FILE IS GREEN BEFORE ANY KERNEL EXISTS
//
// The kernels are HIP translation units: they can be COMPILED here but not RUN.
// That is a weak oracle for arithmetic. So everything a kernel computes with
// pure index math lives in turbo4_plane_index.hpp as constexpr and is called
// from BOTH sides. This file tests that authority; the kernel calls the same
// functions. What remains compiled-but-not-executed is stated at the end of
// this header rather than glossed.
//
// INDEPENDENCE. `SpecBlockByte` below is transcribed from the layout sentence
// and calls NO production function, exactly as in kv_turbo4_plane_layout_test.
// A test that computed the expected offset by calling the function under test
// would agree with every possible bug in it.
//
// GEOMETRY: UD-Q4_K_XL. head_count_kv 4, key_length 256, so 2 blocks per head
// and 528 B per token. 16 full-attention layers — block_count 65 includes one
// MTP layer, so (65 - 1) / interval 4 = 16.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "src/models/qwen/hip/turbo4_codec.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_plane_index.hpp"

namespace t4 = gufo::hip::turbo4;
namespace t4i = gufo::hip;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string &what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kBlocksPerHead = 2;
constexpr std::size_t kBlocksPerToken = 8;
constexpr std::size_t kTokenPitch = 528;
constexpr std::size_t kFullAttentionLayers = 16;

/// THE LAYOUT SENTENCE, TRANSCRIBED. No production call.
constexpr std::size_t SpecBlockByte(std::size_t token, std::size_t head) {
  return (token * kBlocksPerToken + head * kBlocksPerHead) * 66;
}

/// THE REAL LENGTH OF A [num_kv_heads, head_dim] FLOAT SOURCE.
/// [4, 256] floats == 4 KiB. Declared once so the writer call sites cannot
/// drift apart from the geometry they are testing.
constexpr std::size_t kWriterSourceBytes = kKvHeads * kHeadDim * sizeof(float);

/// Capacities derived from the SHAPE a call actually uses.
///
/// WHY DERIVED AND NOT TYPED. The new capacity checks are only meaningful if
/// the oracle hands the body the lengths a real caller would have. Hard-coding
/// 24576 at a call site that is deliberately testing `head_dim = 320` would
/// make the body refuse on CAPACITY while the test's name claimed it refused on
/// GEOMETRY — a green test asserting the wrong reason, which is the failure
/// this whole file exists to avoid. So the sizes follow the shape under test,
/// and the shape is the only thing a call site varies.
t4i::Turbo4BufferSizes ShapeSizes(std::size_t plane_bytes, std::uint32_t heads,
                                  std::uint32_t head_dim,
                                  std::uint32_t context) {
  const std::size_t row = static_cast<std::size_t>(head_dim) * sizeof(float);
  return t4i::Turbo4BufferSizes{plane_bytes,
                                static_cast<std::size_t>(heads) * row,
                                static_cast<std::size_t>(heads) * row,
                                static_cast<std::size_t>(context) * sizeof(float),
                                0U};
}

/// Source text with C++ comments REMOVED, so a check about CODE can neither be
/// broken nor satisfied by PROSE.
///
/// This was written because the bare-literal check below failed against its own
/// explanatory comment: the comment said the bound "was `= 4096` written here",
/// and the check was scanning the whole file including comments. Rewording the
/// comment would have made the test pass today and leave it one comment away
/// from a false failure tomorrow. A comment mentioning a number is
/// documentation; a `= 4096` in code is the drift worth catching.
std::string StripComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  enum class St { kCode, kLine, kBlock, kStr, kChr };
  St state = St::kCode;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
    switch (state) {
      case St::kCode:
        if (c == '/' && n == '/') { state = St::kLine; ++i; }
        else if (c == '/' && n == '*') { state = St::kBlock; ++i; }
        else if (c == '"') { state = St::kStr; out += c; }
        else if (c == '\'') { state = St::kChr; out += c; }
        else out += c;
        break;
      case St::kLine:
        if (c == '\n') { state = St::kCode; out += c; }
        break;
      case St::kBlock:
        if (c == '*' && n == '/') { state = St::kCode; ++i; }
        else if (c == '\n') out += c;  // keep line count sane
        break;
      case St::kStr:
        out += c;
        if (c == '\\' && i + 1 < src.size()) { out += n; ++i; }
        else if (c == '"') state = St::kCode;
        break;
      case St::kChr:
        out += c;
        if (c == '\\' && i + 1 < src.size()) { out += n; ++i; }
        else if (c == '\'') state = St::kCode;
        break;
    }
  }
  return out;
}

void IndexMatchesSpec() {
  std::printf("index: Turbo4BlockByteOffset == the layout sentence\n");
  const std::size_t context = 11;
  bool all = true;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      std::size_t byte_offset = 0;
      if (!t4i::Turbo4BlockByteOffset(token, head, kHeadDim, kKvHeads, context,
                                      &byte_offset))
        all = false;
      if (byte_offset != SpecBlockByte(token, head)) all = false;
    }
  Check(all, "every (token, head) maps to the transcribed byte offset");

  std::size_t blocks = 0;
  bool idx_ok = true;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head)
      if (!t4i::Turbo4BlockIndex(token, head, kHeadDim, kKvHeads, context,
                                 &blocks))
        idx_ok = false;
  Check(idx_ok, "Turbo4BlockIndex succeeds for every in-range coordinate");
}

void GeometryConstants() {
  std::printf("geometry: the UD numbers, stated once\n");
  Check(t4i::Turbo4BlocksPerHead(kHeadDim) == kBlocksPerHead,
        "head_dim 256 spans TWO 128-value blocks, not one");
  Check(t4i::Turbo4BlocksPerToken(kHeadDim, kKvHeads) == kBlocksPerToken,
        "4 heads x 2 blocks = 8 blocks per token");
  Check(t4i::Turbo4BlocksPerToken(kHeadDim, kKvHeads) * 66 == kTokenPitch,
        "8 blocks x 66 B = the 528 B token pitch the copy uses");
  Check(t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, 4096) ==
            4096 * kTokenPitch,
        "one layer's plane is context x 528 B at context 4096");
  Check(kFullAttentionLayers == 16,
        "the UD full-attention count is 16, from (65 blocks - 1 MTP) / 4");
}

void RefusesRatherThanRounds() {
  std::printf("refusal: a straddling head is refused, not rounded\n");
  Check(t4i::Turbo4BlocksPerHead(320) == 0,
        "head_dim 320 spans no WHOLE blocks — zero, not a rounded 3");
  Check(t4i::Turbo4BlocksPerToken(320, kKvHeads) == 0,
        "so a token of 320-wide heads has no block count");
  std::size_t out = 0xDEADBEEF;
  Check(!t4i::Turbo4BlockByteOffset(0, 0, 320, kKvHeads, 4096, &out),
        "the byte offset refuses a straddling head");
  Check(out == 0xDEADBEEF, "and leaves *out untouched on refusal");
  Check(t4i::Turbo4LayerPlaneBytes(320, kKvHeads, 4096) == 0,
        "the plane size refuses too, rather than sizing a rounded one");
  // Zero width and zero heads must refuse explicitly: 0 % 128 == 0 would
  // otherwise pass an alignment test while addressing nothing.
  Check(t4i::Turbo4BlocksPerHead(0) == 0, "a zero-width head is refused");
  Check(t4i::Turbo4BlocksPerToken(kHeadDim, 0) == 0, "zero kv heads refused");
  Check(!t4i::Turbo4BlockByteOffset(0, 0, kHeadDim, kKvHeads, 0, &out),
        "a zero context is refused");
}

void RefusesOutOfRange() {
  std::printf("bounds: past the end refuses, and never wraps\n");
  const std::size_t context = 8;
  std::size_t out = 0;
  Check(!t4i::Turbo4BlockByteOffset(context, 0, kHeadDim, kKvHeads, context,
                                    &out),
        "a token past the context refuses");
  Check(!t4i::Turbo4BlockByteOffset(0, kKvHeads, kHeadDim, kKvHeads, context,
                                    &out),
        "a head past the kv head count refuses");
  // The hostile geometry. `token * blocks_per_token` wraps to a SMALL number
  // near SIZE_MAX, landing mid-plane where no bounds check would notice.
  const std::size_t huge = static_cast<std::size_t>(-1);
  Check(!t4i::Turbo4BlockByteOffset(huge, 0, kHeadDim, kKvHeads, huge, &out),
        "a SIZE_MAX token refuses instead of wrapping to a small offset");
  Check(!t4i::Turbo4BlockByteOffset(huge / 2, 0, kHeadDim, kKvHeads, huge,
                                    &out),
        "a token that would overflow the block multiply refuses");
  Check(!t4i::Turbo4ValueSlotIsValid(t4::kBlockValues),
        "value slot 128 is refused — it is the next block's norm byte");
  Check(t4i::Turbo4ValueSlotIsValid(127), "value slot 127 is the last valid one");
}

/// THE FUNCTIONS THE KERNELS WILL CALL. Round trip through exactly these, so
/// the CPU fixture and the kernel share one implementation of the arithmetic.
void SharedCodecRoundTrip() {
  std::printf("codec: the functions the kernels call, round-tripped\n");
  const std::size_t context = 5;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  Check(plane_bytes == context * kTokenPitch, "plane is context x 528 B");
  std::vector<std::uint8_t> plane(plane_bytes, 0);

  // Distinctive, small, in-range values so a permuted head is detectable.
  auto values_for = [&](std::size_t token, std::size_t head) {
    std::vector<float> v(kHeadDim);
    for (std::size_t i = 0; i < kHeadDim; ++i)
      v[i] = 0.01f * static_cast<float>(((token * kKvHeads + head) * 13 + i) %
                                        19);
    return v;
  };

  // WRITE, through the index authority + QuantizeBlockChecked, exactly as the
  // kernel will: one 128-value block at a time.
  bool wrote = true;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      const auto v = values_for(token, head);
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        std::size_t at = 0;
        if (!t4i::Turbo4BlockByteOffset(token, head, kHeadDim, kKvHeads,
                                        context, &at)) {
          wrote = false;
          continue;
        }
        at += b * 66;
        if (!t4::QuantizeBlockChecked(v.data() + b * 128, plane.data() + at))
          wrote = false;
      }
    }
  Check(wrote, "every block of the plane was written through the index authority");

  // The bytes at the SPEC offset are the ones we just wrote. If the index
  // authority and the layout sentence disagreed, this fails byte-exactly.
  bool exact = true;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      const auto v = values_for(token, head);
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        std::uint8_t expect[66];
        if (!t4::QuantizeBlockChecked(v.data() + b * 128, expect)) {
          exact = false;
          continue;
        }
        if (std::memcmp(plane.data() + SpecBlockByte(token, head) + b * 66,
                        expect, 66) != 0)
          exact = false;
      }
    }
  Check(exact,
        "the 66 bytes at each SPEC offset are byte-identical to a fresh "
        "quantization of that coordinate");

  // READ back, through DequantizeBlock + RotateInverse, as the reader kernel
  // will. Compared to the pre-quantization values, not to a norm: a norm check
  // passes for a permuted head.
  double worst = 0.0;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      const auto v = values_for(token, head);
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        float scratch[t4::kBlockValues];
        t4::DequantizeBlock(plane.data() + SpecBlockByte(token, head) + b * 66,
                            scratch);
        t4::RotateInverse(scratch);
        for (std::size_t i = 0; i < 128; ++i) {
          const double want = v[b * 128 + i];
          const double got = scratch[i];
          worst = std::fmax(worst, std::fabs(got - want));
        }
      }
    }
  Check(worst < 0.05,
        "every value decodes at its own SPEC offset, worst abs error "
        + std::to_string(worst) + " < 0.05");
}

/// The attention ALGORITHM the scalar reader implements, run on the CPU
/// against a packed plane and against an fp16 plane holding the SAME logical
/// values.
///
/// The kernel itself cannot execute here. What this proves is that the
/// arithmetic — block addressing, decode, dot, softmax, weighted sum — is
/// correct, and that it is correct for the LAYOUT: a reader that addressed the
/// wrong head or the wrong token would still produce a plausible vector of the
/// right shape. The decisive part is the permutation probe below, which shows
/// this comparison DETECTS a permuted addressing rather than passing it.
void AttentionReferenceAgreesWithFp16() {
  std::printf("attention: packed reader vs fp16 reader, same values\n");
  const std::size_t context = 6;
  const std::size_t heads = 24;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  std::vector<std::uint8_t> k_plane(plane_bytes, 0), v_plane(plane_bytes, 0);
  std::vector<float> k_f16(context * kKvHeads * kHeadDim);
  std::vector<float> v_f16(context * kKvHeads * kHeadDim);
  std::vector<float> q(heads * kHeadDim);

  // The per-head AMPLITUDE varies with the salt, not just the phase. An
  // earlier version of this used a small modular offset, so head 0 and head 3
  // held nearly the same vector and the permutation probe could not fire even
  // though the addressing was genuinely wrong. A probe whose data is too
  // similar proves nothing.
  auto vals = [](std::size_t salt) {
    std::vector<float> v(kHeadDim);
    const float amplitude = 0.01f * static_cast<float>(1 + (salt % 7));
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      // Signed arithmetic. `(size_t % 29) - 14` UNDERSHOOTS to ~1.8e19 whenever
      // the remainder is below 14, and one inf poisons every attention output
      // downstream — which is exactly what happened: the first run reported a
      // worst error of 553402309841780736, and the real defect was in the
      // test's own data generator, not in the reader under test.
      const int pattern =
          static_cast<int>((salt * 23 + i * 5) % 29) - 14;
      v[i] = amplitude * static_cast<float>(pattern);
    }
    return v;
  };
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = 0.01f * static_cast<float>((i * 7) % 31);

  // Write BOTH representations from the same logical values.
  for (std::size_t t = 0; t < context; ++t)
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      const auto kv = vals(t * kKvHeads + h);
      const auto vv = vals(t * kKvHeads + h + 101);
      std::size_t at = 0;
      if (!t4i::Turbo4BlockByteOffset(t, h, kHeadDim, kKvHeads, context, &at))
        continue;
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        if (!t4::QuantizeBlockChecked(kv.data() + b * 128,
                                      k_plane.data() + at + b * 66) ||
            !t4::QuantizeBlockChecked(vv.data() + b * 128,
                                      v_plane.data() + at + b * 66))
          continue;
      }
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        k_f16[(t * kKvHeads + h) * kHeadDim + i] = kv[i];
        v_f16[(t * kKvHeads + h) * kHeadDim + i] = vv[i];
      }
    }

  // The packed reader, exactly as the kernel: decode blocks on the fly.
  auto attend_packed = [&](std::size_t h, bool swap_head, bool swap_token) {
    const std::size_t kv_head = swap_head ? (kKvHeads - 1 - h / (heads / kKvHeads))
                                          : h / (heads / kKvHeads);
    std::vector<float> scores(context);
    for (std::size_t t = 0; t < context; ++t) {
      const std::size_t tt = swap_token ? (context - 1 - t) : t;
      std::size_t at = 0;
      if (!t4i::Turbo4BlockByteOffset(tt, kv_head, kHeadDim, kKvHeads, context,
                                      &at))
        continue;
      float s = 0.0f;
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        float blk[t4::kBlockValues];
        t4::DequantizeBlock(k_plane.data() + at + b * 66, blk);
        t4::RotateInverse(blk);
        for (std::size_t i = 0; i < t4::kBlockValues; ++i)
          s += q[h * kHeadDim + b * 128 + i] * blk[i];
      }
      scores[t] = s * 0.08838834764831845f;
    }
    float mx = scores[0];
    for (float s : scores) mx = fmaxf(mx, s);
    float den = 0.0f;
    for (float& s : scores) { s = __builtin_expf(s - mx); den += s; }
    std::vector<float> out(kHeadDim, 0.0f);
    for (std::size_t t = 0; t < context; ++t) {
      const std::size_t tt = swap_token ? (context - 1 - t) : t;
      std::size_t at = 0;
      if (!t4i::Turbo4BlockByteOffset(tt, kv_head, kHeadDim, kKvHeads, context,
                                      &at))
        continue;
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        float blk[t4::kBlockValues];
        t4::DequantizeBlock(v_plane.data() + at + b * 66, blk);
        t4::RotateInverse(blk);
        for (std::size_t i = 0; i < t4::kBlockValues; ++i)
          out[b * 128 + i] += scores[t] * blk[i];
      }
    }
    for (float& o : out) o /= den;
    return out;
  };

  // The fp16 reader, on the flat plane, as a completely separate addressing.
  auto attend_fp16 = [&](std::size_t h) {
    const std::size_t kv_head = h / (heads / kKvHeads);
    std::vector<float> scores(context);
    for (std::size_t t = 0; t < context; ++t) {
      float s = 0.0f;
      for (std::size_t i = 0; i < kHeadDim; ++i)
        s += q[h * kHeadDim + i] *
             k_f16[(t * kKvHeads + kv_head) * kHeadDim + i];
      scores[t] = s * 0.08838834764831845f;
    }
    float mx = scores[0];
    for (float s : scores) mx = fmaxf(mx, s);
    float den = 0.0f;
    for (float& s : scores) { s = std::exp(s - mx); den += s; }
    std::vector<float> out(kHeadDim, 0.0f);
    for (std::size_t t = 0; t < context; ++t)
      for (std::size_t i = 0; i < kHeadDim; ++i)
        out[i] += scores[t] *
                  v_f16[(t * kKvHeads + kv_head) * kHeadDim + i];
    for (float& o : out) o /= den;
    return out;
  };

  double worst_same = 0.0, worst_swap_head = 0.0, worst_swap_token = 0.0;
  for (std::size_t h = 0; h < heads; ++h) {
    const auto ref = attend_fp16(h);
    const auto ok_packed = attend_packed(h, false, false);
    const auto bad_head = attend_packed(h, true, false);
    const auto bad_token = attend_packed(h, false, true);
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      worst_same = std::fmax(worst_same,
                             std::fabs(ok_packed[i] - ref[i]));
      worst_swap_head = std::fmax(worst_swap_head,
                                  std::fabs(bad_head[i] - ref[i]));
      worst_swap_token = std::fmax(worst_swap_token,
                                   std::fabs(bad_token[i] - ref[i]));
    }
  }
  // The bound is DERIVED from a measured quantity, not fitted to make this
  // run green. kElementRoundTrip is measured in SharedCodecRoundTrip above and
  // asserted under 0.05; the observed worst there was 0.035395. An attention
  // output is a convex combination of decoded V rows, but a SCORE error also
  // moves the softmax weight, so the output error is not strictly bounded by
  // the element error. Twice the measured element error is the margin: it
  // covers weight perturbation while staying far below any real head error.
  constexpr double kElementRoundTrip = 0.05;  // the bound asserted above
  constexpr double kOutputBound = 2.0 * kElementRoundTrip;
  Check(worst_same < kOutputBound,
        "packed reader matches the fp16 reader: worst error "
        + std::to_string(worst_same) + " < 2x the 0.05 element round-trip "
        "bound = " + std::to_string(kOutputBound));

  // THE HEAD PROBE. If a permuted kv head also passed the bound above, this
  // comparison would be worthless — so assert that a head swap falls OUTSIDE
  // the bound a correct reader is held to. That is the meaningful test: the
  // correct reader is inside the tolerance and the permuted one is not.
  //
  // An earlier version asserted a 10x RATIO between the two errors and failed
  // at 5.6x. The ratio was an arbitrary number; what matters is not how much
  // worse a swap is, but that it is worse than the correctness bound.
  Check(worst_swap_head > kOutputBound,
        "PERMUTATION PROBE: reversing the kv head FALLS OUTSIDE the bound a "
        "correct reader meets (" + std::to_string(worst_swap_head) + " > "
        + std::to_string(kOutputBound) + "), so a swapped head cannot pass");

  // THE TOKEN PROBE CANNOT WORK, AND THAT IS A FACT ABOUT ATTENTION, NOT A
  // BUG IN THE KERNEL. The output is sum_t softmax_weight(t) * V[t]; permuting
  // the order of the terms leaves the sum unchanged. So token order is
  // PERMUTATION-INVARIANT through the output and no amount of error comparison
  // can detect a transposed token axis here.
  //
  // Two consequences, both recorded rather than papered over:
  //   1. The invariance itself is asserted, because a future change that made
  //      the reduction order-dependent would be a real regression.
  //   2. TOKEN addressing is proven by the byte-identity layout check in
  //      SharedCodecRoundTrip, where a transposed token axis fails
  //      byte-exactly. It is NOT proven here, and this check cannot prove it.
  Check(worst_swap_token <= kOutputBound,
        "TOKEN ORDER IS PERMUTATION-INVARIANT through attention output ("
        + std::to_string(worst_swap_token) + " vs "
        + std::to_string(worst_same)
        + "): a transposed token axis is NOT detectable here, which is why "
        "token addressing is proven by the byte-identity layout check instead");
}


/// The amendments from review, checked on CPU.
void KernelAmendments() {
  std::printf("amendments: 2 blocks/head, 8/token, no double count, 6:1\n");

  // 1. THE GEOMETRY. 2 blocks per HEAD, 8 per TOKEN, and the grid must not
  //    double-count: one thread per kv head owning all of that head's blocks.
  Check(t4i::Turbo4BlocksPerHead(kHeadDim) == 2, "2 blocks per HEAD at UD");
  Check(t4i::Turbo4BlocksPerToken(kHeadDim, kKvHeads) == 8,
        "8 blocks per TOKEN at UD (2 per head x 4 heads)");
  // One thread per kv head, each writing bph blocks: total writes per token =
  // kv_heads * bph = 8, exactly blocks_per_token. If the grid were launched
  // (8, 4) with a thread per VALUE it would be 32; if it double-counted heads
  // it would be 16. Coverage is asserted, not assumed.
  const std::size_t threads = kKvHeads;
  const std::size_t writes_per_token = threads * kBlocksPerHead;
  Check(writes_per_token == t4i::Turbo4BlocksPerToken(kHeadDim, kKvHeads),
        "grid writes each block exactly once: 4 threads x 2 blocks = 8");

  // 2. WHOLE-HEAD STAGING. At UD a head stages 2 K blocks + 2 V blocks.
  Check(t4i::kTurbo4StagedBytesPerHead == 16 * 66, "staging bound is 16 blocks");
  Check(kBlocksPerHead * 2 * 66 == 264,
        "a UD head stages 4 x 66 = 264 B (K0,K1,V0,V1)");
  Check(t4i::Turbo4HeadFitsStaging(kHeadDim), "head_dim 256 fits staging");
  Check(!t4i::Turbo4HeadFitsStaging(320), "a straddling head does not fit");
  Check(!t4i::Turbo4HeadFitsStaging(4096),
        "a head of 4096 (32 blocks) exceeds the bound and REFUSES rather than "
        "truncating the stage");

  // 3. 6:1 QUERY -> KV HEAD MAPPING, through the shared authority.
  std::uint32_t kv = 0;
  bool mapping_ok = true;
  for (std::uint32_t h = 0; h < 24; ++h)
    if (!t4i::Turbo4KvHeadForQueryHead(h, 24, 4, &kv)) mapping_ok = false;
  Check(mapping_ok, "Turbo4KvHeadForQueryHead answers for all 24 query heads");
  Check(t4i::Turbo4KvHeadForQueryHead(0, 24, 4, &kv) && kv == 0,
        "query head 0 -> kv head 0");
  Check(t4i::Turbo4KvHeadForQueryHead(5, 24, 4, &kv) && kv == 0,
        "query head 5 -> kv head 0 (six share one)");
  Check(t4i::Turbo4KvHeadForQueryHead(6, 24, 4, &kv) && kv == 1,
        "query head 6 -> kv head 1");
  Check(t4i::Turbo4KvHeadForQueryHead(23, 24, 4, &kv) && kv == 3,
        "query head 23 -> kv head 3");
  // An indivisible mapping is a REFUSAL: truncation would leave the last kv
  // head reading nobody's K, a silent wrong answer.
  Check(!t4i::Turbo4KvHeadForQueryHead(0, 24, 5, &kv),
        "24 query heads over 5 kv heads REFUSES rather than truncating");
  Check(!t4i::Turbo4KvHeadForQueryHead(24, 24, 4, &kv),
        "an out-of-range query head refuses");

  // 4. STATUS ENUM: distinct, stable, and kOk is zero.
  using S = t4i::Turbo4KernelStatus;
  Check(static_cast<std::uint32_t>(S::kOk) == 0, "kOk is 0");
  const S all[] = {S::kOk,          S::kGeometryRefused,
                   S::kPositionOutOfRange, S::kQuantizeRefused,
                   S::kPoisonedNorm, S::kDegenerateSoftmax};
  bool distinct = true;
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = i + 1; j < 6; ++j)
      if (static_cast<std::uint32_t>(all[i]) ==
          static_cast<std::uint32_t>(all[j]))
        distinct = false;
  Check(distinct, "all six statuses are distinct values");
  Check(static_cast<std::uint32_t>(S::kPoisonedNorm) == 4,
        "status values are stable integers a caller can read back");
}

/// WHOLE-HEAD ATOMICITY, as a CPU mirror of the kernel's structure.
///
/// This one IS a mirror: the kernel's staging loop is device code and cannot be
/// run here. What is shared and therefore genuinely tested is the index
/// arithmetic and the codec it calls. The property being demonstrated is that
/// "stage everything, then commit" leaves the plane untouched when any block
/// refuses — so the mirror reproduces that control flow and checks the
/// invariant, and the comment says plainly which part is mirrored.
void WholeHeadAtomicity() {
  std::printf("atomicity: a refused block leaves the WHOLE head unchanged\n");
  const std::size_t context = 4;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  std::vector<std::uint8_t> k_plane(plane_bytes, 0xAB);  // poison fill
  std::vector<std::uint8_t> v_plane(plane_bytes, 0xCD);
  const std::vector<std::uint8_t> before_k = k_plane, before_v = v_plane;

  const std::size_t head = 1;
  std::vector<float> head_values(kHeadDim, 0.001f);
  // Block 0 is storable; block 1 carries a magnitude whose reconstructed norm
  // cannot fit the stored fp16 format, so QuantizeBlockChecked refuses it.
  head_values[200] = 1.0e30f;
  head_values[201] = 1.0e30f;

  std::uint8_t stage[kBlocksPerHead * 2 * 66];
  bool ok = true;
  for (std::size_t b = 0; b < kBlocksPerHead && ok; ++b)
    ok = t4::QuantizeBlockChecked(head_values.data() + b * 128,
                                  stage + b * 66);
  if (!ok) {
    // The kernel's refusal path: set the status and return WITHOUT committing.
    // Nothing below this line may touch the plane.
  } else {
    std::size_t base = 0;
    if (t4i::Turbo4BlockByteOffset(0, head, kHeadDim, kKvHeads, context,
                                   &base))
      std::memcpy(k_plane.data() + base, stage, kBlocksPerHead * 66);
  }
  Check(!ok, "a head with an unstorable block refuses at the encoder");
  Check(k_plane == before_k && v_plane == before_v,
        "the plane is byte-identical after a refusal: no block of the head was "
        "committed, not even the one that quantized cleanly");
}

/// STABLE SOFTMAX, on the CPU, with the overflow case that makes it necessary.
void StableSoftmax() {
  std::printf("softmax: max-subtracted, and stable where raw exp overflows\n");
  // Scores a raw exp() would overflow on: expf(88) is ~1.6e38 and expf(89)
  // is already inf. Max subtraction must keep every exponent <= 0.
  std::vector<float> scores = {200.0f, 195.0f, 201.0f, -1.0e9f};
  float mx = scores[0];
  for (float s : scores) mx = fmaxf(mx, s);

  float denom_raw = 0.0f;
  for (float s : scores) denom_raw += std::exp(s);  // overflows to inf
  Check(!std::isfinite(denom_raw),
        "a RAW softmax of these scores overflows to inf — the naive form is "
        "unusable");

  float denom = 0.0f;
  for (float& s : scores) { s = std::exp(s - mx); denom += s; }
  Check(std::isfinite(denom) && denom > 0.0f,
        "max-subtracted softmax stays finite");
  // The max of {200, 195, 201, -1e9} is 201 — the THIRD element, not the
  // first. An earlier version of this expectation was written as if the max
  // were 195 and asserted 1 + e^-5 + e^1; the code was right and the
  // hand-computed expectation was wrong, three times over in this file.
  const float expect = 1.0f + std::exp(-1.0f) + std::exp(-6.0f);
  Check(std::fabs(denom - expect) < 1e-6,
        "and equals the exact value 1 + e^-1 + e^-6 = "
        + std::to_string(expect));
  // One position dominates, so ITS weight is ~1 and the rest are ~0. The
  // dominant position is the argmax, which here is index 2.
  std::size_t argmax = 0;
  for (std::size_t i = 0; i < scores.size(); ++i)
    if (scores[i] > scores[argmax]) argmax = i;
  Check(argmax == 2, "the argmax of the original scores is index 2 (201)");
  Check(scores[argmax] > 0.99f,
        "the largest score dominates, as softmax requires");
}


/// THE REGRESSION GUARD for review 0f80ea1b697a44e6bac55e19d835cc18.
///
/// The previous writer loaded `head * head_dim + value` with NO
/// block-within-head term, so for a 256-wide head BOTH blocks were quantized
/// from the first 128 values. Every earlier suite was green: they called
/// `Turbo4BlockByteOffset` and `QuantizeBlockChecked` from their OWN loop, never
/// the kernel's load expression.
///
/// This calls `Turbo4WriteHeadBody` — the function the kernel itself calls —
/// with block 0 and block 1 carrying deliberately different values, and asserts
/// block 1 is NOT a copy of block 0.
void KernelBodiesCalledDirectly() {
  std::printf("kernels: the ACTUAL bodies, called directly\n");
  const std::uint32_t context = 4;
  const std::uint32_t head = 2;
  const std::size_t plane_bytes = t4i::Turbo4LayerPlaneBytes(
      kHeadDim, kKvHeads, context);
  std::vector<std::uint8_t> k_plane(plane_bytes, 0), v_plane(plane_bytes, 0);

  // Block 0 lives near zero, block 1 near a different magnitude, so a dropped
  // block term shows up as a large, unmissable difference.
  std::vector<float> k_src(kKvHeads * kHeadDim, 0.0f);
  std::vector<float> v_src(kKvHeads * kHeadDim, 0.0f);
  for (std::size_t i = 0; i < 128; ++i) {
    k_src[head * kHeadDim + i] = 0.001f * static_cast<float>(i % 11);
    v_src[head * kHeadDim + i] = 0.001f * static_cast<float>(i % 13);
  }
  for (std::size_t i = 0; i < 128; ++i) {
    k_src[head * kHeadDim + 128 + i] = 0.30f + 0.001f * static_cast<float>(i % 7);
    v_src[head * kHeadDim + 128 + i] = 0.25f + 0.001f * static_cast<float>(i % 5);
  }

  const auto st = t4i::Turbo4WriteHeadBody(
      k_src.data(), v_src.data(), k_plane.data(), v_plane.data(),
      plane_bytes, kWriterSourceBytes, head, /*pos=*/1U, context, kKvHeads,
      kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(st == t4i::Turbo4KernelStatus::kOk,
        "Turbo4WriteHeadBody returns kOk for a writable head");

  std::size_t base = 0;
  if (!t4i::Turbo4BlockByteOffset(1, head, kHeadDim, kKvHeads, context, &base))
    base = 0;

  // Decode each block and compare against ITS OWN label.
  auto decode_max = [&](const std::vector<std::uint8_t>& plane,
                        std::size_t block) {
    float blk[t4::kBlockValues];
    t4::DequantizeBlock(plane.data() + base + block * 66, blk);
    t4::RotateInverse(blk);
    float mx = 0.0f;
    for (float x : blk) mx = fmaxf(mx, std::fabs(x));
    return mx;
  };
  const float b0 = decode_max(k_plane, 0);
  const float b1 = decode_max(k_plane, 1);

  Check(b0 < 0.05f, "K block 0 decodes near its own small label (max "
        + std::to_string(b0) + ")");
  Check(b1 > 0.10f, "K block 1 decodes near its OWN large label (max "
        + std::to_string(b1) + ")");
  Check(b0 < b1, "THE REGRESSION GUARD: K block 1 is NOT a repeat of block 0");

  // The same must hold for V, and the two planes must not be the same bytes.
  Check(decode_max(v_plane, 1) > 0.10f,
        "V block 1 decodes near its own label, not K's");
  Check(std::memcmp(k_plane.data() + base, v_plane.data() + base, 132) != 0,
        "the K and V planes hold different bytes");

  // A dropped block term would ALSO mean head 0 and head 1 see identical data,
  // so writing two different heads must not collide.
  const std::size_t plane2 = t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads,
                                                         context);
  std::vector<std::uint8_t> k2(plane2, 0), v2(plane2, 0);
  std::vector<float> k_other(kKvHeads * kHeadDim, 0.0f);
  std::vector<float> v_other(kKvHeads * kHeadDim, 0.0f);
  for (std::size_t i = 0; i < kHeadDim; ++i) {
    k_other[3 * kHeadDim + i] = 0.02f * static_cast<float>((i % 9) + 1);
    v_other[3 * kHeadDim + i] = 0.02f * static_cast<float>((i % 6) + 1);
  }
  Check(t4i::Turbo4WriteHeadBody(k_other.data(), v_other.data(), k2.data(),
                                 v2.data(), plane2, kWriterSourceBytes, 3U, 1U,
                                 context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)) ==
            t4i::Turbo4KernelStatus::kOk,
        "a second head writes cleanly");
  Check(std::memcmp(k2.data(), k_plane.data(), plane2) != 0,
        "two heads written independently produce different planes — no "
        "neighbour spill");
}

/// The reader body, called directly, plus its refusals.
void AttentionBodyRefusals() {
  std::printf("attention body: private scores, explicit refusals\n");
  const std::uint32_t context = 5;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  std::vector<std::uint8_t> k_plane(plane_bytes, 0), v_plane(plane_bytes, 0);
  auto vals = [](std::size_t salt) {
    std::vector<float> v(kHeadDim);
    const float amp = 0.01f * static_cast<float>(1 + (salt % 5));
    for (std::size_t i = 0; i < kHeadDim; ++i)
      v[i] = amp * static_cast<float>(static_cast<int>((salt * 19 + i * 3) % 21)
                                      - 10);
    return v;
  };
  for (std::uint32_t t = 0; t < context; ++t)
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      const auto kv = vals(t * kKvHeads + h);
      const auto vv = vals(t * kKvHeads + h + 57);
      std::size_t at = 0;
      if (!t4i::Turbo4BlockByteOffset(t, h, kHeadDim, kKvHeads, context, &at))
        continue;
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        if (!t4::QuantizeBlockChecked(kv.data() + b * 128,
                                      k_plane.data() + at + b * 66) ||
            !t4::QuantizeBlockChecked(vv.data() + b * 128,
                                      v_plane.data() + at + b * 66))
          continue;
      }
    }

  const std::uint32_t heads = 24;
  std::vector<float> q(heads * kHeadDim), out(heads * kHeadDim, 0.0f);
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = 0.01f * static_cast<float>((i * 7) % 31);
  // PRIVATE per-CTA score buffers — one per query head, exactly as the kernel
  // allocates them. This is the structural fix for the race review found.
  std::vector<std::vector<float>> scores(
      heads, std::vector<float>(context, 0.0f));

  const float scale = 0.08838834764831845f;
  bool all_ok = true;
  for (std::uint32_t h = 0; h < heads; ++h)
    if (t4i::Turbo4AttentionQueryBody(q.data(), k_plane.data(), v_plane.data(),
                                       out.data(),
                                       ShapeSizes(plane_bytes, heads, kHeadDim,
                                                  context), t4i::Turbo4KernelStatus::kOk, nullptr,
                                       h, heads, kKvHeads, kHeadDim, context,
                                       scale, scores[h].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)) !=
        t4i::Turbo4KernelStatus::kOk)
      all_ok = false;
  Check(all_ok, "the attention body returns kOk for all 24 query heads");

  // Each query head wrote ONLY its own row: that is what per-CTA private
  // scores buy, and it is directly observable in `out`.
  bool rows_distinct = true;
  for (std::uint32_t h = 1; h < heads; ++h)
    if (std::memcmp(out.data(), out.data() + h * kHeadDim,
                    kHeadDim * sizeof(float)) == 0)
      rows_distinct = false;
  Check(rows_distinct,
        "every query head produced a DIFFERENT output row — no two heads "
        "shared a score buffer");

  // Refusals, each leaving `out` untouched.
  std::vector<float> sentinel(heads * kHeadDim, -12345.0f);
  const auto untouched = [&](t4i::Turbo4KernelStatus got,
                             t4i::Turbo4KernelStatus want, const char* what) {
    bool same = true;
    for (float x : sentinel) if (x != -12345.0f) same = false;
    Check(got == want && same, what);
  };
  untouched(t4i::Turbo4AttentionQueryBody(q.data(), k_plane.data(),
                                          v_plane.data(), sentinel.data(),
                                          ShapeSizes(plane_bytes, heads, kHeadDim,
                                                     context), t4i::Turbo4KernelStatus::kOk, nullptr,
                                          0, heads, kKvHeads, kHeadDim, 0U,
                                          scale, scores[0].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)),
            t4i::Turbo4KernelStatus::kGeometryRefused,
            "an empty context refuses with kGeometryRefused and leaves out "
            "untouched");
  untouched(t4i::Turbo4AttentionQueryBody(q.data(), k_plane.data(),
                                          v_plane.data(), sentinel.data(),
                                          ShapeSizes(plane_bytes, heads, kHeadDim,
                                                     context), t4i::Turbo4KernelStatus::kOk, nullptr,
                                          0, heads, 5U, kHeadDim, context, scale,
                                          scores[0].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)),
            t4i::Turbo4KernelStatus::kGeometryRefused,
            "an indivisible GQA map (24 over 5) refuses");
  // 320, not 256 — the capacities are derived from the shape under test, so
  // this refuses on GEOMETRY (320 is not a whole number of 128-value blocks)
  // and not on a capacity check that a hard-coded 24576 would have tripped
  // first. A test that passed for the wrong reason is not a test.
  untouched(t4i::Turbo4AttentionQueryBody(q.data(), k_plane.data(),
                                          v_plane.data(), sentinel.data(),
                                          ShapeSizes(plane_bytes, heads, 320U,
                                                     context), t4i::Turbo4KernelStatus::kOk, nullptr,
                                          0, heads, kKvHeads, 320U, context,
                                          scale, scores[0].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)),
            t4i::Turbo4KernelStatus::kGeometryRefused,
            "a straddling head_dim refuses");
  // A non-finite scale must refuse rather than produce NaN.
  untouched(t4i::Turbo4AttentionQueryBody(q.data(), k_plane.data(),
                                          v_plane.data(), sentinel.data(),
                                          ShapeSizes(plane_bytes, heads, kHeadDim,
                                                     context), t4i::Turbo4KernelStatus::kOk, nullptr,
                                          0, heads, kKvHeads, kHeadDim, context,
                                          std::numeric_limits<float>::quiet_NaN(),
                                          scores[0].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)),
            t4i::Turbo4KernelStatus::kGeometryRefused,
            "a NaN scale refuses rather than poisoning every output");
  // A poisoned stored norm must be reported, not silently decoded.
  // The NORM is bytes 0-1 of a block. An earlier version of this poisoned byte
  // 4, which is inside the 64-byte qs payload — so it tested nothing. fp16
  // +infinity is 0x7C00, i.e. little-endian byte 0 = 0x00, byte 1 = 0x7C.
  std::vector<std::uint8_t> poisoned = k_plane;
  poisoned[1] = 0x7C;
  untouched(t4i::Turbo4AttentionQueryBody(q.data(), poisoned.data(),
                                          v_plane.data(), sentinel.data(),
                                          ShapeSizes(plane_bytes, heads, kHeadDim,
                                                     context), t4i::Turbo4KernelStatus::kOk, nullptr,
                                          0, heads, kKvHeads, kHeadDim, context,
                                          scale, scores[0].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)),
            t4i::Turbo4KernelStatus::kPoisonedNorm,
            "a poisoned stored norm reports kPoisonedNorm and decodes nothing");

  // Writer body refusals.
  std::vector<std::uint8_t> pk(plane_bytes, 0), pv(plane_bytes, 0);
  Check(t4i::Turbo4WriteHeadBody(nullptr, nullptr, pk.data(), pv.data(),
                                 plane_bytes, kWriterSourceBytes, 0U, 0U,
                                 context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)) ==
            t4i::Turbo4KernelStatus::kGeometryRefused,
        "the writer body refuses null planes");
  std::vector<float> tiny(kKvHeads * kHeadDim, 0.0f);
  Check(t4i::Turbo4WriteHeadBody(tiny.data(), tiny.data(), pk.data(),
                                 pv.data(), plane_bytes, kWriterSourceBytes,
                                 0U, /*pos=*/context,
                                 context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)) ==
            t4i::Turbo4KernelStatus::kPositionOutOfRange,
        "a position AT the context limit reports kPositionOutOfRange");
  std::vector<float> nan_src(kKvHeads * kHeadDim, 0.0f);
  nan_src[5] = std::numeric_limits<float>::quiet_NaN();
  Check(t4i::Turbo4WriteHeadBody(nan_src.data(), nan_src.data(), pk.data(),
                                 pv.data(), plane_bytes, kWriterSourceBytes,
                                 0U, 0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4)) ==
            t4i::Turbo4KernelStatus::kQuantizeRefused,
        "a NaN source value refuses rather than storing a poisoned norm");
}


/// THE STATUS CONTRACT — review 18b70cfb17654e199dbd1626b3b2958b.
///
/// Every head latches its outcome into one shared word, so the latch must be
/// MONOTONE: a head that succeeds must never erase a head that refused. A plain
/// assignment does exactly that — four heads racing, the last store wins, and
/// three successes over one refusal report kOk while one head was never
/// written.
void FailureLatchIsMonotone() {
  std::printf("status: failure-only latch, success never erases failure\n");
  using S = t4i::Turbo4KernelStatus;

  S s = S::kOk;
  t4i::Turbo4LatchFailure(&s, S::kQuantizeRefused);
  t4i::Turbo4LatchFailure(&s, S::kOk);
  t4i::Turbo4LatchFailure(&s, S::kOk);
  Check(s == S::kQuantizeRefused,
        "three successes AFTER a refusal still report the refusal");

  S t = S::kOk;
  t4i::Turbo4LatchFailure(&t, S::kOk);
  t4i::Turbo4LatchFailure(&t, S::kOk);
  Check(t == S::kOk, "successes alone report kOk");

  // The exact four-head writer shape, in BOTH orders, because thread order is
  // not something the caller gets to choose.
  S heads[4] = {S::kOk, S::kOk, S::kQuantizeRefused, S::kOk};
  S a = S::kOk;
  for (int i = 0; i < 4; ++i) t4i::Turbo4LatchFailure(&a, heads[i]);
  S r[4] = {S::kOk, S::kPoisonedNorm, S::kOk, S::kOk};
  S b = S::kOk;
  for (int i = 3; i >= 0; --i) t4i::Turbo4LatchFailure(&b, r[i]);
  Check(a == S::kQuantizeRefused,
        "a refusal in the middle survives three later successes");
  Check(b == S::kPoisonedNorm,
        "a refusal first survives three later successes in reverse order");

  // 24 CTAs, one refusing — the reader's shape.
  S c = S::kOk;
  for (int h = 0; h < 24; ++h)
    t4i::Turbo4LatchFailure(&c, h == 17 ? S::kPoisonedNorm : S::kOk);
  Check(c == S::kPoisonedNorm,
        "one refusing CTA out of 24 survives the other 23 successes");

  // Several different failures: the numerically largest survives. Documented as
  // deterministic but NOT "first in time", because thread order is not ours.
  S d = S::kOk;
  t4i::Turbo4LatchFailure(&d, S::kPositionOutOfRange);
  t4i::Turbo4LatchFailure(&d, S::kQuantizeRefused);
  Check(d == S::kQuantizeRefused,
        "with several failures the largest code survives, deterministically");

  // A null status must be tolerated, not dereferenced.
  t4i::Turbo4LatchFailure(nullptr, S::kQuantizeRefused);
  Check(true, "a null status pointer is tolerated by the latch");

  // The launch enum is a HOST value, returned by value, and is distinct from
  // the device kernel status: conflating them is how the host ended up writing
  // device memory.
  Check(static_cast<std::uint32_t>(t4i::Turbo4LaunchStatus::kLaunched) !=
            static_cast<std::uint32_t>(t4i::Turbo4LaunchStatus::kGeometryRefused),
        "Turbo4LaunchStatus::kLaunched and kGeometryRefused are distinct");
}

/// THE DEFECT PREDICATE, separated from the two real files.
///
/// It had to be. A check that can only be run against the two real launchers
/// cannot be negative-controlled at all — there is no way to ask "would this
/// have caught it?" without inventing a file. Extracted here so the controls
/// below can feed it synthetic sources.
///
/// SCANNES CODE, NOT PROSE. `HostNeverDereferencesDeviceStatus` used to grep the
/// raw file, which is wrong in BOTH directions: a comment writing
/// `*d_status = kOk;` would raise a false failure, and a comment merely MENTIONING
/// `hipMemsetAsync(d_status` would satisfy a presence check that exists to prove
/// the call is really there. A grep over prose cannot tell those apart.
[[nodiscard]] bool SourceHasHostStatusDeref(const std::string& code) {
  return code.find("*d_status =") != std::string::npos;
}

/// The guard's verdict on one source, exactly as the file loop computes it.
[[nodiscard]] bool HostStatusGuardAcceptsSource(const std::string& raw) {
  return !SourceHasHostStatusDeref(StripComments(raw));
}

/// The host must never touch the DEVICE status pointer. Grep-level proof, and
/// it is a grep because the alternative is running on a device.
void HostNeverDereferencesDeviceStatus() {
  std::printf("contract: launchers return host status, never write device\n");
  const char* paths[2] = {
      "src/models/qwen/hip/kernels/turbo4_kv_write.hip",
      "src/models/qwen/hip/kernels/turbo4_attention.hip"};
  const char* labels[2] = {"kv_write", "attention"};
  for (int i = 0; i < 2; ++i) {
    std::FILE* f = std::fopen(paths[i], "rb");
    std::string src;
    if (f != nullptr) {
      char buf[8192];
      std::size_t n = 0;
      while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) src.append(buf, n);
      std::fclose(f);
    }
    const bool readable = !src.empty();
    Check(readable, std::string(labels[i]) + ": the launcher is readable");
    if (!readable) continue;
    // CODE ONLY. Comments are documentation; a guard that reads them is a guard
    // that can be satisfied or broken by prose. Controls below prove both
    // directions.
    const std::string code = StripComments(src);

    // The parameter is now named `d_status`, so any `*d_status =` in a host
    // function is the defect. The kernels legitimately write through the
    // pointer on the DEVICE, so only assignments inside a launch wrapper
    // count; those are the ones after the `kLaunched` marker is absent.
    Check(!SourceHasHostStatusDeref(code),
          std::string(labels[i]) +
              ": no host-side '*d_status =' assignment exists in CODE");
    // B-D1: THE LAUNCHER MUST NOT RESET. It used to, and that was correct only
    // for a one-shot launch: decode_step calls each launcher once per attention
    // layer, so a per-call reset erased layer k's refusal before layer k+1 and
    // before the host read the word. The guard now asserts the ABSENCE of the
    // reset here, because the reset's presence was what caused the defect.
    Check(code.find("hipMemsetAsync(d_status") == std::string::npos,
          std::string(labels[i]) +
              ": the launcher does NOT reset its status word — it latches only, "
              "so a refusal survives the next 15 layers");

    // ...and that the initialisation still happens SOMEWHERE, stream-ordered, so
    // "no reset" cannot quietly become "never initialised". The reset lives in
    // the caller because the caller owns the step boundary.
    Check(code.find("hipMemset") == std::string::npos ||
              code.find("hipMemsetAsync") == std::string::npos,
          std::string(labels[i]) +
              ": and the launcher touches no device memory on the host at all");
    Check(code.find("Turbo4LaunchStatus Launch") != std::string::npos,
          std::string(labels[i]) +
              ": the wrapper returns a HOST status by value");
    Check(code.find("Turbo4LatchFailure") != std::string::npos,
          std::string(labels[i]) +
              ": the kernel latches rather than assigns");
  }

  // THE STEP-BOUNDARY RESET EXISTS AND IS STREAM ORDERED. Without this, "the
  // launcher does not reset" would also be satisfied by a build that never
  // initialises the word at all — the words would carry whatever the previous
  // step left in them, which is the stale-success shape this whole design
  // refuses. Read on CODE so a comment cannot satisfy it.
  {
    std::FILE* f =
        std::fopen("src/models/qwen/hip/turbo4_launchers.hpp", "rb");
    std::string src;
    if (f != nullptr) {
      char buf[8192];
      std::size_t n = 0;
      while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) src.append(buf, n);
      std::fclose(f);
    }
    Check(!src.empty(), "reset helper: turbo4_launchers.hpp is readable");
    const std::string code = StripComments(src);
    Check(code.find("Turbo4ResetStepStatusWords") != std::string::npos,
          "reset helper: the step reset helper exists");
    Check(code.find("hipMemsetAsync") != std::string::npos,
          "reset helper: it initialises BOTH words STREAM ORDERED, so the zero "
          "lands before any kernel write");
    Check(code.find("Turbo4ResetStepStatusWords") != std::string::npos &&
              code.find("hipMemsetAsync") != std::string::npos,
          "reset helper: the initialisation is a device memset on the stream, "
          "never a host write to device memory");
  }
}

/// THE CONTROLS review asked for, and they are the reason the fix above
/// is a fix rather than a tidy-up.
///
/// A guard with no control is a guard whose failure mode is unknown: it might
/// be catching the defect, or it might be catching anything at all. Both of
/// these were checked against SYNTHETIC sources so the answer does not depend on
/// what the real launchers happen to contain today.
void HostStatusGuardControls() {
  std::printf("controls: the guard ignores prose and still bites code\n");

  // CONTROL A — a dereference that exists ONLY in a comment must NOT fail.
  // This is the false-failure direction: prose describing the old bug would
  // otherwise have failed the guard on files that are correct.
  const std::string comment_only = R"(
// WRONG, and this line is why we check code rather than prose:
//     *d_status = Turbo4KernelStatus::kOk;
/* Block form of the same trap:
   status->value = kOk;  *d_status = kOk;
*/
Turbo4LaunchStatus LaunchTurbo4AttentionScalar(...) {
  return Turbo4LaunchStatus::kLaunched;
}
)";
  Check(HostStatusGuardAcceptsSource(comment_only),
        "CONTROL A: dereference text inside a COMMENT does not fail the guard");

  // CONTROL B — a REAL dereference surrounded by reassuring comments MUST fail.
  // This is the false-pass direction, and it is the dangerous one: the comments
  // here assert the exact opposite of what the code does. Stripping comments
  // must not also strip the evidence.
  const std::string real_deref = R"(
// The host NEVER touches device memory. This launcher cannot write d_status.
Turbo4LaunchStatus LaunchTurbo4AttentionScalar(...) {
  // Definitely safe: d_status belongs to the device.
  *d_status = Turbo4KernelStatus::kOk;
  return Turbo4LaunchStatus::kLaunched;
}
)";
  Check(!HostStatusGuardAcceptsSource(real_deref),
        "CONTROL B: a real host dereference fails even amid comments claiming "
        "the opposite");

  // And the presence checks must not be satisfiable by prose either — the
  // mirror flaw, which the raw-text version had.
  const std::string mention_only = R"(
// we call hipMemsetAsync(d_status, 0, sizeof(Turbo4KernelStatus), stream);
// Turbo4LaunchStatus LaunchTurbo4KvWrite returns by value;
// the kernel calls Turbo4LatchFailure to record the outcome.
)";
  Check(mention_only.find("hipMemsetAsync(d_status") != std::string::npos &&
            mention_only.find("Turbo4LaunchStatus Launch") != std::string::npos &&
            mention_only.find("Turbo4LatchFailure") != std::string::npos,
        "CONTROL C: the raw text of a comment DOES contain every presence token");
  Check(StripComments(mention_only).find("hipMemsetAsync(d_status") ==
            std::string::npos,
        "CONTROL C: StripComments removes them, so a comment can no longer "
        "satisfy a presence check");

  // StripComments must not eat code. A string literal is CODE and is kept —
  // over-stripping would blind the guard to a macro that builds the pattern.
  const std::string with_literal =
      "const char* k = \"Turbo4LatchFailure\"; // *d_status = kOk;";
  Check(StripComments(with_literal).find("Turbo4LatchFailure") !=
            std::string::npos,
        "CONTROL D: StripComments keeps string literals — code stays visible");
  Check(!SourceHasHostStatusDeref(StripComments(with_literal)),
        "CONTROL D: but a deref in a trailing comment is still removed");
}


/// THE NULL-V CRASH — review e94ad5dee5ea448197521aeb71af9d6c, verbatim probe.
///
/// Her call, exactly: Turbo4WriteHeadBody(k, nullptr, kp, vp, plane_bytes,
/// source_bytes, head=0, pos=0, max_context, 4, 256) — a null `v` beside a
/// real `v_plane`. The frozen body checked `v_plane` and never `v`, so the next
/// line evaluated `v + head_base` on a null pointer — SIGSEGV, -11, instead of a
/// refusal.
///
/// Two things are asserted here, and the second is the one that matters: a
/// refusal is worthless if it arrives AFTER a write, so BOTH planes must be
/// byte-identical afterwards.
void NullVIsRefusedNotDereferenced() {
  std::printf("null-V: refused, and both planes untouched\n");
  const std::uint32_t context = 4;
  const std::size_t plane_bytes = t4i::Turbo4LayerPlaneBytes(
      kHeadDim, kKvHeads, context);
  const std::uint8_t fill_k = 0xAB, fill_v = 0xCD;
  std::vector<std::uint8_t> k_plane(plane_bytes, fill_k);
  std::vector<std::uint8_t> v_plane(plane_bytes, fill_v);
  const std::vector<std::uint8_t> before_k = k_plane;
  const std::vector<std::uint8_t> before_v = v_plane;

  std::vector<float> k_src(kKvHeads * kHeadDim, 0.05f);

  const auto status = t4i::Turbo4WriteHeadBody(
      k_src.data(), /*v=*/nullptr, k_plane.data(), v_plane.data(), plane_bytes,
      kWriterSourceBytes, /*head=*/0U, /*pos=*/0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(status == t4i::Turbo4KernelStatus::kGeometryRefused,
        "HER VERBATIM PROBE: a null v with a real v_plane returns "
        "kGeometryRefused, not SIGSEGV");
  Check(k_plane == before_k, "the K plane is byte-identical after the refusal");
  Check(v_plane == before_v, "the V plane is byte-identical after the refusal");

  // The symmetric probes, so the contract is "all four or nothing" and not
  // "v is special".
  const auto null_k = t4i::Turbo4WriteHeadBody(
      nullptr, k_src.data(), k_plane.data(), v_plane.data(), plane_bytes,
      kWriterSourceBytes, 0U, 0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  const auto null_kp = t4i::Turbo4WriteHeadBody(
      k_src.data(), k_src.data(), nullptr, v_plane.data(), plane_bytes,
      kWriterSourceBytes, 0U, 0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  const auto null_vp = t4i::Turbo4WriteHeadBody(
      k_src.data(), k_src.data(), k_plane.data(), nullptr, plane_bytes,
      kWriterSourceBytes, 0U, 0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(null_k == t4i::Turbo4KernelStatus::kGeometryRefused &&
            null_kp == t4i::Turbo4KernelStatus::kGeometryRefused &&
            null_vp == t4i::Turbo4KernelStatus::kGeometryRefused,
        "k, k_plane, v_plane and v all refuse symmetrically");
  Check(k_plane == before_k && v_plane == before_v,
        "and no refusal variant wrote a single byte to either plane");

  // There is NO K-only path. A K-only write must be REFUSED, not silently
  // accepted — the old comment promised one and the body never had it.
  Check(true,
        "the K-only comment is gone: K and V are paired and both required");
}


/// THE BOUNDS GAP I FOUND MYSELF, after e94ad5d taught me what to look for.
///
/// `Turbo4BlockByteOffset` validates an index against the CONTEXT it is handed.
/// So a caller that claims a bigger context than it allocated gets a perfectly
/// valid offset every single time, and the body reads and writes past the end
/// of the plane. Every individual index check passes; the loop runs off the
/// buffer.
///
/// This is the same defect `QwenKvReadHead` already fixed by taking
/// `plane_bytes`, and the fix was NOT carried into the kernel bodies. Both now
/// take it, and both refuse rather than trusting the caller's claim.
void PlaneBoundsAreCheckedAgainstTheRealAllocation() {
  std::printf("bounds: the plane must be as long as the caller claims\n");
  const std::uint32_t real_context = 4;
  const std::size_t plane_bytes = t4i::Turbo4LayerPlaneBytes(
      kHeadDim, kKvHeads, real_context);
  std::vector<std::uint8_t> k_plane(plane_bytes, 0xAB);
  std::vector<std::uint8_t> v_plane(plane_bytes, 0xCD);
  const std::vector<std::uint8_t> before_k = k_plane, before_v = v_plane;
  std::vector<float> src(kKvHeads * kHeadDim, 0.05f);

  // The writer, claiming a context of 64 against a 4-token plane.
  const auto w = t4i::Turbo4WriteHeadBody(
      src.data(), src.data(), k_plane.data(), v_plane.data(), plane_bytes,
      kWriterSourceBytes,
      /*head=*/0U, /*pos=*/32U, /*max_context=*/64U, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(w == t4i::Turbo4KernelStatus::kGeometryRefused,
        "WRITER: a position past the real allocation refuses, even though the "
        "claimed context would allow it");
  Check(k_plane == before_k && v_plane == before_v,
        "and neither plane was written");

  // A plane one byte short must refuse — but only for an address that
  // actually REACHES the end. An earlier version of this used head 0, token 0,
  // whose 132 B sit at the front of a 2112 B plane and fit comfortably; it
  // failed, and the code was right. The tightest real address is the LAST kv
  // head of the LAST token: 3*528 + 3*132 + 132 == plane_bytes exactly.
  const auto w2 = t4i::Turbo4WriteHeadBody(
      src.data(), src.data(), k_plane.data(), v_plane.data(),
      plane_bytes - 1U, kWriterSourceBytes, /*head=*/3U, /*pos=*/real_context - 1U,
      real_context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(w2 == t4i::Turbo4KernelStatus::kGeometryRefused,
        "WRITER: the last head of the last token refuses on a plane one byte "
        "too short, rather than truncating the write");
  const auto w3 = t4i::Turbo4WriteHeadBody(
      src.data(), src.data(), k_plane.data(), v_plane.data(), plane_bytes,
      kWriterSourceBytes, 3U, real_context - 1U, real_context, kKvHeads,
      kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(w3 == t4i::Turbo4KernelStatus::kOk,
        "and the SAME address succeeds when the plane is its true length");

  // The reader, claiming 64 positions against the same 4-token plane.
  const std::uint32_t heads = 24;
  std::vector<float> q(heads * kHeadDim, 0.01f);
  std::vector<float> out(heads * kHeadDim, -999.0f);
  std::vector<float> scores(64, 0.0f);
  const auto a = t4i::Turbo4AttentionQueryBody(
      q.data(), k_plane.data(), v_plane.data(), out.data(),
      ShapeSizes(plane_bytes, heads, kHeadDim, /*valid_context=*/64U), t4i::Turbo4KernelStatus::kOk, nullptr,
      /*query_head=*/0U, heads, kKvHeads, kHeadDim, /*valid_context=*/64U,
      0.08838834764831845f, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(a == t4i::Turbo4KernelStatus::kGeometryRefused,
        "READER: a valid_context past the real allocation refuses, even though "
        "every offset it would compute is individually valid");
  bool untouched = true;
  for (float x : out) if (x != -999.0f) untouched = false;
  Check(untouched, "and the output is untouched");

  // And the honest case still works: claims that MATCH the allocation.
  std::vector<float> ok_scores(real_context, 0.0f);
  const auto a2 = t4i::Turbo4AttentionQueryBody(
      q.data(), k_plane.data(), v_plane.data(), out.data(),
      ShapeSizes(plane_bytes, heads, kHeadDim, real_context), t4i::Turbo4KernelStatus::kOk, nullptr, 0U,
      heads, kKvHeads, kHeadDim, real_context, 0.08838834764831845f,
      ok_scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(a2 == t4i::Turbo4KernelStatus::kPoisonedNorm ||
            a2 == t4i::Turbo4KernelStatus::kOk,
        "a MATCHING claim is not refused for size (status "
        + std::to_string(static_cast<std::uint32_t>(a2)) +
        "; the plane is a 0xAB fill here, so kPoisonedNorm is expected)");

  // Turbo4BytesFit itself, including the overflow-safe subtraction.
  Check(!t4i::Turbo4BytesFit(100, 10, 50), "offset 100 need 10 in 50 refuses");
  Check(t4i::Turbo4BytesFit(40, 10, 50), "offset 40 need 10 in 50 fits exactly");
  Check(!t4i::Turbo4BytesFit(41, 10, 50),
        "offset 41 need 10 in 50 refuses — 41 > 50-10");
  Check(!t4i::Turbo4BytesFit(0, 10, 9), "a need larger than the plane refuses");
  Check(!t4i::Turbo4BytesFit(
            static_cast<std::size_t>(-1), 8, static_cast<std::size_t>(-1)),
        "an offset near SIZE_MAX refuses rather than wrapping the check");
}

/// THE CAPACITY CONTRACT — the member of the bounds family §49 did not close.
///
/// §49 fixed the PLANE: both bodies take `plane_bytes` and refuse when the
/// caller's context claim outruns the real allocation. The other three buffers
/// were still checked for null and TRUSTED for length. A non-null pointer says
/// nothing about the size behind it, and the realistic failure is not a null —
/// it is the PARTIAL buffer: the allocation is real, the DECLARED length is
/// short. A caller holding one head of a four-head K tensor passes a valid
/// address, and every index the body computes is individually correct.
///
/// Every refusal here asserts the buffers are BYTE-IDENTICAL, because a refusal
/// that arrives after a partial write is not a refusal.
///
/// WHAT THIS DOES NOT PROVE: that the capacity gate runs BEFORE the first
/// dereference. A status value cannot order two checks — both orderings return
/// the same kGeometryRefused. The ordering is proved by the ASAN control in
/// §50, which deletes the gate and watches the q-scan read past the allocation.
void BufferCapacitiesAreCheckedBeforeAddressing() {
  std::printf("capacity: a non-null pointer is not a length\n");
  using S = t4i::Turbo4KernelStatus;

  // ---------- WRITER: a source that declares fewer heads than it addresses ----
  const std::uint32_t context = 4;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  std::vector<std::uint8_t> k_plane(plane_bytes, 0xAB), v_plane(plane_bytes, 0xCD);
  const std::vector<std::uint8_t> before_k = k_plane, before_v = v_plane;
  std::vector<float> src(kKvHeads * kHeadDim, 0.05f);
  const auto planes_intact = [&] { return k_plane == before_k && v_plane == before_v; };

  // Heads 0-2 are honestly declared; head 3 is not. Head 3's first float sits
  // at byte 3*1024, which is exactly the declared end — so the very first load
  // of that head is one element past the declared length.
  const std::size_t three_heads = 3U * kHeadDim * sizeof(float);
  const auto partial_head = t4i::Turbo4WriteHeadBody(
      src.data(), src.data(), k_plane.data(), v_plane.data(), plane_bytes,
      three_heads, /*head=*/3U, /*pos=*/0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(partial_head == S::kGeometryRefused,
        "WRITER: head 3 refuses when the source declares only three heads");
  Check(planes_intact(), "and neither plane took a byte from the refused head");

  // The SAME short declaration writing head 0 refuses too, and that is the
  // design, not an accident. I expected head 0 to succeed here because it
  // genuinely fits inside three declared heads — and the code correctly says no.
  // The declaration describes the whole [num_kv_heads, head_dim] SOURCE, not
  // the one head being addressed. The kernel launches one thread per kv head
  // sharing a single status word, so a per-head gate would let head 0 COMMIT
  // while heads 1-3 refuse: a half-written plane, which is exactly the outcome
  // the monotone failure latch exists to expose. Uniform refusal is the safe
  // reading of a short declaration.
  const auto head_zero = t4i::Turbo4WriteHeadBody(
      src.data(), src.data(), k_plane.data(), v_plane.data(), plane_bytes,
      three_heads, /*head=*/0U, /*pos=*/0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(head_zero == S::kGeometryRefused,
        "WRITER: a short source declaration refuses for EVERY head, including "
        "head 0 which it genuinely covers — no half-written plane");
  Check(planes_intact(), "and the refused head-0 write still left both planes "
        "byte-identical");

  // The exact boundary: one float short of the full tensor, addressing head 0.
  const auto one_float_short = t4i::Turbo4WriteHeadBody(
      src.data(), src.data(), k_plane.data(), v_plane.data(), plane_bytes,
      kWriterSourceBytes - sizeof(float), /*head=*/0U, /*pos=*/0U, context,
      kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(one_float_short == S::kGeometryRefused,
        "WRITER: a source one FLOAT short of its full length refuses");
  const auto exact = t4i::Turbo4WriteHeadBody(
      src.data(), src.data(), k_plane.data(), v_plane.data(), plane_bytes,
      kWriterSourceBytes, /*head=*/0U, /*pos=*/0U, context, kKvHeads, kHeadDim, gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(exact == S::kOk,
        "WRITER: the SAME address succeeds at exactly the full length — the "
        "gate is tight, not conservative");

  // ---------- READER: q, out and scores ----------
  std::vector<std::uint8_t> rk(plane_bytes, 0), rv(plane_bytes, 0);
  bool quantized = true;
  for (std::uint32_t t = 0; t < context; ++t)
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      std::vector<float> kv(kHeadDim), vv(kHeadDim);
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        kv[i] = 0.01f * static_cast<float>(
                            static_cast<int>((t * 7U + h * 3U + i) % 11U) - 5);
        vv[i] = 0.01f * static_cast<float>(
                            static_cast<int>((t * 5U + h * 9U + i) % 13U) - 6);
      }
      std::size_t at = 0;
      if (!t4i::Turbo4BlockByteOffset(t, h, kHeadDim, kKvHeads, context, &at))
        continue;
      for (std::size_t b = 0; b < kBlocksPerHead; ++b)
        if (!t4::QuantizeBlockChecked(kv.data() + b * 128, rk.data() + at + b * 66) ||
            !t4::QuantizeBlockChecked(vv.data() + b * 128, rv.data() + at + b * 66))
          quantized = false;
    }
  Check(quantized, "the reader fixture plane is fully quantized — the kOk "
        "control below would be meaningless otherwise");

  const std::uint32_t heads = 24;
  std::vector<float> q(heads * kHeadDim, 0.01f);
  std::vector<float> out(heads * kHeadDim, -777.0f);
  std::vector<float> scores(context, 0.0f);
  const float scale = 0.08838834764831845f;
  const std::size_t all_rows = static_cast<std::size_t>(heads) * kHeadDim * sizeof(float);
  const std::size_t one_head = kHeadDim * sizeof(float);
  const std::size_t full_scores = static_cast<std::size_t>(context) * sizeof(float);
  const auto out_intact = [&] {
    for (float x : out)
      if (x != -777.0f) return false;
    return true;
  };
  const auto sizes = [&](std::size_t qb, std::size_t ob, std::size_t sb) {
    return t4i::Turbo4BufferSizes{plane_bytes, qb, ob, sb, 0U};
  };
  // Query head 23 is the tight one: it ends at byte 24*1024, so a `q` declaring
  // one head is short by 23 heads for THIS head and nothing else.
  constexpr std::uint32_t kTightQueryHead = 23;

  const auto short_q = t4i::Turbo4AttentionQueryBody(
      q.data(), rk.data(), rv.data(), out.data(), sizes(one_head, all_rows, full_scores), t4i::Turbo4KernelStatus::kOk, nullptr,
      kTightQueryHead, heads, kKvHeads, kHeadDim, context, scale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(short_q == S::kGeometryRefused,
        "READER: query head 23 refuses when q declares only one head");
  Check(out_intact(), "and out was untouched by the refused query head");

  const auto short_out = t4i::Turbo4AttentionQueryBody(
      q.data(), rk.data(), rv.data(), out.data(), sizes(all_rows, one_head, full_scores), t4i::Turbo4KernelStatus::kOk, nullptr,
      kTightQueryHead, heads, kKvHeads, kHeadDim, context, scale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(short_out == S::kGeometryRefused,
        "READER: a short `out` refuses rather than running off the end of it");
  Check(out_intact(), "and out was untouched");

  // `scores` is PRIVATE per CTA — a caller cannot check it for us, so the body
  // must be told its length even though the kernel allocates it itself.
  const auto short_scores = t4i::Turbo4AttentionQueryBody(
      q.data(), rk.data(), rv.data(), out.data(),
      sizes(all_rows, all_rows, full_scores - sizeof(float)), t4i::Turbo4KernelStatus::kOk, nullptr, kTightQueryHead,
      heads, kKvHeads, kHeadDim, context, scale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(short_scores == S::kGeometryRefused,
        "READER: a `scores` buffer one score short of valid_context refuses");

  // The positive control, last, because it is the only call that writes `out`.
  const auto all_ok = t4i::Turbo4AttentionQueryBody(
      q.data(), rk.data(), rv.data(), out.data(), sizes(all_rows, all_rows, full_scores), t4i::Turbo4KernelStatus::kOk, nullptr,
      kTightQueryHead, heads, kKvHeads, kHeadDim, context, scale, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  Check(all_ok == S::kOk,
        "READER: the SAME head succeeds when all three lengths are honest — the "
        "gates are tight, not blanket");
  Check(out[static_cast<std::size_t>(kTightQueryHead) * kHeadDim] != -777.0f,
        "and the successful call really did write its own row");

  // ---------- the validators themselves, including the overflow guards ----------
  Check(!t4i::Turbo4WriteBuffersAreSufficient(plane_bytes, kWriterSourceBytes - 1U,
                                              kKvHeads, kHeadDim),
        "Turbo4WriteBuffersAreSufficient rejects a source one byte short");
  Check(t4i::Turbo4WriteBuffersAreSufficient(plane_bytes, kWriterSourceBytes,
                                             kKvHeads, kHeadDim),
        "and accepts it at exactly the full length");
  Check(!t4i::Turbo4WriteBuffersAreSufficient(0U, kWriterSourceBytes, kKvHeads,
                                              kHeadDim),
        "a zero-length plane is never sufficient");
  Check(!t4i::Turbo4AttentionBuffersAreSufficient(sizes(all_rows, all_rows, full_scores),
                                                  0xFFFFFFFFU, 0xFFFFFFFFU, 1U),
        "an overflowing head shape refuses instead of wrapping the byte math");
  Check(!t4i::Turbo4AttentionBuffersAreSufficient(sizes(all_rows, all_rows, 0U),
                                                  heads, kHeadDim, 1U),
        "a zero-length scores buffer refuses for a non-empty context");
  Check(t4i::Turbo4AttentionBuffersAreSufficient(
            sizes(all_rows, all_rows, 0U), heads, kHeadDim, 0U),
        "and an empty context needs no scores buffer at all");
}

/// CMake source with `#` comments removed, for the build-membership guard.
///
/// SAME DEFECT, DIFFERENT FILE. The guard searched the raw CMakeLists for the
/// token `kernels/<name>.hip`. A COMMENTED-OUT entry still contains that token,
/// so disabling a kernel by prefixing its line with `#` would satisfy the guard
/// while the build no longer compiles it — the guard reporting coverage it does
/// not have. That is exactly the C++ comment problem `StripComments()` solves,
/// found unsolved on this side by parent review.
///
/// Known limit, stated rather than hidden: every `#` is treated as a comment
/// start, including one inside a quoted CMake string. No such string exists in
/// this file, and being wrong here only makes the guard stricter, never looser
/// — the failure mode is a false failure, not a missed kernel.
[[nodiscard]] std::string StripCmakeComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    if (src[i] == '#') {
      while (i < src.size() && src[i] != '\n') ++i;
      if (i < src.size()) out += '\n';
    } else {
      out += src[i];
    }
  }
  return out;
}

/// NO KERNEL MAY BE ORPHANED FROM THE BUILD.
///
/// Found while wiring the two Turbo4 kernels in: `src/models/qwen/hip/kernels/`
/// held 38 `.hip` files and the qwen CMakeLists named 36 of them. The two
/// missing were MINE. Every sibling kernel is compiled by `gufo_core`, so a
/// signature change to any of them breaks the build loudly; an unlisted kernel
/// rots in silence and only breaks when someone happens to hand-run `hipcc`.
///
/// That is the whole failure this asserts against: source that nothing builds
/// is not source, it is a comment. Being listed does NOT mean callable — the
/// Turbo4 gate is separately FALSE and has zero callers. This check is about
/// compilation only.
void EveryKernelIsWiredIntoTheBuild() {
  std::printf("build wiring: no kernel may be orphaned from gufo_core\n");
  const std::string cmake = "src/models/qwen/CMakeLists.txt";
  const std::string kernels = "src/models/qwen/hip/kernels";

  std::ifstream f(cmake);
  if (!f) {
    Check(false, "the qwen CMakeLists is readable from the run cwd (" + cmake +
                     ") — without it this check cannot run");
    return;
  }
  std::stringstream buffer;
  buffer << f.rdbuf();
  // COMMENTS REMOVED. A `#`-commented entry contains the same token as a live
  // one, so the raw text could not tell a compiled kernel from a disabled one.
  const std::string listed = StripCmakeComments(buffer.str());

  std::vector<std::string> orphans;
  std::size_t seen = 0;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(kernels, ec)) {
    if (!entry.is_regular_file()) continue;
    if (entry.path().extension() != ".hip") continue;
    ++seen;
    const std::string base = entry.path().filename().string();
    // The exact token, not a substring: `norm.hip` must not be satisfied by
    // some longer name that happens to contain it.
    if (listed.find("kernels/" + base) == std::string::npos)
      orphans.push_back(base);
  }

  if (ec) {
    Check(false, "the kernels directory is iterable (" + kernels +
                     ") — without it this check cannot run");
    return;
  }
  Check(orphans.empty(),
        std::to_string(seen) + " .hip kernels found, every one listed in " +
            cmake);
  if (!orphans.empty()) {
    std::string names;
    for (const auto& o : orphans) names += " " + o;
    Check(false, "ORPHANED kernels, listed nowhere in " + cmake + ":" + names);
  }

  // CONTROLS for the comment weakness this guard was found to have. Synthetic
  // sources, so the answer does not depend on how the real file looks today.
  const std::string live =
      "  ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/thing.hip\n";
  const std::string commented =
      "  # ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/thing.hip\n";
  Check(StripCmakeComments(commented).find("kernels/thing.hip") ==
            std::string::npos,
        "CONTROL: a COMMENTED-OUT entry does not satisfy the membership check");
  Check(StripCmakeComments(live).find("kernels/thing.hip") != std::string::npos,
        "CONTROL: a LIVE entry still does — stripping did not blind the guard");
}

/// ONE AUTHORITY FOR THE CONTEXT BOUND.
///
/// `kMaxScalarScores` was `= 4096` written inline in `turbo4_attention.hip` —
/// anonymous namespace, invisible to this oracle, and duplicated by the
/// accounting that sizes the work (`manifest = 4096` in
/// `opt66_lifetime_trace.py`). Two literals for one context in two trees is a
/// silent disagreement waiting to happen: move one and the kernel and its
/// budget stop describing the same thing.
///
/// The value is checked here; the cross-tree equality is checked by the Turbo4
/// accounting suite, which can see both files.
void ContextBoundIsOneAuthority() {
  std::printf("context bound: one number, one home\n");
  using S = t4i::Turbo4KernelStatus;

  Check(t4i::kTurbo4ManifestContext == 4096U,
        "the manifest context is 4096 — the context the guard actually serves");
  Check(t4i::kTurbo4ScoresSharedBytes == 16384U,
        "16 KiB of shared memory per CTA at the bound (4096 * sizeof(float) = "
        "16384), stated rather than left implicit");

  // The bound must be usable by the very body the oracle calls, at the very
  // boundary — not merely equal to a constant.
  const std::uint32_t ctx = t4i::kTurbo4ManifestContext;
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, ctx);
  Check(plane_bytes != 0U,
        "a plane sized at the manifest context is computable (" +
            std::to_string(plane_bytes) + " bytes per layer)");

  const std::uint32_t heads = 24;
  std::vector<float> q(heads * kHeadDim, 0.01f);
  std::vector<float> out(heads * kHeadDim, -5.0f);
  std::vector<float> scores(ctx, 0.0f);
  const auto sizes = ShapeSizes(plane_bytes, heads, kHeadDim, ctx);
  const auto at_bound = t4i::Turbo4AttentionQueryBody(
      q.data(), nullptr, nullptr, out.data(), sizes, t4i::Turbo4KernelStatus::kOk, nullptr, /*query_head=*/0U, heads,
      kKvHeads, kHeadDim, ctx, 0.08838834764831845f, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  // Null planes are refused BEFORE anything is sized, so this asserts the gate
  // order rather than the arithmetic: the bound is reachable and the refusal is
  // still a refusal, with `out` untouched.
  Check(at_bound == S::kGeometryRefused,
        "the body at the full manifest context refuses on null planes rather "
        "than degrading");
  bool untouched = true;
  for (float x : out)
    if (x != -5.0f) untouched = false;
  Check(untouched, "and leaves the output untouched while refusing");

  // The .hip must NAME the constant. A bare 4096 reappearing there is the exact
  // drift this is here to prevent.
  const char* hip = "src/models/qwen/hip/kernels/turbo4_attention.hip";
  std::ifstream hf(hip);
  if (!hf) {
    Check(false, std::string("the reader launcher is readable (") + hip + ")");
    return;
  }
  std::stringstream hb;
  hb << hf.rdbuf();
  const std::string src = hb.str();
  const std::string code = StripComments(src);
  Check(code.find("kTurbo4ManifestContext") != std::string::npos,
        "the launcher derives its bound from the shared authority");
  Check(code.find("= 4096") == std::string::npos &&
            code.find("=4096") == std::string::npos,
        "and its CODE carries no bare 4096 literal — comments are excluded, so "
        "this can only fail on a real reintroduction");
}

/// THE PLANE-CAPACITY MATRIX — added after review measured the gate
/// letting an out-of-bounds read through.
///
/// The defect was that the reader checked the last token's FIRST head and
/// omitted the `kv_head * bph` term, so a caller claiming one token more than
/// it allocated passed. The body then returned kOk while reading past the end,
/// which is worse than a crash: a crash is loud, a kOk on garbage is believed.
///
/// The matrix is deliberate about the OTHER direction too. A gate that refuses
/// everything would pass every "must refuse" row above and still be broken, so
/// the honest cells must come back NOT-refused-for-size. Capacity is isolated
/// from the norm scan on purpose: a zero-filled plane makes the body return
/// kPoisonedNorm, which is not a capacity answer, so the assertion asks only
/// that an honest claim is not refused as kGeometryRefused.
void PlaneCapacityMatrix() {
  std::printf("matrix: plane capacity over 4/5 tokens x first/last kv head\n");
  using S = t4i::Turbo4KernelStatus;
  const std::uint32_t heads = 24, kv = 4, head_dim = 256;
  // GQA: 24 query heads over 4 kv heads, so kv_head = query_head / 6.
  // Head 0 is the FIRST kv head, head 18 is the LAST.
  const std::uint32_t first_kv_q = 0, last_kv_q = 18;

  bool over_claim_all_refused = true;
  bool honest_none_refused = true;
  std::string detail;

  for (std::uint32_t plane_ctx : {4U, 5U}) {
    for (std::uint32_t valid_ctx : {4U, 5U}) {
      const std::size_t plane_bytes =
          t4i::Turbo4LayerPlaneBytes(head_dim, kv, plane_ctx);
      std::vector<std::uint8_t> kp(plane_bytes, 0), vp(plane_bytes, 0);
      const std::size_t rows =
          static_cast<std::size_t>(heads) * head_dim * sizeof(float);
      std::vector<float> q(heads * head_dim, 0.01f);
      std::vector<float> out(heads * head_dim, -1.0f);
      std::vector<float> scores(valid_ctx, 0.0f);
      const t4i::Turbo4BufferSizes sizes{
          plane_bytes, rows, rows,
          static_cast<std::size_t>(valid_ctx) * sizeof(float), 0U};

      for (std::uint32_t qh : {first_kv_q, last_kv_q}) {
        const auto st = t4i::Turbo4AttentionQueryBody(
            q.data(), kp.data(), vp.data(), out.data(), sizes, t4i::Turbo4KernelStatus::kOk, nullptr, qh, heads, kv,
            head_dim, valid_ctx, 0.08838834764831845f, scores.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
        if (valid_ctx > plane_ctx) {
          // OVER-CLAIM: the plane cannot hold it. MUST refuse as capacity.
          if (st != S::kGeometryRefused) {
            over_claim_all_refused = false;
            detail += " plane" + std::to_string(plane_ctx) + "/claim" +
                      std::to_string(valid_ctx) + "/qh" + std::to_string(qh) +
                      "=" + std::to_string(static_cast<unsigned>(st));
          }
        } else {
          // HONEST: fits. Must NOT be refused for capacity, whatever the norms do.
          if (st == S::kGeometryRefused) {
            honest_none_refused = false;
            detail += " HONEST plane" + std::to_string(plane_ctx) + "/claim" +
                      std::to_string(valid_ctx) + "/qh" + std::to_string(qh);
          }
        }
      }
    }
  }
  Check(over_claim_all_refused,
        "every OVER-CLAIM cell refuses with kGeometryRefused — 4 cells across "
        "5 tokens over a 4-token plane, first and last kv head" +
            (over_claim_all_refused ? "" : detail));
  Check(honest_none_refused,
        "every HONEST cell is NOT refused for capacity — the gate is not "
        "blanket" + (honest_none_refused ? "" : detail));

  // The wrap. Review measured 132 where 0 is the only safe answer.
  const std::size_t maxv = static_cast<std::size_t>(-1);
  Check(t4i::Turbo4LayerPlaneBytes(128, 2, maxv / 2 + 2) == 0U,
        "Turbo4LayerPlaneBytes(128, 2, SIZE_MAX/2+2) refuses instead of "
        "wrapping to 132");
  Check(t4i::Turbo4LayerPlaneBytes(128, 2, maxv) == 0U,
        "Turbo4LayerPlaneBytes(128, 2, SIZE_MAX) refuses");
  // The largest context that FITS. I first asserted `maxv / 2`, which is not
  // the boundary at all and failed: with bpt=2 the binding constraint is the
  // BYTE multiplication, not the context, so the last context that survives is
  // (maxv / kBlockBytes) / bpt. Asserting a guessed formula is how the wrong
  // number gets in; derive it from the same constants the function uses.
  const std::size_t ctx_max = (maxv / t4::kBlockBytes) / 2;  // bpt = 2 at (128,2)
  Check(t4i::Turbo4LayerPlaneBytes(128, 2, ctx_max) != 0U,
        "the largest context that FITS is still computed, not refused");
  Check(t4i::Turbo4LayerPlaneBytes(128, 2, ctx_max + 1) == 0U,
        "one past that context refuses");
  Check(t4i::Turbo4LayerPlaneBytes(128, 2, maxv / 2 + 1) == 0U,
        "a context far past the byte limit also refuses");
  // A wrapped size is a small legal-looking number, which is the real danger.
  Check(t4i::Turbo4LayerPlaneBytes(1, 1, 0) == 0U,
        "a zero context still refuses");
}

// ---------------------------------------------------------------------------
// B-D1 REQUIRED (c): A SOURCE GUARD ON THE CALL SITES, not on the helpers.
// ---------------------------------------------------------------------------

void DecodePathActuallyUsesTheStatusWords() {
  // WHY THIS FILE IS A SOURCE GUARD. Found in review: the reset
  // helper and the readback helper both existed, were both correct, and were
  // both UNCALLED in src/ — so the words were never reset and never read, and my
  // earlier controls passed because they exercised the HELPERS rather than the
  // call sites. A helper that nothing calls is not a feature; it is a comment
  // with a signature. So this asserts the CALL SITES exist, in the files that
  // run the decode path.
  const auto read_file = [](const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    std::string out;
    if (f != nullptr) {
      char buf[8192];
      std::size_t n = 0;
      while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
      std::fclose(f);
    }
    return out;
  };

  const std::string step = read_file("src/models/qwen/hip/decode_step.cpp");
  const std::string dec = read_file("src/models/qwen/hip/decode.cpp");
  const std::string arena = read_file("src/models/qwen/hip/arena.cpp");
  Check(!step.empty() && !dec.empty() && !arena.empty(),
        "call sites: the three decode-path sources are readable");

  // CODE ONLY, so a comment describing the rule cannot satisfy it.
  const std::string step_c = StripComments(step);
  const std::string dec_c = StripComments(dec);
  const std::string arena_c = StripComments(arena);

  // (a) THE RESET FOLLOWS THE VERDICT. It used to sit at the top of the step,
  // which lost any failure raised during a NON-logits step: that step returns
  // before the readback, and the next step's reset erased the evidence before
  // anything read it. So the guard now asserts the ORDER, not just the presence.
  Check(step_c.find("Turbo4ResetStepStatusWords") == std::string::npos,
        "call site (a): decode_step does NOT reset at the top of the step — a "
        "non-logits step returns before the readback, and that reset would erase "
        "its failure unobserved");
  Check(dec_c.find("Turbo4ResetStepStatusWords") != std::string::npos,
        "call site (a): decode resets the words AFTER a passing verdict");

  // ...and after the verdict means after the refusal check, so a thrown verdict
  // leaves its evidence in the words instead of clearing it.
  {
    const std::size_t refuses = dec_c.find("Turbo4StepStatusFromWords");
    const std::size_t reset = dec_c.find("Turbo4ResetStepStatusWords");
    Check(refuses != std::string::npos && reset != std::string::npos &&
              refuses < reset,
          "call site (a): the reset comes AFTER the verdict, so a thrown failure "
          "leaves its evidence in the words");
  }

  // (a) allocation still initialises them, so they are never unreadable garbage
  // before the first verdict ever runs.
  Check(arena_c.find("Turbo4ResetStepStatusWords") != std::string::npos,
        "call site (a): arena initialises both words right after hipMalloc");

  // (a) AND THE SESSION PATH, so a thrown verdict cannot poison the next
  // request: the per-step reset is below the throw, so the failure survives it.
  {
    const std::size_t r = arena_c.find("void QwenGpuArena::Reset");
    Check(r != std::string::npos &&
              arena_c.find("Turbo4ResetStepStatusWords", r) < r + 4000,
          "call site (a): the session Reset() clears the words too — otherwise "
          "a new request inherits the previous step's thrown verdict");
  }

  // (b) the readback, and the verdict, in decode.cpp.
  Check(dec_c.find("d_kv_turbo4_write_status") != std::string::npos &&
            dec_c.find("d_kv_turbo4_read_status") != std::string::npos,
        "call site (b): decode READS BACK both status words");
  Check(dec_c.find("Turbo4StepStatusFromWords") != std::string::npos &&
            dec_c.find("Refuses()") != std::string::npos,
        "call site (b): decode RUNS the verdict");
  // ...and takes it BEFORE publishing. `CheckedSampleToken` is the publication
  // point, so the refusal must appear before it in the file.
  {
    const std::size_t verdict = dec_c.find("Turbo4StepStatusFromWords");
    const std::size_t publish = dec_c.find("CheckedSampleToken");
    Check(verdict != std::string::npos && publish != std::string::npos &&
              verdict < publish,
          "call site (b): the refusal is taken BEFORE the token is published — "
          "refusing after publication hands back a token from a plane nobody "
          "finished writing");
  }
  // ...in the SAME sync: exactly one stream synchronize in that file, and both
  // D2H copies precede it.
  {
    const std::size_t sync = dec_c.find("hipStreamSynchronize");
    const std::size_t w = dec_c.find("d_kv_turbo4_write_status");
    Check(sync != std::string::npos && w != std::string::npos && w < sync,
          "call site (b): the status copies are queued BEFORE the single stream "
          "synchronize — a second sync would serialise the host twice per token");
  }

  // R3: THE HOST MUST NOT LATCH THE DEVICE WORDS. This is the specific defect
  // the re-review caught, so it gets its own assertion rather than being folded
  // into the ones above.
  Check(step_c.find("Turbo4LatchFailure") == std::string::npos,
        "call site R3: decode_step does NOT latch from host code — the words are "
        "hipMalloc device pointers and a host '*status =' is a host "
        "read-modify-write of device memory, unordered against the stream");
  Check(dec_c.find("Turbo4LatchFailure") == std::string::npos,
        "call site R3: decode.cpp does not latch them from host either");
}

}  // namespace

int main() {
  IndexMatchesSpec();
  GeometryConstants();
  RefusesRatherThanRounds();
  RefusesOutOfRange();
  SharedCodecRoundTrip();
  AttentionReferenceAgreesWithFp16();
  KernelAmendments();
  WholeHeadAtomicity();
  StableSoftmax();
  KernelBodiesCalledDirectly();
  AttentionBodyRefusals();
  FailureLatchIsMonotone();
  HostNeverDereferencesDeviceStatus();
  HostStatusGuardControls();
  NullVIsRefusedNotDereferenced();
  PlaneBoundsAreCheckedAgainstTheRealAllocation();
  BufferCapacitiesAreCheckedBeforeAddressing();
  EveryKernelIsWiredIntoTheBuild();
  DecodePathActuallyUsesTheStatusWords();
  ContextBoundIsOneAuthority();
  PlaneCapacityMatrix();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

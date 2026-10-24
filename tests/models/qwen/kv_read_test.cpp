// The K READ path, CPU only.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/kv_read_test.cpp -o /tmp/r
//
// Geometry is UD-Q4_K_XL: head_dim 256, kv_heads 4, so kv_width 1024 per token
// per plane, 2 blocks per head and 8 blocks (528 B) per token.
//
// The sharpest check in here is the FIRST one. DequantizeBlock returns the
// ROTATED domain, so a reader that forgets RotateInverse still produces a
// vector with the right length, the right norm, and plausible-looking floats —
// silently wrong. So the round trip is asserted against the ORIGINAL
// pre-quantization values, not against a norm or a checksum.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace t4 = gufo::hip::turbo4;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string &what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvPlaneGeometry;
using gufo::hip::QwenKvPlaneIsReadable;
using gufo::hip::QwenKvReadHead;

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kKvWidth = kHeadDim * kKvHeads;  // 1024
constexpr std::size_t kContext = 8;

/// Distinct, reproducible values so a wrong offset shows up as the wrong
/// head's data rather than as a plausible-but-similar vector.
float TokenValue(std::size_t token, std::size_t head, std::size_t i) {
  return static_cast<float>(1 + token * 17 + head * 3) *
         (1.0f + 0.01f * static_cast<float>(i % 13)) *
         ((i % 5 == 0) ? -1.0f : 1.0f);
}

/// Fill a Turbo4 plane token-major, head-minor.
std::vector<std::uint8_t> WriteTurbo4Plane(
    const QwenKvPlaneGeometry& g, const std::vector<float>& source) {
  const std::size_t plane_bytes =
      gufo::hip::QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4,
                                        g.head_dim * g.kv_heads * g.context);
  std::vector<std::uint8_t> plane(plane_bytes);
  const std::size_t bph = g.head_dim / t4::kBlockValues;
  for (std::size_t t = 0; t < g.context; ++t) {
    for (std::size_t h = 0; h < g.kv_heads; ++h) {
      const std::size_t first = t * bph * g.kv_heads + h * bph;
      for (std::size_t b = 0; b < bph; ++b) {
        const std::size_t src = (t * g.kv_heads + h) * g.head_dim +
                                b * t4::kBlockValues;
        t4::QuantizeBlock(source.data() + src,
                          plane.data() + (first + b) * t4::kBlockBytes);
      }
    }
  }
  return plane;
}

}  // namespace

int main() {
  std::printf("K read path (CPU, HIP-free)\n");
  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, kContext};

  // Source values, laid out exactly as WriteTurbo4Plane expects.
  std::vector<float> source(kKvWidth * kContext);
  for (std::size_t t = 0; t < kContext; ++t) {
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        source[(t * kKvHeads + h) * kHeadDim + i] = TokenValue(t, h, i);
      }
    }
  }
  const std::vector<std::uint8_t> plane = WriteTurbo4Plane(g, source);

  // ---- 1. THE READ LANDS IN THE ORIGINAL DOMAIN ------------------------
  // The check that catches a missing RotateInverse: a rotated-domain read has
  // the right norm and the right length, so only comparing VALUES against the
  // pre-quantization input finds it.
  //
  // THE METRIC MATTERS, and getting it wrong first time is instructive. A
  // per-element relative error (|e_i| / |x_i|) divides by a near-zero element
  // and reported 196370x on a sin() input whose norm was correct to 3e-4. The
  // codec was fine; the metric was not. The block-scale metrics below are the
  // ones that mean something for a transform codec.
  auto L2 = [](const std::vector<float>& v, std::size_t from, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = v[from + i];
      acc += d * d;
    }
    return std::sqrt(acc);
  };
  {
    double worst_l2 = 0.0;
    double worst_norm = 0.0;
    std::vector<float> out(kHeadDim);
    bool all_read = true;
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        if (!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(),
                            plane.size(), g, t, h, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) {
          all_read = false;
          continue;
        }
        const std::size_t src = (t * kKvHeads + h) * kHeadDim;
        const double in_l2 = L2(source, src, kHeadDim);
        double err = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
          const double d = out[i] - source[src + i];
          err += d * d;
        }
        worst_l2 = std::fmax(worst_l2, std::sqrt(err) / in_l2);
        worst_norm = std::fmax(worst_norm, std::fabs(L2(out, 0, kHeadDim) / in_l2 - 1.0));
      }
    }
    std::printf("  note  worst block L2 reconstruction error = %.5f\n", worst_l2);
    std::printf("  note  worst block NORM ratio deviation   = %.6f\n", worst_norm);
    Check(all_read, "every (token, head) in the plane reads back");
    // A rotated-domain read scores ~8.0 here, not ~0.09.
    Check(worst_l2 < 0.30, "the read returns ORIGINAL-domain values, not rotated ones");
    // The norm is stored explicitly in fp16, so preservation is nearly free and
    // is a property of the FORMAT, not evidence of value accuracy.
    Check(worst_norm < 1e-3, "the block norm survives the round trip to fp16 precision");
  }

  // ---- 1b. THE ERROR IS ~9% AND THAT IS THE FORMAT, NOT A BUG ---------
  {
    std::vector<float> out(kHeadDim);
    std::vector<double> per_type;
    // Same measurement on four unrelated shapes. If Turbo4 were healthy the
    // number would not move; a large spread would mean the centroid table is
    // tuned for one distribution and wrong for others.
    for (int kind = 0; kind < 4; ++kind) {
      std::vector<float> probe(t4::kBlockValues);
      for (std::size_t i = 0; i < probe.size(); ++i) {
        switch (kind) {
          case 0: probe[i] = 1.0f + 0.01f * static_cast<float>(i); break;
          case 1: probe[i] = std::sin(0.05f * static_cast<float>(i)) * 3.0f; break;
          case 2: probe[i] = (i % 5 == 0) ? -1.0f : 1.0f; break;
          default: probe[i] = static_cast<float>((i * 37 % 101)) / 40.0f - 1.0f; break;
        }
      }
      std::uint8_t blk[t4::kBlockBytes];
      const bool ok = t4::QuantizeBlockChecked(probe.data(), blk);
      if (!ok) { per_type.push_back(NAN); continue; }
      std::vector<float> back(t4::kBlockValues);
      t4::DequantizeBlock(blk, back.data());
      t4::RotateInverse(back.data());
      double err = 0.0;
      for (std::size_t i = 0; i < probe.size(); ++i) {
        const double d = back[i] - probe[i];
        err += d * d;
      }
      const double rel = std::sqrt(err) / L2(probe, 0, probe.size());
      per_type.push_back(rel);
      if (!std::isfinite(rel)) {
        std::printf("  note    shape %d produced a NON-FINITE relative error\n", kind);
      }
    }
    // Reject non-finite results BEFORE comparing them. std::fmax silently
    // ignores a NaN, so a garbage measurement used to pass all three spread
    // checks below while the note line printed -nan.
    bool any_bad = false;
    for (double e : per_type) if (!std::isfinite(e)) any_bad = true;
    Check(!any_bad, "every input shape produced a FINITE error measurement");
    double lo = per_type[0], hi = per_type[0];
    for (double e : per_type) {
      if (!std::isfinite(e)) continue;
      lo = std::fmin(lo, e);
      hi = std::fmax(hi, e);
    }
    std::printf("  note  L2 error across 4 input shapes: min %.5f max %.5f\n", lo, hi);
    for (size_t k = 0; k < per_type.size(); ++k) {
      std::printf("  note    shape %zu: %.5f\n", k, per_type[k]);
    }
    // 4 bits = 16 levels; a companded table of that width cannot beat roughly
    // 2^-4.5 ~ 4.4% and lands near 9% in practice. The bound is the FORMAT's,
    // not fitted to these four outputs.
    Check(lo > 0.02, "the error is at 4-bit scale, not a broken round trip");
    Check(hi < 0.15, "and it stays inside the 4-bit format's error scale");
    Check(hi - lo < 0.05,
          "the error does not depend on input shape: one number describes it");
    (void)out;
  }

  // ---- 2. ROTATED-DOMAIN READS WOULD BE CAUGHT --------------------------
  {
    // Prove the guard in check 1 has teeth: read WITHOUT the inverse rotation.
    const std::size_t bph = kHeadDim / t4::kBlockValues;
    std::vector<float> scratch(t4::kBlockValues);
    double worst_no_inverse = 0.0;
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      std::vector<float> flat(kHeadDim);
      for (std::size_t b = 0; b < bph; ++b) {
        const std::size_t block = h * bph + b;
        t4::DequantizeBlock(plane.data() + block * t4::kBlockBytes, scratch.data());
        for (std::size_t j = 0; j < t4::kBlockValues; ++j) {
          flat[b * t4::kBlockValues + j] = scratch[j];
        }
      }
      const std::size_t src = h * kHeadDim;
      double err = 0.0;
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        const double d = flat[i] - source[src + i];
        err += d * d;
      }
      worst_no_inverse =
          std::fmax(worst_no_inverse, std::sqrt(err) / L2(source, src, kHeadDim));
    }
    std::printf("  note  worst L2 error WITHOUT inverse rotation = %.5f\n",
                worst_no_inverse);
    Check(worst_no_inverse > 1.0,
          "omitting RotateInverse really does break the values (control)");
  }

  // ---- 3. OFFSETS: EACH HEAD IS ITS OWN DATA ---------------------------
  {
    std::vector<float> out(kHeadDim);
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    const float v_head0 = out[0];
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g, 0, 1, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    Check(std::fabs(out[0] - v_head0) > 1e-3f,
          "head 1 does not return head 0's data (offset is head-strided)");
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g, 1, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    Check(std::fabs(out[0] - v_head0) > 1e-3f,
          "token 1 does not return token 0's data (offset is token-strided)");
    // Head h of token 0 must equal head 0 of the shifted token h.
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g, 0, 2, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    const float h2 = out[0];
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    Check(std::fabs(out[0] - h2) > 1e-3f, "head 0 and head 2 carry different data");
  }

  // ---- 4. REFUSALS: OUT OF RANGE ----------------------------------------
  {
    std::vector<float> out(kHeadDim, -12345.0f);
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g,
                          kContext, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "token == context is refused");
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g,
                          kContext + 100, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "token far past the end is refused");
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g, 0,
                          kKvHeads, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "head == kv_heads is refused");
    Check(std::fabs(out[0] + 12345.0f) < 1e-6f,
          "a refused read leaves the output untouched");
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, nullptr, 0, g, 0, 0,
                          out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a null plane is refused");
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(), g, 0, 0,
                          nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a null output is refused");
  }

  // ---- 5. REFUSALS: A PLANE TOO SMALL ----------------------------------
  {
    std::vector<float> out(kHeadDim);
    // A buffer holding only the first two tokens cannot answer for token 5.
    std::vector<std::uint8_t> tiny(plane.size() / kContext * 2);
    std::copy(plane.begin(), plane.begin() + tiny.size(), tiny.begin());
    // BEHAVIOUR CHANGE, made by a review probe. This used to assert
    // that a short plane "still answers for the tokens it contains" — partial
    // reads from a truncated buffer. The probe required full-plane bounds to
    // be checked BEFORE addressing, so a buffer that cannot hold the whole
    // plane is now refused outright, even for index 0. Partial reads from a
    // truncated plane are exactly what made the wrapping case look valid.
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, tiny.data(), tiny.size(),
                          g, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a short plane is refused outright, including for index 0");
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, tiny.data(), tiny.size(),
                          g, 2, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a short plane refuses an index past its end");
  }

  // ---- 6. REFUSALS: BAD GEOMETRY ---------------------------------------
  {
    Check(!QwenKvPlaneIsReadable(QwenKvCacheStorage::kTurbo4,
                                 QwenKvPlaneGeometry{0, kKvHeads, kContext}),
          "zero head_dim is refused");
    Check(!QwenKvPlaneIsReadable(QwenKvCacheStorage::kTurbo4,
                                 QwenKvPlaneGeometry{kHeadDim, 0, kContext}),
          "zero kv_heads is refused");
    Check(!QwenKvPlaneIsReadable(QwenKvCacheStorage::kTurbo4,
                                 QwenKvPlaneGeometry{kHeadDim, kKvHeads, 0}),
          "zero context is refused");
    Check(!QwenKvPlaneIsReadable(QwenKvCacheStorage::kTurbo4,
                                 QwenKvPlaneGeometry{100, kKvHeads, kContext}),
          "head_dim 100 is refused: it is not whole blocks");
    Check(!QwenKvPlaneIsReadable(QwenKvCacheStorage::kTurbo4,
                                 QwenKvPlaneGeometry{64, kKvHeads, kContext}),
          "head_dim 64 is refused: a half block would be shared with a neighbour");
    Check(QwenKvPlaneIsReadable(QwenKvCacheStorage::kTurbo4, g),
          "the UD-Q4 geometry is readable");
    std::vector<float> out(kHeadDim);
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane.size(),
                          QwenKvPlaneGeometry{100, kKvHeads, kContext}, 0, 0,
                          out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "reading with a bad geometry is refused, not approximated");
  }

  // ---- 7. ELEMENT-ADDRESSED STORAGE STILL READS ------------------------
  {
    std::vector<std::uint16_t> half(kKvWidth * kContext);
    for (std::size_t i = 0; i < half.size(); ++i) {
      half[i] = t4::FloatToHalfBits(source[i]);
    }
    std::vector<float> out(kHeadDim);
    Check(QwenKvReadHead(QwenKvCacheStorage::kFp16,
                         reinterpret_cast<const std::uint8_t*>(half.data()),
                         half.size() * sizeof(std::uint16_t), g, 3, 2, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kFp16)), true),
          "fp16 planes read back");
    const std::size_t src = (3 * kKvHeads + 2) * kHeadDim;
    // RELATIVE error. An absolute bound is wrong here: this head's values run
    // to about 52, where fp16's own resolution is ~0.03, so an absolute 1e-2
    // bound is below the format's precision and fails on a perfect round trip.
    double worst_rel = 0.0;
    double worst_abs = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      const double denom = std::fmax(1e-6, std::fabs(static_cast<double>(source[src + i])));
      worst_abs = std::fmax(worst_abs, std::fabs(out[i] - source[src + i]));
      worst_rel = std::fmax(worst_rel, std::fabs(out[i] - source[src + i]) / denom);
    }
    std::printf("  note  fp16 read: worst abs %.6f, worst rel %.6f\n", worst_abs, worst_rel);
    Check(worst_rel < 1e-3, "fp16 round trip is exact to fp16 relative precision");

    std::vector<float> raw(kKvWidth * kContext);
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = source[i];
    Check(QwenKvReadHead(QwenKvCacheStorage::kFp32,
                         reinterpret_cast<const std::uint8_t*>(raw.data()),
                         raw.size() * sizeof(float), g, 3, 2, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kFp32)), true),
          "fp32 planes read back");
    Check(out[0] == source[src] && out[kHeadDim - 1] == source[src + kHeadDim - 1],
          "fp32 round trip is bit-exact");
  }

  // ---- 8. THE PLANE SIZE THE READER ASSUMES IS THE WRITER'S SIZE ------
  {
    const std::size_t reader_bytes = gufo::hip::QwenKvBytesPerToken(
        QwenKvCacheStorage::kTurbo4, g);
    const std::size_t per_token_blocks =
        gufo::hip::QwenKvBlocksPerHead(g) * g.kv_heads;
    Check(per_token_blocks == 8, "8 blocks per token at this geometry");
    Check(reader_bytes == per_token_blocks * t4::kBlockBytes,
          "the read path's byte count equals blocks x block size");
    Check(reader_bytes == 528U, "528 B per token per plane");
    Check(plane.size() == reader_bytes * kContext,
          "the written plane is exactly reader_bytes x context");
  }

  // ---- 9. THE fp16 NORM OVERFLOWS, AND NOW IT REFUSES ------------------
  // The block norm is stored as a ggml_half, so a group whose L2 norm exceeds
  // 65504 becomes inf and EVERY value in the block dequantizes to NaN. This
  // was unguarded on both sides: the writer stored the poison and the reader
  // handed NaN back into arithmetic that never trips a bounds check.
  {
    Check(t4::kNormMaxFinite == 65504.0f, "the fp16 ceiling is the documented 65504");

    // The write side refuses.
    std::vector<float> huge(t4::kBlockValues, 100000.0f);
    std::uint8_t blk[t4::kBlockBytes];
    std::memset(blk, 0xA5, sizeof(blk));
    Check(!t4::QuantizeBlockChecked(huge.data(), blk),
          "a block whose norm overflows fp16 is REFUSED by the checked encoder");
    bool untouched = true;
    for (std::size_t i = 0; i < t4::kBlockBytes; ++i) {
      if (blk[i] != 0xA5) untouched = false;
    }
    Check(untouched, "a refused encode writes nothing at all");

    // The predicate agrees with the boundary.
    Check(t4::GroupNormIsStorable(1.0f), "norm 1.0 is storable");
    Check(t4::GroupNormIsStorable(t4::kNormMaxFinite), "norm exactly 65504 is storable");
    Check(!t4::GroupNormIsStorable(t4::kNormMaxFinite + 1.0f), "65505 is not");
    Check(!t4::GroupNormIsStorable(std::numeric_limits<float>::infinity()),
          "infinity is not storable");
    Check(!t4::GroupNormIsStorable(std::nanf("")), "NaN is not storable");
    Check(!t4::StoredNormIsUsable(std::numeric_limits<float>::infinity()),
          "an inf stored norm is refused on read");
    Check(t4::StoredNormIsUsable(3.5f), "a normal stored norm is usable");
  }

  // ---- 10. THE READ PATH REFUSES A POISONED BLOCK ---------------------
  {
    // Hand-build a block whose stored norm is inf, put it in a valid plane,
    // and confirm the reader refuses rather than returning NaN.
    QwenKvPlaneGeometry one{kHeadDim, kKvHeads, 1};
    const std::size_t plane_bytes = gufo::hip::QwenKvBytesPerToken(
        QwenKvCacheStorage::kTurbo4, one);
    std::vector<std::uint8_t> bad_plane(plane_bytes, 0);
    const std::uint16_t inf_half = 0x7C00;  // fp16 infinity
    std::memcpy(bad_plane.data(), &inf_half, sizeof(inf_half));
    std::vector<float> out(kHeadDim, 7.0f);
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, bad_plane.data(),
                          bad_plane.size(), one, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a plane whose first block norm is inf is REFUSED");
    Check(std::fabs(out[0] - 7.0f) < 1e-6f,
          "the refusal leaves the output untouched, not filled with NaN");

    // A finite norm in the same position still reads.
    const std::uint16_t one_half = t4::FloatToHalfBits(12.5f);
    std::memcpy(bad_plane.data(), &one_half, sizeof(one_half));
    Check(QwenKvReadHead(QwenKvCacheStorage::kTurbo4, bad_plane.data(),
                         bad_plane.size(), one, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "the same plane with a finite norm reads normally");
  }

  // ---- 11. A POISONED SECOND BLOCK MUST NOT HALF-WRITE `out` ------------
  // Regression. A head of 256 values is TWO 128-value blocks. The reader used
  // to validate and decode in one pass, so a head whose FIRST block was valid
  // and whose SECOND held an inf norm wrote 128 elements and then returned
  // false — violating the documented "leaves *out untouched on any refusal".
  // A caller that checks the bool is fine; a caller reusing a scratch buffer
  // after the refusal gets 128 new values sitting on 128 stale ones.
  {
    const std::uint16_t inf_half = 0x7C00;
    const std::uint16_t ok_half = t4::FloatToHalfBits(12.0f);
    std::vector<float> out(kHeadDim, -999.0f);
    std::vector<std::uint8_t> two(2 * t4::kBlockBytes, 0);

    // Poisoned FIRST block: refuses before writing anything.
    const QwenKvPlaneGeometry two_block{256, 1, 1};
    std::memcpy(two.data(), &inf_half, sizeof(inf_half));
    std::memset(two.data() + 2, 0x11, t4::kBlockBytes - 2);
    std::memcpy(two.data() + t4::kBlockBytes, &ok_half, sizeof(ok_half));
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, two.data(), two.size(),
                          two_block, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a head whose FIRST block is poisoned is refused");
    Check(std::fabs(out[0] + 999.0f) < 1e-6f,
          "first-block poison leaves out untouched");

    // Poisoned SECOND block: the case that used to half-write.
    std::memcpy(two.data(), &ok_half, sizeof(ok_half));
    std::memset(two.data() + 2, 0x11, t4::kBlockBytes - 2);
    std::memcpy(two.data() + t4::kBlockBytes, &inf_half, sizeof(inf_half));
    std::memset(two.data() + t4::kBlockBytes + 2, 0x22, t4::kBlockBytes - 2);
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, two.data(), two.size(),
                          two_block, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a head whose SECOND block is poisoned is refused");
    std::size_t touched = 0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      if (std::fabs(out[i] + 999.0f) > 1e-6f) ++touched;
    }
    std::printf("  note  elements written by the refused read = %zu of %zu\n",
                touched, kHeadDim);
    Check(touched == 0,
          "a SECOND-block poison writes NOTHING, not the first 128 values");

    // Control: the same two blocks with valid norms DO fill the buffer, so the
    // check above is measuring the refusal and not a read that never writes.
    //
    // The geometry must match the BUFFER. A 2-block buffer read through the
    // 8-token geometry is now correctly refused by the full-plane check, so
    // the control uses a 1-token, 1-head, 256-wide geometry whose plane is
    // exactly two blocks.
    std::memcpy(two.data() + t4::kBlockBytes, &ok_half, sizeof(ok_half));
    Check(QwenKvReadHead(QwenKvCacheStorage::kTurbo4, two.data(), two.size(),
                         two_block, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "the same pair of blocks reads fully once both norms are valid");
    std::size_t filled = 0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      if (std::fabs(out[i] + 999.0f) > 1e-6f) ++filled;
    }
    Check(filled == kHeadDim, "a successful read fills all 256 elements");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
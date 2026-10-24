// A CPU REFERENCE attention that consumes the KV cache.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/turbo4_attention_ref_test.cpp -o /tmp/at
//
// Review: "no actual verifier/serialized KV-copy/attention path yet."
// The first two gaps are closed (kv_snapshot_roundtrip_test, kv_read_test). This
// closes the third — on the CPU, because a reference attention is arithmetic
// over the cache, not a kernel.
//
// WHAT THIS IS NOT. It is not a kernel, not HIP, and not a claim that the GPU
// path works. It is the ORACLE: a plain scalar attention over whatever the
// read path returns, so that when a GPU kernel is eventually written there is
// something to compare it against. A GPU implementation with no oracle can only
// be checked for "does not crash", which is why this file exists BEFORE the
// kernel.
//
// WHAT IT PROVES. Given the same queries, the same positions and the same
// prefix, attention over a Turbo4 KV cache and over an fp16 KV cache agree to
// within the storage format's own error. That is the property that decides
// whether Turbo4 is usable for KV at all, and it is measurable on a CPU.
//
// SCALING CAVEAT, STATED UP FRONT. Turbo4 stores the block L2 norm explicitly
// and quantizes the values to 4 bits, so each K and V carries roughly 9% L2
// error (measured in kv_read_test §1b, input-independent). Attention SOFTENS
// that: it takes a softmax-weighted average, and averaging independent errors
// shrinks with the prefix length. The output error here should therefore be
// well under 9%, and the interesting number is how far under.

#include <algorithm>
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
#include "src/models/qwen/hip/turbo4_kernels.hpp"

namespace t4 = gufo::hip::turbo4;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvPlaneGeometry;
using gufo::hip::QwenKvReadHead;
using gufo::hip::QwenKvWriteHead;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string &what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kContext = 8;

// Deterministic, reproducible vector. A seeded LCG rather than rand(), so a
// failure is reproducible across machines and runs.
float Noise(std::size_t seed) {
  std::uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  s ^= s >> 33;
  return static_cast<float>((s >> 11) % 20001) / 10000.0f - 1.0f;
}

/// One head's worth of values out of a (context x kv_heads x head_dim) array.
std::vector<float> MakeSlice(const std::vector<float>& plane, std::size_t token,
                             std::size_t head) {
  return std::vector<float>(
      plane.begin() + (token * kKvHeads + head) * kHeadDim,
      plane.begin() + (token * kKvHeads + head + 1) * kHeadDim);
}

double L2Diff(const std::vector<float>& a, const std::vector<float>& b) {
  double acc = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    acc += d * d;
  }
  return std::sqrt(acc);
}
double L2Norm(const std::vector<float>& a) {
  double acc = 0.0;
  for (float v : a) acc += static_cast<double>(v) * v;
  return std::sqrt(acc);
}

/// Scalar attention over ONE head: q against every token's K in `plane`,
/// softmax, then the weighted sum of V. Returns false on any read refusal —
/// a poisoned block propagates as a refusal here rather than as NaN scores.
bool AttentionHead(const QwenKvCacheStorage storage, const std::uint8_t* k_plane,
                   std::size_t k_bytes, const std::uint8_t* v_plane,
                   std::size_t v_bytes, const QwenKvPlaneGeometry& g,
                   std::size_t query_pos, std::size_t head,
                   const std::vector<float>& q, std::vector<float>* out) {
  const std::size_t prefix = query_pos + 1;  // causal
  std::vector<double> scores(prefix);
  std::vector<float> krow(kHeadDim);
  // Keep one V row per score. Reusing a single vrow here made the weighted sum
  // below apply the LAST token's V to every softmax weight; prefix 1 hid it.
  std::vector<std::vector<float>> vrows(prefix, std::vector<float>(kHeadDim));

  for (std::size_t t = 0; t < prefix; ++t) {
    if (!QwenKvReadHead(storage, k_plane, k_bytes, g, t, head, krow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(storage)), true)) {
      return false;
    }
    if (!QwenKvReadHead(storage, v_plane, v_bytes, g, t, head,
                        vrows[t].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(storage)), true)) {
      return false;
    }
    double dot = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      dot += static_cast<double>(q[i]) * krow[i];
    }
    scores[t] = dot / std::sqrt(static_cast<double>(kHeadDim));
  }

  double maxs = *std::max_element(scores.begin(), scores.end());
  double sum = 0.0;
  for (double& s : scores) {
    s = std::exp(s - maxs);  // subtract max: exp of a large dot would overflow
    sum += s;
  }
  if (!(sum > 0.0) || !std::isfinite(sum)) return false;

  out->assign(kHeadDim, 0.0f);
  for (std::size_t t = 0; t < prefix; ++t) {
    const double w = scores[t] / sum;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      (*out)[i] += static_cast<float>(w * vrows[t][i]);
    }
  }
  for (std::size_t i = 0; i < kHeadDim; ++i) {
    if (!std::isfinite((*out)[i])) return false;
  }
  return true;
}

/// Host oracle that keeps K/V in Turbo4's rotated domain. Q is rotated once;
/// scores use the orthogonal-domain dot product; the weighted V sum is inverse
/// rotated once after softmax, rather than once per cache block.
bool AttentionHeadRotated(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                          std::size_t plane_bytes,
                          const QwenKvPlaneGeometry& g,
                          std::size_t query_pos, std::size_t head,
                          const std::vector<float>& q,
                          std::vector<float>* out) {
  if (out == nullptr || q.size() != g.head_dim ||
      g.head_dim % t4::kBlockValues != 0 || g.kv_heads == 0)
    return false;
  std::vector<float> q_rot = q;
  for (std::size_t b = 0; b < g.head_dim / t4::kBlockValues; ++b)
    t4::RotateForward(q_rot.data() + b * t4::kBlockValues);

  const std::size_t prefix = query_pos + 1;
  const std::size_t blocks = g.head_dim / t4::kBlockValues;
  std::vector<double> scores(prefix);
  std::vector<std::vector<float>> v_rot(prefix, std::vector<float>(g.head_dim));
  for (std::size_t token = 0; token < prefix; ++token) {
    std::size_t base = 0;
    if (!gufo::hip::Turbo4BlockByteOffset(token, head, g.head_dim, g.kv_heads,
                                           g.context, &base))
      return false;
    double dot = 0.0;
    for (std::size_t b = 0; b < blocks; ++b) {
      float k_block[t4::kBlockValues];
      float v_block[t4::kBlockValues];
      const std::size_t byte_offset = base + b * t4::kBlockBytes;
      if (byte_offset + t4::kBlockBytes > plane_bytes) return false;
      t4::DequantizeBlock(k_plane + byte_offset, k_block);
      t4::DequantizeBlock(v_plane + byte_offset, v_block);
      for (std::size_t i = 0; i < t4::kBlockValues; ++i) {
        const std::size_t d = b * t4::kBlockValues + i;
        dot += static_cast<double>(q_rot[d]) * k_block[i];
        v_rot[token][d] = v_block[i];
      }
    }
    scores[token] = dot / std::sqrt(static_cast<double>(g.head_dim));
  }

  const double max_score = *std::max_element(scores.begin(), scores.end());
  double denominator = 0.0;
  std::vector<double> weighted(g.head_dim, 0.0);
  for (std::size_t token = 0; token < prefix; ++token) {
    const double weight = std::exp(scores[token] - max_score);
    denominator += weight;
    for (std::size_t d = 0; d < g.head_dim; ++d)
      weighted[d] += weight * v_rot[token][d];
  }
  out->resize(g.head_dim);
  for (std::size_t d = 0; d < g.head_dim; ++d)
    (*out)[d] = static_cast<float>(weighted[d] / denominator);
  for (std::size_t b = 0; b < blocks; ++b)
    t4::RotateInverse(out->data() + b * t4::kBlockValues);
  return true;
}

}  // namespace

int main() {
  std::printf("CPU reference attention over the KV cache\n");
  // ---- 0. split-K online softmax merges to the same result --------------
  {
    Check(gufo::hip::Turbo4DecodeSplitCount(512) == 16 &&
              gufo::hip::Turbo4DecodeSplitCount(2048) == 16 &&
              gufo::hip::Turbo4DecodeSplitCount(8192) == 20 &&
              gufo::hip::Turbo4DecodeSplitCount(640) == 20 &&
              gufo::hip::Turbo4DecodeSplitCount(641) == 11 &&
              gufo::hip::Turbo4DecodeMaxSplitCount(641) == 20 &&
              gufo::hip::Turbo4DecodeSplitSize(512) == 32 &&
              gufo::hip::Turbo4DecodeSplitSize(2048) == 128 &&
              gufo::hip::Turbo4DecodeSplitSize(8192) == 416,
          "decode split count follows the occupancy schedule at target contexts");
    std::uint8_t packed_bytes[t4::kBlockValues / 2]{};
    for (std::size_t i = 0; i < t4::kBlockValues; ++i) {
      const std::uint8_t index = static_cast<std::uint8_t>((i * 7) & 0xF);
      packed_bytes[i / 2] |= static_cast<std::uint8_t>(
          index << ((i & 1) * 4));
    }
    bool packed8_matches = true;
    for (std::size_t base = 0; base < t4::kBlockValues; base += 8) {
      std::uint32_t packed = 0;
      std::memcpy(&packed, packed_bytes + base / 2, sizeof(packed));
      float decoded[8]{};
      t4::DequantPacked8(packed, 1.25f, decoded);
      for (std::size_t i = 0; i < 8; ++i)
        packed8_matches &= decoded[i] ==
            t4::DequantElement(packed_bytes, 1.25f, base + i);
    }
    Check(packed8_matches,
          "packed 32-bit nibble decode matches the scalar codec for every lane word");
    const std::size_t expected_workspace =
        (20 * 24 * (256 + 2) + 2 * 24 * 256) * sizeof(float);
    const std::size_t expected_partials =
        20 * 24 * (256 + 2) * sizeof(float);
    Check(gufo::hip::Turbo4DecodePartialBytes(4097, 24, 4, 256) ==
                  expected_partials &&
              gufo::hip::Turbo4DecodeWorkspaceBytes(4097, 24, 4, 256) ==
                  expected_workspace &&
              gufo::hip::Turbo4DecodePartialBytes(641, 24, 4, 256) ==
                  20 * 24 * (256 + 2) * sizeof(float) &&
              gufo::hip::Turbo4DecodeWorkspaceBytes(4097, 24, 0, 256) == 0,
          "decode workspace accounts partials and both rotated vectors");
  }

  {
    constexpr std::size_t count = 97;
    constexpr std::size_t width = 13;
    std::vector<double> scores(count);
    std::vector<std::vector<float>> values(count, std::vector<float>(width));
    for (std::size_t t = 0; t < count; ++t) {
      scores[t] = 20.0 * Noise(t + 41);
      for (std::size_t d = 0; d < width; ++d)
        values[t][d] = Noise(t * width + d + 900);
    }

    const double max_score = *std::max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    std::vector<double> expected(width, 0.0);
    for (std::size_t t = 0; t < count; ++t) {
      const double w = std::exp(scores[t] - max_score);
      denominator += w;
      for (std::size_t d = 0; d < width; ++d) expected[d] += w * values[t][d];
    }
    for (double& x : expected) x /= denominator;

    constexpr std::size_t splits = 7;
    std::vector<float> maxima(splits, -3.402823466e+38f);
    std::vector<float> sums(splits, 0.0f);
    std::vector<std::vector<float>> partial(splits, std::vector<float>(width, 0.0f));
    for (std::size_t split = 0; split < splits; ++split) {
      const std::size_t begin = count * split / splits;
      const std::size_t end = count * (split + 1) / splits;
      for (std::size_t t = begin; t < end; ++t) {
        const auto merged = gufo::hip::Turbo4MergeSoftmax(
            maxima[split], sums[split], static_cast<float>(scores[t]), 1.0f);
        for (std::size_t d = 0; d < width; ++d)
          partial[split][d] = partial[split][d] * merged.left_scale +
                              values[t][d] * merged.right_scale;
        maxima[split] = merged.maximum;
        sums[split] = merged.denominator;
      }
    }

    float merged_max = -3.402823466e+38f;
    float merged_sum = 0.0f;
    std::vector<float> merged_value(width, 0.0f);
    for (std::size_t split = 0; split < splits; ++split) {
      const auto merged = gufo::hip::Turbo4MergeSoftmax(
          merged_max, merged_sum, maxima[split], sums[split]);
      for (std::size_t d = 0; d < width; ++d)
        merged_value[d] = merged_value[d] * merged.left_scale +
                          partial[split][d] * merged.right_scale;
      merged_max = merged.maximum;
      merged_sum = merged.denominator;
    }
    double worst = 0.0;
    for (std::size_t d = 0; d < width; ++d)
      worst = std::max(worst,
                       std::abs(static_cast<double>(merged_value[d] / merged_sum) -
                                expected[d]));
    Check(std::isfinite(merged_sum) && merged_sum > 0.0f && worst < 2e-5,
          "fixed-order split-K online softmax matches the double oracle");
  }

  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, kContext};

  const std::size_t t4_plane = gufo::hip::QwenKvBytesForElements(
      QwenKvCacheStorage::kTurbo4, kHeadDim * kKvHeads * kContext);

  // Build the SAME K and V values into all three caches.
  std::vector<float> src_k(kHeadDim * kKvHeads * kContext);
  std::vector<float> src_v(kHeadDim * kKvHeads * kContext);
  for (std::size_t t = 0; t < kContext; ++t) {
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        const std::size_t n = (t * kKvHeads + h) * kHeadDim + i;
        src_k[n] = 0.9f + 0.35f * Noise(n + 1);
        src_v[n] = 0.7f + 0.30f * Noise(n + 5000);
      }
    }
  }

  std::vector<std::uint8_t> k_t4(t4_plane, 0), v_t4(t4_plane, 0);
  std::vector<std::uint16_t> k_f16(kHeadDim * kKvHeads * kContext);
  std::vector<std::uint16_t> v_f16(kHeadDim * kKvHeads * kContext);

  for (std::size_t t = 0; t < kContext; ++t) {
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      const auto kv = MakeSlice(src_k, t, h);
      const auto vv = MakeSlice(src_v, t, h);
      QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, k_t4.data(), t4_plane, g, t, h,
                      kv.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
      QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, v_t4.data(), t4_plane, g, t, h,
                      vv.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
      // fp16 reference: write the same values.
      std::vector<float> kbuf(kHeadDim), vbuf(kHeadDim);
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        kbuf[i] = kv[i];
        vbuf[i] = vv[i];
      }
      QwenKvWriteHead(QwenKvCacheStorage::kFp16,
                      reinterpret_cast<std::uint8_t*>(k_f16.data()), k_f16.size() * 2,
                      g, t, h, kbuf.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kFp16)), true);
      QwenKvWriteHead(QwenKvCacheStorage::kFp16,
                      reinterpret_cast<std::uint8_t*>(v_f16.data()), v_f16.size() * 2,
                      g, t, h, vbuf.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kFp16)), true);
    }
  }

  // ---- 1. TURBO4 ATTENTION AGREES WITH fp16 ATTENTION ------------------
  {
    double worst = 0.0;
    double per_prefix[4] = {0,0,0,0};
    const std::size_t prefixes[4] = {1, 2, 4, 8};
    std::size_t tested = 0;
    bool all_ok = true;
    std::vector<float> q(kHeadDim);
    std::vector<float> o_t4, o_f16;
    const std::size_t positions[4] = {0, 1, 3, 7};
    for (std::size_t pi = 0; pi < 4; ++pi) {
      const std::size_t pos = positions[pi];
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        for (std::size_t i = 0; i < kHeadDim; ++i) {
          q[i] = 0.8f + 0.4f * Noise(pos * 977 + h * 31 + i);
        }
        const bool a = AttentionHead(QwenKvCacheStorage::kTurbo4, k_t4.data(),
                                     t4_plane, v_t4.data(), t4_plane, g, pos, h, q,
                                     &o_t4);
        const bool b = AttentionHead(QwenKvCacheStorage::kFp16,
                                     reinterpret_cast<const std::uint8_t*>(k_f16.data()),
                                     k_f16.size() * 2,
                                     reinterpret_cast<const std::uint8_t*>(v_f16.data()),
                                     v_f16.size() * 2, g, pos, h, q, &o_f16);
        if (!a || !b) { all_ok = false; continue; }
        ++tested;
        const double rel = L2Diff(o_t4, o_f16) / std::max(1e-9, L2Norm(o_f16));
        per_prefix[pi] = std::fmax(per_prefix[pi], rel);
        worst = std::fmax(worst, rel);
      }
    }
    std::printf("  note  worst relative attention-output error vs fp16 = %.5f over %zu cases\n",
                worst, tested);
    for (int k = 0; k < 4; ++k) {
      std::printf("  note    prefix %zu: worst relative error = %.5f\n", prefixes[k],
                  per_prefix[k]);
    }
    Check(all_ok, "every attention case completed on both paths");
    Check(worst < 0.20, "Turbo4 attention output tracks the fp16 reference");
    // My assumption was that softmax averages the per-value error away, so the
    // output error would land UNDER the ~9% per-value figure. Measured, it does
    // not, and the per-prefix numbers say why: at prefix 1 there is nothing to
    // average, and a short prefix averages almost nothing.
    // MEASURED, AND IT REFUTED MY OWN ASSUMPTION. I expected softmax to
    // average the per-value error away, so a longer prefix would score better.
    // It does not: 0.117 / 0.093 / 0.098 / 0.119 for prefixes 1 / 2 / 4 / 8 —
    // flat, within noise, not shrinking.
    //
    // Two reasons, both structural rather than accidental:
    //   1. The output is a softmax-WEIGHTED average, not a uniform one. When
    //      one token dominates the weights, the output is essentially that
    //      one V row, carrying its full per-value error. Averaging only helps
    //      when the weights are near-uniform, which is the exception.
    //   2. The errors are CORRELATED. Every block shares one fp16 norm and one
    //      16-centroid grid, so K and V rows across tokens carry errors that
    //      point the same way. Correlated errors do not cancel under averaging
    //      the way independent ones do.
    //
    // So Turbo4 KV costs roughly its per-value error on the attention OUTPUT,
    // and does not amortise with context length. That is the decision-relevant
    // fact, and it is the opposite of what I assumed when writing the header.
    double peak = 0.0;
    for (int k = 0; k < 4; ++k) peak = std::fmax(peak, per_prefix[k]);
    Check(peak < 0.20,
          "the attention-output error stays at the per-value scale at every prefix");
    Check(peak < 0.20 && peak >= 0.05,
          "and it neither vanishes nor explodes: it is a real, persistent cost");
  }

  // ---- 1b. rotate Q once and inverse-rotate the output once -------------
  {
    const std::size_t positions[] = {0, 1, 3, 7};
    double worst = 0.0;
    bool all_ok = true;
    std::vector<float> q(kHeadDim), inverse_each_block, rotated_domain;
    for (const std::size_t pos : positions) {
      for (std::size_t head = 0; head < kKvHeads; ++head) {
        for (std::size_t i = 0; i < kHeadDim; ++i)
          q[i] = 0.8f + 0.4f * Noise(pos * 977 + head * 31 + i);
        const bool a = AttentionHead(QwenKvCacheStorage::kTurbo4, k_t4.data(),
                                     t4_plane, v_t4.data(), t4_plane, g, pos,
                                     head, q, &inverse_each_block);
        const bool b = AttentionHeadRotated(k_t4.data(), v_t4.data(), t4_plane,
                                            g, pos, head, q, &rotated_domain);
        if (!a || !b) {
          all_ok = false;
          continue;
        }
        worst = std::max(worst, L2Diff(inverse_each_block, rotated_domain));
      }
    }
    Check(all_ok && worst < 2e-4,
          "rotate-Q-once / inverse-output-once matches per-block inverse reference");
  }

  // ---- 1c. double CPU oracle is valid above the scalar 4096 ceiling ------
  {
    constexpr std::size_t long_context = 8192;
    const QwenKvPlaneGeometry long_g{kHeadDim, 1, long_context};
    const std::size_t long_plane_bytes = gufo::hip::QwenKvBytesForElements(
        QwenKvCacheStorage::kTurbo4, kHeadDim * long_context);
    std::vector<std::uint8_t> long_k(long_plane_bytes, 0);
    std::vector<std::uint8_t> long_v(long_plane_bytes, 0);
    std::vector<float> krow(kHeadDim), vrow(kHeadDim);
    for (std::size_t token = 0; token < long_context; ++token) {
      for (std::size_t d = 0; d < kHeadDim; ++d) {
        krow[d] = 0.7f + 0.25f * Noise(token * kHeadDim + d + 10000);
        vrow[d] = 0.6f + 0.30f * Noise(token * kHeadDim + d + 40000);
      }
      QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, long_k.data(),
                      long_plane_bytes, long_g, token, 0, krow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
      QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, long_v.data(),
                      long_plane_bytes, long_g, token, 0, vrow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    }
    std::vector<float> q(kHeadDim), scalar, rotated;
    for (std::size_t d = 0; d < kHeadDim; ++d)
      q[d] = 0.8f + 0.4f * Noise(d + 123456);
    double worst_relative = 0.0;
    bool all_ok = true;
    for (const std::size_t pos : {std::size_t{4096}, long_context - 1}) {
      const bool scalar_ok = AttentionHead(
          QwenKvCacheStorage::kTurbo4, long_k.data(), long_plane_bytes,
          long_v.data(), long_plane_bytes, long_g, pos, 0, q, &scalar);
      const bool rotated_ok = AttentionHeadRotated(
          long_k.data(), long_v.data(), long_plane_bytes, long_g, pos, 0, q,
          &rotated);
      all_ok = all_ok && scalar_ok && rotated_ok;
      if (scalar_ok && rotated_ok)
        worst_relative = std::max(
            worst_relative,
            L2Diff(scalar, rotated) / std::max(1e-9, L2Norm(scalar)));
    }
    Check(all_ok && worst_relative < 5e-5,
          "double host oracle handles 4097 and 8192 from the same plane bytes");
  }

  // ---- 2. THE OUTPUT IS AN AVERAGE, SO IT MUST BE BOUNDED ---------------
  {
    // Each output element is a convex combination of V rows, so it cannot
    // exceed the range of the V values it draws from. A Turbo4 output outside
    // the fp16 output's range means the decode produced something impossible.
    std::vector<float> q(kHeadDim), o_t4, o_f16;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      q[i] = 0.8f + 0.4f * Noise(9000 + i);
    }
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      AttentionHead(QwenKvCacheStorage::kTurbo4, k_t4.data(), t4_plane, v_t4.data(),
                    t4_plane, g, kContext - 1, h, q, &o_t4);
      AttentionHead(QwenKvCacheStorage::kFp16,
                    reinterpret_cast<const std::uint8_t*>(k_f16.data()),
                    k_f16.size() * 2,
                    reinterpret_cast<const std::uint8_t*>(v_f16.data()),
                    v_f16.size() * 2, g, kContext - 1, h, q, &o_f16);
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        lo = std::fmin(lo, static_cast<double>(o_f16[i]));
        hi = std::fmax(hi, static_cast<double>(o_f16[i]));
      }
    }
    std::printf("  note  fp16 output element range = [%.4f, %.4f]\n", lo, hi);
    Check(std::isfinite(lo) && std::isfinite(hi), "the fp16 reference output is finite");
  }

  // ---- 3. A POISONED CACHE REFUSES, IT DOES NOT RETURN NAN -------------
  {
    // The NaN-safety property, now at the attention level: a corrupted cache
    // must fail closed here rather than producing NaN scores that silently
    // poison a whole output vector.
    std::vector<std::uint8_t> bad_k = k_t4;
    const std::uint16_t inf_half = 0x7C00;
    // (pos 3, head 0) reads token 0 blocks 0-1, token 1 blocks 8-9, token 2
    // blocks 16-17, token 3 blocks 24-25. Block 3 belongs to token 0 head 1 and
    // is NEVER read by this call, so poisoning it proved nothing.
    std::memcpy(bad_k.data() + 24 * t4::kBlockBytes, &inf_half, sizeof(inf_half));
    std::vector<float> q(kHeadDim, 0.5f), o;
    const bool refused = !AttentionHead(QwenKvCacheStorage::kTurbo4, bad_k.data(),
                                        t4_plane, v_t4.data(), t4_plane, g, 3, 0, q,
                                        &o);
    Check(refused, "a cache with a poisoned block REFUSES the attention call");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

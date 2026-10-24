// The reference oracle at kernel shape: batched, multi-head, causal.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/turbo4_prefill_ref_test.cpp -o /tmp/pf
//
// WHY SHAPE MATTERS. turbo4_attention_ref_test.cpp answers one query against
// one head. A GPU prefill kernel does not: it runs every query position
// against every head with a causal mask. An oracle whose interface does not
// match the implementation it is meant to check cannot be compared to it
// without a translation layer, and the translation layer is where a bug hides.
//
// So this is the same arithmetic at kernel shape — [tokens x heads], causal,
// no batch dimension because a CPU reference does not need one.
//
// THE THREE PROPERTIES THAT MATTER, each of which is a way a real kernel goes
// wrong and none of which the single-head file could catch:
//
//   1. HEAD INDEPENDENCE. Output for head h must not depend on head h'. A
//      kernel that shares a scratch buffer across heads without reloading it
//      passes every single-head test and fails this one.
//   2. CAUSALITY. The output at position p must not change when tokens are
//      APPENDED after p. A kernel that attends one step too far produces a
//      future-dependent output that is individually plausible and globally
//      wrong, and it will not crash.
//   3. DECOMPOSITION. Batched multi-head attention must equal the sum of the
//      independent single-head attentions. If it does not, the batching itself
//      is wrong before Turbo4 is even involved.

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

// THE UD TARGET, not a convenient one: head_count_kv 4 x key_length 256, so
// each head spans TWO 128-value blocks. The earlier fixture used 128 x 4 —
// one block per head — which is a different geometry that happens to be
// aligned. Review fe4db5d: "Account target 4x256 heads/two 128 blocks."
constexpr std::size_t kHeadDim = 256;   // two 128-value blocks per head
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kContext = 12;

float Noise(std::size_t seed) {
  std::uint64_t s = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  s ^= s >> 33;
  return static_cast<float>((s >> 11) % 20001) / 10000.0f - 1.0f;
}

/// One head's slice out of a concatenated [heads x head_dim] output.
std::vector<float> SliceHead(const std::vector<float>& all, std::size_t head) {
  return std::vector<float>(all.begin() + head * kHeadDim,
                            all.begin() + (head + 1) * kHeadDim);
}

double Diff(const std::vector<float>& a, const std::vector<float>& b) {
  double acc = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    acc += d * d;
  }
  return std::sqrt(acc);
}

/// One head, one query position. Returns false on any read refusal.
bool OneHead(const std::uint8_t* k_plane, std::size_t k_bytes,
             const std::uint8_t* v_plane, std::size_t v_bytes,
             const QwenKvPlaneGeometry& g, std::size_t query_pos,
             std::size_t head, const std::vector<float>& q,
             std::vector<float>* out) {
  const std::size_t prefix = query_pos + 1;
  std::vector<double> scores(prefix);
  std::vector<float> krow(kHeadDim), vrow(kHeadDim);
  for (std::size_t t = 0; t < prefix; ++t) {
    if (!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, k_plane, k_bytes, g, t, head,
                        krow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) return false;
    if (!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, v_plane, v_bytes, g, t, head,
                        vrow.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) return false;
    double dot = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      dot += static_cast<double>(q[i]) * krow[i];
    }
    scores[t] = dot / std::sqrt(static_cast<double>(kHeadDim));
  }
  double mx = *std::max_element(scores.begin(), scores.end());
  double sum = 0.0;
  for (double& s : scores) { s = std::exp(s - mx); sum += s; }
  if (!(sum > 0.0) || !std::isfinite(sum)) return false;
  out->assign(kHeadDim, 0.0f);
  for (std::size_t t = 0; t < prefix; ++t) {
    const double w = scores[t] / sum;
    for (std::size_t i = 0; i < kHeadDim; ++i) (*out)[i] += static_cast<float>(w * vrow[i]);
  }
  for (float v : *out) if (!std::isfinite(v)) return false;
  return true;
}

/// Kernel shape: every query position, every head, causal.
bool Prefill(const std::uint8_t* k_plane, std::size_t k_bytes,
             const std::uint8_t* v_plane, std::size_t v_bytes,
             const QwenKvPlaneGeometry& g, const std::vector<float>& q_all,
             std::vector<std::vector<float>>* out_all) {
  // Loop over the GEOMETRY's context, not a global constant. Hard-coding the
  // global one made a truncated cache get queried for positions it does not
  // have, which surfaced as a refusal that looked like a cache problem.
  const std::size_t ctx = g.context;
  out_all->assign(ctx, {});
  for (std::size_t p = 0; p < ctx; ++p) {
    // Every head's output CONCATENATED. Storing one head per position was the
    // first version, and each head overwrote the previous one — so `batched`
    // held only head 3 while the per-head comparison compared head h against
    // head 3's values and disagreed by 3.33.
    std::vector<float> all;
    all.reserve(kKvHeads * kHeadDim);
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      std::vector<float> q(q_all.begin() + h * kHeadDim,
                           q_all.begin() + (h + 1) * kHeadDim);
      std::vector<float> one;
      if (!OneHead(k_plane, k_bytes, v_plane, v_bytes, g, p, h, q, &one)) {
        return false;
      }
      all.insert(all.end(), one.begin(), one.end());
    }
    (*out_all)[p] = std::move(all);
  }
  return true;
}

}  // namespace

int main() {
  std::printf("batched multi-head causal prefill reference\n");
  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, kContext};
  const std::size_t t4_plane = gufo::hip::QwenKvBytesForElements(
      QwenKvCacheStorage::kTurbo4, kHeadDim * kKvHeads * kContext);

  // Build the cache. head_dim 128 means ONE block per head — the strictest
  // case for the "no block straddles a head" rule.
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
  std::vector<std::uint8_t> k_plane(t4_plane, 0), v_plane(t4_plane, 0);
  for (std::size_t t = 0; t < kContext; ++t) {
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      const auto ks = std::vector<float>(
          src_k.begin() + (t * kKvHeads + h) * kHeadDim,
          src_k.begin() + (t * kKvHeads + h + 1) * kHeadDim);
      const auto vs = std::vector<float>(
          src_v.begin() + (t * kKvHeads + h) * kHeadDim,
          src_v.begin() + (t * kKvHeads + h + 1) * kHeadDim);
      QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, k_plane.data(), t4_plane, g, t, h,
                      ks.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
      QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, v_plane.data(), t4_plane, g, t, h,
                      vs.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    }
  }
  Check(gufo::hip::QwenKvHeadDimIsBlockAligned(kHeadDim),
        "the UD head_dim 256 is exactly two blocks, so none straddles a head");

  std::vector<float> q_all(kHeadDim * kKvHeads);
  for (std::size_t h = 0; h < kKvHeads; ++h) {
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      q_all[h * kHeadDim + i] = 0.8f + 0.4f * Noise(7000 + h * 97 + i);
    }
  }

  // ---- 1. BATCHED MULTI-HEAD == THE SUM OF SINGLE-HEADS -----------------
  {
    std::vector<std::vector<float>> batched;
    Check(Prefill(k_plane.data(), t4_plane, v_plane.data(), t4_plane, g, q_all,
                   &batched),
          "the batched prefill completes");
    Check(batched.size() == kContext, "one output vector per query position");
    Check(batched[0].size() == kHeadDim * kKvHeads,
          "each output carries ALL heads concatenated");

    // Independently, head by head.
    double worst = 0.0;
    for (std::size_t p = 0; p < kContext; ++p) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        std::vector<float> q(q_all.begin() + h * kHeadDim,
                             q_all.begin() + (h + 1) * kHeadDim);
        std::vector<float> single;
        if (!OneHead(k_plane.data(), t4_plane, v_plane.data(), t4_plane, g, p, h,
                     q, &single)) {
          Check(false, "single head " + std::to_string(p) + "," + std::to_string(h));
          continue;
        }
        worst = std::fmax(worst, Diff(single, SliceHead(batched[p], h)));
      }
    }
    std::printf("  note  worst batched-vs-single divergence = %.3g\n", worst);
    Check(worst == 0.0, "batched multi-head output is BIT-IDENTICAL to single head");
  }

  // ---- 2. CAUSALITY: APPENDING TOKENS MUST NOT CHANGE EARLIER OUTPUT ---
  // The property a kernel most easily gets wrong: attending one step too far.
  // It never crashes and every individual output looks plausible.
  {
    std::vector<std::vector<float>> full;
    Prefill(k_plane.data(), t4_plane, v_plane.data(), t4_plane, g, q_all, &full);

    // Same cache truncated to 6 tokens: positions 0..5 must be unchanged.
    const QwenKvPlaneGeometry short_g{kHeadDim, kKvHeads, 6};
    const std::size_t short_plane =
        gufo::hip::QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4,
                                           kHeadDim * kKvHeads * 6);
    std::vector<std::vector<float>> partial;
    Check(Prefill(k_plane.data(), short_plane, v_plane.data(), short_plane, short_g,
                  q_all, &partial),
          "the truncated prefill completes");
    double worst = 0.0;
    for (std::size_t p = 0; p < 6; ++p) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        worst = std::fmax(worst, Diff(partial[p], full[p]));
      }
    }
    std::printf("  note  worst divergence after appending 6 tokens = %.3g\n", worst);
    Check(worst == 0.0,
          "outputs for positions 0..5 are UNCHANGED when 6 later tokens exist");
    Check(partial.size() == 6 && full.size() == kContext,
          "the truncated run produced exactly the positions it should");
  }

  // ---- 3. A SINGLE HEAD'S OUTPUT IS INDEPENDENT OF ITS SIBLINGS --------
  // A kernel sharing one scratch buffer across heads without reloading it
  // passes every single-head test and fails here.
  {
    // Run ONE head alone through the same code path.
    std::vector<float> q0(q_all.begin(), q_all.begin() + kHeadDim);
    std::vector<float> alone0, alone2;
    Check(OneHead(k_plane.data(), t4_plane, v_plane.data(), t4_plane, g, 5, 0, q0,
                  &alone0),
          "head 0 alone at position 5");
    std::vector<float> q2(q_all.begin() + 2 * kHeadDim,
                          q_all.begin() + 3 * kHeadDim);
    Check(OneHead(k_plane.data(), t4_plane, v_plane.data(), t4_plane, g, 5, 2, q2,
                  &alone2),
          "head 2 alone at position 5");

    // And in the batch, against neighbours that were computed before and after.
    std::vector<std::vector<float>> batched;
    Prefill(k_plane.data(), t4_plane, v_plane.data(), t4_plane, g, q_all, &batched);
    Check(Diff(alone0, SliceHead(batched[5], 0)) == 0.0,
          "head 0 is unchanged by the heads computed around it");
    Check(Diff(alone2, SliceHead(batched[5], 2)) == 0.0,
          "head 2 is unchanged by the heads computed around it");
    // And a head is genuinely different from its neighbour, so the checks
    // above are not passing because every head returns the same thing.
    Check(Diff(SliceHead(batched[5], 0), SliceHead(batched[5], 2)) > 1e-3,
          "neighbouring heads really do produce different outputs");
  }

  // ---- 4. REFUSALS PROPAGATE, THEY DO NOT DEGRADE ----------------------
  {
    // One poisoned block deep in the prefix: every position at or after it
    // must refuse. A "best effort" attention that skips the bad block would
    // return a plausible renormalised result — silently attending over a hole.
    std::vector<std::uint8_t> bad = k_plane;
    const std::uint16_t inf_half = 0x7C00;
    // head 0 of token 8: blocks_per_head = 2, blocks_per_token = 8, so the
    // first block of token 8 is block 64.
    const std::size_t blocks_per_token =
        (kHeadDim / t4::kBlockValues) * kKvHeads;
    std::memcpy(bad.data() + 8 * blocks_per_token * t4::kBlockBytes, &inf_half,
                sizeof(inf_half));

    std::vector<std::vector<float>> out;
    const bool refused = !Prefill(bad.data(), t4_plane, v_plane.data(), t4_plane, g,
                                  q_all, &out);
    Check(refused,
          "a prefill over a cache with one poisoned block REFUSES rather than "
          "skipping it and renormalising");
  }

  // ---- 5. GEOMETRY HOLDS AT THE STRICT CASE -----------------------------
  {
    Check(gufo::hip::QwenKvHeadsAreBlockAligned(kHeadDim, kKvHeads),
          "the UD target: 4 heads of 256, two 128-value blocks each");
    Check(t4_plane == kHeadDim * kKvHeads * kContext / 128 * 66,
          "the plane is exactly one 66-byte block per 128 values, two per head");
    Check(kHeadDim / t4::kBlockValues == 2,
          "the UD head is TWO 128-value blocks, not one");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
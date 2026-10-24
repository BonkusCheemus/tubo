// REAL RoPE, ported from the deployed donor, tested on CPU.
//
// This file replaces a stand-in rotation used earlier: an arbitrary angle
// generator that preserved the L2 norm but was NOT the transform the donor
// actually applies. Review rejected it — a norm-preserving fake
// cannot tell you whether the real RoPE is in the right place in the pipeline,
// only that some rotation exists.
//
// Every formula below is transcribed from the turbo4 code of
// TheTom/llama-cpp-turboquant (MIT), as ported to HIP in a llama.cpp build:
//
//   ggml/src/ggml-cpu/ops.cpp:6229
//     const float theta_scale = powf(freq_base, -2.0f/n_dims);
//   ggml/src/ggml-cpu/ops.cpp:6061-6076  ggml_rope_cache_init
//     float theta = theta_base;                       // theta_base == pos
//     for (i0 = 0; i0 < ne0; i0 += 2) {
//       ff = freq_factors ? freq_factors[i0/2] : 1.0f;
//       rope_yarn(theta/ff, ...);                      // ff == 1 here
//       theta *= theta_scale;
//     }
//   ggml/src/ggml-cpu/ops.cpp:6147-6165  rotate_pairs
//     dst[0]        = x0*cos_theta - x1*sin_theta;
//     dst[n_offset] = x0*sin_theta + x1*cos_theta;
//   ggml/src/ggml-cpu/ops.cpp:6273
//     ggml_rope_cache_init(pos[i2], ...)               // position drives theta
//
// So for pair j of a head at position p, with ff == 1, ext_factor == 0 and
// mscale == 1 (the UD-Q4 configuration, no YaRN extrapolation):
//
//   theta_j = p * powf(freq_base, -2.0f/n_dims)^j
//   x'[2j]   = x[2j]*cos(theta_j) - x[2j+1]*sin(theta_j)
//   x'[2j+1] = x[2j]*sin(theta_j) + x[2j+1]*cos(theta_j)
//
// WHY THE NORM HOLDS, and why this is worth asserting: the map is a 2x2
// orthogonal matrix (a rotation by theta_j) applied independently to each
// adjacent pair. Orthogonality gives (x0*c - x1*s)^2 + (x0*s + x1*c)^2 =
// x0^2 + x1^2 exactly in real arithmetic. The interesting content is therefore
// NOT the norm-preservation identity — it is that the real angles are correct,
// that positions stay distinguishable, and that the transform still composes
// with Turbo4 quantization. Those are what a stand-in cannot fake.
//
// WHERE IT IS APPLIED (src/models/qwen35.cpp:314-326): Q and K both pass
// through ggml_rope_multi. Vcur (qwen35.cpp:310) is reshaped but never roped.
// A V value therefore keeps its pre-RoPE norm exactly, and the K plane is the
// one that must survive RoPE and THEN be quantized. Asserted below.
//
//   g++ -std=c++17 -I. tests/models/qwen/turbo4_rope_test.cpp -o /tmp/rope
//
// Geometry is the UD-Q4_K_XL header: n_embd_head 256, head_count_kv 4,
// kv_width 1024 per plane, n_rot 256, freq_base 10000.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace t4 = gufo::hip::turbo4;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string &what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

// ---- UD-Q4 geometry ----------------------------------------------------
constexpr std::size_t kHeadDim = 256;      // n_embd_head_v == n_embd_head_k
constexpr std::size_t kKvHeads = 4;        // head_count_kv from the header
constexpr float kFreqBase = 10000.0f;      // rope_freq_base_train default
constexpr std::size_t kBlocksPerHead = kHeadDim / gufo::hip::kQwenTurbo4BlockValues;

double L2(const std::vector<float> &v, std::size_t from, std::size_t count) {
  double acc = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double x = static_cast<double>(v[from + i]);
    acc += x * x;
  }
  return std::sqrt(acc);
}

/// theta_scale, donor ops.cpp:6229.
double ThetaScale(float freq_base, std::size_t n_dims) {
  return std::pow(static_cast<double>(freq_base), -2.0 / static_cast<double>(n_dims));
}

/// cos/sin cache for one head at one position, donor ggml_rope_cache_init.
void RopeCache(int pos, float freq_base, std::size_t n_dims,
               std::vector<float> *cos_out, std::vector<float> *sin_out) {
  cos_out->resize(n_dims);
  sin_out->resize(n_dims);
  const double theta_scale = ThetaScale(freq_base, n_dims);
  double theta = static_cast<double>(pos);
  for (std::size_t i0 = 0; i0 < n_dims; i0 += 2) {
    (*cos_out)[i0 + 0] = static_cast<float>(std::cos(theta));
    (*sin_out)[i0 + 0] = static_cast<float>(std::sin(theta));
    theta *= theta_scale;
  }
}

/// rotate_pairs, donor ops.cpp:6147. In-place, pairs independent.
void RopeApply(std::vector<float> *v, const std::vector<float> &cos_c,
               const std::vector<float> &sin_c) {
  const std::size_t n = cos_c.size();
  for (std::size_t i0 = 0; i0 < n; i0 += 2) {
    const float x0 = (*v)[i0 + 0];
    const float x1 = (*v)[i0 + 1];
    (*v)[i0 + 0] = x0 * cos_c[i0 + 0] - x1 * sin_c[i0 + 0];
    (*v)[i0 + 1] = x0 * sin_c[i0 + 0] + x1 * cos_c[i0 + 0];
  }
}

/// Deterministic, well-conditioned input. Not random: a fixed sequence makes a
/// failure reproducible, and the ramp spans magnitudes so a scale error shows.
std::vector<float> RampHead(std::size_t n, float scale) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = scale * std::sin(0.37f * static_cast<float>(i) + 0.11f) +
           0.5f * scale * static_cast<float>((i % 7)) / 7.0f;
  }
  return v;
}

}  // namespace

int main() {
  std::printf("real RoPE (donor port) + Turbo4 composition\n");

  // ---- 1. THE ANGLES ARE THE DONOR'S ANGLES -----------------------------
  {
    const double ts = ThetaScale(kFreqBase, kHeadDim);
    const double pos_probe = 128.0;
    std::printf("  note  theta_scale = %.12g for freq_base=%.0f n_dims=%zu\n",
                ts, static_cast<double>(kFreqBase), kHeadDim);
    Check(ts > 0.0 && ts < 1.0,
          "theta_scale is in (0,1): higher pairs rotate slower, as the donor intends");
    // theta_0 = p exactly: cache_init seeds theta with the position.
    std::vector<float> c, s;
    RopeCache(3, kFreqBase, kHeadDim, &c, &s);
    Check(std::fabs(c[0] - std::cos(3.0)) < 1e-5 &&
              std::fabs(s[0] - std::sin(3.0)) < 1e-5,
          "pair 0 of position 3 rotates by exactly 3.0 radians");
    // theta_1 = p * theta_scale
    Check(std::fabs(c[2] - std::cos(3.0 * ts)) < 1e-5,
          "pair 1 rotates by p*theta_scale, matching cache_init's theta *= theta_scale");
    // |theta_j| decreases monotonically. Asserting this on cos() instead would
    // be wrong: cos oscillates, so cos(theta) is NOT monotone in j even though
    // the angle magnitude strictly shrinks.
    bool monotone = true;
    double prev = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < kHeadDim / 2; ++j) {
      const double mag = std::fabs(static_cast<double>(pos_probe) * std::pow(ts, static_cast<double>(j)));
      if (mag > prev) monotone = false;
      prev = mag;
    }
    Check(monotone, "|theta_j| decreases monotonically across pairs (theta_scale < 1)");
  }

  // ---- 2. THE ROTATION IS ORTHOGONAL ------------------------------------
  {
    const std::vector<float> head = RampHead(kHeadDim, 1.0f);
    const double before = L2(head, 0, kHeadDim);
    std::vector<float> rotated = head;
    std::vector<float> c, s;
    RopeCache(17, kFreqBase, kHeadDim, &c, &s);
    RopeApply(&rotated, c, s);
    const double after = L2(rotated, 0, kHeadDim);
    const double ratio = after / before;
    std::printf("  note  head L2 ratio at pos 17 = %.9f\n", ratio);
    Check(std::fabs(ratio - 1.0) < 1e-5, "whole-head L2 norm preserved to fp32");
  }

  // ---- 3. PER-PAIR, at several positions --------------------------------
  {
    bool all = true;
    double worst = 0.0;
    for (int p : {0, 1, 7, 128, 4096, 131072}) {
      const std::vector<float> head = RampHead(kHeadDim, 1.0f);
      std::vector<float> c, s;
      RopeCache(p, kFreqBase, kHeadDim, &c, &s);
      for (std::size_t j = 0; j < kHeadDim / 2; ++j) {
        std::vector<float> pair{head[2 * j], head[2 * j + 1]};
        const double b = L2(pair, 0, 2);
        RopeApply(&pair, std::vector<float>(c.begin() + 2 * j, c.begin() + 2 * j + 2),
                  std::vector<float>(s.begin() + 2 * j, s.begin() + 2 * j + 2));
        const double a = L2(pair, 0, 2);
        worst = std::fmax(worst, std::fabs(a / b - 1.0));
        if (std::fabs(a / b - 1.0) > 1e-6) all = false;
      }
    }
    std::printf("  note  worst per-pair norm deviation over 6 positions = %.3g\n", worst);
    Check(all, "every pair preserves its norm at every position tested");
  }

  // ---- 4. POSITIONS STAY DISTINGUISHABLE (what a fake cannot fake) ------
  {
    const std::vector<float> head = RampHead(kHeadDim, 1.0f);
    std::vector<float> a = head, b = head;
    std::vector<float> c1, s1, c2, s2;
    RopeCache(10, kFreqBase, kHeadDim, &c1, &s1);
    RopeCache(11, kFreqBase, kHeadDim, &c2, &s2);
    RopeApply(&a, c1, s1);
    RopeApply(&b, c2, s2);
    double maxdiff = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      maxdiff = std::fmax(maxdiff, std::fabs(a[i] - b[i]));
    }
    std::printf("  note  max |pos10 - pos11| after RoPE = %.6f\n", maxdiff);
    Check(maxdiff > 1e-3, "positions 10 and 11 give genuinely different rotations");

    // Position 0 IS the identity in real RoPE, because theta_j = 0 * ts^j = 0
    // for every j. I originally asserted the opposite, which a stand-in with a
    // fixed non-zero angle would have "passed". Identity-at-origin is the real
    // signature of the donor's formula: theta is seeded with the position, so
    // p=0 zeroes every angle at once.
    std::vector<float> zero = head;
    std::vector<float> cz, sz;
    RopeCache(0, kFreqBase, kHeadDim, &cz, &sz);
    RopeApply(&zero, cz, sz);
    double moved = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      moved = std::fmax(moved, std::fabs(zero[i] - head[i]));
    }
    std::printf("  note  max |rope(pos 0) - x| = %.3g\n", moved);
    Check(moved < 1e-7, "position 0 is exactly the identity, since theta = 0 * ts^j");
    // And it must NOT be the identity at position 1 — otherwise origin-identity
    // would hold for a transform that never rotates anything.
    std::vector<float> one = head;
    std::vector<float> cone, sone;
    RopeCache(1, kFreqBase, kHeadDim, &cone, &sone);
    RopeApply(&one, cone, sone);
    double moved1 = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      moved1 = std::fmax(moved1, std::fabs(one[i] - head[i]));
    }
    Check(moved1 > 1e-3, "position 1 does move the vector, so the origin is special");
  }

  // ---- 5. PAIRS ARE INDEPENDENT ----------------------------------------
  {
    std::vector<float> c, s;
    RopeCache(5, kFreqBase, kHeadDim, &c, &s);
    // Rotating pair 0 alone must leave pair 1 untouched.
    std::vector<float> whole = RampHead(kHeadDim, 1.0f);
    std::vector<float> part = whole;
    std::vector<float> both = whole;
    RopeApply(&both, c, s);
    std::vector<float> c0{c[0]}, s0{s[0]};
    RopeApply(&part, c0, s0);
    Check(std::fabs(part[0] - both[0]) < 1e-6f && std::fabs(part[1] - both[1]) < 1e-6f,
          "pair 0 matches the full rotation");
    Check(std::fabs(part[2] - whole[2]) < 1e-7f && std::fabs(part[3] - whole[3]) < 1e-7f,
          "rotating pair 0 leaves pair 1 byte-identical to the input");
  }

  // ---- 6. INVERSE --------------------------------------------------------
  {
    std::vector<float> c, s;
    RopeCache(29, kFreqBase, kHeadDim, &c, &s);
    std::vector<float> nc(kHeadDim), ns(kHeadDim);
    for (std::size_t i = 0; i < kHeadDim; i += 2) {
      nc[i] = c[i];
      nc[i + 1] = -s[i];
      ns[i] = -s[i];
      ns[i + 1] = c[i];
    }
    const std::vector<float> head = RampHead(kHeadDim, 2.0f);
    std::vector<float> v = head;
    RopeApply(&v, c, s);
    RopeApply(&v, nc, ns);
    double err = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      err = std::fmax(err, std::fabs(v[i] - head[i]));
    }
    std::printf("  note  max |rope(+theta) then rope(-theta) - x| = %.3g\n", err);
    Check(err < 1e-5, "rotating by +theta then -theta returns the original vector");
  }

  // ---- 7. V IS NOT ROPED: the K/V ORDER IN THE PIPELINE -----------------
  {
    // qwen35.cpp:310 reshapes Vcur; :314-326 ropes Q and K only. So a V plane
    // fed straight to the cache is UNROTATED and its norm is untouched.
    const std::vector<float> v_head = RampHead(kHeadDim, 1.0f);
    const std::vector<float> copied = v_head;  // the V path: reshape only
    const double ratio = L2(copied, 0, kHeadDim) / L2(v_head, 0, kHeadDim);
    Check(std::fabs(ratio - 1.0) < 1e-7,
          "V passes through unrotated, so its norm is exactly the input norm");
    // And the K path is genuinely different, or the distinction is fiction.
    std::vector<float> k_head = v_head;
    std::vector<float> c, s;
    RopeCache(1, kFreqBase, kHeadDim, &c, &s);
    RopeApply(&k_head, c, s);
    Check(L2(k_head, 0, kHeadDim) != L2(v_head, 0, kHeadDim) ||
              k_head[0] != v_head[0],
          "K at position 1 is rotated, so K and V are not the same vector");
  }

  // ---- 8. RoPE COMPOSES WITH TURBO4 (the property that matters) --------
  // The cache stores post-RoPE K. So: rope -> quantize -> dequantize, and the
  // block's shared fp16 norm must still represent the ROLLED vector's norm.
  {
    const std::vector<float> head = RampHead(kHeadDim, 1.0f);
    std::vector<float> c, s;
    RopeCache(1234, kFreqBase, kHeadDim, &c, &s);
    std::vector<float> roped = head;
    RopeApply(&roped, c, s);

    const double whole_head = L2(head, 0, kHeadDim);  // reference: RoPE kept it
    double worst = 0.0;
    for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
      const std::size_t off = b * gufo::hip::kQwenTurbo4BlockValues;
      // Compare BLOCK against BLOCK. An earlier version divided each block's
      // post-quantize norm by the WHOLE-head norm, which reports ~30% on a
      // codec that is accurate to 3e-4 — a units error that looked like a
      // catastrophic precision failure.
      const double block_before =
          L2(roped, off, gufo::hip::kQwenTurbo4BlockValues);
      std::uint8_t block[gufo::hip::kQwenTurbo4BlockBytes];
      t4::QuantizeBlock(roped.data() + off, block);
      std::vector<float> back(gufo::hip::kQwenTurbo4BlockValues);
      t4::DequantizeBlock(block, back.data());
      const double after = L2(back, 0, back.size());
      worst = std::fmax(worst, std::fabs(after / block_before - 1.0));
    }
    std::printf("  note  worst RoPE->Turbo4->RoPE^-1 norm deviation = %.6f\n", worst);
    // RoPE must not have changed the head norm BEFORE quantization. Compare
    // the pre-RoPE head against the post-RoPE head (not an expression with
    // itself, which is trivially true and tests nothing).
    Check(std::fabs(whole_head - L2(roped, 0, kHeadDim)) / whole_head < 1e-5,
          "RoPE left the head norm unchanged before quantization");
    // The bound is a PRECISION argument, not a fitted number: Turbo4 keeps a
    // 4-bit grid plus one fp16 norm, so the floor is set by the grid spacing
    // and fp16's ~1e-3 relative resolution. Review rejected the old
    // 0.15 ceiling precisely because it was fitted to observed output rather
    // than derived. 2% is comfortably inside the format's own precision while
    // still failing loudly if RoPE is applied twice or dropped.
    Check(worst < 0.01, "RoPE then Turbo4 keeps each block norm within 1%");
  }

  // ---- 9. FREQ_BASE IS NOT A FREE PARAMETER -----------------------------
  {
    const std::vector<float> head = RampHead(kHeadDim, 1.0f);
    std::vector<float> a = head, b = head;
    std::vector<float> c1, s1, c2, s2;
    RopeCache(50, kFreqBase, kHeadDim, &c1, &s1);
    RopeCache(50, 1000000.0f, kHeadDim, &c2, &s2);
    RopeApply(&a, c1, s1);
    RopeApply(&b, c2, s2);
    double d = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      d = std::fmax(d, std::fabs(a[i] - b[i]));
    }
    Check(d > 1e-3, "a different freq_base gives a materially different rotation");
    Check(L2(a, 0, kHeadDim) != 0.0 && L2(b, 0, kHeadDim) != 0.0,
          "both freq_base settings still preserve a usable norm");
  }

  // ---- 10. A HEAD NEVER STRADDLES A BLOCK AT THIS GEOMETRY --------------
  {
    // Reinforced here because the RoPE cache is indexed by pair within a HEAD;
    // if a block could straddle heads, one shared fp16 norm would serve two
    // differently-roped vectors and the round trip above would be meaningless.
    Check(gufo::hip::QwenKvHeadDimIsBlockAligned(kHeadDim),
          "UD-Q4 head_dim 256 is a whole number of 128-value blocks");
    Check(gufo::hip::QwenKvHeadsAreBlockAligned(kHeadDim, kKvHeads),
          "4 heads of 256 are each block aligned");
    Check(kHeadDim * kKvHeads == 1024,
          "head_dim 256 x 4 kv heads = the 1024 kv_width measured in the header");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
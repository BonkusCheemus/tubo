// Turbo5 / Turbo6 codec test (CPU only, no HIP).
//
//   g++ -std=c++20 -I. tests/models/qwen/turbo56_codec_test.cpp -o
//   /tmp/t56codec
//
// What this establishes, and from where each oracle comes:
//   1. TABLES. The six codebook arrays hash to the value computed from the
//      source of TheTom/llama-cpp-turboquant PR #381 (ggml-turbo-quant.c
//      TURBO5_/TURBO6_ CENTROIDS and MID) by an independent script. A typo in
//      the header changes the hash.
//   2. LAYOUT. An independent decoder written from the PR's
//      dequantize_row_turbo5_0 / dequantize_row_turbo6_0 (not from the header's
//      CodeAt) reads every block the codec writes, and each code it recovers
//      must be the nearest centroid to the double-precision rotated value.
//   3. GOLDEN BYTES. One fixed vector per width. The expected bytes were
//   produced
//      by an independent numpy implementation of the PR's quantize math
//      (float32, PR tables, searchsorted over the PR midpoints); its Turbo4
//      output equals the baseline C++ header's byte for byte, which ties the
//      script to the deployed format.
//   4. TURBO4 UNCHANGED. The expected bytes and the FNV hash over 1000 blocks
//      were produced by the BASELINE turbo4_codec.hpp (git index, before
//      Turbo5/6 were added) and are frozen below.
//   5. QUALITY. Relative reconstruction error is ordered turbo6 < turbo5 <
//      turbo4 and sits near the Lloyd-Max values (0.0095, 0.0025, 0.00065 of
//      the block energy), and the norm correction reproduces the input L2 norm.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "src/models/qwen/hip/turbo4_codec.hpp"
#include "tests/models/qwen/turbo56_writer_diff.hpp"

namespace t4 = gufo::hip::turbo4;
using t4::Bits;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("  FAIL  %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("  ok    %s\n", what.c_str());
  }
}

struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float Uni() { return static_cast<float>(Next() >> 40) / 16777216.0f; }
  float Gauss() {
    float a = 0;
    for (int i = 0; i < 12; ++i)
      a += Uni();
    return a - 6.0f;
  }
};

void FixedVector(float* x) {
  for (int j = 0; j < 128; ++j)
    x[j] = static_cast<float>(((j * 37 + 11) % 61) - 30) / 17.0f;
}

std::uint64_t FnvFloats(const float* a, std::size_t n) {
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < n; ++i) {
    unsigned char b[4];
    std::memcpy(b, &a[i], 4);
    for (unsigned char c : b) {
      h ^= c;
      h *= 1099511628211ull;
    }
  }
  return h;
}

std::string Hex(const std::uint8_t* p, std::size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (std::size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

// ---- independent decoder, from the PR's dequantize_row_turbo{5,6}_0
// ----------
int RefCode(Bits bits, const std::uint8_t* b, int j) {
  const std::uint8_t* qs = b + 2;
  const int nib = (qs[j / 2] >> ((j % 2) * 4)) & 0xF;
  if (bits == Bits::k5) {
    const int neg = (b[66 + j / 8] >> (j % 8)) & 1;
    return neg ? 15 - nib : 16 + nib;
  }
  if (bits == Bits::k6) {
    const int hi = (b[66 + j / 4] >> ((j % 4) * 2)) & 3;
    return nib | (hi << 4);
  }
  return nib;
}

double Centroid(Bits bits, int code) {
  return t4::CentroidValue(bits, code);
}
int Levels(Bits bits) {
  return 1 << static_cast<int>(bits);
}

// Rotated (signs1 -> Hadamard -> signs2) normalized vector, in DOUBLE, by the
// definition H[i][j] = (-1)^popcount(i & j) rather than butterflies.
void RotateDouble(const float* x, double norm, double* w) {
  double xs[128];
  for (int j = 0; j < 128; ++j)
    xs[j] = (norm > 1e-10 ? x[j] / norm : 0.0) * t4::TURBO_WHT_SIGNS1[j];
  for (int i = 0; i < 128; ++i) {
    double acc = 0.0;
    for (int j = 0; j < 128; ++j)
      acc += ((__builtin_popcount(i & j) & 1) ? -xs[j] : xs[j]);
    w[i] = acc * 0.08838834764831845 * t4::TURBO_WHT_SIGNS2[i];
  }
}

// Is `code` the nearest centroid to `w`, allowing the 1e-6 slack the 6-decimal
// tables and float32 arithmetic can disagree by right at a midpoint?
bool IsNearest(Bits bits, int code, double w) {
  const int n = Levels(bits);
  const double eps = 1e-6;
  if (code > 0) {
    const double mid = 0.5 * (Centroid(bits, code - 1) + Centroid(bits, code));
    if (w < mid - eps)
      return false;
  }
  if (code < n - 1) {
    const double mid = 0.5 * (Centroid(bits, code) + Centroid(bits, code + 1));
    if (w > mid + eps)
      return false;
  }
  return true;
}

void TestTables() {
  std::printf("tables\n");
  // Hashes of the PR #381 float32 arrays, computed by an independent script.
  Check(FnvFloats(t4::TURBO_CENTROIDS_5BIT, 32) == 13960790351939628271ull,
        "TURBO_CENTROIDS_5BIT == PR TURBO5_CENTROIDS");
  Check(FnvFloats(t4::TURBO_MID_5BIT, 31) == 3512519013697824011ull,
        "TURBO_MID_5BIT == PR TURBO5_MID");
  Check(FnvFloats(t4::TURBO_CENTROIDS_6BIT, 64) == 4214503388383282683ull,
        "TURBO_CENTROIDS_6BIT == PR TURBO6_CENTROIDS");
  Check(FnvFloats(t4::TURBO_MID_6BIT, 63) == 12337097727562134307ull,
        "TURBO_MID_6BIT == PR TURBO6_MID");

  for (Bits bits : {Bits::k4, Bits::k5, Bits::k6}) {
    const int n = Levels(bits);
    bool anti = true, inc = true, mid_ok = true;
    for (int i = 0; i < n; ++i) {
      if (Centroid(bits, i) != -Centroid(bits, n - 1 - i))
        anti = false;
      if (i + 1 < n && !(Centroid(bits, i) < Centroid(bits, i + 1)))
        inc = false;
    }
    // Midpoint i must be the average of centroids i and i+1 up to the 6-decimal
    // rounding of both tables.
    const float* mids = bits == Bits::k4   ? t4::TURBO_MID_4BIT
                        : bits == Bits::k5 ? t4::TURBO_MID_5BIT
                                           : t4::TURBO_MID_6BIT;
    for (int i = 0; i + 1 < n; ++i) {
      const double want = 0.5 * (Centroid(bits, i) + Centroid(bits, i + 1));
      if (std::fabs(mids[i] - want) > 2e-6)
        mid_ok = false;
    }
    const std::string name = "bits " + std::to_string(static_cast<int>(bits));
    Check(anti, name + ": codebook antisymmetric c[N-1-i] == -c[i]");
    Check(inc, name + ": codebook strictly increasing");
    Check(mid_ok, name + ": midpoints are centroid averages");
  }
}

void TestSizes() {
  std::printf("sizes\n");
  Check(t4::BlockBytes(Bits::k4) == 66, "turbo4 block is 66 B");
  Check(t4::BlockBytes(Bits::k5) == 82, "turbo5 block is 82 B");
  Check(t4::BlockBytes(Bits::k6) == 98, "turbo6 block is 98 B");
  Check(t4::kBlockBytes == 66, "turbo4::kBlockBytes unchanged");
  Check(t4::BitsAreValid(4) && t4::BitsAreValid(5) && t4::BitsAreValid(6) &&
            !t4::BitsAreValid(0) && !t4::BitsAreValid(3) &&
            !t4::BitsAreValid(7),
        "BitsAreValid accepts exactly 4,5,6");
}

void TestPackUnpack() {
  std::printf("pack/unpack\n");
  for (Bits bits : {Bits::k4, Bits::k5, Bits::k6}) {
    const int n = Levels(bits);
    bool all = true;
    for (int code = 0; code < n; ++code) {
      for (int j = 0; j < 128; ++j) {
        std::uint8_t blk[t4::kMaxBlockBytes] = {};
        std::memset(blk, 0, sizeof(blk));
        blk[t4::kNormBytes + j / 2] |= static_cast<std::uint8_t>(
            t4::QsNibbleOfCode(bits, code) << ((j % 2) * 4));
        if (bits == Bits::k5)
          blk[66 + j / 8] |=
              static_cast<std::uint8_t>(t4::QhBitsOfCode(bits, code, j));
        if (bits == Bits::k6)
          blk[66 + j / 4] |=
              static_cast<std::uint8_t>(t4::QhBitsOfCode(bits, code, j));
        if (t4::CodeAt(bits, blk, j) != static_cast<std::uint32_t>(code))
          all = false;
        if (RefCode(bits, blk, j) != code)
          all = false;
        // The header decoder and the independent decoder agree on every
        // position, not just the one written (catches a stray bit landing in a
        // neighbour's field).
        for (int k = 0; k < 128; ++k)
          if (t4::CodeAt(bits, blk, k) !=
              static_cast<std::uint32_t>(RefCode(bits, blk, k)))
            all = false;
      }
    }
    Check(all, "bits " + std::to_string(static_cast<int>(bits)) +
                   ": every (code, position) round-trips through the layout");
  }
  // Turbo5 sign-magnitude spot values straight from the PR comment:
  // "Centroid code = neg ? 15-m : 16+m".
  Check(t4::QsNibbleOfCode(Bits::k5, 16) == 0 &&
            t4::QsNibbleOfCode(Bits::k5, 31) == 15 &&
            t4::QsNibbleOfCode(Bits::k5, 15) == 0 &&
            t4::QsNibbleOfCode(Bits::k5, 0) == 15,
        "turbo5 magnitude nibble: code 16..31 -> 0..15, code 15..0 -> 0..15");
  Check(t4::QhBitsOfCode(Bits::k5, 15, 3) == (1u << 3) &&
            t4::QhBitsOfCode(Bits::k5, 16, 3) == 0u &&
            t4::QhBitsOfCode(Bits::k5, 0, 9) == (1u << 1),
        "turbo5 sign bit: set for negative, at bit j%8");
  Check(t4::QhBitsOfCode(Bits::k6, 0x3A, 5) == (3u << 2) &&
            t4::QhBitsOfCode(Bits::k6, 0x0F, 2) == 0u,
        "turbo6 high bits: code>>4 at bit 2*(j%4)");
}

void TestNearestMatchesLinearScan() {
  std::printf("nearest code\n");
  Rng r(12345);
  bool agree = true;
  for (Bits bits : {Bits::k5, Bits::k6}) {
    const float* mids =
        bits == Bits::k5 ? t4::TURBO_MID_5BIT : t4::TURBO_MID_6BIT;
    const int n = Levels(bits);
    for (int t = 0; t < 200000; ++t) {
      float v = (r.Uni() - 0.5f) * 0.8f;
      if (t < 2 * n) {  // exact midpoints and neighbours: strict '<' contract
        const int i = (t / 2) % (n - 1);
        v = t % 2 == 0 ? mids[i] : std::nextafter(mids[i], -1.0f);
      }
      int want = n - 1;
      for (int i = 0; i < n - 1; ++i)
        if (v < mids[i]) {
          want = i;
          break;
        }
      if (t4::NearestCode(bits, v) != want)
        agree = false;
    }
  }
  Check(agree,
        "binary search == strict-'<' linear scan (incl. exact midpoints)");
  Check(t4::NearestCode(Bits::k4, 0.011349f) == t4::NearestCentroid(0.011349f),
        "k4 NearestCode is NearestCentroid");
}

struct Golden {
  Bits bits;
  const char* hex;
};

// Produced by the independent numpy reference (see header comment, point 3).
const Golden kGolden[] = {
    {Bits::k4,
     "c34950d94327de962949bd7b369a58b89974e750361188a6899dc9a8c849c38638778b517"
     "1c46c"
     "45a50a757d7748d94aa68764baad66a8e9a76e55b39da296a7c467"},
    {Bits::k5,
     "df493ea278a09c23b2625a17922450703316c15e83bc0152122b9250916388038010063c0"
     "c9628"
     "6555e5050911709364430027634a3331b3412c43773b5923507720f3a438c2fd0480e57c9"
     "ebb58"
     "c39047d5"},
    {Bits::k6,
     "df4984341faf38488435b5dedb4840f077c37c43f98613aa344624a022371f18f0de0c86f"
     "723b0"
     "35953ae4e3dd103648991fc1c7939972768da868f076bc489ef1aee4149f626ba1a65a4d0"
     "19aba"
     "ae6e9c524ad4572975656e99a55bea79958b9959"},
};

void TestGolden() {
  std::printf("golden bytes\n");
  float x[128];
  FixedVector(x);
  for (const Golden& g : kGolden) {
    std::uint8_t blk[t4::kMaxBlockBytes];
    t4::QuantizeBlock(x, blk, g.bits);
    const std::string got = Hex(blk, t4::BlockBytes(g.bits));
    const bool ok = got == g.hex;
    Check(ok, "bits " + std::to_string(static_cast<int>(g.bits)) +
                  ": fixed vector bytes match the independent reference");
    if (!ok)
      std::printf("        got  %s\n        want %s\n", got.c_str(), g.hex);
  }
}

void TestTurbo4Unchanged() {
  std::printf("turbo4 unchanged vs baseline\n");
  // Frozen from the baseline header (git index) by basedump.cpp.
  float x[128];
  FixedVector(x);
  std::uint8_t blk[t4::kBlockBytes];
  t4::QuantizeBlock(x, blk);  // default argument path
  Check(Hex(blk, 66) == kGolden[0].hex,
        "default QuantizeBlock(x, dst) == baseline fixed-vector bytes");
  std::uint8_t blk2[t4::kBlockBytes];
  t4::QuantizeBlock(x, blk2, Bits::k4);
  Check(std::memcmp(blk, blk2, 66) == 0, "explicit Bits::k4 == default");

  Rng r(0x9E3779B97F4A7C15ull);
  std::uint64_t h = 1469598103934665603ull;
  for (int v = 0; v < 1000; ++v) {
    float y[128];
    const float sc = (v % 7 == 0) ? 100.0f : (v % 5 == 0 ? 0.01f : 1.0f);
    for (int j = 0; j < 128; ++j)
      y[j] = r.Gauss() * sc;
    std::uint8_t o[66];
    t4::QuantizeBlock(y, o);
    for (int i = 0; i < 66; ++i) {
      h ^= o[i];
      h *= 1099511628211ull;
    }
  }
  Check(h == 6449984692014687429ull,
        "FNV over 1000 turbo4 blocks == baseline hash");

  // Decode side: the baseline DequantizeBlock is DequantElement per j; the new
  // default path must return identical floats.
  float a[128], b[128];
  t4::DequantizeBlock(blk, a);
  const float norm =
      t4::HalfBitsToFloat(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
  bool same = true;
  for (int j = 0; j < 128; ++j) {
    b[j] = t4::TURBO_CENTROIDS_4BIT[(blk[2 + j / 2] >> ((j % 2) * 4)) & 0xF] *
           norm;
    if (std::memcmp(&a[j], &b[j], 4) != 0)
      same = false;
    const float e = t4::DequantizeElement(blk, j);
    if (std::memcmp(&a[j], &e, 4) != 0)
      same = false;
  }
  Check(same,
        "turbo4 DequantizeBlock/DequantizeElement bit-identical to baseline "
        "formula");
}

void TestAgainstIndependentDecoder() {
  std::printf("independent decoder\n");
  Rng r(777);
  for (Bits bits : {Bits::k5, Bits::k6}) {
    int bad_code = 0, bad_norm = 0, bad_norm_pres = 0;
    const int trials = 120;
    for (int t = 0; t < trials; ++t) {
      float x[128];
      const float sc = (t % 3 == 0) ? 50.0f : 0.37f;
      for (int j = 0; j < 128; ++j)
        x[j] = r.Gauss() * sc;
      std::uint8_t blk[t4::kMaxBlockBytes];
      t4::QuantizeBlock(x, blk, bits);

      double sum = 0;
      for (int j = 0; j < 128; ++j)
        sum += static_cast<double>(x[j]) * x[j];
      const double norm = std::sqrt(sum);
      double w[128];
      RotateDouble(x, norm, w);
      double recon = 0;
      for (int j = 0; j < 128; ++j) {
        const int c = RefCode(bits, blk, j);
        if (!IsNearest(bits, c, w[j]))
          ++bad_code;
        recon += Centroid(bits, c) * Centroid(bits, c);
      }
      const double want_norm = norm / std::sqrt(recon);
      const double got_norm = t4::HalfBitsToFloat(
          static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
      if (std::fabs(got_norm - want_norm) > 6e-4 * want_norm)
        ++bad_norm;

      // The norm correction exists so that dequantized blocks keep the input L2
      // norm: ||dequant|| == ||x||, to fp16 rounding of the stored norm.
      float d[128];
      t4::DequantizeBlock(blk, d, bits);
      double dn = 0;
      for (int j = 0; j < 128; ++j)
        dn += static_cast<double>(d[j]) * d[j];
      if (std::fabs(std::sqrt(dn) - norm) > 1e-3 * norm)
        ++bad_norm_pres;
    }
    const std::string name = "bits " + std::to_string(static_cast<int>(bits));
    Check(bad_code == 0,
          name + ": every code is the nearest centroid (independent decoder)");
    Check(bad_norm == 0,
          name + ": stored norm == group_norm / recon_norm to fp16 rounding");
    Check(bad_norm_pres == 0,
          name + ": dequantized L2 norm reproduces the input norm");
  }
}

void TestQuality() {
  std::printf("round-trip quality\n");
  Rng r(2024);
  double rel[3] = {0, 0, 0};
  const int trials = 300;
  const Bits all[3] = {Bits::k4, Bits::k5, Bits::k6};
  for (int t = 0; t < trials; ++t) {
    float x[128];
    for (int j = 0; j < 128; ++j)
      x[j] = r.Gauss() * 2.0f;
    double energy = 0;
    for (int j = 0; j < 128; ++j)
      energy += static_cast<double>(x[j]) * x[j];
    for (int k = 0; k < 3; ++k) {
      std::uint8_t blk[t4::kMaxBlockBytes];
      t4::QuantizeBlock(x, blk, all[k]);
      float d[128];
      t4::DequantizeBlock(blk, d, all[k]);
      t4::RotateInverse(d);
      double err = 0;
      for (int j = 0; j < 128; ++j) {
        const double e = static_cast<double>(x[j]) - d[j];
        err += e * e;
      }
      rel[k] += err / energy;
    }
  }
  for (double& v : rel)
    v /= trials;
  std::printf("        relative MSE: turbo4 %.5f  turbo5 %.5f  turbo6 %.5f\n",
              rel[0], rel[1], rel[2]);
  Check(rel[2] < rel[1] && rel[1] < rel[0],
        "MSE ordering turbo6 < turbo5 < turbo4");
  // Lloyd-Max for N(0,1): 0.0095, 0.0025, 0.00065. Allow a factor of 2 each way
  // (finite sample, norm correction, a Gaussian input rather than the exact
  // rotated-coordinate law).
  Check(rel[0] > 0.0095 / 2 && rel[0] < 0.0095 * 2,
        "turbo4 relative MSE near 0.0095");
  Check(rel[1] > 0.0025 / 2 && rel[1] < 0.0025 * 2,
        "turbo5 relative MSE near 0.0025");
  Check(rel[2] > 0.00065 / 2 && rel[2] < 0.00065 * 2,
        "turbo6 relative MSE near 0.00065");
  Check(rel[0] / rel[1] > 2.0 && rel[1] / rel[2] > 2.0,
        "each extra bit at least halves the error");
}

void TestChecked() {
  std::printf("checked encode / edge inputs\n");
  for (Bits bits : {Bits::k4, Bits::k5, Bits::k6}) {
    const std::string name = "bits " + std::to_string(static_cast<int>(bits));
    float huge[128];
    for (int j = 0; j < 128; ++j)
      huge[j] = (j % 2 ? 1.0f : -1.0f) * 1.0e6f;
    std::uint8_t dst[t4::kMaxBlockBytes];
    std::memset(dst, 0xAB, sizeof(dst));
    const bool ok = t4::QuantizeBlockChecked(huge, dst, bits);
    bool untouched = true;
    for (unsigned char c : dst)
      untouched &= (c == 0xAB);
    Check(!ok && untouched,
          name + ": norm overflowing fp16 is refused and dst untouched");

    float zeros[128] = {};
    std::uint8_t zb[t4::kMaxBlockBytes];
    std::memset(zb, 0xCD, sizeof(zb));
    Check(t4::QuantizeBlockChecked(zeros, zb, bits),
          name + ": zero block is storable");
    float d[128];
    t4::DequantizeBlock(zb, d, bits);
    bool z = true;
    for (float v : d)
      z &= (v == 0.0f);
    Check(z, name + ": zero block decodes to zeros");
    bool no_spill = true;
    for (std::size_t i = t4::BlockBytes(bits); i < sizeof(zb); ++i)
      no_spill &= (zb[i] == 0xCD);
    Check(no_spill, name + ": encode writes exactly BlockBytes(bits)");
  }
}

void TestWriterDiffGate() {
  using gufo::test::Turbo56WriterByteDiffAllowed;
  using gufo::test::Turbo56WriterDiffCountAllowed;
  std::uint8_t gpu[t4::kMaxBlockBytes]{};
  std::uint8_t host[t4::kMaxBlockBytes]{};
  gpu[1] = host[1] = 0x3c;  // Norm 1.0.
  for (unsigned ulp : {1U, 2U, 3U}) {
    gpu[0] = static_cast<std::uint8_t>(ulp);
    Check(Turbo56WriterByteDiffAllowed(gpu, host, 0, Bits::k4) == (ulp <= 2),
          "writer norm ULP bound " + std::to_string(ulp));
  }
  gpu[0] = 0; gpu[1] = 0x7c;
  Check(!Turbo56WriterByteDiffAllowed(gpu, host, 1, Bits::k4),
        "writer rejects infinite norm");
  gpu[0] = 1;
  Check(!Turbo56WriterByteDiffAllowed(gpu, host, 0, Bits::k4),
        "writer rejects NaN norm");
  gpu[0] = 0; gpu[1] = 0xbc;
  Check(!Turbo56WriterByteDiffAllowed(gpu, host, 1, Bits::k4),
        "writer rejects negative norm");
  gpu[1] = 0x3c;
  host[2] = 0x87; gpu[2] = 0x98;
  Check(Turbo56WriterByteDiffAllowed(gpu, host, 2, Bits::k4),
        "writer checks both neighbouring nibble codes");
  gpu[2] = 0x9a;
  Check(!Turbo56WriterByteDiffAllowed(gpu, host, 2, Bits::k4),
        "writer rejects non-neighbour nibble");
  host[2] = 0x0f; gpu[2] = 0x10;
  Check(!Turbo56WriterByteDiffAllowed(gpu, host, 2, Bits::k4),
        "raw byte distance one is not decoded code distance one");
  host[2] = gpu[2] = 0; host[66] = 1; gpu[66] = 0;
  Check(Turbo56WriterByteDiffAllowed(gpu, host, 66, Bits::k5),
        "writer accepts Turbo5 central sign transition code15 to16");
  host[2] = 1;
  Check(!Turbo56WriterByteDiffAllowed(gpu, host, 66, Bits::k5),
        "writer rejects Turbo5 sign change code14 to16");
  host[2] = 15; host[66] = 0; gpu[2] = 0; gpu[66] = 1;
  Check(Turbo56WriterByteDiffAllowed(gpu, host, 2, Bits::k6) &&
            Turbo56WriterByteDiffAllowed(gpu, host, 66, Bits::k6),
        "writer accepts Turbo6 adjacent15 to16 across qs and qh");
  gpu[66] = 2;
  Check(!Turbo56WriterByteDiffAllowed(gpu, host, 66, Bits::k6),
        "writer rejects non-neighbour high bits");
  for (Bits bits : {Bits::k4, Bits::k5})
    Check(!Turbo56WriterByteDiffAllowed(gpu, host, t4::BlockBytes(bits), bits),
          "writer rejects padding even in a wider slot");
  Check(Turbo56WriterDiffCountAllowed(0, 9999) &&
            !Turbo56WriterDiffCountAllowed(1, 9999) &&
            Turbo56WriterDiffCountAllowed(1, 10000) &&
            !Turbo56WriterDiffCountAllowed(2, 10000) &&
            Turbo56WriterDiffCountAllowed(2, 20000),
        "writer byte rate cap is inclusive and never rounded upward");
}

}  // namespace

int main() {
  TestTables();
  TestSizes();
  TestPackUnpack();
  TestNearestMatchesLinearScan();
  TestGolden();
  TestTurbo4Unchanged();
  TestAgainstIndependentDecoder();
  TestQuality();
  TestChecked();
  TestWriterDiffGate();
  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}

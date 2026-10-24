// CPU exactness test of the two v_dual_dot2acc arms of
// OpusIq3sB8VopdKernel (src/models/qwen/hip/kernels/sn_iq3.hip) against k4's
// int32 math. No GPU, no HIP: it emulates, operation by operation, what one lane
// does for one 256-weight block and one token.
//
//   g++ -O2 -std=c++20 -o /tmp/vopd_exact tests/kernels/sn_iq3_vopd_exact_test.cc
//   /tmp/vopd_exact src/models/qwen/hip/quant_ops.hpp [blocks]
//
// k4        isum = sum_sb (2n+1) * sum_i dot4(w_i, x_i)             (int32)
// t16       table entry = 2 signed f16 pairs; pk_mul_f16 by (s,s); dot2acc lo/hi
// cv        table byte = w+128; perm -> f16 0x64bb (= 1152+w); pk_fma_f16(p, s,
//           -1152 s) (ONE rounding); dot2acc lo/hi
// Both: per word two accumulators (lo = weights 0,1, hi = weights 2,3 of every
// word of the block), lo + hi in f32 at the end; equal to float(isum) required.
// dot2acc is emulated in three orders ((p0+p1)+acc, (acc+p0)+p1 with a float
// rounding after every add, and one exact rounding): the sums are integers
// below 2^24, so all three must agree.
// The grid is read from quant_ops.hpp (kDeviceIq3sGrid), so the real values are
// used. Returns 0 only if every comparison matched.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

float HalfToFloat(std::uint16_t h) {
  const int sign = (h >> 15) & 1, exp = (h >> 10) & 31, man = h & 1023;
  double v;
  if (exp == 0) v = std::ldexp(static_cast<double>(man), -24);
  else if (exp == 31) v = man ? NAN : INFINITY;
  else v = std::ldexp(static_cast<double>(1024 + man), exp - 25);
  return static_cast<float>(sign ? -v : v);
}

// Round a double to the nearest f16 (ties to even), as the hardware does for
// an exact intermediate. Finite, in-range inputs only (asserted by caller).
std::uint16_t DoubleToHalf(double x) {
  if (x == 0.0) return std::signbit(x) ? 0x8000 : 0;
  const std::uint16_t sign = x < 0 ? 0x8000 : 0;
  x = std::fabs(x);
  int e;
  std::frexp(x, &e);  // x = m * 2^e, m in [0.5, 1)
  int exp = e - 1 + 15;  // biased exponent of the leading bit
  if (exp >= 31) return sign | 0x7C00;
  double scaled;  // in units of the quantum
  if (exp <= 0) { scaled = std::ldexp(x, 24); exp = 0; }
  else scaled = std::ldexp(x, 10 - (e - 1));  // [1024, 2048)
  double fl = std::floor(scaled), diff = scaled - fl;
  std::uint32_t q = static_cast<std::uint32_t>(fl);
  if (diff > 0.5 || (diff == 0.5 && (q & 1))) ++q;
  if (exp == 0) return sign | static_cast<std::uint16_t>(q);  // q may roll into exp 1
  if (q >= 2048) { q >>= 1; ++exp; if (exp >= 31) return sign | 0x7C00; }
  return sign | static_cast<std::uint16_t>((exp << 10) | (q - 1024));
}

std::uint32_t Perm(std::uint32_t s0, std::uint32_t s1, std::uint32_t sel) {
  const std::uint64_t pool = (static_cast<std::uint64_t>(s0) << 32) | s1;
  std::uint32_t out = 0;
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t c = (sel >> (8 * i)) & 0xFF;
    std::uint32_t b;
    if (c <= 7) b = (pool >> (8 * c)) & 0xFF;
    else if (c == 12) b = 0x00;
    else { std::fprintf(stderr, "perm selector %u unused here\n", c); std::exit(2); }
    out |= b << (8 * i);
  }
  return out;
}

// quant_ops.hpp NegateSelectedBytes, verbatim arithmetic.
std::uint32_t NegateSelectedBytes(std::uint32_t v, std::uint32_t bits) {
  const std::uint32_t selector =
      (((bits & 0x0FU) * 0x00810204U) & 0x04040404U) | 0x03020100U;
  return Perm(0x01010100U - v, v, selector);
}

std::vector<std::uint32_t> LoadGrid(const char* path) {
  std::ifstream f(path);
  std::stringstream ss; ss << f.rdbuf();
  const std::string s = ss.str();
  const auto p = s.find("kDeviceIq3sGrid[512] = {");
  if (p == std::string::npos) { std::fprintf(stderr, "grid not found\n"); std::exit(2); }
  const auto e = s.find("};", p);
  const std::string body = s.substr(p, e - p);
  std::vector<std::uint32_t> g;
  const std::regex re("0x([0-9a-fA-F]{8})U");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), re); it != std::sregex_iterator(); ++it)
    g.push_back(static_cast<std::uint32_t>(std::stoul((*it)[1].str(), nullptr, 16)));
  if (g.size() != 512) { std::fprintf(stderr, "grid has %zu entries\n", g.size()); std::exit(2); }
  return g;
}

float FloatAdd(float a, float b) { volatile float r = a + b; return r; }

// dot2acc: acc + a0*w0 + a1*w1, three evaluation orders.
float Dot2Acc(int order, float acc, float a0, float w0, float a1, float w1) {
  const float p0 = a0 * w0, p1 = a1 * w1;  // exact: |p| <= 59055 < 2^24
  if (order == 0) return FloatAdd(FloatAdd(p0, p1), acc);
  if (order == 1) return FloatAdd(FloatAdd(acc, p0), p1);
  return static_cast<float>(static_cast<double>(acc) + p0 + p1);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s quant_ops.hpp [blocks]\n", argv[0]); return 2; }
  const auto grid = LoadGrid(argv[1]);
  const long blocks = argc > 2 ? std::atol(argv[2]) : 200000;
  long bad = 0;

  // 0. f16 helpers: every finite half survives half -> double -> half.
  for (std::uint32_t h = 0; h < 65536; ++h) {
    if (((h >> 10) & 31) == 31) continue;
    if (DoubleToHalf(HalfToFloat(h)) != h && !((h & 0x7FFF) == 0)) { ++bad; }
  }
  for (int v = -127; v <= 127; ++v)  // activations are exact in f16
    if (HalfToFloat(DoubleToHalf(v)) != static_cast<float>(v)) ++bad;
  for (int w = -15; w <= 15; ++w)    // scaled weights are exact in f16
    for (int s = 1; s <= 31; s += 2)
      if (HalfToFloat(DoubleToHalf(static_cast<double>(w) * s)) != static_cast<float>(w * s)) ++bad;
  std::printf("f16 helper self-test mismatches: %ld\n", bad);

  // 1. The three tables, entry by entry, against k4's signed bytes.
  long table_bad = 0;
  std::vector<std::uint32_t> k4_table(8192), cv_table(8192);
  std::vector<std::uint32_t> t16_table(16384);
  for (std::uint32_t sign = 0; sign < 16; ++sign)
    for (std::uint32_t i = 0; i < 512; ++i) {
      const std::uint32_t idx = (sign << 9) | i;
      const std::uint32_t wd = NegateSelectedBytes(grid[i], sign);
      k4_table[idx] = wd;
      cv_table[idx] = wd ^ 0x80808080U;
      auto hb = [&](std::uint32_t j) {
        return static_cast<std::uint32_t>(DoubleToHalf(static_cast<std::int8_t>((wd >> (8 * j)) & 0xFF)));
      };
      t16_table[2 * idx] = hb(0) | (hb(1) << 16);
      t16_table[2 * idx + 1] = hb(2) | (hb(3) << 16);
      for (int j = 0; j < 4; ++j) {
        const int w = static_cast<std::int8_t>((wd >> (8 * j)) & 0xFF);
        const int expect_sign = ((sign >> j) & 1) ? -1 : 1;
        if (w != expect_sign * static_cast<int>((grid[i] >> (8 * j)) & 0xFF)) ++table_bad;
        if (w == 0 || (w & 1) == 0 || w < -15 || w > 15) ++table_bad;
        if (static_cast<int>((cv_table[idx] >> (8 * j)) & 0xFF) != w + 128) ++table_bad;
      }
    }
  std::printf("table mismatches (k4 sign fold / t16 / cv bias): %ld\n", table_bad);
  bad += table_bad;


  // 3. Grid-free worst case: every weight +-15, every activation +-127, s = 31,
  // all 256 weights of the block (|isum| = 15,118,080 < 2^24).
  {
    long worst_bad = 0;
    for (int wsign = 0; wsign < 2; ++wsign)
      for (int xsign = 0; xsign < 2; ++xsign)
        for (int order = 0; order < 3; ++order) {
          const int w = wsign ? -15 : 15, xv = xsign ? -127 : 127;
          const std::uint32_t wd = static_cast<std::uint32_t>(static_cast<std::uint8_t>(w)) * 0x01010101U;
          const std::uint32_t e = wd ^ 0x80808080U;
          const std::uint32_t pa = Perm(0x64646464U, e, 0x05010400U);
          const std::uint16_t sh = DoubleToHalf(31), ch = DoubleToHalf(-1152.0 * 31);
          const std::uint16_t w_t = DoubleToHalf(static_cast<double>(w) * 31);
          const std::uint16_t w_c = DoubleToHalf(static_cast<double>(HalfToFloat(pa & 0xFFFF)) * HalfToFloat(sh) + HalfToFloat(ch));
          float lo_t = 0, hi_t = 0, lo_c = 0, hi_c = 0;
          for (int word = 0; word < 64; ++word) {
            lo_t = Dot2Acc(order, lo_t, xv, HalfToFloat(w_t), xv, HalfToFloat(w_t));
            hi_t = Dot2Acc(order, hi_t, xv, HalfToFloat(w_t), xv, HalfToFloat(w_t));
            lo_c = Dot2Acc(order, lo_c, xv, HalfToFloat(w_c), xv, HalfToFloat(w_c));
            hi_c = Dot2Acc(order, hi_c, xv, HalfToFloat(w_c), xv, HalfToFloat(w_c));
          }
          const float ref = static_cast<float>(256LL * w * xv * 31);
          if (FloatAdd(lo_t, hi_t) != ref) ++worst_bad;
          if (FloatAdd(lo_c, hi_c) != ref) ++worst_bad;
        }
    std::printf("worst-case (|w|=15, |x|=127, s=31, 256 weights, |isum| = 15118080) mismatches: %ld\n", worst_bad);
    bad += worst_bad;
  }

  // 2. Blocks.
  std::mt19937_64 rng(0x5eed0b8ULL);
  long mism_t16[3] = {0, 0, 0}, mism_cv[3] = {0, 0, 0}, nonfused_cv = 0;
  long extreme_blocks = 0;
  double max_abs = 0;
  for (long blk = 0; blk < blocks; ++blk) {
    // kind: random, or an extreme corner (every 16th block).
    const int kind = (blk % 16 == 15) ? static_cast<int>((blk / 16) % 6) : -1;
    std::uint32_t qs[64], signs[64];  // per word: grid index (9 bit), sign nibble
    int scale[8], x[256];
    for (int w = 0; w < 64; ++w) {
      qs[w] = rng() & 511;
      signs[w] = rng() & 15;
    }
    for (int s = 0; s < 8; ++s) scale[s] = 1 + 2 * static_cast<int>(rng() & 15);
    for (int i = 0; i < 256; ++i) x[i] = static_cast<int>(rng() % 255) - 127;
    if (kind >= 0) {
      ++extreme_blocks;
      // pick the grid entry with the largest magnitude (0x0f0f0f0f if present)
      std::uint32_t big = 0;
      for (std::uint32_t i = 0; i < 512; ++i) {
        int sum = 0;
        for (int j = 0; j < 4; ++j) sum += (grid[i] >> (8 * j)) & 0xFF;
        int bsum = 0;
        for (int j = 0; j < 4; ++j) bsum += (grid[big] >> (8 * j)) & 0xFF;
        if (sum > bsum) big = i;
      }
      for (int w = 0; w < 64; ++w) { qs[w] = big; signs[w] = (kind & 1) ? 0 : (kind & 2) ? 15 : 0; }
      for (int s = 0; s < 8; ++s) scale[s] = 31;
      for (int i = 0; i < 256; ++i) x[i] = (kind >= 4) ? ((i & 1) ? 127 : -127) : ((kind & 1) ? -127 : 127);
      if (kind == 2 || kind == 3) for (int i = 0; i < 256; ++i) x[i] = 127;
    }
    // k4 reference.
    std::int64_t isum = 0;
    for (int s = 0; s < 8; ++s) {
      int dot = 0;
      for (int wi = 0; wi < 8; ++wi) {
        const std::uint32_t wd = k4_table[(signs[8 * s + wi] << 9) | qs[8 * s + wi]];
        for (int j = 0; j < 4; ++j)
          dot += static_cast<std::int8_t>((wd >> (8 * j)) & 0xFF) * x[32 * s + 4 * wi + j];
      }
      isum += static_cast<std::int64_t>(scale[s]) * dot;
    }
    const float ref = static_cast<float>(isum);
    if (std::fabs(static_cast<double>(isum)) > max_abs) max_abs = std::fabs(static_cast<double>(isum));
    // arms, three dot2acc orders each.
    for (int order = 0; order < 3; ++order) {
      float lo_t = 0, hi_t = 0, lo_c = 0, hi_c = 0;
      for (int s = 0; s < 8; ++s) {
        const std::uint16_t sh = DoubleToHalf(scale[s]);
        const std::uint16_t ch = DoubleToHalf(-1152.0 * scale[s]);
        for (int wi = 0; wi < 8; ++wi) {
          const std::uint32_t idx = (signs[8 * s + wi] << 9) | qs[8 * s + wi];
          const int base = 32 * s + 4 * wi;
          // t16
          std::uint16_t wp[4];  // after v_pk_mul_f16 (exact product, one rounding)
          for (int h = 0; h < 4; ++h) {
            const std::uint16_t bits = (t16_table[2 * idx + (h >> 1)] >> (16 * (h & 1))) & 0xFFFF;
            wp[h] = DoubleToHalf(static_cast<double>(HalfToFloat(bits)) * HalfToFloat(sh));
          }
          lo_t = Dot2Acc(order, lo_t, x[base + 0], HalfToFloat(wp[0]), x[base + 1], HalfToFloat(wp[1]));
          hi_t = Dot2Acc(order, hi_t, x[base + 2], HalfToFloat(wp[2]), x[base + 3], HalfToFloat(wp[3]));
          // cv
          const std::uint32_t e = cv_table[idx];
          const std::uint32_t pa = Perm(0x64646464U, e, 0x05010400U), pb = Perm(0x64646464U, e, 0x05030402U);
          std::uint16_t wc[4], wn[4];
          for (int h = 0; h < 4; ++h) {
            const std::uint32_t p = h < 2 ? pa : pb;
            const std::uint16_t bits = (p >> (16 * (h & 1))) & 0xFFFF;
            const double a = HalfToFloat(bits);
            wc[h] = DoubleToHalf(a * HalfToFloat(sh) + HalfToFloat(ch));   // fused: one rounding
            wn[h] = DoubleToHalf(static_cast<double>(HalfToFloat(DoubleToHalf(a * HalfToFloat(sh)))) + HalfToFloat(ch));  // mul, round, add, round
          }
          lo_c = Dot2Acc(order, lo_c, x[base + 0], HalfToFloat(wc[0]), x[base + 1], HalfToFloat(wc[1]));
          hi_c = Dot2Acc(order, hi_c, x[base + 2], HalfToFloat(wc[2]), x[base + 3], HalfToFloat(wc[3]));
          if (order == 0)
            for (int h = 0; h < 4; ++h)
              if (HalfToFloat(wn[h]) != HalfToFloat(wc[h])) ++nonfused_cv;
        }
      }
      if (FloatAdd(lo_t, hi_t) != ref) ++mism_t16[order];
      if (FloatAdd(lo_c, hi_c) != ref) ++mism_cv[order];
    }
  }
  std::printf("blocks %ld (%ld extreme corners), max |isum| = %.0f (2^24 = 16777216)\n", blocks, extreme_blocks, max_abs);
  for (int o = 0; o < 3; ++o)
    std::printf("dot2acc order %d: t16 mismatches %ld, cv mismatches %ld\n", o, mism_t16[o], mism_cv[o]);
  std::printf("info: cv weights that would differ if mul and add were TWO roundings (not fma): %ld\n", nonfused_cv);
  bad += mism_t16[0] + mism_t16[1] + mism_t16[2] + mism_cv[0] + mism_cv[1] + mism_cv[2];
  std::printf(bad == 0 ? "PASS\n" : "FAIL (%ld)\n", bad);
  return bad == 0 ? 0 : 1;
}

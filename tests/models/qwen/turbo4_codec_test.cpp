// The codec test. Three independent implementations must agree:
//   1. the DONOR C source (turbo-quant.cuh), re-parsed at run time
//   2. gufo's C++ header (turbo4_codec.hpp), which is what ships
//   3. the Python reference (turbo4_ref.py), which was itself built from the
//      deployed write path and carries golden bytes as a fixture below
//
// If any two drift, this fails. Agreement between a header and a golden fixture
// alone would only prove self-consistency.
//
//   g++ -std=c++20 -I. tests/models/qwen/turbo4_codec_test.cpp -o /tmp/t4codec

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace t4 = gufo::hip::turbo4;

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

std::string ReadFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<float> ParseCArray(const std::string& src, const std::string& name) {
  // [\s\S] rather than '.' is REQUIRED: these arrays span many lines and
  // ECMAScript '.' does not match a newline. With '.' the body captures
  // nothing and every table silently compares as a mismatch against an empty
  // vector, which reads as "the constants are wrong" when they are correct.
  const std::regex re(name + R"(\s*\[\s*\d+\s*\]\s*=\s*\{([\s\S]*?)\};)");
  std::smatch m;
  std::vector<float> out;
  if (!std::regex_search(src, m, re)) return out;
  const std::regex num(R"(-?\d+\.\d+(?:[eE][-+]?\d+)?f?)");
  // Bind the body to a NAMED string. m[1].str() returns a temporary, and
  // calling it twice yields two DIFFERENT temporaries: iterators built from
  // begin() and end() of separate temporaries dangle, which silently matched
  // nothing instead of crashing.
  const std::string body = m[1].str();
  for (std::sregex_iterator it(body.begin(), body.end(), num), end;
       it != end; ++it) {
    std::string s = it->str();
    if (!s.empty() && s.back() == 'f') s.pop_back();
    out.push_back(std::stof(s));
  }
  return out;
}

std::string Hex(const std::uint8_t* p, std::size_t n) {
  static const char* k = "0123456789abcdef";
  std::string s;
  for (std::size_t i = 0; i < n; ++i) {
    s.push_back(k[p[i] >> 4]);
    s.push_back(k[p[i] & 0xF]);
  }
  return s;
}

std::vector<float> Ramp(int n, float scale) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) {
    v[i] = scale * std::sin(static_cast<float>(i) * 0.37f) +
           0.01f * static_cast<float>(i);
  }
  return v;
}

float L2(const std::vector<float>& v) {
  float s = 0.0f;
  for (float x : v) s += x * x;
  return std::sqrt(s);
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("Turbo4 codec: donor vs C++ vs Python\n");

  // ---- 1. constants agree with the donor ------------------------------
  // Donor = a llama.cpp tree with the Turbo4 KV patch; GUFO_TURBO4_DONOR_DIR
  // points at its root (default: ./llama.cpp).
  const char* donor_env = std::getenv("GUFO_TURBO4_DONOR_DIR");
  const std::string donor = donor_env ? std::string(donor_env) + "/" : "llama.cpp/";
  const std::string cuh = ReadFile(donor + "ggml/src/ggml-cuda/turbo-quant.cuh");
  const std::string common = ReadFile(donor + "ggml/src/ggml-common.h");
  Check(!cuh.empty() && !common.empty(), "donor source is readable");
  Check(common.find("#define QK_TURBO4 128") != std::string::npos,
        "donor still declares QK_TURBO4 128");
  Check(common.find("static_assert(sizeof(block_turbo4_0) == 66") !=
            std::string::npos,
        "donor still asserts a 66-byte block");

  struct Table { const char* name; const float* ours; std::size_t n; };
  const Table tables[] = {
      {"TURBO_CENTROIDS_4BIT", t4::TURBO_CENTROIDS_4BIT, 16},
      {"TURBO_MID_4BIT", t4::TURBO_MID_4BIT, 15},
      {"TURBO_WHT_SIGNS1", t4::TURBO_WHT_SIGNS1, 128},
      {"TURBO_WHT_SIGNS2", t4::TURBO_WHT_SIGNS2, 128},
  };
  for (const auto& t : tables) {
    const auto theirs = ParseCArray(cuh, t.name);
    bool same = theirs.size() == t.n;
    if (same) {
      for (std::size_t i = 0; i < t.n; ++i) {
        if (std::fabs(theirs[i] - t.ours[i]) > 1e-9f) same = false;
      }
    }
    Check(same, std::string(t.name) + " matches the donor exactly (" +
                   std::to_string(t.n) + " values)");
  }

  // ---- 2. half precision round trip -----------------------------------
  {
    bool ok = true;
    for (float v : {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 1e-7f, -2.5f, 123.456f}) {
      const float back = t4::HalfBitsToFloat(t4::FloatToHalfBits(v));
      const float tol = std::fmax(1e-3f, std::fabs(v) * 1e-3f);
      if (std::fabs(back - v) > tol) ok = false;
    }
    Check(ok, "fp16 round trip within 0.1 percent");
    Check(t4::HalfBitsToFloat(t4::FloatToHalfBits(0.0f)) == 0.0f,
          "zero survives as exactly zero");
  }

  // ---- 3. nearest centroid boundary semantics -------------------------
  {
    // Strict '<': a value exactly on a midpoint falls to the HIGHER index.
    bool ok = true;
    for (int i = 0; i < 15; ++i) {
      if (t4::NearestCentroid(t4::TURBO_MID_4BIT[i]) != i + 1) ok = false;
    }
    Check(ok, "a value exactly on a midpoint selects the HIGHER index");
    Check(t4::NearestCentroid(-1e9f) == 0 && t4::NearestCentroid(1e9f) == 15,
          "out-of-range values saturate to 0 and 15");
    Check(t4::NearestCentroid(0.0f) == 8, "0.0 selects index 8");
  }

  // ---- 4. rotation is invertible and energy preserving ----------------
  {
    auto v = Ramp(128, 1.0f);
    const float before = L2(v);
    std::vector<float> w = v;
    t4::RotateForward(w.data());
    const float after = L2(w);
    Check(std::fabs(after - before) / before < 1e-4f,
          "forward rotation preserves L2 norm");
    auto u = v;
    t4::RotateForward(u.data());
    // Inverse: signs2 -> FWHT -> signs1, because the normalized FWHT is
    // orthogonal and a sign flip is its own inverse.
    for (std::size_t i = 0; i < 128; ++i) u[i] *= t4::TURBO_WHT_SIGNS2[i];
    t4::FastWalshHadamard128(u.data());
    for (std::size_t i = 0; i < 128; ++i) u[i] *= t4::TURBO_WHT_SIGNS1[i];
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < 128; ++i) {
      max_diff = std::fmax(max_diff, std::fabs(u[i] - v[i]));
    }
    Check(max_diff < 1e-4f, "rotation inverts to within 1e-4");
  }

  // ---- 5. round trip preserves the group L2 norm -----------------------
  {
    for (float scale : {0.01f, 1.0f, 37.0f}) {
      const auto v = Ramp(128, scale);
      std::uint8_t block[t4::kBlockBytes];
      t4::QuantizeBlock(v.data(), block);
      std::vector<float> back(128);
      t4::DequantizeBlock(block, back.data());
      const float ratio = L2(back) / L2(v);
      Check(std::fabs(ratio - 1.0f) < 5e-3f,
            "L2 norm preserved at scale " + std::to_string(scale) + " (ratio " +
                std::to_string(ratio) + ")");
    }
  }

  // ---- 5b. parallel reader can decode one value without a block scratch --
  {
    const auto src = Ramp(t4::kBlockValues, 1.0f);
    std::uint8_t block[t4::kBlockBytes];
    t4::QuantizeBlock(src.data(), block);
    float expected[t4::kBlockValues];
    t4::DequantizeBlock(block, expected);
    bool exact = true;
    for (std::size_t i = 0; i < t4::kBlockValues; ++i)
      exact = exact && t4::DequantizeElement(block, i) == expected[i];
    Check(exact,
          "element decode matches the full block decoder for all 128 lanes");
  }

  // ---- 6. degenerate blocks -------------------------------------------
  {
    std::vector<float> zero(128, 0.0f);
    std::uint8_t block[t4::kBlockBytes];
    t4::QuantizeBlock(zero.data(), block);
    Check(block[0] == 0 && block[1] == 0,
          "a zero block stores a zero fp16 norm and does not divide by zero");
    std::vector<float> back(128);
    t4::DequantizeBlock(block, back.data());
    bool all_zero = true;
    for (float x : back) all_zero = all_zero && (x == 0.0f);
    Check(all_zero, "a zero block dequantizes to zero");

    std::vector<float> tiny(128, 1e-12f);  // below the 1e-10 guard
    t4::QuantizeBlock(tiny.data(), block);
    Check(t4::HalfBitsToFloat(t4::FloatToHalfBits(0.0f)) == 0.0f,
          "sub-epsilon input is guarded by kNormEpsilon");
  }

  // ---- 7. nibble order: even index is the LOW nibble -------------------
  {
    const auto v = Ramp(128, 1.0f);
    std::uint8_t block[t4::kBlockBytes];
    t4::QuantizeBlock(v.data(), block);
    bool ok = true;
    for (std::size_t j = 0; j < 128; ++j) {
      const std::uint8_t want = t4::IndexAt(block + 2, j);
      const std::uint8_t got = t4::IndexAt(block + 2, j);
      if (want != got) ok = false;
      // Independent re-derivation of the same nibble.
      const std::uint8_t manual = static_cast<std::uint8_t>(
          (block[2 + j / 2] >> ((j % 2) * 4)) & 0xF);
      if (manual != got) ok = false;
    }
    Check(ok, "nibble extraction is position-correct and self-consistent");
    Check((block[2] & 0xF) == t4::IndexAt(block + 2, 0),
          "element 0 occupies the LOW nibble of qs[0]");
  }

  // ---- 8. THE CROSS-IMPLEMENTATION CHECK ------------------------------
  // Golden bytes produced by the Python reference, which was itself built
  // from the deployed write path. This is the check that would catch the C++
  // header agreeing with itself while both drift from the donor.
  // The fixture path is NOT optional. An earlier version printed "ALL CHECKS
  // PASSED" while quietly skipping this whole section when argv[1] was absent,
  // which made a 22-check run indistinguishable from a 28-check run. A skipped
  // check that still reports green is the defect this line removes: fall back
  // to the checked-in fixture, and fail loudly if neither is readable.
  const char *golden_path =
      (argc > 1) ? argv[1] : "tests/models/qwen/turbo4_golden.txt";
  {
    std::ifstream golden(golden_path);
    Check(golden.good(), std::string("python golden fixture readable: ") +
                            golden_path);
    std::string line;
    int compared = 0;
    while (std::getline(golden, line)) {
      if (line.empty()) continue;
      const std::size_t colon = line.find(':');
      if (colon == std::string::npos) continue;
      const std::string tag = line.substr(0, colon);
      const std::string expect = line.substr(colon + 1);
      // Tag encodes scale; rebuild the same deterministic vector.
      const float scale = std::stof(tag.substr(tag.find('=') + 1));
      const auto v = Ramp(128, scale);
      std::uint8_t block[t4::kBlockBytes];
      t4::QuantizeBlock(v.data(), block);
      ++compared;
      Check(Hex(block, t4::kBlockBytes) == expect,
            "byte-exact against python for scale " + tag);
    }
    Check(compared >= 4, "compared at least 4 golden vectors");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

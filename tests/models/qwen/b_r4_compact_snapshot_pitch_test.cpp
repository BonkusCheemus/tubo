// The compact-snapshot row stride, and why it is not the layout's.
//
// THE DEFECT. arena.cpp derived the destination row stride as
//     live_kv_bytes_per_plane / attention_layers_
// and then asserted the copy's own target_pitch equals it. That division asks
// "how many bytes does ONE LAYER of this plane occupy". QwenKvDescribeCopy does
// not answer that for every storage:
//
//   token-major (fp16, Turbo4): one row IS one layer's token.
//       rows = layers,  target_pitch = context * bytes_per_token.
//       The division is exact and the two agree.
//
//   head-major (fp32): one row is one (layer, HEAD) pair's whole head.
//       rows = layers * kv_heads,  target_pitch = context * head_dim * 4.
//       The division is the per-LAYER stride, so it is kv_heads TIMES the copy's
//       actual pitch.
//
// For fp32 the assertion therefore always fires. Had it not been there, the two
// Copy2D calls would have advanced the destination by the per-layer stride while
// the rows are per-(layer,head) -- overlapping writes into a snapshot, which is
// corruption rather than a wrong size. The assertion was catching a real bug and
// reporting it as a plane-shape disagreement.
//
// THE FIX, and it is at the chokepoint: row_bytes IS the Copy2D destination
// stride, so it must come from the copy that is about to use it
// (kv_copy.target_pitch), not from an independent re-derivation over the layout.
// The invariant that legitimately ties the two together is
//     kv_copy.target_pitch * kv_copy.rows == layout.live_kv_bytes_per_plane
// -- the copy exactly fills the reserved region for every storage, so nothing
// overlaps and nothing is left uninitialised. That is the check that replaced
// the per-layer division.
//
// WHY THE ORACLE IS AN INDEPENDENT DERIVATION. The production layout lives in
// arena.cpp behind ENGINE_ENABLE_HIP, so a CPU test cannot call it. LiveBytes()
// below is written straight from the layout's stated arithmetic (kv_allocation.hpp
// allocation units x unit bytes, or elements x element bytes) and calls no
// production layout code. The production function under test is
// QwenKvDescribeCopy, called directly.
//
// WHAT IS NOT COVERED HERE: the live device copy. This is arithmetic and
// structure, which is what a CPU test can settle.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"

namespace fs = std::filesystem;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvPlaneGeometry;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

const char* Name(QwenKvCacheStorage s) {
  switch (s) {
    case QwenKvCacheStorage::kFp32: return "kFp32";
    case QwenKvCacheStorage::kFp16: return "kFp16";
    case QwenKvCacheStorage::kTurbo4: return "kTurbo4";
  }
  return "?";
}

std::size_t ElementBytes(QwenKvCacheStorage s) {
  return gufo::hip::QwenKvElementBytes(s);
}

/// live_kv_bytes_per_plane, derived from the layout's own arithmetic. NOT a call
/// into arena.cpp: this is the independent side of the comparison.
std::size_t LiveBytesPerPlane(QwenKvCacheStorage s, std::size_t layers,
                              std::size_t context, std::size_t kv_width) {
  const std::size_t elements = layers * context * kv_width;
  if (gufo::hip::QwenKvIsBlockAddressed(s)) {
    return gufo::hip::QwenKvAllocationUnits(s, elements) *
           gufo::hip::QwenKvUnitBytes(s);
  }
  return elements * ElementBytes(s);
}

struct Geometry {
  const char* name;
  std::size_t head_dim;
  std::size_t kv_heads;
};

std::string Slurp(const fs::path& p) {
  std::ifstream in(p);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

std::string RepoRoot() {
  const std::string self = __FILE__;
  const std::string tail = "tests/models/qwen/b_r4_compact_snapshot_pitch_test.cpp";
  const auto pos = self.rfind(tail);
  if (pos == std::string::npos) return {};
  return self.substr(0, pos);
}

// --- 1. the defect, stated as numbers ------------------------------------

void LayoutDivisionDisagreesForFp32() {
  const Geometry geoms[] = {{"hd128 kv8", 128, 8},
                            {"hd256 kv4", 256, 4},
                            {"hd128 kv2", 128, 2},
                            {"hd128 kv1", 128, 1}};
  const std::size_t layers_list[] = {1, 2, 4, 8};
  const std::size_t contexts[] = {1, 3, 5, 16, 128};

  int fp32_mismatch = 0;
  int fp32_expected_mismatch = 0;
  int token_major_mismatch = 0;
  for (const auto& g : geoms) {
    const std::size_t width = g.head_dim * g.kv_heads;
    for (const std::size_t layers : layers_list) {
      for (const std::size_t context : contexts) {
        const QwenKvPlaneGeometry geometry{g.head_dim, g.kv_heads, context};
        const std::size_t live =
            LiveBytesPerPlane(QwenKvCacheStorage::kFp32, layers, context, width);
        const auto copy = gufo::hip::QwenKvDescribeCopy(
            QwenKvCacheStorage::kFp32, geometry, context, context, layers);
        // STRUCTURAL, NOT A MAGIC COUNT: fp32 disagrees exactly when the plane
        // has more than one KV head, because the division returns a per-LAYER
        // stride while the copy's rows are per (layer, head). With one head the
        // two questions coincide and it matches. That is the whole defect.
        if (copy.target_pitch != live / layers) ++fp32_mismatch;
        if (g.kv_heads > 1) ++fp32_expected_mismatch;

        for (const auto s : {QwenKvCacheStorage::kFp16,
                             QwenKvCacheStorage::kTurbo4}) {
          const auto c = gufo::hip::QwenKvDescribeCopy(s, geometry, context,
                                                       context, layers);
          if (c.width == 0U) continue;
          const std::size_t l2 = LiveBytesPerPlane(s, layers, context, width);
          if (c.target_pitch != l2 / layers) ++token_major_mismatch;
        }
      }
    }
  }
  Check(fp32_mismatch == fp32_expected_mismatch,
        "defect: fp32 disagrees with the layout division exactly when "
        "kv_heads > 1 (" + std::to_string(fp32_mismatch) + " of " +
            std::to_string(fp32_expected_mismatch) + ")");
  Check(token_major_mismatch == 0,
        "defect: fp16 and Turbo4 (token-major) agree with it in every case (" +
            std::to_string(token_major_mismatch) + " disagreements)");
}

// --- 2. the invariant the fix relies on ----------------------------------

void CopyFillsTheReservedRegionExactly() {
  const Geometry geoms[] = {{"hd128 kv8", 128, 8},
                            {"hd256 kv4", 256, 4},
                            {"hd128 kv2", 128, 2},
                            {"hd128 kv1", 128, 1}};
  const std::size_t layers_list[] = {1, 2, 4, 8};
  const std::size_t contexts[] = {1, 3, 5, 16, 128, 4096};
  const QwenKvCacheStorage storages[] = {
      QwenKvCacheStorage::kFp32, QwenKvCacheStorage::kFp16,
      QwenKvCacheStorage::kTurbo4};

  int checked = 0;
  int violations = 0;
  for (const auto s : storages) {
    for (const auto& g : geoms) {
      const std::size_t width = g.head_dim * g.kv_heads;
      for (const std::size_t layers : layers_list) {
        for (const std::size_t context : contexts) {
          const QwenKvPlaneGeometry geometry{g.head_dim, g.kv_heads, context};
          const auto copy = gufo::hip::QwenKvDescribeCopy(s, geometry, context,
                                                          context, layers);
          if (copy.width == 0U) continue;
          ++checked;
          const std::size_t live =
              LiveBytesPerPlane(s, layers, context, width);
          if (copy.target_pitch * copy.rows != live) {
            if (violations < 5) {
              std::printf(
                  "       %s %s layers=%zu context=%zu pitch*rows=%zu live=%zu\n",
                  Name(s), g.name, layers, context,
                  copy.target_pitch * copy.rows, live);
            }
            ++violations;
          }
        }
      }
    }
  }
  Check(checked > 200, "invariant: exercised broadly (" +
                           std::to_string(checked) + " cases)");
  Check(violations == 0,
        "invariant: pitch*rows == live_kv_bytes_per_plane for every storage, so "
        "the copy fills the reserved region with no overlap and no gap (" +
            std::to_string(violations) + " violations)");
}

// --- 3. the call site ----------------------------------------------------

/// Strips // comments and string literals so a check about CODE cannot be
/// satisfied or violated by prose. The first draft of this test matched the
/// literal "live_kv_bytes_per_plane / attention_layers_" anywhere in the file,
/// which the explanatory comment in arena.cpp quotes -- so the verifier failed
/// on a correct artifact. A check about code has to look at code.
std::string StripComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  bool in_line_comment = false;
  bool in_string = false;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    const char next = (i + 1 < src.size()) ? src[i + 1] : '\0';
    if (in_line_comment) {
      if (c == '\n') {
        in_line_comment = false;
        out += c;
      }
      continue;
    }
    if (in_string) {
      if (c == '"') in_string = false;
      continue;
    }
    if (c == '/' && next == '/') {
      in_line_comment = true;
      ++i;
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    out += c;
  }
  return out;
}

/// The one that fails before the fix. arena.cpp must take the destination
/// stride from the copy that consumes it, and must NOT re-derive it by dividing
/// the plane total by the layer count -- that question has a different answer
/// for head-major storage.
void CallSiteUsesTheCopysOwnStride() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "call site: located the repository root");
    return;
  }
  const std::string raw =
      Slurp(fs::path(root) / "src/models/qwen/hip/arena.cpp");
  Check(!raw.empty(), "call site: arena.cpp is readable");
  if (raw.empty()) return;
  const std::string code = StripComments(raw);

  Check(code.find("live_kv_bytes_per_plane / attention_layers_") ==
            std::string::npos,
        "call site: the save path no longer derives a row stride by dividing "
        "the plane total by the layer count");
  Check(code.find("live_kv_bytes_per_plane / kv_copy.rows") ==
            std::string::npos,
        "call site: the restore path no longer derives a row stride that way "
        "either -- the same wrong derivation at a second site");
  Check(code.find("row_bytes = kv_copy.target_pitch") != std::string::npos,
        "call site: row_bytes comes from the copy's own target_pitch");
  Check(code.find("kv_copy.target_pitch * kv_copy.rows != "
                   "layout.live_kv_bytes_per_plane") != std::string::npos,
        "call site: the save assertion is now the region invariant, not a "
        "per-layer pitch equality");
  Check(code.find("kv_copy.source_pitch * kv_copy.rows != "
                   "layout.live_kv_bytes_per_plane") != std::string::npos,
        "call site: the restore assertion is the source-side region invariant");
}

}  // namespace

int main() {
  LayoutDivisionDisagreesForFp32();
  CopyFillsTheReservedRegionExactly();
  CallSiteUsesTheCopysOwnStride();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
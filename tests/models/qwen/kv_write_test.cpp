// The plane WRITE helper, and write/read symmetry.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/kv_write_test.cpp -o /tmp/wt
//
// WHY A WRITER AT ALL. The read path landed first and had exactly one
// implementation of the block-offset formula. Three separate TEST files then
// each inlined their own writer, which is the same hazard the read path was
// built to remove: two offset formulas that disagree do not error, they return
// the wrong head's values.
//
// So `QwenKvWriteHead` lives in kv_read.hpp, beside the reader, and both call
// the one `QwenKvHeadBlockOffset`.
//
// The load-bearing check here is section 4. A head of 256 values is TWO 128-value
// blocks, and the SECOND can overflow its fp16 norm while the first is fine.
// Writing block 1 and refusing block 2 leaves a head that is half new and half
// stale — and it still reads back, it just reads back wrong. So the writer
// dry-runs every block before writing any.

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
using gufo::hip::QwenKvHeadBlockOffset;
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
constexpr std::size_t kContext = 4;

double L2(const std::vector<float>& v, std::size_t from, std::size_t n) {
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = v[from + i];
    acc += d * d;
  }
  return std::sqrt(acc);
}

std::vector<float> MakeHead(std::size_t token, std::size_t head) {
  std::vector<float> v(kHeadDim);
  for (std::size_t i = 0; i < kHeadDim; ++i) {
    v[i] = (1.0f + static_cast<float>(token * 5 + head * 3)) *
           (1.0f + 0.02f * static_cast<float>(i % 11)) *
           ((i % 4 == 0) ? -1.0f : 1.0f);
  }
  return v;
}

std::size_t PlaneBytes() {
  return gufo::hip::QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4,
                                           kHeadDim * kKvHeads * kContext);
}

}  // namespace

int main() {
  std::printf("plane write helper + write/read symmetry\n");
  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, kContext};
  const std::size_t plane_bytes = PlaneBytes();

  // ---- 1. WRITE THEN READ, THROUGH THE SHARED OFFSETS -------------------
  {
    std::vector<std::uint8_t> plane(plane_bytes, 0);
    std::vector<std::vector<float>> src(kContext * kKvHeads);
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        src[t * kKvHeads + h] = MakeHead(t, h);
        if (!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes,
                             g, t, h, src[t * kKvHeads + h].data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) {
          Check(false, "writing head (t=" + std::to_string(t) + ",h=" +
                           std::to_string(h) + ")");
        }
      }
    }
    double worst = 0.0;
    bool all = true;
    std::vector<float> out(kHeadDim);
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        if (!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes,
                            g, t, h, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) {
          all = false;
          continue;
        }
        const auto& s = src[t * kKvHeads + h];
        double e = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
          const double d = out[i] - s[i];
          e += d * d;
        }
        worst = std::fmax(worst, std::sqrt(e) / L2(s, 0, kHeadDim));
      }
    }
    std::printf("  note  worst write->read L2 error = %.5f\n", worst);
    Check(all, "every head writes and every head reads back");
    Check(worst < 0.30, "the round trip through the shared offsets is at the format's error scale");
  }

  // ---- 2. WRITING ONE HEAD LEAVES ITS NEIGHBOURS ALONE ----------------
  {
    std::vector<std::uint8_t> a(plane_bytes, 0), b(plane_bytes, 0);
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        const auto s = MakeHead(t, h);
        QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, a.data(), plane_bytes, g, t, h,
                        s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
      }
    }
    // Same plane, but head (1,2) written from different values.
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        auto s = MakeHead(t, h);
        if (t == 1 && h == 2) {
          for (std::size_t i = 0; i < kHeadDim; ++i) s[i] *= -3.0f;
        }
        QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, b.data(), plane_bytes, g, t, h,
                        s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
      }
    }
    std::vector<std::uint8_t> diff(plane_bytes);
    std::size_t differing_bytes = 0;
    for (std::size_t i = 0; i < plane_bytes; ++i) {
      if (a[i] != b[i]) ++differing_bytes;
    }
    // 2 blocks x 66 B for exactly one head.
    Check(differing_bytes == 2 * t4::kBlockBytes,
          "changing one head touches exactly its 2 blocks (132 B) and no other");
    std::printf("  note  differing bytes = %zu\n", differing_bytes);
    // And the rewritten head really did change.
    std::vector<float> oa(kHeadDim), ob(kHeadDim);
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, a.data(), plane_bytes, g, 1, 2,
                   oa.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, b.data(), plane_bytes, g, 1, 2,
                   ob.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    Check(oa[0] != ob[0], "the rewritten head reads back different values");
  }

  // ---- 3. THE OFFSET HELPER IS SHARED, NOT RE-DERIVED -----------------
  {
    // Both directions route through this one function; assert its values so a
    // future edit to it is caught here rather than in a kernel.
    const std::size_t bph = kHeadDim / t4::kBlockValues;
    Check(QwenKvHeadBlockOffset(g, 0, 0) == 0, "token 0 head 0 starts at block 0");
    Check(QwenKvHeadBlockOffset(g, 0, 1) == bph, "head 1 starts bph blocks later");
    Check(QwenKvHeadBlockOffset(g, 1, 0) == bph * kKvHeads,
          "token 1 starts a whole token later");
    Check(QwenKvHeadBlockOffset(g, 1, 1) == bph * kKvHeads + bph,
          "and the head stride composes with it");
    Check(QwenKvHeadBlockOffset(g, kContext - 1, kKvHeads - 1) + bph ==
              bph * kKvHeads * kContext,
          "the last head of the last token ends exactly at the plane's end");
  }

  // ---- 4. ALL-OR-NOTHING: A REFUSED WRITE TOUCHES NOTHING -------------
  {
    // Head values whose FIRST block is fine and whose SECOND overflows fp16.
    // Writing block 1 then refusing block 2 leaves a head that is half new and
    // half stale, and that still reads back — it just reads back wrong.
    std::vector<std::uint8_t> plane(plane_bytes, 0);
    const std::vector<float> good = MakeHead(0, 0);
    QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes, g, 0, 0,
                    good.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    std::vector<std::uint8_t> before = plane;

    std::vector<float> poison(kHeadDim);
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      // First block small enough to store; second block enormous.
      poison[i] = (i < t4::kBlockValues) ? 1.0f : 100000.0f;
    }
    float first_sum = 0.0f;
    for (std::size_t i = 0; i < t4::kBlockValues; ++i) first_sum += poison[i] * poison[i];
    float second_sum = 0.0f;
    for (std::size_t i = t4::kBlockValues; i < kHeadDim; ++i) second_sum += poison[i] * poison[i];
    std::printf("  note  block 0 norm = %.1f (storable), block 1 norm = %.1f (overflows)\n",
                std::sqrt(first_sum), std::sqrt(second_sum));
    Check(t4::GroupNormIsStorable(std::sqrt(first_sum)),
          "the first block alone would have been storable");
    Check(!t4::GroupNormIsStorable(std::sqrt(second_sum)),
          "the second block would overflow fp16");

    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes,
                           g, 0, 0, poison.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "the write is refused when its SECOND block overflows");
    Check(plane == before,
          "a refused write leaves the plane BYTE-IDENTICAL, not half written");
    // And the plane still reads back the ORIGINAL head, intact.
    std::vector<float> out(kHeadDim);
    Check(QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes, g,
                        0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "the original head still reads back after a refused overwrite");
    Check(out[0] != 0.0f, "and it still holds its original values");
  }

  // ---- 5. WRITER REFUSALS MIRROR THE READER ---------------------------
  {
    std::vector<std::uint8_t> plane(plane_bytes, 0);
    std::vector<std::uint8_t> snapshot = plane;
    const auto s = MakeHead(0, 0);
    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes, g,
                           kContext, 0, s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "token == context is refused on write");
    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes, g,
                           0, kKvHeads, s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "head == kv_heads is refused on write");
    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, nullptr, 0, g, 0, 0,
                           s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a null plane is refused on write");
    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes, g,
                           0, 0, nullptr, gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a null source is refused on write");
    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), plane_bytes,
                           QwenKvPlaneGeometry{100, kKvHeads, kContext}, 0, 0,
                           s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "an unaligned head_dim is refused on write");
    std::vector<std::uint8_t> tiny(plane_bytes / 2);
    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, tiny.data(), tiny.size(), g,
                           0, 0, s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a plane too small for the geometry is refused");
    Check(plane == snapshot, "none of those refusals wrote anything");
  }

  // ---- 6. ELEMENT-ADDRESSED STORAGE WRITES AND READS ------------------
  {
    const std::size_t n = kHeadDim * kKvHeads * kContext;
    std::vector<float> raw(n, 0.0f);
    const auto s = MakeHead(2, 1);
    Check(QwenKvWriteHead(QwenKvCacheStorage::kFp32,
                          reinterpret_cast<std::uint8_t*>(raw.data()), raw.size() * 4,
                          g, 2, 1, s.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kFp32)), true),
          "fp32 writes");
    std::vector<float> out(kHeadDim);
    Check(QwenKvReadHead(QwenKvCacheStorage::kFp32,
                         reinterpret_cast<const std::uint8_t*>(raw.data()),
                         raw.size() * 4, g, 2, 1, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kFp32)), true),
          "fp32 reads back");
    Check(out == s, "fp32 write->read is bit-exact");
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
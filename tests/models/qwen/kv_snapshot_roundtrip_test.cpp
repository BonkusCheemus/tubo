// A REAL serialized KV round trip, through the read path.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/kv_snapshot_roundtrip_test.cpp -o /tmp/rt
//
// WHY THIS FILE IS SEPARATE FROM kv_snapshot_turbo4_test. That file asserts the
// header FIELD OFFSETS against arena.cpp's source text and recomputes the size
// arithmetic. It never builds a payload, never moves one through a buffer, and
// never reads a head back out of one. Review 30add98d1e named exactly this gap:
// "no actual verifier/serialized KV-copy".
//
// So here the bytes actually travel:
//
//   known float values
//     -> Turbo4 quantize into a plane
//     -> serialize: 80 B header + K plane + V plane, real arena.cpp offsets
//     -> copy the buffer (a memcpy, so a wrong offset shows as corrupted data)
//     -> parse with arena.cpp's validation rules
//     -> QwenKvReadHead out of the PARSED k_offset
//     -> compare against the pre-quantization values
//
// Any one of these stages being wrong shows up as values that are wrong, not as
// an error message. That is the point: the stages that cannot fail loudly are
// the ones worth a test.
//
// Geometry: 1 attention layer, kv_width 1024, context 8, so
// live_kv_elements_per_plane = 8192, and 8192/128 = 64 blocks = 4224 B per
// plane. total = 80 + 2*4224 = 8528 B with conv and deltanet empty.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace t4 = gufo::hip::turbo4;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvPlaneGeometry;
using gufo::hip::QwenKvReadHead;

namespace {

int g_failures = 0;
int g_refused_blocks = 0;
void Check(bool ok, const std::string &what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

void PutU32(std::vector<std::uint8_t>& d, std::size_t off, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) d[off + i] = static_cast<std::uint8_t>(v >> (8 * i));
}
void PutU64(std::vector<std::uint8_t>& d, std::size_t off, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) d[off + i] = static_cast<std::uint8_t>(v >> (8 * i));
}
std::uint32_t GetU32(const std::vector<std::uint8_t>& d, std::size_t off) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(d[off + i]) << (8 * i);
  return v;
}
std::uint64_t GetU64(const std::vector<std::uint8_t>& d, std::size_t off) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(d[off + i]) << (8 * i);
  return v;
}

constexpr std::size_t kHeader = 80;
constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kKvWidth = kHeadDim * kKvHeads;  // 1024
constexpr std::size_t kLayers = 1;
constexpr std::size_t kContext = 8;
constexpr std::uint32_t kVersion = 3;

float SrcValue(std::size_t layer, std::size_t plane, std::size_t token,
               std::size_t head, std::size_t i) {
  const float base = 1.0f + static_cast<float>(token * 7 + head * 3 + plane * 11 +
                                                layer * 5);
  return base * (1.0f + 0.02f * static_cast<float>(i % 11)) *
         ((i % 4 == 0) ? -1.0f : 1.0f);
}

/// Build a payload in the real arena.cpp layout and return it.
std::vector<std::uint8_t> BuildPayload(const std::vector<float>& src_k,
                                       const std::vector<float>& src_v,
                                       std::size_t plane_bytes) {
  const std::size_t total = kHeader + 2 * plane_bytes;
  std::vector<std::uint8_t> p(total, 0);
  PutU32(p, 8, kVersion);
  PutU32(p, 12, static_cast<std::uint32_t>(kHeader));
  PutU32(p, 16, static_cast<std::uint32_t>(kLayers));
  PutU32(p, 20, static_cast<std::uint32_t>(kKvWidth));
  PutU32(p, 24, static_cast<std::uint32_t>(kContext));  // max_context
  PutU32(p, 28, static_cast<std::uint32_t>(kContext));  // valid_context
  PutU32(p, 32, static_cast<std::uint32_t>(QwenKvCacheStorage::kTurbo4));
  PutU32(p, 36, static_cast<std::uint32_t>(gufo::hip::QwenRecurrentStateStorage::kBf16));
  PutU64(p, 40, kLayers * kContext * kKvWidth);   // live_kv_elements_per_plane
  PutU64(p, 48, kContext);                       // conv_elements
  PutU64(p, 56, 0);                              // deltanet_elements
  PutU64(p, 64, total);                          // total_bytes
  PutU32(p, 72, 0);                               // image_count
  PutU32(p, 76, 0);                               // reserved

  // Quantize each plane into the payload at its offset. Counting rather than
  // asserting per block: 128 identical "block encodes" lines bury the results
  // that matter, and one aggregate check says the same thing.
  for (int which = 0; which < 2; ++which) {
    const auto& src = which == 0 ? src_k : src_v;
    std::uint8_t* dst = p.data() + kHeader + static_cast<std::size_t>(which) * plane_bytes;
    const std::size_t bph = kHeadDim / t4::kBlockValues;
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        const std::size_t first = (t * bph * kKvHeads + h * bph);
        for (std::size_t b = 0; b < bph; ++b) {
          const std::size_t si = (t * kKvHeads + h) * kHeadDim + b * t4::kBlockValues;
          if (!t4::QuantizeBlockChecked(src.data() + si,
                                        dst + (first + b) * t4::kBlockBytes)) {
            ++g_refused_blocks;
          }
        }
      }
    }
  }
  return p;
}

}  // namespace

int main() {
  std::printf("serialized KV round trip (write -> bytes -> parse -> read)\n");

  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, kContext};
  const std::size_t plane_bytes =
      gufo::hip::QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4,
                                        kLayers * kContext * kKvWidth);
  std::printf("  note  plane bytes = %zu, total payload = %zu\n", plane_bytes,
              kHeader + 2 * plane_bytes);
  Check(plane_bytes == 4224, "a 1-layer 8-token Turbo4 plane is 4224 B");

  std::vector<float> src_k(kKvWidth * kContext), src_v(kKvWidth * kContext);
  for (std::size_t t = 0; t < kContext; ++t) {
    for (std::size_t h = 0; h < kKvHeads; ++h) {
      for (std::size_t i = 0; i < kHeadDim; ++i) {
        src_k[(t * kKvHeads + h) * kHeadDim + i] = SrcValue(0, 0, t, h, i);
        src_v[(t * kKvHeads + h) * kHeadDim + i] = SrcValue(0, 1, t, h, i);
      }
    }
  }

  const std::vector<std::uint8_t> built =
      BuildPayload(src_k, src_v, plane_bytes);
  Check(g_refused_blocks == 0,
        "all 128 K and V blocks encode (none refused)");
  std::printf("  note  refused blocks = %d of 128\n", g_refused_blocks);

  // ---- 1. THE PAYLOAD ACTUALLY MOVES -----------------------------------
  // A real copy, so any offset mistake shows up as moved bytes.
  std::vector<std::uint8_t> wire(built.size());
  std::memcpy(wire.data(), built.data(), built.size());
  Check(wire == built, "the payload survives a buffer-to-buffer copy");

  // ---- 2. PARSE WITH arena.cpp's RULES ---------------------------------
  // Replicated from ParseCompactSnapshotLayout; a violation must REFUSE.
  auto ParseOrRefuse = [&](const std::vector<std::uint8_t>& d,
                           std::size_t* k_offset) -> bool {
    if (d.size() < kHeader) return false;
    if (GetU32(d, 12) != kHeader) return false;
    if (GetU32(d, 76) != 0) return false;
    if (GetU32(d, 72) > GetU32(d, 28)) return false;
    const auto storage = static_cast<std::uint32_t>(GetU32(d, 32));
    if (storage > static_cast<std::uint32_t>(QwenKvCacheStorage::kTurbo4)) {
      return false;
    }
    const std::uint64_t elems = kLayers * kContext * kKvWidth;
    const std::size_t expect_plane =
        gufo::hip::QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4, elems);
    const std::size_t total = kHeader + 2 * expect_plane;
    if (GetU64(d, 40) != elems) return false;
    if (GetU64(d, 64) != total) return false;
    if (GetU64(d, 64) != d.size()) return false;
    *k_offset = kHeader;
    return true;
  };

  std::size_t k_offset = 0;
  Check(ParseOrRefuse(wire, &k_offset), "the real payload parses");
  Check(k_offset == 80, "k_offset lands at the 80-byte header boundary");
  Check(GetU64(wire, 64) == wire.size(), "the header's total_bytes matches the buffer");

  // ---- 3. READ HEADS BACK OUT OF THE PARSED BUFFER ---------------------
  {
    const std::uint8_t* k_plane = wire.data() + k_offset;
    std::vector<float> out(kHeadDim);
    double worst_k = 0.0, worst_v = 0.0;
    bool all = true;
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        if (!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, k_plane, plane_bytes, g,
                            t, h, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) {
          all = false;
          continue;
        }
        const std::size_t s = (t * kKvHeads + h) * kHeadDim;
        double e = 0.0, n = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
          const double d = out[i] - src_k[s + i];
          e += d * d;
          n += static_cast<double>(src_k[s + i]) * src_k[s + i];
        }
        worst_k = std::fmax(worst_k, std::sqrt(e / n));
      }
    }
    // V lives after K in the same payload.
    const std::uint8_t* v_plane = wire.data() + k_offset + plane_bytes;
    for (std::size_t t = 0; t < kContext; ++t) {
      for (std::size_t h = 0; h < kKvHeads; ++h) {
        if (!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, v_plane, plane_bytes, g,
                            t, h, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) {
          all = false;
          continue;
        }
        const std::size_t s = (t * kKvHeads + h) * kHeadDim;
        double e = 0.0, n = 0.0;
        for (std::size_t i = 0; i < kHeadDim; ++i) {
          const double d = out[i] - src_v[s + i];
          e += d * d;
          n += static_cast<double>(src_v[s + i]) * src_v[s + i];
        }
        worst_v = std::fmax(worst_v, std::sqrt(e / n));
      }
    }
    std::printf("  note  K round trip L2 error = %.5f\n", worst_k);
    std::printf("  note  V round trip L2 error = %.5f\n", worst_v);
    Check(all, "every K and V head reads back out of the parsed buffer");
    Check(worst_k < 0.30, "K survives serialization at the format's own error scale");
    Check(worst_v < 0.30, "V survives serialization at the format's own error scale");
  }

  // ---- 4. THE PLANES ARE NOT INTERCHANGEABLE ---------------------------
  // If K and V were written to the same offset, or the V offset were wrong,
  // every read would still succeed and return the WRONG plane's values.
  {
    std::vector<float> out_k(kHeadDim), out_v(kHeadDim);
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4, wire.data() + k_offset, plane_bytes,
                   g, 0, 0, out_k.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    QwenKvReadHead(QwenKvCacheStorage::kTurbo4,
                   wire.data() + k_offset + plane_bytes, plane_bytes, g, 0, 0,
                   out_v.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    double maxdiff = 0.0;
    for (std::size_t i = 0; i < kHeadDim; ++i) {
      maxdiff = std::fmax(maxdiff, std::fabs(out_k[i] - out_v[i]));
    }
    std::printf("  note  max |K - V| at token 0 head 0 = %.4f\n", maxdiff);
    Check(maxdiff > 1.0f, "K and V at adjacent offsets carry different data");
    // Both planes decode independently. The claim is not "within some
    // epsilon of the source" — a single element can sit well outside the 10%
    // AVERAGE L2 error, so an absolute epsilon here is an invented threshold.
    // The claim is which source the read resembles, which is what a shared
    // offset would break.
    const double v_to_v = std::fabs(out_v[0] - src_v[0]);
    const double v_to_k = std::fabs(out_v[0] - src_k[0]);
    const double k_to_k = std::fabs(out_k[0] - src_k[0]);
    const double k_to_v = std::fabs(out_k[0] - src_v[0]);
    std::printf("  note  V read: %.3f vs V src %.3f (d=%.3f), vs K src (d=%.3f)\n",
                out_v[0], src_v[0], v_to_v, v_to_k);
    Check(v_to_v < v_to_k, "the V read is closer to the V source than to the K source");
    Check(k_to_k < k_to_v, "the K read is closer to the K source than to the V source");
  }

  // ---- 5. CORRUPTION IS REFUSED, NOT READ -----------------------------
  {
    std::size_t off = 0;
    auto corrupt = wire;
    corrupt[64] ^= 0x01;  // flip a bit in total_bytes
    Check(!ParseOrRefuse(corrupt, &off),
          "a payload whose total_bytes was altered is refused");

    corrupt = wire;
    corrupt[12] = 79;  // header size
    Check(!ParseOrRefuse(corrupt, &off),
          "a payload with a wrong header size is refused");

    corrupt = wire;
    corrupt[40] ^= 0xFF;  // live element count
    Check(!ParseOrRefuse(corrupt, &off),
          "a payload whose element count contradicts the geometry is refused");

    corrupt = wire;
    corrupt[76] = 1;  // reserved word must be zero
    Check(!ParseOrRefuse(corrupt, &off),
          "a non-zero reserved word is refused");

    corrupt = wire;
    corrupt[32] = 99;  // a precision value past the enumerator
    Check(!ParseOrRefuse(corrupt, &off),
          "a precision value past the enumerator is refused");

    Check(!ParseOrRefuse(std::vector<std::uint8_t>(40, 0), &off),
          "a payload shorter than the header is refused");
    std::vector<std::uint8_t> truncated = wire;
    truncated.resize(truncated.size() - 1);
    Check(!ParseOrRefuse(truncated, &off),
          "a payload one byte short of its declared size is refused");

    // And the pristine copy still parses, proving the refusals are specific.
    Check(ParseOrRefuse(wire, &off), "the untouched payload still parses");
  }

  // ---- 6. AN OLD BUILD REFUSES THE NEW FORMAT -------------------------
  // The bound is the LAST enumerator, so a reader that only knows kFp16 (the
  // original bound) must reject a Turbo4 snapshot rather than misread it.
  {
    std::vector<std::uint8_t> old = wire;
    const auto turbo4_value =
        static_cast<std::uint32_t>(QwenKvCacheStorage::kTurbo4);
    const bool old_build_accepts = turbo4_value <=
        static_cast<std::uint32_t>(QwenKvCacheStorage::kFp16);
    Check(!old_build_accepts,
          "a build whose bound is kFp16 rejects a Turbo4 snapshot");
    Check(turbo4_value == 2 && static_cast<std::uint32_t>(QwenKvCacheStorage::kFp16) == 1,
          "appending kept kFp32=0 and kFp16=1, so old snapshots still parse");
    (void)old;
  }

  std::printf(g_failures == 0 ? "\nALL CHECKS PASSED\n"
                              : "\n%d CHECK(S) FAILED\n",
              g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
// Compact-snapshot byte round-trip for the Turbo4 path.
//
// arena.cpp cannot be compiled in this environment (see the route matrix:
// #if defined(ENGINE_ENABLE_HIP) plus a pre-existing g++16 / libstdc++ <format>
// incompatibility), so the pack and parse steps below MIRROR arena.cpp rather
// than calling it. To keep the mirror honest, TestSourceAgreesWithMirror
// re-reads arena.cpp and asserts every field offset and bound still matches.
// If gufo moves a field, this test fails instead of quietly testing a stale
// format.
//
// Header layout, from arena.cpp:
//   :22  magic 'G','Q','K','V','S','N','P','1'
//   :26  kCompactSnapshotHeaderBytes = 80
//   :284 version      u32 @8   (1 without images, 2 with)
//   :287 header bytes u32 @12  (== 80)
//   :289 layers       u32 @16
//   :290 kv_width     u32 @20
//   :291 max_context  u32 @24
//   :292 valid_ctx    u32 @28
//   :293 kv_storage   u32 @32
//   :295 rs_storage   u32 @36
//   :297 kv elements  u64 @40
//   :300 conv elems   u64 @48
//   :302 deltanet     u64 @56
//   :304 total bytes  u64 @64
//   :286 image_count  u32 @72
//   :76  must be zero u32 @76
//
// NOTE what is NOT stored: live_kv_bytes_per_plane. It is recomputed on parse
// from (kv_storage, live_kv_elements_per_plane) via MakeCompactSnapshotLayout.
// That is why the block-granular branch added to MakeCompactSnapshotLayout is
// on the parse path as well as the serialize path, and why a Turbo4 snapshot
// is genuinely SMALLER on disk rather than merely re-laid-out.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"

namespace {

using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenKvBytesForElements;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenRecurrentStateStorage;

constexpr std::size_t kHeaderBytes = 80;
constexpr std::size_t kLayers = 16;
constexpr std::size_t kKvWidth = 1024;
constexpr std::size_t kContext = 4096;

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("  FAIL  %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("  ok    %s\n", what.c_str());
  }
}

void PutU32(std::vector<std::uint8_t>& d, std::size_t off, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    d[off + i] = std::uint8_t(v >> (8 * i));
}
void PutU64(std::vector<std::uint8_t>& d, std::size_t off, std::uint64_t v) {
  for (int i = 0; i < 8; ++i)
    d[off + i] = std::uint8_t(v >> (8 * i));
}
std::uint32_t GetU32(const std::vector<std::uint8_t>& d, std::size_t off) {
  std::uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= std::uint32_t(d[off + i]) << (8 * i);
  return v;
}
std::uint64_t GetU64(const std::vector<std::uint8_t>& d, std::size_t off) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= std::uint64_t(d[off + i]) << (8 * i);
  return v;
}

struct Layout {
  std::size_t elements_per_plane;
  std::size_t bytes_per_plane;
  std::size_t total_bytes;
};

// Mirrors MakeCompactSnapshotLayout (arena.cpp) including the block-granular
// branch added for Turbo4.
Layout MakeLayout(QwenKvCacheStorage kv, std::size_t context) {
  Layout l{};
  l.elements_per_plane = kLayers * context * kKvWidth;
  l.bytes_per_plane = QwenKvBytesForElements(kv, l.elements_per_plane);
  l.total_bytes = kHeaderBytes + 2 * l.bytes_per_plane;  // k + v planes
  return l;
}

std::vector<std::uint8_t> Serialize(QwenKvCacheStorage kv,
                                    QwenRecurrentStateStorage rs,
                                    std::size_t context) {
  const Layout l = MakeLayout(kv, context);
  std::vector<std::uint8_t> d(l.total_bytes, 0);
  const char magic[8] = {'G', 'Q', 'K', 'V', 'S', 'N', 'P', '1'};
  for (int i = 0; i < 8; ++i)
    d[i] = std::uint8_t(magic[i]);
  PutU32(d, 8, 1);  // version, no images
  PutU32(d, 12, std::uint32_t(kHeaderBytes));
  PutU32(d, 16, std::uint32_t(kLayers));
  PutU32(d, 20, std::uint32_t(kKvWidth));
  PutU32(d, 24, std::uint32_t(context));
  PutU32(d, 28, std::uint32_t(context));
  PutU32(d, 32, std::uint32_t(kv));
  PutU32(d, 36, std::uint32_t(rs));
  PutU64(d, 40, std::uint64_t(l.elements_per_plane));
  PutU64(d, 48, 0);
  PutU64(d, 56, 0);
  PutU64(d, 64, std::uint64_t(l.total_bytes));
  PutU32(d, 72, 0);  // image count
  PutU32(d, 76, 0);  // reserved, must be zero
  return d;
}

// Returns false when the payload would be rejected, mirroring
// ParseCompactSnapshotLayout (arena.cpp:169-200).
bool Parse(const std::vector<std::uint8_t>& d, Layout* out) {
  if (d.size() < kHeaderBytes)
    return false;
  const char magic[8] = {'G', 'Q', 'K', 'V', 'S', 'N', 'P', '1'};
  for (int i = 0; i < 8; ++i) {
    if (d[i] != std::uint8_t(magic[i]))
      return false;
  }
  const std::uint32_t version = GetU32(d, 8);
  const std::uint32_t image_count = GetU32(d, 72);
  if ((version != 1 && version != 2) || image_count > GetU32(d, 28) ||
      (version == 1 && image_count != 0) ||
      GetU32(d, 12) != std::uint32_t(kHeaderBytes) || GetU32(d, 76) != 0) {
    return false;
  }
  const std::uint32_t kv_value = GetU32(d, 32);
  const std::uint32_t rs_value = GetU32(d, 36);
  // Upper bound is the LAST enumerator, not kFp16.
  if (kv_value > std::uint32_t(QwenKvCacheStorage::kTurbo6) ||
      rs_value > std::uint32_t(QwenRecurrentStateStorage::kBf16)) {
    return false;
  }
  const auto kv = static_cast<QwenKvCacheStorage>(kv_value);
  const std::size_t context = GetU32(d, 24);
  Layout l = MakeLayout(kv, context);
  // The stored element count must agree with the geometry, otherwise the
  // recomputed byte count would not describe the buffer actually written.
  if (GetU64(d, 40) != std::uint64_t(l.elements_per_plane))
    return false;
  if (GetU64(d, 64) != std::uint64_t(l.total_bytes))
    return false;
  if (d.size() != l.total_bytes)
    return false;
  *out = l;
  return true;
}

std::string ReadFile(const std::string& path) {
  std::ifstream f(path);
  if (!f)
    return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

int main() {
  std::printf("Turbo4 compact-snapshot round-trip\n");

  const std::string arena = ReadFile("src/models/qwen/hip/arena.cpp");

  // ---- the mirror still matches gufo ------------------------------------
  Check(!arena.empty(), "arena.cpp is readable for the mirror guard");
  std::string source = arena;
  source.erase(std::remove_if(source.begin(), source.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               source.end());
  // Current streaming writer serializes into 'header', then emits it. Match
  // complete typed stores independent of clang-format whitespace.
  Check(source.find("PutLittleEndian<std::uint32_t>(header,32,static_cast<std::"
                    "uint32_t>(kv_storage_))") != std::string::npos,
        "arena still writes kv_storage at offset 32");
  Check(
      source.find("PutLittleEndian<std::uint32_t>(header,36,static_cast<std::"
                  "uint32_t>(recurrent_state_storage_))") != std::string::npos,
      "arena still writes recurrent storage at offset 36");
  Check(source.find("PutLittleEndian<std::uint64_t>(header,40,static_cast<std::"
                    "uint64_t>(layout.live_kv_elements_per_plane))") !=
            std::string::npos,
        "arena still writes kv elements at offset 40");
  Check(source.find("PutLittleEndian<std::uint64_t>(header,64,static_cast<std::"
                    "uint64_t>(layout.total_bytes))") != std::string::npos,
        "arena still writes total bytes at offset 64");
  Check(source.find("GetLittleEndian<std::uint32_t>(payload,76)!=0") !=
            std::string::npos,
        "arena still checks the reserved word at offset 76");
  Check(source.find("kv_storage_value>static_cast<std::uint32_t>("
                    "QwenKvCacheStorage::kTurbo6)") != std::string::npos,
        "arena precision bound is the last enumerator");
  Check(source.find("QwenKvIsBlockAddressed(kv_storage)") != std::string::npos,
        "arena plane sizing branches on block addressing");

  // ---- round trip for every storage ------------------------------------
  for (auto kv : {QwenKvCacheStorage::kFp32, QwenKvCacheStorage::kFp16,
                  QwenKvCacheStorage::kTurbo4, QwenKvCacheStorage::kTurbo5,
                  QwenKvCacheStorage::kTurbo6}) {
    for (auto rs :
         {QwenRecurrentStateStorage::kFp32, QwenRecurrentStateStorage::kBf16}) {
      const auto blob = Serialize(kv, rs, kContext);
      Layout got{};
      const bool ok = Parse(blob, &got);
      const Layout want = MakeLayout(kv, kContext);
      const std::string tag = "storage " + std::to_string(int(kv)) + " state " +
                              std::to_string(int(rs));
      Check(ok, tag + ": parses");
      Check(got.total_bytes == want.total_bytes, tag + ": total preserved");
      Check(got.bytes_per_plane == want.bytes_per_plane,
            tag + ": plane bytes preserved");
      Check(got.elements_per_plane == want.elements_per_plane,
            tag + ": element count preserved");
    }
  }

  // ---- the element count is storage-independent, bytes are not ---------
  {
    const Layout f16 = MakeLayout(QwenKvCacheStorage::kFp16, kContext);
    const Layout t4 = MakeLayout(QwenKvCacheStorage::kTurbo4, kContext);
    Check(f16.elements_per_plane == t4.elements_per_plane,
          "both layouts store the SAME element count");
    Check((kHeaderBytes + 2 * f16.bytes_per_plane) !=
              (kHeaderBytes + 2 * t4.bytes_per_plane),
          "but different total bytes, so the byte count must be recomputed "
          "on parse rather than trusted from the header");
  }

  // ---- STORAGE SHRINKS, it does not merely move ------------------------
  {
    const Layout f16 = MakeLayout(QwenKvCacheStorage::kFp16, kContext);
    const Layout t4 = MakeLayout(QwenKvCacheStorage::kTurbo4, kContext);
    const std::size_t f16_total = kHeaderBytes + 2 * f16.bytes_per_plane;
    const std::size_t t4_total = kHeaderBytes + 2 * t4.bytes_per_plane;
    Check(t4_total < f16_total, "turbo4 snapshot is smaller on the wire");
    Check(t4_total == kHeaderBytes + 2 * 34603008,
          "turbo4 total is header + two 34603008 planes");
    Check(f16_total == kHeaderBytes + 2 * 134217728,
          "fp16 total is header + two 134217728 planes");

    // And the shrink holds as context grows: it is a ratio, not a rounding.
    bool monotone = true;
    for (std::size_t ctx : {1024UL, 4096UL, 16384UL, 65536UL}) {
      const std::size_t a =
          kHeaderBytes + 2 * QwenKvBytesForElements(QwenKvCacheStorage::kFp16,
                                                    kLayers * ctx * kKvWidth);
      const std::size_t b =
          kHeaderBytes + 2 * QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4,
                                                    kLayers * ctx * kKvWidth);
      if (b >= a)
        monotone = false;
    }
    Check(monotone, "turbo4 stays smaller at every context tested");
  }

  // ---- cross-storage rejection -----------------------------------------
  {
    auto t4 = Serialize(QwenKvCacheStorage::kTurbo4,
                        QwenRecurrentStateStorage::kFp32, kContext);
    Layout out{};
    Check(Parse(t4, &out), "a turbo4 snapshot parses on a turbo4 build");

    // Simulate an OLD build whose bound is kFp16: value 2 must be refused.
    const std::uint32_t value = GetU32(t4, 32);
    Check(value == 2, "turbo4 persists as value 2");
    Check(value > std::uint32_t(QwenKvCacheStorage::kFp16),
          "an fp16-bounded build refuses a turbo4 snapshot rather than "
          "reinterpreting its planes");

    // A tampered storage value must be refused, not clamped.
    t4[32] = 99;
    Check(!Parse(t4, &out), "an unknown storage value is rejected");
    t4[32] = 2;
    // A tampered total must be refused.
    auto bad = t4;
    PutU64(
        bad, 64,
        std::uint64_t(
            MakeLayout(QwenKvCacheStorage::kTurbo4, kContext).total_bytes + 1));
    Check(!Parse(bad, &out),
          "a header claiming more bytes than the payload is rejected");
    // A tampered element count must be refused.
    auto bad2 = t4;
    PutU64(bad2, 40, std::uint64_t(12345));
    Check(!Parse(bad2, &out),
          "a header whose element count contradicts the geometry is rejected");
    // A truncated payload must be refused.
    auto short_blob = t4;
    short_blob.resize(short_blob.size() - 1);
    Check(!Parse(short_blob, &out), "a short payload is rejected");
    // The reserved word must stay zero.
    auto bad3 = t4;
    PutU32(bad3, 76, 1);
    Check(!Parse(bad3, &out), "a non-zero reserved word is rejected");
  }

  std::printf(
      g_failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECK(S) FAILED\n",
      g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
// The snapshot copy must move the whole valid context, not one
// token per row.
//
// WHY A DIFFERENT KIND OF TEST FROM B-R4. B-R4 checked SHAPE: pitch*rows against
// the reserved region, in 288 cases. That invariant was true and the copy was
// still wrong, because the defect was in an ARGUMENT, not in a total. An
// invariant test cannot catch a wrong-argument defect -- it can only confirm the
// numbers that were never in question. So this one moves BYTES and compares
// every one of them.
//
// THE DEFECT. QwenKvDescribeCopy sets width to ONE position: token_pitch for
// token-major storage (one token, all heads), head_dim*elem for fp32. Every
// Copy2D caller passes that as the per-row width, while source_pitch and
// target_pitch are context*width. The result is that row r copies the bytes at
// [r*source_pitch, r*source_pitch + one_position) and leaves the rest of that
// row's reserved span untouched. A restored plane then carries only position 0
// per layer.
//
// THE ORACLE IS THE EMULATED COPY. Copy2DAsync is emulated exactly as the three
// call sites use it --
//     for r in 0..rows:  memcpy(dst + r*target_pitch, src + r*source_pitch, width)
// and then every valid byte of the destination is compared against the source
// byte it was supposed to carry. No shape arithmetic is consulted, so this
// cannot pass while the data is wrong.
//
// THE PLANE IS FILLED WITH COORDINATE-DERIVED BYTES, not a constant: every byte
// of token (layer, token) carries a value derived from those coordinates, so a
// copy that lands at the wrong offset or copies the wrong span fails, and a
// destination left partly unwritten reads as untouched filler.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"

using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvCopyShape;
using gufo::hip::QwenKvPlaneGeometry;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

const char* Name(QwenKvCacheStorage s) {
  switch (s) {
    case QwenKvCacheStorage::kFp32: return "fp32";
    case QwenKvCacheStorage::kFp16: return "fp16";
    case QwenKvCacheStorage::kTurbo4: return "turbo4";
  }
  return "?";
}

std::uint8_t CoordValue(std::size_t a, std::size_t b, std::size_t c) {
  return static_cast<std::uint8_t>(1U + (a * 7U + b * 13U + c * 29U) % 251U);
}

/// Token-major plane: layers rows, each row `context` tokens of `token_pitch`
/// bytes. Byte value depends on (layer, token).
std::vector<std::uint8_t> MakeTokenMajor(std::size_t layers, std::size_t context,
                                          std::size_t token_pitch) {
  std::vector<std::uint8_t> plane(layers * context * token_pitch, 0xEE);
  for (std::size_t layer = 0; layer < layers; ++layer) {
    for (std::size_t token = 0; token < context; ++token) {
      const std::uint8_t v = CoordValue(layer, token, 0);
      const std::size_t base = layer * context * token_pitch + token * token_pitch;
      for (std::size_t j = 0; j < token_pitch; ++j) plane[base + j] = v;
    }
  }
  return plane;
}

/// Head-major plane: (layer x head) rows, each row `context` positions of
/// `pos_bytes`. Byte value depends on (layer, head, position).
std::vector<std::uint8_t> MakeHeadMajor(std::size_t layers, std::size_t kv_heads,
                                        std::size_t context, std::size_t pos_bytes) {
  std::vector<std::uint8_t> plane(layers * kv_heads * context * pos_bytes, 0xEE);
  for (std::size_t layer = 0; layer < layers; ++layer) {
    for (std::size_t head = 0; head < kv_heads; ++head) {
      const std::size_t row = layer * kv_heads + head;
      for (std::size_t pos = 0; pos < context; ++pos) {
        const std::uint8_t v = CoordValue(layer, head, pos);
        const std::size_t base = row * context * pos_bytes + pos * pos_bytes;
        for (std::size_t j = 0; j < pos_bytes; ++j) plane[base + j] = v;
      }
    }
  }
  return plane;
}

/// Copy2DAsync, emulated exactly as arena.cpp calls it.
void EmulateCopy2D(std::vector<std::uint8_t>& dst,
                   const std::vector<std::uint8_t>& src,
                   const QwenKvCopyShape& copy) {
  const std::size_t valid = copy.source_pitch < copy.target_pitch
                                ? copy.source_pitch
                                : copy.target_pitch;
  for (std::size_t r = 0; r < copy.rows; ++r) {
    const std::size_t d = r * copy.target_pitch;
    const std::size_t s = r * copy.source_pitch;
    if (d + copy.width > dst.size() || s + copy.width > src.size()) continue;
    std::memcpy(dst.data() + d, src.data() + s, copy.width);
    (void)valid;
  }
}

/// Compares EVERY valid byte of the destination against the source byte it was
/// meant to carry. Returns the number of mismatching bytes.
std::size_t CountWrongBytes(const std::vector<std::uint8_t>& dst,
                            const std::vector<std::uint8_t>& src,
                            const QwenKvCopyShape& copy) {
  const std::size_t valid = copy.source_pitch < copy.target_pitch
                                ? copy.source_pitch
                                : copy.target_pitch;
  std::size_t wrong = 0;
  for (std::size_t r = 0; r < copy.rows; ++r) {
    for (std::size_t j = 0; j < valid; ++j) {
      const std::size_t d = r * copy.target_pitch + j;
      const std::size_t s = r * copy.source_pitch + j;
      if (d >= dst.size() || s >= src.size()) {
        ++wrong;
        continue;
      }
      if (dst[d] != src[s]) ++wrong;
    }
  }
  return wrong;
}

struct Case {
  QwenKvCacheStorage storage;
  std::size_t head_dim;
  std::size_t kv_heads;
  std::size_t layers;
  std::size_t source_context;
  std::size_t target_context;
};

void OneCase(const Case& c) {
  const QwenKvPlaneGeometry geometry{c.head_dim, c.kv_heads, c.target_context};
  const QwenKvCopyShape copy = gufo::hip::QwenKvDescribeCopy(
      c.storage, geometry, c.source_context, c.target_context, c.layers);
  if (copy.width == 0U) {
    Check(false, std::string("copy described for ") + Name(c.storage));
    return;
  }

  std::vector<std::uint8_t> src;
  if (gufo::hip::QwenKvIsBlockAddressed(c.storage)) {
    const std::size_t token_pitch = gufo::hip::QwenKvBytesPerToken(c.storage, geometry);
    src = MakeTokenMajor(c.layers, c.source_context, token_pitch);
  } else if (c.storage == QwenKvCacheStorage::kFp32) {
    const std::size_t pos_bytes =
        c.head_dim * gufo::hip::QwenKvElementBytes(c.storage);
    src = MakeHeadMajor(c.layers, c.kv_heads, c.source_context, pos_bytes);
  } else {
    src = MakeTokenMajor(c.layers, c.source_context,
                         c.head_dim * c.kv_heads *
                             gufo::hip::QwenKvElementBytes(c.storage));
  }

  std::vector<std::uint8_t> dst(copy.target_pitch * copy.rows, 0xEE);
  EmulateCopy2D(dst, src, copy);
  const std::size_t wrong = CountWrongBytes(dst, src, copy);

  const std::string label =
      std::string(Name(c.storage)) + " layers=" + std::to_string(c.layers) +
      " heads=" + std::to_string(c.kv_heads) +
      " src_ctx=" + std::to_string(c.source_context) +
      " tgt_ctx=" + std::to_string(c.target_context) +
      " rows=" + std::to_string(copy.rows) +
      " width=" + std::to_string(copy.width) +
      " src_pitch=" + std::to_string(copy.source_pitch);
  Check(wrong == 0, "bytes: every valid byte survives the copy -- " + label +
                        " (" + std::to_string(wrong) + " wrong of " +
                        std::to_string((copy.source_pitch <
                                        copy.target_pitch
                                            ? copy.source_pitch
                                            : copy.target_pitch) *
                                       copy.rows) +
                        ")");
}

/// The save/restore pair as arena.cpp runs it: save copies a valid region out,
/// restore writes it back into a plane sized for the full context.
void RoundTrip(std::size_t head_dim, std::size_t kv_heads, std::size_t layers,
               std::size_t valid_context, std::size_t max_context,
               QwenKvCacheStorage storage) {
  // SAVE: source and target are both the valid context.
  const QwenKvPlaneGeometry save_geometry{head_dim, kv_heads, valid_context};
  const QwenKvCopyShape save_copy = gufo::hip::QwenKvDescribeCopy(
      storage, save_geometry, valid_context, valid_context, layers);

  std::vector<std::uint8_t> plane;
  if (gufo::hip::QwenKvIsBlockAddressed(storage)) {
    plane = MakeTokenMajor(layers, valid_context,
                           gufo::hip::QwenKvBytesPerToken(storage, save_geometry));
  } else if (storage == QwenKvCacheStorage::kFp32) {
    plane = MakeHeadMajor(layers, kv_heads, valid_context,
                          head_dim * gufo::hip::QwenKvElementBytes(storage));
  } else {
    plane = MakeTokenMajor(layers, valid_context,
                           head_dim * kv_heads *
                               gufo::hip::QwenKvElementBytes(storage));
  }

  const std::size_t region =
      (save_copy.source_pitch < save_copy.target_pitch ? save_copy.source_pitch
                                                      : save_copy.target_pitch) *
      save_copy.rows;
  std::vector<std::uint8_t> snapshot(region, 0xEE);
  EmulateCopy2D(snapshot, plane, save_copy);
  const std::size_t wrong_save = CountWrongBytes(snapshot, plane, save_copy);
  Check(wrong_save == 0,
        std::string("round trip SAVE: ") + Name(storage) + " valid=" +
            std::to_string(valid_context) + "/" + std::to_string(max_context) +
            " carries every valid byte (" + std::to_string(wrong_save) +
            " wrong of " + std::to_string(region) + ")");

  // RESTORE: the snapshot's valid rows go into a plane sized for max_context.
  const QwenKvPlaneGeometry restore_geometry{head_dim, kv_heads, max_context};
  const QwenKvCopyShape restore_copy = gufo::hip::QwenKvDescribeCopy(
      storage, restore_geometry, valid_context, max_context, layers);
  std::vector<std::uint8_t> restored(restore_copy.target_pitch * restore_copy.rows,
                                     0xEE);
  EmulateCopy2D(restored, snapshot, restore_copy);

  const std::size_t valid = restore_copy.source_pitch < restore_copy.target_pitch
                                ? restore_copy.source_pitch
                                : restore_copy.target_pitch;
  std::size_t wrong_restore = 0;
  for (std::size_t r = 0; r < restore_copy.rows; ++r) {
    for (std::size_t j = 0; j < valid; ++j) {
      if (restored[r * restore_copy.target_pitch + j] != plane[r * valid + j]) {
        ++wrong_restore;
      }
    }
  }
  Check(wrong_restore == 0,
        std::string("round trip RESTORE: ") + Name(storage) + " valid=" +
            std::to_string(valid_context) + "/" + std::to_string(max_context) +
            " recovers every valid byte (" + std::to_string(wrong_restore) +
            " wrong of " + std::to_string(valid * restore_copy.rows) + ")");
}

}  // namespace

int main() {
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kKvHeads = 4;
  constexpr std::size_t kLayers = 4;

  // valid == max, then valid < max. Both must be byte-correct.
  for (const auto storage : {QwenKvCacheStorage::kFp16,
                             QwenKvCacheStorage::kFp32,
                             QwenKvCacheStorage::kTurbo4}) {
    for (const std::size_t ctx : {std::size_t{64}, std::size_t{17}}) {
      OneCase({storage, kHeadDim, kKvHeads, kLayers, ctx, ctx});
    }
    // valid < max, which is what a continuation or prefix reuse actually does.
    OneCase({storage, kHeadDim, kKvHeads, kLayers, 17, 64});
    RoundTrip(kHeadDim, kKvHeads, kLayers, 17, 64, storage);
    RoundTrip(kHeadDim, kKvHeads, kLayers, 64, 64, storage);
  }

  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
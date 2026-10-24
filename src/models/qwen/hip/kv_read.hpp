#ifndef GUFO_MODELS_QWEN_HIP_KV_READ_HPP_  // plane access, both directions
#define GUFO_MODELS_QWEN_HIP_KV_READ_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "execution_policy.hpp"
#include "kv_allocation.hpp"
#include "turbo4_codec.hpp"

/// Plane access in BOTH directions, sharing one offset formula.
///
/// The read side came first; the writer joins it here rather than in its own
/// file, because the hazard is not a duplicated writer or a duplicated reader —
/// it is a duplicated OFFSET. Two offset formulas that disagree do not raise an
/// error; they return the wrong head's values, which decode just as plausibly.
/// `QwenKvHeadBlockOffset` is therefore called by both directions and is the
/// only place the layout arithmetic exists.
///
/// WHY THIS FILE EXISTS. Everything landed so far is the write side — allocate,
/// quantize, serialize, identify. Nothing could read a plane back. That gap is
/// exactly where a kernel author invents a second block-offset formula, and two
/// formulas disagree silently: one plane, two readers, no error. So the read
/// path is written once, on the CPU, before any kernel exists to copy it.
///
/// SCOPE, STATED PLAINLY. This is arithmetic over a byte buffer. It is NOT an
/// attention path, and it does not claim to be one: nothing here does a dot
/// product, and no claim is made about GPU behaviour, residency or parity.
/// What it does establish is that the offsets a kernel will need are derived
/// once, are bounds-checked, and are the same numbers the allocator reserved.
///
/// ADDRESSING. For block-addressed storage a plane is
///     kv_width = head_dim * kv_heads values per token,
/// laid out token-major, head-minor, each head occupying a whole number of
/// 128-value blocks. So:
//     blocks_per_head = head_dim / 128
///     blocks_per_token = blocks_per_head * kv_heads
///     block(t, h) = t * blocks_per_token + h * blocks_per_head
///
/// The head-major-within-token order is what the writer must also use, which is
/// why QwenKvWriteHeadOffset below and this file agree by construction.

namespace gufo::hip {

/// Static shape of one KV plane. Validated once by QwenKvPlaneIsReadable.
struct QwenKvPlaneGeometry {
  std::size_t head_dim;  ///< values per head (n_embd_head_k == _v for Qwen)
  std::size_t kv_heads;  ///< head_count_kv
  std::size_t context;   ///< positions in the plane
};

/// Blocks one head spans (width-independent: every width is 128 values per
/// block). Zero for element-addressable storage.
[[nodiscard]] constexpr std::size_t QwenKvBlocksPerHead(
    const QwenKvPlaneGeometry& g) noexcept {
  return QwenKvIsBlockAddressed(QwenKvCacheStorage::kTurbo4) &&
                 g.head_dim != 0U && g.head_dim % turbo4::kBlockValues == 0U
             ? g.head_dim / turbo4::kBlockValues
             : 0U;
}

/// Bytes one token of one plane occupies.
[[nodiscard]] constexpr std::size_t QwenKvBytesPerToken(
    QwenKvCacheStorage storage, const QwenKvPlaneGeometry& g) noexcept {
  return QwenKvBytesForElements(storage, g.head_dim * g.kv_heads);
}

/// Bytes the WHOLE plane occupies for this geometry, or 0 if the arithmetic
/// would overflow. The read and write paths both require the caller's buffer
/// to be at least this large, so a geometry that claims more than exists is
/// refused before anything is addressed rather than after.
[[nodiscard]] constexpr std::size_t QwenKvWholePlaneBytes(
    QwenKvCacheStorage storage, const QwenKvPlaneGeometry& g) noexcept {
  if (!QwenKvStorageIsKnown(storage))
    return 0U;
  if (g.head_dim == 0U || g.kv_heads == 0U || g.context == 0U)
    return 0U;
  const std::size_t maxv = static_cast<std::size_t>(-1);
  if (g.kv_heads > maxv / g.head_dim)
    return 0U;
  const std::size_t per_token =
      QwenKvBytesForElements(storage, g.head_dim * g.kv_heads);
  if (per_token == 0U)
    return 0U;
  if (g.context > maxv / per_token)
    return 0U;
  return per_token * g.context;
}

/// True when the geometry can be addressed at all. A zero width, a zero head
/// count, a zero context, or a head that is not a whole number of blocks all
/// REFUSE — they do not clamp, and they do not round up to a whole block.
/// Rounding up would silently allocate a head a block it does not own, and two
/// neighbouring heads would then share the same norm.
[[nodiscard]] constexpr bool QwenKvPlaneIsReadable(
    QwenKvCacheStorage storage, const QwenKvPlaneGeometry& g) noexcept {
  if (!QwenKvStorageIsKnown(storage))
    return false;
  if (g.head_dim == 0U || g.kv_heads == 0U || g.context == 0U)
    return false;
  if (QwenKvIsBlockAddressed(storage)) {
    if (!QwenKvHeadDimIsBlockAligned(g.head_dim))
      return false;
    // head_dim * kv_heads must not wrap.
    if (g.kv_heads > (static_cast<std::size_t>(-1) / g.head_dim))
      return false;
  }
  // The plane must have a real, non-zero size for this geometry.
  return QwenKvBytesPerToken(storage, g) != 0U;
}

/// Block index of head `head` at token `token`. Assumes the geometry passed
/// QwenKvPlaneIsReadable; callers use QwenKvReadHead, which checks.
/// CHECKED block offset. Returns false and leaves *out alone if the index
/// arithmetic would wrap.
///
/// The unchecked form `token * blocks_per_head * kv_heads + head *
/// blocks_per_head` overflows to a SMALL number on a hostile geometry. Given
/// {head_dim 128, kv_heads 2, context SIZE_MAX+2} and token SIZE_MAX+1, it
/// wrapped to block 0 and the read then happily answered from the start of
/// the plane — a refusal that returned success, for the worst possible index.
/// Both directions must use the checked form.
[[nodiscard]] constexpr bool QwenKvHeadBlockOffsetChecked(
    const QwenKvPlaneGeometry& g, std::size_t token, std::size_t head,
    std::size_t* out) noexcept {
  const std::size_t maxv = static_cast<std::size_t>(-1);
  const std::size_t bph = g.head_dim / turbo4::kBlockValues;
  if (bph == 0U)
    return false;
  if (head >= g.kv_heads)
    return false;
  if (token >= g.context)
    return false;
  if (bph != 0U && g.kv_heads > maxv / bph)
    return false;
  const std::size_t per_token = bph * g.kv_heads;
  if (token > maxv / per_token)
    return false;
  const std::size_t base = token * per_token;
  if (head > maxv / bph)
    return false;
  const std::size_t off = base + head * bph;
  if (off < base)
    return false;  // addition wrapped
  *out = off;
  return true;
}

[[nodiscard]] constexpr std::size_t QwenKvHeadBlockOffset(
    const QwenKvPlaneGeometry& g, std::size_t token,
    std::size_t head) noexcept {
  const std::size_t bph = g.head_dim / turbo4::kBlockValues;
  return token * bph * g.kv_heads + head * bph;
}

/// Read one head of one token back to floats. `plane_bytes` is the actual
/// length of the buffer being read, NOT the length the geometry implies.
///
/// Returns false and leaves `*out` untouched on any refusal: bad geometry,
/// token or head past the end, a null plane, or a plane too small to contain
/// the index requested. It is deliberately a bool, not a throw: the GPU caller
/// wants a branch, not an exception, and a bool keeps this usable from HIP.
bool QwenKvReadHead(QwenKvCacheStorage storage, const std::uint8_t* plane,
                    std::size_t plane_bytes, const QwenKvPlaneGeometry& g,
                    std::size_t token, std::size_t head, float* out,
                    turbo4::PlaneFormats formats, bool for_k) noexcept {
  if (QwenKvIsBlockAddressed(storage) &&
      (!formats.Valid() || formats.stride != QwenKvTurboBits(storage)))
    return false;
  if (!QwenKvPlaneIsReadable(storage, g))
    return false;
  if (plane == nullptr || out == nullptr)
    return false;
  if (token >= g.context || head >= g.kv_heads)
    return false;

  // `plane_bytes` is the REAL length of the buffer, not the length the geometry
  // implies. An earlier signature omitted it and therefore could not refuse a
  // short plane: it derived the size it checked against from the same geometry
  // it was asked to validate, so a truncated buffer passed every check and the
  // read ran off the end. The caller knows the buffer; this function cannot.
  const std::size_t per_token = QwenKvBytesPerToken(storage, g);
  if (per_token == 0U)
    return false;

  // The index arithmetic must be proven BEFORE it is used to address. The
  // whole-plane length is derived with the same checked discipline.
  const std::size_t whole_plane = QwenKvWholePlaneBytes(storage, g);
  if (whole_plane == 0U || plane_bytes < whole_plane)
    return false;

  if (QwenKvIsBlockAddressed(storage)) {
    // Block size and codec width follow the storage (66/82/98).
    const turbo4::Bits bits = for_k ? formats.k : formats.v;
    const std::size_t block_bytes = turbo4::BlockBytes(formats.stride);
    const std::size_t bph = g.head_dim / turbo4::kBlockValues;
    const std::size_t need = bph * block_bytes;
    std::size_t first_block = 0U;
    if (!QwenKvHeadBlockOffsetChecked(g, token, head, &first_block))
      return false;
    if (need > plane_bytes)
      return false;
    if (first_block > (plane_bytes - need) / block_bytes)
      return false;

    const std::uint8_t* base = plane + first_block * block_bytes;

    // PHASE 1: validate EVERY block before decoding ANY of them.
    //
    // A block whose fp16 norm overflowed to inf dequantizes to NaN in all 128
    // of its values, and a NaN propagates through a whole attention score
    // without tripping a bounds check. So the norm is inspected first.
    //
    // The two phases are separate because a head of 256 values is TWO blocks,
    // and the SECOND can be poisoned while the first is fine. Validating and
    // decoding in one loop wrote block 0 into `out` and then refused on block
    // 1 — a refusal that left 128 elements written and 128 untouched, in a
    // buffer documented to be untouched. That is the same half-written hazard
    // the writer refuses to produce, and the reader must not produce it either.
    for (std::size_t b = 0; b < bph; ++b) {
      std::uint16_t stored_half = 0;
      std::memcpy(&stored_half, base + b * block_bytes, sizeof(stored_half));
      if (!turbo4::StoredNormIsUsable(turbo4::HalfBitsToFloat(stored_half))) {
        return false;
      }
    }

    // PHASE 2: nothing can refuse now, so `out` is either wholly written or
    // wholly untouched.
    for (std::size_t b = 0; b < bph; ++b) {
      float scratch[turbo4::kBlockValues];
      turbo4::DequantizeBlock(base + b * block_bytes, scratch, bits);
      // DequantizeBlock lands in the ROTATED domain; the read path owes the
      // caller original-domain values, so the inverse rotation is applied here
      // and nowhere else.
      turbo4::RotateInverse(scratch);
      for (std::size_t j = 0; j < turbo4::kBlockValues; ++j) {
        out[b * turbo4::kBlockValues + j] = scratch[j];
      }
    }
    return true;
  }

  // Element-addressable: a flat copy, still bounds-checked against the plane.
  const std::size_t elem = QwenKvElementBytes(storage);
  if (elem == 0U)
    return false;  // the refusal, honoured
  const std::size_t maxv = static_cast<std::size_t>(-1);
  if (g.kv_heads > maxv / g.head_dim)
    return false;
  const std::size_t elems_per_token = g.head_dim * g.kv_heads;
  if (token > maxv / elems_per_token)
    return false;
  const std::size_t tok_base = token * elems_per_token;
  if (head > maxv / g.head_dim)
    return false;
  const std::size_t head_off = head * g.head_dim;
  if (head_off > maxv - tok_base)
    return false;
  const std::size_t base_index = tok_base + head_off;
  const std::size_t need_bytes = g.head_dim * elem;
  if (need_bytes > plane_bytes)
    return false;
  if (base_index > (plane_bytes - need_bytes) / elem)
    return false;
  for (std::size_t i = 0; i < g.head_dim; ++i) {
    const std::uint8_t* p = plane + (base_index + i) * elem;
    if (elem == sizeof(std::uint16_t)) {
      std::uint16_t half = 0;
      std::memcpy(&half, p, sizeof(half));
      out[i] = turbo4::HalfBitsToFloat(half);
    } else {
      std::memcpy(&out[i], p, sizeof(float));
    }
  }
  return true;
}

/// Write one head of one token into a plane.
///
/// Returns false and writes NOTHING on any refusal: bad geometry, token or head
/// past the end, a null pointer, a plane too small, or a block whose fp16 norm
/// will not fit the stored format. Same contract as QwenKvReadHead, and the
/// asymmetry matters — a partial write would leave a plane that reads back as
/// plausible garbage.
bool QwenKvWriteHead(QwenKvCacheStorage storage, std::uint8_t* plane,
                     std::size_t plane_bytes, const QwenKvPlaneGeometry& g,
                     std::size_t token, std::size_t head,
                     const float* src, turbo4::PlaneFormats formats, bool for_k) noexcept {
  if (QwenKvIsBlockAddressed(storage) &&
      (!formats.Valid() || formats.stride != QwenKvTurboBits(storage)))
    return false;
  if (!QwenKvPlaneIsReadable(storage, g))
    return false;
  if (plane == nullptr || src == nullptr)
    return false;
  if (token >= g.context || head >= g.kv_heads)
    return false;

  const std::size_t per_token = QwenKvBytesPerToken(storage, g);
  if (per_token == 0U)
    return false;
  const std::size_t plane_total =
      (g.context <= (static_cast<std::size_t>(-1) / per_token))
          ? per_token * g.context
          : 0U;
  if (plane_total == 0U || plane_total > plane_bytes)
    return false;

  if (QwenKvIsBlockAddressed(storage)) {
    const std::size_t bph = g.head_dim / turbo4::kBlockValues;
    std::size_t first_block = 0U;
    if (!QwenKvHeadBlockOffsetChecked(g, token, head, &first_block)) {
      return false;
    }
    const turbo4::Bits bits = for_k ? formats.k : formats.v;
    const std::size_t block_bytes = turbo4::BlockBytes(formats.stride);
    std::uint8_t* base = plane + first_block * block_bytes;

    // STAGE every block, then commit. Two reasons, both learned the hard way:
    //
    // 1. Atomicity. Encoding is the only step that can refuse at runtime.
    //    Encoding as we go writes block 0 and then refuses block 1, leaving
    //    half a head new and half stale.
    // 2. The refusal test belongs to the ENCODER, not here. A dry run over
    //    raw group norms checked the wrong quantity: QuantizeBlock stores
    //    group_norm / recon, which is LARGER, so a dry run passed on a block
    //    that then stored inf. Re-deriving the norm here would just repeat
    //    that mistake. QuantizeBlockChecked validates the value it actually
    //    writes, so a single authority decides.
    //
    // kHeadDim is unbounded here, so the staging buffer is sized per head by
    // the caller's geometry rather than assumed to be one block.
    constexpr std::size_t kMaxStageBlocks = 32;  // 32 blocks = 3136 B at Turbo6
    const std::size_t stage_blocks = bph;
    if (stage_blocks > kMaxStageBlocks)
      return false;
    std::uint8_t stage[kMaxStageBlocks * turbo4::kMaxBlockBytes]{};
    for (std::size_t b = 0; b < stage_blocks; ++b) {
      if (!turbo4::QuantizeBlockChecked(src + b * turbo4::kBlockValues,
                                        stage + b * block_bytes, bits)) {
        return false;  // nothing committed
      }
    }
    std::memcpy(base, stage, stage_blocks * block_bytes);
    return true;
  }

  const std::size_t elem = QwenKvElementBytes(storage);
  if (elem == 0U)
    return false;
  const std::size_t elems_per_token = g.head_dim * g.kv_heads;
  const std::size_t base_index = token * elems_per_token + head * g.head_dim;
  for (std::size_t i = 0; i < g.head_dim; ++i) {
    std::uint8_t* p = plane + (base_index + i) * elem;
    if (elem == sizeof(std::uint16_t)) {
      const std::uint16_t half = turbo4::FloatToHalfBits(src[i]);
      std::memcpy(p, &half, sizeof(half));
    } else {
      std::memcpy(p, &src[i], sizeof(float));
    }
  }
  return true;
}

/// Shape of a plane-to-plane copy: how many rows, how wide each row is in
/// BYTES, and the pitch on each side.
///
/// This exists because the arena had THREE copy sites (save, restore, compact
/// restore) and each one recomputed width and rows from the storage. Three
/// derivations of one layout is three chances to disagree, and two of them
/// agreeing is worse than none — the copy still succeeds and returns the wrong
/// head. One descriptor, three call sites.
///
/// THE LAYOUT CLAIM, STATED SO IT CAN BE FALSIFIED. A Turbo4 plane is
/// token-major with head-minor blocks, which is the SAME logical order as an
/// fp16 plane: within token t, head h's values come before head h+1's, and
/// head h's values are in index order. So Turbo4 copies as a token-major plane
/// exactly like fp16 and needs NO head reordering — the only difference is the
/// number of bytes one token occupies (528 for the UD geometry, against 2048
/// for fp16). rows and token_major are therefore IDENTICAL for the two, and
/// `QwenKvCopyShapeAgreesWithFp16` below is the assertion that says so.
struct QwenKvCopyShape {
  std::size_t width;  ///< bytes moved per row (B-R5: the whole valid row)
  std::size_t rows;   ///< rows in the copy
  std::size_t source_pitch;  ///< bytes between source rows
  std::size_t target_pitch;  ///< bytes between target rows
  bool token_major;          ///< rows are (layer) of a token-major plane
};

/// Describe the copy between two planes of `layers` attention layers holding
/// `source_context` and `target_context` positions.
///
/// Returns width == 0 as the REFUSAL for an unknown storage, an unaligned head
/// or any overflow — the same convention as QwenKvWholePlaneBytes.
[[nodiscard]] constexpr QwenKvCopyShape QwenKvDescribeCopy(
    QwenKvCacheStorage storage, const QwenKvPlaneGeometry& g,
    std::size_t source_context, std::size_t target_context,
    std::size_t layers) noexcept {
  QwenKvCopyShape shape{0U, 0U, 0U, 0U, false};
  if (!QwenKvStorageIsKnown(storage))
    return shape;
  if (layers == 0U)
    return shape;
  const std::size_t maxv = static_cast<std::size_t>(-1);

  if (storage == QwenKvCacheStorage::kFp32) {
    // FP32 is HEAD-MAJOR: one row is one (layer, head) pair's whole head.
    const std::size_t elem = QwenKvElementBytes(storage);
    if (elem == 0U || g.head_dim > maxv / elem)
      return shape;
    const std::size_t position_bytes = g.head_dim * elem;
    if (g.kv_heads > maxv / layers)
      return shape;
    const std::size_t rows = layers * g.kv_heads;
    if (source_context > maxv / position_bytes)
      return shape;
    if (target_context > maxv / position_bytes)
      return shape;
    shape.rows = rows;
    shape.source_pitch = source_context * position_bytes;
    shape.target_pitch = target_context * position_bytes;
    // B-R5. `width` is BYTES MOVED PER ROW, so it is the whole valid overlap of
    // the two spans, not a single position. It used to be one position while
    // the pitches were context*position, so every caller handed Copy2D a width
    // that moved position 0 of each row and left the rest of that row's
    // reserved span untouched -- a restored plane carrying only position 0 per
    // (layer, head). This is the source of that defect: all three callers read
    // this one field, so correcting it here corrects every call site at once.
    shape.width =
        (source_context < target_context ? source_context : target_context) *
        position_bytes;
    shape.token_major = false;
    return shape;
  }

  // FP16 and Turbo4 are both TOKEN-MAJOR: one row is one layer's token.
  //
  // The per-head alignment check is NOT optional here. QwenKvBytesPerToken
  // works from a FLAT element count, so a head_dim of 320 gives 1280 elements,
  // which is 10 whole blocks — a perfectly shaped answer for a head that
  // straddles a block boundary. Rounding up is how two heads end up sharing one
  // norm and both decoding wrongly, with every size check passing.
  if (QwenKvIsBlockAddressed(storage)) {
    if (g.kv_heads == 0U)
      return shape;
    if (!QwenKvHeadDimIsBlockAligned(g.head_dim))
      return shape;
  }
  const std::size_t token_pitch = QwenKvBytesPerToken(storage, g);
  if (token_pitch == 0U)
    return shape;
  if (source_context > maxv / token_pitch)
    return shape;
  if (target_context > maxv / token_pitch)
    return shape;
  shape.rows = layers;
  shape.source_pitch = source_context * token_pitch;
  shape.target_pitch = target_context * token_pitch;
  // B-R5, as above: the per-row width is the whole valid overlap, not one
  // token. min(source_context, target_context) * bytes-per-token, which is the
  // same as min(source_pitch, target_pitch). Both pitches are already
  // overflow-checked above, so this cannot exceed target_pitch.
  shape.width =
      (source_context < target_context ? source_context : target_context) *
      token_pitch;
  shape.token_major = true;
  return shape;
}

/// True when a Turbo4 copy has the same SHAPE as the fp16 copy — same rows,
/// same token-major ordering — differing only in bytes per token.
///
/// The predicate that would let a kernel author skip a head reorder. If this is
/// false for a real geometry, the "reuse the fp16 branch" shortcut is wrong and
/// the layout needs its own branch after all.
[[nodiscard]] constexpr bool QwenKvCopyShapeAgreesWithFp16(
    const QwenKvPlaneGeometry& g, std::size_t source_context,
    std::size_t target_context, std::size_t layers) noexcept {
  const QwenKvCopyShape t4 = QwenKvDescribeCopy(
      QwenKvCacheStorage::kTurbo4, g, source_context, target_context, layers);
  const QwenKvCopyShape f16 = QwenKvDescribeCopy(
      QwenKvCacheStorage::kFp16, g, source_context, target_context, layers);
  return t4.width != 0U && f16.width != 0U && t4.rows == f16.rows &&
         t4.token_major == f16.token_major && t4.rows == layers;
}

/// The selection receipt. Always printed at engine start.
///
/// The reason this exists: a "Turbo4 arm" once ran f16 end to end and its dump
/// came out byte-identical to the f16 arm, with nothing anywhere saying so. The
/// cure is not another test, it is a line at engine start that NAMES the
/// storage and states the bytes it allocated. Two runs that claim to differ and
/// print the same two numbers are visibly the same run.
///
/// Host-only, and it lives here rather than beside the resolver because it
/// needs the allocation arithmetic, which lives in kv_allocation.hpp and
/// includes execution_policy.hpp -- so putting it there would be a circular
/// include.
[[nodiscard]] inline std::string DescribeQwenKvSelection(
    const QwenExecutionPolicy& policy, std::size_t kv_elements) {
  const QwenKvCacheStorage storage = policy.kv_cache_storage;
  const std::size_t bytes = QwenKvStorageIsKnown(storage)
                                ? QwenKvBytesForElements(storage, kv_elements)
                                : 0U;
  std::string out = "gufo: qwen kv storage=";
  switch (storage) {
    case QwenKvCacheStorage::kFp32:
      out += "fp32";
      break;
    case QwenKvCacheStorage::kFp16:
      out += "fp16";
      break;
    case QwenKvCacheStorage::kTurbo4:
      out += "turbo4";
      break;
    case QwenKvCacheStorage::kTurbo5:
      out += "turbo5";
      break;
    case QwenKvCacheStorage::kTurbo6:
      out += "turbo6";
      break;
    default:
      out += "unknown";
      break;
  }
  out += " elements=" + std::to_string(kv_elements);
  out += " bytes=" + std::to_string(bytes);
  out += " bits_per_value=";
  out +=
      std::to_string(kv_elements == 0U ? 0.0
                                       : static_cast<double>(bytes) * 8.0 /
                                             static_cast<double>(kv_elements));
  if (policy.kv_layers.layer_count != 0 &&
      !policy.kv_layers.IsUniform(QwenKvLayerFormatForStorage(storage))) {
    out += " padded_layer_codecs=";
    for (std::uint32_t i = 0; i < policy.kv_layers.layer_count; ++i) {
      if (i != 0)
        out += ",";
      const auto f = QwenKvTurboLayerFormats(policy, i);
      out += std::to_string(static_cast<unsigned>(f.k)) + "/" +
             std::to_string(static_cast<unsigned>(f.v));
    }
  }
  return out;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KV_READ_HPP_
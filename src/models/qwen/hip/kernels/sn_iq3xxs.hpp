#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3XXS_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3XXS_HPP_

// Strix-native (SN) family: a Strix-native IQ3_XXS weight layout, its inverse, and
// the row/plane geometry the GPU kernel (sn_iq3xxs.hip) reads. Host-only, so
// the CPU repack test does not need a HIP compiler.
//
// ---- the ggml block, for reference -----------------------------------------
//
// block_iq3_xxs is 98 bytes (prior_art/ggml-common.h):
//   fp16 d     @ 0
//   uint8 qs[96] @ 2
// qs[0..63] is one 8-bit iq3xxs_grid index per 4 weights (16 per 32-weight
// group, 8 groups). qs[64..95] is one uint32 per 32-weight group: four 7-bit
// ksigns_iq2xs indices in bits 0..27 and the 4-bit scale `ls` in bits 28..31.
// For the 16-weight sub-block s (0..15), ib32 = s/2, p = s%2; the four grid
// index bytes are qs[ib32*8 + 4p + 0..3] and the aux word is qs[64 + 4*ib32].
//
// ---- math, identical to DecodeQuantSub16's kIQ3_XXS case --------------------
//
// For w = 0..3 (the w-th grid entry of the sub-block), grid entry g_w, and the
// sign index from aux bits 7*(2p + w/2) selecting nibble w%2:
//   sgn_w  = ksigns_iq2xs[sign_index] >> (4 * (w % 2))
//   q_w    = NegateSelectedBytes(iq3xxs_grid[g_w], sgn_w)   // 4 signed int8
//   scale  = d * (0.5 + ls) * 0.5
//   w[4w + k] = scale * int8(byte k of q_w)
//
// ---- SN layout -------------------------------------------------------------
//
// The tensor (rows x k, k a multiple of 256, nb = k/256 blocks per row) is
// split into three planes, each starting 16-byte aligned:
//   qs  plane: rows * nb * 64 bytes   (one 8-bit grid index per 4 weights)
//   aux plane: rows * nb * 32 bytes   (one uint32 per 32 weights)
//   d   plane: rows * nb * 2  bytes   (one fp16 per 256 weights)
//
// Rows are interleaved in groups of kSnIq3xxsRowInterleave = 4. Inside a group
// of g rows (g = 4, except the last group, which holds g = rows % 4) every
// plane is block-major: the group's g blocks are laid out contiguously, so a
// wave that owns up to four adjacent rows reads one contiguous 64*g-byte qs
// chunk (and a 32*g-byte aux chunk) per super-block. The last group is stored
// with its real height, so the packed tensor is exactly rows * nb * 98 bytes:
// the same size as the ggml tensor, no padding.
//
// Every qs/aux access the kernel makes is a 16-byte (or wider) aligned load:
// within a group the (block, row) pairs are (b*g + lr), so a row's chunk sits
// at `base + (b*g + lr) * 64` (qs) or `... * 32` (aux), and g*64 / g*32 are
// multiples of 16. The d plane is only ever read as single fp16s.
//
// Deterministic deinterleave: the repack is a pure byte gather, so
// SnIq3xxsUnpack(SnIq3xxsRepack(x)) == x byte for byte.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace gufo::hip::sn {

inline constexpr std::size_t kSnIq3xxsBlockK = 256;
inline constexpr std::size_t kSnIq3xxsBlockBytes = 98;
inline constexpr std::size_t kSnIq3xxsQsBytes = 64;
inline constexpr std::size_t kSnIq3xxsAuxBytes = 32;
inline constexpr std::size_t kSnIq3xxsHalfQsBytes = 32;
inline constexpr std::size_t kSnIq3xxsHalfAuxBytes = 16;
inline constexpr std::size_t kSnIq3xxsRowInterleave = 4;

inline constexpr std::size_t SnIq3xxsBlocks(std::size_t k) noexcept {
  return k / kSnIq3xxsBlockK;
}

/// Exact packed size: identical to the ggml tensor.
inline constexpr std::size_t SnIq3xxsPackedBytes(std::size_t rows,
                                                 std::size_t k) noexcept {
  return rows * SnIq3xxsBlocks(k) * kSnIq3xxsBlockBytes;
}

inline constexpr std::size_t SnIq3xxsQsPlaneBytes(std::size_t rows,
                                                  std::size_t k) noexcept {
  return rows * SnIq3xxsBlocks(k) * kSnIq3xxsQsBytes;
}

inline constexpr std::size_t SnIq3xxsAuxPlaneBytes(std::size_t rows,
                                                   std::size_t k) noexcept {
  return rows * SnIq3xxsBlocks(k) * kSnIq3xxsAuxBytes;
}

/// Where one row's three planes begin, and the in-plane stride from one
/// super-block to the next. `qs`/`aux` point at the row's byte 0 of block 0
/// (the caller adds block * stride and the half offset).
struct SnIq3xxsRowRef {
  const std::uint8_t* qs;
  const std::uint8_t* aux;
  const std::uint8_t* d;
  std::size_t qs_block_stride;
  std::size_t aux_block_stride;
  std::size_t d_block_stride;
};

/// Host mirror of the addressing the kernel computes. Kept as the written
/// spec the CPU test compares the repack against.
inline SnIq3xxsRowRef SnIq3xxsRowRefAt(const void* packed, std::size_t rows,
                                       std::size_t k,
                                       std::size_t row) noexcept {
  const auto* const base = static_cast<const std::uint8_t*>(packed);
  const std::size_t nb = SnIq3xxsBlocks(k);
  const std::size_t num_full = rows / kSnIq3xxsRowInterleave;
  const std::size_t rem = rows % kSnIq3xxsRowInterleave;
  const std::uint8_t* const qs_plane = base;
  const std::uint8_t* const aux_plane = base + SnIq3xxsQsPlaneBytes(rows, k);
  const std::uint8_t* const d_plane =
      aux_plane + SnIq3xxsAuxPlaneBytes(rows, k);

  const bool full = row < num_full * kSnIq3xxsRowInterleave;
  const std::size_t group = row / kSnIq3xxsRowInterleave;
  const std::size_t lr = row % kSnIq3xxsRowInterleave;
  const std::size_t g = full ? kSnIq3xxsRowInterleave : rem;
  const std::size_t base_rows =
      full ? group * kSnIq3xxsRowInterleave : num_full * kSnIq3xxsRowInterleave;
  return SnIq3xxsRowRef{
      qs_plane + base_rows * nb * kSnIq3xxsQsBytes + lr * kSnIq3xxsQsBytes,
      aux_plane + base_rows * nb * kSnIq3xxsAuxBytes + lr * kSnIq3xxsAuxBytes,
      d_plane + base_rows * nb * 2 + lr * 2,
      g * kSnIq3xxsQsBytes,
      g * kSnIq3xxsAuxBytes,
      g * 2};
}

/// Gather every row of a ggml IQ3_XXS tensor into the SN layout. `src` and
/// `dst` must not overlap; `dst` must have room for SnIq3xxsPackedBytes.
inline void SnIq3xxsRepack(const void* src, std::size_t rows, std::size_t k,
                           void* dst) noexcept {
  assert(k % kSnIq3xxsBlockK == 0U);
  const auto* const s = static_cast<const std::uint8_t*>(src);
  auto* const base = static_cast<std::uint8_t*>(dst);
  const std::size_t nb = SnIq3xxsBlocks(k);
  std::uint8_t* const qs_plane = base;
  std::uint8_t* const aux_plane = base + SnIq3xxsQsPlaneBytes(rows, k);
  std::uint8_t* const d_plane = aux_plane + SnIq3xxsAuxPlaneBytes(rows, k);
  std::size_t row = 0;
  std::size_t qs_g = 0;
  std::size_t aux_g = 0;
  std::size_t d_g = 0;
  while (row < rows) {
    const std::size_t g = std::min(kSnIq3xxsRowInterleave, rows - row);
    for (std::size_t b = 0; b < nb; ++b) {
      for (std::size_t lr = 0; lr < g; ++lr) {
        const std::uint8_t* const blk =
            s + (((row + lr) * nb) + b) * kSnIq3xxsBlockBytes;
        std::memcpy(qs_plane + qs_g + ((b * g) + lr) * kSnIq3xxsQsBytes,
                    blk + 2, kSnIq3xxsQsBytes);
        std::memcpy(aux_plane + aux_g + ((b * g) + lr) * kSnIq3xxsAuxBytes,
                    blk + 2 + kSnIq3xxsQsBytes, kSnIq3xxsAuxBytes);
        std::memcpy(d_plane + d_g + ((b * g) + lr) * 2, blk, 2);
      }
    }
    qs_g += g * nb * kSnIq3xxsQsBytes;
    aux_g += g * nb * kSnIq3xxsAuxBytes;
    d_g += g * nb * 2;
    row += g;
  }
}

/// Inverse of SnIq3xxsRepack: exact byte-for-byte reconstruction.
inline void SnIq3xxsUnpack(const void* src, std::size_t rows, std::size_t k,
                           void* dst) noexcept {
  assert(k % kSnIq3xxsBlockK == 0U);
  const auto* const base = static_cast<const std::uint8_t*>(src);
  auto* const d = static_cast<std::uint8_t*>(dst);
  const std::size_t nb = SnIq3xxsBlocks(k);
  const std::uint8_t* const qs_plane = base;
  const std::uint8_t* const aux_plane = base + SnIq3xxsQsPlaneBytes(rows, k);
  const std::uint8_t* const d_plane =
      aux_plane + SnIq3xxsAuxPlaneBytes(rows, k);
  std::size_t row = 0;
  std::size_t qs_g = 0;
  std::size_t aux_g = 0;
  std::size_t d_g = 0;
  while (row < rows) {
    const std::size_t g = std::min(kSnIq3xxsRowInterleave, rows - row);
    for (std::size_t b = 0; b < nb; ++b) {
      for (std::size_t lr = 0; lr < g; ++lr) {
        std::uint8_t* const blk =
            d + (((row + lr) * nb) + b) * kSnIq3xxsBlockBytes;
        std::memcpy(blk, d_plane + d_g + ((b * g) + lr) * 2, 2);
        std::memcpy(blk + 2,
                    qs_plane + qs_g + ((b * g) + lr) * kSnIq3xxsQsBytes,
                    kSnIq3xxsQsBytes);
        std::memcpy(blk + 2 + kSnIq3xxsQsBytes,
                    aux_plane + aux_g + ((b * g) + lr) * kSnIq3xxsAuxBytes,
                    kSnIq3xxsAuxBytes);
      }
    }
    qs_g += g * nb * kSnIq3xxsQsBytes;
    aux_g += g * nb * kSnIq3xxsAuxBytes;
    d_g += g * nb * 2;
    row += g;
  }
}

}  // namespace gufo::hip::sn

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SN_IQ3XXS_HPP_

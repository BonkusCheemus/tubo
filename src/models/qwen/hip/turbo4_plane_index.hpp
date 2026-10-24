#ifndef GUFO_MODELS_QWEN_HIP_TURBO4_PLANE_INDEX_HPP_
#define GUFO_MODELS_QWEN_HIP_TURBO4_PLANE_INDEX_HPP_

/// Thread-indexing authority for the Turbo4 KV kernels.
///
/// WHY THIS IS A HEADER AND NOT KERNEL CODE.
///
/// The kernels below are real HIP translation units: they cannot be executed
/// on this machine, only compiled. That is a weak oracle for anything involving
/// arithmetic. The one part of a kernel that is pure index math — "which byte
/// does (token, head, block) address" — is therefore lifted out here as
/// `constexpr`, callable identically on host and device, so a CPU test checks
/// the SAME function the kernel calls rather than a re-implementation of it
/// that could agree with every bug in the original.
///
/// THE REMAINING UNTESTED-IN-SILICON PART, STATED PLAINLY: block/thread
/// dimension assignment and the quantize/dequantize calls themselves are only
/// COMPILED here, not executed. Compilation catches a type error; it does not
/// catch a wrong grid dimension. Final real-plane qualification stays held.
///
/// LAYOUT, one layer's plane:
///     block(t, h)  = t * blocks_per_token + h * blocks_per_head
///     blocks_per_head = head_dim / 128
///     blocks_per_token = blocks_per_head * kv_heads
///     byte offset  = block * turbo4::BlockBytes(bits)   (66 / 82 / 98)
/// At the UD-Q4_K_XL geometry: head_dim 256, kv_heads 4, so 2 blocks per head,
/// 8 blocks and 528 bytes per token. 16 full-attention layers — block_count 65
/// includes one MTP layer, so the count is (65 - 1) / interval 4.

#include <cstddef>
#include <cstdint>

#include "turbo4_codec.hpp"

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define GUFO_TURBO4_HD __host__ __device__
#else
#define GUFO_TURBO4_HD
#endif

namespace gufo::hip {

/// Explicit device-side outcome of a Turbo4 kernel.
///
/// WHY THIS EXISTS. Review finding: a kernel that returns silently after a
/// refusal is a "silent stale success" — the caller sees a plane that still
/// holds the PREVIOUS token's blocks and no indication that this token never
/// landed. That is worse than an error, because it is invisible. Every refusal
/// path therefore writes a status, and `kOk` is the only value that means the
/// plane was written.
///
/// It is a plain integer enum with stable values so a caller can read it back
/// without a shared header dependency, and it lives here — the host/device
/// authority — so the CPU oracle can assert the same decisions the device
/// makes.
enum class Turbo4KernelStatus : std::uint32_t {
  kOk = 0,
  /// A shape that cannot be addressed at all: an unaligned head, no kv heads,
  /// no query heads, or a head wider than the staging bound.
  kGeometryRefused = 1,
  /// The token position is at or past max_context.
  kPositionOutOfRange = 2,
  /// A block's reconstructed norm will not fit the stored fp16 format, so the
  /// WHOLE head is left unchanged.
  kQuantizeRefused = 3,
  /// A stored norm in the plane is unusable (inf/NaN), so nothing was decoded.
  kPoisonedNorm = 4,
  /// Every score underflowed, so the softmax has no defined denominator.
  kDegenerateSoftmax = 5,
  /// The writer for this context did not succeed, so there is nothing coherent
  /// to attend over. APPENDED, never renumbered: the values above are part of
  /// the device/host contract and have shipped in tests.
  ///
  /// A NOTE ON THE LATCH, because this value is the largest one. The failure
  /// latch is a MAXIMUM, so kWriterFailed (6) now WINS over kDegenerateSoftmax
  /// (5) and every other code when both occur. That is the correct reading —
  /// a failed writer makes the other refusals downstream symptoms — but it does
  /// mean a mixed-head case reports this rather than the specific defect, so it
  /// is recorded here rather than discovered in a diff.
  kWriterFailed = 6,
};

/// A head must fit the per-thread staging buffer. At the UD geometry head_dim
/// 256 is 2 blocks per plane, so a head stages 4 x 66 = 264 B — K's two blocks
/// and V's two, committed together or not at all.
///
/// The bound exists so the staging array is a compile-time size. Exceeding it
/// is a REFUSAL, never a truncation: a partial stage that then commits is
/// exactly the half-written head this design exists to prevent.
inline constexpr std::size_t kTurbo4MaxBlocksPerHead = 8;
inline constexpr std::size_t kTurbo4StagedBlocksPerHead =
    2 * kTurbo4MaxBlocksPerHead;  // K and V
inline constexpr std::size_t kTurbo4StagedBytesPerHead =
    kTurbo4StagedBlocksPerHead * turbo4::kBlockBytes;
/// The staging array is sized for the WIDEST block (Turbo6, 98 B) so
/// one array serves every width; the Turbo4 figure above is kept for existing
/// users.
inline constexpr std::size_t kTurboMaxStagedBytesPerHead =
    kTurbo4StagedBlocksPerHead * turbo4::kMaxBlockBytes;

/// THE MANIFEST CONTEXT — one number, one home.
///
/// The scalar reader stages one score per valid token in SHARED memory, so the
/// context it can serve is bounded by the LDS the device will actually hand the
/// CTA. At the manifest context this is 4096 * sizeof(float) = 16384 B = 16 KiB
/// per CTA, which is a request a gfx1151 workgroup can satisfy — but that is a
/// DEVICE property, still compile-time-believed and not measured here.
///
/// WHY IT LIVES HERE RATHER THAN IN THE .hip. It was an anonymous-namespace
/// literal inside `turbo4_attention.hip`, which made it invisible to the CPU
/// oracle and gave it no name beyond "the number in that file". The accounting
/// that sizes this work carries its own `4096` (`opt66_lifetime_trace.py`
/// `manifest`, `opt66_boundary.py` `CONTEXT`), so the SAME context existed as
/// separate literals in two trees. One of them moving is a silent disagreement
/// between the kernel and the budget it was sized from.
///
/// A cross-tree test now holds these equal; see the KV accounting suite.
inline constexpr std::uint32_t kTurbo4ManifestContext = 4096;

/// Shared bytes the scalar reader requests per CTA at the manifest context.
/// Stated, not left implicit: 16 KiB is the figure a reviewer should be able to
/// check without reconstructing the multiplication.
inline constexpr std::size_t kTurbo4ScoresSharedBytes =
    static_cast<std::size_t>(kTurbo4ManifestContext) * sizeof(float);

/// Blocks one head spans, or 0 when the head is not a whole number of blocks.
///
/// Zero is a REFUSAL, not a rounding. Rounding up lets two neighbouring heads
/// share one block's norm, and every value then decodes as a legal number with
/// a plausible magnitude — the worst possible failure.
[[nodiscard]] GUFO_TURBO4_HD constexpr std::size_t Turbo4BlocksPerHead(
    std::size_t head_dim) noexcept {
  return (head_dim != 0U && head_dim % turbo4::kBlockValues == 0U)
             ? head_dim / turbo4::kBlockValues
             : 0U;
}

/// Blocks one token spans, across every kv head.
[[nodiscard]] GUFO_TURBO4_HD constexpr std::size_t Turbo4BlocksPerToken(
    std::size_t head_dim, std::size_t kv_heads) noexcept {
  const std::size_t bph = Turbo4BlocksPerHead(head_dim);
  if (bph == 0U || kv_heads == 0U)
    return 0U;
  const std::size_t maxv = static_cast<std::size_t>(-1);
  if (kv_heads > maxv / bph)
    return 0U;  // refuse rather than wrap
  return bph * kv_heads;
}

/// Block index of (token, head) within one layer's plane.
///
/// Returns false and leaves `*out` untouched on any refusal: a head that is not
/// block-aligned, a zero head count, a token or head past the end, or any
/// overflow. The overflow form matters: `token * blocks_per_token` wraps to a
/// SMALL number near SIZE_MAX, so the wrapped address lands in the middle of a
/// live plane and no bounds check notices.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4BlockIndex(
    std::size_t token, std::size_t head, std::size_t head_dim,
    std::size_t kv_heads, std::size_t context, std::size_t* out) noexcept {
  const std::size_t bph = Turbo4BlocksPerHead(head_dim);
  if (bph == 0U || kv_heads == 0U || context == 0U)
    return false;
  if (head >= kv_heads || token >= context)
    return false;
  const std::size_t maxv = static_cast<std::size_t>(-1);
  const std::size_t bpt = Turbo4BlocksPerToken(head_dim, kv_heads);
  if (bpt == 0U)
    return false;
  if (token > maxv / bpt)
    return false;
  const std::size_t base = token * bpt;
  if (head > maxv / bph)
    return false;
  const std::size_t off = base + head * bph;
  if (off < base)
    return false;  // addition wrapped
  *out = off;
  return true;
}

/// Byte offset of one block within one layer's plane.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4BlockByteOffset(
    std::size_t token, std::size_t head, std::size_t head_dim,
    std::size_t kv_heads, std::size_t context, std::size_t* out,
    turbo4::Bits bits = turbo4::Bits::k4) noexcept {
  std::size_t block = 0;
  if (!Turbo4BlockIndex(token, head, head_dim, kv_heads, context, &block))
    return false;
  const std::size_t maxv = static_cast<std::size_t>(-1);
  const std::size_t block_bytes = turbo4::BlockBytes(bits);
  if (block > maxv / block_bytes)
    return false;
  *out = block * block_bytes;
  return true;
}

/// Value offset inside a block's 128 slots. Refuses past 127 for the same
/// reason it refuses a bad block: a slot past the end is a write into the next
/// block's norm.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4ValueSlotIsValid(
    std::size_t value_in_block) noexcept {
  return value_in_block < turbo4::kBlockValues;
}

/// Whole-plane byte size for one layer, or 0 on any refusal.
[[nodiscard]] GUFO_TURBO4_HD constexpr std::size_t Turbo4LayerPlaneBytes(
    std::size_t head_dim, std::size_t kv_heads, std::size_t context,
    turbo4::Bits bits = turbo4::Bits::k4) noexcept {
  const std::size_t bpt = Turbo4BlocksPerToken(head_dim, kv_heads);
  if (bpt == 0U || context == 0U)
    return 0U;
  const std::size_t maxv = static_cast<std::size_t>(-1);
  const std::size_t block_bytes = turbo4::BlockBytes(bits);
  if (bpt > maxv / block_bytes)
    return 0U;
  // THE MISSING GUARD. `bpt * context` was computed with no check that the
  // product fits, so it WRAPPED: a review probe measured
  // Turbo4LayerPlaneBytes(128, 2, SIZE_MAX/2 + 2) returning **132** instead of
  // 0. bpt=2, context=2^63+1, so 2*context = 2^64+2 -> 2, and 2*66 = 132. A
  // wrapped size is worse than a refused one: the caller believes it has a
  // plane, and every offset it derives from it is a small legal-looking number
  // pointing into the middle of a live allocation. The previous line checks
  // `bpt` against the BLOCK multiplier, which says nothing about `context`.
  if (context > maxv / bpt)
    return 0U;
  const std::size_t blocks = bpt * context;
  if (blocks > maxv / block_bytes)
    return 0U;
  return blocks * block_bytes;
}

/// True when one head fits the per-thread staging buffer.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4HeadFitsStaging(
    std::size_t head_dim) noexcept {
  const std::size_t bph = Turbo4BlocksPerHead(head_dim);
  return bph != 0U && bph <= kTurbo4MaxBlocksPerHead;
}

/// Query-head -> kv-head mapping for grouped-query attention.
///
/// UD is 24 query heads over 4 kv heads, so SIX query heads share one kv head.
/// `num_heads % num_kv_heads != 0` REFUSES: a truncated map would leave the
/// last kv head reading nobody's K, which is a silent wrong answer rather than
/// a wrong count of answers.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4KvHeadForQueryHead(
    std::uint32_t query_head, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t* out) noexcept {
  if (num_heads == 0U || num_kv_heads == 0U)
    return false;
  if (query_head >= num_heads)
    return false;
  if (num_heads % num_kv_heads != 0U)
    return false;
  const std::uint32_t groups = num_heads / num_kv_heads;
  if (groups == 0U)
    return false;
  *out = query_head / groups;
  return true;
}

/// True when `need` bytes starting at `offset` fit inside a plane of
/// `plane_bytes`.
///
/// THE BOUNDS QUESTION NOBODY ASKED THE CALLER. `Turbo4BlockByteOffset`
/// validates an index against the CONTEXT it is handed. A caller that claims a
/// larger context than it allocated therefore gets a perfectly valid offset and
/// an out-of-bounds access — the same defect class `QwenKvReadHead` already
/// fixed by taking `plane_bytes`, and the same one that was NOT carried into
/// the kernel bodies. The check is subtraction-based so it cannot itself
/// overflow.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4BytesFit(
    std::size_t offset, std::size_t need, std::size_t plane_bytes) noexcept {
  if (need > plane_bytes)
    return false;
  return offset <= plane_bytes - need;
}

/// Declared buffer capacities for the attention body, validated ONCE before
/// anything is addressed.
///
/// WHY A STRUCT AND NOT THREE MORE size_t PARAMETERS. The plane-length fix in
/// §49 closed ONE member of a family: `q`, `out` and `scores` are checked for
/// null and trusted for length, so a caller passing a short buffer gets an
/// out-of-bounds access with every individual pointer check satisfied. Three
/// separate `if (x_bytes < ...)` calls in a body is three places to forget
/// one; one struct with one validator is one place.
///
/// THE PRINCIPLE: a pointer being non-null says nothing about how long the
/// allocation behind it is. Only the caller knows that, so the caller must
/// state it, and the callee must check the statement.
struct Turbo4BufferSizes {
  std::size_t plane_bytes;   ///< one packed Turbo4 plane
  std::size_t q_bytes;       ///< num_heads * head_dim * sizeof(float)
  std::size_t out_bytes;     ///< num_heads * head_dim * sizeof(float)
  std::size_t scores_bytes;  ///< valid_context * sizeof(float), PRIVATE
  std::size_t
      gate_bytes;  ///< num_heads * head_dim * sizeof(float); 0 == NO GATE
};

/// THE GATE ABSENCE RULE — answered with a rule rather
/// than a separate argument.
///
/// `gate_bytes == 0` means there is no gate, and that is the ONLY way to say
/// it. A separate `bool has_gate` would let a caller state a size and pass a
/// null pointer (or the reverse) and leave the callee to guess which was meant
/// — the same class of impossible state §50 spent a unit removing. One number
/// carries the whole declaration: zero is absent, non-zero is a gate of that
/// many bytes. A gate over zero heads is genuinely absent, so nothing real is
/// lost.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4GateIsAbsent(
    const Turbo4BufferSizes& sizes) noexcept {
  return sizes.gate_bytes == 0U;
}

[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4AttentionBuffersAreSufficient(
    const Turbo4BufferSizes& sizes, std::uint32_t num_heads,
    std::uint32_t head_dim, std::uint32_t valid_context) noexcept {
  if (num_heads == 0U || head_dim == 0U)
    return false;
  const std::size_t maxv = static_cast<std::size_t>(-1);
  // `head_dim * sizeof(float)` cannot overflow on a 64-bit size_t (the largest
  // uint32 times 4 is ~17 GB), so there is deliberately NO guard on it. An
  // earlier draft had `head_dim > maxv / sizeof(float)` here and hipcc was
  // right to call it a tautology: a uint32 can never exceed 4.6e18. A check
  // that cannot fire while reading as an overflow guard is worse than none —
  // it invites the reader to believe a hazard is handled here.
  const std::size_t row = static_cast<std::size_t>(head_dim) * sizeof(float);
  // THIS one is live: a large head_dim makes `row` large enough that
  // maxv / row falls below the largest num_heads.
  if (static_cast<std::size_t>(num_heads) > maxv / row)
    return false;
  const std::size_t all_rows = num_heads * row;
  if (sizes.q_bytes < all_rows)
    return false;
  if (sizes.out_bytes < all_rows)
    return false;
  // The gate covers the SAME [num_heads, head_dim] shape as q and out, so it is
  // checked against the same figure — and only when it is present at all.
  if (!Turbo4GateIsAbsent(sizes) && sizes.gate_bytes < all_rows)
    return false;
  if (valid_context != 0U) {
    // Same reasoning: `valid_context * sizeof(float)` cannot overflow a 64-bit
    // size_t, so there is no guard here to be tautological.
    if (sizes.scores_bytes <
        static_cast<std::size_t>(valid_context) * sizeof(float))
      return false;
  }
  return true;
}

/// The writer's equivalent: two [num_kv_heads, head_dim] float sources and two
/// packed planes.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4WriteBuffersAreSufficient(
    std::size_t plane_bytes, std::size_t source_bytes,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) noexcept {
  if (num_kv_heads == 0U || head_dim == 0U)
    return false;
  const std::size_t maxv = static_cast<std::size_t>(-1);
  // No `head_dim > maxv / sizeof(float)` guard, for the same reason as the
  // attention validator: it is a tautology against a uint32 on a 64-bit
  // size_t, and a guard that cannot fire is not a guard.
  const std::size_t row = static_cast<std::size_t>(head_dim) * sizeof(float);
  if (static_cast<std::size_t>(num_kv_heads) > maxv / row)
    return false;
  if (source_bytes < num_kv_heads * row)
    return false;
  // Each head's span must fit the plane at its tightest position; the caller
  // supplies the real length and the body re-checks at the exact offset.
  if (plane_bytes == 0U)
    return false;
  return true;
}

/// Where ONE attention layer's K plane and V plane live inside one arena.
///
/// Both regions are LAYER-MAJOR and the two regions are K-then-V, which is the
/// shape the live arena and the compact snapshot already use:
///
///   live    K region at 0,                V region at region_bytes
///   compact K region at 80,               V region at 80 + region_bytes
///           (kCompactSnapshotHeaderBytes, arena.cpp:27)
///
/// The only difference is the 80-byte base the compact layout puts in front of
/// it. That is why `base_bytes` is an explicit argument here rather than a
/// constant: one constant would have to be wrong for one of the two layouts,
/// and a slice that silently addresses 80 bytes early reads the header as a
/// norm.
struct Turbo4LayerSlices {
  std::size_t k_offset{0};      ///< byte offset of this layer's K plane
  std::size_t v_offset{0};      ///< byte offset of this layer's V plane
  std::size_t plane_bytes{0};   ///< bytes of ONE layer's plane
  std::size_t region_bytes{0};  ///< bytes of ONE region = plane_bytes * layers
};

/// Resolve one layer's K and V plane offsets inside `arena_bytes`.
///
/// `context` is the context the REGION was sized at — max_context for the live
/// arena, valid_context for a compact snapshot. It is the caller's number, not
/// this function's: the live arena allocates for max_context and a snapshot for
/// the context it captured, and passing the wrong one silently yields offsets
/// that are individually valid and wrong.
///
/// REFUSES, never guesses: no output pointer, a zero or past-the-end layer, a
/// head/context that yields no plane, or any slice that does not fit the arena.
/// Every product is overflow-checked, because a wrapped offset is a small legal
/// number pointing into the middle of a live allocation — the failure mode this
/// header exists to prevent.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4LayerPlaneSlices(
    std::size_t arena_bytes, std::size_t base_bytes, std::size_t head_dim,
    std::size_t kv_heads, std::size_t context, std::size_t n_attn_layers,
    std::size_t attn_layer, Turbo4LayerSlices* out,
    turbo4::Bits bits = turbo4::Bits::k4) noexcept {
  if (out == nullptr)
    return false;
  if (n_attn_layers == 0U || attn_layer >= n_attn_layers)
    return false;
  const std::size_t plane =
      Turbo4LayerPlaneBytes(head_dim, kv_heads, context, bits);
  if (plane == 0U)
    return false;

  const std::size_t maxv = static_cast<std::size_t>(-1);
  if (n_attn_layers > maxv / plane)
    return false;  // region product
  const std::size_t region = plane * n_attn_layers;
  // Two regions, K then V. Checked against the SUBTRACTION so the sum below
  // cannot wrap into something that looks like it fits.
  if (region > maxv / 2U)
    return false;
  const std::size_t two_regions = region * 2U;
  if (base_bytes > arena_bytes)
    return false;
  if (two_regions > arena_bytes - base_bytes)
    return false;
  if (attn_layer > maxv / plane)
    return false;  // unreachable, stated for the reader

  Turbo4LayerSlices slices;
  slices.plane_bytes = plane;
  slices.region_bytes = region;
  slices.k_offset = base_bytes + attn_layer * plane;
  slices.v_offset = base_bytes + region + attn_layer * plane;
  // The whole-buffer fit above already implies these two, but the body's
  // invariant is "check the slice you are about to hand out", and a caller that
  // recomputes one offset by hand deserves the same guarantee.
  if (!Turbo4BytesFit(slices.k_offset, plane, arena_bytes))
    return false;
  if (!Turbo4BytesFit(slices.v_offset, plane, arena_bytes))
    return false;
  *out = slices;
  return true;
}

/// One layer's valid prefix, in both the live and the snapshot buffer.
///
/// A snapshot holds `snapshot_context` positions; the live arena holds
/// `arena_context`. A layer's VALID PREFIX is therefore contiguous inside its
/// own region, but the prefix of the WHOLE buffer is not: the live buffer's
/// layer stride is `arena_context` tokens while the snapshot's is
/// `snapshot_context`, so "the first N tokens of everything" is a set of N
/// per-layer copies and never one contiguous span. That is the whole reason
/// this struct exists instead of a single offset.
struct Turbo4LayerPrefix {
  std::size_t layer{0};
  std::size_t k_src{0};  ///< live K offset of this layer
  std::size_t v_src{0};  ///< live V offset of this layer
  std::size_t k_dst{0};  ///< snapshot K offset of this layer
  std::size_t v_dst{0};  ///< snapshot V offset of this layer
  std::size_t bytes{0};  ///< bytes copied per plane
};

/// Per-layer prefix plan from a live arena into a snapshot buffer.
///
/// `arena_context` sizes the LIVE regions, `snapshot_context` sizes the
/// SNAPSHOT ones, and `live_base` / `snapshot_base` place them (0 for live, the
/// compact header size for a snapshot). Returns false on any refusal — see
/// `Turbo4LayerPlaneSlices` — and writes nothing.
///
/// THIS IS THE HONEST ANSWER TO "what does the snapshot transfer do today". For
/// a UNIFORM Turbo4 cache the existing 2D copy is already per-layer: its `rows`
/// is the layer count and its pitch is the layer stride, which is the same
/// number this function computes. `Turbo4LayerPrefixesAgreeWithUniformStride`
/// proves the two agree rather than assuming it, because "they look the same"
/// is how a snapshot silently restores layer 7's tokens into layer 8's bytes.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool Turbo4LayerPrefixPlan(
    std::size_t arena_bytes, std::size_t arena_context,
    std::size_t snapshot_bytes, std::size_t snapshot_context,
    std::size_t live_base, std::size_t snapshot_base, std::size_t head_dim,
    std::size_t kv_heads, std::size_t n_attn_layers, Turbo4LayerPrefix* out,
    std::size_t out_capacity, turbo4::Bits bits = turbo4::Bits::k4) noexcept {
  if (out == nullptr || n_attn_layers == 0U)
    return false;
  if (out_capacity < n_attn_layers)
    return false;
  if (snapshot_context == 0U || snapshot_context > arena_context)
    return false;
  for (std::size_t layer = 0; layer < n_attn_layers; ++layer) {
    Turbo4LayerSlices live{};
    Turbo4LayerSlices snap{};
    if (!Turbo4LayerPlaneSlices(arena_bytes, live_base, head_dim, kv_heads,
                                arena_context, n_attn_layers, layer, &live,
                                bits))
      return false;
    if (!Turbo4LayerPlaneSlices(snapshot_bytes, snapshot_base, head_dim,
                                kv_heads, snapshot_context, n_attn_layers,
                                layer, &snap, bits))
      return false;
    // The bytes actually moved are the SNAPSHOT's plane size — the valid
    // prefix — not the live one. Copying live.plane_bytes would read past the
    // end of the snapshot and write past the end of nothing.
    out[layer] =
        Turbo4LayerPrefix{layer,         live.k_offset, live.v_offset,
                          snap.k_offset, snap.v_offset, snap.plane_bytes};
  }
  return true;
}

/// True when the uniform-stride 2D copy describes exactly the per-layer plan.
///
/// THE CLAIM UNDER TEST: for a uniform Turbo4 cache, `width = token_pitch`,
/// `rows = layers` and `pitch = context * token_pitch` already address layer i
/// at `i * plane_bytes`, which is what `Turbo4LayerPrefixPlan` computes layer
/// by layer. If that holds, the fast 2D path may stay for the uniform case and
/// the slice path is only needed once the table is MIXED. If it does NOT hold,
/// the existing snapshot path is wrong for Turbo4 and the uniform shortcut has
/// to go.
[[nodiscard]] GUFO_TURBO4_HD constexpr bool
Turbo4LayerPrefixesAgreeWithUniformStride(std::size_t head_dim,
                                          std::size_t kv_heads,
                                          std::size_t n_attn_layers,
                                          std::size_t token_pitch) noexcept {
  const std::size_t plane =
      Turbo4LayerPlaneBytes(head_dim, kv_heads, kTurbo4ManifestContext);
  if (plane == 0U)
    return false;
  // One layer's plane is exactly `context` tokens of `token_pitch` bytes. That
  // identity is what makes rows=pitch/layer true for every context at once,
  // rather than only at the manifest context.
  for (std::size_t ctx = 1; ctx <= kTurbo4ManifestContext; ++ctx) {
    const std::size_t p = Turbo4LayerPlaneBytes(head_dim, kv_heads, ctx);
    if (p != ctx * token_pitch)
      return false;
  }
  (void)plane;
  (void)n_attn_layers;
  return true;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_TURBO4_PLANE_INDEX_HPP_

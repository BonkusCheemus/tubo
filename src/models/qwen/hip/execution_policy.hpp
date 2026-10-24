#ifndef GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include "turbo4_codec.hpp"

namespace gufo::hip {

enum class QwenKvCacheStorage : std::uint8_t {
  kFp32,
  kFp16,
  /// Turbo4: 128 values per block sharing one fp16 norm plus 64 packed nibble
  /// bytes (66 B, 4.125 bits/value). APPENDED, not inserted: the persisted
  /// snapshot header stores this enumerator's numeric value, so keeping
  /// kFp32=0 and kFp16=1 means an existing FP16 snapshot written by an older
  /// build still parses. Inserting kTurbo4 in the middle would renumber every
  /// snapshot already on disk.
  ///
  /// The arena.cpp precision bound was RAISED from kFp16 to this last
  /// enumerator, otherwise a Turbo4 snapshot would be written by one build and
  /// rejected as invalid by that same build. An older build still refuses it,
  /// which is the intended direction: new format, old reader says no.
  kTurbo4,
  /// Turbo5 (82 B per 128 values, 5.125 bits/value) and Turbo6 (98 B,
  /// 6.125). Same block-addressed family as kTurbo4, APPENDED after it for the
  /// same persistence reason (the snapshot header stores the numeric value), so
  /// kFp32=0, kFp16=1, kTurbo4=2 are unchanged. The arena.cpp precision bound
  /// moved to kTurbo6, the last member.
  kTurbo5,
  kTurbo6,
};

/// Precision of ONE plane — one layer's K, or one layer's V.
///
/// A whole-cache enumerator cannot express a MIXED cache: "the last N
/// KV-bearing layers are f16, the rest are Turbo4" is not a value of
/// `QwenKvCacheStorage`, it is a table. That table is this unit.
enum class QwenKvLayerFormat : std::uint8_t {
  kTurbo4 = 0,
  kFp16 = 1,
  /// APPENDED: the table cells persist these values.
  kTurbo5 = 2,
  kTurbo6 = 3,
};

[[nodiscard]] constexpr bool QwenKvLayerFormatIsBlockAddressed(
    QwenKvLayerFormat format) noexcept {
  return format == QwenKvLayerFormat::kTurbo4 ||
         format == QwenKvLayerFormat::kTurbo5 ||
         format == QwenKvLayerFormat::kTurbo6;
}

[[nodiscard]] constexpr turbo4::Bits QwenKvLayerBits(QwenKvLayerFormat f) noexcept {
  return f == QwenKvLayerFormat::kTurbo4 ? turbo4::Bits::k4
       : f == QwenKvLayerFormat::kTurbo5 ? turbo4::Bits::k5
       : f == QwenKvLayerFormat::kTurbo6 ? turbo4::Bits::k6
                                        : static_cast<turbo4::Bits>(0);
}

/// Preserve all legacy identities; only new uniform formats add a suffix.
[[nodiscard]] constexpr std::string_view QwenKvPersistenceWidthSuffix(
    QwenKvCacheStorage storage) noexcept {
  return storage == QwenKvCacheStorage::kTurbo5 ? "kv_storage=turbo5\n"
       : storage == QwenKvCacheStorage::kTurbo6 ? "kv_storage=turbo6\n" : "";
}

/// The per-layer format a whole-cache storage fills its table with.
[[nodiscard]] constexpr QwenKvLayerFormat QwenKvLayerFormatForStorage(
    QwenKvCacheStorage storage) noexcept {
  return storage == QwenKvCacheStorage::kTurbo4
             ? QwenKvLayerFormat::kTurbo4
             : (storage == QwenKvCacheStorage::kTurbo5
                    ? QwenKvLayerFormat::kTurbo5
                    : (storage == QwenKvCacheStorage::kTurbo6
                           ? QwenKvLayerFormat::kTurbo6
                           : QwenKvLayerFormat::kFp16));
}

// NO PER-ELEMENT SIZE HELPER, ON PURPOSE. kFp16 has one (2 bytes). kTurbo4 does
// not: it is block-addressed, 128 values sharing one fp16 norm plus 64 packed
// nibble bytes. A function returning sizeof(float) for it would be exactly the
// kind of plausible-looking constant that sizes a buffer 7.76x wrong, which is
// the defect class kv_allocation.hpp was written to end. Use
// QwenKvBytesForElements / Turbo4LayerPlaneBytes instead.

/// PER-LAYER, PER-K/V FORMAT TABLE.
///
/// The index is the ATTENTION-LAYER index (0..attention_layer_count-1), NOT the
/// block index. That distinction is the whole trap: this model has 65 blocks of
/// which only 16 carry a KV cache, so "the last 4 layers" taken over BLOCKS
/// selects the last 4 blocks, not the last 4 KV-bearing layers, and on this
/// geometry the two differ. The identical bug was seen on the strix
/// side (mode 2 of an earlier experiment hit only 3 KV layers of 8) — the same
/// mistake reaches gufo through the same requirement.
///
/// K and V are stored in one 16-bit cell per layer (low byte K, high byte V) so
/// the table cannot grow a third plane by accident, and so a copy is a memcpy.
struct QwenKvFormatTable {
  /// HARD CAP, and exceeding it is a REFUSAL, never a truncation. A truncated
  /// table would silently drop the tail the caller asked for — which is the
  /// half of the model that was meant to be the accurate half.
  static constexpr std::uint32_t kMaxLayers = 256;

  std::uint32_t layer_count{0};
  std::uint16_t cells[kMaxLayers]{};

  [[nodiscard]] constexpr QwenKvLayerFormat K(
      std::uint32_t layer) const noexcept {
    return static_cast<QwenKvLayerFormat>(
        static_cast<std::uint8_t>(cells[layer] & 0xFFU));
  }
  [[nodiscard]] constexpr QwenKvLayerFormat V(
      std::uint32_t layer) const noexcept {
    return static_cast<QwenKvLayerFormat>(
        static_cast<std::uint8_t>((cells[layer] >> 8U) & 0xFFU));
  }
  constexpr void Set(std::uint32_t layer, QwenKvLayerFormat k,
                     QwenKvLayerFormat v) noexcept {
    cells[layer] = static_cast<std::uint16_t>(
        static_cast<std::uint8_t>(k) |
        (static_cast<std::uint16_t>(static_cast<std::uint8_t>(v)) << 8U));
  }
  /// True when every layer holds `format` in BOTH K and V — i.e. the table is
  /// indistinguishable from a whole-cache enumerator, and the fast uniform path
  /// is allowed to run.
  [[nodiscard]] constexpr bool IsUniform(
      QwenKvLayerFormat format) const noexcept {
    for (std::uint32_t i = 0; i < layer_count; ++i)
      if (K(i) != format || V(i) != format)
        return false;
    return true;
  }
  [[nodiscard]] constexpr bool HasMixedPlanes() const noexcept {
    for (std::uint32_t i = 0; i < layer_count; ++i)
      if (K(i) != V(i))
        return true;
    return false;
  }
  /// Count of layers whose K is block-addressed. The allocation is sized from
  /// this, per K and per V separately.
  [[nodiscard]] constexpr std::uint32_t BlockAddressedCount(
      bool for_k) const noexcept {
    std::uint32_t n = 0;
    for (std::uint32_t i = 0; i < layer_count; ++i)
      if (QwenKvLayerFormatIsBlockAddressed(for_k ? K(i) : V(i)))
        ++n;
    return n;
  }
  /// WHY THE TABLE IS IN THE FINGERPRINT. The existing fingerprint encodes one
  /// whole-cache enumerator. Two tables that differ only in WHICH layers are
  /// f16 produced the same fingerprint, so a captured graph could be replayed
  /// against a cache whose layout no longer matched — the silent route switch
  /// the single-enumerator fingerprint was written to prevent, arriving through
  /// the table instead of the enumerator. Every cell is folded in, in order, so
  /// a different table is a different graph identity.
  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    h = (h ^ static_cast<std::uint64_t>(layer_count)) * 1099511628211ULL;
    for (std::uint32_t i = 0; i < layer_count; ++i)
      h = (h ^ static_cast<std::uint64_t>(cells[i])) * 1099511628211ULL;
    return h;
  }
};

[[nodiscard]] constexpr bool QwenKvFormatTablesEqual(
    const QwenKvFormatTable& a, const QwenKvFormatTable& b) noexcept {
  if (a.layer_count != b.layer_count ||
      a.layer_count > QwenKvFormatTable::kMaxLayers)
    return false;
  for (std::uint32_t i = 0; i < a.layer_count; ++i)
    if (a.cells[i] != b.cells[i])
      return false;
  return true;
}

/// The existing compact header identifies only the whole-cache format.
/// Refuse a table it cannot identify; RAM snapshots retain the complete table.
[[nodiscard]] constexpr bool QwenKvTableSupportsCompactSnapshot(
    const QwenKvFormatTable& table, QwenKvCacheStorage storage) noexcept {
  return table.layer_count <= QwenKvFormatTable::kMaxLayers &&
         (table.layer_count == 0 ||
          table.IsUniform(QwenKvLayerFormatForStorage(storage)));
}

inline void RequireQwenKvFormatTablesEqual(
    const QwenKvFormatTable& source, const QwenKvFormatTable& target) {
  if (!QwenKvFormatTablesEqual(source, target))
    throw std::invalid_argument("Qwen RAM snapshot per-layer KV formats mismatch");
}

inline void RequireCompactKvFormats(const QwenKvFormatTable& table,
                                    QwenKvCacheStorage storage) {
  if (!QwenKvTableSupportsCompactSnapshot(table, storage))
    throw std::logic_error(
        "mixed Turbo KV compact snapshots need per-layer format metadata");
}

/// Build a tail table: the last `tail_n` KV-BEARING layers take `tail_k` /
/// `tail_v`, every other KV-bearing layer takes `base_k` / `base_v`.
///
/// K and V are named SEPARATELY because one of the arms this table exists for
/// is asymmetric on purpose: giving K more bits than V is the largest single
/// lever in the KV-quantisation literature (the asymmetric K>V arm), and a builder that
/// set both planes from one enumerator could not state it at all.
///
/// `full_attention[il]` says whether BLOCK il carries a KV cache. This is the
/// only thing that makes "last N" mean what a reader thinks it means.
///
/// REFUSES, never guesses: no output, a null output pointer, zero blocks, more
/// blocks than the table cap, or a `tail_n` larger than the number of
/// KV-bearing layers — which would silently mean "all of them, not the N asked
/// for".
[[nodiscard]] constexpr bool MakeQwenKvTailFormatTable(
    const bool* full_attention, std::uint32_t block_count, std::uint32_t tail_n,
    QwenKvLayerFormat tail_k, QwenKvLayerFormat tail_v,
    QwenKvLayerFormat base_k, QwenKvLayerFormat base_v,
    QwenKvFormatTable* out) noexcept {
  if (out == nullptr || full_attention == nullptr)
    return false;
  if (block_count == 0U || block_count > QwenKvFormatTable::kMaxLayers)
    return false;
  std::uint32_t kv_layers = 0;
  for (std::uint32_t il = 0; il < block_count; ++il)
    if (full_attention[il])
      ++kv_layers;
  if (kv_layers == 0U)
    return false;
  // REFUSE rather than clamp. Clamping is how "the last 8 layers" silently
  // becomes "the last 3 that exist" and the arm is reported as run.
  if (tail_n > kv_layers)
    return false;

  out->layer_count = kv_layers;
  // Walk the KV-BEARING layers in order and index the table by ATTENTION-LAYER
  // ORDINAL. Blocks without a KV cache contribute nothing and take no cell.
  std::uint32_t attn = 0;
  for (std::uint32_t il = 0; il < block_count; ++il) {
    if (!full_attention[il])
      continue;
    // "The last tail_n of them". Compared as an ordinal against the count, so a
    // table of 16 with tail 4 selects exactly 4. An earlier form added the two
    // offsets together and selected 7 — the kind of off-by-N that produces a
    // plausible table and a wrong experiment.
    const bool is_tail = tail_n != 0U && attn >= kv_layers - tail_n;
    out->Set(attn, is_tail ? tail_k : base_k, is_tail ? tail_v : base_v);
    ++attn;
  }
  // The count must agree with the count that was measured, or a block was
  // classified differently between the two passes.
  if (attn != kv_layers)
    return false;
  return true;
}

enum class QwenRecurrentStateStorage : std::uint8_t {
  kFp32,
  kBf16,
};

[[nodiscard]] constexpr std::size_t QwenRecurrentStateElementBytes(
    QwenRecurrentStateStorage storage) noexcept {
  return storage == QwenRecurrentStateStorage::kBf16 ? sizeof(std::uint16_t)
                                                     : sizeof(float);
}

/// Immutable route policy for one Qwen GPU executor. Resolve this before HIP
/// graph capture. Alternate storage is used by independent operator controls.
enum class QwenExecutionMode : std::uint8_t {
  kDecode,
  kPrefill,
};

/// FNV-1a-style composition helpers for a deterministic executor-local graph
/// workload identity. Callers add configuration fields and resolved route
/// fingerprints; raw addresses must never participate.
[[nodiscard]] constexpr std::uint64_t
BeginQwenGraphWorkloadIdentity() noexcept {
  return 14695981039346656037ULL;
}

[[nodiscard]] constexpr std::uint64_t ExtendQwenGraphWorkloadIdentity(
    std::uint64_t identity, std::uint64_t value) noexcept {
  return (identity ^ value) * 1099511628211ULL;
}

struct QwenExecutionPolicy {
  bool fuse_qk_norm_rope_kv{true};
  QwenKvCacheStorage kv_cache_storage{QwenKvCacheStorage::kFp16};
  /// PER-LAYER, PER-K/V FORMAT. Empty (`layer_count == 0`) means "no table":
  /// the whole cache is `kv_cache_storage`, which is every existing caller's
  /// meaning and keeps their fingerprint untouched. Populated means the cache
  /// is MIXED and `kv_cache_storage` no longer describes the layout on its own.
  QwenKvFormatTable kv_layers{};
  bool kv_layers_repeat{
      false};  ///< runtime singleton pattern, resolved before allocation
  QwenRecurrentStateStorage recurrent_state_storage{
      QwenRecurrentStateStorage::kFp32};

  [[nodiscard]] static constexpr QwenExecutionPolicy Production() noexcept {
    return {};
  }

  [[nodiscard]] constexpr bool UsesFp16AttentionKv() const noexcept {
    return kv_cache_storage == QwenKvCacheStorage::kFp16;
  }

  [[nodiscard]] constexpr bool UsesBf16RecurrentState() const noexcept {
    return recurrent_state_storage == QwenRecurrentStateStorage::kBf16;
  }

  /// Stable bit fingerprint suitable for telemetry and graph-cache identity.
  ///
  /// The storage enumerator is encoded, NOT a boolean derived from it. An
  /// earlier version packed UsesFp16AttentionKv() at bit 8, which is injective
  /// only while the enumerator has two members: with kTurbo4 appended, kFp32
  /// and kTurbo4 both return false and produced the SAME fingerprint, so a
  /// graph captured for one storage layout could be replayed against another.
  /// That is a silent route switch inside the component meant to prevent them.
  ///
  /// kv bits are split across bit 8 (value bit 0) and bit 16 (value bit 1)
  /// rather than packed into a byte at 8..15, because bit 9 already belongs to
  /// UsesBf16RecurrentState: a whole byte would let kv=2 light the
  /// recurrent-state bit, aliasing Turbo4+fp32 state with fp32+bf16 state.
  /// Splitting keeps kFp32 and kFp16 at exactly the values they had before, so
  /// graph-cache entries written by an existing build stay reachable.
  ///
  /// kTurbo6 = 4 needs a third bit. Bit 17 is free (used bits: 0, 8, 9,
  /// 16) and (kv >> 2) is 0 for kFp32..kTurbo5, so every fingerprint a build
  /// with kFp32..kTurbo4 could write is unchanged.
  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    const auto kv = static_cast<std::uint64_t>(kv_cache_storage);
    std::uint64_t fp =
        (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 0U) |
        ((kv & 0x1U) << 8U) |
        (static_cast<std::uint64_t>(UsesBf16RecurrentState()) << 9U) |
        (((kv >> 1U) & 0x1U) << 16U) | (((kv >> 2U) & 0x1U) << 17U);
    // FOLDED ONLY WHEN A TABLE EXISTS. An empty table contributes nothing, so
    // every fingerprint an existing build could have written is unchanged and
    // its graph-cache entries stay reachable. When a table IS present the
    // layout is mixed and a graph captured under one table must not replay
    // under another.
    if (kv_layers.layer_count != 0U)
      fp = ExtendQwenGraphWorkloadIdentity(fp, kv_layers.Fingerprint());
    return fp;
  }
};

enum class QwenKvReaderKind : std::uint8_t {
  kElementWise,  ///< fp16/fp32 plane: LaunchAttention
  kBlockPacked,  ///< Turbo4 66-byte block pitch: writer -> block reader (+gate)
};

/// THE PER-LAYER ROUTE DECISION, resolved as pure data so it can be tested
/// without a device. Everything here was previously implicit in one `if` at the
/// decode call site, which is exactly where a wrong choice becomes a confident
/// wrong answer.
struct QwenLayerKvPlan {
  /// Whether the fused QK-norm + RoPE + KV-write kernel may run for this layer.
  bool fuse_qk_norm_rope_kv{false};
  QwenKvReaderKind reader{QwenKvReaderKind::kElementWise};
  /// Turbo4 is NOT graph-eligible, ever.
  bool graph_eligible{true};
};

/// Is attention layer `attn_layer_idx` block-addressed in EITHER plane?
///
/// Either plane, not both: a layer with Turbo4 K and fp16 V still needs a block
/// reader for K, and routing it to the element-wise reader because "V is fp16"
/// would read 66-byte blocks as fp16 pairs.
[[nodiscard]] constexpr bool QwenKvLayerIsBlockAddressed(
    const QwenExecutionPolicy& policy, std::uint32_t attn_layer_idx) noexcept {
  if (policy.kv_layers.layer_count == 0U)
    return false;
  if (attn_layer_idx >= policy.kv_layers.layer_count)
    return false;
  return QwenKvLayerFormatIsBlockAddressed(
             policy.kv_layers.K(attn_layer_idx)) ||
         QwenKvLayerFormatIsBlockAddressed(policy.kv_layers.V(attn_layer_idx));
}

/// Resolve the route for one attention layer.
///
/// THREE RULES, and the third is the one that is easiest to forget:
///
///  1. A BLOCK-PACKED LAYER CANNOT USE THE FUSED KERNEL. `opt-c010` writes the
///     KV cache at an element pitch and has no 66-byte-block variant, so
///     running it for a Turbo4 layer writes fp16-shaped bytes into a block
///     plane. The policy toggle is therefore OVERRIDDEN, not consulted: a
///     caller that set `fuse_qk_norm_rope_kv = true` and a Turbo4 table still
///     gets the unfused path, because the alternative is a silent corruption.
///  2. The reader follows the table, not the whole-cache enumerator.
///  3. A BLOCK-PACKED LAYER IS GRAPH-INELIGIBLE. The reader is dispatched with
///     the HOST's `pos + 1` rather than the device position, because the valid
///     context is a launch-time value the device cannot know. A captured graph
///     would replay with the context frozen at capture time — reading a shorter
///     context than the plane holds and silently dropping the tokens after it.
///     This is stated as a field rather than left as a comment at the call
///     site, because a comment is not something the graph guard can check.
[[nodiscard]] constexpr QwenLayerKvPlan ResolveQwenLayerKvPlan(
    const QwenExecutionPolicy& policy, std::uint32_t attn_layer_idx,
    bool head_dim_fusion_supported) noexcept {
  const bool block = QwenKvLayerIsBlockAddressed(policy, attn_layer_idx);
  QwenLayerKvPlan plan;
  plan.reader =
      block ? QwenKvReaderKind::kBlockPacked : QwenKvReaderKind::kElementWise;
  plan.fuse_qk_norm_rope_kv =
      !block && policy.fuse_qk_norm_rope_kv && head_dim_fusion_supported;
  plan.graph_eligible = !block;
  return plan;
}

/// Resolved route decisions for one layer invocation. This record is pure data
/// so policy selection can be tested independently from HIP launches.
struct QwenLayerRoutePlan {
  QwenExecutionMode mode{QwenExecutionMode::kDecode};
  bool full_attention{false};
  bool fuse_qk_norm_rope_kv{false};

  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    return (static_cast<std::uint64_t>(mode == QwenExecutionMode::kPrefill)
            << 0U) |
           (static_cast<std::uint64_t>(full_attention) << 1U) |
           (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 2U);
  }
};

enum class QwenRouteRejection : std::uint32_t {
  kNone = 0,
  kQkNormRopeKvRequiresAttention = 1U << 0U,
};

[[nodiscard]] constexpr QwenRouteRejection operator|(
    QwenRouteRejection lhs, QwenRouteRejection rhs) noexcept {
  return static_cast<QwenRouteRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenRouteRejection& operator|=(QwenRouteRejection& lhs,
                                         QwenRouteRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenRouteRejection(
    QwenRouteRejection reasons, QwenRouteRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

struct QwenLayerRouteResolution {
  QwenLayerRoutePlan plan;
  QwenRouteRejection rejected{QwenRouteRejection::kNone};
};

[[nodiscard]] constexpr QwenLayerRouteResolution
ResolveQwenLayerRouteWithReasons(const QwenExecutionPolicy& policy,
                                 QwenExecutionMode mode,
                                 bool full_attention) noexcept {
  const auto rejected = policy.fuse_qk_norm_rope_kv && !full_attention
                            ? QwenRouteRejection::kQkNormRopeKvRequiresAttention
                            : QwenRouteRejection::kNone;
  return {
      .plan =
          {
              .mode = mode,
              .full_attention = full_attention,
              .fuse_qk_norm_rope_kv =
                  full_attention && policy.fuse_qk_norm_rope_kv,
          },
      .rejected = rejected,
  };
}

[[nodiscard]] constexpr QwenLayerRoutePlan ResolveQwenLayerRoute(
    const QwenExecutionPolicy& policy, QwenExecutionMode mode,
    bool full_attention) noexcept {
  return ResolveQwenLayerRouteWithReasons(policy, mode, full_attention).plan;
}

enum class QwenGraphRejection : std::uint32_t {
  kNone = 0,
  kLogitsNotRequested = 1U << 0U,
  kSplitKAttentionRequired = 1U << 1U,
  /// A BLOCK-PACKED LAYER IS PRESENT, so capture must be refused.
  /// `ResolveQwenGraphRejections` knew nothing of block layers,
  /// so the one field that recorded the danger was set and never read — and a
  /// captured graph would have frozen the block reader's host `pos + 1`,
  /// reading a shorter context than the plane holds.
  ///
  /// Bit 2 is the only free one: 0, 1 and 3 are taken above.
  kBlockKvGraphIneligible = 1U << 2U,
  kGraphDisabled = 1U << 3U,
};

[[nodiscard]] constexpr QwenGraphRejection operator|(
    QwenGraphRejection lhs, QwenGraphRejection rhs) noexcept {
  return static_cast<QwenGraphRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenGraphRejection& operator|=(QwenGraphRejection& lhs,
                                         QwenGraphRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenGraphRejection(
    QwenGraphRejection reasons, QwenGraphRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

[[nodiscard]] constexpr QwenGraphRejection ResolveQwenGraphRejections(
    bool compute_logits, bool split_k_attention, bool graph_enabled) noexcept {
  QwenGraphRejection rejected = QwenGraphRejection::kNone;
  if (!compute_logits) {
    rejected |= QwenGraphRejection::kLogitsNotRequested;
  }
  if (split_k_attention) {
    rejected |= QwenGraphRejection::kSplitKAttentionRequired;
  }
  if (!graph_enabled) {
    rejected |= QwenGraphRejection::kGraphDisabled;
  }
  return rejected;
}

/// Block-family planes share the whole-cache slot pitch, but not the codec.
[[nodiscard]] constexpr bool QwenKvLayerFormatFitsStorage(
    QwenKvLayerFormat format, QwenKvCacheStorage storage) noexcept {
  const auto slot = QwenKvLayerFormatForStorage(storage);
  if (!QwenKvLayerFormatIsBlockAddressed(slot))
    return format == slot;
  return QwenKvLayerFormatIsBlockAddressed(format) &&
         QwenKvLayerBits(format) <= QwenKvLayerBits(slot);
}

/// DOES THE TABLE NEED A PLANE THE ARENA NEVER ALLOCATED?
///
/// `HasMixedPlanes` was written, tested and then never
/// consumed. So a table with an f16 tail under `kTurbo4` storage routed those
/// layers to `LaunchAttention` — the ELEMENT-WISE reader — with
/// `d_attention_kv_f16 == nullptr`, because the arena allocates only
/// `d_kv_turbo4` for that storage. That is a null pointer dereference on the
/// decode path, reached by configuring the thing we just built.
///
/// Block-family codecs may share uniform slots of the widest allocated format.
/// A mixed element/block table still needs an element plane the arena does not
/// allocate, so that combination remains refused. One predicate serves both
/// construction and the production guard.
[[nodiscard]] constexpr bool QwenKvTableIsServableByStorage(
    const QwenKvFormatTable& table, QwenKvCacheStorage storage) noexcept {
  if (table.layer_count == 0U)
    return true;  // no table: the enumerator rules
  // Block codecs can be narrower than the slot, never wider. Element/block
  // mixtures still cannot be represented by the single allocation.
  for (std::uint32_t i = 0; i < table.layer_count; ++i)
    if (!QwenKvLayerFormatFitsStorage(table.K(i), storage) ||
        !QwenKvLayerFormatFitsStorage(table.V(i), storage))
      return false;
  return true;
}

/// The first layer that needs a plane the arena did not allocate, or
/// `table.layer_count` when there is none. The index is what makes the refusal
/// message name a layer instead of saying "invalid".
[[nodiscard]] constexpr std::uint32_t QwenKvTableFirstUnservableLayer(
    const QwenKvFormatTable& table, QwenKvCacheStorage storage) noexcept {
  if (table.layer_count == 0U)
    return 0U;
  for (std::uint32_t i = 0; i < table.layer_count; ++i)
    if (!QwenKvLayerFormatFitsStorage(table.K(i), storage) ||
        !QwenKvLayerFormatFitsStorage(table.V(i), storage))
      return i;
  return table.layer_count;
}

/// Graph capture, now knowing about block layers.
///
/// `ResolveQwenGraphRejections` is kept as the plain predicate it was, and this
/// is the one the executor calls: it adds the block-layer refusal that the
/// plain version could not express. Every layer in the table is checked, not
/// just one, because a single block layer anywhere is enough to freeze a
/// context.
[[nodiscard]] constexpr QwenGraphRejection ResolveQwenGraphRejectionsForPolicy(
    const QwenExecutionPolicy& policy, bool compute_logits,
    bool split_k_attention, bool graph_enabled) noexcept {
  QwenGraphRejection rejected = ResolveQwenGraphRejections(
      compute_logits, split_k_attention, graph_enabled);
  for (std::uint32_t il = 0; il < policy.kv_layers.layer_count; ++il)
    if (QwenKvLayerIsBlockAddressed(policy, il)) {
      rejected |= QwenGraphRejection::kBlockKvGraphIneligible;
      break;
    }
  return rejected;
}

/// The pure part of the selection, so a CPU test can reach every case
/// without touching the process environment.
///
/// `block_kv` is GUFO_DEV_TURBO4_KV and `bits` is GUFO_DEV_TURBO_KV_BITS (null
/// = unset). Only block_kv == "1" opens a block format at all; `bits` then
/// picks the width: unset or "4" = Turbo4 (the behaviour before Turbo5/6,
/// unchanged), "5" = Turbo5, "6" = Turbo6. Any OTHER value of `bits` fails
/// CLOSED to `fallback` (production f16), for the same reason a typo in
/// block_kv does: the caller asked for a width we do not have, and silently
/// giving them Turbo4 would make a 5- or 6-bit A/B arm run as 4-bit.
[[nodiscard]] constexpr QwenKvCacheStorage ResolveQwenKvStorageFromEnv(
    const char* block_kv, const char* bits,
    QwenKvCacheStorage fallback) noexcept {
  if (block_kv == nullptr || std::string_view(block_kv) != "1")
    return fallback;
  if (bits == nullptr)
    return QwenKvCacheStorage::kTurbo4;
  const std::string_view b(bits);
  if (b == "4")
    return QwenKvCacheStorage::kTurbo4;
  if (b == "5")
    return QwenKvCacheStorage::kTurbo5;
  if (b == "6")
    return QwenKvCacheStorage::kTurbo6;
  return fallback;
}

/// Parse a singleton or CSV of per-attention-layer widths. No truncation,
/// whitespace aliases, empty fields, or mismatched K/V lengths are accepted.
[[nodiscard]] inline bool ConfigureQwenTurboKvFormats(
    QwenExecutionPolicy* policy, const char* k_text, const char* v_text) {
  if (k_text == nullptr && v_text == nullptr)
    return true;
  if (policy == nullptr)
    return false;
  QwenKvLayerFormat keys[QwenKvFormatTable::kMaxLayers]{};
  QwenKvLayerFormat values[QwenKvFormatTable::kMaxLayers]{};
  auto parse = [](const char* text, QwenKvLayerFormat* out,
                  std::uint32_t* count) {
    *count = 0;
    if (text == nullptr)
      return true;
    std::string_view rest(text);
    if (rest.empty())
      return false;
    for (;;) {
      if (*count == QwenKvFormatTable::kMaxLayers)
        return false;
      if (rest[0] < '4' || rest[0] > '6' || (rest.size() > 1 && rest[1] != ','))
        return false;
      out[(*count)++] = rest[0] == '4'   ? QwenKvLayerFormat::kTurbo4
                        : rest[0] == '5' ? QwenKvLayerFormat::kTurbo5
                                         : QwenKvLayerFormat::kTurbo6;
      if (rest.size() == 1)
        return true;
      rest.remove_prefix(2);
      if (rest.empty())
        return false;
    }
  };
  std::uint32_t nk = 0, nv = 0;
  if (!parse(k_text, keys, &nk) || !parse(v_text, values, &nv))
    return false;
  const std::uint32_t n = nk > nv ? nk : nv;
  if ((nk > 1 && nk != n) || (nv > 1 && nv != n))
    return false;
  const auto fallback = QwenKvLayerFormatForStorage(policy->kv_cache_storage);
  if (!QwenKvLayerFormatIsBlockAddressed(fallback))
    return false;
  auto widest = QwenKvLayerFormat::kTurbo4;
  QwenKvFormatTable table{};
  for (std::uint32_t i = 0; i < n; ++i) {
    const auto k = nk == 0 ? fallback : keys[nk == 1 ? 0 : i];
    const auto v = nv == 0 ? fallback : values[nv == 1 ? 0 : i];
    table.Set(i, k, v);
    if (QwenKvLayerBits(k) > QwenKvLayerBits(widest))
      widest = k;
    if (QwenKvLayerBits(v) > QwenKvLayerBits(widest))
      widest = v;
  }
  table.layer_count = n;
  policy->kv_layers = table;
  policy->kv_layers_repeat = n == 1;
  policy->kv_cache_storage =
      widest == QwenKvLayerFormat::kTurbo6   ? QwenKvCacheStorage::kTurbo6
      : widest == QwenKvLayerFormat::kTurbo5 ? QwenKvCacheStorage::kTurbo5
                                             : QwenKvCacheStorage::kTurbo4;
  return true;
}

/// The ONE runtime policy resolver. DEV ONLY, default OFF.
///
/// Production() stays constexpr and untouched: it is the compile-time default
/// for every declaration in the tree, and a getenv inside it would make all of
/// them depend on process state. Instead every RUNTIME construction path
/// resolves through this function, so GUFO_DEV_TURBO4_KV cannot be honoured on
/// one path and silently ignored on another. CLI entry points resolve inside
/// an error boundary and pass the result explicitly. Direct executor callers
/// still resolve through their default argument when no policy is supplied.
///
/// All runtime env selection is resolved here. Per-plane overrides must not
/// silently become an FP16 run when their parent gate is disabled or invalid.
///
/// Only the exact string "1" selects Turbo4. A typo fails CLOSED to production
/// f16 rather than opening a format the caller did not ask for.
[[nodiscard]] inline QwenExecutionPolicy ResolveRuntimeQwenExecutionPolicy() {
  QwenExecutionPolicy policy = QwenExecutionPolicy::Production();
  const char* gate = std::getenv("GUFO_DEV_TURBO4_KV");
  const char* bits = std::getenv("GUFO_DEV_TURBO_KV_BITS");
  const char* keys = std::getenv("GUFO_DEV_TURBO_KV_K_BITS");
  const char* values = std::getenv("GUFO_DEV_TURBO_KV_V_BITS");
  const bool enabled = gate != nullptr && std::string_view(gate) == "1";
  if (enabled && bits != nullptr && std::string_view(bits) != "4" &&
      std::string_view(bits) != "5" && std::string_view(bits) != "6")
    throw std::invalid_argument("invalid GUFO_DEV_TURBO_KV_BITS (expected 4, 5 or 6)");
  if ((keys != nullptr || values != nullptr) && !enabled)
    throw std::invalid_argument("per-layer Turbo KV overrides require an enabled block KV gate");
  const QwenKvCacheStorage chosen =
      ResolveQwenKvStorageFromEnv(gate, bits, policy.kv_cache_storage);
  if (chosen != policy.kv_cache_storage) {
    policy.kv_cache_storage = chosen;
    if (!ConfigureQwenTurboKvFormats(&policy, keys, values)) {
      throw std::invalid_argument("invalid per-layer Turbo KV K/V bit list");
    }
    // kv_layers is left EMPTY HERE ON PURPOSE, and that is now a documented
    // hand-off rather than a claim about meaning: C1c's comment here claimed
    // empty means "the whole cache", and that was FALSE - the resolvers read an
    // empty table as "no layer is block-addressed" and routed everything to the
    // element-wise f16 reader. The table is filled at executor construction by
    // ResolveQwenKvLayerTableForStorage (C1f), the one place that knows the
    // model's attention-layer count. This function only selects the storage.
  }
  return policy;
}

[[nodiscard]] constexpr bool QwenKvCacheStorageIsBlockAddressed(
    QwenKvCacheStorage storage) noexcept {
  return storage == QwenKvCacheStorage::kTurbo4 ||
         storage == QwenKvCacheStorage::kTurbo5 ||
         storage == QwenKvCacheStorage::kTurbo6;
}

/// The ONE place the per-layer table meets the whole-cache
/// enumerator, and it runs BEFORE the arena allocates a single plane.
///
/// WHY THIS EXISTS. An earlier version left `kv_layers` EMPTY on the theory
/// that empty means "the whole cache". The resolver says the opposite:
/// `QwenKvLayerIsBlockAddressed` returns false when `layer_count == 0`, so an
/// empty table routes EVERY layer to the element-wise f16 reader. Under Turbo4
/// the f16 planes are null, and the engine ran FusedQKNormRoPEKvWriteKernel and
/// then handed the f16 attention kernel a nil pointer. The empty table also
/// fed graph rejection (an empty loop sets no bit) and the slice math, so
/// Turbo4 stayed graph-eligible while reading null planes. C1c's comment was
/// the defect's cover story: it read like an intentional invariant.
///
/// UNIFORM, NOT MIXED. A block-addressed whole cache fills every attention
/// layer K=V=kTurbo4. A genuine mixed cache stays a caller-supplied table and
/// never reaches here with an empty one.
///
/// f16/fp32 with an empty table is UNCHANGED: element-wise is the correct route
/// for those, and the table is still unused by every resolver above.
[[nodiscard]] inline QwenExecutionPolicy ResolveQwenKvLayerTableForStorage(
    QwenExecutionPolicy policy, std::uint32_t attention_layer_count) {
  if (policy.kv_layers_repeat) {
    if (policy.kv_layers.layer_count != 1 || attention_layer_count == 0 ||
        attention_layer_count > QwenKvFormatTable::kMaxLayers)
      throw std::invalid_argument("invalid Turbo KV singleton layer pattern");
    const auto k = policy.kv_layers.K(0), v = policy.kv_layers.V(0);
    for (std::uint32_t i = 0; i < attention_layer_count; ++i)
      policy.kv_layers.Set(i, k, v);
    policy.kv_layers.layer_count = attention_layer_count;
    policy.kv_layers_repeat = false;
  }
  if (policy.kv_layers.layer_count != 0 &&
      policy.kv_layers.layer_count != attention_layer_count)
    throw std::invalid_argument(
        "Turbo KV bit list must match attention-layer count");
  if (QwenKvCacheStorageIsBlockAddressed(policy.kv_cache_storage) &&
      policy.kv_layers.layer_count == 0U) {
    if (attention_layer_count == 0U ||
        attention_layer_count > QwenKvFormatTable::kMaxLayers) {
      // Cannot fill, so refuse rather than fall through to the silent f16 path
      // this function exists to delete.
      throw std::invalid_argument(
          "Qwen block-addressed KV storage needs a per-layer table, and the "
          "model reports " +
          std::to_string(attention_layer_count) + " attention layers (cap " +
          std::to_string(QwenKvFormatTable::kMaxLayers) + ")");
    }
    for (std::uint32_t i = 0; i < attention_layer_count; ++i) {
      const QwenKvLayerFormat layer_format =
          QwenKvLayerFormatForStorage(policy.kv_cache_storage);
      policy.kv_layers.Set(i, layer_format, layer_format);
    }
    policy.kv_layers.layer_count = attention_layer_count;
  }
  // C1f invariant, and the backstop for any OTHER route that reaches
  // construction with a block-addressed cache and no table. Cheap, and it turns
  // a silent wrong-read into a refusal that names itself.
  if (QwenKvCacheStorageIsBlockAddressed(policy.kv_cache_storage) &&
      policy.kv_layers.layer_count == 0U) {
    throw std::invalid_argument(
        "Qwen block-addressed KV storage reached the executor with an EMPTY "
        "per-layer table; that routes every layer to the element-wise reader "
        "against null planes");
  }
  return policy;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_

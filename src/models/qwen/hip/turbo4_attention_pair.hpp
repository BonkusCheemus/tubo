#ifndef GUFO_MODELS_QWEN_HIP_TURBO4_ATTENTION_PAIR_HPP_
#define GUFO_MODELS_QWEN_HIP_TURBO4_ATTENTION_PAIR_HPP_

/// THE f16 pair a BLOCK-PACKED attention layer reads, built once.
///
/// WHY A HELPER AND NOT A CALL-SITE EXPRESSION. All
/// three block-layer call sites get their pair from here, and this function has
/// NO LAYER PARAMETER. The three facts that must hold together are therefore not
/// three things a caller can get independently wrong — they are one return
/// value:
///
///   * K is the scratch base and V is scratch + max_context*kv_heads*head_dim.
///   * the layer index handed to the attention launcher is 0, because the
///     scratch already HOLDS this layer — the dequant put it there;
///   * skip_kv_write is true, because the KV went into 66-byte blocks via the
///     E1 writer and there is nothing for the route to repack.
///
/// WHY LAYER 0 IS CORRECT RATHER THAN CONVENIENT. The three routes do not apply
/// the layer offset the same way — wmma (attention_wmma.hip:657-661) and tile
/// (attention_tile.hip:432-435) offset the read on the HOST, batched
/// (attention_batched.hip:264-266 + :116-118) recomputes it inside the kernel,
/// and wmma/tile also hand the BASE pointer plus layer_idx to their repack
/// kernels. Against a ONE-layer scratch, layer_idx 1 addresses exactly the V
/// half's first element — inside the allocation, so nothing faults, and the
/// answer is confidently wrong. Layer 2 is exactly one past the end. Pinning the
/// index to a constant here is what makes that unreachable rather than merely
/// discouraged; tests/models/qwen/turbo4_layer_offset_test.cpp pins the
/// arithmetic.
///
/// The f32 pair is null on BOTH members and must stay that way.
/// attention_batched.hip:36-62 stores raw f32 into k_cache/v_cache whenever
/// they are non-null, which on a block layer would write f32 over the blocks.
/// On a Turbo4 run d_kv_cache is null anyway (arena.cpp allocates only
/// d_kv_turbo4), so this returns null rather than relying on that.

#include <cstddef>
#include <cstdint>

namespace gufo::hip {

/// The layer index a block-layer route passes to an attention launcher.
///
/// A CONSTANT, not a default: a caller that wanted a different index cannot
/// pass one here, because there is no parameter to pass it in.
inline constexpr std::uint32_t kTurbo4AttentionLayerIdx = 0;

/// Whether a block-layer route asks its launcher to write KV back.
///
/// Also a constant, for the same reason. The KV for this chunk was written into
/// blocks by the E1 writer before the dequant ran; letting the attention
/// launcher repack would rewrite f16-shaped bytes into the 66-byte plane.
inline constexpr bool kTurbo4SkipKvWrite = true;

/// The four cache pointers one attention call receives, for a block layer.
struct Turbo4AttentionF16Pair {
  float* k_f32{nullptr};
  float* v_f32{nullptr};
  std::uint16_t* k{nullptr};
  std::uint16_t* v{nullptr};
  /// False when there is no usable scratch. A caller must REFUSE on this rather
  /// than launch with a null f16 pair, because the launcher's own refusal for a
  /// block layer fires for a different reason and the two would be
  /// indistinguishable in a log.
  bool valid{false};
};

/// Build the pair. Pure pointer arithmetic, host-only, no HIP dependency.
///
/// `scratch` must be the arena's `d_turbo4_attention_scratch`; the V offset
/// below is exactly half that buffer, so the last element written is the last
/// element the allocation owns.
///
/// That buffer covers `max_context` positions by default and only the
/// prefilled prefix under GUFO_DEV_SCRATCH_TILED. This helper does not care
/// which -- it takes the count from the arena either way, and "V starts at half
/// the allocation" is the only invariant it needs.
///
/// `layer_elements` is ONE LAYER's element count, and the caller passes
/// `arena.GetAttentionKvLayerElements()` rather than recomputing it. That is
/// the design rule and it is the safer shape: this header, the arena's scratch
/// sizer and the arena accessor would otherwise each hold their own copy of
/// `kv_heads * max_context * head_dim`, and the copy that is wrong is the one
/// nobody tests. The all-layer count is a DIFFERENT number and must never be
/// passed here -- it is 16x too large at the production geometry, which is a
/// silent overrun rather than a refusal.
[[nodiscard]] constexpr Turbo4AttentionF16Pair MakeTurbo4AttentionF16Pair(
    void* scratch, std::size_t layer_elements) noexcept {
  Turbo4AttentionF16Pair pair;
  if (scratch == nullptr || layer_elements == 0U) return pair;
  auto* base = static_cast<std::uint16_t*>(scratch);
  pair.k = base;
  pair.v = base + layer_elements;
  pair.valid = true;
  return pair;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_TURBO4_ATTENTION_PAIR_HPP_
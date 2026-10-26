#ifndef GUFO_MODELS_QWEN_HIP_TURBO4_LAUNCHERS_HPP_
#define GUFO_MODELS_QWEN_HIP_TURBO4_LAUNCHERS_HPP_

/// Host-side declarations of the two Turbo4 launchers.
///
/// WHY THIS FILE EXISTS. `LaunchTurbo4KvWrite` and
/// `LaunchTurbo4AttentionScalar` were declared only inside their own `.hip`
/// translation units, which meant the decode step could not call them without
/// inventing its own declarations of them — and an invented declaration is how
/// a caller passes the arguments in the wrong order and gets a kernel that
/// launches cleanly over the wrong buffers. The declarations live here, and
/// BOTH `.hip` files include this header alongside their definitions, so a
/// signature change that forgets the caller is a compile error rather than a
/// silent corruption.
///
/// Requires HIP. This header is not part of the CPU oracle surface: the oracle
/// tests the bodies in turbo4_kernels.hpp, which are host/device, not these
/// wrappers.
///
/// Every launcher requires explicit PlaneFormats: physical slot
/// stride and independent K/V codecs. Plane bases/capacities use the stride;
/// readers and writers use each plane's codec. No implicit Turbo4 selection.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/vision/rope.hpp"

namespace gufo::hip {

/// Write one position of K and V for EVERY kv head into a single layer's plane.
///
/// `k_plane` and `v_plane` are the LAYER's plane bases — the caller has already
/// applied the layer offset — and `plane_bytes` is that layer's plane size. The
/// writer does not add a layer offset of its own and must never be handed the
/// arena base and an offset separately, or the two are applied twice.
///
/// `d_pos` is a DEVICE pointer read on the device, so the position stays
/// correct under graph capture. `d_status` is the writer's own status word and
/// is NEVER dereferenced on the host.
Turbo4LaunchStatus LaunchTurbo4KvWrite(
    const float* k_post_rope, const float* v, std::uint8_t* k_plane,
    std::uint8_t* v_plane, const std::uint32_t* d_pos, std::size_t plane_bytes,
    std::size_t source_bytes, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    Turbo4KernelStatus* d_status, hipStream_t stream,
    turbo4::PlaneFormats formats);

/// E7 fused decode pre-step. One heterogeneous launch computes Q norm/RoPE and
/// its Turbo4 rotated query, while each KV-head CTA computes K norm/RoPE and
/// stages/commits the four K/V Turbo4 blocks for that head.
Turbo4LaunchStatus LaunchTurbo4DecodePrepare(
    const float* q, const float* k, const float* v, const float* q_norm_weight,
    const float* k_norm_weight, float* q_rot, std::uint8_t* k_plane,
    std::uint8_t* v_plane, const std::uint32_t* d_pos, std::size_t q_bytes,
    std::size_t kv_bytes, std::size_t q_rot_bytes, std::size_t plane_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t max_context, float rope_theta,
    float eps, Turbo4KernelStatus* d_write_status,
    Turbo4KernelStatus* d_read_status,
    const models::qwen::vision::DeviceRope* rope, hipStream_t stream,
    turbo4::PlaneFormats formats);

/// Write a whole CHUNK of post-RoPE K and V into one layer's Turbo4 planes.
///
/// The chunk is [count, num_kv_heads, head_dim] — note the extra leading axis
/// the single-position writer does not have. A Turbo4 block lives inside ONE
/// position of ONE head, so positions are independent and this is a grid over
/// (position, head) rather than a loop per head; each thread runs the
/// single-position body, so the packing stays in one place.
///
/// `d_first_pos` is a DEVICE pointer, as in the single-position writer, so the
/// position stays correct under graph capture. `count` is a HOST value because
/// it sizes the grid, and the WHOLE range is range-checked on the host before
/// the grid is dispatched.
///
/// `d_status` is REQUIRED and is never dereferenced on the host. It is LATCHED,
/// never reset: the caller resets once per step.
Turbo4LaunchStatus LaunchTurbo4KvWriteBatched(
    const float* k_post_rope, const float* v, std::uint8_t* k_plane,
    std::uint8_t* v_plane, const std::uint32_t* d_first_pos,
    std::uint32_t count, std::size_t plane_bytes, std::size_t source_bytes,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, Turbo4KernelStatus* d_status, hipStream_t stream,
    turbo4::PlaneFormats formats);

/// Dequantize ONE LAYER's Turbo4 blocks into an f16 scratch, positions
/// [0, count). Grid (count, num_kv_heads): one thread per (position, head),
/// each running the same `Turbo4DequantHeadUnitBody` the CPU range body loops.
///
/// `k_plane`/`v_plane` are this LAYER's block bases — `Turbo4BlockByteOffset`
/// takes no layer argument, so that base is the one address nothing checks.
///
/// `k_dst`/`v_dst` are the scratch's K and V halves. For a one-layer scratch
/// the caller supplies v_dst = k_dst + max_context * kv_heads * head_dim halfs
/// and passes layer_idx 0 to the attention launcher.
///
/// `d_status` is REQUIRED and LATCHED, never reset: the caller resets once per
/// chunk, the same contract the KV write follows.
Turbo4LaunchStatus LaunchTurbo4DequantLayer(
    const std::uint8_t* k_plane, const std::uint8_t* v_plane, void* k_dst,
    void* v_dst, std::uint32_t count, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    Turbo4KernelStatus* d_status, hipStream_t stream,
    turbo4::PlaneFormats formats);

/// Read every query head out of one layer's packed plane, optionally gated.
///
/// `k_plane` / `v_plane` are again the LAYER's plane bases, and `plane_bytes`
/// that layer's plane size. `valid_context` is a HOST value and is NOT read
/// from the device, which is what makes this route graph-ineligible: a captured
/// graph would replay with the context frozen at capture time. See
/// `ResolveQwenLayerKvPlan`.
///
/// `d_write_status` is REQUIRED (null returns kBadArguments) and is read on the
/// DEVICE; `d_status` receives the reader's own outcome and is never the same
/// word. The reader does not write `d_write_status`.
Turbo4LaunchStatus LaunchTurbo4AttentionScalar(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, std::size_t plane_bytes, std::size_t q_bytes,
    std::size_t out_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t valid_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats);

/// E5 parallel split-K decode reader. Q is rotated once per head; each CTA is
/// one (KV head, 128-token split), serves all six GQA query heads, and writes
/// (m,l,o) partials. A deterministic reducer merges splits, inverse-rotates the
/// output once, applies the sigmoid gate, then publishes `out`.
Turbo4LaunchStatus LaunchTurbo4AttentionSplit(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t valid_context, std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, bool query_is_rotated, turbo4::PlaneFormats formats);

/// `rows` verify rows in ONE launch per stage (rotate, split, reduce);
/// the row is a grid dimension. Row r reads first_valid_context + r positions
/// with its own split plan and merge order, so every row is bit-identical to
/// LaunchTurbo4AttentionSplit called with that context. q/out/gate are
/// [rows, num_heads * head_dim] contiguous; q_rot/out_rot likewise; `partials`
/// is `rows` slices of Turbo4BatchedDecodePartialBytes. All *_bytes are totals
/// for `rows` rows. The query is always rotated here.
Turbo4LaunchStatus LaunchTurbo4AttentionSplitBatched(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats);

/// Same contract and buffers as LaunchTurbo4AttentionSplitBatched, but
/// the split stage is one WMMA tile kernel per (KV head, split): each K/V tile
/// is dequantized to LDS once and all rows x 6 GQA heads loop over it.
/// rows must be <= 8 (kGeometryRefused otherwise: fall back to the batched
/// scalar reader). NOT bit-identical (f16 operands); dev-gated by the caller.
/// Serves Turbo4 and the Turbo5/6 qh / padded-slot formats (the block
/// codec is decoded per plane, exactly as the split reader does); the caller
/// gates Turbo5/6 behind GUFO_DEV_VERIFY_WMMA_WIDE.
Turbo4LaunchStatus LaunchTurbo4AttentionVerifyWmma(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats);

/// ATTN6: the K = V = Turbo6 verify reader (Turbo4VerifyAttn6Kernel), same
/// arguments and partial/output layout as LaunchTurbo4AttentionVerifyWmma.
/// splits_cap 0 = the WMMA reader's split plan (bitwise identical partials);
/// N > 0 = at most N splits per KV head (Attn6SplitSize). Refuses
/// (kGeometryRefused, nothing enqueued) anything but K = V = Turbo6 in 98-byte
/// slots with a 32-bit addressable plane, or GUFO_DEV_V_SKIP. Reached from
/// LaunchTurbo4AttentionVerifyWmma under GUFO_DEV_ATTN6_V3=1.
/// stages: kAttn6StagesAll (default) = rotate, reader, reduce as three
/// launches; a subset launches only those (bench timing, P4; the skipped
/// stages' buffers must already hold their results). kAttn6StageFused = ONE
/// launch (Turbo4VerifyAttn6FuseKernel, GUFO_DEV_ATTN_FUSE=1): on kOk the
/// same partials, out_rot and out bit for bit; on failure the same status for
/// any single failure class (buffers then unspecified). q_rot not written,
/// q and out_rot 16-byte aligned, at most 64 kv heads, one such launch in
/// flight per device (static arrival counters). kAttn6StageFusedB = the same plus design B step 2a
/// (Turbo4VerifyAttn6BKernel, GUFO_DEV_ATTN_B=1), same bits again.
inline constexpr std::uint32_t kAttn6StageRotate = 1U;
inline constexpr std::uint32_t kAttn6StageMain = 2U;
inline constexpr std::uint32_t kAttn6StageReduce = 4U;
inline constexpr std::uint32_t kAttn6StagesAll = 7U;
inline constexpr std::uint32_t kAttn6StageFused = 8U;
inline constexpr std::uint32_t kAttn6StageFusedB = 16U;
Turbo4LaunchStatus LaunchTurbo4AttentionVerifyAttn6(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats,
    std::uint32_t splits_cap, std::uint32_t stages = kAttn6StagesAll,
    int attn_c = -1);

/// VRD6: the K = V = Turbo6 verify reader built for memory speed
/// (Turbo4VerifyVrd6Kernel), same arguments and partial/output layout as
/// LaunchTurbo4AttentionVerifyWmma. splits_cap as for
/// LaunchTurbo4AttentionVerifyAttn6 (0 = the WMMA reader's plan, N = at most
/// N splits per KV head). Refuses (kGeometryRefused, nothing enqueued) what
/// the attn6 reader refuses. NOT bitwise vs either reader (norms folded, exp2,
/// different P rounding). Reached from LaunchTurbo4AttentionVerifyWmma under
/// GUFO_DEV_VRD6=1.
Turbo4LaunchStatus LaunchTurbo4AttentionVerifyVrd6(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats,
    std::uint32_t splits_cap);

/// Reset BOTH status words, once per decode step, ahead of the first layer.
///
/// The launchers deliberately do not reset: they latch
/// only, because a per-call reset erased a refusal from the previous attention
/// layer. The STEP is the unit at which a failure becomes meaningful, so the
/// step is where the reset belongs.
///
/// This is the only place either word is zeroed. It MUST be called before the
/// first attention layer of a step and NOT again inside the layer loop.
inline void Turbo4ResetStepStatusWords(Turbo4KernelStatus* write_status,
                                       Turbo4KernelStatus* read_status,
                                       hipStream_t stream) noexcept {
  if (write_status != nullptr)
    (void)hipMemsetAsync(write_status, 0, sizeof(Turbo4KernelStatus), stream);
  if (read_status != nullptr)
    (void)hipMemsetAsync(read_status, 0, sizeof(Turbo4KernelStatus), stream);
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_TURBO4_LAUNCHERS_HPP_

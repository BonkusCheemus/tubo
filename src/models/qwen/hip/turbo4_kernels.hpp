// Port of TheTom/llama-cpp-turboquant (MIT, Copyright (c) 2023-2026 The ggml authors; turbo5/turbo6 from PR #381
// by JakeATX), modified for HIP/gfx1151 in tubo. License text: licenses/llama-cpp-turboquant.txt.
#ifndef GUFO_MODELS_QWEN_HIP_TURBO4_KERNELS_HPP_
#define GUFO_MODELS_QWEN_HIP_TURBO4_KERNELS_HPP_

/// The COMMON host/device bodies of the Turbo4 writer and reader.
///
/// WHY THESE ARE NOT INSIDE THE .hip FILES — review
/// 0f80ea1b697a44e6bac55e19d835cc18.
///
/// A previous packet tested `Turbo4BlockByteOffset` and `QuantizeBlockChecked`
/// from a CPU fixture and called the kernels covered. They were not. The
/// fixture re-implemented the load-and-stage loop ITSELF, so a bug in the
/// kernel's own indexing was invisible to it. That is precisely the defect this
/// review found: the writer loaded `head*head_dim + tid` without the
/// block-within-head term, so both 256-wide head blocks repeated the first 128
/// values — and the CPU suite was green throughout, because the CPU suite was
/// never running the code it claimed to cover.
///
/// So the bodies live HERE, annotated host/device, and BOTH the kernels and the
/// CPU fixture call them. The arithmetic has one home. A kernel bug is now a
/// fixture failure by construction.
///
/// WHAT IS STILL NOT COVERED: grid dimension choice and launch geometry. Those
/// remain compile-only, which is why the reader's shared-memory partitioning is
/// now structural — one CTA per query head with a private scores buffer —
/// rather than an index arithmetic a test can only reason about.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "turbo4_codec.hpp"
#include "turbo4_plane_index.hpp"

namespace gufo::hip {

/// Host-side launch outcome, returned BY VALUE.
///
/// Review 18b70cfb17654e199dbd1626b3b2958b: the previous launch wrapper
/// dereferenced the device status pointer ON THE HOST (`*status = kOk`) before
/// passing the same pointer to the device. That is a host read-modify-write of
/// device memory — undefined at best, a fault at worst. The host must never
/// touch `d_status`; it learns geometry failures from THIS return value, and
/// only the device writes `d_status`.
enum class Turbo4LaunchStatus : std::uint32_t {
  kLaunched = 0,  ///< enqueued; d_status holds the ASYNC device outcome
  kGeometryRefused =
      1,               ///< host-side geometry check failed; nothing dispatched
  kBadArguments = 2,   ///< a required pointer was null, or plane_bytes == 0
  kEnqueueFailed = 3,  ///< HIP refused the enqueue; nothing will run
};

/// Record a failure WITHOUT letting a success erase it.
///
/// Every thread that runs a head latches its outcome here. A plain assignment
/// is wrong twice: the threads RACE on the same word, and `kOk == 0` from a
/// head that succeeded will happily overwrite another head's refusal — so the
/// final status reads "success" while one head was never written. That is the
/// silent stale success this whole design exists to prevent, arriving through
/// the status word itself.
///
/// The latch is a MAXIMUM, so it is monotone: `kOk` can never lower an already
/// recorded failure, and no interleaving of the threads can lose one. Which
/// failure code survives when there are several is the numerically largest —
/// deterministic, though not necessarily the first in time.
GUFO_TURBO4_HD inline void Turbo4LatchFailure(
    Turbo4KernelStatus* status, Turbo4KernelStatus value) noexcept {
  if (status == nullptr)
    return;
#if defined(__HIP_DEVICE_COMPILE__)
  atomicMax(reinterpret_cast<unsigned int*>(status),
            static_cast<unsigned int>(value));
#else
  // HOST BUILD — including hipcc's HOST PASS, which is the part this used to
  // get wrong. The guard read `__HIPCC__`, and hipcc defines that in BOTH
  // passes, so host code compiled by hipcc took the device branch and
  // referenced `atomicMax`, a __device__-only function: a hard compile error
  // the moment the host latched a status, which is what decode_step now does
  // once per step.
  //
  // No concurrency on the host, so a plain maximum IS the atomic maximum, and
  // the CPU oracle tests the same monotone rule the device executes.
  // `__HIPCC__` on its own is never a sufficient condition for "we are on the
  // device" —
  // `__HIP_DEVICE_COMPILE__` is, and it is defined in the device pass only.
  if (static_cast<unsigned int>(value) > static_cast<unsigned int>(*status))
    *status = value;
#endif
}

/// Finite check that works on host and device. A NaN or inf reaching the
/// quantizer produces a norm that stores fine and decodes to garbage, so this
/// is a refusal, not a warning.
[[nodiscard]] GUFO_TURBO4_HD inline bool Turbo4IsFinite(float value) noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
  return isfinite(value) != 0;
#else
  return std::isfinite(value);
#endif
}

/// WRITE ONE KV HEAD for one token, atomically.
///
/// Stages every block of the head — K then V — before any of them reaches the
/// plane, so a single unstorable block leaves the WHOLE head unchanged instead
/// of committing half a new head beside half a stale one.
///
/// The load is `head * head_dim + block_within_head * 128 + value`. The
/// block-within-head term is the one a previous draft omitted, which made both
/// blocks of a 256-wide head repeat the first 128 values. It is written out
/// longhand here rather than factored, because this is the expression that was
/// wrong and a factored form would hide it again.
///
/// K and V ARE PAIRED AND BOTH REQUIRED. There is no K-only write. An earlier
/// version of this comment claimed one while the body had no such path, and the
/// body then dereferenced a null `v` — review e94ad5dee5ea448197521aeb71af9d6c
/// reproduced SIGSEGV with `v = nullptr` against a non-null `v_plane`. The
/// contract is now the one the body always had: both planes or a refusal.
///
/// `formats` selects the block width (66/82/98 B). Default k4 is the
/// original behaviour.
GUFO_TURBO4_HD inline Turbo4KernelStatus Turbo4WriteHeadBody(
    const float* k_post_rope, const float* v, std::uint8_t* k_plane,
    std::uint8_t* v_plane, std::size_t plane_bytes, std::size_t source_bytes,
    std::uint32_t head, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    turbo4::PlaneFormats formats) noexcept {
  if (!formats.Valid())
    return Turbo4KernelStatus::kGeometryRefused;
  if (head >= num_kv_heads || num_kv_heads == 0U)
    return Turbo4KernelStatus::kGeometryRefused;
  // ALL FOUR buffers or nothing. `v` was checked for `v_plane` and not for
  // itself, so a caller passing a null `v` with a real `v_plane` got a null
  // pointer arithmetic on the very next line — `v + head_base` — and a
  // SIGSEGV instead of a refusal. The check is now on the pointer that is
  // actually dereferenced, not on its destination.
  if (k_post_rope == nullptr || v == nullptr || k_plane == nullptr ||
      v_plane == nullptr)
    return Turbo4KernelStatus::kGeometryRefused;
  // A non-null source pointer says nothing about how long it is. Without this
  // the head is read from a caller that allocated fewer than
  // num_kv_heads * head_dim floats.
  if (!Turbo4WriteBuffersAreSufficient(plane_bytes, source_bytes, num_kv_heads,
                                       head_dim))
    return Turbo4KernelStatus::kGeometryRefused;
  const std::size_t bph = Turbo4BlocksPerHead(head_dim);
  if (bph == 0U || !Turbo4HeadFitsStaging(head_dim))
    return Turbo4KernelStatus::kGeometryRefused;
  if (pos >= max_context)
    return Turbo4KernelStatus::kPositionOutOfRange;

  std::size_t base_byte = 0;
  if (!Turbo4BlockByteOffset(pos, head, head_dim, num_kv_heads, max_context,
                             &base_byte, formats.stride))
    return Turbo4KernelStatus::kGeometryRefused;
  // The head must fit the plane the caller ACTUALLY allocated. Without this the
  // only thing bounding the write is the caller's claim about its own context.
  const std::size_t block_bytes = turbo4::BlockBytes(formats.stride);
  const std::size_t head_span = bph * block_bytes;
  if (!Turbo4BytesFit(base_byte, head_span, plane_bytes))
    return Turbo4KernelStatus::kGeometryRefused;

  // ---- STAGE: all of K, then all of V, before touching the plane ----
  std::uint8_t stage[kTurboMaxStagedBytesPerHead]{};
  const std::size_t head_base = static_cast<std::size_t>(head) * head_dim;

  for (std::size_t b = 0; b < bph; ++b) {
    // head * head_dim + block_within_head * 128 + value_in_block
    const float* src = k_post_rope + head_base + b * turbo4::kBlockValues;
    for (std::size_t i = 0; i < turbo4::kBlockValues; ++i)
      if (!Turbo4IsFinite(src[i]))
        return Turbo4KernelStatus::kQuantizeRefused;
    if (!turbo4::QuantizeBlockChecked(src, stage + b * block_bytes, formats.k))
      return Turbo4KernelStatus::kQuantizeRefused;
  }
  for (std::size_t b = 0; b < bph; ++b) {
    const float* src = v + head_base + b * turbo4::kBlockValues;
    for (std::size_t i = 0; i < turbo4::kBlockValues; ++i)
      if (!Turbo4IsFinite(src[i]))
        return Turbo4KernelStatus::kQuantizeRefused;
    if (!turbo4::QuantizeBlockChecked(src, stage + (bph + b) * block_bytes,
                                      formats.v))
      return Turbo4KernelStatus::kQuantizeRefused;
  }

  // ---- COMMIT: only now, and all of it ----
  for (std::size_t b = 0; b < bph; ++b)
    std::memcpy(k_plane + base_byte + b * block_bytes, stage + b * block_bytes,
                block_bytes);
  for (std::size_t b = 0; b < bph; ++b)
    std::memcpy(v_plane + base_byte + b * block_bytes,
                stage + (bph + b) * block_bytes, block_bytes);
  return Turbo4KernelStatus::kOk;
}

/// THE BATCHED WRITER — one call, N positions of one head.
///
/// WHAT IT IS FOR. E1: prefill hands a CHUNK to the plane, and launching the
/// single-position writer N times costs N launches. This takes the whole chunk
/// so the kernel can spread it over a (position, head) grid.
///
/// WHY IT DELEGATES. A Turbo4 block lives inside ONE position
/// of ONE head — blocks run along head_dim, 2 per 256-wide head — so positions
/// are independent and a batch has no packing logic to own. What the batch DOES
/// add is the translation: the chunk-major source stride
/// `i * num_kv_heads * head_dim`, the position map `first_pos + i`, and the
/// whole-range refusal. Those are the failure modes, and they are exactly what
/// the CPU test checks. A second, independent quantizer would be a second
/// packing authority to keep equal forever: the earlier block_within_head
/// defect would have to be fixed twice, and only one copy would be in the
/// kernel's review path.
///
/// SOURCE LAYOUT. `Turbo4WriteHeadBody` receives ONE position,
/// [num_kv_heads, head_dim]. This receives the chunk,
/// [count, num_kv_heads, head_dim].
///
/// REFUSE THE WHOLE RANGE OR NONE OF IT. The range is validated BEFORE any
/// position is written, so a range that runs past max_context leaves the plane
/// untouched instead of landing a chunk that silently ends early.
///
/// MID-RANGE REFUSAL IS NOT ALL-OR-NOTHING, and that is deliberate. Once the
/// range is valid, a later position whose block cannot be stored (an unstorable
/// group norm, a non-finite value) stops the loop with the EARLIER positions
/// already committed. This is the decode contract too: a token that reached the
/// plane stays there, and the latched status word is what tells the caller the
/// chunk did not finish. A whole-batch rollback would mean re-quantizing on the
/// failure path, and would still not un-write a block a previous decode already
/// committed.
[[nodiscard]] GUFO_TURBO4_HD inline Turbo4KernelStatus Turbo4WriteHeadRangeBody(
    const float* k_post_rope, const float* v, std::uint8_t* k_plane,
    std::uint8_t* v_plane, std::size_t plane_bytes, std::size_t source_bytes,
    std::uint32_t head, std::uint32_t first_pos, std::uint32_t count,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    turbo4::PlaneFormats formats) noexcept {
  if (!formats.Valid())
    return Turbo4KernelStatus::kGeometryRefused;
  if (count == 0U)
    return Turbo4KernelStatus::kGeometryRefused;
  // Unsigned compare, not first_pos + count > max_context: the sum can wrap.
  if (first_pos >= max_context)
    return Turbo4KernelStatus::kPositionOutOfRange;
  if (count > max_context - first_pos)
    return Turbo4KernelStatus::kPositionOutOfRange;
  if (k_post_rope == nullptr || v == nullptr || k_plane == nullptr ||
      v_plane == nullptr)
    return Turbo4KernelStatus::kGeometryRefused;
  if (head >= num_kv_heads || num_kv_heads == 0U)
    return Turbo4KernelStatus::kGeometryRefused;

  const std::size_t bph = Turbo4BlocksPerHead(head_dim);
  if (bph == 0U || !Turbo4HeadFitsStaging(head_dim))
    return Turbo4KernelStatus::kGeometryRefused;

  const std::size_t pos_stride = static_cast<std::size_t>(num_kv_heads) *
                                 static_cast<std::size_t>(head_dim);
  const std::size_t needed =
      pos_stride * static_cast<std::size_t>(count) * sizeof(float);
  if (source_bytes < needed)
    return Turbo4KernelStatus::kGeometryRefused;

  // The plane must hold the LAST position's span; if it holds the first it
  // holds all of them, since offsets only increase with position.
  std::size_t last_base = 0;
  if (!Turbo4BlockByteOffset(first_pos + count - 1U, head, head_dim,
                             num_kv_heads, max_context, &last_base,
                             formats.stride))
    return Turbo4KernelStatus::kGeometryRefused;
  if (!Turbo4BytesFit(last_base, bph * turbo4::BlockBytes(formats.stride),
                      plane_bytes))
    return Turbo4KernelStatus::kGeometryRefused;

  // ONE packing authority: the single-position body, per position.
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::size_t off = static_cast<std::size_t>(i) * pos_stride;
    const auto st = Turbo4WriteHeadBody(
        k_post_rope + off, v + off, k_plane, v_plane, plane_bytes,
        pos_stride * sizeof(float), head, first_pos + i, max_context,
        num_kv_heads, head_dim, formats);
    if (st != Turbo4KernelStatus::kOk)
      return st;
  }
  return Turbo4KernelStatus::kOk;
}

/// DEQUANTIZE ONE POSITION OF ONE HEAD into an f16 scratch. THE UNIT.
///
/// k_plane and v_plane are PER-LAYER BASES. Turbo4BlockByteOffset takes no
/// layer argument, so the caller's layer base is the one part of this address
/// that nothing here checks: point it at the wrong layer and every block
/// decodes cleanly from the wrong data. Stated here rather than left implied.
///
/// `dst_offset` is where this (position, head) lands, which the range body
/// computes as t * kv_width + h * head_dim — the consumer's expression
/// (attention_wmma.hip:143 kHeadMajor=false, so :253-255).
[[nodiscard]] GUFO_TURBO4_HD inline Turbo4KernelStatus
Turbo4DequantHeadUnitBody(
    const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    std::uint16_t* k_dst, std::uint16_t* v_dst, std::size_t dst_offset,
    std::uint32_t t, std::uint32_t h, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    turbo4::PlaneFormats formats) noexcept {
  if (!formats.Valid())
    return Turbo4KernelStatus::kGeometryRefused;
  const std::size_t bph = Turbo4BlocksPerHead(head_dim);
  if (bph == 0U || h >= num_kv_heads)
    return Turbo4KernelStatus::kGeometryRefused;
  std::size_t base_byte = 0;
  if (!Turbo4BlockByteOffset(t, h, head_dim, num_kv_heads, max_context,
                             &base_byte, formats.stride))
    return Turbo4KernelStatus::kGeometryRefused;
  const std::size_t block_bytes = turbo4::BlockBytes(formats.stride);
  float scratch[turbo4::kBlockValues];
  for (std::size_t b = 0; b < bph; ++b) {
    for (int plane = 0; plane < 2; ++plane) {
      const std::uint8_t* blk =
          (plane == 0 ? k_plane : v_plane) + base_byte + b * block_bytes;
      // The SAME decode the single-position READ body runs (turbo4_kernels.hpp
      // :451-453 for K, :485-487 for V). Not a second definition of a block.
      turbo4::DequantizeBlock(blk, scratch, plane == 0 ? formats.k : formats.v);
      turbo4::RotateInverse(scratch);
      std::uint16_t* out =
          (plane == 0 ? k_dst : v_dst) + dst_offset + b * turbo4::kBlockValues;
      for (std::size_t i = 0; i < turbo4::kBlockValues; ++i)
        out[i] = turbo4::FloatToHalfBits(scratch[i]);
    }
  }
  return Turbo4KernelStatus::kOk;
}

/// DEQUANTIZE ONE LAYER'S BLOCKS INTO AN f16 SCRATCH, positions [0, count).
///
/// E2. The f16 attention routes cannot read a 66-byte block pitch, so a layer
/// staged in Turbo4 has to be materialised as f16 before they run.
///
/// THE RANGE BODY LOOPS THE UNIT, it does not carry its own copy of the decode.
/// The unit above is what the GPU kernel calls too (one thread per
/// (position, head), grid (count, num_kv_heads)), so there is exactly one place
/// that knows which block belongs where.
///
/// K AND V ARE TWO POINTERS ON PURPOSE. The launchers receive k_cache_f16 and
/// v_cache_f16 separately, and for a one-layer scratch the caller supplies
/// v_dst = scratch + max_context*kKvWidth halfs with layer_idx 0. Which pointer
/// the caller passes is its decision; pinning it here would bury a layout
/// question in the arithmetic.
[[nodiscard]] GUFO_TURBO4_HD inline Turbo4KernelStatus Turbo4DequantLayerBody(
    const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    std::uint16_t* k_dst, std::uint16_t* v_dst, std::uint32_t count,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    turbo4::PlaneFormats formats) noexcept {
  if (!formats.Valid())
    return Turbo4KernelStatus::kGeometryRefused;
  if (k_plane == nullptr || v_plane == nullptr || k_dst == nullptr ||
      v_dst == nullptr)
    return Turbo4KernelStatus::kGeometryRefused;
  if (count == 0U)
    return Turbo4KernelStatus::kGeometryRefused;
  if (count > max_context)
    return Turbo4KernelStatus::kPositionOutOfRange;
  if (num_kv_heads == 0U)
    return Turbo4KernelStatus::kGeometryRefused;
  if (Turbo4BlocksPerHead(head_dim) == 0U)
    return Turbo4KernelStatus::kGeometryRefused;

  const std::size_t kv_width = static_cast<std::size_t>(num_kv_heads) *
                               static_cast<std::size_t>(head_dim);
  for (std::uint32_t t = 0; t < count; ++t) {
    for (std::uint32_t h = 0; h < num_kv_heads; ++h) {
      const std::size_t dst_offset = static_cast<std::size_t>(t) * kv_width +
                                     static_cast<std::size_t>(h) * head_dim;
      const auto st = Turbo4DequantHeadUnitBody(
          k_plane, v_plane, k_dst, v_dst, dst_offset, t, h, max_context,
          num_kv_heads, head_dim, formats);
      if (st != Turbo4KernelStatus::kOk)
        return st;
    }
  }
  return Turbo4KernelStatus::kOk;
}

/// READ THE WRITER'S OUTCOME, FAILING CLOSED WHEN THERE IS NO WORD.
///
/// The launcher refuses a null `d_write_status`, but this
/// kernel is reachable without the launcher — the body comment says so itself —
/// so the decision belongs where the CPU oracle can reach it. A previous
/// version read `(d_write_status != nullptr) ? *d_write_status : kOk`, which
/// treats the ONE case the reader cannot evaluate as proof that the writer
/// succeeded. That is the silent stale success this design has refused since
/// §31, arriving through a null check.
///
/// Absent evidence of a successful write is a writer FAILURE. The read is a
/// plain device load on the writer's stream, so there is no synchronisation
/// here and no host round trip.
[[nodiscard]] GUFO_TURBO4_HD inline Turbo4KernelStatus Turbo4ReadWriterStatus(
    const Turbo4KernelStatus* d_write_status) noexcept {
  return (d_write_status != nullptr) ? *d_write_status
                                     : Turbo4KernelStatus::kWriterFailed;
}

/// THE SIGMOID GATE — the single arithmetic definition in the Turbo4 path.
///
/// EXPRESSION OF RECORD is `kernels/attention_decode.hip:165-168`:
///
///     const float sig = 1.0F / (1.0F + expf(-g));
///     ctx *= sig;
///
/// THE SAME WRONG FORMULA WAS HERE FIRST, and it is worth recording why nobody
/// caught it for a unit. This helper originally read `g / (1.0f + expf(-g))` —
/// that is `g * sigmoid(g)`, SiLU, not a gate. The CPU reference in
/// turbo4_decode_adapter_test.cpp had the identical expression written into it,
/// so REF1 passed at 0.000000 relative against a helper that was wrong. The
/// negative control ("delete the gate line and the suite goes red") PROVED THE
/// GATE WAS USED and said nothing about whether it was RIGHT: both sides moved
/// together because both were typed from the same wrong memory of the source.
///
/// THE RULE THAT FOLLOWS, and it is the one the test now enforces: a reference
/// is transcribed from the EXPRESSION OF RECORD, never from the code under
/// test, and it is pinned to literal values so the two cannot drift apart
/// silently again. `Turbo4SigmoidGatePinnedValues` holds gate(0)=0.5,
/// gate(2)=0.8807971, gate(-2)=0.1192029.
///
/// The shape matches the fp16 path: applied AFTER the softmax-weighted V sum,
/// not before, and multiplying rather than adding.
[[nodiscard]] GUFO_TURBO4_HD inline float Turbo4SigmoidGate(float g) noexcept {
  return 1.0f / (1.0f + expf(-g));
}

/// Stable merge for two online-softmax partials. Each input has a maximum m,
/// denominator l = sum(exp(score-m)), and an unnormalised value numerator o.
/// The caller applies the returned scales to both l and every component of o.
/// Keeping this arithmetic in the host/device header makes the split reducer
/// testable against a double CPU oracle without duplicating its equations.
struct Turbo4SoftmaxMerge {
  float maximum;
  float denominator;
  float left_scale;
  float right_scale;
};

[[nodiscard]] GUFO_TURBO4_HD inline Turbo4SoftmaxMerge Turbo4MergeSoftmax(
    float left_max, float left_sum, float right_max, float right_sum) noexcept {
  if (left_sum <= 0.0f)
    return {right_max, right_sum, 0.0f, right_sum > 0.0f ? 1.0f : 0.0f};
  if (right_sum <= 0.0f)
    return {left_max, left_sum, 1.0f, 0.0f};
  const float maximum = fmaxf(left_max, right_max);
  const float left_scale = expf(left_max - maximum);
  const float right_scale = expf(right_max - maximum);
  // Match the numerator merge: round the right product, fuse the left one.
  return {maximum, fmaf(left_sum, left_scale, right_sum * right_scale),
          left_scale, right_scale};
}

/// Stable split geometry shared by the arena estimator, allocator, launcher,
/// and CPU tests. Aim for about 20 splits, round each split to 32 positions,
/// and keep at least four iterations per warp at short contexts.
inline constexpr std::size_t kTurbo4DecodePositionsPerSplit = 1024;
inline constexpr std::size_t kTurbo4DecodeMinPositionsPerSplit = 32;
inline constexpr std::size_t kTurbo4DecodeSplitTargetCount = 20;
inline constexpr std::size_t kTurbo4DecodeWarpsPerSplit = 8;

[[nodiscard]] constexpr std::size_t Turbo4DecodeSplitSize(
    std::size_t context) noexcept {
  if (context == 0)
    return 0;
  const std::size_t target = context / kTurbo4DecodeSplitTargetCount +
                             (context % kTurbo4DecodeSplitTargetCount != 0);
  const std::size_t rounded = ((target + 31) / 32) * 32;
  return rounded < kTurbo4DecodeMinPositionsPerSplit
             ? kTurbo4DecodeMinPositionsPerSplit
             : (rounded > kTurbo4DecodePositionsPerSplit
                    ? kTurbo4DecodePositionsPerSplit
                    : rounded);
}

[[nodiscard]] constexpr std::size_t Turbo4DecodeSplitCount(
    std::size_t context) noexcept {
  const std::size_t split_size = Turbo4DecodeSplitSize(context);
  return split_size == 0 ? 0
                         : context / split_size + (context % split_size != 0);
}

/// Workspace must cover the largest split count at any valid context up to
/// max_context. The rounded split size has boundaries where the count drops.
[[nodiscard]] constexpr std::size_t Turbo4DecodeMaxSplitCount(
    std::size_t max_context) noexcept {
  std::size_t maximum = 0;
  for (std::size_t split_size = kTurbo4DecodeMinPositionsPerSplit;
       split_size <= kTurbo4DecodePositionsPerSplit; split_size += 32) {
    const std::size_t first_context =
        (split_size - kTurbo4DecodeMinPositionsPerSplit) *
            kTurbo4DecodeSplitTargetCount +
        1;
    const std::size_t last_context =
        split_size == kTurbo4DecodePositionsPerSplit
            ? max_context
            : split_size * kTurbo4DecodeSplitTargetCount;
    const std::size_t context =
        max_context < last_context ? max_context : last_context;
    if (context < first_context)
      continue;
    const std::size_t count =
        context / split_size + (context % split_size != 0);
    if (count > maximum)
      maximum = count;
  }
  return maximum;
}

[[nodiscard]] constexpr std::size_t Turbo4DecodePartialBytes(
    std::size_t max_context, std::size_t num_heads, std::size_t num_kv_heads,
    std::size_t head_dim) noexcept {
  const std::size_t max_value = static_cast<std::size_t>(-1);
  if (max_context == 0 || num_heads == 0 || num_kv_heads == 0 ||
      num_heads % num_kv_heads != 0 || head_dim == 0 ||
      head_dim % turbo4::kBlockValues != 0)
    return 0;
  const std::size_t splits = Turbo4DecodeMaxSplitCount(max_context);
  const std::size_t state_width = head_dim + 2;  // m, l, and o[d]
  if (state_width < head_dim || num_heads > max_value / state_width)
    return 0;
  const std::size_t head_state = num_heads * state_width;
  if (splits > max_value / head_state)
    return 0;
  const std::size_t partial_slots = splits;
  if (partial_slots > max_value / head_state)
    return 0;
  const std::size_t partial_values = partial_slots * head_state;
  if (partial_values > max_value / sizeof(float))
    return 0;
  return partial_values * sizeof(float);
}

[[nodiscard]] constexpr std::size_t Turbo4DecodeWorkspaceBytes(
    std::size_t max_context, std::size_t num_heads, std::size_t num_kv_heads,
    std::size_t head_dim) noexcept {
  const std::size_t max_value = static_cast<std::size_t>(-1);
  const std::size_t partial_bytes =
      Turbo4DecodePartialBytes(max_context, num_heads, num_kv_heads, head_dim);
  if (partial_bytes == 0)
    return 0;
  if (num_heads > max_value / head_dim)
    return 0;
  const std::size_t vector_values = num_heads * head_dim;
  if (vector_values > max_value / (2 * sizeof(float)))
    return 0;
  const std::size_t vector_bytes = vector_values * 2 * sizeof(float);
  if (vector_bytes > max_value - partial_bytes)
    return 0;
  return partial_bytes + vector_bytes;
}

/// Batched verify. Row r attends over first_context + r positions and
/// keeps its own split plan, so a row's partial slice only needs the largest
/// split count among the rows (not the max_context one). Returns bytes PER ROW,
/// or 0 on a geometry the single-row sizing would also refuse.
[[nodiscard]] constexpr std::size_t Turbo4BatchedDecodePartialBytes(
    std::size_t first_context, std::size_t rows, std::size_t num_heads,
    std::size_t head_dim) noexcept {
  if (first_context == 0 || rows == 0 || num_heads == 0 || head_dim == 0)
    return 0;
  std::size_t splits = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t count = Turbo4DecodeSplitCount(first_context + row);
    if (count > splits)
      splits = count;
  }
  return splits * num_heads * (head_dim + 2) * sizeof(float);
}

/// Scratch for `rows` rows: [rows x partials][rows x q_rot][rows x out_rot].
[[nodiscard]] constexpr std::size_t Turbo4BatchedDecodeWorkspaceBytes(
    std::size_t first_context, std::size_t rows, std::size_t num_heads,
    std::size_t head_dim) noexcept {
  const std::size_t partial =
      Turbo4BatchedDecodePartialBytes(first_context, rows, num_heads, head_dim);
  if (partial == 0)
    return 0;
  return rows * (partial + 2 * num_heads * head_dim * sizeof(float));
}

/// ATTEND for ONE query head, over a whole packed plane.
///
/// `scores` is a PRIVATE buffer of at least `valid_context` floats belonging to
/// the calling CTA alone. That is the whole fix for the race review
/// found: the previous kernel launched 24 threads in ONE CTA against a single
/// `extern __shared__` array, so all 24 query heads wrote `s_scores[t]` at the
/// same index. Private per CTA cannot collide.
///
/// STABLE SOFTMAX: the max is subtracted before exp, and every accumulation
/// runs in strictly increasing position order. Max subtraction keeps each
/// exponent
/// <= 0 so exp cannot overflow; strict order keeps the result independent of
/// how a compiler would schedule a tree reduction.
GUFO_TURBO4_HD inline Turbo4KernelStatus Turbo4AttentionQueryBody(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, const Turbo4BufferSizes& sizes,
    Turbo4KernelStatus writer_status, const float* gate,
    std::uint32_t query_head, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t valid_context, float scale, float* scores,
    turbo4::PlaneFormats formats) noexcept {
  if (!formats.Valid())
    return Turbo4KernelStatus::kGeometryRefused;
  if (num_heads == 0U || num_kv_heads == 0U || valid_context == 0U ||
      head_dim == 0U || scores == nullptr)
    return Turbo4KernelStatus::kGeometryRefused;
  if (q == nullptr || k_plane == nullptr || v_plane == nullptr ||
      out == nullptr)
    return Turbo4KernelStatus::kGeometryRefused;
  if (!Turbo4IsFinite(scale))
    return Turbo4KernelStatus::kGeometryRefused;

  // ---- WRITER FAILURE FIRST, BEFORE `out` IS EVER ADDRESSED ----
  //
  // The writer's word is a DEVICE pointer read here on the
  // DEVICE, in stream order, so there is no host synchronisation and no second
  // round trip. If the writer refused, the plane holds whatever was there
  // before — a stale or half-written context — and attending over it would
  // produce a confident number from data nobody wrote. That is the silent stale
  // success this whole design has refused since §31, arriving through the
  // reader.
  //
  // This is checked BEFORE the capacity gate on purpose: a caller whose buffers
  // are also short should hear about the writer, because that is the reason the
  // context is not coherent.
  if (writer_status != Turbo4KernelStatus::kOk)
    return Turbo4KernelStatus::kWriterFailed;

  // THE GATE AND ITS POINTER MUST AGREE. `gate_bytes == 0` is the single way to
  // say "no gate" (see Turbo4GateIsAbsent). A caller that states a gate size
  // and passes no pointer, or the reverse, has made a statement the body cannot
  // honour, and guessing which half was meant is how a gate silently disappears
  // from a model. Refuse instead.
  const bool gate_absent = Turbo4GateIsAbsent(sizes);
  if (gate_absent != (gate == nullptr))
    return Turbo4KernelStatus::kGeometryRefused;

  // ---- CAPACITY BEFORE ADDRESSING. NOTHING BELOW HERE DEREFERENCES ----
  //
  // ORDER IS THE WHOLE POINT. The first draft of this check sat below the
  // `q_head` finite-scan loop, where it guarded nothing: that loop had already
  // read `head_dim` floats past a short `q`. A check placed after the first
  // dereference is a comment, not a gate. So both capacity checks run here,
  // above every pointer arithmetic and every load, and the body below may
  // assume q, out, scores and the plane are as long as it has been told.
  if (!Turbo4AttentionBuffersAreSufficient(sizes, num_heads, head_dim,
                                           valid_context))
    return Turbo4KernelStatus::kGeometryRefused;

  const std::size_t bph = Turbo4BlocksPerHead(head_dim);
  if (bph == 0U)
    return Turbo4KernelStatus::kGeometryRefused;
  // THE PLANE MUST ACTUALLY HOLD EVERY (token, kv head) THIS LOOP ADDRESSES.
  //
  // This line was wrong, and a review probe measured it. The previous form was
  //   Turbo4BytesFit((valid_context - 1) * head_span, head_span, plane_bytes)
  // which computes the last token's FIRST head and omits the `kv_head * bph`
  // term entirely. A caller claiming one token more than it allocated therefore
  // passed the gate: 2112 bytes for 4 tokens, valid_context 5 -> 4*132 = 528,
  // and 528 <= 2112-132, so the check said yes. The loop then addressed token
  // 4, kv head 3 at byte 2508 and read 132 bytes from a 2112-byte plane, and
  // the body returned **kOk** while doing it — an out-of-bounds read reported
  // as success, which is the worst of the two failure modes.
  //
  // The whole-plane size is the correct requirement and it is the SAME number
  // the caller allocated with, so it is one authority rather than a second
  // hand-computed offset. The writer's own Turbo4BytesFit call is left alone:
  // its `base_byte` already carries both the token and the kv-head term.
  const std::size_t need_plane = Turbo4LayerPlaneBytes(
      head_dim, num_kv_heads, valid_context, formats.stride);
  if (need_plane == 0U || sizes.plane_bytes < need_plane)
    return Turbo4KernelStatus::kGeometryRefused;
  const std::size_t block_bytes = turbo4::BlockBytes(formats.stride);

  std::uint32_t kv_head = 0;
  if (!Turbo4KvHeadForQueryHead(query_head, num_heads, num_kv_heads, &kv_head))
    return Turbo4KernelStatus::kGeometryRefused;

  const float* q_head = q + static_cast<std::size_t>(query_head) * head_dim;
  float* out_head = out + static_cast<std::size_t>(query_head) * head_dim;
  for (std::size_t i = 0; i < head_dim; ++i)
    if (!Turbo4IsFinite(q_head[i]))
      return Turbo4KernelStatus::kGeometryRefused;

  // THE GATE IS SCANNED HERE, NOT WHERE IT IS APPLIED. Phase 4 is the first
  // place `out` is written, so a check sitting after it would refuse with the
  // caller's buffer already clobbered — the contract this body has kept since
  // the review that moved the capacity gate above the first dereference. A NaN
  // gate would otherwise reach `out` through a multiply and poison a whole head
  // with no status anyone reads.
  const float* gate_head = nullptr;
  if (gate != nullptr) {
    gate_head = gate + static_cast<std::size_t>(query_head) * head_dim;
    for (std::size_t i = 0; i < head_dim; ++i)
      if (!Turbo4IsFinite(gate_head[i]))
        return Turbo4KernelStatus::kGeometryRefused;
  }

  // PHASE 1 — validate every stored norm before decoding a single value. A
  // poisoned norm dequantizes to NaN in all 128 values, and a NaN propagates
  // through a score without tripping any bounds check.
  for (std::uint32_t t = 0; t < valid_context; ++t) {
    std::size_t base_byte = 0;
    if (!Turbo4BlockByteOffset(t, kv_head, head_dim, num_kv_heads,
                               valid_context, &base_byte, formats.stride))
      return Turbo4KernelStatus::kGeometryRefused;
    for (std::size_t b = 0; b < bph; ++b) {
      for (int plane = 0; plane < 2; ++plane) {
        const std::uint8_t* blk =
            (plane == 0 ? k_plane : v_plane) + base_byte + b * block_bytes;
        std::uint16_t stored_half = 0;
        std::memcpy(&stored_half, blk, sizeof(stored_half));
        if (!turbo4::StoredNormIsUsable(turbo4::HalfBitsToFloat(stored_half)))
          return Turbo4KernelStatus::kPoisonedNorm;
      }
    }
  }

  // PHASE 2 — scores.
  float running_max = -3.402823466e+38f;
  for (std::uint32_t t = 0; t < valid_context; ++t) {
    std::size_t base_byte = 0;
    if (!Turbo4BlockByteOffset(t, kv_head, head_dim, num_kv_heads,
                               valid_context, &base_byte, formats.stride))
      return Turbo4KernelStatus::kGeometryRefused;
    float score = 0.0f;
    for (std::size_t b = 0; b < bph; ++b) {
      float scratch[turbo4::kBlockValues];
      turbo4::DequantizeBlock(k_plane + base_byte + b * block_bytes, scratch,
                              formats.k);
      turbo4::RotateInverse(scratch);
      for (std::size_t i = 0; i < turbo4::kBlockValues; ++i)
        score += q_head[b * turbo4::kBlockValues + i] * scratch[i];
    }
    score *= scale;
    if (!Turbo4IsFinite(score))
      return Turbo4KernelStatus::kGeometryRefused;
    scores[t] = score;
    running_max = fmaxf(running_max, score);
  }

  // PHASE 3 — stable softmax.
  float denom = 0.0f;
  for (std::uint32_t t = 0; t < valid_context; ++t) {
    const float e = expf(scores[t] - running_max);
    scores[t] = e;
    denom += e;
  }
  if (!Turbo4IsFinite(denom) || denom <= 0.0f)
    return Turbo4KernelStatus::kDegenerateSoftmax;
  const float inv = 1.0f / denom;

  // PHASE 4 — weighted sum. `out` is first touched HERE, after every refusal
  // path has returned, so a refusal leaves the caller's buffer untouched.
  for (std::size_t i = 0; i < head_dim; ++i)
    out_head[i] = 0.0f;
  for (std::uint32_t t = 0; t < valid_context; ++t) {
    std::size_t base_byte = 0;
    if (!Turbo4BlockByteOffset(t, kv_head, head_dim, num_kv_heads,
                               valid_context, &base_byte, formats.stride))
      return Turbo4KernelStatus::kGeometryRefused;
    const float w = scores[t];
    for (std::size_t b = 0; b < bph; ++b) {
      float scratch[turbo4::kBlockValues];
      turbo4::DequantizeBlock(v_plane + base_byte + b * block_bytes, scratch,
                              formats.v);
      turbo4::RotateInverse(scratch);
      for (std::size_t i = 0; i < turbo4::kBlockValues; ++i)
        out_head[b * turbo4::kBlockValues + i] += w * scratch[i];
    }
  }
  for (std::size_t i = 0; i < head_dim; ++i)
    out_head[i] *= inv;

  // PHASE 5 — the sigmoid gate. AFTER the softmax-normalised sum, MULTIPLYING,
  // which is the shape the fp16 path uses (attention_decode.hip:165-168). Both
  // halves of that ordering are load-bearing: gating the SCORES would change
  // which tokens win the softmax, and adding rather than multiplying would make
  // the gate a bias. The gate is the plain logistic `1/(1+exp(-g))` — NOT
  // `g*sigmoid(g)`; that was a defect caught in review.
  //
  // `gate_head` is this query head's own row of a [num_heads, head_dim] tensor
  // — the same indexing the gate pointer has in attention_decode.hip:166. It
  // was bounds-checked once, above, against `gate_bytes`.
  if (gate_head != nullptr)
    for (std::size_t i = 0; i < head_dim; ++i)
      out_head[i] *= Turbo4SigmoidGate(gate_head[i]);
  return Turbo4KernelStatus::kOk;
}

/// THE TWO STATUS WORDS, READ BACK AS ONE VERDICT.
///
/// The failure mode this closes: the launchers used to
/// reset their word on every call, decode_step calls them once per attention
/// layer, so a refusal in layer k was erased by layer k+1 — and no host code
/// read either word at all, so a reader refusal left the caller's output buffer
/// holding the PREVIOUS layer's output, which the output projection then
/// consumed silently.
///
/// The words are now reset once per STEP by the caller, and read back together
/// here, at the same point as the sampled token. One predicate decides whether
/// the step may publish, so there is no path where one word is consulted and
/// the other is not.
struct Turbo4StepStatus {
  Turbo4KernelStatus write{Turbo4KernelStatus::kOk};
  Turbo4KernelStatus read{Turbo4KernelStatus::kOk};

  /// TRUE IF EITHER PLANE FAILED. Deliberately not "the read failed": a writer
  /// refusal is the more fundamental one, because the reader would otherwise be
  /// attending over a context nobody wrote.
  [[nodiscard]] constexpr bool Refuses() const noexcept {
    return write != Turbo4KernelStatus::kOk || read != Turbo4KernelStatus::kOk;
  }
};

/// Decode the two raw device words into a verdict.
///
/// Words are raw `std::uint32_t` because they are memcpy'd back from device
/// memory; an out-of-range value is NOT coerced into a valid enumerator, it is
/// reported as a failure. A word holding a number this build does not know
/// about means the device and the host disagree about the contract, and
/// treating that as "fine" is the silent-stale-success shape again.
[[nodiscard]] constexpr Turbo4StepStatus Turbo4StepStatusFromWords(
    std::uint32_t write_word, std::uint32_t read_word) noexcept {
  const auto decode = [](std::uint32_t raw) noexcept {
    switch (raw) {
      case 0U:
        return Turbo4KernelStatus::kOk;
      case 1U:
        return Turbo4KernelStatus::kGeometryRefused;
      case 2U:
        return Turbo4KernelStatus::kPositionOutOfRange;
      case 3U:
        return Turbo4KernelStatus::kQuantizeRefused;
      case 4U:
        return Turbo4KernelStatus::kPoisonedNorm;
      case 5U:
        return Turbo4KernelStatus::kDegenerateSoftmax;
      case 6U:
        return Turbo4KernelStatus::kWriterFailed;
      default:
        return Turbo4KernelStatus::kGeometryRefused;
    }
  };
  return Turbo4StepStatus{decode(write_word), decode(read_word)};
}


// ATTN6 (GUFO_DEV_ATTN6_V3=1, Turbo4VerifyAttn6Kernel): Turbo6 code -> byte
// offset into a 64-entry f32 centroid table, packed four codes per word.
// One slice = 32 values = nibble words w[4] (value j at w[j/8] bits 4(j%8))
// and qh words qh[2] (value j at qh[j/16] bits 8((j/4)%4) + 2(j%4)), exactly
// LoadTurbo4Slice<k6>. Out: words[h][r] byte k = 4 * code(16h + 4k + r).
// One v_perm per (h, nibble pair) + ~4 shift/mask ops per word = ~1 VALU per
// value, then 1 byte extract per value (was ~5 ops per value).
[[nodiscard]] GUFO_TURBO4_HD inline std::uint32_t Attn6Perm(
    std::uint32_t s0, std::uint32_t s1, std::uint32_t sel) noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_perm(s0, s1, sel);
#else
  // Selector bytes 0-3 pick s1 bytes, 4-7 pick s0 bytes (v_perm_b32).
  const std::uint64_t pool = (static_cast<std::uint64_t>(s0) << 32) | s1;
  std::uint32_t d = 0;
  for (std::uint32_t i = 0; i < 4; ++i)
    d |= static_cast<std::uint32_t>((pool >> (8U * ((sel >> (8U * i)) & 7U))) &
                                    0xFFU)
         << (8U * i);
  return d;
#endif
}

GUFO_TURBO4_HD inline void Attn6PackCodes(const std::uint32_t* w,
                                          const std::uint32_t* qh,
                                          std::uint32_t words[2][4]) noexcept {
  for (std::uint32_t h = 0; h < 2; ++h) {
    // Bytes holding nibble pairs (0,1) and (2,3) of the four 16-bit halves.
    const std::uint32_t pa = Attn6Perm(w[2 * h + 1], w[2 * h], 0x06040200U);
    const std::uint32_t pb = Attn6Perm(w[2 * h + 1], w[2 * h], 0x07050301U);
    const std::uint32_t q = qh[h];
    words[h][0] = ((pa << 2) & 0x3C3C3C3CU) | ((q << 6) & 0xC0C0C0C0U);
    words[h][1] = ((pa >> 2) & 0x3C3C3C3CU) | ((q << 4) & 0xC0C0C0C0U);
    words[h][2] = ((pb << 2) & 0x3C3C3C3CU) | ((q << 2) & 0xC0C0C0C0U);
    words[h][3] = ((pb >> 2) & 0x3C3C3C3CU) | (q & 0xC0C0C0C0U);
  }
}

/// ATTN6 split plan. splits_cap 0 = the WMMA reader's plan
/// (Turbo4DecodeSplitSize, so partials are bitwise comparable); otherwise at
/// most splits_cap splits per KV head (fewer, longer CTAs: one full wave of
/// resident CTAs instead of 2.1, and fewer partial records), rounded to 32.
/// Never more splits than Turbo4DecodeSplitCount, so the arena's partial
/// buffer always fits.
[[nodiscard]] GUFO_TURBO4_HD constexpr std::size_t Attn6SplitSize(
    std::size_t context, std::size_t splits_cap) noexcept {
  const std::size_t plan = Turbo4DecodeSplitSize(context);
  const std::size_t count = Turbo4DecodeSplitCount(context);
  if (plan == 0 || splits_cap == 0 || splits_cap >= count)
    return plan;
  const std::size_t target = (context + splits_cap - 1) / splits_cap;
  return ((target + 31) / 32) * 32;
}

// VRD6 (GUFO_DEV_VRD6=1, Turbo4VerifyVrd6Kernel): Turbo6 PAIR table. A value
// pair (2m, 2m + 1) of a block is one nibble byte (value 2m in the low
// nibble) plus one qh nibble (bits 0-1 = value 2m's high code bits, bits 2-3
// = value 2m + 1's). Index = nibble byte | qh nibble << 8 (12 bits); the
// entry holds both values as f16 CENTROIDS, without the norm (the reader
// folds the K norm into the score and the V norm into P):
//   low half  = c16[(idx & 0xF) | ((idx >> 8) & 3) << 4]
//   high half = c16[((idx >> 4) & 0xF) | ((idx >> 10) & 3) << 4],
// c16[code] = FloatToHalfBits(TURBO_CENTROIDS_6BIT[code]) (64 entries, built
// once per CTA, so the 4096 entries cost two table reads each).
inline constexpr std::uint32_t kVrd6LutEntries = 4096;

[[nodiscard]] GUFO_TURBO4_HD inline std::uint32_t Vrd6LutEntry(
    const std::uint16_t* c16, std::uint32_t idx) noexcept {
  const std::uint32_t a = (idx & 0xFU) | (((idx >> 8) & 3U) << 4);
  const std::uint32_t b = ((idx >> 4) & 0xFU) | (((idx >> 10) & 3U) << 4);
  return static_cast<std::uint32_t>(c16[a]) |
         (static_cast<std::uint32_t>(c16[b]) << 16);
}

/// v_perm_b32 for the selectors VRD6 uses: 0-7 pick a byte of {s0 (4-7),
/// s1 (0-3)}, 12 gives 0x00, 13-15 give 0xFF (8-11, the sign bytes, are not
/// used and not emulated: 0xFF here).
[[nodiscard]] GUFO_TURBO4_HD inline std::uint32_t Vrd6Perm(
    std::uint32_t s0, std::uint32_t s1, std::uint32_t sel) noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_perm(s0, s1, sel);
#else
  const std::uint64_t pool = (static_cast<std::uint64_t>(s0) << 32) | s1;
  std::uint32_t d = 0;
  for (std::uint32_t i = 0; i < 4; ++i) {
    const std::uint32_t s = (sel >> (8U * i)) & 0xFFU;
    const std::uint32_t byte =
        s < 8U ? static_cast<std::uint32_t>((pool >> (8U * s)) & 0xFFU)
               : (s == 12U ? 0U : 0xFFU);
    d |= byte << (8U * i);
  }
  return d;
#endif
}

/// LDS byte offsets into the pair table of the 16 pairs of one 32-value
/// Turbo6 slice (w = its 16 nibble bytes, qh = its 8 qh bytes, exactly as
/// LoadTurbo4Slice<k6> / LoadTurbo6Raw read them): out[m] = 4 * index of
/// values (2m, 2m + 1). Pair m's nibble byte is w byte m; its qh nibble is
/// nibble m of the 64 qh bits, i.e. byte (m % 8) / 2 of qh[m / 8] masked to
/// even (he) or odd (ho) nibbles. One v_perm + one shift per pair.
GUFO_TURBO4_HD inline void Vrd6PairOffsets(const std::uint32_t* w,
                                           const std::uint32_t* qh,
                                           std::uint32_t* out) noexcept {
  for (std::uint32_t r = 0; r < 2; ++r) {
    const std::uint32_t he = qh[r] & 0x0F0F0F0FU;         // nibbles 0, 2, 4, 6
    const std::uint32_t ho = (qh[r] >> 4) & 0x0F0F0F0FU;  // nibbles 1, 3, 5, 7
    for (std::uint32_t n = 0; n < 8; ++n) {
      const std::uint32_t m = 8U * r + n;
      const std::uint32_t sel = (m & 3U) | ((4U + n / 2U) << 8) | 0x0C0C0000U;
      out[m] = Vrd6Perm((n & 1U) != 0U ? ho : he, w[m >> 2], sel) << 2;
    }
  }
}

/// Column of key `key` (0..31 of a 32-key tile) in the VRD6 P and V^T tiles
/// and in its norm rows: within each 16-key half, even keys first, then odd
/// (key 2i + h of half kb -> 16 kb + 8 h + i). The S^T WMMA leaves keys
/// 2i + h (i = 0..7) of one query in one lane, so this order makes them one
/// contiguous 16-byte P store; V^T and the norms use the same order so the PV
/// product pairs every P column with its own key.
[[nodiscard]] GUFO_TURBO4_HD constexpr std::uint32_t Vrd6KeyColumn(
    std::uint32_t key) noexcept {
  return (key & 16U) | ((key & 1U) << 3) | ((key & 15U) >> 1);
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_TURBO4_KERNELS_HPP_

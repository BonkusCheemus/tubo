#ifndef GUFO_MODELS_QWEN_HIP_KV_ALLOCATION_HPP_
#define GUFO_MODELS_QWEN_HIP_KV_ALLOCATION_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include "execution_policy.hpp"
#include "turbo4_codec.hpp"

/// Block-granular KV allocation arithmetic.
///
/// arena.cpp sizes its KV cache as ELEMENTS x UNIFORM BYTES PER ELEMENT, which
/// cannot express a format whose bytes are shared across a group. Turbo4 packs
/// 128 values into ONE block carrying a single fp16 norm plus 64 packed nibble
/// bytes: 66 bytes per 128 values, 4.125 bits per value. Two consequences, and
/// they are the reason this header exists rather than a one-line ternary:
///
///   1. Bytes-per-element does not exist for Turbo4. A single K value cannot be
///      read without its block's norm, so the allocation UNIT has to become the
///      block. `elements * 4.125 / 8` is algebraically the same number when the
///      count is aligned and truncates when it is not, so it looks correct in
///      review and loses the last block's bytes on a tail.
///   2. The norm is shared, which is why a whole-cache dequantize is never
///      required: any single block is independently readable.
///
/// The Turbo4 constants below mirror the deployed turbo4 code of
/// TheTom/llama-cpp-turboquant (MIT), as ported to HIP in a llama.cpp build:
///   ggml/src/ggml-common.h:330,336-340
///     QK_TURBO4 128, ggml_half norm + uint8_t qs[64], static_assert 66 bytes
/// The 68-byte comment at ggml-common.h:334-335 is stale and is contradicted by
/// the static_assert on the following line; the assert is what is followed
/// here.

namespace gufo::hip {

inline constexpr std::size_t kQwenTurbo4BlockValues = 128;
inline constexpr std::size_t kQwenTurbo4BlockBytes = 66;
/// From TheTom/llama-cpp-turboquant PR #381 ggml-common.h
/// (block_turbo5_0, block_turbo6_0 static_asserts). Tied to the codec below so
/// the two authorities cannot drift.
inline constexpr std::size_t kQwenTurbo5BlockBytes = 82;
inline constexpr std::size_t kQwenTurbo6BlockBytes = 98;
static_assert(kQwenTurbo4BlockBytes == turbo4::BlockBytes(turbo4::Bits::k4) &&
                  kQwenTurbo5BlockBytes ==
                      turbo4::BlockBytes(turbo4::Bits::k5) &&
                  kQwenTurbo6BlockBytes == turbo4::BlockBytes(turbo4::Bits::k6),
              "kv_allocation block sizes must equal the codec's");

/// Effective bits per value, 4.125. Named because it is a documentation figure
/// and NOT an allocation term: multiplying by it is the mistake above.
inline constexpr double kQwenTurbo4BitsPerValue =
    static_cast<double>(kQwenTurbo4BlockBytes) * 8.0 /
    static_cast<double>(kQwenTurbo4BlockValues);

/// Overflow-safe ceiling division. The obvious form `n + d - 1` wraps when
/// `n` is within `d` of SIZE_MAX, returning 0 or 1 instead of the true count —
/// and it returns the *smaller* wrong answer, so the error survives review.
/// Review: division first, remainder as the increment.
[[nodiscard]] constexpr std::size_t CeilDiv(std::size_t numerator,
                                            std::size_t denominator) noexcept {
  if (denominator == 0U)
    return 0U;
  return numerator / denominator + (numerator % denominator != 0U ? 1U : 0U);
}

/// True when `a * b` cannot be represented. Checked before every allocation
/// size is formed; an undetected wrap would allocate a tiny buffer for a large
/// cache and corrupt on first write.
[[nodiscard]] constexpr bool QwenKvMultiplyOverflows(std::size_t a,
                                                     std::size_t b) noexcept {
  return b != 0U && a > (static_cast<std::size_t>(-1) / b);
}

/// Checked multiply. Returns false and leaves `*out` untouched on overflow.
[[nodiscard]] constexpr bool QwenKvMulChecked(std::size_t a, std::size_t b,
                                              std::size_t* out) noexcept {
  if (QwenKvMultiplyOverflows(a, b))
    return false;
  *out = a * b;
  return true;
}

/// True when this storage cannot address a single element and must be addressed
/// in whole blocks.
[[nodiscard]] constexpr bool QwenKvIsBlockAddressed(
    QwenKvCacheStorage storage) noexcept {
  return storage == QwenKvCacheStorage::kTurbo4 ||
         storage == QwenKvCacheStorage::kTurbo5 ||
         storage == QwenKvCacheStorage::kTurbo6;
}

/// The codec width of a block-addressed storage. Meaningful only when
/// QwenKvIsBlockAddressed(storage); anything else answers k4, which no caller
/// reaches because every use sits behind the block-addressed check.
[[nodiscard]] constexpr turbo4::Bits QwenKvTurboBits(
    QwenKvCacheStorage storage) noexcept {
  return storage == QwenKvCacheStorage::kTurbo5
             ? turbo4::Bits::k5
             : (storage == QwenKvCacheStorage::kTurbo6 ? turbo4::Bits::k6
                                                       : turbo4::Bits::k4);
}

/// The codec widths for one attention layer, inside the uniform slot pitch.
[[nodiscard]] inline turbo4::PlaneFormats QwenKvTurboLayerFormats(
    const QwenExecutionPolicy& policy, std::uint32_t layer) {
  if (!QwenKvIsBlockAddressed(policy.kv_cache_storage))
    throw std::invalid_argument(
        "requested block codecs for element KV storage");
  const auto stride = QwenKvTurboBits(policy.kv_cache_storage);
  if (layer >= policy.kv_layers.layer_count)
    throw std::invalid_argument("Turbo KV format table has no requested layer");
  auto bits = [](QwenKvLayerFormat f) {
    return f == QwenKvLayerFormat::kTurbo6   ? turbo4::Bits::k6
           : f == QwenKvLayerFormat::kTurbo5 ? turbo4::Bits::k5
                                             : turbo4::Bits::k4;
  };
  if (!QwenKvLayerFormatFitsStorage(policy.kv_layers.K(layer),
                                    policy.kv_cache_storage) ||
      !QwenKvLayerFormatFitsStorage(policy.kv_layers.V(layer),
                                    policy.kv_cache_storage))
    throw std::invalid_argument(
        "Turbo KV plane codec does not fit storage slot");
  return {stride, bits(policy.kv_layers.K(layer)),
          bits(policy.kv_layers.V(layer))};
}

/// Bytes per block for a block-addressed storage (66 / 82 / 98).
[[nodiscard]] constexpr std::size_t QwenKvBlockBytes(
    QwenKvCacheStorage storage) noexcept {
  return turbo4::BlockBytes(QwenKvTurboBits(storage));
}

/// True when `storage` is a member of the enumerator at all.
///
/// A caller can arrive with an arbitrary value — a truncated read from a
/// header, a cast from a wire byte. The old `storage == kFp16 ? 2 :
/// sizeof(float)` answered 4 for ANY value that was not fp16, including 99,
/// so an unknown precision silently reported a valid element size and every
/// size derived from it was confidently wrong. Unknown must mean unknown.
[[nodiscard]] constexpr bool QwenKvStorageIsKnown(
    QwenKvCacheStorage storage) noexcept {
  return storage == QwenKvCacheStorage::kFp32 ||
         storage == QwenKvCacheStorage::kFp16 ||
         storage == QwenKvCacheStorage::kTurbo4 ||
         storage == QwenKvCacheStorage::kTurbo5 ||
         storage == QwenKvCacheStorage::kTurbo6;
}

/// True when the executor can actually ALLOCATE a plane for this storage.
///
/// This exists as a predicate rather than as an `if` inside arena.cpp so the
/// decision is TESTABLE on a CPU. arena.cpp is HIP-only and cannot be run
/// here, so an inline `if (storage == kFp16) ... else fp32` could only ever be
/// asserted by grepping its own source text — which is how the bug survived
/// review in the first place: `UsesFp16AttentionKv()` is true for fp16 only, so
/// kTurbo4 silently took the fp32 branch and was handed a buffer 7.757576x
/// larger per value than a Turbo4 plane needs (fp32 4 B/value, Turbo4
/// 66/128 = 0.515625 B/value). Over-allocation is the SAFE direction: no
/// Turbo4-shaped offset can run past the end of an fp32-sized buffer. The bug
/// wasted memory and mis-reported sizes; it did not corrupt memory.
///
/// DELIBERATELY NOT the same predicate as QwenKvHasElementBytes. A block plane
/// is allocatable while having no bytes-per-element, so welding the two
/// together makes one of them wrong the moment the other is right. They answer
/// different questions: "can the executor hold it" vs "does a single value
/// have a size". The storage that must be refused by BOTH is the unknown one.
[[nodiscard]] constexpr bool QwenKvIsAllocatableByExecutor(
    QwenKvCacheStorage storage) noexcept {
  return QwenKvStorageIsKnown(storage);
}

/// True when a KERNEL can consume a plane of this storage — i.e. the executor
/// can both hold it and read it. Holding a block plane and being able to
/// execute against it are separate achievements, and flipping only the first
/// is how an engine ends up running against a buffer nothing reads. Block
/// storage is held (see QwenKvIsAllocatableByExecutor) but not yet consumed:
/// no attention kernel takes a 66-byte block pitch.
[[nodiscard]] constexpr bool QwenKvIsExecutableByExecutor(
    QwenKvCacheStorage storage) noexcept {
  return QwenKvStorageIsKnown(storage) && !QwenKvIsBlockAddressed(storage);
}

/// Which execution entry is asking. The switch below is phase-scoped: decode
/// and prefill are different answers, not a stricter and a laxer version of
/// one.
///
/// `kNoKvAccess` is for an entry that touches NO KV plane at
/// all — vision configuration with no image prompt, for instance. It is not a
/// licence to execute against block storage: the only way in is the same
/// GUFO_DEV_TURBO4_DECODE-gated branch that admits kDecode, and only for KNOWN
/// block-addressed storage. The distinction that matters is that an entry
/// claiming kNoKvAccess while it does read the KV plane is now a lie the guard
/// cannot detect, so the call site carries the proof (a null prompt) and the
/// next reader can check it.
///
/// `kBatchedToken` and `kVerify` are for the two BATCHED
/// entries, ForwardTokenBatch and ForwardVerificationBatch. They exist because
/// the kPrefill default stopped being a refusal: unit E gave kPrefill its own
/// GUFO_DEV_TURBO4_PREFILL admission arm, so an entry that simply inherited the
/// default became ADMITTED on a Turbo4 session. Both batched entries then went
/// on to launch f16 kernels against `d_kv_cache` / `d_attention_kv_f16`, which
/// are nullptr on a Turbo4 run — the confident garbage this whole guard exists
/// to prevent, reopened by a widening three units away from it.
///
/// Naming them separately is the fix. It is not a stricter kPrefill: a distinct
/// token means the entry can be found by grep, so the next phase that does earn
/// a Turbo4 route (E6.2 admits kVerify under the DECODE switch) is a
/// deliberate, visible edit to this one place rather than a silent
/// re-broadening of the default. Neither phase has an admission arm here — both
/// REFUSE on block storage until that ruling lands.
enum class QwenKvExecutionPhase : std::uint8_t {
  kPrefill,
  kDecode,
  kNoKvAccess,
  kBatchedToken,
  kVerify
};

/// Turbo4 decode DEV SWITCH. Default OFF.
///
/// Read from the environment on EVERY call, deliberately not cached in a
/// function-local static: a cached read is correct in production and makes the
/// switch untestable, which is how a dev switch outlives its dev window.
///
/// Any value other than exactly "1" is OFF. An empty string, "0" and "true" are
/// all refusals, because a switch that treats a typo as "on" fails open.
[[nodiscard]] inline bool QwenKvDevTurbo4DecodeEnabled() noexcept {
  const char* const value = std::getenv("GUFO_DEV_TURBO4_DECODE");
  return value != nullptr && std::string_view(value) == "1";
}

/// Turbo4 PREFILL DEV SWITCH. Default OFF, and SEPARATE from the decode
/// switch on purpose.
///
/// WHY A SECOND SWITCH RATHER THAN WIDENING THE FIRST.
/// `GUFO_DEV_TURBO4_DECODE=1` admits block storage for kDecode and kNoKvAccess
/// only; kPrefill still throws. That is not an oversight to be tidied away —
/// prefill reaches a DIFFERENT route: it writes a chunk of blocks with E1's
/// batched writer and then reads them back through the three PRODUCTION
/// attention kernels via the f16 scratch. Decode still uses the scalar block
/// reader. So the two phases have genuinely different kernels behind them, and
/// one switch would let a prefill claim a permission its route has not earned.
///
/// With only DECODE=1 the prompt still goes through the C1d token-feeding
/// fallback and the new prefill wiring never executes — an arm that cannot be
/// switched on is an arm that cannot be tested. Two switches keep BOTH arms
/// reachable: unset keeps C1d as the fallback, and each admits exactly its own
/// phase.
///
/// Same fail-closed rule as the decode switch: only the exact string "1" is ON,
/// so a typo refuses rather than admitting a route that was not asked for.
[[nodiscard]] inline bool QwenKvDevTurbo4PrefillEnabled() noexcept {
  const char* const value = std::getenv("GUFO_DEV_TURBO4_PREFILL");
  return value != nullptr && std::string_view(value) == "1";
}

/// Throws unless a KERNEL can consume a plane of this storage.
///
/// The wall behind the allocatable/executable split. Allocating a
/// Turbo4 plane is now legitimate — the member exists and the pitch is derived
/// from one authority — so an engine holding one would otherwise run its
/// attention against `d_attention_kv_f16 == nullptr` and produce confident
/// garbage. The message names the two missing things rather than "unsupported",
/// because a refusal that does not say what would make it work gets deleted by
/// the next reader who assumes it is stale.
///
/// `phase` defaults to kPrefill, so every pre-existing call site keeps the
/// refusal it has today, verbatim. Only entry points that have been classified
/// as decode pass kDecode explicitly.
inline void RequireExecutableKvStorage(
    QwenKvCacheStorage storage,
    QwenKvExecutionPhase phase = QwenKvExecutionPhase::kPrefill) {
  // DEV ONLY, and the ONLY behavioural difference when the switch is unset is
  // that this branch does not return. With the variable absent the guard falls
  // through to exactly the code below, unchanged.
  //
  // Scoped to block-addressed-but-KNOWN storage on purpose: an unknown
  // precision is not a Turbo4 policy, and it keeps its own distinct refusal.
  //
  // C1e: kNoKvAccess rides this SAME branch, not a looser one. An entry that
  // touches no KV plane has no attention to run, so the missing-kernel reason
  // does not apply to it — but it gets no new permission either, because the
  // switch is still the only thing that admits block-addressed storage here.
  // kVerify is added here: its execution uses the same
  // decode-equivalent kernels and is admitted only by the DECODE switch, never
  // PREFILL alone.
  if ((phase == QwenKvExecutionPhase::kDecode ||
       phase == QwenKvExecutionPhase::kNoKvAccess ||
       phase == QwenKvExecutionPhase::kVerify) &&
      QwenKvDevTurbo4DecodeEnabled() && QwenKvIsBlockAddressed(storage) &&
      QwenKvStorageIsKnown(storage)) {
    return;
  }
  // The prefill arm. Its OWN switch, for the reason given on
  // QwenKvDevTurbo4PrefillEnabled: prefill's route is the E1 batched write plus
  // the production attention kernels reading the f16 scratch, which is not the
  // route the decode switch authorises. Scoped to block-addressed-and-KNOWN for
  // the same reason as the decode arm above.
  if (phase == QwenKvExecutionPhase::kPrefill &&
      QwenKvDevTurbo4PrefillEnabled() && QwenKvIsBlockAddressed(storage) &&
      QwenKvStorageIsKnown(storage)) {
    return;
  }
  if (QwenKvIsExecutableByExecutor(storage))
    return;
  if (!QwenKvStorageIsKnown(storage)) {
    throw std::invalid_argument("Qwen KV storage is not a known precision");
  }
  // C1e: the refusal names the phase that reached the guard. The old body
  // blamed two missing kernels for every refusal, which was false for a no-KV
  // entry and useless for a prefill one — the reader who most needs to know
  // WHERE they landed was told the same sentence in all three cases. The rest
  // of the sentence is kept verbatim so anything matching the old text still
  // does.
  //
  // E6.0: the two batched phases name the ENTRY as well as the phase. The
  // guard cannot see a call site, so this is a fixed mapping asserted here and
  // kept honest by e6_batched_phase_refusal_test, which reads the phase out of
  // the source and would fail if an entry stopped naming its own. Naming the
  // symbol is what turns "somebody was refused" into a line the next reader can
  // go and look at. Note these two are refused by NO switch — saying "with
  // GUFO_DEV_TURBO4_DECODE unset" would blame a switch that never had the power
  // to admit them.
  const char* const refusing_phase =
      phase == QwenKvExecutionPhase::kPrefill
          ? "the prefill phase, with GUFO_DEV_TURBO4_PREFILL unset"
          : (phase == QwenKvExecutionPhase::kDecode
                 ? "the decode phase, with GUFO_DEV_TURBO4_DECODE unset"
                 : (phase == QwenKvExecutionPhase::kBatchedToken
                        ? "the batched-token phase (QwenGpuExecutor::"
                          "ForwardTokenBatch), which no dev switch admits: its "
                          "callees write scratch.attention.k/v through the "
                          "LaunchBatched* kernels and never address a block "
                          "plane"
                        : (phase == QwenKvExecutionPhase::kVerify
                               ? "the DSpark verification phase "
                                 "(QwenGpuExecutor::ForwardVerificationBatch), "
                                 "with GUFO_DEV_TURBO4_DECODE unset"
                               : "a no-KV-access entry, with "
                                 "GUFO_DEV_TURBO4_DECODE unset")));
  throw std::logic_error(
      std::string("Qwen cannot EXECUTE against a block-addressed (Turbo4) KV "
                  "plane at ") +
      refusing_phase +
      ": the executor allocates and serializes one, but no attention kernel "
      "takes a 66-byte block pitch, so the f16 and fp32 plane members stay "
      "null. Two things are missing: a KV write kernel that emits blocks, and "
      "an attention read kernel that decodes them.");
}

/// True when a meaningful bytes-per-ELEMENT exists for this storage. False for
/// Turbo4 (no per-value size) AND for an unknown value (no meaning at all).
[[nodiscard]] constexpr bool QwenKvHasElementBytes(
    QwenKvCacheStorage storage) noexcept {
  return QwenKvStorageIsKnown(storage) && !QwenKvIsBlockAddressed(storage);
}

/// Bytes per element. Returns 0 for block-addressable storage — a REFUSAL, not
/// a size. It previously fell through to `sizeof(float)` and answered 4 for
/// Turbo4, which is how a caller asking the one question with no answer got a
/// confident wrong number and shrank a 66-byte block to a 4-byte element.
/// Review: refuse; do not approximate.
/// Guard with QwenKvHasElementBytes before relying on a non-zero result.
[[nodiscard]] constexpr std::size_t QwenKvElementBytes(
    QwenKvCacheStorage storage) noexcept {
  // Both the block-addressed case and the UNKNOWN case refuse. The unknown one
  // used to fall through to sizeof(float) and answer 4.
  if (!QwenKvStorageIsKnown(storage))
    return 0U;
  if (QwenKvIsBlockAddressed(storage))
    return 0U;
  return storage == QwenKvCacheStorage::kFp16 ? sizeof(std::uint16_t)
                                              : sizeof(float);
}

/// Blocks required to hold `elements`. For element-addressable storage this is
/// the element count itself, so callers can keep one code path.
[[nodiscard]] constexpr std::size_t QwenKvAllocationUnits(
    QwenKvCacheStorage storage, std::size_t elements) noexcept {
  return QwenKvIsBlockAddressed(storage)
             ? CeilDiv(elements, kQwenTurbo4BlockValues)
             : elements;
}

/// Bytes per allocation unit.
[[nodiscard]] constexpr std::size_t QwenKvUnitBytes(
    QwenKvCacheStorage storage) noexcept {
  return QwenKvIsBlockAddressed(storage) ? QwenKvBlockBytes(storage)
                                         : QwenKvElementBytes(storage);
}

/// Total bytes for one plane of `elements` values.
///
/// Returns 0 rather than a wrapped small number if the multiply overflows, and
/// 0 if the geometry is not block-aligned for block-addressable storage. A
/// caller that ignores 0 gets a refusal, not a buffer.
[[nodiscard]] constexpr std::size_t QwenKvBytesForElements(
    QwenKvCacheStorage storage, std::size_t elements) noexcept {
  // A block must not straddle a head. Head alignment is checked by
  // QwenKvHeadDimIsBlockAligned against the head dimension, not inferable from
  // a flat element count: 16384 elements is 128 whole blocks ONLY when each
  // head's dimension is itself a whole number of blocks.
  std::size_t bytes = 0U;
  if (!QwenKvMulChecked(QwenKvAllocationUnits(storage, elements),
                        QwenKvUnitBytes(storage), &bytes)) {
    return 0U;
  }
  return bytes;
}

/// True when one head's dimension is a whole number of 128-value blocks.
///
/// Review: a flat element count can be perfectly block-aligned while
/// every individual head still straddles a block boundary, which would let one
/// block's shared norm serve values from two different heads. The flat check
/// cannot see that; this one can.
///
/// THE UD TARGET IS 4 KV HEADS x 256, NOT 8 x 128. The header gives
/// head_count_kv 4 and key_length 256, so kv_width is 1024 per plane and each
/// head spans TWO 128-value blocks. An earlier comment here claimed "8 KV
/// heads = 128 values per head = exactly one block" — that is the wrong
/// geometry, it happened to pass the same alignment test, and it would have
/// mis-sized every block offset for the real model.
[[nodiscard]] constexpr bool QwenKvHeadDimIsBlockAligned(
    std::size_t head_dim) noexcept {
  // Zero must be refused explicitly. `0 % 128 == 0` is true, so a zero-width
  // head would otherwise pass this check and claim to be a whole number of
  // blocks while addressing nothing at all.
  return head_dim != 0U && head_dim % kQwenTurbo4BlockValues == 0U;
}

/// True when every head in a plane of `head_dim`-wide heads fits whole blocks.
[[nodiscard]] constexpr bool QwenKvHeadsAreBlockAligned(
    std::size_t head_dim, std::size_t head_count) noexcept {
  if (head_count == 0U)
    return false;
  return QwenKvHeadDimIsBlockAligned(head_dim) &&
         QwenKvHeadDimIsBlockAligned(head_dim * head_count);
}

/// Padding values a partial block consumes. Zero for element-addressable
/// storage. A caller that cannot mask padded slots must REFUSE a context whose
/// tail is non-zero rather than write into them.
[[nodiscard]] constexpr std::size_t QwenKvTailElements(
    QwenKvCacheStorage storage, std::size_t elements) noexcept {
  return QwenKvIsBlockAddressed(storage)
             ? QwenKvAllocationUnits(storage, elements) *
                       kQwenTurbo4BlockValues -
                   elements
             : 0U;
}

/// True when no padding is needed. For the UD-Q4 geometry
/// (attention_layers x kv_width = 16384 elements per token per plane) this
/// holds at every context, because 16384 is 128 whole blocks.
[[nodiscard]] constexpr bool QwenKvIsExactlyBlockAligned(
    QwenKvCacheStorage storage, std::size_t elements) noexcept {
  return QwenKvTailElements(storage, elements) == 0U;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KV_ALLOCATION_HPP_
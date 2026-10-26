#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace gufo::hip {

// Dev switch GUFO_DEV_NO_EXTRA_SYNC=1: skip host waits whose only
// consumer is later work on the same stream (stream order already holds).
[[nodiscard]] inline bool GufoDevSkipExtraSync() noexcept {
  static const bool skip = [] {
    const char* v = std::getenv("GUFO_DEV_NO_EXTRA_SYNC");
    return v != nullptr && v[0] == '1';
  }();
  return skip;
}

// SUPERSEDED by upstream #650 (merged 2026-10-20): the copy-free
// rollback is now unconditional, so nothing reads this switch any more. Kept
// so served env files that still set it stay valid.
// Dev switch GUFO_DEV_SSM_NO_SNAPSHOT=1: do not copy the recurrent state
// before a speculative verify round. The verify kernels already load the state
// once and store it once, so they can load from the live buffer and store into
// the snapshot buffer instead of in place. The round-start state then survives
// in the live buffer at zero cost, a rejection restores by keeping it, and a
// commit is a pointer swap. Removes 2 x 151.5 MiB of device-to-device copy per
// round (Qwen3.8-27B: 48 recurrent layers) and adds no memory traffic.
[[nodiscard]] inline bool GufoDevSsmNoSnapshot() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_SSM_NO_SNAPSHOT");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

// Host-side waste in the speculative round ("1% stack"). Three switches
// rather than one because the risk differs: (1) only drops a host memcpy,
// (2) moves a device->host->device feature round trip onto the device and so
// touches the draft's pending-feature bookkeeping, (3) changes which buffer the
// frontier is sampled from. All default OFF. None may alter greedy output.

// GUFO_DEV_HOSTOPS=1: stop copying buffers the producer still owns. The
// verification feature rows travel as a span into the target's own host buffer
// instead of being assigned into VerificationChunkResult, and a retained
// frontier row lands directly in its destination vector.
[[nodiscard]] inline bool GufoDevHostops() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_HOSTOPS");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

// GUFO_DEV_HOSTOPS_FEAT=1: hand the target's captured feature rows to the
// drafter on the device. Removes one D2H and one H2D of rows x feature_width
// floats per round, both out of unpinned host memory.
[[nodiscard]] inline bool GufoDevHostopsFeatures() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_HOSTOPS_FEAT");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

// GUFO_DEV_DFLASH_INJECT_WMMA: DFlash prompt injection runs its
// quantized projections through the int8 WMMA prefill GEMM, one launch per
// injection chunk instead of one per 8 rows. 1 = attn_k/attn_v only (normed
// input); 2 = also the fc projection, whose input is RAW target hidden states
// (outlier channels can flatten an int8 block: check acceptance). Needs
// GUFO_DEV_DFLASH_KEEP_QUANT=1, else the weights are BF16 and nothing changes.
[[nodiscard]] inline int GufoDevDFlashInjectWmma() noexcept {
  static const int level = [] {
    const char* v = std::getenv("GUFO_DEV_DFLASH_INJECT_WMMA");
    return v != nullptr && (v[0] == '1' || v[0] == '2') ? v[0] - '0' : 0;
  }();
  return level;
}

// GUFO_DEV_HOSTOPS_FRONTIER=1: keep the retained frontier as a row index into
// the verification logits instead of a host copy, so a sampled request samples
// that row in place rather than copying a vocabulary down and back up.
[[nodiscard]] inline bool GufoDevHostopsFrontier() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_HOSTOPS_FRONTIER");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

// GUFO_DEV_V_SKIP=<threshold>: in the split-K decode attention reader a key
// whose online-softmax weight exp(score - running_max) is below the threshold
// contributes less than the threshold to the normalised output. The running
// maximum only grows, so the weight only falls, and the key cannot recover.
// Set to a positive value the reader skips dequantising that key's V row and
// its V multiply-add. Unset or 0 returns exactly the pre-switch path.
[[nodiscard]] inline float GufoDevVSkipThreshold() noexcept {
  static const float threshold = [] {
    const char* v = std::getenv("GUFO_DEV_V_SKIP");
    if (v == nullptr)
      return 0.0f;
    // A weight relative to the running max is in (0, 1]; a threshold outside
    // (0, 1) (or NaN) would drop every key, so it means off (review).
    const float t = std::strtof(v, nullptr);
    return t > 0.0f && t < 1.0f ? t : 0.0f;
  }();
  return threshold;
}

// GUFO_DEV_VERIFY_WMMA_WIDE=1: let the WMMA verify reader serve the
// Turbo5/Turbo6 qh / padded-slot formats as well as Turbo4. That reader
// dequantizes each K/V block once per (kv head, split) and reuses it for every
// draft row; the scalar split reader it replaces re-reads the whole K+V plane
// once per row (R x the traffic). Read once on the host like
// GufoDevVSkipThreshold. Unset, Turbo5/6 keep falling back to the split reader
// with the one-time banner, so Turbo4 output is unchanged.
[[nodiscard]] inline bool GufoDevVerifyWmmaWide() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_VERIFY_WMMA_WIDE");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}
// GUFO_DEV_IQ3_PACKED=1: let IQ3_S / IQ3_XXS FFN gate+up run as one
// packed 34816-row projection, as Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0 already do. EF-apex
// stores every layer's gate and up as adjacent same-type IQ3 tensors, so the
// kTable's m34816 IQ3_S B4/B6 rows (measured on EF-apex gate/up) become
// reachable under GUFO_DEV_KQUANT_V3=1 (the m34816 B1 rows stay unreachable:
// packing needs batch >= 2); other widths run iq_fast/dot4 once at m34816
// instead of twice at 17408. Unset, IQ3 keeps two 17408-row projections. Changes the kernel picked, so
// greedy text may differ from unset.
[[nodiscard]] inline bool GufoDevIq3Packed() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_IQ3_PACKED");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}

// GUFO_DEV_WSTRAG: straggler kernels of the served round that sit far below the
// DRAM band for a reason that is not bandwidth. Every level produces the SAME
// values as the default path, bit for bit; only the grid, the memory walk or
// the register/scratch placement changes. Unset (0), every site keeps the
// default instantiation.
//   1  DFlash selector: top-k length fixed at compile time, so the per-thread
//      candidate lists stay in VGPRs instead of scratch (the dynamic
//      `insertion` index is what pushes them to scratch today).
//   2  as 1, plus the 1952->16 reduction moved out of the single-workgroup
//      finalize into its own kernel, one workgroup per draft row. Same
//      thread->candidate partition and same merge tree, so the reduced list is
//      identical; it just stops being redone serially inside the one-WG kernel
//      once per draft step.
// The two levels are separate so an A/B says WHICH cause was real.
[[nodiscard]] inline unsigned GufoDevWstragLevel() noexcept {
  static const unsigned level = [] {
    const char* v = std::getenv("GUFO_DEV_WSTRAG");
    if (v == nullptr || v[1] != '\0') return 0U;
    if (v[0] == '1') return 1U;
    if (v[0] == '2') return 2U;
    return 0U;
  }();
  return level;
}
[[nodiscard]] inline bool GufoDevWstrag() noexcept {
  return GufoDevWstragLevel() != 0U;
}

// GUFO_DEV_BATCH_INVARIANT=1 (read once, default OFF): make a verify row's
// logits independent of the verify width. Every width-keyed CHOICE on the
// verify path -- which kernel family, which tuned geometry, which attention
// split plan -- is resolved as if the width were kBatchInvariantWidth, then the
// call runs at its real width. The per-row arithmetic of these kernels does not
// depend on the width itself (the Batch loop is over independent accumulators);
// it depends on the geometry the width SELECTS, which is what this pins.
//
// Why it matters: --draft-tokens 6, lookup_route, the drafter bf16 attention
// and the drafter vocab cut only change WHICH rows and widths the verify sees.
// With an invariant verify they stop changing greedy text.
//
// B8 is unchanged by construction (8 is the width everything is pinned to), so
// the served round costs the same; widths 1-7 may move either way.
[[nodiscard]] inline bool GufoDevBatchInvariant() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_BATCH_INVARIANT");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}

// The width every verify-path choice is resolved at. DFlash2's block is 8 rows
// (drafter proposes 7), so 8 is the widest verify this engine ever launches and
// the only width whose speed is on the critical path.
inline constexpr std::size_t kBatchInvariantWidth = 8;

/// The width to RESOLVE a verify-path choice at: the real width normally, and
/// kBatchInvariantWidth for every 1..8 verify once the switch is on. Widths
/// above 8 are cross-request prefill-style batches, not verify rows, and keep
/// their own routing.
[[nodiscard]] inline std::size_t BatchInvariantSelect(
    std::size_t batch) noexcept {
  return (batch >= 1 && batch <= kBatchInvariantWidth &&
          GufoDevBatchInvariant())
             ? kBatchInvariantWidth
             : batch;
}

/// Context length the verify ATTENTION split plan is laid out for. The split
/// SIZE is what moves the key -> split boundaries, and it is derived from the
/// last row's context, so it changes with the verify width. Pinning it to the
/// width-8 last row (clamped to the window) keeps every row's keys in the same
/// split, with the same tile order, at every width. The split COUNT still
/// follows the real last row, so no split lands entirely past the last row;
/// splits that hold only masked keys already write an empty (m = -inf, l = 0)
/// record, which the sequential merge adds as an exact zero.
[[nodiscard]] inline std::uint32_t BatchInvariantPlanContext(
    std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context) noexcept {
  const std::uint32_t last = first_context + rows - 1U;
  if (rows == 0U || rows > kBatchInvariantWidth || !GufoDevBatchInvariant())
    return last;
  const std::uint32_t pinned =
      first_context + static_cast<std::uint32_t>(kBatchInvariantWidth) - 1U;
  return pinned > max_context ? max_context : pinned;
}

// IQ3 X: balanced persistent tiles for packed B7/B8 gate+up only. The
// launchers check the shape; every other value retains the existing kernels.
// Its Balanced instantiation static_asserts Batch == 7 || Batch == 8, so it
// cannot be pinned for widths 1-6: batch-invariant mode turns it off instead
// (it is not in the served env, so nothing served loses a measured gain).
[[nodiscard]] inline bool GufoDevIq3X() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_IQ3_X");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on && !GufoDevBatchInvariant();
}

// GUFO_DEV_VERIFY_T6_PIPE=0: keep the K = V = Turbo6 WMMA verify reader
// on its earlier instance (prefetch that waits on itself, checked per-load
// addressing). Default (unset) takes the pipelined instance, whose output is
// bitwise identical; the switch exists for the A/B and the bitwise compare.
// Only read in -DGUFO_T4_PREFETCH=1 builds.
[[nodiscard]] inline bool GufoDevVerifyT6Pipe() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_VERIFY_T6_PIPE");
    return v == nullptr || v[0] != '0';
  }();
  return on;
}

// ATTN6 GUFO_DEV_ATTN6_V3=1: the K = V = Turbo6 verify reader takes
// Turbo4VerifyAttn6Kernel (Q in registers, unchunked K/V tiles, 5 barriers per
// tile, packed code decode). Partials are bitwise identical to the pipelined
// WMMA reader while GUFO_DEV_ATTN6_SPLITS is unset. Needs a
// -DGUFO_T4_PREFETCH=1 build like the pipelined reader it replaces.
[[nodiscard]] inline bool GufoDevAttn6V3() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_ATTN6_V3");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}
// ATTN FUSE GUFO_DEV_ATTN_FUSE=1 (with GUFO_DEV_ATTN6_V3=1): attn6's rotate,
// reader and reduce as one launch (Turbo4VerifyAttn6FuseKernel); partials,
// out_rot, out and status bitwise equal to the three-launch attn6.
[[nodiscard]] inline bool GufoDevAttnFuse() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_ATTN_FUSE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}
// ATTN B GUFO_DEV_ATTN_B=1 (with GUFO_DEV_ATTN6_V3=1): the fused kernel plus
// design B step 2a (live-row softmax, PV fragment reuse), bitwise equal too.
[[nodiscard]] inline bool GufoDevAttnB() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_ATTN_B");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}
// ATTN6 GUFO_DEV_ATTN6_SPLITS=N: at most N splits per KV head (Attn6SplitSize).
// Unset / 0 = the WMMA reader's split plan. A different plan changes the
// cross-split merge order, so output is then close, not bitwise.
[[nodiscard]] inline std::uint32_t GufoDevAttn6Splits() noexcept {
  static const std::uint32_t n = [] {
    const char* v = std::getenv("GUFO_DEV_ATTN6_SPLITS");
    return v == nullptr ? 0U
                        : static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
  }();
  return n;
}
// ATTN_C GUFO_DEV_ATTN_C=1 (with GUFO_DEV_ATTN6_V3=1): the three bit-exact
// cuts of attnC inside the attn6 verify reader and the fused / ATTN_B
// kernels.
//   L1 a 32x interleaved centroid table, so lane l always reads LDS bank l
//      whatever its code is (7.95 -> 4.01 cycles per lookup per SIMD, P3).
//      This one is attnC's own and stacks with GUFO_DEV_ATTN_B.
//   L2 the PV fragments hoisted: 48 -> 20 ds_load_b128 and 24 -> 12
//      ds_load_2addr_b32 per PV wave per tile.
//   L3 the online softmax gated to the live rows instead of all 48, with the
//      padding rows' P / row_scale pre-filled (0 at rows 8, ~12% at rows 1).
// L2 and L3 are the same two changes GUFO_DEV_ATTN_B makes, found
// independently; inside the fused kernel ATTN_C reuses ATTN_B's code for them
// rather than duplicating it, so ATTN_C adds only L1 on top of ATTN_B.
// Addresses, load scheduling and dropped padding work only: no float
// operation, operand or per-accumulator order changes, and no padding row
// reaches a partial record, so partials and out stay bitwise equal to attn6
// with the same split plan.
[[nodiscard]] inline bool GufoDevAttnC() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_ATTN_C");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}
}  // namespace gufo::hip

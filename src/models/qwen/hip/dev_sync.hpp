#pragma once
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

// SUPERSEDED by upstream #650 (tubo-up merge 2026-10-20): the copy-free
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

// IQ3 X: balanced persistent tiles for packed B7/B8 gate+up only. The
// launchers check the shape; every other value retains the existing kernels.
[[nodiscard]] inline bool GufoDevIq3X() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_IQ3_X");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
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
}  // namespace gufo::hip

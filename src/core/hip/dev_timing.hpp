#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <time.h>
#include <utility>
#include <vector>

namespace gufo::core::hip::dev_timing {

enum class Phase : std::size_t {
  kInject,
  kDraftForward,
  kMarkovConfidence,
  kVerify,
  // gaps: the target's recurrent replay after a rejection. It is issued on the
  // target's arena stream and nothing waits on it before the next verify, so
  // its span is expected to hide under draft_forward + markov_confidence on
  // the draft stream. Measuring it is the only way to tell hidden from
  // exposed: round wall - verify - draft_forward - markov_confidence left
  // ~4.2 ms unattributed and this was the largest candidate inside it.
  kSsmReplay,
  // Qwen prefill. Same GUFO_DEV_DSPARK_TIMING switch. Attn/Dequant
  // are bucketed by chunk start_pos: <8192, <16384, <24576, >=24576.
  kPrefillChunk,
  kPrefillGdn,
  kPrefillFfn,
  kPrefillKvWrite,
  kPrefillDequantB0,
  kPrefillDequantB1,
  kPrefillDequantB2,
  kPrefillDequantB3,
  kPrefillAttnB0,
  kPrefillAttnB1,
  kPrefillAttnB2,
  kPrefillAttnB3,
  // gdn4: prefill_gdn split. In = qkv/gate/alpha/beta projections, Rec = conv
  // + preps + recurrence + post-norm-gate, Out = ssm_out. Scan = the
  // recurrence kernel alone (token-serial or chunked), nested inside Rec.
  kPrefillGdnIn,
  kPrefillGdnRec,
  kPrefillGdnOut,
  kPrefillGdnScan,
};

inline constexpr std::size_t kPhaseCount =
    static_cast<std::size_t>(Phase::kPrefillGdnScan) + 1;

/// gaps: HOST wall windows, not GPU stream spans.
///
/// Every `Phase` above is a pair of hipEvents, so it measures time on one
/// stream and is blind to host CPU time and to work on the other stream. On
/// the served NEWNR stack that left a hole: round wall 123.2 ms minus verify
/// 100.8 minus draft_forward 12.3 minus markov_confidence 5.9 = ~4.2 ms with
/// no owner, and `inject`'s event total could not fill it because its 225
/// calls include the prefill injections. These scopes time the host windows
/// with a plain steady_clock, so the residual gets a name instead of a guess.
enum class HostSpan : std::size_t {
  kRound,       ///< one whole VerifyStep, host wall. The denominator.
  kPropose,     ///< drafter proposal: inject + draft_forward + markov + cut.
  kVerifyCall,  ///< ForwardVerificationChunk, including its closing sync.
  kPostVerify,  ///< acceptance, frontier, rollback/replay issue, feedback.
};

inline constexpr std::size_t kHostSpanCount =
    static_cast<std::size_t>(HostSpan::kPostVerify) + 1;

inline constexpr std::array<std::string_view, kHostSpanCount> kHostSpanNames{
    "round", "propose", "verify_call", "post_verify"};

inline constexpr std::array<std::string_view, kPhaseCount> kPhaseNames{
    "inject",
    "draft_forward",
    "markov_confidence",
    "verify",
    "ssm_replay",
    "prefill_chunk",
    "prefill_gdn",
    "prefill_ffn",
    "prefill_kv_write",
    "prefill_dequant_p0_8k",
    "prefill_dequant_p8k_16k",
    "prefill_dequant_p16k_24k",
    "prefill_dequant_p24k_up",
    "prefill_attn_p0_8k",
    "prefill_attn_p8k_16k",
    "prefill_attn_p16k_24k",
    "prefill_attn_p24k_up",
    "prefill_gdn_in",
    "prefill_gdn_rec",
    "prefill_gdn_out",
    "prefill_gdn_scan"};

/// `base` must be kPrefillDequantB0 or kPrefillAttnB0.
[[nodiscard]] inline Phase PrefillDepthPhase(Phase base,
                                             std::uint32_t start_pos) noexcept {
  return static_cast<Phase>(static_cast<std::size_t>(base) +
                            std::min<std::uint32_t>(start_pos >> 13, 3));
}

[[nodiscard]] inline int Mode() noexcept {
  static const int mode = [] {
    const char* value = std::getenv("GUFO_DEV_DSPARK_TIMING");
    if (value == nullptr || value[1] != '\0') return 0;
    if (value[0] == '1') return 1;
    if (value[0] == '2') return 2;
    return 0;
  }();
  return mode;
}

[[nodiscard]] inline bool Enabled() noexcept { return Mode() != 0; }
[[nodiscard]] inline bool Detailed() noexcept { return Mode() == 2; }

[[nodiscard]] inline std::uint64_t RealtimeNs() noexcept {
  timespec now{};
  if (::clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
  return static_cast<std::uint64_t>(now.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

class Recorder final {
 public:
  struct Events {
    hipEvent_t start{};
    hipEvent_t stop{};
    std::uint64_t start_realtime_ns{};
  };

  static Recorder& Instance() noexcept {
    static Recorder recorder;
    static const bool registered = [] {
      (void)std::atexit([] { Recorder::Instance().Report(); });
      return true;
    }();
    (void)registered;
    return recorder;
  }

  void BeginWall() noexcept {
    wall_start_ = std::chrono::steady_clock::now();
    wall_started_ = true;
    Stamp("start");
  }

  void Stamp(const char* phase) const noexcept {
    if (!wall_started_) return;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - wall_start_)
                                  .count();
    std::fprintf(stderr,
                 "GUFO_DEV_DSPARK_TIMING wall=%s elapsed_ms=%.3f\n",
                 phase, elapsed_ms);
  }

  void AddHost(HostSpan span, double elapsed_ms) noexcept {
    const auto index = static_cast<std::size_t>(span);
    if (index >= kHostSpanCount) return;
    std::lock_guard lock(mutex_);
    host_total_ms_[index] += elapsed_ms;
    ++host_calls_[index];
  }

  void Add(Phase phase, Events events) noexcept {
    try {
      std::lock_guard lock(mutex_);
      events_[static_cast<std::size_t>(phase)].push_back(events);
    } catch (...) {
      (void)hipEventDestroy(events.start);
      (void)hipEventDestroy(events.stop);
      std::fprintf(stderr,
                   "GUFO_DEV_DSPARK_TIMING error=event_accumulator\n");
    }
  }

  void Report() noexcept {
    Stamp("exit");
    std::array<std::vector<Events>, kPhaseCount> events;
    std::array<double, kHostSpanCount> host_total_ms{};
    std::array<std::size_t, kHostSpanCount> host_calls{};
    {
      std::lock_guard lock(mutex_);
      if (reported_) return;
      reported_ = true;
      events.swap(events_);
      host_total_ms = host_total_ms_;
      host_calls = host_calls_;
    }
    for (std::size_t span = 0; span < kHostSpanCount; ++span) {
      if (host_calls[span] == 0) continue;
      std::fprintf(stderr,
                   "GUFO_DEV_DSPARK_TIMING host=%.*s total_ms=%.3f "
                   "calls=%zu ms_per_call=%.3f\n",
                   static_cast<int>(kHostSpanNames[span].size()),
                   kHostSpanNames[span].data(), host_total_ms[span],
                   host_calls[span],
                   host_total_ms[span] /
                       static_cast<double>(host_calls[span]));
    }

    const auto& names = kPhaseNames;
    struct ScopeSample {
      std::size_t phase;
      std::uint64_t start_realtime_ns;
      float elapsed_ms;
    };
    std::vector<ScopeSample> scope_samples;
    for (std::size_t phase = 0; phase < events.size(); ++phase) {
      double total_ms = 0.0;
      std::size_t calls = 0;
      for (const auto& pair : events[phase]) {
        float elapsed_ms = 0.0F;
        if (hipEventSynchronize(pair.stop) == hipSuccess &&
            hipEventElapsedTime(&elapsed_ms, pair.start, pair.stop) ==
                hipSuccess) {
          total_ms += elapsed_ms;
          ++calls;
          if (Detailed() && pair.start_realtime_ns != 0) {
            scope_samples.push_back(
                {phase, pair.start_realtime_ns, elapsed_ms});
          }
        }
        (void)hipEventDestroy(pair.start);
        (void)hipEventDestroy(pair.stop);
      }
      if (calls != 0) {
        std::fprintf(stderr,
                     "GUFO_DEV_DSPARK_TIMING phase=%.*s total_ms=%.3f "
                     "calls=%zu ms_per_call=%.3f\n",
                     static_cast<int>(names[phase].size()), names[phase].data(),
                     total_ms, calls, total_ms / static_cast<double>(calls));
      }
    }
    if (Detailed()) {
      std::sort(scope_samples.begin(), scope_samples.end(),
                [](const auto& a, const auto& b) {
                  return a.start_realtime_ns < b.start_realtime_ns;
                });
      for (const auto& scope : scope_samples) {
        std::fprintf(stderr,
                     "GUFO_DEV_DSPARK_TIMING scope phase=%.*s "
                     "start_unix_ns=%llu ms=%.3f\n",
                     static_cast<int>(names[scope.phase].size()),
                     names[scope.phase].data(),
                     static_cast<unsigned long long>(scope.start_realtime_ns),
                     scope.elapsed_ms);
      }
    }
  }

 private:
  std::array<std::vector<Events>, kPhaseCount> events_{};
  std::array<double, kHostSpanCount> host_total_ms_{};
  std::array<std::size_t, kHostSpanCount> host_calls_{};
  std::mutex mutex_;
  std::chrono::steady_clock::time_point wall_start_{};
  bool wall_started_{false};
  bool reported_{false};
};

class EventScope final {
 public:
  EventScope(Phase phase, hipStream_t stream) noexcept
      : phase_(phase), stream_(stream) {
    if (!Enabled()) return;
    if (Detailed()) events_.start_realtime_ns = RealtimeNs();
    if (hipEventCreate(&events_.start) != hipSuccess ||
        hipEventCreate(&events_.stop) != hipSuccess ||
        hipEventRecord(events_.start, stream_) != hipSuccess) {
      if (events_.start != nullptr) (void)hipEventDestroy(events_.start);
      if (events_.stop != nullptr) (void)hipEventDestroy(events_.stop);
      events_ = {};
      std::fprintf(stderr, "GUFO_DEV_DSPARK_TIMING error=event_start\n");
      return;
    }
    active_ = true;
  }

  ~EventScope() { Stop(); }

  EventScope(const EventScope&) = delete;
  EventScope& operator=(const EventScope&) = delete;

  void Stop() noexcept {
    if (!active_) return;
    if (hipEventRecord(events_.stop, stream_) != hipSuccess) {
      (void)hipEventDestroy(events_.start);
      (void)hipEventDestroy(events_.stop);
      events_ = {};
      active_ = false;
      std::fprintf(stderr, "GUFO_DEV_DSPARK_TIMING error=event_stop\n");
      return;
    }
    Recorder::Instance().Add(phase_, events_);
    events_ = {};
    active_ = false;
  }

 private:
  Phase phase_;
  hipStream_t stream_;
  Recorder::Events events_{};
  bool active_{false};
};

/// gaps: times one host window. No hip call, no stream, no sync: with the
/// switch off it is two branches and nothing else, so it is safe on the
/// served path.
class HostScope final {
 public:
  explicit HostScope(HostSpan span) noexcept : span_(span) {
    if (!Enabled()) return;
    begin_ = std::chrono::steady_clock::now();
    active_ = true;
  }

  ~HostScope() { Stop(); }

  HostScope(const HostScope&) = delete;
  HostScope& operator=(const HostScope&) = delete;

  void Stop() noexcept {
    if (!active_) return;
    active_ = false;
    Recorder::Instance().AddHost(
        span_, std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - begin_)
                   .count());
  }

 private:
  HostSpan span_;
  std::chrono::steady_clock::time_point begin_{};
  bool active_{false};
};

inline void BeginWall() noexcept {
  if (Enabled()) Recorder::Instance().BeginWall();
}

inline void WallStamp(const char* phase) noexcept {
  if (Enabled()) Recorder::Instance().Stamp(phase);
}

}  // namespace gufo::core::hip::dev_timing

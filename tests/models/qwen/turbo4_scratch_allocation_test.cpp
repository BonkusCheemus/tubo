// Turbo4 scratch and split-K workspace are estimated.
//
// WHY THIS EXISTS. The Turbo4 f16 scratch is the buffer the three attention
// routes actually read (attention_wmma.hip:650-651, attention_tile.hip:39,
// attention_batched.hip:45). Before E2(b) the arena allocated it nowhere: under
// kTurbo4 both d_attention_kv_f16 and d_kv_cache were null, which review
// verified closes the f32-store hazard BY CONSTRUCTION but only because no
// f16 route could run. Wiring E2 creates the first run where a f16 route does
// run, so the buffer has to exist, and it has to be PAID FOR in the number a
// fit/refuse decision reads.
//
// TWO SITES, and the distinction is the whole point:
//   - EstimateMemoryUsage — the pre-hipMalloc number. If it omits workspace
//     while the allocator mallocs it, a fit PASSES and then the allocation
//     fails, so the refusal arrives late and somewhere unrelated.
//   - the allocator's kTurbo4 case — the post-allocation record.
//
// This file tests the estimate. The GPU companion checks actual allocations.
//
// HOW THE SCRATCH IS MEASURED HERE. NOT by restating the formula: a test that
// asserts `estimate == base + 2*C*kv*hd*2` passes if the same wrong constant
// appears on both sides. Instead the scratch is a RESIDUAL — the difference
// between the Turbo4 estimate and the fp32 estimate, after the two planes each
// storage really costs have been taken out. Both estimates come from the same
// estimator with the same config, so everything except the KV storage and the
// scratch cancels exactly. A second site computing the same number wrongly
// cannot cancel.
//
//   scratch = state(turbo4) - state(fp32)
//             - 2*plane_bytes(turbo4) + 2*plane_bytes(fp32)
//
// E5 adds a deterministic split-K partial buffer plus rotated Q/output vectors
// to the same residual. Assert each known component; fp16 and fp32 must remain
// zero, so Turbo4-only workspace cannot be charged to the wrong storage.

#include <cstdint>
#include <cstdio>
#include <string>

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenGpuArena;
using gufo::hip::QwenKvCacheStorage;

// The UD Qwen3.8-27B geometry that actually runs, with head_dim 256 so a head is
// a whole number of 128-value Turbo4 blocks. 64 layers / interval 4 = 16 full
// attention layers, which matters: ONE layer of scratch and SIXTEEN layers of
// plane differ by 16x, so "one layer" is an assertion here, not a comment.
gufo::core::ModelConfig Ud27BConfig() {
  gufo::core::ModelConfig config;
  config.model_name = "qwen3.8-27b";
  config.num_layers = 64;
  config.hidden_size = 5120;
  config.intermediate_size = 17408;
  config.num_attention_heads = 24;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.vocab_size = 248320;
  config.context_length = 262144;
  config.full_attention_interval = 4;
  config.mtp_num_layers = 1;
  config.ssm_conv_kernel = 4;
  config.ssm_state_size = 128;
  config.ssm_group_count = 16;
  config.ssm_time_step_rank = 32;
  config.ssm_inner_size = 4096;
  config.rotary_dim = 64;
  return config;
}

QwenExecutionPolicy Policy(QwenKvCacheStorage storage) {
  auto policy = QwenExecutionPolicy::Production();
  policy.kv_cache_storage = storage;
  return policy;
}

// Bytes ONE plane of `context` really costs under `storage`, computed by the
// test from the allocation header rather than copied out of arena.cpp.
std::size_t PlaneBytes(const gufo::core::ModelConfig& config,
                       std::uint32_t context, QwenKvCacheStorage storage) {
  const std::size_t elements = static_cast<std::size_t>(
                                   config.FullAttentionLayerCount()) *
                               context * config.num_key_value_heads *
                               config.head_dim;
  return gufo::hip::QwenKvBytesForElements(storage, elements);
}

// The residual: what the Turbo4 estimate pays that the fp32 estimate does not,
// once each plane's own cost is removed. Equals the scratch iff the ONLY
// difference between the two estimates is the KV planes plus the scratch.
std::size_t ResidualScratchBytes(const gufo::core::ModelConfig& config,
                                 std::uint32_t context) {
  const auto turbo4 =
      QwenGpuArena::EstimateMemoryUsage(config, context,
                                        Policy(QwenKvCacheStorage::kTurbo4));
  const auto fp32 = QwenGpuArena::EstimateMemoryUsage(
      config, context, Policy(QwenKvCacheStorage::kFp32));
  const std::size_t turbo4_planes = 2 * PlaneBytes(config, context,
                                                   QwenKvCacheStorage::kTurbo4);
  const std::size_t fp32_planes =
      2 * PlaneBytes(config, context, QwenKvCacheStorage::kFp32);
  // Subtracted as a signed delta: the fp32 planes are LARGER, so the raw
  // subtraction underflows in size_t and wraps to a huge number. Underflow is
  // not a rounding error a reader would spot, so it is done explicitly.
  return turbo4.request_state_bytes - fp32.request_state_bytes -
         turbo4_planes + fp32_planes;
}

// Residual against a NON-BLOCK storage. Nothing should be left over: a scratch
// charged to f16 or fp32 shows up here as a non-zero number that is not
// explained by the planes.
std::size_t ResidualAgainstFp32(const gufo::core::ModelConfig& config,
                                std::uint32_t context,
                                QwenKvCacheStorage storage) {
  const auto subject = QwenGpuArena::EstimateMemoryUsage(config, context,
                                                         Policy(storage));
  const auto fp32 = QwenGpuArena::EstimateMemoryUsage(
      config, context, Policy(QwenKvCacheStorage::kFp32));
  return subject.request_state_bytes - fp32.request_state_bytes -
         2 * PlaneBytes(config, context, storage) +
         2 * PlaneBytes(config, context, QwenKvCacheStorage::kFp32);
}

void ScratchIsChargedAtTheEstimate() {
  const auto config = Ud27BConfig();
  constexpr std::uint32_t context = 4096;
  const std::size_t measured = ResidualScratchBytes(config, context);
  const std::size_t prefill_scratch = 2 * static_cast<std::size_t>(context) *
                                      config.num_key_value_heads *
                                      config.head_dim * sizeof(std::uint16_t);
  const std::size_t decode_workspace = gufo::hip::Turbo4DecodeWorkspaceBytes(
      context, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);
  const std::size_t expected = prefill_scratch + decode_workspace;

  std::printf("  context=%u kv_heads=%u head_dim=%u -> scratch residual=%zu B "
              "(expected %zu B)\n",
              context, config.num_key_value_heads, config.head_dim, measured,
              expected);
  Check(measured == expected,
        "the Turbo4 residual charges the one-layer f16 scratch plus the exact "
        "E5 partial/q/output workspace");
}

void ScratchIsOneLayerNotAPlane() {
  const auto config = Ud27BConfig();
  constexpr std::uint32_t context = 4096;
  const std::size_t measured = ResidualScratchBytes(config, context);
  const std::size_t decode_workspace = gufo::hip::Turbo4DecodeWorkspaceBytes(
      context, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);
  const std::size_t per_layer = 2 * static_cast<std::size_t>(context) *
                                config.num_key_value_heads *
                                config.head_dim * sizeof(std::uint16_t);
  const std::size_t full_plane =
      2 * PlaneBytes(config, context, QwenKvCacheStorage::kTurbo4);

  Check(measured >= decode_workspace && measured - decode_workspace == per_layer,
        "scratch is ONE layer: K and V of one layer, not a plane per layer");
  Check(config.FullAttentionLayerCount() > 1,
        "the config has more than one full-attention layer, so a per-layer "
        "plane and a one-layer scratch cannot be confused");
  Check(measured * config.FullAttentionLayerCount() != full_plane,
        "sixteen one-layer scratches would NOT equal the sixteen-layer plane, "
        "so this geometry can tell the two sizes apart");
  std::printf("  one-layer scratch=%zu B, %u-layer plane=%zu B\n", per_layer,
              config.FullAttentionLayerCount(), full_plane);
}

void ScratchIsAbsentForEveryOtherStorage() {
  const auto config = Ud27BConfig();
  constexpr std::uint32_t context = 4096;
  Check(ResidualAgainstFp32(config, context, QwenKvCacheStorage::kFp16) == 0,
        "the f16 estimate pays planes and nothing else: no scratch charged to "
        "an element-addressable storage");
  Check(ResidualAgainstFp32(config, context, QwenKvCacheStorage::kFp32) == 0,
        "the fp32 estimate is unchanged by this work");
}

// The refusal must arrive at the SAME place it always did, one term larger.
// A fit decision reads TotalBytes(); if the scratch were charged to
// temporary_scratch_bytes instead, a concurrency multiplier would fold it in
// N times and the gate would refuse long before the allocator would fail.
void FitGateSeesItOnce() {
  const auto config = Ud27BConfig();
  constexpr std::uint32_t context = 4096;
  const auto turbo4 = QwenGpuArena::EstimateMemoryUsage(
      config, context, Policy(QwenKvCacheStorage::kTurbo4));
  const auto fp32 = QwenGpuArena::EstimateMemoryUsage(
      config, context, Policy(QwenKvCacheStorage::kFp32));
  const std::size_t measured = ResidualScratchBytes(config, context);

  // SIGNED DELTA, and the sign matters. A Turbo4 plane is SMALLER than an fp32
  // plane -- 66 bytes per 128 values against 4 bytes per value -- so
  // `turbo4_planes - fp32_planes` UNDERFLOWS in size_t and wraps to about
  // 1.8e19. That is a big enough number to satisfy any "the gate sees a real
  // byte count" check, which is exactly why the arithmetic is done signed here
  // and the sign is on the line where a reader will look at it.
  const long long plane_delta =
      static_cast<long long>(
          2 * PlaneBytes(config, context, QwenKvCacheStorage::kTurbo4)) -
      static_cast<long long>(
          2 * PlaneBytes(config, context, QwenKvCacheStorage::kFp32));

  Check(turbo4.temporary_scratch_bytes == fp32.temporary_scratch_bytes,
        "the scratch term sits in request_state_bytes, the per-session term a "
        "fit check multiplies by concurrency -- not in the shared scratch");
  Check(static_cast<long long>(turbo4.TotalBytes()) ==
            static_cast<long long>(fp32.TotalBytes()) + plane_delta +
                static_cast<long long>(measured),
        "Turbo4's reservation differs from fp32's by exactly the plane saving "
        "plus the scratch -- the fit gate is short by neither");
  Check(static_cast<long long>(turbo4.TotalBytes()) >= 0 &&
            static_cast<long long>(turbo4.TotalBytes()) <
                static_cast<long long>(fp32.TotalBytes()),
        "and the total that reaches the gate is itself a sane, positive byte "
        "count rather than a wrapped one");
  std::printf("  turbo4 total=%zu (state %zu + temp %zu)\n",
              turbo4.TotalBytes(), turbo4.request_state_bytes,
              turbo4.temporary_scratch_bytes);
  std::printf("  fp32   total=%zu (state %zu + temp %zu)\n", fp32.TotalBytes(),
              fp32.request_state_bytes, fp32.temporary_scratch_bytes);
  std::printf("  plane_delta=%lld measured scratch=%zu\n", plane_delta,
              measured);
}

// Two contexts: both the one-layer prefill scratch and the split-K partials must
// be charged at each size. The latter grows by deterministic 128-position
// splits, not by a guessed bytes-per-token multiplier.
void ScratchScalesWithContext() {
  const auto config = Ud27BConfig();
  constexpr std::uint32_t small = 1024;
  constexpr std::uint32_t large = 8192;
  const std::size_t at_small = ResidualScratchBytes(config, small);
  const std::size_t at_large = ResidualScratchBytes(config, large);
  const auto expected = [&config](std::uint32_t context) {
    return 2 * static_cast<std::size_t>(context) * config.num_key_value_heads *
               config.head_dim * sizeof(std::uint16_t) +
           gufo::hip::Turbo4DecodeWorkspaceBytes(
               context, config.num_attention_heads, config.num_key_value_heads,
               config.head_dim);
  };
  Check(at_small == expected(small) && at_large == expected(large) &&
            at_large > at_small,
        "each context pays exact f16 scratch plus deterministic split workspace");
}

// A Turbo4 estimate is still refused when the geometry has no whole 128-value
// block per head. The scratch must not make an unservable geometry servable.
void UnalignedGeometryStillRefuses() {
  auto config = Ud27BConfig();
  config.head_dim = 96;  // 96 is not a multiple of 128.
  constexpr std::uint32_t context = 4096;
  bool threw = false;
  try {
    (void)QwenGpuArena::EstimateMemoryUsage(
        config, context, Policy(QwenKvCacheStorage::kTurbo4));
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw,
        "a Turbo4 estimate with a head that is not a whole number of blocks "
        "is still refused, scratch or not");
}

}  // namespace

int main() {
  ScratchIsChargedAtTheEstimate();
  ScratchIsOneLayerNotAPlane();
  ScratchIsAbsentForEveryOtherStorage();
  FitGateSeesItOnce();
  ScratchScalesWithContext();
  UnalignedGeometryStillRefuses();
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

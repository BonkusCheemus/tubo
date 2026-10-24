// Turbo4 scratch/workspace are allocated and backed.
//
// WHY A GPU TEST AT ALL. The companion CPU test
// (turbo4_scratch_allocation_test.cpp) proves the ESTIMATE charges both the
// prefill scratch and E5 decode workspace. This file proves the allocator
// performs those hipMallocs and that the recorded ranges are backed.
//
// WHAT THE HISTORICAL DEFECT WAS. Under kTurbo4 the arena allocated
// d_kv_turbo4 and left BOTH d_attention_kv_f16 and d_kv_cache null. That closed
// the f32-store hazard by construction, but it also meant no f16 attention
// route could run at all. E2 creates the first run where one does, so the
// buffer has to exist before E3 wires the routes to it.
//
// WHAT IS ASSERTED.
//   0. Under Turbo4 d_attention_kv_f16 is NULL and the scratch is in its OWN
//      member. That nullness is a tripwire, not an oversight: see the note on
//      the member in executor.hpp.
//   1. Under Turbo4 the scratch is non-null and the RECORDED bytes are exactly
//      2*max_context*kv_heads*head_dim*2 -- computed here from the geometry,
//      not read back out of the code under test.
//   2. The recorded bytes EQUAL the scratch the ESTIMATE charged. This is the
//      check that matters most: the two sites are written at different times
//      with different inputs (config vs a member), and a drift between them is
//      exactly the "fit passed, allocation failed" failure the CPU test alone
//      cannot see.
//   3. The buffer is backed. hipMemset, then write the FIRST and the LAST half
//      and read both back. A recorded size with nothing behind it passes (1).
//   4. The chosen V split has exactly zero slack: V starts at
//      max_context*kv_heads*head_dim halfs, so the buffer must be exactly twice
//      that. One half less and E3 slices past the end.
//   5. f16 and fp32 allocate NO scratch: the recorded scratch is 0 for both.
//      fp32 keeps its pointer null exactly as before; f16 keeps its plane.
//      These are the "bytes unchanged" half of the brief.
//   6. Turbo4's other members are unchanged: the block plane is still the
//      plane, d_kv_cache is still null, and the two status words still exist.
//
// NOT ASSERTED HERE: that the routes READ this buffer. That is E3's wiring and
// E4's GPU check; asserting it here would be a test for code that does not
// exist yet.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"

namespace {

int g_failures = 0;

// Flushed per line on purpose. If a later assertion aborts the process -- which
// a missing allocation will, at the first hipMemset -- an unflushed buffer
// throws away every line printed before it, so the reader gets a core dump and
// no idea WHICH check had already failed. Observed while mutation-testing this
// file: the null-scratch mutation produced a SIGABRT and zero output.
void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
  if (!ok) ++g_failures;
}

using gufo::hip::QwenExecutionPolicy;
using gufo::hip::QwenGpuArena;
using gufo::hip::QwenKvCacheStorage;

// head_dim 256 so a head is a whole number of 128-value Turbo4 blocks; two
// full-attention layers out of eight so "one layer of scratch" and "a plane"
// are numbers a test can tell apart.
gufo::core::ModelConfig TestConfig() {
  gufo::core::ModelConfig config;
  config.model_name = "turbo4-scratch-alloc";
  config.num_layers = 8;
  config.hidden_size = 512;
  config.intermediate_size = 1024;
  config.num_attention_heads = 8;
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.vocab_size = 512;
  config.context_length = 64;
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

std::size_t PlaneBytes(const gufo::core::ModelConfig& config,
                       std::uint32_t context, QwenKvCacheStorage storage) {
  const std::size_t elements = static_cast<std::size_t>(
                                   config.FullAttentionLayerCount()) *
                               context * config.num_key_value_heads *
                               config.head_dim;
  return gufo::hip::QwenKvBytesForElements(storage, elements);
}

// The scratch the ESTIMATE charges, measured the same way the CPU test does it:
// as a residual against the fp32 estimate, so this is an independent number and
// not a restatement of the allocator's formula.
std::size_t EstimatedScratchBytes(const gufo::core::ModelConfig& config,
                                  std::uint32_t context) {
  const auto turbo4 = QwenGpuArena::EstimateMemoryUsage(
      config, context, Policy(QwenKvCacheStorage::kTurbo4));
  const auto fp32 = QwenGpuArena::EstimateMemoryUsage(
      config, context, Policy(QwenKvCacheStorage::kFp32));
  return turbo4.request_state_bytes - fp32.request_state_bytes -
         2 * PlaneBytes(config, context, QwenKvCacheStorage::kTurbo4) +
         2 * PlaneBytes(config, context, QwenKvCacheStorage::kFp32);
}

void Turbo4AllocatesTheScratch() {
  const auto config = TestConfig();
  QwenGpuArena arena(config, config.context_length,
                     Policy(QwenKvCacheStorage::kTurbo4));
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  // From the geometry, independently of the code under test.
  const std::size_t expected = 2 * static_cast<std::size_t>(config.context_length) *
                               config.num_key_value_heads *
                               config.head_dim * sizeof(std::uint16_t);
  const std::size_t recorded = arena.Turbo4ScratchBytesRecorded();
  const std::size_t expected_workspace = gufo::hip::Turbo4DecodeWorkspaceBytes(
      config.context_length, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim);
  const std::size_t expected_partial = gufo::hip::Turbo4DecodePartialBytes(
      config.context_length, config.num_attention_heads,
      config.num_key_value_heads, config.head_dim);
  const std::size_t recorded_workspace =
      arena.Turbo4DecodeWorkspaceBytesRecorded();

  std::printf("  scratch recorded=%zu B expected=%zu B\n", recorded, expected);
  // THE TRIPWIRE, stated as an assertion. d_attention_kv_f16 must stay NULL on a
  // block layer: all 25 `OffsetIfPresent(d_attention_kv_f16, total_k)` call
  // sites rely on that null to refuse, and `total_k` is the ALL-LAYER element
  // count. Parking the scratch there would hand every one of them a non-null
  // pointer at base + all-layers, which is silently past the end of a one-layer
  // buffer.
  Check(arena.d_attention_kv_f16 == nullptr,
        "Turbo4 leaves d_attention_kv_f16 NULL -- the tripwire the f16 call "
        "sites refuse on must survive the scratch allocation");
  Check(arena.d_turbo4_attention_scratch != nullptr,
        "the scratch lives in its OWN member, d_turbo4_attention_scratch");
  Check(recorded == expected,
        "the recorded scratch is exactly 2*max_context*kv_heads*head_dim*2 "
        "bytes -- the size the design fixes, computed from the geometry here");
  Check(recorded_workspace == expected_workspace &&
            arena.Turbo4DecodePartialBytesRecorded() == expected_partial &&
            recorded + recorded_workspace ==
                EstimatedScratchBytes(config, config.context_length),
        "allocated prefill scratch plus E5 workspace equals the estimate, "
        "including split partials exactly once");
  Check(arena.d_turbo4_decode_workspace != nullptr,
        "E5 partials and rotated vectors are backed by an arena allocation");
  Check(arena.Turbo4KvBytesRecorded() ==
            2 * PlaneBytes(config, config.context_length,
                           QwenKvCacheStorage::kTurbo4),
        "the block plane is unchanged: still both planes of blocks");

  // The chosen V split: V starts at max_context*kv_heads*head_dim halfs.
  // Exactly twice that, so the split fits with no slack to spare.
  const std::size_t one_plane_halfs =
      static_cast<std::size_t>(config.context_length) *
      config.num_key_value_heads * config.head_dim;
  Check(recorded == 2 * one_plane_halfs * sizeof(std::uint16_t),
        "the buffer is exactly K+V of one layer, so v = k + "
        "max_context*kv_heads*head_dim halfs lands on the last element");

  // Unchanged neighbours: the block plane is the plane, f32 is still absent.
  Check(arena.d_kv_turbo4 != nullptr, "the Turbo4 block plane is still there");
  Check(arena.d_kv_cache == nullptr,
        "d_kv_cache is still null under Turbo4 -- no fp32 plane is allocated");
}

void ScratchIsRealMemory() {
  const auto config = TestConfig();
  QwenGpuArena arena(config, config.context_length,
                     Policy(QwenKvCacheStorage::kTurbo4));
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  const std::size_t recorded = arena.Turbo4ScratchBytesRecorded();
  // REFUSED, not attempted. hipMemset on a null pointer throws, and a throw
  // here would abort the whole binary before any later check ran -- turning a
  // readable "the scratch was never allocated" into a core dump.
  if (recorded == 0 || arena.d_turbo4_attention_scratch == nullptr) {
    Check(false,
          "scratch is backed by real memory: nothing was allocated to back the "
          "recorded byte count");
    return;
  }
  auto* base = static_cast<std::uint16_t*>(arena.d_turbo4_attention_scratch);

  // A recorded byte count is a claim about the buffer; writing at the far end
  // and reading it back is a measurement of it. An allocator that returned a
  // pointer with fewer bytes behind it than it recorded passes every size
  // assertion above and fails here.
  //
  // Both ends go through hipMemcpy, never a direct store: `base` is a DEVICE
  // pointer. A host-side store to it is undefined behaviour that an iGPU with
  // shared memory may well appear to honour, which would make this check pass
  // without ever touching the allocation.
  HIP_CHECK(hipMemset(arena.d_turbo4_attention_scratch, 0, recorded));
  const std::uint16_t kFirst = 0x1234;
  const std::uint16_t kLast = 0xBEEF;
  const std::size_t halfs = recorded / sizeof(std::uint16_t);
  HIP_CHECK(hipMemcpy(base, &kFirst, sizeof(kFirst),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(base + halfs - 1, &kLast, sizeof(kLast),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  std::uint16_t read_first = 0;
  std::uint16_t read_last = 0;
  HIP_CHECK(hipMemcpy(&read_first, base, sizeof(read_first),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&read_last, base + halfs - 1, sizeof(read_last),
                      hipMemcpyDeviceToHost));
  Check(read_first == kFirst, "the FIRST half of the scratch is writable");
  Check(read_last == kLast,
        "the LAST half of the scratch is writable: the whole recorded range is "
        "backed, not just the front of it");

  const std::size_t decode_bytes = arena.Turbo4DecodeWorkspaceBytesRecorded();
  if (decode_bytes == 0 || arena.d_turbo4_decode_workspace == nullptr) {
    Check(false, "E5 workspace is backed by real memory");
    return;
  }
  auto* decode = static_cast<std::uint8_t*>(arena.d_turbo4_decode_workspace);
  const std::uint8_t kFirstByte = 0x12;
  const std::uint8_t kLastByte = 0xBE;
  HIP_CHECK(hipMemset(arena.d_turbo4_decode_workspace, 0, decode_bytes));
  HIP_CHECK(hipMemcpy(decode, &kFirstByte, 1, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(decode + decode_bytes - 1, &kLastByte, 1,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipStreamSynchronize(arena.stream));
  std::uint8_t decode_read_first = 0;
  std::uint8_t decode_read_last = 0;
  HIP_CHECK(hipMemcpy(&decode_read_first, decode, 1, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&decode_read_last, decode + decode_bytes - 1, 1,
                      hipMemcpyDeviceToHost));
  Check(decode_read_first == kFirstByte && decode_read_last == kLastByte,
        "first and last E5 workspace bytes are writable");
}

void OtherStoragesAllocateNoScratch() {
  const auto config = TestConfig();

  {
    QwenGpuArena arena(config, config.context_length,
                       Policy(QwenKvCacheStorage::kFp16));
    HIP_CHECK(hipStreamSynchronize(arena.stream));
    Check(arena.Turbo4ScratchBytesRecorded() == 0,
          "an f16 KV table allocates NO Turbo4 scratch: its bytes are unchanged");
    Check(arena.d_attention_kv_f16 != nullptr,
          "an f16 KV table still gets its own f16 plane, exactly as before");
    Check(arena.d_kv_turbo4 == nullptr,
          "an f16 KV table allocates no block plane, as before");
    Check(arena.d_turbo4_attention_scratch == nullptr,
          "an f16 KV table allocates no Turbo4 scratch in its own member "
          "either -- the two allocations are independent");
    Check(arena.Turbo4DecodeWorkspaceBytesRecorded() == 0 &&
              arena.d_turbo4_decode_workspace == nullptr,
          "an f16 KV table allocates no E5 decode workspace");
  }
  {
    QwenGpuArena arena(config, config.context_length,
                       Policy(QwenKvCacheStorage::kFp32));
    HIP_CHECK(hipStreamSynchronize(arena.stream));
    Check(arena.Turbo4ScratchBytesRecorded() == 0,
          "an fp32 KV table allocates NO Turbo4 scratch: its bytes are "
          "unchanged");
    Check(arena.d_attention_kv_f16 == nullptr,
          "an fp32 KV table leaves the f16 pointer null, exactly as before -- "
          "the scratch work must not leak an allocation into this arm");
    Check(arena.d_kv_cache != nullptr,
          "an fp32 KV table still gets its fp32 plane");
    Check(arena.d_turbo4_attention_scratch == nullptr,
          "an fp32 KV table allocates no Turbo4 scratch, as before");
    Check(arena.Turbo4DecodeWorkspaceBytesRecorded() == 0 &&
              arena.d_turbo4_decode_workspace == nullptr,
          "an fp32 KV table allocates no E5 decode workspace");
  }
}

// The REAL accessor, on a REAL arena. The CPU pair test pins the arithmetic
// RELATION, which a reader could satisfy with a local recomputation while the
// shipped accessor said something else -- and mutating the accessor to the
// all-layers count passed every check until this was added.
void TheAccessorIsTheRealOne() {
  const auto config = TestConfig();
  QwenGpuArena arena(config, config.context_length,
                     Policy(QwenKvCacheStorage::kTurbo4));
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  const std::size_t layer = arena.GetAttentionKvLayerElements();
  const std::size_t plane = arena.GetAttentionKvPlaneElements();
  const std::size_t expected_layer =
      static_cast<std::size_t>(config.num_key_value_heads) *
      config.context_length * config.head_dim;
  std::printf("  layer_elements=%zu plane_elements=%zu (x%u layers)\n", layer,
              plane, config.FullAttentionLayerCount());

  Check(layer == expected_layer,
        "GetAttentionKvLayerElements() is kv_heads*max_context*head_dim");
  Check(plane == layer * config.FullAttentionLayerCount(),
        "and it is exactly one layer of the plane, so the scratch's V offset "
        "and the f16 cache's plane count cannot be confused");
  Check(layer != plane,
        "and the two accessors are NOT the same number at this geometry -- "
        "which is the whole reason the second one exists");
  Check(2 * layer * sizeof(std::uint16_t) ==
            arena.Turbo4ScratchBytesRecorded(),
        "twice the layer count IS the scratch size: the accessor and the "
        "allocation agree");
}

}  // namespace

int main() {
  TheAccessorIsTheRealOne();
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  std::printf("E2(b) device: Turbo4 f16 scratch allocation\n");
  Turbo4AllocatesTheScratch();
  ScratchIsRealMemory();
  OtherStoragesAllocateNoScratch();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

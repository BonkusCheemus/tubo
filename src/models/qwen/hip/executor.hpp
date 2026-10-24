#ifndef GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/models/qwen/hip/embedding_rows.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::hip {

class HipblasLtGemm;
class QwenGpuExecutor;

struct QwenGpuBatchItem {
  QwenGpuExecutor* executor{nullptr};
  tokenization::TokenId token_id{0};
  std::uint32_t position{0};
};

struct QwenGpuVerificationItem {
  QwenGpuExecutor* executor{nullptr};
  std::span<const tokenization::TokenId> tokens;
  std::uint32_t position{0};
  bool capture_logits{false};
};

struct QwenSampledVerificationResult {
  tokenization::TokenId token{0};
  bool accepted{false};
};

struct QwenGpuWeightRegion {
  // Original reader address for tensor offsets; host_copy owns GPU-visible
  // immutable bytes, backed by transparent huge pages when available.
  const void* host_data{nullptr};
  void* device_data{nullptr};
  void* host_copy{nullptr};
  std::size_t size{0};
  // > 0 = host_copy was registered as consecutive chunks of this many
  // bytes (GUFO_DEV_REGISTER_CHUNK_MIB); 0 = one registration.
  std::size_t register_chunk{0};
};

struct QwenGpuMemoryUsage {
  std::size_t request_state_bytes{0};
  std::size_t temporary_scratch_bytes{0};

  [[nodiscard]] std::size_t TotalBytes() const noexcept {
    return request_state_bytes + temporary_scratch_bytes;
  }
};

/// Immutable GPU-visible Qwen model resources shared by executor sessions.
class QwenGpuModel {
public:
  QwenGpuModel(
      std::shared_ptr<const core::GgufReader> reader,
      models::QwenModelWeights weights,
      std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
      std::vector<QwenGpuWeightRegion> weight_regions,
      std::shared_ptr<models::qwen::vision::Encoder> vision = {},
      std::unique_ptr<models::qwen::EmbeddingRowStore> embedding_rows = {});
  ~QwenGpuModel();

  QwenGpuModel(const QwenGpuModel&) = delete;
  QwenGpuModel& operator=(const QwenGpuModel&) = delete;
  QwenGpuModel(QwenGpuModel&&) = delete;
  QwenGpuModel& operator=(QwenGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenGpuModel> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr,
      std::shared_ptr<models::qwen::vision::Encoder> vision = {});

  [[nodiscard]] const std::shared_ptr<models::qwen::vision::Encoder>&
  VisionEncoder() const noexcept {
    return vision_;
  }
  [[nodiscard]] const models::QwenModelWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] std::span<const QwenGpuWeightRegion> GetWeightRegions()
      const noexcept {
    return weight_regions_;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetWeightRegionCount() const noexcept {
    return weight_regions_.size();
  }
  [[nodiscard]] std::size_t GetResidentBytes() const noexcept;
  models::qwen::EmbeddingRowStore* GetEmbeddingRows() const noexcept {
    return embedding_rows_.get();
  }

private:
  // Keep the mapped GGUF storage alive until every registered region is
  // released.
  std::shared_ptr<const core::GgufReader> reader_;
  models::QwenModelWeights weights_;
  std::shared_ptr<const tokenization::QwenTokenizer> tokenizer_;
  std::vector<QwenGpuWeightRegion> weight_regions_;
  std::unique_ptr<models::qwen::EmbeddingRowStore> embedding_rows_;
  std::shared_ptr<models::qwen::vision::Encoder> vision_;
};

class QwenGpuSnapshot final {
public:
  ~QwenGpuSnapshot();

  QwenGpuSnapshot(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot& operator=(const QwenGpuSnapshot&) = delete;
  QwenGpuSnapshot(QwenGpuSnapshot&&) = delete;
  QwenGpuSnapshot& operator=(QwenGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_ + vision_layout_.images.size() *
                                sizeof(models::qwen::vision::ImageGrid);
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] QwenKvCacheStorage KvStorage() const noexcept {
    return kv_storage_;
  }
  [[nodiscard]] QwenRecurrentStateStorage RecurrentStateStorage()
      const noexcept {
    return recurrent_state_storage_;
  }
  [[nodiscard]] std::size_t CompactPayloadBytes() const;
  [[nodiscard]] std::size_t SerializeCompact(
      std::span<std::uint8_t> destination) const;
  /// The SerializeCompact bytes, in order, through `sink` in pieces of at
  /// most `piece_bytes`: one small host buffer instead of a whole-payload copy.
  void StreamCompact(
      const std::function<void(std::span<const std::uint8_t>)>& sink,
      std::size_t piece_bytes = std::size_t{64} << 20U) const;
  [[nodiscard]] static models::qwen::vision::RopeLayout ReadRopeLayout(
      std::span<const std::uint8_t> payload);

private:
  // One body for both: straight into `destination` when `sink` is null,
  // else piecewise through a reused host buffer.
  void EmitCompact(
      std::span<std::uint8_t> destination,
      const std::function<void(std::span<const std::uint8_t>)>* sink,
      std::size_t piece_bytes) const;
  QwenGpuSnapshot() = default;

  void* d_kv_f32_{nullptr};
  void* d_kv_f16_{nullptr};
  /// Block-addressed (Turbo4) plane: K and V, two planes in one allocation,
  /// exactly as the fp16 and fp32 sites do. A THIRD member, not a
  /// reinterpretation of either — a Turbo4 plane's bytes are not fp16 bytes at
  /// any offset, so aliasing one onto the other would be a silent corruption
  /// that passes every size check.
  void* d_kv_turbo4_{nullptr};
  void* d_ssm_conv_{nullptr};
  void* d_ssm_deltanet_{nullptr};
  std::size_t kv_elements_per_plane_{0};
  std::size_t conv_elements_{0};
  std::size_t deltanet_elements_{0};
  std::size_t conv_elements_per_layer_{0};
  std::size_t deltanet_elements_per_layer_{0};
  std::uint32_t num_layers_{0};
  std::uint32_t attention_layers_{0};
  std::uint32_t full_attention_interval_{0};
  std::uint32_t kv_width_{0};
  /// Head geometry the plane was WRITTEN with, recorded rather than derived.
  ///
  /// SerializeCompact has no QwenGpuArena and therefore no config, yet it must
  /// describe a copy of the very plane SaveSnapshot wrote. Reconstructing the
  /// geometry from kv_width_ alone cannot work: kv_width is kv_heads *
  /// head_dim, and the fp32 plane is head-major, so its row pitch depends on
  /// the split. These are in-memory fields; the compact WIRE header is
  /// unchanged.
  std::uint32_t kv_heads_{0};
  std::uint32_t head_dim_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::size_t payload_bytes_{0};
  QwenKvCacheStorage kv_storage_{QwenKvCacheStorage::kFp32};
  QwenKvFormatTable kv_formats_{};
  QwenRecurrentStateStorage recurrent_state_storage_{
      QwenRecurrentStateStorage::kFp32};

  friend class QwenGpuArena;
  friend class QwenGpuExecutor;
  models::qwen::vision::RopeLayout vision_layout_;
};

/// Shared decode-lifetime workspaces over stable arena allocations.
struct QwenDecodeScratch {
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> logits;
  std::span<hip_bfloat16> bf16;
  std::span<hip_bfloat16> weight_bf16;
  std::span<std::uint32_t> prompt_tokens;

  /// Aliases the first element of QwenSsmScratch::alpha. It is legal only
  /// after all layer execution has completed, during the sampling/output epoch.
  std::span<std::uint32_t> sampled_token;
};

struct QwenAttentionScratch {
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> output;
  std::span<float> split_k;
};

struct QwenSsmScratch {
  std::span<float> qkv;
  std::span<float> conv_out;
  std::span<float> gate;
  std::span<float> out;
  std::span<float> alpha;
  std::span<float> beta;
};

struct QwenFfnScratch {
  std::span<float> gate;
  std::span<float> up;
  std::span<float> activation;
  std::span<float> out;
};

/// Typed non-owning capability views over stable Qwen GPU arena allocations.
/// The nested aggregates carry element counts without owning memory or changing
/// any address used by kernels or graph capture.
struct QwenGpuScratchView {
  QwenDecodeScratch decode;
  QwenAttentionScratch attention;
  QwenSsmScratch ssm;
  QwenFfnScratch ffn;
};

static_assert(std::is_trivially_copyable_v<QwenDecodeScratch>);
static_assert(std::is_trivially_copyable_v<QwenAttentionScratch>);
static_assert(std::is_trivially_copyable_v<QwenSsmScratch>);
static_assert(std::is_trivially_copyable_v<QwenFfnScratch>);
static_assert(std::is_trivially_copyable_v<QwenGpuScratchView>);
static_assert(std::is_same_v<decltype(QwenDecodeScratch::sampled_token),
                             std::span<std::uint32_t>>);

template<typename T>
[[nodiscard]] constexpr T* OffsetIfPresent(T* pointer,
                                           std::size_t offset) noexcept {
  return pointer == nullptr ? nullptr : pointer + offset;
}

/// Dev switch: GUFO_DEV_SCRATCH_TILED=1 sizes the Turbo4 f16 prefill
/// scratch from the PREFILLED prefix instead of from `--context`.
///
/// WHY A SWITCH AND NOT THE DEFAULT. The scratch is the one allocation whose
/// lifetime shrank without its size shrinking: it exists so the f16 attention
/// routes can read a 66-byte-block layer, it covers positions
/// [0, start_pos + batch) only, and the decode routes never touch it -- they
/// read the blocks directly through the split-K reader. Sizing it at load for
/// `--context` therefore reserves 4 KiB per token of a window that may never be
/// filled: 360 MiB at `-c 92160`, on a 31 GiB UMA box where 1-2 GiB free is
/// what produces multi-minute KFD stalls.
///
/// NOT A NUMERICS CHANGE. The dequant destination offset is
/// `t * kv_width + h * head_dim` (turbo4_kernels.hpp:336) and the attention
/// routes read `base + kv_head * head_dim + position * kKvWidth`
/// (attention_wmma.hip:253-258). Neither contains `max_context`. The ONE place
/// the configured context enters this buffer is the V base offset,
/// `scratch + layer_elements`, which the arena now reports from the allocation
/// it actually made. Same bytes, same positions, same order of operations.
[[nodiscard]] inline bool Turbo4ScratchTiledEnabled() noexcept {
  static const bool enabled = [] {
    const char* const value = std::getenv("GUFO_DEV_SCRATCH_TILED");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

/// Preallocated, zero-allocation GPU execution arena on gfx1151.
class QwenGpuArena {
public:
  explicit QwenGpuArena(
      const core::ModelConfig& config, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = ResolveRuntimeQwenExecutionPolicy());
  ~QwenGpuArena();

  QwenGpuArena(const QwenGpuArena&) = delete;
  QwenGpuArena& operator=(const QwenGpuArena&) = delete;
  QwenGpuArena(QwenGpuArena&&) noexcept;
  QwenGpuArena& operator=(QwenGpuArena&&) noexcept;

  void Reset();
  /// Pins the live recurrent state for rollback. Neither save nor restore
  /// copies anything: the state lives in one of two buffers, and the first
  /// pass after either call reads the pinned buffer and writes the other one.
  void SaveState(std::uint32_t valid_context);
  void RestoreState();
  /// Names the buffers one recurrent-state pass reads and writes. Call exactly
  /// once per forward, replay or prefill pass, before its first launch; every
  /// launch of the pass then writes `*_out`, so the pinned buffer survives
  /// until the next SaveState.
  [[nodiscard]] QwenRecurrentStateIo BeginRecurrentWrite();
  /// Buffer indices of the most recent BeginRecurrentWrite, for graph keys.
  [[nodiscard]] std::uint64_t RecurrentIoIdentity() const noexcept;
  [[nodiscard]] bool BeginSsmReplayCapture();
  void DisableSsmReplayCapture();
  void MarkSsmReplayPosition(std::uint32_t position) noexcept;
  [[nodiscard]] bool CanReplaySsmPosition(
      std::uint32_t position) const noexcept;
  [[nodiscard]] SsmReplayCapture GetSsmReplayCapture() const noexcept;
  [[nodiscard]] bool IsSsmReplayCaptureActive() const noexcept {
    return replay_capture_active_;
  }
  [[nodiscard]] const float* GetReplayQkv(std::uint32_t layer,
                                          std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayAlpha(std::uint32_t layer,
                                            std::uint32_t position) const;
  [[nodiscard]] const float* GetReplayBeta(std::uint32_t layer,
                                           std::uint32_t position) const;


  float* d_hidden{nullptr};
  float* d_normed{nullptr};
  float* d_q{nullptr};
  float* d_k{nullptr};
  float* d_v{nullptr};
  float* d_attn_out{nullptr};
  float* d_ffn_gate{nullptr};
  float* d_ffn_up{nullptr};
  float* d_ffn_act{nullptr};
  float* d_ffn_out{nullptr};
  float* d_ssm_qkv{nullptr};
  float* d_conv_out{nullptr};
  float* d_ssm_gate{nullptr};
  float* d_ssm_out{nullptr};
  float* d_alpha_buf{nullptr};
  float* d_beta_buf{nullptr};
  // opt-c170-deltanet-rowsplit prologue scratch: 3 floats per (token, key head)
  // for the k/q norms and the unnormalized k . q, and 2 floats per (token,
  // value head) for the decay and beta gates.
  float* d_ssm_kq_scales{nullptr};
  float* d_ssm_alpha_beta{nullptr};
  float* d_logits{nullptr};
  void* d_attention_kv_f16{nullptr};
  /// The Turbo4 f16 scratch, and DELIBERATELY NOT d_attention_kv_f16.
  ///
  /// d_attention_kv_f16 stays NULL under block storage, and that nullness is a
  /// TRIPWIRE the code relies on: kv_allocation.hpp:152-160 and
  /// execution_policy.hpp:436-447 treat "f16 pair is null on a block layer" as
  /// the refusal, so a path that wrongly reaches the f16 reader faults loudly
  /// instead of quietly reading something.
  ///
  /// Putting the scratch in d_attention_kv_f16 would destroy that tripwire.
  /// All 25 call sites pass
  /// `OffsetIfPresent(d_attention_kv_f16, total_k)` — and `total_k` is the
  /// ALL-LAYER plane element count (GetAttentionKvPlaneElements multiplies by
  /// FullAttentionLayerCount). With a real buffer sitting there, every one of
  /// those would receive a non-null pointer at base + all-layers, which is
  /// silently past the end of a ONE-layer scratch, and a layer_idx > 0 on top
  /// overruns again. Nobody has to write that bug for it to happen.
  ///
  /// E3 passes THIS pointer explicitly, on block layers only, through one
  /// helper that takes no layer index. See the note there for why.
  void* d_turbo4_attention_scratch{nullptr};
  /// E5 split-K workspace: partial (m,l,o) rows, rotated Q, and reduced output.
  /// One allocation keeps its lifetime identical to the Turbo4 KV plane.
  void* d_turbo4_decode_workspace{nullptr};
  /// ONE uint32 device word holding the chunk's start position, for E1's
  /// batched writer, which reads it on the device so the position survives
  /// graph capture.
  ///
  /// WHY A WORD AND NOT A HOST BUFFER. The first version of this staged the
  /// position with hipMalloc + a BLOCKING hipMemcpy + hipFree per prefill
  /// chunk. hipFree synchronises the device, so that was a device-wide sync
  /// once per chunk, and it needed the host value to stay alive across the
  /// launch. The word is written with hipMemsetD32Async on the arena's own
  /// stream instead, which is stream-ordered with the kernels that read it.
  ///
  /// Allocated beside the scratch, and carried at the same bookkeeping sites.
  std::uint32_t* d_turbo4_first_pos{nullptr};
  float* d_kv_cache{nullptr};
  /// THE TWO TURBO4 STATUS WORDS — one for the writer's outcome, one for the
  /// reader's. They are SEPARATE allocations, not one word shared by both.
  ///
  /// A single word would be wrong in the direction that matters: the reader
  /// memsets the word before it dispatches, and the writer latches into it
  /// before that. If they shared, the reader's own reset would erase the
  /// writer's failure between the write and the read — which is exactly the
  /// window the writer-failure check exists to cover, and the failure would
  /// present as a clean reader over a stale plane.
  ///
  /// The reader never writes the writer's word (see Turbo4ReadWriterStatus), so
  /// whatever is in it after a decode is the writer's real outcome and can be
  /// read back by anyone who wants the receipt.
  ///
  /// ALLOCATED ONLY FOR kTurbo4. For every other storage these stay null, and a
  /// null here is a refusal rather than a default.
  void* d_kv_turbo4_write_status{nullptr};
  void* d_kv_turbo4_read_status{nullptr};
  /// THE BYTES THE ARENA ACTUALLY ALLOCATED, recorded at the hipMalloc call.
  /// The decode step was recomputing the plane size from
  /// the config, so the length it passed to the capacity checks was a CLAIM
  /// about the buffer rather than a measurement of it — and a capacity check
  /// only means something against the real length, which is what the launcher
  /// comments say.
  std::size_t turbo4_kv_bytes_recorded{0};
  std::size_t turbo4_status_bytes_recorded{0};
  /// The f16 scratch the arena actually allocated, recorded at its hipMalloc.
  /// Same reason as turbo4_kv_bytes_recorded: a consumer that has to know the
  /// scratch is big enough (the E3 wiring slicing V = k + max_context*kKvWidth)
  /// needs a MEASUREMENT of the buffer, not a recomputation of the formula that
  /// produced it. 0 means no scratch was allocated.
  std::size_t turbo4_scratch_bytes_recorded{0};
  std::size_t turbo4_decode_workspace_bytes_recorded{0};
  std::size_t turbo4_decode_partial_bytes_recorded{0};
  /// Stated rather than implied, so a memory report can fold it in and a
  /// reviewer can check the size without reconstructing the multiplication.
  [[nodiscard]] static constexpr std::size_t Turbo4StatusWordBytes() noexcept {
    return sizeof(std::uint32_t);
  }
  [[nodiscard]] static constexpr std::size_t Turbo4StatusWordsBytes() noexcept {
    return 2 * Turbo4StatusWordBytes();
  }
  /// The recorded plane size, or 0 when no Turbo4 plane was allocated. A caller
  /// that passes 0 here is asking for a refusal, not for a default.
  [[nodiscard]] std::size_t Turbo4KvBytesRecorded() const noexcept {
    return turbo4_kv_bytes_recorded;
  }
  [[nodiscard]] std::size_t Turbo4StatusBytesRecorded() const noexcept {
    return turbo4_status_bytes_recorded;
  }
  [[nodiscard]] std::size_t Turbo4ScratchBytesRecorded() const noexcept {
    return turbo4_scratch_bytes_recorded;
  }
  [[nodiscard]] std::size_t Turbo4DecodeWorkspaceBytesRecorded()
      const noexcept {
    return turbo4_decode_workspace_bytes_recorded;
  }
  [[nodiscard]] std::size_t Turbo4DecodePartialBytesRecorded() const noexcept {
    return turbo4_decode_partial_bytes_recorded;
  }

  /// Block-addressed (Turbo4) plane, K and V in one allocation of
  /// QwenKvBytesForElements(kTurbo4, total_kv) * 2 bytes. Populated only when
  /// policy_.kv_cache_storage == kTurbo4; the fp16 and fp32 members stay null
  /// in that case, which is why execution refuses block storage until a kernel
  /// exists that takes a 66-byte block pitch.
  void* d_kv_turbo4{nullptr};
  // Mirrors of the live recurrent buffers (see BeginRecurrentWrite); reassigned
  // whenever the live buffer changes, so snapshots and resets read and write
  // the current state without knowing about the rollback pair.
  float* d_ssm_conv_state{nullptr};
  void* d_ssm_deltanet_state{nullptr};
  std::uint32_t* d_prompt_tokens{nullptr};
  float* d_target_layer_features{nullptr};

  hipStream_t stream{nullptr};
  hipblasHandle_t hipblas_handle{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm;
  void* d_scratch_bf16{nullptr};
  float* d_split_k_attention{nullptr};
  hip_bfloat16* d_weights_bf16{nullptr};
  void* d_scratch_q8_act{nullptr};

  [[nodiscard]] std::uint32_t GetMaxBatch() const noexcept {
    return max_batch_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::size_t GetAttentionKvPlaneElements() const noexcept {
    // Qwen27B reaches 2^32 elements per plane at context 262144.
    return static_cast<std::size_t>(config_.FullAttentionLayerCount()) *
           config_.num_key_value_heads * max_context_ * config_.head_dim;
  }
  /// ONE LAYER's worth of KV elements — the plane above divided by the number
  /// of full-attention layers.
  ///
  /// WHY IT EXISTS, and why the division is not done here. This is the count
  /// the Turbo4 f16 scratch is sized from and the V offset is taken from: the
  /// scratch holds K and V of ONE layer, so its two halves are one layer apart.
  /// `GetAttentionKvPlaneElements` is a DIFFERENT number — every layer — and
  /// passing that one is a 16x-too-far offset into a one-layer buffer,
  /// silently, because the scratch sits inside a larger arena. Two design rules
  /// follow: the helper builds V itself, and this accessor
  /// exists and the helper uses it.
  ///
  /// Written as the product rather than `plane / layers` on purpose: the plane
  /// count is deliberately the product of FullAttentionLayerCount, and dividing
  /// it back would make the two accessors agree only while the multiplication
  /// has no remainder. A layer table that is not uniform would silently change
  /// the answer, and "the scratch is half the plane" is not a fact worth
  /// deriving at the call site.
  ///
  /// tests/models/qwen/turbo4_attention_pair_test.cpp pins that the two are the
  /// same rule at the production geometry.
  ///
  /// Under GUFO_DEV_SCRATCH_TILED this answers from the SCRATCH THE
  /// ARENA ACTUALLY ALLOCATED rather than from `max_context_`, because under
  /// that switch the two are no longer the same number. It is the measurement,
  /// not the formula -- the same reason `turbo4_scratch_bytes_recorded` exists
  /// at all. Half the recorded buffer, in halfs, is one layer by construction:
  /// the allocation is `2 * tokens * kv_heads * head_dim` halfs and the V base
  /// is exactly its midpoint. A recorded 0 means no scratch is allocated yet,
  /// and returning 0 makes MakeTurbo4AttentionF16Pair refuse rather than hand a
  /// caller an offset into a buffer that does not exist.
  [[nodiscard]] std::size_t GetAttentionKvLayerElements() const noexcept {
    if (Turbo4ScratchTiledEnabled() &&
        QwenKvCacheStorageIsBlockAddressed(policy_.kv_cache_storage)) {
      return turbo4_scratch_bytes_recorded / (2U * sizeof(std::uint16_t));
    }
    return static_cast<std::size_t>(config_.num_key_value_heads) *
           max_context_ * config_.head_dim;
  }
  /// Make the Turbo4 f16 prefill scratch cover `visible_context`
  /// positions, growing it if it does not already. No-op unless
  /// GUFO_DEV_SCRATCH_TILED is set and the KV storage is block-addressed.
  ///
  /// MONOTONE, AND NEVER PER CHUNK. The high-water mark only rises, so a
  /// prefill of a prompt no longer than any previous one allocates nothing, and
  /// the growth step is a whole prompt batch (`max_batch_` positions).
  /// Per-chunk allocate/free was deliberately removed from this path once
  /// already -- see the note on `d_turbo4_first_pos` -- because `hipFree`
  /// synchronises the device, and on this box large allocation churn during
  /// prefill is itself a stall source.
  void EnsureTurbo4PrefillScratch(std::size_t visible_context);
  [[nodiscard]] QwenRecurrentStateStorage GetRecurrentStateStorage()
      const noexcept {
    return policy_.recurrent_state_storage;
  }
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = ResolveRuntimeQwenExecutionPolicy());
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const;
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      std::uint32_t valid_context) const;

  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);
  [[nodiscard]] QwenGpuScratchView GetScratchView(
      std::size_t batch_size = 1) noexcept;
  void SetTargetLayerCapture(std::span<const std::uint32_t> target_layer_ids);
  [[nodiscard]] std::span<const std::uint32_t> GetTargetLayerCapture()
      const noexcept {
    return target_layer_ids_;
  }
  [[nodiscard]] std::optional<std::size_t> GetTargetLayerCaptureIndex(
      std::uint32_t layer) const noexcept;

private:
  void FreeAll() noexcept;
  void AllocateRecurrentSnapshot();
  void SyncLivePointers() noexcept;
  [[nodiscard]] bool AllocateSsmReplayLog();

  core::ModelConfig config_;
  std::uint32_t max_context_;
  std::uint32_t max_batch_;
  QwenExecutionPolicy policy_;
  std::vector<std::uint32_t> target_layer_ids_;
  // Buffer 0 comes from the constructor; buffer 1 is allocated by the first
  // SaveState. `live_recurrent_` holds the current state, `saved_recurrent_`
  // the pinned one (meaningful while has_saved_state_), `source_recurrent_`
  // the buffer the most recent pass read.
  std::array<float*, 2> d_ssm_conv_buffers_{};
  std::array<void*, 2> d_ssm_deltanet_buffers_{};
  std::uint8_t live_recurrent_{0};
  std::uint8_t saved_recurrent_{0};
  std::uint8_t source_recurrent_{0};
  float* d_ssm_replay_qkv_{nullptr};
  float* d_ssm_replay_alpha_{nullptr};
  float* d_ssm_replay_beta_{nullptr};
  std::uint32_t* d_ssm_replay_enabled_{nullptr};
  std::uint32_t saved_context_{0};
  std::uint32_t replay_last_position_{0};
  std::size_t replay_captured_positions_{0};
  bool has_saved_state_{false};
  bool replay_capture_active_{false};
};

/// End-to-end GPU model executor running directly on the gfx1151 RDNA 3.5 CUs.
class QwenGpuExecutor {
public:
  explicit QwenGpuExecutor(
      std::shared_ptr<const QwenGpuModel> model,
      std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = ResolveRuntimeQwenExecutionPolicy());
  ~QwenGpuExecutor();

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> Create(
      std::shared_ptr<const QwenGpuModel> model,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = ResolveRuntimeQwenExecutionPolicy());

  [[nodiscard]] static std::unique_ptr<QwenGpuExecutor> CreateFromGguf(
      std::shared_ptr<const core::GgufReader> reader,
      std::string* error_msg = nullptr, std::uint32_t max_context = 4096,
      QwenExecutionPolicy policy = ResolveRuntimeQwenExecutionPolicy());

  /// Generates tokens auto-regressively on GPU with streaming callback.
  std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  /// Continues generation from an exact retained prompt prefix.
  ///
  /// The executor state must represent prompt_tokens[0:cached_prefix_tokens].
  /// A zero prefix is a cold prefill over the complete prompt.
  std::vector<tokenization::TokenId> GenerateFromPrefix(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::size_t cached_prefix_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }

  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return *tokenizer_;
  }

  [[nodiscard]] const QwenGpuModel& GetModel() const noexcept {
    return *model_;
  }

  [[nodiscard]] const std::shared_ptr<const QwenGpuModel>& GetSharedModel()
      const noexcept {
    return model_;
  }

  [[nodiscard]] const QwenExecutionPolicy& GetExecutionPolicy() const noexcept {
    return policy_;
  }

  /// Runs one single token forward step on GPU, returning next token ID.
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true);

  /// Advances two to eight independent retained sessions in one
  /// layer-synchronous decode step. Stateless projections share weight reads;
  /// every row still addresses its own KV cache and recurrent state.
  [[nodiscard]] static std::vector<tokenization::TokenId> ForwardTokenBatch(
      std::span<const QwenGpuBatchItem> items);

  /// Runs batched prompt prefill on GPU, returning the first predicted token
  /// ID.
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos = 0, bool compute_logits = true);

  /// Verifies a chunk of speculative draft candidate tokens in a single
  /// parallel prefill forward pass on the GPU.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits = false);
  /// Shares exact projections across independent speculative chunks. Each
  /// sequence retains its own causal attention, recurrence and replay state.
  [[nodiscard]] static std::vector<std::vector<tokenization::TokenId>>
  ForwardVerificationBatch(std::span<const QwenGpuVerificationItem> items);
  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos);
  [[nodiscard]] std::span<const float> GetVerificationHiddenStates()
      const noexcept {
    return h_verification_hidden_;
  }
  [[nodiscard]] std::span<const float> GetVerificationLogits() const noexcept {
    return h_verification_logits_;
  }
  [[nodiscard]] std::span<const float> CopyVerificationLogits(std::size_t row);

  /// The captured target-layer feature rows of the most recent
  /// verification batch, still on the device. Null unless that batch captured
  /// tapped layers for this session. `VerificationFeatureWidth()` is the row
  /// stride in floats, the rows are contiguous, and `GetVerificationStream()`
  /// is the stream that produced them.
  [[nodiscard]] const float* GetVerificationFeatures() const noexcept {
    return last_verification_features_;
  }
  [[nodiscard]] std::size_t VerificationFeatureWidth() const noexcept {
    return last_verification_feature_width_;
  }
  [[nodiscard]] hipStream_t GetVerificationStream() const noexcept {
    return arena_.stream;
  }
  /// When set, a verification batch leaves the captured feature rows
  /// on the device and skips the host copy into h_verification_hidden_. The
  /// caller must consume GetVerificationFeatures() before the next batch.
  void SetDeviceFeatureHandoff(bool enabled) noexcept {
    device_feature_handoff_ = enabled;
  }
  [[nodiscard]] bool DeviceFeatureHandoff() const noexcept {
    return device_feature_handoff_;
  }

  /// Copies one verification logit row straight into `out`, so a
  /// caller that keeps the row does not also pay a host copy out of the
  /// shared `h_logits_` staging vector.
  void CopyVerificationLogitsInto(std::size_t row, std::vector<float>& out);

  /// Copies the logits produced by the most recent forward pass to host memory.
  [[nodiscard]] std::span<const float> CopyLastLogits();

  /// Samples the most recent device-resident logit row and transfers one token.
  [[nodiscard]] tokenization::TokenId SampleLastLogits(
      sampling::SamplerState& sampler);

  /// Restores a saved frontier row and samples with the same GPU operations
  /// as a fresh autoregressive request.
  [[nodiscard]] tokenization::TokenId SampleCachedLogits(
      std::span<const float> logits, sampling::SamplerState& sampler);

  /// Samples one row from the most recent verification batch without copying
  /// its vocabulary-sized logits to the host.
  [[nodiscard]] tokenization::TokenId SampleVerificationLogits(
      std::size_t row, sampling::SamplerState& sampler);

  /// Performs one exact sampled-speculation accept/residual decision on the
  /// GPU, returning only the decision and selected token.
  [[nodiscard]] QwenSampledVerificationResult VerifySampledToken(
      std::size_t row, tokenization::TokenId draft_token,
      std::span<const tokenization::TokenId> draft_candidate_ids,
      std::span<const float> draft_candidate_probabilities,
      double draft_token_probability, sampling::SamplerState& sampler);

  /// Enables host capture of prompt hidden states from an absolute position.
  /// Disabled by default so ordinary prefill avoids device-to-host copies.
  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids = {},
      std::uint32_t capture_start_position = 0);

  /// Returns the flattened capture suffix from the most recent
  /// ForwardPromptBatch.
  [[nodiscard]] std::span<const float> GetPromptHiddenStates() const noexcept {
    return h_prompt_hidden_;
  }

  /// Copies the final-layer hidden state from the most recent forward pass.
  [[nodiscard]] std::span<const float> CopyLastHidden();

  [[nodiscard]] std::uint32_t GetMaxPromptBatch() const noexcept {
    return arena_.GetMaxBatch();
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return arena_.GetMaxContext();
  }
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const core::ModelConfig& config, std::uint32_t max_context,
      QwenExecutionPolicy policy = ResolveRuntimeQwenExecutionPolicy());
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const;
  void SetCancellationCheck(std::function<bool()> check) {
    cancellation_check_ = std::move(check);
    vision_input_.SetCancellationCheck(cancellation_check_);
  }
  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      std::uint32_t valid_context) const;
  void ConfigureVision(
      std::shared_ptr<const models::qwen::vision::Prompt> prompt,
      std::shared_ptr<models::qwen::vision::Encoder> encoder);
  [[nodiscard]] models::qwen::vision::DeviceInput& VisionInput() noexcept {
    return vision_input_;
  }
  [[nodiscard]] const models::qwen::vision::RopeLayout& VisionLayout()
      const noexcept {
    return vision_input_.layout();
  }
  [[nodiscard]] std::unique_ptr<QwenGpuSnapshot> SaveSnapshot(
      std::uint32_t valid_context);
  void RestoreSnapshot(const QwenGpuSnapshot& snapshot);
  void RestoreCompactSnapshot(std::span<const std::uint8_t> payload,
                              std::uint32_t expected_valid_context);

  /// Resets GPU state. Any reset failure is retained and rethrown by the next
  /// fallible operation; invalidation/cleanup itself remains noexcept.
  void Reset() noexcept;
  /// `allow_divert` is ignored: upstream #650 makes the copy-free rollback
  /// (formerly opt-in) the only behaviour. Kept for the SpeculativeTarget API.
  void SaveState(std::uint32_t valid_context, bool allow_divert = false);
  void RestoreState();
  /// Ends replay capture after the last verification chunk. `commit_state`
  /// is ignored since upstream #650 (no diverted state left to commit).
  void FinishVerification(bool commit_state = true);

private:
  void CheckPrefillCancellation() const;
  void PrefillLayerCheckpoint(std::uint32_t layer);
  std::function<bool()> cancellation_check_;
  std::array<hipEvent_t, 2> prefill_events_{};
  void CheckReset(
      QwenKvExecutionPhase phase = QwenKvExecutionPhase::kPrefill) const {
    // Every public execution entry opens with this call, so it is the ONE place
    // an executor precondition can be enforced. A guard added at the five call
    // sites instead would be five edits that can drift apart — and a guard
    // added to only one of them is worse than none, because it looks covered.
    //
    // Order matters: a retained reset failure is rethrown FIRST, unchanged, so
    // the caller sees the original fault that actually broke rather than a
    // standing policy refusal that would mask it behind different text.
    //
    // `phase` defaults to kPrefill, so an entry that has not been CLASSIFIED as
    // decode refuses exactly as before. The default is the safe direction on
    // purpose: the cost of forgetting to classify a decode entry is a confusing
    // refusal, and the cost of wrongly classifying a prefill entry is silent
    // wrong output.
    if (reset_failure_)
      std::rethrow_exception(reset_failure_);
    RequireExecutableKvStorage(policy_.kv_cache_storage, phase);
  }
  std::exception_ptr reset_failure_;
  void ReplaySsmState(std::uint32_t position, std::uint32_t count = 1);
  void EnsureVerificationLogits(std::size_t batch_size);
  [[nodiscard]] GpuSamplingParameters PrepareGpuSamplingParameters(
      const sampling::SamplerState& sampler);

  [[nodiscard]] tokenization::TokenId ForwardPromptChunk(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t start_pos, bool compute_logits);
  [[nodiscard]] std::vector<tokenization::TokenId>
  ForwardDecodeEquivalentVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_logits);

  std::shared_ptr<const QwenGpuModel> model_;
  const models::QwenModelWeights& weights_;
  const tokenization::QwenTokenizer* tokenizer_;
  const QwenExecutionPolicy policy_;
  QwenGpuArena arena_;
  EmbeddingRowStage decode_embedding_;
  EmbeddingRowStage batch_embedding_;
  const detail::HipGraphCaptureKey graph_key_;
  detail::HipGraphDecodeExecutor graph_executor_;
  models::qwen::vision::DeviceInput vision_input_;
  std::vector<float> h_logits_;
  std::vector<float> h_prompt_hidden_;
  std::vector<float> h_verification_hidden_;
  std::vector<float> h_verification_logits_;
  std::vector<float> h_last_hidden_;
  float* d_verification_logits_{nullptr};
  std::size_t verification_logits_capacity_{0};
  std::size_t last_verification_rows_{0};
  std::size_t last_hidden_offset_{0};
  // This session's slice of the coordinator's feature buffer from the
  // most recent verification batch. Valid until the next batch overwrites it.
  const float* last_verification_features_{nullptr};
  std::size_t last_verification_feature_width_{0};
  bool device_feature_handoff_{false};
  // Features normally share d_verification_logits_, which the output
  // projection then overwrites. A device handoff has to outlive that GEMM, so
  // the handoff route captures into this dedicated buffer instead.
  float* d_verification_features_{nullptr};
  std::size_t verification_features_capacity_{0};
  void EnsureVerificationFeatures(std::size_t elements);
  float* d_target_layer_features_{nullptr};
  GpuSamplingWorkspace sampling_workspace_;
  std::optional<tokenization::TokenId> next_token_;
  bool capture_prompt_hidden_{false};
  std::uint32_t prompt_hidden_capture_start_position_{0};
  bool replaying_ssm_state_{false};
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTOR_HPP_

#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/dflash_policy.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Optional synchronous host observation for the independent reference test.
/// The span is valid only during the callback; production leaves it empty.
using DFlashTrace =
    std::function<void(std::string_view, std::span<const float>)>;

class QwenDFlashGpuExecutor;

struct QwenDFlashContextRequest {
  QwenDFlashGpuExecutor* executor{nullptr};
  std::span<const float> features;
  std::uint32_t position{0};
  DFlashTrace trace;
};

struct QwenDFlashBlockRequest {
  QwenDFlashGpuExecutor* executor{nullptr};
  tokenization::TokenId anchor{0};
  std::uint32_t position{0};
  std::uint32_t draft_count{0};
  float temperature{0.0F};
  std::span<const float> uniforms;
  DFlashTrace trace;
};

class QwenDFlashGpuSnapshot final {
public:
  ~QwenDFlashGpuSnapshot();

  QwenDFlashGpuSnapshot(const QwenDFlashGpuSnapshot&) = delete;
  QwenDFlashGpuSnapshot& operator=(const QwenDFlashGpuSnapshot&) = delete;
  QwenDFlashGpuSnapshot(QwenDFlashGpuSnapshot&&) = delete;
  QwenDFlashGpuSnapshot& operator=(QwenDFlashGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const;
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const;

private:
  QwenDFlashGpuSnapshot() = default;

  void* d_k_{nullptr};
  void* d_v_{nullptr};
  std::size_t elements_per_layer_{0};
  std::uint32_t num_layers_{0};
  std::uint32_t kv_width_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::uint32_t history_capacity_{0};
  std::size_t payload_bytes_{0};

  friend class QwenDFlashGpuExecutor;
};

/// Immutable GPU-visible DFlash / DFlash-2 model weights and topology.
class QwenDFlashGpuModel final {
public:
  ~QwenDFlashGpuModel();

  QwenDFlashGpuModel(const QwenDFlashGpuModel&) = delete;
  QwenDFlashGpuModel& operator=(const QwenDFlashGpuModel&) = delete;
  QwenDFlashGpuModel(QwenDFlashGpuModel&&) = delete;
  QwenDFlashGpuModel& operator=(QwenDFlashGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenDFlashGpuModel> Create(
      std::shared_ptr<const core::GgufReader> dflash_reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const speculative::QwenDFlashWeights& GetWeights()
      const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] const speculative::QwenDFlashConfig& GetDFlashConfig()
      const noexcept {
    return weights_.dflash_config;
  }
  [[nodiscard]] std::size_t GetPackedWeightBytes() const noexcept {
    return packed_weight_bytes_;
  }
  models::qwen::EmbeddingRowStore* GetEmbeddingRows() const noexcept {
    return target_model_->GetEmbeddingRows();
  }
  /// GUFO_DEV_DFLASH_DROP_PAGECACHE: drop the sidecar file's pages again and log
  /// page-cache residency before/after. Create() drops once after packing; the
  /// executor calls this after its prewarm, so pages re-faulted by anything in
  /// between go too (RAM ledger 10-13: 1.0 GiB stayed resident). No-op when the
  /// env is off. `where` labels the log line.
  void DropFilePages(const char* where) const noexcept;

  /// GUFO_DEV_DRAFT_VOCAB (draft_vocab.hpp): the LM-head rows of the kept ids,
  /// gathered once at load, and those ids on the device. Size 0 = no cut.
  struct DraftVocabCut {
    models::QwenTensorRef head{};
    const std::uint32_t* d_ids{nullptr};
    std::uint32_t size{0};
  };
  [[nodiscard]] const DraftVocabCut& GetDraftVocabCut() const noexcept {
    return draft_vocab_cut_;
  }

private:
  QwenDFlashGpuModel(std::shared_ptr<const core::GgufReader> dflash_reader,
                     std::shared_ptr<const QwenGpuModel> target_model,
                     speculative::QwenDFlashWeights weights,
                     std::vector<void*> allocations,
                     std::size_t packed_weight_bytes,
                     DraftVocabCut draft_vocab_cut);

  std::shared_ptr<const core::GgufReader> dflash_reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  speculative::QwenDFlashWeights weights_;
  std::vector<void*> allocations_;
  std::size_t packed_weight_bytes_{0};
  DraftVocabCut draft_vocab_cut_{};
};

/// Session executor for GPU DFlash / DFlash-2 parallel block drafting.
class QwenDFlashGpuExecutor final {
public:
  ~QwenDFlashGpuExecutor();

  QwenDFlashGpuExecutor(const QwenDFlashGpuExecutor&) = delete;
  QwenDFlashGpuExecutor& operator=(const QwenDFlashGpuExecutor&) = delete;
  QwenDFlashGpuExecutor(QwenDFlashGpuExecutor&&) = delete;
  QwenDFlashGpuExecutor& operator=(QwenDFlashGpuExecutor&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuExecutor> Create(
      std::shared_ptr<const QwenDFlashGpuModel> model,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  void Reset() noexcept;

  /// Ingests target multi-layer hidden states and injects K/V into the draft
  /// cache.
  bool InjectTargetContext(std::span<const float> target_features,
                           std::uint32_t position, std::uint32_t num_tokens,
                           const DFlashTrace& trace = {},
                           bool allow_aligned_prefix_skip = false);

  /// GUFO_DEV_HOSTOPS_FEAT. Appends `num_rows` target feature rows to
  /// the device staging buffer, copied device to device from the target's own
  /// capture buffer once `src_stream` has produced them. Nothing reaches the
  /// draft cache yet: staged rows stay pending exactly as the host vector's
  /// rows do, so a cancelled round can still discard them.
  [[nodiscard]] bool StageTargetFeaturesDevice(const float* device_rows,
                                               std::size_t row_stride,
                                               hipStream_t src_stream,
                                               std::uint32_t staged_rows,
                                               std::uint32_t num_rows);
  /// Injects `num_tokens` previously staged rows into the draft K/V cache.
  [[nodiscard]] bool InjectStagedTargetFeatures(std::uint32_t position,
                                                std::uint32_t num_tokens);
  /// Reads staged rows back to the host, for the snapshot path only.
  [[nodiscard]] bool ReadStagedTargetFeatures(std::uint32_t num_rows,
                                              std::vector<float>& out) const;
  [[nodiscard]] std::uint32_t StagingCapacityRows() const noexcept {
    return staging_capacity_;
  }

  [[nodiscard]] speculative::PromptCaptureWindow GetTargetPromptCaptureWindow(
      std::uint32_t segment_start, std::size_t token_count) const noexcept {
    return speculative::MakeAlignedPromptCaptureWindow(
        segment_start, token_count, history_capacity_, injection_capacity_);
  }

  /// Projects committed features together and writes each private cache at
  /// its original absolute positions. Reuses the coordinator's scalar scratch.
  static void InjectTargetContextBatch(
      std::span<const QwenDFlashContextRequest> requests);

  /// Executes non-causal parallel block diffusion drafting.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardBlock(
      tokenization::TokenId anchor_token, std::uint32_t current_pos,
      std::uint32_t draft_count, float temperature = 0.0F,
      std::span<const float> sample_uniforms = {},
      std::vector<float>* out_confidences = nullptr,
      std::vector<tokenization::TokenId>* out_candidate_ids = nullptr,
      std::vector<float>* out_candidate_probabilities = nullptr,
      const DFlashTrace& trace = {});

  /// Reads one completed drafter-logit row without changing draft state.
  [[nodiscard]] bool TryGetDraftRankStats(
      std::size_t draft_row, tokenization::TokenId target_token,
      std::uint32_t& rank, float& top1_probability) const noexcept;

  /// Shared projections with independent attention, convolution and selector
  /// boundaries. A single request retains the ordinary block execution.
  [[nodiscard]] static std::vector<speculative::DraftProposal>
  ForwardBlockBatch(std::span<const QwenDFlashBlockRequest> requests);

  [[nodiscard]] std::unique_ptr<QwenDFlashGpuSnapshot> SaveSnapshot() const;
  void RestoreSnapshot(const QwenDFlashGpuSnapshot& snapshot);
  void RestorePersistentSnapshot(std::span<const std::uint8_t> payload);
  [[nodiscard]] std::size_t StateBytes() const noexcept;
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const noexcept;
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept;
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const QwenDFlashGpuModel& model, std::uint32_t max_context,
      std::size_t max_batch_width = 1);

  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return model_->GetConfig().hidden_size;
  }
  [[nodiscard]] std::size_t GetTargetFeaturesSize() const noexcept {
    return model_->GetDFlashConfig().target_layer_ids.size() * GetHiddenSize();
  }
  [[nodiscard]] std::uint32_t GetInjectedContextLength() const noexcept {
    return injected_context_len_;
  }
  [[nodiscard]] const QwenDFlashGpuModel& GetModel() const noexcept {
    return *model_;
  }

private:
  QwenDFlashGpuExecutor(std::shared_ptr<const QwenDFlashGpuModel> model,
                        std::uint32_t max_context);

  bool InjectTargetContextChunk(std::span<const float> target_features,
                                std::uint32_t position,
                                std::uint32_t num_tokens,
                                const DFlashTrace& trace);
  // The projection, norm, RoPE and ring-write chain, reading feature
  // rows that are already on the device. The host entry uploads into
  // d_target_features_ and then calls this with that pointer; the device entry
  // passes its staging buffer, so neither duplicates the chain.
  bool InjectChunkFromDeviceFeatures(const float* device_features,
                                     std::uint32_t position,
                                     std::uint32_t num_tokens,
                                     const DFlashTrace& trace);
  void Allocate();
  void Free() noexcept;
  [[nodiscard]] static std::size_t BatchScratchBytes(
      const QwenDFlashGpuModel& model, std::size_t rows);
  void PrewarmBlockGemms();
  void RunBlockGemm(const models::QwenTensorRef& weight, const float* input,
                    float* output, std::size_t batch_size,
                    std::size_t output_size, std::size_t input_size);
  void RunInjectGemm(const models::QwenTensorRef& weight, const float* input,
                     float* output, std::size_t num_tokens,
                     std::size_t output_size, std::size_t input_size,
                     bool raw_input = false);

  std::shared_ptr<const QwenDFlashGpuModel> model_;
  EmbeddingRowStage embedding_stage_;
  std::uint32_t max_context_{4096};
  std::uint32_t injected_context_len_{0};
  std::uint32_t history_capacity_{0};
  std::uint32_t injection_capacity_{0};
  // GUFO_DEV_HOSTOPS_FEAT: pending target feature rows, device side.
  // Sized at one block (one round can stage at most draft_count + 1 rows); a
  // request for more is refused and the caller keeps the host route.
  std::uint32_t staging_capacity_{0};
  float* d_staged_features_{nullptr};
  hipEvent_t feature_ready_{nullptr};
  // GUFO_DEV_DFLASH_INJECT_WMMA: tiled Q8_1 activations for one
  // injection chunk; null when the flag is off.
  void* d_inject_q8_{nullptr};
  std::size_t inject_q8_bytes_{0};
  hipStream_t stream_{nullptr};
  hipblasHandle_t hipblas_handle_{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm_;

  // GPU Allocations for draft KV cache
  std::vector<float*> d_injected_k_;
  std::vector<float*> d_injected_v_;

  // GPU Scratch buffers
  float* d_target_features_{nullptr};
  float* d_fused_features_{nullptr};
  float* d_block_hidden_{nullptr};
  float* d_block_normed_{nullptr};
  float* d_conv_hidden_{nullptr};
  float* d_dynamic_coefficients_{nullptr};
  float* d_q_{nullptr};
  float* d_k_block_{nullptr};
  float* d_v_block_{nullptr};
  float* d_attn_out_{nullptr};
  /// GUFO_DEV_DRAFTK=2 score buffer: one f32 per (draft row, query head,
  /// attended key). Allocated only when the level is set, so the default
  /// route carries no extra bytes. Size: DFlashDraftKScoreElements.
  float* d_draftk_scores_{nullptr};
  float* d_ffn_gate_{nullptr};
  float* d_ffn_up_{nullptr};
  float* d_ffn_down_{nullptr};
  float* d_logits_{nullptr};
  /// GUFO_DEV_DRAFT_VOCAB: the cut head writes here, one row of cut.size per
  /// draft position, then scatters into d_logits_ (kept -inf elsewhere).
  float* d_draft_vocab_logits_{nullptr};
  std::size_t last_draft_logits_rows_{0};
  float* d_selector_hidden_{nullptr};
  std::uint32_t* d_selector_candidate_ids_{nullptr};
  float* d_selector_candidate_probabilities_{nullptr};
  float* d_selector_uniforms_{nullptr};
  float* d_confidences_{nullptr};
  /// One bias dump per process, step 0 only. Not a step
  /// counter — see the dump site for why the file's existence is the gate.
  bool bias_dump_written_{false};
  std::uint32_t* d_out_token_{nullptr};
  // DSpark Markov head. d_markov_prev_ holds the token the next bias row is
  // conditioned on, so the chain runs entirely on the device.
  float* d_markov_bias_{nullptr};
  float* d_markov_w1_prev_{nullptr};
  std::uint32_t* d_markov_prev_{nullptr};
  std::uint32_t* d_draft_tokens_{nullptr};
  hip_bfloat16* d_bf16_input_{nullptr};
  std::size_t scratch_bytes_{0};
  void* d_batch_scratch_{nullptr};
  std::size_t batch_scratch_bytes_{0};
};

struct QwenDFlashGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{7};
  speculative::DFlashDraftPolicy policy{
      speculative::DFlashDraftPolicy::kAdaptive};
};

/// Adapts the GPU DFlash executor to the repository's IDraftBackend speculative
/// interface.
class QwenDFlashGpuDraftBackend final : public speculative::IDraftBackend {
public:
  ~QwenDFlashGpuDraftBackend() override = default;

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend> Create(
      std::shared_ptr<const QwenDFlashGpuModel> model,
      QwenDFlashGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend>
  CreateFromGguf(std::string_view dflash_model_path,
                 std::shared_ptr<const QwenGpuModel> target_model,
                 QwenDFlashGpuDraftConfig config = {},
                 std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenDFlashGpuDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return executor_->GetModel().GetDFlashConfig().target_layer_ids;
  }

  [[nodiscard]] speculative::PromptCaptureWindow TargetPromptCaptureWindow(
      std::uint32_t segment_start,
      std::size_t token_count) const noexcept override {
    return executor_->GetTargetPromptCaptureWindow(segment_start, token_count);
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;

  /// GUFO_DEV_HOSTOPS_FEAT. The device-side twin of
  /// UpdateTargetHidden: appends `num_rows` committed target feature rows from
  /// the target's own capture buffer, without a host round trip. Returns false
  /// when this backend cannot take the rows that way, and the caller then
  /// falls back to UpdateTargetHidden.
  [[nodiscard]] bool UpdateTargetHiddenDevice(const float* device_rows,
                                              std::size_t row_stride,
                                              hipStream_t src_stream,
                                              std::uint32_t num_rows);

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;
  [[nodiscard]] speculative::DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override;
  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return true;
  }
  [[nodiscard]] std::vector<speculative::DraftProposal> ProposeBatch(
      std::span<const speculative::DraftProposalRequest> requests) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override;

  void Reset() noexcept override;
  void BeginRequest() noexcept override { controller_.Reset(); }
  /// The most recent verification has not yet injected these target rows.
  /// Discard them when the target returns to its saved verification frontier.
  void DiscardPendingTargetContext(std::uint32_t position);

  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept {
    return executor_->GetMemoryUsage();
  }

  /// How many device feature rows UpdateTargetHiddenDevice can take.
  /// Zero means the device route is unavailable and the caller must stay on
  /// UpdateTargetHidden. The verifier asks before it offers the rows.
  [[nodiscard]] std::uint32_t StagingCapacityRows() const noexcept {
    return executor_->StagingCapacityRows();
  }

private:
  void InjectPendingFeatures(std::uint32_t position);
  [[nodiscard]] std::uint32_t PendingFeatureCount(std::uint32_t position) const;
  QwenDFlashGpuDraftBackend(std::unique_ptr<QwenDFlashGpuExecutor> executor,
                            QwenDFlashGpuDraftConfig config);
  [[nodiscard]] speculative::DraftProposal ProposeImpl(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state);

  std::unique_ptr<QwenDFlashGpuExecutor> executor_;
  QwenDFlashGpuDraftConfig config_;
  speculative::DFlashLengthController controller_;
  std::vector<float> pending_target_features_;
  // Committed rows staged on the device instead of in
  // pending_target_features_. Exactly one of the two is non-empty.
  std::uint32_t pending_device_rows_{0};
  std::vector<tokenization::TokenId> proposed_tokens_;
  /// proposed_tokens_ = drafter rows, then lookup rows. Only the
  /// drafter rows feed the length controller.
  std::size_t drafter_count_{0};
  std::size_t lookup_count_{0};
  std::uint32_t lookup_position_{0};
  /// Positions below this need a long match (PR #570 cooldown).
  std::uint32_t lookup_cooldown_until_{0};
  void ResetLookupState() noexcept {
    drafter_count_ = 0;
    lookup_count_ = 0;
    lookup_position_ = 0;
    lookup_cooldown_until_ = 0;
  }
  std::uint32_t draft_len_log_position_{0};
  std::uint32_t draft_len_log_chosen_{0};
  float draft_len_log_mean_before_{0.0F};
  bool draft_len_log_pending_{false};
  // GUFO_DEV_DRAFT_CONF_LOG: the proposal's per-row confidences, for the
  // offline stop-rule fit. Printed only when it matches `drafted`.
  std::vector<float> draft_len_log_confidences_;
  std::uint32_t draft_rank_log_position_{0};
  bool draft_rank_log_pending_{false};
  /// GUFO_DEV_DRAFT_BANDIT: end of the previous round. Default-built
  /// means "no previous round", so the first round contributes no cost sample.
  std::chrono::steady_clock::time_point last_round_end_{};
  /// False when the last proposal was verified in a shared cohort, whose round
  /// time measures the cohort's row count rather than this request's width.
  bool round_cost_is_comparable_{false};
  bool primed_{false};
  bool proposal_active_{false};
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_HPP_

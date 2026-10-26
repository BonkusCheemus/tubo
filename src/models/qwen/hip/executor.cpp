#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/dev_timing.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"

namespace gufo::hip {
namespace {

const QwenGpuModel& RequireModel(
    const std::shared_ptr<const QwenGpuModel>& model) {
  if (model == nullptr) {
    throw std::invalid_argument("Qwen GPU model must not be null");
  }
  // Step2 may load Rows32 bytes for inspection, but no old raw-pointer
  // GPU reader may consume them until all reader paths are integrated.
  // This runs in the initializer list, before the arena allocates or launches.
  // Only the default-OFF dev switch admits IQ3 gate/up tags (C9 profile).
  if (const char* why = models::TaggedGpuServeRefusal(model->GetWeights()))
    throw std::invalid_argument(why);
  return *model;
}

[[nodiscard]] detail::HipGraphCaptureKey BuildGraphCaptureKey(
    const models::QwenModelWeights& weights, const QwenExecutionPolicy& policy,
    std::uint32_t max_context) noexcept {
  const auto& config = weights.config;
  std::uint64_t workload = BeginQwenGraphWorkloadIdentity();
  const std::uint64_t fields[] = {
      config.num_layers,
      config.hidden_size,
      config.intermediate_size,
      config.num_attention_heads,
      config.num_key_value_heads,
      config.head_dim,
      config.vocab_size,
      max_context,
      1U,  // Captured decode always computes logits.
  };
  for (const std::uint64_t field : fields) {
    workload = ExtendQwenGraphWorkloadIdentity(workload, field);
  }
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    const auto resolution = ResolveQwenLayerRouteWithReasons(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    workload = ExtendQwenGraphWorkloadIdentity(workload,
                                               resolution.plan.Fingerprint());
  }
  return {
      .execution_identity = policy.Fingerprint(),
      .workload_identity = workload,
  };
}

}  // namespace

QwenGpuExecutor::QwenGpuExecutor(std::shared_ptr<const QwenGpuModel> model,
                                 std::uint32_t max_context,
                                 QwenExecutionPolicy policy)
    : model_(std::move(model)),
      weights_(RequireModel(model_).GetWeights()),
      tokenizer_(&model_->GetTokenizer()),
      // C1f: resolve the per-layer table HERE, where the policy and the model
      // config meet and BEFORE arena_ allocates a plane. In the init list, not
      // the body: the arena takes policy_ by value, so a body assignment would
      // arrive one allocation too late to matter.
      policy_(ResolveQwenKvLayerTableForStorage(
          std::move(policy), weights_.config.FullAttentionLayerCount())),
      arena_(weights_.config, max_context, policy_),
      graph_key_(BuildGraphCaptureKey(weights_, policy_, max_context)),
      h_logits_(weights_.config.vocab_size, 0.0F) {
  AllocateGpuSamplingWorkspace(&sampling_workspace_, weights_.config.vocab_size,
                               max_context);
  // Every Dot4 consumer (V3/dot4 small-batch GEMMs, the IQ3 pair)
  // runs at B<=8 with k = one projection's input width. Reserve the widest on
  // arena_.stream here, before ForwardToken's first capture, so a captured
  // graph's scratch is never grown or freed (27B: 8 * 17408, the down input).
  const auto& c = weights_.config;
  ReserveDot4Scratch(8U * std::max({c.hidden_size, c.intermediate_size,
                                    c.AttentionSize(), c.ssm_inner_size}),
                     arena_.stream);
  // NormQ only fuses the dim == hidden_size norms at width <= 8, so the widest
  // call is 8 * hidden_size. No-op unless GUFO_DEV_NORMQ=1.
  ReserveNormQScratch(8U * c.hidden_size, arena_.stream);
  detail::EmitQwenExecutionPolicy(policy_.Fingerprint());
}

QwenGpuExecutor::~QwenGpuExecutor() {
  (void)hipStreamSynchronize(arena_.stream);
  for (const auto event : prefill_events_) {
    if (event)
      (void)hipEventDestroy(event);
  }
  if (d_verification_logits_ != nullptr) {
    (void)hipFree(d_verification_logits_);
  }
  if (d_verification_features_ != nullptr) {
    (void)hipFree(d_verification_features_);
  }
  FreeGpuSamplingWorkspace(&sampling_workspace_);
}

void QwenGpuExecutor::EnsureVerificationLogits(std::size_t batch_size) {
  if (batch_size <= verification_logits_capacity_) {
    return;
  }
  if (d_verification_logits_ != nullptr) {
    HIP_CHECK(hipFree(d_verification_logits_));
    d_verification_logits_ = nullptr;
  }
  verification_logits_capacity_ = 0;
  HIP_CHECK(hipMalloc(&d_verification_logits_,
                      batch_size * weights_.config.vocab_size * sizeof(float)));
  verification_logits_capacity_ = batch_size;
}

void QwenGpuExecutor::EnsureVerificationFeatures(std::size_t elements) {
  // Rows x feature_width floats, at most kMaxDecodeBatch^2 rows of
  // taps x hidden_size. Allocated only on the device-handoff route.
  if (elements <= verification_features_capacity_) {
    return;
  }
  if (d_verification_features_ != nullptr) {
    HIP_CHECK(hipFree(d_verification_features_));
    d_verification_features_ = nullptr;
  }
  verification_features_capacity_ = 0;
  HIP_CHECK(hipMalloc(&d_verification_features_, elements * sizeof(float)));
  verification_features_capacity_ = elements;
}

QwenGpuMemoryUsage QwenGpuExecutor::EstimateMemoryUsage(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy) {
  auto usage = QwenGpuArena::EstimateMemoryUsage(config, max_context, policy);
  usage.temporary_scratch_bytes +=
      EstimateGpuSamplingWorkspaceBytes(config.vocab_size, max_context);
  // Reserve the bounded fallback for eight requests with eight rows each.
  // Large arenas reuse FFN scratch and normally retain fewer rows.
  const std::size_t rows = std::min<std::size_t>(max_context, 64);
  const std::size_t features = rows * 5 * config.hidden_size;
  const std::size_t feature_rows =
      (features + config.vocab_size - 1) / config.vocab_size;
  usage.temporary_scratch_bytes +=
      std::max(rows, feature_rows) * config.vocab_size * sizeof(float);
  return usage;
}

std::size_t QwenGpuExecutor::SnapshotPayloadBytes(
    std::uint32_t valid_context) const {
  return arena_.SnapshotPayloadBytes(valid_context);
}

QwenGpuMemoryUsage QwenGpuExecutor::GetMemoryUsage() const {
  auto usage = arena_.GetMemoryUsage();
  usage.temporary_scratch_bytes += verification_logits_capacity_ *
                                   weights_.config.vocab_size * sizeof(float);
  usage.temporary_scratch_bytes +=
      verification_features_capacity_ * sizeof(float);
  usage.temporary_scratch_bytes += sampling_workspace_.SizeBytes();
  usage.temporary_scratch_bytes += vision_input_.Bytes();
  usage.temporary_scratch_bytes +=
      decode_embedding_.Bytes() + batch_embedding_.Bytes();
  return usage;
}

void QwenGpuExecutor::Reset() noexcept {
  replaying_ssm_state_ = false;
  next_token_.reset();
  prompt_hidden_capture_start_position_ = 0;
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  last_verification_features_ = nullptr;
  last_verification_feature_width_ = 0;
  try {
    arena_.Reset();
    reset_failure_ = nullptr;
  } catch (...) {
    reset_failure_ = std::current_exception();
  }
  graph_executor_.Reset();
}

std::unique_ptr<QwenGpuSnapshot> QwenGpuExecutor::SaveSnapshot(
    std::uint32_t valid_context) {
  CheckReset();
  auto snapshot = arena_.SaveSnapshot(valid_context);
  snapshot->vision_layout_ = vision_input_.layout().Prefix(valid_context);
  return snapshot;
}

void QwenGpuExecutor::ConfigureVision(
    std::shared_ptr<const models::qwen::vision::Prompt> prompt,
    std::shared_ptr<models::qwen::vision::Encoder> encoder) {
  // C1e: with no image prompt this entry configures the vision rope and nothing
  // else — it reads and writes no KV plane, which is why the measured T refusal
  // (a null prompt reaching the guard at kPrefill, at admission, before any
  // token) was a refusal about the wrong phase. A REAL image prompt keeps
  // kPrefill and stays refused, because C1d feeds vision through decode: an
  // image embedding admitted here would be silently wrong rather than refused.
  CheckReset(prompt ? QwenKvExecutionPhase::kPrefill
                    : QwenKvExecutionPhase::kNoKvAccess);
  const auto* previous = vision_input_.rope();
  if (prompt)
    prompt->rope.Validate(arena_.GetMaxContext());
  vision_input_.Configure(std::move(prompt), std::move(encoder), arena_.stream);
  if (previous != vision_input_.rope())
    graph_executor_.Reset();
}

void QwenGpuExecutor::RestoreSnapshot(const QwenGpuSnapshot& snapshot) {
  CheckReset();
  replaying_ssm_state_ = false;
  next_token_.reset();
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  last_verification_features_ = nullptr;
  last_verification_feature_width_ = 0;
  arena_.RestoreSnapshot(snapshot);
  vision_input_.RestoreLayout(snapshot.vision_layout_, arena_.stream);
  graph_executor_.Reset();
}

void QwenGpuExecutor::RestoreCompactSnapshot(
    std::span<const std::uint8_t> payload,
    std::uint32_t expected_valid_context) {
  CheckReset();
  replaying_ssm_state_ = false;
  next_token_.reset();
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  last_verification_features_ = nullptr;
  last_verification_feature_width_ = 0;
  arena_.RestoreCompactSnapshot(payload, expected_valid_context);
  vision_input_.RestoreLayout(QwenGpuSnapshot::ReadRopeLayout(payload),
                              arena_.stream);
  graph_executor_.Reset();
}

void QwenGpuExecutor::SaveState(std::uint32_t valid_context,
                                bool allow_divert) {
  // The opt-in divert is upstream #650's default now: every SaveState
  // pins the live buffer and the next pass writes the other one, so the flag
  // no longer changes anything. Kept for the SpeculativeTarget interface.
  (void)allow_divert;
  CheckReset();
  replaying_ssm_state_ = false;
  arena_.SaveState(valid_context);
  if (arena_.BeginSsmReplayCapture()) {
    graph_executor_.Reset();
  }
}

void QwenGpuExecutor::RestoreState() {
  CheckReset();
  if (arena_.IsSsmReplayCaptureActive())
    arena_.DisableSsmReplayCapture();
  arena_.RestoreState();
  replaying_ssm_state_ = true;
}

void QwenGpuExecutor::FinishVerification(bool commit_state) {
  CheckReset();
  arena_.DisableSsmReplayCapture();
  // The commit swap is gone with upstream #650: the pass already wrote
  // the live buffer, so there is nothing left to commit.
  (void)commit_state;
}

void QwenGpuExecutor::ReplaySsmState(std::uint32_t position,
                                     std::uint32_t count) {
  const auto& config = weights_.config;
  auto scratch = arena_.GetScratchView();
  // gaps + k95round (same hook, merged): the replay is issued here and nothing
  // waits on arena_.stream before the next verification batch, so these
  // launches are expected to run under the drafter's
  // inject/draft_forward/markov_confidence on the other stream. Hidden and
  // exposed cost the same on the host clock, so only this event pair can tell
  // them apart; its span is the replay's GPU time, and whether it is on the
  // round's critical path shows as a change in the round wall when the replay
  // is removed, not here.
  ::gufo::core::hip::dev_timing::EventScope ssm_replay_timer(
      ::gufo::core::hip::dev_timing::Phase::kSsmReplay, arena_.stream);
  // The replay is one pass: it reads the restored (pinned) state and leaves it
  // intact for a second RestoreState.
  const auto recurrent_state = arena_.BeginRecurrentWrite();
  for (std::uint32_t layer_idx = 0; layer_idx < config.num_layers;
       ++layer_idx) {
    const auto& layer = weights_.layers[layer_idx];
    if (layer.is_full_attention) {
      continue;
    }

    // Layers have independent recorded inputs and recurrent state. Replay each
    // layer's committed rows together, splitting only at ring/scratch bounds;
    // a layer's later splits continue from what its first launch wrote.
    auto layer_state = recurrent_state;
    for (std::uint32_t offset = 0; offset < count;) {
      const auto start = position + offset;
      const auto until_wrap = static_cast<std::uint32_t>(
          kSsmReplayCapacity - (start % kSsmReplayCapacity));
      const auto rows =
          std::min({count - offset, until_wrap, arena_.GetMaxBatch()});
      LaunchSSMConvRecurrenceRows(
          arena_.GetReplayQkv(layer_idx, start),
          static_cast<const float*>(layer.ssm_conv1d.data), layer_state,
          scratch.ssm.conv_out.data(), arena_.GetReplayAlpha(layer_idx, start),
          arena_.GetReplayBeta(layer_idx, start),
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data), nullptr, nullptr,
          nullptr, config.SsmLayerIndex(layer_idx), config.SsmQkvSize(),
          config.ssm_group_count, config.ssm_time_step_rank,
          config.ssm_state_size, config.SsmValueSize(), rows,
          config.ssm_time_step_rank, config.ssm_inner_size, arena_.stream, {},
          arena_.GetRecurrentStateStorage());
      layer_state = layer_state.Continued();
      offset += rows;
    }
  }
}

std::span<const float> QwenGpuExecutor::CopyLastLogits() {
  // Classified DECODE readback: the buffer a decode step just produced.
  CheckReset(QwenKvExecutionPhase::kDecode);
  auto scratch = arena_.GetScratchView();
  HIP_CHECK(hipMemcpyAsync(h_logits_.data(), scratch.decode.logits.data(),
                           h_logits_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_logits_;
}

tokenization::TokenId QwenGpuExecutor::SampleLastLogits(
    sampling::SamplerState& sampler) {
  // Classified DECODE readback: sampling a decode step's frontier.
  CheckReset(QwenKvExecutionPhase::kDecode);
  if (sampler.NeedsConstraintMask())
    return sampler.Sample(CopyLastLogits());
  auto parameters = PrepareGpuSamplingParameters(sampler);
  if (sampler.config().uses_random_sampling()) {
    parameters.uniform = sampler.Uniform();
  }

  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSampling(scratch.decode.logits.data(), d_out_token,
                    weights_.config.vocab_size, parameters,
                    sampler.penalties().data(), sampler.penalties().size(),
                    &sampling_workspace_, arena_.stream);

  tokenization::TokenId token = 0;
  HIP_CHECK(hipMemcpyAsync(&token, d_out_token, sizeof(token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return CheckedSampleToken(token, weights_.config.vocab_size);
}

tokenization::TokenId QwenGpuExecutor::SampleCachedLogits(
    std::span<const float> logits, sampling::SamplerState& sampler) {
  // Classified DECODE readback: sampling a frontier retained from decode.
  CheckReset(QwenKvExecutionPhase::kDecode);
  if (logits.size() != weights_.config.vocab_size)
    throw std::invalid_argument(
        "cached Qwen frontier has the wrong vocabulary");
  if (sampler.NeedsConstraintMask())
    return sampler.Sample(logits);
  auto scratch = arena_.GetScratchView();
  HIP_CHECK(hipMemcpyAsync(scratch.decode.logits.data(), logits.data(),
                           logits.size_bytes(), hipMemcpyHostToDevice,
                           arena_.stream));
  return SampleLastLogits(sampler);
}

GpuSamplingParameters QwenGpuExecutor::PrepareGpuSamplingParameters(
    const sampling::SamplerState& sampler) {
  const auto& config = sampler.config();

  return GpuSamplingParameters{
      .temperature = config.temperature,
      .top_k = config.top_k,
      .top_p = config.top_p,
      .min_p = config.min_p,
      .min_keep = config.min_keep,
      .repeat_penalty = config.repeat_penalty,
      .frequency_penalty = config.frequency_penalty,
      .presence_penalty = config.presence_penalty,
  };
}

tokenization::TokenId QwenGpuExecutor::SampleVerificationLogits(
    std::size_t row, sampling::SamplerState& sampler) {
  CheckReset();
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  if (sampler.NeedsConstraintMask())
    return sampler.Sample(CopyVerificationLogits(row));
  auto parameters = PrepareGpuSamplingParameters(sampler);
  if (sampler.config().uses_random_sampling()) {
    parameters.uniform = sampler.Uniform();
  }
  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSampling(d_verification_logits_ + (row * weights_.config.vocab_size),
                    d_out_token, weights_.config.vocab_size, parameters,
                    sampler.penalties().data(), sampler.penalties().size(),
                    &sampling_workspace_, arena_.stream);

  tokenization::TokenId token = 0;
  HIP_CHECK(hipMemcpyAsync(&token, d_out_token, sizeof(token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return CheckedSampleToken(token, weights_.config.vocab_size);
}

QwenSampledVerificationResult QwenGpuExecutor::VerifySampledToken(
    std::size_t row, tokenization::TokenId draft_token,
    std::span<const tokenization::TokenId> draft_candidate_ids,
    std::span<const float> draft_candidate_probabilities,
    double draft_token_probability, sampling::SamplerState& sampler) {
  CheckReset();
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  if (draft_candidate_ids.empty() ||
      draft_candidate_ids.size() != draft_candidate_probabilities.size() ||
      !std::isfinite(draft_token_probability) ||
      draft_token_probability <= 0.0 || draft_token_probability > 1.0) {
    throw std::invalid_argument(
        "Qwen sampled verification proposal is malformed");
  }

  if (sampler.NeedsConstraintMask()) {
    const auto target = sampler.Distribution(CopyVerificationLogits(row));
    if (sampler.Uniform() * draft_token_probability <
        target.probability(draft_token)) {
      sampler.Accept(draft_token);
      return {.token = draft_token, .accepted = true};
    }
    return {.token = target.SampleResidual(draft_candidate_ids,
                                           draft_candidate_probabilities,
                                           sampler.mutable_rng_state()),
            .accepted = false};
  }

  auto parameters = PrepareGpuSamplingParameters(sampler);
  // DEV GUFO_DEV_LOOSE_ACCEPT=lambda (idea: a close drafted word is
  // worth trying): LOSSY. The device accepts when
  // u * q(x) < p(x); passing u / lambda accepts with min(1, lambda * p/q), so a
  // draft token the target finds plausible but less likely than the drafter did
  // is kept more often (FLy / ASD family). lambda = 1 or unset
  // = exact rejection sampling. The residual draw is unchanged.
  // ponytail: device path only; the constraint-mask and host paths stay exact.
  static const double loose_accept = [] {
    const char* v = std::getenv("GUFO_DEV_LOOSE_ACCEPT");
    if (v == nullptr || v[0] == '\0') return 1.0;
    char* end = nullptr;
    const double l = std::strtod(v, &end);
    if (end == v || *end != '\0' || !(l > 1.0) || !std::isfinite(l)) {
      std::fprintf(stderr, "GUFO_DEV_LOOSE_ACCEPT='%s' ignored (exact verify)\n", v);
      return 1.0;
    }
    std::fprintf(stderr, "GUFO_DEV_LOOSE_ACCEPT=%.3f active (lossy verify)\n", l);
    return l;
  }();
  const double acceptance_uniform = sampler.Uniform() / loose_accept;
  const std::uint64_t residual_rng_checkpoint = sampler.rng_state();
  const double residual_uniform = sampler.Uniform();

  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSpeculativeSampling(
      d_verification_logits_ + (row * weights_.config.vocab_size), d_out_token,
      sampling_workspace_.speculative_accepted, weights_.config.vocab_size,
      parameters, draft_token, static_cast<float>(draft_token_probability),
      draft_candidate_ids.data(), draft_candidate_probabilities.data(),
      draft_candidate_ids.size(), acceptance_uniform, residual_uniform,
      sampler.penalties().data(), sampler.penalties().size(),
      &sampling_workspace_, arena_.stream);

  QwenSampledVerificationResult result;
  std::uint32_t accepted = 0;
  HIP_CHECK(hipMemcpyAsync(&result.token, d_out_token, sizeof(result.token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipMemcpyAsync(&accepted, sampling_workspace_.speculative_accepted,
                           sizeof(accepted), hipMemcpyDeviceToHost,
                           arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  if (result.token >= weights_.config.vocab_size || accepted > 1)
    throw std::runtime_error(
        "target logit distribution contains no finite values");
  result.accepted = accepted != 0U;
  if (result.accepted) {
    // Lossless rejection sampling consumes a residual draw only when the
    // proposal is rejected. The GPU receives it eagerly so acceptance remains
    // a single synchronization, then restores the exact host RNG state here.
    sampler.SetRngState(residual_rng_checkpoint);
    sampler.Accept(draft_token);
  }
  return result;
}

std::span<const float> QwenGpuExecutor::CopyVerificationLogits(
    std::size_t row) {
  CheckReset();
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  HIP_CHECK(hipMemcpyAsync(
      h_logits_.data(),
      d_verification_logits_ + (row * weights_.config.vocab_size),
      h_logits_.size() * sizeof(float), hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_logits_;
}

void QwenGpuExecutor::CopyVerificationLogitsInto(std::size_t row,
                                                 std::vector<float>& out) {
  // Same transfer as CopyVerificationLogits, landing in the caller's
  // retained vector. The old route copied into the shared h_logits_ staging
  // vector and the caller then assigned it out, so every retained frontier
  // paid a second vocabulary-sized host memcpy and a fresh allocation.
  CheckReset();
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  const std::size_t vocab_size = weights_.config.vocab_size;
  out.resize(vocab_size);
  HIP_CHECK(hipMemcpyAsync(out.data(),
                           d_verification_logits_ + (row * vocab_size),
                           vocab_size * sizeof(float), hipMemcpyDeviceToHost,
                           arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
}

void QwenGpuExecutor::SetPromptHiddenCapture(
    bool enabled, std::span<const std::uint32_t> target_layer_ids,
    std::uint32_t capture_start_position) {
  capture_prompt_hidden_ = enabled;
  prompt_hidden_capture_start_position_ =
      enabled ? capture_start_position : 0;
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  arena_.SetTargetLayerCapture(enabled ? target_layer_ids
                                       : std::span<const std::uint32_t>{});
}

std::span<const float> QwenGpuExecutor::CopyLastHidden() {
  // Classified DECODE readback: the hidden state a decode step just produced.
  CheckReset(QwenKvExecutionPhase::kDecode);
  const std::size_t hidden_size = weights_.config.hidden_size;
  const std::size_t target_layer_count = arena_.GetTargetLayerCapture().size();
  if (target_layer_count > 0) {
    h_last_hidden_.resize(target_layer_count * hidden_size);
    HIP_CHECK(hipMemcpyAsync(h_last_hidden_.data(),
                             arena_.d_target_layer_features,
                             h_last_hidden_.size() * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
  } else {
    h_last_hidden_.resize(hidden_size);
    auto scratch = arena_.GetScratchView(arena_.GetMaxBatch());
    HIP_CHECK(hipMemcpyAsync(h_last_hidden_.data(),
                             scratch.decode.hidden.data() + last_hidden_offset_,
                             hidden_size * sizeof(float), hipMemcpyDeviceToHost,
                             arena_.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_last_hidden_;
}

std::vector<tokenization::TokenId> QwenGpuExecutor::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  next_token_.reset();
  Reset();
  return GenerateFromPrefix(prompt_tokens, 0, options, on_token);
}

std::vector<tokenization::TokenId> QwenGpuExecutor::GenerateFromPrefix(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::size_t cached_prefix_tokens, const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  CheckReset();
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }
  if (cached_prefix_tokens > prompt_tokens.size()) {
    throw std::invalid_argument(
        "cached Qwen prefix exceeds the rendered prompt");
  }

  tokenization::TokenId next_token = 0;
  if (cached_prefix_tokens < prompt_tokens.size()) {
    next_token =
        ForwardPromptBatch(prompt_tokens.subspan(cached_prefix_tokens),
                           static_cast<std::uint32_t>(cached_prefix_tokens));
  } else if (next_token_.has_value()) {
    next_token = *next_token_;
  } else {
    throw std::logic_error("Qwen retained prefix has no next-token frontier");
  }

  std::size_t cur_pos = prompt_tokens.size();
  sampling::SamplerState sampler(options.sampling, prompt_tokens);
  if (!options.sampling.can_use_unmodified_argmax()) {
    next_token = SampleLastLogits(sampler);
  }

  // 2. Auto-regressive decode generation loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (tokenizer_->IsStopToken(next_token)) {
      break;
    }

    output_tokens.push_back(next_token);
    sampler.Accept(next_token);
    if (on_token) {
      const auto piece = tokenizer_->DecodeToken(next_token);
      if (!on_token(next_token, piece)) {
        break;
      }
    }

    next_token = ForwardToken(next_token, static_cast<std::uint32_t>(cur_pos));
    if (!options.sampling.can_use_unmodified_argmax()) {
      next_token = SampleLastLogits(sampler);
    }
    ++cur_pos;
  }

  next_token_ = next_token;
  return output_tokens;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

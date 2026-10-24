#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip::kernels {

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream);

/// One DSpark block position: bias `logits` with
/// `bias[c] = sum_r W2[c, r] * W1[*previous_token, r]` (the arithmetic
/// specification is speculative::ComputeMarkovBiasRow), write that row of
/// `markov_w1` into `w1_prev`, add the bias to `logits` in place, then take the
/// token. `token` is the argmax when `temperature <= 0` and one inverse-CDF
/// draw otherwise; `confidence` is 1 for the greedy path and the sampled
/// softmax probability otherwise. `confidence_projection` is F32 of width
/// `hidden_size + markov_rank` over the concatenated [hidden ; w1_prev].
void LaunchDFlashMarkovStep(
    const void* markov_w1, const void* markov_w2,
    const std::uint32_t* previous_token, float* bias, float* w1_prev,
    float* logits, const float* hidden, const void* confidence_projection,
    const void* confidence_bias, float* confidence, std::uint32_t* token,
    std::uint32_t vocab_size, std::uint32_t markov_rank,
    std::uint32_t hidden_size, float temperature,
    const float* sample_uniform, hipStream_t stream);

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream);

void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Thread-per-key reference route of the call above. Exposed so an equivalence
/// test can run both routes in one process; the production entry point picks
/// between them.
void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Wave-per-key route of the call above. Requires `head_dim` to be a multiple
/// of four so a K row can be read as `float4`.
void LaunchDFlashNonCausalAttentionWave(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity);

/// Quantizes a BF16 weight matrix into a Q8_0 copy for the draft-only LM head.
/// `total_elements` must be a multiple of the Q8_0 block size.
void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream);

[[nodiscard]] std::size_t DFlashSelectorScratchElements(
    std::uint32_t vocab_size) noexcept;

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream);

struct DFlashSelectorSequence {
  std::uint32_t count;
  float temperature;
  /// GUFO_DEV_DRAFT_CONF: greedy rows report the chosen path's top-k softmax
  /// weight instead of 1.0. Sampled rows always report their probability.
  std::uint32_t greedy_confidence{0};
};

/// GUFO_DEV_DRAFT_VOCAB: logits[r * vocab + ids[j]] = cut_logits[r * cut + j].
void LaunchDFlashScatterDraftLogits(const float* cut_logits,
                                    const std::uint32_t* ids, float* logits,
                                    std::uint32_t rows, std::uint32_t cut_size,
                                    std::uint32_t vocab_size,
                                    hipStream_t stream);

/// An optional Markov transition bias added to each selector
/// candidate, `scale * sum_r W2[candidate, r] * W1[previous, r]` -- the DSpark
/// bias (speculative::ComputeMarkovBiasRow) restricted to the top-k the
/// selector reranks, so only k + 1 rank-length rows are read per step instead
/// of all of W2. Both matrices are BF16 [vocab, rank], rank contiguous. A null
/// `w1` is the old selector exactly.
struct DFlashMarkovRerank {
  const void* w1{nullptr};
  const void* w2{nullptr};
  std::uint32_t rank{0};
  float scale{0.0F};
};

// Proposal rows are contiguous by sequence. Each token sequence includes its
// anchor before the proposal slots; all other buffers contain proposals only.
// Partial buffers need DFlashSelectorScratchElements(vocab_size) per proposal.
// Requested candidates include one-hot probabilities for greedy sequences.
void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream, DFlashMarkovRerank markov = {});

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_DFLASH_KERNELS_HPP_

#ifndef GUFO_MODELS_QWEN_DFLASH_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN_DFLASH_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::speculative {

/// Configuration for a DFlash-2 block-diffusion drafting model.
struct QwenDFlashConfig {
  /// Zero-based target layer outputs from which hidden representations are
  /// extracted.
  std::vector<std::uint32_t> target_layer_ids;

  /// Total diffusion block size, including the committed anchor token.
  std::uint32_t block_size{0};

  /// Diffusion mask token ID.
  tokenization::TokenId mask_token_id{0};

  /// Dynamic causal-convolution kernel width and channel grouping.
  std::uint32_t conv_kernel_size{0};
  std::uint32_t conv_group_size{0};

  /// Low-rank candidate selector topology. DFlash-2 only; DSpark carries a
  /// Markov head instead and leaves these zero.
  std::uint32_t selector_rank{0};
  std::uint32_t selector_top_k{0};

  /// DSpark = DFlash backbone + a semi-autoregressive Markov head and a
  /// confidence head. Set by the loader from `markov_w1.weight`; the GPU
  /// executor never reads it and discriminates on the tensors instead
  /// (IsDSparkDraft).
  bool is_dspark{false};
  /// Vanilla Markov rank: `markov_w1`/`markov_w2` are [markov_rank, vocab].
  std::uint32_t markov_rank{0};
  /// `dflash.has_confidence_head`; the confidence projection is optional.
  bool has_confidence_head{false};

  /// `dflash.sample_from_anchor`, default true when the key is absent, exactly
  /// as the reference reads it (src/models/dflash.cpp:316-318). True drafts the
  /// anchor slot itself, so the Markov chain starts at block position 0; false
  /// treats slot 0 as a bonus anchor and the chain starts at position 1.
  bool sample_from_anchor{true};

  /// `dflash.attention.layer_norm_rms_epsilon`. The draft RMSNorm kernels take
  /// this value as a literal (see kDraftRmsEps in dflash_weights.cpp), so the
  /// loader asserts equality instead of threading a field through the kernels.
  float rms_eps{0.0F};

  /// Attention topology. DFlash-2 is non-causal inside the current block.
  std::uint32_t sliding_window{0};

  /// Per-layer attention window taken from
  /// `<arch>.attention.sliding_window_pattern`. One entry per layer, in layer
  /// order; `true` means the layer uses the sliding window above and `false`
  /// means unbounded (full) attention, so an entry is the window value or 0.
  /// Empty means the artifact declared no pattern and every layer is treated
  /// as sliding-window, which is what the kernels did before the key was read.
  std::vector<std::uint32_t> layer_sliding_windows;

  /// Number of transformer blocks in the draft model.
  std::uint32_t num_layers{0};
};

/// Attention window for one layer. 0 means unbounded (full) attention, and is
/// what the draft attention kernels take for an unconstrained layer.
[[nodiscard]] std::uint32_t LayerSlidingWindow(
    const QwenDFlashConfig& config, std::uint32_t layer) noexcept;

/// Largest window across the layers, or 0 when every layer is unbounded. Used
/// for KV history capacity: a sliding layer needs `window` of history, an
/// unbounded layer needs all of it.
[[nodiscard]] std::uint32_t MaxSlidingWindow(
    const QwenDFlashConfig& config) noexcept;

/// DFlash-2 transformer weights plus the layer-local dynamic convolutions.
struct QwenDFlashLayerWeights {
  models::QwenLayerWeights transformer;

  /// [hidden_size, conv_kernel_size, 2]
  models::QwenTensorRef attention_conv_base;
  models::QwenTensorRef ffn_conv_base;

  /// [hidden_size, 2 * conv_kernel_size * (hidden_size / conv_group_size)]
  models::QwenTensorRef attention_conv_projection;
  models::QwenTensorRef ffn_conv_projection;
};

/// Immutable weights for DFlash-2 models.
struct QwenDFlashWeights {
  core::ModelConfig config;
  QwenDFlashConfig dflash_config;

  // Feature fusion encoder: compresses multi-layer target hidden states
  models::QwenTensorRef
      fc_projection;  // [n_target_layers * hidden_size, draft_hidden_size]
  models::QwenTensorRef fc_norm;  // [draft_hidden_size] RMSNorm

  // Shared / tied target embeddings and LM head
  models::QwenTensorRef token_embedding;
  models::QwenTensorRef output_norm;
  models::QwenTensorRef output;

  // Draft transformer decoder layers and layer-local dynamic convolutions.
  std::vector<QwenDFlashLayerWeights> layers;

  // DFlash-2 low-rank candidate path selector.
  models::QwenTensorRef selector_predecessor;  // [selector_rank, vocab_size]
  models::QwenTensorRef selector_successor;    // [selector_rank, vocab_size]
  models::QwenTensorRef selector_hidden;       // [hidden_size, selector_rank]

  // DSpark semi-autoregressive Markov head. Both matrices are [markov_rank,
  // vocab_size] and the bias is W2 * W1[previous token id].
  models::QwenTensorRef markov_w1;
  models::QwenTensorRef markov_w2;

  // DSpark confidence head: sigmoid(conf_proj . [hidden ; markov_w1[prev]] + b).
  // Stored F32 even when the artifact ships F16; `conf_projection_data` owns
  // the converted copy in that case.
  models::QwenTensorRef conf_projection;  // [hidden_size + markov_rank]
  models::QwenTensorRef conf_bias;        // [1]
  std::vector<float> conf_projection_data;

  // A Markov bias borrowed from another artifact
  // (GUFO_DEV_DFLASH_MARKOV) that a DFlash-2 draft adds to its selector top-k.
  // Kept apart from markov_w1/markov_w2 because their presence IS the DSpark
  // mode (IsDSparkDraft). Device BF16 [vocab, rank] once packed; empty = off.
  models::QwenTensorRef rerank_markov_w1;
  models::QwenTensorRef rerank_markov_w2;
  std::uint32_t rerank_markov_rank{0};
  float rerank_markov_scale{1.0F};

  [[nodiscard]] static std::optional<QwenDFlashWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Which drafting head a loaded draft model carries, decided the way the
/// reference decides it: by tensor presence, never by a flag. Strix creates the
/// conv/selector tensors only under `if (selector_meta)`
/// (src/models/dflash.cpp:138, :249) and dispatches the Markov head under
/// `if (model.dspark_markov_w1)` (:843), so an absent tensor IS the mode.
enum class DFlashDraftHead : std::uint8_t {
  /// Convolution + candidate selector tensors present.
  kDFlash2,
  /// Markov + confidence tensors present, no convolutions.
  kDSpark,
  /// Convolution tensors without the selector, the selector without the
  /// convolutions, a half-loaded Markov head, or no draft head at all.
  kIncomplete,
};

[[nodiscard]] DFlashDraftHead ClassifyDFlashDraftHead(
    const QwenDFlashWeights& weights) noexcept;

/// True when the draft carries the DSpark Markov head. Equivalent to
/// `ClassifyDFlashDraftHead(...) == kDSpark` but reads only the Markov tensor,
/// because a half-loaded artifact must classify as DSpark and be refused rather
/// than drafted as DFlash-2.
[[nodiscard]] bool IsDSparkDraft(const QwenDFlashWeights& weights) noexcept;

/// Why the Qwen DFlash GPU executor cannot serve this drafter, or an empty
/// string when it can. Callers MUST run this before constructing a
/// `QwenDFlashGpuExecutor`: the executor divides `hidden_size` by
/// `conv_group_size`, and a refused artifact must never reach that division.
///
/// This is the single copy of every DFlash/DSpark drafter refusal message; the
/// CLI routes surface the string it returns rather than repeating the text.
/// It refuses a MIXED artifact (some convolutions, or convolutions without the
/// selector, or a Markov head without its confidence projection), because the
/// executor's discriminator is tensor emptiness and a half-migrated artifact
/// would otherwise load and then draft as the wrong head.
[[nodiscard]] std::string DFlashGpuExecutorSupportRefusal(
    const QwenDFlashWeights& weights);

/// Host-side guard for `LaunchDFlashGroupedDynamicConv`, which divides
/// `hidden_size` by `group_size` on the device (hip/kernels/dflash_kernels.hip
/// :78-79). A zero group size is undefined behaviour there, not a clean error,
/// and DSpark never launches the kernel at all. Non-empty means refuse.
[[nodiscard]] std::string DFlashGroupedConvLaunchRefusal(
    std::size_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size);

/// Per-token dynamic-convolution coefficients, or 0 for a draft without
/// convolutions. Never divides by `conv_group_size`, so the DSpark zeros the
/// loader leaves there cannot fault.
[[nodiscard]] std::size_t DFlashDynamicCoefficientCount(
    std::size_t hidden_size, std::uint32_t conv_kernel_size,
    std::uint32_t conv_group_size) noexcept;

/// Scratch elements one draft executor allocates, matching `Allocate()` and
/// `EstimateMemoryUsage()` element for element. `markov_head` is the caller's
/// IsDSparkDraft answer: a DSpark draft reserves the Markov head's bias row,
/// `markov_w1` row, predecessor id and emitted token row instead of the
/// convolution coefficients and the selector buffers, and never touches
/// conv_group_size. `selector_scratch_elements` is the selector partial width
/// and must be 0 when `markov_head` is true.
[[nodiscard]] std::size_t DFlashDraftScratchElements(
    const core::ModelConfig& config, const QwenDFlashConfig& draft,
    std::uint32_t max_context, std::size_t selector_scratch_elements,
    bool markov_head) noexcept;

/// Which block rows one DSpark Markov chain drafts, and where it starts.
struct DSparkChainRange {
  /// Block rows the chain produces a token for.
  std::size_t rows{0};
  /// First block row the chain reads; the chain never writes before it.
  std::size_t first_row{0};
};

/// Chain geometry for a DSpark proposal of `draft_count` tokens over a block
/// of `block_count` rows (the executor's block is always `draft_count + 1`, the
/// committed anchor plus the proposal slots).
///
/// `dflash.sample_from_anchor` decides where the chain STARTS, exactly as the
/// reference does (src/models/dflash.cpp:316-318): true drafts the anchor slot
/// itself and starts at row 0, false treats row 0 as a bonus anchor and starts
/// at row 1. It never decides how MANY tokens the caller receives. The
/// verifier contract is `draft_count` (QwenDFlashGpuDraftBackend::Propose
/// passes the count it asked for and the verifier scores what comes back), so
/// `rows` is capped at `draft_count` here, AFTER the anchor has been handled:
/// an anchor-predicted block has `draft_count + 1` rows and must still yield
/// `draft_count` tokens.
///
/// The reference agrees by construction rather than by a second cap: for the
/// same `n_max` it builds a block of `n_draft` tokens, not `n_draft + 1`
/// (common/speculative.cpp:1297) and the head emits `block_drafts` columns
/// (src/models/dflash.cpp:323, :348), so its proposal is exactly `n_max` long.
[[nodiscard]] DSparkChainRange DSparkChainRangeFor(
    std::size_t block_count, std::size_t draft_count,
    bool sample_from_anchor) noexcept;

/// One DSpark Markov transition bias row for the token that precedes the
/// drafted position: `bias[c] = sum_r W2[c, r] * W1[previous, r]`, accumulated
/// in F32, for `c` in [0, vocab_size). Both matrices are [markov_rank, vocab]
/// with the rank contiguous, so the row for token `t` starts at element
/// `t * markov_rank` -- the reference takes the same row with
/// `ggml_get_rows(w1, prev)` and a rank product by w2
/// (src/models/dflash.cpp:349-350).
///
/// Returns nullopt, never a partial row, when the artifact is not DSpark,
/// either tensor is absent, its element count is not rank * vocab, the types
/// are unsupported, or `previous` is outside the draft vocabulary: a previous
/// token past the end would read another matrix's bytes.
[[nodiscard]] std::optional<std::vector<float>> ComputeMarkovBiasRow(
    const QwenDFlashWeights& weights, tokenization::TokenId previous);

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_DFLASH_WEIGHTS_HPP_

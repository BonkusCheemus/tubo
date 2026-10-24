#include "src/models/qwen/dflash_weights.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::speculative {
namespace {

std::uint32_t LayerSlidingWindowImpl(const QwenDFlashConfig& config,
                                     std::uint32_t layer) noexcept {
  if (config.layer_sliding_windows.empty()) {
    return config.sliding_window;
  }
  return layer < config.layer_sliding_windows.size()
             ? config.layer_sliding_windows[layer]
             : 0U;
}

std::uint32_t MaxSlidingWindowImpl(const QwenDFlashConfig& config) noexcept {
  std::uint32_t widest = 0;
  for (std::uint32_t layer = 0; layer < config.num_layers; ++layer) {
    const std::uint32_t window = LayerSlidingWindowImpl(config, layer);
    if (window > widest) {
      widest = window;
    }
  }
  return widest;
}

/// RMSNorm epsilon the draft kernels are compiled with. Asserted against the
/// artifact rather than plumbed through: hip/dflash/executor.cpp:840, 979,
/// 1033, 1070 and hip/dflash/batch.cpp:395, 448, 473 pass 1e-6F literally.
constexpr float kDraftRmsEps = 1e-6F;

models::QwenTensorRef TensorRef(const core::GgufReader& reader,
                                std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address < region_address) {
      continue;
    }
    const auto offset = tensor_address - region_address;
    if (offset < region.size) {
      available_bytes = region.size - offset;
      break;
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

[[nodiscard]] bool IsSupportedMatrixType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ6_K ||
         // Community low-bit drafters (Anbeeld Q2_K/Q3_K_M) also
         // quantize the selector; PackMatrixBf16 dequantizes these on the host.
         type == core::GgmlType::kQ2_K || type == core::GgmlType::kQ3_K ||
         type == core::GgmlType::kQ5_K || type == core::GgmlType::kIQ4_XS;
}

[[nodiscard]] bool IsSupportedVectorType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
}

[[nodiscard]] bool BindRequiredTensor(const core::GgufReader& reader,
                                      std::string_view name,
                                      std::size_t expected_elements,
                                      bool matrix,
                                      models::QwenTensorRef& destination,
                                      std::string* error_msg) {
  destination = TensorRef(reader, name);
  if (destination.empty()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required tensor '" +
                            std::string(name) + "'");
    return false;
  }
  if (destination.num_elements != expected_elements) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) + "' has " +
                            std::to_string(destination.num_elements) +
                            " elements; expected " +
                            std::to_string(expected_elements));
    return false;
  }
  const bool supported = matrix ? IsSupportedMatrixType(destination.type)
                                : IsSupportedVectorType(destination.type);
  if (!supported) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' has unsupported type " +
                            std::string(core::ToString(destination.type)));
    return false;
  }
  if (!destination.FitsAvailableStorage()) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' exceeds its mapped GGUF storage");
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::uint32_t> RequiredUint32(
    const core::GgufReader& reader, std::string_view key,
    std::string* error_msg) {
  auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required metadata '" +
                            std::string(key) + "'");
  }
  return value;
}

[[nodiscard]] std::optional<std::vector<std::uint64_t>>
GetNonnegativeIntegerArray(const core::GgufReader& reader,
                           std::string_view key) {
  const auto* metadata = reader.FindMetadata(key);
  if (metadata == nullptr) {
    return std::nullopt;
  }
  if (const auto* values =
          std::get_if<std::vector<std::uint64_t>>(&metadata->value)) {
    return *values;
  }
  if (const auto* values =
          std::get_if<std::vector<std::int64_t>>(&metadata->value)) {
    std::vector<std::uint64_t> converted;
    converted.reserve(values->size());
    for (const std::int64_t value : *values) {
      if (value < 0) {
        return std::nullopt;
      }
      converted.push_back(static_cast<std::uint64_t>(value));
    }
    return converted;
  }
  return std::nullopt;
}

}  // namespace

namespace {

struct ConvTensorPresence {
  std::size_t present{0};
  std::size_t expected{0};
};

[[nodiscard]] ConvTensorPresence CountConvTensors(
    const QwenDFlashWeights& weights) noexcept {
  ConvTensorPresence counts;
  for (const auto& layer : weights.layers) {
    const models::QwenTensorRef* const tensors[]{
        &layer.attention_conv_base, &layer.attention_conv_projection,
        &layer.ffn_conv_base, &layer.ffn_conv_projection};
    for (const auto* tensor : tensors) {
      ++counts.expected;
      counts.present += tensor->empty() ? 0U : 1U;
    }
  }
  return counts;
}

[[nodiscard]] std::size_t CountSelectorTensors(
    const QwenDFlashWeights& weights) noexcept {
  const models::QwenTensorRef* const tensors[]{
      &weights.selector_predecessor, &weights.selector_successor,
      &weights.selector_hidden};
  std::size_t present = 0;
  for (const auto* tensor : tensors) {
    present += tensor->empty() ? 0U : 1U;
  }
  return present;
}

[[nodiscard]] std::size_t CountMarkovTensors(
    const QwenDFlashWeights& weights) noexcept {
  const models::QwenTensorRef* const tensors[]{
      &weights.markov_w1, &weights.markov_w2, &weights.conf_projection};
  std::size_t present = 0;
  for (const auto* tensor : tensors) {
    present += tensor->empty() ? 0U : 1U;
  }
  return present;
}

}  // namespace

DFlashDraftHead ClassifyDFlashDraftHead(
    const QwenDFlashWeights& weights) noexcept {
  const ConvTensorPresence conv = CountConvTensors(weights);
  const std::size_t selector = CountSelectorTensors(weights);
  const std::size_t markov = CountMarkovTensors(weights);
  const bool conv_complete =
      conv.expected != 0U && conv.present == conv.expected;
  const bool conv_absent = conv.present == 0U;
  const bool selector_present = selector != 0U;
  const bool markov_present = markov != 0U;

  if (conv_complete && selector_present && !markov_present) {
    return DFlashDraftHead::kDFlash2;
  }
  if (conv_absent && !selector_present && markov == 3U) {
    return DFlashDraftHead::kDSpark;
  }
  return DFlashDraftHead::kIncomplete;
}

bool IsDSparkDraft(const QwenDFlashWeights& weights) noexcept {
  // The Markov matrix alone decides, so a half-loaded DSpark artifact is still
  // recognised as DSpark and refused rather than drafted as DFlash-2.
  return !weights.markov_w1.empty() || !weights.markov_w2.empty();
}

std::string DFlashGpuExecutorSupportRefusal(const QwenDFlashWeights& weights) {
  const auto& config = weights.config;
  const auto& draft = weights.dflash_config;
  const ConvTensorPresence conv = CountConvTensors(weights);
  const std::size_t selector = CountSelectorTensors(weights);
  const std::size_t markov = CountMarkovTensors(weights);

  switch (ClassifyDFlashDraftHead(weights)) {
    case DFlashDraftHead::kDFlash2:
      break;
    case DFlashDraftHead::kDSpark:
      break;
    case DFlashDraftHead::kIncomplete:
      return "Qwen DFlash drafting is refused: this drafter is neither a "
             "complete DFlash-2 nor a complete DSpark. It carries " +
             std::to_string(conv.present) + " of " +
             std::to_string(conv.expected) + " per-layer convolution tensors, " +
             std::to_string(selector) +
             " of 3 candidate-selector tensors and " + std::to_string(markov) +
             " of 3 Markov/confidence tensors. Complete DFlash-2 ships every "
             "convolution and selector tensor and no Markov head; complete "
             "DSpark ships none of them and carries markov_w1, markov_w2 and "
             "conf_proj. A mixed artifact would load and then draft as the "
             "wrong head.";
  }

  if (config.hidden_size == 0 || config.vocab_size == 0 ||
      draft.num_layers == 0 || weights.layers.size() != draft.num_layers) {
    return "Qwen DFlash drafting is refused: this drafter has an incomplete "
           "transformer topology (hidden " +
           std::to_string(config.hidden_size) + ", vocabulary " +
           std::to_string(config.vocab_size) + ", layers " +
           std::to_string(draft.num_layers) + ", loaded " +
           std::to_string(weights.layers.size()) + ").";
  }
  if (draft.block_size < 2 || draft.sliding_window < draft.block_size) {
    return "Qwen DFlash drafting is refused: this drafter declares "
           "dflash.block_size " +
           std::to_string(draft.block_size) + " with sliding window " +
           std::to_string(draft.sliding_window) +
           "; the block size must be at least 2 and fit inside the window.";
  }

  if (IsDSparkDraft(weights)) {
    // `hidden / conv_group_size` is never evaluated on this path, so the zero
    // the loader left in conv_group_size is a mode, not a fault.
    if (draft.markov_rank == 0 ||
        weights.markov_w1.num_elements !=
            static_cast<std::size_t>(draft.markov_rank) * config.vocab_size ||
        weights.markov_w2.num_elements !=
            static_cast<std::size_t>(draft.markov_rank) * config.vocab_size) {
      return "Qwen DSpark drafting is refused: markov_w1/markov_w2 are not "
             "both [" +
             std::to_string(draft.markov_rank) + ", " +
             std::to_string(config.vocab_size) +
             "], so the Markov bias row cannot be addressed.";
    }
    return {};
  }

  if (draft.conv_group_size == 0 || draft.conv_kernel_size == 0) {
    return "Qwen DFlash drafting is refused: this DFlash-2 drafter declares "
           "dflash.conv_group_size " +
           std::to_string(draft.conv_group_size) +
           " or dflash.conv_kernel_size " +
           std::to_string(draft.conv_kernel_size) +
           ", which the Qwen DFlash GPU executor needs to size its dynamic "
           "convolutions.";
  }
  if (config.hidden_size % draft.conv_group_size != 0) {
    return "Qwen DFlash drafting is refused: hidden size " +
           std::to_string(config.hidden_size) +
           " is not a multiple of dflash.conv_group_size " +
           std::to_string(draft.conv_group_size) +
           ", so the grouped convolution cannot divide.";
  }
  if (draft.selector_rank == 0 || draft.selector_top_k == 0 ||
      draft.selector_top_k > 16U) {
    return "Qwen DFlash drafting is refused: this drafter declares "
           "dflash.selector_rank " +
           std::to_string(draft.selector_rank) +
           " and dflash.selector_top_k " +
           std::to_string(draft.selector_top_k) +
           ", which the Qwen DFlash GPU executor requires for candidate "
           "selection (top_k must be 1..16).";
  }
  return {};
}

std::string DFlashGroupedConvLaunchRefusal(std::size_t hidden_size,
                                           std::uint32_t kernel_size,
                                           std::uint32_t group_size) {
  if (group_size == 0U) {
    return "DFlash grouped convolution is refused: conv_group_size is 0. The "
           "kernel divides hidden_size " +
           std::to_string(hidden_size) +
           " by group_size on the device, and a zero divisor is undefined "
           "behaviour there rather than a clean error. A DSpark draft has no "
           "convolutions and must not launch this kernel at all.";
  }
  if (kernel_size == 0U) {
    return "DFlash grouped convolution is refused: conv_kernel_size is 0.";
  }
  if (hidden_size % group_size != 0U) {
    return "DFlash grouped convolution is refused: hidden size " +
           std::to_string(hidden_size) +
           " is not a multiple of conv_group_size " + std::to_string(group_size) +
           ".";
  }
  return {};
}

std::size_t DFlashDynamicCoefficientCount(std::size_t hidden_size,
                                          std::uint32_t conv_kernel_size,
                                          std::uint32_t conv_group_size) noexcept {
  if (conv_group_size == 0U || conv_kernel_size == 0U || hidden_size == 0U) {
    return 0U;
  }
  return std::size_t{2} * conv_kernel_size *
         (hidden_size / conv_group_size);
}

std::size_t DFlashDraftScratchElements(
    const core::ModelConfig& config, const QwenDFlashConfig& draft,
    std::uint32_t max_context, std::size_t selector_scratch_elements,
    bool markov_head) noexcept {
  const std::size_t hidden = config.hidden_size;
  const std::size_t kv = static_cast<std::size_t>(config.num_key_value_heads) *
                         config.head_dim;
  const std::size_t block = draft.block_size;
  const std::size_t injection = std::min<std::size_t>(max_context, 256U);
  const std::size_t shared = std::max(injection, block);
  const std::size_t dynamic =
      markov_head ? 0U
                  : DFlashDynamicCoefficientCount(
                        hidden, draft.conv_kernel_size, draft.conv_group_size);
  const std::size_t ffn_width =
      std::max<std::size_t>(config.intermediate_size, selector_scratch_elements);
  const bool markov = markov_head && draft.markov_rank != 0;
  // Markov scratch: one bias row, one markov_w1 row, one running predecessor
  // token id and the emitted DSpark token row. The emitted row cannot reuse the
  // block token buffer, because step zero reads the anchor out of it.
  const std::size_t markov_elements =
      markov ? config.vocab_size + draft.markov_rank + 1U + block : 0U;
  const std::size_t proposal =
      block * (3U * hidden + dynamic + 2U * config.AttentionSize() +
               2U * ffn_width + config.vocab_size + 3U);
  const std::size_t selector_terms =
      markov ? 0U : block * (draft.selector_rank + 2U * draft.selector_top_k);
  return injection * draft.target_layer_ids.size() * hidden +
         shared * (2U * hidden + 2U * kv) + proposal + selector_terms +
         markov_elements;
}

std::optional<QwenDFlashWeights> QwenDFlashWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "dflash") {
    SetError(error_msg, "DFlash-2 GGUF requires general.architecture='dflash'");
    return std::nullopt;
  }

  core::ModelConfig config;
  config.architecture = "dflash";
  const std::string prefix = "dflash.";

  const auto hidden_size =
      RequiredUint32(reader, prefix + "embedding_length", error_msg);
  const auto intermediate_size =
      RequiredUint32(reader, prefix + "feed_forward_length", error_msg);
  const auto num_attention_heads =
      RequiredUint32(reader, prefix + "attention.head_count", error_msg);
  const auto num_key_value_heads =
      RequiredUint32(reader, prefix + "attention.head_count_kv", error_msg);
  const auto head_dim =
      RequiredUint32(reader, prefix + "attention.key_length", error_msg);
  auto rotary_dim = reader.GetMetadataUint32(prefix + "rope.dimension_count");
  if (!rotary_dim.has_value()) {
    const auto sections =
        GetNonnegativeIntegerArray(reader, prefix + "rope.dimension_sections");
    if (sections.has_value()) {
      const auto& values = *sections;
      if (values.size() == 4 && values[0] > 0 && values[1] == 0 &&
          values[2] == 0 && values[3] == 0 &&
          values[0] <= std::numeric_limits<std::uint32_t>::max() / 2U) {
        rotary_dim = static_cast<std::uint32_t>(values[0] * 2U);
      }
    }
  }
  const auto context_length =
      RequiredUint32(reader, prefix + "context_length", error_msg);
  const auto num_layers =
      RequiredUint32(reader, prefix + "block_count", error_msg);
  const auto rope_theta = reader.GetMetadataFloat32(prefix + "rope.freq_base");
  if (!hidden_size || !intermediate_size || !num_attention_heads ||
      !num_key_value_heads || !head_dim || !rotary_dim || !context_length ||
      !num_layers || !rope_theta) {
    if (!rotary_dim) {
      SetError(error_msg,
               "DFlash-2 GGUF requires a valid rotary dimension or "
               "degenerate four-section M-RoPE layout");
    }
    if (!rope_theta) {
      SetError(
          error_msg,
          "DFlash-2 GGUF is missing required metadata 'dflash.rope.freq_base'");
    }
    return std::nullopt;
  }

  config.hidden_size = *hidden_size;
  config.intermediate_size = *intermediate_size;
  config.num_attention_heads = *num_attention_heads;
  config.num_key_value_heads = *num_key_value_heads;
  config.head_dim = *head_dim;
  config.rotary_dim = *rotary_dim;
  config.rope_theta = *rope_theta;
  config.context_length = *context_length;
  config.num_layers = *num_layers;
  config.full_attention_interval = 1;
  config.mtp_num_layers = 0;

  if (config.hidden_size == 0 || config.intermediate_size == 0 ||
      config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
      config.head_dim == 0 || config.num_layers == 0 ||
      config.num_attention_heads % config.num_key_value_heads != 0 ||
      config.rotary_dim == 0 || config.rotary_dim > config.head_dim ||
      (config.rotary_dim % 2U) != 0) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid transformer topology");
    return std::nullopt;
  }

  QwenDFlashWeights weights;
  weights.config = config;
  auto& df_cfg = weights.dflash_config;

  // DSpark = DFlash backbone + Markov head + confidence head. The Markov
  // matrix is the discriminator, matching the reference loader, which keys
  // draft-dspark off `markov_w1.weight` and draft-dflash off its absence.
  df_cfg.is_dspark = reader.FindTensor("markov_w1.weight") != nullptr;

  const auto block_size =
      RequiredUint32(reader, prefix + "block_size", error_msg);
  // Conv and selector keys describe the DFlash-2 block-diffusion path only.
  // DSpark exports none of them and must not be asked for them.
  const auto conv_kernel_size = df_cfg.is_dspark
                                    ? std::optional<std::uint32_t>{}
                                    : RequiredUint32(
                                          reader, prefix + "conv_kernel_size",
                                          error_msg);
  const auto conv_group_size = df_cfg.is_dspark
                                   ? std::optional<std::uint32_t>{}
                                   : RequiredUint32(
                                         reader, prefix + "conv_group_size",
                                         error_msg);
  const auto selector_rank = df_cfg.is_dspark
                                 ? std::optional<std::uint32_t>{}
                                 : RequiredUint32(reader,
                                                  prefix + "selector_rank",
                                                  error_msg);
  const auto selector_top_k = df_cfg.is_dspark
                                  ? std::optional<std::uint32_t>{}
                                  : RequiredUint32(reader,
                                                   prefix + "selector_top_k",
                                                   error_msg);
  const auto sliding_window =
      RequiredUint32(reader, prefix + "attention.sliding_window", error_msg);
  // DSpark reads the mask id from this same drafter-side key; the target
  // artifact carries no tokenizer keys at all, so there is nowhere else to
  // look.
  const auto mask_token_id =
      RequiredUint32(reader, "tokenizer.ggml.mask_token_id", error_msg);
  // Only DFlash-2 declares block-internal causality. DSpark is non-causal,
  // which is also how the draft kernels are written.
  const auto is_causal = df_cfg.is_dspark
                             ? std::optional<bool>{false}
                             : reader.GetMetadataBool(
                                   prefix + "attention.causal");
  // The gate must ask for what each topology actually needs: testing the four
  // DFlash-2 keys unconditionally refuses every DSpark draft, because their
  // values are deliberately absent rather than merely unread.
  const bool missing_core =
      !block_size || !sliding_window || !mask_token_id || !is_causal.has_value();
  const bool missing_dflash2 =
      !df_cfg.is_dspark &&
      (!conv_kernel_size || !conv_group_size || !selector_rank ||
       !selector_top_k);
  if (missing_core || missing_dflash2) {
    if (!is_causal.has_value()) {
      SetError(error_msg,
               "DFlash-2 GGUF is missing required metadata "
               "'dflash.attention.causal'");
    }
    return std::nullopt;
  }

  df_cfg.block_size = *block_size;
  df_cfg.conv_kernel_size = conv_kernel_size.value_or(0U);
  df_cfg.conv_group_size = conv_group_size.value_or(0U);
  df_cfg.selector_rank = selector_rank.value_or(0U);
  df_cfg.selector_top_k = selector_top_k.value_or(0U);
  df_cfg.sliding_window = *sliding_window;

  // M8 sliding_window_pattern. TRUE means the layer uses the sliding window,
  // FALSE means full attention: the reference loader reads the array straight
  // into its per-layer is_swa table (src/models/dflash.cpp:99) and is_swa()
  // returns that entry verbatim (src/llama-hparams.cpp:290-296), while the
  // converters write `true` for layers whose layer_type is "sliding_attention"
  // (conversion/muse_glimmer.py:51, conversion/gemma.py:549). The DSpark
  // draft declares dflash.attention.sliding_window=2048 with
  // [true, true, true, true, true], i.e. all five layers sliding.
  //
  // The reference only reads the pattern when the scalar window is non-zero
  // (src/models/dflash.cpp:97) and hard-errors when the pattern is then
  // missing. Gufo keeps the pre-existing all-sliding fallback instead of
  // adding that requirement: a DFlash-2 artifact without the key must keep
  // loading and drafting identically.
  const auto swa_pattern = GetNonnegativeIntegerArray(
      reader, prefix + "attention.sliding_window_pattern");
  if (swa_pattern.has_value()) {
    if (swa_pattern->empty() ||
        swa_pattern->size() != config.num_layers) {
      SetError(error_msg,
               "DFlash GGUF sliding_window_pattern has " +
                   std::to_string(swa_pattern->size()) +
                   " entries; expected one per layer (" +
                   std::to_string(config.num_layers) + ")");
      return std::nullopt;
    }
    df_cfg.layer_sliding_windows.reserve(swa_pattern->size());
    for (const std::uint64_t entry : *swa_pattern) {
      if (entry > 1U) {
        SetError(error_msg,
                 "DFlash GGUF sliding_window_pattern entry is neither true "
                 "nor false");
        return std::nullopt;
      }
      df_cfg.layer_sliding_windows.push_back(entry != 0U ? *sliding_window
                                                        : 0U);
    }
  }
  df_cfg.mask_token_id = *mask_token_id;
  df_cfg.num_layers = config.num_layers;
  df_cfg.has_confidence_head =
      reader.GetMetadataBool(prefix + "has_confidence_head")
          .value_or(false);
  // `dflash.sample_from_anchor`, read the way the reference reads it: absent
  // means true (src/models/dflash.cpp:316-318). The executor must not guess --
  // it decides whether the anchor slot is itself a drafted position.
  {
    const auto* anchor_meta = reader.FindMetadata(prefix + "sample_from_anchor");
    if (anchor_meta != nullptr) {
      if (const auto* flag = std::get_if<bool>(&anchor_meta->value)) {
        df_cfg.sample_from_anchor = *flag;
      } else if (const auto* text =
                     std::get_if<std::string_view>(&anchor_meta->value)) {
        if (*text != "true" && *text != "false") {
          SetError(error_msg,
                   "DFlash GGUF dflash.sample_from_anchor must be true or "
                   "false");
          return std::nullopt;
        }
        df_cfg.sample_from_anchor = *text == "true";
      } else {
        SetError(error_msg,
                 "DFlash GGUF dflash.sample_from_anchor must be a bool or the "
                 "string \"true\" or \"false\"");
        return std::nullopt;
      }
    }
  }

  // The draft RMSNorm kernels take the epsilon as a literal, so the artifact
  // must agree with them rather than the value being plumbed through. A
  // mismatch would silently normalise with the wrong epsilon.
  const auto rms_eps = reader.GetMetadataFloat32(
      prefix + "attention.layer_norm_rms_epsilon");
  if (rms_eps.has_value() && *rms_eps != kDraftRmsEps) {
    SetError(error_msg,
             "DFlash GGUF declares layer_norm_rms_epsilon=" +
                 std::to_string(*rms_eps) + " but the draft kernels are "
                 "compiled for 1e-06");
    return std::nullopt;
  }
  df_cfg.rms_eps = kDraftRmsEps;

  if (df_cfg.block_size < 2 || df_cfg.sliding_window < df_cfg.block_size ||
      *is_causal ||
      (!df_cfg.is_dspark &&
       (df_cfg.conv_kernel_size == 0 || df_cfg.conv_group_size == 0 ||
        config.hidden_size % df_cfg.conv_group_size != 0 ||
        df_cfg.selector_rank == 0 || df_cfg.selector_top_k == 0 ||
        df_cfg.selector_top_k > 16U))) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid diffusion topology");
    return std::nullopt;
  }

  // GGUF target_layers are layer-input indices (the convention used by the
  // converter). Gufo taps layer outputs, so normalize N to zero-based N - 1.
  const auto encoded_target_layers =
      GetNonnegativeIntegerArray(reader, prefix + "target_layers");
  if (!encoded_target_layers.has_value()) {
    SetError(error_msg,
             "DFlash-2 GGUF is missing required integer array "
             "'dflash.target_layers'");
    return std::nullopt;
  }
  if (encoded_target_layers->empty()) {
    SetError(error_msg, "DFlash-2 target layer list must not be empty");
    return std::nullopt;
  }
  df_cfg.target_layer_ids.reserve(encoded_target_layers->size());
  for (const std::uint64_t encoded_layer : *encoded_target_layers) {
    if (encoded_layer == 0 ||
        !std::in_range<std::uint32_t>(encoded_layer - 1U)) {
      SetError(error_msg,
               "DFlash-2 target layer indices must use positive GGUF "
               "layer-input numbering");
      return std::nullopt;
    }
    const auto normalized = static_cast<std::uint32_t>(encoded_layer - 1U);
    if (!df_cfg.target_layer_ids.empty() &&
        normalized <= df_cfg.target_layer_ids.back()) {
      SetError(error_msg,
               "DFlash-2 target layer indices must be strictly increasing");
      return std::nullopt;
    }
    df_cfg.target_layer_ids.push_back(normalized);
  }

  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  // DSpark has no dynamic convolutions, so its group and coefficient counts
  // are zero and must not be divided by.
  const std::size_t groups = df_cfg.is_dspark
                                 ? 0U
                                 : hidden / df_cfg.conv_group_size;
  const std::size_t dynamic_coefficients =
      std::size_t{2} * df_cfg.conv_kernel_size * groups;

  // DFlash-2 uses the target model's embedding and output head. Private copies
  // are accepted but not required.
  if (reader.FindTensor("d2t") != nullptr) {
    SetError(error_msg,
             "DFlash-2 reduced-vocabulary d2t exports are not supported yet");
    return std::nullopt;
  }

  std::size_t inferred_vocab = 0;
  if (df_cfg.is_dspark) {
    // DSpark infers its vocabulary from the Markov matrix, not from the
    // DFlash-2 selector tensors it does not carry. `markov_w1` is
    // [markov_rank, vocab]; the rank comes from ne[0].
    const auto* markov_meta = reader.FindTensor("markov_w1.weight");
    if (markov_meta == nullptr || markov_meta->data == nullptr ||
        markov_meta->dimensions.size() != 2 || markov_meta->dimensions[0] == 0) {
      SetError(error_msg,
               "DSpark markov_w1.weight must be a 2-D [markov_rank, vocab] "
               "tensor");
      return std::nullopt;
    }
    const auto rank = markov_meta->dimensions[0];
    if (!std::in_range<std::uint32_t>(rank)) {
      SetError(error_msg, "DSpark Markov rank is out of range");
      return std::nullopt;
    }
    df_cfg.markov_rank = static_cast<std::uint32_t>(rank);
    inferred_vocab =
        static_cast<std::size_t>(markov_meta->dimensions[1]);
    if (inferred_vocab == 0 || !std::in_range<std::uint32_t>(inferred_vocab) ||
        markov_meta->ElementCount() != rank * inferred_vocab) {
      SetError(error_msg, "DSpark Markov vocabulary size is invalid");
      return std::nullopt;
    }
    if (!BindRequiredTensor(reader, "markov_w1.weight",
                            markov_meta->ElementCount(), true,
                            weights.markov_w1, error_msg) ||
        !BindRequiredTensor(reader, "markov_w2.weight",
                            markov_meta->ElementCount(), true,
                            weights.markov_w2, error_msg)) {
      return std::nullopt;
    }
    const auto w2_meta = reader.FindTensor("markov_w2.weight");
    if (w2_meta == nullptr || w2_meta->dimensions.size() != 2 ||
        w2_meta->dimensions[0] != rank ||
        w2_meta->dimensions[1] != inferred_vocab) {
      SetError(error_msg,
               "DSpark markov_w2.weight must match markov_w1.weight");
      return std::nullopt;
    }

    // Confidence head: [hidden_size + markov_rank], optional. F16 is accepted
    // and converted to F32 here; leaving it F16 would hand the GEMM path a
    // tensor it reads as F32 and silently produce garbage.
    const std::size_t confidence_input = hidden + df_cfg.markov_rank;
    const auto* conf_meta = reader.FindTensor("conf_proj.weight");
    if (conf_meta != nullptr && conf_meta->data != nullptr) {
      weights.conf_projection = TensorRef(reader, "conf_proj.weight");
      if (weights.conf_projection.num_elements != confidence_input) {
        SetError(error_msg,
                 "DSpark conf_proj.weight has " +
                     std::to_string(weights.conf_projection.num_elements) +
                     " elements; expected " +
                     std::to_string(confidence_input));
        return std::nullopt;
      }
      if (weights.conf_projection.type == core::GgmlType::kF16) {
        const auto* source =
            static_cast<const std::uint16_t*>(weights.conf_projection.data);
        weights.conf_projection_data.resize(confidence_input);
        for (std::size_t i = 0; i < confidence_input; ++i) {
          weights.conf_projection_data[i] =
              quant::Fp16ToFloat(source[i]);
        }
        weights.conf_projection = models::QwenTensorRef{
            .data = weights.conf_projection_data.data(),
            .type = core::GgmlType::kF32,
            .num_elements = confidence_input,
        };
      } else if (!IsSupportedVectorType(weights.conf_projection.type) ||
                 !weights.conf_projection.FitsAvailableStorage()) {
        SetError(error_msg,
                 "DSpark conf_proj.weight has unsupported storage " +
                     std::string(core::ToString(
                         weights.conf_projection.type)));
        return std::nullopt;
      }
      if (!BindRequiredTensor(reader, "conf_proj.bias", 1, false,
                              weights.conf_bias, error_msg)) {
        return std::nullopt;
      }
    }
  } else {
    const auto* predecessor_tensor =
        reader.FindTensor("selector_predecessor.weight");
    if (predecessor_tensor == nullptr || predecessor_tensor->data == nullptr ||
        predecessor_tensor->ElementCount() % df_cfg.selector_rank != 0) {
      SetError(error_msg,
               "DFlash-2 selector_predecessor.weight has invalid dimensions");
      return std::nullopt;
    }
    weights.selector_predecessor =
        TensorRef(reader, "selector_predecessor.weight");
    if (!IsSupportedMatrixType(weights.selector_predecessor.type) ||
        !weights.selector_predecessor.FitsAvailableStorage()) {
      SetError(error_msg,
               "DFlash-2 selector_predecessor.weight has unsupported storage");
      return std::nullopt;
    }
    inferred_vocab =
        weights.selector_predecessor.num_elements / df_cfg.selector_rank;
    if (inferred_vocab == 0 ||
        !std::in_range<std::uint32_t>(inferred_vocab)) {
      SetError(error_msg, "DFlash-2 selector vocabulary size is invalid");
      return std::nullopt;
    }
    if (!BindRequiredTensor(reader, "selector_successor.weight",
                            inferred_vocab * df_cfg.selector_rank, true,
                            weights.selector_successor, error_msg) ||
        !BindRequiredTensor(reader, "selector_hidden.weight",
                            hidden * df_cfg.selector_rank, true,
                            weights.selector_hidden, error_msg)) {
      return std::nullopt;
    }
  }
  config.vocab_size = static_cast<std::uint32_t>(inferred_vocab);
  weights.config.vocab_size = config.vocab_size;

  const std::size_t encoder_input = df_cfg.target_layer_ids.size() * hidden;
  if (!BindRequiredTensor(reader, "fc.weight", hidden * encoder_input, true,
                          weights.fc_projection, error_msg) ||
      !BindRequiredTensor(reader, "enc.output_norm.weight", hidden, false,
                          weights.fc_norm, error_msg) ||
      !BindRequiredTensor(reader, "output_norm.weight", hidden, false,
                          weights.output_norm, error_msg)) {
    return std::nullopt;
  }
  const auto fc_scale = TensorRef(reader, "fc.scale");
  if (!fc_scale.empty() &&
      (fc_scale.num_elements != 1 || !IsSupportedVectorType(fc_scale.type) ||
       !fc_scale.FitsAvailableStorage() || fc_scale.Get(0) != 1.0F)) {
    SetError(error_msg,
             "DFlash-2 fc.scale must be absent or a unit F32/BF16 scalar");
    return std::nullopt;
  }

  weights.layers.resize(df_cfg.num_layers);
  for (std::uint32_t i = 0; i < df_cfg.num_layers; ++i) {
    const std::string layer_prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    auto& transformer = layer.transformer;
    transformer.is_full_attention = true;

    if (!BindRequiredTensor(reader, layer_prefix + "attn_norm.weight", hidden,
                            false, transformer.attn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q.weight",
                            attention * hidden, true, transformer.attn_q,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k.weight", kv * hidden,
                            true, transformer.attn_k, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_v.weight", kv * hidden,
                            true, transformer.attn_v, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_output.weight",
                            hidden * attention, true, transformer.attn_output,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q_norm.weight",
                            config.head_dim, false, transformer.attn_q_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k_norm.weight",
                            config.head_dim, false, transformer.attn_k_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_norm.weight", hidden,
                            false, transformer.ffn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_gate.weight",
                            intermediate * hidden, true, transformer.ffn_gate,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_up.weight",
                            intermediate * hidden, true, transformer.ffn_up,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_down.weight",
                            hidden * intermediate, true, transformer.ffn_down,
                            error_msg)) {
      return std::nullopt;
    }
    // The per-layer dynamic convolutions belong to DFlash-2's block diffusion;
    // DSpark's layers are plain transformer blocks.
    if (!df_cfg.is_dspark &&
        (!BindRequiredTensor(reader, layer_prefix + "attn_conv_base",
                             hidden * df_cfg.conv_kernel_size * 2U, false,
                             layer.attention_conv_base, error_msg) ||
         !BindRequiredTensor(reader, layer_prefix + "attn_conv_proj.weight",
                             dynamic_coefficients * hidden, true,
                             layer.attention_conv_projection, error_msg) ||
         !BindRequiredTensor(reader, layer_prefix + "ffn_conv_base",
                             hidden * df_cfg.conv_kernel_size * 2U, false,
                             layer.ffn_conv_base, error_msg) ||
         !BindRequiredTensor(reader, layer_prefix + "ffn_conv_proj.weight",
                             dynamic_coefficients * hidden, true,
                             layer.ffn_conv_projection, error_msg))) {
      return std::nullopt;
    }
  }

  return weights;
}

std::uint32_t LayerSlidingWindow(const QwenDFlashConfig& config,
                                 std::uint32_t layer) noexcept {
  return LayerSlidingWindowImpl(config, layer);
}

std::uint32_t MaxSlidingWindow(const QwenDFlashConfig& config) noexcept {
  return MaxSlidingWindowImpl(config);
}

std::optional<std::vector<float>> ComputeMarkovBiasRow(
    const QwenDFlashWeights& weights, tokenization::TokenId previous) {
  const auto& config = weights.dflash_config;
  const std::size_t rank = config.markov_rank;
  const std::size_t vocab = weights.config.vocab_size;
  if (rank == 0 || vocab == 0 || !config.is_dspark ||
      weights.markov_w1.empty() || weights.markov_w2.empty()) {
    return std::nullopt;
  }
  if (weights.markov_w1.num_elements != rank * vocab ||
      weights.markov_w2.num_elements != rank * vocab) {
    return std::nullopt;
  }
  if (!IsSupportedMatrixType(weights.markov_w1.type) ||
      !IsSupportedMatrixType(weights.markov_w2.type)) {
    return std::nullopt;
  }
  if (static_cast<std::size_t>(previous) >= vocab) {
    return std::nullopt;
  }

  std::vector<float> bias(vocab, 0.0F);
  const std::size_t w1_offset = static_cast<std::size_t>(previous) * rank;

  // w1's prev row is the SAME rank floats for every candidate. Reading it once
  // instead of vocab*rank times is a pure hoist: `Get` is deterministic, so
  // the value and the operand order below are unchanged.
  std::vector<float> w1_row(rank);
  for (std::size_t component = 0; component < rank; ++component) {
    w1_row[component] = weights.markov_w1.Get(w1_offset + component);
  }

  // w2 read block-cached. `TensorRef::Get` dequantizes the whole 256-element
  // Q4_K block to hand back ONE element (state.hpp), so the element-wise loop
  // paid rank=256 dequants per candidate: vocab*rank Gets, each dequantizing
  // 256 elements. Caching the containing block does one dequant per block and
  // is the same float, from the same canonical DequantizeQ4_K, so the results
  // are BIT-IDENTICAL, not merely close. Other types keep the element-wise
  // path: they are not the drafter's type and correctness outranks speed.
  constexpr std::size_t kQ4KBlock = 256;
  const bool q4k_w2 =
      weights.markov_w2.type == gufo::core::GgmlType::kQ4_K;
  const auto* w2_bytes = static_cast<const std::uint8_t*>(weights.markov_w2.data);
  float block_buf[kQ4KBlock];
  std::size_t cached_block = std::numeric_limits<std::size_t>::max();
  const auto read_w2 = [&](std::size_t index) -> float {
    if (!q4k_w2) return weights.markov_w2.Get(index);
    const std::size_t block = index / kQ4KBlock;
    if (block != cached_block) {
      gufo::quant::DequantizeQ4_K(
          w2_bytes + block * sizeof(gufo::quant::block_q4_K), block_buf,
          kQ4KBlock);
      cached_block = block;
    }
    return block_buf[index % kQ4KBlock];
  };

  for (std::size_t candidate = 0; candidate < vocab; ++candidate) {
    const std::size_t w2_offset = candidate * rank;
    float partial = 0.0F;
    for (std::size_t component = 0; component < rank; ++component) {
      partial += read_w2(w2_offset + component) * w1_row[component];
    }
    bias[candidate] = partial;
  }
  return bias;
}

DSparkChainRange DSparkChainRangeFor(std::size_t block_count,
                                     std::size_t draft_count,
                                     bool sample_from_anchor) noexcept {
  // Where the chain starts is the anchor's business: the anchor slot is a
  // prediction slot when sample_from_anchor is true, a bonus token when it is
  // false (src/models/dflash.cpp:316-318).
  const std::size_t first_row = sample_from_anchor ? 0U : 1U;
  // How many tokens the caller gets is NOT. The caller asked for draft_count
  // and the verifier scores that many, so the cap is applied here, after the
  // anchor, instead of letting an anchor-predicted block spend its extra row.
  const std::size_t rows =
      first_row < block_count ? std::min(draft_count, block_count - first_row)
                              : 0U;
  return DSparkChainRange{rows, first_row};
}

}  // namespace gufo::speculative

#ifndef GUFO_MODELS_QWEN_STATE_HPP_
#define GUFO_MODELS_QWEN_STATE_HPP_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/iq3_layout.hpp"
#include "src/models/qwen/rows32_layout.hpp"

namespace gufo::models {

/// Non-owning reference to a mapped tensor. Consumers validate the exact
/// formats supported for each tensor role.
struct QwenTensorRef {
  const void* data = nullptr;
  core::GgmlType type = core::GgmlType::kF32;
  std::size_t num_elements = 0;
  /// Bytes available from `data` to the end of its mapped storage region.
  /// Synthetic/in-process tensors default to unbounded trusted storage.
  std::size_t available_bytes = std::numeric_limits<std::size_t>::max();
  QwenWeightLayout layout = QwenWeightLayout::kNative;
  std::size_t layout_columns = 0;
  // IQ3 P/G/U refs share one pair root; logical size remains num_elements.
  std::size_t layout_rows = 0;
  std::size_t layout_row_begin = 0;

  [[nodiscard]] bool HasValidWeightLayout() const noexcept {
    if (layout == QwenWeightLayout::kNative)
      return true;
    if (IsIq3Layout(layout)) {
      if (!data || reinterpret_cast<std::uintptr_t>(data) % 16 ||
          !Iq3ShapeValid(layout, layout_rows, layout_columns) ||
          (layout == QwenWeightLayout::kIq3SRows8
               ? type != core::GgmlType::kIQ3_S
               : type != core::GgmlType::kIQ3_XXS) ||
          !num_elements || num_elements % layout_columns ||
          layout_row_begin > layout_rows)
        return false;
      const auto rows = num_elements / layout_columns;
      return rows <= layout_rows - layout_row_begin &&
             rows % Iq3GroupRows(layout) == 0 &&
             layout_row_begin % Iq3GroupRows(layout) == 0;
    }
    return layout == QwenWeightLayout::kRows32 &&
           (type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K) &&
           layout_columns != 0 && layout_columns % 256 == 0 &&
           num_elements % layout_columns == 0 &&
           num_elements / layout_columns % 32 == 0 && num_elements != 0 &&
           num_elements / 256 <=
               std::numeric_limits<std::size_t>::max() /
                   (type == core::GgmlType::kQ5_K ? 176 : 210);
  }

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || num_elements == 0;
  }

  [[nodiscard]] std::size_t EncodedSizeBytes() const noexcept {
    return gufo::quant::EncodedSizeBytes(type, num_elements);
  }

  [[nodiscard]] bool FitsAvailableStorage() const noexcept {
    const std::size_t encoded_bytes =
        IsIq3Layout(layout)
            ? Iq3PackedBytes(layout, layout_rows, layout_columns)
            : EncodedSizeBytes();
    return HasValidWeightLayout() && encoded_bytes != 0 &&
           encoded_bytes <= available_bytes;
  }

  [[nodiscard]] float Get(std::size_t index) const noexcept {
    if (IsIq3Layout(layout)) {
      assert(false && "tagged IQ3 CPU consumers are not supported");
      std::abort();
    }
    if (layout != QwenWeightLayout::kNative) {
      if (!data || !FitsAvailableStorage() || index >= num_elements) {
        assert(false && "QwenTensorRef::Get: invalid Rows32 view");
        std::abort();
      }
      const std::size_t block_bytes = type == core::GgmlType::kQ5_K ? 176 : 210;
      alignas(16) std::uint8_t block[210];
      const auto* bytes = static_cast<const std::uint8_t*>(data);
      for (std::size_t o = 0; o < block_bytes; ++o)
        block[o] = bytes[Rows32ByteOffset(
            num_elements / layout_columns, layout_columns / 256,
            index / layout_columns, index % layout_columns / 256, o,
            block_bytes)];
      auto native = *this;
      native.data = block;
      native.num_elements = 256;
      native.available_bytes = block_bytes;
      native.layout = QwenWeightLayout::kNative;
      native.layout_columns = 0;
      native.layout_rows = 0;
      native.layout_row_begin = 0;
      return native.Get(index % 256);
    }
    if (type == core::GgmlType::kF32) {
      return static_cast<const float*>(data)[index];
    }
    if (type == core::GgmlType::kBF16) {
      const auto u16 = static_cast<const std::uint16_t*>(data)[index];
      const std::uint32_t u32 = static_cast<std::uint32_t>(u16) << 16;
      float f = 0.0F;
      std::memcpy(&f, &u32, sizeof(float));
      return f;
    }
    if (type == core::GgmlType::kF16) {
      return gufo::quant::Fp16ToFloat(
          static_cast<const std::uint16_t*>(data)[index]);
    }
    if (type == core::GgmlType::kQ8_K) {
      constexpr std::size_t kBlockSize = 256;
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const auto* block =
          reinterpret_cast<const gufo::quant::block_q8_K*>(data) + block_idx;
      return block->d * static_cast<float>(block->qs[pos]);
    }
    if (type == core::GgmlType::kQ8_0) {
      // Q8_0 block: fp16 scale + 32 int8 values (QK=32, 34 bytes).
      constexpr std::size_t kBlockSize = 32;
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const auto* block =
          reinterpret_cast<const gufo::quant::block_q8_0*>(data) + block_idx;
      return gufo::quant::Fp16ToFloat(block->d) *
             static_cast<float>(block->qs[pos]);
    }
    if (type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ4_K ||
        type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K) {
      // QK=256 block-packed matmul weights. Dequantize the containing block via
      // the canonical (parity-tested) full-row dequant helpers, then index into
      // it. Previously this fell through to a silent 0.0F.
      constexpr std::size_t kBlockSize = 256;
      const std::size_t block_idx = index / kBlockSize;
      const std::size_t pos = index % kBlockSize;
      const std::size_t block_bytes =
          type == core::GgmlType::kQ3_K   ? sizeof(gufo::quant::block_q3_K)
          : type == core::GgmlType::kQ4_K ? sizeof(gufo::quant::block_q4_K)
          : type == core::GgmlType::kQ5_K ? sizeof(gufo::quant::block_q5_K)
                                          : sizeof(gufo::quant::block_q6_K);
      const auto* block_ptr =
          static_cast<const std::uint8_t*>(data) + block_idx * block_bytes;
      float block_buf[kBlockSize];
      switch (type) {
        case core::GgmlType::kQ3_K:
          gufo::quant::DequantizeQ3_K(block_ptr, block_buf, kBlockSize);
          break;
        case core::GgmlType::kQ4_K:
          gufo::quant::DequantizeQ4_K(block_ptr, block_buf, kBlockSize);
          break;
        case core::GgmlType::kQ5_K:
          gufo::quant::DequantizeQ5_K(block_ptr, block_buf, kBlockSize);
          break;
        case core::GgmlType::kQ6_K:
          gufo::quant::DequantizeQ6_K(block_ptr, block_buf, kBlockSize);
          break;
        default:
          break;
      }
      return block_buf[pos];
    }
    // Unsupported (or non-block-aligned) type: fail loudly instead of silently
    // returning 0.0F.
    assert(false && "QwenTensorRef::Get: unsupported GgmlType");
    std::abort();
  }

  [[nodiscard]] std::span<const float> AsFloatSpan() const noexcept {
    if (layout == QwenWeightLayout::kNative && type == core::GgmlType::kF32 &&
        data != nullptr) {
      return {static_cast<const float*>(data), num_elements};
    }
    return {};
  }
};

/// Tensor weight references for a single transformer layer block (Linear SSM or
/// Full Attention).
struct QwenLayerWeights {
  bool is_full_attention = false;

  // Common Layer Norms
  QwenTensorRef attn_norm;
  QwenTensorRef ffn_norm;

  // Full Attention Weights (every 4th block)
  QwenTensorRef attn_q;
  QwenTensorRef attn_k;
  QwenTensorRef attn_v;
  QwenTensorRef attn_output;
  QwenTensorRef attn_q_norm;
  QwenTensorRef attn_k_norm;

  // Linear Attention / SSM Weights (3 of 4 blocks)
  QwenTensorRef attn_qkv;
  QwenTensorRef attn_gate;
  QwenTensorRef ssm_a;
  QwenTensorRef ssm_conv1d;
  QwenTensorRef ssm_dt;
  QwenTensorRef ssm_alpha;
  QwenTensorRef ssm_beta;
  QwenTensorRef ssm_norm;
  QwenTensorRef ssm_out;

  // Feed Forward Network
  QwenTensorRef ffn_gate;
  QwenTensorRef ffn_up;
  QwenTensorRef ffn_down;
};

/// Full model tensor references mapped directly from GGUF storage.
struct QwenModelWeights {
  core::ModelConfig config;
  QwenTensorRef token_embd;
  std::vector<QwenLayerWeights> layers;
  QwenTensorRef output_norm;
  QwenTensorRef output;

  [[nodiscard]] bool HasNonNativeWeights() const noexcept {
    const auto native = [](const QwenTensorRef& t) {
      return t.layout == QwenWeightLayout::kNative;
    };
    if (!native(token_embd) || !native(output_norm) || !native(output))
      return true;
    for (const auto& l : layers) {
      const QwenTensorRef* refs[] = {
          &l.attn_norm, &l.ffn_norm,    &l.attn_q,      &l.attn_k,
          &l.attn_v,    &l.attn_output, &l.attn_q_norm, &l.attn_k_norm,
          &l.attn_qkv,  &l.attn_gate,   &l.ssm_a,       &l.ssm_conv1d,
          &l.ssm_dt,    &l.ssm_alpha,   &l.ssm_beta,    &l.ssm_norm,
          &l.ssm_out,   &l.ffn_gate,    &l.ffn_up,      &l.ffn_down};
      for (const auto* t : refs)
        if (!native(*t))
          return true;
    }
    return false;
  }

  /// Loads and binds weights from a GgufReader.
  [[nodiscard]] static std::optional<QwenModelWeights> LoadFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// Contiguous Key-Value Cache for auto-regressive generation.
class QwenKvCache {
public:
  QwenKvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
              std::uint32_t max_context, std::uint32_t head_dim);

  void Reset() noexcept { current_pos_ = 0; }
  [[nodiscard]] std::uint32_t GetCurrentPos() const noexcept {
    return current_pos_;
  }
  void AdvancePos() noexcept { ++current_pos_; }

  [[nodiscard]] std::span<float> GetKeySlice(std::uint32_t layer,
                                             std::uint32_t kv_head,
                                             std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetKeySlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

  [[nodiscard]] std::span<float> GetValueSlice(std::uint32_t layer,
                                               std::uint32_t kv_head,
                                               std::uint32_t pos) noexcept;
  [[nodiscard]] std::span<const float> GetValueSlice(
      std::uint32_t layer, std::uint32_t kv_head,
      std::uint32_t pos) const noexcept;

private:
  std::uint32_t num_layers_;
  std::uint32_t num_kv_heads_;
  std::uint32_t max_context_;
  std::uint32_t head_dim_;
  std::uint32_t current_pos_{0};
  std::vector<float> k_data_;
  std::vector<float> v_data_;
};

/// Preallocated activation scratch arena for zero-heap-allocation decode
/// passes.
struct QwenScratchArena {
  explicit QwenScratchArena(const core::ModelConfig& config);

  std::vector<float> buffer;
  std::span<float> hidden;
  std::span<float> normed;
  std::span<float> q;
  std::span<float> k;
  std::span<float> v;
  std::span<float> attn_scores;
  std::span<float> attn_out;
  std::span<float> mlp_gate;
  std::span<float> mlp_up;
  std::span<float> mlp_act;
  std::span<float> mlp_out;
  std::span<float> ssm_qkv;
  std::span<float> ssm_gate;
  std::span<float> ssm_out_buf;
  std::span<float> logits;
};

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_STATE_HPP_

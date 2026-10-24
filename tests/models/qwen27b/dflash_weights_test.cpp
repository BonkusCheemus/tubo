// DFlash / DSpark drafter weight binding, CPU only.
//
// M1  DSpark binding: markov_w1/w2 + conf_proj, vocab inferred from markov_w1,
//     no conv / selector / causal keys required.
// M2  target tap cap 8 (fc.weight proves 8 taps: 40960 = 8 * 5120).
// M4  conf_proj F16 -> F32 converted at load; the converted values are compared
//     against an INDEPENDENT IEEE-754 binary16 -> binary32 reference written
//     below, bit for bit. The reference does not call
//     gufo::quant::Fp16ToFloat, so the code under test never checks itself.
// M5  artifact rms epsilon asserted equal to the 1e-6F the kernels use.
// M6  mask id read from the drafter's own header.
//
// Two fixtures are built in memory so the DFlash-2 branch has something that
// loads: no DFlash-2 artifact in the model store is loadable at HEAD (the
// ROCmFP4 one fails inside the reader, the Q2_K one has no rope dimension key).
// The real DSpark artifact is loaded too, with RSS measured around it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/dflash_weights.hpp"

namespace {

constexpr int kSkipExit = 77;

int g_failures = 0;

void Check(bool condition, std::string_view what) {
  if (condition) {
    std::cout << "  ok   " << what << "\n";
    return;
  }
  std::cerr << "  FAIL " << what << "\n";
  ++g_failures;
}

std::size_t ResidentKilobytes() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      std::size_t value = 0;
      status >> value;
      return value;
    }
    status.ignore(1 << 16, '\n');
  }
  return 0;
}

// IEEE-754 binary16 -> binary32, written from the standard's bit layout.
// Independent of src/core/quant/ggml_dequant.cpp on purpose.
float ReferenceHalfToFloat(std::uint16_t half) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half >> 15) << 31;
  const std::uint32_t exponent = (half >> 10) & 0x1FU;
  const std::uint32_t mantissa = half & 0x3FFU;

  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;  // signed zero
    } else {
      std::uint32_t m = mantissa;
      std::uint32_t shift = 0;
      while ((m & 0x400U) == 0U) {
        m <<= 1;
        ++shift;
      }
      m &= 0x3FFU;
      bits = sign | ((113U - shift) << 23) | (m << 13);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }

  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::string_view EnvOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return (value != nullptr && *value != '\0') ? std::string_view(value)
                                             : std::string_view(fallback);
}

// ---------------------------------------------------------------------------
// Minimal synthetic GGUF writer. KV section and tensor-info table are appended
// first with offsets computed against the (32-byte aligned) data section start,
// so nothing needs patching except the two counts.
// ---------------------------------------------------------------------------
class GgufBuilder {
 public:
  GgufBuilder() {
    const char magic[4] = {'G', 'G', 'U', 'F'};
    AppendBytes(magic, 4);
    AppendPod(std::uint32_t{3});
    tensor_count_pos_ = buffer_.size();
    AppendPod(std::uint64_t{0});
    metadata_count_pos_ = buffer_.size();
    AppendPod(std::uint64_t{0});
  }

  void String(std::string_view key, std::string_view value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(value);
    ++metadata_count_;
  }

  void Uint32(std::string_view key, std::uint32_t value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
    AppendPod(value);
    ++metadata_count_;
  }

  void Float32(std::string_view key, float value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kFloat32));
    AppendPod(value);
    ++metadata_count_;
  }

  void Bool(std::string_view key, bool value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kBool));
    AppendPod(static_cast<std::uint8_t>(value ? 1 : 0));
    ++metadata_count_;
  }

  void Uint64Array(std::string_view key,
                   const std::vector<std::uint64_t>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint64));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
      AppendPod(value);
    }
    ++metadata_count_;
  }

  /// Bool arrays are one byte per element in GGUF, not eight.
  void BoolArray(std::string_view key, const std::vector<bool>& values) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kArray));
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kBool));
    AppendPod(static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
      AppendPod(static_cast<std::uint8_t>(value ? 1 : 0));
    }
    ++metadata_count_;
  }

  /// Registers a tensor and returns its element count. The payload is filled
  /// with a recognisable non-zero F32 pattern; F16 tensors carry the halves
  /// verbatim.
  std::uint64_t TensorF32(std::string_view name,
                          const std::vector<std::uint64_t>& dims) {
    std::uint64_t elements = 1;
    for (const auto dimension : dims) {
      elements *= dimension;
    }
    tensors_.push_back({std::string(name), dims,
                        gufo::core::GgmlType::kF32, elements * 4, {}});
    return elements;
  }

  std::uint64_t TensorF16(std::string_view name,
                          const std::vector<std::uint64_t>& dims,
                          const std::vector<std::uint16_t>& halves) {
    std::uint64_t elements = 1;
    for (const auto dimension : dims) {
      elements *= dimension;
    }
    tensors_.push_back({std::string(name), dims,
                        gufo::core::GgmlType::kF16, halves.size() * 2, halves});
    return elements;
  }

  std::vector<std::uint8_t> Build() {
    const std::uint64_t tensor_count = tensors_.size();
    std::memcpy(buffer_.data() + tensor_count_pos_, &tensor_count,
                sizeof(tensor_count));
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));

    // Data offsets are relative to the start of the data section.
    std::uint64_t cursor = 0;
    for (auto& tensor : tensors_) {
      tensor.offset = cursor;
      cursor += tensor.bytes;
      const std::uint64_t pad = (32U - (cursor % 32U)) % 32U;
      cursor += pad;
    }

    for (const auto& tensor : tensors_) {
      AppendString(tensor.name);
      AppendPod(static_cast<std::uint32_t>(tensor.dims.size()));
      for (const auto dimension : tensor.dims) {
        AppendPod(dimension);
      }
      AppendPod(static_cast<std::uint32_t>(tensor.type));
      AppendPod(tensor.offset);
    }

    while (buffer_.size() % 32 != 0) {
      buffer_.push_back(0);
    }
    const std::size_t data_start = buffer_.size();
    for (const auto& tensor : tensors_) {
      buffer_.resize(data_start + tensor.offset + tensor.bytes, 0);
      if (tensor.type == gufo::core::GgmlType::kF16) {
        std::memcpy(buffer_.data() + data_start + tensor.offset,
                    tensor.halves.data(), tensor.bytes);
      } else {
        auto* floats = reinterpret_cast<float*>(buffer_.data() + data_start +
                                         tensor.offset);
        const std::size_t count = tensor.bytes / sizeof(float);
        for (std::size_t i = 0; i < count; ++i) {
          floats[i] = static_cast<float>(i % 97);
        }
      }
    }
    return buffer_;
  }

 private:
  struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dims;
    gufo::core::GgmlType type;
    std::size_t bytes;
    std::vector<std::uint16_t> halves;
    std::uint64_t offset{0};
  };

  void AppendBytes(const void* data, std::size_t length) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), bytes, bytes + length);
  }

  template <typename T>
  void AppendPod(T value) {
    AppendBytes(&value, sizeof(T));
  }

  void AppendString(std::string_view value) {
    AppendPod(static_cast<std::uint64_t>(value.size()));
    AppendBytes(value.data(), value.size());
  }

  std::vector<std::uint8_t> buffer_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
  std::vector<TensorRecord> tensors_;
};

// --- fixture geometry ------------------------------------------------------
constexpr std::uint32_t kHidden = 128;
constexpr std::uint32_t kIntermediate = 256;
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kHeadDim = 32;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kAttention = kHeads * kHeadDim;   // 128
constexpr std::uint32_t kKv = kKvHeads * kHeadDim;        // 64
constexpr std::uint32_t kRank = 8;
constexpr std::uint32_t kVocab = 512;
constexpr std::uint32_t kKernel = 2;
constexpr std::uint32_t kGroup = 16;
constexpr std::uint32_t kLayers = 5;
constexpr std::uint32_t kTaps = 2;
constexpr std::uint32_t kGroups = kHidden / kGroup;                    // 8
constexpr std::uint32_t kDynamic = 2U * kKernel * kGroups;              // 32

void AddCommonTransformerMetadata(GgufBuilder& builder) {
  builder.String("general.architecture", "dflash");
  builder.Uint32("dflash.block_count", kLayers);
  builder.Uint32("dflash.context_length", 4096);
  builder.Uint32("dflash.embedding_length", kHidden);
  builder.Uint32("dflash.feed_forward_length", kIntermediate);
  builder.Uint32("dflash.attention.head_count", kHeads);
  builder.Uint32("dflash.attention.head_count_kv", kKvHeads);
  builder.Uint32("dflash.attention.key_length", kHeadDim);
  builder.Uint32("dflash.attention.value_length", kHeadDim);
  builder.Uint32("dflash.rope.dimension_count", kHeadDim);
  builder.Float32("dflash.rope.freq_base", 10000000.0F);
  builder.Float32("dflash.attention.layer_norm_rms_epsilon", 1e-6F);
  builder.Uint32("dflash.block_size", 8);
  builder.Uint32("dflash.attention.sliding_window", 64);
  builder.Uint32("tokenizer.ggml.mask_token_id", 42);
}

void AddPlainLayers(GgufBuilder& builder) {
  builder.TensorF32("enc.output_norm.weight", {kHidden});
  builder.TensorF32("output_norm.weight", {kHidden});
  builder.TensorF32("fc.weight", {kTaps * kHidden, kHidden});
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    builder.TensorF32(prefix + "attn_norm.weight", {kHidden});
    builder.TensorF32(prefix + "attn_q.weight", {kAttention, kHidden});
    builder.TensorF32(prefix + "attn_k.weight", {kKv, kHidden});
    builder.TensorF32(prefix + "attn_v.weight", {kKv, kHidden});
    builder.TensorF32(prefix + "attn_output.weight", {kHidden, kAttention});
    builder.TensorF32(prefix + "attn_q_norm.weight", {kHeadDim});
    builder.TensorF32(prefix + "attn_k_norm.weight", {kHeadDim});
    builder.TensorF32(prefix + "ffn_norm.weight", {kHidden});
    builder.TensorF32(prefix + "ffn_gate.weight", {kIntermediate, kHidden});
    builder.TensorF32(prefix + "ffn_up.weight", {kIntermediate, kHidden});
    builder.TensorF32(prefix + "ffn_down.weight", {kHidden, kIntermediate});
  }
}

// DFlash-2: every conv + selector + causal key present, no Markov tensor.
std::vector<std::uint8_t> BuildDFlash2Fixture() {
  GgufBuilder builder;
  AddCommonTransformerMetadata(builder);
  builder.Uint32("dflash.conv_kernel_size", kKernel);
  builder.Uint32("dflash.conv_group_size", kGroup);
  builder.Uint32("dflash.selector_rank", kRank);
  builder.Uint32("dflash.selector_top_k", 4);
  builder.Bool("dflash.attention.causal", false);
  builder.Uint64Array("dflash.target_layers", {2, 4});
  AddPlainLayers(builder);
  builder.TensorF32("selector_predecessor.weight", {kRank, kVocab});
  builder.TensorF32("selector_successor.weight", {kRank, kVocab});
  builder.TensorF32("selector_hidden.weight", {kHidden, kRank});
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    builder.TensorF32(prefix + "attn_conv_base", {kHidden, kKernel, 2});
    builder.TensorF32(prefix + "attn_conv_proj.weight", {kHidden, kDynamic});
    builder.TensorF32(prefix + "ffn_conv_base", {kHidden, kKernel, 2});
    builder.TensorF32(prefix + "ffn_conv_proj.weight", {kHidden, kDynamic});
  }
  return builder.Build();
}

// DSpark with a hand-picked F16 conf_proj: zeros, subnormals, normals, inf,
// NaN. The Markov matrices are F32 here because the loader must accept either.
std::vector<std::uint8_t> BuildDSparkF16Fixture(
    const std::vector<std::uint16_t>& conf_halves) {
  GgufBuilder builder;
  AddCommonTransformerMetadata(builder);
  builder.Bool("dflash.has_confidence_head", true);
  builder.Uint64Array("dflash.target_layers", {2, 4});
  AddPlainLayers(builder);
  builder.TensorF32("markov_w1.weight", {kRank, kVocab});
  builder.TensorF32("markov_w2.weight", {kRank, kVocab});
  builder.TensorF16("conf_proj.weight", {kHidden + kRank}, conf_halves);
  builder.TensorF32("conf_proj.bias", {1});
  return builder.Build();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

std::uint64_t HeaderElementCount(const gufo::core::GgufReader& reader,
                                 std::string_view name) {
  const auto* info = reader.FindTensor(name);
  return info == nullptr ? 0U : info->ElementCount();
}

void CheckShape(const gufo::core::GgufReader& reader,
                const gufo::models::QwenTensorRef& bound, std::string_view name,
                std::uint64_t expected_elements) {
  const std::uint64_t header = HeaderElementCount(reader, name);
  Check(header != 0U, std::string(name) + ": present in header");
  Check(bound.num_elements == header,
        std::string(name) + ": bound elements == header elements (" +
            std::to_string(bound.num_elements) + " vs " +
            std::to_string(header) + ")");
  Check(bound.num_elements == expected_elements,
        std::string(name) + ": bound elements == expected (" +
            std::to_string(expected_elements) + ")");
  Check(bound.FitsAvailableStorage(),
        std::string(name) + ": fits mapped storage");
}

void TestSyntheticDFlash2NoRegression() {
  std::cout << "synthetic DFlash-2 no-regression\n";
  const auto buffer = BuildDFlash2Fixture();
  std::string error;
  auto reader = gufo::core::GgufReader::OpenMemory(buffer.data(), buffer.size(),
                                                   &error);
  Check(reader != nullptr, "fixture opens: " + error);
  if (reader == nullptr) {
    return;
  }
  auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Check(weights.has_value(), "DFlash-2 fixture loads: " + error);
  if (!weights.has_value()) {
    return;
  }
  const auto& df = weights->dflash_config;
  Check(!df.is_dspark, "DFlash-2 fixture is NOT classified as DSpark");
  Check(df.markov_rank == 0U, "DFlash-2 carries no Markov rank");
  Check(weights->markov_w1.empty() && weights->markov_w2.empty() &&
            weights->conf_projection.empty() && weights->conf_bias.empty(),
        "DFlash-2 binds no Markov or confidence tensor");
  Check(weights->conf_projection_data.empty(),
        "DFlash-2 owns no converted confidence buffer");
  Check(!weights->selector_predecessor.empty() &&
            !weights->selector_successor.empty() &&
            !weights->selector_hidden.empty(),
        "DFlash-2 selector still bound");
  Check(df.conv_kernel_size == kKernel && df.conv_group_size == kGroup,
        "DFlash-2 conv config still read");
  Check(df.selector_rank == kRank && df.selector_top_k == 4U,
        "DFlash-2 selector config still read");
  Check(reader->GetMetadataBool("dflash.attention.causal").has_value(),
        "DFlash-2 causal key still required and read");
  Check(weights->config.vocab_size == kVocab,
        "DFlash-2 vocab still inferred from the selector");
  Check(df.rms_eps == 1e-6F, "DFlash-2 rms_eps == 1e-6F");
  Check(df.mask_token_id == 42U, "DFlash-2 mask id still read");
  const std::vector<std::uint32_t> expected_taps{1, 3};
  Check(df.target_layer_ids == expected_taps,
        "DFlash-2 taps still normalized to [1,3]");
  for (std::size_t i = 0; i < weights->layers.size(); ++i) {
    const auto& layer = weights->layers[i];
    if (layer.attention_conv_base.empty() || layer.ffn_conv_base.empty() ||
        layer.attention_conv_projection.empty() ||
        layer.ffn_conv_projection.empty()) {
      Check(false, "DFlash-2 conv tensors bound on layer " + std::to_string(i));
      return;
    }
  }
  Check(true, "DFlash-2 conv tensors bound on all " +
                  std::to_string(weights->layers.size()) + " layers");
}

// Ruling A: the discriminator is tensor emptiness, and the construction
// invariant refuses a MIXED artifact. No is_dspark flag reaches the executor.
gufo::speculative::QwenDFlashWeights BuildHandWeights(
    bool conv, bool selector, bool markov) {
  using gufo::models::QwenTensorRef;
  using gufo::speculative::QwenDFlashLayerWeights;
  using gufo::speculative::QwenDFlashWeights;
  QwenDFlashWeights weights;
  weights.config.hidden_size = kHidden;
  weights.config.vocab_size = kVocab;
  weights.config.intermediate_size = kIntermediate;
  weights.config.num_attention_heads = 32;
  weights.config.num_key_value_heads = 4;
  weights.config.head_dim = 128;
  auto& draft = weights.dflash_config;
  draft.block_size = 8;
  draft.conv_kernel_size = conv ? kKernel : 0U;
  draft.conv_group_size = conv ? kGroup : 0U;
  draft.selector_rank = selector ? kRank : 0U;
  draft.selector_top_k = selector ? 4U : 0U;
  draft.num_layers = 2;
  draft.target_layer_ids = {0U, 40U};
  draft.sliding_window = 2048;
  draft.layer_sliding_windows = {2048U, 2048U};
  draft.markov_rank = markov ? kRank : 0U;
  // Deliberately left at whatever the caller implies: the discriminator must
  // not be able to read it.
  draft.is_dspark = false;
  weights.layers.assign(draft.num_layers, QwenDFlashLayerWeights{});
  const QwenTensorRef filled{.data = reinterpret_cast<const void*>(0x1),
                              .type = gufo::core::GgmlType::kF32,
                              .num_elements = 64};
  for (auto& layer : weights.layers) {
    layer.attention_conv_base = conv ? filled : QwenTensorRef{};
    layer.ffn_conv_base = conv ? filled : QwenTensorRef{};
    layer.attention_conv_projection = conv ? filled : QwenTensorRef{};
    layer.ffn_conv_projection = conv ? filled : QwenTensorRef{};
  }
  weights.selector_predecessor = selector ? filled : QwenTensorRef{};
  weights.selector_successor = selector ? filled : QwenTensorRef{};
  weights.selector_hidden = selector ? filled : QwenTensorRef{};
  // The Markov matrices are [markov_rank, vocab] and the confidence projection
  // is [hidden + markov_rank]; the refusal checks those shapes by value.
  const QwenTensorRef markov_matrix{
      .data = reinterpret_cast<const void*>(0x1),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = std::size_t{kRank} * kVocab};
  weights.markov_w1 = markov ? markov_matrix : QwenTensorRef{};
  weights.markov_w2 = markov ? markov_matrix : QwenTensorRef{};
  weights.conf_projection =
      markov ? QwenTensorRef{
                   .data = reinterpret_cast<const void*>(0x1),
                   .type = gufo::core::GgmlType::kF32,
                   .num_elements = std::size_t{kHidden} + kRank}
             : QwenTensorRef{};
  weights.conf_bias = markov ? filled : QwenTensorRef{};
  return weights;
}

void TestDraftHeadDiscriminator() {
  std::cout << "ruling A: tensor-emptiness discriminator and construction "
               "invariant\n";
  using gufo::speculative::ClassifyDFlashDraftHead;
  using gufo::speculative::DFlashDraftHead;
  using gufo::speculative::DFlashGpuExecutorSupportRefusal;
  using gufo::speculative::IsDSparkDraft;

  const auto complete_2 = BuildHandWeights(true, true, false);
  const auto complete_spark = BuildHandWeights(false, false, true);
  Check(ClassifyDFlashDraftHead(complete_2) == DFlashDraftHead::kDFlash2,
        "conv + selector, no markov classifies as DFlash-2");
  Check(ClassifyDFlashDraftHead(complete_spark) == DFlashDraftHead::kDSpark,
        "no conv, no selector, markov + conf classifies as DSpark");
  Check(IsDSparkDraft(complete_spark), "DSpark reads as DSpark");
  Check(!IsDSparkDraft(complete_2), "DFlash-2 does not read as DSpark");

  // The flag cannot decide: is_dspark is false on BOTH hand-built artifacts and
  // the classification is still right.
  Check(complete_spark.dflash_config.is_dspark == false,
        "the fixture proves the flag is not consulted");

  // Values: a working DSpark artifact is NOT refused.
  const std::string spark_refusal =
      DFlashGpuExecutorSupportRefusal(complete_spark);
  Check(spark_refusal.empty(),
        "a complete DSpark artifact is NOT refused: " + spark_refusal);
  const std::string dflash2_refusal =
      DFlashGpuExecutorSupportRefusal(complete_2);
  Check(dflash2_refusal.empty(),
        "a complete DFlash-2 artifact is NOT refused: " + dflash2_refusal);

  // MIXED artifacts refuse, and the message says which tensors are wrong.
  // {0, 0} with a complete Markov head IS complete DSpark, so it is not in
  // the mixed list: every other combination mixes a conv or selector tensor
  // into a DSpark artifact, or leaves a DFlash-2 artifact half-built.
  const std::size_t mixed_cases[][2] = {{1, 0}, {0, 1}, {1, 1}};
  for (const auto& combination : mixed_cases) {
    const auto mixed =
        BuildHandWeights(combination[0] != 0, combination[1] != 0, true);
    const std::string refusal = DFlashGpuExecutorSupportRefusal(mixed);
    Check(!refusal.empty(),
          "conv=" + std::to_string(combination[0]) +
              " selector=" + std::to_string(combination[1]) +
              " with markov refuses");
    Check(refusal.find("neither a complete DFlash-2 nor a complete DSpark") !=
              std::string::npos,
          "mixed refusal names the invariant it broke");
  }
  // The real DFlash-2 fixture, loaded: it must still classify as DFlash-2 and
  // must still construct. Proved by loading, not by reading.
  {
    const auto buffer = BuildDFlash2Fixture();
    std::string error;
    auto reader = gufo::core::GgufReader::OpenMemory(buffer.data(),
                                                      buffer.size(), &error);
    Check(reader != nullptr, "DFlash-2 fixture opens: " + error);
    if (reader != nullptr) {
      auto loaded = gufo::speculative::QwenDFlashWeights::LoadFromGguf(
          *reader, &error);
      Check(loaded.has_value(), "DFlash-2 fixture loads: " + error);
      if (loaded.has_value()) {
        Check(ClassifyDFlashDraftHead(*loaded) == DFlashDraftHead::kDFlash2,
              "the loaded DFlash-2 artifact classifies as DFlash-2");
        Check(!IsDSparkDraft(*loaded),
              "the loaded DFlash-2 artifact does not read as DSpark");
        const std::string refusal =
            DFlashGpuExecutorSupportRefusal(*loaded);
        Check(refusal.empty(),
              "the loaded DFlash-2 artifact still constructs: " + refusal);
      }
    }
  }

  // Convolution tensors present but the selector missing: refused.
  {
    auto mixed = BuildHandWeights(true, false, false);
    Check(!DFlashGpuExecutorSupportRefusal(mixed).empty(),
          "convolutions without the selector refuse");
  }
  // Half a Markov head: markov_w2 present, conf_proj absent.
  {
    auto mixed = BuildHandWeights(false, false, true);
    mixed.conf_projection = {};
    Check(!DFlashGpuExecutorSupportRefusal(mixed).empty(),
          "a half-loaded Markov head refuses");
    Check(IsDSparkDraft(mixed),
          "a half-loaded Markov head still reads as DSpark, not DFlash-2");
  }
  // A structurally complete DFlash-2 with a broken divisor still refuses.
  {
    auto broken = BuildHandWeights(true, true, false);
    broken.dflash_config.conv_group_size = 0;
    const std::string refusal = DFlashGpuExecutorSupportRefusal(broken);
    Check(!refusal.empty(), "DFlash-2 with conv_group_size 0 refuses");
    Check(refusal.find("conv_group_size") != std::string::npos,
          "the refusal names the zero divisor");
  }
  {
    auto broken = BuildHandWeights(true, true, false);
    broken.dflash_config.selector_top_k = 0;
    Check(!DFlashGpuExecutorSupportRefusal(broken).empty(),
          "DFlash-2 with selector_top_k 0 refuses");
  }
  {
    auto broken = BuildHandWeights(false, false, true);
    broken.dflash_config.markov_rank = 8;
    broken.markov_w2.num_elements = 16;
    Check(!DFlashGpuExecutorSupportRefusal(broken).empty(),
          "a Markov matrix that is not [rank, vocab] refuses");
  }
}

// Ruling B: the host guard the conv launcher uses, so the device division can
// never observe zero.
void TestGroupedConvLaunchRefusal() {
  std::cout << "ruling B: conv launcher refuses group_size 0 on the host\n";
  using gufo::speculative::DFlashDynamicCoefficientCount;
  using gufo::speculative::DFlashGroupedConvLaunchRefusal;

  const std::string zero = DFlashGroupedConvLaunchRefusal(5120, 4, 0);
  Check(!zero.empty(), "group_size 0 is refused");
  Check(zero.find("conv_group_size") != std::string::npos,
        "the refusal names conv_group_size");
  Check(zero.size() > 40U, "the refusal explains why, not a stub");
  Check(DFlashGroupedConvLaunchRefusal(5120, 4, kGroup).empty(),
        "a real DFlash-2 launch is not refused");
  Check(!DFlashGroupedConvLaunchRefusal(5120, 4, 7).empty(),
        "hidden 5120 with group 7 (not a divisor) is refused");
  Check(!DFlashGroupedConvLaunchRefusal(5120, 0, kGroup).empty(),
        "kernel_size 0 is refused");

  // The sizing helper a DSpark artifact goes through instead of dividing.
  Check(DFlashDynamicCoefficientCount(5120, 4, kGroup) ==
            std::size_t{2} * 4 * (5120 / kGroup),
        "DFlash-2 coefficient count is 2 * kernel * groups");
  Check(DFlashDynamicCoefficientCount(5120, 0, 0) == 0U,
        "DSpark coefficient count is 0 without dividing");
  Check(DFlashDynamicCoefficientCount(5120, 4, 0) == 0U,
        "kernel 4 with group 0 is 0, not a division by zero");
}

// Ruling E: the memory estimate, asserted by VALUE for both heads.
void TestDraftScratchElements() {
  std::cout << "ruling E: DSpark memory estimate\n";
  using gufo::speculative::DFlashDraftScratchElements;

  constexpr std::uint32_t kContext = 512;
  const auto weights2 = BuildHandWeights(true, true, false);
  const auto weights_spark = BuildHandWeights(false, false, true);
  const std::size_t selector_scratch = 8 * 16;

  // Hand-computed expectation for a DFlash-2 draft.
  const std::size_t hidden = kHidden;
  const std::size_t kv = std::size_t{4} * 128;
  const std::size_t attention = std::size_t{32} * 128;
  static_assert(kHidden == 128 && kIntermediate == 256 && kRank == 8 &&
                    kVocab == 512 && kKernel == 2 && kGroup == 16,
                "this expectation is hand-computed from the fixture sizes");
  const std::size_t dynamic = std::size_t{2} * kKernel * (hidden / kGroup);
  const std::size_t block = 8;
  const std::size_t injection = std::min<std::size_t>(kContext, 256);
  const std::size_t shared = std::max(injection, block);
  const std::size_t ffn =
      std::max<std::size_t>(kIntermediate, selector_scratch);
  const std::size_t expected_2 =
      injection * 2 * hidden + shared * (2 * hidden + 2 * kv) +
      block * (3 * hidden + dynamic + 2 * attention + 2 * ffn + kVocab + 3) +
      block * (kRank + 2 * 4);
  const std::size_t got_2 = DFlashDraftScratchElements(
      weights2.config, weights2.dflash_config, kContext, selector_scratch,
      false);
  Check(got_2 == expected_2,
        "DFlash-2 scratch is " + std::to_string(expected_2) + ", got " +
            std::to_string(got_2));

  // DSpark: no convolution coefficients, no selector terms, plus the Markov
  // bias row, the markov_w1 row, the predecessor id and the emitted token row.
  const std::size_t markov_terms = kVocab + kRank + 1 + block;
  const std::size_t expected_spark =
      injection * 2 * hidden + shared * (2 * hidden + 2 * kv) +
      block * (3 * hidden + 2 * attention + 2 * kIntermediate + kVocab + 3) +
      markov_terms;
  const std::size_t got_spark = DFlashDraftScratchElements(
      weights_spark.config, weights_spark.dflash_config, kContext, 0, true);
  Check(got_spark == expected_spark,
        "DSpark scratch is " + std::to_string(expected_spark) + ", got " +
            std::to_string(got_spark));

  // The zero divisor the estimate used to compute is never evaluated: a DSpark
  // config with conv_group_size set to a real value must not change anything.
  {
    auto moved = weights_spark;
    moved.dflash_config.conv_group_size = kGroup;
    moved.dflash_config.conv_kernel_size = kKernel;
    Check(DFlashDraftScratchElements(moved.config, moved.dflash_config,
                                     kContext, 0, true) == expected_spark,
          "DSpark sizing ignores conv_group_size entirely");
  }
  // And the DSpark estimate is not the DFlash-2 one with a divisor.
  Check(got_spark != got_2, "DSpark and DFlash-2 sizes differ");
  {
    auto broken = weights_spark;
    broken.dflash_config.markov_rank = 0;
    Check(DFlashDraftScratchElements(broken.config, broken.dflash_config,
                                     kContext, 0, true) ==
              expected_spark - markov_terms,
          "an artifact with no Markov rank reserves no Markov scratch");
    Check(DFlashDraftScratchElements(weights_spark.config,
                                     weights_spark.dflash_config, kContext, 0,
                                     false) ==
              expected_spark - markov_terms,
          "the same config sized as DFlash-2 reserves no Markov scratch either");
  }
}

void TestSyntheticDSparkF16Conversion() {
  std::cout << "synthetic DSpark F16 -> F32 conversion\n";
  // IEEE-754 binary16 bit patterns chosen to cover every class.
  const std::vector<std::uint16_t> conf_halves{
      0x0000,  // +0
      0x8000,  // -0
      0x0001,  // smallest positive subnormal
      0x03FF,  // largest subnormal
      0x0400,  // smallest normal
      0x3C00,  // 1.0
      0xBC00,  // -1.0
      0x3555,  // 0.333251953125
      0x7BFF,  // 65504
      0xFBFF,  // -65504
      0x7C00,  // +inf
      0xFC00,  // -inf
      0x7E00,  // NaN
  };
  const std::size_t wanted = kHidden + kRank;
  std::vector<std::uint16_t> halves = conf_halves;
  halves.resize(wanted);
  for (std::size_t i = conf_halves.size(); i < wanted; ++i) {
    halves[i] = static_cast<std::uint16_t>(0x3C00 + (i & 0x3F));  // ~1..1.5
  }

  const auto buffer = BuildDSparkF16Fixture(halves);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenMemory(buffer.data(), buffer.size(),
                                                   &error);
  Check(reader != nullptr, "fixture opens: " + error);
  if (reader == nullptr) {
    return;
  }
  auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Check(weights.has_value(), "DSpark fixture loads: " + error);
  if (!weights.has_value()) {
    return;
  }
  const auto& df = weights->dflash_config;
  Check(df.is_dspark, "DSpark fixture classified as DSpark (no conv/selector/"
                      "causal keys present)");
  Check(weights->config.vocab_size == kVocab,
        "vocab inferred from markov_w1, not the selector");
  Check(df.has_confidence_head, "has_confidence_head read");

  // M4: converted to F32 at load, values bit-identical to the reference.
  Check(weights->conf_projection.type == gufo::core::GgmlType::kF32,
        "conf_projection is F32 after load (artifact stores F16)");
  Check(weights->conf_projection_data.size() == wanted,
        "owned conversion buffer holds " + std::to_string(wanted) + " floats");
  const auto* conf_meta = reader->FindTensor("conf_proj.weight");
  Check(conf_meta != nullptr && conf_meta->type == gufo::core::GgmlType::kF16,
        "artifact conf_proj.weight really is F16");

  const auto* raw = static_cast<const std::uint16_t*>(conf_meta->data);
  std::size_t bit_mismatch = 0;
  std::size_t nan_mismatch = 0;
  for (std::size_t i = 0; i < wanted; ++i) {
    const float want = ReferenceHalfToFloat(raw[i]);
    const float got = weights->conf_projection_data[i];
    const bool want_is_nan = std::isnan(want);
    if (want_is_nan || std::isnan(got)) {
      if (!(want_is_nan && std::isnan(got))) {
        ++nan_mismatch;
      }
      continue;
    }
    if (FloatBits(want) != FloatBits(got)) {
      ++bit_mismatch;
    }
  }
  Check(bit_mismatch == 0U,
        "every converted value is BIT-IDENTICAL to the independent half->float "
        "reference (mismatches=" + std::to_string(bit_mismatch) + ")");
  Check(nan_mismatch == 0U,
        "NaN patterns stay NaN (mismatches=" + std::to_string(nan_mismatch) +
            ")");
  Check(FloatBits(weights->conf_projection_data[1]) == 0x80000000U,
        "negative zero keeps its sign bit");
  Check(weights->conf_projection_data[2] ==
            ReferenceHalfToFloat(0x0001U),
        "smallest subnormal matches");
  Check(std::isinf(weights->conf_projection_data[10]) &&
            weights->conf_projection_data[10] > 0,
        "+inf preserved");
  Check(std::isnan(weights->conf_projection_data[12]), "NaN preserved");

  // Negative control for the B-R4/B-R5 shape: reading the same bytes as F32
  // must NOT agree with the reference, or this test could not detect it.
  const auto* as_f32 = static_cast<const float*>(conf_meta->data);
  std::size_t as_f32_matches = 0;
  for (std::size_t i = 0; i < wanted; ++i) {
    if (FloatBits(as_f32[i]) == FloatBits(ReferenceHalfToFloat(raw[i]))) {
      ++as_f32_matches;
    }
  }
  Check(as_f32_matches < 4U,
        "negative control: interpreting the F16 bytes as F32 disagrees with "
        "the reference (" + std::to_string(as_f32_matches) +
            " accidental matches out of " + std::to_string(wanted) + ")");

  // The production F16 reader must agree with the conversion too.
  const gufo::models::QwenTensorRef raw_ref{
      .data = conf_meta->data,
      .type = gufo::core::GgmlType::kF16,
      .num_elements = conf_meta->ElementCount(),
  };
  std::size_t reader_mismatch = 0;
  for (std::size_t i = 0; i < wanted; ++i) {
    const float want = raw_ref.Get(i);
    const float got = weights->conf_projection_data[i];
    if (std::isnan(want) || std::isnan(got)) {
      continue;
    }
    if (FloatBits(want) != FloatBits(got)) {
      ++reader_mismatch;
    }
  }
  Check(reader_mismatch == 0U,
        "conversion agrees with the production F16 reader (mismatches=" +
            std::to_string(reader_mismatch) + ")");
}

void TestEpsilonMismatchRefused() {
  std::cout << "epsilon mismatch refusal (M5)\n";
  GgufBuilder builder;
  builder.String("general.architecture", "dflash");
  builder.Uint32("dflash.block_count", kLayers);
  builder.Uint32("dflash.context_length", 4096);
  builder.Uint32("dflash.embedding_length", kHidden);
  builder.Uint32("dflash.feed_forward_length", kIntermediate);
  builder.Uint32("dflash.attention.head_count", kHeads);
  builder.Uint32("dflash.attention.head_count_kv", kKvHeads);
  builder.Uint32("dflash.attention.key_length", kHeadDim);
  builder.Uint32("dflash.attention.value_length", kHeadDim);
  builder.Uint32("dflash.rope.dimension_count", kHeadDim);
  builder.Float32("dflash.rope.freq_base", 10000000.0F);
  builder.Float32("dflash.attention.layer_norm_rms_epsilon", 1e-5F);
  builder.Uint32("dflash.block_size", 8);
  builder.Uint32("dflash.attention.sliding_window", 64);
  builder.Uint32("tokenizer.ggml.mask_token_id", 42);
  builder.Bool("dflash.has_confidence_head", true);
  builder.Uint64Array("dflash.target_layers", {2, 4});
  AddPlainLayers(builder);
  builder.TensorF32("markov_w1.weight", {kRank, kVocab});
  builder.TensorF32("markov_w2.weight", {kRank, kVocab});
  const auto buffer = builder.Build();

  std::string error;
  auto reader = gufo::core::GgufReader::OpenMemory(buffer.data(), buffer.size(),
                                                   &error);
  Check(reader != nullptr, "fixture opens: " + error);
  if (reader == nullptr) {
    return;
  }
  auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Check(!weights.has_value(),
        "a draft declaring epsilon 1e-5 is REFUSED (kernels are 1e-6)");
  Check(error.find("layer_norm_rms_epsilon") != std::string::npos,
        "refusal names the epsilon key: " + error);
}

void TestRealDSpark(const std::filesystem::path& path) {
  std::cout << "real DSpark artifact: " << path << "\n";
  const std::size_t rss_before = ResidentKilobytes();
  std::cout << "  RSS before open: " << rss_before << " KiB\n";

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Check(reader != nullptr, "reader opens: " + error);
  if (reader == nullptr) {
    return;
  }
  const std::size_t rss_mapped = ResidentKilobytes();

  auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  const std::size_t rss_after = ResidentKilobytes();
  std::cout << "  RSS after mmap:  " << rss_mapped << " KiB\n";
  std::cout << "  RSS after load:  " << rss_after << " KiB\n";
  Check(weights.has_value(), "DSpark artifact loads: " + error);
  if (!weights.has_value()) {
    return;
  }

  const auto& cfg = weights->config;
  const auto& df = weights->dflash_config;

  Check(df.is_dspark, "classified as DSpark");
  Check(df.markov_rank == 256U, "markov_rank == 256");
  Check(weights->markov_w1.type == gufo::core::GgmlType::kQ4_K,
        "markov_w1 is Q4_K");
  Check(weights->markov_w2.type == gufo::core::GgmlType::kQ4_K,
        "markov_w2 is Q4_K");
  Check(weights->selector_predecessor.empty() &&
            weights->selector_successor.empty() &&
            weights->selector_hidden.empty(),
        "no DFlash-2 selector bound on a DSpark draft");
  bool conv_unbound = true;
  for (const auto& layer : weights->layers) {
    conv_unbound = conv_unbound && layer.attention_conv_base.empty() &&
                   layer.attention_conv_projection.empty() &&
                   layer.ffn_conv_base.empty() &&
                   layer.ffn_conv_projection.empty();
  }
  Check(conv_unbound, "no conv tensor bound on any layer");
  Check(df.conv_kernel_size == 0U && df.conv_group_size == 0U &&
            df.selector_rank == 0U && df.selector_top_k == 0U,
        "conv/selector config left zero");
  Check(df.rms_eps == 1e-6F, "rms_eps == 1e-6F (M5)");
  // M8 on the real artifact: sliding_window 2048 with the pattern [true x5],
  // so all five layers are sliding-window. The reference reaches the same
  // state (n_swa 2048, is_swa_any 1) for this file.
  Check(df.sliding_window == 2048U,
        "artifact sliding_window == 2048 (M8)");
  Check(df.layer_sliding_windows ==
            std::vector<std::uint32_t>(df.num_layers, 2048U),
        "artifact pattern [true x5] puts window 2048 on every layer (got " +
            std::to_string(df.layer_sliding_windows.size()) + " entries)");
  Check(gufo::speculative::MaxSlidingWindow(df) == 2048U,
        "artifact widest layer window == 2048");
  // The Markov row is NOT computed on the real artifact here: QwenTensorRef::Get
  // re-dequantizes a whole 256-element block per element read
  // (src/models/qwen/state.hpp:78-96), so a full row would be ~16 GFLOP of
  // pointless work. ComputeMarkovBiasRow is the arithmetic spec for the GPU
  // kernel, exercised on F32 fixtures above.

  Check(cfg.hidden_size == 5120U, "hidden_size == 5120 (U1 draft)");
  Check(cfg.vocab_size == 248320U,
        "vocab_size == 248320 inferred from markov_w1 (M1)");
  Check(cfg.num_attention_heads == 20U && cfg.num_key_value_heads == 4U &&
            cfg.head_dim == 256U,
        "heads 20 / kv 4 / head_dim 256");
  Check(cfg.intermediate_size == 17408U, "ffn width 17408");
  Check(df.num_layers == 5U, "block_count == 5");
  Check(df.block_size == 8U, "block_size == 8");
  Check(df.sliding_window == 2048U, "sliding_window == 2048");
  Check(df.mask_token_id == 248077U, "mask_token_id == 248077 (M6)");
  Check(df.has_confidence_head, "has_confidence_head");
  Check(reader->FindTensor("dflash.attention.causal") == nullptr,
        "artifact carries no causal key and the load still succeeded");
  Check(reader->FindTensor("dflash.conv_kernel_size") == nullptr,
        "artifact carries no conv key and the load still succeeded");
  Check(reader->FindTensor("selector_predecessor.weight") == nullptr,
        "negative control: no selector_predecessor.weight in this artifact");

  Check(df.target_layer_ids.size() == 8U, "8 target taps (M2)");
  const std::vector<std::uint32_t> expected_taps{3, 11, 19, 27, 35,
                                                43, 51, 59};
  Check(df.target_layer_ids == expected_taps,
        "taps normalized to layer OUTPUT indices [3,11,...,59] (M9)");

  CheckShape(*reader, weights->markov_w1, "markov_w1.weight", 63569920U);
  CheckShape(*reader, weights->markov_w2, "markov_w2.weight", 63569920U);
  CheckShape(*reader, weights->fc_projection, "fc.weight", 209715200U);
  CheckShape(*reader, weights->fc_norm, "enc.output_norm.weight", 5120U);
  CheckShape(*reader, weights->output_norm, "output_norm.weight", 5120U);
  CheckShape(*reader, weights->conf_bias, "conf_proj.bias", 1U);
  for (std::size_t i = 0; i < weights->layers.size(); ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto& t = weights->layers[i].transformer;
    CheckShape(*reader, t.attn_norm, prefix + "attn_norm.weight", 5120U);
    CheckShape(*reader, t.attn_q, prefix + "attn_q.weight", 5120U * 5120U);
    CheckShape(*reader, t.attn_k, prefix + "attn_k.weight", 5120U * 1024U);
    CheckShape(*reader, t.attn_v, prefix + "attn_v.weight", 5120U * 1024U);
    CheckShape(*reader, t.attn_output, prefix + "attn_output.weight",
               5120U * 5120U);
    CheckShape(*reader, t.attn_q_norm, prefix + "attn_q_norm.weight", 256U);
    CheckShape(*reader, t.attn_k_norm, prefix + "attn_k_norm.weight", 256U);
    CheckShape(*reader, t.ffn_norm, prefix + "ffn_norm.weight", 5120U);
    CheckShape(*reader, t.ffn_gate, prefix + "ffn_gate.weight", 5120U * 17408U);
    CheckShape(*reader, t.ffn_up, prefix + "ffn_up.weight", 5120U * 17408U);
    CheckShape(*reader, t.ffn_down, prefix + "ffn_down.weight", 5120U * 17408U);
  }

  // F16 -> F32 on the real artifact, against the same independent reference.
  Check(weights->conf_projection.type == gufo::core::GgmlType::kF32,
        "conf_projection converted to F32, not left F16");
  const auto* conf_meta = reader->FindTensor("conf_proj.weight");
  Check(conf_meta != nullptr && conf_meta->type == gufo::core::GgmlType::kF16,
        "artifact conf_proj.weight is F16");
  Check(weights->conf_projection_data.size() == 5376U,
        "owned conversion buffer has 5376 floats");
  std::size_t mismatches = 0;
  std::size_t nonzero = 0;
  if (conf_meta != nullptr &&
      conf_meta->type == gufo::core::GgmlType::kF16) {
    const auto* raw = static_cast<const std::uint16_t*>(conf_meta->data);
    for (std::size_t i = 0; i < weights->conf_projection_data.size(); ++i) {
      const float want = ReferenceHalfToFloat(raw[i]);
      const float got = weights->conf_projection_data[i];
      if (FloatBits(want) != FloatBits(got)) {
        ++mismatches;
      }
      if (want != 0.0F) {
        ++nonzero;
      }
    }
  }
  std::cout << "  conf_proj: " << weights->conf_projection_data.size()
            << " values, " << nonzero << " non-zero\n";
  Check(mismatches == 0U,
        "real artifact: every converted value is bit-identical to the "
        "independent half->float reference (mismatches=" +
            std::to_string(mismatches) + ")");
  Check(nonzero > 5000U,
        "conversion is not all zeros (" + std::to_string(nonzero) +
            " non-zero of 5376)");
}

// ---------------------------------------------------------------------------
// M8 sliding_window_pattern and M3 Markov bias. Both take the pattern / bias
// from a DSpark fixture whose window is kWindow (=64), matching the shape the
// real artifact declares (sliding_window 2048, pattern [true x5]).
// ---------------------------------------------------------------------------
constexpr std::uint32_t kWindow = 64;

std::vector<std::uint8_t> BuildDSparkPatternFixture(
    const std::vector<bool>& pattern) {
  GgufBuilder builder;
  // AddCommonTransformerMetadata already declares
  // dflash.attention.sliding_window = kWindow; adding it again would make the
  // fixture unreadable (duplicate metadata key).
  AddCommonTransformerMetadata(builder);
  static_assert(kWindow == 64U);
  // An empty vector means the artifact declares no pattern key at all, which
  // is the DFlash-2 case, not a zero-length array.
  if (!pattern.empty()) {
    builder.BoolArray("dflash.attention.sliding_window_pattern", pattern);
  }
  builder.Uint64Array("dflash.target_layers", {2, 4});
  AddPlainLayers(builder);
  builder.TensorF32("markov_w1.weight", {kRank, kVocab});
  builder.TensorF32("markov_w2.weight", {kRank, kVocab});
  return builder.Build();
}

// Expected Markov bias, computed here from the builder's fill pattern
// (element i == i % 97) with the indexing written out by hand: row t of a
// [rank, vocab] matrix starts at element t * rank, NOT t * vocab.
float ReferenceMarkovBias(std::size_t candidate, std::size_t previous,
                          std::size_t rank) {
  float total = 0.0F;
  for (std::size_t component = 0; component < rank; ++component) {
    const auto w2 = static_cast<float>(((candidate * rank) + component) % 97);
    const auto w1 = static_cast<float>(((previous * rank) + component) % 97);
    total += w2 * w1;
  }
  return total;
}

void TestSlidingWindowPattern() {
  std::cout << "sliding_window_pattern semantics\n";
  std::string error;
  // Mixed pattern: true == sliding window, false == full (unbounded).
  const auto mixed = BuildDSparkPatternFixture({true, true, false, true, false});
  auto reader = gufo::core::GgufReader::OpenMemory(mixed.data(), mixed.size(),
                                                   &error);
  Check(reader != nullptr, "mixed-pattern fixture opens: " + error);
  if (reader == nullptr) {
    return;
  }
  auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Check(weights.has_value(), "mixed-pattern fixture loads: " + error);
  if (!weights.has_value()) {
    return;
  }
  const auto& df = weights->dflash_config;
  const std::vector<std::uint32_t> expected{kWindow, kWindow, 0U, kWindow, 0U};
  Check(df.layer_sliding_windows == expected,
        "pattern [T,T,F,T,F] with window 64 gives [64,64,0,64,0] (got " +
            std::to_string(df.layer_sliding_windows.size()) + " entries)");
  for (std::uint32_t layer = 0; layer < df.num_layers; ++layer) {
    Check(gufo::speculative::LayerSlidingWindow(df, layer) == expected[layer],
          "layer " + std::to_string(layer) + " window is " +
              std::to_string(expected[layer]));
  }
  Check(gufo::speculative::MaxSlidingWindow(df) == kWindow,
        "widest layer window is 64");
  // The layers stay attention layers: this flag means window, not SSM.
  for (const auto& layer : weights->layers) {
    if (!layer.transformer.is_full_attention) {
      Check(false, "sliding_window_pattern must not clear is_full_attention");
      return;
    }
  }
  Check(true, "all layers remain attention layers");

  // All-true is the real artifact's pattern: every layer sliding.
  const auto all_true = BuildDSparkPatternFixture({true, true, true, true, true});
  reader = gufo::core::GgufReader::OpenMemory(all_true.data(), all_true.size(),
                                              &error);
  Check(reader != nullptr, "all-true fixture opens: " + error);
  if (reader != nullptr) {
    auto all_sliding =
        gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
    Check(all_sliding.has_value(), "all-true fixture loads: " + error);
    if (all_sliding) {
      Check(all_sliding->dflash_config.layer_sliding_windows ==
                std::vector<std::uint32_t>(kLayers, kWindow),
            "pattern [T x5] gives window 64 on all five layers");
    }
  }

  // Absent key: unchanged behaviour, every layer sliding.
  const auto absent = BuildDSparkPatternFixture({});
  reader = gufo::core::GgufReader::OpenMemory(absent.data(), absent.size(),
                                              &error);
  Check(reader != nullptr, "pattern-less fixture opens: " + error);
  if (reader != nullptr) {
    auto legacy =
        gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
    Check(legacy.has_value(), "pattern-less fixture loads: " + error);
    if (legacy) {
      Check(legacy->dflash_config.layer_sliding_windows.empty(),
            "no pattern key leaves the per-layer table empty");
      Check(gufo::speculative::LayerSlidingWindow(
                legacy->dflash_config, 3) == 64U,
            "absent pattern still drafts with the scalar window (64)");
      Check(gufo::speculative::MaxSlidingWindow(legacy->dflash_config) == 64U,
            "absent pattern keeps the scalar window as the widest");
    }
  }

  // Wrong length: refused, with a message that names both counts.
  const auto short_pattern = BuildDSparkPatternFixture({true, false});
  reader = gufo::core::GgufReader::OpenMemory(short_pattern.data(),
                                              short_pattern.size(), &error);
  Check(reader != nullptr, "short-pattern fixture opens: " + error);
  if (reader != nullptr) {
    error.clear();
    auto refused =
        gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
    Check(!refused.has_value(), "2-entry pattern for 5 layers is refused");
    Check(error.find("sliding_window_pattern") != std::string::npos,
          "refusal names the key: " + error);
    Check(error.find('5') != std::string::npos,
          "refusal names the layer count: " + error);
  }
}

// Ruling 2: the proposal is capped at the caller-requested `draft_count`
// AFTER the anchor is handled. The anchor slot still counts as a prediction
// slot; it just must not push the proposal past draft_count. Every case below
// asserts the two VALUES the executor reads (`rows` to size the GEMM and the
// download, `first_row` to offset the chain), not a shape.
//
// The bug this replaces: rows was `anchor_predicted ? block_count :
// draft_count`, and block_count == draft_count + 1, so an anchor-predicted
// DSpark artifact returned one token more than the caller asked for.
void TestDSparkChainRangeCap() {
  std::cout << "ruling 2: proposal capped at draft_count after the anchor\n";
  using gufo::speculative::DSparkChainRange;
  using gufo::speculative::DSparkChainRangeFor;

  struct Case {
    std::size_t draft_count;
    bool sample_from_anchor;
    std::size_t want_rows;
    std::size_t want_first_row;
  };
  // block_size is 8 in the fixtures, so the executor clamps draft_count to 7
  // before this is called; 0..7 is the whole reachable range.
  const Case cases[] = {
      // Anchor predicted (the reference default, key absent): the chain starts
      // at row 0 and still yields exactly what the caller asked for.
      {7, true, 7, 0},   // block_size - 1, the widest a real artifact reaches
      {1, true, 1, 0},   // a single requested token
      {3, true, 3, 0},   // an interior value
      {0, true, 0, 0},   // nothing asked, nothing proposed
      // Anchor NOT predicted: row 0 is a bonus anchor, chain starts at row 1.
      // Unchanged geometry, and still exactly draft_count long.
      {7, false, 7, 1},
      {1, false, 1, 1},
      {3, false, 3, 1},
      {0, false, 0, 1},
  };
  for (const Case& c : cases) {
    const std::size_t block_count = c.draft_count + 1U;
    const DSparkChainRange got =
        DSparkChainRangeFor(block_count, c.draft_count, c.sample_from_anchor);
    const std::string label =
        "draft_count=" + std::to_string(c.draft_count) +
        " anchor=" + (c.sample_from_anchor ? "true" : "false");
    Check(got.rows == c.want_rows,
          label + ": rows == " + std::to_string(c.want_rows) + ", got " +
              std::to_string(got.rows));
    Check(got.first_row == c.want_first_row,
          label + ": first_row == " + std::to_string(c.want_first_row) +
              ", got " + std::to_string(got.first_row));
    // The invariant, stated independently of the table: the caller asked for
    // draft_count and never gets more, under either anchor setting.
    Check(got.rows <= c.draft_count,
          label + ": rows " + std::to_string(got.rows) +
              " never exceeds the requested " + std::to_string(c.draft_count));
  }

  // The anchor switch is a START offset only. Flipping it must never change the
  // length the caller receives -- that is the whole ruling.
  for (std::size_t n = 0; n <= 7U; ++n) {
    const DSparkChainRange on = DSparkChainRangeFor(n + 1U, n, true);
    const DSparkChainRange off = DSparkChainRangeFor(n + 1U, n, false);
    Check(on.rows == off.rows,
          "n=" + std::to_string(n) + ": anchor setting changes rows (" +
              std::to_string(on.rows) + " vs " + std::to_string(off.rows) +
              ")");
    Check(on.first_row == 0U && off.first_row == 1U,
          "n=" + std::to_string(n) + ": anchor setting changes the start row");
  }

  // The regression itself, spelled out: an anchor-predicted block is one row
  // WIDER than the request, and the cap is what keeps the proposal at the
  // requested length. Removing the cap makes rows == block_count.
  {
    const std::size_t draft_count = 7;
    const std::size_t block_count = draft_count + 1U;
    const DSparkChainRange got =
        DSparkChainRangeFor(block_count, draft_count, true);
    Check(block_count == draft_count + 1U && got.rows == draft_count,
          "anchor-predicted block of 8 rows proposes 7, not 8");
    Check(got.rows != block_count,
          "the proposal is shorter than the block it was computed from");
  }

  // Degenerate input: a block with no row to start from must propose nothing
  // rather than underflow the length.
  {
    const DSparkChainRange got = DSparkChainRangeFor(0U, 4U, false);
    Check(got.rows == 0U && got.first_row == 1U,
          "an empty block with a bonus anchor proposes 0 rows");
  }
  // A block too small for the anchor-predicted start row still proposes what
  // it can, never more.
  {
    const DSparkChainRange got = DSparkChainRangeFor(1U, 9U, true);
    Check(got.rows == 1U, "a 1-row anchor-predicted block proposes its 1 row");
  }

  // DFlash-2 must be untouched by any of this: it never reaches this helper,
  // and its selector geometry still drafts from row 1 over draft_count rows.
  // Asserted through the discriminator, not by reading the executor.
  {
    const auto two = BuildHandWeights(true, true, false);
    const auto spark = BuildHandWeights(false, false, true);
    Check(!gufo::speculative::IsDSparkDraft(two),
          "DFlash-2 does not take the DSpark chain path");
    Check(gufo::speculative::IsDSparkDraft(spark),
          "DSpark does take the DSpark chain path");
    Check(DSparkChainRangeFor(8, 7, true).rows ==
              DSparkChainRangeFor(8, 7, false).rows,
          "DFlash-2's draft_count-row geometry is unaffected by the cap");
  }
}

void TestMarkovBiasValues() {
  std::cout << "Markov transition bias values\n";
  std::string error;
  const auto buffer = BuildDSparkPatternFixture({true, true, true, true, true});
  auto reader = gufo::core::GgufReader::OpenMemory(buffer.data(), buffer.size(),
                                                   &error);
  Check(reader != nullptr, "fixture opens: " + error);
  if (reader == nullptr) {
    return;
  }
  auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Check(weights.has_value(), "fixture loads: " + error);
  if (!weights.has_value()) {
    return;
  }
  const auto& df = weights->dflash_config;
  Check(df.markov_rank == kRank, "rank read from markov_w1.ne[0]");
  Check(weights->config.vocab_size == kVocab, "vocab read from markov_w1.ne[1]");

  const std::vector<std::uint32_t> probes{0U, 1U, 7U, kVocab - 1U};
  std::size_t mismatches = 0;
  std::size_t compared = 0;
  for (const auto previous : probes) {
    const auto bias = gufo::speculative::ComputeMarkovBiasRow(*weights, previous);
    if (!bias.has_value()) {
      Check(false, "bias row for previous=" + std::to_string(previous));
      return;
    }
    if (bias->size() != kVocab) {
      Check(false, "bias row length is vocab for previous=" +
                       std::to_string(previous));
      return;
    }
    for (std::size_t candidate = 0; candidate < kVocab; ++candidate) {
      const float want = ReferenceMarkovBias(candidate, previous, kRank);
      if (FloatBits((*bias)[candidate]) != FloatBits(want)) {
        ++mismatches;
      }
      ++compared;
    }
  }
  Check(mismatches == 0U,
        "every bias value matches the hand-written reference (" +
            std::to_string(compared) + " compared, " +
            std::to_string(mismatches) + " mismatched)");

  // Two values worked out by hand, so a whole-class indexing error (row-major
  // vs column-major) cannot pass as "close enough".
  const auto zero = gufo::speculative::ComputeMarkovBiasRow(*weights, 0);
  Check(zero.has_value() && FloatBits((*zero)[0]) == FloatBits(140.0F),
        "bias(prev=0, cand=0) == 140.0 exactly (sum r*r, r=0..7)");
  Check(zero.has_value() && FloatBits((*zero)[1]) == FloatBits(364.0F),
        "bias(prev=0, cand=1) == 364.0 exactly (sum (8+r)*r)");
  // A transposed read would give the candidate 0 row for candidate 1.
  Check(zero.has_value() && (*zero)[1] != (*zero)[0],
        "candidate 1 is not the candidate 0 row");

  // Out-of-range previous token must be refused, not clamped or read past.
  Check(!gufo::speculative::ComputeMarkovBiasRow(*weights, kVocab).has_value(),
        "previous == vocab is refused");
  Check(!gufo::speculative::ComputeMarkovBiasRow(*weights, kVocab + 999U)
             .has_value(),
        "previous past the vocabulary is refused");

  // A DFlash-2 config carries no Markov head: no row, no crash.
  gufo::speculative::QwenDFlashWeights plain;
  plain.dflash_config.sliding_window = 64U;
  plain.dflash_config.num_layers = kLayers;
  Check(!gufo::speculative::ComputeMarkovBiasRow(plain, 0).has_value(),
        "a non-DSpark config produces no Markov row");
  // Mismatched matrices: w2 one element short must not produce a row.
  auto truncated = *weights;
  truncated.markov_w2.num_elements -= 1;
  Check(!gufo::speculative::ComputeMarkovBiasRow(truncated, 0).has_value(),
        "a markov_w2 that does not match markov_w1 is refused");
}

constexpr const char* kDSparkPath =
    "models/"
    "RedHat-DSpark-Q4_K_M.gguf";

}  // namespace

int main() {
  const std::filesystem::path dspark_path(EnvOr("GUFO_DSPARK_GGUF", kDSparkPath));

  // Hermetic fixtures first: they need no model store at all.
  TestSyntheticDFlash2NoRegression();
  TestDraftHeadDiscriminator();
  TestGroupedConvLaunchRefusal();
  TestDraftScratchElements();
  TestSyntheticDSparkF16Conversion();
  TestEpsilonMismatchRefused();
  TestSlidingWindowPattern();
  TestDSparkChainRangeCap();
  TestMarkovBiasValues();

  if (!std::filesystem::exists(dspark_path)) {
    std::cout << "SKIP: real DSpark artifact not found: " << dspark_path << "\n";
    if (g_failures != 0) {
      std::cerr << g_failures << " FAILURES before the skip\n";
      return 1;
    }
    return kSkipExit;
  }
  TestRealDSpark(dspark_path);

  if (g_failures != 0) {
    std::cerr << "qwen_dflash_weights_test: " << g_failures << " FAILURES\n";
    return 1;
  }
  std::cout << "qwen_dflash_weights_test: ALL TESTS PASSED\n";
  return 0;
}
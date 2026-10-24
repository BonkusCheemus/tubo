#ifndef GUFO_MODELS_QWEN_IQ3_PAIR_VIEW_HPP_
#define GUFO_MODELS_QWEN_IQ3_PAIR_VIEW_HPP_

#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include "src/models/qwen/state.hpp"

namespace gufo::models {

struct QwenIq3PairViews {
  QwenTensorRef pair;
  QwenTensorRef gate;
  QwenTensorRef up;
};

inline QwenIq3PairViews MakeIq3PairViews(const void* root, core::GgmlType type,
                                         std::size_t full_rows,
                                         std::size_t columns,
                                         std::size_t available_bytes) {
  const auto layout =
      type == core::GgmlType::kIQ3_S     ? QwenWeightLayout::kIq3SRows8
      : type == core::GgmlType::kIQ3_XXS ? QwenWeightLayout::kIq3XxsRows4
                                         : QwenWeightLayout::kNative;
  if (!Iq3ShapeValid(layout, full_rows, columns) || full_rows % 2 ||
      full_rows / 2 % Iq3GroupRows(layout))
    throw std::invalid_argument("invalid IQ3 paired shape/type");
  QwenTensorRef pair{.data = root,
                     .type = type,
                     .num_elements = full_rows * columns,
                     .available_bytes = available_bytes,
                     .layout = layout,
                     .layout_columns = columns,
                     .layout_rows = full_rows,
                     .layout_row_begin = 0};
  if (!pair.FitsAvailableStorage())
    throw std::invalid_argument("invalid IQ3 pair root/alignment/storage");
  auto gate = pair;
  gate.num_elements /= 2;
  auto up = gate;
  up.layout_row_begin = full_rows / 2;
  return {pair, gate, up};
}

// Serving geometry is deliberately narrower than the CPU byte-layout oracle.
// No device pointer is dereferenced by either validation helper.
inline bool IsSnIq3VerifyView(const QwenTensorRef& view) noexcept {
  return IsIq3Layout(view.layout) && view.FitsAvailableStorage() &&
         view.layout_columns == 5120 && view.layout_rows == 34816 &&
         (view.num_elements == 17408ULL * 5120 ||
          view.num_elements == 34816ULL * 5120) &&
         (view.layout_row_begin == 0 || view.layout_row_begin == 17408);
}

inline bool IsSnIq3VerifyPair(const QwenTensorRef& gate,
                              const QwenTensorRef& up) noexcept {
  return IsSnIq3VerifyView(gate) && IsSnIq3VerifyView(up) &&
         gate.num_elements == 17408ULL * 5120 &&
         gate.num_elements == up.num_elements && gate.data == up.data &&
         gate.type == up.type && gate.layout == up.layout &&
         gate.layout_row_begin == 0 && up.layout_row_begin == 17408;
}

// Executor fence profile: nullptr when `weights` may reach the GPU executor, else
// why not. Native models always pass. A tagged model passes ONLY with the
// default-OFF dev switch (GUFO_DEV_SN_IQ3_SERVE=1), only when every tag is an
// IQ3 gate/up pair of the one shape all tagged readers cover (prefill unpack,
// SN B1..8; C8 CPU readers refuse tags at their own load), and only when the
// run profile has no unimplemented tagged reader (NPU B-cache, C7, unless
// GUFO_NPU_BCACHE_TAGGED=1 selects its logical-tensor key). Rows32 (P1) or any
// other tag keeps the fence.
[[nodiscard]] inline const char* TaggedGpuServeRefusal(
    const QwenModelWeights& weights, bool serve_switch, bool npu_bcache) {
  if (!weights.HasNonNativeWeights())
    return nullptr;
  if (!serve_switch)
    return "Rows32 Qwen GPU readers are not integrated yet";
  if (npu_bcache)
    return "tagged IQ3 serving refuses GUFO_NPU_BCACHE without GUFO_NPU_BCACHE_TAGGED=1";
  auto rest = weights;  // ponytail: one load-time copy of ~64 layer refs
  for (auto& layer : rest.layers) {
    if (layer.ffn_gate.layout == QwenWeightLayout::kNative &&
        layer.ffn_up.layout == QwenWeightLayout::kNative)
      continue;
    if (!IsSnIq3VerifyPair(layer.ffn_gate, layer.ffn_up))
      return "tagged FFN gate/up is not a served IQ3 pair";
    layer.ffn_gate.layout = layer.ffn_up.layout = QwenWeightLayout::kNative;
  }
  return rest.HasNonNativeWeights()
             ? "only IQ3 gate/up tags have GPU readers; another tag remains"
             : nullptr;
}

// C7: GUFO_NPU_BCACHE is set and its tagged (logical-tensor) key was not
// selected with exactly GUFO_NPU_BCACHE_TAGGED=1. Executor fence and prefill.
[[nodiscard]] inline bool TaggedNpuBcacheRefused() {
  const char* const bcache = std::getenv("GUFO_NPU_BCACHE");
  const char* const tagged = std::getenv("GUFO_NPU_BCACHE_TAGGED");
  return bcache != nullptr && bcache[0] != '\0' &&
         !(tagged != nullptr && std::string_view(tagged) == "1");
}

[[nodiscard]] inline const char* TaggedGpuServeRefusal(
    const QwenModelWeights& weights) {
  const char* const serve = std::getenv("GUFO_DEV_SN_IQ3_SERVE");
  return TaggedGpuServeRefusal(
      weights, serve != nullptr && std::string_view(serve) == "1",
      TaggedNpuBcacheRefused());
}

// Numeric address oracle only: never dereferences a possibly device-only root.
inline std::size_t Iq3ViewByteOffset(const QwenTensorRef& view, std::size_t row,
                                     std::size_t block, std::size_t byte) {
  if (!IsIq3Layout(view.layout) || !view.FitsAvailableStorage() ||
      row >= view.num_elements / view.layout_columns ||
      block >= view.layout_columns / 256 || byte >= Iq3BlockBytes(view.layout))
    throw std::out_of_range("invalid IQ3 logical byte address");
  return Iq3ByteOffset(view.layout, view.layout_rows, view.layout_columns / 256,
                       view.layout_row_begin + row, block, byte);
}

// Shared by the CPU oracle and the GPU unpack launcher. Pointer arithmetic only.
inline void ValidateIq3Unpack(const QwenTensorRef& view, const void* native,
                              std::size_t capacity) {
  if (!IsIq3Layout(view.layout) || !view.FitsAvailableStorage() || !native ||
      view.EncodedSizeBytes() > capacity)
    throw std::invalid_argument("invalid IQ3 unpack view/output");
  const auto span =
      Iq3PackedBytes(view.layout, view.layout_rows, view.layout_columns);
  const auto s = reinterpret_cast<std::uintptr_t>(view.data);
  const auto d = reinterpret_cast<std::uintptr_t>(native);
  const auto output_bytes = view.EncodedSizeBytes();
  if ((s <= d && d - s < span) || (d < s && s - d < output_bytes))
    throw std::invalid_argument("overlapping IQ3 unpack view");
}

// Prefill readers: the tagged G/U pair is unpacked whole
// (view P) into `scratch`; native gate/up then point at its two halves, so the
// unchanged native kernels read the original GGUF bytes. Returns {P, native
// gate, native up}. Host arithmetic only; never dereferences either buffer.
inline QwenIq3PairViews Iq3NativeScratchViews(const QwenTensorRef& gate,
                                              const QwenTensorRef& up,
                                              void* scratch,
                                              std::size_t capacity) {
  if (!IsIq3Layout(gate.layout) || !gate.FitsAvailableStorage() ||
      !up.FitsAvailableStorage() || gate.data != up.data ||
      gate.type != up.type || gate.layout != up.layout ||
      gate.layout_rows != up.layout_rows ||
      gate.layout_columns != up.layout_columns ||
      gate.num_elements != up.num_elements ||
      2 * gate.num_elements != gate.layout_rows * gate.layout_columns ||
      gate.layout_row_begin != 0 || up.layout_row_begin != gate.layout_rows / 2)
    throw std::invalid_argument("tagged prefill gate/up is not one IQ3 pair");
  auto pair = gate;
  pair.num_elements *= 2;
  const auto half = gate.EncodedSizeBytes();
  if (!scratch || pair.EncodedSizeBytes() > capacity)
    throw std::length_error("IQ3 prefill unpack scratch too small");
  const QwenTensorRef native{.data = scratch,
                             .type = gate.type,
                             .num_elements = gate.num_elements,
                             .available_bytes = capacity};
  auto native_up = native;
  native_up.data = static_cast<const std::uint8_t*>(scratch) + half;
  native_up.available_bytes = capacity - half;
  return {pair, native, native_up};
}

// CPU oracle: both buffers must be host-addressable (never a GTT device root).
// Native output contains ONLY this logical view, starting at output row0.
inline void UnpackIq3View(const QwenTensorRef& view, void* native,
                          std::size_t capacity) {
  ValidateIq3Unpack(view, native, capacity);
  const auto* in = static_cast<const std::uint8_t*>(view.data);
  auto* out = static_cast<std::uint8_t*>(native);
  const auto bb = Iq3BlockBytes(view.layout);
  const auto nb = view.layout_columns / 256;
  for (std::size_t r = 0; r < view.num_elements / view.layout_columns; ++r)
    for (std::size_t b = 0; b < nb; ++b)
      for (std::size_t o = 0; o < bb; ++o)
        out[(r * nb + b) * bb + o] =
            in[Iq3ByteOffset(view.layout, view.layout_rows, nb,
                             view.layout_row_begin + r, b, o)];
}

}  // namespace gufo::models
#endif

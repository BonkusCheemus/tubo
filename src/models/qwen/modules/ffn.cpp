#include "src/models/qwen/modules/ffn.hpp"

#include <cmath>
#include <stdexcept>

#include "src/models/qwen/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#endif

namespace gufo::models::qwen {

void FfnForward(const CpuModuleContext&, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) noexcept {
  if (!view.gate.empty()) {
    TensorGEMV(view.gate, x, view.intermediate_size, view.hidden_size,
               gate_scratch);
  }
  if (!view.up.empty()) {
    TensorGEMV(view.up, x, view.intermediate_size, view.hidden_size,
               up_scratch);
  }

  for (std::size_t i = 0; i < view.intermediate_size; ++i) {
    const float g = gate_scratch[i];
    const float silu_g = g / (1.0F + std::exp(-g));
    act_scratch[i] = silu_g * up_scratch[i];
  }

  if (!view.down.empty()) {
    TensorGEMV(view.down, act_scratch, view.hidden_size, view.intermediate_size,
               out);
  }
}

#if defined(ENGINE_ENABLE_HIP)
void FfnForward(const HipModuleContext& ctx, const FfnLayerView& view,
                std::span<const float> x, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> out) {
  const auto stream = static_cast<hipStream_t>(ctx.Stream());
  if (view.gate.layout != QwenWeightLayout::kNative ||
      view.up.layout != QwenWeightLayout::kNative) {
    // Tagged IQ3 gate/up take the SN B1 route (one ActK quantize,
    // two logical views; LaunchSnIq3Pair refuses any other tag or shape) and
    // the pointwise SwiGLU. Not bit-exact vs the FP32 fused kernel (ed6).
    const std::size_t n = view.intermediate_size;
    if (gate_scratch.size() < n || up_scratch.size() < n ||
        act_scratch.size() < n || gate_scratch.data() == up_scratch.data())
      throw std::invalid_argument(
          "tagged IQ3 FFN needs distinct gate/up scratch");
    ::gufo::hip::LaunchSnIq3Pair(view.gate, view.up, x.data(),
                                 gate_scratch.data(), up_scratch.data(),
                                 nullptr, 1, stream);
    ::gufo::hip::LaunchBatchedSwiGLUActivation(
        gate_scratch.data(), up_scratch.data(), act_scratch.data(), nullptr, n,
        stream);
  } else {
    ::gufo::hip::LaunchFusedSwiGLUGEMV(
        view.gate.data, view.gate.type, view.up.data, view.up.type, x.data(),
        act_scratch.data(), view.intermediate_size, view.hidden_size, stream);
  }
  ::gufo::hip::LaunchGEMV(view.down.data, view.down.type, act_scratch.data(),
                          out.data(), view.hidden_size, view.intermediate_size,
                          stream);
}
#endif

}  // namespace gufo::models::qwen

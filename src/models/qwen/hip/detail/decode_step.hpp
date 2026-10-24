#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

#include <cstdint>
#include <string_view>

#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::models {
struct QwenModelWeights;
}

namespace gufo::hip {

class QwenGpuArena;
struct QwenExecutionPolicy;

/// Emits pure route decisions before entering graph capture. `mode` names the
/// executor entry, not the layer policy mode: distinct entry modes are required
/// because decode, server-batch and verification entries may share a policy.
/// This must never be called from the capture callback.
void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy,
                              std::string_view mode = "decode");

/// Runs one graph-capture-safe decode layer stack using stable arena storage.
/// `recurrent_state` is the pass's QwenGpuArena::BeginRecurrentWrite result;
/// a captured graph bakes its pointers, so the graph key must carry
/// QwenGpuArena::RecurrentIoIdentity.
void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits,
                       QwenRecurrentStateIo recurrent_state,
                       models::qwen::vision::DeviceInput* image_input = nullptr,
                       const void* embedding_row = nullptr);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

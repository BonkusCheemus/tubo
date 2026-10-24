#pragma once
// DEV: split one f16 prefill GEMM between the GPU and the XDNA2 NPU.
// GUFO_NPU_SPLIT=<flm GEMM .xclbin> turns it on (built with GUFO_ENABLE_NPU).
// Split ops: ffn_down (k 17408 -> m 5120) and attn_output/ssm_out (k 6144 -> m 5120),
// residual epilogue; ffn gate/up (k 5120 -> m 17408 each, SwiGLU epilogue); store-epilogue
// GEMMs with k 5120 (attn_qkv, attn_gate, attn_q).
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/npu_bcache_key.hpp"

namespace gufo::hip::npu {
// Returns how many leading rows the GPU must compute. m = no split. Below m,
// rows [result, m) are queued on the NPU as job *seq, and FinishResidual(seq)
// must follow the GPU launch on the same stream. One stream, no graph capture.
std::size_t BeginSplit(core::GgmlType type, const void* w, const void* x,
                       std::size_t batch, std::size_t m, std::size_t k,
                       hipStream_t stream, std::uint32_t* seq);
// Waits for the NPU rows on the stream (no host sync) and adds them to y.
void FinishResidual(float* y, std::size_t batch, std::size_t m,
                    std::size_t m_gpu, hipStream_t stream, std::uint32_t seq);
// Same contract for store-epilogue GEMMs (k 5120, 4096 <= m < 17408; GUFO_NPU_STORE_PCT):
// FinishStore writes NPU rows [result, m) of y.
std::size_t BeginStore(core::GgmlType type, const void* w, const void* x,
                       std::size_t batch, std::size_t m, std::size_t k,
                       hipStream_t stream, std::uint32_t* seq);
void FinishStore(float* y, std::size_t batch, std::size_t m, std::size_t m_gpu,
                 hipStream_t stream, std::uint32_t seq);
// Same contract for the paired gate/up GEMM: the GPU computes gate and up
// rows [0, result); FinishGateUp writes SwiGLU(gate, up) rows [result, m) of y (f16).
std::size_t BeginGateUp(core::GgmlType gate_type, core::GgmlType up_type,
                        const void* gate_w, const void* up_w, const void* x,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream, std::uint32_t* seq);
void FinishGateUp(void* y, std::size_t batch, std::size_t m, std::size_t m_gpu,
                  hipStream_t stream, std::uint32_t seq);
// Tagged IQ3 prefill: until ClearScratchAliases, a weight pointer equal to
// gate_scratch / up_scratch is that logical tensor for the B-cache key and fingerprint
// (npu_bcache_key.hpp). Prefill thread only, like every other call here.
void SetScratchAliases(const void* gate_scratch, const LogicalWeight& gate,
                       const void* up_scratch, const LogicalWeight& up);
void ClearScratchAliases();
}  // namespace gufo::hip::npu

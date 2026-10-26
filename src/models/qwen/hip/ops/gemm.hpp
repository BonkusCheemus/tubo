#ifndef GUFO_MODELS_QWEN_HIP_OPS_GEMM_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_GEMM_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::hip {

struct HipblasLtDispatchInfo {
  int algorithm_id{-1};
  std::string solution_name;
  std::string kernel_name;
  std::string plan_source{"heuristic"};
  std::string persistent_cache_status{"disabled"};
  std::size_t workspace_bytes{0};
  double plan_resolution_us{0.0};
  bool plan_cache_hit{false};
};

struct HipblasLtGemmOptions {
  std::string plan_database_path;
  std::size_t tuning_workspace_bytes{0};
  bool ignore_environment{false};
};

struct HipblasLtTuningOptions {
  std::uint32_t warmup{2};
  std::uint32_t repetitions{5};
  std::size_t max_algorithms{64};
};

struct HipblasLtTuningResult {
  int algorithm_id{-1};
  int heuristic_algorithm_id{-1};
  std::string solution_name;
  std::string kernel_name;
  std::size_t workspace_bytes{0};
  double median_us{0.0};
  double heuristic_median_us{0.0};
  double verified_speedup{1.0};
  std::size_t supported_algorithms{0};
  std::size_t measured_algorithms{0};
  bool retained_tuned_algorithm{false};
};

/// Cached hipBLASLt BF16 GEMM plans for prompt-processing projections.
class HipblasLtGemm {
public:
  explicit HipblasLtGemm(HipblasLtGemmOptions options = {});
  ~HipblasLtGemm();

  HipblasLtGemm(const HipblasLtGemm&) = delete;
  HipblasLtGemm& operator=(const HipblasLtGemm&) = delete;
  HipblasLtGemm(HipblasLtGemm&&) noexcept;
  HipblasLtGemm& operator=(HipblasLtGemm&&) noexcept;

  /// Computes Y[B, M] = X_bf16[B, K] * A_bf16[M, K]^T.
  /// Returns false when hipBLASLt cannot provide a supported plan.
  [[nodiscard]] bool RunBf16(const void* a_bf16, const void* x_bf16, float* y,
                             std::size_t batch_size, std::size_t m,
                             std::size_t k, hipStream_t stream = nullptr,
                             HipblasLtDispatchInfo* dispatch_info = nullptr);

  /// Benchmarks supported algorithms, installs the fastest plan, and retains
  /// it for SavePlans(). Buffers must use the same layout as RunBf16().
  [[nodiscard]] bool TuneBf16(const void* a_bf16, const void* x_bf16, float* y,
                              std::size_t batch_size, std::size_t m,
                              std::size_t k,
                              const HipblasLtTuningOptions& options,
                              HipblasLtTuningResult* result,
                              hipStream_t stream = nullptr);

  /// Atomically writes all resolved plans using the current hardware/ROCm key.
  [[nodiscard]] bool SavePlans(const std::string& path,
                               std::string* error = nullptr) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Pre-reserve the per-stream Dot4 scratch (elems = batch * k int8 values) for
/// every later consumer on `stream` before graph capture. No GPU work. Any
/// growth during capture throws; it never grows, frees or falls back there.
void ReserveDot4Scratch(std::size_t elems, hipStream_t stream = nullptr);

/// ---- NormQ (GUFO_DEV_NORMQ=1) ----
/// A fused RMSNorm (prefill_norm.hip) writes the int8 activations and scales
/// its consumers would have quantized, in BOTH quantizer kinds, and publishes
/// a per-stream mark; a consumer whose (x, batch, k) matches reads them
/// instead of launching its own quantize kernel. Bitwise equal to
/// norm-then-QuantizeActivations{,Q8K}Dot4Kernel.
///
/// The buffers are NOT the Dot4 scratch: every other quantize on the stream
/// overwrites that one, so a match would have to prove nothing in between
/// touched it. These are written by the fused norm alone, which is why no
/// invalidation is needed anywhere but the norm launchers themselves (each
/// either publishes or clears).
struct NormQTarget {
  std::int8_t* q[2]{nullptr, nullptr};  //< [0] = plain, [1] = ActK
  float2* ds[2]{nullptr, nullptr};
};

[[nodiscard]] bool GufoDevNormQ() noexcept;

/// Buffers for a fused norm of `elems` (= batch * k) activations, or false
/// (nothing is allocated during a graph capture: the caller must then run the
/// unfused norm). Call ReserveNormQScratch before capture.
[[nodiscard]] bool NormQAcquire(hipStream_t stream, std::size_t elems,
                                NormQTarget* target);
void ReserveNormQScratch(std::size_t elems, hipStream_t stream = nullptr);

/// The fused norm wrote `x` ([batch][k]) on `stream`, both kinds.
void NormQPublish(hipStream_t stream, const float* x, std::size_t batch,
                  std::size_t k);
/// Any other write to `x` on `stream` drops the mark.
void NormQClear(hipStream_t stream, const float* x);
/// True and `q`/`ds` set when the mark covers exactly (x, batch, k) in the
/// `act_k` kind: the caller skips its quantize and reads these instead.
[[nodiscard]] bool NormQLookup(hipStream_t stream, const float* x,
                               std::size_t batch, std::size_t k, bool act_k,
                               std::int8_t** q, float2** ds);

/// Tagged IQ3 FFN projection only, B1..8 (B1: single-token decode, C1/C2).
/// Quantizes ONCE then consumes both
/// logical views, or the full pair if packed_output != nullptr. Outputs are
/// FP32 [B,17408] each, or packed [B,34816]. No native fallback or allocation
/// during capture. Caller orders all uses of this stream's scratch.
void LaunchSnIq3Pair(const models::QwenTensorRef& gate,
                     const models::QwenTensorRef& up, const float* input,
                     float* gate_output, float* up_output, float* packed_output,
                     std::size_t batch, hipStream_t stream = nullptr);

/// k95small (GUFO_DEV_SMALL_K95=1; the caller gates on the env). Each call
/// launches its whole group or nothing (returns false: run the old calls).
/// Bitwise equal to the per-projection LaunchBatchedQuantGEMMFp32 /
/// LaunchGEMVRows calls it replaces (kernels/k95_small.hpp).
/// Attention Q, K, V of one x at width 8: one int8 quantization, Q's V3 GEMM,
/// then K (Q5_K) + V (Q6_K) as one launch. Outputs [8, q_m] / [8, kv_m].
[[nodiscard]] bool TryLaunchK95AttentionQkv(
    core::GgmlType q_type, const void* wq, core::GgmlType k_type,
    const void* wk, core::GgmlType v_type, const void* wv, const float* x,
    float* yq, float* yk, float* yv, std::size_t batch, std::size_t q_m,
    std::size_t kv_m, std::size_t k, hipStream_t stream = nullptr);
/// Two V3 projections of one x (SSM qkv + gate), width 1-8: one quantization.
[[nodiscard]] bool TryLaunchK95SharedQuantPair(
    core::GgmlType a_type, const void* wa, float* ya, std::size_t a_m,
    core::GgmlType b_type, const void* wb, float* yb, std::size_t b_m,
    const float* x, std::size_t batch, std::size_t k,
    hipStream_t stream = nullptr);
/// SSM controls: F32 alpha + Q8_0 beta [48, 5120], width 1-8 except 6, in
/// one launch. Outputs [batch, m] each.
[[nodiscard]] bool TryLaunchK95SsmControls(
    core::GgmlType alpha_type, const void* alpha, core::GgmlType beta_type,
    const void* beta, const float* x, float* y_alpha, float* y_beta,
    std::size_t batch, std::size_t m, std::size_t k,
    hipStream_t stream = nullptr);

/// Tagged IQ3 view (P, G or U) -> native IQ3 rows at `native`, byte-equal to
/// the GGUF source (inverse of the SN copy plan). Device-to-device, async.
void LaunchUnpackIq3View(const models::QwenTensorRef& view, void* native,
                         std::size_t capacity, hipStream_t stream = nullptr);

/// Computes Matrix-Vector Multiplication: y = A * x
/// Supports both F32 and BF16 weights A
void LaunchGEMV(
    const void* A, core::GgmlType a_type, const float* x, float* y,
    std::size_t M, std::size_t K, hipStream_t stream = nullptr,
    models::qwen::QwenGemmMode mode = models::qwen::QwenGemmMode::kHipDecode);

/// `rows` independent LaunchGEMV calls on row-major x (stride K) and
/// y (stride M), kHipDecode mode. The F32/BF16 block routes run as ONE launch
/// (grid M x rows), bitwise equal to the per-row loop; other routes loop.
void LaunchGEMVRows(const void* A, core::GgmlType a_type, const float* x,
                    float* y, std::size_t rows, std::size_t M, std::size_t K,
                    hipStream_t stream = nullptr);

/// opt-c1xx-q8k-gemv: computes y = A * x where A is stored as block_q8_K
/// ({ float d; int8_t qs[256]; int16_t bsums[16]; }, QK_K=256). The dot runs
/// at Q8 (activation quantized to int8, integer MAC, single fp scale at block
/// end); weights are not dequantized to fp16. Requires K % 256 == 0.
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream = nullptr);

/// Dequantizes a block_q8_K weight region ({ float d; int8_t qs[256]; int16_t
/// bsums[16]; }, QK=256) into a BF16 scratch buffer for the prefill hipblas
/// GEMM path (opt-c162-q8k-prefill-dequant). hipBLAS cannot consume Q8_K
/// blocks, so prefill dequantizes to BF16 and runs the GF16 GEMM on the
/// scratch. out[b*256 + j] = (hip_bfloat16)(d * qs[j]) per block, matching the
/// CPU DequantizeQ8_K oracle. Requires n_elems % 256 == 0.
void LaunchDequantizeQ8KToBf16(const void* w, hip_bfloat16* out,
                               std::size_t n_elems,
                               hipStream_t stream = nullptr);

/// Type-dispatched dequantize-to-BF16 for the prefill hipblas GEMM path
/// (opt-c162-q8k-prefill-dequant). Converts a quantized weight region
/// (block_q8_0, block_q5_K, block_q6_K, block_q8_K) into a BF16 scratch
/// buffer. out[b*qk + j] = (hip_bfloat16)(per_value_dequant(block[b], j)) per
/// block, matching the CPU DequantizeQ8_0/Q5_K/Q6_K/Q8_K oracles. Q8_0 uses
/// QK=32, Q5_K/Q6_K/Q8_K use QK=256; requires n_elems to be a whole number of
/// blocks. Unsupported types are a no-op.
namespace detail {

/// Launches the exact wave64 kernel for measured Qwen27B verification shapes.
/// Returns false without launching for other widths, shapes and formats.
[[nodiscard]] bool TryLaunchKQuantSmallBatchWave64(
    core::GgmlType type, const void* w, const float* x, float* y,
    std::size_t batch, std::size_t m, std::size_t k, hipStream_t stream,
    std::size_t groups = 1);

[[nodiscard]] bool TryLaunchQ8SmallBatchWave64(const void* w, const float* x,
                                               float* y, std::size_t batch,
                                               std::size_t m, std::size_t k,
                                               hipStream_t stream,
                                               std::size_t groups = 1);

/// Exact BF16 projection for selected shared widths; the caller selects shapes.
[[nodiscard]] bool TryLaunchBf16SmallBatchWave64(const void* w, const float* x,
                                                 float* y, std::size_t batch,
                                                 std::size_t m, std::size_t k,
                                                 hipStream_t stream,
                                                 std::size_t groups = 1);

/// opt-q4kxl: true for the formats that run natively through the K-quant GPU
/// kernels -- the blocked WMMA GEMM at prefill batch, the exact shared-weight
/// kernel at draft width, and the routing and activation-fusion gates that feed
/// them. That is everything with a DecodeQuantSub16 implementation except Q8_0
/// (which keeps its own untouched kernels) and Q8_K (whose row layout carries
/// block sums the WMMA staging does not use).
///
/// The name says Wmma for the kernel it was introduced for; the predicate is
/// now the general "decoded in-kernel from packed form" test.
[[nodiscard]] constexpr bool IsNativeWmmaQuant(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kIQ4_NL:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kQ2_K:
    case core::GgmlType::kIQ2_XXS:
    case core::GgmlType::kIQ2_XS:
    case core::GgmlType::kIQ2_S:
    case core::GgmlType::kIQ3_XXS:
    case core::GgmlType::kIQ1_M:
      return true;
    default:
      return false;
  }
}

}  // namespace detail

void LaunchDequantizeToBf16(core::GgmlType type, const void* w,
                            hip_bfloat16* out, std::size_t n_elems,
                            hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_bf16[B, K] * W_quant[M, K]^T on quantized
/// weights (block_q8_0 / block_q8_K / block_q5_K / block_q6_K) with NO
/// dequantize-to-BF16 and NO BF16 GEMM (opt-c162-prefill-quant-direct). Each
/// output row is computed by one warp reading the quant blocks directly.
void LaunchBatchedQuantGEMM(core::GgmlType type, const void* w,
                            const void* bf16_x, float* y, std::size_t batch,
                            std::size_t m, std::size_t k,
                            hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_bf16[B, K] * W_quant[M, K]^T without
/// quantizing the activation. The Q8_0 small-batch route shares each weight
/// load across up to eight verification rows.
void LaunchBatchedQuantGEMMBf16(core::GgmlType type, const void* w,
                                const void* bf16_x, float* y, std::size_t batch,
                                std::size_t m, std::size_t k,
                                hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_fp32[B, K] * W_quant[M, K]^T using the same
/// quant-block dot product and FP32 activation precision as decode. Intended
/// for short verification batches where exact target-path numerics matter.
/// Groups of up to sixteen rows share weight loads while preserving the
/// isolated production GEMV arithmetic order.
void LaunchBatchedQuantGEMMFp32(core::GgmlType type, const void* w,
                                const float* fp32_x, float* y,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream = nullptr);

/// Quantizes BF16 activation tensor X[B, K] to block_q8_1 activation blocks
void LaunchQuantizeActivationQ8_1(const void* bf16_x, void* q8_1_out,
                                  std::size_t batch, std::size_t k,
                                  hipStream_t stream = nullptr);

/// Quantizes FP32 activation tensor X[B, K] to block_q8_1 activation blocks
/// True when the fused SSM post-norm/gate + Q8_1 quantize kernel can take this
/// head shape.
[[nodiscard]] bool IsFusedSSMEpilogueQuantizeQ8_1Supported(
    std::uint32_t val_dim, std::size_t inner_size) noexcept;

/// SSM per-head post-RMSNorm + SiLU gate writing the tiled Q8_1 activation
/// directly (opt-c174-ssm-epilogue-quant). Use when `ssm_out` is Q8_0, so the
/// FP32 gated row has no other consumer. Bit-identical to
/// BatchedSSMPostNormGateKernel followed by an FP32 quantize.
void LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1(
    const float* raw_out, const float* ssm_norm, const float* gate,
    void* q8_1_out, std::size_t batch_size, std::size_t inner_size,
    std::uint32_t num_heads, std::uint32_t val_dim,
    hipStream_t stream = nullptr);

/// True when the fused RMSNorm + Q8_1 quantize kernel can take this row length.
[[nodiscard]] bool IsFusedRMSNormQuantizeQ8_1Supported(
    std::size_t dim) noexcept;

/// Optional residual add + RMSNorm + tiled Q8_1 quantize in one pass
/// (opt-c173-norm-quant). Use when every projection reading the norm is Q8_0,
/// so neither the FP32 normed row nor the BF16 staging copy is needed. With
/// `residual` non-null the sum `x + residual` is normalized and also written to
/// `sum_out` for the next residual link. The normed values are bit-identical to
/// LaunchBatchedResidualAdd followed by LaunchBatchedRMSNorm.
void LaunchBatchedFusedRMSNormQuantizeQ8_1(
    const float* x, const float* residual, const float* weight, float* sum_out,
    void* q8_1_out, std::size_t batch_size, std::size_t dim, float eps,
    hipStream_t stream = nullptr);

void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream = nullptr);

/// Byte size of the tiled Q8_1 activation buffer a [batch, k] activation needs
/// before LaunchBatchedQuantGEMMPreQuantized can consume it.
[[nodiscard]] std::size_t QuantizedActivationBytes(std::size_t batch,
                                                   std::size_t k);

/// Fuses SwiGLU activation (SiLU(gate) * up) with Q8_1 quantization into one
/// kernel
void LaunchBatchedFusedSwiGLUQuantizeQ8_1(const float* gate, const float* up,
                                          void* q8_1_out, std::size_t batch,
                                          std::size_t k,
                                          hipStream_t stream = nullptr);

/// Directly computes Y[B, M] = X_q8_1[B, K] * W_q8_0[M, K]^T using LDS-staged
/// Matrix Core (WMMA) operations with pre-quantized activation blocks.
void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k,
                                        hipStream_t stream = nullptr);

/// Directly computes dual GEMM:
///   Y_gate[B, M] = X_q8_1[B, K] * W_gate[M, K]^T
///   Y_up[B, M]   = X_q8_1[B, K] * W_up[M, K]^T
/// in a single fused kernel pass sharing activation memory reads and registers.
void LaunchBatchedDualQuantGEMMPreQuantized(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, float* y_up, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream = nullptr);

/// True when LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1 can replace the
/// gate/up GEMM pair plus the separate SwiGLU-quantize pass
/// (opt-c192-swiglu-epilogue). `q8_out_bytes` is the capacity of the buffer the
/// fused epilogue will write the [batch, m] Q8_1 activation into; a caller that
/// reuses a differently shaped scratch has to pass its real size.
[[nodiscard]] bool IsFusedSwiGluGemmEpilogueSupported(
    core::GgmlType type, std::size_t batch, std::size_t m,
    std::size_t q8_out_bytes) noexcept;

/// Computes Y_gate[B, M] = X_q8_1[B, K] * W_gate[M, K]^T and then, in the up
/// projection's own epilogue, the tiled Q8_1 encoding of
/// SiLU(Y_gate) * (X_q8_1 * W_up^T) -- so the FP32 [B, M] up intermediate is
/// never stored and never read back. Bit-identical to the pair of GEMMs
/// followed by LaunchBatchedFusedSwiGLUQuantizeQ8_1. `q8_1_out` must not alias
/// `q8_1_x`.
void LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, void* q8_1_out, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream = nullptr);

/// Batched GEMM: Y[B, M] = X[B, K] * A[M, K]^T
void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
                       hipStream_t stream = nullptr);

/// Exact BF16-weight GEMM with FP32 activations, grouped into at most sixteen
/// rows. The per-output accumulation order matches the decode GEMV and does
/// not depend on the batch width.
void LaunchExactBf16GEMMFp32SmallBatch(const void* A, const float* X, float* Y,
                                       std::size_t batch_size, std::size_t M,
                                       std::size_t K,
                                       hipStream_t stream = nullptr);

/// Converts float buffer to bfloat16 buffer on GPU
void LaunchFloatToBfloat16(const float* in, void* out, std::size_t num_elements,
                           hipStream_t stream = nullptr);

/// Hardware-accelerated Batched GEMM: Y[B, M] = X[B, K] * A[M, K]^T using
/// hipBLAS
void LaunchHipblasGEMM(hipblasHandle_t handle, const void* A, bool is_bf16,
                       const float* X, float* Y, std::size_t batch_size,
                       std::size_t M, std::size_t K, void* d_x_bf16_buf,
                       hipStream_t stream = nullptr);

/// Direct BF16 GEMM without input conversion: Y[B, M] = X_bf16[B, K] *
/// A_bf16[M, K]^T
void LaunchHipblasGEMMBF16(hipblasHandle_t handle, const void* A_bf16,
                           const void* d_x_bf16, float* Y,
                           std::size_t batch_size, std::size_t M, std::size_t K,
                           hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_GEMM_HPP_

#define ENGINE_ENABLE_HIP 1
#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3.hip"
namespace gufo::hip {
#define P(B,M) template __global__ void OpusIq3sB8FloorProbeKernel<B,4,4,OpusFloorProbe::M>(const std::uint8_t*, float*, std::uint32_t, std::uint32_t);
#define A(B) P(B,kDot4) P(B,kRateDot4S) P(B,kRateDot4V) P(B,kRateDot4U8) P(B,kRateDot8) P(B,kRateDot2) P(B,kRateDot2Acc) P(B,kRateFma) P(B,kRateFmac) P(B,kRateWmma8) P(B,kRateWmma4)
A(1) A(8)
}

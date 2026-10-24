// The V3 dot4 kernels for IQ3_S / IQ3_XXS / IQ4_XS,
// compiled with -mwavefrontsize64. Bench-only. ceil_B (10-18) found these
// cells issue ~0.5 VALU/cycle/SIMD; a wave64 VALU op occupies the SIMD two
// passes, so a dependent op waits less and scalar/branch work is amortised
// over 64 lanes. The kernel is wave-size safe: lanes are threadIdx.x & 31,
// cross-lane traffic is XorLane (DPP row_xmask / permlanex16, both within a
// 32-lane half) and every barrier is block-wide. Same kernel source, so the
// bench requires it to match its wave32 twin bitwise.
#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/small_batch_dot4.hpp"

namespace gufo::hip::detail {
namespace {

template<core::GgmlType Type, std::size_t Batch>
bool Go(std::uint32_t rows, std::uint32_t waves, std::uint32_t per_lane,
        bool act_k, const void* w, const float* x, float* y, std::int8_t* xq,
        float2* xds, std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  bool hit = false;
  const auto g = [&]<std::uint32_t R, std::uint32_t W, std::uint32_t P = 1,
                     bool K = false>() {
    if (hit || R != rows || W != waves || P != per_lane || K != act_k) return;
    LaunchSmallBatchDot4V3<Type, Batch, R, W, P, K, 64>(w, x, y, xq, xds, m,
                                                        k, stream);
    hit = true;
  };
  // Same geometry list as the bench's wave32 V3 arm.
  g.template operator()<4, 4>();
  g.template operator()<4, 8>();
  g.template operator()<8, 4>();
  g.template operator()<2, 4>();
  g.template operator()<4, 2>();
  constexpr bool kIq4 = Type == core::GgmlType::kIQ4_XS;
  if constexpr (kIq4 || Batch >= 4) {
    g.template operator()<4, 4, 2>();
    g.template operator()<4, 8, 2>();
    g.template operator()<8, 4, 2>();
    g.template operator()<2, 4, 2>();
  }
  if constexpr (Batch >= 6) {
    g.template operator()<4, 4, 4>();
    g.template operator()<4, 8, 4>();
    g.template operator()<8, 4, 4>();
    g.template operator()<2, 4, 4>();
  }
  if constexpr (kIq4) {
    g.template operator()<4, 4, 1, true>();
    g.template operator()<4, 8, 1, true>();
    g.template operator()<8, 4, 1, true>();
    g.template operator()<2, 4, 1, true>();
    g.template operator()<4, 4, 2, true>();
    g.template operator()<4, 8, 2, true>();
    g.template operator()<8, 4, 2, true>();
    g.template operator()<2, 4, 2, true>();
    if constexpr (Batch >= 6) {
      g.template operator()<4, 4, 4, true>();
      g.template operator()<4, 8, 4, true>();
      g.template operator()<8, 4, 4, true>();
      g.template operator()<2, 4, 4, true>();
    }
  }
  return hit;
}

template<core::GgmlType Type>
bool ByBatch(std::size_t batch, std::uint32_t rows, std::uint32_t waves,
             std::uint32_t per_lane, bool act_k, const void* w, const float* x,
             float* y, std::int8_t* xq, float2* xds, std::uint32_t m,
             std::uint32_t k, hipStream_t stream) {
  switch (batch) {
#define GUFO_W64_B(B)                                                        \
  case B:                                                                    \
    return Go<Type, B>(rows, waves, per_lane, act_k, w, x, y, xq, xds, m, k, \
                       stream);
    GUFO_W64_B(1) GUFO_W64_B(2) GUFO_W64_B(3) GUFO_W64_B(4)
    GUFO_W64_B(5) GUFO_W64_B(6) GUFO_W64_B(7) GUFO_W64_B(8)
#undef GUFO_W64_B
  }
  return false;
}

}  // namespace

// false = no wave64 instantiation for this (type, batch, geometry).
bool TryLaunchDot4V3Wave64(core::GgmlType type, std::size_t batch,
                           std::uint32_t rows, std::uint32_t waves,
                           std::uint32_t per_lane, bool act_k, const void* w,
                           const float* x, float* y, std::int8_t* xq,
                           float2* xds, std::uint32_t m, std::uint32_t k,
                           hipStream_t stream) {
  switch (type) {
    case core::GgmlType::kIQ3_S:
      return ByBatch<core::GgmlType::kIQ3_S>(batch, rows, waves, per_lane,
                                             act_k, w, x, y, xq, xds, m, k,
                                             stream);
    case core::GgmlType::kIQ3_XXS:
      return ByBatch<core::GgmlType::kIQ3_XXS>(batch, rows, waves, per_lane,
                                               act_k, w, x, y, xq, xds, m, k,
                                               stream);
    case core::GgmlType::kIQ4_XS:
      return ByBatch<core::GgmlType::kIQ4_XS>(batch, rows, waves, per_lane,
                                              act_k, w, x, y, xq, xds, m, k,
                                              stream);
    default:
      return false;
  }
}

}  // namespace gufo::hip::detail
#endif  // ENGINE_ENABLE_HIP

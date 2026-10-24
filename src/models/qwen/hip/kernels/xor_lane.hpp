#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_XOR_LANE_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_XOR_LANE_HPP_

// From upstream gufo PR #518: lane i receives lane
// (i ^ offset)'s value, as __shfl_xor(v, offset) does. On gfx1151 it is a DPP
// row_xmask (offsets below 16) or v_permlanex16 (16): a register move instead
// of an LDS crossbar round trip (ds_bpermute / ds_swizzle). Only the transport
// differs, so reductions stay bitwise. Every lane of the 16/32-lane group must
// be active, as the shuffle's callers already require.

#include <hip/hip_runtime.h>

namespace gufo::hip {

__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_XOR_LANE_HPP_

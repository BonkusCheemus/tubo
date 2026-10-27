// Port of TheTom/llama-cpp-turboquant (MIT, Copyright (c) 2023-2026 The ggml authors; turbo5/turbo6 from PR #381
// by JakeATX), modified for HIP/gfx1151 in tubo. License text: licenses/llama-cpp-turboquant.txt.
// No FP contraction in this TU. Proved by
// tests/models/qwen/turbo4_dequant_f16_gpu_test.cpp, which requires the device
// scratch to be BIT-EQUAL to the host body: with contraction on, ~0.17% of
// elements differed, spread over many positions, sampled cases 1 ULP apart.
// After inlining, DequantElement's c*norm times sign2 feeds FWHT stage 1 (a+b)
// and HIP's default -ffp-contract=fast-honor-pragmas fuses that into an fma.
//
// SCOPE: THIS FILE ONLY, and it must precede the includes so the inlined codec
// bodies compile under it. Deliberately NOT on turbo4_kernels.hpp (that would
// change every consumer including the C2 decode numbers) and NOT on the decode
// read kernel: prefill dequant writes an f16 scratch while decode reads into
// registers, so those paths never needed to agree bit for bit.
//
// NOT a CMake option: set_source_files_properties is directory-scoped and
// gufo_core is created outside src/models/qwen, so it never reached this file.
#pragma clang fp contract(off)

// The Turbo4 -> f16 dequant KERNEL, one thread per
// (position, head).
//
//   /opt/rocm/bin/hipcc -fsyntax-only -std=c++20 -DENGINE_ENABLE_HIP -I. -x hip
//   \
//       src/models/qwen/hip/kernels/turbo4_kv_dequant.hip
//
// SPLIT THE SAME WAY E1 WAS SPLIT. The host body `Turbo4DequantLayerBody` is a
// serial loop over count x heads x blocks — about 131k blocks per layer per
// chunk at 8k, all on one thread. So the work itself lives in
// `Turbo4DequantHeadUnitBody`, which is ONE (position, head). This kernel is a
// grid over (count, num_kv_heads); the host body loops the same unit. One
// definition of what a block decodes to, reached from both sides.
//
// WHAT IS STILL NOT PROVEN HERE, PLAINLY: the grid choice is compile-only.
// Compilation catches a type error, not a wrong grid. The unit body, the index
// arithmetic and the decode have a CPU oracle; the launch does not, and the
// GPU test exists because of that, not instead of it.
//
// PER-LAYER BASES. `k_plane`/`v_plane` are this LAYER's block bases.
// Turbo4BlockByteOffset takes no layer argument, so the layer base is the one
// address nothing inside this file can check.

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace gufo::hip {
namespace {

/// grid (ceil(count / kDequantThreads), num_kv_heads); one THREAD per
/// (position, head). Was one thread per BLOCK: 1 of 32 lanes busy, and a 32k
/// prefill re-dequantizes the whole prefix every 4096-token chunk.
/// Each thread still runs the unchanged serial unit, so output is
/// bit-identical.
constexpr std::uint32_t kDequantThreads = 64;

__global__ void Turbo4KvDequantKernel(
    const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, std::uint16_t* __restrict__ k_dst,
    std::uint16_t* __restrict__ v_dst, std::uint32_t count,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, Turbo4KernelStatus* __restrict__ status,
    turbo4::PlaneFormats formats) {
  const std::uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t h = blockIdx.y;
  if (t >= count)
    return;
  const std::size_t kv_width = static_cast<std::size_t>(num_kv_heads) *
                               static_cast<std::size_t>(head_dim);
  const std::size_t dst_offset = static_cast<std::size_t>(t) * kv_width +
                                 static_cast<std::size_t>(h) * head_dim;
  // LATCH, never assign: count * num_kv_heads threads share this one word.
  Turbo4LatchFailure(
      status, Turbo4DequantHeadUnitBody(k_plane, v_plane, k_dst, v_dst,
                                        dst_offset, t, h, max_context,
                                        num_kv_heads, head_dim, formats));
}

// Four independent wave32s per CTA. Each wave owns one (position, head)
// and visits its 128-value K/V blocks; each lane owns four adjacent values.
constexpr std::uint32_t kDequantV2Waves = 4;

__device__ __forceinline__ void Turbo4DequantWaveBlock(
    const std::uint8_t* src, std::uint16_t* dst, turbo4::Bits bits,
    const float* centroids, const float* signs1, const float* signs2) {
  const unsigned lane = threadIdx.x & 31U;
  unsigned norm_bits = 0;
  if (lane == 0U) {
    std::uint16_t half;
    std::memcpy(&half, src, sizeof(half));
    norm_bits = half;
  }
  const float norm = turbo4::HalfBitsToFloat(
      static_cast<std::uint16_t>(__shfl(norm_bits, 0, 32)));

  // Coalesced, exact-width reads: 64 nibble bytes plus the 2-byte norm,
  // then 16/32 qh bytes for Turbo5/6. No read past a compact final block.
  // memcpy also accepts the byte-aligned bases accepted by the old kernel.
  std::uint16_t qs;
  std::memcpy(&qs, src + 2U + lane * 2U, sizeof(qs));
  unsigned qh = 0;
  if (bits == turbo4::Bits::k5) {
    if ((lane & 1U) == 0U)
      qh = src[turbo4::kQhOffset + lane / 2U];
    qh = __shfl(qh, lane & ~1U, 32) >> ((lane & 1U) * 4U);
  } else if (bits == turbo4::Bits::k6) {
    qh = src[turbo4::kQhOffset + lane];
  }

  float x[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    unsigned code = (qs >> (4U * j)) & 15U;
    if (bits == turbo4::Bits::k5)
      code = ((qh >> j) & 1U) != 0U ? 15U - code : 16U + code;
    else if (bits == turbo4::Bits::k6)
      code |= ((qh >> (2U * j)) & 3U) << 4U;
    x[j] = centroids[code] * norm;
    x[j] *= signs2[4U * lane + j];
  }

  // FastWalshHadamard128's h=1 and h=2 stages, in that order.
  const float a0 = x[0] + x[1], a1 = x[0] - x[1];
  const float a2 = x[2] + x[3], a3 = x[2] - x[3];
  x[0] = a0 + a2;
  x[1] = a1 + a3;
  x[2] = a0 - a2;
  x[3] = a1 - a3;
  // h=4..64 crosses lanes. In the upper half, the lower-index operand is
  // the peer: peer - self, never self - peer or a reassociated sum.
#pragma unroll
  for (unsigned mask = 1; mask <= 16; mask *= 2) {
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const float peer = __shfl_xor(x[j], mask, 32);
      x[j] = (lane & mask) != 0U ? peer - x[j] : x[j] + peer;
    }
  }
  std::uint64_t packed = 0;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    x[j] *= turbo4::kInvSqrt128;
    x[j] *= signs1[4U * lane + j];
    packed |= static_cast<std::uint64_t>(turbo4::FloatToHalfBits(x[j]))
              << (16U * j);
  }
  // The normal scratch is 8-byte aligned: one 8-byte store per lane.
  // Keep the old API's unaligned-pointer support without imposing a new
  // geometry refusal. The alias type permits stores into a half allocation.
  using PackedHalfs = std::uint64_t __attribute__((__may_alias__));
  if ((reinterpret_cast<std::uintptr_t>(dst) & 7U) == 0U)
    reinterpret_cast<PackedHalfs*>(dst)[lane] = packed;
  else
    std::memcpy(dst + 4U * lane, &packed, sizeof(packed));
}

__global__ void Turbo4KvDequantV2Kernel(
    const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, std::uint16_t* __restrict__ k_dst,
    std::uint16_t* __restrict__ v_dst, std::uint32_t count,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, Turbo4KernelStatus* __restrict__ status,
    turbo4::PlaneFormats formats) {
  __shared__ float centroids[2][64];
  __shared__ float signs1[128];
  __shared__ float signs2[128];
  const unsigned tid = threadIdx.x;
  if (tid < 64U) {
    centroids[0][tid] = turbo4::CentroidValue(formats.k, tid);
    centroids[1][tid] = turbo4::CentroidValue(formats.v, tid);
  }
  signs1[tid] = turbo4::TURBO_WHT_SIGNS1[tid];
  signs2[tid] = turbo4::TURBO_WHT_SIGNS2[tid];
  // Even tail waves participate: no early exit before this CTA barrier.
  __syncthreads();

  const std::size_t t =
      static_cast<std::size_t>(blockIdx.x) * kDequantV2Waves + tid / 32U;
  if (t >= count)
    return;
  const std::uint32_t h = blockIdx.y;
  std::size_t base_byte = 0;
  if (!Turbo4BlockByteOffset(t, h, head_dim, num_kv_heads, max_context,
                             &base_byte, formats.stride)) {
    if ((tid & 31U) == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  const std::size_t dst_offset =
      (t * num_kv_heads + h) * static_cast<std::size_t>(head_dim);
  const std::size_t block_bytes = turbo4::BlockBytes(formats.stride);
  // Dequant has no writer staging bound: retain every multiple-of-128 head.
  for (std::size_t b = 0; b < Turbo4BlocksPerHead(head_dim); ++b) {
    const std::size_t src_offset = base_byte + b * block_bytes;
    const std::size_t out_offset = dst_offset + b * turbo4::kBlockValues;
    Turbo4DequantWaveBlock(k_plane + src_offset, k_dst + out_offset, formats.k,
                           centroids[0], signs1, signs2);
    Turbo4DequantWaveBlock(v_plane + src_offset, v_dst + out_offset, formats.v,
                           centroids[1], signs1, signs2);
  }
  // Success does not touch the latch, including an earlier writer failure.
}

}  // namespace

/// Dequantize one layer's Turbo4 blocks into an f16 scratch, positions
/// [0, count).
///
/// `k_dst` and `v_dst` are the scratch's K and V halves. For a ONE-LAYER
/// scratch the caller supplies v_dst = k_dst + max_context * kv_heads *
/// head_dim halfs and passes layer_idx 0 to the attention launcher; the V
/// offset is the caller's because it is a property of the allocation it owns.
///
/// `d_status` is REQUIRED and is never dereferenced on the host. It is LATCHED,
/// never reset — the caller resets once per chunk, the same contract the KV
/// write follows.
Turbo4LaunchStatus LaunchTurbo4DequantLayer(
    const std::uint8_t* k_plane, const std::uint8_t* v_plane, void* k_dst,
    void* v_dst, std::uint32_t count, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    Turbo4KernelStatus* d_status, hipStream_t stream,
    turbo4::PlaneFormats formats) {
  // HOST-SIDE checks only; d_status is device memory and is never written here.
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  if (k_plane == nullptr || v_plane == nullptr || k_dst == nullptr ||
      v_dst == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  if (d_status == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  if (count == 0U)
    return Turbo4LaunchStatus::kBadArguments;
  // Unsigned compare, so count + offset cannot wrap.
  if (count > max_context)
    return Turbo4LaunchStatus::kGeometryRefused;  // no such enum value;
                                                  // geometry is the closest
                                                  // host check
  if (num_kv_heads == 0U || max_context == 0U ||
      Turbo4BlocksPerHead(head_dim) == 0U)
    return Turbo4LaunchStatus::kGeometryRefused;

  const char* deq_v2 = std::getenv("GUFO_DEV_DEQ_V2");
  if (deq_v2 != nullptr && deq_v2[0] == '1' && deq_v2[1] == '\0') {
    const dim3 grid((static_cast<std::size_t>(count) + kDequantV2Waves - 1U) /
                        kDequantV2Waves,
                    num_kv_heads, 1U);
    Turbo4KvDequantV2Kernel<<<grid, dim3(32U * kDequantV2Waves, 1U, 1U), 0,
                              stream>>>(
        k_plane, v_plane, static_cast<std::uint16_t*>(k_dst),
        static_cast<std::uint16_t*>(v_dst), count, max_context, num_kv_heads,
        head_dim, d_status, formats);
  } else {
    const dim3 grid((count + kDequantThreads - 1U) / kDequantThreads,
                    num_kv_heads, 1U);
    Turbo4KvDequantKernel<<<grid, dim3(kDequantThreads, 1U, 1U), 0, stream>>>(
        k_plane, v_plane, static_cast<std::uint16_t*>(k_dst),
        static_cast<std::uint16_t*>(v_dst), count, max_context, num_kv_heads,
        head_dim, d_status, formats);
  }
  // PEEK, not get: hipGetLastError clears, and clearing an error the caller
  // still needs is its own silent loss.
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
// Port of TheTom/llama-cpp-turboquant (MIT, Copyright (c) 2023-2026 The ggml authors; turbo5/turbo6 from PR #381
// by JakeATX), modified for HIP/gfx1151 in tubo. License text: licenses/llama-cpp-turboquant.txt.
// Post-RoPE Turbo4 KV WRITE kernel.
//
//   /opt/rocm/bin/hipcc -fsyntax-only -std=c++20 -DENGINE_ENABLE_HIP -I. -x hip
//   \
//       src/models/qwen/hip/kernels/turbo4_kv_write.hip
//
// The host/device `Turbo4WriteHeadBody` remains the serial CPU oracle. This GPU
// path uses 128 threads per KV head, parallelizes one 128-value block at a
// time, and stages the head's K0/K1/V0/V1 bytes before committing any of them.
//
// WHOLE-HEAD ATOMICITY: a refused block leaves the head's old plane bytes
// intact.
//
// EXPLICIT STATUS: every path returns a Turbo4KernelStatus, including success.
// A silent return is a silent stale success — the previous token's blocks with
// no indication this token never landed.
//
// WHAT IS NOT PROVEN: the grid choice itself is compile-only. The body, the
// index arithmetic and the codec all have a CPU oracle.
//
// Every entry point takes a `turbo4::Bits` (default k4). The 128-thread
// layout is width-independent: thread j owns value j of the block. Only the
// code lookup, the qs nibble, the qh bits and the block pitch change with
// `formats`; the staging array is sized for the widest block.

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace gufo::hip {
namespace {

template<turbo4::Bits kBits>
struct Turbo4WriterShared {
  float values[turbo4::kBlockValues];
  float warp_sums[4];
  float group_norm;
  float inv_group_norm;
  float recon_norm;
  std::uint32_t pos;
  std::size_t base_byte;
  std::size_t bph;
  std::uint32_t result;
  std::uint32_t head_failed;
  std::uint32_t block_failed;
  std::uint8_t indices[turbo4::kBlockValues];
  std::uint8_t stage[kTurbo4StagedBlocksPerHead * turbo4::BlockBytes(kBits)];
};

static_assert(sizeof(Turbo4WriterShared<turbo4::Bits::k4>::stage) ==
                  kTurbo4StagedBytesPerHead,
              "default Turbo4 writer staging must not grow");

template<turbo4::Bits kBits>
struct Turbo4FusedShared {
  Turbo4WriterShared<kBits> writer;
  float head_values[2 * turbo4::kBlockValues];
  float norm_sums[turbo4::kBlockValues];
  float inv_rms;
};

/// Match PerHeadRMSNormKernel's 256-value tree with 128 lanes: the first
/// reduction stage is the pair x[i]^2 + x[i+128]^2, then the same 64..1 tree.
template<turbo4::Bits kBits>
__device__ bool Turbo4NormalizeHead(const float* src, const float* weight,
                                    std::uint32_t head_dim, float eps,
                                    Turbo4FusedShared<kBits>& shared) {
  const std::uint32_t tid = threadIdx.x;
  if (tid == 0U)
    shared.writer.block_failed = 0U;
  __syncthreads();

  const float x0 = src[tid];
  const float x1 = src[tid + turbo4::kBlockValues];
  if (!Turbo4IsFinite(x0) || !Turbo4IsFinite(x1))
    atomicExch(&shared.writer.block_failed, 1U);
  shared.head_values[tid] = x0;
  shared.head_values[tid + turbo4::kBlockValues] = x1;
  volatile float sq0 = x0 * x0;
  volatile float sq1 = x1 * x1;
  shared.norm_sums[tid] = sq0 + sq1;
  __syncthreads();

  for (std::uint32_t stride = turbo4::kBlockValues / 2U; stride > 0U;
       stride >>= 1U) {
    if (tid < stride)
      shared.norm_sums[tid] += shared.norm_sums[tid + stride];
    __syncthreads();
  }
  if (tid == 0U) {
    const float mean_sq = shared.norm_sums[0] / static_cast<float>(head_dim);
    shared.inv_rms = rsqrtf(mean_sq + eps);
    if (!Turbo4IsFinite(shared.inv_rms))
      shared.writer.block_failed = 1U;
  }
  __syncthreads();
  if (shared.writer.block_failed != 0U)
    return false;

  const float w0 = weight != nullptr ? weight[tid] : 1.0F;
  const float w1 =
      weight != nullptr ? weight[tid + turbo4::kBlockValues] : 1.0F;
  shared.head_values[tid] = (x0 * shared.inv_rms) * w0;
  shared.head_values[tid + turbo4::kBlockValues] = (x1 * shared.inv_rms) * w1;
  __syncthreads();
  return true;
}

__device__ void Turbo4ApplyRopeHead(
    float* head_values, std::uint32_t pos, std::uint32_t rotary_dim,
    float rope_theta, const models::qwen::vision::DeviceRope* rope) {
  const std::uint32_t pair = threadIdx.x;
  const std::uint32_t half_rot = rotary_dim / 2U;
  if (pair < half_rot) {
    const float exponent =
        static_cast<float>(2U * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle =
        models::qwen::vision::RopePosition(rope, pos, pair) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float x0 = head_values[pair];
    const float x1 = head_values[pair + half_rot];
    head_values[pair] = x0 * cos_val - x1 * sin_val;
    head_values[pair + half_rot] = x0 * sin_val + x1 * cos_val;
  }
  __syncthreads();
}

template<turbo4::Bits kBits>
__device__ bool Turbo4RotateQueryHead(const float* head_values, float* q_rot,
                                      std::uint32_t head,
                                      std::uint32_t head_dim,
                                      Turbo4FusedShared<kBits>& shared) {
  const std::uint32_t tid = threadIdx.x;
  const std::size_t head_base = static_cast<std::size_t>(head) * head_dim;
  for (std::uint32_t offset = 0; offset < head_dim;
       offset += turbo4::kBlockValues) {
    if (tid == 0U)
      shared.writer.block_failed = 0U;
    __syncthreads();
    const float value = head_values[offset + tid];
    if (!Turbo4IsFinite(value))
      atomicExch(&shared.writer.block_failed, 1U);
    shared.writer.values[tid] = value;
    __syncthreads();
    if (shared.writer.block_failed != 0U)
      return false;

    shared.writer.values[tid] *= turbo4::TURBO_WHT_SIGNS1[tid];
    __syncthreads();
    for (std::uint32_t stride = 1U; stride < turbo4::kBlockValues;
         stride *= 2U) {
      const std::uint32_t group = tid / (2U * stride);
      const std::uint32_t within = tid % (2U * stride);
      if (within < stride) {
        const std::uint32_t a = group * 2U * stride + within;
        const std::uint32_t b = a + stride;
        const float av = shared.writer.values[a];
        const float bv = shared.writer.values[b];
        shared.writer.values[a] = av + bv;
        shared.writer.values[b] = av - bv;
      }
      __syncthreads();
    }
    shared.writer.values[tid] *=
        turbo4::kInvSqrt128 * turbo4::TURBO_WHT_SIGNS2[tid];
    q_rot[head_base + offset + tid] = shared.writer.values[tid];
    __syncthreads();
  }
  return true;
}

/// One 128-thread CTA quantizes one block with the donor's warp/tree order.
/// All threads call this uniformly; staged_bytes points inside shared memory.
template<turbo4::Bits kBits>
__device__ void Turbo4QuantizeBlockParallel(const float* __restrict__ src,
                                            std::uint8_t* staged_bytes,
                                            Turbo4WriterShared<kBits>& shared,
                                            turbo4::Bits bits) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  if (tid == 0U)
    shared.block_failed = 0U;
  __syncthreads();

  const float original = src[tid];
  shared.values[tid] = original;
  if (!Turbo4IsFinite(original))
    atomicExch(&shared.block_failed, 1U);

  float squared = original * original;
  for (int offset = 16; offset > 0; offset >>= 1)
    squared += __shfl_xor(squared, offset, 32);
  if (lane == 0U)
    shared.warp_sums[tid / 32U] = squared;
  __syncthreads();

  if (tid == 0U) {
    float total = 0.0F;
    for (std::uint32_t warp = 0; warp < 4U; ++warp)
      total += shared.warp_sums[warp];
    shared.group_norm = sqrtf(total);
    shared.inv_group_norm = shared.group_norm > turbo4::kNormEpsilon
                                ? 1.0F / shared.group_norm
                                : 0.0F;
    if (!Turbo4IsFinite(shared.group_norm))
      shared.block_failed = 1U;
  }
  __syncthreads();

  float value = shared.values[tid] * shared.inv_group_norm;
  value *= turbo4::TURBO_WHT_SIGNS1[tid];
  for (int h = 1; h < 32; h <<= 1) {
    const float peer = __shfl_xor(value, h, 32);
    value = (lane & static_cast<std::uint32_t>(h)) != 0U ? peer - value
                                                         : value + peer;
  }
  shared.values[tid] = value;
  __syncthreads();

  // The last two FWHT stages cross wave boundaries. Match strix set-rows.cu
  // 1072-1084: 32-value and then 64-value butterfly, with a barrier each time.
  if ((tid & 63U) < 32U) {
    const float a = shared.values[tid];
    const float b = shared.values[tid + 32U];
    shared.values[tid] = a + b;
    shared.values[tid + 32U] = a - b;
  }
  __syncthreads();
  if (tid < 64U) {
    const float a = shared.values[tid];
    const float b = shared.values[tid + 64U];
    shared.values[tid] = a + b;
    shared.values[tid + 64U] = a - b;
  }
  __syncthreads();

  value = shared.values[tid] * turbo4::kInvSqrt128;
  value *= turbo4::TURBO_WHT_SIGNS2[tid];
  const std::uint8_t index = turbo4::NearestCode(bits, value);
  shared.indices[tid] = index;
  __syncthreads();
  if ((tid & 1U) == 0U) {
    staged_bytes[2U + tid / 2U] = static_cast<std::uint8_t>(
        (turbo4::QsNibbleOfCode(bits, index) & 0x0FU) |
        (turbo4::QsNibbleOfCode(bits, shared.indices[tid + 1U]) << 4U));
  }
  // qh plane (Turbo5: 8 sign bits per byte, Turbo6: 4 x 2 high bits per byte).
  // Each gathering thread reads only `indices`, which every thread wrote before
  // the barrier above, and writes a byte no other thread touches.
  if (bits == turbo4::Bits::k5 && (tid & 7U) == 0U) {
    std::uint32_t qh = 0U;
    for (std::uint32_t i = 0; i < 8U; ++i)
      qh |= turbo4::QhBitsOfCode(bits, shared.indices[tid + i], tid + i);
    staged_bytes[turbo4::kQhOffset + tid / 8U] = static_cast<std::uint8_t>(qh);
  } else if (bits == turbo4::Bits::k6 && (tid & 3U) == 0U) {
    std::uint32_t qh = 0U;
    for (std::uint32_t i = 0; i < 4U; ++i)
      qh |= turbo4::QhBitsOfCode(bits, shared.indices[tid + i], tid + i);
    staged_bytes[turbo4::kQhOffset + tid / 4U] = static_cast<std::uint8_t>(qh);
  }

  const float centroid = turbo4::CentroidValue(bits, index);
  float recon_squared = centroid * centroid;
  for (int offset = 16; offset > 0; offset >>= 1)
    recon_squared += __shfl_xor(recon_squared, offset, 32);
  if (lane == 0U)
    shared.warp_sums[tid / 32U] = recon_squared;
  __syncthreads();

  if (tid == 0U) {
    float total = 0.0F;
    for (std::uint32_t warp = 0; warp < 4U; ++warp)
      total += shared.warp_sums[warp];
    shared.recon_norm = sqrtf(total);
    const float corrected = shared.recon_norm > turbo4::kNormEpsilon
                                ? shared.group_norm / shared.recon_norm
                                : shared.group_norm;
    const std::uint16_t half = turbo4::FloatToHalfBits(corrected);
    const float stored = turbo4::HalfBitsToFloat(half);
    if (!turbo4::GroupNormIsStorable(stored))
      shared.block_failed = 1U;
    staged_bytes[0] = static_cast<std::uint8_t>(half & 0xFFU);
    staged_bytes[1] = static_cast<std::uint8_t>(half >> 8U);
  }
  __syncthreads();
}

/// Quantize a head's K/V into shared staging, then commit the complete head.
template<turbo4::Bits kBits>
__device__ Turbo4KernelStatus Turbo4StageCommitHead(
    const float* __restrict__ k_head, const float* __restrict__ v_head,
    std::uint8_t* __restrict__ k_plane, std::uint8_t* __restrict__ v_plane,
    std::size_t base_byte, std::size_t bph, Turbo4WriterShared<kBits>& shared,
    turbo4::PlaneFormats formats) {
  const std::uint32_t tid = threadIdx.x;
  const std::size_t block_bytes = turbo4::BlockBytes(formats.stride);
  if (tid == 0U)
    shared.head_failed = 0U;
  // Clear every padding byte too: snapshots retain the complete slot.
  for (std::size_t byte = tid; byte < 2U * bph * block_bytes;
       byte += blockDim.x)
    shared.stage[byte] = 0;
  __syncthreads();

  for (std::size_t block = 0; block < bph; ++block) {
    Turbo4QuantizeBlockParallel(k_head + block * turbo4::kBlockValues,
                                shared.stage + block * block_bytes, shared,
                                formats.k);
    if (tid == 0U && shared.block_failed != 0U)
      shared.head_failed = 1U;
    __syncthreads();
  }
  for (std::size_t block = 0; block < bph; ++block) {
    Turbo4QuantizeBlockParallel(v_head + block * turbo4::kBlockValues,
                                shared.stage + (bph + block) * block_bytes,
                                shared, formats.v);
    if (tid == 0U && shared.block_failed != 0U)
      shared.head_failed = 1U;
    __syncthreads();
  }

  if (shared.head_failed == 0U) {
    const std::size_t k_stage_bytes = bph * block_bytes;
    const std::size_t all_stage_bytes = 2U * k_stage_bytes;
    for (std::size_t byte = tid; byte < all_stage_bytes; byte += blockDim.x) {
      if (byte < k_stage_bytes) {
        k_plane[base_byte + byte] = shared.stage[byte];
      } else {
        v_plane[base_byte + byte - k_stage_bytes] = shared.stage[byte];
      }
    }
  } else if (tid == 0U) {
    shared.result =
        static_cast<std::uint32_t>(Turbo4KernelStatus::kQuantizeRefused);
  }
  __syncthreads();
  return static_cast<Turbo4KernelStatus>(shared.result);
}

/// Stage every K/V block for one head; commit only if every block is storable.
template<turbo4::Bits kBits>
__device__ Turbo4KernelStatus Turbo4WriteHeadParallel(
    const float* __restrict__ k_post_rope, const float* __restrict__ v,
    std::uint8_t* __restrict__ k_plane, std::uint8_t* __restrict__ v_plane,
    std::size_t plane_bytes, std::size_t source_bytes, std::uint32_t head,
    std::uint32_t pos, std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, Turbo4WriterShared<kBits>& shared,
    turbo4::PlaneFormats formats) {
  const std::uint32_t tid = threadIdx.x;
  if (tid == 0U) {
    shared.result = static_cast<std::uint32_t>(Turbo4KernelStatus::kOk);
    shared.head_failed = 0U;
    shared.bph = Turbo4BlocksPerHead(head_dim);
    if (!formats.Valid() || formats.stride != kBits || head >= num_kv_heads ||
        num_kv_heads == 0U || k_post_rope == nullptr || v == nullptr ||
        k_plane == nullptr || v_plane == nullptr ||
        !Turbo4WriteBuffersAreSufficient(plane_bytes, source_bytes,
                                         num_kv_heads, head_dim) ||
        shared.bph == 0U || !Turbo4HeadFitsStaging(head_dim)) {
      shared.result =
          static_cast<std::uint32_t>(Turbo4KernelStatus::kGeometryRefused);
    } else if (pos >= max_context) {
      shared.result =
          static_cast<std::uint32_t>(Turbo4KernelStatus::kPositionOutOfRange);
    } else {
      const std::size_t head_span =
          shared.bph * turbo4::BlockBytes(formats.stride);
      if (!Turbo4BlockByteOffset(pos, head, head_dim, num_kv_heads, max_context,
                                 &shared.base_byte, formats.stride) ||
          !Turbo4BytesFit(shared.base_byte, head_span, plane_bytes)) {
        shared.result =
            static_cast<std::uint32_t>(Turbo4KernelStatus::kGeometryRefused);
      }
    }
  }
  __syncthreads();
  if (shared.result != static_cast<std::uint32_t>(Turbo4KernelStatus::kOk))
    return static_cast<Turbo4KernelStatus>(shared.result);

  const std::size_t head_base = static_cast<std::size_t>(head) * head_dim;
  return Turbo4StageCommitHead(k_post_rope + head_base, v + head_base, k_plane,
                               v_plane, shared.base_byte, shared.bph, shared,
                               formats);
}

/// One CTA per KV head, 128 threads for each 128-value block.
template<turbo4::Bits kBits>
__global__ void Turbo4KvWriteKernel(
    const float* __restrict__ k_post_rope, const float* __restrict__ v,
    std::uint8_t* __restrict__ k_plane, std::uint8_t* __restrict__ v_plane,
    const std::uint32_t* __restrict__ d_pos, std::size_t plane_bytes,
    std::size_t source_bytes, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    Turbo4KernelStatus* __restrict__ status, turbo4::PlaneFormats formats) {
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  __shared__ Turbo4WriterShared<kBits> shared;
  if (d_pos == nullptr) {
    if (tid == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  if (tid == 0U)
    shared.pos = *d_pos;
  __syncthreads();
  const Turbo4KernelStatus result = Turbo4WriteHeadParallel(
      k_post_rope, v, k_plane, v_plane, plane_bytes, source_bytes, head,
      shared.pos, max_context, num_kv_heads, head_dim, shared, formats);
  if (tid == 0U)
    Turbo4LatchFailure(status, result);
}

/// One 128-thread CTA per POSITION and KV HEAD: grid (count, num_kv_heads).
///
/// `k_post_rope` / `v` are the CHUNK, [count, num_kv_heads, head_dim]; the
/// source rows are position-major; each CTA stages and commits one whole head.
template<turbo4::Bits kBits>
__global__ void Turbo4KvWriteBatchedKernel(
    const float* __restrict__ k_post_rope, const float* __restrict__ v,
    std::uint8_t* __restrict__ k_plane, std::uint8_t* __restrict__ v_plane,
    const std::uint32_t* __restrict__ d_first_pos, std::uint32_t count,
    std::size_t plane_bytes, std::size_t source_bytes,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, Turbo4KernelStatus* __restrict__ status,
    turbo4::PlaneFormats formats) {
  const std::uint32_t index = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  __shared__ Turbo4WriterShared<kBits> shared;
  if (index >= count)
    return;
  if (d_first_pos == nullptr) {
    if (tid == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  const std::size_t pos_stride = static_cast<std::size_t>(num_kv_heads) *
                                 static_cast<std::size_t>(head_dim);
  const std::size_t off = static_cast<std::size_t>(index) * pos_stride;
  if (tid == 0U) {
    const std::uint32_t first = *d_first_pos;
    shared.pos = first + index;
    shared.result = first > static_cast<std::uint32_t>(-1) - index
                        ? static_cast<std::uint32_t>(
                              Turbo4KernelStatus::kPositionOutOfRange)
                        : static_cast<std::uint32_t>(Turbo4KernelStatus::kOk);
  }
  __syncthreads();
  if (shared.result != static_cast<std::uint32_t>(Turbo4KernelStatus::kOk)) {
    if (tid == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kPositionOutOfRange);
    return;
  }
  const Turbo4KernelStatus result = Turbo4WriteHeadParallel(
      k_post_rope + off, v + off, k_plane, v_plane, plane_bytes,
      source_bytes - off * sizeof(float), head, shared.pos, max_context,
      num_kv_heads, head_dim, shared, formats);
  if (tid == 0U)
    Turbo4LatchFailure(status, result);
}

/// One fused pre-step launch: one CTA per Q head plus one CTA per KV head.
/// Q CTAs emit the Turbo4-rotated query. KV CTAs normalize/rotate K, stage K/V,
/// and commit the complete head. The following stream-ordered attention reader
/// sees either all valid K/V blocks or a latched refusal.
template<turbo4::Bits kBits>
__global__ void Turbo4DecodePrepareKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, float* __restrict__ q_rot,
    std::uint8_t* __restrict__ k_plane, std::uint8_t* __restrict__ v_plane,
    const std::uint32_t* __restrict__ d_pos, std::size_t plane_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t max_context, float rope_theta,
    float eps, Turbo4KernelStatus* __restrict__ write_status,
    Turbo4KernelStatus* __restrict__ read_status,
    const models::qwen::vision::DeviceRope* __restrict__ rope,
    turbo4::PlaneFormats formats) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t cta = blockIdx.x;
  const bool query_cta = cta < num_heads;
  __shared__ Turbo4FusedShared<kBits> shared;
  Turbo4KernelStatus* const own_status = query_cta ? read_status : write_status;

  if (tid == 0U) {
    shared.writer.result = static_cast<std::uint32_t>(Turbo4KernelStatus::kOk);
    if (!formats.Valid() || formats.stride != kBits || d_pos == nullptr ||
        own_status == nullptr || write_status == nullptr ||
        read_status == nullptr) {
      shared.writer.result =
          static_cast<std::uint32_t>(Turbo4KernelStatus::kGeometryRefused);
    } else {
      const auto read_code = Turbo4ReadWriterStatus(read_status);
      const auto write_code = Turbo4ReadWriterStatus(write_status);
      const auto prior = static_cast<std::uint32_t>(read_code) >
                                 static_cast<std::uint32_t>(write_code)
                             ? read_code
                             : write_code;
      shared.writer.result = static_cast<std::uint32_t>(prior);
      shared.writer.pos = *d_pos;
      if (prior == Turbo4KernelStatus::kOk &&
          shared.writer.pos >= max_context) {
        shared.writer.result =
            static_cast<std::uint32_t>(Turbo4KernelStatus::kPositionOutOfRange);
      }
    }
  }
  __syncthreads();
  if (shared.writer.result !=
      static_cast<std::uint32_t>(Turbo4KernelStatus::kOk)) {
    if (tid == 0U)
      Turbo4LatchFailure(own_status,
                         static_cast<Turbo4KernelStatus>(shared.writer.result));
    return;
  }

  if (query_cta) {
    const std::uint32_t head = cta;
    const float* q_head = q + static_cast<std::size_t>(head) * head_dim;
    if (!Turbo4NormalizeHead(q_head, q_weight, head_dim, eps, shared)) {
      if (tid == 0U)
        Turbo4LatchFailure(read_status, Turbo4KernelStatus::kGeometryRefused);
      return;
    }
    Turbo4ApplyRopeHead(shared.head_values, shared.writer.pos, rotary_dim,
                        rope_theta, rope);
    if (!Turbo4RotateQueryHead(shared.head_values, q_rot, head, head_dim,
                               shared)) {
      if (tid == 0U)
        Turbo4LatchFailure(read_status, Turbo4KernelStatus::kGeometryRefused);
    }
    return;
  }

  const std::uint32_t kv_head = cta - num_heads;
  const float* k_head = k + static_cast<std::size_t>(kv_head) * head_dim;
  const float* v_head = v + static_cast<std::size_t>(kv_head) * head_dim;
  if (!Turbo4NormalizeHead(k_head, k_weight, head_dim, eps, shared)) {
    if (tid == 0U)
      Turbo4LatchFailure(write_status, Turbo4KernelStatus::kQuantizeRefused);
    return;
  }
  Turbo4ApplyRopeHead(shared.head_values, shared.writer.pos, rotary_dim,
                      rope_theta, rope);
  if (tid == 0U) {
    shared.writer.result = static_cast<std::uint32_t>(Turbo4KernelStatus::kOk);
    shared.writer.bph = Turbo4BlocksPerHead(head_dim);
    const std::size_t head_span =
        shared.writer.bph * turbo4::BlockBytes(formats.stride);
    if (kv_head >= num_kv_heads || shared.writer.bph == 0U ||
        !Turbo4HeadFitsStaging(head_dim) ||
        !Turbo4BlockByteOffset(shared.writer.pos, kv_head, head_dim,
                               num_kv_heads, max_context,
                               &shared.writer.base_byte, formats.stride) ||
        !Turbo4BytesFit(shared.writer.base_byte, head_span, plane_bytes)) {
      shared.writer.result =
          static_cast<std::uint32_t>(Turbo4KernelStatus::kGeometryRefused);
    }
  }
  __syncthreads();
  if (shared.writer.result !=
      static_cast<std::uint32_t>(Turbo4KernelStatus::kOk)) {
    if (tid == 0U)
      Turbo4LatchFailure(write_status,
                         static_cast<Turbo4KernelStatus>(shared.writer.result));
    return;
  }
  const auto result = Turbo4StageCommitHead(
      shared.head_values, v_head, k_plane, v_plane, shared.writer.base_byte,
      shared.writer.bph, shared.writer, formats);
  if (tid == 0U)
    Turbo4LatchFailure(write_status, result);
}

}  // namespace

/// Write a whole CHUNK of post-RoPE K and V into a layer's Turbo4 planes.
///
/// The chunk is [count, num_kv_heads, head_dim]. Positions are independent,
/// so this is one 128-thread CTA per (position, KV head); each CTA keeps the
/// whole-head staging and commit boundary.
///
/// `d_first_pos` is a DEVICE pointer for the same reason the single-position
/// writer takes one: the position must stay correct under graph capture.
/// `count` is a HOST value because it sizes the grid.
Turbo4LaunchStatus LaunchTurbo4KvWriteBatched(
    const float* k_post_rope, const float* v, std::uint8_t* k_plane,
    std::uint8_t* v_plane, const std::uint32_t* d_first_pos,
    std::uint32_t count, std::size_t plane_bytes, std::size_t source_bytes,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, Turbo4KernelStatus* d_status, hipStream_t stream,
    turbo4::PlaneFormats formats) {
  // HOST-SIDE checks only; `d_status` is never dereferenced here.
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  if (k_post_rope == nullptr || v == nullptr || k_plane == nullptr ||
      v_plane == nullptr || d_first_pos == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  if (d_status == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  if (count == 0U || plane_bytes == 0U)
    return Turbo4LaunchStatus::kBadArguments;
  if (num_kv_heads == 0U || max_context == 0U ||
      !Turbo4HeadFitsStaging(head_dim))
    return Turbo4LaunchStatus::kGeometryRefused;
  // The WHOLE range is checked on the host, so an out-of-range chunk never
  // dispatches a grid that would partly overwrite the plane.
  const std::size_t pos_stride = static_cast<std::size_t>(num_kv_heads) *
                                 static_cast<std::size_t>(head_dim);
  if (source_bytes < pos_stride * count * sizeof(float))
    return Turbo4LaunchStatus::kBadArguments;
  if (Turbo4BlocksPerHead(head_dim) == 0U)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (source_bytes < pos_stride * sizeof(float))
    return Turbo4LaunchStatus::kBadArguments;
  // NO RESET of d_status: the caller resets once per step. See the
  // single-position launcher for why a per-call memset erases a refusal.

  const dim3 grid(count, num_kv_heads, 1U);
  switch (formats.stride) {
    case turbo4::Bits::k4:
      Turbo4KvWriteBatchedKernel<turbo4::Bits::k4>
          <<<grid, dim3(turbo4::kBlockValues, 1U, 1U), 0, stream>>>(
              k_post_rope, v, k_plane, v_plane, d_first_pos, count, plane_bytes,
              source_bytes, max_context, num_kv_heads, head_dim, d_status,
              formats);
      break;
    case turbo4::Bits::k5:
      Turbo4KvWriteBatchedKernel<turbo4::Bits::k5>
          <<<grid, dim3(turbo4::kBlockValues, 1U, 1U), 0, stream>>>(
              k_post_rope, v, k_plane, v_plane, d_first_pos, count, plane_bytes,
              source_bytes, max_context, num_kv_heads, head_dim, d_status,
              formats);
      break;
    case turbo4::Bits::k6:
      Turbo4KvWriteBatchedKernel<turbo4::Bits::k6>
          <<<grid, dim3(turbo4::kBlockValues, 1U, 1U), 0, stream>>>(
              k_post_rope, v, k_plane, v_plane, d_first_pos, count, plane_bytes,
              source_bytes, max_context, num_kv_heads, head_dim, d_status,
              formats);
      break;
  }
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

/// Write one token's post-RoPE K and V into a layer's Turbo4 planes.
///
/// `k_post_rope` and `v` are [num_kv_heads, head_dim] device floats, and
/// `source_bytes` is how long EACH of them really is. A non-null pointer does
/// not say that: a caller that allocated one head of a four-head tensor gets a
/// valid address and a three-head overflow. `k_plane` and `v_plane` are that
/// layer's planes, each at least `plane_bytes`, which is likewise the real
/// length and not a claim about it.
///
/// `status` receives a Turbo4KernelStatus on every path. Launched as one CTA
/// per KV head with 128 threads; each CTA parallelizes its 128-value blocks and
/// commits the head only after all four UD blocks succeed.
Turbo4LaunchStatus LaunchTurbo4KvWrite(
    const float* k_post_rope, const float* v, std::uint8_t* k_plane,
    std::uint8_t* v_plane, const std::uint32_t* d_pos, std::size_t plane_bytes,
    std::size_t source_bytes, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    Turbo4KernelStatus* d_status, hipStream_t stream,
    turbo4::PlaneFormats formats) {
  // HOST-SIDE checks only. `d_status` is NEVER dereferenced here: it is device
  // memory, and writing it from the host is what review called out.
  // A host refusal is reported through this return value instead.
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  if (k_post_rope == nullptr || v == nullptr || k_plane == nullptr ||
      v_plane == nullptr || d_pos == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  // FAIL-CLOSED: a status destination is REQUIRED, not optional. Review
  // found: an optional status leaves a path on which nothing records what
  // the device did, and "no status" then reads as "nothing went wrong".
  if (d_status == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  if (plane_bytes == 0U)
    return Turbo4LaunchStatus::kBadArguments;
  if (num_kv_heads == 0U || max_context == 0U ||
      !Turbo4HeadFitsStaging(head_dim))
    return Turbo4LaunchStatus::kGeometryRefused;
  // SAME GATE, SAME FUNCTION as the body runs. Checking here means a caller
  // with a short source never dispatches 24 threads that all have to discover
  // the same problem on silicon; the body keeps its own copy because the kernel
  // can be reached without passing through this launcher.
  if (!Turbo4WriteBuffersAreSufficient(plane_bytes, source_bytes, num_kv_heads,
                                       head_dim))
    return Turbo4LaunchStatus::kBadArguments;

  // NO RESET HERE. This used to memset the word to kOk on every call, which was
  // correct for a one-shot launch and wrong for the real one: decode_step calls
  // the writer once per attention layer, 16 times per token, so a refusal in
  // layer k was erased by layer k+1 and the step reported success (found in
  // review).
  //
  // The word is now reset ONCE PER STEP by the caller, which owns the step
  // boundary and is the only party that knows where it is. This launcher only
  // LATCHES, which is what makes the max-latch useful across layers: a failure
  // recorded by layer 3 survives layers 4..15.

  switch (formats.stride) {
    case turbo4::Bits::k4:
      Turbo4KvWriteKernel<turbo4::Bits::k4>
          <<<num_kv_heads, turbo4::kBlockValues, 0, stream>>>(
              k_post_rope, v, k_plane, v_plane, d_pos, plane_bytes,
              source_bytes, max_context, num_kv_heads, head_dim, d_status,
              formats);
      break;
    case turbo4::Bits::k5:
      Turbo4KvWriteKernel<turbo4::Bits::k5>
          <<<num_kv_heads, turbo4::kBlockValues, 0, stream>>>(
              k_post_rope, v, k_plane, v_plane, d_pos, plane_bytes,
              source_bytes, max_context, num_kv_heads, head_dim, d_status,
              formats);
      break;
    case turbo4::Bits::k6:
      Turbo4KvWriteKernel<turbo4::Bits::k6>
          <<<num_kv_heads, turbo4::kBlockValues, 0, stream>>>(
              k_post_rope, v, k_plane, v_plane, d_pos, plane_bytes,
              source_bytes, max_context, num_kv_heads, head_dim, d_status,
              formats);
      break;
  }
  // ENQUEUE is not EXECUTION. The launch can still be refused — a grid beyond
  // the device limit, an unlaunchable configuration — and returning kLaunched
  // without asking means the caller believes a kernel is running when none is.
  // PEEK, not get: hipGetLastError clears, and clearing an error the caller
  // still needs is its own silent loss.
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

Turbo4LaunchStatus LaunchTurbo4DecodePrepare(
    const float* q, const float* k, const float* v, const float* q_norm_weight,
    const float* k_norm_weight, float* q_rot, std::uint8_t* k_plane,
    std::uint8_t* v_plane, const std::uint32_t* d_pos, std::size_t q_bytes,
    std::size_t kv_bytes, std::size_t q_rot_bytes, std::size_t plane_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t max_context, float rope_theta,
    float eps, Turbo4KernelStatus* d_write_status,
    Turbo4KernelStatus* d_read_status,
    const models::qwen::vision::DeviceRope* rope, hipStream_t stream,
    turbo4::PlaneFormats formats) {
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  if (q == nullptr || k == nullptr || v == nullptr || q_rot == nullptr ||
      k_plane == nullptr || v_plane == nullptr || d_pos == nullptr ||
      d_write_status == nullptr || d_read_status == nullptr ||
      d_write_status == d_read_status || plane_bytes == 0U)
    return Turbo4LaunchStatus::kBadArguments;
  if (num_heads == 0U || num_kv_heads == 0U || max_context == 0U ||
      head_dim != 2U * turbo4::kBlockValues || rotary_dim == 0U ||
      rotary_dim > head_dim || (rotary_dim % 2U) != 0U ||
      !Turbo4HeadFitsStaging(head_dim) || !std::isfinite(rope_theta) ||
      rope_theta <= 0.0F || !std::isfinite(eps) || eps < 0.0F)
    return Turbo4LaunchStatus::kGeometryRefused;
  const std::size_t q_values = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_values =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t required_plane = Turbo4LayerPlaneBytes(
      head_dim, num_kv_heads, max_context, formats.stride);
  if (q_values > static_cast<std::size_t>(-1) / sizeof(float) ||
      kv_values > static_cast<std::size_t>(-1) / sizeof(float) ||
      q_bytes < q_values * sizeof(float) ||
      q_rot_bytes < q_values * sizeof(float) ||
      kv_bytes < kv_values * sizeof(float) || required_plane == 0U ||
      plane_bytes < required_plane)
    return Turbo4LaunchStatus::kBadArguments;

  switch (formats.stride) {
    case turbo4::Bits::k4:
      Turbo4DecodePrepareKernel<turbo4::Bits::k4>
          <<<num_heads + num_kv_heads, turbo4::kBlockValues, 0, stream>>>(
              q, k, v, q_norm_weight, k_norm_weight, q_rot, k_plane, v_plane,
              d_pos, plane_bytes, num_heads, num_kv_heads, head_dim, rotary_dim,
              max_context, rope_theta, eps, d_write_status, d_read_status, rope,
              formats);
      break;
    case turbo4::Bits::k5:
      Turbo4DecodePrepareKernel<turbo4::Bits::k5>
          <<<num_heads + num_kv_heads, turbo4::kBlockValues, 0, stream>>>(
              q, k, v, q_norm_weight, k_norm_weight, q_rot, k_plane, v_plane,
              d_pos, plane_bytes, num_heads, num_kv_heads, head_dim, rotary_dim,
              max_context, rope_theta, eps, d_write_status, d_read_status, rope,
              formats);
      break;
    case turbo4::Bits::k6:
      Turbo4DecodePrepareKernel<turbo4::Bits::k6>
          <<<num_heads + num_kv_heads, turbo4::kBlockValues, 0, stream>>>(
              q, k, v, q_norm_weight, k_norm_weight, q_rot, k_plane, v_plane,
              d_pos, plane_bytes, num_heads, num_kv_heads, head_dim, rotary_dim,
              max_context, rope_theta, eps, d_write_status, d_read_status, rope,
              formats);
      break;
  }
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

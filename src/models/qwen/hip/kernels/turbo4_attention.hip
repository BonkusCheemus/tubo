// SCALAR packed-K/V Turbo4 attention reader.
//
//   /opt/rocm/bin/hipcc -fsyntax-only -std=c++20 -DENGINE_ENABLE_HIP -I. -x hip
//   \
//       src/models/qwen/hip/kernels/turbo4_attention.hip
//
// ONE CTA PER QUERY HEAD, ONE SERIAL THREAD PER CTA.
//
// This is the fix for review 0f80ea1b697a44e6bac55e19d835cc18. The previous
// kernel launched 24 threads in ONE CTA against a single `extern __shared__`
// `s_scores` array, so all 24 query heads wrote `s_scores[t]` at the same index
// — a genuine cross-query data race, and one that passes every compile and
// every CPU test. Now `gridDim = num_heads`, `blockDim = 1`, and each CTA's
// shared array belongs to exactly one query head. Private per CTA cannot
// collide, so the partitioning is structural rather than an index expression a
// test can only reason about.
//
// A THIN LAUNCHER. The body is `Turbo4AttentionQueryBody` in
// turbo4_kernels.hpp, annotated host/device, and the CPU fixture calls the SAME
// function — which is what the previous packet failed to do.
//
// WHAT IS NOT PROVEN: gridDim/blockDim choice is compile-only, as is the shared
// memory allocation itself. The body, the index arithmetic and the codec all
// have a CPU oracle. Real-plane qualification stays held and nothing calls
// this.
//
// Both readers take a `turbo4::Bits` (default k4). The scalar reader
// passes it to the shared host/device body. The split reader is a template on
// it, so the Turbo4 instantiation is the pre-Turbo5/6 kernel with the same loads,
// the same 16-entry shared table and the same arithmetic, and Turbo5/6 add only
// the qh read (1 / 2 bytes per lane: a lane owns 8 values, so its qh formats
// are exactly one / two aligned bytes) and a 32 / 64-entry shared table.

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "src/models/qwen/hip/dev_sync.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace gufo::hip {
namespace {

/// Shared score staging, private to ONE CTA, which serves ONE query head.
///
/// NOT A LOCAL LITERAL. This was `= 4096` written here, invisible to the CPU
/// oracle and duplicated by the accounting that sizes the work. It now names
/// the single authority in turbo4_plane_index.hpp, so the bound cannot drift
/// away from the manifest context it exists to serve.
constexpr std::uint32_t kMaxScalarScores = kTurbo4ManifestContext;

__global__ void Turbo4AttentionScalarKernel(
    const float* __restrict__ q, const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, float* __restrict__ out,
    const float* __restrict__ gate, const Turbo4BufferSizes sizes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t valid_context, float scale,
    const Turbo4KernelStatus* __restrict__ d_write_status,
    Turbo4KernelStatus* __restrict__ status, turbo4::PlaneFormats formats) {
  // One CTA per query head. blockDim.x is 1, so this CTA touches exactly one
  // element of `out` and owns this CTA's shared buffer outright.
  extern __shared__ float s_scores[];

  const std::uint32_t query_head = blockIdx.x;
  if (query_head >= num_heads)
    return;
  if (valid_context == 0U || valid_context > kMaxScalarScores) {
    if (status != nullptr)
      *status = Turbo4KernelStatus::kGeometryRefused;
    return;
  }
  // THE WRITER'S WORD, READ ON THE DEVICE. This is a plain load from device
  // memory on the same stream that wrote it, so stream order already guarantees
  // the value is the writer's final one — there is no host synchronisation and
  // no second round trip here. The host never dereferences `d_write_status`
  // either: it only checks the pointer for null, exactly as it does for
  // `d_status`. The reader LATCHES into `status` and NEVER writes
  // `d_write_status`, so the writer's word survives for whoever wants it.
  //
  // NULL IS A WRITER FAILURE, NOT AN ASSUMED SUCCESS. Found in review: the
  // launcher already refuses a null word, but this kernel is reachable without
  // passing through the launcher — the body comment says so explicitly. The
  // decision lives in Turbo4ReadWriterStatus so the CPU oracle can reach it,
  // rather than here where it could only be read.
  const Turbo4KernelStatus writer_status =
      Turbo4ReadWriterStatus(d_write_status);
  // LATCH, never assign: 24 CTAs share this word.
  Turbo4LatchFailure(
      status, Turbo4AttentionQueryBody(
                  q, k_plane, v_plane, out, sizes, writer_status, gate,
                  query_head, num_heads, num_kv_heads, head_dim, valid_context,
                  scale, s_scores, formats));
}

}  // namespace

/// Reference scalar attention over a packed Turbo4 KV plane.
///
/// `q` and `out` are [num_heads, head_dim] device floats, and `q_bytes` /
/// `out_bytes` are their REAL lengths. This is not ceremony: a caller holding
/// one head of a 24-head tensor passes a perfectly non-null `q`, every
/// individual index the body computes is valid, and the scan reads 23 heads
/// past the end of the allocation. `plane_bytes` is the real plane length for
/// the same reason. The scores buffer needs no argument — this function
/// allocates it, so it is the only party that knows how long it is.
///
/// `status` receives a Turbo4KernelStatus on every path, and `out` is left
/// untouched on every refusal.
///
/// Launched as one CTA per query head, one thread each.
Turbo4LaunchStatus LaunchTurbo4AttentionScalar(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, std::size_t plane_bytes, std::size_t q_bytes,
    std::size_t out_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t valid_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats) {
  // Host-side checks only; `d_status` and `d_write_status` are device memory
  // and are never dereferenced here. Same contract as the writer.
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  if (q == nullptr || k_plane == nullptr || v_plane == nullptr ||
      out == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  // THE WRITER'S WORD IS REQUIRED. Making it optional would
  // recreate the exact hole this closes: a caller that forgets the writer gets
  // a reader that attends over a plane nobody wrote and reports kOk. Fail
  // closed.
  if (d_write_status == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  // FAIL-CLOSED, same contract as the writer: the device error destination is
  // required. Without it there is a path where nothing records the outcome and
  // "no status" is indistinguishable from "nothing went wrong".
  if (d_status == nullptr)
    return Turbo4LaunchStatus::kBadArguments;
  if (plane_bytes == 0U)
    return Turbo4LaunchStatus::kBadArguments;
  // GATE ABSENCE IS DECLARED BY BYTES: zero bytes IS "no gate" (see
  // Turbo4GateIsAbsent). A real pointer with zero bytes, or a null pointer with
  // a non-zero size, is a statement the body cannot honour; refuse at the door
  // rather than dispatching 24 CTAs to guess which half was meant.
  if ((gate_bytes == 0U) != (gate == nullptr))
    return Turbo4LaunchStatus::kBadArguments;
  if (num_heads == 0U || num_kv_heads == 0U || head_dim == 0U ||
      valid_context == 0U || valid_context > kMaxScalarScores)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (Turbo4BlocksPerHead(head_dim) == 0U)
    return Turbo4LaunchStatus::kGeometryRefused;
  std::uint32_t probe = 0;
  if (!Turbo4KvHeadForQueryHead(0, num_heads, num_kv_heads, &probe))
    return Turbo4LaunchStatus::kGeometryRefused;

  // The scores length is not a caller claim, it is the shared allocation made
  // three lines below. Assembling it HERE means the struct the device receives
  // describes the same buffer the device will read.
  const std::size_t scores_bytes =
      static_cast<std::size_t>(valid_context) * sizeof(float);
  const Turbo4BufferSizes sizes{plane_bytes, q_bytes, out_bytes, scores_bytes,
                                gate_bytes};
  // Refuse at the door rather than dispatching 24 CTAs that would each
  // independently discover the same short buffer on silicon. The body repeats
  // the check because the kernel is reachable without passing through here.
  if (!Turbo4AttentionBuffersAreSufficient(sizes, num_heads, head_dim,
                                           valid_context))
    return Turbo4LaunchStatus::kBadArguments;
  // Same requirement as the body, and for the same reason: the previous form
  // omitted the kv-head term and let a caller claim one token more than it
  // allocated. Refusing here means 24 CTAs never launch on a plane that cannot
  // hold the context they were told to read (found by measurement in review).
  const std::size_t need_plane = Turbo4LayerPlaneBytes(
      head_dim, num_kv_heads, valid_context, formats.stride);
  if (need_plane == 0U || plane_bytes < need_plane)
    return Turbo4LaunchStatus::kBadArguments;

  // NO RESET HERE — same reason as the writer, and worse here: this launcher
  // runs once per query head GROUP per attention layer, so a per-call reset
  // erased a reader refusal before the next layer could see it (found in
  // review). The caller resets both words once per step; this launcher only
  // latches.
  const std::size_t shared_bytes = scores_bytes;
  Turbo4AttentionScalarKernel<<<num_heads, 1, shared_bytes, stream>>>(
      q, k_plane, v_plane, out, gate, sizes, num_heads, num_kv_heads, head_dim,
      valid_context, scale, d_write_status, d_status, formats);
  // ENQUEUE != EXECUTION. 4096 scores x num_heads is a real shared-memory
  // request that a device can refuse; report that rather than kLaunched.
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

namespace {

constexpr std::uint32_t kSplitHeadDim = 256;
constexpr std::uint32_t kSplitQHeadsPerKv = 6;
constexpr std::uint32_t kSplitWarpLanes = 32;
constexpr std::uint32_t kSplitWarps = 8;
constexpr std::uint32_t kSplitElementsPerLane = kSplitHeadDim / kSplitWarpLanes;
constexpr std::size_t kSplitRecordWidth = kSplitHeadDim + 2;
// The epilogue merge no longer stages every warp's full numerator
// (that was kSplitWarps * kSplitQHeadsPerKv * kSplitRecordWidth floats, 49.5 KB
// of LDS, which capped the kernel at 2 CTAs per WGP). Each warp keeps only its
// (maximum, denominator) pair for all heads, and the numerators go through LDS
// kSplitMergeQPerPass heads at a time. Per output value the merge arithmetic is
// unchanged: the same eight products added in the same warp order.
constexpr std::uint32_t kSplitMergeQPerPass = 2;
static_assert(kSplitQHeadsPerKv % kSplitMergeQPerPass == 0,
              "merge passes must tile the query heads");
constexpr std::size_t kSplitWarpStatValues =
    kSplitWarps * kSplitQHeadsPerKv * 2;
constexpr std::size_t kSplitMergePassValues =
    kSplitWarps * kSplitMergeQPerPass * kSplitHeadDim;

__global__ void Turbo4RotateQueryKernel(
    const float* q, float* q_rot, std::uint32_t num_heads,
    std::uint32_t head_dim, std::size_t q_bytes, std::size_t q_rot_bytes,
    Turbo4KernelStatus* status, std::size_t row_stride = 0) {
  __shared__ float s_q[turbo4::kBlockValues];
  __shared__ unsigned int s_error;
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  if (head >= num_heads)
    return;
  // blockIdx.y is the verify row; row_stride is 0 for a single launch.
  q += static_cast<std::size_t>(blockIdx.y) * row_stride;
  q_rot += static_cast<std::size_t>(blockIdx.y) * row_stride;
  if (head_dim != kSplitHeadDim ||
      q_bytes <
          static_cast<std::size_t>(num_heads) * head_dim * sizeof(float) ||
      q_rot_bytes <
          static_cast<std::size_t>(num_heads) * head_dim * sizeof(float)) {
    if (tid == 0)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  if (status == nullptr)
    return;
  if (tid == 0)
    s_error = static_cast<unsigned int>(*status);
  __syncthreads();
  if (s_error != 0U)
    return;

  const std::size_t head_base = static_cast<std::size_t>(head) * head_dim;
  for (std::uint32_t offset = 0; offset < head_dim;
       offset += turbo4::kBlockValues) {
    if (tid == 0)
      s_error = 0;
    __syncthreads();
    const std::size_t index = head_base + offset + tid;
    const float value = q[index];
    if (!Turbo4IsFinite(value))
      atomicMax(&s_error, static_cast<unsigned int>(
                              Turbo4KernelStatus::kGeometryRefused));
    s_q[tid] = value;
    __syncthreads();
    if (s_error != 0U) {
      if (tid == 0)
        Turbo4LatchFailure(status, static_cast<Turbo4KernelStatus>(s_error));
      return;
    }

    s_q[tid] *= turbo4::TURBO_WHT_SIGNS1[tid];
    __syncthreads();
    for (std::uint32_t stride = 1; stride < turbo4::kBlockValues; stride *= 2) {
      const std::uint32_t group = tid / (2 * stride);
      const std::uint32_t within = tid % (2 * stride);
      if (within < stride) {
        const std::uint32_t a = group * 2 * stride + within;
        const std::uint32_t b = a + stride;
        const float av = s_q[a];
        const float bv = s_q[b];
        s_q[a] = av + bv;
        s_q[b] = av - bv;
      }
      __syncthreads();
    }
    s_q[tid] *= turbo4::kInvSqrt128 * turbo4::TURBO_WHT_SIGNS2[tid];
    q_rot[index] = s_q[tid];
    __syncthreads();
  }
}

/// Explicit 2-byte-aligned load. Blocks are 66 / 82 / 98 B, so every
/// block starts on an even byte (plane bases and layer plane sizes are even)
/// and every field read below sits at an even offset inside it. The old
/// byte-wise std::memcpy told the compiler nothing about that alignment.
__device__ __forceinline__ std::uint16_t Turbo4LoadU16(const std::uint8_t* p) {
  std::uint16_t v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), sizeof(v));
  return v;
}

/// Decode the eight values one lane owns from one stored block (Turbo4/5/6).
///
/// `slot_start` is the lane's first value inside the 128-value block, always a
/// multiple of 8, which is what makes every plane read below aligned and
/// whole-byte: 4 bytes of nibbles, then for Turbo5 one qh byte (8 sign bits) or
/// for Turbo6 two qh bytes (8 x 2 high bits). `centroids` is the shared table
/// of 16 / 32 / 64 entries indexed by CODE; for Turbo5 the code is rebuilt from
/// the sign-magnitude pair exactly as turbo4::CodeAt does.
template<turbo4::Bits kBits>
__device__ __forceinline__ void Turbo4DecodeLane8(const std::uint8_t* block,
                                                  std::size_t slot_start,
                                                  float norm,
                                                  const float* centroids,
                                                  float* out) {
  // Four nibble bytes at an even offset: two u16 loads, little-endian join
  // (the same value the old 4-byte memcpy produced).
  const std::uint8_t* nibbles = block + turbo4::kNormBytes + slot_start / 2;
  const std::uint32_t packed =
      static_cast<std::uint32_t>(Turbo4LoadU16(nibbles)) |
      (static_cast<std::uint32_t>(Turbo4LoadU16(nibbles + 2)) << 16U);
  if constexpr (kBits == turbo4::Bits::k5) {
    const std::uint32_t signs = block[turbo4::kQhOffset + slot_start / 8];
    for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e) {
      const std::uint32_t mag = (packed >> (4U * e)) & 0xFU;
      const std::uint32_t neg = (signs >> e) & 1U;
      const float c = neg != 0U ? -centroids[mag] : centroids[mag];
      out[e] = c * norm;
    }
  } else if constexpr (kBits == turbo4::Bits::k6) {
    const std::uint16_t high =
        Turbo4LoadU16(block + turbo4::kQhOffset + slot_start / 4);
    for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e) {
      const std::uint32_t lo = (packed >> (4U * e)) & 0xFU;
      const std::uint32_t hi =
          (static_cast<std::uint32_t>(high) >> (2U * e)) & 3U;
      out[e] = centroids[lo | (hi << 4U)] * norm;
    }
  } else {
    for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
      out[e] = centroids[(packed >> (4U * e)) & 0xFU] * norm;
  }
}

__device__ __forceinline__ void Turbo4DecodePlaneLane8(
    turbo4::Bits bits, const std::uint8_t* block, std::size_t slot_start,
    float norm, const float* centroids, float* out) {
  if (bits == turbo4::Bits::k5)
    Turbo4DecodeLane8<turbo4::Bits::k5>(block, slot_start, norm, centroids,
                                        out);
  else if (bits == turbo4::Bits::k6)
    Turbo4DecodeLane8<turbo4::Bits::k6>(block, slot_start, norm, centroids,
                                        out);
  else
    Turbo4DecodeLane8<turbo4::Bits::k4>(block, slot_start, norm, centroids,
                                        out);
}

/// One CTA per (KV head, context split). Each of its eight 32-lane warps owns
/// one token at a time, then advances by eight positions. That lets the warp
/// reduce six QK dots with shuffles while each K/V element is loaded once and
/// reused across all six GQA query heads.
// kVSkipTinyV is a COMPILE-TIME switch, not a runtime one. The false
// instantiation is byte-for-byte the pre-GUFO_DEV_V_SKIP kernel, so leaving the
// env unset pays nothing. The true instantiation defers the V dequant behind
// the softmax and drops it when every query head's weight is below
// v_skip_threshold; the decision is warp-uniform because all lanes of a warp
// share one key and read the same warp-reduced score.
template<turbo4::Bits kBits, bool kVSkipTinyV = false>
__global__ void Turbo4AttentionSplitKernel(
    const float* q_rot, const std::uint8_t* k_plane,
    const std::uint8_t* v_plane, float* partials, std::size_t plane_bytes,
    std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t valid_context,
    std::uint32_t max_context, std::uint32_t split_count,
    std::uint32_t max_splits, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    turbo4::PlaneFormats formats, std::size_t q_rot_row_stride = 0,
    std::size_t partials_row_stride = 0, float v_skip_threshold = 0.0f) {
  (void)v_skip_threshold;  // read only by the kVSkipTinyV instantiation
  __shared__ unsigned int s_split_error;
  __shared__ float s_warp_stats[kSplitWarpStatValues];
  __shared__ float s_merge_numerators[kSplitMergePassValues];
  __shared__ float s_merge_weights[kSplitWarps * kSplitQHeadsPerKv];
  __shared__ float s_merge_maxima[kSplitQHeadsPerKv];
  __shared__ float s_merge_denominators[kSplitQHeadsPerKv];
  constexpr std::size_t kBlockBytes = turbo4::BlockBytes(kBits);
  constexpr std::uint32_t kCodes = 1U << static_cast<std::uint32_t>(kBits);
  __shared__ float s_centroids[kCodes];
  __shared__ float s_v_centroids[kBits == turbo4::Bits::k4 ? 1 : kCodes];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t warp = tid / kSplitWarpLanes;
  const std::uint32_t lane = tid % kSplitWarpLanes;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  // blockIdx.z is the verify row. Row r attends over valid_context + r
  // positions with its own split plan; a single launch has gridDim.z == 1, so
  // row == 0 and this is the old per-row kernel. split_count is the grid
  // extent (max over rows); blocks past this row's own count have no work.
  const std::uint32_t row = blockIdx.z;
  const std::uint32_t row_context = valid_context + row;
  const std::uint32_t row_splits =
      static_cast<std::uint32_t>(Turbo4DecodeSplitCount(row_context));
  if (row_splits != 0U && split >= row_splits && split < split_count)
    return;
  const std::size_t record_width = kSplitRecordWidth;
  const std::size_t partial_slots = max_splits;
  const std::size_t required_bytes =
      partial_slots * num_heads * record_width * sizeof(float);
  if (!formats.Valid() || formats.stride != kBits ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv ||
      kv_head >= num_kv_heads || row_context == 0 ||
      row_context > max_context || split_count == 0 ||
      split_count > max_splits || split >= split_count ||
      plane_bytes < Turbo4LayerPlaneBytes(kSplitHeadDim, num_kv_heads,
                                          max_context, kBits) ||
      partial_bytes < required_bytes || q_rot == nullptr ||
      partials == nullptr || status == nullptr || write_status == nullptr) {
    if (tid == 0)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  q_rot += static_cast<std::size_t>(row) * q_rot_row_stride;
  partials += static_cast<std::size_t>(row) * partials_row_stride;
  if (tid == 0) {
    s_split_error = static_cast<unsigned int>(*status);
    if (*write_status != Turbo4KernelStatus::kOk) {
      s_split_error =
          static_cast<unsigned int>(Turbo4KernelStatus::kWriterFailed);
      Turbo4LatchFailure(status, Turbo4KernelStatus::kWriterFailed);
    }
  }
  __syncthreads();
  if (s_split_error != 0U)
    return;
  // kCodes <= 64 <= blockDim.x (256), so one pass fills the table.
  if constexpr (kBits == turbo4::Bits::k4) {
    if (tid < 16U) s_centroids[tid] = turbo4::TURBO_CENTROIDS_4BIT[tid];
  } else {
    const unsigned k_codes = formats.k == turbo4::Bits::k5
                                 ? 16U
                                 : 1U << static_cast<unsigned>(formats.k);
    const unsigned v_codes = formats.v == turbo4::Bits::k5
                                 ? 16U
                                 : 1U << static_cast<unsigned>(formats.v);
    if (tid < k_codes)
      s_centroids[tid] = turbo4::CentroidValue(
          formats.k, tid + (formats.k == turbo4::Bits::k5 ? 16U : 0U));
    if (formats.v != formats.k && tid < v_codes)
      s_v_centroids[tid] = turbo4::CentroidValue(
          formats.v, tid + (formats.v == turbo4::Bits::k5 ? 16U : 0U));
  }
  __syncthreads();

  float maxima[kSplitQHeadsPerKv];
  float denominators[kSplitQHeadsPerKv];
  float numerators[kSplitQHeadsPerKv][kSplitElementsPerLane];
  for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q) {
    maxima[q] = -3.402823466e+38f;
    denominators[q] = 0.0f;
    for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
      numerators[q][e] = 0.0f;
  }

  const std::uint32_t split_size =
      static_cast<std::uint32_t>(Turbo4DecodeSplitSize(row_context));
  const std::uint32_t begin = split * split_size;
  const std::uint32_t split_end = begin + split_size;
  const std::uint32_t end = row_context < split_end ? row_context : split_end;
  std::uint32_t local_error = 0;

  // q_rot for this CTA's six query heads, read once. It is constant
  // across tokens; the loop used to re-read all 48 floats per lane per token.
  float q_regs[kSplitQHeadsPerKv][kSplitElementsPerLane];
  for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q) {
    const std::uint32_t query_head = kv_head * kSplitQHeadsPerKv + q;
    const std::size_t q_base =
        static_cast<std::size_t>(query_head) * kSplitHeadDim +
        lane * kSplitElementsPerLane;
    for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
      q_regs[q][e] = q_rot[q_base + e];
  }

  // Validate the token addressing once per CTA instead of per token.
  // Turbo4BlockByteOffset is monotonic in `token` (every refusal is a bound or
  // an overflow that only gets worse as token grows; head and geometry are
  // fixed and were validated above), so the split's LAST token passing proves
  // every earlier token passes. Then the byte offset is
  //   token * bytes_per_token + head_byte
  // in exact integer arithmetic (no overflow, by the last-token check), which
  // is stepped by kSplitWarps tokens per iteration. If the last token fails,
  // the loop falls back to the original per-token call so the per-token refusal
  // and error reporting stay exactly as before.
  const std::size_t bytes_per_token =
      Turbo4BlocksPerToken(kSplitHeadDim, num_kv_heads) * kBlockBytes;
  const std::size_t head_byte =
      static_cast<std::size_t>(kv_head) * Turbo4BlocksPerHead(kSplitHeadDim) *
      kBlockBytes;
  std::size_t last_byte = 0;
  const bool offsets_ok =
      end <= begin ||
      Turbo4BlockByteOffset(end - 1U, kv_head, kSplitHeadDim, num_kv_heads,
                            max_context, &last_byte, kBits);
  // Unsigned wrap past the last token is harmless: it is never read.
  std::size_t token_byte =
      static_cast<std::size_t>(begin + warp) * bytes_per_token + head_byte;
  for (std::uint32_t token = begin + warp; token < end;
       token += kSplitWarps, token_byte += kSplitWarps * bytes_per_token) {
    std::size_t base_byte = token_byte;
    if (!offsets_ok &&
        !Turbo4BlockByteOffset(token, kv_head, kSplitHeadDim, num_kv_heads,
                               max_context, &base_byte, kBits)) {
      local_error =
          static_cast<unsigned int>(Turbo4KernelStatus::kGeometryRefused);
      continue;
    }
    const std::uint32_t first_d = lane * kSplitElementsPerLane;
    const std::size_t block_index = first_d / turbo4::kBlockValues;
    const std::size_t slot_start = first_d % turbo4::kBlockValues;
    const std::size_t block_byte = base_byte + block_index * kBlockBytes;
    const std::uint8_t* k_block = k_plane + block_byte;
    const std::uint8_t* v_block = v_plane + block_byte;
    float k_norm = 0.0f;
    float v_norm = 0.0f;
    if ((lane & 15U) == 0U) {
      const std::uint16_t k_norm_bits = Turbo4LoadU16(k_block);
      const std::uint16_t v_norm_bits = Turbo4LoadU16(v_block);
      k_norm = turbo4::HalfBitsToFloat(k_norm_bits);
      v_norm = turbo4::HalfBitsToFloat(v_norm_bits);
    }
    k_norm = __shfl(k_norm, lane & ~15U, kSplitWarpLanes);
    v_norm = __shfl(v_norm, lane & ~15U, kSplitWarpLanes);
    if (!turbo4::StoredNormIsUsable(k_norm) ||
        !turbo4::StoredNormIsUsable(v_norm)) {
      local_error =
          static_cast<unsigned int>(Turbo4KernelStatus::kPoisonedNorm);
      k_norm = 0.0f;
      v_norm = 0.0f;
    }
    float kval[kSplitElementsPerLane];
    float vval[kSplitElementsPerLane];
    if constexpr (kBits == turbo4::Bits::k4) {
      Turbo4DecodeLane8<turbo4::Bits::k4>(k_block, slot_start, k_norm, s_centroids, kval);
      if constexpr (!kVSkipTinyV)
        Turbo4DecodeLane8<turbo4::Bits::k4>(v_block, slot_start, v_norm, s_centroids, vval);
    } else {
      Turbo4DecodePlaneLane8(formats.k, k_block, slot_start, k_norm, s_centroids, kval);
      if constexpr (!kVSkipTinyV)
        Turbo4DecodePlaneLane8(formats.v, v_block, slot_start, v_norm,
            formats.v == formats.k ? s_centroids : s_v_centroids, vval);
    }
    if constexpr (!kVSkipTinyV) {
      for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
        if (!Turbo4IsFinite(kval[e]) || !Turbo4IsFinite(vval[e]))
          local_error =
              static_cast<unsigned int>(Turbo4KernelStatus::kPoisonedNorm);
    } else {
      // V is not decoded yet, so only K is checked here; the V check runs in
      // the block that decodes it.
      for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
        if (!Turbo4IsFinite(kval[e]))
          local_error =
              static_cast<unsigned int>(Turbo4KernelStatus::kPoisonedNorm);
    }
    float dots[kSplitQHeadsPerKv] = {};
    for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q)
      for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
        dots[q] += q_regs[q][e] * kval[e];
    for (int offset = static_cast<int>(kSplitWarpLanes / 2); offset > 0;
         offset /= 2)
      for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q)
        dots[q] += __shfl_down(dots[q], offset, kSplitWarpLanes);
    if constexpr (kVSkipTinyV) {
      // Two passes: the softmax merge (identical arithmetic to the base path)
      // decides per head whether the key is worth its V row, then the V row is
      // decoded once if any head wants it and used by those heads.
      float left_scales[kSplitQHeadsPerKv];
      float right_scales[kSplitQHeadsPerKv];
      bool keep_any = false;
      for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q) {
        const float score = __shfl(dots[q], 0, kSplitWarpLanes) * scale;
        left_scales[q] = 0.0f;
        right_scales[q] = 0.0f;
        if (!Turbo4IsFinite(score)) {
          local_error =
              static_cast<unsigned int>(Turbo4KernelStatus::kGeometryRefused);
          continue;
        }
        const auto merged =
            Turbo4MergeSoftmax(maxima[q], denominators[q], score, 1.0f);
        left_scales[q] = merged.left_scale;
        right_scales[q] = merged.right_scale;
        // right_scale = exp(score - maximum) is this key's weight relative to
        // the new running maximum. Below 1 the maximum did not move, so a
        // skipped key leaves the running (m, l) untouched and is dropped whole
        // from that head's normaliser and numerator.
        if (merged.right_scale >= v_skip_threshold) {
          maxima[q] = merged.maximum;
          denominators[q] = merged.denominator;
          keep_any = true;
        }
      }
      if (keep_any) {
        if constexpr (kBits == turbo4::Bits::k4) {
          Turbo4DecodeLane8<turbo4::Bits::k4>(v_block, slot_start, v_norm,
                                              s_centroids, vval);
        } else {
          Turbo4DecodePlaneLane8(
              formats.v, v_block, slot_start, v_norm,
              formats.v == formats.k ? s_centroids : s_v_centroids, vval);
        }
        for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
          if (!Turbo4IsFinite(vval[e]))
            local_error =
                static_cast<unsigned int>(Turbo4KernelStatus::kPoisonedNorm);
      }
      for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q) {
        if (right_scales[q] < v_skip_threshold)
          continue;
        for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
          numerators[q][e] = fmaf(numerators[q][e], left_scales[q],
                                  vval[e] * right_scales[q]);
      }
    } else {
      for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q) {
        const float score = __shfl(dots[q], 0, kSplitWarpLanes) * scale;
        if (!Turbo4IsFinite(score)) {
          local_error =
              static_cast<unsigned int>(Turbo4KernelStatus::kGeometryRefused);
          continue;
        }
        const auto merged =
            Turbo4MergeSoftmax(maxima[q], denominators[q], score, 1.0f);
        maxima[q] = merged.maximum;
        denominators[q] = merged.denominator;
        // Pin the shipped contraction: round V * right, then fuse N * left.
        // An equivalent two-product expression lets codegen swap the rounding.
        for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
          numerators[q][e] = fmaf(numerators[q][e], merged.left_scale,
                                  vval[e] * merged.right_scale);
      }
    }
  }

  // Only the (maximum, denominator) pair is staged here; the numerators go
  // through s_merge_numerators in passes below.
  if (lane == 0) {
    for (std::uint32_t q = 0; q < kSplitQHeadsPerKv; ++q) {
      const std::size_t stat_row =
          (static_cast<std::size_t>(warp) * kSplitQHeadsPerKv + q) * 2U;
      s_warp_stats[stat_row] = maxima[q];
      s_warp_stats[stat_row + 1] = denominators[q];
    }
  }
  if (local_error != 0U)
    atomicMax(&s_split_error, local_error);
  // Review: one thread folds the global status in BEFORE the barrier,
  // so every wave takes the same return below; the merge passes after it have
  // barriers, and a return some waves skip would leave them reading slots the
  // returned warps never wrote.
  if (tid == 0 && *status != Turbo4KernelStatus::kOk)
    atomicMax(&s_split_error, static_cast<unsigned int>(*status));
  __syncthreads();
  if (s_split_error != 0U) {
    if (tid == 0)
      Turbo4LatchFailure(status,
                         static_cast<Turbo4KernelStatus>(s_split_error));
    return;
  }

  // Merge the eight warp-local online-softmax states before they leave this
  // CTA. One lane per query head computes the eight scales and denominator.
  if (warp == 0 && lane < kSplitQHeadsPerKv) {
    const std::uint32_t q = lane;
    float maximum = -3.402823466e+38f;
    bool any = false;
    for (std::uint32_t w = 0; w < kSplitWarps; ++w) {
      const std::size_t row =
          (static_cast<std::size_t>(w) * kSplitQHeadsPerKv + q) * 2U;
      const float sum = s_warp_stats[row + 1];
      if (!Turbo4IsFinite(sum) || sum < 0.0f) {
        atomicMax(&s_split_error, static_cast<unsigned int>(
                                      Turbo4KernelStatus::kDegenerateSoftmax));
      } else if (sum > 0.0f) {
        const float m = s_warp_stats[row];
        if (!Turbo4IsFinite(m)) {
          atomicMax(&s_split_error,
                    static_cast<unsigned int>(
                        Turbo4KernelStatus::kDegenerateSoftmax));
        } else {
          maximum = fmaxf(maximum, m);
          any = true;
        }
      }
    }
    if (!any) {
      atomicMax(&s_split_error, static_cast<unsigned int>(
                                    Turbo4KernelStatus::kDegenerateSoftmax));
    } else {
      float denominator = 0.0f;
      for (std::uint32_t w = 0; w < kSplitWarps; ++w) {
        const std::size_t row =
            (static_cast<std::size_t>(w) * kSplitQHeadsPerKv + q) * 2U;
        const float sum = s_warp_stats[row + 1];
        const float weight =
            sum > 0.0f ? expf(s_warp_stats[row] - maximum) : 0.0f;
        s_merge_weights[w * kSplitQHeadsPerKv + q] = weight;
        denominator += weight * sum;
      }
      s_merge_maxima[q] = maximum;
      s_merge_denominators[q] = denominator;
      if (!Turbo4IsFinite(denominator) || denominator <= 0.0f)
        atomicMax(&s_split_error, static_cast<unsigned int>(
                                      Turbo4KernelStatus::kDegenerateSoftmax));
    }
  }
  __syncthreads();
  if (s_split_error != 0U) {
    if (tid == 0)
      Turbo4LatchFailure(status,
                         static_cast<Turbo4KernelStatus>(s_split_error));
    return;
  }

  // Numerator merge, kSplitMergeQPerPass query heads per pass. Thread `tid`
  // owns output dimension tid; each warp stores its lane*8+e values, the CTA
  // syncs, and thread tid sums the eight warps in order w = 0..7 exactly as the
  // all-heads-resident version did. The second barrier keeps the next pass from
  // overwriting a buffer slot that is still being read. The loops are unrolled
  // so numerators[q][*] keeps compile-time indices (registers, no scratch).
#pragma unroll
  for (std::uint32_t q0 = 0; q0 < kSplitQHeadsPerKv; q0 += kSplitMergeQPerPass) {
#pragma unroll
    for (std::uint32_t qi = 0; qi < kSplitMergeQPerPass; ++qi) {
      float* slot = s_merge_numerators +
                    (static_cast<std::size_t>(warp) * kSplitMergeQPerPass + qi) *
                        kSplitHeadDim +
                    lane * kSplitElementsPerLane;
#pragma unroll
      for (std::uint32_t e = 0; e < kSplitElementsPerLane; ++e)
        slot[e] = numerators[q0 + qi][e];
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t qi = 0; qi < kSplitMergeQPerPass; ++qi) {
      const std::uint32_t q = q0 + qi;
      const std::uint32_t query_head = kv_head * kSplitQHeadsPerKv + q;
      float* row =
          partials +
          (static_cast<std::size_t>(split) * num_heads + query_head) *
              record_width;
      float value = 0.0f;
      for (std::uint32_t w = 0; w < kSplitWarps; ++w) {
        value += s_merge_weights[w * kSplitQHeadsPerKv + q] *
                 s_merge_numerators[(static_cast<std::size_t>(w) *
                                         kSplitMergeQPerPass +
                                     qi) *
                                        kSplitHeadDim +
                                    tid];
      }
      if (tid == 0) {
        row[0] = s_merge_maxima[q];
        row[1] = s_merge_denominators[q];
      }
      row[2 + tid] = value;
    }
    if (q0 + kSplitMergeQPerPass < kSplitQHeadsPerKv)
      __syncthreads();
  }
}

__global__ void Turbo4AttentionReduceInverseKernel(
    const float* partials, float* out_rot, float* out, const float* gate,
    std::size_t partial_bytes, std::size_t out_rot_bytes, std::size_t out_bytes,
    std::size_t gate_bytes, std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t split_count, std::uint32_t max_splits, std::uint32_t head_dim,
    Turbo4KernelStatus* status, std::uint32_t valid_context = 0,
    std::size_t partials_row_stride = 0, std::size_t vector_row_stride = 0,
    bool uniform_splits = false) {
  __shared__ float s_output[kSplitHeadDim];
  __shared__ float s_tree[kSplitHeadDim];
  __shared__ float s_denominator;
  __shared__ unsigned int s_error;
  extern __shared__ float s_weights[];
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::size_t record_width = kSplitRecordWidth;
  // blockIdx.y is the verify row. valid_context is only read when
  // vector_row_stride != 0 (batched); the single launch keeps split_count.
  // uniform_splits = every row shares one split plan (the WMMA verify
  // kernel pads rows that end early with empty m=-FLT_MAX, l=0 records).
  const std::uint32_t row = blockIdx.y;
  const std::size_t partial_count =
      (vector_row_stride == 0U || uniform_splits)
          ? split_count
          : Turbo4DecodeSplitCount(valid_context + row);
  const std::size_t max_partial_count = max_splits;
  const std::size_t required_partial =
      max_partial_count * num_heads * record_width * sizeof(float);
  const std::size_t required_output =
      static_cast<std::size_t>(num_heads) * head_dim * sizeof(float);
  if (head >= num_heads || head_dim != kSplitHeadDim ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv || split_count == 0 ||
      split_count > max_splits || max_splits > kSplitHeadDim ||
      partial_count == 0 || partial_count > split_count ||
      partials == nullptr || out_rot == nullptr || out == nullptr ||
      status == nullptr || partial_bytes < required_partial ||
      out_rot_bytes < required_output || out_bytes < required_output ||
      ((gate_bytes == 0) != (gate == nullptr)) ||
      (gate != nullptr && gate_bytes < required_output)) {
    if (tid == 0)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  partials += static_cast<std::size_t>(row) * partials_row_stride;
  out_rot += static_cast<std::size_t>(row) * vector_row_stride;
  out += static_cast<std::size_t>(row) * vector_row_stride;
  if (gate != nullptr)
    gate += static_cast<std::size_t>(row) * vector_row_stride;
  if (*status != Turbo4KernelStatus::kOk)
    return;

  if (tid == 0)
    s_error = 0;
  __syncthreads();
  const float* first_row = nullptr;
  float local_maximum = -3.402823466e+38f;
  if (tid < partial_count) {
    first_row = partials + (static_cast<std::size_t>(tid) * num_heads + head) *
                               record_width;
    if (!Turbo4IsFinite(first_row[0]) || !Turbo4IsFinite(first_row[1]) ||
        first_row[1] < 0.0f) {
      atomicMax(&s_error, static_cast<unsigned int>(
                              Turbo4KernelStatus::kDegenerateSoftmax));
    } else {
      local_maximum = first_row[0];
    }
  }
  s_tree[tid] = local_maximum;
  __syncthreads();
  if (s_error != 0U) {
    if (tid == 0)
      Turbo4LatchFailure(status, static_cast<Turbo4KernelStatus>(s_error));
    return;
  }

  for (std::uint32_t stride = kSplitHeadDim / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      s_tree[tid] = fmaxf(s_tree[tid], s_tree[tid + stride]);
    __syncthreads();
  }
  const float maximum = s_tree[0];
  if (tid < partial_count) {
    const float* row =
        partials +
        (static_cast<std::size_t>(tid) * num_heads + head) * record_width;
    s_weights[tid] = expf(row[0] - maximum);
    if (!Turbo4IsFinite(s_weights[tid]))
      atomicMax(&s_error, static_cast<unsigned int>(
                              Turbo4KernelStatus::kDegenerateSoftmax));
  } else if (tid < max_splits) {
    s_weights[tid] = 0.0f;
  }
  __syncthreads();
  if (s_error != 0U) {
    if (tid == 0)
      Turbo4LatchFailure(status, static_cast<Turbo4KernelStatus>(s_error));
    return;
  }

  if (tid == 0) {
    float denominator = 0.0f;
    for (std::size_t slot = 0; slot < partial_count; ++slot) {
      const float* row = partials + (slot * num_heads + head) * record_width;
      denominator += s_weights[slot] * row[1];
    }
    s_denominator = denominator;
    if (!Turbo4IsFinite(denominator) || denominator <= 0.0f)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kDegenerateSoftmax);
  }
  __syncthreads();
  if (*status != Turbo4KernelStatus::kOk)
    return;

  float numerator = 0.0f;
  for (std::size_t slot = 0; slot < partial_count; ++slot) {
    const float* row = partials + (slot * num_heads + head) * record_width;
    numerator += s_weights[slot] * row[2 + tid];
  }
  s_output[tid] = numerator / s_denominator;
  if (!Turbo4IsFinite(s_output[tid]))
    Turbo4LatchFailure(status, Turbo4KernelStatus::kDegenerateSoftmax);
  __syncthreads();
  if (*status != Turbo4KernelStatus::kOk)
    return;

  const std::size_t output_base = static_cast<std::size_t>(head) * head_dim;
  out_rot[output_base + tid] = s_output[tid];
  __syncthreads();

  // One thread per output value transforms both independent 128-value blocks.
  const std::uint32_t block = tid / turbo4::kBlockValues;
  const std::uint32_t element = tid % turbo4::kBlockValues;
  const std::uint32_t base = block * turbo4::kBlockValues;
  s_output[tid] *= turbo4::TURBO_WHT_SIGNS2[element];
  __syncthreads();
  for (std::uint32_t stride = 1; stride < turbo4::kBlockValues; stride *= 2) {
    const std::uint32_t group = element / (2 * stride);
    const std::uint32_t within = element % (2 * stride);
    if (within < stride) {
      const std::uint32_t a = base + group * 2 * stride + within;
      const std::uint32_t b = a + stride;
      const float av = s_output[a];
      const float bv = s_output[b];
      s_output[a] = av + bv;
      s_output[b] = av - bv;
    }
    __syncthreads();
  }
  s_output[tid] *= turbo4::kInvSqrt128 * turbo4::TURBO_WHT_SIGNS1[element];
  __syncthreads();

  if (gate != nullptr && !Turbo4IsFinite(gate[output_base + tid]))
    Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
  __syncthreads();
  if (*status != Turbo4KernelStatus::kOk)
    return;
  out[output_base + tid] =
      gate == nullptr
          ? s_output[tid]
          : s_output[tid] * Turbo4SigmoidGate(gate[output_base + tid]);
}

}  // namespace

namespace {

/// OPT-GUFO_DEV_V_SKIP: pick the split-K reader instantiation. `v_skip` false
/// launches the unchanged kernel; true launches the V-skipping twin with the
/// threshold. One place so the single and batched launchers cannot drift.
template<turbo4::Bits kBits>
void LaunchTurbo4AttentionSplitStage(
    bool v_skip, float v_skip_threshold, const float* q_rot_input,
    const std::uint8_t* k_plane, const std::uint8_t* v_plane, float* partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t valid_context,
    std::uint32_t max_context, std::uint32_t split_count,
    std::uint32_t max_splits, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    turbo4::PlaneFormats formats, dim3 grid, hipStream_t stream,
    std::size_t q_rot_row_stride, std::size_t partials_row_stride) {
  if (v_skip) {
    Turbo4AttentionSplitKernel<kBits, true><<<grid, kSplitHeadDim, 0, stream>>>(
        q_rot_input, k_plane, v_plane, partials, plane_bytes, partial_bytes,
        num_heads, num_kv_heads, valid_context, max_context, split_count,
        max_splits, scale, d_write_status, d_status, formats, q_rot_row_stride,
        partials_row_stride, v_skip_threshold);
  } else {
    Turbo4AttentionSplitKernel<kBits, false>
        <<<grid, kSplitHeadDim, 0, stream>>>(
            q_rot_input, k_plane, v_plane, partials, plane_bytes, partial_bytes,
            num_heads, num_kv_heads, valid_context, max_context, split_count,
            max_splits, scale, d_write_status, d_status, formats,
            q_rot_row_stride, partials_row_stride, 0.0f);
  }
}

}  // namespace

Turbo4LaunchStatus LaunchTurbo4AttentionSplit(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t valid_context, std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, bool query_is_rotated, turbo4::PlaneFormats formats) {
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  if (q == nullptr || k_plane == nullptr || v_plane == nullptr ||
      out == nullptr || q_rot == nullptr || out_rot == nullptr ||
      partials == nullptr || d_write_status == nullptr || d_status == nullptr ||
      plane_bytes == 0 || num_heads == 0 || num_kv_heads == 0)
    return Turbo4LaunchStatus::kBadArguments;
  if (max_context == 0 || valid_context == 0 || valid_context > max_context)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (head_dim != kSplitHeadDim ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv || !std::isfinite(scale))
    return Turbo4LaunchStatus::kGeometryRefused;
  if ((gate_bytes == 0U) != (gate == nullptr))
    return Turbo4LaunchStatus::kBadArguments;

  const std::size_t vector_bytes =
      static_cast<std::size_t>(num_heads) * head_dim * sizeof(float);
  const std::size_t required_plane = Turbo4LayerPlaneBytes(
      head_dim, num_kv_heads, max_context, formats.stride);
  const std::size_t required_partials =
      Turbo4DecodePartialBytes(max_context, num_heads, num_kv_heads, head_dim);
  if (required_plane == 0U || plane_bytes < required_plane ||
      required_partials == 0U || partial_bytes < required_partials ||
      q_bytes < vector_bytes || out_bytes < vector_bytes ||
      q_rot_bytes < vector_bytes || out_rot_bytes < vector_bytes ||
      (gate != nullptr && gate_bytes < vector_bytes))
    return Turbo4LaunchStatus::kBadArguments;

  const std::uint32_t split_count =
      static_cast<std::uint32_t>(Turbo4DecodeSplitCount(valid_context));
  const std::uint32_t max_splits =
      static_cast<std::uint32_t>(Turbo4DecodeMaxSplitCount(max_context));
  if (split_count == 0 || max_splits < split_count ||
      max_splits > kSplitHeadDim)
    return Turbo4LaunchStatus::kGeometryRefused;

  const float* q_rot_input = q_rot;
  if (query_is_rotated) {
    q_rot_input = q;
  } else {
    Turbo4RotateQueryKernel<<<num_heads, turbo4::kBlockValues, 0, stream>>>(
        q, q_rot, num_heads, head_dim, q_bytes, q_rot_bytes, d_status);
    if (hipPeekAtLastError() != hipSuccess)
      return Turbo4LaunchStatus::kEnqueueFailed;
  }

  const dim3 split_grid(num_kv_heads, split_count);
  const float v_skip_threshold = GufoDevVSkipThreshold();
  const bool v_skip = v_skip_threshold > 0.0f;
  switch (formats.stride) {
    case turbo4::Bits::k5:
      LaunchTurbo4AttentionSplitStage<turbo4::Bits::k5>(
          v_skip, v_skip_threshold, q_rot_input, k_plane, v_plane, partials,
          plane_bytes, partial_bytes, num_heads, num_kv_heads, valid_context,
          max_context, split_count, max_splits, scale, d_write_status, d_status,
          formats, split_grid, stream, 0, 0);
      break;
    case turbo4::Bits::k6:
      LaunchTurbo4AttentionSplitStage<turbo4::Bits::k6>(
          v_skip, v_skip_threshold, q_rot_input, k_plane, v_plane, partials,
          plane_bytes, partial_bytes, num_heads, num_kv_heads, valid_context,
          max_context, split_count, max_splits, scale, d_write_status, d_status,
          formats, split_grid, stream, 0, 0);
      break;
    default:
      LaunchTurbo4AttentionSplitStage<turbo4::Bits::k4>(
          v_skip, v_skip_threshold, q_rot_input, k_plane, v_plane, partials,
          plane_bytes, partial_bytes, num_heads, num_kv_heads, valid_context,
          max_context, split_count, max_splits, scale, d_write_status, d_status,
          formats, split_grid, stream, 0, 0);
      break;
  }
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;

  Turbo4AttentionReduceInverseKernel<<<num_heads, kSplitHeadDim,
                                       max_splits * sizeof(float), stream>>>(
      partials, out_rot, out, gate, partial_bytes, out_rot_bytes, out_bytes,
      gate_bytes, num_heads, num_kv_heads, split_count, max_splits, head_dim,
      d_status);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

Turbo4LaunchStatus LaunchTurbo4AttentionSplitBatched(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats) {
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  if (q == nullptr || k_plane == nullptr || v_plane == nullptr ||
      out == nullptr || q_rot == nullptr || out_rot == nullptr ||
      partials == nullptr || d_write_status == nullptr || d_status == nullptr ||
      plane_bytes == 0 || num_heads == 0 || num_kv_heads == 0 || rows == 0)
    return Turbo4LaunchStatus::kBadArguments;
  if (max_context == 0 || first_valid_context == 0 || rows > max_context ||
      first_valid_context > max_context - rows + 1U)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (head_dim != kSplitHeadDim ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv || !std::isfinite(scale))
    return Turbo4LaunchStatus::kGeometryRefused;
  if ((gate_bytes == 0U) != (gate == nullptr))
    return Turbo4LaunchStatus::kBadArguments;

  const std::size_t vector_floats =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t vector_bytes = vector_floats * sizeof(float);
  const std::size_t rows_bytes = rows * vector_bytes;
  const std::size_t row_partial_bytes = Turbo4BatchedDecodePartialBytes(
      first_valid_context, rows, num_heads, head_dim);
  const std::size_t required_plane = Turbo4LayerPlaneBytes(
      head_dim, num_kv_heads, max_context, formats.stride);
  if (required_plane == 0U || plane_bytes < required_plane ||
      row_partial_bytes == 0U || partial_bytes < rows * row_partial_bytes ||
      q_bytes < rows_bytes || out_bytes < rows_bytes ||
      q_rot_bytes < rows_bytes || out_rot_bytes < rows_bytes ||
      (gate != nullptr && gate_bytes < rows_bytes))
    return Turbo4LaunchStatus::kBadArguments;

  // Grid extent = largest split count over the rows. Each row still derives its
  // OWN split count/size on the device from first_valid_context + row.
  const std::uint32_t max_splits = static_cast<std::uint32_t>(
      row_partial_bytes / (num_heads * (head_dim + 2) * sizeof(float)));
  if (max_splits == 0 || max_splits > kSplitHeadDim)
    return Turbo4LaunchStatus::kGeometryRefused;

  Turbo4RotateQueryKernel<<<dim3(num_heads, rows), turbo4::kBlockValues, 0,
                            stream>>>(q, q_rot, num_heads, head_dim, q_bytes,
                                      q_rot_bytes, d_status, vector_floats);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;

  const std::size_t partials_row_stride = row_partial_bytes / sizeof(float);
  const dim3 batched_grid(num_kv_heads, max_splits, rows);
  const float v_skip_threshold = GufoDevVSkipThreshold();
  const bool v_skip = v_skip_threshold > 0.0f;
  switch (formats.stride) {
    case turbo4::Bits::k4:
      LaunchTurbo4AttentionSplitStage<turbo4::Bits::k4>(
          v_skip, v_skip_threshold, q_rot, k_plane, v_plane, partials,
          plane_bytes, row_partial_bytes, num_heads, num_kv_heads,
          first_valid_context, max_context, max_splits, max_splits, scale,
          d_write_status, d_status, formats, batched_grid, stream, vector_floats,
          partials_row_stride);
      break;
    case turbo4::Bits::k5:
      LaunchTurbo4AttentionSplitStage<turbo4::Bits::k5>(
          v_skip, v_skip_threshold, q_rot, k_plane, v_plane, partials,
          plane_bytes, row_partial_bytes, num_heads, num_kv_heads,
          first_valid_context, max_context, max_splits, max_splits, scale,
          d_write_status, d_status, formats, batched_grid, stream, vector_floats,
          partials_row_stride);
      break;
    case turbo4::Bits::k6:
      LaunchTurbo4AttentionSplitStage<turbo4::Bits::k6>(
          v_skip, v_skip_threshold, q_rot, k_plane, v_plane, partials,
          plane_bytes, row_partial_bytes, num_heads, num_kv_heads,
          first_valid_context, max_context, max_splits, max_splits, scale,
          d_write_status, d_status, formats, batched_grid, stream, vector_floats,
          partials_row_stride);
      break;
  }
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;

  Turbo4AttentionReduceInverseKernel<<<dim3(num_heads, rows), kSplitHeadDim,
                                       max_splits * sizeof(float), stream>>>(
      partials, out_rot, out, gate, row_partial_bytes, out_rot_bytes, out_bytes,
      gate_bytes, num_heads, num_kv_heads, max_splits, max_splits, head_dim,
      d_status, first_valid_context, partials_row_stride, vector_floats);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

// ---------------------------------------------------------------------------
// Matrix-core (WMMA) multi-row verify reader.
//
// The scalar split reader above makes every verify row decode the WHOLE Turbo4
// KV with scalar FMA + warp shuffles (~2.3 ms per row per layer at 32k, launch
// count is not the cost: batching all rows into one launch gained 0%). This kernel
// dequantizes each K/V tile into LDS ONCE and lets all rows x all six GQA heads
// of the KV head loop over it as a 48-row matrix product (design copied from
// WmmaCausalAttention in attention_wmma.hip and TheTom fattn-mma-turbo.cuh).
//
// One CTA per (KV head, KV split), 256 threads = 8 wave32. Q = rows x 6 heads
// <= 48 vectors = 3 WMMA row tiles (rotated Q from Turbo4RotateQueryKernel,
// converted to f16 into LDS once). Per 32-position tile:
//   1. every thread dequantizes 32 K dims and 32 V dims of one position
//      (centroid * norm -> f16): K to LDS [pos][d], V transposed to [d][pos];
//      K and V are staged one 128-dim half at a time (waves 0-3 own
//      dims 0-127, waves 4-7 dims 128-255), and the S tile reuses that buffer;
//   2. waves 0-5: S[16x16] = Q K^T, 16 WMMA each (qt = wave/2, key block =
//      wave%2), scaled, to an fp32 LDS tile;
//   3. 192 threads, 4 per query vector: per-row causal mask (row r sees
//      positions < first_context + r), online softmax in fp32, P as f16 to LDS;
//   4. all waves: O[qt][dim tile wave and wave+8] += P V, 2 WMMA each; the
//      fp32 accumulators (48 VGPR) stay in registers across the whole split.
// The (m, l, o) partial records match the scalar reader's layout
// [row][split][head][2 + 256] in the ROTATED basis, so the existing reduce
// kernel does the cross-split merge, inverse rotation and gate. All rows share
// the split plan of the LAST row's context; rows that end earlier emit empty
// records (m = -FLT_MAX, l = 0, o = 0) for splits past their context.
//
// The same kernel serves Turbo5/6. `Turbo4Slice` now also carries the
// qh bytes, and DequantTurbo4SlicePlane dispatches on the plane codec, exactly
// as the scalar split reader's Turbo4DecodePlaneLane8 does, so a decoded K/V
// value is identical whichever reader produced it. Only the block address
// (82 / 98-byte slot) and the code reconstruction change; the WMMA/softmax body
// is width-independent.
//
// LDS (static, bytes): q 48x264 f16 = 25344; kv union 10240 (K chunk 32x136
// f16 = 8704, V^T chunk 128x40 f16 = 10240, and the S tile 48x33 f32 = 6336
// all share it); P 48x40 f16 = 3840; row_scale 192; centroids 2x64 f32 = 512;
// error word 4  ->  40132 (39.2 KiB), was 56708 before the half-dim staging. K and V^T are staged
// in two 128-dim halves and S reuses the K/V^T buffer; see the barrier comments
// in the tile loop. 3 CTAs fit one WGP's 128 KiB (was 2).
// VGPR estimate (NOT measured, compile-only): o_acc 48 + S acc 8 + Q/K frags 16
// + raw K/V slices 14 + dequant temps ~16 + softmax/address ~20  ->  ~125.
//
// NOT bit-identical to the scalar reader: f16 Q/K/V/P operands and a different
// summation order. The gate is turbo4_verify_wmma_gpu_test (L2 rel < 2e-3); the
// Turbo5/6 path has the same structure, so the same bound applies.
// ponytail: K/V global loads are issued at the tile top (V's latency hides
// behind the chunked S phase + softmax, K's does not) and there is no cross-tile prefetch; add
// one if a profile shows the dequant stage dominating.
namespace {

constexpr std::uint32_t kVerifyMaxRows = 8;
constexpr std::uint32_t kVerifyQTiles = 3;
constexpr std::uint32_t kVerifyQVecs = kVerifyQTiles * 16;
constexpr std::uint32_t kVerifyKeys = 32;
constexpr std::uint32_t kVerifyRowStride = kSplitHeadDim + 8;  // q and K rows
constexpr std::uint32_t kVerifyVtStride = kVerifyKeys + 8;     // V^T and P rows
constexpr std::uint32_t kVerifySStride = kVerifyKeys + 1;
// K and V^T are staged in kVerifyChunks halves of the head dim, so the
// shared buffer holds 128 dims of K ([key][d], 136-half rows) or 128 dims of
// V^T ([d][key], 40-half rows), never the full 256.
constexpr std::uint32_t kVerifyChunks = 2;
constexpr std::uint32_t kVerifyChunkDims = kSplitHeadDim / kVerifyChunks;
constexpr std::uint32_t kVerifyKStride = kVerifyChunkDims + 8;  // K chunk rows
constexpr std::uint32_t kVerifyKChunkHalves = kVerifyKeys * kVerifyKStride;
constexpr std::uint32_t kVerifyVtChunkHalves = kVerifyChunkDims * kVerifyVtStride;
// The S tile (fp32) shares this buffer once the K chunks are consumed.
constexpr std::uint32_t kVerifyKvHalves =
    kVerifyKChunkHalves > kVerifyVtChunkHalves ? kVerifyKChunkHalves
                                               : kVerifyVtChunkHalves;
static_assert(kVerifyQVecs * kVerifySStride * sizeof(float) <=
                  kVerifyKvHalves * sizeof(__half),
              "S tile must fit the shared K / V^T buffer");
static_assert(kVerifyChunks == 2 && kVerifyChunkDims % 16 == 0 &&
                  kVerifyChunkDims == 4 * 32,
              "a chunk is four 32-dim dequant groups");
static_assert(kVerifyKStride % 8 == 0, "K chunk rows start on 16 bytes");
static_assert(kVerifyQVecs == kVerifyMaxRows * kSplitQHeadsPerKv, "48 vectors");
static_assert(kVerifyRowStride % 8 == 0 && kVerifyVtStride % 8 == 0,
              "WMMA fragment rows start on a 16-byte boundary");

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

// Wave32 layout (verified in attention_wmma.hip): A lane L holds row L%16 and
// 16 contiguous k, B lane L holds column L%16 and 16 contiguous k, and C
// element i is row 2i + L/16, column L%16.
__device__ __forceinline__ v8f VerifyWmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h VerifyLoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

/// Raw bytes of 32 consecutive dims (16 nibble bytes) of one KV block, plus the
/// qh bytes that carry the sign (Turbo5) or the two high code bits (Turbo6) for
/// those same 32 dims. `qh[0..1]` are the 4 / 8 bytes starting at kQhOffset for
/// the group; k4 leaves them zero.
struct Turbo4Slice {
  std::uint32_t w[4];
  std::uint32_t qh[2];
  float norm;
};

/// dgroup 0..7 selects dims [32*dgroup, 32*dgroup + 32): block dgroup/4, byte
/// 2 + 16*(dgroup%4). Block bytes are only 2-aligned (66/82/98-byte blocks), so
/// the nibbles and qh bytes are read as u16/u32. `kSlot` is the physical slot
/// width (plane stride); `codec` is the plane's own codec, which selects how
/// many qh bytes to read (0 / 4 / 8) and may differ from the slot width.
template<turbo4::Bits kSlot>
__device__ __forceinline__ bool LoadTurbo4Slice(
    const std::uint8_t* plane, std::uint32_t token, std::uint32_t kv_head,
    std::uint32_t dgroup, std::uint32_t num_kv_heads, std::uint32_t max_context,
    turbo4::Bits codec, Turbo4Slice* out) {
  std::size_t base = 0;
  if (!Turbo4BlockByteOffset(token, kv_head, kSplitHeadDim, num_kv_heads,
                             max_context, &base, kSlot))
    return false;
  const std::uint8_t* block =
      plane + base + (dgroup / 4U) * turbo4::BlockBytes(kSlot);
  std::uint16_t norm_bits = 0;
  std::memcpy(&norm_bits, block, sizeof(norm_bits));
  out->norm = __half2float(__ushort_as_half(norm_bits));
  const std::uint8_t* nibbles = block + 2 + 16U * (dgroup % 4U);
  for (std::uint32_t i = 0; i < 4; ++i) {
    std::uint16_t lo = 0, hi = 0;
    std::memcpy(&lo, nibbles + 4U * i, sizeof(lo));
    std::memcpy(&hi, nibbles + 4U * i + 2U, sizeof(hi));
    out->w[i] =
        static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16);
  }
  out->qh[0] = 0;
  out->qh[1] = 0;
  // The 32 values sit entirely inside the qh group `dgroup % 4`: 4 bytes for
  // Turbo5 (one sign bit per value) or 8 bytes for Turbo6 (two bits per value).
  // kQhOffset is 66 for every width, so the qh read is slot-independent.
  const std::uint32_t group = dgroup & 3U;
  if (codec == turbo4::Bits::k5)
    std::memcpy(out->qh, block + turbo4::kQhOffset + 4U * group, 4U);
  else if (codec == turbo4::Bits::k6)
    std::memcpy(out->qh, block + turbo4::kQhOffset + 8U * group, 8U);
  return true;
}

/// Even element in the LOW nibble, as the writer packs it. k4 is the original
/// nibble-only path; k5 rebuilds sign * magnitude; k6 appends the two qh bits.
template<turbo4::Bits kCodec>
__device__ __forceinline__ void DequantTurbo4Slice(const Turbo4Slice& slice,
                                                   const float* centroids,
                                                   __half* out) {
  if constexpr (kCodec == turbo4::Bits::k5) {
#pragma unroll
    for (std::uint32_t j = 0; j < 32; ++j) {
      const std::uint32_t mag = (slice.w[j >> 3] >> (4U * (j & 7U))) & 0xFU;
      const std::uint32_t neg = (slice.qh[0] >> j) & 1U;
      const float c = neg != 0U ? -centroids[mag] : centroids[mag];
      out[j] = __float2half(c * slice.norm);
    }
  } else if constexpr (kCodec == turbo4::Bits::k6) {
#pragma unroll
    for (std::uint32_t j = 0; j < 32; ++j) {
      const std::uint32_t lo = (slice.w[j >> 3] >> (4U * (j & 7U))) & 0xFU;
      const std::uint32_t hi =
          (slice.qh[j >> 4] >> (8U * ((j >> 2) & 3U) + 2U * (j & 3U))) & 3U;
      out[j] = __float2half(centroids[lo | (hi << 4U)] * slice.norm);
    }
  } else {
#pragma unroll
    for (std::uint32_t j = 0; j < 32; ++j)
      out[j] = __float2half(
          centroids[(slice.w[j >> 3] >> (4U * (j & 7U))) & 0xFU] * slice.norm);
  }
}

/// Runtime codec dispatcher, mirroring Turbo4DecodePlaneLane8: the codec bits
/// are uniform across the CTA, so this branch costs no per-lane divergence.
__device__ __forceinline__ void DequantTurbo4SlicePlane(
    turbo4::Bits codec, const Turbo4Slice& slice, const float* centroids,
    __half* out) {
  if (codec == turbo4::Bits::k5)
    DequantTurbo4Slice<turbo4::Bits::k5>(slice, centroids, out);
  else if (codec == turbo4::Bits::k6)
    DequantTurbo4Slice<turbo4::Bits::k6>(slice, centroids, out);
  else
    DequantTurbo4Slice<turbo4::Bits::k4>(slice, centroids, out);
}

// Served K = V = Turbo6 instance only (-DGUFO_T4_PREFETCH=1): the
// prefetch keeps the next tile's slice as RAW words. Nothing reads them until
// the tile that uses them, so no s_waitcnt lands between the next tile's load
// issue and this tile's WMMA. Same bytes as LoadTurbo4Slice<k6>: w = the 16
// nibble bytes, qh = the 8 qh bytes, norm = the fp16 norm bits.
struct Turbo6RawSlice {
  std::uint32_t w[4];
  std::uint32_t qh[2];
  // Block bytes 0..3: the fp16 norm bits in the low half, the block's first
  // two nibble bytes (unused) in the high half. Loaded as a whole 32-bit word
  // on purpose: as a 16-bit value gfx1151's true16 allocator puts it in half a
  // VGPR (global_load_d16) and uses the other half for f16 temporaries, and
  // every such write waits on the next tile's load.
  std::uint32_t norm;
};

/// One slice from a block whose bytes are only 2-aligned (98-byte slots): one
/// unaligned 32-bit (norm), one 128-bit and one 64-bit load. `base` is the
/// plane (SGPR), the offsets are 32-bit, so each load is SGPR base + VGPR
/// offset with no 64-bit address math.
__device__ __forceinline__ void LoadTurbo6Raw(const std::uint8_t* base,
                                              std::uint32_t norm_off,
                                              std::uint32_t nib_off,
                                              std::uint32_t qh_off,
                                              Turbo6RawSlice* out) {
  __builtin_memcpy(&out->norm, __builtin_assume_aligned(base + norm_off, 2),
                   sizeof(out->norm));
  __builtin_memcpy(out->w, __builtin_assume_aligned(base + nib_off, 2),
                   sizeof(out->w));
  __builtin_memcpy(out->qh, __builtin_assume_aligned(base + qh_off, 2),
                   sizeof(out->qh));
}

/// The decode-ready slice of `self`, with load_tile's two rules applied at
/// the point of use instead of at load time:
///  - past the end (`valid` false) the slice is all zero, exactly the
///    Turbo4Slice{} load_tile leaves (the raw words were loaded from a clamped
///    token so the load needs no branch, and are discarded here);
///  - if either the K or the V norm of this (token, dgroup) is unusable both
///    become 0 and `*poisoned` is set (the caller latches it).
__device__ __forceinline__ Turbo4Slice Turbo6Unpack(const Turbo6RawSlice& self,
                                                    const Turbo6RawSlice& other,
                                                    bool valid,
                                                    bool* poisoned) {
  std::uint32_t bits = self.norm, other_bits = other.norm;
  // Empty asm: the norm stays a whole 32-bit VGPR from load to here, so the
  // compiler cannot narrow it back to a half register (see Turbo6RawSlice).
  // No instruction, no value change.
  asm("" : "+v"(bits), "+v"(other_bits));
  bits = valid ? bits : 0U;
  other_bits = valid ? other_bits : 0U;
  const float norm =
      __half2float(__ushort_as_half(static_cast<std::uint16_t>(bits)));
  const float other_norm =
      __half2float(__ushort_as_half(static_cast<std::uint16_t>(other_bits)));
  *poisoned = !turbo4::StoredNormIsUsable(norm) ||
              !turbo4::StoredNormIsUsable(other_norm);
  Turbo4Slice s;
  for (std::uint32_t i = 0; i < 4; ++i)
    s.w[i] = valid ? self.w[i] : 0U;
  s.qh[0] = valid ? self.qh[0] : 0U;
  s.qh[1] = valid ? self.qh[1] : 0U;
  s.norm = *poisoned ? 0.0f : norm;
  return s;
}

/// Where the K = V = Turbo6 pipelined instance may run: both codecs Turbo6 in
/// 98-byte slots, and the layer plane non-empty and addressable with 32 bits.
/// A non-zero Turbo4LayerPlaneBytes means token * bytes_per_token + head bytes
/// never overflows for token < max_context, so the per-token checked
/// Turbo4BlockByteOffset can never refuse and is hoisted out entirely.
__host__ __device__ inline bool VerifyT6PipeFits(turbo4::PlaneFormats formats,
                                                 std::uint32_t num_kv_heads,
                                                 std::uint32_t max_context) {
  const std::size_t bytes = Turbo4LayerPlaneBytes(
      kSplitHeadDim, num_kv_heads, max_context, turbo4::Bits::k6);
  return formats.stride == turbo4::Bits::k6 &&
         formats.k == turbo4::Bits::k6 && formats.v == turbo4::Bits::k6 &&
         bytes != 0U && bytes <= 0xFFFFFFFFULL;
}

// GUFO_DEV_V_SKIP for the WMMA verify reader. The unit here is a whole 32-key
// tile: V is transposed into shared once per tile and reused by the P V matmul,
// so it can only be dropped tile-wide. `kVSkipTinyV` is a COMPILE-TIME switch;
// the false instantiation is the pre-GUFO_DEV_V_SKIP kernel, so an unset env
// pays nothing. The true instantiation drops a tile's V dequant and V matmul
// only when every key in the tile is below v_skip_threshold for every query row
// in the tile; the decision is block-uniform (__syncthreads_or), so the skipped
// path is a uniform branch. Both instantiations are emitted into the binary;
// only the launch choice depends on the env.
//
// `kSlot` is the physical slot width (66/82/98 B). The kernel decodes
// each K/V block once per (kv head, split) and applies it to all rows x 6 GQA
// heads, so the width only changes how a block is addressed and decoded; the
// WMMA/softmax body is width-independent. k4 keeps the original 66-byte path.
#ifndef GUFO_T4_PREFETCH
#define GUFO_T4_PREFETCH 0  // -DGUFO_T4_PREFETCH=1: verify-reader software pipeline
#endif
// Barrier for the verify tile loop. With the prefetch on, __syncthreads()
// would not do: on gfx11 its workgroup fence waits for every outstanding
// global load (vmcnt 0), i.e. for the next tile's slices, at the first barrier.
// Inside the loop only LDS is shared between threads (partials are written
// after it, s_error is an LDS atomic), so a fence on the local address space
// plus s_barrier orders everything the loop needs (CK's block_sync_lds).
__device__ __forceinline__ void VerifyTileBarrier() {
#if GUFO_T4_PREFETCH
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
  __syncthreads();
#endif
}
// `kT6Pipe` (K = V = Turbo6 only, launched only with GUFO_T4_PREFETCH):
// the same kernel with a prefetch that does not wait on itself. The next
// tile's slices are loaded as raw words (Turbo6RawSlice), unconditionally
// (token clamped to end - 1, zeroed at use), at the plane base plus a 32-bit
// offset (the checked 64-bit Turbo4BlockByteOffset is hoisted:
// VerifyT6PipeFits proves it never refuses), and the norm check moves from
// load time to the K dequant. Same bytes, same
// decode (DequantTurbo4Slice<k6>), same WMMA/softmax order: bitwise identical
// partials and status to the kT6Pipe = false instance.
template<turbo4::Bits kSlot, bool kVSkipTinyV = false, bool kT6Pipe = false>
__global__ __launch_bounds__(256) void Turbo4VerifyWmmaSplitKernel(
    const float* __restrict__ q_rot, const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, float* __restrict__ partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t max_splits, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    std::size_t q_rot_row_stride, std::size_t partials_row_stride,
    turbo4::PlaneFormats formats, float v_skip_threshold = 0.0f,
    std::uint32_t plan_context = 0U) {
  (void)v_skip_threshold;  // read only by the kVSkipTinyV instantiation
  static_assert(!kT6Pipe || (kSlot == turbo4::Bits::k6 && !kVSkipTinyV),
                "the pipelined reader is the K = V = Turbo6 no-V-skip instance");
  __shared__ __attribute__((
      aligned(16))) __half q_lds[kVerifyQVecs * kVerifyRowStride];
  // K chunk, then the S tile, then V^T chunks, one after another (see the loop).
  __shared__ __attribute__((aligned(16))) __half kv_lds[kVerifyKvHalves];
  __shared__
      __attribute__((aligned(16))) __half p_lds[kVerifyQVecs][kVerifyVtStride];
  __shared__ float row_scale[kVerifyQVecs];
  // One 16 / 32 / 64-entry table per plane. Turbo4 and Turbo6 index by code;
  // Turbo5 holds the 16 nonnegative centroids and applies the qh sign. When
  // K and V share a codec the second table is unused but costs 256 B, well
  // under the LDS bound.
  __shared__ float s_centroids[64];
  __shared__ float s_v_centroids[64];
  __shared__ unsigned int s_error;

  float(*const s_lds)[kVerifySStride] =
      reinterpret_cast<float(*)[kVerifySStride]>(kv_lds);

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;

  // Same split plan for every row: the LAST row's (longest) context.
  // plan_context != 0 (batch-invariant mode) lays the split SIZE out for the
  // width-8 last row instead, so the key -> split boundaries stop moving with
  // the verify width. The COUNT still follows this launch's last row, so no
  // split starts past it.
  const std::uint32_t last_context = first_context + rows - 1U;
  const std::uint32_t split_size = static_cast<std::uint32_t>(
      Turbo4DecodeSplitSize(plan_context != 0U ? plan_context : last_context));
  const std::uint32_t split_count =
      split_size == 0U ? 0U : (last_context + split_size - 1U) / split_size;
  const std::size_t required_bytes = static_cast<std::size_t>(max_splits) *
                                     num_heads * kSplitRecordWidth *
                                     sizeof(float);
  if (num_heads != num_kv_heads * kSplitQHeadsPerKv ||
      kv_head >= num_kv_heads || rows == 0 || rows > kVerifyMaxRows ||
      first_context == 0 || last_context > max_context || split_count == 0 ||
      split_count > max_splits || split >= split_count ||
      !formats.Valid() || formats.stride != kSlot ||
      plane_bytes < Turbo4LayerPlaneBytes(kSplitHeadDim, num_kv_heads,
                                          max_context, kSlot) ||
      (kT6Pipe && !VerifyT6PipeFits(formats, num_kv_heads, max_context)) ||
      partial_bytes < required_bytes || q_rot == nullptr ||
      partials == nullptr || status == nullptr || write_status == nullptr) {
    if (tid == 0)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  if (tid == 0) {
    s_error = static_cast<unsigned int>(*status);
    if (*write_status != Turbo4KernelStatus::kOk) {
      s_error = static_cast<unsigned int>(Turbo4KernelStatus::kWriterFailed);
      Turbo4LatchFailure(status, Turbo4KernelStatus::kWriterFailed);
    }
  }
  // Width-generic table fill. For Turbo5 only the 16 nonnegative
  // centroids are stored and the sign is applied at decode; k4 and k6 store
  // every code. Same fill as the scalar split kernel, so the decoded values are
  // identical for a given block.
  const unsigned k_codes = formats.k == turbo4::Bits::k5
                               ? 16U
                               : 1U << static_cast<unsigned>(formats.k);
  const unsigned v_codes = formats.v == turbo4::Bits::k5
                               ? 16U
                               : 1U << static_cast<unsigned>(formats.v);
  if (tid < k_codes)
    s_centroids[tid] = turbo4::CentroidValue(
        formats.k, tid + (formats.k == turbo4::Bits::k5 ? 16U : 0U));
  if (formats.v != formats.k && tid < v_codes)
    s_v_centroids[tid] = turbo4::CentroidValue(
        formats.v, tid + (formats.v == turbo4::Bits::k5 ? 16U : 0U));
  __syncthreads();
  if (s_error != 0U)
    return;

  // Query vector m = row * 6 + gqa_head; vectors past rows * 6 are zero rows
  // that no softmax row ever reads and no record ever receives.
  const std::uint32_t nvec = rows * kSplitQHeadsPerKv;
  for (std::uint32_t m = 0; m < kVerifyQVecs; ++m) {
    float value = 0.0f;
    if (m < nvec)
      value = q_rot[static_cast<std::size_t>(m / kSplitQHeadsPerKv) *
                        q_rot_row_stride +
                    (static_cast<std::size_t>(kv_head) * kSplitQHeadsPerKv +
                     m % kSplitQHeadsPerKv) *
                        kSplitHeadDim +
                    tid];
    q_lds[m * kVerifyRowStride + tid] = __float2half(value);
  }

  // Softmax ownership: 4 consecutive threads per query vector (waves 0-5).
  const std::uint32_t rg = tid >> 2U;
  const std::uint32_t seg = tid & 3U;
  const std::uint32_t row_context =
      rg < nvec ? first_context + rg / kSplitQHeadsPerKv : 0U;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  v8f o_acc[kVerifyQTiles][2] = {};

  // Dequant ownership: position pos, dims [32*dgroup, 32*dgroup + 32).
  const std::uint32_t pos = tid & 31U;
  const std::uint32_t dgroup = tid >> 5U;

  const std::uint32_t begin = split * split_size;
  const std::uint32_t end =
      last_context < begin + split_size ? last_context : begin + split_size;
  // This thread's K and V slice of the tile starting at `at` (zero past end).
  const auto load_tile = [&](std::uint32_t at, Turbo4Slice* k,
                             Turbo4Slice* v) {
    *k = Turbo4Slice{};
    *v = Turbo4Slice{};
    if (at + pos < end) {
      if (!LoadTurbo4Slice<kSlot>(k_plane, at + pos, kv_head, dgroup,
                                  num_kv_heads, max_context, formats.k, k) ||
          !LoadTurbo4Slice<kSlot>(v_plane, at + pos, kv_head, dgroup,
                                  num_kv_heads, max_context, formats.v, v))
        atomicMax(&s_error, static_cast<unsigned int>(
                                Turbo4KernelStatus::kGeometryRefused));
      if (!turbo4::StoredNormIsUsable(k->norm) ||
          !turbo4::StoredNormIsUsable(v->norm)) {
        atomicMax(&s_error,
                  static_cast<unsigned int>(Turbo4KernelStatus::kPoisonedNorm));
        k->norm = 0.0f;
        v->norm = 0.0f;
      }
    }
  };
  // kT6Pipe addressing, hoisted out of the loop (the scalar reader's
  // Pattern): byte offset of (token, kv_head, dgroup's block) is
  //   token * t6_token_bytes + kv_head * t6_head_bytes + (dgroup / 4) * 98,
  // exactly Turbo4BlockByteOffset(token, kv_head, ..., k6) + the block step,
  // in 32-bit arithmetic: every offset is below the layer plane size, which
  // VerifyT6PipeFits bounds by 2^32. The per-lane part after the token is
  // fixed for the split. Unused by the other instances (dead code there).
  const std::uint32_t t6_token_bytes = static_cast<std::uint32_t>(
      Turbo4BlocksPerToken(kSplitHeadDim, num_kv_heads) *
      turbo4::BlockBytes(kSlot));
  const std::uint32_t t6_norm_off =
      kv_head * static_cast<std::uint32_t>(Turbo4BlocksPerHead(kSplitHeadDim) *
                                           turbo4::BlockBytes(kSlot)) +
      (dgroup / 4U) * static_cast<std::uint32_t>(turbo4::BlockBytes(kSlot));
  const std::uint32_t t6_nib_off = t6_norm_off + 2U + 16U * (dgroup & 3U);
  const std::uint32_t t6_qh_off = t6_norm_off +
                                  static_cast<std::uint32_t>(turbo4::kQhOffset) +
                                  8U * (dgroup & 3U);
  // No branch and no check here consumes the loads: a lane past the end reads
  // token end - 1 (a valid token: begin < end <= last_context <= max_context)
  // and Turbo6Unpack zeroes it at use, so the in-flight load count is the same
  // on every path and the waitcnt pass never has to assume a skipped load.
  const auto load_raw = [&](std::uint32_t at, Turbo6RawSlice* k,
                            Turbo6RawSlice* v) {
    const std::uint32_t token = at + pos < end ? at + pos : end - 1U;
    const std::uint32_t token_off = token * t6_token_bytes;
    LoadTurbo6Raw(k_plane, token_off + t6_norm_off, token_off + t6_nib_off,
                  token_off + t6_qh_off, k);
    LoadTurbo6Raw(v_plane, token_off + t6_norm_off, token_off + t6_nib_off,
                  token_off + t6_qh_off, v);
  };
  using TileSlice =
      std::conditional_t<kT6Pipe, Turbo6RawSlice, Turbo4Slice>;
  const auto load_any = [&](std::uint32_t at, TileSlice* k, TileSlice* v) {
    if constexpr (kT6Pipe)
      load_raw(at, k, v);
    else
      load_tile(at, k, v);
  };
#if GUFO_T4_PREFETCH
  // Software pipeline. The next tile's slices are
  // requested before this tile's 5 barriers and WMMA work, so the global
  // latency overlaps compute instead of stalling the first dequant. Same
  // tiles, same values, same order: output is bitwise identical; s_error is
  // read only after the loop, so latching it one tile early changes nothing.
  TileSlice next_ks{}, next_vs{};
  load_any(begin, &next_ks, &next_vs);
#endif
  for (std::uint32_t tile = begin; tile < end; tile += kVerifyKeys) {
    VerifyTileBarrier();  // previous tile's PV is done with kv_lds and p_lds
#if GUFO_T4_PREFETCH
    const TileSlice ks = next_ks, vs = next_vs;
    load_any(tile + kVerifyKeys, &next_ks, &next_vs);
#else
    TileSlice ks{}, vs{};
    load_any(tile, &ks, &vs);
#endif
    // --- S = Q K^T (scaled), waves 0-5, K staged in two 128-dim chunks ----
    // kv_lds holds K dims [128c, 128c + 128) for c = 0 then 1; the S
    // accumulator carries across chunks, so the 16 k-steps still add in order
    // 0..15 (bit-identical to the unchunked loop). Waves 0-3 / 4-7 dequant
    // chunk 0 / 1 (dgroup = wave, so the branch is wave-uniform).
    const std::uint32_t s_qt = wave >> 1U;
    const std::uint32_t s_kb = wave & 1U;
    const bool s_wave = wave < 6U && s_qt * 16U < nvec;
    v8f s_acc = {};
#pragma unroll 1
    for (std::uint32_t c = 0; c < kVerifyChunks; ++c) {
      if ((dgroup >> 2U) == c) {
        __attribute__((aligned(16))) __half vals[32];
        if constexpr (kT6Pipe) {
          // The norm check load_tile did at load time, now at first use; the
          // latch still precedes the barrier below, and s_error is read only
          // after the loop.
          bool poisoned = false;
          const Turbo4Slice kslice =
              Turbo6Unpack(ks, vs, tile + pos < end, &poisoned);
          if (poisoned)
            atomicMax(&s_error, static_cast<unsigned int>(
                                    Turbo4KernelStatus::kPoisonedNorm));
          DequantTurbo4Slice<turbo4::Bits::k6>(kslice, s_centroids, vals);
        } else {
          DequantTurbo4SlicePlane(formats.k, ks, s_centroids, vals);
        }
        uint4* dst = reinterpret_cast<uint4*>(
            &kv_lds[pos * kVerifyKStride + (dgroup & 3U) * 32U]);
#pragma unroll
        for (std::uint32_t i = 0; i < 4; ++i)
          dst[i] = reinterpret_cast<const uint4*>(vals)[i];
      }
      // K chunk c staged. For c = 0 this also orders the writes after the top
      // barrier's guarantee that the previous tile's PV is done with kv_lds.
      VerifyTileBarrier();
      if (s_wave) {
#pragma unroll
        for (std::uint32_t kstep = 0; kstep < kVerifyChunkDims / 16U; ++kstep) {
          s_acc = VerifyWmma(
              VerifyLoadFrag(
                  &q_lds[(s_qt * 16U + sub) * kVerifyRowStride +
                         c * kVerifyChunkDims + kstep * 16U]),
              VerifyLoadFrag(&kv_lds[(s_kb * 16U + sub) * kVerifyKStride +
                                     kstep * 16U]),
              s_acc);
        }
      }
      VerifyTileBarrier();  // K chunk c fully read: kv_lds is free again
    }
    // kv_lds now holds the S tile (fp32). The barrier above separates every
    // wave's last K read from these writes.
    if (s_wave) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s_lds[s_qt * 16U + 2U * i + half_id][s_kb * 16U + sub] =
            s_acc[i] * scale;
    }
    VerifyTileBarrier();  // S visible to the vote / softmax threads

    // GUFO_DEV_V_SKIP: this tile's largest score per row, reduced across the
    // four softmax threads, is the only thing that decides whether some key is
    // worth its V row. A key is kept when exp(score - running_max) reaches the
    // threshold; that weight is monotonic in score, so the row's largest score
    // bounds it from above. Keep the tile if any row keeps any key.
    bool tile_skip = false;
    if constexpr (kVSkipTinyV) {
      bool row_keep = false;
      // Only real query rows vote: padded rows (rg >= nvec) and empty rows have
      // part_max = -inf, and -inf - (-inf) = NaN would keep every tile (review).
      if (rg < nvec) {
        float part_max = -INFINITY;
#pragma unroll
        for (std::uint32_t m = 0; m < 8; ++m) {
          const std::uint32_t col = seg * 8U + m;
          const float v =
              (tile + col < row_context) ? s_lds[rg][col] : -INFINITY;
          part_max = fmaxf(part_max, v);
        }
        part_max = fmaxf(part_max, __shfl_xor(part_max, 1));
        part_max = fmaxf(part_max, __shfl_xor(part_max, 2));
        // exp(min(part_max - running_max, 0)) is the row's largest weight. An
        // empty row (part_max = -inf) gives 0, so it cannot keep the tile; a
        // -inf running_max (first tile) gives 1, so it always keeps.
        const float d = part_max - running_max;
        const float rel = d < 0.0f ? d : 0.0f;
        row_keep = part_max != -INFINITY && expf(rel) >= v_skip_threshold;
      }
      tile_skip = __syncthreads_or(row_keep ? 1 : 0) == 0;
    }

    // --- online softmax, per-row causal mask ------------------------------
    if (!tile_skip && rg < kVerifyQVecs) {
      float vals[8];
      float part_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < 8; ++m) {
        const std::uint32_t col = seg * 8U + m;
        vals[m] = (tile + col < row_context) ? s_lds[rg][col] : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
      part_max = fmaxf(part_max, __shfl_xor(part_max, 1));
      part_max = fmaxf(part_max, __shfl_xor(part_max, 2));
      const float next_max = fmaxf(running_max, part_max);
      const float prior_scale =
          isfinite(running_max) ? expf(running_max - next_max) : 0.0f;
      float part_sum = 0.0f;
#pragma unroll
      for (std::uint32_t m = 0; m < 8; ++m) {
        const __half w =
            __float2half(isfinite(vals[m]) ? expf(vals[m] - next_max) : 0.0f);
        p_lds[rg][seg * 8U + m] = w;
        part_sum += __half2float(w);  // l sums what PV multiplies
      }
      part_sum += __shfl_xor(part_sum, 1);
      part_sum += __shfl_xor(part_sum, 2);
      running_max = next_max;
      running_sum = running_sum * prior_scale + part_sum;
      if (seg == 0U)
        row_scale[rg] = prior_scale;
    }

    // --- O += P V: V^T staged in two 128-dim chunks (t = chunk). Wave `w`
    // owns local dim tile w of chunk t, i.e. dims (w + 8t) * 16 + [0, 16), the
    // same tiles as before. tile_skip is CTA-uniform, so the barriers inside
    // are reached by every wave or by none.
    if (!tile_skip) {
#pragma unroll
      for (std::uint32_t t = 0; t < kVerifyChunks; ++t) {
        // t = 0: every softmax thread has read S (kv_lds) and written P /
        // row_scale. t = 1: every wave has finished PV on chunk 0 in kv_lds.
        VerifyTileBarrier();
        if ((dgroup >> 2U) == t) {
          __attribute__((aligned(16))) __half vals[32];
          if constexpr (kT6Pipe) {
            bool poisoned = false;  // latched by the K dequant of this tile
            DequantTurbo4Slice<turbo4::Bits::k6>(
                Turbo6Unpack(vs, ks, tile + pos < end, &poisoned), s_centroids,
                vals);
          } else {
            DequantTurbo4SlicePlane(
                formats.v, vs,
                formats.v == formats.k ? s_centroids : s_v_centroids, vals);
          }
#pragma unroll
          for (std::uint32_t j = 0; j < 32; ++j)
            kv_lds[((dgroup & 3U) * 32U + j) * kVerifyVtStride + pos] = vals[j];
        }
        VerifyTileBarrier();  // V^T chunk t staged
#pragma unroll
        for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
          if (qt * 16U >= nvec)
            continue;
          float row_rescale[8];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            row_rescale[i] = row_scale[qt * 16U + 2U * i + half_id];
          v8f acc = o_acc[qt][t];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            acc[i] *= row_rescale[i];
#pragma unroll
          for (std::uint32_t kb = 0; kb < kVerifyKeys / 16U; ++kb)
            acc = VerifyWmma(
                VerifyLoadFrag(&p_lds[qt * 16U + sub][kb * 16U]),
                VerifyLoadFrag(&kv_lds[(wave * 16U + sub) * kVerifyVtStride +
                                       kb * 16U]),
                acc);
          o_acc[qt][t] = acc;
        }
      }
    }
    // No trailing barrier: the next iteration's top barrier orders this tile's
    // last kv_lds / p_lds / row_scale reads before the next writes.
  }

  if (s_error != 0U) {  // set before a barrier inside the loop (or by tid 0)
    if (tid == 0)
      Turbo4LatchFailure(status, static_cast<Turbo4KernelStatus>(s_error));
    return;
  }

  // --- partial records [row][split][head]{m, l, o[256]}, rotated basis -----
  // Empty (row ends before this split): m = -FLT_MAX, l = 0, o = 0.
  if (rg < nvec && seg == 0U) {
    float* rec =
        partials +
        static_cast<std::size_t>(rg / kSplitQHeadsPerKv) * partials_row_stride +
        (static_cast<std::size_t>(split) * num_heads +
         kv_head * kSplitQHeadsPerKv + rg % kSplitQHeadsPerKv) *
            kSplitRecordWidth;
    rec[0] = isfinite(running_max) ? running_max : -3.402823466e+38f;
    rec[1] = running_sum;
  }
#pragma unroll
  for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t m = qt * 16U + 2U * i + half_id;
        if (m >= nvec)
          continue;
        float* rec = partials +
                     static_cast<std::size_t>(m / kSplitQHeadsPerKv) *
                         partials_row_stride +
                     (static_cast<std::size_t>(split) * num_heads +
                      kv_head * kSplitQHeadsPerKv + m % kSplitQHeadsPerKv) *
                         kSplitRecordWidth;
        rec[2 + (wave + 8U * t) * 16U + sub] = o_acc[qt][t][i];
      }
    }
  }
}

/// Pick the slot-width instantiation of the WMMA verify kernel and its
/// GUFO_DEV_V_SKIP twin. One place so the two cannot drift; mirrors
/// LaunchTurbo4AttentionSplitStage.
template<turbo4::Bits kSlot>
void LaunchTurbo4VerifyWmmaStage(
    bool v_skip, float v_skip_threshold, const float* q_rot,
    const std::uint8_t* k_plane, const std::uint8_t* v_plane, float* partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t max_splits, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    std::size_t q_rot_row_stride, std::size_t partials_row_stride,
    turbo4::PlaneFormats formats, dim3 grid, hipStream_t stream) {
  const std::uint32_t plan_context =
      GufoDevBatchInvariant()
          ? BatchInvariantPlanContext(first_context, rows, max_context)
          : 0U;
#if GUFO_T4_PREFETCH
  // The served K = V = Turbo6 case takes the pipelined instance
  // (bitwise identical output); GUFO_DEV_VERIFY_T6_PIPE=0 keeps the old one.
  if constexpr (kSlot == turbo4::Bits::k6) {
    if (!v_skip && GufoDevVerifyT6Pipe() &&
        VerifyT6PipeFits(formats, num_kv_heads, max_context)) {
      Turbo4VerifyWmmaSplitKernel<kSlot, false, true>
          <<<grid, 256, 0, stream>>>(
              q_rot, k_plane, v_plane, partials, plane_bytes, partial_bytes,
              num_heads, num_kv_heads, first_context, rows, max_context,
              max_splits, scale, write_status, status, q_rot_row_stride,
              partials_row_stride, formats, 0.0f, plan_context);
      return;
    }
  }
#endif
  if (v_skip)
    Turbo4VerifyWmmaSplitKernel<kSlot, true><<<grid, 256, 0, stream>>>(
        q_rot, k_plane, v_plane, partials, plane_bytes, partial_bytes,
        num_heads, num_kv_heads, first_context, rows, max_context, max_splits,
        scale, write_status, status, q_rot_row_stride, partials_row_stride,
        formats, v_skip_threshold, plan_context);
  else
    Turbo4VerifyWmmaSplitKernel<kSlot, false><<<grid, 256, 0, stream>>>(
        q_rot, k_plane, v_plane, partials, plane_bytes, partial_bytes,
        num_heads, num_kv_heads, first_context, rows, max_context, max_splits,
        scale, write_status, status, q_rot_row_stride, partials_row_stride,
        formats, 0.0f, plan_context);
}

// VRD6 GUFO_DEV_VRD6=1: K = V = Turbo6 verify reads take
// Turbo4VerifyVrd6Kernel (checked before GUFO_DEV_ATTN6_V3). Unset: the code
// path below is unchanged. Read here, not in dev_sync.hpp, so no other
// worker's file changes.
[[nodiscard]] bool GufoDevVrd6() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("GUFO_DEV_VRD6");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}
// GUFO_DEV_VRD6_SPLITS=N: at most N splits per KV head (Attn6SplitSize).
// Unset = 10 (40 CTAs over 4 KV heads: one CTA wave); 0 = the WMMA plan.
[[nodiscard]] std::uint32_t GufoDevVrd6Splits() noexcept {
  static const std::uint32_t n = [] {
    const char* v = std::getenv("GUFO_DEV_VRD6_SPLITS");
    return v == nullptr
               ? 10U
               : static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
  }();
  return n;
}

}  // namespace

Turbo4LaunchStatus LaunchTurbo4AttentionVerifyWmma(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats) {
  // The reader now serves the qh / padded-slot formats (Turbo5/6)
  // through Turbo4DecodePlaneLane-style decode; only an invalid format is
  // refused here. The caller gates Turbo5/6 behind GUFO_DEV_VERIFY_WMMA_WIDE.
  if (!formats.Valid())
    return Turbo4LaunchStatus::kGeometryRefused;
  // VRD6: same fall-through contract as ATTN6 below.
  if (GufoDevVrd6()) {
    const Turbo4LaunchStatus vrd6 = LaunchTurbo4AttentionVerifyVrd6(
        q, k_plane, v_plane, out, q_rot, out_rot, partials, plane_bytes,
        q_bytes, out_bytes, q_rot_bytes, out_rot_bytes, partial_bytes, gate,
        gate_bytes, num_heads, num_kv_heads, head_dim, first_valid_context,
        rows, max_context, scale, d_write_status, d_status, stream, formats,
        GufoDevVrd6Splits());
    if (vrd6 != Turbo4LaunchStatus::kGeometryRefused)
      return vrd6;
  }
  // ATTN6: K = V = Turbo6 takes the restructured reader when asked; it
  // refuses (enqueuing nothing) every geometry it does not serve, which then
  // falls through to this reader unchanged.
  if (GufoDevAttn6V3()) {
    // ATTN FUSE (GUFO_DEV_ATTN_FUSE=1): attn6 as one launch; a geometry the
    // fused kernel refuses takes the three-launch attn6.
    // ATTN B (GUFO_DEV_ATTN_B=1) first: the fused kernel plus B step 2a.
    for (const std::uint32_t stages :
         {kAttn6StageFusedB, kAttn6StageFused, kAttn6StagesAll}) {
      if ((stages == kAttn6StageFusedB && !GufoDevAttnB()) ||
          (stages == kAttn6StageFused && !GufoDevAttnFuse()))
        continue;
      const Turbo4LaunchStatus attn6 = LaunchTurbo4AttentionVerifyAttn6(
          q, k_plane, v_plane, out, q_rot, out_rot, partials, plane_bytes,
          q_bytes, out_bytes, q_rot_bytes, out_rot_bytes, partial_bytes, gate,
          gate_bytes, num_heads, num_kv_heads, head_dim, first_valid_context,
          rows, max_context, scale, d_write_status, d_status, stream, formats,
          GufoDevAttn6Splits(), stages);
      if (attn6 != Turbo4LaunchStatus::kGeometryRefused)
        return attn6;
    }
  }
  if (q == nullptr || k_plane == nullptr || v_plane == nullptr ||
      out == nullptr || q_rot == nullptr || out_rot == nullptr ||
      partials == nullptr || d_write_status == nullptr || d_status == nullptr ||
      plane_bytes == 0 || num_heads == 0 || num_kv_heads == 0 || rows == 0)
    return Turbo4LaunchStatus::kBadArguments;
  if (max_context == 0 || first_valid_context == 0 || rows > kVerifyMaxRows ||
      rows > max_context || first_valid_context > max_context - rows + 1U)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (head_dim != kSplitHeadDim ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv || !std::isfinite(scale))
    return Turbo4LaunchStatus::kGeometryRefused;
  if ((gate_bytes == 0U) != (gate == nullptr))
    return Turbo4LaunchStatus::kBadArguments;

  const std::size_t vector_floats =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t rows_bytes = rows * vector_floats * sizeof(float);
  const std::size_t row_partial_bytes = Turbo4BatchedDecodePartialBytes(
      first_valid_context, rows, num_heads, head_dim);
  const std::size_t required_plane =
      Turbo4LayerPlaneBytes(head_dim, num_kv_heads, max_context,
                            formats.stride);
  if (required_plane == 0U || plane_bytes < required_plane ||
      row_partial_bytes == 0U || partial_bytes < rows * row_partial_bytes ||
      q_bytes < rows_bytes || out_bytes < rows_bytes ||
      q_rot_bytes < rows_bytes || out_rot_bytes < rows_bytes ||
      (gate != nullptr && gate_bytes < rows_bytes))
    return Turbo4LaunchStatus::kBadArguments;

  // Per-row slice = the largest split count over the rows; this kernel's shared
  // plan (the last row's count) never exceeds it.
  const std::uint32_t max_splits = static_cast<std::uint32_t>(
      row_partial_bytes / (num_heads * (head_dim + 2) * sizeof(float)));
  // The grid must agree with the kernel's own split_count, so it is derived
  // from the same (possibly pinned) split size.
  const std::uint32_t wmma_last_context = first_valid_context + rows - 1U;
  const std::uint32_t wmma_plan_context =
      GufoDevBatchInvariant()
          ? BatchInvariantPlanContext(first_valid_context, rows, max_context)
          : wmma_last_context;
  const std::uint32_t wmma_split_size =
      static_cast<std::uint32_t>(Turbo4DecodeSplitSize(wmma_plan_context));
  const std::uint32_t split_count =
      wmma_split_size == 0U
          ? 0U
          : (wmma_last_context + wmma_split_size - 1U) / wmma_split_size;
  if (max_splits == 0 || max_splits > kSplitHeadDim || split_count == 0 ||
      split_count > max_splits)
    return Turbo4LaunchStatus::kGeometryRefused;

  Turbo4RotateQueryKernel<<<dim3(num_heads, rows), turbo4::kBlockValues, 0,
                            stream>>>(q, q_rot, num_heads, head_dim, q_bytes,
                                      q_rot_bytes, d_status, vector_floats);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;

  const std::size_t partials_row_stride = row_partial_bytes / sizeof(float);
  const float v_skip_threshold = GufoDevVSkipThreshold();
  const bool v_skip = v_skip_threshold > 0.0f;
  const dim3 verify_grid(num_kv_heads, split_count);
  switch (formats.stride) {
    case turbo4::Bits::k5:
      LaunchTurbo4VerifyWmmaStage<turbo4::Bits::k5>(
          v_skip, v_skip_threshold, q_rot, k_plane, v_plane, partials,
          plane_bytes, row_partial_bytes, num_heads, num_kv_heads,
          first_valid_context, rows, max_context, max_splits, scale,
          d_write_status, d_status, vector_floats, partials_row_stride, formats,
          verify_grid, stream);
      break;
    case turbo4::Bits::k6:
      LaunchTurbo4VerifyWmmaStage<turbo4::Bits::k6>(
          v_skip, v_skip_threshold, q_rot, k_plane, v_plane, partials,
          plane_bytes, row_partial_bytes, num_heads, num_kv_heads,
          first_valid_context, rows, max_context, max_splits, scale,
          d_write_status, d_status, vector_floats, partials_row_stride, formats,
          verify_grid, stream);
      break;
    default:
      LaunchTurbo4VerifyWmmaStage<turbo4::Bits::k4>(
          v_skip, v_skip_threshold, q_rot, k_plane, v_plane, partials,
          plane_bytes, row_partial_bytes, num_heads, num_kv_heads,
          first_valid_context, rows, max_context, max_splits, scale,
          d_write_status, d_status, vector_floats, partials_row_stride, formats,
          verify_grid, stream);
      break;
  }
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;

  Turbo4AttentionReduceInverseKernel<<<dim3(num_heads, rows), kSplitHeadDim,
                                       max_splits * sizeof(float), stream>>>(
      partials, out_rot, out, gate, row_partial_bytes, out_rot_bytes, out_bytes,
      gate_bytes, num_heads, num_kv_heads, split_count, max_splits, head_dim,
      d_status, first_valid_context, partials_row_stride, vector_floats,
      /*uniform_splits=*/true);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

// ---------------------------------------------------------------------------
// ATTN6 (GUFO_DEV_ATTN6_V3=1): K = V = Turbo6 verify reader, restructured for
// issue rate. Same arithmetic as the pipelined WMMA reader above
// (Turbo4VerifyWmmaSplitKernel<k6, false, true>), element for element and in
// the same order, so with the same split plan its partial records are bitwise
// identical:
//   - K/V value = __float2half(centroid_f32[code] * norm), same table, same
//     poison rule (both planes' norms of the (token, block));
//   - S(qt, kb) = WMMA over ksteps 0..15 in order on one accumulator, * scale;
//   - softmax: same 4-threads-per-row ownership, same expf, same f16 P, same
//     part_sum tree (sequential 8, then xor 1, xor 2);
//   - O[qt][t] *= row_scale, then WMMA kb 0, kb 1.
// What changes is who holds what, and how often the CTA waits:
//   1. 16 waves, two roles with their own loops (so their registers do not
//      add up). S group = waves 0-7: decodes K (slot = 32-dim group), slots
//      0-5 run S with Q held in registers (16 f16 fragments, 128 VGPRs) and
//      the softmax. PV group = waves 8-15: decodes V, owns O (dim tiles
//      w-8 and w), runs PV. Q leaves LDS (25 KB), so K (32 x 256) and V^T
//      (256 x 32) are staged whole, not in two 128-dim chunks.
//   2. 4 barriers per 32-key tile instead of 10, and the groups overlap: the
//      PV group decodes V while the S group runs S, and runs PV(t) while the
//      S group decodes K(t + 1).
//   3. Code decode: Attn6PackCodes builds the table byte offsets four at a
//      time (~1 op per value + 1 extract) instead of ~5 ops per value.
//   4. Softmax: branch-free mask (no exec-masked LDS load per element), two
//      ds_load_b128 per thread, quad DPP instead of ds_bpermute shuffles.
//   5. Split plan is a parameter (Attn6SplitSize): 0 keeps the WMMA plan;
//      N caps splits per KV head (one CTA wave on the GPU, fewer partials).
// LDS (static): K tile / S tile union 32 x 264 f16 = 16896, V^T 256 x 40 f16 =
// 20480, P 48 x 40 f16 = 3840, row_scale 192, centroids 256, error 4 -> 41668 B.
// VGPR budget <= 192 (amdgpu_waves_per_eu 8): 2 CTAs = 32 waves per WGP.
// Barrier ledger (both loops execute B1..B4 once per tile, same order):
//   S group : decode K(t) | B1 | S(t)        | B2 | S -> LDS | B3 | softmax | B4
//   PV group:             | B1 | decode V(t) | B2 |          | B3 |         | B4 | PV(t)
// K(t+1) written after B4(t): every S(t) read (before B2) and softmax read
// (before B4) is done. V^T(t+1) written after B1(t+1), which no PV wave
// reaches before finishing PV(t). P / row_scale(t+1) written after B3(t+1),
// same argument. S tile written after B2: every K read is done.
// ponytail: the S-wave rotation (slot) assumes waves go to SIMD wave % 4 and
// that even/odd splits share WGPs; a profile would show if neither holds.
namespace {

constexpr std::uint32_t kA6Threads = 512;
constexpr std::uint32_t kA6KStride = kSplitHeadDim + 8;  // K tile rows (halves)
constexpr std::uint32_t kA6KHalves = kVerifyKeys * kA6KStride;
constexpr std::uint32_t kA6VtStride = kVerifyKeys + 8;  // V^T rows (halves)
// S tile rows: 48 floats, so the two half-waves of a WMMA C store (rows 2i and
// 2i + 1, 16 columns each) hit disjoint banks, and rows are 16-byte aligned
// for the softmax's two b128 loads.
constexpr std::uint32_t kA6SStride = 48;
constexpr std::uint32_t kA6KSteps = kSplitHeadDim / 16U;
static_assert(kVerifyQVecs * kA6SStride * sizeof(float) <=
                  kA6KHalves * sizeof(__half),
              "the S tile reuses the K tile buffer");
static_assert(kA6KStride % 8 == 0 && kA6VtStride % 8 == 0 &&
                  kA6SStride % 4 == 0,
              "fragment and b128 rows start on 16 bytes");

/// LDS-only barrier (CK block_sync_lds): the next tile's raw K/V loads are in
/// flight across every barrier of this tile, and __syncthreads' workgroup
/// fence would wait for them (vmcnt 0). Only LDS is shared inside the loop.
__device__ __forceinline__ void Attn6Barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/// DequantTurbo4Slice<k6> with the codes from Attn6PackCodes: the same
/// table entry and the same `__float2half(c * norm)` for every value.
///
/// kRep (ATTN_C L1): the table holds 32 interleaved copies, entry
/// [code * 32 + lane] (8 KB instead of 256 B), so lane l always lands in LDS
/// bank l whatever its code is (the FLUTE / QuIP# duplication rule). A single
/// 64-dword table spans 64 banks, but a wave reaches only its SIMD pair's
/// 32-bank set, so codes c and c + 32 collide and a real code stream costs
/// ~4 ways. P3 (attnB_20261025/probes/p3.log, real turbo6 code streams,
/// 8 waves/SIMD = the in-situ occupancy, packed mode): 7.95 -> 4.01 cycles
/// per lookup per SIMD. `lane_dword` is 4 * lane, hoisted to the caller, so
/// the whole extra cost is one address op per value and every loaded float is
/// the same bits at the same point in the same order.
template <bool kRep>
__device__ __forceinline__ void Attn6Dequant32(const Turbo4Slice& slice,
                                               const float* centroids,
                                               std::uint32_t lane_dword,
                                               __half* out) {
  std::uint32_t words[2][4];
  Attn6PackCodes(slice.w, slice.qh, words);
  const char* const table = reinterpret_cast<const char*>(centroids);
#pragma unroll
  for (std::uint32_t h = 0; h < 2; ++h)
#pragma unroll
    for (std::uint32_t k = 0; k < 4; ++k)
#pragma unroll
      for (std::uint32_t r = 0; r < 4; ++r) {
        std::uint32_t offset = (words[h][r] >> (8U * k)) & 0xFFU;
        if constexpr (kRep)  // 4 * (32 * code + lane)
          offset = (offset << 5U) | lane_dword;
        const float c = *reinterpret_cast<const float*>(table + offset);
        out[16U * h + 4U * k + r] = __float2half(c * slice.norm);
      }
}

/// __shfl_xor(v, 1 / 2) inside an aligned group of four lanes, as quad DPP
/// (no LDS round trip). Same value as the shuffle; every lane is active.
/// Dwords of the ATTN_C L1 centroid table: 32 copies of the 64 entries.
inline constexpr std::uint32_t kA6CentroidDwords = 64U * 32U;

__device__ __forceinline__ float Attn6QuadXor1(float v) {
  return __int_as_float(
      __builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));
}
__device__ __forceinline__ float Attn6QuadXor2(float v) {
  return __int_as_float(
      __builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false));
}

/// One plane's slice of a (token, 32-dim group) plus the other plane's norm
/// word, which the poison rule needs (Turbo6Unpack zeroes both if either norm
/// is unusable).
struct Attn6Raw {
  Turbo6RawSlice self;
  Turbo6RawSlice other;  // only .norm is loaded / read
};

// ---------------------------------------------------------------------------
// ATTN FUSE (GUFO_DEV_ATTN_FUSE=1 with GUFO_DEV_ATTN6_V3=1; attnB): attn6 as
// ONE launch. Turbo4RotateQueryKernel moves into a prologue of every
// (kv head, split) CTA and Turbo4AttentionReduceInverseKernel into the last
// CTA of each kv head to finish (arrival counter). The main body is attn6's,
// unchanged, so partials are attn6's bit for bit. Contract: on a kOk launch
// partials, out_rot and out are bitwise attn6's; on a failed launch the
// status is attn6's for any single failure class, the buffers are not
// (attn6 writes none of them after a failure; fused CTAs of other kv heads
// may already have). Why out_rot / out stay bitwise equal (CPU proof:
// attnB_20261025/fuse_proof.cpp):
//  - rotate: the same 7-stage butterfly network on the same values. A
//    butterfly output is one IEEE add or subtract of the same two operands
//    (lower = a + b, upper = a - b, a = the lower element), whichever lane
//    computes it, and stages run in the same order (stride 1 .. 64). Signs
//    are exact (+-1); the last step is x * (kInvSqrt128 * sign) as written
//    there. Then __float2half, as attn6 did on q_rot.
//  - reduce, one wave per (row, q head): max is order-free (fmaxf is exact;
//    the records are finite-checked first; a +-0 max gives the same
//    expf(m - max)); the weights are the same expf(m - max); the denominator
//    is the same LEFT fold over splits 0..n-1 as fma(w, l, d) (the reduce
//    kernel's thread 0 compiles it to v_fmac_f32); every numerator is the
//    same left fold over splits as fma(w, o, num) (v_fmac_f32 there too);
//    the same IEEE division; the inverse WHT as the rotate; the same gate.
//  - status: the same codes at the same checks. Only when two DIFFERENT
//    failure classes hit one launch can the latched maximum differ (attn6's
//    three launches already order those by launch, not by value).
// Arrival: every CTA past the geometry check arrives exactly once (also on
// its early-exit paths), after a release fence that covers its partial
// stores and its status latch. atomicInc(limit n - 1) returns n - 1 to
// exactly one CTA and stores 0, so the counter is clean for the next launch
// without a memset. The last CTA acquires, then reduces this kv head's
// 6 x rows items. Every geometry refusal in the kernel is grid-uniform, so
// no completed launch leaves a partial count. A count found above n - 1
// (only a concurrent or aborted launch can leave one) latches
// kGeometryRefused; it does NOT heal (the counter then rests at n - 1 and
// later launches reduce early): after such a failure the process must not
// trust fused results again (restart). Not reachable on the served path,
// which runs one verify launch at a time on one stream.
// ponytail: one counter per kv head in a static device array: one fused
// launch in flight per device. Two streams running it at once would need
// per-call counters (e.g. in the caller's workspace).
constexpr std::uint32_t kAttn6FuseMaxKvHeads = 64;
__device__ unsigned int g_attn6_fuse_arrivals[kAttn6FuseMaxKvHeads];
// Both reuse attn6's K tile (kv_lds) and V^T (vt_lds) buffers. Before the
// loop the q tile: 128-value block 0 of every vector in kv_lds, block 1 in
// vt_lds, rows of kA6FuseQStride halves. After it the reduce's per-wave
// w / l arrays (2 x 256 floats, any split count the launcher allows):
// waves 0-7 in kv_lds, waves 8-15 in vt_lds.
constexpr std::uint32_t kA6FuseQStride = turbo4::kBlockValues + 8U;
constexpr std::uint32_t kA6VtHalves = kSplitHeadDim * kA6VtStride;
static_assert(kVerifyQVecs * kA6FuseQStride <= kA6KHalves &&
                  kVerifyQVecs * kA6FuseQStride <= kA6VtHalves &&
                  kA6FuseQStride % 8U == 0U,
              "q tile halves fit, rows on 16 bytes");
static_assert(8U * 2U * kSplitHeadDim * sizeof(float) <=
                      kA6KHalves * sizeof(__half) &&
                  8U * 2U * kSplitHeadDim * sizeof(float) <=
                      kA6VtHalves * sizeof(__half),
              "reduce scratch fits");

struct Attn6FuseArgs {
  float* out_rot = nullptr;
  float* out = nullptr;
  const float* gate = nullptr;   // null = no gate
  std::size_t vector_row_stride = 0;  // floats between rows of out / gate
};

/// The status word, wave-uniform (one lane's load), so early exits on it
/// never split a wave that shuffles afterwards.
__device__ __forceinline__ unsigned int Attn6FuseStatus(
    Turbo4KernelStatus* status) {
  return static_cast<unsigned int>(__builtin_amdgcn_readfirstlane(
      static_cast<int>(__hip_atomic_load(
          reinterpret_cast<unsigned int*>(status), __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT))));
}

/// The WHT butterflies of one 128-value block held kV values per lane:
/// element e = kV * (lane % (128 / kV)) + j is v[j]. Strides below kV are
/// inside the lane, the rest cross lanes (partner = lane ^ (stride / kV)).
/// Stride order 1, 2, .., 64 as Turbo4RotateQueryKernel.
template<std::uint32_t kV>
__device__ __forceinline__ void Attn6FuseButterflies(float (&v)[kV],
                                                     std::uint32_t lane) {
#pragma unroll
  for (std::uint32_t s = 1; s < kV; s *= 2U)
#pragma unroll
    for (std::uint32_t j = 0; j < kV; ++j)
      if ((j & s) == 0U) {
        const float a = v[j];
        const float b = v[j + s];
        v[j] = a + b;
        v[j + s] = a - b;
      }
#pragma unroll
  for (std::uint32_t s = kV; s < turbo4::kBlockValues; s *= 2U) {
    const std::uint32_t x = s / kV;
    const bool upper = (lane & x) != 0U;
#pragma unroll
    for (std::uint32_t j = 0; j < kV; ++j) {
      const float p = __shfl_xor(v[j], static_cast<int>(x), 32);
      v[j] = upper ? p - v[j] : v[j] + p;
    }
  }
}

/// Prologue: this CTA's 6 x rows query vectors, rotated exactly as
/// Turbo4RotateQueryKernel, stored as __float2half: block 0 into q_lo, block
/// 1 into q_hi (row m = vector m, kA6FuseQStride halves). One wave per (vector, 128-block), 4 values
/// per lane; a wave's (at most 6) blocks are all loaded before the first is
/// rotated, so the prologue pays one L2 round trip, not six. A non-finite q
/// value latches kGeometryRefused (the rotate kernel's code) into s_error
/// and the status word.
constexpr std::uint32_t kA6FuseQItemsPerWave =
    2U * kVerifyQVecs / (kA6Threads / 32U);
static_assert(kA6FuseQItemsPerWave * (kA6Threads / 32U) == 2U * kVerifyQVecs,
              "q blocks tile the waves");
__device__ __forceinline__ void Attn6FuseRotateQ(
    const float* q, std::size_t row_stride, std::uint32_t kv_head,
    std::uint32_t nvec, std::uint32_t wave, std::uint32_t lane, __half* q_lo,
    __half* q_hi, unsigned int* s_error, Turbo4KernelStatus* status) {
  float4 f[kA6FuseQItemsPerWave];
#pragma unroll
  for (std::uint32_t k = 0; k < kA6FuseQItemsPerWave; ++k) {
    const std::uint32_t item = wave + k * (kA6Threads / 32U);
    const std::uint32_t m = item >> 1U;
    f[k] = item < 2U * nvec
               ? *reinterpret_cast<const float4*>(
                     q + static_cast<std::size_t>(m / kSplitQHeadsPerKv) *
                             row_stride +
                     (static_cast<std::size_t>(kv_head) * kSplitQHeadsPerKv +
                      m % kSplitQHeadsPerKv) *
                         kSplitHeadDim +
                     (item & 1U) * turbo4::kBlockValues + 4U * lane)
               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  bool bad = false;
#pragma unroll
  for (std::uint32_t k = 0; k < kA6FuseQItemsPerWave; ++k) {
    const std::uint32_t item = wave + k * (kA6Threads / 32U);
    if (item >= 2U * nvec)  // wave-uniform
      break;
    const std::uint32_t m = item >> 1U;
    const std::uint32_t blk = item & 1U;
    float v[4] = {f[k].x, f[k].y, f[k].z, f[k].w};
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      bad = bad || !Turbo4IsFinite(v[j]);
      v[j] *= turbo4::TURBO_WHT_SIGNS1[4U * lane + j];
    }
    Attn6FuseButterflies<4>(v, lane);
    __attribute__((aligned(8))) __half h[4];
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      float r = v[j] * (turbo4::kInvSqrt128 *
                        turbo4::TURBO_WHT_SIGNS2[4U * lane + j]);
      // Round to f32 FIRST, as the rotate kernel does (it stores q_rot as
      // f32; attn6 converts that). Without this pin the compiler fuses the
      // multiply and the conversion into v_fma_mix*_f16 (ONE rounding,
      // straight to f16), which differs on f16 ties: GPU gate 10-25 12:10,
      // 1 query vector of 48 off. No instruction, no value change.
      asm volatile("" : "+v"(r));
      h[j] = __float2half(r);
    }
    *reinterpret_cast<uint2*>(&(blk == 0U ? q_lo : q_hi)[m * kA6FuseQStride +
                                                         4U * lane]) =
        *reinterpret_cast<const uint2*>(h);
  }
  if (__any(bad) && lane == 0U) {
    atomicMax(s_error,
              static_cast<unsigned int>(Turbo4KernelStatus::kGeometryRefused));
    Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
  }
}

/// One (row, q head) of Turbo4AttentionReduceInverseKernel (uniform splits)
/// on one wave: lane = split for the max and the weights, lane = 8 output
/// dims for the numerators and the inverse WHT. w_lds / l_lds: this wave's
/// n floats each. Same checks and codes as that kernel. Its mid-kernel
/// re-reads of the status word are dropped (the caller checks it once):
/// they only decide which of two DIFFERENT codes wins a race, never whether
/// a single failure class is reported.
/// Latency: everything that does not need the weights (the (m, l) heads,
/// the first kBatch splits' numerator words, the gate) is loaded first, so
/// an item costs ceil(n / kBatch) L2 round trips (4 at 32k), not ~n.
__device__ __forceinline__ void Attn6FuseReduceItem(
    const float* partials, const Attn6FuseArgs& fz, Turbo4KernelStatus* status,
    std::uint32_t num_heads, std::uint32_t n, std::size_t partials_row_stride,
    std::uint32_t row, std::uint32_t head, std::uint32_t lane, float* w_lds,
    float* l_lds) {
  constexpr std::uint32_t kBatch = 8;
  const std::size_t rec_stride =
      static_cast<std::size_t>(num_heads) * kSplitRecordWidth;
  const float* const base = partials +
                            static_cast<std::size_t>(row) * partials_row_stride +
                            static_cast<std::size_t>(head) * kSplitRecordWidth;
  const float* const o = base + 2U + 8U * lane;
  const std::size_t at = static_cast<std::size_t>(row) * fz.vector_row_stride +
                         static_cast<std::size_t>(head) * kSplitHeadDim +
                         8U * lane;
  // --- loads ---
  float x[kBatch][8];
#pragma unroll
  for (std::uint32_t u = 0; u < kBatch; ++u)
    if (u < n)
      __builtin_memcpy(x[u], __builtin_assume_aligned(o + u * rec_stride, 8),
                       sizeof(x[u]));
  float g[8] = {};
  if (fz.gate != nullptr)
    __builtin_memcpy(g, fz.gate + at, sizeof(g));
  float maximum = -3.402823466e+38f;
  bool bad = false;
  for (std::uint32_t s = lane; s < n; s += 32U) {
    const float m = base[s * rec_stride];
    const float l = base[s * rec_stride + 1U];
    if (!Turbo4IsFinite(m) || !Turbo4IsFinite(l) || l < 0.0f)
      bad = true;
    else
      maximum = fmaxf(maximum, m);
    w_lds[s] = m;
    l_lds[s] = l;
  }
  // --- max, weights, denominator ---
  if (__any(bad)) {
    if (lane == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kDegenerateSoftmax);
    return;
  }
#pragma unroll
  for (std::uint32_t xm = 16; xm > 0; xm >>= 1U)
    maximum = fmaxf(maximum, __shfl_xor(maximum, static_cast<int>(xm), 32));
  for (std::uint32_t s = lane; s < n; s += 32U) {
    const float w = expf(w_lds[s] - maximum);
    bad = bad || !Turbo4IsFinite(w);
    w_lds[s] = w;
  }
  if (__any(bad)) {
    if (lane == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kDegenerateSoftmax);
    return;
  }
  // Same wave: its LDS writes above land before these reads (in order).
  float denominator = 0.0f;
  for (std::uint32_t s = 0; s < n; ++s)
    denominator = fmaf(w_lds[s], l_lds[s], denominator);
  if (__any(!Turbo4IsFinite(denominator) || denominator <= 0.0f)) {
    if (lane == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kDegenerateSoftmax);
    return;
  }
  // --- numerators: left folds over splits 0..n-1, batch by batch ---
  float v[8] = {};
  for (std::uint32_t s0 = 0;; ) {
#pragma unroll
    for (std::uint32_t u = 0; u < kBatch; ++u) {
      if (s0 + u >= n)
        break;
      const float w = w_lds[s0 + u];
#pragma unroll
      for (std::uint32_t j = 0; j < 8; ++j)
        v[j] = fmaf(w, x[u][j], v[j]);
    }
    s0 += kBatch;
    if (s0 >= n)
      break;
#pragma unroll
    for (std::uint32_t u = 0; u < kBatch; ++u)
      if (s0 + u < n)
        __builtin_memcpy(x[u],
                         __builtin_assume_aligned(o + (s0 + u) * rec_stride, 8),
                         sizeof(x[u]));
  }
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j) {
    v[j] = v[j] / denominator;
    bad = bad || !Turbo4IsFinite(v[j]);
  }
  if (__any(bad)) {
    if (lane == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kDegenerateSoftmax);
    return;
  }
  reinterpret_cast<float4*>(fz.out_rot + at)[0] =
      make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(fz.out_rot + at)[1] =
      make_float4(v[4], v[5], v[6], v[7]);
  // --- inverse WHT, gate ---
  const std::uint32_t e0 = 8U * (lane & 15U);  // element of v[0] in its block
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j)
    v[j] *= turbo4::TURBO_WHT_SIGNS2[e0 + j];
  Attn6FuseButterflies<8>(v, lane);
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j)
    v[j] *= turbo4::kInvSqrt128 * turbo4::TURBO_WHT_SIGNS1[e0 + j];
  if (fz.gate != nullptr)
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j)
      bad = bad || !Turbo4IsFinite(g[j]);
  if (__any(bad)) {
    if (lane == 0U)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j)
    fz.out[at + j] =
        fz.gate == nullptr ? v[j] : v[j] * Turbo4SigmoidGate(g[j]);
}

/// Arrival + (last CTA of this kv head only) the reduce. Every thread of the
/// CTA calls it once, at one point, after its last partial store and latch.
/// scratch_lo / scratch_hi = the CTA's K tile / V^T LDS (free by now):
/// 512 floats per wave, waves 0-7 / 8-15.
__device__ __forceinline__ void Attn6FuseTail(
    const float* partials, const Attn6FuseArgs& fz, Turbo4KernelStatus* status,
    std::uint32_t num_heads, std::uint32_t kv_head, std::uint32_t split_count,
    std::uint32_t rows, std::size_t partials_row_stride, std::uint32_t tid,
    float* scratch_lo, float* scratch_hi) {
  __shared__ unsigned int s_last;
  __threadfence();  // release: this thread's partial stores and latches
  __syncthreads();
  if (tid == 0U) {
    const unsigned int seen =
        atomicInc(&g_attn6_fuse_arrivals[kv_head], split_count - 1U);
    if (seen > split_count - 1U)  // stale count: never a silent early reduce
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    s_last = seen == split_count - 1U;
  }
  __syncthreads();
  if (s_last == 0U)
    return;
  __threadfence();  // acquire: every split's partials and latches
  // attn6's reduce does nothing once any CTA of the launch latched a code.
  if (Attn6FuseStatus(status) != 0U)
    return;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t lane = tid & 31U;
  float* const w_lds =
      (wave < 8U ? scratch_lo : scratch_hi) + (wave & 7U) * 2U * kSplitHeadDim;
  for (std::uint32_t item = wave; item < rows * kSplitQHeadsPerKv;
       item += kA6Threads / 32U)
    Attn6FuseReduceItem(partials, fz, status, num_heads, split_count,
                        partials_row_stride, item / kSplitQHeadsPerKv,
                        kv_head * kSplitQHeadsPerKv + item % kSplitQHeadsPerKv,
                        lane, w_lds, w_lds + kSplitHeadDim);
}

/// kC (GUFO_DEV_ATTN_C=1): the three bit-exact cuts of attnC.
///   L1 centroid table: 32 interleaved copies, conflict-free (Attn6Dequant32).
///   L2 PV fragments: the A operand p_lds(qt, kb) and the B operand
///      vt_lds(t, kb) were re-read for all 24 (t, qt, kb) steps though only 10
///      distinct fragments exist. Hoisted, q tile outer: 48 -> 20
///      ds_load_b128 and 24 -> 12 ds_load_2addr_b32 per PV wave per tile.
///   L3 softmax: gated to the live rows instead of all 48, with the padding
///      rows' P / row_scale pre-filled once.
/// L2 and L3 are the same two changes ATTN_B makes inside the fused kernel,
/// reached here without the fused launch; L1 is attnC's own and stacks with
/// ATTN_B (see Attn6Body). No float operation, operand, or per-accumulator
/// order changes, so partials and out are bitwise attn6's.
/// kC = false must stay ISA-identical to the base kernel.
template <bool kC>
__global__ __launch_bounds__(kA6Threads)
__attribute__((amdgpu_waves_per_eu(8))) void Turbo4VerifyAttn6Kernel(
    const float* __restrict__ q_rot, const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, float* __restrict__ partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t max_splits,
    std::uint32_t split_size, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    std::size_t q_rot_row_stride, std::size_t partials_row_stride,
    turbo4::PlaneFormats formats) {
  // K tile [key][d] during decode + S, then the S tile [q][key] (fp32).
  __shared__ __attribute__((aligned(16))) __half kv_lds[kA6KHalves];
  __shared__ __attribute__((
      aligned(16))) __half vt_lds[kSplitHeadDim * kA6VtStride];
  __shared__
      __attribute__((aligned(16))) __half p_lds[kVerifyQVecs][kVerifyVtStride];
  __shared__ float row_scale[kVerifyQVecs];
  __shared__ float s_centroids[kC ? kA6CentroidDwords : 64U];
  __shared__ unsigned int s_error;
  float(*const s_lds)[kA6SStride] =
      reinterpret_cast<float(*)[kA6SStride]>(kv_lds);

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;

  const std::uint32_t last_context = first_context + rows - 1U;
  const std::uint32_t split_count =
      split_size == 0U ? 0U : (last_context + split_size - 1U) / split_size;
  const std::size_t required_bytes = static_cast<std::size_t>(max_splits) *
                                     num_heads * kSplitRecordWidth *
                                     sizeof(float);
  if (blockDim.x != kA6Threads ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv ||
      kv_head >= num_kv_heads || rows == 0 || rows > kVerifyMaxRows ||
      first_context == 0 || last_context > max_context || split_size == 0U ||
      split_size % kVerifyKeys != 0U || split_count > max_splits ||
      split >= split_count ||
      !VerifyT6PipeFits(formats, num_kv_heads, max_context) ||
      plane_bytes < Turbo4LayerPlaneBytes(kSplitHeadDim, num_kv_heads,
                                          max_context, turbo4::Bits::k6) ||
      partial_bytes < required_bytes || q_rot == nullptr ||
      partials == nullptr || status == nullptr || write_status == nullptr) {
    if (tid == 0)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  if (tid == 0) {
    s_error = static_cast<unsigned int>(*status);
    if (*write_status != Turbo4KernelStatus::kOk) {
      s_error = static_cast<unsigned int>(Turbo4KernelStatus::kWriterFailed);
      Turbo4LatchFailure(status, Turbo4KernelStatus::kWriterFailed);
    }
  }
  if constexpr (kC) {
    // 32 copies: entry [code * 32 + lane]. 4 iterations of 512 threads, once
    // per CTA against 32 tiles of lookups.
#pragma unroll
    for (std::uint32_t i = tid; i < kA6CentroidDwords; i += kA6Threads)
      s_centroids[i] = turbo4::CentroidValue(turbo4::Bits::k6, i >> 5U);
  } else if (tid < 64U) {
    s_centroids[tid] = turbo4::CentroidValue(turbo4::Bits::k6, tid);
  }
  __syncthreads();
  if (s_error != 0U)
    return;

  const std::uint32_t nvec = rows * kSplitQHeadsPerKv;
  // L3: the online softmax runs all kVerifyQVecs = 48 rows whatever `nvec`
  // is, though at rows < 8 only nvec of them are query vectors. A padding
  // row's O accumulator is never written to a partial record (the record
  // writes are gated on `rg < nvec` and `m >= nvec`), so skipping its softmax
  // changes no output -- but PV still reads P and row_scale for every row of
  // a live q tile, so zero them once here instead of reading whatever the
  // last CTA left in LDS. Worth 0 at rows 8 (nvec = 48) and ~12% of the tile
  // at rows 1, where 5 of the 6 softmax slots go idle.
  const std::uint32_t soft_rows = kC ? nvec : kVerifyQVecs;
  if constexpr (kC) {
    for (std::uint32_t i = tid; i < (kVerifyQVecs - nvec) * kVerifyVtStride;
         i += kA6Threads)
      p_lds[nvec + i / kVerifyVtStride][i % kVerifyVtStride] =
          __float2half(0.0f);
    for (std::uint32_t r = nvec + tid; r < kVerifyQVecs; r += kA6Threads)
      row_scale[r] = 0.0f;
    // Visible to the PV group by B1 of the first tile (Attn6Barrier is an
    // LDS release/acquire + s_barrier, and PV reads P only after B4).
  }
  const bool s_group = wave < 8U;
  // S-group role slot: the S / softmax units (slots 0-5) sit on waves 0-5 for
  // even splits and on waves 2-7 for odd ones, so across co-resident CTAs no
  // SIMD (wave % 4) carries two S waves every time. Any bijection works: the
  // arithmetic per unit does not depend on which wave runs it.
  const std::uint32_t slot = (wave + 8U - 2U * (split & 1U)) & 7U;
  // Decode ownership: position pos, dims [32 * dgroup, 32 * dgroup + 32) of
  // K (S group) or V (PV group).
  const std::uint32_t pos = lane;
  const std::uint32_t dgroup = s_group ? slot : wave - 8U;
  const std::uint32_t begin = split * split_size;
  const std::uint32_t end =
      last_context < begin + split_size ? last_context : begin + split_size;

  // Raw-word prefetch of the next tile, the kT6Pipe instance's addressing
  // (VerifyT6PipeFits makes every offset 32-bit).
  const std::uint32_t t6_token_bytes = static_cast<std::uint32_t>(
      Turbo4BlocksPerToken(kSplitHeadDim, num_kv_heads) *
      turbo4::BlockBytes(turbo4::Bits::k6));
  const std::uint32_t t6_norm_off =
      kv_head * static_cast<std::uint32_t>(
                    Turbo4BlocksPerHead(kSplitHeadDim) *
                    turbo4::BlockBytes(turbo4::Bits::k6)) +
      (dgroup / 4U) *
          static_cast<std::uint32_t>(turbo4::BlockBytes(turbo4::Bits::k6));
  const std::uint32_t t6_nib_off = t6_norm_off + 2U + 16U * (dgroup & 3U);
  const std::uint32_t t6_qh_off =
      t6_norm_off + static_cast<std::uint32_t>(turbo4::kQhOffset) +
      8U * (dgroup & 3U);
  const std::uint8_t* const self_plane = s_group ? k_plane : v_plane;
  const std::uint8_t* const other_plane = s_group ? v_plane : k_plane;
  const auto load_raw = [&](std::uint32_t at, Attn6Raw* raw) {
    const std::uint32_t token = at + pos < end ? at + pos : end - 1U;
    const std::uint32_t token_off = token * t6_token_bytes;
    LoadTurbo6Raw(self_plane, token_off + t6_norm_off, token_off + t6_nib_off,
                  token_off + t6_qh_off, &raw->self);
    __builtin_memcpy(&raw->other.norm,
                     __builtin_assume_aligned(
                         other_plane + token_off + t6_norm_off, 2),
                     sizeof(raw->other.norm));
  };
  // Unpack + decode this lane's 32 values of the tile at `tile`.
  const auto decode = [&](const Attn6Raw& raw, std::uint32_t tile,
                          __half* vals) {
    bool poisoned = false;
    const Turbo4Slice slice =
        Turbo6Unpack(raw.self, raw.other, tile + pos < end, &poisoned);
    // K and V share the rule, so the S group's latch covers both planes.
    if (poisoned && s_group)
      atomicMax(&s_error,
                static_cast<unsigned int>(Turbo4KernelStatus::kPoisonedNorm));
    Attn6Dequant32<kC>(slice, s_centroids, 4U * lane, vals);
  };
  Attn6Raw next{};
  load_raw(begin, &next);

  if (s_group) {
    // ======================= S group: waves 0-7 ===========================
    // S ownership (as the WMMA reader): slot = 2 qt + kb for slots 0-5.
    const std::uint32_t s_qt = slot >> 1U;
    const std::uint32_t s_kb = slot & 1U;
    const bool s_wave = slot < 6U && s_qt * 16U < nvec;
    // This lane's Q operands: vector s_qt * 16 + sub, all 256 dims, as
    // __float2half(q_rot), i.e. the values q_lds held. Zero past nvec.
    // Unconditional loads (src is a valid vector for every lane), then select.
    v16h q_frag[kA6KSteps];
    {
      const std::uint32_t m = s_qt * 16U + sub;
      const bool live = s_wave && m < nvec;
      const float4* src = reinterpret_cast<const float4*>(
          q_rot +
          static_cast<std::size_t>(live ? m / kSplitQHeadsPerKv : 0U) *
              q_rot_row_stride +
          (static_cast<std::size_t>(kv_head) * kSplitQHeadsPerKv +
           (live ? m % kSplitQHeadsPerKv : 0U)) *
              kSplitHeadDim);
      // One k-step at a time: the empty asm (memory clobber) pins each
      // fragment's conversion before the next k-step's loads. Without it all
      // 64 b128 loads are hoisted (256 VGPRs before the f16 conversion) and
      // the kernel spills. Once per CTA, so the exposed latency is small.
#pragma unroll
      for (std::uint32_t ks = 0; ks < kA6KSteps; ++ks) {
#pragma unroll
        for (std::uint32_t e4 = 0; e4 < 4U; ++e4) {
          const float4 f = src[ks * 4U + e4];
          const float in[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t e = 0; e < 4U; ++e)
            q_frag[ks][e4 * 4U + e] = __builtin_bit_cast(
                _Float16, __half_as_ushort(__float2half(live ? in[e] : 0.0f)));
        }
        asm volatile("" : "+v"(q_frag[ks])::"memory");
      }
    }
    // Softmax ownership: 4 consecutive threads per query vector (slots 0-5).
    const std::uint32_t rg = (slot * 32U + lane) >> 2U;
    const std::uint32_t seg = lane & 3U;
    const std::uint32_t row_context =
        rg < nvec ? first_context + rg / kSplitQHeadsPerKv : 0U;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (std::uint32_t tile = begin; tile < end; tile += kVerifyKeys) {
      {
        const Attn6Raw raw = next;
        load_raw(tile + kVerifyKeys, &next);
        __attribute__((aligned(16))) __half vals[32];
        decode(raw, tile, vals);
        uint4* dst = reinterpret_cast<uint4*>(
            &kv_lds[pos * kA6KStride + dgroup * 32U]);
#pragma unroll
        for (std::uint32_t i = 0; i < 4; ++i)
          dst[i] = reinterpret_cast<const uint4*>(vals)[i];
      }
      Attn6Barrier();  // B1: K tile staged

      v8f s_acc = {};
      if (s_wave) {
        // Groups of 4 k-steps with a scheduling fence between them: without
        // it the scheduler hoists all 16 K fragment loads (128 VGPRs) ahead
        // of the chain, on top of the 128 VGPRs of Q, and spills.
#pragma unroll
        for (std::uint32_t kstep = 0; kstep < kA6KSteps; ++kstep) {
          if (kstep % 4U == 0U)
            __builtin_amdgcn_sched_barrier(0);
          s_acc = VerifyWmma(
              q_frag[kstep],
              VerifyLoadFrag(
                  &kv_lds[(s_kb * 16U + sub) * kA6KStride + kstep * 16U]),
              s_acc);
        }
      }
      Attn6Barrier();  // B2: every K read done; the S tile may overwrite it
      if (s_wave) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          s_lds[s_qt * 16U + 2U * i + half_id][s_kb * 16U + sub] =
              s_acc[i] * scale;
      }
      Attn6Barrier();  // B3: S visible to the softmax threads

      // Online softmax, per-row causal mask (slots 0-5, wave-uniform).
      if (rg < soft_rows) {
        const float4* srow =
            reinterpret_cast<const float4*>(&s_lds[rg][seg * 8U]);
        const float4 lo4 = srow[0], hi4 = srow[1];
        const float raw[8] = {lo4.x, lo4.y, lo4.z, lo4.w,
                              hi4.x, hi4.y, hi4.z, hi4.w};
        float vals[8];
        float part_max = -INFINITY;
#pragma unroll
        for (std::uint32_t m = 0; m < 8; ++m) {
          const std::uint32_t col = seg * 8U + m;
          vals[m] = (tile + col < row_context) ? raw[m] : -INFINITY;
          part_max = fmaxf(part_max, vals[m]);
        }
        part_max = fmaxf(part_max, Attn6QuadXor1(part_max));
        part_max = fmaxf(part_max, Attn6QuadXor2(part_max));
        const float next_max = fmaxf(running_max, part_max);
        const float prior_scale =
            isfinite(running_max) ? expf(running_max - next_max) : 0.0f;
        float part_sum = 0.0f;
        __attribute__((aligned(16))) __half w[8];
#pragma unroll
        for (std::uint32_t m = 0; m < 8; ++m) {
          w[m] = __float2half(isfinite(vals[m]) ? expf(vals[m] - next_max)
                                                : 0.0f);
          part_sum += __half2float(w[m]);  // l sums what PV multiplies
        }
        *reinterpret_cast<uint4*>(&p_lds[rg][seg * 8U]) =
            *reinterpret_cast<const uint4*>(w);
        part_sum += Attn6QuadXor1(part_sum);
        part_sum += Attn6QuadXor2(part_sum);
        running_max = next_max;
        running_sum = running_sum * prior_scale + part_sum;
        if (seg == 0U)
          row_scale[rg] = prior_scale;
      }
      Attn6Barrier();  // B4: P and row_scale visible to the PV group
    }

    // Partial record heads {m, l} [row][split][head], rotated basis; empty
    // (row ends before this split): m = -FLT_MAX, l = 0. s_error is final:
    // every latch precedes a B1.
    if (s_error == 0U && rg < nvec && seg == 0U) {
      float* rec =
          partials +
          static_cast<std::size_t>(rg / kSplitQHeadsPerKv) *
              partials_row_stride +
          (static_cast<std::size_t>(split) * num_heads +
           kv_head * kSplitQHeadsPerKv + rg % kSplitQHeadsPerKv) *
              kSplitRecordWidth;
      rec[0] = isfinite(running_max) ? running_max : -3.402823466e+38f;
      rec[1] = running_sum;
    }
  } else {
    // ======================= PV group: waves 8-15 =========================
    const std::uint32_t pv_wave = wave - 8U;  // dim tiles pv_wave, pv_wave + 8
    v8f o_acc[kVerifyQTiles][2] = {};
    for (std::uint32_t tile = begin; tile < end; tile += kVerifyKeys) {
      const Attn6Raw raw = next;
      load_raw(tile + kVerifyKeys, &next);
      Attn6Barrier();  // B1: every wave is past PV(t - 1): V^T is free
      {
        __attribute__((aligned(16))) __half vals[32];
        decode(raw, tile, vals);
#pragma unroll
        for (std::uint32_t j = 0; j < 32; ++j)
          vt_lds[(dgroup * 32U + j) * kA6VtStride + pos] = vals[j];
      }
      Attn6Barrier();  // B2
      Attn6Barrier();  // B3
      Attn6Barrier();  // B4: V^T (ours, before B2), P, row_scale visible

      if constexpr (kC) {
        // L2: the B operand depends on (t, kb) and the A operand on (qt, kb),
        // so the 24 (t, qt, kb) steps hold only 4 + 6 distinct fragments.
        // Load each once, q tile outer. Every accumulator o_acc[qt][t] still
        // gets row_rescale, then kb 0, then kb 1, with the same operands, so
        // the floats are attn6's. Safe between B4(t) and B1(t + 1): vt_lds is
        // rewritten only after B1 and p_lds / row_scale only after B3.
        v16h vt_frag[2][kVerifyKeys / 16U];
#pragma unroll
        for (std::uint32_t t = 0; t < 2; ++t)
#pragma unroll
          for (std::uint32_t kb = 0; kb < kVerifyKeys / 16U; ++kb)
            vt_frag[t][kb] = VerifyLoadFrag(
                &vt_lds[((pv_wave + 8U * t) * 16U + sub) * kA6VtStride +
                        kb * 16U]);
#pragma unroll
        for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
          if (qt * 16U >= nvec)
            continue;
          v16h p_frag[kVerifyKeys / 16U];
#pragma unroll
          for (std::uint32_t kb = 0; kb < kVerifyKeys / 16U; ++kb)
            p_frag[kb] = VerifyLoadFrag(&p_lds[qt * 16U + sub][kb * 16U]);
          float row_rescale[8];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            row_rescale[i] = row_scale[qt * 16U + 2U * i + half_id];
#pragma unroll
          for (std::uint32_t t = 0; t < 2; ++t) {
            v8f acc = o_acc[qt][t];
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i)
              acc[i] *= row_rescale[i];
#pragma unroll
            for (std::uint32_t kb = 0; kb < kVerifyKeys / 16U; ++kb)
              acc = VerifyWmma(p_frag[kb], vt_frag[t][kb], acc);
            o_acc[qt][t] = acc;
          }
        }
      } else {
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
          if (qt * 16U >= nvec)
            continue;
          float row_rescale[8];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            row_rescale[i] = row_scale[qt * 16U + 2U * i + half_id];
          v8f acc = o_acc[qt][t];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            acc[i] *= row_rescale[i];
#pragma unroll
          for (std::uint32_t kb = 0; kb < kVerifyKeys / 16U; ++kb)
            acc = VerifyWmma(
                VerifyLoadFrag(&p_lds[qt * 16U + sub][kb * 16U]),
                VerifyLoadFrag(&vt_lds[((pv_wave + 8U * t) * 16U + sub) *
                                           kA6VtStride +
                                       kb * 16U]),
                acc);
          o_acc[qt][t] = acc;
        }
      }
      }
    }

    if (s_error == 0U) {
#pragma unroll
      for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
#pragma unroll
        for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const std::uint32_t m = qt * 16U + 2U * i + half_id;
            if (m >= nvec)
              continue;
            float* rec = partials +
                         static_cast<std::size_t>(m / kSplitQHeadsPerKv) *
                             partials_row_stride +
                         (static_cast<std::size_t>(split) * num_heads +
                          kv_head * kSplitQHeadsPerKv +
                          m % kSplitQHeadsPerKv) *
                             kSplitRecordWidth;
            rec[2 + (pv_wave + 8U * t) * 16U + sub] = o_acc[qt][t][i];
          }
        }
      }
    }
  }
  if (s_error != 0U && tid == 0)
    Turbo4LatchFailure(status, static_cast<Turbo4KernelStatus>(s_error));
}

// The ATTN FUSE kernel body: Turbo4VerifyAttn6Kernel's body (above) with the
// fuse blocks added; q_rot is the UNROTATED q (rotated in the prologue), and
// fz carries the reduce's outputs. A copy, not a shared template, on purpose:
// moving the served kernel's body into a template changed its ISA (register
// allocation / scheduling), and the served kernel must stay byte-identical.
// ponytail: two copies of the main body; an edit to one must go to the other
// (the GPU bench "fuse" arm compares them bitwise).
// kB (with kFuse; GUFO_DEV_ATTN_B=1): design B step 2a, the two pieces of
// the attn7 WGP pipeline that run unchanged inside attn6's shape, both
// bitwise (proof: attnB_20261025/b_proof.cpp):
//  - live-row softmax: only rows < 6 x rows run it (attn6 runs all 48). Rows
//    never interact (a row's 4 threads exchange only inside their quad), so
//    live rows are unchanged; a dead row's P / row_scale is left stale and
//    reaches only O rows >= 6 x rows, which no partial record stores.
//  - PV fragment reuse: per tile a PV wave loads its 4 V^T fragments
//    (t, kb) once and each live q tile's 2 P fragments once (attn6 reloads
//    both for every (t, qt): 24 fragment loads -> 10 at 8 rows). Every
//    accumulator still sees rescale, then WMMA kb 0, then kb 1, on the same
//    operands.
// kC (ATTN_C, GUFO_DEV_ATTN_C=1): attnC's L1 centroid table (32 interleaved
// copies, lane l always in bank l; see Attn6Dequant32). L1 is orthogonal to
// kB, so ATTN_C stacks on ATTN_B. attnC's other two cuts ARE kB's two pieces
// -- live-row softmax and PV fragment reuse, found independently -- so kC
// reuses kB's code rather than duplicating it: both are enabled by
// `kB || kC`. kC additionally pre-fills the padding rows' P / row_scale, so a
// dead row reads a deterministic zero instead of stale LDS; kB alone keeps
// its proven stale-read behaviour unchanged.
template<bool kFuse, bool kB, bool kC>
__device__ __forceinline__ void Attn6Body(
    const float* __restrict__ q_rot, const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, float* __restrict__ partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t max_splits,
    std::uint32_t split_size, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    std::size_t q_rot_row_stride, std::size_t partials_row_stride,
    turbo4::PlaneFormats formats, const Attn6FuseArgs& fz) {
  // K tile [key][d] during decode + S, then the S tile [q][key] (fp32).
  __shared__ __attribute__((aligned(16))) __half kv_lds[kA6KHalves];
  __shared__ __attribute__((
      aligned(16))) __half vt_lds[kSplitHeadDim * kA6VtStride];
  __shared__
      __attribute__((aligned(16))) __half p_lds[kVerifyQVecs][kVerifyVtStride];
  __shared__ float row_scale[kVerifyQVecs];
  __shared__ float s_centroids[kC ? kA6CentroidDwords : 64U];
  __shared__ unsigned int s_error;
  float(*const s_lds)[kA6SStride] =
      reinterpret_cast<float(*)[kA6SStride]>(kv_lds);

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;

  const std::uint32_t last_context = first_context + rows - 1U;
  const std::uint32_t split_count =
      split_size == 0U ? 0U : (last_context + split_size - 1U) / split_size;
  const std::size_t required_bytes = static_cast<std::size_t>(max_splits) *
                                     num_heads * kSplitRecordWidth *
                                     sizeof(float);
  if (blockDim.x != kA6Threads ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv ||
      kv_head >= num_kv_heads || rows == 0 || rows > kVerifyMaxRows ||
      first_context == 0 || last_context > max_context || split_size == 0U ||
      split_size % kVerifyKeys != 0U || split_count > max_splits ||
      split >= split_count ||
      !VerifyT6PipeFits(formats, num_kv_heads, max_context) ||
      plane_bytes < Turbo4LayerPlaneBytes(kSplitHeadDim, num_kv_heads,
                                          max_context, turbo4::Bits::k6) ||
      partial_bytes < required_bytes || q_rot == nullptr ||
      partials == nullptr || status == nullptr || write_status == nullptr) {
    if (tid == 0)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  if (tid == 0) {
    s_error = static_cast<unsigned int>(*status);
    if (*write_status != Turbo4KernelStatus::kOk) {
      s_error = static_cast<unsigned int>(Turbo4KernelStatus::kWriterFailed);
      Turbo4LatchFailure(status, Turbo4KernelStatus::kWriterFailed);
    }
  }
  if constexpr (kC) {
#pragma unroll
    for (std::uint32_t i = tid; i < kA6CentroidDwords; i += kA6Threads)
      s_centroids[i] = turbo4::CentroidValue(turbo4::Bits::k6, i >> 5U);
  } else if (tid < 64U) {
    s_centroids[tid] = turbo4::CentroidValue(turbo4::Bits::k6, tid);
  }
  __syncthreads();
  // Fuse: the arrival on every exit path past the geometry check.
  const auto fuse_tail = [&]() {
    if constexpr (kFuse)
      Attn6FuseTail(partials, fz, status, num_heads, kv_head, split_count,
                    rows, partials_row_stride, tid,
                    reinterpret_cast<float*>(kv_lds),
                    reinterpret_cast<float*>(vt_lds));
  };
  if (s_error != 0U) {
    fuse_tail();
    return;
  }

  const std::uint32_t nvec = rows * kSplitQHeadsPerKv;
  if constexpr (kC) {
    // kB leaves a dead row's P / row_scale stale (proved bitwise: it reaches
    // only O rows >= nvec, which no partial record stores). kC zeroes them
    // once instead, so the discarded values are deterministic rather than
    // whatever the previous CTA left in LDS. ~4 stores per thread per CTA.
    for (std::uint32_t i = tid; i < (kVerifyQVecs - nvec) * kVerifyVtStride;
         i += kA6Threads)
      p_lds[nvec + i / kVerifyVtStride][i % kVerifyVtStride] =
          __float2half(0.0f);
    for (std::uint32_t r = nvec + tid; r < kVerifyQVecs; r += kA6Threads)
      row_scale[r] = 0.0f;
  }
  const bool s_group = wave < 8U;
  // S-group role slot: the S / softmax units (slots 0-5) sit on waves 0-5 for
  // even splits and on waves 2-7 for odd ones, so across co-resident CTAs no
  // SIMD (wave % 4) carries two S waves every time. Any bijection works: the
  // arithmetic per unit does not depend on which wave runs it.
  const std::uint32_t slot = (wave + 8U - 2U * (split & 1U)) & 7U;
  // Decode ownership: position pos, dims [32 * dgroup, 32 * dgroup + 32) of
  // K (S group) or V (PV group).
  const std::uint32_t pos = lane;
  const std::uint32_t dgroup = s_group ? slot : wave - 8U;
  const std::uint32_t begin = split * split_size;
  const std::uint32_t end =
      last_context < begin + split_size ? last_context : begin + split_size;

  // Raw-word prefetch of the next tile, the kT6Pipe instance's addressing
  // (VerifyT6PipeFits makes every offset 32-bit).
  const std::uint32_t t6_token_bytes = static_cast<std::uint32_t>(
      Turbo4BlocksPerToken(kSplitHeadDim, num_kv_heads) *
      turbo4::BlockBytes(turbo4::Bits::k6));
  const std::uint32_t t6_norm_off =
      kv_head * static_cast<std::uint32_t>(
                    Turbo4BlocksPerHead(kSplitHeadDim) *
                    turbo4::BlockBytes(turbo4::Bits::k6)) +
      (dgroup / 4U) *
          static_cast<std::uint32_t>(turbo4::BlockBytes(turbo4::Bits::k6));
  const std::uint32_t t6_nib_off = t6_norm_off + 2U + 16U * (dgroup & 3U);
  const std::uint32_t t6_qh_off =
      t6_norm_off + static_cast<std::uint32_t>(turbo4::kQhOffset) +
      8U * (dgroup & 3U);
  const std::uint8_t* const self_plane = s_group ? k_plane : v_plane;
  const std::uint8_t* const other_plane = s_group ? v_plane : k_plane;
  const auto load_raw = [&](std::uint32_t at, Attn6Raw* raw) {
    const std::uint32_t token = at + pos < end ? at + pos : end - 1U;
    const std::uint32_t token_off = token * t6_token_bytes;
    LoadTurbo6Raw(self_plane, token_off + t6_norm_off, token_off + t6_nib_off,
                  token_off + t6_qh_off, &raw->self);
    __builtin_memcpy(&raw->other.norm,
                     __builtin_assume_aligned(
                         other_plane + token_off + t6_norm_off, 2),
                     sizeof(raw->other.norm));
  };
  // Unpack + decode this lane's 32 values of the tile at `tile`.
  const auto decode = [&](const Attn6Raw& raw, std::uint32_t tile,
                          __half* vals) {
    bool poisoned = false;
    const Turbo4Slice slice =
        Turbo6Unpack(raw.self, raw.other, tile + pos < end, &poisoned);
    // K and V share the rule, so the S group's latch covers both planes.
    if (poisoned && s_group)
      atomicMax(&s_error,
                static_cast<unsigned int>(Turbo4KernelStatus::kPoisonedNorm));
    Attn6Dequant32<kC>(slice, s_centroids, 4U * lane, vals);
  };
  Attn6Raw next{};
  load_raw(begin, &next);
  if constexpr (kFuse) {
    // Rotate this CTA's query vectors into the K tile buffer (as f16) while
    // the first tile's raw loads are in flight.
    Attn6FuseRotateQ(q_rot, q_rot_row_stride, kv_head, nvec, wave, lane,
                     kv_lds, vt_lds, &s_error, status);
    Attn6Barrier();  // QA: q tile written, s_error final for Q
    if (s_error != 0U) {
      fuse_tail();
      return;
    }
  }

  if (s_group) {
    // ======================= S group: waves 0-7 ===========================
    // S ownership (as the WMMA reader): slot = 2 qt + kb for slots 0-5.
    const std::uint32_t s_qt = slot >> 1U;
    const std::uint32_t s_kb = slot & 1U;
    const bool s_wave = slot < 6U && s_qt * 16U < nvec;
    // This lane's Q operands: vector s_qt * 16 + sub, all 256 dims, as
    // __float2half(q_rot), i.e. the values q_lds held. Zero past nvec.
    // Unconditional loads (src is a valid vector for every lane), then select.
    v16h q_frag[kA6KSteps];
    if constexpr (kFuse) {
      // The same halves, from the prologue's q tile (row m: dims 0-127 in
      // kv_lds, 128-255 in vt_lds).
      const std::uint32_t m = s_qt * 16U + sub;
      const bool live = s_wave && m < nvec;
      const std::uint32_t row = (live ? m : 0U) * kA6FuseQStride;
#pragma unroll
      for (std::uint32_t ks = 0; ks < kA6KSteps; ++ks) {
        q_frag[ks] = VerifyLoadFrag(
            (ks < kA6KSteps / 2U ? kv_lds : vt_lds) + row +
            (ks % (kA6KSteps / 2U)) * 16U);
        if (!live)
          q_frag[ks] = v16h{};
      }
      Attn6Barrier();  // QB: every q read done before K(t) / V^T(t) overwrite it
    } else {
      const std::uint32_t m = s_qt * 16U + sub;
      const bool live = s_wave && m < nvec;
      const float4* src = reinterpret_cast<const float4*>(
          q_rot +
          static_cast<std::size_t>(live ? m / kSplitQHeadsPerKv : 0U) *
              q_rot_row_stride +
          (static_cast<std::size_t>(kv_head) * kSplitQHeadsPerKv +
           (live ? m % kSplitQHeadsPerKv : 0U)) *
              kSplitHeadDim);
      // One k-step at a time: the empty asm (memory clobber) pins each
      // fragment's conversion before the next k-step's loads. Without it all
      // 64 b128 loads are hoisted (256 VGPRs before the f16 conversion) and
      // the kernel spills. Once per CTA, so the exposed latency is small.
#pragma unroll
      for (std::uint32_t ks = 0; ks < kA6KSteps; ++ks) {
#pragma unroll
        for (std::uint32_t e4 = 0; e4 < 4U; ++e4) {
          const float4 f = src[ks * 4U + e4];
          const float in[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t e = 0; e < 4U; ++e)
            q_frag[ks][e4 * 4U + e] = __builtin_bit_cast(
                _Float16, __half_as_ushort(__float2half(live ? in[e] : 0.0f)));
        }
        asm volatile("" : "+v"(q_frag[ks])::"memory");
      }
    }
    // Softmax ownership: 4 consecutive threads per query vector (slots 0-5).
    const std::uint32_t rg = (slot * 32U + lane) >> 2U;
    const std::uint32_t seg = lane & 3U;
    const std::uint32_t row_context =
        rg < nvec ? first_context + rg / kSplitQHeadsPerKv : 0U;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (std::uint32_t tile = begin; tile < end; tile += kVerifyKeys) {
      {
        const Attn6Raw raw = next;
        load_raw(tile + kVerifyKeys, &next);
        __attribute__((aligned(16))) __half vals[32];
        decode(raw, tile, vals);
        uint4* dst = reinterpret_cast<uint4*>(
            &kv_lds[pos * kA6KStride + dgroup * 32U]);
#pragma unroll
        for (std::uint32_t i = 0; i < 4; ++i)
          dst[i] = reinterpret_cast<const uint4*>(vals)[i];
      }
      Attn6Barrier();  // B1: K tile staged

      v8f s_acc = {};
      if (s_wave) {
        // Groups of 4 k-steps with a scheduling fence between them: without
        // it the scheduler hoists all 16 K fragment loads (128 VGPRs) ahead
        // of the chain, on top of the 128 VGPRs of Q, and spills.
#pragma unroll
        for (std::uint32_t kstep = 0; kstep < kA6KSteps; ++kstep) {
          if (kstep % 4U == 0U)
            __builtin_amdgcn_sched_barrier(0);
          s_acc = VerifyWmma(
              q_frag[kstep],
              VerifyLoadFrag(
                  &kv_lds[(s_kb * 16U + sub) * kA6KStride + kstep * 16U]),
              s_acc);
        }
      }
      Attn6Barrier();  // B2: every K read done; the S tile may overwrite it
      if (s_wave) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          s_lds[s_qt * 16U + 2U * i + half_id][s_kb * 16U + sub] =
              s_acc[i] * scale;
      }
      Attn6Barrier();  // B3: S visible to the softmax threads

      // Online softmax, per-row causal mask (slots 0-5; wave-uniform, kB:
      // quad-uniform: a row is one aligned quad, its DPP stays inside it).
      if (rg < ((kB || kC) ? nvec : kVerifyQVecs)) {
        const float4* srow =
            reinterpret_cast<const float4*>(&s_lds[rg][seg * 8U]);
        const float4 lo4 = srow[0], hi4 = srow[1];
        const float raw[8] = {lo4.x, lo4.y, lo4.z, lo4.w,
                              hi4.x, hi4.y, hi4.z, hi4.w};
        float vals[8];
        float part_max = -INFINITY;
#pragma unroll
        for (std::uint32_t m = 0; m < 8; ++m) {
          const std::uint32_t col = seg * 8U + m;
          vals[m] = (tile + col < row_context) ? raw[m] : -INFINITY;
          part_max = fmaxf(part_max, vals[m]);
        }
        part_max = fmaxf(part_max, Attn6QuadXor1(part_max));
        part_max = fmaxf(part_max, Attn6QuadXor2(part_max));
        const float next_max = fmaxf(running_max, part_max);
        const float prior_scale =
            isfinite(running_max) ? expf(running_max - next_max) : 0.0f;
        float part_sum = 0.0f;
        __attribute__((aligned(16))) __half w[8];
#pragma unroll
        for (std::uint32_t m = 0; m < 8; ++m) {
          w[m] = __float2half(isfinite(vals[m]) ? expf(vals[m] - next_max)
                                                : 0.0f);
          part_sum += __half2float(w[m]);  // l sums what PV multiplies
        }
        *reinterpret_cast<uint4*>(&p_lds[rg][seg * 8U]) =
            *reinterpret_cast<const uint4*>(w);
        part_sum += Attn6QuadXor1(part_sum);
        part_sum += Attn6QuadXor2(part_sum);
        running_max = next_max;
        running_sum = running_sum * prior_scale + part_sum;
        if (seg == 0U)
          row_scale[rg] = prior_scale;
      }
      Attn6Barrier();  // B4: P and row_scale visible to the PV group
    }

    // Partial record heads {m, l} [row][split][head], rotated basis; empty
    // (row ends before this split): m = -FLT_MAX, l = 0. s_error is final:
    // every latch precedes a B1.
    if (s_error == 0U && rg < nvec && seg == 0U) {
      float* rec =
          partials +
          static_cast<std::size_t>(rg / kSplitQHeadsPerKv) *
              partials_row_stride +
          (static_cast<std::size_t>(split) * num_heads +
           kv_head * kSplitQHeadsPerKv + rg % kSplitQHeadsPerKv) *
              kSplitRecordWidth;
      rec[0] = isfinite(running_max) ? running_max : -3.402823466e+38f;
      rec[1] = running_sum;
    }
  } else {
    // ======================= PV group: waves 8-15 =========================
    const std::uint32_t pv_wave = wave - 8U;  // dim tiles pv_wave, pv_wave + 8
    v8f o_acc[kVerifyQTiles][2] = {};
    if constexpr (kFuse)
      Attn6Barrier();  // QB (the S group's q reads)
    for (std::uint32_t tile = begin; tile < end; tile += kVerifyKeys) {
      const Attn6Raw raw = next;
      load_raw(tile + kVerifyKeys, &next);
      Attn6Barrier();  // B1: every wave is past PV(t - 1): V^T is free
      {
        __attribute__((aligned(16))) __half vals[32];
        decode(raw, tile, vals);
#pragma unroll
        for (std::uint32_t j = 0; j < 32; ++j)
          vt_lds[(dgroup * 32U + j) * kA6VtStride + pos] = vals[j];
      }
      Attn6Barrier();  // B2
      Attn6Barrier();  // B3
      Attn6Barrier();  // B4: V^T (ours, before B2), P, row_scale visible

      if constexpr (kB || kC) {
        v16h vt_frag[2][2];
#pragma unroll
        for (std::uint32_t t = 0; t < 2; ++t)
#pragma unroll
          for (std::uint32_t kb = 0; kb < kVerifyKeys / 16U; ++kb)
            vt_frag[t][kb] = VerifyLoadFrag(
                &vt_lds[((pv_wave + 8U * t) * 16U + sub) * kA6VtStride +
                        kb * 16U]);
#pragma unroll
        for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
          if (qt * 16U >= nvec)
            continue;
          const v16h p0 = VerifyLoadFrag(&p_lds[qt * 16U + sub][0]);
          const v16h p1 = VerifyLoadFrag(&p_lds[qt * 16U + sub][16]);
          float row_rescale[8];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            row_rescale[i] = row_scale[qt * 16U + 2U * i + half_id];
#pragma unroll
          for (std::uint32_t t = 0; t < 2; ++t) {
            v8f acc = o_acc[qt][t];
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i)
              acc[i] *= row_rescale[i];
            acc = VerifyWmma(p0, vt_frag[t][0], acc);
            acc = VerifyWmma(p1, vt_frag[t][1], acc);
            o_acc[qt][t] = acc;
          }
        }
      } else {
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
          if (qt * 16U >= nvec)
            continue;
          float row_rescale[8];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            row_rescale[i] = row_scale[qt * 16U + 2U * i + half_id];
          v8f acc = o_acc[qt][t];
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            acc[i] *= row_rescale[i];
#pragma unroll
          for (std::uint32_t kb = 0; kb < kVerifyKeys / 16U; ++kb)
            acc = VerifyWmma(
                VerifyLoadFrag(&p_lds[qt * 16U + sub][kb * 16U]),
                VerifyLoadFrag(&vt_lds[((pv_wave + 8U * t) * 16U + sub) *
                                           kA6VtStride +
                                       kb * 16U]),
                acc);
          o_acc[qt][t] = acc;
        }
      }
      }  // kB
    }

    if (s_error == 0U) {
#pragma unroll
      for (std::uint32_t qt = 0; qt < kVerifyQTiles; ++qt) {
#pragma unroll
        for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const std::uint32_t m = qt * 16U + 2U * i + half_id;
            if (m >= nvec)
              continue;
            float* rec = partials +
                         static_cast<std::size_t>(m / kSplitQHeadsPerKv) *
                             partials_row_stride +
                         (static_cast<std::size_t>(split) * num_heads +
                          kv_head * kSplitQHeadsPerKv +
                          m % kSplitQHeadsPerKv) *
                             kSplitRecordWidth;
            rec[2 + (pv_wave + 8U * t) * 16U + sub] = o_acc[qt][t][i];
          }
        }
      }
    }
  }
  if (s_error != 0U && tid == 0)
    Turbo4LatchFailure(status, static_cast<Turbo4KernelStatus>(s_error));
  fuse_tail();
}

// ATTN FUSE: q (unrotated) -> out / out_rot / partials in one launch.
// kC = ATTN_C on top (L1 table + kB's two pieces, which kC also enables).
template <bool kC>
__global__ __launch_bounds__(kA6Threads)
__attribute__((amdgpu_waves_per_eu(8))) void Turbo4VerifyAttn6FuseKernel(
    const float* __restrict__ q, const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, float* __restrict__ partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t max_splits,
    std::uint32_t split_size, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    std::size_t q_row_stride, std::size_t partials_row_stride,
    turbo4::PlaneFormats formats, Attn6FuseArgs fz) {
  Attn6Body<true, false, kC>(q, k_plane, v_plane, partials, plane_bytes,
                  partial_bytes, num_heads, num_kv_heads, first_context, rows,
                  max_context, max_splits, split_size, scale, write_status,
                  status, q_row_stride, partials_row_stride, formats, fz);
}

// ATTN B step 2a: the fused kernel plus live-row softmax and PV fragment
// reuse (see Attn6Body). kC = ATTN_C's L1 table on top of it.
template <bool kC>
__global__ __launch_bounds__(kA6Threads)
__attribute__((amdgpu_waves_per_eu(8))) void Turbo4VerifyAttn6BKernel(
    const float* __restrict__ q, const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, float* __restrict__ partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t max_splits,
    std::uint32_t split_size, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    std::size_t q_row_stride, std::size_t partials_row_stride,
    turbo4::PlaneFormats formats, Attn6FuseArgs fz) {
  Attn6Body<true, true, kC>(q, k_plane, v_plane, partials, plane_bytes,
                        partial_bytes, num_heads, num_kv_heads, first_context,
                        rows, max_context, max_splits, split_size, scale,
                        write_status, status, q_row_stride,
                        partials_row_stride, formats, fz);
}

}  // namespace

Turbo4LaunchStatus LaunchTurbo4AttentionVerifyAttn6(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats,
    std::uint32_t splits_cap, std::uint32_t stages, int attn_c) {
  if (!formats.Valid() || !VerifyT6PipeFits(formats, num_kv_heads, max_context) ||
      GufoDevVSkipThreshold() > 0.0f)
    return Turbo4LaunchStatus::kGeometryRefused;
  const bool fused =
      stages == kAttn6StageFused || stages == kAttn6StageFusedB;
  // The fused kernel reads q and writes out_rot as float4.
  if (fused && ((reinterpret_cast<std::uintptr_t>(q) |
                 reinterpret_cast<std::uintptr_t>(out_rot)) % 16U) != 0U)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (fused ? num_kv_heads > kAttn6FuseMaxKvHeads
            : (stages == 0U || stages > kAttn6StagesAll))
    return Turbo4LaunchStatus::kGeometryRefused;
  if (q == nullptr || k_plane == nullptr || v_plane == nullptr ||
      out == nullptr || q_rot == nullptr || out_rot == nullptr ||
      partials == nullptr || d_write_status == nullptr || d_status == nullptr ||
      plane_bytes == 0 || num_heads == 0 || num_kv_heads == 0 || rows == 0)
    return Turbo4LaunchStatus::kBadArguments;
  if (max_context == 0 || first_valid_context == 0 || rows > kVerifyMaxRows ||
      rows > max_context || first_valid_context > max_context - rows + 1U)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (head_dim != kSplitHeadDim ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv || !std::isfinite(scale))
    return Turbo4LaunchStatus::kGeometryRefused;
  if ((gate_bytes == 0U) != (gate == nullptr))
    return Turbo4LaunchStatus::kBadArguments;

  const std::size_t vector_floats =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t rows_bytes = rows * vector_floats * sizeof(float);
  const std::size_t row_partial_bytes = Turbo4BatchedDecodePartialBytes(
      first_valid_context, rows, num_heads, head_dim);
  const std::size_t required_plane = Turbo4LayerPlaneBytes(
      head_dim, num_kv_heads, max_context, formats.stride);
  if (required_plane == 0U || plane_bytes < required_plane ||
      row_partial_bytes == 0U || partial_bytes < rows * row_partial_bytes ||
      q_bytes < rows_bytes || out_bytes < rows_bytes ||
      q_rot_bytes < rows_bytes || out_rot_bytes < rows_bytes ||
      (gate != nullptr && gate_bytes < rows_bytes))
    return Turbo4LaunchStatus::kBadArguments;

  const std::uint32_t last_context = first_valid_context + rows - 1U;
  const std::uint32_t max_splits = static_cast<std::uint32_t>(
      row_partial_bytes / (num_heads * (head_dim + 2) * sizeof(float)));
  // Batch-invariant mode lays the split SIZE out for the width-8 last row; the
  // COUNT stays on this launch's real last row.
  const std::uint32_t split_size = static_cast<std::uint32_t>(Attn6SplitSize(
      BatchInvariantPlanContext(first_valid_context, rows, max_context),
      splits_cap));
  const std::uint32_t split_count =
      split_size == 0U ? 0U : (last_context + split_size - 1U) / split_size;
  if (max_splits == 0 || max_splits > kSplitHeadDim || split_count == 0 ||
      split_count > max_splits)
    return Turbo4LaunchStatus::kGeometryRefused;

  const std::size_t partials_row_stride = row_partial_bytes / sizeof(float);
  // attn_c: -1 = GUFO_DEV_ATTN_C, 0/1 = explicit (the bench needs several
  // arms in one process; the env flag is a static const read once).
  const bool want_c = attn_c < 0 ? GufoDevAttnC() : attn_c != 0;
  if (fused) {
    // One launch: rotate, reader and reduce (Turbo4VerifyAttn6FuseKernel).
    // q_rot is not written.
    const Attn6FuseArgs fz{out_rot, out, gate, vector_floats};
    const auto launch_fused = [&](auto kc) {
      if (stages == kAttn6StageFusedB)
        Turbo4VerifyAttn6BKernel<decltype(kc)::value>
            <<<dim3(num_kv_heads, split_count), kA6Threads, 0, stream>>>(
                q, k_plane, v_plane, partials, plane_bytes, row_partial_bytes,
                num_heads, num_kv_heads, first_valid_context, rows,
                max_context, max_splits, split_size, scale, d_write_status,
                d_status, vector_floats, partials_row_stride, formats, fz);
      else
        Turbo4VerifyAttn6FuseKernel<decltype(kc)::value>
            <<<dim3(num_kv_heads, split_count), kA6Threads, 0, stream>>>(
                q, k_plane, v_plane, partials, plane_bytes, row_partial_bytes,
                num_heads, num_kv_heads, first_valid_context, rows,
                max_context, max_splits, split_size, scale, d_write_status,
                d_status, vector_floats, partials_row_stride, formats, fz);
    };
    if (want_c)
      launch_fused(std::true_type{});
    else
      launch_fused(std::false_type{});
    return hipPeekAtLastError() == hipSuccess
               ? Turbo4LaunchStatus::kLaunched
               : Turbo4LaunchStatus::kEnqueueFailed;
  }

  // stages: the bench times each launch alone (P4); every caller passes all.
  if ((stages & kAttn6StageRotate) != 0U) {
    Turbo4RotateQueryKernel<<<dim3(num_heads, rows), turbo4::kBlockValues, 0,
                              stream>>>(q, q_rot, num_heads, head_dim, q_bytes,
                                        q_rot_bytes, d_status, vector_floats);
    if (hipPeekAtLastError() != hipSuccess)
      return Turbo4LaunchStatus::kEnqueueFailed;
  }

  if ((stages & kAttn6StageMain) != 0U) {
    const auto launch_main = [&](auto kc) {
      Turbo4VerifyAttn6Kernel<decltype(kc)::value>
          <<<dim3(num_kv_heads, split_count), kA6Threads, 0, stream>>>(
              q_rot, k_plane, v_plane, partials, plane_bytes,
              row_partial_bytes, num_heads, num_kv_heads, first_valid_context,
              rows, max_context, max_splits, split_size, scale, d_write_status,
              d_status, vector_floats, partials_row_stride, formats);
    };
    if (want_c)
      launch_main(std::true_type{});
    else
      launch_main(std::false_type{});
    if (hipPeekAtLastError() != hipSuccess)
      return Turbo4LaunchStatus::kEnqueueFailed;
  }

  if ((stages & kAttn6StageReduce) != 0U) {
    Turbo4AttentionReduceInverseKernel<<<dim3(num_heads, rows), kSplitHeadDim,
                                         max_splits * sizeof(float), stream>>>(
        partials, out_rot, out, gate, row_partial_bytes, out_rot_bytes,
        out_bytes, gate_bytes, num_heads, num_kv_heads, split_count,
        max_splits, head_dim, d_status, first_valid_context,
        partials_row_stride, vector_floats,
        /*uniform_splits=*/true);
    if (hipPeekAtLastError() != hipSuccess)
      return Turbo4LaunchStatus::kEnqueueFailed;
  }
  return Turbo4LaunchStatus::kLaunched;
}

// ---------------------------------------------------------------------------
// VRD6 (GUFO_DEV_VRD6=1): K = V = Turbo6 verify reader built to read the KV
// planes at memory speed. Same contract, partial layout and reduce as the
// readers above; NOT bitwise vs them. Why the attn6 reader stays under
// ~90 GB/s at 32k / 8 rows (cap 10): per 32-key tile and CTA it issues ~4300
// VALU beside its 192 WMMA (6144 SIMD cycles; WMMA and VALU share the SIMD):
// one 64-entry f32 table read + v_fma_mix per value, libm expf (~12 VALU
// each), four barriers per tile with phases where half the waves idle.
// What changes here:
//   1. Decode = one LDS read per value PAIR from a 4096-entry table of f16
//      centroid pairs (Vrd6PairOffsets: 1 v_perm + 1 shift per pair), and no
//      norm multiply: the K norm (x scale x log2 e) multiplies the f32 score
//      after the WMMA, s = n0' S0 + n1' S1 (one accumulator per 128-dim
//      block), and the V norm multiplies P: P_b = f16(p * n_v,b), one P tile
//      per 128-dim V block.
//   2. S^T = K Q^T on the WMMA (keys are rows): a lane holds 8 keys of one
//      query, so the row max is 7 v_max + one permlanex16, and its 8 P values
//      are one 16-byte store (Vrd6KeyColumn order, which V^T shares).
//   3. exp2 (v_exp_f32) on log2-scaled scores; l is the f32 sum of p.
//   4. Two LDS-only barriers per tile; every wave issues 16 WMMA per tile, so
//      any 3-waves-per-SIMD placement puts 48 WMMA on each SIMD:
//        S wave (qt, kb), 0-5:   S^T(t), fold, max | Y | softmax(t), P(t) | X
//        PV wave (qt, blk), 6-11: rescale, PV(t-1)  | Y | decode K(t+1), V(t) | X
//   5. Split plan: Attn6SplitSize, default cap 10 (4 KV heads x 10 = 40 CTAs =
//      one CTA wave at 2 CTAs per WGP).
// Buffer lifetimes (single buffers unless noted): K(t+1) and its norms are
// written in B(t) and read in A(t+1) (A(t) read K(t) before Y(t)); V^T(t),
// P(t), alpha(t), flags(t) are written in B(t) and read in A(t+1) (A(t) read
// the previous ones before Y(t)); rowmax is written in A and read in B; the V
// norms of tile t are read in B(t) while B(t) writes tile t + 1's: double.
// LDS: pair table 16384 + K 32x264 f16 16896 + V^T 256x40 f16 20480 + P
// 2x48x40 f16 7680 + norms 768 + rowmax 384 + alpha 192 + flags 16 + l 192 +
// error 4 = 62996 B, 2 CTAs per WGP.
// Numerics vs the WMMA / attn6 readers: K and V values are f16(c), not
// f16(c * norm); s is folded in f32; P carries the V norm in f16 (subnormal
// only where p * n_v < 6.1e-5, the class of loss f16(c * n_v) has for
// n_v < ~0.02); l sums f32 p. Checked by a CPU error study.
// ponytail: the 16 decode groups per tile split 3/3/2 over three K and three
// V decoder waves (the 2-group waves carry the norms); a profile would show
// whether that imbalance costs anything.
namespace {

constexpr std::uint32_t kV6Threads = 384;  // 12 waves: 6 S, 6 PV
constexpr std::uint32_t kV6SWaves = 6;
constexpr std::uint32_t kV6Rounds = 3;  // decode batches per PV wave, at most
constexpr std::uint32_t kV6KStride = kSplitHeadDim + 8;  // K tile rows (halves)
constexpr std::uint32_t kV6TStride = kVerifyKeys + 8;    // V^T and P rows
constexpr float kV6Log2e = 1.44269504088896341f;
constexpr float kV6Ln2 = 0.693147180559945309f;
static_assert(kV6KStride % 8 == 0 && kV6TStride % 8 == 0,
              "fragment and b128 rows start on 16 bytes");
static_assert(kV6SWaves * kV6Rounds >= 16, "16 decode batches per tile");

/// LDS-only barrier (as Attn6Barrier): the next tiles' raw loads stay in
/// flight across it.
__device__ __forceinline__ void Vrd6Barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/// The same lane of the other 16-lane half (v_permlanex16, identity select).
__device__ __forceinline__ float Vrd6OtherHalf(float v) {
  return __uint_as_float(__builtin_amdgcn_permlanex16(
      __float_as_uint(v), __float_as_uint(v), 0x76543210U, 0xFEDCBA98U, false,
      false));
}

/// 16 nibble bytes + 8 qh bytes of one (token, 32-dim group): LoadTurbo6Raw
/// without the norm word. Blocks are 2-aligned, offsets 32-bit
/// (VerifyT6PipeFits).
__device__ __forceinline__ void Vrd6LoadSlice(const std::uint8_t* base,
                                              std::uint32_t block_off,
                                              std::uint32_t group,
                                              std::uint32_t* w,
                                              std::uint32_t* qh) {
  __builtin_memcpy(
      w, __builtin_assume_aligned(base + block_off + 2U + 16U * group, 2), 16);
  __builtin_memcpy(qh,
                   __builtin_assume_aligned(
                       base + block_off + turbo4::kQhOffset + 8U * group, 2),
                   8);
}

/// One PV wave's raw words for one tile job: three 32-dim slices of its
/// plane, plus the K and V norm words of one block.
struct Vrd6Job {
  std::uint32_t w[kV6Rounds][4];
  std::uint32_t qh[kV6Rounds][2];
  std::uint32_t kn;
  std::uint32_t vn;
};

__global__ __launch_bounds__(kV6Threads)
__attribute__((amdgpu_waves_per_eu(6))) void Turbo4VerifyVrd6Kernel(
    const float* __restrict__ q_rot, const std::uint8_t* __restrict__ k_plane,
    const std::uint8_t* __restrict__ v_plane, float* __restrict__ partials,
    std::size_t plane_bytes, std::size_t partial_bytes, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t first_context, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t max_splits,
    std::uint32_t split_size, float scale,
    const Turbo4KernelStatus* write_status, Turbo4KernelStatus* status,
    std::size_t q_rot_row_stride, std::size_t partials_row_stride,
    turbo4::PlaneFormats formats) {
  __shared__ std::uint32_t lut[kVrd6LutEntries];
  __shared__ __attribute__((aligned(16))) __half kf[kVerifyKeys * kV6KStride];
  __shared__ __attribute__((
      aligned(16))) __half vt[kSplitHeadDim * kV6TStride];
  __shared__ __attribute__((aligned(16))) __half pt[2][kVerifyQVecs][kV6TStride];
  __shared__ __attribute__((aligned(16))) float kn_lds[2][kVerifyKeys];
  __shared__ __attribute__((aligned(16))) float vn_lds[2][2][kVerifyKeys];
  __shared__ float rowmax[2][kVerifyQVecs];
  __shared__ __attribute__((aligned(16))) float alpha_il[kVerifyQTiles][16];
  __shared__ unsigned int flags[4];
  __shared__ float l_x[kVerifyQVecs];
  __shared__ unsigned int s_error;

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  // Wave index as a scalar: every role / q-tile / decoder choice below is
  // wave-uniform, so it branches on SGPRs and the plane pointer stays a
  // scalar base for the loads.
  const std::uint32_t wave =
      static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(
          static_cast<int>(tid >> 5U)));
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;

  const std::uint32_t last_context = first_context + rows - 1U;
  const std::uint32_t split_count =
      split_size == 0U ? 0U : (last_context + split_size - 1U) / split_size;
  const std::size_t required_bytes = static_cast<std::size_t>(max_splits) *
                                     num_heads * kSplitRecordWidth *
                                     sizeof(float);
  if (blockDim.x != kV6Threads ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv ||
      kv_head >= num_kv_heads || rows == 0 || rows > kVerifyMaxRows ||
      first_context == 0 || last_context > max_context || split_size == 0U ||
      split_size % kVerifyKeys != 0U || split_count > max_splits ||
      split >= split_count ||
      !VerifyT6PipeFits(formats, num_kv_heads, max_context) ||
      plane_bytes < Turbo4LayerPlaneBytes(kSplitHeadDim, num_kv_heads,
                                          max_context, turbo4::Bits::k6) ||
      partial_bytes < required_bytes || q_rot == nullptr ||
      partials == nullptr || status == nullptr || write_status == nullptr) {
    if (tid == 0)
      Turbo4LatchFailure(status, Turbo4KernelStatus::kGeometryRefused);
    return;
  }
  if (tid == 0) {
    s_error = static_cast<unsigned int>(*status);
    if (*write_status != Turbo4KernelStatus::kOk) {
      s_error = static_cast<unsigned int>(Turbo4KernelStatus::kWriterFailed);
      Turbo4LatchFailure(status, Turbo4KernelStatus::kWriterFailed);
    }
  }
  // Pair table from the 64 f16 centroids, staged in the K tile (free until
  // the first K decode).
  std::uint16_t* const c16 = reinterpret_cast<std::uint16_t*>(kf);
  if (tid < 64U)
    c16[tid] = turbo4::FloatToHalfBits(turbo4::TURBO_CENTROIDS_6BIT[tid]);
  __syncthreads();
  for (std::uint32_t e = tid; e < kVrd6LutEntries; e += kV6Threads)
    lut[e] = Vrd6LutEntry(c16, e);
  __syncthreads();
  if (s_error != 0U)
    return;

  const std::uint32_t nvec = rows * kSplitQHeadsPerKv;
  const std::uint32_t begin = split * split_size;
  const std::uint32_t end =
      last_context < begin + split_size ? last_context : begin + split_size;
  const std::uint32_t tiles = (end - begin + kVerifyKeys - 1U) / kVerifyKeys;
  // Both roles: q-tile qt of the 3 (wave-uniform); waves of an unused q-tile
  // (rows <= 5 leave qt 2 idle) skip their WMMA but keep every barrier.
  const std::uint32_t qt =
      (wave < kV6SWaves ? wave : wave - kV6SWaves) >> 1U;
  const bool active = qt * 16U < nvec;
  const auto record = [&](std::uint32_t vec) {
    return partials +
           static_cast<std::size_t>(vec / kSplitQHeadsPerKv) *
               partials_row_stride +
           (static_cast<std::size_t>(split) * num_heads +
            kv_head * kSplitQHeadsPerKv + vec % kSplitQHeadsPerKv) *
               kSplitRecordWidth;
  };

  if (wave < kV6SWaves) {
    // ======================= S waves 0-5: (qt, kb) =========================
    const std::uint32_t kb = wave & 1U;
    const std::uint32_t qv = qt * 16U + sub;  // this lane's query vector
    const bool live = active && qv < nvec;
    const std::uint32_t row_context =
        live ? first_context + qv / kSplitQHeadsPerKv : 0U;
    // B operand of S^T = K Q^T: query qv, all 256 dims, as
    // __float2half(q_rot) (the attn6 reader's q_frag). Zero past nvec.
    v16h q_frag[kA6KSteps];
    {
      const float4* src = reinterpret_cast<const float4*>(
          q_rot +
          static_cast<std::size_t>(live ? qv / kSplitQHeadsPerKv : 0U) *
              q_rot_row_stride +
          (static_cast<std::size_t>(kv_head) * kSplitQHeadsPerKv +
           (live ? qv % kSplitQHeadsPerKv : 0U)) *
              kSplitHeadDim);
#pragma unroll
      for (std::uint32_t ks = 0; ks < kA6KSteps; ++ks) {
#pragma unroll
        for (std::uint32_t e4 = 0; e4 < 4U; ++e4) {
          const float4 f = src[ks * 4U + e4];
          const float in[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
          for (std::uint32_t e = 0; e < 4U; ++e)
            q_frag[ks][e4 * 4U + e] = __builtin_bit_cast(
                _Float16, __half_as_ushort(__float2half(live ? in[e] : 0.0f)));
        }
        // As in the attn6 reader: keeps the 64 b128 loads from all being
        // hoisted ahead of the conversions.
        asm volatile("" : "+v"(q_frag[ks])::"memory");
      }
    }
    float m_run = -INFINITY;  // log2 units
    float l_run = 0.0f;       // this lane's keys only (half, kb)
    float m_loc = -INFINITY;
    float s[8] = {};
    Vrd6Barrier();  // X(-1): K(0) and its norms staged

    for (std::uint32_t t = 0; t < tiles; ++t) {
      const std::uint32_t tb = begin + t * kVerifyKeys;
      // ---- A(t): S^T over keys 16 kb .. 16 kb + 15, fold, mask, max ----
      if (active) {
        v8f s0 = {}, s1 = {};  // dims 0-127 / 128-255
        const __half* const krow = &kf[(kb * 16U + sub) * kV6KStride];
#pragma unroll
        for (std::uint32_t ks = 0; ks < 8U; ++ks) {
          // Fence every two k-steps: without it the scheduler hoists all 16
          // K fragments (128 VGPRs) on top of the 128 of Q.
          if (ks % 2U == 0U)
            __builtin_amdgcn_sched_barrier(0);
          s0 = VerifyWmma(VerifyLoadFrag(krow + ks * 16U), q_frag[ks], s0);
          s1 = VerifyWmma(VerifyLoadFrag(krow + (ks + 8U) * 16U),
                          q_frag[ks + 8U], s1);
        }
        // Element i = key 16 kb + 2i + half_id: its norms sit at column
        // 16 kb + 8 half_id + i (Vrd6KeyColumn), 8 floats in a row.
        const float4* n0 = reinterpret_cast<const float4*>(
            &kn_lds[0][kb * 16U + 8U * half_id]);
        const float4* n1 = reinterpret_cast<const float4*>(
            &kn_lds[1][kb * 16U + 8U * half_id]);
        const float4 a0 = n0[0], a1 = n0[1], b0 = n1[0], b1 = n1[1];
        const float k0[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float k1[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (std::uint32_t i = 0; i < 8U; ++i)
          s[i] = fmaf(k1[i], s1[i], k0[i] * s0[i]);
        if (tb + kVerifyKeys > first_context) {  // a row's end is in reach
#pragma unroll
          for (std::uint32_t i = 0; i < 8U; ++i)
            s[i] = tb + kb * 16U + 2U * i + half_id < row_context ? s[i]
                                                                  : -INFINITY;
        }
        float mx = s[0];
#pragma unroll
        for (std::uint32_t i = 1; i < 8U; ++i)
          mx = fmaxf(mx, s[i]);
        m_loc = fmaxf(mx, Vrd6OtherHalf(mx));
        if (half_id == 0U)
          rowmax[kb][qv] = m_loc;
      }
      Vrd6Barrier();  // Y(t)

      // ---- B(t): online softmax over the tile's 32 keys, P(t), alpha(t) ----
      if (active) {
        const float m_tile = fmaxf(m_loc, rowmax[kb ^ 1U][qv]);
        const float m_new = fmaxf(m_run, m_tile);
        // Fully masked so far: any finite pivot (p = 0, alpha unused).
        const float m_safe = m_new == -INFINITY ? 0.0f : m_new;
        const bool moved = m_new != m_run;
        const float alpha =
            moved ? __builtin_amdgcn_exp2f(m_run - m_safe) : 1.0f;
        float p[8];
#pragma unroll
        for (std::uint32_t i = 0; i < 8U; ++i)
          p[i] = __builtin_amdgcn_exp2f(s[i] - m_safe);
        l_run = fmaf(l_run, alpha,
                     ((p[0] + p[1]) + (p[2] + p[3])) +
                         ((p[4] + p[5]) + (p[6] + p[7])));
        m_run = m_new;
        const float4* v0 = reinterpret_cast<const float4*>(
            &vn_lds[t & 1U][0][kb * 16U + 8U * half_id]);
        const float4* v1 = reinterpret_cast<const float4*>(
            &vn_lds[t & 1U][1][kb * 16U + 8U * half_id]);
        const float4 c0 = v0[0], c1 = v0[1], d0 = v1[0], d1 = v1[1];
        const float n_v0[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        const float n_v1[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
        __attribute__((aligned(16))) __half w0[8];
        __attribute__((aligned(16))) __half w1[8];
#pragma unroll
        for (std::uint32_t i = 0; i < 8U; ++i) {
          w0[i] = __float2half(p[i] * n_v0[i]);
          w1[i] = __float2half(p[i] * n_v1[i]);
        }
        *reinterpret_cast<uint4*>(&pt[0][qv][kb * 16U + 8U * half_id]) =
            *reinterpret_cast<const uint4*>(w0);
        *reinterpret_cast<uint4*>(&pt[1][qv][kb * 16U + 8U * half_id]) =
            *reinterpret_cast<const uint4*>(w1);
        if (kb == 0U) {  // both kb waves hold the same m, alpha
          // PV lanes read alpha of rows 2i + h as 8 floats: [qt][h][i].
          if (half_id == 0U)
            alpha_il[qt][(sub & 1U) * 8U + (sub >> 1U)] = alpha;
          const bool any_moved = __builtin_amdgcn_ballot_w32(moved) != 0U;
          if (lane == 0U)
            flags[qt] = any_moved ? 1U : 0U;
        }
      }
      Vrd6Barrier();  // X(t)
    }

    // A(tiles) is the PV waves' last PV; here the four partial sums of l
    // (two halves, two kb waves) meet: kb 1 posts its pair sum.
    float l_kb = 0.0f;
    if (active) {
      l_kb = l_run + Vrd6OtherHalf(l_run);
      if (kb == 1U && half_id == 0U)
        l_x[qv] = l_kb;
    }
    Vrd6Barrier();  // Z
    // Partial record head {m, l}, natural-log m for the reduce kernel;
    // empty (row ends before this split): m = -FLT_MAX, l = 0. s_error is
    // final: every latch precedes an X barrier.
    if (live && kb == 0U && half_id == 0U && s_error == 0U) {
      float* const rec = record(qv);
      rec[0] = isfinite(m_run) ? m_run * kV6Ln2 : -3.402823466e+38f;
      rec[1] = l_kb + l_x[qv];
    }
  } else {
    // ======================= PV waves 6-11: (qt, blk) ======================
    const std::uint32_t w6 = wave - kV6SWaves;
    const std::uint32_t blk = w6 & 1U;  // V dims [128 blk, 128 blk + 128)
    // Decoders (wave-uniform): w6 0-2 decode K of tile t + 1, w6 3-5 V of
    // tile t, one key per lane; dk = w6 % 3 takes 32-dim groups 3 dk + r
    // (r = 0, 1, 2; dk 2 has groups 6, 7 only). dk 2 also stages the norms
    // of tile t + 1, block 0 (K decoder) or 1 (V decoder): the waves with two
    // groups carry the norm work. Every wave issues the same 8 loads per job
    // (dk 2 reloads group 7, every wave loads the two norm words), so the
    // load counter is static and a decode waits only for its own job.
    const bool k_dec = w6 < 3U;
    const std::uint32_t dk = k_dec ? w6 : w6 - 3U;
    const bool norm_dec = dk == 2U;
    const std::uint32_t norm_blk = k_dec ? 0U : 1U;
    const std::uint8_t* const plane = k_dec ? k_plane : v_plane;
    const float k_scale = scale * kV6Log2e;
    const std::uint32_t t6_block_bytes =
        static_cast<std::uint32_t>(turbo4::BlockBytes(turbo4::Bits::k6));
    const std::uint32_t t6_token_bytes = static_cast<std::uint32_t>(
        Turbo4BlocksPerToken(kSplitHeadDim, num_kv_heads) * t6_block_bytes);
    const std::uint32_t t6_head_bytes =
        kv_head *
        static_cast<std::uint32_t>(Turbo4BlocksPerHead(kSplitHeadDim)) *
        t6_block_bytes;
    // Byte offset of this lane's token in tile `tile` (past the end: clamped
    // to the last live token; zeroed by its norm, masked by the softmax).
    const auto token_off = [&](std::uint32_t tile) {
      const std::uint32_t tok = begin + tile * kVerifyKeys + lane;
      return (tok < end ? tok : end - 1U) * t6_token_bytes + t6_head_bytes;
    };
    // Raw words of job jt: this wave's groups of K(jt + 1) or V(jt), and the
    // K and V norm words of block norm_blk of tile jt + 1. Unconditional.
    const auto load_job = [&](std::uint32_t jt, Vrd6Job* job) {
      const std::uint32_t off = token_off(k_dec ? jt + 1U : jt);
#pragma unroll
      for (std::uint32_t r = 0; r < kV6Rounds; ++r) {
        const std::uint32_t g = 3U * dk + r < 8U ? 3U * dk + r : 7U;
        Vrd6LoadSlice(plane, off + (g >> 2U) * t6_block_bytes, g & 3U,
                      job->w[r], job->qh[r]);
      }
      const std::uint32_t noff = token_off(jt + 1U) + norm_blk * t6_block_bytes;
      __builtin_memcpy(&job->kn, __builtin_assume_aligned(k_plane + noff, 2),
                       4);
      __builtin_memcpy(&job->vn, __builtin_assume_aligned(v_plane + noff, 2),
                       4);
    };
    // One group: pair-table gathers, then K row (key = lane) into kf, or V
    // column (Vrd6KeyColumn(lane)) into vt.
    const auto decode_group = [&](std::uint32_t g, const std::uint32_t* w,
                                  const std::uint32_t* qh) {
      const char* const table = reinterpret_cast<const char*>(lut);
      std::uint32_t off[16], pairs[16];
      Vrd6PairOffsets(w, qh, off);
#pragma unroll
      for (std::uint32_t m = 0; m < 16U; ++m)
        pairs[m] = *reinterpret_cast<const std::uint32_t*>(table + off[m]);
      if (k_dec) {
        uint4* const dst =
            reinterpret_cast<uint4*>(&kf[lane * kV6KStride + g * 32U]);
#pragma unroll
        for (std::uint32_t i = 0; i < 4U; ++i)
          dst[i] = make_uint4(pairs[4U * i], pairs[4U * i + 1U],
                              pairs[4U * i + 2U], pairs[4U * i + 3U]);
      } else {
        __half* const dst = &vt[g * 32U * kV6TStride + Vrd6KeyColumn(lane)];
#pragma unroll
        for (std::uint32_t m = 0; m < 16U; ++m) {
          dst[2U * m * kV6TStride] =
              __ushort_as_half(static_cast<unsigned short>(pairs[m]));
          dst[(2U * m + 1U) * kV6TStride] =
              __ushort_as_half(static_cast<unsigned short>(pairs[m] >> 16));
        }
      }
    };
    // Decode job jt (B phase of tile jt): groups of K(jt + 1) (if that tile
    // exists) or V(jt) (if do_v), and the norms of tile jt + 1.
    const auto decode_job = [&](std::uint32_t jt, const Vrd6Job& job,
                                bool do_v) {
      const bool k_live = jt + 1U < tiles;
      if (k_dec ? k_live : do_v) {
        decode_group(3U * dk, job.w[0], job.qh[0]);
        decode_group(3U * dk + 1U, job.w[1], job.qh[1]);
        if (dk < 2U)
          decode_group(3U * dk + 2U, job.w[2], job.qh[2]);
      }
      if (norm_dec && k_live) {
        // Turbo6Unpack's rule: past the end both norms are 0 (not poison); a
        // non-finite K or V norm zeroes both and latches kPoisonedNorm.
        const bool valid = begin + (jt + 1U) * kVerifyKeys + lane < end;
        std::uint32_t kbits = job.kn, vbits = job.vn;
        asm("" : "+v"(kbits), "+v"(vbits));  // keep whole 32-bit VGPRs
        kbits = valid ? kbits : 0U;
        vbits = valid ? vbits : 0U;
        const float k_norm =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(kbits)));
        const float v_norm =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(vbits)));
        const bool poisoned = !turbo4::StoredNormIsUsable(k_norm) ||
                              !turbo4::StoredNormIsUsable(v_norm);
        if (poisoned)
          atomicMax(&s_error, static_cast<unsigned int>(
                                  Turbo4KernelStatus::kPoisonedNorm));
        const std::uint32_t col = Vrd6KeyColumn(lane);
        kn_lds[norm_blk][col] = poisoned ? 0.0f : k_norm * k_scale;
        vn_lds[(jt + 1U) & 1U][norm_blk][col] = poisoned ? 0.0f : v_norm;
      }
    };

    v8f o[8] = {};  // O rows qt*16 + 2i + half_id, dims 128 blk + 16 j + sub
    // A phase: rescale by alpha(t - 1) when any row's max moved, then
    // O += P_blk(t - 1) V(t - 1), one 16-key k-step at a time: its 8 V
    // fragments are loaded together, so the LDS latency shows once per
    // k-step, not once per WMMA.
    const auto pv = [&]() {
      const unsigned int moved = static_cast<unsigned int>(
          __builtin_amdgcn_readfirstlane(static_cast<int>(flags[qt])));
      if (moved != 0U) {
        const float4* a =
            reinterpret_cast<const float4*>(&alpha_il[qt][8U * half_id]);
        const float4 a0 = a[0], a1 = a[1];
        const float al[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (std::uint32_t j = 0; j < 8U; ++j)
#pragma unroll
          for (std::uint32_t i = 0; i < 8U; ++i)
            o[j][i] *= al[i];
      }
#pragma unroll
      for (std::uint32_t kb = 0; kb < 2U; ++kb) {
        const v16h pa = VerifyLoadFrag(&pt[blk][qt * 16U + sub][kb * 16U]);
        v16h vf[8];
#pragma unroll
        for (std::uint32_t j = 0; j < 8U; ++j)
          vf[j] = VerifyLoadFrag(
              &vt[(blk * 128U + j * 16U + sub) * kV6TStride + kb * 16U]);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (std::uint32_t j = 0; j < 8U; ++j)
          o[j] = VerifyWmma(pa, vf[j], o[j]);
        __builtin_amdgcn_sched_barrier(0);
      }
    };

    Vrd6Job ja, jb;
    load_job(0xFFFFFFFFU, &ja);  // job -1: K(0) and the norms of tile 0
    decode_job(0xFFFFFFFFU, ja, false);
    load_job(0U, &ja);
    load_job(1U, &jb);
    Vrd6Barrier();  // X(-1)
    const auto step = [&](std::uint32_t t, Vrd6Job* job) {
      if (t > 0U && active)
        pv();  // A(t): PV(t - 1)
      Vrd6Barrier();  // Y(t)
      decode_job(t, *job, true);
      load_job(t + 2U, job);
      Vrd6Barrier();  // X(t)
    };
    // Two jobs in flight; the x2 unroll keeps each in its own registers. Both
    // steps of the body are unconditional and an odd tail is peeled: with a
    // conditional second step the load counter merges two paths at the loop
    // head, and the first step's decode then waited for the other job too.
    std::uint32_t t = 0;
    for (; t + 1U < tiles; t += 2U) {
      step(t, &ja);
      step(t + 1U, &jb);
    }
    if (t < tiles)
      step(t, &ja);
    if (active)
      pv();  // A(tiles): PV(tiles - 1)
    Vrd6Barrier();  // Z

    if (active && s_error == 0U) {
#pragma unroll
      for (std::uint32_t j = 0; j < 8U; ++j)
#pragma unroll
        for (std::uint32_t i = 0; i < 8U; ++i) {
          const std::uint32_t vec = qt * 16U + 2U * i + half_id;
          if (vec < nvec)
            record(vec)[2U + blk * 128U + j * 16U + sub] = o[j][i];
        }
    }
  }
  if (s_error != 0U && tid == 0)
    Turbo4LatchFailure(status, static_cast<Turbo4KernelStatus>(s_error));
}

}  // namespace

Turbo4LaunchStatus LaunchTurbo4AttentionVerifyVrd6(
    const float* q, const std::uint8_t* k_plane, const std::uint8_t* v_plane,
    float* out, float* q_rot, float* out_rot, float* partials,
    std::size_t plane_bytes, std::size_t q_bytes, std::size_t out_bytes,
    std::size_t q_rot_bytes, std::size_t out_rot_bytes,
    std::size_t partial_bytes, const float* gate, std::size_t gate_bytes,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t first_valid_context, std::uint32_t rows,
    std::uint32_t max_context, float scale,
    const Turbo4KernelStatus* d_write_status, Turbo4KernelStatus* d_status,
    hipStream_t stream, turbo4::PlaneFormats formats,
    std::uint32_t splits_cap) {
  if (!formats.Valid() || !VerifyT6PipeFits(formats, num_kv_heads, max_context) ||
      GufoDevVSkipThreshold() > 0.0f)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (q == nullptr || k_plane == nullptr || v_plane == nullptr ||
      out == nullptr || q_rot == nullptr || out_rot == nullptr ||
      partials == nullptr || d_write_status == nullptr || d_status == nullptr ||
      plane_bytes == 0 || num_heads == 0 || num_kv_heads == 0 || rows == 0)
    return Turbo4LaunchStatus::kBadArguments;
  if (max_context == 0 || first_valid_context == 0 || rows > kVerifyMaxRows ||
      rows > max_context || first_valid_context > max_context - rows + 1U)
    return Turbo4LaunchStatus::kGeometryRefused;
  if (head_dim != kSplitHeadDim ||
      num_heads != num_kv_heads * kSplitQHeadsPerKv || !std::isfinite(scale))
    return Turbo4LaunchStatus::kGeometryRefused;
  if ((gate_bytes == 0U) != (gate == nullptr))
    return Turbo4LaunchStatus::kBadArguments;

  const std::size_t vector_floats =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t rows_bytes = rows * vector_floats * sizeof(float);
  const std::size_t row_partial_bytes = Turbo4BatchedDecodePartialBytes(
      first_valid_context, rows, num_heads, head_dim);
  const std::size_t required_plane = Turbo4LayerPlaneBytes(
      head_dim, num_kv_heads, max_context, formats.stride);
  if (required_plane == 0U || plane_bytes < required_plane ||
      row_partial_bytes == 0U || partial_bytes < rows * row_partial_bytes ||
      q_bytes < rows_bytes || out_bytes < rows_bytes ||
      q_rot_bytes < rows_bytes || out_rot_bytes < rows_bytes ||
      (gate != nullptr && gate_bytes < rows_bytes))
    return Turbo4LaunchStatus::kBadArguments;

  const std::uint32_t last_context = first_valid_context + rows - 1U;
  const std::uint32_t max_splits = static_cast<std::uint32_t>(
      row_partial_bytes / (num_heads * (head_dim + 2) * sizeof(float)));
  // Batch-invariant mode lays the split SIZE out for the width-8 last row; the
  // COUNT stays on this launch's real last row.
  const std::uint32_t split_size = static_cast<std::uint32_t>(Attn6SplitSize(
      BatchInvariantPlanContext(first_valid_context, rows, max_context),
      splits_cap));
  const std::uint32_t split_count =
      split_size == 0U ? 0U : (last_context + split_size - 1U) / split_size;
  if (max_splits == 0 || max_splits > kSplitHeadDim || split_count == 0 ||
      split_count > max_splits)
    return Turbo4LaunchStatus::kGeometryRefused;

  Turbo4RotateQueryKernel<<<dim3(num_heads, rows), turbo4::kBlockValues, 0,
                            stream>>>(q, q_rot, num_heads, head_dim, q_bytes,
                                      q_rot_bytes, d_status, vector_floats);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;

  const std::size_t partials_row_stride = row_partial_bytes / sizeof(float);
  Turbo4VerifyVrd6Kernel<<<dim3(num_kv_heads, split_count), kV6Threads, 0,
                           stream>>>(
      q_rot, k_plane, v_plane, partials, plane_bytes, row_partial_bytes,
      num_heads, num_kv_heads, first_valid_context, rows, max_context,
      max_splits, split_size, scale, d_write_status, d_status, vector_floats,
      partials_row_stride, formats);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;

  Turbo4AttentionReduceInverseKernel<<<dim3(num_heads, rows), kSplitHeadDim,
                                       max_splits * sizeof(float), stream>>>(
      partials, out_rot, out, gate, row_partial_bytes, out_rot_bytes, out_bytes,
      gate_bytes, num_heads, num_kv_heads, split_count, max_splits, head_dim,
      d_status, first_valid_context, partials_row_stride, vector_floats,
      /*uniform_splits=*/true);
  if (hipPeekAtLastError() != hipSuccess)
    return Turbo4LaunchStatus::kEnqueueFailed;
  return Turbo4LaunchStatus::kLaunched;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

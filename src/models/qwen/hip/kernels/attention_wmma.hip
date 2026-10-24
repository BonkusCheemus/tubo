// Masked prefill attention on the WMMA matrix cores. One pass covers the
// visible prefix and causal diagonal, using the canonical FP16 KV cache.
// Validated against the CPU double-precision reference below.
//
// Design. Wave32 fragment layout, verified against a CPU double-precision
// reference: A holds row L%16 and 16 contiguous k, B holds column L%16 and 16
// contiguous k (so B is fed a transposed tile), and C element i is row
// 2i + L/16, column L%16.
//
// Three things set the shape of this kernel, each found by ablation:
//
//   * Global request count dominates, not bytes. Every block re-reads K and V
//     for its key range, so the cost per query row falls as the block covers
//     more rows. 32 rows per block spends 2.43 ms of a 5.03 ms call on global
//     memory; 64 rows halves that to 0.94 ms of 3.24 ms. 64 is the most the
//     eight-wave split supports without spilling.
//   * V has to reach LDS transposed, because both WMMA operands want 16
//     contiguous values of the contracted dimension and V is stored
//     [position][head][dim]. Writing it with each lane owning 8 contiguous dims
//     of one key -- what the coalesced global read wants -- puts all 32 lanes
//     of a wave on LDS rows kVtStride halves apart, which is a single bank and
//     a 32-way conflict. Giving each lane one key instead costs a strided
//     global read but makes the LDS writes conflict-free, and was worth 1.8x.
//   * V's global load must not sit at its point of use, behind the barriers it
//     needs, where its latency is fully exposed. Hoisting it to the top of the
//     same key-tile iteration was worth 1.14x; `opt-c178-attn-prefetch` below
//     goes further and moves it a whole tile ahead.
//
// Barriers are not a cost here in the sense of bank or issue cost: removing all
// four per key tile changes nothing measurable at depth 0. They do serialize
// the phases, which is a different and real cost -- see the latency note below.
//
// opt-c178-attn-prefetch. What the ablations above actually price is *exposed
// latency*, not work. Per SIMD the kernel issued one WMMA every 223 cycles
// against a ~64-cycle issue cost, so the SIMD sat idle 71% of the time: 23 KB
// of LDS fits only two blocks per CU, giving four waves per SIMD to cover a
// dependency chain that all eight waves of a block walk in lockstep, and whose
// head is a global load. Loading tile i+1's K and V into a second set of
// registers immediately after tile i's staging barrier puts that load behind
// tile i's S, softmax and PV phases instead of in front of them, for 16 VGPRs
// and 16 v_mov per tile. One layer call at batch 2048, interleaved with the
// unpipelined kernel to cancel APU throttling:
//
//   depth      before    after   speedup
//       0       3.250    2.500     1.30x
//    4096      15.082   13.022     1.16x
//    8192      27.026   23.955     1.13x
//   16384      51.595   46.107     1.12x
//
// 15.9 -> 20.6 TFLOPS at depth 0 and 17.2 -> 19.4 at depth 8192. The staged
// bytes, their order and the WMMA sequence are all unchanged, so the result is
// bit-identical, which the oracle test asserts directly.
//
// Two things measured alongside it and rejected, both recorded in the model
// card: remapping V's global read so four lanes cover 32 contiguous dims of one
// key (fewer, fully-used line requests, and it lets the transpose write 8
// packed dwords instead of 16 halves) is 4% faster at depth 0 but 30% *slower*
// at depth 16384; and dropping the V^T row padding to fit three blocks per CU
// costs more in fragment-read bank conflicts than the occupancy returns.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace gufo::hip {

// Defined in kernels/attention_tile.hip; the WMMA route reuses the same FP16
// cache and the same pack/sync pair unchanged.
__global__ void PackTiledAttentionKvKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context);

__global__ void SyncTiledAttentionKvPrefixKernel(
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t prefix_length,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim);

namespace {

constexpr std::uint32_t kHeadDim = detail::kTiledAttentionHeadDim;
constexpr std::uint32_t kQueryHeads = detail::kTiledAttentionQueryHeads;
constexpr std::uint32_t kKvHeads = detail::kTiledAttentionKvHeads;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A fragment is 16 contiguous halves = 32 bytes. Reading it element by element
// costs 16 ds_read_u16 plus 16 __half -> _Float16 conversions; the two formats
// are bit-identical, so two 16-byte loads and a bitcast do the same work in two
// instructions. Every fragment base here is 16-byte aligned by construction.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kWmmaHeads =
    2;  // query heads per block, must divide GQA
// Measured tile: 32 queries x 2 heads = 64 rows per block against 16 keys. See
// the header note on why more rows per block is the lever that matters.
constexpr std::uint32_t kWmmaQueryRows = 32;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;

// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

}  // namespace

// External linkage on purpose: an anonymous-namespace kernel is dispatched with
// an empty name, and rocprofv3 then merges it into an unattributed "other"
// bucket -- 7.3% of a depth-8192 profile in this kernel's case.
//
// kQueryRows queries x kWmmaHeads heads form (kQueryRows / 16) * kWmmaHeads
// 16-row blocks, and kKeys keys form kKeys / 16 key blocks. Their product is
// pinned to 4 so the eight waves split as four S tiles x two halves of the head
// dimension, which is what keeps Q in registers at 64 VGPRs. Growing kQueryRows
// and shrinking kKeys in step therefore trades nothing in the tiling but halves
// the K/V bytes each block pulls per query row -- the dominant cost.
template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kHeadMajor = false>
__launch_bounds__(256, 2) __global__
    void WmmaCausalAttention(const float* __restrict__ q,
                             const float* __restrict__ gate,
                             const __half* __restrict__ k_cache_f16,
                             const __half* __restrict__ v_cache_f16,
                             float* __restrict__ out_context,
                             std::uint32_t start_pos, std::size_t batch_size,
                             float* __restrict__ lse_out,
                             std::uint32_t key_begin) {
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == 4, "eight waves cover four S tiles in two halves");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 8 / kSTiles == kWmmaKSteps, "k split");
  static_assert(!kHeadMajor || kKeys == 16, "packed value tile width");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row block -> (query head, 16-query sub-block of this block's rows).
  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves =
      kHeadMajor || kKeys * kWmmaKStride > kHeadDim * kVtStride
          ? kKeys * kWmmaKStride
          : kHeadDim * kVtStride;
  __shared__ __half kv_lds[kKvLdsHalves];
  // Padding avoids repeated banks in softmax reads and probability fragments.
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block, so both
  // halves of the head dimension stay evenly loaded.
  v8f o_acc[kRowBlocks][2] = {};
  // Each softmax row keeps the same four owner lanes across the key sweep.
  // Only the final statistics and each tile's rescale factor cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // One key per lane for V, a 16-dim slice per thread. This costs a strided
  // global read but makes the transpose's LDS writes conflict-free, which is
  // worth 1.8x; a coalesced remapping was measured and is slower at depth.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  constexpr std::size_t kv_stride = kHeadMajor ? kHeadDim : kKvWidth;
  const std::size_t head_offset = static_cast<std::size_t>(kv_head) * kHeadDim *
                                  (kHeadMajor ? context_end : 1);
  const auto* v_base = v_cache_f16 + head_offset + v_slice;
  const auto* k_base = k_cache_f16 + head_offset;

  using ValueFragment = std::conditional_t<kHeadMajor, v16h, uint4>;
  constexpr std::uint32_t kValueFragments = kHeadMajor ? 2 : kVRegs;
  const auto load_v = [&](std::uint32_t key_start, ValueFragment* dst) {
    if constexpr (kHeadMajor) {
      // Each wave consumes two dimension tiles. The temporary value layout
      // supplies their WMMA fragments directly, avoiding a repeated LDS
      // transpose and barrier for every 16-key tile.
      const std::size_t padded = (context_end + 15u) & ~std::size_t{15};
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
        const std::uint32_t dim = (wave + t * 8u) * 16u + sub;
        dst[t] = LoadFrag(
            v_cache_f16 +
            static_cast<std::size_t>(kv_head) * padded * kHeadDim +
            static_cast<std::size_t>(key_start / 16u) * kHeadDim * 16u +
            dim * 16u);
      }
    } else {
      const std::uint32_t key_position = key_start + v_key;
      const auto* src =
          v_base + (static_cast<std::size_t>(key_position) * kv_stride);
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                      : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] =
          (key_position < context_end)
              ? *reinterpret_cast<const uint4*>(
                    k_base +
                    (static_cast<std::size_t>(key_position) * kv_stride) + d8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  ValueFragment v_cur[kValueFragments];
  uint4 k_pre[kKRegs] = {};
  ValueFragment v_pre[kValueFragments] = {};
  load_k(key_begin, k_cur);
  load_v(key_begin, v_cur);

  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kKeys) {
    // --- stage K from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    __syncthreads();

    // --- prefetch the next key tile. Everything below covers its latency.
    const std::uint32_t next_start = key_start + kKeys;
    if (next_start < max_visible) {
      load_k(next_start, k_pre);
      if constexpr (!kHeadMajor)
        load_v(next_start, v_pre);
    }

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
      // Each half writes its own slot, so the reader sums them instead of a
      // read-modify-write that would need a barrier between the halves.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();

    // The unpacked path shares each scale across both dimension tiles. Packed
    // V instead rescales immediately before PV, shortening register lifetimes
    // enough to retain six waves per SIMD with the causal-tail correction.
    if constexpr (!kHeadMajor) {
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        float scale[8];
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
        }
#pragma unroll
        for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            o_acc[rb][t][i] *= scale[i];
          }
        }
      }
    }

    // --- stage V transposed. No barrier is needed here: the barrier after the
    // softmax already separates every read of K in the S phase from these
    // writes to the same buffer.
    //
    // Lane L owns key L % kKeys and each thread a dim slice, so consecutive
    // lanes write consecutive *columns* of V^T: two 16-bit writes share a dword
    // and the wave covers 16 banks. The mapping the coalesced global read would
    // prefer -- each lane taking 8 contiguous dims of one key -- puts all 32
    // lanes on rows kVtStride halves apart, which is one bank, and costs a
    // 32-way conflict on every write.
    if constexpr (!kHeadMajor) {
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
      __syncthreads();
    }

    // --- O += P V ---
    // Packed V is in registers: visit each row block's two dimension tiles
    // together, sharing its scales. Unpacked V comes from LDS: visit dimension
    // tiles first so the compiler can share their fragments across row blocks.
    // The order is compile-time constant and both paths keep the same
    // arithmetic.
#pragma unroll
    for (std::uint32_t outer = 0; outer < (kHeadMajor ? kRowBlocks : 2);
         ++outer) {
      float scale[8];
      if constexpr (kHeadMajor) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          scale[i] = row_scale[outer * 16 + 2 * i + half_id];
      }
#pragma unroll
      for (std::uint32_t inner = 0; inner < (kHeadMajor ? 2 : kRowBlocks);
           ++inner) {
        const std::uint32_t rb = kHeadMajor ? outer : inner;
        const std::uint32_t t = kHeadMajor ? inner : outer;
        const std::uint32_t dim_tile = wave + t * 8;
        v16h v_frag[kKeyBlocks];
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          if constexpr (kHeadMajor)
            v_frag[kb] = v_cur[t];
          else
            v_frag[kb] =
                LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride + kb * 16]);
        }
        if constexpr (kHeadMajor) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            o_acc[rb][t][i] *= scale[i];
        }
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          const auto first_key = key_start + kb * 16;
          // Like Flash-Next, sum a query's final partial tile over visible
          // keys only. WMMA can round differently when a zero probability
          // multiplies a populated future value instead of chunk padding.
          // Complete tiles keep the matrix-core path and its original order.
          if (first_key + 15 > start_pos + query_start) {
            v8f tail = o_acc[rb][t];
            // Keep the position difference uniform across the wave; only the
            // even/odd row owner varies between its two half-waves.
            const int relative_key = static_cast<int>(first_key) -
                                     static_cast<int>(start_pos + query_start +
                                                      row_block_offset(rb));
#pragma unroll
            for (std::uint32_t key = 0; key < 16; ++key) {
              const float value = static_cast<float>(v_frag[kb][key]);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                if (static_cast<int>(half_id) <
                        relative_key + 15 - static_cast<int>(2 * i) &&
                    static_cast<int>(half_id) >= relative_key +
                                                     static_cast<int>(key) -
                                                     static_cast<int>(2 * i)) {
                  tail[i] =
                      fmaf(__half2float(
                               p_lds[rb * 16 + 2 * i + half_id][kb * 16 + key]),
                           value, tail[i]);
                }
              }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i)
              if (static_cast<int>(half_id) <
                  relative_key + 15 - static_cast<int>(2 * i))
                next[i] = tail[i];
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
    if constexpr (kHeadMajor) {
      // Packed V is already in WMMA order. Load it after the current PV work;
      // the next QK/softmax hides its latency without retaining a second set
      // of fragments across the causal-tail calculation.
      if (next_start < max_visible)
        load_v(next_start, v_cur);
    } else {
#pragma unroll
      for (std::uint32_t n = 0; n < kValueFragments; ++n)
        v_cur[n] = v_pre[n];
    }
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr && lse_out == nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
  if (lse_out != nullptr && wave == 0) {
    for (std::uint32_t rg = lane; rg < kRows; rg += 32) {
      const std::uint32_t rb = rg / 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + (rg % 16);
      if (local_query >= batch_size) {
        continue;
      }
      const float denominator = row_sum[rg];
      lse_out[(static_cast<std::size_t>(row_block_head(rb)) * batch_size) +
              local_query] =
          (denominator > 0.0F) ? (row_max[rg] + logf(denominator)) : -INFINITY;
    }
  }
}

// ---------------------------------------------------------------------------
// attn5 V2 (GUFO_DEV_ATTN_V2=1): the packed-heads kernel above, rescheduled.
//
// Every row sees the same S tiles (the same 16 queries x 16 keys, the same two
// eight-step half-dimension WMMA chains summed as s0 + s1), the same softmax
// arithmetic, and per O tile the same rescale -> WMMA -> visible-key FMA tail
// in the same key order. Only data movement and scheduling differ:
//
//   * LDS-only barriers. __syncthreads in WGP mode is s_waitcnt vmcnt(0) +
//     s_barrier + buffer_gl0_inv: every barrier waited for the register
//     prefetch, so the next tile's V load (issued at the end of a tile) was
//     exposed in full at the first barrier of the next one. All cross-wave
//     traffic here is LDS; the global inputs are read-only.
//   * Two barriers per 16-key tile instead of four. Phase A holds tile j's PV
//     and tile j + 1's S, phase B tile j + 1's softmax and the staging of
//     K(j + 2). One K buffer suffices: S(j + 1) has read it before the phase A
//     barrier, and K(j + 2) is written after it. V(j + 1) is loaded during
//     phase A and K(j + 3) during phase B, each a full phase before its use.
//   * K loads are unconditional (rows past the context read the last row and
//     are staged as zeros), so every path into the loop has the same loads in
//     flight and the wait counts stay exact: with a conditional load the
//     compiler had to wait for every load, the next K included, before PV.
//   * Score rows padded to 20 floats and read as two float4, rescale factors
//     stored by row parity and read as two float4, the 4-lane max/sum
//     shuffles as DPP moves instead of ds_bpermute round trips.
//   * The causal tail is decided once per tile. Inside a diagonal tile each
//     row block either keeps the WMMA (every row sees the whole tile), copies
//     the rescaled O (no row sees any of it), or runs one predicated FMA chain
//     per row from two 16-byte probability loads, instead of 128 single-half
//     LDS round trips behind 128 exec-mask branches.
//   * Block order: the three head pairs that share a KV head and a query
//     block are dispatched together, heaviest query block first.
// ---------------------------------------------------------------------------
namespace {

// A barrier for waves that exchange data through LDS only (as in the
// flash-next port of this kernel).
__device__ __forceinline__ __attribute__((convergent)) void SyncLds() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

// Lane i receives lane (i ^ offset)'s value, as __shfl_xor(v, offset) does.
// On gfx1151 offsets 1 and 2 are a DPP row_xmask register move instead of a
// ds_bpermute round trip; only the transport differs. All lanes are active.
__device__ __forceinline__ float XorLane(float v, int offset) {
#if defined(__gfx1151__)
  const int bits = __builtin_bit_cast(int, v);
  switch (offset) {
    case 1:
      return __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, bits, 0x161, 0xF, 0xF, false));
    case 2:
      return __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, bits, 0x162, 0xF, 0xF, false));
    default:
      break;
  }
#endif
  return __shfl_xor(v, offset);
}

}  // namespace

template<std::uint32_t kQueryRows, std::uint32_t kKeys>
__launch_bounds__(256, 2) __global__
    void WmmaCausalAttentionV2(const float* __restrict__ q,
                               const float* __restrict__ gate,
                               const __half* __restrict__ k_cache_f16,
                               const __half* __restrict__ v_cache_f16,
                               float* __restrict__ out_context,
                               std::uint32_t start_pos, std::size_t batch_size,
                               float* __restrict__ lse_out,
                               std::uint32_t key_begin) {
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
  constexpr std::uint32_t kScoreStride = 20;
  static_assert(kKeys == 16 && kSTiles == 4, "packed value tile, 4 S tiles");
  static_assert(kSoftmaxLanes == 4 && kPerLane == 4, "one float4 per lane");
  static_assert(kRowBlocks * 2 * 8 == 64, "two dimension tiles per wave");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  // Dispatch order (linear block id, x fastest): the three head pairs of one
  // KV head and query block are consecutive, so their identical key sweeps
  // run side by side and two of every three K/V tile reads can hit L2; and
  // the heaviest query block (most causal keys) goes first, so the short
  // blocks fill the end of the dispatch. A bijection onto the same blocks.
  constexpr std::uint32_t kPairs = kGqa / kWmmaHeads;
  const std::uint32_t linear = (blockIdx.y * gridDim.x) + blockIdx.x;
  const std::uint32_t group = linear / kPairs;
  const std::uint32_t pair_in_group = linear % kPairs;
  const std::uint32_t kv_head = group / gridDim.x;
  const std::uint32_t query_start =
      (gridDim.x - 1u - (group % gridDim.x)) * kQueryRows;
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  __shared__ __attribute__((aligned(16))) __half k_lds[kKeys * kWmmaKStride];
  __shared__
      __attribute__((aligned(16))) float s_lds[2][kSTiles][16][kScoreStride];
  __shared__ __attribute__((aligned(16))) __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  // [row block][row % 2][row / 2]: a PV lane needs the eight rows of its
  // parity, two float4 reads.
  __shared__ __attribute__((aligned(16))) float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  v8f o_acc[kRowBlocks][2] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  // V(j) in WMMA B order for this wave's two dimension tiles; K(j + 2) in
  // flight between its load in phase B and its staging one tile later.
  v16h v_cur[2] = {};
  uint4 k_regs[(kKeys * (kHeadDim / 8)) / 256] = {};

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);
  const std::uint32_t n_tiles =
      key_begin < max_visible ? (max_visible - key_begin + kKeys - 1) / kKeys
                              : 0u;
  const auto tile_key = [&](std::uint32_t j) { return key_begin + j * kKeys; };
  // A tile needs the visible-key tail iff a query of this block has keys of
  // it in its future; tiles only move forward, so once true it stays true.
  const auto diagonal = [&](std::uint32_t key_start) {
    return key_start + 15 > start_pos + query_start;
  };

  // Packed layouts (PackAttentionHeads): K [head][context_end][dim], V
  // [head][tile][dim][16 keys].
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const auto* k_base = k_cache_f16 + static_cast<std::size_t>(kv_head) *
                                         kHeadDim * context_end;
  const std::size_t padded = (context_end + 15u) & ~std::size_t{15};
  const auto* v_head =
      v_cache_f16 + static_cast<std::size_t>(kv_head) * padded * kHeadDim;

  // Rows past the context read its last row; stage_k writes zeros for them,
  // as the served kernel's conditional load does.
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position =
          min(key_start + (idx / (kHeadDim / 8)), context_end - 1u);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] = *reinterpret_cast<const uint4*>(
          k_base + (static_cast<std::size_t>(key_position) * kHeadDim) + d8);
    }
  };
  const auto load_v = [&](std::uint32_t key_start, v16h* dst) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim = (wave + t * 8u) * 16u + sub;
      dst[t] = LoadFrag(v_head +
                        static_cast<std::size_t>(key_start / 16u) * kHeadDim *
                            16u +
                        dim * 16u);
    }
  };
  const auto stage_k = [&](std::uint32_t key_start, const uint4* regs) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&k_lds[(key_row * kWmmaKStride) + d8]) =
          (key_start + key_row < context_end) ? regs[n]
                                              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  // S = Q K^T for the tile in k_lds; each half writes its own slot.
  const auto scores = [&] {
    v8f s_acc = {};
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const v16h k_frag =
          LoadFrag(&k_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
      s_acc = Wmma(q_frag[ks], k_frag, s_acc);
    }
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
    }
  };

  // Online softmax of the scores in s_lds, kSoftmaxLanes threads per row.
  const auto softmax = [&](std::uint32_t key_start) {
    const std::uint32_t rg = tid / kSoftmaxLanes;
    const std::uint32_t seg = tid % kSoftmaxLanes;
    const std::uint32_t rb = rg / 16;
    const std::uint32_t row = rg % 16;
    const std::uint32_t local_query = query_start + row_block_offset(rb) + row;
    const std::uint32_t absolute_query = start_pos + local_query;
    // col / 16 == 0, so a row's scores live in S tile rb.
    const float4 s0 =
        *reinterpret_cast<const float4*>(&s_lds[0][rb][row][seg * kPerLane]);
    const float4 s1 =
        *reinterpret_cast<const float4*>(&s_lds[1][rb][row][seg * kPerLane]);
    const float s0v[kPerLane] = {s0.x, s0.y, s0.z, s0.w};
    const float s1v[kPerLane] = {s1.x, s1.y, s1.z, s1.w};
    float part_max = -INFINITY;
    float vals[kPerLane];
#pragma unroll
    for (std::uint32_t m = 0; m < kPerLane; ++m) {
      const std::uint32_t col = (seg * kPerLane) + m;
      const std::uint32_t key_position = key_start + col;
      const bool valid = local_query < batch_size &&
                         key_position <= absolute_query &&
                         key_position < context_end;
      vals[m] = valid ? (s0v[m] + s1v[m]) : -INFINITY;
      part_max = fmaxf(part_max, vals[m]);
    }
#pragma unroll
    for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
      part_max = fmaxf(part_max, XorLane(part_max, static_cast<int>(off)));
    }
    const float prev_max = running_max;
    const float next_max = fmaxf(prev_max, part_max);
    const float prior_scale =
        isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
    float part_sum = 0.0F;
#pragma unroll
    for (std::uint32_t m = 0; m < kPerLane; ++m) {
      const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
      part_sum += w;
      p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
    }
#pragma unroll
    for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
      part_sum += XorLane(part_sum, static_cast<int>(off));
    }
    running_max = next_max;
    // The served kernel contracts (running_sum * prior_scale) + part_sum
    // into one v_fmac; spelled out so a scheduling change cannot unfuse it.
    running_sum = __fmaf_rn(running_sum, prior_scale, part_sum);
    if (seg == 0)
      row_scale[(rb * 16) + ((row % 2) * 8) + (row / 2)] = prior_scale;
  };

  // O = O * scale + P V for tile key_start, with the visible-key tail on a
  // diagonal tile. A lane owns rows 2i + half_id of every row block.
  const auto accumulate = [&](std::uint32_t key_start, auto diag_tag) {
    constexpr bool kDiag = decltype(diag_tag)::value;
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      const auto* factors =
          reinterpret_cast<const float4*>(&row_scale[(rb * 16) + (half_id * 8)]);
      const float4 f0 = factors[0];
      const float4 f1 = factors[1];
      const float scale[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
      const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][0]);
      v8f next[2];
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          o_acc[rb][t][i] *= scale[i];
        next[t] = Wmma(p_frag, v_cur[t], o_acc[rb][t]);
      }
      if constexpr (kDiag) {
        // Like the served kernel: a row whose final partial tile has future
        // keys sums only its visible keys, in key order, from the rescaled O.
        const int relative_key =
            static_cast<int>(key_start) -
            static_cast<int>(start_pos + query_start + row_block_offset(rb));
        if (relative_key >= 16) {
          // No row of this block sees a key of the tile.
          next[0] = o_acc[rb][0];
          next[1] = o_acc[rb][1];
        } else if (relative_key + 15 > 0) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const int row = static_cast<int>(2 * i + half_id);
            const int last = row - relative_key;  // last visible key
            union {
              uint4 u[2];
              __half h[16];
            } pr;
            const auto* prow = &p_lds[(rb * 16) + row][0];
            pr.u[0] = *reinterpret_cast<const uint4*>(prow);
            pr.u[1] = *reinterpret_cast<const uint4*>(prow + 8);
            float acc0 = o_acc[rb][0][i];
            float acc1 = o_acc[rb][1][i];
#pragma unroll
            for (std::uint32_t key = 0; key < 16; ++key) {
              const float p = __half2float(pr.h[key]);
              const bool visible = static_cast<int>(key) <= last;
              const float t0 =
                  fmaf(p, static_cast<float>(v_cur[0][key]), acc0);
              const float t1 =
                  fmaf(p, static_cast<float>(v_cur[1][key]), acc1);
              acc0 = visible ? t0 : acc0;
              acc1 = visible ? t1 : acc1;
            }
            if (row < relative_key + 15) {
              next[0][i] = acc0;
              next[1][i] = acc1;
            }
          }
        }
      }
      o_acc[rb][0] = next[0];
      o_acc[rb][1] = next[1];
    }
  };
  // Keep each tile's PV WMMAs ahead of the phase barrier; left alone the
  // compiler may sink them behind the next softmax (flash-next port note).
  const auto pin_o = [&] {
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb)
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t)
        asm volatile("" : "+v"(o_acc[rb][t]));
  };
  using Plain = std::false_type;
  using Diag = std::true_type;

  // Prologue: K(0) staged, S(0), softmax(0) and K(1) staged.
  // K loads and stages run unconditionally (past the last tile they read the
  // clamped last row and stage a buffer nobody reads), so every path into the
  // loop has the same loads in flight and the wait counts stay exact.
  if (n_tiles != 0) {
    load_k(tile_key(0), k_regs);
    load_v(tile_key(0), v_cur);
    stage_k(tile_key(0), k_regs);
    load_k(tile_key(1), k_regs);
  }
  SyncLds();
  if (n_tiles != 0)
    scores();
  SyncLds();
  if (n_tiles != 0) {
    softmax(tile_key(0));
    stage_k(tile_key(1), k_regs);
    load_k(tile_key(2), k_regs);
  }
  SyncLds();

  // Phase B of step j: softmax(j + 1) into p_lds / row_scale (PV(j) read them
  // before the phase A barrier), K(j + 2) into k_lds (S(j + 1) read K(j + 1)
  // before it), and the load of K(j + 3).
  const auto phase_b = [&](std::uint32_t j) {
    softmax(tile_key(j + 1));
    stage_k(tile_key(j + 2), k_regs);
    load_k(tile_key(j + 3), k_regs);
  };

  std::uint32_t j = 0;
  // Every tile that has a successor. Diagonal tiles take the causal tail
  // through one uniform branch: a separate loop for them made the register
  // allocator spill inside the main loop.
  for (; j + 1 < n_tiles; ++j) {
    if (diagonal(tile_key(j)))
      accumulate(tile_key(j), Diag{});
    else
      accumulate(tile_key(j), Plain{});
    load_v(tile_key(j + 1), v_cur);
    scores();
    pin_o();
    SyncLds();
    phase_b(j);
    SyncLds();
  }
  // The last tile: PV only. Nothing reads p_lds, row_scale or k_lds after
  // it, so it needs no barrier before the epilogue.
  if (j < n_tiles) {
    if (diagonal(tile_key(j)))
      accumulate(tile_key(j), Diag{});
    else
      accumulate(tile_key(j), Plain{});
  }

  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  SyncLds();

  // --- epilogue (unchanged) ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr && lse_out == nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
  if (lse_out != nullptr && wave == 0) {
    for (std::uint32_t rg = lane; rg < kRows; rg += 32) {
      const std::uint32_t rb = rg / 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + (rg % 16);
      if (local_query >= batch_size) {
        continue;
      }
      const float denominator = row_sum[rg];
      lse_out[(static_cast<std::size_t>(row_block_head(rb)) * batch_size) +
              local_query] =
          (denominator > 0.0F) ? (row_max[rg] + logf(denominator)) : -INFINITY;
    }
  }
}

// FFN scratch is idle during attention. Keep K head-major and transpose each
// 16-key V tile into the layout consumed by WMMA. Persistent KV is unchanged.
__global__ void PackAttentionHeads(const __half* keys, const __half* values,
                                   __half* packed_keys, __half* packed_values,
                                   std::uint32_t length) {
  __shared__ __half tile[16][kHeadDim + 8];
  // Adjacent blocks read adjacent heads of the same source token tile.
  const std::uint32_t head = blockIdx.x, base = blockIdx.y * 16;
  for (std::uint32_t i = threadIdx.x; i < 16 * kHeadDim / 8; i += 256) {
    const std::uint32_t key = i / (kHeadDim / 8);
    const std::uint32_t dim = (i % (kHeadDim / 8)) * 8;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (base + key < length) {
      const auto src =
          (static_cast<std::size_t>(base + key) * kKvHeads + head) * kHeadDim +
          dim;
      const auto dst =
          (static_cast<std::size_t>(head) * length + base + key) * kHeadDim +
          dim;
      *reinterpret_cast<uint4*>(packed_keys + dst) =
          *reinterpret_cast<const uint4*>(keys + src);
      value = *reinterpret_cast<const uint4*>(values + src);
    }
    *reinterpret_cast<uint4*>(&tile[key][dim]) = value;
  }
  __syncthreads();
  const std::size_t padded = (length + 15u) & ~std::size_t{15};
  for (std::uint32_t i = threadIdx.x; i < 16 * kHeadDim / 8; i += 256) {
    const std::uint32_t dim = i / 2, key = (i % 2) * 8;
    union {
      uint4 packed;
      __half halves[8];
    } value;
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j)
      value.halves[j] = tile[key + j][dim];
    *reinterpret_cast<uint4*>(
        packed_values + static_cast<std::size_t>(head) * padded * kHeadDim +
        static_cast<std::size_t>(base) * kHeadDim + dim * 16 + key) =
        value.packed;
  }
}

bool LaunchQwenWmmaAttention(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream,
    float* lse_out, std::uint32_t key_begin, bool skip_kv_write,
    std::span<float> packed_k_workspace, std::span<float> packed_v_workspace) {
  const detail::AttentionSupportParams support_params{
      .batch_size = batch_size,
      .start_pos = start_pos,
      .max_context = max_context,
      .num_heads = num_heads,
      .num_kv_heads = num_kv_heads,
      .head_dim = head_dim,
      .has_k_cache_f16 = k_cache_f16 != nullptr,
      .has_v_cache_f16 = v_cache_f16 != nullptr,
  };
  if (!detail::IsTiledAttentionSupported(support_params)) {
    return false;
  }

  constexpr std::size_t block_size = 256;
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kKvWidth;
  auto* layer_k = static_cast<__half*>(k_cache_f16) + layer_cache_offset;
  auto* layer_v = static_cast<__half*>(v_cache_f16) + layer_cache_offset;

  if (!skip_kv_write) {
    if (start_pos > 0 && k_cache != nullptr && v_cache != nullptr) {
      const dim3 sync_grid(
          static_cast<unsigned int>((kKvWidth + block_size - 1) / block_size),
          static_cast<unsigned int>(start_pos));
      SyncTiledAttentionKvPrefixKernel<<<sync_grid, block_size, 0, stream>>>(
          k_cache, v_cache, static_cast<__half*>(k_cache_f16),
          static_cast<__half*>(v_cache_f16), layer_idx, start_pos, max_context,
          num_kv_heads, head_dim);
    }
    const dim3 pack_grid(
        static_cast<unsigned int>((kKvWidth + block_size - 1) / block_size),
        static_cast<unsigned int>(batch_size));
    PackTiledAttentionKvKernel<<<pack_grid, block_size, 0, stream>>>(
        k, v, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), k_cache, v_cache, layer_idx,
        start_pos, batch_size, max_context);
  }

  const dim3 grid(static_cast<unsigned int>((batch_size + kWmmaQueryRows - 1) /
                                            kWmmaQueryRows),
                  kKvHeads * (kGqa / kWmmaHeads));
  const auto length = start_pos + static_cast<std::uint32_t>(batch_size);
  const std::size_t key_head_bytes =
      static_cast<std::size_t>(length) * kHeadDim * sizeof(__half);
  const std::size_t value_head_bytes =
      ((static_cast<std::size_t>(length) + 15u) & ~std::size_t{15}) * kHeadDim *
      sizeof(__half);
  const auto heads_per_group = static_cast<std::uint32_t>(std::min(
      {std::size_t{kKvHeads}, packed_k_workspace.size_bytes() / key_head_bytes,
       packed_v_workspace.size_bytes() / value_head_bytes}));
  // attn5: GUFO_DEV_ATTN_V2=1 runs WmmaCausalAttentionV2 on the packed route
  // and takes that route from 256 tokens (the op test checks V2 byte for
  // byte against the packed and the canonical route). Unset: unchanged.
  const char* attn_v2_env = std::getenv("GUFO_DEV_ATTN_V2");
  const bool attn_v2 = attn_v2_env != nullptr && attn_v2_env[0] == '1' &&
                       attn_v2_env[1] == '\0';
  const std::size_t packed_min_batch = attn_v2 ? 256 : 1024;
  // Copying the prefix pays off only for large chunks. Small prefill tails
  // and speculative commit replays retain the canonical cache directly.
  // Reuse the same workspace for bounded groups at depths where all four
  // heads no longer fit; every head keeps its original key order.
  if (batch_size >= packed_min_batch && heads_per_group != 0 &&
      key_begin < length && key_begin % kWmmaKeys == 0) {
    auto* packed_k = reinterpret_cast<__half*>(packed_k_workspace.data());
    auto* packed_v = reinterpret_cast<__half*>(packed_v_workspace.data());
    for (std::uint32_t first = 0; first < kKvHeads; first += heads_per_group) {
      const auto heads = std::min(heads_per_group, kKvHeads - first);
      const std::size_t query_offset =
          static_cast<std::size_t>(first) * kGqa * kHeadDim;
      PackAttentionHeads<<<dim3(heads, (length + 15u) / 16u), block_size, 0,
                           stream>>>(layer_k + first * kHeadDim,
                                     layer_v + first * kHeadDim, packed_k,
                                     packed_v, length);
      const dim3 group_grid(grid.x, heads * (kGqa / kWmmaHeads));
      if (attn_v2) {
        WmmaCausalAttentionV2<kWmmaQueryRows, kWmmaKeys>
            <<<group_grid, block_size, 0, stream>>>(
                q + query_offset, gate ? gate + query_offset : nullptr,
                packed_k, packed_v, out_context + query_offset, start_pos,
                batch_size,
                lse_out ? lse_out + first * kGqa * batch_size : nullptr,
                key_begin);
        continue;
      }
      WmmaCausalAttention<kWmmaQueryRows, kWmmaKeys, true>
          <<<group_grid, block_size, 0, stream>>>(
              q + query_offset, gate ? gate + query_offset : nullptr, packed_k,
              packed_v, out_context + query_offset, start_pos, batch_size,
              lse_out ? lse_out + first * kGqa * batch_size : nullptr,
              key_begin);
    }
    return true;
  }
  WmmaCausalAttention<kWmmaQueryRows, kWmmaKeys>
      <<<grid, block_size, 0, stream>>>(q, gate, layer_k, layer_v, out_context,
                                        start_pos, batch_size, lse_out,
                                        key_begin);
  return true;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

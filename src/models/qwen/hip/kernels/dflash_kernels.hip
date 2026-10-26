#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <string>

#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/dev_sync.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip::kernels {

// The DFlash-2 draft head is the target LM head, which the Q8 target keeps in
// BF16 at 2.54 GB -- half of everything the draft graph reads. Drafting is
// purely weight-bandwidth bound, and the draft only steers which tokens the
// target verifies, so a draft-private Q8_0 copy cannot change an emitted token:
// the verifier still runs the target's BF16 head.
__global__ void dflash_quantize_bf16_to_q8_0_kernel(
    const hip_bfloat16* __restrict__ source,
    Q8_0Block* __restrict__ destination, uint64_t total_blocks) {
  const uint64_t block =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= total_blocks) {
    return;
  }
  const hip_bfloat16* values = source + (block * kQ8_0BlockSize);
  float absolute_max = 0.0F;
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    absolute_max =
        fmaxf(absolute_max, fabsf(static_cast<float>(values[index])));
  }
  const float scale = absolute_max / 127.0F;
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  Q8_0Block& out = destination[block];
  out.d = __float2half(scale);
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    const float quantized = roundf(static_cast<float>(values[index]) * inverse);
    out.qs[index] =
        static_cast<int8_t>(fminf(fmaxf(quantized, -127.0F), 127.0F));
  }
}

void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream) {
  if (bf16_source == nullptr || q8_destination == nullptr ||
      total_elements % kQ8_0BlockSize != 0) {
    return;
  }
  const std::uint64_t total_blocks = total_elements / kQ8_0BlockSize;
  constexpr std::uint32_t kThreads = 256;
  const std::uint64_t grid = (total_blocks + kThreads - 1) / kThreads;
  dflash_quantize_bf16_to_q8_0_kernel<<<static_cast<unsigned int>(grid),
                                        kThreads, 0, stream>>>(
      static_cast<const hip_bfloat16*>(bf16_source),
      static_cast<Q8_0Block*>(q8_destination), total_blocks);
}

__global__ void dflash_grouped_dynamic_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dynamic_coefficients,
    const float* __restrict__ base_kernel, float* __restrict__ output,
    uint32_t num_tokens, uint32_t hidden_size, uint32_t kernel_size,
    uint32_t group_size, uint32_t side) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  if (index >= total) {
    return;
  }

  const uint32_t token = index / hidden_size;
  const uint32_t channel = index % hidden_size;
  const uint32_t groups = hidden_size / group_size;
  const uint32_t group = channel / group_size;
  const uint32_t coefficients_per_token = 2U * kernel_size * groups;

  float sum = 0.0F;
  for (uint32_t tap = 0; tap < kernel_size; ++tap) {
    if (token < tap) {
      continue;
    }
    const uint64_t dynamic_index =
        static_cast<uint64_t>(token) * coefficients_per_token +
        (side * kernel_size + tap) * groups + group;
    const uint64_t base_index =
        static_cast<uint64_t>(side * kernel_size + tap) * hidden_size + channel;
    const uint64_t input_index =
        static_cast<uint64_t>(token - tap) * hidden_size + channel;
    sum += (base_kernel[base_index] + dynamic_coefficients[dynamic_index]) *
           input[input_index];
  }
  output[index] = sum;
}

__global__ void dflash_silu_mul_kernel(float* __restrict__ gate,
                                       const float* __restrict__ up,
                                       uint32_t total_elements) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const float g = gate[idx];
  const float silu_g = g / (1.0F + expf(-g));
  gate[idx] = silu_g * up[idx];
}

// Coalesced QK loads and shared K/V reads for adjacent query heads. Each
// head retains its original dot-product and 256-thread softmax reduction.
//
// `Tokens` folds that many draft rows into one CTA (GUFO_DEV_DRAFTK=1). Every
// arithmetic step is still indexed by one (row, head) pair and runs in the same
// order, so folding rows changes only which CTA owns a chain, never the chain
// itself: the K float4 a lane reads, the five-step xor butterfly, the strided
// 256-thread softmax reduction and the ascending-key PV accumulation are all
// unchanged. What it buys is the KV read. The layer's KV planes are requested
// `block_count * heads_per_kv / (Tokens * QueryHeads)` times; at the shipped
// DFlash2 shape (8 rows, 4 heads per KV head) that is 16x at <2,1> and 8x at
// <2,2>. It also fills the PV phase: `head_dim` is 128 and the CTA is 256
// threads, so <2,1> leaves four of eight waves idle there and <2,2> does not.
//
// The cap on this route is LDS: `scores` is Tokens * QueryHeads * total_k
// floats, and 8 combos at the shipped sliding window of 2048 need 65,792 B
// against a 65,536 B workgroup limit. Past a factor of two the score array has
// to leave LDS -- that is the split route below.
template<uint32_t QueryHeads, uint32_t Tokens>
__global__ void dflash_noncausal_attention_wave_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  constexpr uint32_t kPrefetch = 32;
  constexpr uint32_t kCombos = QueryHeads * Tokens;
  const uint32_t tok = blockIdx.x * Tokens, h = blockIdx.y * QueryHeads;
  if (tok + Tokens > block_count || h + QueryHeads > num_q_heads)
    return;
  const uint32_t kv_h = h / (num_q_heads / num_kv_heads),
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count = history_length > history_start
                                     ? history_length - history_start
                                     : 0U,
                 total_k = history_count + block_count;
  const float4* q_vec[Tokens];
#pragma unroll
  for (unsigned t = 0; t < Tokens; ++t)
    q_vec[t] = reinterpret_cast<const float4*>(
        q + uint64_t(tok + t) * q_dim + uint64_t(h) * head_dim);
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U, waves = blockDim.x >> 5U;
  extern __shared__ float scores[];
  __shared__ float reduce[kCombos][256];
  for (uint32_t local = wave; local < history_count; local += waves) {
    const uint32_t position = history_start + local;
    const float4* key = reinterpret_cast<const float4*>(
        injected_k + uint64_t(position % history_capacity) * kv_dim +
        uint64_t(kv_h) * head_dim);
    float dot[kCombos] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned t = 0; t < Tokens; ++t) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g) {
          const float4 qv = q_vec[t][g * num_vec + d];
          dot[(t * QueryHeads) + g] +=
              (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
        }
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned c = 0; c < kCombos; ++c)
        dot[c] += __shfl_xor(dot[c], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (unsigned t = 0; t < Tokens; ++t) {
        const uint32_t query_position = current_pos + tok + t;
        const bool visible = query_position - position < sliding_window;
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g) {
          const unsigned c = (t * QueryHeads) + g;
          scores[c * total_k + local] = visible ? dot[c] * scale : -1e20F;
        }
      }
    }
  }
  for (uint32_t p = wave; p < block_count; p += waves) {
    const float4* key = reinterpret_cast<const float4*>(
        block_k + uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim);
    float dot[kCombos] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned t = 0; t < Tokens; ++t) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g) {
          const float4 qv = q_vec[t][g * num_vec + d];
          dot[(t * QueryHeads) + g] +=
              (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
        }
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned c = 0; c < kCombos; ++c)
        dot[c] += __shfl_xor(dot[c], offset);
    }
    if (lane == 0) {
      const uint32_t key_position = current_pos + p;
#pragma unroll
      for (unsigned t = 0; t < Tokens; ++t) {
        const uint32_t query_position = current_pos + tok + t;
        const uint32_t distance = query_position > key_position
                                      ? query_position - key_position
                                      : key_position - query_position;
        const bool visible = distance < sliding_window;
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g) {
          const unsigned c = (t * QueryHeads) + g;
          scores[c * total_k + history_count + p] =
              visible ? dot[c] * scale : -1e20F;
        }
      }
    }
  }
  __syncthreads();
  float maxima[kCombos];
#pragma unroll
  for (unsigned c = 0; c < kCombos; ++c)
    maxima[c] = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < kCombos; ++c)
      maxima[c] = fmaxf(maxima[c], scores[c * total_k + p]);
  }
#pragma unroll
  for (unsigned c = 0; c < kCombos; ++c)
    reduce[c][threadIdx.x] = maxima[c];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned c = 0; c < kCombos; ++c)
        reduce[c][threadIdx.x] =
            fmaxf(reduce[c][threadIdx.x], reduce[c][threadIdx.x + stride]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned c = 0; c < kCombos; ++c)
    maxima[c] = reduce[c][0];
  __syncthreads();
  float sums[kCombos] = {};
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < kCombos; ++c) {
      const float e = expf(scores[c * total_k + p] - maxima[c]);
      scores[c * total_k + p] = e;
      sums[c] += e;
    }
  }
#pragma unroll
  for (unsigned c = 0; c < kCombos; ++c)
    reduce[c][threadIdx.x] = sums[c];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned c = 0; c < kCombos; ++c)
        reduce[c][threadIdx.x] += reduce[c][threadIdx.x + stride];
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned c = 0; c < kCombos; ++c)
    sums[c] = 1.f / (reduce[c][0] + 1e-9F);
  __syncthreads();
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < kCombos; ++c)
      scores[c * total_k + p] *= sums[c];
  }
  __syncthreads();
  // One thread owns one (draft row, query head, dim) accumulator and walks the
  // keys ascending, exactly as the single-row route does. At Tokens == 1 the
  // index collapses to the original `d = threadIdx.x; d < head_dim` walk.
  for (uint32_t index = threadIdx.x; index < Tokens * head_dim;
       index += blockDim.x) {
    const uint32_t row = Tokens == 1U ? 0U : index / head_dim;
    const uint32_t d = Tokens == 1U ? index : index % head_dim;
    const uint32_t cbase = row * QueryHeads;
    float acc[QueryHeads] = {};
    uint32_t slot = history_count > 0 ? history_start % history_capacity : 0;
    // Issue independent value loads together, then retain the original
    // ascending-key FMA sequence. Advancing the ring slot avoids a division
    // for every value without changing the persistent KV layout.
    uint32_t local = 0;
    for (; history_count - local >= kPrefetch; local += kPrefetch) {
      float values[kPrefetch];
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
        values[i] =
            injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
        slot = slot + 1 == history_capacity ? 0 : slot + 1;
      }
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          acc[g] += scores[(cbase + g) * total_k + local + i] * values[i];
      }
    }
    for (; local < history_count; ++local) {
      const float value =
          injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[(cbase + g) * total_k + local] * value;
      slot = slot + 1 == history_capacity ? 0 : slot + 1;
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float value =
          block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[(cbase + g) * total_k + history_count + p] * value;
    }
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      out[uint64_t(tok + row) * q_dim + uint64_t(h + g) * head_dim + d] =
          acc[g];
  }
}

// ---------------------------------------------------------------------------
// GUFO_DEV_DRAFTK=2: the split route.
//
// The fused route above re-reads the layer's KV planes once per CTA, and the
// CTA count is pinned by the softmax: one CTA has to own every key of a combo,
// so `CTAs = (block_count / Tokens) * (num_q_heads / QueryHeads)` and the
// requested KV bytes are `CTAs * (K + V) per kv head`. LDS caps Tokens *
// QueryHeads at 7 combos (see above), hence a factor of two.
//
// Splitting the three phases lifts that, because each phase can take the
// decomposition it wants and still keep every float chain in one thread:
//   scores  - split over KEYS (a score is one independent dot) -> K read 1.0x
//   softmax - 256-thread CTA per combo group, the only shape the reduction tree
//             permits, on a 2.1 MB global score array instead of LDS
//   PV      - split over DIM (each (row, head, d) chain is intact) -> V ~1.0x
// Scores live in `scores` laid out [draft row * num_q_heads + head][total_k];
// that row-major-by-combo order is what keeps the softmax and PV reads
// contiguous.
/// `KRegs` = ceil(head_dim / 4 / 32), the K float4s one lane keeps. It is a
/// template parameter and not a runtime count because a runtime-indexed local
/// array is promoted to LDS by AMDGPUPromoteAlloca (16 KB per CTA, measured),
/// and the whole point of this kernel is that the key row sits in registers
/// while every combo reuses it.
template<uint32_t KRegs>
__global__ __launch_bounds__(256) void dflash_draftk_scores_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ block_k, float* __restrict__ scores,
    uint32_t current_pos, uint32_t history_start, uint32_t history_count,
    uint32_t block_count, uint32_t sliding_window, uint32_t num_q_heads,
    uint32_t num_kv_heads, uint32_t head_dim, float scale,
    uint32_t history_capacity, uint32_t keys_per_block) {
  const uint32_t kv_h = blockIdx.y;
  const uint32_t heads_per_kv = num_q_heads / num_kv_heads;
  const uint32_t q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t total_k = history_count + block_count;
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U, waves = blockDim.x >> 5U;
  const uint32_t key_begin = blockIdx.x * keys_per_block;
  const uint32_t key_limit = key_begin + keys_per_block;
  const uint32_t key_end = key_limit < total_k ? key_limit : total_k;

  for (uint32_t key = key_begin + wave; key < key_end; key += waves) {
    const bool in_block = key >= history_count;
    const uint32_t block_index = in_block ? key - history_count : 0U;
    const uint32_t key_position =
        in_block ? current_pos + block_index : history_start + key;
    const float4* source =
        in_block ? reinterpret_cast<const float4*>(
                       block_k + uint64_t(block_index) * kv_dim +
                       uint64_t(kv_h) * head_dim)
                 : reinterpret_cast<const float4*>(
                       injected_k +
                       uint64_t(key_position % history_capacity) * kv_dim +
                       uint64_t(kv_h) * head_dim);
    // One key row per wave, read once and held in registers: every combo below
    // reuses it. This is the whole point of the split.
    float4 held[KRegs];
#pragma unroll
    for (uint32_t i = 0; i < KRegs; ++i) {
      const uint32_t d = lane + (i * 32U);
      if (d < num_vec)
        held[i] = source[d];
    }

    for (uint32_t row = 0; row < block_count; ++row) {
      const uint32_t query_position = current_pos + row;
      bool visible;
      if (in_block) {
        const uint32_t distance = query_position > key_position
                                      ? query_position - key_position
                                      : key_position - query_position;
        visible = distance < sliding_window;
      } else {
        visible = query_position - key_position < sliding_window;
      }
      // Both pointers advance by one head per step, so the per-combo address
      // is an add and not a 64-bit multiply: the ISA charged 20 scalar ops per
      // combo for recomputing `row * q_dim + head * head_dim` and
      // `(row * num_q_heads + head) * total_k`, against 10 for the butterfly.
      const float4* q_vec = reinterpret_cast<const float4*>(
          q + uint64_t(row) * q_dim +
          uint64_t(kv_h * heads_per_kv) * head_dim);
      float* out_row =
          scores +
          (uint64_t(row) * num_q_heads + (kv_h * heads_per_kv)) * total_k + key;
      for (uint32_t r = 0; r < heads_per_kv; ++r) {
        float dot = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < KRegs; ++i) {
          const uint32_t d = lane + (i * 32U);
          if (d >= num_vec)
            break;
          const float4 kv = held[i];
          const float4 qv = q_vec[d];
          dot += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
          dot += __shfl_xor(dot, offset);
        if (lane == 0)
          *out_row = visible ? dot * scale : -1e20F;
        q_vec += num_vec;
        out_row += total_k;
      }
    }
  }
}

/// The exact softmax of the fused route, on the global score array. The
/// per-thread strided partial and the LDS tree below are the reduction order
/// the fused kernel uses, which is why this CTA must stay 256 threads wide.
template<uint32_t Combos>
__global__ __launch_bounds__(256) void dflash_draftk_softmax_kernel(
    float* __restrict__ scores, uint32_t total_k, uint32_t num_q_heads) {
  __shared__ float reduce[Combos][256];
  float* row[Combos];
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    row[c] = scores + (uint64_t(blockIdx.x) * num_q_heads +
                       (blockIdx.y * Combos) + c) *
                          total_k;
  float maxima[Combos];
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    maxima[c] = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < Combos; ++c)
      maxima[c] = fmaxf(maxima[c], row[c][p]);
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    reduce[c][threadIdx.x] = maxima[c];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned c = 0; c < Combos; ++c)
        reduce[c][threadIdx.x] =
            fmaxf(reduce[c][threadIdx.x], reduce[c][threadIdx.x + stride]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    maxima[c] = reduce[c][0];
  __syncthreads();
  float sums[Combos] = {};
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < Combos; ++c) {
      const float e = expf(row[c][p] - maxima[c]);
      row[c][p] = e;
      sums[c] += e;
    }
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    reduce[c][threadIdx.x] = sums[c];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned c = 0; c < Combos; ++c)
        reduce[c][threadIdx.x] += reduce[c][threadIdx.x + stride];
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    sums[c] = 1.f / (reduce[c][0] + 1e-9F);
  __syncthreads();
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < Combos; ++c)
      row[c][p] *= sums[c];
  }
}

/// PV over one KV head. grid = (combo groups, dim tiles, kv heads); a wave owns
/// one dim tile of 32 and `Accumulators` combos, so the V row of a key is read
/// once per CTA as one 128-byte line per wave. `Accumulators` is the knob
/// between V re-reads (combo groups) and CTA count; at the shipped shape
/// (32 combos per KV head, 8 waves) 2 gives 2 groups, 64 CTAs and a 2.0x V
/// request that the sibling CTA turns into an L2 hit, because grid.x (the combo
/// group) varies fastest and the two land together.
template<uint32_t Accumulators>
__global__ __launch_bounds__(256) void dflash_draftk_pv_kernel(
    const float* __restrict__ injected_v, const float* __restrict__ block_v,
    const float* __restrict__ scores, float* __restrict__ out,
    uint32_t history_start, uint32_t history_count, uint32_t block_count,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t history_capacity) {
  constexpr uint32_t kPrefetch = 32;
  constexpr uint32_t kDimTile = 32;
  const uint32_t kv_h = blockIdx.z;
  const uint32_t heads_per_kv = num_q_heads / num_kv_heads;
  const uint32_t q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t total_k = history_count + block_count;
  const uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U,
                 waves = blockDim.x >> 5U;
  const uint32_t d = (blockIdx.y * kDimTile) + lane;
  // Combo = (draft row, query head within this KV head), numbered
  // row * heads_per_kv + r, the order the launcher's divisibility check sizes.
  const uint32_t combos_per_group = waves * Accumulators;
  const uint32_t combo_base = (blockIdx.x * combos_per_group) + wave;

  const float* srow[Accumulators];
#pragma unroll
  for (unsigned j = 0; j < Accumulators; ++j) {
    const uint32_t combo = combo_base + (j * waves);
    const uint32_t row = combo / heads_per_kv;
    const uint32_t head = (kv_h * heads_per_kv) + (combo % heads_per_kv);
    srow[j] = scores + (uint64_t(row) * num_q_heads + head) * total_k;
  }
  float acc[Accumulators] = {};
  // Same ring walk and the same 32-deep prefetch as the fused route, so the FMA
  // order is the FMA order. Measured in the ISA, a 32-key chunk is 64 v_fmac of
  // real work against 193 scalar ops of ring bookkeeping (the per-key
  // `slot + 1 == capacity ? 0 : slot + 1` and the 64-bit `slot * kv_dim`),
  // 32 V loads and 16 b128 score loads. Walking two CONTIGUOUS ring runs
  // instead removes those 193 but costs more than it saves: `local` then starts
  // a run at an arbitrary offset, the compiler can no longer prove the score
  // rows are 16-byte aligned, so the 16 b128 loads become 64 b32 and the
  // 32-deep prefetch collapses (571 instructions per chunk, VGPR 83 -> 53).
  // The fix that keeps both is a chunk grid aligned to `local` with the one
  // straddling chunk on a slow path; worth ~10-60 us per call, not taken here.
  uint32_t slot = history_count > 0 ? history_start % history_capacity : 0;
  const float* vbase = injected_v + (uint64_t(kv_h) * head_dim) + d;
  uint32_t local = 0;
  for (; history_count - local >= kPrefetch; local += kPrefetch) {
    float values[kPrefetch];
#pragma unroll
    for (unsigned i = 0; i < kPrefetch; ++i) {
      values[i] = vbase[uint64_t(slot) * kv_dim];
      slot = slot + 1 == history_capacity ? 0 : slot + 1;
    }
#pragma unroll
    for (unsigned i = 0; i < kPrefetch; ++i) {
#pragma unroll
      for (unsigned j = 0; j < Accumulators; ++j)
        acc[j] += srow[j][local + i] * values[i];
    }
  }
  for (; local < history_count; ++local) {
    const float value = vbase[uint64_t(slot) * kv_dim];
#pragma unroll
    for (unsigned j = 0; j < Accumulators; ++j)
      acc[j] += srow[j][local] * value;
    slot = slot + 1 == history_capacity ? 0 : slot + 1;
  }
  for (uint32_t p = 0; p < block_count; ++p) {
    const float value =
        block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
    for (unsigned j = 0; j < Accumulators; ++j)
      acc[j] += srow[j][history_count + p] * value;
  }
#pragma unroll
  for (unsigned j = 0; j < Accumulators; ++j) {
    const uint32_t combo = combo_base + (j * waves);
    const uint32_t row = combo / heads_per_kv;
    const uint32_t head = (kv_h * heads_per_kv) + (combo % heads_per_kv);
    out[(uint64_t(row) * q_dim) + (uint64_t(head) * head_dim) + d] = acc[j];
  }
}

__global__ void dflash_noncausal_attention_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= block_count || h >= num_q_heads)
    return;

  const uint32_t gqa_group = num_q_heads / num_kv_heads;
  const uint32_t kv_h = h / gqa_group;
  const uint32_t q_dim = num_q_heads * head_dim;
  const uint32_t kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  const uint32_t total_k = history_count + block_count;

  const float* q_ptr = q + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);
  float* out_ptr = out + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);

  extern __shared__ float s_scores[];

  for (uint32_t local = threadIdx.x; local < history_count;
       local += blockDim.x) {
    const uint32_t position = history_start + local;
    const float* k_ptr =
        injected_k + ((uint64_t)(position % history_capacity) * kv_dim +
                      (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    s_scores[local] =
        query_position - position < sliding_window ? dot * scale : -1e20F;
  }

  for (uint32_t p = threadIdx.x; p < block_count; p += blockDim.x) {
    const float* k_ptr =
        block_k + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    const uint32_t key_position = current_pos + p;
    const uint32_t distance = query_position > key_position
                                  ? query_position - key_position
                                  : key_position - query_position;
    s_scores[history_count + p] =
        distance < sliding_window ? dot * scale : -1e20F;
  }
  __syncthreads();

  // Find max score
  __shared__ float s_reduce[256];
  float local_max = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_reduce[threadIdx.x] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] =
          fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_val = s_reduce[0];
  __syncthreads();

  // Exponentiate & sum denominator
  float local_sum = 0.0F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    const float exp_score = expf(s_scores[p] - max_val);
    s_scores[p] = exp_score;
    local_sum += exp_score;
  }
  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0F / (s_reduce[0] + 1e-9F);
  __syncthreads();

  // Normalize scores
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    s_scores[p] *= inv_sum;
  }
  __syncthreads();

  // Multiply by V
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0F;
    for (uint32_t local = 0; local < history_count; ++local) {
      const uint32_t position = history_start + local;
      const float* v_ptr =
          injected_v + ((uint64_t)(position % history_capacity) * kv_dim +
                        (uint64_t)kv_h * head_dim);
      acc += s_scores[local] * v_ptr[d];
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float* v_ptr =
          block_v + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
      acc += s_scores[history_count + p] * v_ptr[d];
    }
    out_ptr[d] = acc;
  }
}

constexpr uint32_t kSelectorThreads = 128;
constexpr uint32_t kMaxSelectorTopK = 16;
constexpr uint32_t kSelectorTokensPerBlock = 2048;
/// The served dflash.selector_rank. GUFO_DEV_WSTRAG instantiates the finalize
/// path with this fixed so the codebook gather unrolls; any other rank keeps
/// the runtime loop.
constexpr uint32_t kSelectorRankFast = 256;
/// Threads that cooperate on one candidate's low-rank transition score.
constexpr uint32_t kSelectorTransitionLanes =
    kSelectorThreads / kMaxSelectorTopK;

// TopK == 0 keeps the runtime-length walk below, statement for statement: that
// is what the default build instantiates, so GUFO_DEV_WSTRAG unset changes
// nothing.
//
// TopK != 0 (GUFO_DEV_WSTRAG=1, TopK == kMaxSelectorTopK == the served
// dflash.selector_top_k) takes the branch-free form instead. WHY: `insertion`
// is a runtime index into the caller's private `local_scores[16]` /
// `local_ids[16]`, and LLVM cannot promote a dynamically indexed alloca, so
// both arrays live in SCRATCH and every candidate pays scratch load/store
// latency -- the measured cost of both selector kernels (0.55 + 0.52 ms/round,
// 1.8-13 GB/s of effective read bandwidth). With TopK a constant every index
// below is a compile-time constant after unrolling, so the arrays stay in
// VGPRs.
//
// Same result, always. The list is sorted descending (invariant: it starts all
// -INFINITY and each insert preserves it), so (score > scores[j]) is monotone
// non-decreasing in j; the serial shift walk therefore stops after exactly
// ahead = #{j in [0, TopK-1) : score > scores[j]} steps and the new pair lands
// at TopK-1-ahead. Ties keep the incumbent, because the walk test is strict
// `>` and so is the count. NaN keeps the original placement (last slot): the
// early-out uses the same `<=` test, and every `>` compare against NaN is
// false, giving ahead == 0.
template <uint32_t TopK>
__device__ __forceinline__ void InsertSelectorCandidate(
    float score, uint32_t id, float* scores, uint32_t* ids,
    [[maybe_unused]] uint32_t selector_top_k) {
  if constexpr (TopK == 0U) {
    if (score <= scores[selector_top_k - 1U]) {
      return;
    }
    uint32_t insertion = selector_top_k - 1U;
    while (insertion > 0U && score > scores[insertion - 1U]) {
      scores[insertion] = scores[insertion - 1U];
      ids[insertion] = ids[insertion - 1U];
      --insertion;
    }
    scores[insertion] = score;
    ids[insertion] = id;
  } else {
    if (score <= scores[TopK - 1U]) {
      return;
    }
    uint32_t ahead = 0U;
#pragma unroll
    for (uint32_t j = 0; j + 1U < TopK; ++j) {
      ahead += score > scores[j] ? 1U : 0U;
    }
    const uint32_t insertion = (TopK - 1U) - ahead;
#pragma unroll
    for (int i = static_cast<int>(TopK) - 1; i >= 0; --i) {
      const float shifted = i > 0 ? scores[i - 1] : 0.0F;
      const uint32_t shifted_id = i > 0 ? ids[i - 1] : 0U;
      const auto slot = static_cast<uint32_t>(i);
      scores[i] =
          slot > insertion ? shifted : (slot == insertion ? score : scores[i]);
      ids[i] =
          slot > insertion ? shifted_id : (slot == insertion ? id : ids[i]);
    }
  }
}

/// Tree-merges the per-thread descending top-k lists held in shared memory into
/// slot zero. The previous implementation had thread zero fold all
/// kSelectorThreads lists serially, which dominated both selector kernels; a
/// pairwise merge does the same work in log2(threads) rounds. Ties keep the
/// lower slot, which is the same preference the serial fold had.
///
/// TopK != 0 only fixes the list length at compile time: the tree shape, the
/// visiting order and the `>=`-takes-mine tie rule are untouched, so the output
/// is identical. It matters because merged_scores/merged_ids are indexed by a
/// runtime loop counter otherwise, which puts them in scratch too.
template <uint32_t TopK>
__device__ void MergeSelectorLists(float* scores, uint32_t* ids,
                                   uint32_t selector_top_k_runtime) {
  const uint32_t selector_top_k =
      TopK != 0U ? TopK : selector_top_k_runtime;
  for (uint32_t stride = kSelectorThreads / 2U; stride > 0U; stride >>= 1U) {
    __syncthreads();
    if (threadIdx.x < stride) {
      float* mine_scores = scores + (threadIdx.x * kMaxSelectorTopK);
      uint32_t* mine_ids = ids + (threadIdx.x * kMaxSelectorTopK);
      const float* other_scores =
          scores + ((threadIdx.x + stride) * kMaxSelectorTopK);
      const uint32_t* other_ids =
          ids + ((threadIdx.x + stride) * kMaxSelectorTopK);
      float merged_scores[kMaxSelectorTopK];
      uint32_t merged_ids[kMaxSelectorTopK];
      uint32_t mine = 0;
      uint32_t other = 0;
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        const bool take_mine =
            other >= selector_top_k ||
            (mine < selector_top_k && mine_scores[mine] >= other_scores[other]);
        if (take_mine) {
          merged_scores[index] = mine_scores[mine];
          merged_ids[index] = mine_ids[mine];
          ++mine;
        } else {
          merged_scores[index] = other_scores[other];
          merged_ids[index] = other_ids[other];
          ++other;
        }
      }
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        mine_scores[index] = merged_scores[index];
        mine_ids[index] = merged_ids[index];
      }
    }
  }
  __syncthreads();
}

template <uint32_t TopK>
__global__ void dflash_selector_partial_topk_kernel(
    const float* __restrict__ logits, float* __restrict__ partial_scores,
    uint32_t* __restrict__ partial_ids, uint32_t vocab_size,
    uint32_t selector_top_k_runtime) {
  const uint32_t selector_top_k =
      TopK != 0U ? TopK : selector_top_k_runtime;
  logits += uint64_t(blockIdx.y) * vocab_size;
  partial_scores += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  partial_ids += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  __shared__ float shared_scores[kSelectorThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * kMaxSelectorTopK];

  float local_scores[kMaxSelectorTopK];
  uint32_t local_ids[kMaxSelectorTopK];
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }

  const uint32_t begin = blockIdx.x * kSelectorTokensPerBlock;
  const uint32_t requested_end = begin + kSelectorTokensPerBlock;
  const uint32_t end = requested_end < vocab_size ? requested_end : vocab_size;
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    InsertSelectorCandidate<TopK>(logits[token], token, local_scores,
                                  local_ids, selector_top_k);
  }

  const uint32_t shared_offset = threadIdx.x * kMaxSelectorTopK;
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }

  MergeSelectorLists<TopK>(shared_scores, shared_ids, selector_top_k);

  const uint32_t output_offset = blockIdx.x * selector_top_k;
  for (uint32_t index = threadIdx.x; index < selector_top_k;
       index += blockDim.x) {
    partial_scores[output_offset + index] = shared_scores[index];
    partial_ids[output_offset + index] = shared_ids[index];
  }
}

// The 1952 -> 16 reduction, exactly as FinalizeSelector does it: the same
// `index = threadIdx.x; index += blockDim.x` partition over the candidates, the
// same per-thread insert, the same pairwise tree. Pulled out so it can run once
// per draft row in parallel instead of once per draft STEP inside the
// single-workgroup finalize. One workgroup per row; row = blockIdx.x.
template <uint32_t TopK>
__global__ void dflash_selector_reduce_kernel(
    const float* __restrict__ partial_scores,
    const uint32_t* __restrict__ partial_ids,
    float* __restrict__ reduced_scores, uint32_t* __restrict__ reduced_ids,
    uint32_t partial_count, uint32_t selector_top_k_runtime) {
  const uint32_t selector_top_k = TopK != 0U ? TopK : selector_top_k_runtime;
  const uint64_t base =
      uint64_t(blockIdx.x) * partial_count * selector_top_k;
  __shared__ float shared_scores[kSelectorThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * kMaxSelectorTopK];

  float local_scores[kMaxSelectorTopK];
  uint32_t local_ids[kMaxSelectorTopK];
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }
  const uint32_t candidate_count = partial_count * selector_top_k;
  for (uint32_t index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    InsertSelectorCandidate<TopK>(partial_scores[base + index],
                                  partial_ids[base + index], local_scores,
                                  local_ids, selector_top_k);
  }
  const uint32_t shared_offset = threadIdx.x * kMaxSelectorTopK;
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }
  MergeSelectorLists<TopK>(shared_scores, shared_ids, selector_top_k);
  const uint64_t out = uint64_t(blockIdx.x) * selector_top_k;
  for (uint32_t index = threadIdx.x; index < selector_top_k;
       index += blockDim.x) {
    reduced_scores[out + index] = shared_scores[index];
    reduced_ids[out + index] = shared_ids[index];
  }
}

// Prereduced == true: `partial_scores`/`partial_ids` already hold this row's
// final top-k (written by dflash_selector_reduce_kernel), so the insert pass
// and the merge tree are skipped and the list is read straight into the slots
// MergeSelectorLists would have left it in. Everything after that point --
// transition scoring, the pick, the sampling -- is untouched.
// Rank != 0 fixes dflash.selector_rank at compile time as well (served value
// 256). It only gives the transition loop a constant trip count, 256/8 = 32,
// so the 32 strided bf16 loads of one candidate's codebook row are issued
// before the first s_waitcnt instead of one load per stalled iteration. That
// gather is 16 scattered 512-B rows of a 127 MB table per draft step, read by
// four waves on one CU with nothing to hide the latency. The `partial +=`
// chain keeps its ascending-rank order, so the sum is unchanged; unrolling
// never reassociates without fast-math, which is off.
template <uint32_t TopK, bool Prereduced = false, uint32_t Rank = 0>
__device__ __forceinline__ void FinalizeSelector(
    const float* __restrict__ partial_scores,
    const uint32_t* __restrict__ partial_ids,
    const float* __restrict__ projected_hidden,
    const hip_bfloat16* __restrict__ predecessor_codebook,
    const hip_bfloat16* __restrict__ successor_codebook,
    const uint32_t* __restrict__ predecessor_token,
    uint32_t* __restrict__ output_token, float* __restrict__ output_confidence,
    float temperature, const float* __restrict__ sample_uniform,
    uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank_runtime, uint32_t selector_top_k_runtime,
    bool greedy_confidence = false,
    const hip_bfloat16* __restrict__ markov_w1 = nullptr,
    const hip_bfloat16* __restrict__ markov_w2 = nullptr,
    uint32_t markov_rank = 0, float markov_scale = 0.0F) {
  const uint32_t selector_top_k =
      TopK != 0U ? TopK : selector_top_k_runtime;
  const uint32_t selector_rank = Rank != 0U ? Rank : selector_rank_runtime;
  __shared__ float shared_scores[kSelectorThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * kMaxSelectorTopK];
  __shared__ float
      shared_transition[kMaxSelectorTopK * kSelectorTransitionLanes];

  if constexpr (Prereduced) {
    for (uint32_t index = threadIdx.x; index < selector_top_k;
         index += blockDim.x) {
      shared_scores[index] = partial_scores[index];
      shared_ids[index] = partial_ids[index];
    }
    __syncthreads();
  } else {
    float local_scores[kMaxSelectorTopK];
    uint32_t local_ids[kMaxSelectorTopK];
#pragma unroll
    for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
      local_scores[index] = -INFINITY;
      local_ids[index] = 0;
    }
    const uint32_t candidate_count = partial_count * selector_top_k;
    for (uint32_t index = threadIdx.x; index < candidate_count;
         index += blockDim.x) {
      InsertSelectorCandidate<TopK>(partial_scores[index], partial_ids[index],
                                    local_scores, local_ids, selector_top_k);
    }

    const uint32_t shared_offset = threadIdx.x * kMaxSelectorTopK;
#pragma unroll
    for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
      shared_scores[shared_offset + index] = local_scores[index];
      shared_ids[shared_offset + index] = local_ids[index];
    }

    MergeSelectorLists<TopK>(shared_scores, shared_ids, selector_top_k);
  }

  const uint32_t predecessor = *predecessor_token;
  const hip_bfloat16* predecessor_row =
      predecessor_codebook + (uint64_t)predecessor * selector_rank;

  // Score the low-rank transitions in parallel: kSelectorTransitionLanes
  // threads share each candidate and reduce through shared memory. Thread zero
  // used to walk all selector_top_k x selector_rank products by itself.
  {
    const uint32_t candidate_index = threadIdx.x / kSelectorTransitionLanes;
    const uint32_t part = threadIdx.x % kSelectorTransitionLanes;
    if (candidate_index < selector_top_k) {
      const uint32_t candidate = shared_ids[candidate_index];
      const hip_bfloat16* successor_row =
          successor_codebook + (uint64_t)candidate * selector_rank;
      float partial = 0.0F;
      if constexpr (Rank != 0U && Rank % kSelectorTransitionLanes == 0U) {
        // Identical rank sequence, identical accumulation order. Only the trip
        // count becomes a literal (Rank / lanes), which `rank < selector_rank`
        // does not give the compiler: `part` is runtime, so it cannot see that
        // the count is the same 32 for every part in [0, lanes). With the
        // literal the loop unrolls and all 32 strided bf16 loads of the
        // candidate's codebook row are in flight before the first wait,
        // instead of one stalled load per iteration on four waves of one CU.
#pragma unroll
        for (uint32_t i = 0; i < Rank / kSelectorTransitionLanes; ++i) {
          const uint32_t rank = part + (i * kSelectorTransitionLanes);
          partial += static_cast<float>(predecessor_row[rank]) *
                     projected_hidden[rank] *
                     static_cast<float>(successor_row[rank]);
        }
      } else {
        for (uint32_t rank = part; rank < selector_rank;
             rank += kSelectorTransitionLanes) {
          partial += static_cast<float>(predecessor_row[rank]) *
                     projected_hidden[rank] *
                     static_cast<float>(successor_row[rank]);
        }
      }
      // Optional borrowed Markov bias (GUFO_DEV_DFLASH_MARKOV),
      // scale * sum_r W2[candidate, r] * W1[predecessor, r], the DSpark bias
      // (speculative::ComputeMarkovBiasRow) restricted to the top-k it
      // reranks. Absent = the old sum, operand for operand.
      if (markov_w1 != nullptr) {
        const hip_bfloat16* w1_row =
            markov_w1 + (uint64_t)predecessor * markov_rank;
        const hip_bfloat16* w2_row =
            markov_w2 + (uint64_t)candidate * markov_rank;
        float markov = 0.0F;
        for (uint32_t rank = part; rank < markov_rank;
             rank += kSelectorTransitionLanes) {
          markov += static_cast<float>(w1_row[rank]) *
                    static_cast<float>(w2_row[rank]);
        }
        partial += markov_scale * markov;
      }
      shared_transition[(candidate_index * kSelectorTransitionLanes) + part] =
          partial;
    }
  }
  __syncthreads();

  if (threadIdx.x != 0U) {
    return;
  }

  float top_scores[kMaxSelectorTopK];
  uint32_t top_ids[kMaxSelectorTopK];
  for (uint32_t index = 0; index < selector_top_k; ++index) {
    top_scores[index] = shared_scores[index];
    top_ids[index] = shared_ids[index];
  }

  float best_score = -INFINITY;
  uint32_t best_index = 0;
  float path_scores[kMaxSelectorTopK];
  for (uint32_t candidate_index = 0; candidate_index < selector_top_k;
       ++candidate_index) {
    float transition = 0.0F;
    for (uint32_t part = 0; part < kSelectorTransitionLanes; ++part) {
      transition +=
          shared_transition[(candidate_index * kSelectorTransitionLanes) +
                            part];
    }
    const float score = top_scores[candidate_index] + transition;
    path_scores[candidate_index] = score;
    if (score > best_score) {
      best_score = score;
      best_index = candidate_index;
    }
  }

  if (candidate_ids != nullptr) {
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      candidate_ids[index] = top_ids[index];
    }
  }

  if (temperature > 0.0F) {
    const float divisor = fmaxf(temperature, FLT_MIN);
    float denominator = 0.0F;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      // Subtract before division so a finite, very small temperature cannot
      // overflow both logits and their maximum to infinity.
      const float probability_weight =
          expf((path_scores[index] - best_score) / divisor);
      path_scores[index] = probability_weight;
      denominator += probability_weight;
    }
    const float inverse_denominator = 1.0F / fmaxf(denominator, 1e-30F);
    float threshold =
        fminf(fmaxf(*sample_uniform, 0.0F), 0.99999994F) * denominator;
    uint32_t sampled_index = selector_top_k - 1U;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      const float probability = path_scores[index] * inverse_denominator;
      if (candidate_probabilities != nullptr) {
        candidate_probabilities[index] = probability;
      }
      threshold -= path_scores[index];
      if (threshold < 0.0F && sampled_index == selector_top_k - 1U) {
        sampled_index = index;
      }
    }
    *output_token = top_ids[sampled_index];
    if (output_confidence != nullptr) {
      *output_confidence = path_scores[sampled_index] * inverse_denominator;
    }
  } else {
    *output_token = top_ids[best_index];
    if (candidate_probabilities != nullptr) {
      for (uint32_t index = 0; index < selector_top_k; ++index)
        candidate_probabilities[index] = index == best_index ? 1.0F : 0.0F;
    }
    if (output_confidence != nullptr) {
      // GUFO_DEV_DRAFT_CONF: the chosen path's softmax weight over the top-k
      // path scores (temperature 1). It ignores the mass outside the top-k, so
      // it reads high; good enough to stop a draft, not a calibrated p1.
      float denominator = 0.0F;
      if (greedy_confidence) {
        for (uint32_t index = 0; index < selector_top_k; ++index)
          denominator += expf(path_scores[index] - best_score);
      }
      *output_confidence =
          greedy_confidence ? 1.0F / fmaxf(denominator, 1.0F) : 1.0F;
    }
  }
}

template <uint32_t TopK, uint32_t Rank = 0>
__global__ void dflash_selector_finalize_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, const uint32_t* predecessor_token,
    uint32_t* output_token, float* output_confidence, float temperature,
    const float* sample_uniform, uint32_t* candidate_ids,
    float* candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  FinalizeSelector<TopK, false, Rank>(
      partial_scores, partial_ids, projected_hidden,
                   predecessor_codebook, successor_codebook, predecessor_token,
                   output_token, output_confidence, temperature, sample_uniform,
                   candidate_ids, candidate_probabilities, partial_count,
                   selector_rank, selector_top_k);
}

struct SelectorBatch {
  DFlashSelectorSequence sequences[8];
  uint32_t row_offsets[8];
  uint32_t token_offsets[8];
};

template <uint32_t TopK, bool Prereduced = false, uint32_t Rank = 0>
__global__ void dflash_selector_finalize_batch_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, uint32_t* tokens,
    float* confidences, const float* uniforms, uint32_t* candidate_ids,
    float* candidate_probabilities, SelectorBatch batch, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k,
    const hip_bfloat16* markov_w1, const hip_bfloat16* markov_w2,
    uint32_t markov_rank, float markov_scale) {
  const auto sequence = batch.sequences[blockIdx.x];
  __shared__ uint32_t predecessor;
  if (threadIdx.x == 0U)
    predecessor = tokens[batch.token_offsets[blockIdx.x]];
  __syncthreads();
  for (uint32_t step = 0; step < sequence.count; ++step) {
    const uint32_t row = batch.row_offsets[blockIdx.x] + step;
    const uint32_t token = batch.token_offsets[blockIdx.x] + step;
    // Prereduced: partial_* points at the reduce kernel's output, one top-k
    // list per row, so the row stride is selector_top_k, not the full partials.
    const uint64_t partial =
        Prereduced ? uint64_t(row) * selector_top_k
                   : uint64_t(row) * partial_count * selector_top_k;
    const uint64_t candidate = uint64_t(row) * selector_top_k;
    FinalizeSelector<TopK, Prereduced, Rank>(
        partial_scores + partial, partial_ids + partial,
        projected_hidden + uint64_t(row) * selector_rank, predecessor_codebook,
        successor_codebook, &predecessor, tokens + token + 1U,
        confidences == nullptr ? nullptr : confidences + row,
        sequence.temperature,
        sequence.temperature > 0.0F ? uniforms + row : nullptr,
        candidate_ids != nullptr ? candidate_ids + candidate : nullptr,
        candidate_probabilities != nullptr ? candidate_probabilities + candidate
                                           : nullptr,
        partial_count, selector_rank, selector_top_k,
        sequence.greedy_confidence != 0U, markov_w1, markov_w2,
        markov_rank, markov_scale);
    // Thread zero publishes the predecessor for this sequence's next step.
    // Other sequences own separate token, probability and partial buffers.
    if (threadIdx.x == 0U)
      predecessor = tokens[token + 1U];
    __syncthreads();
  }
}

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream) {
  // The kernel divides hidden_size by group_size (dflash_kernels.hip:78-79),
  // which is undefined behaviour on the device rather than a clean error. A
  // DSpark draft has no convolutions and must not reach this launcher at all.
  const std::string refusal = speculative::DFlashGroupedConvLaunchRefusal(
      hidden_size, kernel_size, group_size);
  if (!refusal.empty()) {
    throw std::invalid_argument(refusal);
  }
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  const uint32_t blocks = static_cast<uint32_t>((total + 255U) / 256U);
  dflash_grouped_dynamic_conv_kernel<<<blocks, 256, 0, stream>>>(
      input, dynamic_coefficients, base_kernel, output, num_tokens, hidden_size,
      kernel_size, group_size, side);
}

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream) {
  const uint32_t blocks = (total_elements + 255u) / 256u;
  dflash_silu_mul_kernel<<<blocks, 256, 0, stream>>>(gate, up, total_elements);
}

namespace {

/// Shared launch geometry for both non-causal attention routes: one workgroup
/// per (draft token, query head) and one shared score slot per attended key.
struct DFlashAttentionLaunch {
  dim3 grid;
  std::size_t shared_bytes;
};

[[nodiscard]] DFlashAttentionLaunch DFlashAttentionGeometry(
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads) noexcept {
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  return {.grid = dim3(block_count, num_q_heads),
          .shared_bytes = (history_count + block_count) * sizeof(float)};
}

/// Waves per CTA and dim tile the split route is written for. Both are fixed
/// because the softmax reduction order pins blockDim.x at 256 and the PV lane
/// mapping reads one 128-byte V line per wave.
constexpr std::uint32_t kDraftKThreads = 256;
constexpr std::uint32_t kDraftKWaves = kDraftKThreads / 32U;
constexpr std::uint32_t kDraftKDimTile = 32U;
constexpr std::uint32_t kDraftKSoftmaxCombos = 4U;
constexpr std::uint32_t kDraftKPvAccumulators = 2U;
/// `held[]` in the scores kernel: a lane keeps this many K float4s.
constexpr std::uint32_t kDraftKMaxKRegs = 4U;
/// Target CTA count for the key split. 40 CUs, so aim a little above 4x.
constexpr std::uint32_t kDraftKTargetCtas = 160U;

}  // namespace

unsigned DFlashDraftKLevel() noexcept {
  static const unsigned level = [] {
    const char* const env = std::getenv("GUFO_DEV_DRAFTK");
    if (env == nullptr || env[0] < '0' || env[0] > '9')
      return 0U;
    return static_cast<unsigned>(env[0] - '0');
  }();
  return level;
}

std::size_t DFlashDraftKScoreElements(std::uint32_t block_count,
                                      std::uint32_t num_q_heads,
                                      std::uint32_t max_sliding_window) noexcept {
  return static_cast<std::size_t>(block_count) * num_q_heads *
         (static_cast<std::size_t>(max_sliding_window) + block_count);
}

bool LaunchDFlashNonCausalAttentionDraftK(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, unsigned level, float* scores) {
  if (level == 0U || (head_dim % 4U) != 0U || num_kv_heads == 0U ||
      (num_q_heads % num_kv_heads) != 0U)
    return false;
  const std::uint32_t heads_per_kv = num_q_heads / num_kv_heads;
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  const std::uint32_t total_k = history_count + block_count;
  const std::size_t shared_bytes = total_k * sizeof(float);

  // Level 2: scores leave LDS, every phase takes its own decomposition.
  if (level >= 2U && scores != nullptr && (head_dim % kDraftKDimTile) == 0U &&
      (head_dim / 4U) <= 32U * kDraftKMaxKRegs &&
      (num_q_heads % kDraftKSoftmaxCombos) == 0U && total_k != 0U) {
    const std::uint32_t combos = block_count * heads_per_kv;
    const std::uint32_t per_group = kDraftKWaves * kDraftKPvAccumulators;
    if ((combos % per_group) == 0U) {
      std::uint32_t splits = (kDraftKTargetCtas + num_kv_heads - 1U) /
                             num_kv_heads;
      if (splits == 0U)
        splits = 1U;
      std::uint32_t keys_per_block = (total_k + splits - 1U) / splits;
      keys_per_block =
          ((keys_per_block + kDraftKWaves - 1U) / kDraftKWaves) * kDraftKWaves;
      if (keys_per_block == 0U)
        keys_per_block = kDraftKWaves;
      const std::uint32_t score_blocks =
          (total_k + keys_per_block - 1U) / keys_per_block;
      const auto launch_scores = [&](auto regs) {
        dflash_draftk_scores_kernel<decltype(regs)::value>
            <<<dim3(score_blocks, num_kv_heads), kDraftKThreads, 0, stream>>>(
                q, injected_k, block_k, scores, current_pos, history_start,
                history_count, block_count, sliding_window, num_q_heads,
                num_kv_heads, head_dim, scale, history_capacity,
                keys_per_block);
      };
      if ((head_dim / 4U) <= 32U) {
        launch_scores(std::integral_constant<std::uint32_t, 1U>{});
      } else {
        launch_scores(std::integral_constant<std::uint32_t, kDraftKMaxKRegs>{});
      }
      dflash_draftk_softmax_kernel<kDraftKSoftmaxCombos>
          <<<dim3(block_count, num_q_heads / kDraftKSoftmaxCombos),
             kDraftKThreads, 0, stream>>>(scores, total_k, num_q_heads);
      dflash_draftk_pv_kernel<kDraftKPvAccumulators>
          <<<dim3(combos / per_group, head_dim / kDraftKDimTile, num_kv_heads),
             kDraftKThreads, 0, stream>>>(
              injected_v, block_v, scores, out, history_start, history_count,
              block_count, num_q_heads, num_kv_heads, head_dim,
              history_capacity);
      return true;
    }
  }

  // Level 1: two draft rows and two heads in one CTA. Four score rows is the
  // most the 64 KB workgroup LDS holds beside the two reduction arrays.
  if ((block_count % 2U) == 0U && (heads_per_kv % 2U) == 0U &&
      4U * shared_bytes <=
          64U * 1024U - 4U * kDraftKThreads * sizeof(float)) {
    dflash_noncausal_attention_wave_kernel<2, 2>
        <<<dim3(block_count / 2U, num_q_heads / 2U), kDraftKThreads,
           4U * shared_bytes, stream>>>(
            q, injected_k, injected_v, block_k, block_v, out, current_pos,
            history_length, block_count, sliding_window, num_q_heads,
            num_kv_heads, head_dim, scale, history_capacity);
    return true;
  }
  return false;
}

void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  dflash_noncausal_attention_kernel<<<geometry.grid, 256, geometry.shared_bytes,
                                      stream>>>(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, history_capacity);
}

void LaunchDFlashNonCausalAttentionWave(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  // Wider blocks benefit from two query heads per CTA. Keep narrow blocks
  // independent and leave enough LDS for both 256-thread reduction rows.
  const auto heads_per_kv = num_q_heads / num_kv_heads;
  if (block_count >= 4 && (heads_per_kv % 2U) == 0U &&
      geometry.shared_bytes <= (64U * 1024U - 2U * 256U * sizeof(float)) / 2U) {
    auto grid = geometry.grid;
    grid.y /= 2U;
    dflash_noncausal_attention_wave_kernel<2, 1>
        <<<grid, 256, 2U * geometry.shared_bytes, stream>>>(
            q, injected_k, injected_v, block_k, block_v, out, current_pos,
            history_length, block_count, sliding_window, num_q_heads,
            num_kv_heads, head_dim, scale, history_capacity);
    return;
  }
  dflash_noncausal_attention_wave_kernel<1, 1>
      <<<geometry.grid, 256, geometry.shared_bytes, stream>>>(
          q, injected_k, injected_v, block_k, block_v, out, current_pos,
          history_length, block_count, sliding_window, num_q_heads,
          num_kv_heads, head_dim, scale, history_capacity);
}

void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* draftk_scores) {
  // GUFO_DEV_DRAFTK: opt-in bit-exact routes that cut the KV re-read. Unset,
  // nothing below runs and the default route is the one that shipped.
  if (DFlashDraftKLevel() != 0U &&
      LaunchDFlashNonCausalAttentionDraftK(
          q, injected_k, injected_v, block_k, block_v, out, current_pos,
          history_length, block_count, sliding_window, num_q_heads,
          num_kv_heads, head_dim, scale, stream, history_capacity,
          DFlashDraftKLevel(), draftk_scores))
    return;
  // Coalesced float4 reads cover all shipped DFlash2 head dimensions.
  if ((head_dim % 4U) == 0U) {
    LaunchDFlashNonCausalAttentionWave(
        q, injected_k, injected_v, block_k, block_v, out, current_pos,
        history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
        head_dim, scale, stream, history_capacity);
    return;
  }
  LaunchDFlashNonCausalAttentionScalar(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, stream, history_capacity);
}

std::size_t DFlashSelectorScratchElements(std::uint32_t vocab_size) noexcept {
  const std::size_t partial_count =
      (static_cast<std::size_t>(vocab_size) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock;
  // + kMaxSelectorTopK per row: the tail that GUFO_DEV_WSTRAG=2 writes the
  // reduced per-row top-k into. Callers multiply this by the row count, so the
  // tails of all rows sit together after every row's partials
  // (rows*partial_count*top_k + rows*top_k, which is what this sums to).
  // Reserved unconditionally -- 64 B per row -- so no allocation site needs to
  // know about the flag.
  return partial_count * kMaxSelectorTopK + kMaxSelectorTopK;
}

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream) {
  if (partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_top_k == 0 || selector_top_k > kMaxSelectorTopK ||
      selector_top_k > vocab_size) {
    throw std::invalid_argument("invalid DFlash selector shape or scratch");
  }
  const std::uint32_t partial_count =
      (vocab_size + kSelectorTokensPerBlock - 1U) / kSelectorTokensPerBlock;
  if (GufoDevWstrag() && selector_top_k == kMaxSelectorTopK &&
      selector_rank == kSelectorRankFast) {
    dflash_selector_partial_topk_kernel<kMaxSelectorTopK>
        <<<partial_count, kSelectorThreads, 0, stream>>>(
            logits, partial_scores, partial_ids, vocab_size, selector_top_k);
    dflash_selector_finalize_kernel<kMaxSelectorTopK, kSelectorRankFast>
        <<<1, kSelectorThreads, 0, stream>>>(
            partial_scores, partial_ids, projected_hidden,
            static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
            static_cast<const hip_bfloat16*>(successor_codebook_bf16),
            predecessor_token, output_token, output_confidence, temperature,
            sample_uniform, candidate_ids, candidate_probabilities,
            partial_count, selector_rank, selector_top_k);
    return;
  }
  dflash_selector_partial_topk_kernel<0><<<partial_count, kSelectorThreads, 0,
                                           stream>>>(
      logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  dflash_selector_finalize_kernel<0><<<1, kSelectorThreads, 0, stream>>>(
      partial_scores, partial_ids, projected_hidden,
      static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
      static_cast<const hip_bfloat16*>(successor_codebook_bf16),
      predecessor_token, output_token, output_confidence, temperature,
      sample_uniform, candidate_ids, candidate_probabilities, partial_count,
      selector_rank, selector_top_k);
}

void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream, DFlashMarkovRerank markov) {
  if (markov.w1 != nullptr &&
      (markov.w2 == nullptr || markov.rank == 0 ||
       !std::isfinite(markov.scale)))
    throw std::invalid_argument("invalid DFlash Markov rerank operands");
  if (sequences.empty() || sequences.size() > 8 || logits == nullptr ||
      projected_hidden == nullptr || predecessor_codebook_bf16 == nullptr ||
      successor_codebook_bf16 == nullptr || tokens == nullptr ||
      partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_rank == 0 || selector_top_k == 0 ||
      selector_top_k > kMaxSelectorTopK || selector_top_k > vocab_size)
    throw std::invalid_argument("invalid DFlash selector batch or scratch");
  SelectorBatch batch{};
  std::uint32_t rows = 0;
  for (std::size_t index = 0; index < sequences.size(); ++index) {
    const auto sequence = sequences[index];
    if (sequence.count == 0 ||
        sequence.count >
            std::numeric_limits<std::uint32_t>::max() - 8U - rows ||
        !std::isfinite(sequence.temperature) || sequence.temperature < 0.0F ||
        (sequence.temperature > 0.0F && uniforms == nullptr))
      throw std::invalid_argument("invalid DFlash selector sequence");
    batch.sequences[index] = sequence;
    batch.row_offsets[index] = rows;
    batch.token_offsets[index] = rows + static_cast<std::uint32_t>(index);
    rows += sequence.count;
  }
  const auto partial_count = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(vocab_size) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock);
  // GUFO_DEV_WSTRAG>=1: identical selection, TopK fixed at compile time so the
  // per-thread top-k lists stay in registers instead of scratch. Only the
  // served selector_top_k (== kMaxSelectorTopK) is instantiated; any other
  // length keeps the runtime path.
  const unsigned level = (selector_top_k == kMaxSelectorTopK &&
                          selector_rank == kSelectorRankFast)
                             ? GufoDevWstragLevel()
                             : 0U;
  if (level != 0U) {
    dflash_selector_partial_topk_kernel<kMaxSelectorTopK>
        <<<dim3(partial_count, rows), kSelectorThreads, 0, stream>>>(
            logits, partial_scores, partial_ids, vocab_size, selector_top_k);
    if (level >= 2U) {
      // The reduced lists live in the tail that DFlashSelectorScratchElements
      // reserves: rows * partial_count * top_k partial entries first, then
      // rows * top_k reduced entries.
      const std::uint64_t tail =
          std::uint64_t(rows) * partial_count * selector_top_k;
      float* reduced_scores = partial_scores + tail;
      std::uint32_t* reduced_ids = partial_ids + tail;
      dflash_selector_reduce_kernel<kMaxSelectorTopK>
          <<<rows, kSelectorThreads, 0, stream>>>(
              partial_scores, partial_ids, reduced_scores, reduced_ids,
              partial_count, selector_top_k);
      dflash_selector_finalize_batch_kernel<kMaxSelectorTopK, true,
                                            kSelectorRankFast>
          <<<sequences.size(), kSelectorThreads, 0, stream>>>(
              reduced_scores, reduced_ids, projected_hidden,
              static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
              static_cast<const hip_bfloat16*>(successor_codebook_bf16), tokens,
              confidences, uniforms, candidate_ids, candidate_probabilities,
              batch, partial_count, selector_rank, selector_top_k,
              static_cast<const hip_bfloat16*>(markov.w1),
              static_cast<const hip_bfloat16*>(markov.w2), markov.rank,
              markov.scale);
      return;
    }
    dflash_selector_finalize_batch_kernel<kMaxSelectorTopK, false,
                                          kSelectorRankFast>
        <<<sequences.size(), kSelectorThreads, 0, stream>>>(
            partial_scores, partial_ids, projected_hidden,
            static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
            static_cast<const hip_bfloat16*>(successor_codebook_bf16), tokens,
            confidences, uniforms, candidate_ids, candidate_probabilities,
            batch, partial_count, selector_rank, selector_top_k,
            static_cast<const hip_bfloat16*>(markov.w1),
            static_cast<const hip_bfloat16*>(markov.w2), markov.rank,
            markov.scale);
    return;
  }
  dflash_selector_partial_topk_kernel<0><<<dim3(partial_count, rows),
                                           kSelectorThreads, 0, stream>>>(
      logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  dflash_selector_finalize_batch_kernel<0><<<sequences.size(),
                                             kSelectorThreads, 0, stream>>>(
      partial_scores, partial_ids, projected_hidden,
      static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
      static_cast<const hip_bfloat16*>(successor_codebook_bf16), tokens,
      confidences, uniforms, candidate_ids, candidate_probabilities, batch,
      partial_count, selector_rank, selector_top_k,
      static_cast<const hip_bfloat16*>(markov.w1),
      static_cast<const hip_bfloat16*>(markov.w2), markov.rank, markov.scale);
}

namespace {

// GUFO_DEV_DRAFT_VOCAB: row r of the cut head's output lands at its real token
// ids. Every other logit was set to -inf once at allocation and is never
// written again, so no per-round fill is needed.
__global__ void dflash_scatter_draft_logits_kernel(
    const float* __restrict__ cut_logits, const uint32_t* __restrict__ ids,
    float* __restrict__ logits, uint32_t cut_size, uint32_t vocab_size) {
  const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= cut_size) return;
  const uint64_t row = blockIdx.y;
  logits[row * vocab_size + ids[column]] = cut_logits[row * cut_size + column];
}

}  // namespace

void LaunchDFlashScatterDraftLogits(const float* cut_logits,
                                    const std::uint32_t* ids, float* logits,
                                    std::uint32_t rows, std::uint32_t cut_size,
                                    std::uint32_t vocab_size,
                                    hipStream_t stream) {
  if (rows == 0 || cut_size == 0) return;
  constexpr std::uint32_t kThreads = 256;
  dflash_scatter_draft_logits_kernel<<<dim3((cut_size + kThreads - 1U) / kThreads,
                                            rows),
                                       kThreads, 0, stream>>>(
      cut_logits, ids, logits, cut_size, vocab_size);
}

namespace {

// One DSpark block position: add the Markov head to the base logits, then
// choose the emitted token. Two vocabulary passes, in this order.
//
// 1. logits[c] += sum_r W2[c, r] * W1[previous, r] (fused here).
//    speculative::ComputeMarkovBiasRow is the arithmetic specification and
//    CPU oracle for this kernel. Also write w1_prev[r] = W1[previous, r] for
//    the confidence head.
// 2. Argmax chooses the token chained by the reference. At temperature > 0,
//    an inverse-CDF draw chooses the emitted token; the reference chains on
//    argmax at every temperature (src/models/dflash.cpp:381-384).
constexpr uint32_t kMarkovThreads = 256;
constexpr uint32_t kMarkovWaveSize = 32;  // gfx1151 uses wave32.

__global__ void dflash_markov_bias_kernel(
    const hip_bfloat16* __restrict__ markov_w1,
    const hip_bfloat16* __restrict__ markov_w2,
    const uint32_t* __restrict__ previous_token, float* __restrict__ logits,
    float* __restrict__ w1_prev, uint32_t markov_rank, uint32_t vocab_size) {
  extern __shared__ float w1_shared[];
  const uint32_t previous = *previous_token;
  for (uint32_t component = threadIdx.x; component < markov_rank;
       component += blockDim.x) {
    const float value = static_cast<float>(
        markov_w1[static_cast<uint64_t>(previous) * markov_rank + component]);
    w1_shared[component] = value;
    if (blockIdx.x == 0) {
      w1_prev[component] = value;
    }
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x % kMarkovWaveSize;
  const uint32_t row = blockIdx.x * (blockDim.x / kMarkovWaveSize) +
                       threadIdx.x / kMarkovWaveSize;
  if (row >= vocab_size) {
    return;
  }
  const hip_bfloat16* const w2_row =
      markov_w2 + static_cast<uint64_t>(row) * markov_rank;
  float sum = 0.0F;
  for (uint32_t component = lane; component < markov_rank;
       component += kMarkovWaveSize) {
    sum += static_cast<float>(w2_row[component]) * w1_shared[component];
  }
  for (uint32_t offset = kMarkovWaveSize / 2U; offset > 0U; offset >>= 1U) {
    sum += __shfl_xor(sum, offset);
  }
  if (lane == 0U) {
    logits[row] += sum;
  }
}

// One workgroup reduces the whole vocabulary. The draw is taken by thread zero
// walking the cumulative distribution, the same shape as the DFlash-2 selector
// walking its top-k path scores.
__global__ void dflash_markov_pick_kernel(
    const float* __restrict__ logits, float* __restrict__ confidence,
    uint32_t* __restrict__ token, const float* __restrict__ sample_uniform,
    float temperature, uint32_t vocab_size) {
  __shared__ float reduce[kMarkovThreads];
  __shared__ uint32_t reduce_id[kMarkovThreads];
  const uint32_t tid = threadIdx.x;

  float best = -INFINITY;
  uint32_t best_id = 0;
  for (uint32_t index = tid; index < vocab_size; index += kMarkovThreads) {
    const float value = logits[index];
    if (value > best) {
      best = value;
      best_id = index;
    }
  }
  reduce[tid] = best;
  reduce_id[tid] = best_id;
  __syncthreads();
  for (uint32_t stride = kMarkovThreads / 2U; stride > 0U; stride >>= 1U) {
    if (tid < stride && reduce[tid + stride] > reduce[tid]) {
      reduce[tid] = reduce[tid + stride];
      reduce_id[tid] = reduce_id[tid + stride];
    }
    __syncthreads();
  }
  const float maximum = reduce[0];
  const uint32_t maximum_id = reduce_id[0];

  // The divisor is floored so a finite but tiny temperature cannot overflow
  // both the logits and their maximum to infinity.
  const float divisor = fmaxf(temperature, FLT_MIN);
  float total = 0.0F;
  if (temperature > 0.0F) {
    for (uint32_t index = tid; index < vocab_size; index += kMarkovThreads) {
      total += expf((logits[index] - maximum) / divisor);
    }
    reduce[tid] = total;
    __syncthreads();
    for (uint32_t stride = kMarkovThreads / 2U; stride > 0U; stride >>= 1U) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      __syncthreads();
    }
  }

  if (tid != 0U) {
    return;
  }
  if (temperature <= 0.0F) {
    *token = maximum_id;
    if (confidence != nullptr) {
      *confidence = 1.0F;
    }
    return;
  }
  const float denominator = reduce[0];
  const float threshold = fminf(fmaxf(*sample_uniform, 0.0F), 0.99999994F) *
                          denominator;
  float cumulative = 0.0F;
  uint32_t drawn = maximum_id;
  for (uint32_t index = 0; index < vocab_size; ++index) {
    cumulative += expf((logits[index] - maximum) / divisor);
    if (cumulative > threshold) {
      drawn = index;
      break;
    }
  }
  *token = drawn;
  if (confidence != nullptr) {
    *confidence = expf((logits[drawn] - maximum) / divisor) /
                  fmaxf(denominator, 1e-30F);
  }
}

// sigmoid(conf_proj . [hidden ; w1_prev] + b) over hidden_size + markov_rank.
// The feature vector is the concatenation in that order, so the tail of the
// projection multiplies markov_w1[previous] (src/models/dflash.cpp:371-372).
__global__ void dflash_markov_confidence_kernel(
    const float* __restrict__ confidence_projection,
    const float* __restrict__ confidence_bias,
    const float* __restrict__ hidden, const float* __restrict__ w1_prev,
    float* __restrict__ confidence, uint32_t hidden_size, uint32_t markov_rank) {
  __shared__ float reduce[kMarkovThreads];
  const uint32_t width = hidden_size + markov_rank;
  float sum = 0.0F;
  for (uint32_t index = threadIdx.x; index < width; index += blockDim.x) {
    const float value =
        index < hidden_size ? hidden[index] : w1_prev[index - hidden_size];
    sum += confidence_projection[index] * value;
  }
  reduce[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    const float biased = reduce[0] + confidence_bias[0];
    *confidence = 1.0F / (1.0F + expf(-biased));
  }
}

}  // namespace

void LaunchDFlashMarkovStep(
    const void* markov_w1, const void* markov_w2,
    const std::uint32_t* previous_token, float* bias, float* w1_prev,
    float* logits, const float* hidden, const void* confidence_projection,
    const void* confidence_bias, float* confidence, std::uint32_t* token,
    std::uint32_t vocab_size, std::uint32_t markov_rank,
    std::uint32_t hidden_size, float temperature,
    const float* sample_uniform, hipStream_t stream) {
  if (markov_w1 == nullptr || markov_w2 == nullptr ||
      previous_token == nullptr || bias == nullptr || w1_prev == nullptr ||
      logits == nullptr || hidden == nullptr ||
      confidence_projection == nullptr || confidence_bias == nullptr ||
      confidence == nullptr || token == nullptr || markov_rank == 0 ||
      vocab_size == 0) {
    return;
  }
  constexpr std::uint32_t rows_per_block =
      kMarkovThreads / kMarkovWaveSize;
  const std::uint32_t bias_grids =
      (vocab_size + rows_per_block - 1U) / rows_per_block;
  const std::size_t shared_bytes =
      static_cast<std::size_t>(markov_rank) * sizeof(float);
  dflash_markov_bias_kernel<<<bias_grids, kMarkovThreads, shared_bytes, stream>>>(
      static_cast<const hip_bfloat16*>(markov_w1),
      static_cast<const hip_bfloat16*>(markov_w2), previous_token, logits,
      w1_prev, markov_rank, vocab_size);
  dflash_markov_pick_kernel<<<1, kMarkovThreads, 0, stream>>>(
      logits, confidence, token, sample_uniform, temperature, vocab_size);
  dflash_markov_confidence_kernel<<<1, kMarkovThreads, 0, stream>>>(
      static_cast<const float*>(confidence_projection),
      static_cast<const float*>(confidence_bias), hidden, w1_prev, confidence,
      hidden_size, markov_rank);
}

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)

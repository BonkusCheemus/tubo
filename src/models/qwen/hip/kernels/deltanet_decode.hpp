#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_DELTANET_DECODE_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_DELTANET_DECODE_HPP_

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/models/qwen/hip/detail/recurrent_state.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::hip {

// Value split: Split > 1 spreads a head's 128 value rows over Split
// one-wave blocks (blockIdx.z = part), so 48 heads fill 192 blocks instead of
// 48 on 40 CUs. Row j of the state only ever reads row j, so every state and
// raw-output value keeps its exact arithmetic. The one cross-row step, the
// per-head output RMSNorm, is done by the last block of the head to finish
// (counter below), from the same per-wave partials summed in the same order.
inline constexpr std::uint32_t kDeltaNetSplitMaxSlots = 8 * 64;  // sequences x heads
inline constexpr std::uint32_t kDeltaNetSplitMaxRows = 16;
// Zeroed at module load; the last block of each head resets its counter.
// Keyed by (sequence, head) only: two split launches must never overlap
// (today every recurrence runs on the arena stream, in order).
static __device__ std::uint32_t g_deltanet_split_arrivals[kDeltaNetSplitMaxSlots];
static __device__ float
    g_deltanet_split_sumsq[kDeltaNetSplitMaxSlots * kDeltaNetSplitMaxRows * 4];

// The state tile is loaded once before the row loop and stored once after it.
// A `Diverge` pass loads from `deltanet_in` (the buffer pinned for rollback)
// and stores to `deltanet_state`; in place, `deltanet_in` is unused and the
// single restrict pointer keeps today's aliasing contract and code. The generic
// path keeps per-row storage transitions, so only its first row reads the
// pinned tile. Per-sequence launches take both pointers from the sequence and
// may alias per block.
template<detail::QwenRecurrentStateElement StateT, bool Resident = false,
         bool WriteOutput = true, bool Diverge = false,
         typename Sequences = std::nullptr_t, std::uint32_t Split = 1>
__launch_bounds__(Resident ? 128 : 1024, 1) __global__
    void DeltaNetRecurrenceKernel(
        const float* __restrict__ conv_out, StateT* __restrict__ deltanet_state,
        const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
        const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
        const float* __restrict__ ssm_norm, const float* __restrict__ gate,
        float* __restrict__ out_buf, std::uint32_t layer_idx,
        std::uint32_t num_key_heads, std::uint32_t num_heads,
        std::uint32_t key_dimension, std::uint32_t value_dimension,
        SsmReplayCapture replay_capture, std::uint32_t rows,
        std::size_t conv_row_stride, std::size_t projection_row_stride,
        std::size_t inner_row_stride,
        const StateT* __restrict__ deltanet_in = nullptr,
        Sequences sequences = {}) {
  const StateT* state_in = Diverge ? deltanet_in : deltanet_state;
  StateT* state_out = deltanet_state;
  if constexpr (!std::is_same_v<Sequences, std::nullptr_t>) {
    static_assert(Diverge, "per-sequence launches decide aliasing per block");
    const auto& sequence = sequences.sequences[blockIdx.y];
    state_in = static_cast<const StateT*>(sequence.state.recurrent_in);
    state_out = static_cast<StateT*>(sequence.state.recurrent_out);
    replay_capture = sequence.replay;
    rows = sequence.rows;
    const std::size_t offset = sequence.row_offset;
    conv_out += offset * conv_row_stride;
    if (alpha_buf)
      alpha_buf += offset * projection_row_stride;
    if (beta_buf)
      beta_buf += offset * projection_row_stride;
    if (gate)
      gate += offset * inner_row_stride;
    if (out_buf)
      out_buf += offset * inner_row_stride;
  }
  static_assert(!Resident || std::is_same_v<StateT, float>);
  static_assert(Split == 1 || (Resident && Split == 4));
  const std::uint32_t key_dim = Resident ? 128 : key_dimension;
  const std::uint32_t val_dim = Resident ? 128 : value_dimension;
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;
  // First value row of this block; 0 without the split.
  const std::size_t j0 = (Split > 1) ? blockIdx.z * (128 / Split) : 0;

  // opt-c191-ssm-rows: block `h` owns head h's recurrent state exclusively, so
  // walking `rows` consecutive verification rows here reproduces exactly the
  // sequence of updates that `rows` separate launches produced.
  __shared__ float s_k[128];
  __shared__ float s_q[128];
  __shared__ float s_norm[2];
  __shared__ float s_warp_k[8];
  __shared__ float s_warp_q[8];
  __shared__ float s_val_warp[8];

  const std::size_t head_offset =
      (static_cast<std::size_t>(layer_idx) * num_heads + h) * key_dim * val_dim;
  const StateT* s_matrix_in = state_in + head_offset;
  StateT* s_matrix = state_out + head_offset;

  // Keep each FP32 state row across the verification/replay block. Generic
  // shapes and BF16 storage retain their per-token storage transitions.
  float4 resident[Resident ? 32 : 1];
  if constexpr (Resident) {
#pragma unroll
    for (std::size_t vi = 0; vi < 32; ++vi)
      resident[vi] =
          detail::LoadRecurrentState4(s_matrix_in + (j0 + tid) * 128, vi);
  }

  for (std::uint32_t row = 0; row < rows; ++row) {
    const float* q_ptr =
        conv_out + (static_cast<std::size_t>(row) * conv_row_stride);
    const float* k_ptr = q_ptr + (num_key_heads * key_dim);
    const float* v_ptr = q_ptr + (num_key_heads * key_dim * 2);

    const std::uint32_t kh_idx = h % num_key_heads;
    const float* q_h = q_ptr + (kh_idx * key_dim);
    const float* k_h = k_ptr + (kh_idx * key_dim);
    const float* v_h = v_ptr + (h * val_dim);
    float* o_h = nullptr;
    if constexpr (WriteOutput) {
      o_h = out_buf + (static_cast<std::size_t>(row) * inner_row_stride) +
            (h * val_dim);
    }
    const float* row_alpha = (alpha_buf != nullptr)
                                 ? alpha_buf + (static_cast<std::size_t>(row) *
                                                projection_row_stride)
                                 : nullptr;
    const float* row_beta =
        (beta_buf != nullptr)
            ? beta_buf + (static_cast<std::size_t>(row) * projection_row_stride)
            : nullptr;
    const float* row_gate =
        (gate != nullptr)
            ? gate + (static_cast<std::size_t>(row) * inner_row_stride)
            : nullptr;

    // 1. Decay & learning rate: alpha = exp(ssm_a * softplus(alpha + dt)), beta
    // = sigmoid(beta)
    const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
    const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;
    const float alpha_in = (row_alpha != nullptr) ? row_alpha[h] : 0.0F;
    const float beta_in = (row_beta != nullptr) ? row_beta[h] : 0.0F;

    if (tid == 0 && j0 == 0 && replay_capture.alpha != nullptr &&
        replay_capture.beta != nullptr && replay_capture.position != nullptr &&
        replay_capture.enabled != nullptr && replay_capture.enabled[0] != 0U) {
      const std::size_t slot =
          static_cast<std::size_t>(replay_capture.position[row]) %
          kSsmReplayCapacity;
      const std::size_t offset =
          ((static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity + slot) *
           num_heads) +
          h;
      replay_capture.alpha[offset] = alpha_in;
      replay_capture.beta[offset] = beta_in;
    }

    const float alpha_biased = alpha_in + dt;
    const float alpha_softplus =
        (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
    const float alpha_h = expf(alpha_softplus * a_val);
    const float beta_h = 1.0F / (1.0F + expf(-beta_in));

    // 2. Normalize key ||k_h||_2 and query ||q_h||_2 in parallel
    const std::size_t lane_id = tid & 31u;
    const std::size_t warp_id = tid >> 5u;

    if constexpr (Split > 1) {
      // No FMA contraction here: the unsplit block rounds each square before
      // its first tree add, and unrolled the compiler would fuse the two.
#pragma clang fp contract(off)
      // One wave does the four 32-lane trees the four waves of the unsplit
      // block do, takes lane 0's sum of each (what they store), and adds the
      // four in the same order: identical norms, no cross-wave barrier.
      float k_sq = 0.0F, q_sq = 0.0F;
#pragma unroll
      for (std::size_t w = 0; w < 4; ++w) {
        const std::size_t i = w * 32 + tid;
        const float kv = k_h[i];
        const float qv = q_h[i];
        s_k[i] = kv;
        if constexpr (WriteOutput)
          s_q[i] = qv;
        float part_k = kv * kv;
        float part_q = WriteOutput ? qv * qv : 0.0F;
        for (int off = 16; off > 0; off >>= 1) {
          part_k += __shfl_xor(part_k, off);
          if constexpr (WriteOutput)
            part_q += __shfl_xor(part_q, off);
        }
        k_sq += __shfl(part_k, 0);
        if constexpr (WriteOutput)
          q_sq += __shfl(part_q, 0);
      }
      const float inv_k_norm = 1.0F / sqrtf(k_sq + 1e-6F);
      const float inv_q_norm =
          WriteOutput ? 1.0F / sqrtf(q_sq + 1e-6F) : 0.0F;
      const float q_scale =
          WriteOutput ? (1.0F / sqrtf(static_cast<float>(key_dim))) * inv_q_norm
                      : 0.0F;
#pragma unroll
      for (std::size_t w = 0; w < 4; ++w) {
        s_k[w * 32 + tid] *= inv_k_norm;
        if constexpr (WriteOutput)
          s_q[w * 32 + tid] *= q_scale;
      }
      __syncthreads();
    } else {
    float local_k_sq = 0.0F;
    float local_q_sq = 0.0F;
    if (tid < key_dim) {
      const float kv = k_h[tid];
      const float qv = q_h[tid];
      s_k[tid] = kv;
      if constexpr (WriteOutput)
        s_q[tid] = qv;
      local_k_sq = kv * kv;
      if constexpr (WriteOutput)
        local_q_sq = qv * qv;
    }
    for (int off = 16; off > 0; off >>= 1) {
      local_k_sq += __shfl_xor(local_k_sq, off);
      if constexpr (WriteOutput)
        local_q_sq += __shfl_xor(local_q_sq, off);
    }
    if (lane_id == 0) {
      s_warp_k[warp_id] = local_k_sq;
      if constexpr (WriteOutput)
        s_warp_q[warp_id] = local_q_sq;
    }
    __syncthreads();

    float inv_k_norm = 0.0F;
    float inv_q_norm = 0.0F;
    // Resident waves repeat the same ordered four-term sum. This removes the
    // broadcast barrier without changing the normalization arithmetic.
    if (Resident || tid == 0) {
      float k_sq = 0.0F, q_sq = 0.0F;
      const std::size_t num_warps = Resident ? 4 : blockDim.x >> 5u;
      for (std::size_t w = 0; w < num_warps; ++w) {
        k_sq += s_warp_k[w];
        if constexpr (WriteOutput)
          q_sq += s_warp_q[w];
      }
      inv_k_norm = 1.0F / sqrtf(k_sq + 1e-6F);
      if constexpr (WriteOutput)
        inv_q_norm = 1.0F / sqrtf(q_sq + 1e-6F);
      if constexpr (!Resident) {
        s_norm[0] = inv_k_norm;
        if constexpr (WriteOutput)
          s_norm[1] = inv_q_norm;
      }
    }
    if constexpr (!Resident) {
      __syncthreads();
      inv_k_norm = s_norm[0];
      if constexpr (WriteOutput)
        inv_q_norm = s_norm[1];
    }

    const float q_scale =
        WriteOutput ? (1.0F / sqrtf(static_cast<float>(key_dim))) * inv_q_norm
                    : 0.0F;
    if (tid < key_dim) {
      s_k[tid] *= inv_k_norm;
      if constexpr (WriteOutput)
        s_q[tid] *= q_scale;
    }
    __syncthreads();
    }  // Split == 1

    // 3. Associative retrieval & state update with vectorized float4
    if (tid < val_dim) {
      const std::size_t j = j0 + tid;
      StateT* s_row = s_matrix + (j * key_dim);
      // Only the first row of a diverging pass reads the pinned tile; later
      // rows read what this launch stored.
      const StateT* s_row_in =
          (Diverge && row == 0) ? s_matrix_in + (j * key_dim) : s_row;
      const float4* k4 = reinterpret_cast<const float4*>(s_k);
      const float4* q4 = reinterpret_cast<const float4*>(s_q);

      const std::size_t key_vec = Resident ? 32 : key_dim / 4;

      float u_j = 0.0F;
#pragma unroll
      for (std::size_t vi = 0; vi < key_vec; ++vi) {
        float4 st;
        if constexpr (Resident)
          st = resident[vi];
        else
          st = detail::LoadRecurrentState4(s_row_in, vi);
        const float4 kv = k4[vi];
        u_j += (st.x * alpha_h) * kv.x + (st.y * alpha_h) * kv.y +
               (st.z * alpha_h) * kv.z + (st.w * alpha_h) * kv.w;
        if constexpr (Resident) {
          // Bound shared-memory load hoisting so the 128 state values remain
          // in registers. This compiler barrier emits no GPU instruction.
          if (vi % 4 == 3)
            asm volatile("" ::: "memory");
        }
      }
      const float d_j = (v_h[j] - u_j) * beta_h;

      float o_j = 0.0F;
#pragma unroll
      for (std::size_t vi = 0; vi < key_vec; ++vi) {
        float4 st;
        if constexpr (Resident)
          st = resident[vi];
        else
          st = detail::LoadRecurrentState4(s_row_in, vi);
        const float4 kv = k4[vi];
        const float4 qv = WriteOutput ? q4[vi] : float4{};
        if constexpr (Resident) {
          // Match decode's rounding: delta*key rounds before the state-decay
          // FMA. Reusing the rounded state*alpha here changes the recurrence.
          st.x = fmaf(st.x, alpha_h, d_j * kv.x);
          st.y = fmaf(st.y, alpha_h, d_j * kv.y);
          st.z = fmaf(st.z, alpha_h, d_j * kv.z);
          st.w = fmaf(st.w, alpha_h, d_j * kv.w);
        } else {
          st.x = (st.x * alpha_h) + (d_j * kv.x);
          st.y = (st.y * alpha_h) + (d_j * kv.y);
          st.z = (st.z * alpha_h) + (d_j * kv.z);
          st.w = (st.w * alpha_h) + (d_j * kv.w);
        }
        if constexpr (Resident)
          resident[vi] = st;
        else
          detail::StoreRecurrentState4(s_row, vi, st);
        if constexpr (WriteOutput)
          o_j += st.x * qv.x + st.y * qv.y + st.z * qv.z + st.w * qv.w;
        if constexpr (Resident) {
          // Bound shared-memory load hoisting so the 128 state values remain
          // in registers. This compiler barrier emits no GPU instruction.
          if (vi % 4 == 3)
            asm volatile("" ::: "memory");
        }
      }
      if constexpr (WriteOutput)
        o_h[j] = o_j;
    }
    __syncthreads();

    // Replay only consumes recurrent state. The barrier above prevents the
    // next row from overwriting shared keys while another wave uses them.
    if constexpr (!WriteOutput)
      continue;

    if constexpr (Split > 1) {
#pragma clang fp contract(off)
      // This wave's sum of squares is the unsplit block's warp-(part) partial.
      // Park it; the head's last block normalizes every row after the loop.
      float val_sq = o_h[j0 + tid] * o_h[j0 + tid];
      for (int off = 16; off > 0; off >>= 1)
        val_sq += __shfl_xor(val_sq, off);
      if (tid == 0) {
        const std::size_t slot =
            static_cast<std::size_t>(blockIdx.y) * num_heads + h;
        g_deltanet_split_sumsq[(slot * kDeltaNetSplitMaxRows + row) * 4 +
                               blockIdx.z] = val_sq;
      }
      continue;
    }

    // 4. Per-head RMSNorm on o_h with warp reductions
    float val_sq = (tid < val_dim) ? (o_h[tid] * o_h[tid]) : 0.0F;
    for (int off = 16; off > 0; off >>= 1) {
      val_sq += __shfl_xor(val_sq, off);
    }
    if (lane_id == 0) {
      s_val_warp[warp_id] = val_sq;
    }
    __syncthreads();

    float rms = 0.0F;
    if (Resident || tid == 0) {
      float total_sq = 0.0F;
      const std::size_t num_warps = Resident ? 4 : blockDim.x >> 5u;
      for (std::size_t w = 0; w < num_warps; ++w) {
        total_sq += s_val_warp[w];
      }
      const float mean_sq = total_sq / static_cast<float>(val_dim);
      rms = rsqrtf(mean_sq + 1e-6F);
      if constexpr (!Resident)
        s_norm[0] = rms;
    }
    if constexpr (!Resident) {
      __syncthreads();
      rms = s_norm[0];
    }

    if (tid < val_dim) {
      const float w = (ssm_norm != nullptr) ? ssm_norm[tid] : 1.0F;
      float val = o_h[tid] * rms * w;
      if (row_gate != nullptr) {
        const float g = row_gate[h * val_dim + tid];
        const float sig = 1.0F / (1.0F + expf(-g));
        val *= (g * sig);  // SiLU gating
      }
      o_h[tid] = val;
    }
    __syncthreads();
  }
  if constexpr (Resident) {
    // Transpose four float4 columns within each four-lane group. Each store
    // then writes 64 contiguous bytes per value row, without changing layout.
    const unsigned peer = tid & 3U;
    const std::size_t group_row = ((j0 + tid) & ~std::size_t{3}) * 128;
#pragma unroll
    for (std::size_t vi = 0; vi < 32; vi += 4) {
      // Scalar selections keep the exchanges in registers. Selecting float4
      // aggregates here materializes private storage in the generated code.
      float ax = resident[vi].x, ay = resident[vi].y, az = resident[vi].z,
            aw = resident[vi].w;
      float bx = resident[vi + 1].x, by = resident[vi + 1].y,
            bz = resident[vi + 1].z, bw = resident[vi + 1].w;
      float cx = resident[vi + 2].x, cy = resident[vi + 2].y,
            cz = resident[vi + 2].z, cw = resident[vi + 2].w;
      float dx = resident[vi + 3].x, dy = resident[vi + 3].y,
            dz = resident[vi + 3].z, dw = resident[vi + 3].w;
      // Bound exchange temporaries to this group to prevent hoisting/spills.
      asm volatile(""
                   : "+v"(ax), "+v"(ay), "+v"(az), "+v"(aw), "+v"(bx), "+v"(by),
                     "+v"(bz), "+v"(bw), "+v"(cx), "+v"(cy), "+v"(cz), "+v"(cw),
                     "+v"(dx), "+v"(dy), "+v"(dz), "+v"(dw)
                   :
                   : "memory");
      const auto transpose = [&](float& left, float& right, unsigned bit) {
        const bool other = (tid & bit) != 0;
        const float received = __shfl_xor(other ? left : right, bit);
        left = other ? received : left;
        right = other ? right : received;
      };
      transpose(ax, bx, 1);
      transpose(ay, by, 1);
      transpose(az, bz, 1);
      transpose(aw, bw, 1);
      transpose(cx, dx, 1);
      transpose(cy, dy, 1);
      transpose(cz, dz, 1);
      transpose(cw, dw, 1);
      transpose(ax, cx, 2);
      transpose(ay, cy, 2);
      transpose(az, cz, 2);
      transpose(aw, cw, 2);
      transpose(bx, dx, 2);
      transpose(by, dy, 2);
      transpose(bz, dz, 2);
      transpose(bw, dw, 2);
      detail::StoreRecurrentState4(s_matrix + group_row, vi + peer,
                                   float4{ax, ay, az, aw});
      detail::StoreRecurrentState4(s_matrix + group_row + 128, vi + peer,
                                   float4{bx, by, bz, bw});
      detail::StoreRecurrentState4(s_matrix + group_row + 256, vi + peer,
                                   float4{cx, cy, cz, cw});
      detail::StoreRecurrentState4(s_matrix + group_row + 384, vi + peer,
                                   float4{dx, dy, dz, dw});
      asm volatile("" ::: "memory");
    }
  }
  if constexpr (Split > 1 && WriteOutput) {
    // Last-arriver RMSNorm + SiLU gate for the whole head. No block waits on
    // another: the first Split-1 blocks leave, the last one finishes the head.
    const std::size_t slot =
        static_cast<std::size_t>(blockIdx.y) * num_heads + h;
    __threadfence();  // publish raw o and the partials before arriving
    std::uint32_t arrived = 0;
    if (tid == 0)
      arrived = __hip_atomic_fetch_add(&g_deltanet_split_arrivals[slot], 1U,
                                       __ATOMIC_ACQ_REL,
                                       __HIP_MEMORY_SCOPE_AGENT);
    arrived = __shfl(arrived, 0);
    if (arrived != Split - 1)
      return;
    __threadfence();  // acquire for every lane: see the other blocks' o
    for (std::uint32_t row = 0; row < rows; ++row) {
      float total_sq = 0.0F;
      for (std::size_t w = 0; w < Split; ++w)
        total_sq += g_deltanet_split_sumsq[(slot * kDeltaNetSplitMaxRows + row) *
                                               4 +
                                           w];
      const float mean_sq = total_sq / static_cast<float>(val_dim);
      const float rms = rsqrtf(mean_sq + 1e-6F);
      float* o_h = out_buf + (static_cast<std::size_t>(row) * inner_row_stride) +
                   (h * val_dim);
      const float* row_gate =
          (gate != nullptr)
              ? gate + (static_cast<std::size_t>(row) * inner_row_stride)
              : nullptr;
#pragma unroll
      for (std::size_t w = 0; w < 4; ++w) {
        const std::size_t j = w * 32 + tid;
        const float wn = (ssm_norm != nullptr) ? ssm_norm[j] : 1.0F;
        float val = o_h[j] * rms * wn;
        if (row_gate != nullptr) {
          const float g = row_gate[h * val_dim + j];
          const float sig = 1.0F / (1.0F + expf(-g));
          val *= (g * sig);  // SiLU gating
        }
        o_h[j] = val;
      }
    }
    if (tid == 0)
      __hip_atomic_store(&g_deltanet_split_arrivals[slot], 0U, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  }
}

}  // namespace gufo::hip
#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_DELTANET_DECODE_HPP_

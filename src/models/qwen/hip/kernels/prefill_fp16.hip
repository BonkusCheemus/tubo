#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#if defined(GUFO_ENABLE_NPU)
#include "src/models/qwen/hip/npu_split.hpp"
#endif
namespace gufo::hip {
namespace {
__device__ __forceinline__ __half
RoundActivation(float value) {  // Keep the FP32 rounding boundary before the
                                // FP16 conversion. Without this
  // register dependency, combining the producer with conversion changes rare
  // halfway values relative to the standalone FP32 producer. No instruction
  // or scratch memory is emitted by this compiler barrier.
  asm volatile("" : "+v"(value));
  return __float2half_rn(value);
}
using v16bf = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f WmmaHalf(v16bf a, v16bf b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Permute 16-byte halves and the LDS row to avoid repeated bank conflicts.
__device__ __forceinline__ v16bf LoadSwizzled(const short* p) {
  union {
    v16bf f;
    uint4 q[2];
  } bits;
  const unsigned row = threadIdx.x & 15, shift = (row ^ (row >> 2)) & 1;
  bits.q[0] = reinterpret_cast<const uint4*>(p)[shift];
  bits.q[1] = reinterpret_cast<const uint4*>(p)[shift ^ 1];
  return bits.f;
}
__device__ __forceinline__ void StoreSwizzled(short* p, uint4 lo, uint4 hi,
                                              unsigned row) {
  const unsigned shift = (row ^ (row >> 2)) & 1;
  reinterpret_cast<uint4*>(p)[shift] = lo;
  reinterpret_cast<uint4*>(p)[shift ^ 1] = hi;
}

struct HalfWeights {
  uint4 low, high;
};

struct RawIqWeights {
  unsigned data[4], shift;
  float scale;
};

template<gufo::core::GgmlType Type>
__device__ __forceinline__ RawIqWeights LoadIqRaw(
    const void* row, unsigned sub, const uint2* header = nullptr) {
  static_assert(Type == gufo::core::GgmlType::kIQ4_XS ||
                Type == gufo::core::GgmlType::kIQ4_NL);
  RawIqWeights weights;
  weights.shift = (sub & 1) * 4;
  if constexpr (Type == gufo::core::GgmlType::kIQ4_XS) {
    const auto& block = static_cast<const IQ4XSBlock*>(row)[sub / 16];
    const unsigned sb = (sub / 2) & 7;
    LoadQuantWords16(block.qs + sb * 16, weights.data);
    const unsigned low =
        header ? header->y >> (8 * (sb / 2)) : block.scales_l[sb / 2];
    const unsigned high = header ? header->x >> 16 : block.scales_h;
    const __half d = header ? __ushort_as_half(header->x & 65535) : block.d;
    const int code = static_cast<int>(
                         ((low >> (4 * (sb & 1))) & 15) |
                         (((high >> (2 * sb)) & 3) << 4)) -
                     32;
    weights.scale = __half2float(d) * static_cast<float>(code);
  } else {
    const auto& block = static_cast<const IQ4NLBlock*>(row)[sub / 2];
    LoadQuantWords16(block.qs, weights.data);
    weights.scale = __half2float(block.d);
  }
  return weights;
}

// All 16 IQ4 codebook entries are integers represented exactly in FP16.
// Look up their half bits directly, avoiding signed-byte extraction and
// conversion for each weight. Scaling still uses the original FP32 product.
__device__ __forceinline__ HalfWeights DecodeIqRaw(const RawIqWeights& weights) {
  union {
    v16bf f;
    uint4 u[2];
  } result;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned n = (weights.data[i] >> weights.shift) & 0x0f0f0f0fU;
    const unsigned index = n & 0x07070707U;
    const unsigned select = ((n & 0x08080808U) >> 3) * 255;
    // Low and high bytes of FP16 {-127, -104, -83, -65, -49, -35,
    // -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}, four entries per word.
    const unsigned lo0 =
        __builtin_amdgcn_perm(0x00806020U, 0x103080f0U, index);
    const unsigned lo1 =
        __builtin_amdgcn_perm(0x109050a0U, 0xc0408000U, index);
    const unsigned hi0 =
        __builtin_amdgcn_perm(0xc9cdd0d2U, 0xd4d5d6d7U, index);
    const unsigned hi1 =
        __builtin_amdgcn_perm(0x57555452U, 0x504e4a3cU, index);
    const unsigned lo = (lo0 & ~select) | (lo1 & select);
    const unsigned hi = (hi0 & ~select) | (hi1 & select);
    const unsigned c0 = __builtin_amdgcn_perm(hi, lo, 0x05010400U);
    const unsigned c1 = __builtin_amdgcn_perm(hi, lo, 0x07030602U);
    result.f[4 * i] = static_cast<_Float16>(__fmul_rn(
        weights.scale, __half2float(__ushort_as_half(c0 & 65535))));
    result.f[4 * i + 1] = static_cast<_Float16>(
        __fmul_rn(weights.scale, __half2float(__ushort_as_half(c0 >> 16))));
    result.f[4 * i + 2] = static_cast<_Float16>(__fmul_rn(
        weights.scale, __half2float(__ushort_as_half(c1 & 65535))));
    result.f[4 * i + 3] = static_cast<_Float16>(
        __fmul_rn(weights.scale, __half2float(__ushort_as_half(c1 >> 16))));
  }
  return {result.u[0], result.u[1]};
}

template<gufo::core::GgmlType Type>
__device__ __forceinline__ HalfWeights LoadIqHalf(const void* row,
                                                  unsigned sub) {
  return DecodeIqRaw(LoadIqRaw<Type>(row, sub));
}

template<gufo::core::GgmlType Type>
struct RawWeights {
  static_assert(Type == gufo::core::GgmlType::kQ4_K ||
                Type == gufo::core::GgmlType::kQ5_K ||
                Type == gufo::core::GgmlType::kQ6_K ||
                Type == gufo::core::GgmlType::kQ8_0);
  std::uint32_t low[4], high[Type == gufo::core::GgmlType::kQ5_K ||
                                     Type == gufo::core::GgmlType::kQ6_K
                                 ? 4
                                 : 1];
  __half d, dmin;
  unsigned sc0, sc1, sc2, sub;
  int signed_scale;
};
template<gufo::core::GgmlType Type>
__device__ __forceinline__ RawWeights<Type> LoadRaw(const void* row,
                                                    unsigned sub,
                                                    const uint4* header = nullptr) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  RawWeights<Type> r{};
  r.sub = sub;
  const unsigned sb = (sub / 2) % 8, lane = (sub % 2) * 16;
  if constexpr (Type == GgmlType::kQ8_0) {
    const auto& b = static_cast<const Q8_0Block*>(row)[sub / 2];
    r.d = b.d;
    __builtin_memcpy(r.low, b.qs + lane, 16);
  } else if constexpr (Type == GgmlType::kQ4_K || Type == GgmlType::kQ5_K) {
    using Block =
        std::conditional_t<Type == GgmlType::kQ5_K, Q5KBlock, Q4KBlock>;
    const auto& b = static_cast<const Block*>(row)[sub / 16];
    if (header) {
      r.d = __ushort_as_half(header->x & 65535);
      r.dmin = __ushort_as_half(header->x >> 16);
      const unsigned shift = (sb & 3) * 8;
      r.sc0 = (header->y >> shift) & 255;
      r.sc1 = (header->z >> shift) & 255;
      r.sc2 = (header->w >> shift) & 255;
    } else {
      r.d = b.d;
      r.dmin = b.dmin;
      r.sc0 = b.scales[sb & 3];
      r.sc1 = b.scales[(sb & 3) + 4];
      r.sc2 = b.scales[(sb & 3) + 8];
    }
    LoadQuantWords16(b.qs + (sb / 2) * 32 + lane, r.low);
    if constexpr (Type == GgmlType::kQ5_K)
      LoadQuantWords16(b.qh + lane, r.high);
  } else if constexpr (Type == GgmlType::kQ6_K) {
    const auto& b = static_cast<const Q6KBlock*>(row)[sub / 16];
    unsigned half = sb / 4, segment = sb % 4;
    r.d = b.d;
    r.signed_scale = b.scales[half * 8 + segment * 2 + (sub % 2)];
    LoadQuantWords16(b.ql + half * 64 + (segment % 2) * 32 + lane, r.low);
    LoadQuantWords16(b.qh + half * 32 + lane, r.high);
  }
  return r;
}
template<gufo::core::GgmlType Type>
__device__ __forceinline__ gufo::hip::QuantSub16 DecodeRaw(
    const RawWeights<Type>& r) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  QuantSub16 q;
  q.offset = 0;
  unsigned sb = (r.sub / 2) % 8;
  if constexpr (Type == GgmlType::kQ8_0) {
    __builtin_memcpy(q.w, r.low, 16);
    q.scale = __half2float(r.d);
  } else if constexpr (Type == GgmlType::kQ4_K || Type == GgmlType::kQ5_K) {
    unsigned mask = 0u - (sb >> 2), lo = r.sc0 & 63,
             up = (r.sc2 & 15) | ((r.sc0 >> 6) << 4), lm = r.sc1 & 63,
             um = (r.sc2 >> 4) | ((r.sc1 >> 6) << 4);
    q.scale = __half2float(r.d) * static_cast<float>(lo ^ ((lo ^ up) & mask));
    q.offset =
        __half2float(r.dmin) * static_cast<float>(lm ^ ((lm ^ um) & mask));
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      q.w[i] = (r.low[i] >> (4 * (sb & 1))) & 0x0f0f0f0f;
      if constexpr (Type == GgmlType::kQ5_K)
        q.w[i] |= ((r.high[i] >> sb) & 0x01010101) << 4;
    }
  } else if constexpr (Type == GgmlType::kQ6_K) {
    q.scale = __half2float(r.d) * static_cast<float>(r.signed_scale);
    unsigned seg = sb % 4;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
      q.w[i] = SubBytes(((r.low[i] >> (seg >= 2 ? 4 : 0)) & 0x0f0f0f0f) |
                            (((r.high[i] >> (2 * seg)) & 0x03030303) << 4),
                        0x20202020);
  }
  return q;
}

// Q6_K codes 0..63 become exact FP16 integers without an int->float
// convert: half bits 0x64nn are 1024 + nn, and one packed subtract of 1056
// leaves nn - 32. Scaling keeps DecodeRaw's FP32 product (d * scale * q is
// exact in FP32) and its single FP16 rounding, so the output is bit-identical.
// ISA (10-14, 128x256 kStore): non-WMMA VALU 710 -> 686 per kernel, VGPR 140 -> 137.
__device__ __forceinline__ HalfWeights
DecodeQ6Half(const RawWeights<gufo::core::GgmlType::kQ6_K>& r) {
  using h2 = __attribute__((ext_vector_type(2))) _Float16;
  const float scale = __half2float(r.d) * static_cast<float>(r.signed_scale);
  const unsigned seg = ((r.sub / 2) % 8) % 4;
  union {
    v16bf f;
    uint4 u[2];
  } result;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned codes = ((r.low[i] >> (seg >= 2 ? 4 : 0)) & 0x0f0f0f0fU) |
                           (((r.high[i] >> (2 * seg)) & 0x03030303U) << 4);
    const unsigned pairs[2] = {
        __builtin_amdgcn_perm(0x64646464U, codes, 0x04010400U),
        __builtin_amdgcn_perm(0x64646464U, codes, 0x04030402U)};
#pragma unroll
    for (unsigned j = 0; j < 2; ++j) {
      const h2 v = __builtin_bit_cast(h2, pairs[j]) -
                   h2{static_cast<_Float16>(1056), static_cast<_Float16>(1056)};
      result.f[4 * i + 2 * j] =
          static_cast<_Float16>(__fmul_rn(scale, static_cast<float>(v.x)));
      result.f[4 * i + 2 * j + 1] =
          static_cast<_Float16>(__fmul_rn(scale, static_cast<float>(v.y)));
    }
  }
  return {result.u[0], result.u[1]};
}

// iq3cs DEV (GUFO_DEV_IQ3_CS=1): the IQ3_S / IQ3_XXS gate/up fetch issues only
// the raw block loads; the grid gather and the FP16 decode run at commit, after
// the K-step's WMMAs had the loads in flight (ISA trace: the fetch-stage
// decoder waited 9 vmcnt and ran ~219 VALU before the first WMMA). Same q bytes
// and scale as DecodeQuantSub16<true> (NegateSelectedBytes = its per-element
// negate; checked by a CPU proof), and commit applies the old
// fetch's FP32 product and FP16 rounding, so the LDS tile is byte-identical.
struct RawIq3 {
  std::uint32_t qs;     // four grid-index bytes (IQ3_S: low 8 index bits)
  std::uint32_t signs;  // IQ3_S: the sub-block's 16 sign bits; IQ3_XXS: aux32
  std::uint32_t qh, scale_byte;  // IQ3_S only
  // FP16 d bits, zero-extended. As a __half it became a d16_hi load into a
  // VGPR that the next K-step's LDS reads reuse: a vmcnt wait before WMMAs.
  std::uint32_t d;
  unsigned sub;
};
__device__ __forceinline__ std::uint32_t LoadHalfBits(const __half& h) {
  std::uint16_t bits;
  __builtin_memcpy(&bits, &h, 2);
  return bits;
}
__device__ __forceinline__ float HalfBitsToFloat(std::uint32_t bits) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}
template<gufo::core::GgmlType Type>
__device__ __forceinline__ RawIq3 LoadIq3Raw(const void* row, unsigned sub) {
  static_assert(Type == gufo::core::GgmlType::kIQ3_S ||
                Type == gufo::core::GgmlType::kIQ3_XXS);
  RawIq3 r{};
  r.sub = sub;
  const unsigned sb32 = (sub / 2) % 8, h = sub % 2;
  if constexpr (Type == gufo::core::GgmlType::kIQ3_S) {
    const auto& b = static_cast<const IQ3SBlock*>(row)[sub / 16];
    r.d = LoadHalfBits(b.d);
    __builtin_memcpy(&r.qs, b.qs + sb32 * 8 + 4 * h, 4);
    r.qh = b.qh[sb32];
    std::uint16_t signs;
    __builtin_memcpy(&signs, b.signs + sb32 * 4 + 2 * h, 2);
    r.signs = signs;
    r.scale_byte = b.scales[sb32 / 2];
  } else {
    const auto& b = static_cast<const IQ3XXSBlock*>(row)[sub / 16];
    r.d = LoadHalfBits(b.d);
    __builtin_memcpy(&r.qs, b.qs + sb32 * 8 + 4 * h, 4);
    __builtin_memcpy(&r.signs, b.qs + 64 + 4 * sb32, 4);
  }
  return r;
}
template<gufo::core::GgmlType Type>
__device__ __forceinline__ gufo::hip::QuantSub16 DecodeIq3Raw(const RawIq3& r) {
  using namespace gufo::hip;
  QuantSub16 q;
  q.offset = 0;
  const unsigned sb32 = (r.sub / 2) % 8, h = r.sub % 2;
  if constexpr (Type == gufo::core::GgmlType::kIQ3_S) {
#pragma unroll
    for (unsigned w = 0; w < 4; ++w) {
      // DecodeQuantSub16: l = 2h + w/2, grid_half = w%2, so the qh shift is
      // 8 - 2l - grid_half = 8 - 4h - w and the sign nibble sits at bit 4w.
      const unsigned index = ((r.qs >> (8 * w)) & 255U) |
                             ((r.qh << (8 - 4 * h - w)) & 256U);
      q.w[w] = NegateSelectedBytes(kDeviceIq3sGrid[index], r.signs >> (4 * w));
    }
    const unsigned nibble =
        (sb32 % 2 == 0) ? (r.scale_byte & 0x0FU) : (r.scale_byte >> 4U);
    q.scale = HalfBitsToFloat(r.d) * static_cast<float>(1 + (2 * nibble));
  } else {
#pragma unroll
    for (unsigned g = 0; g < 2; ++g) {
      const unsigned l = 2 * h + g;
      const std::uint32_t signs = Iq2xsSigns((r.signs >> (7U * l)) & 127U);
      q.w[2 * g] = NegateSelectedBytes(
          kDeviceIq3xxsGrid[(r.qs >> (16 * g)) & 255U], signs);
      q.w[2 * g + 1] = NegateSelectedBytes(
          kDeviceIq3xxsGrid[(r.qs >> (16 * g + 8)) & 255U], signs >> 4U);
    }
    q.scale = HalfBitsToFloat(r.d) *
              (0.5F + static_cast<float>(r.signs >> 28U)) * 0.5F;
  }
  return q;
}

enum class HalfEpilogue { kStore, kSwiGLU, kGateUp, kResidual };

// Fully scaled FP16 weights and activations; FP32 accumulation in K16 order.
template<int BM, int BN, int WM, int WN, gufo::core::GgmlType Type,
         HalfEpilogue Epilogue, bool Complete = false,
         gufo::core::GgmlType UpType = Type, bool Iq3Cs = false>
// The 128x256 tile must fit 8 waves/SIMD (2 workgroups per WGP) =
// 192 VGPRs; unconstrained it compiled to 160-256 with spills (code-object dump).
__launch_bounds__(WM * WN * 32) __attribute__((amdgpu_waves_per_eu(
    (BM == 128 && BN == 256) ? 8 : 1))) __global__
    void HalfPrefillGemmKernel(const void* __restrict__ w,
                               const short* __restrict__ x,
                               float* __restrict__ y, int batch, int m, int k,
                               const float* __restrict__ gate,
                               const void* __restrict__ up = nullptr) {
  constexpr bool Paired = Epilogue == HalfEpilogue::kGateUp;
  constexpr bool Mixed = Paired && Type != UpType;
  constexpr bool UpRaw = UpType == gufo::core::GgmlType::kQ4_K ||
                         UpType == gufo::core::GgmlType::kQ5_K;
  constexpr int BK = 4, WaveSize = 32;
  // DEV: the 128x256 / 512-thread tile (GUFO_DEV_HALF_BM128) keeps the
  // 256x256 per-wave tile and weight-side load map (activation loads/thread
  // double), so it takes the same tuned decode paths; 2 WG/WGP needs VGPR<=192.
  constexpr bool Wide = BN == 256 && (BM == 256 || (BM == 128 && WM * WN == 16));
  // A 256-weight Q5 block spans four K64 iterations. Keep its header in
  // registers for the Q5 and Q4/Q5 kernels that benefit from this reuse.
  constexpr bool CacheQ5Header =
      Wide &&
      (Type == gufo::core::GgmlType::kQ5_K ||
       (Type == gufo::core::GgmlType::kQ4_K &&
        UpType == gufo::core::GgmlType::kQ5_K));
  // These IQ4_XS projections benefit from keeping packed weights until
  // LDS commit and reusing the block header. Q3/IQ4_NL mixed pairs retain
  // immediate decoding.
  constexpr bool CacheIqHeader =
      Wide &&
      ((Type == gufo::core::GgmlType::kIQ4_XS &&
        (!Mixed || UpType == gufo::core::GgmlType::kQ4_K ||
         UpType == gufo::core::GgmlType::kQ5_K)) ||
       (Type == gufo::core::GgmlType::kQ4_K &&
        UpType == gufo::core::GgmlType::kIQ4_XS));
  // The mixed Q6 gate is qualified with the fetch-stage decoder; deferring
  // its decode changes dot-product rounding in this kernel specialization.
  constexpr int DecodeStage = (Type == gufo::core::GgmlType::kQ4_K ||
                               Type == gufo::core::GgmlType::kQ5_K ||
                               (Type == gufo::core::GgmlType::kQ6_K && !Mixed) ||
                               Type == gufo::core::GgmlType::kQ8_0)
                                  ? 2
                                  : 0;
  // iq3cs: raw IQ3 loads at fetch, grid gather + decode at commit.
  constexpr bool Iq3Commit =
      Iq3Cs && !Mixed &&
      (Type == gufo::core::GgmlType::kIQ3_S ||
       Type == gufo::core::GgmlType::kIQ3_XXS);
  static_assert(!Iq3Cs || Iq3Commit);
  constexpr int kThreads = WM * WN * WaveSize;
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int KS = 16;
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);

  constexpr int PFA = (BM * BK + kThreads - 1) / kThreads;
  constexpr int PFB = (BN * BK + kThreads - 1) / kThreads;

  __shared__ short s_a[BK][RS][16][KS];
  __shared__ short s_b[BK][TS][16][KS];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WaveSize;
  const int lane = tid % WaveSize;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;

  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    // Wider row groups improve input reuse for these large projections.
    // Long Q5 reductions and mixed gate/up decoders retain smaller groups.
    unsigned group_shift = 3;
    if constexpr (Wide && !Mixed) {
      if constexpr (Type == gufo::core::GgmlType::kIQ4_XS)
        group_shift = 5;
      else if constexpr (Type == gufo::core::GgmlType::kQ5_K ||
                         Type == gufo::core::GgmlType::kQ6_K)
        // Q6_K was untuned at 3; its gate/up fetched 1.86x the
        // DRAM bytes of Q5_K on the same grid (hardware counters).
        group_shift = k <= 8192 ? 5 : 3;
    }
    const unsigned group = 1U << group_shift;
    unsigned first = (blockIdx.y >> group_shift) << group_shift,
             within = (blockIdx.y & (group - 1)) * gridDim.x + blockIdx.x,
             rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = row_tile * (Paired ? BM / 2 : BM),
            t_block = token_tile * BN;
  const int ksteps = k / 16;

  v8f acc[WRS][WTS] = {};

  // Hoisted 64-bit row bases, clamped so an out-of-range row reads row m-1 and
  // is masked out at the store instead of branching in the K loop.
  const std::uint8_t* w_row[PFA];
  const std::size_t row_bytes = gufo::hip::QuantRowBytes(Type, k);
  const std::size_t up_row_bytes = gufo::hip::QuantRowBytes(UpType, k);
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int physical_row = ((p * kThreads) + tid) / BK;
    const int r =
        r_block +
        (Paired ? (physical_row / 32) * 16 + physical_row % 16 : physical_row);
    const void* source = Paired && physical_row % 32 >= 16 ? up : w;
    w_row[p] = static_cast<const std::uint8_t*>(source) +
               (static_cast<std::size_t>((Complete || r < m) ? r : (m - 1)) *
                (Mixed && physical_row % 32 >= 16 ? up_row_bytes : row_bytes));
  }
  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    x_row[p] =
        x +
        (static_cast<std::size_t>((Complete || t < batch) ? t : (batch - 1)) *
         k);
  }

  uint4 ra[PFA][2], rb[PFB][2], headers[PFA];
  uint2 iq_headers[PFA];
  RawIqWeights raw_iq[PFA];
  struct NoRaw {};
  std::conditional_t<DecodeStage == 2, RawWeights<Type>, NoRaw> raw[PFA];
  std::conditional_t<Mixed && UpRaw, RawWeights<UpType>, NoRaw> raw_up[PFA];
  std::conditional_t<Iq3Commit, RawIq3, NoRaw> raw_iq3[PFA];
  const auto fetch_iq = [&](int p, int kk) {
    if ((kk & 12) == 0)
      __builtin_memcpy(
          &iq_headers[p],
          reinterpret_cast<const IQ4XSBlock*>(w_row[p]) + unsigned(kk) / 16,
          8);
    raw_iq[p] = LoadIqRaw<gufo::core::GgmlType::kIQ4_XS>(
        w_row[p], unsigned(kk), &iq_headers[p]);
  };
  const auto fetch_up = [&](int p, int kk) {
    if constexpr (Mixed) {
      if constexpr (UpRaw) {
        if constexpr (CacheQ5Header &&
                      UpType == gufo::core::GgmlType::kQ5_K) {
          if ((kk & 12) == 0)
            __builtin_memcpy(
                &headers[p],
                reinterpret_cast<const Q5KBlock*>(w_row[p]) + unsigned(kk) / 16,
                16);
          raw_up[p] = LoadRaw<UpType>(w_row[p], kk, &headers[p]);
        } else {
          raw_up[p] = LoadRaw<UpType>(w_row[p], kk);
        }
      } else if constexpr (CacheIqHeader &&
                           UpType == gufo::core::GgmlType::kIQ4_XS) {
        fetch_iq(p, kk);
      } else if constexpr (UpType == gufo::core::GgmlType::kIQ4_XS ||
                           UpType == gufo::core::GgmlType::kIQ4_NL) {
        const auto packed = LoadIqHalf<UpType>(w_row[p], unsigned(kk));
        ra[p][0] = packed.low;
        ra[p][1] = packed.high;
      } else {
        gufo::hip::QuantSub16 q;
        gufo::hip::DecodeQuantSub16<true>(UpType, w_row[p], kk, q);
        union { v16bf f; uint4 u[2]; } packed;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          packed.f[j] = static_cast<_Float16>(__fsub_rn(
              __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
        ra[p][0] = packed.u[0];
        ra[p][1] = packed.u[1];
      }
    }
  };
  const auto fetch = [&](int ks0) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BM * BK && (Complete || kk < ksteps)) {
        if (Mixed && (idx / BK) % 32 >= 16)
          fetch_up(p, kk);
        else if constexpr (CacheQ5Header &&
                           Type == gufo::core::GgmlType::kQ5_K) {
          if ((ks0 & 15) == 0)
            __builtin_memcpy(
                &headers[p],
                reinterpret_cast<const Q5KBlock*>(w_row[p]) + unsigned(kk) / 16,
                16);
          raw[p] = LoadRaw<Type>(w_row[p], kk, &headers[p]);
        } else if constexpr (DecodeStage == 2)
          raw[p] = LoadRaw<Type>(w_row[p], kk);
        else if constexpr (CacheIqHeader &&
                           Type == gufo::core::GgmlType::kIQ4_XS) {
          fetch_iq(p, kk);
        } else if constexpr (Type == gufo::core::GgmlType::kIQ4_XS ||
                           Type == gufo::core::GgmlType::kIQ4_NL) {
          const auto packed = LoadIqHalf<Type>(w_row[p], unsigned(kk));
          ra[p][0] = packed.low;
          ra[p][1] = packed.high;
        } else if constexpr (Iq3Commit) {
          raw_iq3[p] = LoadIq3Raw<Type>(w_row[p], unsigned(kk));
        }
        else {
          gufo::hip::QuantSub16 q;
          gufo::hip::DecodeQuantSub16<true>(Type, w_row[p], kk, q);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            packed.f[j] = static_cast<_Float16>(__fsub_rn(
                __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
          ra[p][0] = packed.u[0];
          ra[p][1] = packed.u[1];
        }
      } else {
        if constexpr (CacheIqHeader)
          raw_iq[p] = {};
        if constexpr (Iq3Commit)
          raw_iq3[p] = {};  // decodes to +0 halves, as the zeroed ra below
        ra[p][0] = make_uint4(0, 0, 0, 0);
        ra[p][1] = make_uint4(0, 0, 0, 0);
        if (Mixed && (idx / BK) % 32 >= 16)
          fetch_up(p, kk);
        else if constexpr (DecodeStage == 2)
          raw[p] = {};
        if constexpr (Mixed && UpRaw)
          raw_up[p] = {};
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if (idx < BN * BK && (Complete || kk < ksteps)) {
        const short* src = x_row[p] + static_cast<std::size_t>(kk) * 16;
        rb[p][0] = *reinterpret_cast<const uint4*>(src);
        rb[p][1] = *reinterpret_cast<const uint4*>(src + 8);
      } else {
        rb[p][0] = make_uint4(0, 0, 0, 0);
        rb[p][1] = make_uint4(0, 0, 0, 0);
      }
    }
  };
  const auto commit_up = [&](int p, short* dst, int row) {
    if constexpr (Mixed) {
      if constexpr (UpRaw) {
        gufo::hip::QuantSub16 q = DecodeRaw<UpType>(raw_up[p]);
        union { v16bf f; uint4 u[2]; } packed;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          packed.f[j] = static_cast<_Float16>(__fsub_rn(
              __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
        StoreSwizzled(dst, packed.u[0], packed.u[1], row);
      } else if constexpr (CacheIqHeader &&
                           UpType == gufo::core::GgmlType::kIQ4_XS) {
        const auto packed = DecodeIqRaw(raw_iq[p]);
        StoreSwizzled(dst, packed.low, packed.high, row);
      } else {
        StoreSwizzled(dst, ra[p][0], ra[p][1], row);
      }
    }
  };
  const auto commit = [&]() {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BM * BK) {
        const int placed = (row ^ (ks & 3));
        short* dst = &s_a[ks][placed / 16][placed % 16][0];
        if (Mixed && row % 32 >= 16) {
          commit_up(p, dst, row);
        } else if constexpr (DecodeStage != 0 &&
                             Type == gufo::core::GgmlType::kQ6_K) {
          const auto packed = DecodeQ6Half(raw[p]);
          StoreSwizzled(dst, packed.low, packed.high, row);
        } else if constexpr (DecodeStage != 0) {
          gufo::hip::QuantSub16 q = DecodeRaw<Type>(raw[p]);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            packed.f[j] = static_cast<_Float16>(__fsub_rn(
                __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
          StoreSwizzled(dst, packed.u[0], packed.u[1], row);
        } else if constexpr (CacheIqHeader &&
                             Type == gufo::core::GgmlType::kIQ4_XS) {
          const auto packed = DecodeIqRaw(raw_iq[p]);
          StoreSwizzled(dst, packed.low, packed.high, row);
        } else if constexpr (Iq3Commit) {
          // The old fetch-stage else branch's ops, moved here unchanged.
          const gufo::hip::QuantSub16 q = DecodeIq3Raw<Type>(raw_iq3[p]);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            packed.f[j] = static_cast<_Float16>(__fsub_rn(
                __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
          StoreSwizzled(dst, packed.u[0], packed.u[1], row);
        } else
          StoreSwizzled(dst, ra[p][0], ra[p][1], row);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BN * BK) {
        const int placed = (row ^ (ks & 3));
        short* dst = &s_b[ks][placed / 16][placed % 16][0];
        StoreSwizzled(dst, rb[p][0], rb[p][1], row);
      }
    }
  };
  fetch(0);
  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    commit();
    __syncthreads();
    if (ks0 + BK < ksteps)
      fetch(ks0 + BK);
#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      v16bf a[WRS];
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        a[i] = LoadSwizzled(&s_a[ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
      }
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const v16bf b =
            LoadSwizzled(&s_b[ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int i = 0; i < WRS; ++i) {
          if constexpr (Type == gufo::core::GgmlType::kIQ3_S)
            acc[i][j] = WmmaHalf(a[i], b, acc[i][j]);
          else
            acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
        }
      }
    }
    if constexpr (Wide &&
                  ((Type == gufo::core::GgmlType::kIQ4_XS &&
                    (!Mixed || UpType != gufo::core::GgmlType::kQ5_K)) ||
                   (Type == gufo::core::GgmlType::kIQ4_NL &&
                    UpType == gufo::core::GgmlType::kIQ4_XS))) {
      // Interleave LDS reads with WMMA after IQ4 decoding. Mixed Q5
      // decoders retain the compiler's default instruction schedule.
      __builtin_amdgcn_iglp_opt(0);
    }
    __syncthreads();
  }

  // Orient WMMA output as [token][row] for coalesced direct stores. IQ3_S
  // keeps the original orientation and LDS transpose, which is faster for
  // its decoder's register schedule.
  const auto store = [&](std::size_t index, float value) {
    if constexpr (Epilogue == HalfEpilogue::kSwiGLU) {
      const float g = gate[index];
      const float sig = 1.0F / (1.0F + expf(-g));
      reinterpret_cast<__half*>(y)[index] =
          RoundActivation(__fmul_rn(g * sig, value));
    } else if constexpr (Epilogue == HalfEpilogue::kResidual) {
      // Add only after the dot product, preserving the FP32 residual link.
      y[index] = __fadd_rn(y[index], value);
    } else
      y[index] = value;
  };
  const int hi = lane / 16;
  if constexpr (Paired && Type == gufo::core::GgmlType::kIQ3_S) {
    // IQ3_S fragments hold [row = 2l + hi][token = sl] (swapped WMMA
    // operands). Gate (i) and up (i + 1) share that layout, so SwiGLU runs on
    // the registers, then the LDS transpose of the unpaired IQ3_S store below
    // writes [token][row]. Same FP32 ops as the two-GEMM kSwiGLU path.
    static_assert(WRS % 2 == 0);
    __syncthreads();
    constexpr int Slots =
        (sizeof(s_a) >= sizeof(s_b) ? sizeof(s_a) : sizeof(s_b)) / (256 * 4);
    static_assert(Slots >= WM * WN);
    float* scratch = (sizeof(s_a) >= sizeof(s_b)
                          ? reinterpret_cast<float*>(&s_a[0][0][0][0])
                          : reinterpret_cast<float*>(&s_b[0][0][0][0])) +
                     (wave * 256);
#pragma unroll
    for (int i = 0; i < WRS; i += 2) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
#pragma unroll
        for (int l = 0; l < 256 / WaveSize; ++l) {
          const float g = acc[i][j][l], u = acc[i + 1][j][l];
          const float sig = 1.0F / (1.0F + expf(-g));
          scratch[(sl * 16) + ((WaveSize / 16) * l) + hi] = __fmul_rn(g * sig, u);
        }
        __builtin_amdgcn_wave_barrier();
        const int r0 = r_block + (wr * (WRS / 2) + i / 2) * 16;
        const int t0 = t_block + (((wt * WTS) + j) * 16);
#pragma unroll
        for (int s = 0; s < 256 / WaveSize; ++s) {
          const int flat = (s * WaveSize) + lane;
          const int tok = t0 + (flat >> 4);
          const int r = r0 + (flat & 15);
          if (tok < batch && r < m)
            reinterpret_cast<__half*>(y)[std::size_t(tok) * m + r] =
                RoundActivation(scratch[flat]);
        }
        __builtin_amdgcn_wave_barrier();
      }
    }
  } else if constexpr (Paired) {
    static_assert(WRS % 2 == 0);
    // Adjacent 16-row fragments hold gate and up for the same logical rows.
    // One wave owns both accumulators, so no intermediate store or exchange
    // between waves is needed.
#pragma unroll
    for (int i = 0; i < WRS; i += 2) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int row = r_block + (wr * (WRS / 2) + i / 2) * 16 + sl;
        const int t0 = t_block + (wt * WTS + j) * 16;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int token = t0 + 2 * l + hi;
          if (Complete || (row < m && token < batch)) {
            const float g = acc[i][j][l], u = acc[i + 1][j][l];
            const float sig = 1.0F / (1.0F + expf(-g));
            reinterpret_cast<__half*>(y)[std::size_t(token) * m + row] =
                RoundActivation(__fmul_rn(g * sig, u));
          }
        }
      }
    }
  } else if constexpr (Type != gufo::core::GgmlType::kIQ3_S) {
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int row = r_block + ((wr * WRS + i) * 16) + sl;
        const int t0 = t_block + ((wt * WTS + j) * 16);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int token = t0 + 2 * l + hi;
          if (Complete || (row < m && token < batch))
            store(std::size_t(token) * m + row, acc[i][j][l]);
        }
      }
    }
  } else {
    __syncthreads();
    constexpr int Slots =
        (sizeof(s_a) >= sizeof(s_b) ? sizeof(s_a) : sizeof(s_b)) / (256 * 4);
    static_assert(Slots >= WM * WN);
    float* scratch = (sizeof(s_a) >= sizeof(s_b)
                          ? reinterpret_cast<float*>(&s_a[0][0][0][0])
                          : reinterpret_cast<float*>(&s_b[0][0][0][0])) +
                     (wave * 256);

#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
#pragma unroll
        for (int l = 0; l < 256 / WaveSize; ++l) {
          scratch[(sl * 16) + ((WaveSize / 16) * l) + hi] = acc[i][j][l];
        }
        __builtin_amdgcn_wave_barrier();
        const int r0 = r_block + (((wr * WRS) + i) * 16);
        const int t0 = t_block + (((wt * WTS) + j) * 16);
        if (t0 + 16 <= batch && r0 + 16 <= m) {
          const std::size_t base =
              (static_cast<std::size_t>(t0) * m) + static_cast<std::size_t>(r0);
#pragma unroll
          for (int s = 0; s < 256 / WaveSize; ++s) {
            const int flat = (s * WaveSize) + lane;
            store(
                base + (static_cast<std::size_t>(flat >> 4) * m) + (flat & 15),
                scratch[flat]);
          }
        } else {
#pragma unroll
          for (int s = 0; s < 256 / WaveSize; ++s) {
            const int flat = (s * WaveSize) + lane;
            const int tok = t0 + (flat >> 4);
            const int r = r0 + (flat & 15);
            if (tok < batch && r < m) {
              store((static_cast<std::size_t>(tok) * m) + r, scratch[flat]);
            }
          }
        }
        __builtin_amdgcn_wave_barrier();
      }
    }
  }
}

__global__ void HalfNorm(const float* x, const float* residual,
                         const float* weight, float* sum_out, __half* out,
                         std::size_t dim, float eps) {
  const std::size_t row = blockIdx.x, tid = threadIdx.x, base = row * dim;
  float sum_sq = 0;
  for (std::size_t i = tid; i < dim; i += 256) {
    const float v =
        residual ? __fadd_rn(x[base + i], residual[base + i]) : x[base + i];
    sum_sq += v * v;
  }
  __shared__ float sums[256];
  sums[tid] = sum_sq;
  __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (tid < stride)
      sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(dim) + eps);
  for (std::size_t i = tid; i < dim; i += 256) {
    const float v =
        residual ? __fadd_rn(x[base + i], residual[base + i]) : x[base + i];
    if (sum_out)
      sum_out[base + i] = v;
    out[base + i] = RoundActivation(__fmul_rn(v * inv, weight[i]));
  }
}
__global__ void HalfNorm5120(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              __half* __restrict__ out, float eps) {
  constexpr unsigned Dim = 5120, Elements = Dim / 256;
  const unsigned row = blockIdx.x, tid = threadIdx.x;
  const std::size_t base = std::size_t(row) * Dim;
  float sum_sq = 0;
#pragma unroll
  for (unsigned j = 0; j < Elements; ++j) {
    const float v = x[base + j * 256 + tid];
    // Explicit FMAs preserve which square rounds first when LLVM unrolls
    // the loop; contracting two ordinary products can swap that boundary.
    sum_sq = __fmaf_rn(v, v, sum_sq);
  }
  __shared__ float sums[256];
  sums[tid] = sum_sq;
  __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (tid < stride)
      sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(Dim) + eps);
#pragma unroll
  for (unsigned j = 0; j < Elements; ++j) {
    const unsigned i = j * 256 + tid;
    out[base + i] = RoundActivation(__fmul_rn(x[base + i] * inv, weight[i]));
  }
}

__global__ void HalfCast(const float* x, __half* y, std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __float2half_rn(x[i]);
}
__global__ void HalfWiden(const __half* x, float* y, std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __half2float(x[i]);
}
// DEV quality probe: GUFO_DEV_FAKE_A4 rounds every f16 prefill GEMM
// input in place, per 32-element K group, symmetric, scale amax/qmax: the
// activation side of a low-bit prefill. Measures quality only; adds a pass.
//   1 = int4 (qmax 7), the W4A4 iu4 path as-is
//   2 = int4 after a 32-wide orthonormal Hadamard, undone after rounding:
//       x_hat = H Q(H x), what W' = W H + iu4 on H x computes (QuaRot-style)
//   8 = int8 (qmax 127), per 32-element group (finer than the NPU can use)
//   9 = int8 (qmax 127), ONE scale per token row over all k: what the NPU
//       int8 GEMM computes (it sums the whole row into one int32)
//  11 = bf16 (round to nearest even): an NPU bf16 GEMM with true bf16 inputs
//  12 = bfp16ebs8 (XDNA2 block float, aie_api aie_doc.hpp block_vector table):
//       8 consecutive k share one exponent, 8-bit signed mantissa each = int8
//       with a power-of-two scale per 8 values; what the NPU bf16/bfp16 GEMM
//       (IRON flm, measured 12.4-13.1 TFLOPS) feeds the MAC. RNE assumed (the
//       hardware conversion's rounding is not documented here: optimistic).
// Rows are k-contiguous and k % 32 == 0, so one wave32 = one K group.
__device__ float Hadamard32(float v) {
  const int lane = threadIdx.x & 31;
  for (int o = 1; o < 32; o <<= 1) {
    const float p = __shfl_xor(v, o, 32);
    v = (lane & o) ? p - v : v + p;
  }
  return v * 0.17677669529663687f;  // 1/sqrt(32): H is its own inverse
}
template <int Mode>
__global__ void FakeQuantA4(__half* x, std::size_t n) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  float v = i < n ? __half2float(x[i]) : 0.0f;
  if constexpr (Mode == 11) {
    if (i < n) x[i] = __float2half_rn(__bfloat162float(__float2bfloat16(v)));
    return;
  }
  if constexpr (Mode == 12) {  // 8-lane groups: shuffles must see all 32 lanes
    float a8 = isfinite(v) ? fabsf(v) : INFINITY;
    for (int o = 4; o > 0; o >>= 1) a8 = fmaxf(a8, __shfl_xor(a8, o, 32));
    if (i < n && a8 > 0.0f && isfinite(a8)) {
      const float s = ldexpf(1.0f, ilogbf(a8) - 6);  // a8/s in [64, 128)
      x[i] = __float2half_rn(fminf(fmaxf(rintf(v / s), -127.0f), 127.0f) * s);
    }
    return;
  }
  if constexpr (Mode == 2) v = Hadamard32(v);
  constexpr float qmax = Mode == 8 ? 127.0f : 7.0f;
  float a = isfinite(v) ? fabsf(v) : INFINITY;  // fmaxf drops NaN lanes
  for (int o = 16; o > 0; o >>= 1)
    a = fmaxf(a, __shfl_xor(a, o, 32));
  if (a > 0.0f && isfinite(a)) {  // leave NaN/inf visible
    const float s = a / qmax;
    v = fminf(fmaxf(rintf(v / s), -qmax), qmax) * s;
  }
  if constexpr (Mode == 2) v = Hadamard32(v);  // every lane: shuffles need all 32
  if (i < n && isfinite(a) && a > 0.0f) x[i] = __float2half_rn(v);
}
// Mode 9: one block per row; amax over the row's k values, then int8 round.
__global__ void FakeQuantRowInt8(__half* x, std::size_t k) {
  __half* row = x + std::size_t(blockIdx.x) * k;
  __shared__ float red[256];
  float a = 0.0f;
  for (std::size_t j = threadIdx.x; j < k; j += blockDim.x) {
    const float v = __half2float(row[j]);
    a = fmaxf(a, isfinite(v) ? fabsf(v) : INFINITY);
  }
  red[threadIdx.x] = a;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  a = red[0];
  if (!(a > 0.0f) || !isfinite(a)) return;  // leave NaN/inf visible
  const float s = a / 127.0f;
  for (std::size_t j = threadIdx.x; j < k; j += blockDim.x) {
    const float v = __half2float(row[j]);
    row[j] = __float2half_rn(fminf(fmaxf(rintf(v / s), -127.0f), 127.0f) * s);
  }
}
// mode 10: mode 9 on every GEMM except residual-epilogue ones
// (down_proj, GDN out_proj, attn o_proj) = the NPU share when those stay on GPU.
void MaybeFakeA4(const void* x, std::size_t batch, std::size_t k,
                 hipStream_t stream, bool residual = false) {
  const std::size_t n = batch * k;
  static const int mode = [] {
    const char* e = std::getenv("GUFO_DEV_FAKE_A4");
    if (e == nullptr) return 0;
    const int m = std::atoi(e);
    return m == 1 || m == 2 || m == 8 || m == 9 || m == 10 || m == 11 || m == 12 ? m : 0;
  }();
  if (mode != 0 && !(mode == 10 && residual)) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      std::fprintf(stderr,
                   "GUFO_DEV_FAKE_A4 active mode %d (first f16 prefill GEMM)\n",
                   mode);
    }
    auto* h = static_cast<__half*>(const_cast<void*>(x));
    const auto grid = (n + 255) / 256;
    if (mode == 1) FakeQuantA4<1><<<grid, 256, 0, stream>>>(h, n);
    if (mode == 2) FakeQuantA4<2><<<grid, 256, 0, stream>>>(h, n);
    if (mode == 8) FakeQuantA4<8><<<grid, 256, 0, stream>>>(h, n);
    if (mode == 11) FakeQuantA4<11><<<grid, 256, 0, stream>>>(h, n);
    if (mode == 12) FakeQuantA4<12><<<grid, 256, 0, stream>>>(h, n);
    if (mode == 9 || mode == 10) FakeQuantRowInt8<<<batch, 256, 0, stream>>>(h, k);
    HIP_CHECK(hipGetLastError());
  }
}
// DEV: GUFO_DEV_HALF_BM128=1 launches the large f16 GEMMs as 128x256
// tiles of 512 threads (two workgroups per WGP) instead of 256x256 / 1024.
bool HalfBm128() {
  static const bool on = [] {
    const char* e = std::getenv("GUFO_DEV_HALF_BM128");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// DEV: GUFO_DEV_IQ3_DUAL_GU=1 runs same-type IQ3_S / IQ3_XXS ffn
// gate+up as one dual GEMM, so the NPU split can take their rows (EF-apex:
// all 64 gate/up pairs). IQ3_S uses its own Paired epilogue (swapped WMMA
// orientation, LDS transpose) in HalfPrefillGemmKernel.
bool Iq3DualGateUp() {
  static const bool on = [] {
    const char* e = std::getenv("GUFO_DEV_IQ3_DUAL_GU");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// iq3cs DEV: GUFO_DEV_IQ3_CS=1 runs that IQ3 dual gate/up (256x256 tile) with
// the commit-stage decoder (RawIq3); same bytes. Unset = the fetch-stage one.
bool Iq3CommitStage() {
  static const bool on = [] {
    const char* e = std::getenv("GUFO_DEV_IQ3_CS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
}  // namespace
void LaunchBatchedRMSNormFp16(const float* x, const float* residual,
                              const float* weight, float* sum_out, void* out,
                              std::size_t batch, std::size_t dim, float eps,
                              hipStream_t stream) {
  if (dim == 5120 && residual == nullptr && sum_out == nullptr) {
    HalfNorm5120<<<batch, 256, 0, stream>>>(
        x, weight, static_cast<__half*>(out), eps);
    return;
  }
  HalfNorm<<<batch, 256, 0, stream>>>(x, residual, weight, sum_out,
                                      static_cast<__half*>(out), dim, eps);
}
void LaunchFloatToFp16(const float* x, void* y, std::size_t n,
                       hipStream_t stream) {
  HalfCast<<<(n + 255) / 256, 256, 0, stream>>>(x, static_cast<__half*>(y), n);
}
void LaunchFp16ToFloat(const void* x, float* y, std::size_t n,
                       hipStream_t stream) {
  HalfWiden<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const __half*>(x), y, n);
}
template<HalfEpilogue Epilogue>
void DirectGemm(gufo::core::GgmlType type, const void* w, const void* x,
                float* y, std::size_t batch, std::size_t m, std::size_t k,
                const float* gate, hipStream_t stream) {
  MaybeFakeA4(x, batch, k, stream, Epilogue == HalfEpilogue::kResidual);
  // Rows [m_grid, m) go to the NPU; the GPU grid stops at m_grid
  // (whole 256-row tiles), the row stride stays m.
  std::size_t m_grid = m;
#if defined(GUFO_ENABLE_NPU)
  std::uint32_t npu_seq = 0;
  if constexpr (Epilogue == HalfEpilogue::kResidual)
    m_grid = npu::BeginSplit(type, w, x, batch, m, k, stream, &npu_seq);
  else if constexpr (Epilogue == HalfEpilogue::kStore)
    m_grid = npu::BeginStore(type, w, x, batch, m, k, stream, &npu_seq);
#endif
  const auto launch = [&]<gufo::core::GgmlType Type>() {
    constexpr bool LargeKvTile = Type == gufo::core::GgmlType::kQ4_K ||
                                 Type == gufo::core::GgmlType::kQ5_K ||
                                 Type == gufo::core::GgmlType::kQ6_K ||
                                 Type == gufo::core::GgmlType::kQ8_0;
    const auto large = [&]<bool Full>() {
      if (HalfBm128()) {
        HalfPrefillGemmKernel<128, 256, 4, 4, Type, Epilogue, Full>
            <<<dim3((batch + 255) / 256, (m_grid + 127) / 128), 512, 0, stream>>>(
                w, static_cast<const short*>(x), y, batch, m, k, gate);
        return;
      }
      HalfPrefillGemmKernel<256, 256, 8, 4, Type, Epilogue, Full>
          <<<dim3((batch + 255) / 256, (m_grid + 255) / 256), 1024, 0, stream>>>(
              w, static_cast<const short*>(x), y, batch, m, k, gate);
    };
    if (Epilogue == HalfEpilogue::kSwiGLU || m >= 4096 ||
        (LargeKvTile && m >= 1024 && batch >= 1024)) {
      // Only Q4/Q5 benefit from removing the tail checks on complete tiles.
      // Other formats retain the general kernel's better register schedule.
      if constexpr (Type == gufo::core::GgmlType::kQ4_K ||
                    Type == gufo::core::GgmlType::kQ5_K) {
        if (m % 256 == 0 && batch % 256 == 0) {
          large.template operator()<true>();
          return;
        }
      }
      large.template operator()<false>();
    } else if constexpr (Epilogue != HalfEpilogue::kSwiGLU) {
      if (m >= 1024) {
        HalfPrefillGemmKernel<64, 128, 2, 4, Type, Epilogue>
            <<<dim3((batch + 127) / 128, (m + 63) / 64), 256, 0, stream>>>(
                w, static_cast<const short*>(x), y, batch, m, k, nullptr);
      } else {
        HalfPrefillGemmKernel<32, 64, 2, 2, Type, Epilogue>
            <<<dim3((batch + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(
                w, static_cast<const short*>(x), y, batch, m, k, nullptr);
      }
    }
  };
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      launch.template operator()<GgmlType::kQ4_K>();
      break;
    case GgmlType::kQ5_K:
      launch.template operator()<GgmlType::kQ5_K>();
      break;
    case GgmlType::kQ6_K:
      launch.template operator()<GgmlType::kQ6_K>();
      break;
    case GgmlType::kIQ4_XS:
      launch.template operator()<GgmlType::kIQ4_XS>();
      break;
    case GgmlType::kIQ4_NL:
      launch.template operator()<GgmlType::kIQ4_NL>();
      break;
    case GgmlType::kQ8_0:
      launch.template operator()<GgmlType::kQ8_0>();
      break;
    case GgmlType::kQ3_K:
      launch.template operator()<GgmlType::kQ3_K>();
      break;
    case GgmlType::kIQ3_S:
      launch.template operator()<GgmlType::kIQ3_S>();
      break;
    case GgmlType::kQ2_K:
      launch.template operator()<GgmlType::kQ2_K>();
      break;
    case GgmlType::kIQ2_XXS:
      launch.template operator()<GgmlType::kIQ2_XXS>();
      break;
    case GgmlType::kIQ2_XS:
      launch.template operator()<GgmlType::kIQ2_XS>();
      break;
    case GgmlType::kIQ2_S:
      launch.template operator()<GgmlType::kIQ2_S>();
      break;
    case GgmlType::kIQ3_XXS:
      launch.template operator()<GgmlType::kIQ3_XXS>();
      break;
    case GgmlType::kIQ1_M:
      launch.template operator()<GgmlType::kIQ1_M>();
      break;
    default:
      throw std::runtime_error("unexpected half prefill weight format");
  }
#if defined(GUFO_ENABLE_NPU)
  if constexpr (Epilogue == HalfEpilogue::kResidual) {
    if (m_grid != m) npu::FinishResidual(y, batch, m, m_grid, stream, npu_seq);
  } else if constexpr (Epilogue == HalfEpilogue::kStore) {
    if (m_grid != m) npu::FinishStore(y, batch, m, m_grid, stream, npu_seq);
  }
#endif
}
void LaunchBatchedQuantGEMMFp16(gufo::core::GgmlType type, const void* w,
                                const void* x, float* y, std::size_t batch,
                                std::size_t m, std::size_t k,
                                hipStream_t stream) {
  DirectGemm<HalfEpilogue::kStore>(type, w, x, y, batch, m, k, nullptr, stream);
}
void LaunchBatchedQuantGEMMSwiGLUFp16(gufo::core::GgmlType type, const void* w,
                                      const void* x, const float* gate, void* y,
                                      std::size_t batch, std::size_t m,
                                      std::size_t k, hipStream_t stream) {
  DirectGemm<HalfEpilogue::kSwiGLU>(type, w, x, static_cast<float*>(y), batch,
                                    m, k, gate, stream);
}

void LaunchBatchedQuantGEMMResidualFp16(gufo::core::GgmlType type,
                                        const void* w, const void* x,
                                        float* residual, std::size_t batch,
                                        std::size_t m, std::size_t k,
                                        hipStream_t stream) {
  DirectGemm<HalfEpilogue::kResidual>(type, w, x, residual, batch, m, k,
                                      nullptr, stream);
}

bool TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
    gufo::core::GgmlType gate_type, gufo::core::GgmlType up_type,
    const void* gate_weights, const void* up_weights, const void* x, void* y,
    std::size_t batch, std::size_t m, std::size_t k, hipStream_t stream) {
  const auto launch = [&]<gufo::core::GgmlType Type,
                          gufo::core::GgmlType UpType = Type>() {
    MaybeFakeA4(x, batch, k, stream);
    std::size_t m_grid = m;  // GPU rows; the rest go to the NPU (dev only)
#if defined(GUFO_ENABLE_NPU)
    std::uint32_t npu_seq = 0;
    m_grid = npu::BeginGateUp(gate_type, up_type, gate_weights, up_weights, x,
                              batch, m, k, stream, &npu_seq);
#endif
    const auto tile = [&]<bool Full>() {
      if (HalfBm128()) {
        HalfPrefillGemmKernel<128, 256, 4, 4, Type, HalfEpilogue::kGateUp,
                              Full, UpType>
            <<<dim3((batch + 255) / 256, (m_grid + 63) / 64), 512, 0, stream>>>(
                gate_weights, static_cast<const short*>(x),
                static_cast<float*>(y), batch, m, k, nullptr, up_weights);
        return;
      }
      if constexpr ((Type == gufo::core::GgmlType::kIQ3_S ||
                     Type == gufo::core::GgmlType::kIQ3_XXS) &&
                    Type == UpType) {
        if (Iq3CommitStage()) {
          HalfPrefillGemmKernel<256, 256, 8, 4, Type, HalfEpilogue::kGateUp,
                                Full, UpType, true>
              <<<dim3((batch + 255) / 256, (m_grid + 127) / 128), 1024, 0,
                 stream>>>(gate_weights, static_cast<const short*>(x),
                           static_cast<float*>(y), batch, m, k, nullptr,
                           up_weights);
          return;
        }
      }
      HalfPrefillGemmKernel<256, 256, 8, 4, Type, HalfEpilogue::kGateUp, Full,
                            UpType>
          <<<dim3((batch + 255) / 256, (m_grid + 127) / 128), 1024, 0, stream>>>(
              gate_weights, static_cast<const short*>(x),
              static_cast<float*>(y), batch, m, k, nullptr, up_weights);
    };
    // Complete-tile kernels are instantiated for Q4/Q5 pairs only (as before).
    bool done = false;
    if constexpr ((Type == gufo::core::GgmlType::kQ4_K ||
                   Type == gufo::core::GgmlType::kQ5_K) &&
                  (UpType == gufo::core::GgmlType::kQ4_K ||
                   UpType == gufo::core::GgmlType::kQ5_K)) {
      if (m % 128 == 0 && batch % 256 == 0) {
        tile.template operator()<true>();
        done = true;
      }
    }
    if (!done) tile.template operator()<false>();
#if defined(GUFO_ENABLE_NPU)
    if (m_grid != m) npu::FinishGateUp(y, batch, m, m_grid, stream, npu_seq);
#endif
  };
  using gufo::core::GgmlType;
  // Mixed pairs keep independent row strides and decoders. Qualify each
  // pairing: a shared input alone does not guarantee a faster kernel.
  if (gate_type != up_type) {
    if (gate_type == GgmlType::kQ4_K && up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kQ4_K, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kIQ4_XS &&
               up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kIQ4_XS, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kQ4_K &&
               up_type == GgmlType::kIQ4_XS) {
      launch.template operator()<GgmlType::kQ4_K, GgmlType::kIQ4_XS>();
    } else if (gate_type == GgmlType::kIQ4_XS &&
               up_type == GgmlType::kQ4_K) {
      launch.template operator()<GgmlType::kIQ4_XS, GgmlType::kQ4_K>();
    } else if (gate_type == GgmlType::kQ6_K &&
               up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kQ6_K, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kIQ4_XS &&
               up_type == GgmlType::kQ3_K) {
      launch.template operator()<GgmlType::kIQ4_XS, GgmlType::kQ3_K>();
    } else if (gate_type == GgmlType::kQ3_K &&
               up_type == GgmlType::kIQ4_XS) {
      launch.template operator()<GgmlType::kQ3_K, GgmlType::kIQ4_XS>();
    } else if (gate_type == GgmlType::kIQ4_NL &&
               up_type == GgmlType::kIQ4_XS) {
      launch.template operator()<GgmlType::kIQ4_NL, GgmlType::kIQ4_XS>();
    } else if (gate_type == GgmlType::kIQ4_NL &&
               up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kIQ4_NL, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kQ5_K &&
               up_type == GgmlType::kIQ4_NL) {
      launch.template operator()<GgmlType::kQ5_K, GgmlType::kIQ4_NL>();
    } else {
      return false;
    }
    return true;
  }
  switch (gate_type) {
    case GgmlType::kQ4_K:
      launch.template operator()<GgmlType::kQ4_K>();
      return true;
    case GgmlType::kQ5_K:
      launch.template operator()<GgmlType::kQ5_K>();
      return true;
    case GgmlType::kQ6_K:
      launch.template operator()<GgmlType::kQ6_K>();
      return true;
    case GgmlType::kIQ4_XS:
      launch.template operator()<GgmlType::kIQ4_XS>();
      return true;
    case GgmlType::kIQ3_S:
      if (!Iq3DualGateUp()) return false;
      launch.template operator()<GgmlType::kIQ3_S>();
      return true;
    case GgmlType::kIQ3_XXS:
      if (!Iq3DualGateUp()) return false;
      launch.template operator()<GgmlType::kIQ3_XXS>();
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::hip
#endif

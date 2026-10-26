// dq5.hip -- real Q5_K batch-8 GEMV, decoupled + persistent + R matrices per launch.
//
// RESULT: THIS DESIGN FAILED. Measured on the test machine (pinned clock, quiet machine, validated output):
//   R=4 matrices per launch: best decoupled arm 171 GB/s against 206 GB/s for the coupled baseline kernel;
//   R=1: 140 against 200 GB/s. The decoupled kernel is correct (it passes the checks below) but slower than the
//   coupled kernel it was meant to beat; the extra LDS hop and the producer/consumer synchronisation cost more
//   than the overlap gains. It is published as a negative result and as a reference for the ring protocol.
//
// MEASURES: GB/s of the decoupled kernel against the coupled baseline on a Q5_K matrix (default 5120 x 6144,
//   batch 8), with correctness checked before any timing.
// DEPENDS ON: headers of the tubo/gufo source tree (src/models/qwen/hip/kernels/sn_q4k_k4.hpp and what it
//   includes); this file does not build standalone. Build inside that tree:
//   hipcc -O3 --offload-arch=gfx1151 -std=c++20 -I <tree root> -I <tree root>/src -o dq5 dq5.hip
// RUN:   dq5 [m=5120] [k=6144] [R=1] [host|dev]      (run.sh runs R=1 then R=4 and checks the kernel log)
//
// Arms:
//   a r2k8hp      baseline = the benchmark-best coupled Q5 B8 kernel, q5il_r2k8hp
//                 (SnQ4KK4Kernel<8,2,8,SK0,IL1,Q5,Hoist,Packed>, included VERBATIM from the tree),
//                 R back-to-back launches per round.
//   dq5 g P C S   decoupled: G WGs, each P producer waves (global -> LDS ring of S wave-blocks)
//                 + C consumer waves (LDS -> VGPR, free the slot, Q5 math). One consumer owns one
//                 32-row tile for all of K, then the next tile of its group (persistent over the
//                 R matrices). It walks K phase-major (b = p + 8q) and keeps a running phase total,
//                 so each y is summed in the baseline's fp32 order: ((s0+s1)+...+s7), s_p = f(p) +
//                 f(p+8) + f(p+16). Per-block math = sn_q4k_k4.hpp:596-617, 645-737, 740-759 copied.
//
// Correctness BEFORE timing, on every rotating copy: y vs a CPU double reference built from the
// native ggml blocks (independent decode, ggml get_scale_min_k4 + dequantize_row_q5_K bit layout)
// and the GPU-quantized q8_K activations: max|y-ref|/max|ref| <= 1e-3. Every decoupled arm also
// vs arm a: max|y-ya|/max|ya| <= 1e-5 (designed value-exact; the count of equal values is printed).
// Any failure, spin timeout or unwritten y (prefilled NaN) -> "FAIL" on stderr, exit 3.
//
// WRONG IF: the program exits 3 / prints FAIL (checksum mismatch, spin timeout or unwritten output: the
//   timing is void); the baseline arm a is below its rate in the repository GEMV benchmark at the same size
//   (the machine was loaded or the clock not pinned); or the decoupled arm's equal-count to arm a is far from
//   the number of outputs while relerr_a passes (the value-exact design claim is broken).
#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/qwen/hip/kernels/sn_q4k_k4.hpp"  // baseline, quantizer, pack-sums, Q4KScaleMin


#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2);} } while (0)

namespace {
using gufo::hip::SnQ4KK4Kernel;
constexpr std::uint32_t kB = 8;         // batch (token columns)
constexpr std::uint32_t kKPO = 8;       // K-phase order of the baseline (r2k8: KPhases = 8)
constexpr std::uint32_t kCh = 11;       // 16-B chunks per lane per Q5 wave-block (176 B per row)
constexpr std::uint32_t kWB = kCh * 512;  // 5632 B per wave-block (32 rows x 1 K block)
constexpr unsigned kSpin = 200000;      // bounded wait, as in ../decoupling/dcp.hip (~13 ms)

// One Q5_K K block (256 weights) for one row per lane, 8 tokens: the baseline's body for
// <Q5, Hoist, PackedSums, Probe 0, Batch 8>, sn_q4k_k4.hpp:596-617 (head), 645-737 (dot4),
// 740-759 (min term), with the 11 chunks coming from LDS instead of global memory.
__device__ __forceinline__ void Q5BlockB8(const uint4 (&v)[kCh], const std::int8_t* act_block,
                                          const std::int32_t* rec, std::uint32_t k,
                                          float (&sums)[kB]) {
  constexpr std::size_t Batch = kB;
  const uint4 head = v[0];
  const uint4 qh0 = v[1];
  const uint4 qh1 = v[2];
  const std::uint32_t qhw[8] = {qh0.x, qh0.y, qh0.z, qh0.w, qh1.x, qh1.y, qh1.z, qh1.w};
  const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(head.x & 0xFFFFU)));
  const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(head.x >> 16U)));
  int sc[8];
  int mn[8];
#pragma unroll
  for (std::uint32_t j = 0; j < 8U; ++j) {
    std::uint32_t s = 0;
    std::uint32_t m8 = 0;
    gufo::hip::sn_q4k_k4_detail::Q4KScaleMin(head, j, &s, &m8);
    sc[j] = static_cast<int>(s);
    mn[j] = static_cast<int>(m8);
  }
  int isum[Batch] = {};
#pragma unroll
  for (std::uint32_t gp = 0; gp < 4U; ++gp) {
    const uint4 c0 = v[3U + 2U * gp];
    const uint4 c1 = v[4U + 2U * gp];
    const std::uint32_t gq[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    std::uint32_t lo[8];
    std::uint32_t hi[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8U; ++i) {
      lo[i] = gq[i] & 0x0F0F0F0FU;
      hi[i] = (gq[i] >> 4U) & 0x0F0F0F0FU;
      lo[i] |= ((qhw[i] >> (2U * gp)) & 0x01010101U) << 4U;
      hi[i] |= ((qhw[i] >> ((2U * gp) + 1U)) & 0x01010101U) << 4U;
    }
    const int sc_lo = sc[2U * gp];
    const int sc_hi = sc[(2U * gp) + 1U];
#pragma unroll
    for (std::size_t t0 = 0; t0 < Batch; t0 += 4U) {
      constexpr std::size_t kQuad = 4U;
      uint4 a[kQuad][4];
#pragma unroll
      for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
        const uint4* const p = reinterpret_cast<const uint4*>(act_block + ((t0 + j) * k) + (gp * 64U));
#pragma unroll
        for (std::uint32_t c = 0; c < 4U; ++c) a[j][c] = p[c];
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (std::size_t j = 0; j < kQuad && t0 + j < Batch; ++j) {
        const uint4 alo = a[j][0];
        const uint4 ami = a[j][1];
        const uint4 blo = a[j][2];
        const uint4 bmi = a[j][3];
        int dot_lo = 0;
        int dot_hi = 0;
        dot_lo = __builtin_amdgcn_sudot4(true, lo[0], true, alo.x, dot_lo, false);
        dot_lo = __builtin_amdgcn_sudot4(true, lo[1], true, alo.y, dot_lo, false);
        dot_lo = __builtin_amdgcn_sudot4(true, lo[2], true, alo.z, dot_lo, false);
        dot_lo = __builtin_amdgcn_sudot4(true, lo[3], true, alo.w, dot_lo, false);
        dot_lo = __builtin_amdgcn_sudot4(true, lo[4], true, ami.x, dot_lo, false);
        dot_lo = __builtin_amdgcn_sudot4(true, lo[5], true, ami.y, dot_lo, false);
        dot_lo = __builtin_amdgcn_sudot4(true, lo[6], true, ami.z, dot_lo, false);
        dot_lo = __builtin_amdgcn_sudot4(true, lo[7], true, ami.w, dot_lo, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[0], true, blo.x, dot_hi, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[1], true, blo.y, dot_hi, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[2], true, blo.z, dot_hi, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[3], true, blo.w, dot_hi, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[4], true, bmi.x, dot_hi, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[5], true, bmi.y, dot_hi, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[6], true, bmi.z, dot_hi, false);
        dot_hi = __builtin_amdgcn_sudot4(true, hi[7], true, bmi.w, dot_hi, false);
        isum[t0 + j] += __mul24(sc_lo, dot_lo) + __mul24(sc_hi, dot_hi);
        // Not in the baseline: without this pin LLVM sinks all 512 dot4 below every quad
        // barrier here (IR checked), so the block's 2 KB of activations stay live -> 870
        // SGPR lane spills per block. Integer value unchanged.
        asm volatile("" : "+v"(isum[t0 + j]));
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  std::int32_t pk[5U * Batch];
#pragma unroll
  for (std::uint32_t i = 0; i < 5U * Batch; ++i) pk[i] = rec[i];
#pragma unroll
  for (std::size_t t = 0; t < Batch; ++t) {
    int imin = 0;
#pragma unroll
    for (std::uint32_t q = 0; q < 4U; ++q) {
      const std::int32_t v2 = pk[(4U * t) + q];
      imin += __mul24(mn[2U * q], static_cast<int>(static_cast<short>(v2)));
      imin += __mul24(mn[(2U * q) + 1U], v2 >> 16);
    }
    const float xd = __int_as_float(pk[(4U * Batch) + t]);
    sums[t] += xd * ((d * static_cast<float>(isum[t])) - (dmin * static_cast<float>(imin)));
  }
}

// Ring item i of a WG = (group q, K phase p, step qq, consumer c), c fastest:
//   i = ((q*kKPO + p)*nq + qq)*C + c,  K block b = p + kKPO*qq,  tile u = (q*G + g)*C + c.
// The R matrices are contiguous: weights w + r*mat_bytes, xq + r*8k, xsum + r*blocks*40, y + r*8m.
// seq[s]: 2r = empty for ring round r, 2r+1 = full (as in ../decoupling/dcp.hip).
template <std::uint32_t P, std::uint32_t C, std::uint32_t S>
__launch_bounds__((P + C) * 32U) __global__ void Dq5Kernel(
    const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
    const std::int32_t* __restrict__ xsum, float* __restrict__ y, std::uint32_t m, std::uint32_t k,
    std::uint32_t ngroups, std::uint32_t* __restrict__ err) {
  static_assert(S * kWB + 4U * (S + 1U) <= 65536U, "LDS ring > 64 KiB per WG");
  __shared__ uint4 ring[S][kCh][32];
  __shared__ unsigned seq[S], abort_;
  if (threadIdx.x < S) seq[threadIdx.x] = 0;
  if (threadIdx.x == 0) abort_ = 0;
  __syncthreads();
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wv = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5U);
  const std::uint32_t blocks = k >> 8U, nq = blocks / kKPO, tpm = m >> 5U, g = blockIdx.x, G = gridDim.x;
  const std::size_t mat_bytes = static_cast<std::size_t>(tpm) * blocks * kWB;
  const std::uint32_t ni = ngroups * blocks * C;
  const auto wait_for = [&](unsigned s, unsigned want) {  // same wait loop as ../decoupling/dcp.hip
    for (unsigned n = 0; __hip_atomic_load(&seq[s], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) != want; ++n) {
      if (__hip_atomic_load(&abort_, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)) return false;
      if (n > kSpin) {
        if (lane == 0) { atomicOr(err, 1u); __hip_atomic_store(&abort_, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); }
        return false;
      }
      __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    return true;
  };
  const auto split_tile = [&](std::uint32_t u, std::uint32_t* r) {  // tile u -> (matrix r, tile in r)
    *r = 0;
    while (u >= tpm) { u -= tpm; ++*r; }
    return u;
  };
  if (wv < P) {  // producer: global -> LDS, items wv, wv+P, ...
    std::uint32_t c = 0, qq = 0, p = 0, q = 0;
    const auto advance = [&](std::uint32_t by) {  // (q, p, qq, c) += by items, c fastest
      c += by;
      while (c >= C) {
        c -= C;
        if (++qq == nq) { qq = 0; if (++p == kKPO) { p = 0; ++q; } }
      }
    };
    advance(wv);
    for (std::uint32_t i = wv; i < ni; i += P, advance(P)) {
      const unsigned s = i % S, rd = i / S;
      if (!wait_for(s, 2 * rd)) break;
      std::uint32_t r;
      const std::uint32_t rt = split_tile((q * G + g) * C + c, &r);
      const std::uint8_t* const src = w + r * mat_bytes +
                                      (static_cast<std::size_t>(rt) * blocks + (p + kKPO * qq)) * kWB + lane * 16U;
      uint4 v[kCh];
#pragma unroll
      for (std::uint32_t ch = 0; ch < kCh; ++ch) v[ch] = *reinterpret_cast<const uint4*>(src + ch * 512U);
#pragma unroll
      for (std::uint32_t ch = 0; ch < kCh; ++ch) ring[s][ch][lane] = v[ch];
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // ring stores before "full"
      if (lane == 0) __hip_atomic_store(&seq[s], 2 * rd + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    return;
  }
  const std::uint32_t c = wv - P;  // consumer c: tile (q*G+g)*C + c of every group q
  for (std::uint32_t q = 0; q < ngroups; ++q) {
    std::uint32_t r;
    const std::uint32_t rt = split_tile((q * G + g) * C + c, &r);
    const std::int8_t* const xq_r = xq + static_cast<std::size_t>(r) * kB * k;
    const std::int32_t* const xs_r = xsum + static_cast<std::size_t>(r) * blocks * 5U * kB;
    float tot[kB];
    for (std::uint32_t p = 0; p < kKPO; ++p) {
      float cur[kB] = {};  // = the baseline's per-phase `sums` (sn_q4k_k4.hpp:321)
      for (std::uint32_t qq = 0; qq < nq; ++qq) {
        const std::uint32_t i = ((q * kKPO + p) * nq + qq) * C + c;
        const unsigned s = i % S, rd = i / S;
        if (!wait_for(s, 2 * rd + 1)) return;  // y left unwritten: host sees err=1 / NaN
        uint4 v[kCh];
#pragma unroll
        for (std::uint32_t ch = 0; ch < kCh; ++ch) v[ch] = ring[s][ch][lane];
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // ring reads done before "empty"
        if (lane == 0) __hip_atomic_store(&seq[s], 2 * rd + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        const std::uint32_t b = p + kKPO * qq;
        Q5BlockB8(v, xq_r + static_cast<std::size_t>(b) * 256U, xs_r + static_cast<std::size_t>(b) * 5U * kB, k, cur);
      }
      // Baseline: phase 0's sums, then += partial[p] for p = 1..7 (sn_q4k_k4.hpp:797-803).
#pragma unroll
      for (std::uint32_t t = 0; t < kB; ++t) tot[t] = p == 0 ? cur[t] : tot[t] + cur[t];
    }
    float* const y_r = y + static_cast<std::size_t>(r) * kB * m;
    const std::uint32_t row = rt * 32U + lane;
#pragma unroll
    for (std::uint32_t t = 0; t < kB; ++t) y_r[t * m + row] = tot[t];
  }
}

struct Args {
  const std::uint8_t* w;
  const std::int8_t* xq;
  const float2* xds;
  const std::int32_t* xsum;
  float* y;
  std::uint32_t m, k, R;
  std::uint32_t* err;
};

void BaseR2k8hp(const Args& a) {  // q5il_r2k8hp, R launches (from the repository GEMV benchmark)
  const std::uint32_t blocks = a.k / 256U;
  const std::size_t mb = static_cast<std::size_t>(a.m / 32U) * blocks * kWB;
  for (std::uint32_t r = 0; r < a.R; ++r)
    SnQ4KK4Kernel<kB, 2, 8, false, true, true, true, true, 0, false, false, false><<<(a.m + 63U) / 64U, 512>>>(
        a.w + r * mb, a.xq + static_cast<std::size_t>(r) * kB * a.k, a.xds + static_cast<std::size_t>(r) * kB * (a.k / 16U),
        a.y + static_cast<std::size_t>(r) * kB * a.m, a.m, a.k, nullptr, nullptr,
        a.xsum + static_cast<std::size_t>(r) * blocks * 5U * kB);
}

template <std::uint32_t G, std::uint32_t P, std::uint32_t C, std::uint32_t S>
void Dq5(const Args& a) {
  const std::uint32_t ngroups = a.R * (a.m / 32U) / (G * C);  // host checked divisibility
  Dq5Kernel<P, C, S><<<G, (P + C) * 32U>>>(a.w, a.xq, a.xsum, a.y, a.m, a.k, ngroups, a.err);
}

struct Arm { const char* name; void (*go)(const Args&); std::uint32_t tiles_per_round; };

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

std::uint32_t xs32(std::uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

float half_bits(const std::uint8_t* p) { _Float16 h; std::memcpy(&h, p, 2); return static_cast<float>(h); }

// ggml get_scale_min_k4, byte form (independent of the kernel's uint4 form).
void scale_min_k4(int j, const std::uint8_t* q, int* sc, int* mn) {
  if (j < 4) { *sc = q[j] & 63; *mn = q[j + 4] & 63; }
  else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
}  // namespace

int main(int argc, char** argv) {
  const std::uint32_t m = argc > 1 ? std::atoi(argv[1]) : 5120, k = argc > 2 ? std::atoi(argv[2]) : 6144,
                      R = argc > 3 ? std::atoi(argv[3]) : 1;
  const bool host = argc < 5 || !std::strcmp(argv[4], "host");
  const std::uint32_t blocks = k / 256U, tpm = m / 32U, nsub = k / 16U;
  if (m == 0 || m % 64U || k % 256U || blocks % kKPO || R == 0 || R > 64) {
    std::fprintf(stderr, "need m %% 64 == 0, k %% 2048 == 0, 1 <= R <= 64\n");
    return 1;
  }
  const std::size_t mat_bytes = static_cast<std::size_t>(tpm) * blocks * kWB, bytes = mat_bytes * R;
  const std::size_t copies = std::max<std::size_t>(1, (128ULL << 20) / bytes + ((128ULL << 20) % bytes != 0));
  const std::size_t phys = bytes * copies;

  // Native ggml block_q5_K {half d, dmin; u8 scales[12], qh[32], qs[128]}, random but valid.
  std::uint32_t s = 0x12345678u;
  std::vector<std::uint8_t> nat(bytes);
  for (auto& b : nat) b = static_cast<std::uint8_t>(xs32(s) >> 24);
  for (std::size_t bi = 0; bi < bytes / 176U; ++bi) {
    const _Float16 d = static_cast<_Float16>((0.5f + (xs32(s) >> 8) * (1.0f / 16777216.0f)) * 1e-3f);
    const _Float16 dm = static_cast<_Float16>((0.5f + (xs32(s) >> 8) * (1.0f / 16777216.0f)) * 2e-4f);
    std::memcpy(&nat[bi * 176U], &d, 2);
    std::memcpy(&nat[bi * 176U + 2U], &dm, 2);
  }
  std::vector<std::uint8_t> il(phys);
  for (std::uint32_t r = 0; r < R; ++r)
    gufo::models::InterleaveKQuantRows32(&nat[r * mat_bytes], &il[r * mat_bytes], m, k, 176U);
  for (std::size_t c = 1; c < copies; ++c) std::memcpy(&il[c * bytes], il.data(), bytes);

  std::uint8_t* dev = nullptr;
  CK(hipMalloc(&dev, phys));
  CK(hipMemcpy(dev, il.data(), phys, hipMemcpyHostToDevice));
  const std::uint8_t* base = dev;
  if (host) {  // as ../decoupling/dcp.hip: host-mapped read-only hugepage memory, as the served weights
    void* hp = mmap(nullptr, phys, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (hp == MAP_FAILED) return 1;
    madvise(hp, phys, MADV_HUGEPAGE);
    std::memcpy(hp, il.data(), phys);
    mprotect(hp, phys, PROT_READ);
    CK(hipHostRegister(hp, phys, hipHostRegisterMapped | hipHostRegisterReadOnly));
    void* dp = nullptr;
    CK(hipHostGetDevicePointer(&dp, hp, 0));
    base = static_cast<const std::uint8_t*>(dp);
    CK(hipFree(dev));
  }
  il.clear(); il.shrink_to_fit();

  // Activations: R matrices x 8 tokens, quantized once by the served q8_K quantizer + packed sums.
  const std::size_t nx = static_cast<std::size_t>(R) * kB * k, ny = static_cast<std::size_t>(R) * kB * m;
  std::vector<float> hx(nx);
  for (auto& v : hx) v = static_cast<float>(xs32(s) >> 8) * (2.0f / 16777216.0f) - 1.0f;
  float *x, *y;
  std::int8_t* xq;
  float2* xds;
  std::int32_t* xsum;
  std::uint32_t* err;
  CK(hipMalloc(&x, nx * 4)); CK(hipMalloc(&xq, nx)); CK(hipMalloc(&xds, nx / 16 * sizeof(float2)));
  CK(hipMalloc(&xsum, static_cast<std::size_t>(R) * blocks * 5U * kB * 4)); CK(hipMalloc(&y, ny * 4));
  CK(hipMalloc(&err, 4)); CK(hipMemset(err, 0, 4));
  CK(hipMemcpy(x, hx.data(), nx * 4, hipMemcpyHostToDevice));
  const std::uint32_t subs = static_cast<std::uint32_t>(nx / 16);
  gufo::hip::QuantizeActivationsQ8KDot4Kernel<<<(subs + 255U) / 256U, 256>>>(x, xq, xds, subs);
  for (std::uint32_t r = 0; r < R; ++r)
    gufo::hip::LaunchSnQ4KPackSums<kB>(xds + static_cast<std::size_t>(r) * kB * nsub,
                                       xsum + static_cast<std::size_t>(r) * blocks * 5U * kB, k, nullptr);
  CK(hipGetLastError()); CK(hipDeviceSynchronize());
  std::vector<std::int8_t> hq(nx);
  std::vector<float2> hds(nx / 16);
  CK(hipMemcpy(hq.data(), xq, nx, hipMemcpyDeviceToHost));
  CK(hipMemcpy(hds.data(), xds, hds.size() * sizeof(float2), hipMemcpyDeviceToHost));

  // CPU reference, double, exact integers per block: xd * (d*sum sc_j dot_j - dmin*sum mn_j A_j).
  std::vector<double> ref(ny);
  double ref_max = 0;
  for (std::uint32_t r = 0; r < R; ++r)
    for (std::uint32_t row = 0; row < m; ++row) {
      double acc[kB] = {};
      for (std::uint32_t b = 0; b < blocks; ++b) {
        const std::uint8_t* bp = &nat[((static_cast<std::size_t>(r) * m + row) * blocks + b) * 176U];
        const float d = half_bits(bp), dmin = half_bits(bp + 2);
        const std::uint8_t *scl = bp + 4, *qh = bp + 16, *qs = bp + 48;
        int q[256];
        for (int e = 0; e < 256; ++e) {  // ggml dequantize_row_q5_K bit layout
          const int gp = e / 64, l = e % 64;
          q[e] = l < 32 ? (qs[gp * 32 + l] & 0xF) | (((qh[l] >> (2 * gp)) & 1) << 4)
                        : (qs[gp * 32 + l - 32] >> 4) | (((qh[l - 32] >> (2 * gp + 1)) & 1) << 4);
        }
        int sc[8], mn[8];
        for (int j = 0; j < 8; ++j) scale_min_k4(j, scl, &sc[j], &mn[j]);
        for (std::uint32_t t = 0; t < kB; ++t) {
          const std::size_t tok = static_cast<std::size_t>(r) * kB + t;
          const std::int8_t* a = &hq[tok * k + b * 256U];
          long long isum = 0, imin = 0;
          for (int j = 0; j < 8; ++j) {
            long long dot = 0, asum = 0;
            for (int e = 32 * j; e < 32 * j + 32; ++e) { dot += q[e] * a[e]; asum += a[e]; }
            isum += sc[j] * dot;
            imin += mn[j] * asum;
          }
          acc[t] += static_cast<double>(hds[tok * nsub + b * 16U].x) *
                    (static_cast<double>(d) * isum - static_cast<double>(dmin) * imin);
        }
      }
      for (std::uint32_t t = 0; t < kB; ++t) {
        ref[(static_cast<std::size_t>(r) * kB + t) * m + row] = acc[t];
        ref_max = std::max(ref_max, std::fabs(acc[t]));
      }
    }
  nat.clear(); nat.shrink_to_fit();

  std::vector<Arm> arms;
  arms.push_back({"a r2k8hp", BaseR2k8hp, 1});
#define DQ5(G, P, C, S) arms.push_back({"dq5 g" #G " p" #P " c" #C " s" #S, Dq5<G, P, C, S>, G * C});
  DQ5(40, 4, 4, 8) DQ5(80, 2, 2, 4) DQ5(20, 8, 8, 10)            // G*C = 160: R = 1 and R = 4
  DQ5(80, 4, 8, 4) DQ5(40, 4, 16, 8) DQ5(40, 8, 16, 8)           // G*C = 640: R = 4 only (dcp's best)

  std::printf("# dq5 m=%u k=%u R=%u bytes/launch=%zu copies=%zu mem=%s ref_max=%.4g\n", m, k, R, bytes, copies,
              host ? "host" : "dev", ref_max);
  std::printf("arm,ms,GBps,min_ms,relerr_ref,relerr_a,equal_a,err\n");
  std::vector<float> hy(ny), ya(ny);
  Args a{nullptr, xq, xds, xsum, y, m, k, R, err};
  for (auto& arm : arms) {
    if ((R * tpm) % arm.tiles_per_round) { std::printf("# skip %s: R*m/32 %% (G*C) != 0\n", arm.name); continue; }
    const bool is_a = arm.go == BaseR2k8hp;
    double e_ref = 0, e_a = 0;
    std::size_t equal = 0;
    unsigned e = 0;
    for (std::size_t c = 0; c < copies; ++c) {  // validation: every copy, synced, before any timing
      CK(hipMemset(y, 0xFF, ny * 4));            // NaN: an unwritten y fails below
      a.w = base + c * bytes;
      arm.go(a);
      CK(hipGetLastError()); CK(hipDeviceSynchronize());
      CK(hipMemcpy(hy.data(), y, ny * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost));
      double dr = 0, da = 0, ya_max = 0;
      std::size_t eq = 0;
      for (std::size_t i = 0; i < ny; ++i) {
        const double v = hy[i];
        dr = std::isfinite(v) ? std::max(dr, std::fabs(v - ref[i])) : INFINITY;
        if (!is_a) { da = std::max(da, std::fabs(v - ya[i])); ya_max = std::max(ya_max, std::fabs((double)ya[i])); eq += hy[i] == ya[i]; }
      }
      e_ref = std::max(e_ref, dr / ref_max);
      if (!is_a) { e_a = std::max(e_a, da / ya_max); equal = c ? std::min(equal, eq) : eq; }
      if (is_a && c == 0) ya = hy;
      if (e || !(e_ref <= 1e-3) || (!is_a && !(e_a <= 1e-5))) {
        std::fprintf(stderr, "FAIL %s copy %zu: err=%u relerr_ref=%.3g relerr_a=%.3g (err=1: spin timeout)\n",
                     arm.name, c, e, e_ref, e_a);
        return 3;
      }
    }
    std::vector<double> ms;
    hipEvent_t t0, t1;
    CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    for (int pass = 0; pass < 3; ++pass)
      for (int rep = 0; rep < 3; ++rep) {
        CK(hipEventRecord(t0));
        for (int i = 0; i < 24; ++i) { a.w = base + (i % copies) * bytes; arm.go(a); }
        CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
        float t;
        CK(hipEventElapsedTime(&t, t0, t1));
        ms.push_back(t / 24.0);
      }
    CK(hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost));
    const double med = median(ms);
    std::printf("%s,%.5f,%.2f,%.5f,%.3g,%.3g,%zu/%zu,%u\n", arm.name, med, bytes / (med * 1e6),
                *std::min_element(ms.begin(), ms.end()), e_ref, e_a, is_a ? ny : equal, ny, e);
    if (e) { std::fprintf(stderr, "FAIL %s: spin timeout in the timed launches (err=%u)\n", arm.name, e); return 3; }
  }
  CK(hipGetLastError());
  return 0;
}

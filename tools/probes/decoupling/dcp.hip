// dcp.hip -- load/compute coupling probe ("dcp" = decoupled).
//
// MEASURES: is the loss of a batch-8 weight kernel relative to a pure read caused by COUPLING (the wave that
//   loads a block also computes it, so its next loads wait for its arithmetic), and does a DECOUPLED design
//   (producer waves stream weights into an LDS ring, consumer waves only compute) remove it? Also: what does one
//   launch over R matrices gain over R launches (a persistent multi-matrix kernel), with the same arithmetic?
//   No real weight math: the arithmetic is N64 x 64 dependency-free v_mad_u32_u24 per block (a fixed amount of
//   VALU work). A batch-8 Q5_K block is about 1300-1500 VALU slots per wave-block (512 dot4 x 1.63 + 466 other),
//   so N64 = 20-24 stands in for it. N64 = 0 is the pure read.
//   Memory and timing follow ../ceiling/ceil.hip (the repository's GEMV benchmark): host-mapped read-only
//   hugepage mmap, copies = ceil(128 MiB / launch bytes), 24 launches per round.
//
// BUILD: hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o dcp dcp.hip
// RUN:   dcp <bytes_per_matrix> <R> <N64> [host|dev]     env CH=6|7|9|11|13 (default 11)
//   CH = 16-byte chunks per lane per block: 11 = Q5_K (5632 B/wave-block), 9 = Q4_K,
//   13 = Q6_K (~210 B/row), 7 = IQ3_S, 6 = IQ3_XXS (96 of its 98 B).
//   probes.sh runs the planned series; budget.py turns the results into a loss budget.
// Arms (all persistent, static block interleave):
//   cpl g<G> w<W>          coupled: wave loads CH b128 (all in flight), waits, xor, N VALU, next block
//   cpl ... shp1           same, per-lane-contiguous records (lane stride CH*16 B, the IQ3 per-row
//                          record shape) instead of one contiguous 512 B run per instruction
//   dcp g<G> p<P> c<C> s<S> decoupled: P producer waves global->LDS ring of S blocks;
//                          C consumer waves LDS->VGPR, free the slot, then N VALU
// R matrices per launch: R=1 vs R=4 at the same total bytes per round tells how much
// of the gap is per-launch (ramp/tail/gap) and how much is steady state.
//
// READ IT AS (decide BEFORE building any real decoupled kernel):
//   1. cpl N64=22 vs cpl N64=0 reproduces the slowdown of the real coupled kernel (about 224 -> 212 GB/s at
//      21.6 MB)? If not, the probe does not model the coupling; stop.
//   2. dcp N64=22 >= cpl N64=0 - 1%  -> decoupling removes the batch-8 loss: worth building.
//      dcp N64=0  <  cpl N64=0 - 3%  -> the LDS hop costs more than it saves: do not build.
//   3. R=4 one launch vs R=1: the per-launch share. If dcp R=4 N64=22 < 241 GB/s at
//      21.6 MB x 4, 241 GB/s (95% of 253.4) is out of reach for this design family at 1700 MHz.
//   4. shp1 vs shp0 at CH=6/7, N64=0: the IQ3 load-shape cost at batch 1.
//
// WRONG IF: the program exits 3 or prints FAIL (a spin timeout or a checksum mismatch: the ring protocol is
//   broken and the timing is meaningless); the coupled arm at N64=0 is below the ceil.hip rate for the same
//   size (the probe's own load loop is slow, so every comparison is biased); or two runs of one arm differ by
//   more than the effect being judged (the machine was not quiet).
//
// Hazards: producers + consumers of a ring are waves of ONE workgroup (co-resident by dispatch);
// item i waits only on items < i (DAG, no cycle). Every spin is bounded (kSpin ~13 ms @1.7 GHz):
// timeout -> err=1 + LDS abort flag -> every wave of the WG leaves its loop at its next wait.
// Host validates every copy, syncs, and exits non-zero on err or checksum mismatch BEFORE timing.
// Polls are relaxed (an acquire poll emits buffer_gl0_inv per iteration); one LDS-only acquire
// fence after the loop, LDS-only release fence before each seq store (ISA: s_waitcnt lgkmcnt(0)).
// LDS S*CH*512 B <= 60 KiB per WG; no global atomics in the hot loop.
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2);} } while (0)

constexpr unsigned kSpin = 200000;  // bounded wait: a broken protocol runs slow + err, never hangs

// N64 x 64 independent VALU on 8 accumulators seeded from the data (cannot be hoisted).
__device__ __forceinline__ unsigned valu(unsigned seed, unsigned n64) {
  unsigned acc[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) acc[c] = seed + c;
  for (unsigned it = 0; it < n64; ++it)
#pragma unroll
    for (unsigned r = 0; r < 8; ++r)
#pragma unroll
      for (int c = 0; c < 8; ++c) asm volatile("v_mad_u32_u24 %0, %0, 3, %1" : "+v"(acc[c]) : "v"(r));
  unsigned z = 0;
#pragma unroll
  for (int c = 0; c < 8; ++c) z ^= acc[c];
  return z;
}

// Byte offset of (block j, chunk c, lane l). shp0: chunk c of a block = 512 contiguous B
// (the 32-row interleave). shp1: lane l owns CH*16 contiguous B (SN IQ3 per-row record).
template <int CH, bool SHP1>
__device__ __forceinline__ size_t off(size_t j, unsigned c, unsigned l) {
  return j * (CH * 512) + (SHP1 ? (l * CH * 16 + c * 16) : (c * 512 + l * 16));
}

template <int CH, bool SHP1>
__global__ void cpl(const unsigned char* __restrict__ p, size_t nb, unsigned n64,
                    unsigned* chk, unsigned* sink) {
  const unsigned lane = threadIdx.x & 31, waves = blockDim.x >> 5;
  const size_t slot = (size_t)blockIdx.x * waves + (threadIdx.x >> 5), nslot = (size_t)gridDim.x * waves;
  unsigned x = 0, t = 0;
  for (size_t j = slot; j < nb; j += nslot) {
    uint4 v[CH];
#pragma unroll
    for (int c = 0; c < CH; ++c) v[c] = *reinterpret_cast<const uint4*>(p + off<CH, SHP1>(j, c, lane));
    unsigned y = 0;
#pragma unroll
    for (int c = 0; c < CH; ++c) y ^= v[c].x ^ v[c].y ^ v[c].z ^ v[c].w;
    x ^= y;
    t ^= valu(y, n64);
  }
  if (chk) atomicXor(chk, x);
  if (t == 0x9e3779b9u) sink[0] = t;
}

// Decoupled: P producer waves, C consumer waves, S-slot LDS ring; ring item i of this WG
// = global block blockIdx.x + i*gridDim.x. seq[s]: 2r = empty for round r, 2r+1 = full.
template <int CH, int S>
__global__ void dcp(const unsigned char* __restrict__ p, size_t nb, unsigned n64, unsigned P,
                    unsigned* chk, unsigned* sink, unsigned* err) {
  __shared__ uint4 ring[S][CH][32];
  __shared__ unsigned seq[S], abort_;
  if (threadIdx.x < S) seq[threadIdx.x] = 0;
  if (threadIdx.x == 0) abort_ = 0;
  __syncthreads();
  const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5, C = (blockDim.x >> 5) - P;
  const size_t g = gridDim.x, ni = nb > blockIdx.x ? (nb - blockIdx.x + g - 1) / g : 0;
  const auto wait_for = [&](unsigned s, unsigned want) {
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
  unsigned x = 0, t = 0;
  if (w < P) {  // producer: global -> LDS
    for (size_t i = w; i < ni; i += P) {
      const unsigned s = i % S, r = (unsigned)(i / S);
      if (!wait_for(s, 2 * r)) break;
      const size_t j = blockIdx.x + i * g;
      uint4 v[CH];
#pragma unroll
      for (int c = 0; c < CH; ++c) v[c] = *reinterpret_cast<const uint4*>(p + off<CH, false>(j, c, lane));
#pragma unroll
      for (int c = 0; c < CH; ++c) ring[s][c][lane] = v[c];
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // ring stores before "full"
      if (lane == 0) __hip_atomic_store(&seq[s], 2 * r + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
  } else {      // consumer: LDS -> VGPR, free the slot at once, then compute
    for (size_t i = w - P; i < ni; i += C) {
      const unsigned s = i % S, r = (unsigned)(i / S);
      if (!wait_for(s, 2 * r + 1)) break;
      uint4 v[CH];
#pragma unroll
      for (int c = 0; c < CH; ++c) v[c] = ring[s][c][lane];
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // ring reads done before "empty"
      if (lane == 0) __hip_atomic_store(&seq[s], 2 * r + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
      unsigned y = 0;
#pragma unroll
      for (int c = 0; c < CH; ++c) y ^= v[c].x ^ v[c].y ^ v[c].z ^ v[c].w;
      x ^= y;
      t ^= valu(y, n64);
    }
  }
  if (chk) atomicXor(chk, x);
  if (t == 0x9e3779b9u) sink[0] = t;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <int CH>
int run(size_t mat_bytes, unsigned R, unsigned n64, bool host) {
  constexpr size_t BLK = CH * 512;
  const size_t nb = (mat_bytes * R) / BLK, bytes = nb * BLK;  // read exactly nb whole blocks
  if (nb == 0) { std::fprintf(stderr, "bytes*R < one block (%zu B)\n", BLK); return 1; }
  const size_t copies = std::max<size_t>(1, (128ULL << 20) / bytes + ((128ULL << 20) % bytes != 0));
  const size_t phys = bytes * copies;
  std::vector<unsigned> h(phys / 4);
  unsigned s = 0x12345678u;
  for (auto& x : h) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; x = s; }
  std::vector<unsigned> expect(copies, 0);
  for (size_t i = 0; i < phys / 4; ++i) expect[i / (bytes / 4)] ^= h[i];
  unsigned char* dev = nullptr; CK(hipMalloc(&dev, phys));
  CK(hipMemcpy(dev, h.data(), phys, hipMemcpyHostToDevice));
  const unsigned char* base = dev;
  if (host) {
    void* hp = mmap(nullptr, phys, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (hp == MAP_FAILED) return 1;
    madvise(hp, phys, MADV_HUGEPAGE);
    CK(hipMemcpy(hp, dev, phys, hipMemcpyDeviceToHost));
    mprotect(hp, phys, PROT_READ);
    CK(hipHostRegister(hp, phys, hipHostRegisterMapped | hipHostRegisterReadOnly));
    void* d = nullptr; CK(hipHostGetDevicePointer(&d, hp, 0));
    base = (const unsigned char*)d; CK(hipFree(dev));
  }
  unsigned *chk, *sink, *err;
  CK(hipMalloc(&chk, 4)); CK(hipMalloc(&sink, 4)); CK(hipMalloc(&err, 4)); CK(hipMemset(err, 0, 4));
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  struct Arm { const char* name; void (*go)(const unsigned char*, size_t, unsigned, unsigned*, unsigned*, unsigned*); };
  std::vector<Arm> arms;
  // Launch lambdas cannot be captured into function pointers with state, so each arm is a
  // small static function; the grid/WG shapes are the ones the real kernels use.
#define CPL(G, W, S1, NM) arms.push_back({NM, \
    [](const unsigned char* q, size_t n, unsigned k, unsigned* c, unsigned* z, unsigned*) { \
      cpl<CH, S1><<<G, W * 32>>>(q, n, k, c, z); }});
#define DCP(G, P, C, S) arms.push_back({"dcp g" #G " p" #P " c" #C " s" #S, \
    [](const unsigned char* q, size_t n, unsigned k, unsigned* c, unsigned* z, unsigned* e) { \
      dcp<CH, S><<<G, (P + C) * 32>>>(q, n, k, P, c, z, e); }});
  CPL(80, 16, false, "cpl g80 w16") CPL(40, 32, false, "cpl g40 w32") CPL(160, 8, false, "cpl g160 w8")
  CPL(80, 16, true, "cpl g80 w16 shp1")
  if constexpr (CH * 512 * 8 <= 60 * 1024) { DCP(40, 4, 12, 8) DCP(40, 8, 16, 8) DCP(80, 4, 8, 4) }
  if constexpr (CH * 512 * 12 <= 60 * 1024) { DCP(40, 8, 16, 12) }
  std::printf("# CH=%d blk=%zu R=%u bytes/launch=%zu copies=%zu N64=%u mem=%s\n", CH, BLK, R, bytes,
              copies, n64, host ? "host" : "dev");
  std::printf("arm,ms,GBps,min_ms,ok,err\n");
  for (auto& arm : arms) {
    CK(hipMemset(err, 0, 4));
    unsigned got = 0, e = 0;
    for (size_t k = 0; k < copies; ++k) {  // validation: one launch per copy, synced, before any timing
      CK(hipMemset(chk, 0, 4));
      arm.go(base + k * bytes, nb, n64, chk, sink, err);
      CK(hipGetLastError()); CK(hipDeviceSynchronize());
      CK(hipMemcpy(&got, chk, 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost));
      if (e || got != expect[k]) {
        std::fprintf(stderr, "FAIL %s copy %zu: err=%u chk=%08x expect=%08x (err=1: spin timeout)\n", arm.name, k, e,
                     got, expect[k]);
        std::exit(3);
      }
    }
    std::vector<double> ms;
    for (int pass = 0; pass < 3; ++pass)
      for (int r = 0; r < 3; ++r) {
        CK(hipEventRecord(a));
        for (int i = 0; i < 24; ++i) arm.go(base + (i % copies) * bytes, nb, n64, nullptr, sink, err);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float t; CK(hipEventElapsedTime(&t, a, b)); ms.push_back(t / 24.0);
      }
    CK(hipMemcpy(&e, err, 4, hipMemcpyDeviceToHost));
    const double m = median(ms);
    std::printf("%s,%.5f,%.2f,%.5f,%d,%u\n", arm.name, m, bytes / (m * 1e6),
                *std::min_element(ms.begin(), ms.end()), 1, e);
    if (e) { std::fprintf(stderr, "FAIL %s: spin timeout in the timed launches (err=%u)\n", arm.name, e); std::exit(3); }
  }
  CK(hipGetLastError());
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: dcp <bytes_per_matrix> <R> <N64> [host|dev]  (env CH)\n"); return 1; }
  const size_t mb = std::strtoull(argv[1], nullptr, 10);
  const unsigned R = std::atoi(argv[2]), n64 = std::atoi(argv[3]);
  const bool host = argc < 5 || !std::strcmp(argv[4], "host");
  const int ch = std::getenv("CH") ? std::atoi(std::getenv("CH")) : 11;
  switch (ch) {
    case 6: return run<6>(mb, R, n64, host);
    case 7: return run<7>(mb, R, n64, host);
    case 9: return run<9>(mb, R, n64, host);
    case 11: return run<11>(mb, R, n64, host);
    case 13: return run<13>(mb, R, n64, host);
    default: std::fprintf(stderr, "CH must be 6|7|9|11|13\n"); return 1;
  }
}

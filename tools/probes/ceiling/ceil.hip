// ceil.hip -- DRAM ceiling probe: pure streaming read, 16-byte loads, no math.
//
// MEASURES: the best GB/s a read-only kernel reaches over a buffer of a given size, across launch shapes
//   (workgroup size, grid, unroll, tile vs grid-stride, non-temporal loads or not). This is the upper
//   bound any weight-streaming kernel of that size can hope for on this machine.
//   Memory and timing follow the repository's GEMV benchmark: host = anonymous mmap + MADV_HUGEPAGE,
//   filled by hipMemcpy D->H, mprotect read-only, hipHostRegister(Mapped|ReadOnly), device pointer via
//   hipHostGetDevicePointer. copies = ceil(128 MiB / bytes); launch i reads copy (i % copies), so the
//   data is not served from cache. Timing = one event pair around 24 back-to-back launches, ms/24;
//   3 rounds per pass, median reported.
//
// BUILD: hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o ceil ceil.hip
// RUN:   ./ceil <bytes> <host|dev> [copies(0=bench rule)] [wg(0=all)]    |    ./ceil empty
//        env REPS = passes over all configs (default 3). runall.sh runs the whole sweep; merge.py
//        summarises the logs.
//
// WRONG IF: a config prints ok=0 (its XOR checksum differs from the host value, so the kernel did not
//   read every byte exactly once); or the same config differs by more than a few GB/s between passes
//   (the machine was not quiet, so the ceiling is a floor); or a rate exceeds the theoretical DRAM peak
//   (bytes were not fetched from DRAM; some cache or copy reuse is in the way).
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2);} } while (0)

typedef unsigned v4u __attribute__((ext_vector_type(4)));
template<bool NT> __device__ __forceinline__ uint4 ld(const uint4* p) {
  if (NT) {
    v4u r = __builtin_nontemporal_load(reinterpret_cast<const v4u*>(p));
    return make_uint4(r.x, r.y, r.z, r.w);
  }
  return *p;
}

// n = number of uint4 in the buffer. Every load index < n (guarded). chk != null: validation XOR.
template<int UNR, bool TILE, bool NT>
__global__ void rd(const uint4* __restrict__ p, size_t n, size_t tile, unsigned* chk, unsigned* sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  if (!TILE) {
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    for (size_t base = gtid; base < n; base += nthr * UNR) {
      uint4 v[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const size_t i = base + (size_t)u * nthr;
        v[u] = i < n ? ld<NT>(p + i) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < UNR; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
    }
  } else {
    const size_t beg = (size_t)blockIdx.x * tile;
    const size_t end = beg + tile < n ? beg + tile : n;
    for (size_t base = beg + threadIdx.x; base < end; base += (size_t)blockDim.x * UNR) {
      uint4 v[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const size_t i = base + (size_t)u * blockDim.x;
        v[u] = i < end ? ld<NT>(p + i) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < UNR; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
    }
  }
  const unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (chk) atomicXor(chk, r);
  if (r == 0x9e3779b9u) sink[0] = r;  // keeps the loads alive
}
__global__ void empty_k() {}

using KFn = void (*)(const uint4*, size_t, size_t, unsigned*, unsigned*);
template<int U, bool T, bool N> KFn kf() { return rd<U, T, N>; }
KFn pick(int unr, bool tile, bool nt) {
#define P3(U) (tile ? (nt ? kf<U,true,true>() : kf<U,true,false>()) : (nt ? kf<U,false,true>() : kf<U,false,false>()))
  switch (unr) { case 1: return P3(1); case 2: return P3(2); case 4: return P3(4); default: return P3(8); }
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  if (argc >= 2 && !std::strcmp(argv[1], "empty")) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const int cfg[][2] = {{1, 64}, {40, 256}, {640, 256}, {4320, 256}, {20000, 256}};
    for (auto& c : cfg) {
      empty_k<<<c[0], c[1]>>>(); CK(hipDeviceSynchronize());
      std::vector<double> ms;
      for (int r = 0; r < 9; ++r) {
        CK(hipEventRecord(a));
        for (int i = 0; i < 24; ++i) empty_k<<<c[0], c[1]>>>();
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float t; CK(hipEventElapsedTime(&t, a, b)); ms.push_back(t / 24.0);
      }
      std::printf("empty,grid=%d,wg=%d,launch_to_launch_us_median=%.2f,min=%.2f\n", c[0], c[1],
                  median(ms) * 1000, *std::min_element(ms.begin(), ms.end()) * 1000);
    }
    return 0;
  }
  if (argc < 3) { std::fprintf(stderr, "usage: ceil <bytes> <host|dev> [copies] [wg]\n"); return 1; }
  const size_t bytes = std::strtoull(argv[1], nullptr, 10);
  const bool host = !std::strcmp(argv[2], "host");
  size_t copies = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : 0;
  const int wg_only = argc >= 5 ? std::atoi(argv[4]) : 0;
  if (bytes % 16) { std::fprintf(stderr, "bytes %% 16\n"); return 1; }
  if (!copies) copies = std::max<size_t>(1, (128ULL * 1024 * 1024 + bytes - 1) / bytes);
  const size_t phys = bytes * copies;

  // reference data: random bytes; per-copy XOR for validation
  std::vector<unsigned> h(phys / 4);
  unsigned s = 0x12345678u;
  for (auto& x : h) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; x = s; }
  std::vector<unsigned> expect(copies);
  for (size_t c = 0; c < copies; ++c) {
    unsigned x = 0;
    for (size_t i = 0; i < bytes / 4; ++i) x ^= h[c * bytes / 4 + i];
    expect[c] = x;
  }
  unsigned char* dev = nullptr;
  CK(hipMalloc(&dev, phys));
  CK(hipMemcpy(dev, h.data(), phys, hipMemcpyHostToDevice));
  const unsigned char* base = dev;
  void* hostp = nullptr;
  if (host) {
    hostp = mmap(nullptr, phys, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (hostp == MAP_FAILED) { std::fprintf(stderr, "mmap\n"); return 1; }
    madvise(hostp, phys, MADV_HUGEPAGE);
    CK(hipMemcpy(hostp, dev, phys, hipMemcpyDeviceToHost));
    if (mprotect(hostp, phys, PROT_READ)) { std::fprintf(stderr, "mprotect\n"); return 1; }
    CK(hipHostRegister(hostp, phys, hipHostRegisterMapped | hipHostRegisterReadOnly));
    void* d = nullptr; CK(hipHostGetDevicePointer(&d, hostp, 0));
    base = (const unsigned char*)d;
    CK(hipFree(dev)); dev = nullptr;  // like the bench's temp source: only the mapped copy is read
  }
  unsigned *chk, *sink;
  CK(hipMalloc(&chk, 4)); CK(hipMalloc(&sink, 1 << 20));
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  const size_t n = bytes / 16;
  const int reps = std::getenv("REPS") ? std::atoi(std::getenv("REPS")) : 3;
  std::printf("# bytes=%zu mem=%s copies=%zu phys=%zu\n", bytes, host ? "host" : "dev", copies, phys);
  std::printf("mem,bytes,copies,wg,grid,unr,style,nt,ms,GBps,min_ms,ok\n");
  struct Cfg { int wg, unr, tile, nt; size_t grid, tl; KFn k; int ok; std::vector<double> ms; };
  std::vector<Cfg> cfgs;
  const int wgs[] = {256, 512, 1024}, unrs[] = {1, 2, 4, 8}, grids[] = {40, 80, 160, 320, 640, 0};
  for (int wg : wgs) {
    if (wg_only && wg != wg_only) continue;
    for (int unr : unrs) for (int tile = 0; tile < 2; ++tile) for (int nt = 0; nt < 2; ++nt) {
      for (int g : grids) {
        const size_t cover = (n + (size_t)wg * unr - 1) / ((size_t)wg * unr);
        const size_t grid = g ? (size_t)g : cover;
        if (g && grid > cover) continue;  // more WGs than needed adds nothing
        Cfg c{wg, unr, tile, nt, grid, 0, pick(unr, tile, nt), 1, {}};
        if (tile) { c.tl = (n + grid - 1) / grid; c.tl = (c.tl + 63) / 64 * 64; }
        // validation: copies 0 and 1 once each, XOR must match
        for (size_t cp = 0; cp < copies && cp < 2; ++cp) {
          CK(hipMemset(chk, 0, 4));
          c.k<<<(unsigned)grid, wg>>>((const uint4*)(base + cp * bytes), n, c.tl, chk, sink);
          unsigned got; CK(hipMemcpy(&got, chk, 4, hipMemcpyDeviceToHost));
          if (got != expect[cp]) c.ok = 0;
        }
        cfgs.push_back(c);
      }
    }
  }
  // REPS passes over all configs: slow drift of the machine hits every config alike. Each pass = 3 rounds.
  for (int rep = 0; rep < reps; ++rep) {
    for (auto& c : cfgs) {
      c.k<<<(unsigned)c.grid, c.wg>>>((const uint4*)base, n, c.tl, nullptr, sink);
      for (int r = 0; r < 3; ++r) {
        CK(hipEventRecord(a));
        for (int i = 0; i < 24; ++i)
          c.k<<<(unsigned)c.grid, c.wg>>>((const uint4*)(base + (i % copies) * bytes), n, c.tl, nullptr, sink);
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float t; CK(hipEventElapsedTime(&t, a, b)); c.ms.push_back(t / 24.0);
      }
    }
  }
  CK(hipGetLastError());
  double best = 0; std::string bestcfg;
  for (auto& c : cfgs) {
    const double m = median(c.ms), mn = *std::min_element(c.ms.begin(), c.ms.end());
    const double gbps = bytes / (m * 1e6);
    std::printf("%s,%zu,%zu,%d,%zu,%d,%s,%d,%.5f,%.2f,%.5f,%d\n", host ? "host" : "dev", bytes, copies, c.wg,
                c.grid, c.unr, c.tile ? "tile" : "gstride", c.nt, m, gbps, mn, c.ok);
    if (c.ok && gbps > best) { best = gbps; bestcfg = "wg=" + std::to_string(c.wg) + " grid=" + std::to_string(c.grid) + " unr=" + std::to_string(c.unr) + (c.tile ? " tile" : " gstride") + (c.nt ? " nt" : ""); }
  }
  std::printf("# BEST %s bytes=%zu %.2f GB/s %s\n", host ? "host" : "dev", bytes, best, bestcfg.c_str());
  return 0;
}

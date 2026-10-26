// magic.hip -- launch-overlap probe. Can a pure streaming read of a ~21.6 MB host-mapped buffer reach
// >= 240 GB/s median when consecutive launches are allowed to overlap?
//
// MEASURES: GB/s of 24 back-to-back launches, timed exactly like the repository's GEMV benchmark (one event
//   pair around the 24 launches, divided by 24; copies = ceil(128 MiB / bytes); launch i reads copy i % copies).
//   Memory setup is the same as ../ceiling/ceil.hip. Launch styles ("arms"):
//     plain launches (barrier between them), independent launches with hipExtAnyOrderLaunch (no barrier bit),
//     and dependent windows: launch L starts early, issues its weight loads, then spin-waits until launch L-1
//     has finished and left a token (a stand-in for a matmul that needs the previous output). Order is
//     enforced in-kernel; wrong-order and spin-timeout counters are printed at the end.
//
// BUILD: hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o magic magic.hip      (uses hip/hip_ext.h)
// RUN:
//   magic trace <bytes> <host|dev> <arm>              per-wave start/end timestamps for 24 launches x 3 rounds
//   magic ab    <bytes> <host|dev> <arm,arm,...>      interleaved A/B: REPS reps, each rep = every arm x 3 rounds
//   magic pwr   <bytes> <host|dev> <arm,arm,...>      GB/s and watts per arm (needs GPU_SYSFS_DIR and GPU_POWER_FILE)
//   env: REPS (default 9); METRICS=1 samples gpu_metrics (fclk/uclk/dram counters; needs GPU_SYSFS_DIR);
//        PWR_SEC (default 3); CHECK=1 verifies that no launch was still running when the end event completed;
//        SYNC=1 synchronises every round; TRACE_ROUNDS, TRACE_COMPACT for the trace mode.
//        GPU_SYSFS_DIR = amdgpu device directory in sysfs (contains gpu_metrics)
//        GPU_POWER_FILE = hwmon power1_input file of that device
//   arms are named style_wg_grid_unr[_ao] (see parse_arm and pick below); "_ao" = hipExtAnyOrderLaunch
//   (no barrier bit); styles p, q, r, w are the dependent-launch styles.
//   runall.sh / runfinal.sh run the sweeps; merge.py and trsum.py summarise them.
//
// WRONG IF: an arm prints valid=0 (checksum mismatch); "pdl errors" shows wrong_order or spin_timeout > 0
//   (a dependent launch read a stale token or gave up waiting, so its speed is not a valid result); the
//   end-event check (CHECK=1) reports unfinished launches (the timer stopped before the work did);
//   or the plain arm differs between calls by more than the claimed gain (the machine was not quiet).
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <map>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2);} } while (0)

typedef unsigned long long u64;

// ---- kernels -------------------------------------------------------------------------------
// All kernels: p = n uint4 elements; every load index guarded < n (or < end <= n). chk: XOR validation.
// ts != null: lane 0 of each wave writes {start, end, smid} at ts[3*wave_global]. Stores only; no loops added.
// ctr (steal only): per-launch work counter, zeroed before each round.
#define WAVE_TS_BEGIN                                                                       \
  const unsigned wv = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                           \
  u64 t_beg = 0;                                                                               \
  if (TS) t_beg = (u64)wall_clock64();
#define WAVE_TS_END                                                                         \
  if (TS && (threadIdx.x & 31) == 0) {                                                         \
    ts[3 * (size_t)wv] = t_beg; ts[3 * (size_t)wv + 1] = (u64)wall_clock64(); ts[3 * (size_t)wv + 2] = __smid(); }

#define XOR4(a, v) { a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w; }

template <int UNR, bool TILE, bool TS>
__global__ void rd(const uint4* __restrict__ p, size_t n, size_t tile, unsigned* chk, unsigned* sink, u64* ts, unsigned* ctr) {
  WAVE_TS_BEGIN
  uint4 acc = make_uint4(0, 0, 0, 0);
  if (!TILE) {
    const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nthr = (size_t)gridDim.x * blockDim.x;
    for (size_t base = gtid; base < n; base += nthr * UNR) {
      uint4 v[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) { const size_t i = base + (size_t)u * nthr; v[u] = i < n ? p[i] : make_uint4(0, 0, 0, 0); }
#pragma unroll
      for (int u = 0; u < UNR; ++u) XOR4(acc, v[u]);
    }
  } else {
    const size_t beg = (size_t)blockIdx.x * tile;
    const size_t end = beg + tile < n ? beg + tile : n;
    for (size_t base = beg + threadIdx.x; base < end; base += (size_t)blockDim.x * UNR) {
      uint4 v[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) { const size_t i = base + (size_t)u * blockDim.x; v[u] = i < end ? p[i] : make_uint4(0, 0, 0, 0); }
#pragma unroll
      for (int u = 0; u < UNR; ++u) XOR4(acc, v[u]);
    }
  }
  const unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (chk) atomicXor(chk, r);
  if (r == 0x9e3779b9u) sink[0] = r;
  WAVE_TS_END
}

// Work stealing, per wave. Chunk = 32 lanes x UNR loads x 16 B, consecutive in memory.
// Exit proof: lane 0 takes c = atomicAdd(ctr,1); c is broadcast (readfirstlane), so the whole wave
// takes the same branch. ctr only grows; each iteration consumes one unique value; a wave leaves on the
// first c >= nchunks. Total iterations over all waves <= nchunks + nwaves, so every wave exits.
template <int UNR, bool TS>
__global__ void steal(const uint4* __restrict__ p, size_t n, size_t nchunks, unsigned* chk, unsigned* sink, u64* ts, unsigned* ctr) {
  WAVE_TS_BEGIN
  const unsigned lane = threadIdx.x & 31;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (;;) {
    unsigned c = 0;
    if (lane == 0) c = atomicAdd(ctr, 1u);
    c = __builtin_amdgcn_readfirstlane(__shfl(c, 0, 32));
    if (c >= nchunks) break;
    const size_t base = (size_t)c * 32 * UNR + lane;
    uint4 v[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) { const size_t i = base + (size_t)u * 32; v[u] = i < n ? p[i] : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < UNR; ++u) XOR4(acc, v[u]);
  }
  const unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (chk) atomicXor(chk, r);
  if (r == 0x9e3779b9u) sink[0] = r;
  WAVE_TS_END
}


// Dependent launch (PDL-like), grid-stride read. Launch index L = 3rd arg. With any-order launch, kernel L may
// start while L-1 still runs. It issues its first UNR loads (the weight stream does not depend on L-1), then
// thread 0 of each WG waits until every WG of L-1 has finished (done[L-1] == gridDim.x) and reads the token
// L-1 left (dep[L-1] == L) -- stands in for a matmul's input vector. Wrong token -> err[0]++.
// Exit proof: the wait loop runs at most MAXSPIN iterations (each >= one s_sleep); timeout -> err[1]++.
#define NLMAX 24
#define MAXSPIN 20000u
#ifndef SPIN_SLEEP
#define SPIN_SLEEP 4
#endif
// Bounded wait (exit proof: at most MAXSPIN iterations) for *a >= target, relaxed polls, stats in err[1..3].
__device__ __forceinline__ void spin_until(unsigned* a, unsigned target, unsigned* err) {
  unsigned it = 0;
  while (__hip_atomic_load(a, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target && ++it < MAXSPIN) __builtin_amdgcn_s_sleep(SPIN_SLEEP);
  if (it >= MAXSPIN) atomicAdd(&err[1], 1u);
  atomicMax(&err[2], it);
  if (it > 1000) atomicAdd(&err[3], 1u);
}
// GATE: the prefetch waits until a quarter of L-1's WGs have finished (L-1's tail has begun), so the prefetch
// fills DRAM time L-1 leaves idle instead of competing with L-1's main stream.
template <int UNR, bool TS, int PF = 0, bool GATE = false, int WIN = 1>
__global__ void pdl(const uint4* __restrict__ p, size_t n, size_t L, unsigned* chk, unsigned* sink, u64* ts, unsigned* ctr) {
  WAVE_TS_BEGIN
  unsigned* done = ctr; unsigned* dep = ctr + NLMAX * 64; unsigned* err = ctr + 2 * NLMAX * 64;
  // WIN windows: WG group g = blockIdx % WIN streams segment g (grid % WIN == 0, checked on the host);
  // n below is the segment end, so every load stays < original n.
  const size_t seg = (n + WIN - 1) / WIN, lo = (size_t)(blockIdx.x % WIN) * seg;
  const size_t n_all = n; n = lo + seg < n_all ? lo + seg : n_all;
  const size_t gtid = lo + (size_t)(blockIdx.x / WIN) * blockDim.x + threadIdx.x;
  const size_t nthr = (size_t)(gridDim.x / WIN) * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  uint4 v[UNR];
  size_t base = gtid;
#pragma unroll
  for (int u = 0; u < UNR; ++u) { const size_t i = base + (size_t)u * nthr; v[u] = i < n ? p[i] : make_uint4(0, 0, 0, 0); }
  unsigned pfx = 0;
  if (PF && GATE && L > 0) {
    if (threadIdx.x == 0) spin_until(&done[(L - 1) * 64], (gridDim.x + 3) / 4, err);
    __syncthreads();
  }
  if (PF && L > 0) {  // prefetch: grid-iterations UNR..UNR+PF-1, bytes read again (from cache) by the main loop
#pragma unroll
    for (int k = 0; k < PF; ++k) { const size_t i = base + (size_t)(UNR + k) * nthr; if (i < n) pfx ^= reinterpret_cast<const unsigned*>(p + i)[0]; }
  }
  if (pfx == 0x9e3779b9u) sink[1] = pfx;  // keeps the prefetch loads
  __shared__ unsigned tok_s;
  if (threadIdx.x == 0) {
    unsigned tok = (unsigned)L;
    if (L > 0) {
      unsigned it = 0;
      // relaxed polling (no L0/L1 invalidate per iteration: an acquire load emits buffer_gl0/gl1_inv each time,
      // which flushes the GL1 the predecessor's waves are streaming through); one acquire fence after.
      while (__hip_atomic_load(&done[(L - 1) * 64], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < gridDim.x && ++it < MAXSPIN)
        __builtin_amdgcn_s_sleep(SPIN_SLEEP);
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      if (it >= MAXSPIN) atomicAdd(&err[1], 1u);
      atomicMax(&err[2], it);
      if (it > 1000) atomicAdd(&err[3], 1u);
      tok = __hip_atomic_load(&dep[(L - 1) * 64], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (tok != (unsigned)L) atomicAdd(&err[0], 1u);
    }
    tok_s = tok;
  }
  __syncthreads();
  acc.x ^= tok_s ^ (unsigned)L;  // 0 when the order held, so the validation XOR is unchanged
#pragma unroll
  for (int u = 0; u < UNR; ++u) XOR4(acc, v[u]);
  for (base += nthr * UNR; base < n; base += nthr * UNR) {
#pragma unroll
    for (int u = 0; u < UNR; ++u) { const size_t i = base + (size_t)u * nthr; v[u] = i < n ? p[i] : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < UNR; ++u) XOR4(acc, v[u]);
  }
  const unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (chk) atomicXor(chk, r);
  if (r == 0x9e3779b9u) sink[0] = r;
  __syncthreads();
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) __hip_atomic_store(&dep[L * 64], (unsigned)(L + 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_fetch_add(&done[L * 64], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
  WAVE_TS_END
}

using KFn = void (*)(const uint4*, size_t, size_t, unsigned*, unsigned*, u64*, unsigned*);
struct Arm { std::string name; char style; int wg, grid, unr; bool ao; int graph; };  // graph: 0 stream, 1 captured, 2 manual (no edges), 3 manual two chains (node i after node i-2)
// style: g = grid-stride, t = tile per WG, s = work stealing per wave, p = dependent launch (pdl),
// q = pdl UNR=1 + prefetch of <unr field> grid-iterations into cache while waiting; r = q with the prefetch gated
// on L-1's tail (a quarter of its WGs done); w = pdl UNR=1 with <unr field> address windows
static KFn pick(const Arm& a, bool ts) {
#define RD(U) (a.style == 't' ? (ts ? (KFn)rd<U, true, true> : (KFn)rd<U, true, false>) \
                              : (ts ? (KFn)rd<U, false, true> : (KFn)rd<U, false, false>))
#define ST(U) (ts ? (KFn)steal<U, true> : (KFn)steal<U, false>)
#define PD(U) (ts ? (KFn)pdl<U, true> : (KFn)pdl<U, false>)
#define PQ(K) (ts ? (KFn)pdl<1, true, K> : (KFn)pdl<1, false, K>)
#define PW(W) (ts ? (KFn)pdl<1, true, 0, false, W> : (KFn)pdl<1, false, 0, false, W>)
  if (a.style == 'w') switch (a.unr) { case 1: return PW(1); case 2: return PW(2); case 4: return PW(4); case 8: return PW(8); default: return PW(16); }
#define PR(K) (ts ? (KFn)pdl<1, true, K, true> : (KFn)pdl<1, false, K, true>)
  if (a.style == 'r') switch (a.unr) { case 2: return PR(2); case 4: return PR(4); case 8: return PR(8); case 16: return PR(16); default: return PR(32); }
  if (a.style == 'q') switch (a.unr) { case 2: return PQ(2); case 4: return PQ(4); case 8: return PQ(8); case 16: return PQ(16); default: return PQ(32); }
  if (a.style == 'p') switch (a.unr) { case 1: return PD(1); case 2: return PD(2); case 4: return PD(4); default: return PD(8); }
  if (a.style == 's') switch (a.unr) { case 4: return ST(4); case 8: return ST(8); case 16: return ST(16); default: return ST(32); }
  switch (a.unr) { case 1: return RD(1); case 2: return RD(2); case 4: return RD(4); default: return RD(8); }
}
static Arm parse_arm(const std::string& s) {  // style_wg_grid_unr[_ao]
  Arm a{s, s[0], 0, 0, 0, false, 0};
  if (std::sscanf(s.c_str() + 2, "%d_%d_%d", &a.wg, &a.grid, &a.unr) != 3) { std::fprintf(stderr, "bad arm %s\n", s.c_str()); std::exit(1); }
  if (a.style == 'w' && a.grid % a.unr) { std::fprintf(stderr, "w arm: grid %% windows\n"); std::exit(1); }
  a.ao = s.find("_ao") != std::string::npos;
  a.graph = s.find("_gc") != std::string::npos ? 1 : s.find("_gm") != std::string::npos ? 2 : s.find("_g2") != std::string::npos ? 3 : 0;
  return a;
}

// Device files are never hard-coded: the caller names them through two environment variables.
//   GPU_SYSFS_DIR  = amdgpu device directory in sysfs (the one that contains gpu_metrics)
//   GPU_POWER_FILE = hwmon power1_input file of that device (microwatts)
static std::string req_env(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) { std::fprintf(stderr, "error: environment variable %s is not set (see README.md)\n", name); std::exit(1); }
  return v;
}
static std::string gpu_metrics_path() { return req_env("GPU_SYSFS_DIR") + "/gpu_metrics"; }
static std::string power_path() { return req_env("GPU_POWER_FILE"); }

// ---- gpu_metrics sampler (gpu_metrics_v3_0 layout, offsets checked against a live dump:
// gfxclk avg @174 read 1919 while sclk showed 1917; uclk @186 = 1000 = pp_dpm_mclk) ----
struct Met { double fclk = 0, uclk = 0, gfx = 0, drd = 0, dwr = 0; int n = 0; };
static std::atomic<bool> g_sample{false}, g_quit{false};
static Met g_met; static std::atomic<int> g_metlock{0};
static void sampler() {
  int fd = open(gpu_metrics_path().c_str(), O_RDONLY);
  if (fd < 0) return;
  unsigned char b[512];
  while (!g_quit) {
    if (g_sample) {
      ssize_t k = pread(fd, b, sizeof b, 0);
      if (k >= 190) {
        auto u16 = [&](int o) { return (double)(b[o] | (b[o + 1] << 8)); };
        g_met.fclk += u16(182); g_met.uclk += u16(186); g_met.gfx += u16(174); g_met.drd += u16(94); g_met.dwr += u16(96); g_met.n++;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  close(fd);
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: magic <trace|ab> <bytes> <host|dev> <arms>\n"); return 1; }
  const std::string mode = argv[1];
  // fail early, with a clear message, if the sysfs environment variables the chosen mode needs are missing
  if (mode == "pwr" || (std::getenv("METRICS") && std::atoi(std::getenv("METRICS")))) gpu_metrics_path();
  if (mode == "pwr") power_path();
  const size_t bytes = std::strtoull(argv[2], nullptr, 10);
  const bool host = !std::strcmp(argv[3], "host");
  std::vector<Arm> arms;
  { std::string s = argv[4]; size_t q = 0; while (q <= s.size()) { size_t e = s.find(',', q); if (e == std::string::npos) e = s.size(); arms.push_back(parse_arm(s.substr(q, e - q))); q = e + 1; } }
  if (bytes % 16) { std::fprintf(stderr, "bytes %% 16\n"); return 1; }
  const size_t copies = std::max<size_t>(1, (128ULL * 1024 * 1024 + bytes - 1) / bytes);
  const size_t phys = bytes * copies;

  // ---- memory: identical to ../ceiling/ceil.hip ----
  std::vector<unsigned> h(phys / 4);
  unsigned s = 0x12345678u;
  for (auto& x : h) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; x = s; }
  std::vector<unsigned> expect(copies);
  for (size_t c = 0; c < copies; ++c) { unsigned x = 0; for (size_t i = 0; i < bytes / 4; ++i) x ^= h[c * bytes / 4 + i]; expect[c] = x; }
  unsigned char* dev = nullptr;
  CK(hipMalloc(&dev, phys));
  CK(hipMemcpy(dev, h.data(), phys, hipMemcpyHostToDevice));
  const unsigned char* base = dev;
  if (host) {
    void* hostp = mmap(nullptr, phys, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (hostp == MAP_FAILED) { std::fprintf(stderr, "mmap\n"); return 1; }
    madvise(hostp, phys, MADV_HUGEPAGE);
    CK(hipMemcpy(hostp, dev, phys, hipMemcpyDeviceToHost));
    if (mprotect(hostp, phys, PROT_READ)) { std::fprintf(stderr, "mprotect\n"); return 1; }
    CK(hipHostRegister(hostp, phys, hipHostRegisterMapped | hipHostRegisterReadOnly));
    { // how much of the mapping got 2 MB pages
      FILE* f = std::fopen("/proc/self/smaps", "r"); char ln[512]; bool in = false;
      while (f && std::fgets(ln, sizeof ln, f)) {
        unsigned long lo, hi;
        if (std::sscanf(ln, "%lx-%lx ", &lo, &hi) == 2 && std::strchr(ln, '-') < std::strchr(ln, ' ')) in = (lo <= (unsigned long)hostp && (unsigned long)hostp < hi);
        else if (in && !std::strncmp(ln, "AnonHugePages:", 14)) std::printf("# host map %p (2MB-aligned=%d) %s", hostp, (int)(((unsigned long)hostp & 0x1fffff) == 0), ln);
      }
      if (f) std::fclose(f);
    }
    void* d = nullptr; CK(hipHostGetDevicePointer(&d, hostp, 0));
    base = (const unsigned char*)d;
    CK(hipFree(dev)); dev = nullptr;
  }
  const size_t n = bytes / 16;
  unsigned *chk, *sink, *ctr;
  const int NL = 24; const int MAXR = std::getenv("TRACE_ROUNDS") ? std::atoi(std::getenv("TRACE_ROUNDS")) : 3;
  const bool compact = std::getenv("TRACE_COMPACT") != nullptr; u64 tfirst = 0;
  CK(hipMalloc(&chk, 4)); CK(hipMalloc(&sink, 1 << 20)); CK(hipMalloc(&ctr, (2 * NLMAX * 64 + 64) * 4)); CK(hipMemset(ctr, 0, (2 * NLMAX * 64 + 64) * 4));
  hipEvent_t ea, eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));

  // third kernel arg: tile (t), nchunks (s), unused (g)
  auto karg = [&](const Arm& a, int L) -> size_t {
    if (a.style == 'p' || a.style == 'q' || a.style == 'r' || a.style == 'w') return (size_t)L;
    if (a.style == 't') { size_t tl = (n + a.grid - 1) / a.grid; return (tl + 63) / 64 * 64; }
    if (a.style == 's') return (n + 32 * (size_t)a.unr - 1) / (32 * (size_t)a.unr);
    return 0;
  };
  // ctr slot per launch, 256 B apart (separate cache lines / channels)
  // All work runs on one created stream st (a null stream cannot be captured), stream and graph arms alike.
  hipStream_t st; CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
  auto launch = [&](const Arm& a, KFn k, const unsigned char* src, unsigned* c, unsigned* cnt, u64* ts, int L) {
    const size_t ka = karg(a, L);
    if (a.ao) {
      hipExtLaunchKernelGGL(k, dim3(a.grid), dim3(a.wg), 0, st, nullptr, nullptr, hipExtAnyOrderLaunch,
                            (const uint4*)src, n, ka, c, sink, ts, cnt);
    } else {
      k<<<a.grid, a.wg, 0, st>>>((const uint4*)src, n, ka, c, sink, ts, cnt);
    }
    CK(hipGetLastError());
  };
  auto isdep = [](const Arm& a) { return a.style == 'p' || a.style == 'q' || a.style == 'r' || a.style == 'w'; };
  auto launch24 = [&](const Arm& a, KFn k, u64* tsbuf, size_t nwaves) {
    for (int i = 0; i < NL; ++i)
      launch(a, k, base + (i % copies) * bytes, nullptr, isdep(a) ? ctr : ctr + i * 64, tsbuf ? tsbuf + (size_t)i * nwaves * 3 : nullptr, i);
  };
  // graph per (arm, traced): captured = the 24 launches recorded with hipStreamBeginCapture(ThreadLocal), like the
  // decode executor; manual = 24 kernel nodes added with no dependencies (edges) at all.
  std::map<std::string, hipGraphExec_t> gexec;
  const bool chk_done = std::getenv("CHECK") != nullptr, sync_rounds = std::getenv("SYNC") != nullptr;
  int n_incomplete = 0, n_checked = 0;
  auto get_graph = [&](const Arm& a, KFn k, u64* tsbuf, size_t nwaves) -> hipGraphExec_t {
    const std::string key = a.name + (tsbuf ? "#ts" : "");
    auto it = gexec.find(key); if (it != gexec.end()) return it->second;
    hipGraph_t g;
    if (a.graph == 1) {
      CK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
      launch24(a, k, tsbuf, nwaves);
      CK(hipStreamEndCapture(st, &g));
    } else {
      CK(hipGraphCreate(&g, 0));
      std::vector<hipGraphNode_t> nodes(NL);
      // _g2r: the counter reset is a memset node inside the graph, root of both chains (no host memset before)
      hipGraphNode_t mnode{}; const bool reset_in_graph = a.name.find("_g2r") != std::string::npos;
      if (reset_in_graph) {
        hipMemsetParams mp{}; mp.dst = ctr; mp.value = 0; mp.elementSize = 4; mp.width = 2 * NLMAX * 64; mp.height = 1; mp.pitch = 0;
        CK(hipGraphAddMemsetNode(&mnode, g, nullptr, 0, &mp));
      }
      for (int i = 0; i < NL; ++i) {
        const uint4* src = (const uint4*)(base + (i % copies) * bytes); size_t nn = n, ka = karg(a, i);
        unsigned* c = nullptr; unsigned* cnt = isdep(a) ? ctr : ctr + i * 64; u64* ts = tsbuf ? tsbuf + (size_t)i * nwaves * 3 : nullptr;
        void* args[] = {&src, &nn, &ka, &c, &sink, &ts, &cnt};
        hipKernelNodeParams kp{}; kp.func = (void*)k; kp.gridDim = dim3(a.grid); kp.blockDim = dim3(a.wg); kp.kernelParams = args;
        const bool chain2 = a.graph == 3 && i >= 2, root_after_reset = reset_in_graph && i < 2;
        const hipGraphNode_t* dep = chain2 ? &nodes[i - 2] : root_after_reset ? &mnode : nullptr;
        CK(hipGraphAddKernelNode(&nodes[i], g, dep, dep ? 1 : 0, &kp));
      }
    }
    size_t nn = 0, ne = 0; CK(hipGraphGetNodes(g, nullptr, &nn)); CK(hipGraphGetEdges(g, nullptr, nullptr, &ne));
    hipGraphExec_t ex; CK(hipGraphInstantiate(&ex, g, nullptr, nullptr, 0));
    std::printf("# graph %s %s: nodes=%zu edges=%zu\n", key.c_str(), a.graph == 1 ? "captured" : a.graph == 2 ? "manual-noedges" : "manual-2chains", nn, ne);
    gexec[key] = ex; return ex;
  };
  auto round = [&](const Arm& a, KFn k, u64* tsbuf, size_t nwaves) -> double {
    hipGraphExec_t ex = a.graph ? get_graph(a, k, tsbuf, nwaves) : nullptr;
    if (a.name.find("_g2r") == std::string::npos) CK(hipMemsetAsync(ctr, 0, 2 * NLMAX * 64 * 4, st));
    CK(hipEventRecord(ea, st));
    if (ex) CK(hipGraphLaunch(ex, st)); else launch24(a, k, tsbuf, nwaves);
    CK(hipEventRecord(eb, st)); CK(hipEventSynchronize(eb));
    if (chk_done && isdep(a)) {  // right after the end event: has every WG of every launch finished?
      std::vector<unsigned> dn(NLMAX * 64);
      CK(hipMemcpyAsync(dn.data(), ctr, NLMAX * 64 * 4, hipMemcpyDeviceToHost, st)); CK(hipStreamSynchronize(st));
      int inc = 0; for (int i = 0; i < NL; ++i) inc += dn[i * 64] != (unsigned)a.grid;
      if (inc) ++n_incomplete; ++n_checked;
    }
    if (sync_rounds) CK(hipDeviceSynchronize());
    float t; CK(hipEventElapsedTime(&t, ea, eb)); return t / NL;
  };
  // validation: every arm, plain and traced kernel, copies 0 and 1, XOR of all bytes must match
  auto validate = [&](const Arm& a, KFn k, u64* ts) {  // ts: required (non-null) for a traced kernel
    int ok = 1;
    for (size_t cp = 0; cp < copies && cp < 2; ++cp) {
      CK(hipMemset(chk, 0, 4)); CK(hipMemset(ctr, 0, 4)); CK(hipDeviceSynchronize());
      launch(a, k, base + cp * bytes, chk, ctr, ts, 0);
      CK(hipDeviceSynchronize());
      unsigned got; CK(hipMemcpy(&got, chk, 4, hipMemcpyDeviceToHost));
      if (got != expect[cp]) ok = 0;
    }
    return ok;
  };
  std::printf("# mode=%s bytes=%zu mem=%s copies=%zu phys=%zu\n", mode.c_str(), bytes, host ? "host" : "dev", copies, phys);
  int wclk = 0; CK(hipDeviceGetAttribute(&wclk, hipDeviceAttributeWallClockRate, 0));
  const double tick_us = 1000.0 / wclk;  // wclk in kHz
  std::printf("# wall clock %d kHz (%.1f ns/tick)\n", wclk, tick_us * 1000);
  const bool met = std::getenv("METRICS") && std::atoi(std::getenv("METRICS"));
  std::thread th; if (met) th = std::thread(sampler);

  if (mode == "trace") {
    const Arm& a = arms[0];
    KFn k = pick(a, true);
    const size_t nwaves = (size_t)a.grid * a.wg / 32;
    u64* tsd; CK(hipMalloc(&tsd, (size_t)NL * nwaves * 3 * 8));
    std::printf("# arm %s valid=%d valid_ts=%d nwaves=%zu\n", a.name.c_str(), validate(a, pick(a, false), nullptr), validate(a, k, tsd), nwaves);
    std::vector<u64> tsh((size_t)NL * nwaves * 3);
    round(a, k, tsd, nwaves);  // warm
    for (int r = 0; r < MAXR; ++r) {
      CK(hipMemset(tsd, 0, (size_t)NL * nwaves * 3 * 8)); CK(hipDeviceSynchronize());
      const double ms = round(a, k, tsd, nwaves);
      CK(hipMemcpy(tsh.data(), tsd, tsh.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> t0(NL), t1(NL);
      u64 gmin = ~0ull, gmax = 0;
      // per launch: first start, last start, end percentiles, gap to next launch
      if (compact) {  // one line per launch: round, launch, absolute start (us since first launch of round 0), duration, gap
        for (int i = 0; i < NL; ++i) {
          const u64* w = &tsh[(size_t)i * nwaves * 3]; u64 s0 = ~0ull, e1 = 0;
          for (size_t j = 0; j < nwaves; ++j) { s0 = std::min(s0, w[3 * j]); e1 = std::max(e1, w[3 * j + 1]); }
          if (!tfirst) tfirst = s0;
          std::printf("L,%d,%d,%.2f,%.2f,%.2f\n", r, i, (s0 - tfirst) * tick_us, (e1 - s0) * tick_us, i ? (s0 - t1[i - 1]) * tick_us : 0.0);
          t1[i] = e1;
        }
        std::printf("R,%d,%.2f\n", r, bytes / (ms * 1e6));
        continue;
      }
      std::printf("round %d event_us_per_launch=%.2f GBps=%.1f\n", r, ms * 1000, bytes / (ms * 1e6));
      std::printf("  L  copy  dur_us  inkern_GBps  start_spread_us  end_p10  end_p50  end_p90  end_p100(us from first start)  gap_to_next_us\n");
      for (int i = 0; i < NL; ++i) {
        const u64* w = &tsh[(size_t)i * nwaves * 3];
        u64 s0 = ~0ull, s1 = 0, e1 = 0; std::vector<u64> ends(nwaves);
        for (size_t j = 0; j < nwaves; ++j) { s0 = std::min(s0, w[3 * j]); s1 = std::max(s1, w[3 * j]); e1 = std::max(e1, w[3 * j + 1]); ends[j] = w[3 * j + 1]; }
        std::sort(ends.begin(), ends.end());
        t0[i] = s0; t1[i] = e1; gmin = std::min(gmin, s0); gmax = std::max(gmax, e1);
        auto rel = [&](u64 v) { return (double)(v - s0) * tick_us; };
        const double dur = rel(e1);
        std::printf("  %2d  %zu  %7.2f  %6.1f  %6.2f  %6.2f  %6.2f  %6.2f  %6.2f", i, i % copies, dur, bytes / (dur * 1e3), rel(s1),
                    rel(ends[nwaves / 10]), rel(ends[nwaves / 2]), rel(ends[nwaves * 9 / 10]), rel(ends.back()));
        if (i) std::printf("  prev_gap=%.2f", (t0[i] - t1[i - 1]) * tick_us);
        std::printf("\n");
      }
      double sumdur = 0, sumgap = 0;
      for (int i = 0; i < NL; ++i) { sumdur += (t1[i] - t0[i]) * tick_us; if (i) sumgap += (t0[i] - t1[i - 1]) * tick_us; }
      std::printf("  SUM trace_span_us=%.2f (event %.2f)  mean_dur=%.2f mean_gap=%.2f  inkern_GBps_mean=%.1f\n",
                  (gmax - gmin) * tick_us, ms * 1000 * NL, sumdur / NL, sumgap / (NL - 1), bytes * NL / (sumdur * 1e3));
      // per-CU (smid) wave count and mean end time for launch 12 of this round
      if (r == MAXR - 1 && !compact) {
        const int L = 12; const u64* w = &tsh[(size_t)L * nwaves * 3];
        std::vector<int> cnt(256); std::vector<double> me(256), ms_(256);
        for (size_t j = 0; j < nwaves; ++j) { unsigned id = w[3 * j + 2] & 255; cnt[id]++; me[id] += (w[3 * j + 1] - t0[L]) * tick_us; ms_[id] += (w[3 * j] - t0[L]) * tick_us; }
        std::printf("  launch %d per-smid: id:waves:mean_start_us:mean_end_us\n  ", L);
        for (int id = 0; id < 256; ++id) if (cnt[id]) std::printf("%d:%d:%.1f:%.1f ", id, cnt[id], ms_[id] / cnt[id], me[id] / cnt[id]);
        std::printf("\n");
        // alive-wave histogram, 2 us bins
        std::printf("  launch %d alive waves per 2us bin: ", L);
        const double D = (t1[L] - t0[L]) * tick_us;
        for (double b = 0; b < D + 2; b += 2) {
          int al = 0; for (size_t j = 0; j < nwaves; ++j) { double st = (w[3 * j] - t0[L]) * tick_us, en = (w[3 * j + 1] - t0[L]) * tick_us; if (st <= b + 1 && en >= b + 1) al++; }
          std::printf("%d ", al);
        }
        std::printf("\n");
      }
    }
  } else if (mode == "pwr") {
    const double sec = std::getenv("PWR_SEC") ? std::atof(std::getenv("PWR_SEC")) : 3.0;
    std::vector<KFn> ks; for (auto& a : arms) ks.push_back(pick(a, false));
    std::printf("pwr,arm,cycle,median_GBps,mean_W,samples,GB_per_J\n");
    for (int cyc = 0; cyc < 2; ++cyc)
      for (size_t j = 0; j < arms.size(); ++j) {
        std::atomic<bool> stop{false}; std::vector<double> w, gw;
        auto t0 = std::chrono::steady_clock::now();
        std::thread pt([&] {
          while (!stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500)) continue;
            FILE* f = std::fopen(power_path().c_str(), "r");
            if (f) { double uw; if (std::fscanf(f, "%lf", &uw) == 1) w.push_back(uw / 1e6); std::fclose(f); }
            // gfx-only power: gpu_metrics_v3_0 average_gfx_power (u32 mW @124; layout checked on a live dump)
            int fd = open(gpu_metrics_path().c_str(), O_RDONLY); unsigned char b[256];
            if (fd >= 0) { if (pread(fd, b, sizeof b, 0) >= 128) gw.push_back((b[124] | b[125] << 8 | b[126] << 16 | (unsigned)b[127] << 24) / 1e3); close(fd); }
          }
        });
        std::vector<double> rr;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sec) rr.push_back(round(arms[j], ks[j], nullptr, 0));
        stop = true; pt.join();
        double mw = 0; for (double x : w) mw += x; mw = w.empty() ? 0 : mw / w.size();
        const double g = bytes / (median(rr) * 1e6);
        double mg = 0; for (double x : gw) mg += x; mg = gw.empty() ? 0 : mg / gw.size();
        std::printf("pwr,%s,%d,%.2f,%.2f,%zu,%.2f,gfxW=%.2f,GB_per_J_gfx=%.2f\n", arms[j].name.c_str(), cyc, g, mw, w.size(), mw > 0 ? g / mw : 0, mg, mg > 0 ? g / mg : 0);
        std::fflush(stdout);
      }
  } else {  // ab
    const int reps = std::getenv("REPS") ? std::atoi(std::getenv("REPS")) : 9;
    std::vector<KFn> ks; std::vector<int> ok;
    for (auto& a : arms) { ks.push_back(pick(a, false)); ok.push_back(validate(a, ks.back(), nullptr)); }
    std::vector<std::vector<double>> ms(arms.size()), repmed(arms.size());
    std::vector<Met> mets(arms.size()); std::vector<unsigned> arm_err(arms.size());
    for (int rep = 0; rep < reps; ++rep)
      for (size_t j = 0; j < arms.size(); ++j) {
        unsigned e0[2], e1[2]; CK(hipMemcpy(e0, ctr + 2 * NLMAX * 64, 8, hipMemcpyDeviceToHost));
        round(arms[j], ks[j], nullptr, 0);  // warm
        std::vector<double> rr;
        for (int r = 0; r < 3; ++r) {
          if (met) { g_met = Met{}; g_sample = true; }
          const double t = round(arms[j], ks[j], nullptr, 0);
          if (met) { g_sample = false; std::this_thread::sleep_for(std::chrono::microseconds(600));
            Met m = g_met; if (m.n) { mets[j].fclk += m.fclk / m.n; mets[j].uclk += m.uclk / m.n; mets[j].drd += m.drd / m.n; mets[j].dwr += m.dwr / m.n; mets[j].n++; }
            std::printf("round,%s,%d,%d,%.2f,fclk=%.0f,uclk=%.0f,drd=%.0f,dwr=%.0f,n=%d\n", arms[j].name.c_str(), rep, r, bytes / (t * 1e6),
                        m.n ? m.fclk / m.n : 0, m.n ? m.uclk / m.n : 0, m.n ? m.drd / m.n : 0, m.n ? m.dwr / m.n : 0, m.n); }
          ms[j].push_back(t); rr.push_back(t);
        }
        repmed[j].push_back(median(rr));
        CK(hipMemcpy(e1, ctr + 2 * NLMAX * 64, 8, hipMemcpyDeviceToHost));
        arm_err[j] += (e1[0] - e0[0]) + (e1[1] - e0[1]);
        if (e1[0] != e0[0] || e1[1] != e0[1]) std::printf("# errors in rep %d arm %s (prev arm %s): wrong_order+%u timeout+%u\n", rep, arms[j].name.c_str(),
                                                          arms[(j + arms.size() - 1) % arms.size()].name.c_str(), e1[0] - e0[0], e1[1] - e0[1]);
      }
    std::printf("arm,valid,median_GBps,best_rep_median_GBps,best_round_GBps,worst_round_GBps,rep_medians\n");
    for (size_t j = 0; j < arms.size(); ++j) {
      std::printf("%s,%d,%.2f,%.2f,%.2f,%.2f,", arms[j].name.c_str(), ok[j], bytes / (median(ms[j]) * 1e6),
                  bytes / (*std::min_element(repmed[j].begin(), repmed[j].end()) * 1e6),
                  bytes / (*std::min_element(ms[j].begin(), ms[j].end()) * 1e6), bytes / (*std::max_element(ms[j].begin(), ms[j].end()) * 1e6));
      for (double v : repmed[j]) std::printf("%.1f ", bytes / (v * 1e6));
      if (met && mets[j].n) std::printf(" fclk=%.0f drd=%.0f dwr=%.0f", mets[j].fclk / mets[j].n, mets[j].drd / mets[j].n, mets[j].dwr / mets[j].n);
      std::printf("\n");
    }
  }
  { unsigned e[4]; CK(hipMemcpy(e, ctr + 2 * NLMAX * 64, 16, hipMemcpyDeviceToHost));
    if (chk_done) std::printf("# end-event check: %d of %d rounds had unfinished launches when the end event completed\n", n_incomplete, n_checked);
    std::printf("# pdl errors: wrong_order=%u spin_timeout=%u max_spin_iters=%u waits_over_1000_iters=%u\n", e[0], e[1], e[2], e[3]); }
  g_quit = true; if (th.joinable()) th.join();
  CK(hipGetLastError());
  return 0;
}

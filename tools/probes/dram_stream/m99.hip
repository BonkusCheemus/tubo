// m99.hip -- steady-state DRAM stream probe: pure read of rotating host-mapped buffers, >= 1 GB per timed window.
//
// MEASURES: the highest sustained GB/s a read-only kernel gets from DRAM, over many load/layout configurations
//   (buffer_load or global_load, 4..16 bytes per lane, unroll, wave-contiguous or grid-stride or tiled layout,
//   cache policy bits, wave priority, grid size, dynamic or static chunking). One "round" = ONE persistent
//   launch that reads the whole region R times, timed by an event pair and by in-kernel wall_clock64 (100 MHz).
//   Memory setup is copied from ../ceiling/ceil.hip (anonymous mmap + MADV_HUGEPAGE, hipMemcpy D->H, mprotect RO,
//   hipHostRegister(Mapped|ReadOnly), hipHostGetDevicePointer). copies = ceil(128 MiB / bytes); the copies sit
//   back to back in one mapping of phys = bytes*copies. "trace" mode records per-chunk completion times and
//   bins them (default 50 us) to show dips; "pwr" mode reports watts and DRAM/fabric counters per config.
//   Percent figures printed ("pct_of_256") assume a theoretical peak of 256 GB/s (LPDDR5X-8000 x 256 bit);
//   change the divisor if your memory differs.
//
// BUILD: hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o m99 m99.hip
// RUN:   m99 ab    <bytes> <host|dev> <cfg,cfg,...>    PASSES (3) x every cfg x ROUNDS (7) rounds, interleaved
//        m99 pwr   <bytes> <host|dev> <cfg,cfg,...>    each cfg back to back for PWR_SEC (4) s, gfx + socket watts
//        m99 trace <bytes> <host|dev> <cfg,cfg,...>    per-bin GB/s (cfg layout 4 only); env BIN_US (50), ROUNDS (20), BINS=1
//   <bytes> must be a multiple of 48. Modes ab and pwr need GPU_SYSFS_DIR (amdgpu device directory in sysfs,
//   contains gpu_metrics); mode pwr also needs GPU_POWER_FILE (hwmon power1_input of that device).
//   Use a pinned clock and a quiet machine (see README.md); agg.py, binstat.py, period.py summarise the logs.
// cfg = <mode><W>_u<UNR>_l<LAY>_c<CPOL>_p<PRIO>_g<grid>_t<wg>[_k<KB>]_r<R>
//   mode: bu = buffer_load, gl = global_load, nt = global_load nontemporal (gl/nt: W=16 only, CPOL ignored)
//   W: 4|8|12|16 bytes per lane per load.  UNR: 1|2|4|8 loads in flight per lane per iteration.
//   LAY: 0 grid-stride (each load instr of the grid covers one contiguous nthr*W range)
//        1 wave-contiguous (each wave reads UNR*32*W contiguous bytes per iteration)
//        2 tile per WG (each WG owns a contiguous phys/grid slice)
//        3 software-pipelined wave-contiguous (loads of the next chunk are issued before the current one is consumed)
//        4 dynamic: each WG takes the next chunk of <KB> kilobytes from a global counter (trace mode needs this layout)
//   k<KB>: chunk size in KB for layout 4 (default 64).
//   CPOL (buffer only): bit0 glc, bit1 slc, bit2 dlc (checked in the ISA).  PRIO: s_setprio 0..3.
//
// WRONG IF: a cfg prints ok=0 / "VALIDATION FAIL" (the XOR of every byte read differs from the host value);
//   rounds differ from each other by more than a few GB/s with nothing else running, or the load average is high
//   (another process was using the GPU or memory); event GB/s and in-kernel GB/s ("kern") differ by more than
//   the launch overhead can explain; or a rate exceeds the theoretical peak (cache hit, not DRAM).
#include <hip/hip_runtime.h>
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
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2);} } while (0)
typedef unsigned long long u64;
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef unsigned v3u __attribute__((ext_vector_type(3)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));

// MODE 0 buffer, 1 global plain, 2 global nontemporal. Returns the loaded words folded later (v4u, unused = 0).
template<int MODE, int W, int CP> __device__ __forceinline__ v4u ld(__amdgpu_buffer_rsrc_t r, const unsigned char* p, unsigned off) {
  if constexpr (MODE == 1) { return *reinterpret_cast<const v4u*>(p + off); }
  else if constexpr (MODE == 2) { return __builtin_nontemporal_load(reinterpret_cast<const v4u*>(p + off)); }
  else if constexpr (W == 4) { v4u v = {__builtin_amdgcn_raw_buffer_load_b32(r, off, 0, CP), 0, 0, 0}; return v; }
  else if constexpr (W == 8) { v2u a = __builtin_amdgcn_raw_buffer_load_b64(r, off, 0, CP); v4u v = {a.x, a.y, 0, 0}; return v; }
  else if constexpr (W == 12) { v3u a = __builtin_amdgcn_raw_buffer_load_b96(r, off, 0, CP); v4u v = {a.x, a.y, a.z, 0}; return v; }
  else { return __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, CP); }
}

// p = nbytes valid bytes (phys region). Every load offset is guarded < nbytes (LAY 0/1) or < end <= nbytes (LAY 2);
// buffer loads are additionally bounds-checked in hardware (num_records = nbytes, OOB returns 0).
// Host guarantees nbytes + grid-step*UNR < 2^32 (no unsigned wrap). chk != null: validation XOR.
// ts != null: lane 0 of each wave writes {start, end} at ts[2*wave]; ts holds 2*(grid*wg/32) entries.
template<int MODE, int W, int UNR, int LAY, int CP, int PRIO>
__global__ __launch_bounds__(1024) void sk(const unsigned char* __restrict__ p, unsigned nbytes, unsigned tile, int R,
                                           unsigned* chk, unsigned* sink, u64* ts, unsigned* ctr, u64* tr) {
  if constexpr (PRIO) __builtin_amdgcn_s_setprio(PRIO);
  const u64 t0 = ts ? (u64)wall_clock64() : 0;
  __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc((void*)p, 0, (int)nbytes, 0x31004000);
  const unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x, nthr = gridDim.x * blockDim.x;
  const unsigned lane = threadIdx.x & 31, gw = gtid >> 5, nw = nthr >> 5;
  v4u acc = {0, 0, 0, 0};
  for (int rep = 0; rep < R; ++rep) {
    if constexpr (LAY == 0) {
      const unsigned step = nthr * W;
      for (unsigned o = gtid * W; o < nbytes; o += step * UNR) {
        v4u v[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) { const unsigned off = o + u * step; v[u] = off < nbytes ? ld<MODE, W, CP>(rs, p, off) : v4u{0, 0, 0, 0}; }
#pragma unroll
        for (int u = 0; u < UNR; ++u) acc ^= v[u];
      }
    } else if constexpr (LAY == 1) {
      const unsigned chunk = 32 * W * UNR;
      for (unsigned c = gw * chunk; c < nbytes; c += nw * chunk) {
        v4u v[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) { const unsigned off = c + u * 32 * W + lane * W; v[u] = off < nbytes ? ld<MODE, W, CP>(rs, p, off) : v4u{0, 0, 0, 0}; }
#pragma unroll
        for (int u = 0; u < UNR; ++u) acc ^= v[u];
      }
    } else if constexpr (LAY == 3) {
      // software-pipelined wave-contiguous: loads of chunk k+1 are issued before chunk k is consumed, so a wave
      // never drains to zero loads in flight. Buffer mode drops the guard: hardware OOB returns 0 (num_records).
      const unsigned chunk = 32 * W * UNR, gstep = nw * chunk;
      unsigned c = gw * chunk;
      v4u v[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) { const unsigned off = c + u * 32 * W + lane * W; v[u] = (MODE == 0 || off < nbytes) ? ld<MODE, W, CP>(rs, p, off) : v4u{0, 0, 0, 0}; }
      // two stages per trip so v/nx swap is a register rename, not a copy that would force a wait.
      // chunk c+gstep beyond nbytes loads zeros (guard / OOB) and is consumed harmlessly; the c+2*gstep loads
      // issued on the last trip are never consumed (zeros, no memory access when OOB).
      for (; c < nbytes; c += 2 * gstep) {
        v4u nx[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) { const unsigned off = c + gstep + u * 32 * W + lane * W; nx[u] = (MODE == 0 || off < nbytes) ? ld<MODE, W, CP>(rs, p, off) : v4u{0, 0, 0, 0}; }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int u = 0; u < UNR; ++u) acc ^= v[u];
        __builtin_amdgcn_sched_barrier(0);  // keep stage order: the scheduler otherwise clusters both stages' loads
#pragma unroll
        for (int u = 0; u < UNR; ++u) { const unsigned off = c + 2 * gstep + u * 32 * W + lane * W; v[u] = (MODE == 0 || off < nbytes) ? ld<MODE, W, CP>(rs, p, off) : v4u{0, 0, 0, 0}; }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int u = 0; u < UNR; ++u) acc ^= nx[u];
        __builtin_amdgcn_sched_barrier(0);
      }
    } else if constexpr (LAY == 4) {
      // dynamic: each WG grabs chunk k (tile bytes) from *ctr (zeroed by the host before the launch); chunk k
      // reads phys bytes [(k % nch)*tile, +tile) clipped to nbytes. k >= R*nch -> done. Inside a chunk the WG's
      // waves read wave-contiguous UNR*32*W pieces. The next index is fetched while the current chunk streams.
      // Runs its own loop over all R reps, so it breaks out of the rep loop after the first trip.
      __shared__ unsigned s_k;
      const unsigned nch = (nbytes + tile - 1) / tile, total = nch * (unsigned)R;
      const unsigned wl = threadIdx.x >> 5, nwl = blockDim.x >> 5, cw = 32 * W * UNR;
      if (threadIdx.x == 0) s_k = atomicAdd(ctr, 1u);
      __syncthreads();
      unsigned cur = s_k;
      __syncthreads();
      while (cur < total) {
        unsigned nx = 0;
        if (threadIdx.x == 0) nx = atomicAdd(ctr, 1u);
        const unsigned beg = (cur % nch) * tile;
        const unsigned end = beg + tile < nbytes ? beg + tile : nbytes;
        for (unsigned c = beg + wl * cw; c < end; c += nwl * cw) {
          v4u v[UNR];
#pragma unroll
          for (int u = 0; u < UNR; ++u) { const unsigned off = c + u * 32 * W + lane * W; v[u] = off < end ? ld<MODE, W, CP>(rs, p, off) : v4u{0, 0, 0, 0}; }
#pragma unroll
          for (int u = 0; u < UNR; ++u) acc ^= v[u];
        }
        if (threadIdx.x == 0) s_k = nx;
        __syncthreads();  // every wave has XORed (so received) its loads of chunk cur
        if (tr && threadIdx.x == 0) tr[cur] = (u64)wall_clock64();  // cur < total; tr holds total entries
        cur = s_k;
        __syncthreads();
      }
      break;  // all R reps done above
    } else {
      const unsigned beg = blockIdx.x * tile;
      const unsigned end = beg + tile < nbytes ? beg + tile : nbytes;  // beg >= nbytes -> loop skipped
      const unsigned step = blockDim.x * W;
      for (unsigned o = beg + threadIdx.x * W; o < end; o += step * UNR) {
        v4u v[UNR];
#pragma unroll
        for (int u = 0; u < UNR; ++u) { const unsigned off = o + u * step; v[u] = off < end ? ld<MODE, W, CP>(rs, p, off) : v4u{0, 0, 0, 0}; }
#pragma unroll
        for (int u = 0; u < UNR; ++u) acc ^= v[u];
      }
    }
  }
  const unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (chk) atomicXor(chk, r);
  if (r == 0x9e3779b9u) sink[0] = r;  // keeps the loads alive
  if (ts && lane == 0) {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    ts[2 * (size_t)gw] = t0; ts[2 * (size_t)gw + 1] = (u64)wall_clock64();
  }
}

using KFn = void (*)(const unsigned char*, unsigned, unsigned, int, unsigned*, unsigned*, u64*, unsigned*, u64*);
template<int M, int W, int U, int L> KFn pc(int cp, int pr) {
  if (pr) { switch (pr) { case 1: return sk<M, W, U, L, 0, 1>; case 2: return sk<M, W, U, L, 0, 2>; default: return sk<M, W, U, L, 0, 3>; } }
  if (M != 0) return sk<M, W, U, L, 0, 0>;
  switch (cp) { case 0: return sk<M, W, U, L, 0, 0>; case 1: return sk<M, W, U, L, 1, 0>; case 2: return sk<M, W, U, L, 2, 0>; case 3: return sk<M, W, U, L, 3, 0>;
    case 4: return sk<M, W, U, L, 4, 0>; case 5: return sk<M, W, U, L, 5, 0>; case 6: return sk<M, W, U, L, 6, 0>; default: return sk<M, W, U, L, 7, 0>; }
}
template<int M, int W, int U> KFn pl(int l, int cp, int pr) { return l == 0 ? pc<M, W, U, 0>(cp, pr) : l == 1 ? pc<M, W, U, 1>(cp, pr) : l == 2 ? pc<M, W, U, 2>(cp, pr) : l == 3 ? pc<M, W, U, 3>(cp, pr) : pc<M, W, U, 4>(cp, pr); }
template<int M, int W> KFn pu(int u, int l, int cp, int pr) {
  switch (u) { case 1: return pl<M, W, 1>(l, cp, pr); case 2: return pl<M, W, 2>(l, cp, pr); case 4: return pl<M, W, 4>(l, cp, pr); default: return pl<M, W, 8>(l, cp, pr); }
}
KFn pick(int m, int w, int u, int l, int cp, int pr) {
  if (m == 1) return pu<1, 16>(u, l, cp, pr);
  if (m == 2) return pu<2, 16>(u, l, cp, pr);
  switch (w) { case 4: return pu<0, 4>(u, l, cp, pr); case 8: return pu<0, 8>(u, l, cp, pr); case 12: return pu<0, 12>(u, l, cp, pr); default: return pu<0, 16>(u, l, cp, pr); }
}

struct Cfg { std::string name; int m = 0, w = 16, u = 2, l = 0, cp = 0, pr = 0, grid = 80, wg = 256, R = 7, kb = 64; unsigned tile = 0; KFn k = nullptr; int ok = 0; };
static Cfg parse(const std::string& s) {
  Cfg c; c.name = s;
  c.m = !s.compare(0, 2, "gl") ? 1 : !s.compare(0, 2, "nt") ? 2 : 0;
  c.w = std::atoi(s.c_str() + 2);
  size_t q = 0;
  while ((q = s.find('_', q)) != std::string::npos) {
    ++q; const char f = s[q]; const int v = std::atoi(s.c_str() + q + 1);
    if (f == 'u') c.u = v; else if (f == 'l') c.l = v; else if (f == 'c') c.cp = v; else if (f == 'p') c.pr = v;
    else if (f == 'g') c.grid = v; else if (f == 't') c.wg = v; else if (f == 'r') c.R = v; else if (f == 'k') c.kb = v;
  }
  if (c.m) c.w = 16;
  if ((c.w != 4 && c.w != 8 && c.w != 12 && c.w != 16) || (c.u != 1 && c.u != 2 && c.u != 4 && c.u != 8) || c.l < 0 || c.l > 4 || c.kb < 1 || c.kb > 4096 ||
      c.cp < 0 || c.cp > 7 || c.pr < 0 || c.pr > 3 || c.grid < 1 || c.wg < 32 || c.wg > 1024 || c.wg % 32 || c.R < 1 || (size_t)c.grid * c.wg > (1u << 20)) {
    std::fprintf(stderr, "bad cfg %s\n", s.c_str()); std::exit(1);
  }
  c.k = pick(c.m, c.w, c.u, c.l, c.cp, c.pr);
  return c;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }
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
static double rd_hwmon() { FILE* f = std::fopen(power_path().c_str(), "r"); double x = 0; if (f) { if (std::fscanf(f, "%lf", &x) != 1) x = 0; std::fclose(f); } return x / 1e6; }
static double rd_gfxw() {  // gpu_metrics_v3_0 average_gfx_power, u32 mW @124 (offset checked in ../overlap/magic.hip)
  int fd = open(gpu_metrics_path().c_str(), O_RDONLY); unsigned char b[256]; double w = -1;
  if (fd >= 0) { if (pread(fd, b, sizeof b, 0) >= 128) w = (b[124] | b[125] << 8 | b[126] << 16 | (unsigned)b[127] << 24) / 1e3; close(fd); }
  return w;
}
static double g_fclk = -1, g_uclk = -1, g_endspread = 0, g_startspread = 0;
static void rd_dram(double* r, double* w) {  // gpu_metrics_v3_0 average_dram_reads/writes u16 @94/@96 (units unverified);
  // also average_fclk @182 and average_uclk @186 (offsets from ../overlap/magic.hip, checked there against pp_dpm_*)
  int fd = open(gpu_metrics_path().c_str(), O_RDONLY); unsigned char b[256]; *r = *w = -1;
  if (fd >= 0) { if (pread(fd, b, sizeof b, 0) >= 188) { *r = b[94] | b[95] << 8; *w = b[96] | b[97] << 8; g_fclk = b[182] | b[183] << 8; g_uclk = b[186] | b[187] << 8; } close(fd); }
}
static std::string loadavg() { char b[64] = {0}; FILE* f = std::fopen("/proc/loadavg", "r"); if (f) { if (!std::fgets(b, 15, f)) b[0] = 0; std::fclose(f); } std::string s = b; while (!s.empty() && s.back() == '\n') s.pop_back(); return s; }

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: m99 <ab|pwr> <bytes> <host|dev> <cfg,...>\n"); return 1; }
  const std::string mode = argv[1];
  // fail early, with a clear message, if the sysfs environment variables the chosen mode needs are missing
  if (mode != "trace") gpu_metrics_path();
  if (mode == "pwr") power_path();
  const size_t bytes = std::strtoull(argv[2], nullptr, 10);
  const bool host = !std::strcmp(argv[3], "host");
  std::vector<Cfg> cfgs;
  { std::string s = argv[4]; size_t q = 0; while (q <= s.size()) { size_t e = s.find(',', q); if (e == std::string::npos) e = s.size(); cfgs.push_back(parse(s.substr(q, e - q))); q = e + 1; } }
  if (bytes % 48) { std::fprintf(stderr, "bytes %% 48 (W=12 and W=16 must both divide)\n"); return 1; }
  const size_t copies = std::max<size_t>(1, (128ULL * 1024 * 1024 + bytes - 1) / bytes);
  const size_t phys = bytes * copies;
  if (phys + 1024ULL * 1024 * 16 * 8 * 8 >= (1ULL << 32)) { std::fprintf(stderr, "phys too large for 32-bit offsets\n"); return 1; }
  // ---- memory: identical to ../ceiling/ceil.hip ----
  std::vector<unsigned> h(phys / 4);
  unsigned s = 0x12345678u;
  for (auto& x : h) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; x = s; }
  unsigned expect = 0; for (auto x : h) expect ^= x;  // XOR of the whole phys region (all copies)
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
    void* d = nullptr; CK(hipHostGetDevicePointer(&d, hostp, 0));
    base = (const unsigned char*)d;
    CK(hipFree(dev)); dev = nullptr;
  }
  unsigned *chk, *sink; u64* ts;
  size_t maxw = 0; for (auto& c : cfgs) maxw = std::max(maxw, (size_t)c.grid * c.wg / 32);
  hipStream_t st; CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
  const int NCTR = 1 << 16; unsigned* ctr; int ctr_i = 0;
  CK(hipMalloc(&ctr, NCTR * 4)); CK(hipMemset(ctr, 0, NCTR * 4));
  auto next_ctr = [&]() { if (ctr_i == NCTR) { CK(hipStreamSynchronize(st)); CK(hipMemset(ctr, 0, NCTR * 4)); CK(hipDeviceSynchronize()); ctr_i = 0; } return ctr + ctr_i++; };
  CK(hipMalloc(&chk, 4)); CK(hipMalloc(&sink, 1 << 20)); CK(hipMalloc(&ts, 2 * maxw * sizeof(u64)));
  std::vector<u64> hts(2 * maxw);
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  int dev_id = 0; CK(hipGetDevice(&dev_id)); int wc_khz = 0; CK(hipDeviceGetAttribute(&wc_khz, hipDeviceAttributeWallClockRate, dev_id));
  const double tick_us = 1000.0 / wc_khz;
  std::printf("# mode=%s bytes=%zu mem=%s copies=%zu phys=%zu wallclock=%d kHz\n", mode.c_str(), bytes, host ? "host" : "dev", copies, phys, wc_khz);
  const unsigned nb = (unsigned)phys;
  for (auto& c : cfgs) {
    const unsigned q = 32u * c.w;
    c.tile = (unsigned)(((phys + c.grid - 1) / c.grid + q - 1) / q * q);
    if (c.l == 4) c.tile = (unsigned)(c.kb * 1024u / (unsigned)(c.w * 4) * (unsigned)(c.w * 4));  // multiple of W (and of 4)
    // validation: R=1, XOR of every byte read == host XOR of the phys region
    CK(hipMemset(chk, 0, 4));
    c.k<<<c.grid, c.wg, 0, st>>>(base, nb, c.tile, 1, chk, sink, nullptr, next_ctr(), nullptr);
    CK(hipGetLastError()); CK(hipStreamSynchronize(st));
    unsigned got; CK(hipMemcpy(&got, chk, 4, hipMemcpyDeviceToHost));
    c.ok = got == expect;
    if (!c.ok) std::printf("# VALIDATION FAIL %s got %08x want %08x\n", c.name.c_str(), got, expect);
  }
  // one round = one launch reading R*phys; returns event ms, fills in-kernel span (first wave start -> last wave end)
  auto round = [&](Cfg& c, bool wts, double* span_us) {
    unsigned* cc = next_ctr();
    CK(hipEventRecord(a, st));
    c.k<<<c.grid, c.wg, 0, st>>>(base, nb, c.tile, c.R, nullptr, sink, wts ? ts : nullptr, cc, nullptr);
    CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
    float t; CK(hipEventElapsedTime(&t, a, b));
    if (wts && span_us) {
      const size_t nw = (size_t)c.grid * c.wg / 32;
      CK(hipMemcpy(hts.data(), ts, 2 * nw * sizeof(u64), hipMemcpyDeviceToHost));
      u64 lo = ~0ULL, hi = 0, elo = ~0ULL, shi = 0;
      for (size_t i = 0; i < nw; ++i) { lo = std::min(lo, hts[2 * i]); hi = std::max(hi, hts[2 * i + 1]); elo = std::min(elo, hts[2 * i + 1]); shi = std::max(shi, hts[2 * i]); }
      *span_us = (hi - lo) * tick_us;
      g_endspread = (hi - elo) * tick_us; g_startspread = (shi - lo) * tick_us;
    }
    return (double)t;
  };
  if (mode == "trace") {  // LAY4 only: per-chunk completion times -> GB/s in BIN_US bins over the steady middle
    const double bin_us = std::getenv("BIN_US") ? std::atof(std::getenv("BIN_US")) : 50.0;
    const int nr = std::getenv("ROUNDS") ? std::atoi(std::getenv("ROUNDS")) : 20;
    std::printf("trace,cfg,round,ev_GBps,bins,p05,p25,p50,p75,p95,max,mean_mid\n");
    for (auto& c : cfgs) {
      if (c.l != 4) { std::printf("# skip %s (trace needs l4)\n", c.name.c_str()); continue; }
      const size_t nch = (phys + c.tile - 1) / c.tile, total = nch * (size_t)c.R;
      u64* tr; CK(hipMalloc(&tr, total * sizeof(u64))); std::vector<u64> ht(total);
      std::vector<double> allbins;
      for (int r = 0; r < nr; ++r) {
        CK(hipMemsetAsync(tr, 0, total * sizeof(u64), st));
        unsigned* cw = next_ctr(); unsigned* cc = next_ctr();
        c.k<<<c.grid, c.wg, 0, st>>>(base, nb, c.tile, c.R, nullptr, sink, nullptr, cw, nullptr);  // warm, back to back
        CK(hipEventRecord(a, st));
        c.k<<<c.grid, c.wg, 0, st>>>(base, nb, c.tile, c.R, nullptr, sink, nullptr, cc, tr);
        CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
        float t; CK(hipEventElapsedTime(&t, a, b));
        CK(hipMemcpy(ht.data(), tr, total * sizeof(u64), hipMemcpyDeviceToHost));
        u64 lo = ~0ULL, hi = 0; size_t miss = 0;
        for (auto x : ht) { if (!x) { ++miss; continue; } lo = std::min(lo, x); hi = std::max(hi, x); }
        const double span = (hi - lo) * tick_us; const int nb_ = (int)(span / bin_us);
        std::vector<double> by(nb_ + 1, 0.0);
        for (size_t k = 0; k < total; ++k) if (ht[k]) {
          const size_t beg = (k % nch) * c.tile; const double cb = (double)std::min<size_t>(c.tile, phys - beg);
          by[std::min<int>(nb_, (int)(((ht[k] - lo) * tick_us) / bin_us))] += cb; }
        std::vector<double> g; double sum = 0;  // drop first 2 and last 2 bins (ramp / tail / partial bin)
        for (int i = 2; i + 2 < nb_; ++i) { g.push_back(by[i] / (bin_us * 1e3)); sum += by[i]; allbins.push_back(g.back()); }
        if (std::getenv("BINS")) { std::printf("bins,%s,%d", c.name.c_str(), r); for (double x : g) std::printf(",%.2f", x); std::printf("\n"); }
        std::sort(g.begin(), g.end());
        auto pc = [&](double q) { return g.empty() ? 0 : g[(size_t)(q * (g.size() - 1))]; };
        std::printf("trace,%s,%d,%.2f,%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f%s\n", c.name.c_str(), r, (double)phys * c.R / (t * 1e6), g.size(), pc(.05), pc(.25), pc(.5), pc(.75), pc(.95), pc(1),
                    g.empty() ? 0 : sum / (g.size() * bin_us * 1e3), miss ? ",MISSING" : "");
      }
      std::sort(allbins.begin(), allbins.end());
      auto pa = [&](double q) { return allbins.empty() ? 0 : allbins[(size_t)(q * (allbins.size() - 1))]; };
      std::printf("tracesum,%s,bins=%zu,p01=%.1f,p05=%.1f,p25=%.1f,p50=%.1f,p75=%.1f,p95=%.1f,p99=%.1f,max=%.1f,load=%s\n", c.name.c_str(), allbins.size(),
                  pa(.01), pa(.05), pa(.25), pa(.5), pa(.75), pa(.95), pa(.99), pa(1), loadavg().c_str());
      // histogram, 1 GB/s buckets 230..260
      std::printf("hist,%s", c.name.c_str()); for (int bkt = 230; bkt < 262; bkt += 2) { size_t n = 0; for (double x : allbins) n += (x >= bkt && x < bkt + 2); std::printf(",%d:%zu", bkt, n); } std::printf("\n");
      CK(hipFree(tr));
    }
    return 0;
  }
  if (mode == "pwr") {
    const double sec = std::getenv("PWR_SEC") ? std::atof(std::getenv("PWR_SEC")) : 4.0;
    { // idle baseline: no GPU work from us for 1.5 s
      std::vector<double> gw, sw, dr, dw, fc, uc; auto t0 = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(1500)) { std::this_thread::sleep_for(std::chrono::milliseconds(50));
        double r, w; rd_dram(&r, &w); dr.push_back(r); dw.push_back(w); gw.push_back(rd_gfxw()); sw.push_back(rd_hwmon()); fc.push_back(g_fclk); uc.push_back(g_uclk); }
      std::printf("idle,gfxW=%.2f,socketW=%.2f,dram_rd=%.0f,dram_wr=%.0f,fclk=%.0f,uclk_med=%.0f,uclk_min=%.0f,load=%s\n", median(gw), median(sw), median(dr), median(dw),
                  median(fc), median(uc), *std::min_element(uc.begin(), uc.end()), loadavg().c_str());
    }
    std::printf("pwr,cfg,ok,median_GBps,gfxW,socketW,samples,GB_per_J_gfx,load,dram_rd,dram_wr,fclk_med,fclk_min,uclk_med,uclk_min\n");
    for (int cyc = 0; cyc < 2; ++cyc)
      for (auto& c : cfgs) {
        std::atomic<bool> stop{false}; std::vector<double> gw, sw, dr, dw, fc, uc;
        auto t0 = std::chrono::steady_clock::now();
        std::thread pt([&] { while (!stop) { std::this_thread::sleep_for(std::chrono::milliseconds(50));
          if (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500)) continue; gw.push_back(rd_gfxw()); sw.push_back(rd_hwmon()); double r_, w_; rd_dram(&r_, &w_); dr.push_back(r_); dw.push_back(w_); fc.push_back(g_fclk); uc.push_back(g_uclk); } });
        std::vector<double> ms;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sec) ms.push_back(round(c, false, nullptr));
        stop = true; pt.join();
        double mg = 0, ms_ = 0; for (double x : gw) mg += x; for (double x : sw) ms_ += x;
        mg = gw.empty() ? 0 : mg / gw.size(); ms_ = sw.empty() ? 0 : ms_ / sw.size();
        const double g = (double)phys * c.R / (median(ms) * 1e6);
        std::printf("pwr,%s,%d,%.2f,%.2f,%.2f,%zu,%.2f,%s,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n", c.name.c_str(), c.ok, g, mg, ms_, gw.size(), mg > 0 ? g / mg : 0, loadavg().c_str(), median(dr), median(dw),
                    median(fc), fc.empty() ? 0 : *std::min_element(fc.begin(), fc.end()), median(uc), uc.empty() ? 0 : *std::min_element(uc.begin(), uc.end()));
        std::fflush(stdout);
      }
    return 0;
  }
  // ab: PASSES x (every cfg x ROUNDS rounds), interleaved so slow drift hits every cfg alike
  const int passes = std::getenv("PASSES") ? std::atoi(std::getenv("PASSES")) : 3;
  const int rounds = std::getenv("ROUNDS") ? std::atoi(std::getenv("ROUNDS")) : 7;
  std::vector<std::vector<double>> pm(cfgs.size()), pk(cfgs.size());
  // fclk/uclk sampler: every 2 ms, attributed to the cfg currently timed (cur_cfg; -1 = between cfgs)
  std::atomic<int> cur_cfg{-1}; std::atomic<bool> quit{false};
  std::vector<std::vector<double>> fcs(cfgs.size()), ucs(cfgs.size());
  std::thread smp([&] { while (!quit) { const int j = cur_cfg; if (j >= 0) { double r_, w_; rd_dram(&r_, &w_); if (cur_cfg == j) { fcs[j].push_back(g_fclk); ucs[j].push_back(g_uclk); } }
                        std::this_thread::sleep_for(std::chrono::milliseconds(2)); } });
  { // 1 s warm stream so fclk/uclk DPM are at their top state before the first timed round
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(1000)) round(cfgs[0], false, nullptr);
  }
  std::printf("pass,cfg,ok,ev_GBps_med,ev_GBps_min,ev_GBps_max,kern_GBps_med,ev_ms_med,load,end_spread_us,start_spread_us\n");
  for (int ps = 0; ps < passes; ++ps) {
    const std::string la = loadavg();
    for (size_t j = 0; j < cfgs.size(); ++j) {
      auto& c = cfgs[j];
      round(c, false, nullptr);  // warm
      cur_cfg = (int)j;
      std::vector<double> g, gk, es, ss;
      const double by = (double)phys * c.R;
      for (int r = 0; r < rounds; ++r) { double sp = 0; const double ms = round(c, true, &sp); g.push_back(by / (ms * 1e6)); gk.push_back(by / (sp * 1e3)); es.push_back(g_endspread); ss.push_back(g_startspread); }
      cur_cfg = -1;
      const double m = median(g); pm[j].push_back(m); pk[j].push_back(median(gk));
      std::printf("%d,%s,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%s,%.1f,%.1f\n", ps, c.name.c_str(), c.ok, m, *std::min_element(g.begin(), g.end()),
                  *std::max_element(g.begin(), g.end()), median(gk), by / (m * 1e6), la.c_str(), median(es), median(ss));
      std::fflush(stdout);
    }
  }
  quit = true; smp.join();
  std::printf("final,cfg,ok,GBps_median_of_pass_medians,pass_min,pass_max,kern_GBps_med,pct_of_256,fclk_med,fclk_min,uclk_med,uclk_min\n");
  for (size_t j = 0; j < cfgs.size(); ++j) {
    const double m = median(pm[j]);
    auto mn = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end()); };
    std::printf("final,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.0f,%.0f,%.0f,%.0f\n", cfgs[j].name.c_str(), cfgs[j].ok, m, *std::min_element(pm[j].begin(), pm[j].end()),
                *std::max_element(pm[j].begin(), pm[j].end()), median(pk[j]), m / 2.56, median(fcs[j]), mn(fcs[j]), median(ucs[j]), mn(ucs[j]));
  }
  CK(hipGetLastError());
  return 0;
}

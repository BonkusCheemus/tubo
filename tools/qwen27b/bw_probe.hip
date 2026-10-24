// Pure-read memory bandwidth probe: is 90%+ of the theoretical DRAM bandwidth reachable?
// gfx1151 theory = 256 GB/s (LPDDR5X-8000 x 256 bit); best GEMM cells reached 232-233. This measures
// the box's real ceiling: read-only kernels (4/8/16-byte loads, plain and nontemporal), a grid sweep,
// one copy kernel for reference. Memory: hipMalloc (dev) or hipHostRegister'd host pages (reg, the
// same flags gufo uses for its weights: Mapped|ReadOnly).
// usage: bw_probe [buf_mib=2048] [dev|reg]    -> one CSV row per config on stdout.
// Build both bw_probe (wave32) and bw_probe_w64 (-mwavefrontsize64).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// Native clang vectors, not HIP's uint2/uint4 structs: __builtin_nontemporal_load takes only these.
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
__device__ inline unsigned fold(unsigned v) { return v; }
__device__ inline unsigned fold(u32x2 v) { return v.x ^ v.y; }
__device__ inline unsigned fold(u32x4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

template <typename T, bool NT>
__device__ inline T load(const T* p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

// Grid-stride read, 4 loads in flight per thread, XOR-reduced to one word per block (defeats DCE).
template <typename T, bool NT>
__global__ void read_kernel(const T* __restrict__ src, std::size_t n, unsigned* __restrict__ out) {
  extern __shared__ unsigned red[];
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const T a = load<T, NT>(src + i), b = load<T, NT>(src + i + stride),
            c = load<T, NT>(src + i + 2 * stride), d = load<T, NT>(src + i + 3 * stride);
    acc ^= fold(a) ^ fold(b) ^ fold(c) ^ fold(d);
  }
  for (; i < n; i += stride) acc ^= fold(load<T, NT>(src + i));
  red[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] ^= red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = red[0];
}

__global__ void copy_kernel(const u32x4* __restrict__ src, u32x4* __restrict__ dst, std::size_t n) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

// Hash fill: a constant byte pattern could flatter the number if any layer
// compresses or short-circuits uniform data; hashed words cannot.
__device__ __host__ inline unsigned hash_word(std::size_t i) {
  unsigned x = unsigned(i) * 2654435761u ^ unsigned(i >> 32);
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
  return x;
}
__global__ void fill_kernel(unsigned* dst, std::size_t n) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = hash_word(i);
}

__global__ void wave_size(int* out) { *out = warpSize; }

struct Stat { double median, best; };

template <typename F>
static Stat time_gbs(F launch, double bytes) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  for (int w = 0; w < 3; ++w) { launch(); CHECK(hipGetLastError()); }
  CHECK(hipDeviceSynchronize());
  std::vector<double> gbs;
  for (int r = 0; r < 20; ++r) {
    CHECK(hipEventRecord(a));
    launch();
    CHECK(hipGetLastError());  // a failed launch times ~0 ms and would read as a huge GB/s
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b));
    gbs.push_back(bytes / (ms * 1e-3) / 1e9);
  }
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  std::sort(gbs.begin(), gbs.end());
  return {gbs[gbs.size() / 2], gbs.back()};
}

int main(int argc, char** argv) {
  const std::size_t mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2048;
  const bool reg = argc > 2 && std::strcmp(argv[2], "reg") == 0;
  const bool hashed = argc > 3 && std::strcmp(argv[3], "hash") == 0;
  if (mib < 64 || mib > 4096) { std::fprintf(stderr, "buf_mib must be 64..4096\n"); return 2; }
  const std::size_t bytes = mib << 20;

  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  const int cus = prop.multiProcessorCount;
  int* d_ws;
  int ws = 0;
  CHECK(hipMalloc(&d_ws, sizeof(int)));
  wave_size<<<1, 1>>>(d_ws);
  CHECK(hipMemcpy(&ws, d_ws, sizeof(int), hipMemcpyDeviceToHost));
  CHECK(hipFree(d_ws));

  void* host = nullptr;
  void* src = nullptr;
  if (reg) {
    // ponytail: anonymous 2 MiB-aligned pages; gufo's weight copy may differ in page size/THP, so reg is an estimate.
    host = std::aligned_alloc(2 << 20, bytes);
    if (!host) { std::fprintf(stderr, "aligned_alloc failed\n"); return 1; }
    if (hashed) {
      auto* w = static_cast<unsigned*>(host);
      for (std::size_t i = 0; i < bytes / sizeof(unsigned); ++i) w[i] = hash_word(i);
    } else {
      std::memset(host, 0x5a, bytes);
    }
    CHECK(hipHostRegister(host, bytes, hipHostRegisterMapped | hipHostRegisterReadOnly));
    CHECK(hipHostGetDevicePointer(&src, host, 0));
  } else {
    CHECK(hipMalloc(&src, bytes));
    if (hashed) {
      fill_kernel<<<cus * 16, 256>>>(static_cast<unsigned*>(src), bytes / sizeof(unsigned));
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());
    } else {
      CHECK(hipMemset(src, 0x5a, bytes));
    }
  }
  unsigned* out;
  CHECK(hipMalloc(&out, sizeof(unsigned) * cus * 64));

  // pct_* = share of the 256 GB/s theory; valid=0 flags a row above 270 GB/s (impossible on this bus: a timing fault).
  std::printf("kernel,mem,wave,load_bytes,nontemporal,block,blocks_per_cu,grid,buf_mib,median_gbs,best_gbs,pct_median_of_256,pct_best_of_256,valid\n");
  auto row = [&](const char* k, int lb, int nt, int block, int bpc, Stat s) {
    std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%zu,%.1f,%.1f,%.1f,%.1f,%d\n", k, reg ? "reg" : "dev", ws, lb, nt, block,
                bpc, cus * bpc, mib, s.median, s.best, s.median / 256.0 * 100.0, s.best / 256.0 * 100.0,
                s.best <= 270.0 ? 1 : 0);
    std::fflush(stdout);
  };
  auto sweep = [&](auto tag, int lb) {
    using T = decltype(tag);
    const std::size_t n = bytes / sizeof(T);
    for (int block : {256, 512})
      for (int bpc : {1, 2, 4, 8, 16, 32, 64}) {
        const int grid = cus * bpc;
        const std::size_t shm = sizeof(unsigned) * block;
        row("read", lb, 0, block, bpc, time_gbs([&] {
              read_kernel<T, false><<<grid, block, shm>>>(static_cast<const T*>(src), n, out); }, double(bytes)));
        row("read", lb, 1, block, bpc, time_gbs([&] {
              read_kernel<T, true><<<grid, block, shm>>>(static_cast<const T*>(src), n, out); }, double(bytes)));
      }
  };
  sweep(unsigned{}, 4);
  sweep(u32x2{}, 8);
  sweep(u32x4{}, 16);
  CHECK(hipGetLastError());

  // Copy reference (dev only: reg memory is read-only). Half the buffer -> other half; bytes = read + write.
  if (!reg) {
    const std::size_t n = bytes / 2 / sizeof(u32x4);
    auto* s4 = static_cast<const u32x4*>(src);
    auto* d4 = static_cast<u32x4*>(src) + n;
    for (int bpc : {4, 16, 64})
      row("copy", 16, 0, 256, bpc, time_gbs([&] { copy_kernel<<<cus * bpc, 256>>>(s4, d4, n); }, double(bytes)));
    CHECK(hipGetLastError());
  }

  CHECK(hipFree(out));
  if (reg) { CHECK(hipHostUnregister(host)); std::free(host); }
  else CHECK(hipFree(src));
  return 0;
}

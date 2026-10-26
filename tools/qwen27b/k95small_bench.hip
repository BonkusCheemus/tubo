// k95small bench: the tiny verify projections, served
// route vs the fused k95 launches (src/models/qwen/hip/kernels/k95_small.hpp).
//   kv  : attention K (Q5_K 1024x5120) + V (Q6_K 1024x5120), width 8.
//         served = quant + V3<Q5_K,8,4,4,2> + quant + Dot4Lds<Q6_K,8,2,8,1>;
//         k95    = one K95KvKernel (the quantization is shared with Q's).
//   ctl : SSM alpha (F32 48x5120) + beta (Q8_0 48x5120), widths 1-8 except 6.
//         served = FastGEMV grid (48,B) + Q8_0<8,1,1> grid (6,B); k95 = one
//         launch: "bodies" (the served bodies fused) or "preload" (the
//         served k95 pick, K95SsmControlsPreloadKernel).
//   q   : one activation quantization (the launch k95 removes per shared x).
// Every fused output must be bitwise equal to the served output (memcmp);
// any mismatch prints MISMATCH and the exit code is 1.
// Weights are host-registered read-only like GUFO_DEV_WEIGHTS_GTT=1 and
// rotate over copies totalling > 3x the 32 MB MALL, so every rep reads DRAM.
// Build (production flags; CMake target gufo_k95small_bench):
//   hipcc -O2 -DNDEBUG -std=c++20 -fno-slp-vectorize --offload-arch=gfx1151 \
//     -DENGINE_ENABLE_HIP=1 -I. tools/qwen27b/k95small_bench.hip -o k95small_bench
// Run: k95small_bench [reps=200]
#define ENGINE_ENABLE_HIP 1
#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kernels/k95_small.hpp"

namespace {
using namespace gufo::hip;
using T = gufo::core::GgmlType;

void Check(hipError_t e) {
  if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}

// Host-registered read-only copy, the production GTT weight path.
struct Mapped {
  explicit Mapped(const std::vector<unsigned char>& src) : bytes(src.size()) {
    host = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host == MAP_FAILED) throw std::runtime_error("mmap");
    (void)madvise(host, bytes, MADV_HUGEPAGE);
    std::memcpy(host, src.data(), bytes);
    if (mprotect(host, bytes, PROT_READ) != 0) throw std::runtime_error("mprotect");
    Check(hipHostRegister(host, bytes,
                          hipHostRegisterMapped | hipHostRegisterReadOnly));
    Check(hipHostGetDevicePointer(&dev, host, 0));
  }
  ~Mapped() {
    (void)hipHostUnregister(host);
    (void)munmap(host, bytes);
  }
  const unsigned char* at(std::size_t off) const {
    return static_cast<const unsigned char*>(dev) + off;
  }
  void* host{};
  void* dev{};
  std::size_t bytes{};
};

template<class X> X* DevAlloc(std::size_t n) {
  X* p = nullptr;
  Check(hipMalloc(&p, n * sizeof(X)));
  Check(hipMemset(p, 0, n * sizeof(X)));
  return p;
}

std::mt19937 rng(20261025);
std::uint16_t Half(float v) {
  const _Float16 h = static_cast<_Float16>(v);
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
float Scale() { return std::uniform_real_distribution<float>(1e-4F, 2e-3F)(rng); }

// Random but well-formed blocks: random payload bytes, finite fp16 scales.
std::vector<unsigned char> RandomQuant(T type, std::size_t rows, std::size_t k,
                                       std::size_t copies) {
  const std::size_t bb = type == T::kQ5_K ? 176 : type == T::kQ6_K ? 210 : 34;
  const std::size_t per_block = type == T::kQ8_0 ? 32 : 256;
  const std::size_t blocks = rows * (k / per_block) * copies;
  std::vector<unsigned char> out(blocks * bb);
  std::uniform_int_distribution<int> byte(0, 255);
  for (auto& c : out) c = static_cast<unsigned char>(byte(rng));
  for (std::size_t b = 0; b < blocks; ++b) {
    unsigned char* p = out.data() + b * bb;
    const auto put = [&](std::size_t at, float v) {
      const std::uint16_t h = Half(v);
      std::memcpy(p + at, &h, 2);
    };
    if (type == T::kQ5_K) { put(0, Scale()); put(2, Scale()); }
    if (type == T::kQ6_K) put(208, Scale());
    if (type == T::kQ8_0) put(0, Scale());
  }
  return out;
}

std::vector<unsigned char> RandomF32(std::size_t n) {
  std::vector<unsigned char> out(n * 4);
  std::normal_distribution<float> nd(0.0F, 0.02F);
  for (std::size_t i = 0; i < n; ++i) {
    const float v = nd(rng);
    std::memcpy(out.data() + i * 4, &v, 4);
  }
  return out;
}

struct Timer {
  hipEvent_t a{}, b{};
  Timer() { Check(hipEventCreate(&a)); Check(hipEventCreate(&b)); }
  // Median of 5 trials of `reps` calls of f(rep); microseconds per call.
  template<class F> double Us(int reps, F&& f) {
    for (int r = 0; r < 20; ++r) f(r);
    std::vector<double> t;
    for (int trial = 0; trial < 5; ++trial) {
      Check(hipEventRecord(a));
      for (int r = 0; r < reps; ++r) f(r);
      Check(hipEventRecord(b));
      Check(hipEventSynchronize(b));
      float ms = 0;
      Check(hipEventElapsedTime(&ms, a, b));
      t.push_back(1000.0 * ms / reps);
    }
    std::sort(t.begin(), t.end());
    return t[2];
  }
};

bool Same(const float* d0, const float* d1, std::size_t n, const char* what) {
  std::vector<float> a(n), b(n);
  Check(hipMemcpy(a.data(), d0, n * 4, hipMemcpyDeviceToHost));
  Check(hipMemcpy(b.data(), d1, n * 4, hipMemcpyDeviceToHost));
  if (std::memcmp(a.data(), b.data(), n * 4) == 0) return true;
  std::size_t bad = 0;
  double worst = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (std::memcmp(&a[i], &b[i], 4) != 0) {
      ++bad;
      worst = std::max(worst, static_cast<double>(std::abs(a[i] - b[i])));
    }
  std::printf("MISMATCH %s: %zu of %zu floats differ, max |d| %.3g\n", what, bad,
              n, worst);
  return false;
}

double Gbs(double bytes, double us) { return bytes / (us * 1e3); }

// gemv.hip's FastGEMVBlockKernel, verbatim (runtime a_type, as served).
__global__ void ServedFastGemv(const void* __restrict__ A, T a_type,
                               const float* __restrict__ x,
                               float* __restrict__ y, std::size_t M,
                               std::size_t K) {
  FastGEMVBlockBody(A, a_type, x, y, M, K, blockIdx.x, blockIdx.y);
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
  constexpr std::uint32_t kK = 5120, kKv = 1024, kRank = 48;
  bool ok = true;
  Timer timer;
  hipStream_t st = nullptr;

  // ---- activations + quantization ----
  float* x = DevAlloc<float>(8 * kK);
  {
    std::vector<float> h(8 * kK);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (auto& v : h) v = nd(rng);
    Check(hipMemcpy(x, h.data(), h.size() * 4, hipMemcpyHostToDevice));
  }
  auto* xq = DevAlloc<std::int8_t>(8 * kK);
  auto* xds = DevAlloc<float2>(8 * kK / 16);
  const auto quant = [&](std::uint32_t batch) {
    const std::uint32_t subs = batch * (kK / 16);
    QuantizeActivationsDot4Kernel<<<(subs + 255) / 256, 256, 0, st>>>(x, xq, xds,
                                                                      subs);
  };
  const auto quant_k = [&](std::uint32_t batch) {
    const std::uint32_t subs = batch * (kK / 16);
    QuantizeActivationsQ8KDot4Kernel<<<(subs + 255) / 256, 256, 0, st>>>(
        x, xq, xds, subs);
  };
  for (std::uint32_t b : {1U, 8U}) {
    const double u = timer.Us(reps, [&](int) { quant(b); });
    const double uk = timer.Us(reps, [&](int) { quant_k(b); });
    std::printf("q B%u quant_us=%.2f quant_q8k_us=%.2f\n", b, u, uk);
  }

  // ---- attention K + V, width 8 ----
  {
    const std::size_t kb = std::size_t{kKv} * (kK / 256) * 176;
    const std::size_t vb = std::size_t{kKv} * (kK / 256) * 210;
    const std::size_t copies = 12;  // 12 x 7.9 MB = 95 MB > 3 x MALL
    const Mapped wk(RandomQuant(T::kQ5_K, kKv, kK, copies));
    const Mapped wv(RandomQuant(T::kQ6_K, kKv, kK, copies));
    float* yk0 = DevAlloc<float>(8 * kKv);
    float* yv0 = DevAlloc<float>(8 * kKv);
    float* yk1 = DevAlloc<float>(8 * kKv);
    float* yv1 = DevAlloc<float>(8 * kKv);
    const auto served_k = [&](std::size_t c, float* y) {
      SmallBatchDot4KQuantV3Kernel<T::kQ5_K, 8, 4, 4, 2>
          <<<(kKv + 31) / 32, 128, 0, st>>>(wk.at(c * kb), xq, xds, y, kKv, kK);
    };
    const auto served_v = [&](std::size_t c, float* y) {
      SmallBatchDot4LdsGEMMKernel<T::kQ6_K, 8, 2, 8, 1>
          <<<(kKv + 15) / 16, 256, 0, st>>>(wv.at(c * vb), xq, xds, y, kKv, kK);
    };
    quant(8);
    served_k(0, yk0);
    served_v(0, yv0);
    const double uk = timer.Us(reps, [&](int r) { served_k(r % copies, yk0); });
    const double uv = timer.Us(reps, [&](int r) { served_v(r % copies, yv0); });
    const double ug = timer.Us(reps, [&](int r) {
      quant(8); served_k(r % copies, yk0); quant(8); served_v(r % copies, yv0);
    });
    std::printf("kv B8 served: K_us=%.2f (%.1f GB/s) V_us=%.2f (%.1f GB/s) "
                "group_us=%.2f (2 quant + K + V, %.1f GB/s)\n",
                uk, Gbs(kb, uk), uv, Gbs(vb, uv), ug, Gbs(kb + vb, ug));
    // Reference outputs from copy 0 for the bitwise check.
    quant(8);
    served_k(0, yk0);
    served_v(0, yv0);
    const auto arm = [&]<std::uint32_t W, std::uint32_t KRpl, std::size_t VRpw>(
                         const char* name) {
      const auto run = [&](std::size_t c) {
        LaunchK95Kv<W, KRpl, VRpw>(wk.at(c * kb), wv.at(c * vb), xq, xds, yk1,
                                   yv1, kKv, kK, st);
      };
      Check(hipMemset(yk1, 0xFF, 8 * kKv * 4));
      Check(hipMemset(yv1, 0xFF, 8 * kKv * 4));
      run(0);
      Check(hipDeviceSynchronize());
      const bool exact = Same(yk0, yk1, 8 * kKv, name) &
                         Same(yv0, yv1, 8 * kKv, name);
      ok &= exact;
      const double u = timer.Us(reps, [&](int r) { run(r % copies); });
      std::printf("kv B8 k95 %s: us=%.2f (%.1f GB/s) exact=%d saves_us=%.2f "
                  "vs served group\n",
                  name, u, Gbs(kb + vb, u), exact ? 1 : 0, ug - u);
    };
    arm.template operator()<16, 1, 2>("w16_k1_v2(served)");
    arm.template operator()<8, 1, 2>("w8_k1_v2");
    arm.template operator()<8, 2, 4>("w8_k2_v4");
    arm.template operator()<16, 2, 1>("w16_k2_v1");
    arm.template operator()<4, 1, 1>("w4_k1_v1");
  }

  // ---- SSM controls: alpha F32 + beta Q8_0, widths 1-8 except 6 ----
  {
    const std::size_t ab = std::size_t{kRank} * kK * 4;
    const std::size_t bb = std::size_t{kRank} * (kK / 32) * 34;
    const std::size_t copies = 80;  // 80 x 1.24 MB = 100 MB
    const Mapped wa(RandomF32(std::size_t{kRank} * kK * copies));
    const Mapped wb(RandomQuant(T::kQ8_0, kRank, kK, copies));
    float* ya0 = DevAlloc<float>(8 * kRank);
    float* yb0 = DevAlloc<float>(8 * kRank);
    float* ya1 = DevAlloc<float>(8 * kRank);
    float* yb1 = DevAlloc<float>(8 * kRank);
    for (std::uint32_t b : {1U, 2U, 3U, 4U, 5U, 7U, 8U}) {
      const auto served = [&](std::size_t c) {
        // LaunchGEMVRows (kHipF32Baseline256) then the Q8_0 <8,1,1> route.
        ServedFastGemv<<<dim3(kRank, b), 256, 0, st>>>(wa.at(c * ab), T::kF32,
                                                       x, ya0, kRank, kK);
        SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 1, 1>
            <<<dim3((kRank + 7) / 8, b), 256, 0, st>>>(wb.at(c * bb), x, yb0,
                                                        kRank, kK);
      };
      const auto bodies = [&](std::size_t c) {
        LaunchK95SsmControls(reinterpret_cast<const float*>(wa.at(c * ab)),
                             wb.at(c * bb), x, ya1, yb1, b, kRank, kK, st);
      };
      const auto preload = [&](std::size_t c) {
        LaunchK95SsmControlsPreload(
            reinterpret_cast<const float*>(wa.at(c * ab)), wb.at(c * bb), x,
            ya1, yb1, b, kRank, st);
      };
      const double us = timer.Us(reps, [&](int r) { served(r % copies); });
      std::printf("ctl B%u served_us=%.2f (%.1f GB/s)\n", b, us,
                  Gbs(ab + bb, us));
      served(0);  // reference from copy 0, after the timed loop
      Check(hipDeviceSynchronize());
      const auto arm = [&](const char* tag, const auto& run) {
        Check(hipMemset(ya1, 0xFF, 8 * kRank * 4));
        Check(hipMemset(yb1, 0xFF, 8 * kRank * 4));
        run(0);
        Check(hipDeviceSynchronize());
        const std::string name = "ctl B" + std::to_string(b) + " " + tag;
        const bool exact = Same(ya0, ya1, b * kRank, name.c_str()) &
                           Same(yb0, yb1, b * kRank, name.c_str());
        ok &= exact;
        const double u = timer.Us(reps, [&](int r) { run(r % copies); });
        std::printf("ctl B%u k95 %s: us=%.2f (%.1f GB/s) exact=%d "
                    "saves_us=%.2f\n",
                    b, tag, u, Gbs(ab + bb, u), exact ? 1 : 0, us - u);
      };
      arm("bodies", bodies);
      arm("preload(served)", preload);
    }
  }
  std::printf("%s\n", ok ? "ALL EXACT" : "NOT EXACT");
  return ok ? 0 : 1;
}

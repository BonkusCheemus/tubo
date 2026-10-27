// normq GPU op test: GUFO_DEV_NORMQ=1 is bitwise equal to the unfused norm +
// activation quantize, on both quantizer kinds, at widths 1..8, k = 5120.
//
// Written 2026-10-25.
//
// The env switches are read once into function-local statics, so the two arms
// cannot live in one process: this binary re-execs ITSELF twice, dumps the
// float norm output and both consumers' GEMM outputs, and compares the two
// dumps byte for byte. The GEMM output is a deterministic function of the int8
// activations and their scales, so an equal y over a 5120-wide reduction with
// distinct weights per row pins the quantized bytes; the direct byte-for-byte
// comparison of xq/xds lives in a CPU proof that was run when the kernel was written.
//
// Exit 0 = pass, 1 = mismatch, 77 = no usable GPU (CTest SKIP).

#include <sys/wait.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"

namespace {

constexpr std::size_t kK = 5120;
// The served width-8 cells of each quantizer kind (prefill_quant_gemm.hip
// kTable): attn_q {Q4_K, 12288, 5120, 8} act_k 0, attn_qkv
// {Q4_K, 10240, 5120, 8} act_k 1.
constexpr std::size_t kMPlain = 12288;
constexpr std::size_t kMActk = 10240;
constexpr std::size_t kQ4KBlockBytes = 144;

#define CHECK(expr)                                                          \
  do {                                                                       \
    const hipError_t err_ = (expr);                                          \
    if (err_ != hipSuccess) {                                                \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #expr,    \
                   hipGetErrorString(err_));                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (false)

std::uint16_t FloatToHalf(float v) {
  // Only the few positive, normal scales below go through here.
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, 4);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const int exp = static_cast<int>((bits >> 23) & 0xFFU) - 127 + 15;
  const std::uint32_t mant = (bits >> 13) & 0x3FFU;
  if (exp <= 0 || exp >= 31) return static_cast<std::uint16_t>(sign);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp)
                                            << 10) |
                                    mant);
}

// A Q4_K weight blob with plausible scales, so the GEMM output is finite and
// its bytes depend on every activation byte.
std::vector<std::uint8_t> MakeQ4K(std::size_t m, std::size_t k, unsigned seed) {
  const std::size_t blocks = m * (k / 256);
  std::vector<std::uint8_t> w(blocks * kQ4KBlockBytes);
  std::mt19937 rng(seed);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* p = w.data() + b * kQ4KBlockBytes;
    const std::uint16_t d = FloatToHalf(0.01F + 0.001F * (b % 8));
    const std::uint16_t dmin = FloatToHalf(0.002F + 0.0005F * (b % 5));
    std::memcpy(p, &d, 2);
    std::memcpy(p + 2, &dmin, 2);
    for (int i = 0; i < 12; ++i)
      p[4 + i] = static_cast<std::uint8_t>(rng() & 0x3FU);
    for (int i = 0; i < 128; ++i)
      p[16 + i] = static_cast<std::uint8_t>(rng() & 0xFFU);
  }
  return w;
}

struct Dump {
  std::vector<float> normed, y_plain, y_actk;
};

// One arm: norm then the two consumers, for every width 1..8.
void RunArm(const std::string& path) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
    std::exit(77);

  const std::vector<std::uint8_t> w_plain = MakeQ4K(kMPlain, kK, 11U);
  const std::vector<std::uint8_t> w_actk = MakeQ4K(kMActk, kK, 22U);

  std::mt19937 rng(4242U);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> h_x(8 * kK), h_g(kK);
  for (auto& v : h_x) v = normal(rng);
  for (auto& v : h_g) v = 0.5F + 0.25F * normal(rng);
  // Corners the CPU proof also covers, inside the random field.
  h_x[0] = 0.0F;
  h_x[1] = 3.0e30F;
  h_x[kK + 7] = 7.0e-44F;
  h_g[333] = 0.0F;
  // The clamp's NaN corner, which the CPU proof cannot reach: the reference
  // quantizer's fminf(fmaxf(v, -127), 127) compiles to v_med3_f32 and the
  // fused kernel's to v_maxmin_f32. Both are exact selects, and the RDNA3 ISA
  // makes both return -127 for a NaN input, but that is a reading of the
  // doc, not a measurement -- so drive a NaN in. One plain 16-group gets a
  // denormal amax (id = 127/denormal = inf) and one zero element inside it,
  // so that element's normed * id is 0 * inf = NaN.
  for (std::size_t j = 0; j < 16; ++j)
    h_g[(16 * 9) + j] = 1.0e-40F;  // denormal weights -> denormal normed
  h_g[(16 * 9) + 5] = 0.0F;        // ... and one exact zero in the group

  void *d_wp = nullptr, *d_wa = nullptr;
  float *d_x = nullptr, *d_g = nullptr, *d_normed = nullptr, *d_yp = nullptr,
        *d_ya = nullptr;
  CHECK(hipMalloc(&d_wp, w_plain.size()));
  CHECK(hipMalloc(&d_wa, w_actk.size()));
  CHECK(hipMalloc(&d_x, 8 * kK * sizeof(float)));
  CHECK(hipMalloc(&d_g, kK * sizeof(float)));
  CHECK(hipMalloc(&d_normed, 8 * kK * sizeof(float)));
  CHECK(hipMalloc(&d_yp, 8 * kMPlain * sizeof(float)));
  CHECK(hipMalloc(&d_ya, 8 * kMActk * sizeof(float)));
  CHECK(hipMemcpy(d_wp, w_plain.data(), w_plain.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_wa, w_actk.data(), w_actk.size(), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_g, h_g.data(), h_g.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  // The fused norm needs its buffers before any capture; there is none here,
  // but call it on the same path the engine uses.
  gufo::hip::ReserveNormQScratch(8U * kK, nullptr);

  Dump dump;
  for (std::size_t batch = 1; batch <= 8; ++batch) {
    CHECK(hipMemset(d_normed, 0, 8 * kK * sizeof(float)));
    CHECK(hipMemset(d_yp, 0, 8 * kMPlain * sizeof(float)));
    CHECK(hipMemset(d_ya, 0, 8 * kMActk * sizeof(float)));
    gufo::hip::LaunchBatchedRMSNorm(d_x, d_g, d_normed, nullptr, batch, kK,
                                    1e-6F, nullptr);
    gufo::hip::LaunchBatchedQuantGEMMFp32(gufo::core::GgmlType::kQ4_K, d_wp,
                                          d_normed, d_yp, batch, kMPlain, kK,
                                          nullptr);
    gufo::hip::LaunchBatchedQuantGEMMFp32(gufo::core::GgmlType::kQ4_K, d_wa,
                                          d_normed, d_ya, batch, kMActk, kK,
                                          nullptr);
    CHECK(hipStreamSynchronize(nullptr));
    const std::size_t n = batch * kK;
    const std::size_t np = batch * kMPlain;
    const std::size_t na = batch * kMActk;
    const std::size_t o0 = dump.normed.size();
    const std::size_t p0 = dump.y_plain.size();
    const std::size_t a0 = dump.y_actk.size();
    dump.normed.resize(o0 + n);
    dump.y_plain.resize(p0 + np);
    dump.y_actk.resize(a0 + na);
    CHECK(hipMemcpy(dump.normed.data() + o0, d_normed, n * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(dump.y_plain.data() + p0, d_yp, np * sizeof(float),
                    hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(dump.y_actk.data() + a0, d_ya, na * sizeof(float),
                    hipMemcpyDeviceToHost));
  }

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    std::perror("fopen");
    std::exit(1);
  }
  for (const auto* v : {&dump.normed, &dump.y_plain, &dump.y_actk}) {
    const std::uint64_t count = v->size();
    std::fwrite(&count, sizeof(count), 1, f);
    std::fwrite(v->data(), sizeof(float), count, f);
  }
  std::fclose(f);
  std::fprintf(stderr, "arm dumped %zu+%zu+%zu floats to %s\n",
               dump.normed.size(), dump.y_plain.size(), dump.y_actk.size(),
               path.c_str());
}

std::vector<std::uint8_t> ReadAll(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return {};
  std::vector<std::uint8_t> bytes;
  std::uint8_t buf[1 << 16];
  std::size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0)
    bytes.insert(bytes.end(), buf, buf + got);
  std::fclose(f);
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--arm") == 0) {
    RunArm(argv[2]);
    return 0;
  }
  const char* tmp = std::getenv("TMPDIR");
  const std::string dir = tmp != nullptr ? tmp : "/tmp";
  const std::string off = dir + "/normq_off.bin";
  const std::string on = dir + "/normq_on.bin";
  const std::string self = argv[0];
  // Both arms keep the served KQUANT_V3 route; only NORMQ moves.
  const std::string base =
      "GUFO_DEV_KQUANT_V3=1 '" + self + "' --arm ";
  const int rc_off = std::system(("GUFO_DEV_NORMQ=0 " + base + off).c_str());
  if (rc_off != 0) {
    const int code = WIFEXITED(rc_off) ? WEXITSTATUS(rc_off) : 1;
    std::fprintf(stderr, "arm OFF exited %d\n", code);
    return code == 77 ? 77 : 1;
  }
  const int rc_on = std::system(("GUFO_DEV_NORMQ=1 " + base + on).c_str());
  if (rc_on != 0) {
    std::fprintf(stderr, "arm ON failed\n");
    return 1;
  }
  const auto a = ReadAll(off);
  const auto b = ReadAll(on);
  if (a.empty() || a.size() != b.size()) {
    std::fprintf(stderr, "FAIL dump sizes %zu vs %zu\n", a.size(), b.size());
    return 1;
  }
  if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
    std::size_t first = 0;
    while (first < a.size() && a[first] == b[first]) ++first;
    std::fprintf(stderr, "FAIL first differing byte at %zu of %zu\n", first,
                 a.size());
    return 1;
  }
  std::printf("normq: %zu bytes identical (B1..8, k=5120, plain + ActK)\n",
              a.size());
  return 0;
}

#else
int main() { return 77; }
#endif

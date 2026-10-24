// Turbo4 -> f16 dequant, the DEVICE half: the dequant kernel must produce
// the same scratch BYTES as the host range body.
//
// The CPU half (turbo4_dequant_f16_test.cpp) proves the body. This proves the
// KERNEL: the grid, the per-thread destination offset, and the latched status.
// A body can be right and a kernel still drop the h*head_dim term or use the
// wrong grid dimension, because the kernel computes the destination offset
// itself before calling the unit.
//
/*
   /opt/rocm/bin/hipcc -std=c++20 -DENGINE_ENABLE_HIP -I. -x hip \
       tests/models/qwen/turbo4_dequant_f16_gpu_test.cpp -o /tmp/e2gpu \
       -Lbuild-c1-turbo4-decode -lgufo_core
*/
//
// WHAT IS DELIBERATELY NOT TESTED, SAID OUT LOUD: the LAYER BASE.
// Turbo4BlockByteOffset takes no layer argument, so `k_plane`/`v_plane` are the
// caller's per-layer block base and nothing inside the body or the kernel can
// detect a wrong one. The `layer > 0` case here allocates a plane sized as if
// for a later layer and passes its base, which exercises the OFFSET ARITHMETIC
// at a non-zero layer; it does NOT verify that the arena hands out the right
// base for a given table. That is E3's job.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/turbo4_codec.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"
#include "src/models/qwen/hip/turbo4_plane_index.hpp"

namespace t4i = gufo::hip;
namespace t4 = gufo::hip::turbo4;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok)
    ++g_failures;
}

#define HIP_OK(expr)                                                      \
  do {                                                                    \
    const hipError_t _e = (expr);                                         \
    if (_e != hipSuccess) {                                               \
      std::printf("[FAIL] hip error %d at %s:%d\n", static_cast<int>(_e), \
                  __FILE__, __LINE__);                                    \
      ++g_failures;                                                       \
      return;                                                             \
    }                                                                     \
  } while (0)

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::size_t kKvWidth = kKvHeads * kHeadDim;

/// Order-preserving key for a half, so a bit-pattern difference is a DISTANCE.
/// NaN is not expected here and is not special-cased.
inline int OrderedHalf(std::uint16_t h) {
  const int mag = static_cast<int>(h & 0x7FFF);
  return (h & 0x8000) != 0 ? -mag : mag;
}

float Noise(std::size_t seed) {
  std::uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  const float unit = static_cast<float>(x >> 40) / 16777216.0f;
  return (unit - 0.5f) * 6.0f;
}

/// A plane filled by the real writer, so the blocks are ones the codec made.
std::vector<std::uint8_t> MakePlane(std::uint32_t context, std::uint32_t seed) {
  const std::size_t bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  std::vector<std::uint8_t> plane(bytes, 0);
  std::vector<float> src(kHeadDim);
  const std::size_t bph = kHeadDim / t4::kBlockValues;
  for (std::uint32_t t = 0; t < context; ++t) {
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      for (std::size_t i = 0; i < kHeadDim; ++i)
        src[i] = Noise(seed + t * 7919U + h * 104729U + i);
      std::size_t base = 0;
      if (!t4i::Turbo4BlockByteOffset(t, h, kHeadDim, kKvHeads, context, &base))
        continue;
      for (std::size_t b = 0; b < bph; ++b)
        (void)t4::QuantizeBlockChecked(
            src.data() + b * t4::kBlockValues,
            plane.data() + base + b * t4::kBlockBytes);
    }
  }
  return plane;
}

/// layer_index > 0 offsets the block base by whole planes, which is what a
/// layer table would hand a real caller.
void CompareOnDevice(std::uint32_t context, std::uint32_t count,
                     std::uint32_t layer_index) {
  const std::size_t plane_bytes =
      t4i::Turbo4LayerPlaneBytes(kHeadDim, kKvHeads, context);
  const std::vector<std::uint8_t> k_plane = MakePlane(context, 11);
  const std::vector<std::uint8_t> v_plane = MakePlane(context, 90210);
  // The device is handed the base of THIS layer's region inside a bigger
  // allocation, exactly as a layer table would.
  const std::size_t layer_base =
      static_cast<std::size_t>(layer_index) * plane_bytes;

  std::uint8_t *d_k_plane = nullptr, *d_v_plane = nullptr;
  std::uint16_t *d_k_dst = nullptr, *d_v_dst = nullptr;
  std::uint8_t* d_status = nullptr;
  HIP_OK(hipMalloc(&d_k_plane, plane_bytes * (layer_index + 1)));
  HIP_OK(hipMalloc(&d_v_plane, plane_bytes * (layer_index + 1)));
  // Copy into THIS LAYER'S SLOT, not slot 0: with layer_index > 0 the kernel
  // reads d_k_plane + layer_base, so writing the data at offset 0 would leave
  // it reading uninitialised memory and the host/device compare would fail for
  // a reason that has nothing to do with the kernel.
  HIP_OK(hipMemcpy(d_k_plane + layer_base, k_plane.data(), plane_bytes,
                   hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_v_plane + layer_base, v_plane.data(), plane_bytes,
                   hipMemcpyHostToDevice));

  const std::size_t plane_halfs = static_cast<std::size_t>(context) * kKvWidth;
  HIP_OK(hipMalloc(&d_k_dst, plane_halfs * sizeof(std::uint16_t)));
  HIP_OK(hipMalloc(&d_v_dst, plane_halfs * sizeof(std::uint16_t)));

  HIP_OK(hipMemset(d_k_dst, 0xAB, plane_halfs * sizeof(std::uint16_t)));
  HIP_OK(hipMemset(d_v_dst, 0xAB, plane_halfs * sizeof(std::uint16_t)));
  HIP_OK(hipMalloc(&d_status, 16));
  HIP_OK(hipMemset(d_status, 0, 16));

  const std::string tag = "context=" + std::to_string(context) +
                          " count=" + std::to_string(count) +
                          " layer=" + std::to_string(layer_index);

  const auto st = t4i::LaunchTurbo4DequantLayer(
      d_k_plane + layer_base, d_v_plane + layer_base, d_k_dst, d_v_dst, count,
      context, kKvHeads, kHeadDim,
      reinterpret_cast<t4i::Turbo4KernelStatus*>(d_status), nullptr,
      gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
  HIP_OK(hipDeviceSynchronize());

  // The host body, into HOST scratch. Turbo4DequantLayerBody is a host
  // function: handing it DEVICE pointers would dereference them on the host,
  // and the resulting garbage would look exactly like a kernel bug.
  // SAME sentinel as the device scratch (0xAB). Different sentinels would make
  // the untouched tail past count compare unequal for no reason, and comparing
  // the WHOLE plane with one sentinel is also what proves the kernel did not
  // scribble past count.
  // 0xABAB, not 0x00AB: hipMemset fills BYTES, so the device sentinel is
  // 0xABAB per element. A host vector initialised to 0xAB is 0x00AB and the
  // untouched tail then compares unequal for no reason.
  std::vector<std::uint16_t> h_k(plane_halfs, 0xABAB);
  std::vector<std::uint16_t> h_v(plane_halfs, 0xABAB);
  const auto host_st = t4i::Turbo4DequantLayerBody(
      k_plane.data(), v_plane.data(), h_k.data(), h_v.data(), count, context,
      kKvHeads, kHeadDim,
      gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));

  Check(st == t4i::Turbo4LaunchStatus::kLaunched,
        tag + ": launch reports kLaunched");
  Check(host_st == t4i::Turbo4KernelStatus::kOk,
        tag + ": host body returns kOk");

  std::vector<std::uint16_t> a(plane_halfs), b(plane_halfs);
  HIP_OK(hipMemcpy(a.data(), d_k_dst, plane_halfs * sizeof(std::uint16_t),
                   hipMemcpyDeviceToHost));
  b = h_k;
  std::size_t k_bad = 0;
  int max_ulp = 0;
  std::vector<std::size_t> bad_idx;  // the ACTUAL element indices
  for (std::size_t i = 0; i < plane_halfs; ++i) {
    if (a[i] == b[i])
      continue;
    ++k_bad;
    const int ulp = std::abs(OrderedHalf(a[i]) - OrderedHalf(b[i]));
    if (ulp > max_ulp)
      max_ulp = ulp;
    bad_idx.push_back(i);
  }
  // A cause, not a count: one ULP spread over many positions is rounding,
  // the same ULP on one head or one block is an addressing defect.
  std::vector<std::size_t> uniq;
  for (std::size_t i : bad_idx)
    uniq.push_back(i / kKvWidth);  // distinct POSITION
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  std::printf("  K: bad=%zu max_ulp=%d distinct(pos)=%zu", k_bad, max_ulp,
              uniq.size());
  if (!uniq.empty())
    std::printf(" first_pos=%zu", uniq.front());
  std::printf("\n");
  for (std::size_t n = 0; n < bad_idx.size() && n < 6; ++n) {
    const std::size_t i = bad_idx[n];
    const int ulp = std::abs(OrderedHalf(a[i]) - OrderedHalf(b[i]));
    std::printf(
        "    pos %zu head %zu elem %zu: host=0x%04X dev=0x%04X ulp=%d\n",
        i / kKvWidth, (i % kKvWidth) / kHeadDim, i % kHeadDim, b[i], a[i], ulp);
  }
  HIP_OK(hipMemcpy(a.data(), d_v_dst, plane_halfs * sizeof(std::uint16_t),
                   hipMemcpyDeviceToHost));
  b = h_v;
  std::size_t v_bad = 0;
  for (std::size_t i = 0; i < plane_halfs; ++i)
    if (a[i] != b[i])
      ++v_bad;

  Check(k_bad == 0 && v_bad == 0,
        tag + ": DEVICE scratch is BIT-EQUAL to the host body (k=" +
            std::to_string(k_bad) + " v=" + std::to_string(v_bad) + " bad)");

  // A wrong layer base must produce DIFFERENT bytes: proves the test is not
  // comparing two arrays that both happened to be untouched.
  if (layer_base != 0) {
    std::vector<std::uint8_t> other = MakePlane(context, 4242);
    std::uint8_t* d_other = nullptr;
    HIP_OK(hipMalloc(&d_other, plane_bytes * (layer_index + 1)));
    HIP_OK(hipMemset(d_other, 0, plane_bytes * (layer_index + 1)));
    HIP_OK(
        hipMemcpy(d_other, other.data(), plane_bytes, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_k_dst, 0xAB, plane_halfs * sizeof(std::uint16_t)));
    HIP_OK(hipMemset(d_status, 0, 16));
    (void)t4i::LaunchTurbo4DequantLayer(
        d_other + layer_base, d_v_plane + layer_base, d_k_dst, d_v_dst, count,
        context, kKvHeads, kHeadDim,
        reinterpret_cast<t4i::Turbo4KernelStatus*>(d_status), nullptr,
        gufo::hip::turbo4::PlaneFormats(gufo::hip::turbo4::Bits::k4));
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(a.data(), d_k_dst, plane_halfs * sizeof(std::uint16_t),
                     hipMemcpyDeviceToHost));
    b = h_k;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < plane_halfs; ++i)
      if (a[i] != b[i])
        ++differ;
    Check(differ > 0,
          tag + ": a different layer's blocks give different scratch bytes (" +
              std::to_string(differ) + " differ), so the compare has teeth");
    (void)hipFree(d_other);
  }

  (void)hipFree(d_k_plane);
  (void)hipFree(d_v_plane);
  (void)hipFree(d_k_dst);
  (void)hipFree(d_v_dst);
  (void)hipFree(d_status);
}

// Direct packed fixtures exercise every centroid and extreme norms without
// making these large old/new comparisons depend on the quantizer's cost.
std::vector<std::uint8_t> MakePackedPlane(std::size_t block_count,
                                          t4::PlaneFormats formats,
                                          t4::Bits bits, unsigned seed) {
  const std::size_t stride = t4::BlockBytes(formats.stride);
  std::vector<std::uint8_t> plane(block_count * stride, 0xD7);
  constexpr std::uint16_t norms[] = {0,      1,      0x03FF, 0x0400,
                                     0x3C00, 0x4C00, 0x7BFF, 0x8000};
  for (std::size_t b = 0; b < block_count; ++b) {
    auto* dst = plane.data() + b * stride;
    std::memset(dst, 0, t4::BlockBytes(bits));
    const std::uint16_t norm = norms[(b + seed) % 8];
    std::memcpy(dst, &norm, sizeof(norm));
    std::uint64_t noise = (b + 1) * 6364136223846793005ULL + seed;
    for (unsigned j = 0; j < 128; ++j) {
      noise ^= noise >> 12;
      noise ^= noise << 25;
      noise ^= noise >> 27;
      // The first 64 values cover every centroid. The rest distinguish high
      // token-address bits, so a repeated/aliased source row cannot pass.
      const unsigned code =
          (j < 64 ? j * 13U + seed
                  : unsigned((noise * 2685821657736338717ULL) >> 32)) &
          ((1U << static_cast<unsigned>(bits)) - 1U);
      dst[2 + j / 2] |= t4::QsNibbleOfCode(bits, code) << (4 * (j % 2));
      if (bits != t4::Bits::k4)
        dst[66 + (bits == t4::Bits::k5 ? j / 8 : j / 4)] |=
            t4::QhBitsOfCode(bits, code, j);
    }
  }
  return plane;
}

void CompareOldNew(std::uint32_t count, t4::PlaneFormats formats,
                   std::uint32_t heads = kKvHeads, std::uint32_t dim = kHeadDim,
                   unsigned src_prefix = 0, unsigned dst_prefix = 0) {
  // Some allocations end exactly at the last live row; others have one dirty
  // tail row. Prefixes test the old API's lack of an 8-byte alignment rule.
  const std::uint32_t context = count + (count % 32U != 0U ? 1U : 0U);
  const std::size_t block_count = std::size_t(context) * heads * (dim / 128);
  const auto k = MakePackedPlane(block_count, formats, formats.k, 11);
  const auto v = MakePackedPlane(block_count, formats, formats.v, 29);
  const std::size_t values = std::size_t(context) * heads * dim + dst_prefix;
  const std::size_t out_bytes = values * sizeof(std::uint16_t);
  const std::size_t live_end = dst_prefix + std::size_t(count) * heads * dim;
  const std::string tag =
      "old/new count=" + std::to_string(count) +
      " stride/K/V=" + std::to_string(unsigned(formats.stride)) + "/" +
      std::to_string(unsigned(formats.k)) + "/" +
      std::to_string(unsigned(formats.v)) +
      " heads/dim=" + std::to_string(heads) + "/" + std::to_string(dim) +
      " prefixes=" + std::to_string(src_prefix) + "/" +
      std::to_string(dst_prefix);
  std::uint8_t *dk = nullptr, *dv = nullptr;
  std::uint16_t *out_k = nullptr, *out_v = nullptr;
  t4i::Turbo4KernelStatus* status = nullptr;
  HIP_OK(hipMalloc(&dk, k.size() + src_prefix));
  HIP_OK(hipMalloc(&dv, v.size() + src_prefix));
  HIP_OK(hipMalloc(&out_k, out_bytes));
  HIP_OK(hipMalloc(&out_v, out_bytes));
  HIP_OK(hipMalloc(&status, sizeof(*status)));
  HIP_OK(hipMemcpy(dk + src_prefix, k.data(), k.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dv + src_prefix, v.data(), v.size(), hipMemcpyHostToDevice));

  std::vector<std::uint16_t> old_k(values), old_v(values), actual(values);
  for (unsigned version = 0; version < 2; ++version) {
    Check(setenv("GUFO_DEV_DEQ_V2", version ? "1" : "0", 1) == 0,
          "select dequant version");
    HIP_OK(hipMemset(out_k, 0xAB, out_bytes));
    HIP_OK(hipMemset(out_v, 0xAB, out_bytes));
    // A successful decode must leave both a clean and a failed latch alone.
    const auto initial = (count & 1U) ? t4i::Turbo4KernelStatus::kOk
                                      : t4i::Turbo4KernelStatus::kWriterFailed;
    HIP_OK(hipMemcpy(status, &initial, sizeof(initial), hipMemcpyHostToDevice));
    const auto launched = t4i::LaunchTurbo4DequantLayer(
        dk + src_prefix, dv + src_prefix, out_k + dst_prefix,
        out_v + dst_prefix, count, context, heads, dim, status, nullptr,
        formats);
    HIP_OK(hipDeviceSynchronize());
    Check(launched == t4i::Turbo4LaunchStatus::kLaunched, tag + ": launched");
    t4i::Turbo4KernelStatus latched;
    HIP_OK(hipMemcpy(&latched, status, sizeof(latched), hipMemcpyDeviceToHost));
    Check(latched == initial, tag + ": successful decode preserves latch");
    for (unsigned plane = 0; plane < 2; ++plane) {
      auto& reference = plane == 0 ? old_k : old_v;
      HIP_OK(hipMemcpy(actual.data(), plane == 0 ? out_k : out_v, out_bytes,
                       hipMemcpyDeviceToHost));
      Check(std::all_of(actual.begin(), actual.begin() + dst_prefix,
                        [](auto x) { return x == 0xABAB; }) &&
                std::all_of(actual.begin() + live_end, actual.end(),
                            [](auto x) { return x == 0xABAB; }),
            tag + ": prefix and tail untouched");
      Check(std::any_of(actual.begin() + dst_prefix, actual.begin() + live_end,
                        [](auto x) { return x != 0xABAB; }),
            tag + ": live output written");
      if (version == 0)
        reference = actual;
      else
        Check(actual == reference,
              tag + (plane == 0 ? ": K byte-equal" : ": V byte-equal"));
    }
  }
  HIP_OK(hipFree(dk));
  HIP_OK(hipFree(dv));
  HIP_OK(hipFree(out_k));
  HIP_OK(hipFree(out_v));
  HIP_OK(hipFree(status));
}

void CompareRefusals() {
  std::uint8_t* plane = nullptr;
  std::uint16_t* dst = nullptr;
  t4i::Turbo4KernelStatus* status = nullptr;
  HIP_OK(hipMalloc(&plane, 98));
  HIP_OK(hipMalloc(&dst, 256));
  HIP_OK(hipMalloc(&status, sizeof(*status)));
  struct Case {
    std::uint32_t count, context, heads, dim;
    t4::PlaneFormats formats;
    unsigned null_arg;  // K, V, K-dst, V-dst, status
    t4i::Turbo4LaunchStatus expected;
  };
  const t4::PlaneFormats normal(t4::Bits::k6);
  using L = t4i::Turbo4LaunchStatus;
  const Case cases[] = {
      {0, 1, 1, 128, normal, 0, L::kBadArguments},
      {2, 1, 1, 128, normal, 0, L::kGeometryRefused},
      {1, 0, 1, 128, normal, 0, L::kGeometryRefused},
      {1, 1, 0, 128, normal, 0, L::kGeometryRefused},
      {1, 1, 1, 0, normal, 0, L::kGeometryRefused},
      {1, 1, 1, 127, normal, 0, L::kGeometryRefused},
      {1, 1, 1, 129, normal, 0, L::kGeometryRefused},
      {1,
       1,
       1,
       128,
       {t4::Bits::k4, t4::Bits::k5, t4::Bits::k4},
       0,
       L::kGeometryRefused},
      {1,
       1,
       1,
       128,
       {t4::Bits::k5, t4::Bits::k4, t4::Bits::k6},
       0,
       L::kGeometryRefused},
      {1, 1, 1, 128, t4::PlaneFormats(t4::Bits(3)), 0, L::kGeometryRefused},
      {1,
       1,
       1,
       128,
       {t4::Bits::k6, t4::Bits(7), t4::Bits::k4},
       0,
       L::kGeometryRefused},
      {1,
       1,
       1,
       128,
       {t4::Bits::k6, t4::Bits::k4, t4::Bits(0)},
       0,
       L::kGeometryRefused},
      {1, 1, 1, 128, normal, 1, L::kBadArguments},
      {1, 1, 1, 128, normal, 2, L::kBadArguments},
      {1, 1, 1, 128, normal, 3, L::kBadArguments},
      {1, 1, 1, 128, normal, 4, L::kBadArguments},
      {1, 1, 1, 128, normal, 5, L::kBadArguments},
  };
  for (const char* mode : {"0", "1"}) {
    Check(setenv("GUFO_DEV_DEQ_V2", mode, 1) == 0, "select refusal version");
    for (const auto initial : {t4i::Turbo4KernelStatus::kOk,
                               t4i::Turbo4KernelStatus::kWriterFailed}) {
      for (const auto& c : cases) {
        HIP_OK(hipMemset(dst, 0xAB, 256));
        HIP_OK(hipMemcpy(status, &initial, sizeof(initial),
                         hipMemcpyHostToDevice));
        const auto result = t4i::LaunchTurbo4DequantLayer(
            c.null_arg == 1 ? nullptr : plane,
            c.null_arg == 2 ? nullptr : plane, c.null_arg == 3 ? nullptr : dst,
            c.null_arg == 4 ? nullptr : dst, c.count, c.context, c.heads, c.dim,
            c.null_arg == 5 ? nullptr : status, nullptr, c.formats);
        Check(result == c.expected, "same host refusal in both versions");
        t4i::Turbo4KernelStatus actual;
        HIP_OK(
            hipMemcpy(&actual, status, sizeof(actual), hipMemcpyDeviceToHost));
        Check(actual == initial, "host refusal never writes device latch");
        std::uint16_t output[128];
        HIP_OK(hipMemcpy(output, dst, sizeof(output), hipMemcpyDeviceToHost));
        Check(std::all_of(std::begin(output), std::end(output),
                          [](auto x) { return x == 0xABAB; }),
              "host refusal leaves output untouched");
      }
    }
  }
  HIP_OK(hipFree(plane));
  HIP_OK(hipFree(dst));
  HIP_OK(hipFree(status));
}

}  // namespace

int main() {
  const char* original_env = std::getenv("GUFO_DEV_DEQ_V2");
  const bool had_env = original_env != nullptr;
  const std::string saved_env = had_env ? original_env : "";
  Check(unsetenv("GUFO_DEV_DEQ_V2") == 0, "select original default");
  std::printf("E2 device: dequant kernel vs host body, bit-exact\n");
  for (unsigned version = 0; version < 2; ++version) {
    Check((version ? setenv("GUFO_DEV_DEQ_V2", "1", 1)
                   : unsetenv("GUFO_DEV_DEQ_V2")) == 0,
          "select host-oracle version");
    // counts that are not multiples of anything in particular
    for (std::uint32_t count : {1U, 3U, 5U, 17U, 33U, 64U})
      CompareOnDevice(256, count, 0U);
    CompareOnDevice(256, 256, 0U);   // count == max_context
    CompareOnDevice(1024, 512, 0U);  // deeper context, count == max_context/2
    CompareOnDevice(256, 5U, 1U);    // layer > 0
    CompareOnDevice(256, 64U, 3U);   // layer > 0, larger count
  }
  for (unsigned stride = 4; stride <= 6; ++stride)
    for (unsigned k = 4; k <= stride; ++k)
      for (unsigned v = 4; v <= stride; ++v)
        for (std::uint32_t count :
             {1U, 31U, 32U, 33U, 4095U, 4096U, 4097U, 33712U, 40960U})
          CompareOldNew(count, {t4::Bits(stride), t4::Bits(k), t4::Bits(v)});
  for (std::uint32_t dim : {128U, 384U, 1024U, 1152U, 2048U})
    for (std::uint32_t heads : {1U, 3U, 7U})
      CompareOldNew(33, {t4::Bits::k6, t4::Bits::k5, t4::Bits::k4}, heads, dim);
  for (unsigned offset : {1U, 2U, 3U})
    CompareOldNew(31, t4::PlaneFormats(t4::Bits::k6), 3, 384, offset, offset);
  CompareRefusals();
  Check((had_env ? setenv("GUFO_DEV_DEQ_V2", saved_env.c_str(), 1)
                 : unsetenv("GUFO_DEV_DEQ_V2")) == 0,
        "restore dequant environment");
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
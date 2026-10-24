// Oracle test for the masked WMMA prefill attention (opt-c177-attn-wmma).
//
// The WMMA kernel replaces the tiled `v_dot2_f32_f16` kernel over the same
// visible key range, with the same FP16 K/V cache, the same causal rule and the
// same gate epilogue. Both carry FP16 operands, so they agree to FP16 output
// precision rather than exactly; a CPU reference in tools/bench puts each of
// them at 2.1e-4 against double precision, so the two agreeing to the same
// order is the correct expectation here.
//
// Covered: a depth-0 chunk, a batch that is not a multiple of the 32-query
// block, a chunk at depth, a depth that is not a multiple of the 16-key tile,
// and the log-sum-exp output path.
//
// `opt-c178-attn-prefetch` reads tile i+1's K and V into a second register set
// while tile i computes, so the cases below also cover the loop lengths where
// that can come apart -- a visible range of exactly one key tile, where the
// prefetch never fires, and of exactly two, where it fires once and the tail
// guard has to suppress it. Every case additionally runs the kernel twice and
// requires the two outputs to be byte-identical, which is what would fail if a
// prefetched register were consumed for the wrong tile or read before it was
// written: stale register contents differ between dispatches, an arithmetic
// mistake would not.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kNumHeads = 24;
constexpr std::uint32_t kNumKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  float Uniform(float lo, float hi) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float u = static_cast<float>((state_ >> 8) & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }

private:
  std::uint32_t state_;
};

void Compare(const char* label, const std::vector<float>& got,
             const std::vector<float>& want, double tolerance) {
  double max_abs = 0.0;
  double magnitude = 0.0;
  std::size_t non_finite = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i])) {
      ++non_finite;
      continue;
    }
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                           static_cast<double>(want[i])));
    magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
  }
  const double rel = (magnitude > 0.0) ? (max_abs / magnitude) : max_abs;
  std::cout << "  " << label << ": max_abs=" << max_abs
            << " magnitude=" << magnitude << " rel=" << rel
            << " non_finite=" << non_finite << "\n";
  if (non_finite != 0 || rel > tolerance) {
    std::cerr << label
              << ": WMMA prefill attention disagrees with the tiled kernel\n";
    std::abort();
  }
}

void RunCase(std::uint32_t start_pos, std::size_t batch_size, bool want_lse,
             std::uint32_t key_begin = 0) {
  std::cout << "wmma prefill attention: start_pos=" << start_pos
            << " batch=" << batch_size << " lse=" << (want_lse ? "yes" : "no")
            << "\n";

  const auto kMaxContext = std::max<std::uint32_t>(
      1024, start_pos + static_cast<std::uint32_t>(batch_size) + 64);
  const std::size_t attention_size =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t kv_size = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t total_kv =
      static_cast<std::size_t>(kNumKvHeads) * kMaxContext * kHeadDim;
  const std::size_t q_elements = batch_size * attention_size;
  const std::size_t lse_elements = batch_size * kNumHeads;

  Rng rng(0xA11CEU ^ (start_pos * 131U) ^
          static_cast<std::uint32_t>(batch_size));
  std::vector<float> h_q(q_elements);
  std::vector<float> h_gate(q_elements);
  std::vector<float> h_k(batch_size * kv_size);
  std::vector<float> h_v(batch_size * kv_size);
  std::vector<float> h_cache(total_kv * 2);
  for (auto& x : h_q) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_gate) {
    x = rng.Uniform(-2.0F, 2.0F);
  }
  for (auto& x : h_k) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_v) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_cache) {
    x = rng.Uniform(-1.0F, 1.0F);
  }

  float* d_q = nullptr;
  float* d_gate = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_cache = nullptr;
  void* d_cache_f16 = nullptr;
  float* d_out_ref = nullptr;
  float* d_out_new = nullptr;
  float* d_lse_ref = nullptr;
  float* d_lse_new = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_f16, total_kv * 2 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_new, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_lse_ref, lse_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_lse_new, lse_elements * sizeof(float)));

  const auto upload = [](float* dst, const std::vector<float>& src) {
    HIP_CHECK(hipMemcpy(dst, src.data(), src.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  };
  upload(d_q, h_q);
  upload(d_gate, h_gate);
  upload(d_k, h_k);
  upload(d_v, h_v);
  upload(d_cache, h_cache);
  HIP_CHECK(hipMemset(d_cache_f16, 0, total_kv * 2 * sizeof(std::uint16_t)));

  auto* v_cache = d_cache + total_kv;
  auto* cache_f16_v = static_cast<std::uint16_t*>(d_cache_f16) + total_kv;
  float* lse_ref = want_lse ? d_lse_ref : nullptr;
  float* lse_new = want_lse ? d_lse_new : nullptr;

  // Reference: the tiled kernel, which also packs the FP16 cache.
  HIP_CHECK(hipMemset(d_out_ref, 0, q_elements * sizeof(float)));
  if (!gufo::hip::LaunchBatchedAttentionTile(
          d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out_ref, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, lse_ref, key_begin,
          false)) {
    std::cerr << "tiled attention rejected the production shape\n";
    std::abort();
  }

  // Candidate: the WMMA kernel over the same range. The cache is already
  // packed, so suppress the write and prove it reads the same bytes.
  HIP_CHECK(hipMemset(d_out_new, 0, q_elements * sizeof(float)));
  if (!gufo::hip::LaunchQwenWmmaAttention(
          d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out_new, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin,
          /*skip_kv_write=*/true)) {
    std::cerr << "WMMA attention rejected the production shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref(q_elements);
  std::vector<float> got(q_elements);
  HIP_CHECK(hipMemcpy(ref.data(), d_out_ref, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got.data(), d_out_new, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  Compare("context", got, ref, 2e-3);

  // Independent FP64 attention over the original FP32 inputs, before the
  // kernel rounds Q/K/V and probabilities. Include the first and last query
  // of head zero so partial causal tiles cannot hide behind kernel agreement.
  std::vector<float> oracle, selected;
  for (const auto row : {std::size_t{0}, batch_size - 1}) {
    const auto end = start_pos + row + 1;
    std::vector<double> scores(end, -INFINITY);
    double maximum = -INFINITY;
    for (std::size_t key = key_begin; key < end; ++key) {
      double dot = 0;
      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        const float value = key < start_pos
                                ? h_cache[key * kHeadDim + dim]
                                : h_k[(key - start_pos) * kv_size + dim];
        dot += double(h_q[row * attention_size + dim]) * value;
      }
      scores[key] = dot / std::sqrt(double(kHeadDim));
      maximum = std::max(maximum, scores[key]);
    }
    double sum = 0;
    for (std::size_t key = key_begin; key < end; ++key) {
      scores[key] = std::exp(scores[key] - maximum);
      sum += scores[key];
    }
    for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
      double value = 0;
      for (std::size_t key = key_begin; key < end; ++key) {
        const float v = key < start_pos
                            ? h_cache[total_kv + key * kHeadDim + dim]
                            : h_v[(key - start_pos) * kv_size + dim];
        value += scores[key] * v;
      }
      value = sum > 0 ? value / sum : 0;
      if (!want_lse)
        value /= 1 + std::exp(-double(h_gate[row * attention_size + dim]));
      oracle.push_back(static_cast<float>(value));
      selected.push_back(got[row * attention_size + dim]);
    }
  }
  Compare("FP64 original-input oracle", selected, oracle, 2e-3);

  // Same launch again: the prefetch must not let a register from one tile reach
  // another, which would show up here as run-to-run drift.
  HIP_CHECK(hipMemset(d_out_new, 0, q_elements * sizeof(float)));
  if (!gufo::hip::LaunchQwenWmmaAttention(
          d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
          d_out_new, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin,
          /*skip_kv_write=*/true)) {
    std::cerr << "WMMA attention rejected the production shape on replay\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> again(q_elements);
  HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) != 0) {
    std::size_t differing = 0;
    for (std::size_t i = 0; i < q_elements; ++i) {
      differing += (again[i] != got[i]) ? 1 : 0;
    }
    std::cerr << "WMMA prefill attention is not deterministic: " << differing
              << " of " << q_elements << " elements differ between two runs\n";
    std::abort();
  }
  std::cout << "  replay: byte-identical over " << q_elements << " elements\n";

  if (!want_lse && batch_size > 1) {
    // Populate future KV once, then vary only the chunk boundaries. A masked
    // future value must have exactly the same effect as absent padding.
    for (const std::size_t budget : {std::size_t{31}, std::size_t{512}}) {
      for (std::size_t offset = 0; offset < batch_size; offset += budget) {
        const auto count = std::min(budget, batch_size - offset);
        if (!gufo::hip::LaunchQwenWmmaAttention(
                d_q + offset * attention_size, d_k, d_v,
                d_gate + offset * attention_size, d_cache, v_cache, d_cache_f16,
                cache_f16_v, d_out_new + offset * attention_size, 0,
                start_pos + offset, count, kMaxContext, kNumHeads, kNumKvHeads,
                kHeadDim, nullptr, nullptr, key_begin, true))
          std::abort();
      }
      HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                          hipMemcpyDeviceToHost));
      if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) !=
          0) {
        std::cerr << "WMMA attention changed across chunk boundaries: budget="
                  << budget << '\n';
        for (std::size_t i = 0; i < got.size(); ++i)
          if (again[i] != got[i]) {
            std::cerr << "first row=" << i / attention_size
                      << " dim=" << i % attention_size
                      << " got=" << std::hexfloat << again[i]
                      << " expected=" << got[i] << std::defaultfloat << '\n';
            break;
          }
        std::abort();
      }
    }
    std::cout << "  chunk boundaries: byte-identical\n";
  }

  if (start_pos >= 8192 || batch_size >= 1024) {
    const std::size_t length = start_pos + batch_size;
    const std::array<std::size_t, 2> words{
        length * kv_size / 2,
        ((length + 15u) & ~std::size_t{15}) * kv_size / 2};
    std::array<float*, 2> storage{};
    std::array<std::span<float>, 2> work{};
    for (std::size_t i = 0; i < storage.size(); ++i) {
      HIP_CHECK(hipMalloc(&storage[i], (words[i] + 32) * sizeof(float)));
      HIP_CHECK(hipMemset(storage[i], 0xA5, (words[i] + 32) * sizeof(float)));
      work[i] = {storage[i] + 16, words[i]};
    }
    std::vector<float> lse_expected(lse_elements);
    if (want_lse)
      HIP_CHECK(hipMemcpy(lse_expected.data(), d_lse_new,
                          lse_elements * sizeof(float), hipMemcpyDeviceToHost));
    for (const auto [key_heads, value_heads] :
         {std::pair{4u, 4u}, std::pair{3u, 4u}, std::pair{1u, 4u},
          std::pair{4u, 1u}, std::pair{2u, 3u}, std::pair{3u, 2u},
          std::pair{0u, 4u}, std::pair{4u, 0u}}) {
      for (std::size_t i = 0; i < storage.size(); ++i)
        HIP_CHECK(hipMemset(storage[i], 0xA5, (words[i] + 32) * sizeof(float)));
      const std::array<std::size_t, 2> sizes{
          key_heads ? words[0] / kNumKvHeads * key_heads
                    : words[0] / kNumKvHeads - 1,
          value_heads ? words[1] / kNumKvHeads * value_heads
                      : words[1] / kNumKvHeads - 1};
      const auto k_work = work[0].first(sizes[0]);
      const auto v_work = work[1].first(sizes[1]);
      if (!gufo::hip::LaunchQwenWmmaAttention(
              d_q, d_k, d_v, d_gate, d_cache, v_cache, d_cache_f16, cache_f16_v,
              d_out_new, 0, start_pos, batch_size, kMaxContext, kNumHeads,
              kNumKvHeads, kHeadDim, nullptr, lse_new, key_begin, true, k_work,
              v_work))
        std::abort();
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(again.data(), d_out_new, q_elements * sizeof(float),
                          hipMemcpyDeviceToHost));
      if (std::memcmp(again.data(), got.data(), q_elements * sizeof(float)) !=
          0)
        std::abort();
      if (want_lse) {
        std::vector<float> actual(lse_elements);
        HIP_CHECK(hipMemcpy(actual.data(), d_lse_new,
                            lse_elements * sizeof(float),
                            hipMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), lse_expected.data(),
                        lse_elements * sizeof(float)) != 0)
          std::abort();
      }
      for (std::size_t i = 0; i < storage.size(); ++i) {
        float* data = storage[i];
        std::array<std::uint8_t, 64> first{};
        HIP_CHECK(hipMemcpy(first.data(), data + 16, first.size(),
                            hipMemcpyDeviceToHost));
        const bool untouched = std::all_of(
            first.begin(), first.end(), [](auto byte) { return byte == 0xA5; });
        if (untouched ==
            (key_heads != 0 && value_heads != 0 && batch_size >= 1024 &&
             key_begin < length && key_begin % 16 == 0))
          std::abort();
        for (std::size_t off : {std::size_t{0}, sizes[i] + 16}) {
          std::array<std::uint8_t, 64> guard{};
          HIP_CHECK(hipMemcpy(guard.data(), data + off, guard.size(),
                              hipMemcpyDeviceToHost));
          if (!std::all_of(guard.begin(), guard.end(),
                           [](auto byte) { return byte == 0xA5; }))
            std::abort();
        }
      }
    }
    for (float* data : storage)
      HIP_CHECK(hipFree(data));
    std::cout << "  contiguous heads: exact output/LSE, bounded scratch and "
                 "fallback\n";
  }

  if (want_lse) {
    std::vector<float> lref(lse_elements);
    std::vector<float> lgot(lse_elements);
    HIP_CHECK(hipMemcpy(lref.data(), d_lse_ref, lse_elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(lgot.data(), d_lse_new, lse_elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    Compare("log-sum-exp", lgot, lref, 2e-3);
  }

  for (float* p : {d_q, d_gate, d_k, d_v, d_cache, d_out_ref, d_out_new,
                   d_lse_ref, d_lse_new}) {
    HIP_CHECK(hipFree(p));
  }
  HIP_CHECK(hipFree(d_cache_f16));
}

// attn5: GUFO_DEV_ATTN_V2=1 (WmmaCausalAttentionV2) must be byte-identical to
// the default route. The default runs the packed kernel from 1024 tokens and
// the canonical one below; V2 runs packed from 256 tokens. Each case compares
// V2 with the default route, with itself on replay, and with the canonical
// (no workspace) route, for the context and the log-sum-exp.
std::uint16_t HalfBits(float value) {  // round to nearest even, finite input
  std::uint32_t f;
  std::memcpy(&f, &value, sizeof(f));
  const std::uint32_t sign = (f >> 16) & 0x8000U;
  const std::uint32_t abs = f & 0x7FFFFFFFU;
  if (abs >= 0x477FF000U)  // rounds to >= 65520: clamp (inputs stay below)
    return static_cast<std::uint16_t>(sign | 0x7BFFU);
  if (abs < 0x38800000U) {  // f16 subnormal or zero
    const std::uint32_t mant = (abs & 0x7FFFFFU) | 0x800000U;
    const int shift = 126 - static_cast<int>(abs >> 23);  // 14 + (113 - e)
    if (shift > 24)
      return static_cast<std::uint16_t>(sign);
    const std::uint32_t half_ulp = 1U << (shift - 1);
    std::uint32_t h = mant >> shift;
    const std::uint32_t rest = mant & ((1U << shift) - 1U);
    if (rest > half_ulp || (rest == half_ulp && (h & 1U)))
      ++h;
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = ((abs >> 13) - (112U << 10));
  const std::uint32_t rest = abs & 0x1FFFU;
  if (rest > 0x1000U || (rest == 0x1000U && (h & 1U)))
    ++h;
  return static_cast<std::uint16_t>(sign | h);
}

void SetAttnV2(bool on) {
  const int status =
      on ? setenv("GUFO_DEV_ATTN_V2", "1", 1) : unsetenv("GUFO_DEV_ATTN_V2");
  if (status != 0)
    std::abort();
}

struct V2Problem {
  std::uint32_t start_pos = 0;
  std::size_t batch = 0;
  std::uint32_t max_context = 0;
  float* q = nullptr;
  float* gate = nullptr;
  void* cache_f16 = nullptr;  // K plane then V plane, [position][kv][dim]
  float* k_work = nullptr;
  float* v_work = nullptr;
  std::size_t k_words = 0;
  std::size_t v_words = 0;
  float* out = nullptr;
  float* lse = nullptr;

  V2Problem(std::uint32_t start, std::size_t rows, float q_scale,
            float v_scale)
      : start_pos(start), batch(rows) {
    const std::uint32_t length = start + static_cast<std::uint32_t>(rows);
    max_context = length + 64;
    const std::size_t width = static_cast<std::size_t>(kNumHeads) * kHeadDim;
    const std::size_t kv_width =
        static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
    const std::size_t plane = kv_width * max_context;
    Rng rng(0x5EEDU ^ (start * 2654435761U) ^ static_cast<std::uint32_t>(rows));
    std::vector<float> h_q(rows * width);
    std::vector<float> h_gate(rows * width);
    for (auto& x : h_q)
      x = rng.Uniform(-1.0F, 1.0F) * q_scale;
    for (auto& x : h_gate)
      x = rng.Uniform(-2.0F, 2.0F);
    std::vector<std::uint16_t> h_cache(plane * 2, 0);
    for (std::size_t p = 0; p < length; ++p)
      for (std::size_t i = 0; i < kv_width; ++i) {
        h_cache[p * kv_width + i] = HalfBits(rng.Uniform(-1.0F, 1.0F));
        h_cache[plane + p * kv_width + i] =
            HalfBits(rng.Uniform(-1.0F, 1.0F) * v_scale);
      }
    k_words = static_cast<std::size_t>(length) * kv_width / 2;
    v_words = ((static_cast<std::size_t>(length) + 15U) & ~std::size_t{15}) *
              kv_width / 2;
    HIP_CHECK(hipMalloc(&q, h_q.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&gate, h_gate.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&cache_f16, h_cache.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&k_work, k_words * sizeof(float)));
    HIP_CHECK(hipMalloc(&v_work, v_words * sizeof(float)));
    HIP_CHECK(hipMalloc(&out, h_q.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&lse, rows * kNumHeads * sizeof(float)));
    HIP_CHECK(hipMemcpy(q, h_q.data(), h_q.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gate, h_gate.data(), h_gate.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cache_f16, h_cache.data(),
                        h_cache.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
  }
  V2Problem(const V2Problem&) = delete;
  V2Problem& operator=(const V2Problem&) = delete;
  ~V2Problem() {
    for (void* p : {static_cast<void*>(q), static_cast<void*>(gate), cache_f16,
                    static_cast<void*>(k_work), static_cast<void*>(v_work),
                    static_cast<void*>(out), static_cast<void*>(lse)})
      (void)hipFree(p);
  }

  // One launch; returns the context and (when requested) the LSE.
  std::pair<std::vector<float>, std::vector<float>> Run(
      bool v2, bool workspace, bool want_lse, std::uint32_t key_begin,
      hipStream_t stream = nullptr) {
    SetAttnV2(v2);
    const std::size_t width = static_cast<std::size_t>(kNumHeads) * kHeadDim;
    const std::size_t plane =
        static_cast<std::size_t>(kNumKvHeads) * kHeadDim * max_context;
    auto* k16 = static_cast<std::uint16_t*>(cache_f16);
    HIP_CHECK(hipMemset(out, 0xC3, batch * width * sizeof(float)));
    HIP_CHECK(hipMemset(lse, 0xC3, batch * kNumHeads * sizeof(float)));
    const std::span<float> kw =
        workspace ? std::span<float>{k_work, k_words} : std::span<float>{};
    const std::span<float> vw =
        workspace ? std::span<float>{v_work, v_words} : std::span<float>{};
    if (!gufo::hip::LaunchQwenWmmaAttention(
            q, nullptr, nullptr, want_lse ? nullptr : gate, nullptr, nullptr,
            k16, k16 + plane, out, 0, start_pos, batch, max_context, kNumHeads,
            kNumKvHeads, kHeadDim, stream, want_lse ? lse : nullptr, key_begin,
            /*skip_kv_write=*/true, kw, vw)) {
      std::cerr << "attn5: launcher refused the shape\n";
      std::abort();
    }
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> context(batch * width);
    std::vector<float> lse_host(want_lse ? batch * kNumHeads : 0);
    HIP_CHECK(hipMemcpy(context.data(), out, context.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    if (want_lse)
      HIP_CHECK(hipMemcpy(lse_host.data(), lse,
                          lse_host.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
    return {std::move(context), std::move(lse_host)};
  }
};

void RequireSame(const char* what, const std::vector<float>& a,
                 const std::vector<float>& b, std::size_t row_width) {
  if (a.size() == b.size() &&
      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0)
    return;
  std::size_t differing = 0, first = a.size();
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
      ++differing;
      first = std::min(first, i);
    }
  std::cerr << "attn5 V2 NOT byte-identical (" << what << "): " << differing
            << " of " << a.size() << " differ; first row "
            << first / row_width << " col " << first % row_width << " got "
            << std::hexfloat << b[first] << " want " << a[first]
            << std::defaultfloat << '\n';
  std::abort();
}

void RunV2Case(std::uint32_t start_pos, std::size_t batch, bool want_lse,
               std::uint32_t key_begin = 0, float q_scale = 1.0F,
               float v_scale = 1.0F) {
  std::cout << "attn5 V2 bitwise: start_pos=" << start_pos
            << " batch=" << batch << " lse=" << (want_lse ? "yes" : "no")
            << " key_begin=" << key_begin << " q_scale=" << q_scale
            << " v_scale=" << v_scale << '\n';
  V2Problem problem(start_pos, batch, q_scale, v_scale);
  const std::size_t width = static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const auto base = problem.Run(false, true, want_lse, key_begin);
  const auto canonical = problem.Run(false, false, want_lse, key_begin);
  const auto v2 = problem.Run(true, true, want_lse, key_begin);
  const auto again = problem.Run(true, true, want_lse, key_begin);
  SetAttnV2(false);
  RequireSame("default route vs V2", base.first, v2.first, width);
  RequireSame("canonical route vs V2", canonical.first, v2.first, width);
  RequireSame("V2 replay", v2.first, again.first, width);
  if (want_lse) {
    RequireSame("default LSE vs V2", base.second, v2.second, batch);
    RequireSame("canonical LSE vs V2", canonical.second, v2.second, batch);
    RequireSame("V2 LSE replay", v2.second, again.second, batch);
  }
  std::cout << "  byte-identical over " << base.first.size()
            << " context values\n";
}

// GUFO_ATTN5_BENCH=1: alternate default and V2 launches (ABAB, then BABA) at
// the served shapes and print the median time per call. Includes the head
// packing pass both arms share, as the served prefill timer does.
void BenchV1V2() {
  const char* bench = std::getenv("GUFO_ATTN5_BENCH");
  if (bench == nullptr || bench[0] != '1')
    return;
  const char* reps_env = std::getenv("GUFO_ATTN5_REPS");
  const int reps = reps_env != nullptr ? std::max(2, std::atoi(reps_env)) : 10;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::cout << "attn5 bench: shape, arm, median ms, TFLOPS (causal FLOP), "
               "min ms, max ms\n";
  for (const auto [start_pos, batch] :
       {std::pair{0U, std::size_t{4096}}, std::pair{4096U, std::size_t{4096}},
        std::pair{16384U, std::size_t{4096}},
        std::pair{28672U, std::size_t{4096}},
        std::pair{32768U, std::size_t{472}}}) {
    V2Problem problem(start_pos, batch, 1.0F, 1.0F);
    const double flop = 4.0 * kNumHeads * kHeadDim *
                        (static_cast<double>(batch) * start_pos +
                         static_cast<double>(batch) * (batch + 1) / 2.0);
    const auto time_once = [&](bool v2) {
      SetAttnV2(v2);
      const std::size_t plane =
          static_cast<std::size_t>(kNumKvHeads) * kHeadDim * problem.max_context;
      auto* k16 = static_cast<std::uint16_t*>(problem.cache_f16);
      HIP_CHECK(hipEventRecord(start, nullptr));
      if (!gufo::hip::LaunchQwenWmmaAttention(
              problem.q, nullptr, nullptr, problem.gate, nullptr, nullptr, k16,
              k16 + plane, problem.out, 0, start_pos, batch,
              problem.max_context, kNumHeads, kNumKvHeads, kHeadDim, nullptr,
              nullptr, 0, true, {problem.k_work, problem.k_words},
              {problem.v_work, problem.v_words}))
        std::abort();
      HIP_CHECK(hipEventRecord(stop, nullptr));
      HIP_CHECK(hipEventSynchronize(stop));
      float ms = 0.0F;
      HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
      return static_cast<double>(ms);
    };
    time_once(false);  // warm-up, both arms
    time_once(true);
    std::vector<double> arm[2];
    for (int r = 0; r < reps; ++r) {
      const bool v2_first = (r % 2) != 0;
      arm[v2_first ? 1 : 0].push_back(time_once(v2_first));
      arm[v2_first ? 0 : 1].push_back(time_once(!v2_first));
    }
    for (int a = 0; a < 2; ++a) {
      auto sorted = arm[a];
      std::sort(sorted.begin(), sorted.end());
      const double median = sorted[sorted.size() / 2];
      std::cout << "  s=" << start_pos << " b=" << batch << ' '
                << (a == 0 ? "default" : "V2     ") << ' ' << median << " ms "
                << flop / (median * 1e-3) / 1e12 << " TF " << sorted.front()
                << ' ' << sorted.back() << '\n';
    }
  }
  SetAttnV2(false);
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}

}  // namespace

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen WMMA prefill attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  RunCase(0, 256, false);
  // Not a multiple of the 32-query block, so the last block is partly out of
  // range.
  RunCase(0, 100, false);
  // Exactly one key tile visible, so the prefetch never fires.
  RunCase(0, 16, false);
  // Exactly two, so it fires once and then has to be suppressed.
  RunCase(0, 32, false);
  // Large shallow chunks use packed heads too, including a ragged final tile.
  RunCase(0, 1025, false);
  RunCase(0, 2048, true);
  // At depth: the whole visible range, prefix and diagonal, in one pass.
  RunCase(1024, 512, false);
  // A depth that is not a multiple of the 16-key tile.
  RunCase(1500, 128, false);
  // The log-sum-exp path, which also suppresses the gate.
  RunCase(1024, 256, true);
  RunCase(8192, 100, false);
  RunCase(8192, 1023, false);
  RunCase(8192, 1024, false);
  RunCase(32781, 1025, true);
  RunCase(8192, 1024, true, 16);
  RunCase(8192, 1024, true, 17);
  RunCase(8192, 1024, false, 9216);

  // attn5 V2 against the default route. The served shapes first (8 x 4096
  // chunks; the 944-token tail runs as two 472-token calls), then the edges:
  // shallow chunks, a start that is not a multiple of the 16-key tile (three
  // diagonal tiles per block), a ragged last block, key_begin inside and past
  // a block's range, the V2-only packed range [256, 1024), and score/value
  // scales that move the running maximum often.
  RunV2Case(0, 4096, false);
  RunV2Case(4096, 4096, false);
  RunV2Case(16384, 4096, false);
  RunV2Case(28672, 4096, false);
  RunV2Case(32768, 472, false);
  RunV2Case(33240, 472, false);
  RunV2Case(0, 1025, false);
  RunV2Case(0, 2048, true);
  RunV2Case(1500, 1024, false);
  RunV2Case(1500, 1100, true);
  RunV2Case(8192, 1024, true, 16);
  RunV2Case(8192, 1024, true, 17);
  RunV2Case(8192, 1024, false, 9216);
  RunV2Case(32781, 1025, true);
  RunV2Case(0, 256, false);
  RunV2Case(100, 300, true);
  RunV2Case(8191, 257, false);
  RunV2Case(0, 255, false);
  RunV2Case(0, 4096, false, 0, 8.0F, 16.0F);
  RunV2Case(12288, 4096, true, 0, 6.0F, 1.0F);
  BenchV1V2();

  std::cout << "Qwen WMMA prefill attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen WMMA prefill attention ops test.\n";
  return 77;
#endif
}

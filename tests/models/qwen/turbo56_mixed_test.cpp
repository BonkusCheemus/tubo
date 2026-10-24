// Per-plane codecs in widest-width physical slots. CPU only.
// --negative-control makes the writer use K's codec for V; the byte/padding
// oracle must reject it. This is not a device-launch or quality qualification.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_kernels.hpp"

namespace h = gufo::hip;
namespace t = gufo::hip::turbo4;
namespace {
int checks = 0;
int failures = 0;
void Check(bool ok, const char* what) {
  ++checks;
  if (!ok && ++failures <= 12)
    std::printf("FAIL: %s\n", what);
}

static_assert(!std::is_convertible_v<t::Bits, t::PlaneFormats>);
static_assert(!std::is_default_constructible_v<t::PlaneFormats>);

void Policy() {
  using S = h::QwenKvCacheStorage;
  auto p = h::QwenExecutionPolicy::Production();
  p.kv_cache_storage = S::kTurbo4;
  const auto baseline = h::ResolveQwenKvLayerTableForStorage(p, 3);
  Check(h::ConfigureQwenTurboKvFormats(&p, nullptr, nullptr), "unset formats");
  Check(h::ResolveQwenKvLayerTableForStorage(p, 3).Fingerprint() ==
            baseline.Fingerprint(),
        "unset preserves Turbo4 fingerprint");
  Check(h::ConfigureQwenTurboKvFormats(&p, "5,6,5", "4,5,6"), "CSV parse");
  p = h::ResolveQwenKvLayerTableForStorage(p, 3);
  Check(p.kv_cache_storage == S::kTurbo6, "widest slot selected");
  Check(h::QwenKvTableIsServableByStorage(p.kv_layers, p.kv_cache_storage),
        "mixed block codecs fit allocated slots");
  const auto f0 = h::QwenKvTurboLayerFormats(p, 0);
  const auto f1 = h::QwenKvTurboLayerFormats(p, 1);
  Check(f0.k == t::Bits::k5 && f0.v == t::Bits::k4 && f1.k == t::Bits::k6 &&
            f1.v == t::Bits::k5,
        "per-layer K/V selection");
  Check(!h::QwenKvTableSupportsCompactSnapshot(p.kv_layers, p.kv_cache_storage),
        "mixed compact snapshots refuse missing format identity");
  Check(h::QwenKvTableSupportsCompactSnapshot(baseline.kv_layers,
                                              baseline.kv_cache_storage),
        "uniform Turbo4 compact snapshots unchanged");
  Check(h::QwenKvFormatTablesEqual(p.kv_layers, p.kv_layers),
        "RAM table identity");
  const auto old_table = p.kv_layers;
  const auto fingerprint = p.Fingerprint();
  p.kv_layers.Set(0, h::QwenKvLayerFormat::kTurbo4,
                  h::QwenKvLayerFormat::kTurbo5);
  Check(p.Fingerprint() != fingerprint, "K/V order fingerprints differ");
  Check(!h::QwenKvFormatTablesEqual(old_table, p.kv_layers),
        "RAM snapshot table mismatch refused");
  p = h::QwenExecutionPolicy::Production();
  p.kv_cache_storage = S::kTurbo4;
  Check(h::ConfigureQwenTurboKvFormats(&p, "6", "4"), "singleton parse");
  p = h::ResolveQwenKvLayerTableForStorage(p, 3);
  Check(p.kv_layers.layer_count == 3 && !p.kv_layers_repeat &&
            p.kv_layers.K(2) == h::QwenKvLayerFormat::kTurbo6 &&
            p.kv_layers.V(2) == h::QwenKvLayerFormat::kTurbo4,
        "singleton expands before allocation");
  for (const char* bad : {"", "3", "7", "05", "5,", ",5", "5,,6", "5 6"}) {
    auto q = p;
    Check(!h::ConfigureQwenTurboKvFormats(&q, bad, "4"), "bad CSV refused");
    Check(q.Fingerprint() == p.Fingerprint(), "bad CSV leaves policy intact");
  }
  Check(!h::ConfigureQwenTurboKvFormats(&p, "5,6", "4,5,6"),
        "unequal CSV lengths refused");
  std::string too_many = "5";
  for (unsigned i = 1; i <= h::QwenKvFormatTable::kMaxLayers; ++i)
    too_many += ",5";
  Check(!h::ConfigureQwenTurboKvFormats(&p, too_many.c_str(), "4"),
        "table overflow refused, not truncated");
  bool threw = false;
  try {
    (void)h::ResolveQwenKvLayerTableForStorage(p, 2);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "wrong model layer count refused");
  Check(t::PlaneFormats(t::Bits::k4).IsTurbo4() &&
            !t::PlaneFormats(t::Bits::k5).IsTurbo4() &&
            !t::PlaneFormats(t::Bits::k6, t::Bits::k4, t::Bits::k4).IsTurbo4(),
        "WMMA nibble-only admission excludes qh and padded slots");
  Check(!t::PlaneFormats(t::Bits::k4, t::Bits::k6, t::Bits::k4).Valid(),
        "codec wider than slot refused");
  auto throws = [](auto&& f) {
    try { f(); return false; } catch (const std::exception&) { return true; }
  };
  Check(throws([&] { h::RequireCompactKvFormats(p.kv_layers, p.kv_cache_storage); }),
        "compact guard throws for mixed metadata, not just a false predicate");
  Check(throws([&] { h::RequireQwenKvFormatTablesEqual(old_table, p.kv_layers); }),
        "RAM mismatch guard throws before copy/reset");
  h::RequireCompactKvFormats(baseline.kv_layers, baseline.kv_cache_storage);
  Check(h::QwenKvPersistenceWidthSuffix(S::kFp16).empty() &&
        h::QwenKvPersistenceWidthSuffix(S::kFp32).empty() &&
        h::QwenKvPersistenceWidthSuffix(S::kTurbo4).empty(), "legacy disk identities unchanged");
  Check(h::QwenKvPersistenceWidthSuffix(S::kTurbo5) == "kv_storage=turbo5\n" &&
        h::QwenKvPersistenceWidthSuffix(S::kTurbo6) == "kv_storage=turbo6\n",
        "new uniform disk formats cannot collide");

  auto sixfour = h::QwenExecutionPolicy::Production();
  sixfour.kv_cache_storage = S::kTurbo6;
  auto sixsix = sixfour;
  Check(h::ConfigureQwenTurboKvFormats(&sixfour, "6", "4") &&
        h::ConfigureQwenTurboKvFormats(&sixsix, "6", "6"), "receipt policies configure");
  sixfour = h::ResolveQwenKvLayerTableForStorage(sixfour, 3);
  sixsix = h::ResolveQwenKvLayerTableForStorage(sixsix, 3);
  const auto mixed_receipt = h::DescribeQwenKvSelection(sixfour, 3 * 3 * 2 * 256);
  const auto uniform_receipt = h::DescribeQwenKvSelection(sixsix, 3 * 3 * 2 * 256);
  Check(mixed_receipt.find("bits_per_value=6.125000") != std::string::npos &&
        uniform_receipt.find("bits_per_value=6.125000") != std::string::npos,
        "selection receipts bill K6/V4 and K6/V6 at the same physical width");
  std::printf("%s\n%s\n", mixed_receipt.c_str(), uniform_receipt.c_str());
  for (const auto* name : {"GUFO_DEV_TURBO4_KV", "GUFO_DEV_TURBO_KV_BITS",
                          "GUFO_DEV_TURBO_KV_K_BITS", "GUFO_DEV_TURBO_KV_V_BITS"})
    unsetenv(name);
  Check(h::ResolveRuntimeQwenExecutionPolicy().Fingerprint() ==
        h::QwenExecutionPolicy::Production().Fingerprint(), "default-off resolver unchanged");
  setenv("GUFO_DEV_TURBO4_KV", "1", 1);
  setenv("GUFO_DEV_TURBO_KV_BITS", "bad", 1);
  Check(throws([] { (void)h::ResolveRuntimeQwenExecutionPolicy(); }),
        "enabled gate with invalid uniform width throws without overrides");
  setenv("GUFO_DEV_TURBO4_KV", "01", 1);
  Check(h::ResolveRuntimeQwenExecutionPolicy().kv_cache_storage == S::kFp16,
        "gate typo without overrides still fails closed");
  unsetenv("GUFO_DEV_TURBO4_KV");
  unsetenv("GUFO_DEV_TURBO_KV_BITS");
  setenv("GUFO_DEV_TURBO_KV_K_BITS", "5", 1);
  Check(throws([] { (void)h::ResolveRuntimeQwenExecutionPolicy(); }), "override without gate throws");
  setenv("GUFO_DEV_TURBO4_KV", "0", 1);
  Check(throws([] { (void)h::ResolveRuntimeQwenExecutionPolicy(); }), "disabled gate throws");
  setenv("GUFO_DEV_TURBO4_KV", "1", 1);
  setenv("GUFO_DEV_TURBO_KV_BITS", "bad", 1);
  Check(throws([] { (void)h::ResolveRuntimeQwenExecutionPolicy(); }), "invalid parent width throws");
  setenv("GUFO_DEV_TURBO_KV_BITS", "6", 1);
  setenv("GUFO_DEV_TURBO_KV_V_BITS", "4", 1);
  const auto runtime = h::ResolveQwenKvLayerTableForStorage(h::ResolveRuntimeQwenExecutionPolicy(), 3);
  Check(runtime.kv_layers.K(2) == h::QwenKvLayerFormat::kTurbo5 &&
        runtime.kv_layers.V(2) == h::QwenKvLayerFormat::kTurbo4, "real env path selects independent K/V");
  for (const auto* bad : {"", "7", "5,"}) {
    setenv("GUFO_DEV_TURBO_KV_V_BITS", bad, 1);
    Check(throws([] { (void)h::ResolveRuntimeQwenExecutionPolicy(); }), "invalid V env throws");
  }
  for (const auto* name : {"GUFO_DEV_TURBO4_KV", "GUFO_DEV_TURBO_KV_BITS",
                          "GUFO_DEV_TURBO_KV_K_BITS", "GUFO_DEV_TURBO_KV_V_BITS"})
    unsetenv(name);
}

void HalfCarry() {
  Check(t::FloatToHalfBits(0.124995F) == 0x3000U,
        "normal mantissa carry increments odd exponent (not OR)");
  // Independent compiler binary16 conversion, including both sides of every
  // finite normal/subnormal midpoint. Exclude NaN payload conventions.
  for (unsigned b = 0; b < 0x7BFFU; ++b) {
    const float a = t::HalfBitsToFloat(static_cast<std::uint16_t>(b));
    const float z = t::HalfBitsToFloat(static_cast<std::uint16_t>(b + 1));
    const float mid = (a + z) * 0.5F;
    for (float x : {std::nextafter(mid, a), mid, std::nextafter(mid, z)}) {
      for (float sign : {1.0F, -1.0F}) {
        const float value = x * sign;
        const _Float16 expected = static_cast<_Float16>(value);
        Check(
            t::FloatToHalfBits(value) == std::bit_cast<std::uint16_t>(expected),
            "FloatToHalfBits matches independent binary16 rounding");
      }
    }
  }
}

void Planes(t::PlaneFormats formats, bool negative) {
  constexpr unsigned heads = 2, dim = 256, context = 3;
  constexpr std::size_t width = heads * dim, guard = 16;
  const std::size_t slot = t::BlockBytes(formats.stride);
  const std::size_t bytes =
      h::Turbo4LayerPlaneBytes(dim, heads, context, formats.stride);
  std::vector<std::uint8_t> kb(bytes + 2 * guard, 0xA5), vb(kb);
  auto* k = kb.data() + guard;
  auto* v = vb.data() + guard;
  std::vector<float> keys(context * width), values(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    keys[i] = std::sin(static_cast<float>(i) * 0.021F) * 0.12F;
    values[i] = std::cos(static_cast<float>(i) * 0.013F) * 0.09F;
  }
  auto writer = formats;
  if (negative)
    writer.v = formats.k;
  for (unsigned pos = 0; pos < context; ++pos)
    for (unsigned head = 0; head < heads; ++head)
      Check(h::Turbo4WriteHeadBody(
                keys.data() + pos * width, values.data() + pos * width, k, v,
                bytes, width * sizeof(float), head, pos, context, heads, dim,
                writer) == h::Turbo4KernelStatus::kOk,
            "mixed writer succeeds");
  std::uint8_t expected[t::kMaxBlockBytes]{};
  for (std::size_t block = 0; block < context * width / t::kBlockValues;
       ++block) {
    for (int plane = 0; plane < 2; ++plane) {
      const auto bits = plane == 0 ? formats.k : formats.v;
      const auto* src =
          (plane == 0 ? keys.data() : values.data()) + block * t::kBlockValues;
      const auto* got = (plane == 0 ? k : v) + block * slot;
      Check(t::QuantizeBlockChecked(src, expected, bits),
            "oracle block storable");
      Check(std::memcmp(got, expected, t::BlockBytes(bits)) == 0,
            "each plane uses its own codec and physical slot pitch");
      Check(std::all_of(got + t::BlockBytes(bits), got + slot,
                        [](auto x) { return x == 0; }),
            "slot padding zero");
    }
  }
  for (std::size_t i = 0; i < guard; ++i)
    Check(kb[i] == 0xA5 && vb[i] == 0xA5 && kb[guard + bytes + i] == 0xA5 &&
              vb[guard + bytes + i] == 0xA5,
          "plane boundary guards intact");
  std::vector<std::uint16_t> dk(keys.size()), dv(keys.size());
  Check(h::Turbo4DequantLayerBody(k, v, dk.data(), dv.data(), context, context,
                                  heads, dim,
                                  formats) == h::Turbo4KernelStatus::kOk,
        "prefill dequant handles both formats");
  float scratch[t::kBlockValues];
  for (std::size_t block = 0; block < keys.size() / t::kBlockValues; ++block) {
    for (int plane = 0; plane < 2; ++plane) {
      const auto* data = (plane == 0 ? k : v) + block * slot;
      t::DequantizeBlock(data, scratch, plane == 0 ? formats.k : formats.v);
      t::RotateInverse(scratch);
      const auto* got =
          (plane == 0 ? dk.data() : dv.data()) + block * t::kBlockValues;
      bool same = true;
      for (std::size_t i = 0; i < t::kBlockValues; ++i)
        same = same && got[i] == t::FloatToHalfBits(scratch[i]);
      Check(same, "dequant output matches each plane's block oracle");
    }
  }
  const auto storage = formats.stride == t::Bits::k6 ? h::QwenKvCacheStorage::kTurbo6
      : formats.stride == t::Bits::k5 ? h::QwenKvCacheStorage::kTurbo5 : h::QwenKvCacheStorage::kTurbo4;
  const h::QwenKvPlaneGeometry geometry{dim, heads, context};
  std::vector<float> head(dim);
  Check(h::QwenKvReadHead(storage, k, bytes, geometry, 0, 0, head.data(), formats, true),
        "generic read helper accepts explicit mixed K metadata");
  Check(h::QwenKvReadHead(storage, v, bytes, geometry, 0, 0, head.data(), formats, false),
        "generic read helper accepts explicit mixed V metadata");
  auto copy_k = kb, copy_v = vb;
  Check(h::QwenKvWriteHead(storage, copy_k.data() + guard, bytes, geometry, 0, 0,
                         keys.data(), formats, true), "generic mixed K writer");
  Check(h::QwenKvWriteHead(storage, copy_v.data() + guard, bytes, geometry, 0, 0,
                         values.data(), formats, false), "generic mixed V writer");
  // A failed V block cannot commit already-staged K from the same head.
  const auto before_k = kb, before_v = vb;
  values[dim - 1] = std::numeric_limits<float>::quiet_NaN();
  Check(h::Turbo4WriteHeadBody(keys.data(), values.data(), k, v, bytes,
                               width * sizeof(float), 0, 0, context, heads, dim,
                               formats) ==
            h::Turbo4KernelStatus::kQuantizeRefused,
        "bad V refuses whole head");
  Check(kb == before_k && vb == before_v,
        "refusal leaves both planes unchanged");
  std::vector<float> query(width, 0.02F), out(width, 9.0F), scores(context);
  const h::Turbo4BufferSizes sizes{bytes, width * sizeof(float),
                                   width * sizeof(float),
                                   context * sizeof(float), 0};
  Check(h::Turbo4AttentionQueryBody(
            query.data(), k, v, out.data(), sizes, h::Turbo4KernelStatus::kOk,
            nullptr, 0, heads, heads, dim, context,
            1.0F / std::sqrt(static_cast<float>(dim)), scores.data(),
            formats) == h::Turbo4KernelStatus::kOk,
        "scalar attention reads mixed codecs");
  std::vector<double> decoded_k(context * dim), decoded_v(context * dim);
  for (unsigned pos = 0; pos < context; ++pos)
    for (unsigned b = 0; b < dim / t::kBlockValues; ++b)
      for (unsigned plane = 0; plane < 2; ++plane) {
        const auto* data = (plane == 0 ? k : v) +
                           (pos * heads * (dim / t::kBlockValues) + b) * slot;
        t::DequantizeBlock(data, scratch, plane == 0 ? formats.k : formats.v);
        t::RotateInverse(scratch);
        auto& decoded = plane == 0 ? decoded_k : decoded_v;
        for (unsigned d = 0; d < t::kBlockValues; ++d)
          decoded[pos * dim + b * t::kBlockValues + d] = scratch[d];
      }
  double logits[context]{}, largest = -std::numeric_limits<double>::infinity();
  for (unsigned pos = 0; pos < context; ++pos) {
    for (unsigned d = 0; d < dim; ++d)
      logits[pos] += query[d] * decoded_k[pos * dim + d];
    logits[pos] /= std::sqrt(static_cast<double>(dim));
    largest = std::max(largest, logits[pos]);
  }
  double denom = 0;
  for (auto& score : logits) {
    score = std::exp(score - largest);
    denom += score;
  }
  double max_error = 0;
  for (unsigned d = 0; d < dim; ++d) {
    double expected_value = 0;
    for (unsigned pos = 0; pos < context; ++pos)
      expected_value += logits[pos] * decoded_v[pos * dim + d] / denom;
    max_error = std::max(max_error, std::abs(expected_value - out[d]));
  }
  Check(max_error < 1e-6,
        "mixed scalar attention matches double softmax over decoded planes");
}
}  // namespace

int main(int argc, char** argv) {
  const bool negative =
      argc == 2 && std::string(argv[1]) == "--negative-control";
  Policy();
  HalfCarry();
  for (auto k : {t::Bits::k4, t::Bits::k5, t::Bits::k6})
    for (auto v : {t::Bits::k4, t::Bits::k5, t::Bits::k6}) {
      const auto slot =
          static_cast<unsigned>(k) > static_cast<unsigned>(v) ? k : v;
      Planes({slot, k, v}, negative && k != v);
    }
  // Exercise both codecs narrower than the allocated stride as well.
  Planes({t::Bits::k6, t::Bits::k4, t::Bits::k4}, false);
  std::printf("%s checks=%d failures=%d negative=%d\n",
              failures ? "FAIL" : "PASS", checks, failures, negative);
  return failures ? 1 : 0;
}

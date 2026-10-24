#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/models/qwen/hip/kernels/sn_iq3.hpp"
#include "src/models/qwen/hip/kernels/sn_iq3xxs.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"
#include "src/models/qwen/sn_weight_plan.hpp"

using namespace gufo::models;
namespace {
void Check(bool ok) {
  if (!ok)
    throw std::runtime_error("IQ3 oracle mismatch");
}
template<class F>
void Reject(F f) {
  bool bad = false;
  try {
    f();
  } catch (const std::exception&) {
    bad = true;
  }
  Check(bad);
}
void Reference(const void* in, void* out, QwenWeightLayout layout,
               std::size_t rows, std::size_t k) {
  if (layout == QwenWeightLayout::kIq3SRows8)
    gufo::hip::SnIq3Repack(in, static_cast<std::uint8_t*>(out), rows, k);
  else
    gufo::hip::sn::SnIq3xxsRepack(in, rows, k, out);
}
void Case(QwenWeightLayout layout, std::size_t rows, std::size_t k,
          bool mutant_concat, bool mutant_meta) {
  constexpr std::size_t prefix = 32, suffix = 64;
  const auto bytes = Iq3PackedBytes(layout, rows, k);
  const auto type = layout == QwenWeightLayout::kIq3SRows8
                        ? gufo::core::GgmlType::kIQ3_S
                        : gufo::core::GgmlType::kIQ3_XXS;
  const auto bb = Iq3BlockBytes(layout);
  // Exhaustive address oracle, not merely pack/unpack cancelling each other.
  // Every native byte must own exactly one in-bounds packed byte.
  {
    std::vector<std::uint8_t> seen(bytes);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t b = 0; b < k / 256; ++b)
        for (std::size_t o = 0; o < bb; ++o) {
          const auto at = Iq3ByteOffset(layout, rows, k / 256, r, b, o);
          Check(at < bytes && seen[at] == 0);
          seen[at] = 1;
        }
    Check(std::all_of(seen.begin(), seen.end(), [](auto n) { return n == 1; }));
  }
  std::vector<std::uint8_t> source(prefix + bytes + suffix);
  std::uint64_t seed = 0x8c71fc03b72ba19dULL;
  for (auto& x : source) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    x = static_cast<std::uint8_t>(seed);
  }
  auto expected = source;
  Reference(source.data() + prefix, expected.data() + prefix, layout, rows, k);
  auto byte_oracle = source;
  CopyIq3RowRange(source.data() + prefix, byte_oracle.data() + prefix, layout,
                  rows, k, 0, rows);
  Check(byte_oracle == expected);
  Reject([&] {
    CopyIq3RowRangeFast(source.data(), source.data(), layout, rows, k, 0, rows);
  });
  const std::array spans{SnWeightSpan{prefix, rows, k, bb, layout}};
  const auto jobs = PlanSnWeightCopies(source.size(), spans, 128);
  std::vector<std::uint8_t> actual(source.size(), 0xcd);
  for (auto i = jobs.rbegin(); i != jobs.rend(); ++i)
    ExecuteSnWeightCopyJob(source.data(), actual.data(), *i);
  Check(actual == expected);

  auto naive = source;
  Reference(source.data() + prefix, naive.data() + prefix, layout, rows / 2, k);
  Reference(source.data() + prefix + bytes / 2,
            naive.data() + prefix + bytes / 2, layout, rows / 2, k);
  Check((naive == expected) == (layout == QwenWeightLayout::kIq3SRows8));
  if (mutant_concat && layout == QwenWeightLayout::kIq3XxsRows4)
    actual = naive;
  Check(actual == expected);

  std::size_t capacity = 0;
  for (const auto& j : jobs)
    capacity = std::max(capacity, j.source_bytes);
  std::vector<std::uint8_t> stage(capacity);
  std::fill(actual.begin(), actual.end(), 0xcd);
  for (auto i = jobs.rbegin(); i != jobs.rend(); ++i) {
    const auto writes =
        StageSnWeightCopyJob(source.data(), stage.data(), stage.size(), *i);
    for (std::size_t d = 0; d < writes.size(); ++d) {
      auto w = writes[d];
      if (!w.bytes)
        continue;
      if (mutant_meta && d && i->layout == QwenWeightLayout::kIq3XxsRows4)
        w.destination_offset = i->tensor_offset + w.stage_offset;
      Check(w.stage_offset + w.bytes <= stage.size());
      Check(w.destination_offset + w.bytes <= actual.size());
      std::memcpy(actual.data() + w.destination_offset,
                  stage.data() + w.stage_offset, w.bytes);
    }
  }
  Check(actual == expected);
  // Stage whole coalesced jobs, including XXS full-plane relocation. The
  // canonical fine-grained oracle above stays unchanged.
  const auto coalesced = CoalesceIq3CopyJobs(jobs, 64ULL << 20);
  std::vector<std::uint8_t> big_stage(
      std::min<std::size_t>(bytes, 64ULL << 20));
  std::fill(actual.begin(), actual.end(), 0xcd);
  for (const auto& job : coalesced) {
    Check(!IsIq3Layout(job.layout) || job.source_bytes <= (64ULL << 20));
    const auto writes = StageSnWeightCopyJob(source.data(), big_stage.data(),
                                             big_stage.size(), job);
    for (const auto& w : writes)
      if (w.bytes)
        std::memcpy(actual.data() + w.destination_offset,
                    big_stage.data() + w.stage_offset, w.bytes);
  }
  Check(actual == expected);
  std::vector<std::uint8_t> native(source.size(), 0xcd);
  for (auto i = jobs.rbegin(); i != jobs.rend(); ++i)
    ExecuteSnWeightUnpackJob(actual.data(), native.data(), *i);
  Check(native == source);

  const auto views =
      MakeIq3PairViews(actual.data() + prefix, type, rows, k, bytes);
  Check(views.pair.data == views.gate.data && views.gate.data == views.up.data);
  Check(views.pair.layout_rows == rows &&
        views.up.layout_row_begin == rows / 2);
  Check(views.gate.num_elements == rows / 2 * k &&
        views.up.FitsAvailableStorage());
  Check(views.gate.EncodedSizeBytes() == bytes / 2);
  Check(IsSnIq3VerifyPair(views.gate, views.up) ==
        (rows == 34816 && k == 5120));
  if (rows == 34816 && k == 5120) {
    Check(IsSnIq3VerifyView(views.pair));
    Check(!IsSnIq3VerifyPair(views.up, views.gate));
    Check(!IsSnIq3VerifyPair(views.gate, views.gate));
    auto bad = views.up;
    bad.available_bytes = bytes / 2;
    Check(!IsSnIq3VerifyPair(views.gate, bad));
    bad = views.up;
    bad.data = static_cast<const std::uint8_t*>(bad.data) + 16;
    Check(!IsSnIq3VerifyPair(views.gate, bad));
    bad = views.up;
    bad.type = gufo::core::GgmlType::kQ5_K;
    Check(!IsSnIq3VerifyPair(views.gate, bad));
  }
  for (const auto& view : {views.pair, views.gate, views.up}) {
    const auto logical = view.num_elements / k;
    for (auto r : {std::size_t{0}, logical / 2 - 1, logical / 2, logical - 1})
      for (std::size_t b = 0; b < k / 256; ++b)
        for (std::size_t o = 0; o < bb; ++o) {
          const auto physical = Iq3ViewByteOffset(view, r, b, o);
          const auto original =
              ((view.layout_row_begin + r) * (k / 256) + b) * bb + o;
          Check(actual[prefix + physical] == source[prefix + original]);
        }
    Reject([&] { Iq3ViewByteOffset(view, logical, 0, 0); });
    Reject([&] { Iq3ViewByteOffset(view, 0, k / 256, 0); });
    Reject([&] { Iq3ViewByteOffset(view, 0, 0, bb); });
  }
  std::vector<std::uint8_t> half(bytes / 2);
  UnpackIq3View(views.gate, half.data(), half.size());
  Check(std::equal(half.begin(), half.end(), source.begin() + prefix));
  UnpackIq3View(views.up, half.data(), half.size());
  Check(std::equal(half.begin(), half.end(),
                   source.begin() + prefix + bytes / 2));
  // GPU unpack (UnpackIq3ViewKernel) index math run on the CPU, one u16 per
  // "thread". Mutants: one row off, one S8 record / XXS index off.
  const auto source_offset = [&](const QwenTensorRef& v, std::size_t begin,
                                 std::uint32_t n) {
    return layout == QwenWeightLayout::kIq3SRows8
               ? Iq3UnpackSourceOffset<QwenWeightLayout::kIq3SRows8>(
                     v.layout_rows, k / 256, begin, n)
               : Iq3UnpackSourceOffset<QwenWeightLayout::kIq3XxsRows4>(
                     v.layout_rows, k / 256, begin, n);
  };
  const auto kernel_unpack = [&](const QwenTensorRef& v, std::size_t row_shift,
                                 std::size_t byte_shift) {
    std::vector<std::uint8_t> out(v.EncodedSizeBytes());
    for (std::uint32_t n = 0; n < out.size(); n += 2) {
      const auto at = source_offset(v, v.layout_row_begin, n);
      Check(at % 2 == 0 && source_offset(v, v.layout_row_begin, n + 1) == at + 1);
      const auto from =
          (source_offset(v, v.layout_row_begin + row_shift, n) + byte_shift) %
          bytes;
      out[n] = actual[prefix + from];
      out[n + 1] = actual[prefix + from + 1];
    }
    return out;
  };
  const auto record = layout == QwenWeightLayout::kIq3SRows8 ? 880 : 64;
  for (const auto& view : {views.pair, views.gate, views.up}) {
    const auto at = source.begin() + prefix + (view.layout_row_begin * (k / 256) * bb);
    auto out = kernel_unpack(view, 0, 0);
    Check(std::equal(out.begin(), out.end(), at));
    out = kernel_unpack(view, 1, 0);
    Check(!std::equal(out.begin(), out.end(), at));
    out = kernel_unpack(view, 0, record);
    Check(!std::equal(out.begin(), out.end(), at));
  }
  // Prefill route: unpack P into scratch, native G/U are its two halves.
  std::vector<std::uint8_t> scratch(bytes + 32, 0xcd);
  const auto prefill =
      Iq3NativeScratchViews(views.gate, views.up, scratch.data(), scratch.size());
  Check(prefill.pair.num_elements == rows * k &&
        prefill.pair.layout_row_begin == 0 && prefill.pair.data == views.gate.data);
  const auto unpacked = kernel_unpack(prefill.pair, 0, 0);
  std::copy(unpacked.begin(), unpacked.end(), scratch.begin());
  for (const auto& [ref, offset] :
       {std::pair{prefill.gate, std::size_t{0}}, std::pair{prefill.up, bytes / 2}}) {
    Check(ref.layout == QwenWeightLayout::kNative && ref.type == type &&
          ref.num_elements == rows / 2 * k && ref.FitsAvailableStorage() &&
          ref.EncodedSizeBytes() == bytes / 2);
    const auto* p = static_cast<const std::uint8_t*>(ref.data);
    Check(p == scratch.data() + offset &&
          std::equal(p, p + bytes / 2, source.begin() + prefix + offset));
  }
  Reject([&] { Iq3NativeScratchViews(views.gate, views.up, scratch.data(), bytes - 1); });
  Reject([&] { Iq3NativeScratchViews(views.up, views.gate, scratch.data(), bytes); });
  Reject([&] { Iq3NativeScratchViews(views.gate, views.gate, scratch.data(), bytes); });
  Reject([&] { Iq3NativeScratchViews(views.pair, views.up, scratch.data(), bytes); });
  Reject([&] { Iq3NativeScratchViews(prefill.gate, prefill.up, scratch.data(), bytes); });
  if (rows == 34816 && k == 5120)  // fits the 27B arena BF16 weight scratch
    Check(bytes <= 5120ULL * 17408 * 2);
  Reject([&] { UnpackIq3View(views.up, half.data(), half.size() - 1); });
  Reject([&] { UnpackIq3View(views.pair, actual.data() + prefix, bytes); });
  Reject([&] {
    MakeIq3PairViews(actual.data() + prefix + 1, type, rows, k, bytes);
  });
  Reject([&] {
    MakeIq3PairViews(actual.data() + prefix, type, rows, k, bytes - 1);
  });
  auto invalid = views.up;
  invalid.layout_row_begin = rows;
  Check(!invalid.FitsAvailableStorage());
  invalid = views.up;
  invalid.available_bytes = bytes / 2;
  Check(!invalid.FitsAvailableStorage());
  const auto converted =
      std::find_if(jobs.begin(), jobs.end(),
                   [](const auto& j) { return j.block_bytes != 0; });
  Reject([&] {
    StageSnWeightCopyJob(source.data(), stage.data(), 0, *converted);
  });
  auto wrong = *converted;
  ++wrong.source_offset;
  Reject([&] {
    StageSnWeightCopyJob(source.data(), stage.data(), stage.size(), wrong);
  });
  std::printf(
      "PASS layout=%s rows=%zu k=%zu bytes=%zu jobs=%zu roundtrip=equal "
      "P/G/U=equal\n",
      layout == QwenWeightLayout::kIq3SRows8 ? "S8" : "XXS4", rows, k, bytes,
      jobs.size());
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const bool mutant_concat =
        argc > 1 && std::string_view(argv[1]) == "--mutant-concat";
    const bool mutant_meta =
        argc > 1 && std::string_view(argv[1]) == "--mutant-local-planes";
    Check(argc == 1 || mutant_concat || mutant_meta);
    for (const auto layout :
         {QwenWeightLayout::kIq3SRows8, QwenWeightLayout::kIq3XxsRows4}) {
      Case(layout, 64, 512, mutant_concat, mutant_meta);
      Case(layout, 34816, 5120, mutant_concat, mutant_meta);
      const auto group = Iq3GroupRows(layout);
      Case(layout, 2 * group, 256, mutant_concat, mutant_meta);
      Case(layout, 6 * group, 768, mutant_concat, mutant_meta);  // odd groups/blocks
      Reject([&] { ValidateIq3Range(layout, 2 * group, 255, 0, group); });
      Reject([&] { ValidateIq3Range(layout, 2 * group + 1, 256, 0, group); });
      Reject([&] { ValidateIq3Range(layout, 2 * group, 256, 1, group); });
      Reject([&] { ValidateIq3Range(layout, 2 * group, 256, group, 0); });
      Check(
          !Iq3ShapeValid(layout, std::numeric_limits<std::size_t>::max(), 256));
    }
    std::printf("PASS negatives; sizeof(QwenTensorRef)=%zu; CPU only\n",
                sizeof(QwenTensorRef));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}

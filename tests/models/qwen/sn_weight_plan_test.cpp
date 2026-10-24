#include "src/models/qwen/sn_weight_plan.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

#include "src/models/qwen/state.hpp"

namespace {
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::exit(1);
  }
}
template<class F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected, "invalid copy plan accepted");
}

// Independent full-tensor reference, the original pre-range loop. Do not
// call Rows32ByteOffset here: copy and Get must not validate their own bug.
void Reference(const std::uint8_t* s, std::uint8_t* d,
               const gufo::models::SnWeightSpan& t) {
  const auto blocks = t.columns / 256;
  const std::size_t payload = t.block_bytes == 210 ? 208 : 176;
  for (std::size_t r = 0; r < t.rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      const auto* native = s + (r * blocks + b) * t.block_bytes;
      for (std::size_t c = 0; c < payload / 16; ++c)
        std::memcpy(
            d + ((r / 32 * blocks + b) * payload * 32) + c * 512 + r % 32 * 16,
            native + c * 16, 16);
      if (t.block_bytes == 210)
        std::memcpy(d + t.rows * blocks * 208 +
                        ((r / 32 * blocks + b) * 32 + r % 32) * 2,
                    native + 208, 2);
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  using namespace gufo::models;
  const std::string_view mode = argc > 1 ? argv[1] : "";
  const std::vector<SnWeightSpan> tensors{{37, 96, 512, 176},
                                          {34000, 64, 768, 210}};
  constexpr std::size_t size = 75000;
  std::vector<std::uint8_t> source(size);
  for (std::size_t i = 0; i < size; ++i)
    source[i] = static_cast<std::uint8_t>((i * 71 + i / 251) & 255);
  for (const auto& t : tensors)
    for (std::size_t b = 0; b < t.rows * t.columns / 256; ++b) {
      auto* p = source.data() + t.offset + b * t.block_bytes;
      if (t.block_bytes == 176) {
        p[0] = 0;
        p[1] = 0x3c;
        p[2] = 0;
        p[3] = 0x34;
      } else {
        p[208] = 0;
        p[209] = 0x3c;
      }
    }
  const auto original = source;
  auto expected = source;
  for (const auto& t : tensors)
    Reference(source.data() + t.offset, expected.data() + t.offset, t);
  auto jobs = PlanSnWeightCopies(size, tensors, 127);
  std::reverse(jobs.begin(), jobs.end());
  std::vector<unsigned> coverage(size);
  std::vector<std::uint8_t> output(size, 0xee);
  for (const auto& j : jobs) {
    for (std::size_t i = j.source_offset; i < j.source_offset + j.source_bytes;
         ++i)
      ++coverage[i];
    if (mode == "--mutant-native") {
      std::memcpy(output.data() + j.source_offset,
                  source.data() + j.source_offset, j.source_bytes);
    } else if (mode == "--mutant-local-r" && j.block_bytes) {
      const auto row_bytes = j.columns / 256 * j.block_bytes;
      InterleaveKQuantRows32(
          source.data() + j.source_offset,
          output.data() + j.tensor_offset + j.row_begin * row_bytes, 32,
          j.columns, j.block_bytes);
    } else {
      ExecuteSnWeightCopyJob(source.data(), output.data(), j);
    }
  }
  Check(std::all_of(coverage.begin(), coverage.end(),
                    [](auto n) { return n == 1; }),
        "native gaps/tensor jobs overlap or leave holes");
  Check(source == original, "source changed");
  Check(output == expected,
        "copy differs from independent full-tensor reference");
  std::size_t capacity = 0;
  for (const auto& j : jobs)
    capacity = std::max(capacity, j.source_bytes);
  std::vector<std::uint8_t> stage(capacity), staged_output(size, 0xee);
  for (const auto& j : jobs) {
    const auto writes =
        StageSnWeightCopyJob(source.data(), stage.data(), capacity, j);
    for (std::size_t n = 0; n < writes.size(); ++n) {
      const auto& w = writes[n];
      if (!w.bytes)
        continue;
      const auto offset = mode == "--mutant-stage-meta" && n == 1
                              ? j.tensor_offset + w.stage_offset
                              : w.destination_offset;
      Check(offset <= size && w.bytes <= size - offset,
            "staged destination escapes region");
      std::memcpy(staged_output.data() + offset, stage.data() + w.stage_offset,
                  w.bytes);
    }
  }
  Check(staged_output == expected, "GTT staged scatter differs from reference");
  std::vector<std::uint8_t> restored(size, 0xee);
  for (const auto& j : jobs)
    ExecuteSnWeightUnpackJob(staged_output.data(), restored.data(), j);
  Check(restored == source, "inverse copy plan differs from native bytes");
  Check(source == original, "staging changed source");
  Reject(
      [&] { StageSnWeightCopyJob(source.data(), stage.data(), 0, jobs[0]); });
  for (const auto& t : tensors) {
    QwenTensorRef native{
        .data = source.data() + t.offset,
        .type = t.block_bytes == 176 ? gufo::core::GgmlType::kQ5_K
                                     : gufo::core::GgmlType::kQ6_K,
        .num_elements = t.rows * t.columns,
        .available_bytes = t.rows * t.columns / 256 * t.block_bytes};
    auto packed = native;
    packed.data = output.data() + t.offset;
    packed.layout = QwenWeightLayout::kRows32;
    packed.layout_columns = t.columns;
    Check(packed.FitsAvailableStorage(), "valid tagged tensor refused");
    for (std::size_t i = 0; i < native.num_elements; ++i)
      Check(std::bit_cast<std::uint32_t>(native.Get(i)) ==
                std::bit_cast<std::uint32_t>(packed.Get(i)),
            "Get differs");
    packed.layout_columns = 255;
    Check(!packed.FitsAvailableStorage(), "invalid shape accepted");
    packed.layout = static_cast<QwenWeightLayout>(255);
    Check(!packed.HasValidWeightLayout(), "unknown tag accepted");
  }
  const auto bad = [&](SnWeightSpan t) { return std::vector<SnWeightSpan>{t}; };
  Reject([&] { PlanSnWeightCopies(size, bad({0, 33, 512, 176})); });
  Reject([&] { PlanSnWeightCopies(size, bad({0, 32, 257, 176})); });
  Reject([&] { PlanSnWeightCopies(size, bad({0, 32, 256, 144})); });
  Reject([&] { PlanSnWeightCopies(size, bad({size, 32, 256, 210})); });
  Reject([&] {
    PlanSnWeightCopies(
        size, bad({0, 32, std::numeric_limits<std::size_t>::max() - 255, 210}));
  });
  Reject([&] {
    PlanSnWeightCopies(size, std::span<const SnWeightSpan>(tensors.data(), 1),
                       0);
  });
  auto overlapping = tensors;
  overlapping[1].offset = 38;
  Reject([&] { PlanSnWeightCopies(size, overlapping); });
  Reject([&] {
    InterleaveKQuantRows32(source.data(), source.data(), 32, 256, 210);
  });
  Reject([&] {
    InterleaveKQuantRows32Range(source.data(), output.data(), 64, 256, 210, 32,
                                96);
  });
  QwenModelWeights weights;
  weights.layers.resize(1);
  Check(!weights.HasNonNativeWeights(), "native model refused");
  weights.layers[0].ssm_out.layout = QwenWeightLayout::kRows32;
  Check(weights.HasNonNativeWeights(), "unplumbed reader not fenced");
  weights.layers[0].ssm_out.layout = QwenWeightLayout::kNative;
  weights.output.layout = static_cast<QwenWeightLayout>(255);
  Check(weights.HasNonNativeWeights(), "unknown model tag not fenced");
  auto no_selection = PlanSnWeightCopies(size, {});
  Check(no_selection.size() == 1 && !no_selection[0].block_bytes,
        "native-only plan changed");
  std::printf(
      "PASS: mixed copy plan, independent reference, tagged Get, invalid "
      "inputs; QwenTensorRef=%zu bytes\n",
      sizeof(QwenTensorRef));
}

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"

namespace {

using gufo::speculative::DraftProposal;
using gufo::speculative::IDraftBackend;
using gufo::speculative::MakeAlignedPromptCaptureWindow;
using gufo::speculative::PromptCaptureRowsForChunk;
using gufo::tokenization::TokenId;

int failures = 0;

void Check(bool ok, const char* message) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", message);
  failures += !ok;
}

class FullContextBackend final : public IDraftBackend {
public:
  [[nodiscard]] std::string_view Name() const noexcept override {
    return "full-context-test";
  }
  [[nodiscard]] DraftProposal Propose(std::span<const TokenId>,
                                      std::uint32_t,
                                      std::uint32_t) override {
    return {};
  }
};

void CheckAlignedBoundaries() {
  struct Case {
    std::size_t tokens;
    std::size_t expected_offset;
    std::size_t expected_rows;
  };
  constexpr std::array cases{
      Case{2047, 0, 2047},    Case{2048, 0, 2048},
      Case{2049, 0, 2049},    Case{2303, 0, 2303},
      Case{2304, 256, 2048},  Case{2305, 256, 2049},
      Case{32768, 30720, 2048},
  };
  constexpr std::uint32_t start = 5000;
  for (const auto& test : cases) {
    const auto window =
        MakeAlignedPromptCaptureWindow(start, test.tokens, 2048, 256);
    Check(window.row_offset == test.expected_offset &&
              test.tokens - window.row_offset == test.expected_rows &&
              window.first_position == start + test.expected_offset,
          "aligned window boundary has expected K, W, and absolute position");
  }

  const auto overflow = MakeAlignedPromptCaptureWindow(
      std::numeric_limits<std::uint32_t>::max() - 1U, 10, 1, 1);
  Check(overflow.row_offset == 0 &&
            overflow.first_position ==
                std::numeric_limits<std::uint32_t>::max() - 1U,
        "overflow falls back to full capture");
}

void CheckChunkRangesAndRowIdentity() {
  constexpr std::size_t width = 3;
  constexpr std::size_t rows = 12;
  constexpr std::uint32_t segment_start = 100;
  std::array<float, rows * width> full{};
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < width; ++column) {
      full[row * width + column] = static_cast<float>(row * 10 + column);
    }
  }

  const auto window =
      MakeAlignedPromptCaptureWindow(segment_start, rows, 8, 4);
  Check(window.row_offset == 4 && window.first_position == 104,
        "small window aligns the dropped prefix to an injection chunk");

  std::vector<float> captured;
  for (std::size_t chunk_offset = 0; chunk_offset < rows; chunk_offset += 4) {
    const auto chunk_start =
        segment_start + static_cast<std::uint32_t>(chunk_offset);
    const auto range = PromptCaptureRowsForChunk(
        chunk_start, 4, window.first_position);
    const std::span<const float> chunk{full.data() + chunk_offset * width,
                                       4 * width};
    const auto selected = chunk.subspan(range.first_row * width,
                                        range.row_count * width);
    captured.insert(captured.end(), selected.begin(), selected.end());
  }

  const std::span<const float> expected{full.data() + window.row_offset * width,
                                        (rows - window.row_offset) * width};
  Check(captured.size() == expected.size() &&
            std::memcmp(captured.data(), expected.data(),
                        expected.size_bytes()) == 0,
        "captured suffix feature rows are byte-identical to full capture");

  const auto before = PromptCaptureRowsForChunk(10, 4, 9);
  const auto inside = PromptCaptureRowsForChunk(10, 4, 12);
  const auto after = PromptCaptureRowsForChunk(10, 4, 14);
  Check(before.first_row == 0 && before.row_count == 4 &&
            inside.first_row == 2 && inside.row_count == 2 &&
            after.first_row == 4 && after.row_count == 0,
        "chunk range handles capture starts before, inside, and after chunk");
}

void CheckDefaultBackendKeepsFullPrompt() {
  FullContextBackend backend;
  const auto window = backend.TargetPromptCaptureWindow(77, 1000);
  Check(window.row_offset == 0 && window.first_position == 77,
        "default backend keeps every prompt row (including MTP)");
}

}  // namespace

int main() {
  CheckAlignedBoundaries();
  CheckChunkRangesAndRowIdentity();
  CheckDefaultBackendKeepsFullPrompt();
  return failures == 0 ? 0 : 1;
}

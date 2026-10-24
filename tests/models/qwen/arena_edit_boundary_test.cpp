// HONEST BOUNDARY TEST for the arena.cpp edits — NOT a proof that arena.cpp
// compiles.
//
// WHAT THIS IS: the statements I added to arena.cpp, transcribed verbatim,
// compiled against the REAL headers they depend on (execution_policy.hpp and
// kv_allocation.hpp, both HIP-free), with arena.cpp's own CheckedMultiply
// copied character for character.
//
// WHY A STANDALONE TU IS THE ONLY OPTION HERE: arena.cpp is wrapped in
// `#if defined(ENGINE_ENABLE_HIP)` at line 1 and closed at line 1253, and this
// machine has no HIP toolchain. That makes every whole-file comparison
// vacuous, which was proved by a negative control, not assumed:
//
//   g++ -fsyntax-only                        pristine  0 errors
//   g++ -fsyntax-only                        edited    0 errors
//   g++ -fsyntax-only   + SYNTAX ERROR INJECTED IN MY REGION  -> 0 errors
//                                                       ^ the guard is off,
//                                                         my code is never parsed
//   g++ -fsyntax-only -DENGINE_ENABLE_HIP     all three 1 "error", and it is
//                                                       fatal:
//                                                       hip/hip_runtime.h: No such
//                                                       file or directory
//
// A fatal missing-header error stops the parse before reaching my lines, so the
// broken copy looks identical to the good one in BOTH configurations. An
// earlier report of mine claimed "pristine and edited each give exactly 1
// error, therefore the edit is sound". That comparison carried no information
// and the claim is withdrawn.
//
// WHAT THIS DOES ESTABLISH: that my added expressions are well-formed and
// type-correct against the real enumerators and helper signatures. It does NOT
// establish that arena.cpp as a whole compiles — that needs the HIP toolchain
// and a real build.
//
//   g++ -std=c++17 -Wall -Wextra -I. <this file>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"

namespace gufo::hip {

// arena.cpp:28-32, transcribed character for character.
std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Qwen GPU allocation size overflows");
  }
  return left * right;
}

// The subset of CompactSnapshotLayout the edited statements touch.
struct LayoutSubset {
  std::size_t live_kv_elements_per_plane;
  std::size_t live_kv_bytes_per_plane;
};

}  // namespace gufo::hip

int main() {
  using gufo::hip::CheckedMultiply;
  using gufo::hip::LayoutSubset;
  using gufo::hip::QwenKvAllocationUnits;
  using gufo::hip::QwenKvCacheStorage;
  using gufo::hip::QwenKvElementBytes;
  using gufo::hip::QwenKvIsBlockAddressed;
  using gufo::hip::QwenKvUnitBytes;

  int failures = 0;
  auto Check = [&](bool ok, const char *what) {
    std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
  };

  // UD-Q4 geometry: 16 full-attention layers, kv_width 1024, context 4096.
  const std::size_t attention_layers = 16;
  const std::size_t valid_context = 4096;
  const std::size_t kv_width = 1024;

  for (QwenKvCacheStorage kv_storage :
       {QwenKvCacheStorage::kFp32, QwenKvCacheStorage::kFp16,
        QwenKvCacheStorage::kTurbo4}) {
    // arena.cpp:132-134, verbatim.
    LayoutSubset layout{};
    const std::size_t kv_element_bytes = QwenKvElementBytes(kv_storage);
    layout.live_kv_elements_per_plane = CheckedMultiply(
        CheckedMultiply(attention_layers, valid_context), kv_width);

    // arena.cpp:143-150, the block-granular branch, verbatim in shape.
    if (QwenKvIsBlockAddressed(kv_storage)) {
      layout.live_kv_bytes_per_plane = CheckedMultiply(
          QwenKvAllocationUnits(kv_storage, layout.live_kv_elements_per_plane),
          QwenKvUnitBytes(kv_storage));
    } else {
      layout.live_kv_bytes_per_plane = CheckedMultiply(
          layout.live_kv_elements_per_plane, kv_element_bytes);
    }

    // The snapshot precision bound, arena.cpp:194-195.
    const bool precision_in_range =
        static_cast<std::uint32_t>(kv_storage) <=
            static_cast<std::uint32_t>(QwenKvCacheStorage::kTurbo4);

    std::printf("  note  storage=%u elements=%zu bytes=%zu\n",
                static_cast<unsigned>(kv_storage),
                layout.live_kv_elements_per_plane,
                layout.live_kv_bytes_per_plane);

    Check(precision_in_range,
          "every storage in the enumerator passes the snapshot precision bound");

    if (kv_storage == QwenKvCacheStorage::kFp16) {
      Check(layout.live_kv_bytes_per_plane == 134217728U,
            "fp16 plane is still 134217728 B (unchanged by the branch)");
    }
    if (kv_storage == QwenKvCacheStorage::kTurbo4) {
      Check(layout.live_kv_bytes_per_plane == 34603008U,
            "turbo4 plane is 34603008 B through the block branch");
      // The refused element size must not leak into the plane computation:
      // Turbo4 takes the block branch, where QwenKvUnitBytes is 66, not 0.
      Check(QwenKvUnitBytes(kv_storage) == 66U,
            "the turbo4 branch uses the 66 B block size, not the refused element size");
      Check(layout.live_kv_bytes_per_plane !=
                layout.live_kv_elements_per_plane * 0U,
            "turbo4 bytes are not derived from a zero element size");
    }
  }

  // The refusal is real: the element view is 0 for Turbo4, so any caller that
  // still uses it gets a zero-length plane rather than a wrong one.
  Check(QwenKvElementBytes(QwenKvCacheStorage::kTurbo4) == 0U,
        "the refused Turbo4 element size is visible to a careless caller as 0");

  std::printf(failures == 0 ? "\nEXPRESSIONS TYPE-CHECK "
                              "(does NOT prove arena.cpp compiles)\n"
                            : "\n%d CHECK(S) FAILED\n",
              failures);
  return failures == 0 ? 0 : 1;
}
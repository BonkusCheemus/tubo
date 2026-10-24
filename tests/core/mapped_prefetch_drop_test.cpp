// DropMappedRangeFromPageCache, the two syscalls and the rounding.
//
// WHY THIS NEEDS A TEST AT ALL. The function hands memory back to the kernel
// from a live mapping. Two things can go wrong and neither is visible in a
// throughput number:
//
//   1. THE ROUNDING. MADV_DONTNEED needs a page-aligned start, and a partial
//      page at either edge can hold bytes that belong to something else. Start
//      rounds UP, end rounds DOWN. Round the start DOWN instead and the call
//      discards a page the caller never offered.
//   2. THE LOSSLESSNESS. The whole design rests on "a PROT_READ MAP_PRIVATE
//      file mapping re-faults the same bytes from the same file". If that were
//      ever false the drafter would read garbage weights after the drop, and a
//      garbage drafter is *invisible* under speculative decoding -- the target
//      verifies every row, so the output stays right and only the acceptance
//      rate falls. The bug would present as "the drafter got worse", months
//      later. So it is checked here, on real bytes, through a real mapping.
//
// CPU only: mmap, madvise and posix_fadvise on a temp file. No HIP, no GPU.
//
// Manual compile:
//   g++-15 -std=c++20 -I. -o /tmp/t tests/core/mapped_prefetch_drop_test.cpp

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "src/core/mapped_prefetch.hpp"

namespace fs = std::filesystem;
using gufo::core::DropMappedRangeFromPageCache;
using gufo::core::PrefaultMappedRange;
using gufo::core::ResidentMappedBytes;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
  if (!ok) ++g_failures;
}

std::size_t PageSize() {
  const long value = ::sysconf(_SC_PAGESIZE);
  return value > 0 ? static_cast<std::size_t>(value) : 4096;
}

/// A real file, a real read-only private mapping, filled with a byte pattern
/// that depends on the offset so a wrong page is distinguishable from a
/// zero-filled one.
struct Mapping {
  fs::path path;
  int fd{-1};
  const std::uint8_t* data{nullptr};
  std::size_t size{0};

  explicit Mapping(std::size_t bytes) : size(bytes) {
    path = fs::temp_directory_path() /
           ("gufo_drop_test_" + std::to_string(::getpid()) + ".bin");
    std::vector<std::uint8_t> pattern(bytes);
    for (std::size_t i = 0; i < bytes; ++i)
      pattern[i] = static_cast<std::uint8_t>((i * 31u + 7u) & 0xFFu);
    std::FILE* out = std::fopen(path.c_str(), "wb");
    if (out == nullptr) return;
    (void)std::fwrite(pattern.data(), 1, bytes, out);
    (void)std::fclose(out);
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    void* address = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) return;
    data = static_cast<const std::uint8_t*>(address);
  }
  ~Mapping() {
    if (data != nullptr) (void)::munmap(const_cast<std::uint8_t*>(data), size);
    if (fd >= 0) (void)::close(fd);
    std::error_code ignored;
    fs::remove(path, ignored);
  }
  [[nodiscard]] bool ok() const { return data != nullptr && fd >= 0; }
  /// True when every byte still reads back as the pattern that was written.
  [[nodiscard]] bool Intact() const {
    for (std::size_t i = 0; i < size; ++i) {
      if (data[i] != static_cast<std::uint8_t>((i * 31u + 7u) & 0xFFu))
        return false;
    }
    return true;
  }
};

void WholeAlignedRegionDropsEverything() {
  const std::size_t page = PageSize();
  Mapping map(page * 8);
  if (!map.ok()) {
    Check(false, "could create a temp file and map it");
    return;
  }
  const std::size_t dropped =
      DropMappedRangeFromPageCache(map.data, map.size, map.fd);
  std::printf("  page=%zu size=%zu dropped=%zu\n", page, map.size, dropped);
  Check(dropped == map.size,
        "a page-aligned region of whole pages drops all of it");
  // THE SAFETY PROPERTY. Read every byte back through the same mapping.
  Check(map.Intact(),
        "and every byte re-faults from the file unchanged -- the drop is "
        "lossless on a PROT_READ MAP_PRIVATE mapping");
  // Idempotent: dropping twice is not an error and not a double-free.
  const std::size_t again =
      DropMappedRangeFromPageCache(map.data, map.size, map.fd);
  Check(again == map.size, "dropping an already-dropped range is a no-op, not "
                           "an error");
  Check(map.Intact(), "and the bytes survive the second drop too");
}

void UnalignedStartRoundsUpAndEndRoundsDown() {
  const std::size_t page = PageSize();
  Mapping map(page * 4);
  if (!map.ok()) {
    Check(false, "could create a temp file and map it");
    return;
  }
  // Offer pages 0..3 starting one byte in and ending one byte early. Only the
  // two whole pages fully inside that span may be dropped: page 0 is partial
  // at the head, page 3 is partial at the tail.
  const std::size_t offered = map.size - 2;
  const std::size_t dropped =
      DropMappedRangeFromPageCache(map.data + 1, offered, map.fd);
  std::printf("  offered=%zu (from +1) dropped=%zu expected=%zu\n", offered,
              dropped, page * 2);
  Check(dropped == page * 2,
        "an unaligned span drops only the whole pages strictly inside it: "
        "start rounds up, end rounds down");
  Check(map.Intact(), "and the partial edge pages are untouched and correct");
}

void TooSmallOrDegenerateDropsNothing() {
  const std::size_t page = PageSize();
  Mapping map(page * 2);
  if (!map.ok()) {
    Check(false, "could create a temp file and map it");
    return;
  }
  Check(DropMappedRangeFromPageCache(nullptr, page, map.fd) == 0,
        "a null pointer drops nothing instead of faulting");
  Check(DropMappedRangeFromPageCache(map.data, 0, map.fd) == 0,
        "a zero length drops nothing");
  // A span shorter than one page, offset into a page, contains no whole page.
  Check(DropMappedRangeFromPageCache(map.data + 1, page - 2, map.fd) == 0,
        "a span smaller than the page it sits in drops nothing rather than "
        "rounding outward into a page it was not offered");
  Check(map.Intact(), "and nothing was disturbed");
}

void NoFileDescriptorStillReleasesTheMapping() {
  const std::size_t page = PageSize();
  Mapping map(page * 4);
  if (!map.ok()) {
    Check(false, "could create a temp file and map it");
    return;
  }
  // fd -1 is what GgufMappedRegion carries for an in-memory image. The madvise
  // half must still run; only the cache invalidation is skipped.
  const std::size_t dropped =
      DropMappedRangeFromPageCache(map.data, map.size, -1);
  Check(dropped == map.size,
        "fd -1 (an in-memory GGUF image) still releases the mapping's pages");
  Check(map.Intact(), "and the bytes are still right");
}

/// The drop's return value is the span it ASKED to drop. This checks the pages
/// really left the page cache (measured: a "dropped" receipt came back
/// while the drafter mapping stayed 99% resident). The file must be synced:
/// dirty pages cannot be invalidated, which is the kernel's rule, not ours.
void DropReallyEmptiesThePageCache() {
  const std::size_t page = PageSize();
  Mapping map(page * 256);
  if (!map.ok()) {
    Check(false, "could create a temp file and map it");
    return;
  }
  // tmpfs pages ARE the file: nothing can drop them. Run with TMPDIR on disk.
  struct statfs info {};
  if (::fstatfs(map.fd, &info) == 0 && info.f_type == 0x01021994) {
    std::printf("[SKIP] temp dir is tmpfs; set TMPDIR to a disk dir\n");
    return;
  }
  Check(::fsync(map.fd) == 0, "the file syncs, so its cache pages are clean");
  PrefaultMappedRange(map.data, map.size);
  const std::size_t before = ResidentMappedBytes(map.data, map.size);
  const std::size_t dropped =
      DropMappedRangeFromPageCache(map.data, map.size, map.fd);
  const std::size_t after = ResidentMappedBytes(map.data, map.size);
  std::printf("  resident_before=%zu dropped=%zu resident_after=%zu\n", before,
              dropped, after);
  Check(before >= map.size / 100 * 99, "a prefaulted mapping is (nearly) fully resident");
  Check(after < map.size / 10,
        "after the drop, under a tenth of it is still in the page cache");
  Check(map.Intact(), "and the bytes re-fault unchanged");
  Check(ResidentMappedBytes(nullptr, page) == 0 &&
            ResidentMappedBytes(map.data, 0) == 0,
        "null or empty spans report 0 instead of faulting");
}

}  // namespace

int main() {
  DropReallyEmptiesThePageCache();
  WholeAlignedRegionDropsEverything();
  UnalignedStartRoundsUpAndEndRoundsDown();
  TooSmallOrDegenerateDropsNothing();
  NoFileDescriptorStillReleasesTheMapping();
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

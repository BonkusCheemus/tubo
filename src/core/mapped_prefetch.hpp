#ifndef GUFO_CORE_MAPPED_PREFETCH_HPP_
#define GUFO_CORE_MAPPED_PREFETCH_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace gufo::core {

/// Populate an existing readable mapping before HIP registration or upload.
/// Parallel faults expose disk queue depth without making another weight copy.
/// The mapping must remain alive until this call returns, including on failure.
inline void PrefaultMappedRange(const void* data, std::size_t bytes) {
  if (bytes == 0)
    return;
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (!data || page_size <= 0)
    throw std::invalid_argument("invalid mapped weight range");
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address)
    throw std::length_error("mapped weight address overflows");
  const auto skip = address % static_cast<std::size_t>(page_size);
  if (bytes > std::numeric_limits<std::size_t>::max() - skip)
    throw std::length_error("mapped weight range overflows");
  const auto begin = address - skip;
  bytes += skip;
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  constexpr std::size_t kReaders = 16;
  const auto chunks = bytes / kChunkBytes + (bytes % kChunkBytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto populate = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      if (::madvise(reinterpret_cast<void*>(begin + offset),
                    std::min(kChunkBytes, bytes - offset),
                    MADV_POPULATE_READ) != 0)
        failure.store(errno, std::memory_order_relaxed);
    }
  };
  {
    std::vector<std::jthread> readers;
    for (std::size_t i = 1; i < std::min(kReaders, chunks); ++i)
      readers.emplace_back(populate);
    populate();
    // Join before a caller can register, upload, or unmap any of these pages.
  }
  if (const int code = failure.load(); code != 0)
    throw std::system_error(code, std::generic_category(),
                            "cannot read mapped weights");
}

/// Give back a mapping's pages, and the file's clean page cache, once
/// the bytes have been copied somewhere else. Returns the bytes released.
///
/// WHY THIS EXISTS BESIDE THE PREFAULT. `PrefaultMappedRange` above makes a
/// whole weight file resident so the upload reads it at disk queue depth
/// instead of one fault at a time. After the upload those pages are DEAD: every
/// tensor now lives in a device allocation and nothing reads the mapping again.
/// They are clean and file-backed, so they cannot OOM anything -- but on a
/// 31 GiB UMA box they are a gigabyte of reclaim pressure sitting next to the
/// target's own file-backed weight pages, and reclaiming THOSE is what runs the
/// MMU notifier and makes KFD evict every queue. Free memory, not available
/// memory, is what the stall threshold is measured against here.
///
/// THE ORDER IS THE WHOLE TRICK, and getting it backwards does nothing:
///   1. madvise(MADV_DONTNEED) drops this process's page table entries. On a
///      PROT_READ MAP_PRIVATE file mapping that is LOSSLESS -- a later access
///      re-faults the same bytes from the same file. It never writes back and
///      can never lose data, because a read-only private mapping has no dirty
///      private copies to lose.
///   2. posix_fadvise(POSIX_FADV_DONTNEED) then invalidates the clean cache.
///      This is the step that actually frees memory, and it only works on pages
///      no process still maps -- which is what step 1 just arranged. Called
///      alone it is very nearly a no-op.
///
/// ADVISORY, so it never fails a load: every error path returns 0 and leaves
/// the mapping exactly as it was. A caller that keeps reading the mapping pays
/// re-faults, never a wrong answer.
///
/// START ROUNDS UP AND END ROUNDS DOWN. A partial page at either edge can hold
/// bytes belonging to something else, and MADV_DONTNEED on an unaligned start
/// is EINVAL regardless.
[[nodiscard]] inline std::size_t DropMappedRangeFromPageCache(
    const void* data, std::size_t bytes, int file_descriptor) noexcept {
  if (data == nullptr || bytes == 0)
    return 0;
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return 0;
  const auto page = static_cast<std::size_t>(page_size);
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto begin = (address + (page - 1)) & ~static_cast<std::uintptr_t>(page - 1);
  const auto head = static_cast<std::size_t>(begin - address);
  if (bytes <= head)
    return 0;
  const std::size_t span = (bytes - head) & ~(page - 1);
  if (span == 0)
    return 0;
  if (::madvise(reinterpret_cast<void*>(begin), span, MADV_DONTNEED) != 0)
    return 0;
  // Whole file: one GGUF shard is one fd, and the region above is that shard's
  // only mapping, so there is no other live range in this file to protect.
  if (file_descriptor >= 0)
    (void)::posix_fadvise(file_descriptor, 0, 0, POSIX_FADV_DONTNEED);
  return span;
}

/// Bytes of [data, data+bytes) whose file pages are in the page cache right now
/// (mincore on a file mapping reports cache residency, mapped by us or not).
/// The drop above returns the span it asked to drop; this is the
/// measurement that tells whether the pages actually went (measured RAM ledger:
/// the drafter mapping was 99% resident live despite a "dropped" receipt).
/// Advisory like the drop: any error returns 0. ponytail: one heap vector of
/// bytes/4096 entries (~256 KB for the 1 GB drafter); chunk it if mapped files
/// reach tens of GB.
[[nodiscard]] inline std::size_t ResidentMappedBytes(const void* data,
                                                     std::size_t bytes) noexcept {
  if (data == nullptr || bytes == 0)
    return 0;
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return 0;
  const auto page = static_cast<std::size_t>(page_size);
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto begin = address & ~static_cast<std::uintptr_t>(page - 1);
  const std::size_t span = bytes + (address - begin);
  const std::size_t pages = (span + page - 1) / page;
  std::vector<unsigned char> resident;
  try {
    resident.resize(pages);
  } catch (...) {
    return 0;
  }
  if (::mincore(reinterpret_cast<void*>(begin), span, resident.data()) != 0)
    return 0;
  std::size_t count = 0;
  for (const unsigned char flag : resident)
    count += flag & 1U;
  return count * page;
}

}  // namespace gufo::core

#endif  // GUFO_CORE_MAPPED_PREFETCH_HPP_

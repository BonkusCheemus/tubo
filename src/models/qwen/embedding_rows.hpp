#ifndef GUFO_MODELS_QWEN_EMBEDDING_ROWS_HPP_
#define GUFO_MODELS_QWEN_EMBEDDING_ROWS_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen {

struct EmbeddingResidentSpan {
  std::size_t offset;
  std::size_t size;
};

// Retain shared boundary pages; exclude only pages wholly inside the tensor.
std::vector<EmbeddingResidentSpan> EmbeddingResidentSpans(
    std::size_t file_size, std::size_t tensor_offset, std::size_t tensor_bytes,
    std::size_t page_size);

// Geometric staging growth, shared with the CPU capacity/overflow checks.
std::size_t EmbeddingStageCapacity(std::size_t required_rows,
                                   std::size_t current_rows,
                                   std::size_t row_bytes);

std::uint32_t EmbeddingCrc32c(std::span<const std::uint8_t> bytes);

// CPU-only encoded-row store. File is immutable for its entire open lifetime.
// Failures after Open throw; callers must not substitute stale/resident rows.
class EmbeddingRowStore {
public:
  static constexpr std::size_t kRecordBytes = 4096;
  static constexpr std::size_t kDefaultCacheBytes = 16ULL << 20;
  static std::unique_ptr<EmbeddingRowStore> Open(
      const core::GgufReader& reader, const std::filesystem::path& path,
      std::size_t cache_bytes, std::string* error);
  static bool Export(const core::GgufReader& reader,
                     const std::filesystem::path& path, std::string* error);
  static std::size_t CacheBytesFromEnvironment();
  ~EmbeddingRowStore();
  EmbeddingRowStore(const EmbeddingRowStore&) = delete;
  EmbeddingRowStore& operator=(const EmbeddingRowStore&) = delete;

  // Output is packed encoded rows in input order, including duplicates.
  // Prefill deduplicates misses and joins eight pread workers before return.
  void Fetch(std::span<const std::uint32_t> ids, std::span<std::uint8_t> packed,
             bool prefill);
  std::size_t RowBytes() const noexcept { return row_bytes_; }
  std::size_t Shard() const noexcept { return shard_; }
  std::size_t TensorOffset() const noexcept { return tensor_offset_; }
  std::size_t TensorBytes() const noexcept { return row_bytes_ * rows_; }
  std::size_t CacheBudgetBytes() const noexcept {
    return cache_count_ * kRecordBytes;
  }
  std::size_t CachedBytes() const;
  std::uint64_t Reads() const noexcept { return reads_.load(); }
  std::uint64_t Hits() const noexcept { return hits_.load(); }

private:
  EmbeddingRowStore() = default;
  using Record = std::array<std::uint8_t, kRecordBytes>;
  struct Entry {
    std::uint32_t id;
    Record bytes;
  };
  void ReadOne(std::uint32_t id, std::uint8_t* destination);
  int fd_{-1};
  std::size_t shard_{0};
  std::size_t tensor_offset_{0};
  std::size_t row_bytes_{0};
  std::size_t rows_{0};
  std::size_t cache_count_{0};
  mutable std::mutex mutex_;
  // Serializes batch assembly, not the reads within a batch.
  std::mutex fetch_mutex_;
  std::list<Entry> lru_;
  std::unordered_map<std::uint32_t, std::list<Entry>::iterator> index_;
  std::atomic<std::uint64_t> reads_{0};
  std::atomic<std::uint64_t> hits_{0};
};

}  // namespace gufo::models::qwen
#endif

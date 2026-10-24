#include "src/models/qwen/embedding_rows.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <latch>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_identity.hpp"

namespace gufo::models::qwen {
namespace {

constexpr std::size_t kPage = EmbeddingRowStore::kRecordBytes;
constexpr std::size_t kPayload =
    8;  // CRC32C, token ID, then unchanged GGUF bytes.
constexpr std::size_t kPrefillWorkers = 8;
constexpr char kMagic[8] = {'G', 'U', 'F', 'O', 'E', 'M', 'B', '1'};
static_assert(std::endian::native == std::endian::little);

struct Header {
  char magic[8];
  std::uint32_t version;
  std::uint32_t type;
  std::uint64_t hidden;
  std::uint64_t rows;
  std::uint64_t row_bytes;
  std::uint64_t shard;
  std::uint64_t tensor_offset;
  char model_sha256[64];
  char rows_sha256[64];
  std::uint32_t crc32c;
};
static_assert(sizeof(Header) <= kPage);

struct Geometry {
  const core::GgufTensorInfo* tensor;
  std::size_t shard;
  std::size_t offset;
  std::size_t row_bytes;
};

[[noreturn]] void IoError(const char* message) {
  throw std::system_error(errno, std::generic_category(), message);
}

Geometry Describe(const core::GgufReader& reader) {
  const auto* tensor = reader.FindTensor("token_embd.weight");
  if (!tensor || tensor->dimensions.size() != 2 ||
      tensor->type != core::GgmlType::kQ4_K || tensor->dimensions[0] == 0 ||
      tensor->dimensions[0] % 256 != 0 || tensor->dimensions[1] == 0 ||
      tensor->dimensions[1] > std::numeric_limits<std::uint32_t>::max() ||
      tensor->dimensions[0] / 256 > (kPage - kPayload) / 144) {
    throw std::invalid_argument(
        "SSD embedding requires Q4_K rows fitting 4096 bytes");
  }
  const auto row_bytes =
      static_cast<std::size_t>(tensor->dimensions[0] / 256 * 144);
  if (tensor->dimensions[1] >
          std::numeric_limits<std::size_t>::max() / row_bytes ||
      tensor->size_bytes != row_bytes * tensor->dimensions[1])
    throw std::invalid_argument("invalid embedding tensor extent");
  const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
  const auto regions = reader.GetMappedRegions();
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
    if (address >= base && address - base <= regions[i].size &&
        tensor->size_bytes <= regions[i].size - (address - base)) {
      return {tensor, i, static_cast<std::size_t>(address - base), row_bytes};
    }
  }
  throw std::invalid_argument("embedding does not belong to a GGUF shard");
}

void ReadExact(int fd, void* destination, std::size_t bytes,
               std::uint64_t offset) {
  auto* data = static_cast<std::uint8_t*>(destination);
  for (std::size_t done = 0; done < bytes;) {
    const auto got = ::pread(fd, data + done, bytes - done,
                             static_cast<off_t>(offset + done));
    if (got < 0 && errno == EINTR)
      continue;
    if (got < 0)
      IoError("embedding source read");
    if (got == 0)
      throw std::runtime_error("embedding source truncated");
    done += static_cast<std::size_t>(got);
  }
}

void WriteExact(int fd, const void* source, std::size_t bytes,
                std::uint64_t offset) {
  const auto* data = static_cast<const std::uint8_t*>(source);
  for (std::size_t done = 0; done < bytes;) {
    const auto got = ::pwrite(fd, data + done, bytes - done,
                              static_cast<off_t>(offset + done));
    if (got < 0 && errno == EINTR)
      continue;
    if (got < 0)
      IoError("embedding export write");
    if (got == 0)
      throw std::runtime_error("embedding export short write");
    done += static_cast<std::size_t>(got);
  }
}

// A partial O_DIRECT record is an error. Never retry at an unaligned offset.
void ReadRecord(int fd, std::span<std::uint8_t, kPage> record,
                std::uint64_t offset) {
  ssize_t got;
  do {
    got = ::pread(fd, record.data(), record.size(), static_cast<off_t>(offset));
  } while (got < 0 && errno == EINTR);
  if (got < 0)
    IoError("embedding O_DIRECT read");
  if (got != static_cast<ssize_t>(kPage))
    throw std::runtime_error("embedding row file truncated");
}

}  // namespace

std::uint32_t EmbeddingCrc32c(std::span<const std::uint8_t> bytes) {
  static constexpr auto table = [] {
    std::array<std::uint32_t, 256> result{};
    for (std::uint32_t i = 0; i < result.size(); ++i) {
      auto crc = i;
      for (int bit = 0; bit < 8; ++bit)
        crc = (crc >> 1) ^ ((crc & 1U) ? 0x82f63b78U : 0U);
      result[i] = crc;
    }
    return result;
  }();
  std::uint32_t crc = ~0U;
  for (auto byte : bytes)
    crc = table[(crc ^ byte) & 0xffU] ^ (crc >> 8);
  return ~crc;
}

std::vector<EmbeddingResidentSpan> EmbeddingResidentSpans(
    std::size_t file_size, std::size_t tensor_offset, std::size_t tensor_bytes,
    std::size_t page_size) {
  if (page_size == 0 || tensor_offset > file_size ||
      tensor_bytes > file_size - tensor_offset)
    throw std::invalid_argument("invalid embedding exclusion range");
  const auto end = tensor_offset + tensor_bytes;
  const auto remainder = tensor_offset % page_size;
  const auto adjustment = remainder ? page_size - remainder : 0;
  if (adjustment > tensor_bytes)
    return {{0, file_size}};
  const auto begin = tensor_offset + adjustment;
  const auto stop = end - end % page_size;
  if (stop <= begin)
    return {{0, file_size}};
  std::vector<EmbeddingResidentSpan> result;
  if (begin)
    result.push_back({0, begin});
  if (stop < file_size)
    result.push_back({stop, file_size - stop});
  return result;
}

std::size_t EmbeddingStageCapacity(std::size_t required_rows,
                                   std::size_t current_rows,
                                   std::size_t row_bytes) {
  if (row_bytes == 0 || required_rows == 0)
    throw std::invalid_argument("invalid embedding staging geometry");
  const auto maximum = std::min(
      {std::numeric_limits<std::size_t>::max() / row_bytes,
       std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t),
       static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())});
  if (required_rows > maximum || current_rows > maximum)
    throw std::length_error("embedding staging capacity overflows");
  if (required_rows <= current_rows)
    return current_rows;
  const auto doubled = current_rows > maximum / 2 ? maximum : current_rows * 2;
  return std::max(required_rows, doubled);
}

std::size_t EmbeddingRowStore::CacheBytesFromEnvironment() {
  const char* value = std::getenv("GUFO_DEV_EMBED_SSD_CACHE_MB");
  if (!value)
    return kDefaultCacheBytes;
  std::size_t mib = 0;
  const auto* end = value + std::strlen(value);
  const auto parsed = std::from_chars(value, end, mib);
  if (parsed.ec != std::errc{} || parsed.ptr != end || mib == 0 ||
      mib > (std::numeric_limits<std::size_t>::max() >> 20))
    throw std::invalid_argument(
        "GUFO_DEV_EMBED_SSD_CACHE_MB must be positive MiB");
  return mib << 20;
}

EmbeddingRowStore::~EmbeddingRowStore() {
  if (fd_ >= 0) {
    std::fprintf(
        stderr,
        "GUFO_DEV_EMBED_SSD close reads=%llu hits=%llu cache_bytes=%zu\n",
        static_cast<unsigned long long>(Reads()),
        static_cast<unsigned long long>(Hits()), CachedBytes());
    (void)::close(fd_);
  }
}

std::unique_ptr<EmbeddingRowStore> EmbeddingRowStore::Open(
    const core::GgufReader& reader, const std::filesystem::path& path,
    std::size_t cache_bytes, std::string* error) {
  try {
    const auto geometry = Describe(reader);
    if (reader.GetMappedRegions()[geometry.shard].file_descriptor < 0)
      throw std::invalid_argument(
          "SSD embedding requires file-backed GGUF storage");
    if (!reader.FindTensor("output.weight"))
      throw std::invalid_argument("tied output requires resident embeddings");
    const auto begin = reinterpret_cast<std::uintptr_t>(geometry.tensor->data);
    for (const auto& tensor : reader.GetTensors()) {
      if (tensor.name == "token_embd.weight")
        continue;
      const auto address = reinterpret_cast<std::uintptr_t>(tensor.data);
      if ((address >= begin && address - begin < geometry.tensor->size_bytes) ||
          (address < begin && begin - address < tensor.size_bytes))
        throw std::invalid_argument(
            "tied/aliased embedding storage requires residency");
    }
    if (cache_bytes < kPage)
      throw std::invalid_argument(
          "embedding cache must hold at least one record");
    std::unique_ptr<EmbeddingRowStore> store(new EmbeddingRowStore());
    store->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (store->fd_ < 0)
      IoError("open embedding row file");
    struct stat info{};
    if (::fstat(store->fd_, &info) != 0)
      IoError("stat embedding row file");
    const auto rows = geometry.tensor->dimensions[1];
    const auto expected = (rows + 1) * kPage;
    if (!S_ISREG(info.st_mode) || info.st_size < 0 ||
        static_cast<std::uint64_t>(info.st_size) != expected)
      throw std::invalid_argument("embedding row file extent mismatch");
    alignas(kPage) Record page{};
    ReadRecord(store->fd_, page, 0);  // Also probes actual direct-I/O support.
    Header head{};
    std::memcpy(&head, page.data(), sizeof(head));
    std::memset(page.data() + offsetof(Header, crc32c), 0, sizeof(head.crc32c));
    if (std::memcmp(head.magic, kMagic, sizeof(kMagic)) != 0 ||
        head.version != 1 || head.crc32c != EmbeddingCrc32c(page) ||
        head.type != static_cast<std::uint32_t>(geometry.tensor->type) ||
        head.hidden != geometry.tensor->dimensions[0] || head.rows != rows ||
        head.row_bytes != geometry.row_bytes || head.shard != geometry.shard ||
        head.tensor_offset != geometry.offset)
      throw std::invalid_argument("embedding row file header mismatch");
    const auto identity = core::GgufIdentityHex(reader);
    if (identity.size() != sizeof(head.model_sha256) ||
        std::memcmp(identity.data(), head.model_sha256, identity.size()) != 0)
      throw std::invalid_argument(
          "embedding row file source identity mismatch");
    store->shard_ = geometry.shard;
    store->tensor_offset_ = geometry.offset;
    store->row_bytes_ = geometry.row_bytes;
    store->rows_ = static_cast<std::size_t>(rows);
    store->cache_count_ = cache_bytes / kPage;
    return store;
  } catch (const std::exception& exception) {
    if (error)
      *error = exception.what();
    return nullptr;
  }
}

bool EmbeddingRowStore::Export(const core::GgufReader& reader,
                               const std::filesystem::path& path,
                               std::string* error) {
  int fd = -1;
  std::string temporary;
  try {
    const auto geometry = Describe(reader);
    const auto& tensor = *geometry.tensor;
    const auto identity = core::GgufIdentityHex(reader);
    temporary = path.string() + ".tmp.XXXXXX";
    fd = ::mkstemp(temporary.data());
    if (fd < 0)
      IoError("create embedding export");
    Header head{};
    std::memcpy(head.magic, kMagic, sizeof(kMagic));
    head.version = 1;
    head.type = static_cast<std::uint32_t>(tensor.type);
    head.hidden = tensor.dimensions[0];
    head.rows = tensor.dimensions[1];
    head.row_bytes = geometry.row_bytes;
    head.shard = geometry.shard;
    head.tensor_offset = geometry.offset;
    std::memcpy(head.model_sha256, identity.data(), sizeof(head.model_sha256));
    crypto::Sha256Hasher rows_hash;
    const auto source_fd =
        reader.GetMappedRegions()[geometry.shard].file_descriptor;
    constexpr std::size_t kDropChunk = 16ULL << 20;
    std::size_t advised = 0;
    for (std::uint32_t row = 0; row < head.rows; ++row) {
      Record record{};
      std::memcpy(record.data() + sizeof(std::uint32_t), &row, sizeof(row));
      if (source_fd >= 0) {
        ReadExact(source_fd, record.data() + kPayload, geometry.row_bytes,
                  geometry.offset + row * geometry.row_bytes);
      } else {
        std::memcpy(record.data() + kPayload,
                    static_cast<const std::uint8_t*>(tensor.data) +
                        row * geometry.row_bytes,
                    geometry.row_bytes);
      }
      rows_hash.Update({record.data() + kPayload, geometry.row_bytes});
      const auto checksum =
          EmbeddingCrc32c(std::span(record).subspan(sizeof(std::uint32_t)));
      std::memcpy(record.data(), &checksum, sizeof(checksum));
      WriteExact(fd, record.data(), record.size(),
                 (std::uint64_t{row} + 1) * kPage);
      const auto through =
          (static_cast<std::size_t>(row) + 1) * geometry.row_bytes;
      if (source_fd >= 0 && through - advised >= kDropChunk) {
        (void)::posix_fadvise(
            source_fd, static_cast<off_t>(geometry.offset + advised),
            static_cast<off_t>(through - advised), POSIX_FADV_DONTNEED);
        advised = through;
      }
    }
    const auto digest = rows_hash.FinishHex();
    std::memcpy(head.rows_sha256, digest.data(), sizeof(head.rows_sha256));
    Record header{};
    std::memcpy(header.data(), &head, sizeof(head));
    const auto crc = EmbeddingCrc32c(header);
    std::memcpy(header.data() + offsetof(Header, crc32c), &crc, sizeof(crc));
    WriteExact(fd, header.data(), header.size(), 0);
    if (core::GgufIdentityHex(reader) != identity)
      throw std::runtime_error("GGUF changed during embedding export");
    if (::fsync(fd) != 0)
      IoError("sync embedding export");
    // Atomic publish without overwriting an existing artifact or source GGUF.
    if (::link(temporary.c_str(), path.c_str()) != 0)
      IoError("publish embedding export (destination must not exist)");
    (void)::unlink(temporary.c_str());
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    (void)::close(fd);
    return true;
  } catch (const std::exception& exception) {
    if (fd >= 0) {
      (void)::close(fd);
      (void)::unlink(temporary.c_str());  // Only unlink a temp file we created.
    }
    if (error)
      *error = exception.what();
    return false;
  }
}

std::size_t EmbeddingRowStore::CachedBytes() const {
  std::lock_guard lock(mutex_);
  return lru_.size() * kRecordBytes;
}

void EmbeddingRowStore::ReadOne(std::uint32_t id, std::uint8_t* destination) {
  {
    std::lock_guard lock(mutex_);
    if (const auto found = index_.find(id); found != index_.end()) {
      lru_.splice(lru_.begin(), lru_, found->second);
      std::memcpy(destination, found->second->bytes.data() + kPayload,
                  row_bytes_);
      ++hits_;
      return;
    }
  }
  alignas(kPage) Record record{};
  ReadRecord(fd_, record, (std::uint64_t{id} + 1) * kPage);
  std::uint32_t checksum = 0, stored_id = 0;
  std::memcpy(&checksum, record.data(), sizeof(checksum));
  std::memcpy(&stored_id, record.data() + sizeof(checksum), sizeof(stored_id));
  if (stored_id != id ||
      checksum != EmbeddingCrc32c(std::span(record).subspan(sizeof(checksum))))
    throw std::runtime_error("embedding row ID/CRC32C mismatch");
  ++reads_;
  std::memcpy(destination, record.data() + kPayload, row_bytes_);
  std::lock_guard lock(mutex_);
  if (lru_.size() == cache_count_) {
    index_.erase(lru_.back().id);
    lru_.pop_back();
  }
  lru_.push_front({id, record});
  try {
    index_.emplace(id, lru_.begin());
  } catch (...) {
    lru_.pop_front();
    throw;
  }
}

void EmbeddingRowStore::Fetch(std::span<const std::uint32_t> ids,
                              std::span<std::uint8_t> packed, bool prefill) {
  if (ids.size() > std::numeric_limits<std::size_t>::max() / row_bytes_ ||
      packed.size() != ids.size() * row_bytes_)
    throw std::invalid_argument("embedding packed output extent mismatch");
  std::lock_guard batch_lock(fetch_mutex_);
  std::vector<std::uint32_t> unique;
  std::vector<std::size_t> slots;
  std::unordered_map<std::uint32_t, std::size_t> positions;
  for (const auto id : ids) {
    if (id >= rows_)
      throw std::out_of_range("embedding token ID outside vocabulary");
    const auto [it, inserted] = positions.emplace(id, unique.size());
    if (inserted)
      unique.push_back(id);
    slots.push_back(it->second);
  }
  std::vector<std::uint8_t> rows(unique.size() * row_bytes_);
  std::atomic<std::size_t> next{0};
  std::atomic<bool> stopped{false};
  std::mutex failure_mutex;
  std::exception_ptr failure;
  const auto run = [&] {
    try {
      while (!stopped.load(std::memory_order_acquire)) {
        const auto index = next.fetch_add(1);
        if (index >= unique.size() || stopped.load(std::memory_order_acquire))
          break;
        ReadOne(unique[index], rows.data() + index * row_bytes_);
      }
    } catch (...) {
      stopped.store(true, std::memory_order_release);
      std::lock_guard lock(failure_mutex);
      if (!failure)
        failure = std::current_exception();
    }
  };
  if (prefill && unique.size() > 1) {
    // No liburing dependency. Fixed eight-worker batch, joined before gather.
    std::latch start(1);
    std::vector<std::jthread> workers;
    try {
      for (std::size_t i = 0; i < kPrefillWorkers; ++i)
        workers.emplace_back([&] {
          start.wait();
          run();
        });
    } catch (...) {
      stopped.store(true, std::memory_order_release);
      start.count_down();  // Do not strand workers if thread creation fails.
      throw;
    }
    start.count_down();  // All eight readers exist before the first miss is
                         // issued.
  } else {
    run();  // Decode stays plain synchronous pread.
  }
  if (failure)
    std::rethrow_exception(failure);
  for (std::size_t i = 0; i < ids.size(); ++i)
    std::memcpy(packed.data() + i * row_bytes_,
                rows.data() + slots[i] * row_bytes_, row_bytes_);
}

}  // namespace gufo::models::qwen

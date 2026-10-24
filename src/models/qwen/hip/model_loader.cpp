#if defined(ENGINE_ENABLE_HIP)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/iq3_pair_view.hpp"
#include "src/models/qwen/sn_weight_plan.hpp"

namespace gufo::hip {
namespace {

void ReleaseWeightRegions(std::vector<QwenGpuWeightRegion>& regions) noexcept {
  for (auto& region : regions) {
    if (region.host_copy != nullptr) {
      if (region.register_chunk == 0) {
        (void)hipHostUnregister(region.host_copy);
      } else {
        for (std::size_t offset = 0; offset < region.size;
             offset += region.register_chunk)
          (void)hipHostUnregister(static_cast<char*>(region.host_copy) + offset);
      }
      (void)munmap(region.host_copy, region.size);
    } else if (region.device_data != nullptr) {
      (void)hipFree(region.device_data);  // GUFO_DEV_WEIGHTS_GTT copy
    }
    region = {};
  }
}

struct SnLoadTensor {
  models::QwenTensorRef* tensor;
  std::size_t rows;
  std::size_t columns;
  std::size_t block_bytes;
  models::QwenWeightLayout layout = models::QwenWeightLayout::kRows32;
  models::QwenTensorRef* paired_up = nullptr;
};

std::vector<models::SnWeightCopyJob> PlanRegionCopies(
    const core::GgufMappedRegion& source,
    std::span<const SnLoadTensor> selected,
    std::size_t native_chunk_bytes = 16ULL << 20) {
  std::vector<models::SnWeightSpan> local;
  const auto base = reinterpret_cast<std::uintptr_t>(source.data);
  for (const auto& t : selected) {
    const auto address = reinterpret_cast<std::uintptr_t>(t.tensor->data);
    const auto bytes = t.rows * (t.columns / 256) * t.block_bytes;
    if (address < base) {
      if (base - address < bytes)
        throw std::invalid_argument("Rows32 tensor crosses resident span");
      continue;
    }
    const auto offset = address - base;
    if (offset >= source.size)
      continue;
    if (bytes > source.size - offset)
      throw std::invalid_argument("Rows32 tensor crosses resident span");
    local.push_back({offset, t.rows, t.columns, t.block_bytes, t.layout});
  }
  const auto jobs =
      models::PlanSnWeightCopies(source.size, local, native_chunk_bytes);
  return models::CoalesceIq3CopyJobs(jobs, native_chunk_bytes);
}

void CopyMappedWeights(const core::GgufMappedRegion& source, void* copy,
                       std::size_t file_offset,
                       std::span<const SnLoadTensor> selected) {
  const char* fadv_dontneed = std::getenv("GUFO_DEV_FADV_DONTNEED");
  const bool use_fadv_dontneed =
      fadv_dontneed != nullptr && std::string_view(fadv_dontneed) == "1";
  if (use_fadv_dontneed && source.file_descriptor < 0)
    throw std::system_error(EBADF, std::generic_category(),
                            "FADV_DONTNEED requires file-backed weights");
  const auto jobs = PlanRegionCopies(source, selected);
  const auto chunks = jobs.size();
  const long page = ::sysconf(_SC_PAGESIZE);
  if (!selected.empty() && (source.file_descriptor < 0 || page <= 0))
    throw std::invalid_argument(
        "Rows32 loading requires file-backed GGUF pages");
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto copy_chunk = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto& job = jobs[index];
      const auto offset = job.source_offset;
      const auto bytes = job.source_bytes;
      auto* input = const_cast<std::uint8_t*>(
                        static_cast<const std::uint8_t*>(source.data)) +
                    offset;
      // A tile can begin mid-page. Advise its file-backed pages, while the
      // copy itself touches exactly the validated job span. Native mode keeps
      // its existing page-aligned 16MiB advice unchanged.
      const auto prefix = selected.empty()
                              ? 0
                              : reinterpret_cast<std::uintptr_t>(input) %
                                    static_cast<std::size_t>(page);
      auto* advise = input - prefix;
      if (madvise(advise, bytes + prefix, MADV_POPULATE_READ) != 0) {
        failure.store(errno, std::memory_order_relaxed);
        break;
      }
      models::ExecuteSnWeightCopyJob(source.data, copy, job);
      // The immutable copy now owns these resident bytes. Keep the original
      // mapping valid for the reader without retaining its populated PTEs.
      (void)madvise(advise, bytes + prefix, MADV_DONTNEED);
      if (use_fadv_dontneed) {
        const auto error = ::posix_fadvise(
            source.file_descriptor, static_cast<off_t>(file_offset + offset),
            static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
        if (error != 0) {
          failure.store(error, std::memory_order_relaxed);
          break;
        }
      }
    }
  };
  {
    std::vector<std::jthread> workers;
    for (std::size_t i = 1; i < std::min<std::size_t>(16, chunks); ++i)
      workers.emplace_back(copy_chunk);
    copy_chunk();
  }
  if (const auto error = failure.load(); error != 0)
    throw std::system_error(error, std::generic_category(),
                            "cannot read/advise mapped Qwen weights");
}

// Dev switch GUFO_DEV_WEIGHTS_GTT=1: copy the weights into a driver-owned
// GTT buffer instead of registering anonymous memory (userptr). Userptr pages
// stay on the kernel's LRU, so reclaim or compaction under memory pressure
// fires the HMM notifier and KFD evicts every queue of the process until the
// whole range is revalidated (measured stalls of 40-290 s). GTT pages are not
// on the LRU. Same bytes resident as the mapped path: each 16 MiB source chunk
// is dropped after its copy. Upstream removed a copy mode in 310df10 (#220)
// because it kept the populated file mapping as a second 31 GB copy.
// Drop a completed immutable source span, including boundary pages. A tiled
// conversion must discard its WHOLE tensor after the last tile: later reads
// can leave earlier parts of the original file resident again.
[[nodiscard]] hipError_t DiscardGttSource(
    const core::GgufMappedRegion& source, std::size_t offset,
    std::size_t bytes, std::size_t file_offset, std::size_t page) {
  if (source.file_descriptor < 0)
    return hipSuccess;
  auto* input = const_cast<std::uint8_t*>(
                    static_cast<const std::uint8_t*>(source.data)) + offset;
  const auto prefix = reinterpret_cast<std::uintptr_t>(input) % page;
  if (madvise(input - prefix, bytes + prefix, MADV_DONTNEED) != 0) {
    std::fprintf(stderr, "GTT source discard failed: %s\n", std::strerror(errno));
    return hipErrorInvalidValue;
  }
  const auto error = ::posix_fadvise(
      source.file_descriptor, static_cast<off_t>(file_offset + offset),
      static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
  if (error != 0) {
    std::fprintf(stderr, "GTT source file discard failed: %s\n",
                 std::strerror(error));
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

[[nodiscard]] hipError_t CopyRegionToDevice(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination,
    std::size_t file_offset, std::span<const SnLoadTensor> selected) {
  // Pageable hipMemcpy pins every chunk on the fly (measured: 5 GB in 6.5 min),
  // so stage through two pinned buffers: the CPU fills one while the DMA
  // drains the other.
  constexpr std::size_t kChunkBytes = 64ULL << 20;
  const auto jobs = PlanRegionCopies(source, selected, kChunkBytes);
  for (const auto& job : jobs)
    if (job.source_bytes > kChunkBytes)
      throw std::invalid_argument("Rows32 tile exceeds existing GTT staging");
  const long page = ::sysconf(_SC_PAGESIZE);
  if (page <= 0 || (!selected.empty() && source.file_descriptor < 0))
    throw std::invalid_argument("invalid GTT source pages");
  void* device_data = nullptr;
  if (const auto error = hipMalloc(&device_data, source.size);
      error != hipSuccess)
    return error;
  void* stage[2] = {nullptr, nullptr};
  hipEvent_t drained[2] = {nullptr, nullptr};
  hipError_t error = hipSuccess;
  for (int i = 0; i < 2 && error == hipSuccess; ++i) {
    error = hipHostMalloc(&stage[i], kChunkBytes, hipHostMallocDefault);
    if (error == hipSuccess)
      error = hipEventCreateWithFlags(&drained[i], hipEventDisableTiming);
  }
  for (std::size_t n = 0; error == hipSuccess && n < jobs.size(); ++n) {
    const auto& job = jobs[n];
    const auto offset = job.source_offset;
    const auto bytes = job.source_bytes;
    const auto slot = n % 2;
    auto* input = const_cast<std::uint8_t*>(
                      static_cast<const std::uint8_t*>(source.data)) +
                  offset;
    error = hipEventSynchronize(drained[slot]);  // its previous DMA is done
    if (error != hipSuccess)
      break;
    const auto prefix = reinterpret_cast<std::uintptr_t>(input) %
                        static_cast<std::size_t>(page);
    auto* advise = input - prefix;
    if (source.file_descriptor >= 0)
      (void)madvise(advise, bytes + prefix, MADV_POPULATE_READ);
    std::array<models::SnWeightStagedCopy, 3> writes{};
    try {
      writes = models::StageSnWeightCopyJob(source.data, stage[slot],
                                           kChunkBytes, job);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "SN weight staging failed: %s\n", e.what());
      error = hipErrorInvalidValue;
      break;  // Existing unconditional drain/free handles both slots.
    } catch (...) {
      std::fprintf(stderr, "SN weight staging failed: unknown exception\n");
      error = hipErrorInvalidValue;
      break;
    }
    // Native gaps stay bounded by64MiB. Converted input is retained only
    // until its tensor ends (phase1 tensors are at most25,804,800 bytes),
    // then discarded as one span rather than leaving tiled-file PTEs behind.
    if (!job.block_bytes || job.row_end == job.rows) {
      const auto discard_offset = job.block_bytes ? job.tensor_offset : offset;
      const auto discard_bytes = job.block_bytes
          ? job.rows * (job.columns / 256) * job.block_bytes : bytes;
      error = DiscardGttSource(source, discard_offset, discard_bytes,
                                file_offset, static_cast<std::size_t>(page));
      if (error != hipSuccess)
        break;
    }
    for (const auto& write : writes) {
      if (!write.bytes || error != hipSuccess)
        continue;
      error = hipMemcpyAsync(
          static_cast<std::uint8_t*>(device_data) + write.destination_offset,
          static_cast<const std::uint8_t*>(stage[slot]) + write.stage_offset,
          write.bytes, hipMemcpyHostToDevice, nullptr);
    }
    if (error == hipSuccess)
      error = hipEventRecord(drained[slot], nullptr);
  }
  // Drain even on a failed second (metadata) enqueue, before freeing either
  // pinned buffer. A preceding payload DMA may still be using it.
  const auto sync_error = hipDeviceSynchronize();
  if (error == hipSuccess)
    error = sync_error;
  // No original-file PTEs remain when the one persistent GTT copy is
  // published. Also clears any boundary/read-around pages outside a tensor.
  if (error == hipSuccess)
    error = DiscardGttSource(source, 0, source.size, file_offset,
                              static_cast<std::size_t>(page));
  for (int i = 0; i < 2; ++i) {
    if (drained[i] != nullptr)
      (void)hipEventDestroy(drained[i]);
    if (stage[i] != nullptr)
      (void)hipHostFree(stage[i]);
  }
  if (error != hipSuccess) {
    (void)hipFree(device_data);
    return error;
  }
  std::fprintf(stderr, "GUFO_DEV_WEIGHTS_GTT bytes=%zu\n",
               static_cast<std::size_t>(source.size));
  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = nullptr,
                 .size = source.size};
  return hipSuccess;
}

[[nodiscard]] hipError_t MapRegisteredRegion(
    const core::GgufMappedRegion& source, QwenGpuWeightRegion& destination,
    std::size_t file_offset, std::span<const SnLoadTensor> selected) {
  if (const char* const gtt = std::getenv("GUFO_DEV_WEIGHTS_GTT");
      gtt != nullptr && std::string_view(gtt) == "1")
    return CopyRegionToDevice(source, destination, file_offset, selected);
  void* host_copy = mmap(nullptr, source.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_copy == MAP_FAILED)
    return hipErrorOutOfMemory;
  const auto unmap = [&](void* pointer) { (void)munmap(pointer, source.size); };
  std::unique_ptr<void, decltype(unmap)> owned_copy(host_copy, unmap);
  const char* no_hugepage = std::getenv("GUFO_DEV_NO_HUGEPAGE");
  if (no_hugepage == nullptr || std::string_view(no_hugepage) != "1")
    (void)madvise(host_copy, source.size, MADV_HUGEPAGE);
  CopyMappedWeights(source, host_copy, file_offset, selected);
  // Dev switch GUFO_DEV_THP_FREEZE=1. A live run held only 10.6 of
  // 16.4 GiB of this copy in huge pages; khugepaged then collapses the rest
  // during decode, and every collapse migrates pages under the registered
  // range (HMM notifier -> KFD queue eviction). Collapse now, synchronously,
  // then mark the range NOHUGEPAGE so khugepaged never touches it again.
  // Existing huge pages stay; a range that could not collapse stays 4K.
  if (const char* const freeze = std::getenv("GUFO_DEV_THP_FREEZE");
      freeze != nullptr && std::string_view(freeze) == "1") {
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
    int collapse = madvise(host_copy, source.size, MADV_COLLAPSE);
    if (collapse != 0 && errno == EAGAIN)  // transient LRU/lock race
      collapse = madvise(host_copy, source.size, MADV_COLLAPSE);
    const int collapse_errno = collapse == 0 ? 0 : errno;
    const int nohuge = madvise(host_copy, source.size, MADV_NOHUGEPAGE);
    // A fork would write-protect this private range: another invalidation.
    (void)madvise(host_copy, source.size, MADV_DONTFORK);
    // The outcome, not the return codes: process-wide huge-page total.
    char anon_huge[64] = "?";
    if (FILE* const rollup = std::fopen("/proc/self/smaps_rollup", "r")) {
      char line[256];
      while (std::fgets(line, sizeof(line), rollup) != nullptr)
        if (std::strncmp(line, "AnonHugePages:", 14) == 0)
          std::snprintf(anon_huge, sizeof(anon_huge), "%s", line + 14);
      std::fclose(rollup);
    }
    std::fprintf(stderr,
                 "GUFO_DEV_THP_FREEZE collapse=%d errno=%d nohugepage=%d "
                 "bytes=%zu anon_huge=%s",
                 collapse, collapse_errno, nohuge,
                 static_cast<std::size_t>(source.size), anon_huge);
  }
  // Dev switch: pin the GPU-visible weight copy in RAM so reclaim under
  // memory pressure cannot invalidate it (HMM notifier -> KFD queue eviction).
  // Needs RLIMIT_MEMLOCK >= the weights; a failure is logged and ignored.
  if (const char* const lock = std::getenv("GUFO_DEV_MLOCK");
      lock != nullptr && std::string_view(lock) == "1") {
    const int rc = mlock(host_copy, source.size);
    std::fprintf(stderr, "GUFO_DEV_MLOCK rc=%d errno=%d bytes=%zu\n", rc,
                 rc == 0 ? 0 : errno, static_cast<std::size_t>(source.size));
  }
  // Dev switch GUFO_DEV_REGISTER_CHUNK_MIB=N: register the copy as
  // consecutive N-MiB chunks instead of one range. On gfx1151 ROCr registers
  // host memory as a KFD SVM range; after a notifier hit, restore revalidates
  // the whole range with HMM faults and reschedules on every concurrent
  // invalidation (10-13: one 31 s stall, 1569 retries over 16 GiB). Chunks
  // shrink what one restore walks. Tensors need one contiguous GPU range, so
  // each chunk's device pointer must continue the first one's; otherwise all
  // chunks are dropped and the copy is registered whole, as before.
  std::size_t chunk = 0;
  if (const char* const mib = std::getenv("GUFO_DEV_REGISTER_CHUNK_MIB");
      mib != nullptr)
    chunk = static_cast<std::size_t>(std::strtoull(mib, nullptr, 10)) << 20;
  chunk = chunk / (2ULL << 20) * (2ULL << 20);  // whole huge pages
  constexpr auto kFlags = hipHostRegisterMapped | hipHostRegisterReadOnly;
  void* device_data = nullptr;
  if (chunk != 0 && chunk < source.size) {
    auto* const base = static_cast<char*>(host_copy);
    std::size_t registered = 0;  // bytes covered by successful registrations
    bool ok = true;
    for (std::size_t offset = 0; ok && offset < source.size; offset += chunk) {
      const std::size_t bytes = std::min(chunk, source.size - offset);
      if (hipHostRegister(base + offset, bytes, kFlags) != hipSuccess) {
        ok = false;
        break;
      }
      registered = offset + bytes;
      void* device_chunk = nullptr;
      if (hipHostGetDevicePointer(&device_chunk, base + offset, 0) != hipSuccess)
        ok = false;
      else if (offset == 0)
        device_data = device_chunk;
      else if (device_chunk != static_cast<char*>(device_data) + offset)
        ok = false;
    }
    std::fprintf(stderr,
                 "GUFO_DEV_REGISTER_CHUNK_MIB chunk=%zu chunks=%zu ok=%d\n",
                 chunk >> 20, (registered + chunk - 1) / chunk, ok ? 1 : 0);
    if (ok) {
      destination = {.host_data = source.data,
                     .device_data = device_data,
                     .host_copy = owned_copy.release(),
                     .size = source.size,
                     .register_chunk = chunk};
      return hipSuccess;
    }
    for (std::size_t offset = 0; offset < registered; offset += chunk)
      (void)hipHostUnregister(base + offset);
    device_data = nullptr;
  }

  const auto register_error = hipHostRegister(host_copy, source.size, kFlags);
  if (register_error != hipSuccess)
    return register_error;

  const auto pointer_error =
      hipHostGetDevicePointer(&device_data, host_copy, 0);
  if (pointer_error != hipSuccess) {
    (void)hipHostUnregister(host_copy);
    return pointer_error;
  }

  destination = {.host_data = source.data,
                 .device_data = device_data,
                 .host_copy = owned_copy.release(),
                 .size = source.size};
  return hipSuccess;
}

[[nodiscard]] bool CreateWeightRegions(
    const core::GgufReader& reader,
    std::vector<QwenGpuWeightRegion>& weight_regions,
    const models::qwen::EmbeddingRowStore* embedding_rows,
    std::span<const SnLoadTensor> selected, std::string* error_msg) {
  const auto source_regions = reader.GetMappedRegions();
  if (source_regions.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader has no mapped weight regions";
    }
    return false;
  }

  try {
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
      throw std::runtime_error("cannot determine page size for weight spans");
    for (std::size_t i = 0; i < source_regions.size(); ++i) {
      const auto& source = source_regions[i];
      const auto spans = embedding_rows && embedding_rows->Shard() == i
                             ? models::qwen::EmbeddingResidentSpans(
                                   source.size, embedding_rows->TensorOffset(),
                                   embedding_rows->TensorBytes(),
                                   static_cast<std::size_t>(page_size))
                             : std::vector<models::qwen::EmbeddingResidentSpan>{
                                   {0, source.size}};
      for (const auto& span : spans) {
        const core::GgufMappedRegion slice{
            static_cast<const std::uint8_t*>(source.data) + span.offset,
            span.size, source.file_descriptor};
        auto& destination = weight_regions.emplace_back();
        // Validate every selected span before registration/publication; no
        // native memcpy job may overwrite the converted payload or d-plane.
        (void)PlanRegionCopies(slice, selected);
        const auto error =
            MapRegisteredRegion(slice, destination, span.offset, selected);
        if (!destination.device_data)
          throw std::runtime_error("cannot map GGUF weight span: " +
                                   std::string(hipGetErrorString(error)));
      }
      if (embedding_rows && embedding_rows->Shard() == i) {
        const auto page = static_cast<std::size_t>(page_size);
        const auto offset = embedding_rows->TensorOffset();
        const auto bytes = embedding_rows->TensorBytes();
        const auto extra = (page - offset % page) % page;
        if (extra <= bytes) {
          const auto begin = offset + extra;
          const auto end = offset + bytes - (offset + bytes) % page;
          if (end > begin) {
            auto* address = const_cast<std::uint8_t*>(
                                static_cast<const std::uint8_t*>(source.data)) +
                            begin;
            (void)madvise(address, end - begin, MADV_RANDOM);
            (void)madvise(address, end - begin, MADV_DONTNEED);
            if (source.file_descriptor >= 0)
              (void)::posix_fadvise(
                  source.file_descriptor, static_cast<off_t>(begin),
                  static_cast<off_t>(end - begin), POSIX_FADV_DONTNEED);
          }
        }
      }
    }
  } catch (const std::exception& e) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg)
      *error_msg = e.what();
    return false;
  }
  return true;
}

[[nodiscard]] bool RemapTensor(
    models::QwenTensorRef& tensor,
    std::span<const QwenGpuWeightRegion> regions) noexcept {
  if (tensor.empty()) {
    return true;
  }
  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor.data);
  for (const auto& region : regions) {
    const auto region_address =
        reinterpret_cast<std::uintptr_t>(region.host_data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (offset < region.size && encoded_bytes != 0 &&
          encoded_bytes <= region.size - offset) {
        tensor.data =
            static_cast<const std::uint8_t*>(region.device_data) + offset;
        tensor.available_bytes = region.size - offset;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

QwenGpuModel::QwenGpuModel(
    std::shared_ptr<const core::GgufReader> reader,
    models::QwenModelWeights weights,
    std::shared_ptr<const tokenization::QwenTokenizer> tokenizer,
    std::vector<QwenGpuWeightRegion> weight_regions,
    std::shared_ptr<models::qwen::vision::Encoder> vision,
    std::unique_ptr<models::qwen::EmbeddingRowStore> embedding_rows)
    : reader_(std::move(reader)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      weight_regions_(std::move(weight_regions)),
      embedding_rows_(std::move(embedding_rows)),
      vision_(std::move(vision)) {}

QwenGpuModel::~QwenGpuModel() {
  ReleaseWeightRegions(weight_regions_);
}

std::size_t QwenGpuModel::GetResidentBytes() const noexcept {
  std::size_t total = vision_ ? vision_->ResidentBytes() : 0;
  if (embedding_rows_)
    total += embedding_rows_->CachedBytes();
  for (const auto& region : weight_regions_) {
    if (region.size > std::numeric_limits<std::size_t>::max() - total) {
      return std::numeric_limits<std::size_t>::max();
    }
    total += region.size;
  }
  return total;
}

std::shared_ptr<const QwenGpuModel> QwenGpuModel::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::shared_ptr<models::qwen::vision::Encoder> vision) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "GGUF reader must not be null";
    }
    return nullptr;
  }

  if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                            error_msg)) {
    return nullptr;
  }

  auto weights_opt = models::QwenModelWeights::LoadFromGguf(*reader, error_msg);
  if (!weights_opt.has_value()) {
    return nullptr;
  }

  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*reader, error_msg);
  if (!tokenizer)
    return nullptr;

  std::unique_ptr<models::qwen::EmbeddingRowStore> embedding_rows;
  if (const char* path = std::getenv("GUFO_DEV_EMBED_SSD"); path && *path) {
    std::string refusal;
    try {
      embedding_rows = models::qwen::EmbeddingRowStore::Open(
          *reader, path,
          models::qwen::EmbeddingRowStore::CacheBytesFromEnvironment(),
          &refusal);
    } catch (const std::exception& exception) {
      refusal = exception.what();
      embedding_rows.reset();
    }
    if (!embedding_rows)
      std::fprintf(
          stderr, "GUFO_DEV_EMBED_SSD active=0 resident_fallback=1 reason=%s\n",
          refusal.c_str());
  }

  std::vector<SnLoadTensor> selected;
  const char* const sn_flag = std::getenv("GUFO_SN_K4");
  const std::string_view sn_mode = sn_flag ? sn_flag : "0";
  if (sn_mode == "iq3") {
    // IQ3-only next rollout. P1 (=1) remains a model-only comparison mode;
    // Q5/Q6 stay native here. Both executor fences remain in force.
    try {
      for (auto& layer : weights_opt->layers) {
        auto& gate = layer.ffn_gate;
        auto& up = layer.ffn_up;
        if (gate.type != core::GgmlType::kIQ3_S &&
            gate.type != core::GgmlType::kIQ3_XXS)
          continue;
        const auto bb = gate.type == core::GgmlType::kIQ3_S ? 110U : 98U;
        if (gate.type != up.type || gate.num_elements != 17408ULL * 5120 ||
            up.num_elements != gate.num_elements ||
            !gate.FitsAvailableStorage() || !up.FitsAvailableStorage() ||
            static_cast<const std::byte*>(gate.data) +
                    gate.EncodedSizeBytes() !=
                up.data)
          throw std::invalid_argument("invalid native IQ3 gate/up pair");
        // The canonical helper validates physical span and alignment before any
        // allocation. Descriptor dimensions/types/bytes are checked separately.
        const auto views = models::MakeIq3PairViews(gate.data, gate.type, 34816,
                                                    5120, gate.available_bytes);
        for (const auto* ref : {&gate, &up}) {
          const auto tensors = reader->GetTensors();
          const auto it =
              std::find_if(tensors.begin(), tensors.end(),
                           [&](const auto& t) { return t.data == ref->data; });
          if (it == tensors.end() || it->type != ref->type ||
              it->dimensions != std::vector<std::uint64_t>({5120, 17408}) ||
              it->size_bytes != ref->EncodedSizeBytes())
            throw std::invalid_argument("invalid IQ3 source descriptor");
        }
        selected.push_back({&gate, 34816, 5120, bb, views.pair.layout, &up});
      }
    } catch (const std::exception& e) {
      if (error_msg)
        *error_msg = e.what();
      return nullptr;
    }
  } else if (sn_mode == "1") {
    try {
      const auto add = [&](models::QwenTensorRef& ref, core::GgmlType type) {
        if (ref.empty() || ref.type != type)
          return;
        const auto tensors = reader->GetTensors();
        const auto it =
            std::find_if(tensors.begin(), tensors.end(),
                         [&](const auto& t) { return t.data == ref.data; });
        if (it == tensors.end() || it->dimensions.size() != 2 ||
            !ref.FitsAvailableStorage())
          throw std::invalid_argument("invalid Rows32 tensor descriptor");
        const auto rows = it->dimensions[1], columns = it->dimensions[0];
        // Phase1 is measured/approved only for these output matrices.
        if (rows != 5120 || columns != 6144)
          return;
        const auto block_bytes = type == core::GgmlType::kQ5_K ? 176U : 210U;
        if (it->size_bytes != ref.EncodedSizeBytes())
          throw std::invalid_argument(
              "Rows32 descriptor encoded size mismatch");
        selected.push_back({&ref, static_cast<std::size_t>(rows),
                            static_cast<std::size_t>(columns), block_bytes});
      };
      for (auto& layer : weights_opt->layers) {
        add(layer.ssm_out, core::GgmlType::kQ5_K);
        add(layer.attn_output, core::GgmlType::kQ6_K);
      }
    } catch (const std::exception& e) {
      if (error_msg)
        *error_msg = e.what();
      return nullptr;
    }
  }

  std::vector<QwenGpuWeightRegion> weight_regions;
  if (!CreateWeightRegions(*reader, weight_regions, embedding_rows.get(),
                           selected, error_msg)) {
    return nullptr;
  }

  const auto remap = [&](models::QwenTensorRef& tensor) {
    return RemapTensor(tensor, weight_regions);
  };
  bool remapped = (embedding_rows || remap(weights_opt->token_embd)) &&
                  remap(weights_opt->output_norm) && remap(weights_opt->output);
  for (auto& layer : weights_opt->layers) {
    remapped = remapped && remap(layer.attn_norm) && remap(layer.attn_q) &&
               remap(layer.attn_k) && remap(layer.attn_v) &&
               remap(layer.attn_output) && remap(layer.attn_q_norm) &&
               remap(layer.attn_k_norm) && remap(layer.attn_qkv) &&
               remap(layer.attn_gate) && remap(layer.ssm_out) &&
               remap(layer.ssm_conv1d) && remap(layer.ssm_alpha) &&
               remap(layer.ssm_beta) && remap(layer.ssm_a) &&
               remap(layer.ssm_dt) && remap(layer.ssm_norm) &&
               remap(layer.ffn_norm) && remap(layer.ffn_gate) &&
               remap(layer.ffn_up) && remap(layer.ffn_down);
  }
  if (!remapped) {
    ReleaseWeightRegions(weight_regions);
    if (error_msg != nullptr) {
      *error_msg = "A Qwen tensor does not belong to any mapped GGUF shard";
    }
    return nullptr;
  }

  // All regions copied/registered and all references remapped successfully.
  // Only now publish non-native tags. A failed load never exposes tagged
  // native bytes, nor a partially converted model.
  for (const auto& t : selected) {
    if (t.paired_up) {
      // G/U now share the remapped physical pair root; never advance XXS by
      // half its encoded bytes (its three planes span the FULL pair).
      const auto views =
          models::MakeIq3PairViews(t.tensor->data, t.tensor->type, t.rows,
                                   t.columns, t.tensor->available_bytes);
      *t.tensor = views.gate;
      *t.paired_up = views.up;
    } else {
      t.tensor->layout_columns = t.columns;
      t.tensor->layout = models::QwenWeightLayout::kRows32;
    }
  }

  std::shared_ptr<const tokenization::QwenTokenizer> shared_tokenizer(
      std::move(tokenizer));
  auto model = std::make_shared<const QwenGpuModel>(
      std::move(reader), std::move(*weights_opt), std::move(shared_tokenizer),
      std::move(weight_regions), std::move(vision), std::move(embedding_rows));
  if (auto* store = model->GetEmbeddingRows())
    std::fprintf(stderr,
                 "GUFO_DEV_EMBED_SSD active=1 row_bytes=%zu cache_budget=%zu "
                 "accounted_bytes=%zu\n",
                 store->RowBytes(), store->CacheBudgetBytes(),
                 model->GetResidentBytes());
  return model;
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::Create(
    std::shared_ptr<const QwenGpuModel> model, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  if (model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "Qwen GPU model must not be null";
    }
    return nullptr;
  }

  if (const char* why = models::TaggedGpuServeRefusal(model->GetWeights())) {
    if (error_msg)
      *error_msg = why;
    return nullptr;
  }

  const auto& config = model->GetConfig();
  const std::uint32_t model_context =
      config.context_length > 0 ? config.context_length : max_context;
  if (max_context == 0 || max_context > model_context) {
    if (error_msg != nullptr) {
      *error_msg = "Requested GPU context exceeds the model context length";
    }
    return nullptr;
  }

  return std::make_unique<QwenGpuExecutor>(std::move(model), max_context,
                                           policy);
}

std::unique_ptr<QwenGpuExecutor> QwenGpuExecutor::CreateFromGguf(
    std::shared_ptr<const core::GgufReader> reader, std::string* error_msg,
    std::uint32_t max_context, QwenExecutionPolicy policy) {
  auto model = QwenGpuModel::CreateFromGguf(std::move(reader), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), error_msg, max_context, policy);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

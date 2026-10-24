#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/embedding_rows.hpp"

#include <limits>
#include <numeric>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::hip {

EmbeddingRowStage::~EmbeddingRowStage() {
  Free();
}

void EmbeddingRowStage::Free() noexcept {
  if (copied_)
    (void)hipEventSynchronize(copied_);
  if (device_rows_)
    (void)hipFree(device_rows_);
  if (device_ids_)
    (void)hipFree(device_ids_);
  if (host_rows_)
    (void)hipHostFree(host_rows_);
  if (copied_)
    (void)hipEventDestroy(copied_);
  device_rows_ = nullptr;
  device_ids_ = nullptr;
  host_rows_ = nullptr;
  copied_ = nullptr;
  capacity_bytes_ = 0;
  capacity_rows_ = 0;
}

void EmbeddingRowStage::Prepare(models::qwen::EmbeddingRowStore& store,
                                std::span<const std::uint32_t> ids,
                                hipStream_t stream, bool prefill) {
  if (ids.empty() || ids.size() > std::numeric_limits<std::uint32_t>::max() ||
      ids.size() > std::numeric_limits<std::size_t>::max() / store.RowBytes())
    throw std::invalid_argument("invalid embedding staging extent");
  const auto bytes = ids.size() * store.RowBytes();
  if (copied_)
    HIP_CHECK(
        hipEventSynchronize(copied_));  // Host storage may now be overwritten.
  const auto capacity = models::qwen::EmbeddingStageCapacity(
      ids.size(), capacity_rows_, store.RowBytes());
  const auto allocation_bytes = capacity * store.RowBytes();
  if (allocation_bytes > capacity_bytes_ || capacity > capacity_rows_) {
    if (device_rows_)
      HIP_CHECK(hipStreamSynchronize(stream_));  // Old gathers must finish too.
    Free();
    host_ids_.reserve(capacity);
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_rows_),
                            allocation_bytes, hipHostMallocDefault));
    HIP_CHECK(hipMalloc(&device_rows_, allocation_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ids_),
                        capacity * sizeof(std::uint32_t)));
    HIP_CHECK(hipEventCreateWithFlags(&copied_, hipEventDisableTiming));
    capacity_bytes_ = allocation_bytes;
    capacity_rows_ = capacity;
  }
  stream_ = stream;
  host_ids_.resize(ids.size());
  std::iota(host_ids_.begin(), host_ids_.end(), 0U);
  store.Fetch(ids, {host_rows_, bytes}, prefill);
  HIP_CHECK(hipMemcpyAsync(device_rows_, host_rows_, bytes,
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(device_ids_, host_ids_.data(), ids.size_bytes(),
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipEventRecord(copied_, stream));
}

void EmbeddingRowStage::LaunchBatch(models::qwen::EmbeddingRowStore* store,
                                    const models::QwenTensorRef& table,
                                    std::span<const std::uint32_t> ids,
                                    const std::uint32_t* device_ids,
                                    float* output, std::size_t hidden,
                                    hipStream_t stream, bool prefill) {
  if (store) {
    Prepare(*store, ids, stream, prefill);
    LaunchBatchedEmbeddingLookup(device_rows_, table.type, device_ids_, output,
                                 ids.size(), hidden, stream);
  } else {
    LaunchBatchedEmbeddingLookup(table.data, table.type, device_ids, output,
                                 ids.size(), hidden, stream);
  }
}

void EmbeddingRowStage::LaunchSingle(models::qwen::EmbeddingRowStore* store,
                                     const models::QwenTensorRef& table,
                                     std::uint32_t id, float* output,
                                     std::size_t hidden, hipStream_t stream) {
  if (store) {
    Prepare(*store, {&id, 1}, stream, false);
    LaunchEmbeddingLookup(device_rows_, table.type, 0U, output, hidden, stream);
  } else {
    LaunchEmbeddingLookup(table.data, table.type, id, output, hidden, stream);
  }
}

}  // namespace gufo::hip
#endif

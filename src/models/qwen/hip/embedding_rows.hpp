#ifndef GUFO_MODELS_QWEN_HIP_EMBEDDING_ROWS_HPP_
#define GUFO_MODELS_QWEN_HIP_EMBEDDING_ROWS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/models/qwen/embedding_rows.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::hip {

// Executor-private staging. Single-token graphs use a separate instance from
// batches so growing batch storage cannot invalidate a captured address.
class EmbeddingRowStage {
public:
  EmbeddingRowStage() = default;
  ~EmbeddingRowStage();
  EmbeddingRowStage(const EmbeddingRowStage&) = delete;
  EmbeddingRowStage& operator=(const EmbeddingRowStage&) = delete;
  void Prepare(models::qwen::EmbeddingRowStore& store,
               std::span<const std::uint32_t> ids, hipStream_t stream,
               bool prefill);
  const void* Data() const noexcept { return device_rows_; }
  std::size_t Bytes() const noexcept {
    return 2 * capacity_bytes_ +
           (capacity_rows_ + host_ids_.capacity()) * sizeof(std::uint32_t);
  }
  void LaunchBatch(models::qwen::EmbeddingRowStore* store,
                   const models::QwenTensorRef& table,
                   std::span<const std::uint32_t> ids,
                   const std::uint32_t* device_ids, float* output,
                   std::size_t hidden, hipStream_t stream,
                   bool prefill = false);
  void LaunchSingle(models::qwen::EmbeddingRowStore* store,
                    const models::QwenTensorRef& table, std::uint32_t id,
                    float* output, std::size_t hidden, hipStream_t stream);

private:
  void Free() noexcept;
  std::uint8_t* host_rows_{nullptr};
  void* device_rows_{nullptr};
  std::uint32_t* device_ids_{nullptr};
  std::vector<std::uint32_t> host_ids_;
  std::size_t capacity_bytes_{0};
  std::size_t capacity_rows_{0};
  hipEvent_t copied_{nullptr};
  hipStream_t stream_{nullptr};
};

}  // namespace gufo::hip
#endif

#ifndef GUFO_MODELS_QWEN_SN_WEIGHT_PLAN_HPP_
#define GUFO_MODELS_QWEN_SN_WEIGHT_PLAN_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/models/qwen/iq3_layout.hpp"
#include "src/models/qwen/rows32_layout.hpp"

namespace gufo::models {

// Offsets are relative to ONE resident source/destination span, not GGUF file
// offsets. Selection remains the loader's policy, separate from layout.
struct SnWeightSpan {
  std::size_t offset;
  std::size_t rows;
  std::size_t columns;
  std::size_t block_bytes;
  QwenWeightLayout layout = QwenWeightLayout::kRows32;
};

struct SnWeightCopyJob {
  std::size_t source_offset;
  std::size_t source_bytes;
  // Native jobs use the same source/destination offset. Converted jobs use
  // full tensor bases and row range; destination metadata may be far away.
  std::size_t tensor_offset;
  std::size_t rows;
  std::size_t columns;
  std::size_t block_bytes;
  std::size_t row_begin;
  std::size_t row_end;
  QwenWeightLayout layout = QwenWeightLayout::kRows32;
};

inline std::vector<SnWeightCopyJob> PlanSnWeightCopies(
    std::size_t span_bytes, std::span<const SnWeightSpan> selected,
    std::size_t native_chunk_bytes = 16ULL << 20) {
  if (!native_chunk_bytes)
    throw std::invalid_argument("zero native copy chunk");
  std::vector<SnWeightSpan> tensors(selected.begin(), selected.end());
  std::sort(tensors.begin(), tensors.end(),
            [](const auto& a, const auto& b) { return a.offset < b.offset; });
  std::vector<SnWeightCopyJob> jobs;
  std::size_t cursor = 0;
  const auto gaps = [&](std::size_t end) {
    while (cursor < end) {
      const auto bytes = std::min(native_chunk_bytes, end - cursor);
      jobs.push_back(
          {cursor, bytes, 0, 0, 0, 0, 0, 0, QwenWeightLayout::kNative});
      cursor += bytes;
    }
  };
  for (const auto& t : tensors) {
    const bool iq3 = IsIq3Layout(t.layout);
    if ((!iq3 &&
         (t.layout != QwenWeightLayout::kRows32 || !t.rows || t.rows % 32 ||
          (t.block_bytes != 176 && t.block_bytes != 210))) ||
        (iq3 && (!Iq3ShapeValid(t.layout, t.rows, t.columns) ||
                 t.block_bytes != Iq3BlockBytes(t.layout))) ||
        !t.columns || t.columns % 256 ||
        t.rows > std::numeric_limits<std::size_t>::max() / (t.columns / 256))
      throw std::invalid_argument("unsupported phase1 Rows32 tensor");
    const auto blocks = t.rows * (t.columns / 256);
    if (blocks > std::numeric_limits<std::size_t>::max() / t.block_bytes)
      throw std::invalid_argument("Rows32 encoded size overflow");
    const auto bytes = blocks * t.block_bytes;
    if (t.offset < cursor || t.offset > span_bytes ||
        bytes > span_bytes - t.offset)
      throw std::invalid_argument("Rows32 span overlaps or escapes region");
    gaps(t.offset);
    const auto row_bytes = t.columns / 256 * t.block_bytes;
    for (std::size_t row = 0; row < t.rows; row += 32) {
      const auto end = row + std::min<std::size_t>(32, t.rows - row);
      jobs.push_back({t.offset + row * row_bytes, (end - row) * row_bytes,
                      t.offset, t.rows, t.columns, t.block_bytes, row, end,
                      t.layout});
    }
    cursor = t.offset + bytes;
  }
  gaps(span_bytes);
  return jobs;
}

// Coalesce ONLY IQ3 jobs after the canonical 32-row plan has validated every
// span. Each stage still contains whole 8/4-row groups and at most capacity
// bytes; XXS destinations remain three full-pair planes, not local planes.
inline std::vector<SnWeightCopyJob> CoalesceIq3CopyJobs(
    std::span<const SnWeightCopyJob> jobs, std::size_t capacity) {
  std::vector<SnWeightCopyJob> result;
  for (const auto& job : jobs) {
    if (!result.empty() && IsIq3Layout(job.layout)) {
      auto& last = result.back();
      if (last.layout == job.layout &&
          last.tensor_offset == job.tensor_offset && last.rows == job.rows &&
          last.columns == job.columns && last.block_bytes == job.block_bytes &&
          last.row_end == job.row_begin && job.source_bytes <= capacity &&
          last.source_bytes <= capacity - job.source_bytes &&
          last.source_offset + last.source_bytes == job.source_offset) {
        last.row_end = job.row_end;
        last.source_bytes += job.source_bytes;
        continue;
      }
    }
    result.push_back(job);
  }
  return result;
}

struct SnWeightStagedCopy {
  std::size_t stage_offset;
  std::size_t destination_offset;
  std::size_t bytes;
};

// Reuse a bounded pinned buffer. A local tile's metadata is INTENTIONALLY
// staged after its local payload, then relocated to the FULL tensor plane
// by the second DMA descriptor. Never memcpy the entire local tile to its
// native offset: that would put Q6 d inside the payload.
inline std::array<SnWeightStagedCopy, 3> StageSnWeightCopyJob(
    const void* source, void* stage, std::size_t capacity,
    const SnWeightCopyJob& job) {
  if (!source || !stage || job.source_bytes > capacity)
    throw std::invalid_argument("invalid Rows32 staging buffer");
  const auto* s = static_cast<const std::uint8_t*>(source);
  if (!job.block_bytes) {
    std::memcpy(stage, s + job.source_offset, job.source_bytes);
    return {{{0, job.source_offset, job.source_bytes}, {0, 0, 0}}};
  }
  if (IsIq3Layout(job.layout)) {
    ValidateIq3Range(job.layout, job.rows, job.columns, job.row_begin,
                     job.row_end);
    const auto rows = job.row_end - job.row_begin;
    const auto nb = job.columns / 256;
    if (!rows || job.block_bytes != Iq3BlockBytes(job.layout) ||
        job.source_bytes != rows * nb * job.block_bytes ||
        job.tensor_offset >
            std::numeric_limits<std::size_t>::max() -
                Iq3PackedBytes(job.layout, job.rows, job.columns) ||
        job.source_offset !=
            job.tensor_offset + job.row_begin * nb * job.block_bytes)
      throw std::invalid_argument("invalid staged IQ3 job");
    CopyIq3RowRangeFast(s + job.source_offset, stage, job.layout, rows,
                        job.columns, 0, rows);
    if (job.layout == QwenWeightLayout::kIq3SRows8)
      return {{{0, job.source_offset, job.source_bytes}, {}, {}}};
    const auto q = rows * nb * 64;
    const auto a = rows * nb * 32;
    return {
        {{0, job.tensor_offset + job.row_begin * nb * 64, q},
         {q, job.tensor_offset + job.rows * nb * 64 + job.row_begin * nb * 32,
          a},
         {q + a,
          job.tensor_offset + job.rows * nb * 96 + job.row_begin * nb * 2,
          rows * nb * 2}}};
  }
  if (job.layout != QwenWeightLayout::kRows32 ||
      (job.block_bytes != 176 && job.block_bytes != 210) || !job.columns ||
      job.columns % 256 || job.row_begin % 32 || job.row_end <= job.row_begin ||
      job.row_end > job.rows)
    throw std::invalid_argument("invalid staged Rows32 job");
  const auto rows = job.row_end - job.row_begin;
  if (rows % 32 || rows > capacity / job.block_bytes / (job.columns / 256) ||
      job.source_bytes != rows * (job.columns / 256) * job.block_bytes)
    throw std::invalid_argument("staged Rows32 size mismatch");
  InterleaveKQuantRows32(s + job.source_offset, stage, rows, job.columns,
                         job.block_bytes);
  const std::size_t meta = job.block_bytes == 210 ? 2 : 0;
  const auto payload = job.block_bytes - meta;
  const auto blocks = job.columns / 256;
  const auto payload_bytes = rows * blocks * payload;
  return {
      {{0, job.tensor_offset + job.row_begin * blocks * payload, payload_bytes},
       {payload_bytes,
        job.tensor_offset + job.rows * blocks * payload +
            job.row_begin * blocks * meta,
        rows * blocks * meta}}};
}

// Plan validation precedes allocation/copy/publication. Jobs may run in any
// order: native gaps exclude converted bytes; tile writes are disjoint, also
// in the GLOBAL metadata plane. No tensor-sized scratch or second layout.
inline void ExecuteSnWeightCopyJob(const void* source, void* destination,
                                   const SnWeightCopyJob& job) {
  const auto* s = static_cast<const std::uint8_t*>(source);
  auto* d = static_cast<std::uint8_t*>(destination);
  if (!job.block_bytes) {
    std::memcpy(d + job.source_offset, s + job.source_offset, job.source_bytes);
  } else if (IsIq3Layout(job.layout)) {
    CopyIq3RowRangeFast(s + job.tensor_offset, d + job.tensor_offset,
                        job.layout, job.rows, job.columns, job.row_begin,
                        job.row_end);
  } else {
    InterleaveKQuantRows32Range(s + job.tensor_offset, d + job.tensor_offset,
                                job.rows, job.columns, job.block_bytes,
                                job.row_begin, job.row_end);
  }
}

// Inverse of the SAME plan: SN resident region -> native row-major bytes.
// A pair-local plan (offset0) writes the reusable native prefill layer scratch.
inline void ExecuteSnWeightUnpackJob(const void* source, void* destination,
                                     const SnWeightCopyJob& job) {
  const auto* s = static_cast<const std::uint8_t*>(source);
  auto* d = static_cast<std::uint8_t*>(destination);
  if (!job.block_bytes) {
    std::memcpy(d + job.source_offset, s + job.source_offset, job.source_bytes);
  } else if (IsIq3Layout(job.layout)) {
    CopyIq3RowRange(s + job.tensor_offset, d + job.tensor_offset, job.layout,
                    job.rows, job.columns, job.row_begin, job.row_end, true);
  } else {
    for (auto r = job.row_begin; r < job.row_end; ++r)
      for (std::size_t b = 0; b < job.columns / 256; ++b)
        for (std::size_t o = 0; o < job.block_bytes; ++o)
          d[job.tensor_offset +
            (r * (job.columns / 256) + b) * job.block_bytes + o] =
              s[job.tensor_offset + Rows32ByteOffset(job.rows,
                                                     job.columns / 256, r, b, o,
                                                     job.block_bytes)];
  }
}

}  // namespace gufo::models

#endif  // GUFO_MODELS_QWEN_SN_WEIGHT_PLAN_HPP_

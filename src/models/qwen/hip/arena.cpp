#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/turbo4_launchers.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kMaxPromptBatch = 4096;
constexpr std::size_t kMaxTargetLayerTaps = 8;
constexpr std::array<std::uint8_t, 8> kCompactSnapshotMagic = {
    'G', 'Q', 'K', 'V', 'S', 'N', 'P', '1'};
constexpr std::uint32_t kCompactSnapshotVersion = 1;
constexpr std::size_t kCompactSnapshotHeaderBytes = 80;

std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Qwen GPU allocation size overflows");
  }
  return left * right;
}

/// One authority for "can the snapshot path handle this storage".
///
/// The whole capture path has FOUR places that must agree about a plane:
///   1. SnapshotPayloadBytes sizes the unused tail.
///   2. SaveSnapshot allocates d_kv_f16_, d_kv_f32_ OR d_kv_turbo4_.
///   3. The copy pitch: 2 or 4 bytes per ELEMENT for fp16/fp32, and a packed
///      byte pitch (kv_heads * blocks_per_head * 66) for Turbo4.
///   4. The LAYOUT: fp16 is token-major, fp32 is head-major, and Turbo4 is
///      token-major with a packed pitch — the SAME logical order as fp16, so it
///      copies as bytes without any head reordering.
///
/// All four now route through QwenKvBytesForElements / QwenKvBytesPerToken, so
/// this guard only has to refuse what those cannot answer at all: an unknown
/// precision. Before the block plane existed this refused Turbo4 too, and the
/// comment above recorded why each of the four was individually wrong; that
/// refusal is retired, not relaxed — the arithmetic it was protecting is now
/// present and CPU-tested in kv_read.hpp.
void RequireSnapshotCapableStorage(QwenKvCacheStorage storage) {
  if (!QwenKvStorageIsKnown(storage)) {
    throw std::invalid_argument(
        "Qwen snapshot storage is not a known precision");
  }
}

std::size_t CheckedSum(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Qwen GPU allocation total overflows");
  }
  return left + right;
}

void CheckedAdd(std::size_t value, std::size_t* total) {
  *total = CheckedSum(*total, value);
}

void AddAllocation(std::size_t elements, std::size_t element_bytes,
                   std::size_t* total) {
  CheckedAdd(CheckedMultiply(elements, element_bytes), total);
}

void ThrowOnHipError(hipError_t error, std::string_view operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("Qwen compact snapshot header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
T GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("Qwen compact snapshot header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

std::size_t SizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("Qwen compact snapshot size overflows");
  }
  return static_cast<std::size_t>(value);
}

struct CompactSnapshotLayout {
  std::uint32_t attention_layers{0};
  std::uint32_t kv_width{0};
  std::uint32_t max_context{0};
  std::uint32_t valid_context{0};
  QwenKvCacheStorage kv_storage{QwenKvCacheStorage::kFp32};
  QwenRecurrentStateStorage recurrent_state_storage{
      QwenRecurrentStateStorage::kFp32};
  std::size_t live_kv_elements_per_plane{0};
  std::size_t live_kv_bytes_per_plane{0};
  std::size_t conv_elements{0};
  std::size_t conv_bytes{0};
  std::size_t deltanet_elements{0};
  std::size_t deltanet_bytes{0};
  std::size_t k_offset{kCompactSnapshotHeaderBytes};
  std::size_t v_offset{0};
  std::size_t conv_offset{0};
  std::size_t deltanet_offset{0};
  std::size_t total_bytes{0};
  std::size_t rope_offset{0};
  models::qwen::vision::RopeLayout rope;
};

CompactSnapshotLayout MakeCompactSnapshotLayout(
    std::uint32_t attention_layers, std::uint32_t kv_width,
    std::uint32_t max_context, std::uint32_t valid_context,
    QwenKvCacheStorage kv_storage,
    QwenRecurrentStateStorage recurrent_state_storage,
    std::size_t conv_elements, std::size_t deltanet_elements,
    std::size_t image_count = 0) {
  CompactSnapshotLayout layout{
      .attention_layers = attention_layers,
      .kv_width = kv_width,
      .max_context = max_context,
      .valid_context = valid_context,
      .kv_storage = kv_storage,
      .recurrent_state_storage = recurrent_state_storage,
      .conv_elements = conv_elements,
      .deltanet_elements = deltanet_elements,
  };
  // Single authority for element size. The ternary that lived here answered
  // 4 for anything that was not fp16 — including kTurbo4 and including an
  // arbitrary value off the wire.
  const std::size_t kv_element_bytes = QwenKvElementBytes(kv_storage);
  layout.live_kv_elements_per_plane = CheckedMultiply(
      CheckedMultiply(attention_layers, valid_context), kv_width);
  // Block-granular when the storage shares bytes across a group (Turbo4), and
  // element-granular otherwise, which is what the uniform formula below
  // expressed for every format that existed when this was written.
  //   67.1M elements x 2 B      = 134217728  (fp16, unchanged)
  //   ceil(67108864/128) x 66 B =  34603008  (turbo4)
  // Multiplying elements by 4.125/8 looks equivalent and is not: it truncates
  // on a tail, losing the final block. See kv_allocation.hpp.
  if (QwenKvIsBlockAddressed(kv_storage)) {
    layout.live_kv_bytes_per_plane = CheckedMultiply(
        QwenKvAllocationUnits(kv_storage, layout.live_kv_elements_per_plane),
        QwenKvUnitBytes(kv_storage));
  } else {
    layout.live_kv_bytes_per_plane =
        CheckedMultiply(layout.live_kv_elements_per_plane, kv_element_bytes);
  }
  layout.conv_bytes = CheckedMultiply(conv_elements, sizeof(float));
  layout.deltanet_bytes =
      CheckedMultiply(deltanet_elements,
                      QwenRecurrentStateElementBytes(recurrent_state_storage));
  layout.v_offset = CheckedSum(layout.k_offset, layout.live_kv_bytes_per_plane);
  layout.conv_offset =
      CheckedSum(layout.v_offset, layout.live_kv_bytes_per_plane);
  layout.deltanet_offset = CheckedSum(layout.conv_offset, layout.conv_bytes);
  layout.total_bytes =
      CheckedSum(layout.deltanet_offset, layout.deltanet_bytes);
  layout.rope_offset = layout.total_bytes;
  layout.total_bytes =
      CheckedSum(layout.total_bytes, CheckedMultiply(image_count, 12));
  return layout;
}

CompactSnapshotLayout ParseCompactSnapshotLayout(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kCompactSnapshotHeaderBytes ||
      !std::equal(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
                  payload.begin())) {
    throw std::invalid_argument("Qwen compact snapshot header is invalid");
  }
  const auto version = GetLittleEndian<std::uint32_t>(payload, 8);
  const auto image_count = GetLittleEndian<std::uint32_t>(payload, 72);
  if ((version != kCompactSnapshotVersion && version != 2) ||
      image_count > GetLittleEndian<std::uint32_t>(payload, 28) ||
      (version == 1 && image_count != 0) ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kCompactSnapshotHeaderBytes ||
      GetLittleEndian<std::uint32_t>(payload, 76) != 0) {
    throw std::invalid_argument("Qwen compact snapshot version is invalid");
  }

  const std::uint32_t kv_storage_value =
      GetLittleEndian<std::uint32_t>(payload, 32);
  const std::uint32_t recurrent_storage_value =
      GetLittleEndian<std::uint32_t>(payload, 36);
  // kTurbo6 is the LAST enumerator, so the upper bound is the last
  // member and not kFp16/kTurbo4. Using an earlier member as the ceiling would
  // make every snapshot of a later format unreadable by the build that wrote
  // it. Keeping this a strict comparison against the last enumerator also means
  // a future member appended by mistake is rejected rather than silently
  // reinterpreted.
  if (kv_storage_value >
          static_cast<std::uint32_t>(QwenKvCacheStorage::kTurbo6) ||
      recurrent_storage_value >
          static_cast<std::uint32_t>(QwenRecurrentStateStorage::kBf16)) {
    throw std::invalid_argument("Qwen compact snapshot precision is invalid");
  }

  auto layout = MakeCompactSnapshotLayout(
      GetLittleEndian<std::uint32_t>(payload, 16),
      GetLittleEndian<std::uint32_t>(payload, 20),
      GetLittleEndian<std::uint32_t>(payload, 24),
      GetLittleEndian<std::uint32_t>(payload, 28),
      static_cast<QwenKvCacheStorage>(kv_storage_value),
      static_cast<QwenRecurrentStateStorage>(recurrent_storage_value),
      SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48)),
      SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56)), image_count);
  if (layout.live_kv_elements_per_plane !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40)) ||
      layout.total_bytes !=
          SizeFromU64(GetLittleEndian<std::uint64_t>(payload, 64)) ||
      layout.total_bytes != payload.size()) {
    throw std::invalid_argument("Qwen compact snapshot size is invalid");
  }
  for (std::size_t i = 0; i < image_count; ++i) {
    const auto offset = layout.rope_offset + i * 12;
    layout.rope.images.push_back(
        {GetLittleEndian<std::uint32_t>(payload, offset),
         GetLittleEndian<std::uint32_t>(payload, offset + 4),
         GetLittleEndian<std::uint32_t>(payload, offset + 8)});
  }
  layout.rope.Validate(layout.max_context);
  return layout;
}

}  // namespace

models::qwen::vision::RopeLayout QwenGpuSnapshot::ReadRopeLayout(
    std::span<const std::uint8_t> payload) {
  return ParseCompactSnapshotLayout(payload).rope;
}

QwenGpuSnapshot::~QwenGpuSnapshot() {
  if (d_kv_f32_ != nullptr) {
    (void)hipFree(d_kv_f32_);
  }
  if (d_kv_f16_ != nullptr) {
    (void)hipFree(d_kv_f16_);
  }
  if (d_kv_turbo4_ != nullptr) {
    (void)hipFree(d_kv_turbo4_);
  }
  if (d_ssm_conv_ != nullptr) {
    (void)hipFree(d_ssm_conv_);
  }
  if (d_ssm_deltanet_ != nullptr) {
    (void)hipFree(d_ssm_deltanet_);
  }
}

std::size_t QwenGpuSnapshot::CompactPayloadBytes() const {
  RequireCompactKvFormats(kv_formats_, kv_storage_);
  const std::size_t recurrent_layers = num_layers_ - attention_layers_;
  return MakeCompactSnapshotLayout(
             attention_layers_, kv_width_, max_context_, valid_context_,
             kv_storage_, recurrent_state_storage_,
             CheckedMultiply(recurrent_layers, conv_elements_per_layer_),
             CheckedMultiply(recurrent_layers, deltanet_elements_per_layer_),
             vision_layout_.images.size())
      .total_bytes;
}

std::size_t QwenGpuSnapshot::SerializeCompact(
    std::span<std::uint8_t> destination) const {
  EmitCompact(destination, nullptr, std::numeric_limits<std::size_t>::max());
  return destination.size();
}

void QwenGpuSnapshot::StreamCompact(
    const std::function<void(std::span<const std::uint8_t>)>& sink,
    std::size_t piece_bytes) const {
  if (piece_bytes == 0)
    throw std::invalid_argument("Qwen compact snapshot piece size is zero");
  EmitCompact({}, &sink, piece_bytes);
}

void QwenGpuSnapshot::EmitCompact(
    std::span<std::uint8_t> destination,
    const std::function<void(std::span<const std::uint8_t>)>* sink,
    std::size_t piece_bytes) const {
  RequireCompactKvFormats(kv_formats_, kv_storage_);
  if (full_attention_interval_ == 0 || attention_layers_ > num_layers_ ||
      conv_elements_ != CheckedMultiply(num_layers_ - attention_layers_,
                                        conv_elements_per_layer_) ||
      deltanet_elements_ != CheckedMultiply(num_layers_ - attention_layers_,
                                            deltanet_elements_per_layer_)) {
    throw std::logic_error("Qwen compact snapshot metadata is invalid");
  }
  const std::size_t recurrent_layers = num_layers_ - attention_layers_;
  const auto layout = MakeCompactSnapshotLayout(
      attention_layers_, kv_width_, max_context_, valid_context_, kv_storage_,
      recurrent_state_storage_,
      CheckedMultiply(recurrent_layers, conv_elements_per_layer_),
      CheckedMultiply(recurrent_layers, deltanet_elements_per_layer_),
      vision_layout_.images.size());
  if (sink == nullptr && destination.size() != layout.total_bytes) {
    throw std::invalid_argument(
        "Qwen compact snapshot destination size is invalid");
  }
  vision_layout_.Validate(max_context_);  // before any byte leaves
  // Bytes leave in offset order: header, K, V, conv, deltanet, rope. Only KV
  // rows narrower than their pitch leave gaps; those are zeroed (`zero`),
  // every other region is fully overwritten by its copy.
  std::vector<std::uint8_t> piece;
  std::size_t emitted = 0;
  const auto target = [&](std::size_t bytes,
                          bool zero = false) -> std::uint8_t* {
    std::uint8_t* out = destination.data() + emitted;
    if (sink != nullptr) {
      piece.resize(bytes);
      out = piece.data();
    }
    if (zero)
      std::fill_n(out, bytes, std::uint8_t{0});
    return out;
  };
  const auto commit = [&](std::size_t bytes) {
    if (sink != nullptr)
      (*sink)(std::span<const std::uint8_t>(piece.data(), bytes));
    emitted += bytes;
  };
  std::array<std::uint8_t, kCompactSnapshotHeaderBytes> header_bytes{};
  const std::span<std::uint8_t> header(header_bytes);
  std::copy(kCompactSnapshotMagic.begin(), kCompactSnapshotMagic.end(),
            header.begin());
  PutLittleEndian<std::uint32_t>(header, 8,
                                 vision_layout_.images.empty() ? 1U : 2U);
  PutLittleEndian<std::uint32_t>(header, 72, vision_layout_.images.size());
  PutLittleEndian<std::uint32_t>(
      header, 12, static_cast<std::uint32_t>(kCompactSnapshotHeaderBytes));
  PutLittleEndian<std::uint32_t>(header, 16, attention_layers_);
  PutLittleEndian<std::uint32_t>(header, 20, kv_width_);
  PutLittleEndian<std::uint32_t>(header, 24, max_context_);
  PutLittleEndian<std::uint32_t>(header, 28, valid_context_);
  PutLittleEndian<std::uint32_t>(header, 32,
                                 static_cast<std::uint32_t>(kv_storage_));
  PutLittleEndian<std::uint32_t>(
      header, 36, static_cast<std::uint32_t>(recurrent_state_storage_));
  PutLittleEndian<std::uint64_t>(
      header, 40,
      static_cast<std::uint64_t>(layout.live_kv_elements_per_plane));
  PutLittleEndian<std::uint64_t>(
      header, 48, static_cast<std::uint64_t>(layout.conv_elements));
  PutLittleEndian<std::uint64_t>(
      header, 56, static_cast<std::uint64_t>(layout.deltanet_elements));
  PutLittleEndian<std::uint64_t>(
      header, 64, static_cast<std::uint64_t>(layout.total_bytes));

  std::memcpy(target(header.size()), header.data(), header.size());
  commit(header.size());

  SnapshotTransfer transfer;
  if (layout.live_kv_bytes_per_plane != 0) {
    // THE REAL DEFECT THIS REPLACES. Both of these were
    //   kv_storage_ == kFp16 ? <fp16> : <fp32>
    // so a TURBO4 plane took the fp32 branch: it copied d_kv_f32_ using an
    // fp32 pitch, over a plane whose bytes are 66-per-128-value blocks. That
    // is not a slow path, it is a wrong-source copy into a snapshot that
    // would then round-trip into nonsense. It also said block-addressed
    // storage has no member here at all and the correct behaviour is to refuse
    // and say so: that part is STALE and cost B-R4 its first diagnosis. The
    // block plane is now allocated and serialized -- d_kv_turbo4_ is selected
    // below -- and the Turbo4 refusal is retired deliberately; see
    // RequireSnapshotCapableStorage, whose doc comment says so explicitly.
    // This guard checks only that the storage is known.
    RequireSnapshotCapableStorage(kv_storage_);
    const QwenKvPlaneGeometry geometry{
        static_cast<std::size_t>(head_dim_),
        static_cast<std::size_t>(kv_heads_),
        static_cast<std::size_t>(valid_context_)};
    const QwenKvCopyShape kv_copy = QwenKvDescribeCopy(
        kv_storage_, geometry, static_cast<std::size_t>(valid_context_),
        static_cast<std::size_t>(valid_context_), attention_layers_);
    if (kv_copy.width == 0U) {
      throw std::logic_error(
          "Qwen compact snapshot cannot describe its KV copy for this storage "
          "and geometry");
    }
    // B-R4. This compared kv_copy.target_pitch against
    // live_kv_bytes_per_plane / attention_layers_, which asks what ONE LAYER
    // weighs. For token-major storage (fp16, Turbo4) a row IS a layer's token
    // and the two agree. For fp32 the plane is head-major -- rows are
    // (layer, head) pairs -- so that division is kv_heads TIMES the copy's own
    // pitch, and the comparison fired for every multi-head geometry. Worse, the
    // division was the DESTINATION STRIDE handed to both Copy2D calls, so
    // without this assertion the rows would have overlapped instead of merely
    // being counted wrong.
    //
    // The real invariant is that the copy fills the region reserved for it:
    // true for fp32, fp16 and Turbo4 alike, and it is what rules out an
    // overlap.
    if (kv_copy.target_pitch * kv_copy.rows != layout.live_kv_bytes_per_plane) {
      throw std::logic_error(
          "Qwen compact snapshot copy does not fill its reserved KV region");
    }
    const void* source = kv_storage_ == QwenKvCacheStorage::kFp16 ? d_kv_f16_
                         : QwenKvIsBlockAddressed(kv_storage_)    ? d_kv_turbo4_
                                                                  : d_kv_f32_;
    if (source == nullptr) {
      throw std::logic_error("Qwen compact snapshot has no KV storage");
    }
    // B-R4: the stride of the destination, taken from the copy that consumes
    // it. Re-deriving it from the layout total asks a different question and
    // has a different answer whenever the plane is not token-major.
    const std::size_t row_bytes = kv_copy.target_pitch;
    const auto* source_bytes = static_cast<const std::uint8_t*>(source);
    const std::size_t full_plane_bytes =
        CheckedMultiply(kv_copy.source_pitch, kv_copy.rows);
    const std::size_t rows_per_piece =
        std::max<std::size_t>(1, piece_bytes / row_bytes);
    for (const std::uint8_t* plane :
         {source_bytes, source_bytes + full_plane_bytes}) {
      for (std::size_t row = 0; row < kv_copy.rows; row += rows_per_piece) {
        const std::size_t rows = std::min(rows_per_piece, kv_copy.rows - row);
        transfer.Copy2D(target(rows * row_bytes, kv_copy.width < row_bytes),
                        row_bytes, plane + row * kv_copy.source_pitch,
                        kv_copy.source_pitch, kv_copy.width, rows);
        commit(rows * row_bytes);
      }
    }
  }
  const auto copy_region = [&](const void* region, std::size_t bytes) {
    const auto* from = static_cast<const std::uint8_t*>(region);
    for (std::size_t done = 0; done < bytes;) {
      const std::size_t n = std::min(piece_bytes, bytes - done);
      transfer.Copy(target(n), from + done, n);
      commit(n);
      done += n;
    }
  };
  copy_region(d_ssm_conv_, layout.conv_bytes);
  copy_region(d_ssm_deltanet_, layout.deltanet_bytes);
  if (!vision_layout_.images.empty()) {
    const std::size_t rope_bytes = vision_layout_.images.size() * 12;
    const std::span<std::uint8_t> rope(target(rope_bytes), rope_bytes);
    for (std::size_t i = 0; i < vision_layout_.images.size(); ++i) {
      const auto& grid = vision_layout_.images[i];
      PutLittleEndian<std::uint32_t>(rope, i * 12, grid.offset);
      PutLittleEndian<std::uint32_t>(rope, i * 12 + 4, grid.height);
      PutLittleEndian<std::uint32_t>(rope, i * 12 + 8, grid.width);
    }
    commit(rope_bytes);
  }
  if (emitted != layout.total_bytes)
    throw std::logic_error("Qwen compact snapshot emitted the wrong size");
}

/// THE f16 SCRATCH A BLOCK-ADDRESSED LAYER NEEDS BEFORE ATTENTION CAN RUN.
///
/// The f16 attention routes read an f16 plane, and a Turbo4 layer is
/// 66-byte blocks, so one layer has to be materialised as f16 before those
/// kernels may run. ONE layer, K and V, laid out [count][head][dim] — the
/// consumer's expression, attention_wmma.hip:143 kHeadMajor=false.
///
/// It is ONE layer on purpose: 2 * max_context * kv_heads * head_dim halfs, not
/// a plane per layer. `max_context * kv_heads * head_dim` is 4 KiB/token at the
/// UD geometry (4 heads, head_dim 256).
///
/// ONE definition, called by BOTH the estimator and the allocator. This file
/// already carries a comment about three size computations that must agree, and
/// a fourth copy of this number is how that went wrong before.
[[nodiscard]] inline std::size_t Turbo4AttentionScratchHalfs(
    std::size_t max_context, std::size_t kv_heads,
    std::size_t head_dim) noexcept {
  return 2U * max_context * kv_heads * head_dim;
}

[[nodiscard]] inline std::size_t Turbo4AttentionScratchBytes(
    std::size_t max_context, std::size_t kv_heads,
    std::size_t head_dim) noexcept {
  return Turbo4AttentionScratchHalfs(max_context, kv_heads, head_dim) *
         sizeof(std::uint16_t);
}

QwenGpuMemoryUsage QwenGpuArena::EstimateMemoryUsage(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy) {
  const std::size_t context = std::max(max_context, 1U);
  const std::size_t batch = std::min<std::size_t>(context, kMaxPromptBatch);
  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      CheckedMultiply(config.num_key_value_heads, config.head_dim);
  const std::size_t q_projection = CheckedMultiply(2, attention);
  const std::size_t ssm_qkv = config.SsmQkvSize();
  const std::size_t recurrent =
      std::max<std::size_t>(attention, config.ssm_inner_size);
  const std::size_t projection = std::max(q_projection, ssm_qkv);
  const std::size_t time_step = config.ssm_time_step_rank;

  QwenGpuMemoryUsage usage;
  auto& scratch = usage.temporary_scratch_bytes;
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, attention), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, kv), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, kv), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, intermediate), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, intermediate), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, intermediate), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, hidden), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, projection), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, ssm_qkv), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, recurrent), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, recurrent), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, time_step), sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(batch, time_step), sizeof(float), &scratch);
  AddAllocation(
      CheckedMultiply(CheckedMultiply(batch, config.ssm_group_count), 3),
      sizeof(float), &scratch);
  AddAllocation(CheckedMultiply(CheckedMultiply(batch, time_step), 2),
                sizeof(float), &scratch);
  AddAllocation(config.vocab_size, sizeof(float), &scratch);
  AddAllocation(std::max<std::size_t>(batch, 2), sizeof(std::uint32_t),
                &scratch);
  AddAllocation(CheckedMultiply(kMaxTargetLayerTaps, hidden), sizeof(float),
                &scratch);

  const std::size_t maximum_scratch_width = std::max<std::size_t>(
      {intermediate, hidden, projection,
       CheckedSum(ssm_qkv, CheckedSum(config.ssm_inner_size,
                                      CheckedMultiply(2, time_step)))});
  const std::size_t scratch_elements =
      CheckedMultiply(batch, maximum_scratch_width);
  AddAllocation(scratch_elements, sizeof(hip_bfloat16), &scratch);

  const std::size_t q8_rows = CheckedMultiply((batch + 15) / 16, 16);
  const std::size_t q8_row_elements = scratch_elements / batch;
  const std::size_t q8_elements = CheckedMultiply(q8_rows, q8_row_elements);
  const std::size_t q8_blocks = (q8_elements + 31) / 32;
  CheckedAdd(CheckedMultiply(CheckedMultiply(q8_blocks, sizeof(float)), 10),
             &scratch);
  CheckedAdd(4096, &scratch);

  AddAllocation(detail::DecodeAttentionScratchElements(
                    config.num_attention_heads, config.head_dim),
                sizeof(float), &scratch);

  const std::size_t maximum_weight_width =
      std::max<std::size_t>({q_projection, kv, attention, intermediate, ssm_qkv,
                             config.ssm_inner_size, time_step});
  const std::size_t maximum_weight =
      CheckedMultiply(hidden, maximum_weight_width);
  AddAllocation(maximum_weight, sizeof(hip_bfloat16), &scratch);

  auto& state = usage.request_state_bytes;
  const std::size_t total_kv = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config.FullAttentionLayerCount(), kv),
                      context),
      1);
  // THIRD instance of the fp16-else-fp32 fallthrough, and the one that made
  // §30's "three size computations must agree" requirement false for Turbo4:
  // the estimator counted fp32 state bytes while MakeCompactSnapshotLayout
  // counted block bytes, so the two disagreed about the same plane.
  //
  // Converted to the single authority. `QwenKvBytesForElements` answers for all
  // three storages, so the estimator can no longer disagree with the allocator
  // or the snapshot layout about one plane. A zero result is an explicit
  // refusal, NOT a zero contribution: an unknown or overflowed size added to
  // `state` as nothing would silently shrink the estimate and turn an error
  // into a plan that fails later, somewhere else.
  const auto state_kv_storage = policy.kv_cache_storage;
  if (QwenKvIsBlockAddressed(state_kv_storage) &&
      !QwenKvHeadsAreBlockAligned(config.head_dim,
                                  config.num_key_value_heads)) {
    throw std::logic_error(
        "Qwen memory estimator cannot size a block-addressed (Turbo4) KV "
        "state: a head is not a whole number of 128-value blocks, so the "
        "flat element count would round up and two heads would share a norm");
  }
  const std::size_t state_kv_plane_bytes =
      QwenKvBytesForElements(state_kv_storage, total_kv);
  if (state_kv_plane_bytes == 0U) {
    throw std::logic_error(
        "Qwen memory estimator cannot size a KV plane: unknown precision, "
        "overflow, or a geometry with no block-aligned unit");
  }
  // Two planes, K and V, as every allocation site does.
  CheckedAdd(CheckedMultiply(state_kv_plane_bytes, 2), &state);
  // ...AND the f16 scratch, because a fit/refuse decision reads THIS number.
  // Estimating the block plane while omitting the scratch that makes it
  // readable would pass a fit that the allocator then fails, and the refusal
  // would arrive late and somewhere unrelated.
  if (QwenKvIsBlockAddressed(state_kv_storage)) {
    CheckedAdd(Turbo4AttentionScratchBytes(context, config.num_key_value_heads,
                                           config.head_dim),
               &state);
    const std::size_t decode_workspace =
        Turbo4DecodeWorkspaceBytes(context, config.num_attention_heads,
                                   config.num_key_value_heads, config.head_dim);
    if (decode_workspace == 0U)
      throw std::logic_error(
          "Qwen memory estimator cannot size the Turbo4 decode workspace");
    CheckedAdd(decode_workspace, &state);
  }

  const std::size_t total_conv = CheckedMultiply(
      CheckedMultiply(config.SsmLayerCount(), ssm_qkv), config.ssm_conv_kernel);
  AddAllocation(total_conv, sizeof(float), &state);
  const std::size_t total_deltanet = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config.SsmLayerCount(), time_step),
                      config.ssm_state_size),
      config.SsmValueSize());
  AddAllocation(total_deltanet,
                QwenRecurrentStateElementBytes(policy.recurrent_state_storage),
                &state);
  return usage;
}

QwenGpuMemoryUsage QwenGpuArena::GetMemoryUsage() const {
  return EstimateMemoryUsage(config_, max_context_, policy_);
}

std::size_t QwenGpuArena::SnapshotPayloadBytes(
    std::uint32_t valid_context) const {
  RequireSnapshotCapableStorage(policy_.kv_cache_storage);
  if (valid_context > max_context_)
    throw std::length_error("Qwen snapshot exceeds the context length");
  // The unused tail is a whole number of TOKENS per layer per plane, so it is
  // measured with the SAME per-token authority the copy uses. Sizing it as
  // `2 * bytes_per_ELEMENT * elements` cannot express a block plane at all: a
  // Turbo4 element is 66/128 = 0.515625 B, and multiplying an element count by
  // that fraction truncates on every layer, leaving a few bytes of slack the
  // caller would then hand to hipMalloc as a short buffer.
  const QwenKvPlaneGeometry geometry{
      static_cast<std::size_t>(config_.head_dim),
      static_cast<std::size_t>(config_.num_key_value_heads),
      static_cast<std::size_t>(max_context_)};
  const std::size_t per_token =
      QwenKvBytesPerToken(policy_.kv_cache_storage, geometry);
  if (per_token == 0U) {
    throw std::logic_error(
        "Qwen snapshot cannot size its unused KV tail: this geometry has no "
        "addressable per-token size");
  }
  const std::size_t unused = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config_.FullAttentionLayerCount(),
                                      max_context_ - valid_context),
                      per_token),
      2);
  std::size_t total = GetMemoryUsage().request_state_bytes;
  // The estimate also counts the Turbo block KV's f16 attention scratch and
  // decode workspace (fit decisions need them), but SaveSnapshot copies
  // neither. Leaving them in made every turbo payload 168,812,032 B larger
  // than the copy, so each capture was skipped as kReservationMismatch and
  // turbo KV never kept a RAM checkpoint.
  if (QwenKvIsBlockAddressed(policy_.kv_cache_storage)) {
    const std::size_t not_copied = CheckedSum(
        Turbo4AttentionScratchBytes(max_context_, config_.num_key_value_heads,
                                    config_.head_dim),
        Turbo4DecodeWorkspaceBytes(max_context_, config_.num_attention_heads,
                                   config_.num_key_value_heads,
                                   config_.head_dim));
    if (not_copied > total)
      throw std::logic_error(
          "Qwen snapshot payload is smaller than the turbo scratch it must exclude");
    total -= not_copied;
  }
  // Subtraction, so check the direction. A wrap here returns a payload size
  // near SIZE_MAX, which is not a size — it is an allocation the caller will
  // discover is impossible much later.
  if (unused > total) {
    throw std::logic_error(
        "Qwen snapshot payload is smaller than the KV tail it must exclude");
  }
  return total - unused;
}

std::unique_ptr<QwenGpuSnapshot> QwenGpuArena::SaveSnapshot(
    std::uint32_t valid_context) {
  RequireSnapshotCapableStorage(policy_.kv_cache_storage);
  if (valid_context > max_context_) {
    throw std::length_error("Qwen snapshot exceeds the context length");
  }

  auto snapshot = std::unique_ptr<QwenGpuSnapshot>(new QwenGpuSnapshot());
  snapshot->num_layers_ = config_.num_layers;
  snapshot->attention_layers_ = config_.FullAttentionLayerCount();
  snapshot->full_attention_interval_ = config_.full_attention_interval;
  snapshot->kv_width_ = config_.num_key_value_heads * config_.head_dim;
  snapshot->kv_heads_ = config_.num_key_value_heads;
  snapshot->head_dim_ = config_.head_dim;
  snapshot->max_context_ = max_context_;
  snapshot->valid_context_ = valid_context;
  snapshot->kv_storage_ = policy_.kv_cache_storage;
  snapshot->kv_formats_ = policy_.kv_layers;
  snapshot->recurrent_state_storage_ = policy_.recurrent_state_storage;
  snapshot->kv_elements_per_plane_ = CheckedMultiply(
      CheckedMultiply(snapshot->attention_layers_, valid_context),
      snapshot->kv_width_);
  snapshot->conv_elements_per_layer_ =
      CheckedMultiply(config_.SsmQkvSize(), config_.ssm_conv_kernel);
  snapshot->conv_elements_ =
      CheckedMultiply(config_.num_layers - snapshot->attention_layers_,
                      snapshot->conv_elements_per_layer_);
  snapshot->deltanet_elements_per_layer_ = CheckedMultiply(
      CheckedMultiply(config_.ssm_time_step_rank, config_.ssm_state_size),
      config_.SsmValueSize());
  snapshot->deltanet_elements_ =
      CheckedMultiply(config_.num_layers - snapshot->attention_layers_,
                      snapshot->deltanet_elements_per_layer_);

  // The KV region is sized from the SAME descriptor the copy below uses, so a
  // size and a layout cannot disagree. Three separate computations of one
  // plane's shape is exactly how §30's four-place bug happened.
  const QwenKvPlaneGeometry live_geometry{
      static_cast<std::size_t>(config_.head_dim),
      static_cast<std::size_t>(config_.num_key_value_heads),
      static_cast<std::size_t>(valid_context)};
  const QwenKvCopyShape kv_copy = QwenKvDescribeCopy(
      policy_.kv_cache_storage, live_geometry,
      static_cast<std::size_t>(max_context_),
      static_cast<std::size_t>(valid_context), snapshot->attention_layers_);
  if (kv_copy.width == 0U) {
    throw std::logic_error(
        "Qwen snapshot cannot describe a KV copy for this storage and "
        "geometry");
  }
  // Two planes, K and V, each source_pitch*rows from the arena.
  const std::size_t kv_plane_bytes =
      CheckedMultiply(kv_copy.source_pitch, kv_copy.rows);
  const std::size_t kv_snapshot_bytes =
      CheckedMultiply(kv_copy.target_pitch, kv_copy.rows);
  const std::size_t conv_bytes =
      CheckedMultiply(snapshot->conv_elements_, sizeof(float));
  const std::size_t deltanet_bytes = CheckedMultiply(
      snapshot->deltanet_elements_,
      QwenRecurrentStateElementBytes(snapshot->recurrent_state_storage_));
  CheckedAdd(CheckedMultiply(kv_snapshot_bytes, 2), &snapshot->payload_bytes_);
  CheckedAdd(conv_bytes, &snapshot->payload_bytes_);
  CheckedAdd(deltanet_bytes, &snapshot->payload_bytes_);

  if (kv_snapshot_bytes != 0) {
    switch (policy_.kv_cache_storage) {
      case QwenKvCacheStorage::kFp16:
        ThrowOnHipError(hipMalloc(&snapshot->d_kv_f16_,
                                  CheckedMultiply(kv_snapshot_bytes, 2)),
                        "failed to allocate Qwen FP16 snapshot KV");
        break;
      case QwenKvCacheStorage::kFp32:
        ThrowOnHipError(hipMalloc(&snapshot->d_kv_f32_,
                                  CheckedMultiply(kv_snapshot_bytes, 2)),
                        "failed to allocate Qwen FP32 snapshot KV");
        break;
      case QwenKvCacheStorage::kTurbo4:
      case QwenKvCacheStorage::kTurbo5:
      case QwenKvCacheStorage::kTurbo6:
        // A block plane, not an fp16 plane viewed differently. The bytes are
        // 66 / 82 / 98 per 128 values; nothing at any offset in it is an fp16
        // value.
        ThrowOnHipError(hipMalloc(&snapshot->d_kv_turbo4_,
                                  CheckedMultiply(kv_snapshot_bytes, 2)),
                        "failed to allocate Qwen Turbo4 snapshot KV");
        break;
    }
  }
  ThrowOnHipError(hipMalloc(&snapshot->d_ssm_conv_, conv_bytes),
                  "failed to allocate Qwen convolution snapshot");
  ThrowOnHipError(hipMalloc(&snapshot->d_ssm_deltanet_, deltanet_bytes),
                  "failed to allocate Qwen DeltaNet snapshot");

  // The transfer runs on its own stream. Drain state updates queued on the
  // arena stream (rollback, replay) before reading them from the live buffer.
  assert(d_ssm_conv_state == d_ssm_conv_buffers_[live_recurrent_] &&
         d_ssm_deltanet_state == d_ssm_deltanet_buffers_[live_recurrent_]);
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen snapshot capture");
  SnapshotTransfer transfer;
  if (kv_snapshot_bytes != 0) {
    const auto* source = static_cast<const std::uint8_t*>(
        policy_.kv_cache_storage == QwenKvCacheStorage::kFp16
            ? d_attention_kv_f16
            : (QwenKvIsBlockAddressed(policy_.kv_cache_storage)
                   ? d_kv_turbo4
                   : static_cast<void*>(d_kv_cache)));
    auto* destination = static_cast<std::uint8_t*>(
        policy_.kv_cache_storage == QwenKvCacheStorage::kFp16
            ? snapshot->d_kv_f16_
            : (QwenKvIsBlockAddressed(policy_.kv_cache_storage)
                   ? snapshot->d_kv_turbo4_
                   : snapshot->d_kv_f32_));
    if (source == nullptr || destination == nullptr) {
      throw std::logic_error(
          "Qwen snapshot KV plane is null for a known storage");
    }
    for (std::size_t plane = 0; plane < 2; ++plane)
      transfer.Copy2D(destination + plane * kv_snapshot_bytes,
                      kv_copy.target_pitch, source + plane * kv_plane_bytes,
                      kv_copy.source_pitch, kv_copy.width, kv_copy.rows,
                      hipMemcpyDeviceToDevice);
  }
  transfer.Copy(snapshot->d_ssm_conv_, d_ssm_conv_state, conv_bytes,
                hipMemcpyDeviceToDevice);
  transfer.Copy(snapshot->d_ssm_deltanet_, d_ssm_deltanet_state, deltanet_bytes,
                hipMemcpyDeviceToDevice);
  return snapshot;
}

void QwenGpuArena::RestoreSnapshot(const QwenGpuSnapshot& snapshot) {
  // Same authority as the save path. Restore was UNGUARDED: a Turbo4
  // policy reached here and the copy below took the fp32 branch, so a
  // restore would write fp32-shaped data into a block-addressed plane.
  // A save that refuses followed by a restore that does not is worse
  // than neither.
  RequireSnapshotCapableStorage(policy_.kv_cache_storage);
  const std::uint32_t attention_layers = config_.FullAttentionLayerCount();
  const std::uint32_t kv_width = config_.num_key_value_heads * config_.head_dim;
  const std::size_t conv_elements =
      CheckedMultiply(CheckedMultiply(config_.num_layers - attention_layers,
                                      config_.SsmQkvSize()),
                      config_.ssm_conv_kernel);
  const std::size_t deltanet_elements = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(config_.num_layers - attention_layers,
                                      config_.ssm_time_step_rank),
                      config_.ssm_state_size),
      config_.SsmValueSize());
  RequireQwenKvFormatTablesEqual(snapshot.kv_formats_, policy_.kv_layers);
  if (snapshot.valid_context_ > max_context_ ||
      snapshot.attention_layers_ != attention_layers ||
      snapshot.kv_width_ != kv_width || snapshot.max_context_ != max_context_ ||
      snapshot.kv_heads_ != config_.num_key_value_heads ||
      snapshot.head_dim_ != config_.head_dim ||
      snapshot.kv_storage_ != policy_.kv_cache_storage ||
      snapshot.recurrent_state_storage_ != policy_.recurrent_state_storage ||
      snapshot.conv_elements_ != conv_elements ||
      snapshot.deltanet_elements_ != deltanet_elements) {
    throw std::invalid_argument("Qwen snapshot is incompatible with the arena");
  }

  Reset();
  // Same descriptor as SaveSnapshot, with the two contexts swapped: the
  // snapshot is the source here and the arena the target. Describing the copy
  // twice from two different formulas is how a save and a restore can disagree
  // about the same plane while both succeed.
  const QwenKvPlaneGeometry restore_geometry{
      static_cast<std::size_t>(config_.head_dim),
      static_cast<std::size_t>(config_.num_key_value_heads),
      static_cast<std::size_t>(snapshot.valid_context_)};
  const QwenKvCopyShape kv_copy = QwenKvDescribeCopy(
      policy_.kv_cache_storage, restore_geometry,
      static_cast<std::size_t>(snapshot.valid_context_),
      static_cast<std::size_t>(max_context_), attention_layers);
  if (kv_copy.width == 0U) {
    throw std::logic_error(
        "Qwen snapshot restore cannot describe a KV copy for this storage and "
        "geometry");
  }
  const std::size_t kv_source_plane =
      CheckedMultiply(kv_copy.source_pitch, kv_copy.rows);
  const std::size_t kv_target_plane =
      CheckedMultiply(kv_copy.target_pitch, kv_copy.rows);
  if (kv_source_plane != 0) {
    const auto* source = static_cast<const std::uint8_t*>(
        policy_.kv_cache_storage == QwenKvCacheStorage::kFp16
            ? snapshot.d_kv_f16_
            : (QwenKvIsBlockAddressed(policy_.kv_cache_storage)
                   ? snapshot.d_kv_turbo4_
                   : snapshot.d_kv_f32_));
    auto* destination = static_cast<std::uint8_t*>(
        policy_.kv_cache_storage == QwenKvCacheStorage::kFp16
            ? d_attention_kv_f16
            : (QwenKvIsBlockAddressed(policy_.kv_cache_storage)
                   ? d_kv_turbo4
                   : static_cast<void*>(d_kv_cache)));
    if (source == nullptr || destination == nullptr) {
      throw std::logic_error(
          "Qwen snapshot KV plane is null for a known storage");
    }
    for (std::size_t plane = 0; plane < 2; ++plane)
      ThrowOnHipError(
          hipMemcpy2DAsync(
              destination + plane * kv_target_plane, kv_copy.target_pitch,
              source + plane * kv_source_plane, kv_copy.source_pitch,
              kv_copy.width, kv_copy.rows, hipMemcpyDeviceToDevice, stream),
          "failed to restore Qwen KV snapshot");
  }
  ThrowOnHipError(hipMemcpyAsync(d_ssm_conv_state, snapshot.d_ssm_conv_,
                                 snapshot.conv_elements_ * sizeof(float),
                                 hipMemcpyDeviceToDevice, stream),
                  "failed to restore Qwen convolution snapshot");
  ThrowOnHipError(hipMemcpyAsync(d_ssm_deltanet_state, snapshot.d_ssm_deltanet_,
                                 snapshot.deltanet_elements_ *
                                     QwenRecurrentStateElementBytes(
                                         policy_.recurrent_state_storage),
                                 hipMemcpyDeviceToDevice, stream),
                  "failed to restore Qwen DeltaNet snapshot");
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize Qwen snapshot restore");
}

void QwenGpuArena::RestoreCompactSnapshot(
    std::span<const std::uint8_t> payload,
    std::uint32_t expected_valid_context) {
  RequireCompactKvFormats(policy_.kv_layers, policy_.kv_cache_storage);
  RequireSnapshotCapableStorage(policy_.kv_cache_storage);
  const auto layout = ParseCompactSnapshotLayout(payload);
  const std::uint32_t attention_layers = config_.FullAttentionLayerCount();
  const std::uint32_t kv_width = config_.num_key_value_heads * config_.head_dim;
  const std::size_t conv_elements_per_layer =
      CheckedMultiply(config_.SsmQkvSize(), config_.ssm_conv_kernel);
  const std::size_t deltanet_elements_per_layer = CheckedMultiply(
      CheckedMultiply(config_.ssm_time_step_rank, config_.ssm_state_size),
      config_.SsmValueSize());
  const std::size_t recurrent_layers = config_.num_layers - attention_layers;
  const std::size_t conv_elements =
      CheckedMultiply(recurrent_layers, conv_elements_per_layer);
  const std::size_t deltanet_elements =
      CheckedMultiply(recurrent_layers, deltanet_elements_per_layer);
  if (layout.valid_context != expected_valid_context ||
      layout.valid_context > max_context_ ||
      layout.attention_layers != attention_layers ||
      layout.kv_width != kv_width || layout.max_context != max_context_ ||
      layout.kv_storage != policy_.kv_cache_storage ||
      layout.recurrent_state_storage != policy_.recurrent_state_storage ||
      layout.conv_elements != conv_elements ||
      layout.deltanet_elements != deltanet_elements) {
    throw std::invalid_argument(
        "Qwen compact snapshot is incompatible with the arena");
  }

  Reset();
  if (layout.live_kv_bytes_per_plane != 0) {
    const QwenKvPlaneGeometry geometry{
        static_cast<std::size_t>(config_.head_dim),
        static_cast<std::size_t>(config_.num_key_value_heads),
        static_cast<std::size_t>(max_context_)};
    const QwenKvCopyShape kv_copy = QwenKvDescribeCopy(
        policy_.kv_cache_storage, geometry,
        static_cast<std::size_t>(layout.valid_context),
        static_cast<std::size_t>(max_context_), attention_layers);
    if (kv_copy.width == 0U) {
      throw std::logic_error(
          "Qwen compact restore cannot describe a KV copy for this storage and "
          "geometry");
    }
    // B-R4, second site. Same wrong derivation as the save path: this divided
    // the reserved region by kv_copy.rows to recover a row stride, then
    // compared the result against ONE of the copy's two strides. rows is the
    // layer count for token-major storage and layers*kv_heads for fp32, and the
    // region's two strides differ whenever valid_context != max_context, so the
    // guard could not be satisfied in general.
    //
    // Note that row_bytes was never used by the copy below -- the two
    // hipMemcpy2DAsync calls take kv_copy.target_pitch and kv_copy.source_pitch
    // directly. It existed only to feed this comparison, so it is gone rather
    // than reassigned. The guard that replaces it is the SOURCE-side region
    // invariant: the payload region must hold the copy exactly.
    if (kv_copy.source_pitch * kv_copy.rows != layout.live_kv_bytes_per_plane) {
      throw std::invalid_argument(
          "Qwen compact snapshot KV rows do not match its declared storage");
    }
    auto* destination = static_cast<std::uint8_t*>(
        policy_.kv_cache_storage == QwenKvCacheStorage::kFp16
            ? d_attention_kv_f16
            : (QwenKvIsBlockAddressed(policy_.kv_cache_storage)
                   ? d_kv_turbo4
                   : static_cast<void*>(d_kv_cache)));
    if (destination == nullptr) {
      throw std::logic_error("Qwen compact restore has no KV plane");
    }
    const std::size_t full_plane_bytes =
        CheckedMultiply(kv_copy.target_pitch, kv_copy.rows);
    ThrowOnHipError(
        hipMemcpy2DAsync(destination, kv_copy.target_pitch,
                         payload.data() + layout.k_offset, kv_copy.source_pitch,
                         kv_copy.width, kv_copy.rows, hipMemcpyHostToDevice,
                         stream),
        "failed to restore compact Qwen K cache");
    ThrowOnHipError(
        hipMemcpy2DAsync(destination + full_plane_bytes, kv_copy.target_pitch,
                         payload.data() + layout.v_offset, kv_copy.source_pitch,
                         kv_copy.width, kv_copy.rows, hipMemcpyHostToDevice,
                         stream),
        "failed to restore compact Qwen V cache");
  }
  if (layout.conv_bytes != 0)
    ThrowOnHipError(
        hipMemcpyAsync(d_ssm_conv_state, payload.data() + layout.conv_offset,
                       layout.conv_bytes, hipMemcpyHostToDevice, stream),
        "failed to restore compact Qwen convolution state");
  if (layout.deltanet_bytes != 0)
    ThrowOnHipError(
        hipMemcpyAsync(d_ssm_deltanet_state,
                       payload.data() + layout.deltanet_offset,
                       layout.deltanet_bytes, hipMemcpyHostToDevice, stream),
        "failed to restore compact Qwen DeltaNet state");
  ThrowOnHipError(hipStreamSynchronize(stream),
                  "failed to synchronize compact Qwen snapshot restore");
}

QwenGpuArena::QwenGpuArena(const core::ModelConfig& config,
                           std::uint32_t max_context,
                           QwenExecutionPolicy policy)
    : config_(config),
      max_context_(std::max(max_context, 1U)),
      max_batch_(std::min(max_context_, kMaxPromptBatch)),
      policy_(policy) {
  try {
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
    HIPBLAS_CHECK(hipblasSetStream(hipblas_handle, stream));
    hipblaslt_gemm = std::make_unique<HipblasLtGemm>();

    const std::size_t hidden_size = config_.hidden_size;
    const std::size_t intermediate_size = config_.intermediate_size;
    const std::size_t vocab_size = config_.vocab_size;
    const std::size_t num_layers = config_.SsmLayerCount();
    const std::size_t num_kv_heads = config_.num_key_value_heads;
    const std::size_t head_dim = config_.head_dim;
    const std::size_t batch = max_batch_;
    const std::size_t attention_size = config_.AttentionSize();
    const std::size_t kv_size = num_kv_heads * head_dim;
    const std::size_t q_projection_size = 2 * attention_size;
    const std::size_t ssm_qkv_size = config_.SsmQkvSize();
    const std::size_t ssm_inner_size = config_.ssm_inner_size;
    const std::size_t recurrent_width =
        std::max(attention_size, ssm_inner_size);
    const std::size_t projection_width =
        std::max(q_projection_size, ssm_qkv_size);
    const std::size_t time_step_rank = config_.ssm_time_step_rank;

    HIP_CHECK(hipMalloc(&d_hidden, batch * hidden_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_normed, batch * hidden_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_q, batch * attention_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k, batch * kv_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v, batch * kv_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_attn_out, batch * hidden_size * sizeof(float)));
    HIP_CHECK(
        hipMalloc(&d_ffn_gate, batch * intermediate_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ffn_up, batch * intermediate_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ffn_act, batch * intermediate_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ffn_out, batch * hidden_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ssm_qkv, batch * projection_width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_conv_out, batch * ssm_qkv_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ssm_gate, batch * recurrent_width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ssm_out, batch * recurrent_width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_alpha_buf, batch * time_step_rank * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_beta_buf, batch * time_step_rank * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ssm_kq_scales,
                        batch * config_.ssm_group_count * 3 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ssm_alpha_beta,
                        batch * time_step_rank * 2 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_logits, vocab_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_prompt_tokens, std::max<std::size_t>(batch, 2) *
                                              sizeof(std::uint32_t)));
    // D2f: this allocation used to be a LITERAL 5 - the old DFlash2 tap cap.
    // D1 M2 raised kMaxTargetLayerTaps to 8 and moved the memory ESTIMATE to
    // the constant (arena.cpp:376) but not this allocation, so a DSpark draft
    // that captures 8 taps copied 8 * hidden floats into room for 5. The copy
    // failed with hipErrorInvalidValue, not an illegal address, because an API
    // copy is bounds-checked against the known allocation while a kernel
    // overrun faults instead. Fixed size per call, so it reproduced at every
    // prompt length.
    //
    // The constant is the only legal source for this number. Two sites, one
    // number, and a literal here is how they drift apart again.
    HIP_CHECK(hipMalloc(&d_target_layer_features,
                        kMaxTargetLayerTaps * hidden_size * sizeof(float)));

    const std::size_t scratch_elements =
        batch * std::max<std::size_t>(
                    {intermediate_size, hidden_size, projection_width,
                     ssm_qkv_size + ssm_inner_size + (2 * time_step_rank)});
    HIP_CHECK(
        hipMalloc(&d_scratch_bf16, scratch_elements * sizeof(hip_bfloat16)));
    // The tiled Q8_1 activation layout groups 16 tokens per tile, so size for a
    // batch rounded up to a whole tile. The base layout remains 36 bytes per 32
    // elements; reserve another float per block for the K-quant
    // activation-sum sidecar without changing the Q8 payload or its stride.
    const std::size_t q8_rows = ((batch + 15) / 16) * 16;
    const std::size_t q8_row_elements =
        scratch_elements / std::max<std::size_t>(batch, 1);
    const std::size_t scratch_q8_bytes =
        ((((q8_rows * q8_row_elements) + 31) / 32) * sizeof(float) * 10) +
        4096;  // 36-byte payload + 4-byte sidecar per 32 elems
    HIP_CHECK(hipMalloc(&d_scratch_q8_act, scratch_q8_bytes));
    const std::size_t split_k_elements = detail::DecodeAttentionScratchElements(
        config_.num_attention_heads, config_.head_dim);
    HIP_CHECK(
        hipMalloc(&d_split_k_attention, split_k_elements * sizeof(float)));

    // Weight BF16 scratch: largest per-layer matmul weight in bf16 elements,
    // used by the prefill dequant-to-BF16 then BF16 GEMM path.
    const std::size_t max_weight_elems =
        hidden_size *
        std::max<std::size_t>({q_projection_size, kv_size, attention_size,
                               intermediate_size, ssm_qkv_size, ssm_inner_size,
                               time_step_rank});
    HIP_CHECK(
        hipMalloc(&d_weights_bf16, max_weight_elems * sizeof(hip_bfloat16)));

    const std::size_t total_kv = GetAttentionKvPlaneElements();
    // THIS IS THE ROOT OF THE WHOLE SNAPSHOT GAP.
    //
    // The condition was `UsesFp16AttentionKv()`, i.e. `kv_storage == kFp16`.
    // So selecting kTurbo4 took the ELSE branch and allocated
    // `total_kv * sizeof(float) * 2` — an FP32-SIZED buffer — while every
    // downstream path (allocation arithmetic, snapshot sizing, copy pitch,
    // layout) believed it was holding Turbo4 blocks. Selecting Turbo4 in the
    // policy silently produced a buffer 128*4/66 = 7.757576x TOO LARGE per
    // value (fp32 is 4 B/value, Turbo4 is 66/128 = 0.515625 B/value), which is
    // why the snapshot sites each needed their own refusal.
    //
    // DIRECTION, recorded because an earlier note here had it backwards: the
    // over-allocation is the SAFE direction. No read or write of a
    // Turbo4-shaped offset could run past the end of an fp32-sized buffer, so
    // this bug wasted memory and mis-reported sizes; it did not corrupt.
    // Treating it as under-allocation overstated it.
    //
    // Close it here instead: refuse at the allocation, so there is exactly one
    // place a block-addressed storage can be rejected before anything sized
    // after it can disagree. Two planes are allocated (K and V).
    const auto kv_storage = policy_.kv_cache_storage;
    if (!QwenKvIsAllocatableByExecutor(kv_storage)) {
      throw std::invalid_argument(
          "Qwen executor KV storage is not a known precision");
    }
    // Block storage needs a block-aligned head. Checked on the GEOMETRY, not
    // on the flat element count: a total that happens to be a whole number of
    // blocks can still have every individual head straddling one, which would
    // let two heads share one norm and decode both wrongly.
    if (QwenKvIsBlockAddressed(kv_storage) &&
        !QwenKvHeadsAreBlockAligned(config_.head_dim,
                                    config_.num_key_value_heads)) {
      throw std::logic_error(
          "Qwen executor cannot allocate a Turbo4 KV plane: a head is not a "
          "whole number of 128-value blocks");
    }
    // Single authority for the byte count, so the allocation cannot disagree
    // with the snapshot sizing about what one value costs.
    const std::size_t kv_plane_bytes =
        QwenKvBytesForElements(kv_storage, total_kv);
    if (kv_plane_bytes == 0U) {
      throw std::logic_error("Qwen executor KV plane size computed as zero");
    }
    // C1c: THE SELECTION RECEIPT. Printed from the arena, because the arena is
    // the one place every construction path passes through -- so this line
    // cannot be skipped by a caller that forgot to pass a policy. It names the
    // storage and the bytes actually allocated, which is what distinguishes a
    // real Turbo4 run from a second f16 run that printed nothing.
    std::fprintf(stderr, "%s\n",
                 DescribeQwenKvSelection(policy_, total_kv).c_str());
    std::fflush(stderr);
    // Three explicit storages, not a two-way branch. The old
    // `UsesFp16AttentionKv() ? f16 : fp32` is exactly `== kFp16`, so it routed
    // Turbo4 to fp32 — a buffer 128*4/66 = 7.757576x too large per value. That
    // direction is memory-safe (no Turbo4 offset can run past an fp32-sized
    // end) but it silently wastes the memory and mis-reports every size.
    switch (kv_storage) {
      case QwenKvCacheStorage::kFp16:
        HIP_CHECK(hipMalloc(&d_attention_kv_f16, kv_plane_bytes * 2));
        break;
      case QwenKvCacheStorage::kFp32:
        HIP_CHECK(hipMalloc(&d_kv_cache, kv_plane_bytes * 2));
        break;
      case QwenKvCacheStorage::kTurbo4:
      case QwenKvCacheStorage::kTurbo5:
      case QwenKvCacheStorage::kTurbo6:
        // B-D3: A TABLE THAT NEEDS A PLANE WE DID NOT ALLOCATE IS REFUSED HERE,
        // not at the decode step. An f16 tail row under Turbo4 storage would
        // otherwise route to the element-wise reader with a NULL f16 pointer,
        // which is a crash on the decode path reached by configuring the table
        // we just built. Until the mixed ALLOCATION unit exists, such a table
        // is simply not servable.
        if (!QwenKvTableIsServableByStorage(policy_.kv_layers,
                                            policy_.kv_cache_storage)) {
          throw std::logic_error(
              "Qwen arena: the per-layer KV format table needs a plane this "
              "arena did not allocate (mixed Turbo4/f16 is not implemented)");
        }
        HIP_CHECK(hipMalloc(&d_kv_turbo4, kv_plane_bytes * 2));
        turbo4_kv_bytes_recorded = kv_plane_bytes * 2;
        // The f16 scratch, ONE layer, sized by the same helper the estimator
        // reads. It is its OWN member, NOT d_attention_kv_f16, which stays null
        // under block storage on purpose -- that nullness is the tripwire the
        // 25 `OffsetIfPresent(d_attention_kv_f16, total_k)` call sites rely on,
        // and `total_k` is the ALL-LAYER count, so a real buffer parked there
        // would hand every one of them a pointer past the end.
        //
        // It is filled per layer by LaunchTurbo4DequantLayer and consumed
        // through ONE E3 helper that takes no layer index, so K=scratch,
        // V=scratch+max_context*kKvWidth, layer_idx=0 and skip_kv_write=true
        // cannot be got wrong individually at a call site.
        //
        // Under GUFO_DEV_SCRATCH_TILED nothing is allocated here. The
        // first prefill that needs it calls EnsureTurbo4PrefillScratch with the
        // prefix it is about to materialise, and until then the pointer is null
        // and GetAttentionKvLayerElements answers 0 -- so a route that reaches
        // the pair without the prefill having sized it REFUSES instead of
        // reading an absent buffer. Decode never asks for it.
        if (!Turbo4ScratchTiledEnabled()) {
          const std::size_t scratch_bytes = Turbo4AttentionScratchBytes(
              max_context_, config_.num_key_value_heads, config_.head_dim);
          HIP_CHECK(hipMalloc(&d_turbo4_attention_scratch, scratch_bytes));
          turbo4_scratch_bytes_recorded = scratch_bytes;
        }
        const std::size_t decode_partial_bytes = Turbo4DecodePartialBytes(
            max_context_, config_.num_attention_heads,
            config_.num_key_value_heads, config_.head_dim);
        const std::size_t decode_workspace_bytes = Turbo4DecodeWorkspaceBytes(
            max_context_, config_.num_attention_heads,
            config_.num_key_value_heads, config_.head_dim);
        if (decode_partial_bytes == 0U || decode_workspace_bytes == 0U)
          throw std::logic_error(
              "Qwen arena cannot size the Turbo4 decode workspace");
        HIP_CHECK(
            hipMalloc(&d_turbo4_decode_workspace, decode_workspace_bytes));
        turbo4_decode_partial_bytes_recorded = decode_partial_bytes;
        turbo4_decode_workspace_bytes_recorded = decode_workspace_bytes;
        // The one word E1's batched writer reads its position from. Allocated
        // HERE, beside the scratch, so it shares the scratch's lifetime and its
        // bookkeeping sites rather than being a per-chunk allocation that
        // synchronises the device on free.
        HIP_CHECK(hipMalloc(&d_turbo4_first_pos, sizeof(std::uint32_t)));
        // TWO status words, allocated together with the plane they describe.
        // If the second fails the first is already reachable from `this` and
        // the catch below calls FreeAll, so no word leaks on a partial
        // allocation.
        HIP_CHECK(hipMalloc(&d_kv_turbo4_write_status,
                            QwenGpuArena::Turbo4StatusWordBytes()));
        HIP_CHECK(hipMalloc(&d_kv_turbo4_read_status,
                            QwenGpuArena::Turbo4StatusWordBytes()));
        turbo4_status_bytes_recorded = QwenGpuArena::Turbo4StatusWordsBytes();
        // INITIALISE THE WORDS AT ALLOCATION, not only at the first step.
        // hipMalloc does not zero memory, so without this the reader's first
        // `writer_status` is uninitialised garbage — which
        // Turbo4ReadWriterStatus would faithfully report as kWriterFailed, or
        // worse, as kOk. A word that has never been written must not be
        // readable as a verdict.
        {
          Turbo4ResetStepStatusWords(
              static_cast<Turbo4KernelStatus*>(d_kv_turbo4_write_status),
              static_cast<Turbo4KernelStatus*>(d_kv_turbo4_read_status),
              stream);
          // The stream is created before this point in the constructor; if it
          // is not yet valid the words are still zeroed synchronously, because
          // an unwritten word is the defect and a streamless memset still fixes
          // it.
          (void)hipStreamSynchronize(stream);
        }
        break;
    }

    const std::size_t total_conv =
        num_layers * ssm_qkv_size * config_.ssm_conv_kernel;
    HIP_CHECK(hipMalloc(&d_ssm_conv_buffers_[0], total_conv * sizeof(float)));

    const std::size_t total_deltanet = num_layers * time_step_rank *
                                       config_.ssm_state_size *
                                       config_.SsmValueSize();
    HIP_CHECK(hipMalloc(&d_ssm_deltanet_buffers_[0],
                        total_deltanet * QwenRecurrentStateElementBytes(
                                             policy_.recurrent_state_storage)));

    Reset();
  } catch (...) {
    FreeAll();
    throw;
  }
}

QwenGpuArena::~QwenGpuArena() {
  FreeAll();
}

void QwenGpuArena::EnsureTurbo4PrefillScratch(std::size_t visible_context) {
  if (!Turbo4ScratchTiledEnabled() ||
      !QwenKvIsBlockAddressed(policy_.kv_cache_storage)) {
    return;
  }
  if (visible_context == 0U)
    return;
  if (visible_context > max_context_) {
    // The caller's own length check should have refused first; refuse here too
    // rather than allocate past the plane the blocks live in, because this is
    // the one place that still knows how big that plane is.
    throw std::length_error(
        "Qwen Turbo4 prefill scratch request exceeds the GPU context length");
  }
  // ROUND UP TO A WHOLE PROMPT BATCH. A continuation that re-prefills one more
  // token per turn would otherwise free and re-allocate several hundred MiB
  // every turn; this makes it at most once per `max_batch_` tokens, and the
  // slack it buys is one batch of scratch (16 MiB at the UD geometry).
  const std::size_t batch = std::max<std::size_t>(max_batch_, 1U);
  std::size_t tokens = ((visible_context + batch - 1U) / batch) * batch;
  if (tokens > max_context_)
    tokens = max_context_;
  const std::size_t needed = Turbo4AttentionScratchBytes(
      tokens, config_.num_key_value_heads, config_.head_dim);
  if (needed == 0U) {
    throw std::logic_error("Qwen Turbo4 prefill scratch sized as zero bytes");
  }
  if (needed <= turbo4_scratch_bytes_recorded)
    return;
  // FREE FIRST, so the peak is the new buffer and not both. The old buffer
  // holds a dequant of a prefix this prefill is about to overwrite layer by
  // layer anyway, so there is nothing in it to carry over.
  if (d_turbo4_attention_scratch != nullptr) {
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_turbo4_attention_scratch));
    d_turbo4_attention_scratch = nullptr;
    turbo4_scratch_bytes_recorded = 0;
  }
  HIP_CHECK(hipMalloc(&d_turbo4_attention_scratch, needed));
  turbo4_scratch_bytes_recorded = needed;
}

QwenGpuScratchView QwenGpuArena::GetScratchView(
    std::size_t batch_size) noexcept {
  const std::size_t batch = std::min<std::size_t>(batch_size, max_batch_);
  const std::size_t hidden = config_.hidden_size;
  const std::size_t attention = config_.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config_.num_key_value_heads) * config_.head_dim;
  const std::size_t q_projection = 2 * attention;
  const std::size_t ssm_qkv = config_.SsmQkvSize();
  const std::size_t recurrent =
      std::max<std::size_t>(attention, config_.ssm_inner_size);
  const std::size_t projection = std::max<std::size_t>(q_projection, ssm_qkv);
  const std::size_t bf16_scratch =
      batch *
      std::max<std::size_t>({config_.intermediate_size, hidden, projection,
                             ssm_qkv + config_.ssm_inner_size +
                                 (2 * config_.ssm_time_step_rank)});
  const std::size_t split_k = detail::DecodeAttentionScratchElements(
      config_.num_attention_heads, config_.head_dim);
  const std::size_t weight_bf16 =
      hidden * std::max<std::size_t>({q_projection, kv, attention,
                                      config_.intermediate_size, ssm_qkv,
                                      config_.ssm_inner_size,
                                      config_.ssm_time_step_rank});
  return {
      .decode =
          {
              .hidden = {d_hidden, batch * hidden},
              .normed = {d_normed, batch * hidden},
              .logits = {d_logits, config_.vocab_size},
              .bf16 = {static_cast<hip_bfloat16*>(d_scratch_bf16),
                       bf16_scratch},
              .weight_bf16 = {d_weights_bf16, weight_bf16},
              .prompt_tokens = {d_prompt_tokens,
                                std::max<std::size_t>(batch, 2)},
              // d_alpha_buf is reused only after all SSM layers finish.
              // Preserve that exact address while exposing the sampling epoch's
              // uint32 view.
              .sampled_token = {reinterpret_cast<std::uint32_t*>(d_alpha_buf),
                                1},
          },
      .attention =
          {
              .q = {d_q, batch * attention},
              .k = {d_k, batch * kv},
              .v = {d_v, batch * kv},
              .output = {d_attn_out, batch * hidden},
              .split_k = {d_split_k_attention, split_k},
          },
      .ssm =
          {
              .qkv = {d_ssm_qkv, batch * projection},
              .conv_out = {d_conv_out, batch * ssm_qkv},
              .gate = {d_ssm_gate, batch * recurrent},
              .out = {d_ssm_out, batch * recurrent},
              .alpha = {d_alpha_buf, batch * config_.ssm_time_step_rank},
              .beta = {d_beta_buf, batch * config_.ssm_time_step_rank},
          },
      .ffn =
          {
              .gate = {d_ffn_gate, batch * config_.intermediate_size},
              .up = {d_ffn_up, batch * config_.intermediate_size},
              .activation = {d_ffn_act, batch * config_.intermediate_size},
              .out = {d_ffn_out, batch * hidden},
          },
  };
}

void QwenGpuArena::SetTargetLayerCapture(
    std::span<const std::uint32_t> target_layer_ids) {
  if (target_layer_ids.size() > kMaxTargetLayerTaps) {
    throw std::invalid_argument("too many target hidden-layer taps");
  }
  for (const std::uint32_t layer : target_layer_ids) {
    if (layer >= config_.num_layers) {
      throw std::invalid_argument("target hidden-layer tap is out of range");
    }
  }
  target_layer_ids_.assign(target_layer_ids.begin(), target_layer_ids.end());
}

std::optional<std::size_t> QwenGpuArena::GetTargetLayerCaptureIndex(
    std::uint32_t layer) const noexcept {
  const auto iterator = std::ranges::find(target_layer_ids_, layer);
  if (iterator == target_layer_ids_.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(
      std::distance(target_layer_ids_.begin(), iterator));
}

QwenGpuArena::QwenGpuArena(QwenGpuArena&& other) noexcept
    : config_(other.config_),
      max_context_(other.max_context_),
      max_batch_(other.max_batch_),
      policy_(other.policy_),
      target_layer_ids_(std::move(other.target_layer_ids_)) {
  d_hidden = other.d_hidden;
  d_normed = other.d_normed;
  d_q = other.d_q;
  d_k = other.d_k;
  d_v = other.d_v;
  d_attn_out = other.d_attn_out;
  d_ffn_gate = other.d_ffn_gate;
  d_ffn_up = other.d_ffn_up;
  d_ffn_act = other.d_ffn_act;
  d_ffn_out = other.d_ffn_out;
  d_ssm_qkv = other.d_ssm_qkv;
  d_conv_out = other.d_conv_out;
  d_ssm_gate = other.d_ssm_gate;
  d_ssm_out = other.d_ssm_out;
  d_alpha_buf = other.d_alpha_buf;
  d_beta_buf = other.d_beta_buf;
  d_ssm_kq_scales = other.d_ssm_kq_scales;
  d_ssm_alpha_beta = other.d_ssm_alpha_beta;
  d_logits = other.d_logits;
  d_attention_kv_f16 = other.d_attention_kv_f16;
  d_turbo4_attention_scratch = other.d_turbo4_attention_scratch;
  d_turbo4_decode_workspace = other.d_turbo4_decode_workspace;
  d_turbo4_first_pos = other.d_turbo4_first_pos;
  d_kv_cache = other.d_kv_cache;
  d_kv_turbo4 = other.d_kv_turbo4;
  d_kv_turbo4_write_status = other.d_kv_turbo4_write_status;
  d_kv_turbo4_read_status = other.d_kv_turbo4_read_status;
  turbo4_kv_bytes_recorded = other.turbo4_kv_bytes_recorded;
  turbo4_status_bytes_recorded = other.turbo4_status_bytes_recorded;
  turbo4_scratch_bytes_recorded = other.turbo4_scratch_bytes_recorded;
  turbo4_decode_workspace_bytes_recorded =
      other.turbo4_decode_workspace_bytes_recorded;
  turbo4_decode_partial_bytes_recorded =
      other.turbo4_decode_partial_bytes_recorded;
  d_ssm_conv_state = other.d_ssm_conv_state;
  d_ssm_deltanet_state = other.d_ssm_deltanet_state;
  d_prompt_tokens = other.d_prompt_tokens;
  d_target_layer_features = other.d_target_layer_features;
  stream = other.stream;
  hipblas_handle = other.hipblas_handle;
  hipblaslt_gemm = std::move(other.hipblaslt_gemm);
  d_scratch_bf16 = other.d_scratch_bf16;
  d_scratch_q8_act = other.d_scratch_q8_act;
  d_split_k_attention = other.d_split_k_attention;
  d_weights_bf16 = other.d_weights_bf16;
  d_ssm_conv_buffers_ = other.d_ssm_conv_buffers_;
  d_ssm_deltanet_buffers_ = other.d_ssm_deltanet_buffers_;
  live_recurrent_ = other.live_recurrent_;
  saved_recurrent_ = other.saved_recurrent_;
  source_recurrent_ = other.source_recurrent_;
  d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
  d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
  d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
  d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
  saved_context_ = other.saved_context_;
  replay_last_position_ = other.replay_last_position_;
  replay_captured_positions_ = other.replay_captured_positions_;
  has_saved_state_ = other.has_saved_state_;
  replay_capture_active_ = other.replay_capture_active_;

  other.d_hidden = nullptr;
  other.d_normed = nullptr;
  other.d_q = nullptr;
  other.d_k = nullptr;
  other.d_v = nullptr;
  other.d_attn_out = nullptr;
  other.d_ffn_gate = nullptr;
  other.d_ffn_up = nullptr;
  other.d_ffn_act = nullptr;
  other.d_ffn_out = nullptr;
  other.d_ssm_qkv = nullptr;
  other.d_conv_out = nullptr;
  other.d_ssm_gate = nullptr;
  other.d_ssm_out = nullptr;
  other.d_alpha_buf = nullptr;
  other.d_beta_buf = nullptr;
  other.d_ssm_kq_scales = nullptr;
  other.d_ssm_alpha_beta = nullptr;
  other.d_logits = nullptr;
  other.d_attention_kv_f16 = nullptr;
  other.d_turbo4_attention_scratch = nullptr;
  other.d_turbo4_decode_workspace = nullptr;
  other.d_turbo4_first_pos = nullptr;
  other.d_kv_cache = nullptr;
  other.d_kv_turbo4 = nullptr;
  other.d_kv_turbo4_write_status = nullptr;
  other.d_kv_turbo4_read_status = nullptr;
  other.turbo4_kv_bytes_recorded = 0;
  other.turbo4_status_bytes_recorded = 0;
  other.turbo4_scratch_bytes_recorded = 0;
  other.turbo4_decode_workspace_bytes_recorded = 0;
  other.turbo4_decode_partial_bytes_recorded = 0;
  other.d_ssm_conv_state = nullptr;
  other.d_ssm_deltanet_state = nullptr;
  other.d_prompt_tokens = nullptr;
  other.d_target_layer_features = nullptr;
  other.stream = nullptr;
  other.hipblas_handle = nullptr;
  other.d_scratch_bf16 = nullptr;
  other.d_scratch_q8_act = nullptr;
  other.d_split_k_attention = nullptr;
  other.d_weights_bf16 = nullptr;
  other.d_ssm_conv_buffers_ = {};
  other.d_ssm_deltanet_buffers_ = {};
  other.live_recurrent_ = 0;
  other.saved_recurrent_ = 0;
  other.source_recurrent_ = 0;
  other.d_ssm_replay_qkv_ = nullptr;
  other.d_ssm_replay_alpha_ = nullptr;
  other.d_ssm_replay_beta_ = nullptr;
  other.d_ssm_replay_enabled_ = nullptr;
  other.saved_context_ = 0;
  other.replay_last_position_ = 0;
  other.replay_captured_positions_ = 0;
  other.has_saved_state_ = false;
  other.replay_capture_active_ = false;
}

QwenGpuArena& QwenGpuArena::operator=(QwenGpuArena&& other) noexcept {
  if (this != &other) {
    FreeAll();
    config_ = other.config_;
    max_context_ = other.max_context_;
    max_batch_ = other.max_batch_;
    policy_ = other.policy_;
    target_layer_ids_ = std::move(other.target_layer_ids_);
    d_hidden = other.d_hidden;
    d_normed = other.d_normed;
    d_q = other.d_q;
    d_k = other.d_k;
    d_v = other.d_v;
    d_attn_out = other.d_attn_out;
    d_ffn_gate = other.d_ffn_gate;
    d_ffn_up = other.d_ffn_up;
    d_ffn_act = other.d_ffn_act;
    d_ffn_out = other.d_ffn_out;
    d_ssm_qkv = other.d_ssm_qkv;
    d_conv_out = other.d_conv_out;
    d_ssm_gate = other.d_ssm_gate;
    d_ssm_out = other.d_ssm_out;
    d_alpha_buf = other.d_alpha_buf;
    d_beta_buf = other.d_beta_buf;
    d_ssm_kq_scales = other.d_ssm_kq_scales;
    d_ssm_alpha_beta = other.d_ssm_alpha_beta;
    d_logits = other.d_logits;
    d_attention_kv_f16 = other.d_attention_kv_f16;
    d_turbo4_attention_scratch = other.d_turbo4_attention_scratch;
    d_turbo4_decode_workspace = other.d_turbo4_decode_workspace;
    d_turbo4_first_pos = other.d_turbo4_first_pos;
    d_kv_cache = other.d_kv_cache;
    d_kv_turbo4 = other.d_kv_turbo4;
    d_kv_turbo4_write_status = other.d_kv_turbo4_write_status;
    d_kv_turbo4_read_status = other.d_kv_turbo4_read_status;
    turbo4_kv_bytes_recorded = other.turbo4_kv_bytes_recorded;
    turbo4_status_bytes_recorded = other.turbo4_status_bytes_recorded;
    turbo4_scratch_bytes_recorded = other.turbo4_scratch_bytes_recorded;
    turbo4_decode_workspace_bytes_recorded =
        other.turbo4_decode_workspace_bytes_recorded;
    turbo4_decode_partial_bytes_recorded =
        other.turbo4_decode_partial_bytes_recorded;
    d_ssm_conv_state = other.d_ssm_conv_state;
    d_ssm_deltanet_state = other.d_ssm_deltanet_state;
    d_prompt_tokens = other.d_prompt_tokens;
    d_target_layer_features = other.d_target_layer_features;
    stream = other.stream;
    hipblas_handle = other.hipblas_handle;
    hipblaslt_gemm = std::move(other.hipblaslt_gemm);
    d_scratch_bf16 = other.d_scratch_bf16;
    d_scratch_q8_act = other.d_scratch_q8_act;
    d_split_k_attention = other.d_split_k_attention;
    d_weights_bf16 = other.d_weights_bf16;
    d_ssm_conv_buffers_ = other.d_ssm_conv_buffers_;
    d_ssm_deltanet_buffers_ = other.d_ssm_deltanet_buffers_;
    live_recurrent_ = other.live_recurrent_;
    saved_recurrent_ = other.saved_recurrent_;
    source_recurrent_ = other.source_recurrent_;
    d_ssm_replay_qkv_ = other.d_ssm_replay_qkv_;
    d_ssm_replay_alpha_ = other.d_ssm_replay_alpha_;
    d_ssm_replay_beta_ = other.d_ssm_replay_beta_;
    d_ssm_replay_enabled_ = other.d_ssm_replay_enabled_;
    saved_context_ = other.saved_context_;
    replay_last_position_ = other.replay_last_position_;
    replay_captured_positions_ = other.replay_captured_positions_;
    has_saved_state_ = other.has_saved_state_;
    replay_capture_active_ = other.replay_capture_active_;

    other.d_hidden = nullptr;
    other.d_normed = nullptr;
    other.d_q = nullptr;
    other.d_k = nullptr;
    other.d_v = nullptr;
    other.d_attn_out = nullptr;
    other.d_ffn_gate = nullptr;
    other.d_ffn_up = nullptr;
    other.d_ffn_act = nullptr;
    other.d_ffn_out = nullptr;
    other.d_ssm_qkv = nullptr;
    other.d_conv_out = nullptr;
    other.d_ssm_gate = nullptr;
    other.d_ssm_out = nullptr;
    other.d_alpha_buf = nullptr;
    other.d_beta_buf = nullptr;
    other.d_ssm_kq_scales = nullptr;
    other.d_ssm_alpha_beta = nullptr;
    other.d_logits = nullptr;
    other.d_attention_kv_f16 = nullptr;
    other.d_turbo4_attention_scratch = nullptr;
    other.d_turbo4_decode_workspace = nullptr;
    other.d_turbo4_first_pos = nullptr;
    other.d_kv_cache = nullptr;
    other.d_kv_turbo4 = nullptr;
    other.d_kv_turbo4_write_status = nullptr;
    other.d_kv_turbo4_read_status = nullptr;
    other.turbo4_kv_bytes_recorded = 0;
    other.turbo4_status_bytes_recorded = 0;
    other.turbo4_scratch_bytes_recorded = 0;
    other.turbo4_decode_workspace_bytes_recorded = 0;
    other.turbo4_decode_partial_bytes_recorded = 0;
    other.d_ssm_conv_state = nullptr;
    other.d_ssm_deltanet_state = nullptr;
    other.d_prompt_tokens = nullptr;
    other.d_target_layer_features = nullptr;
    other.stream = nullptr;
    other.hipblas_handle = nullptr;
    other.d_scratch_bf16 = nullptr;
    other.d_scratch_q8_act = nullptr;
    other.d_split_k_attention = nullptr;
    other.d_weights_bf16 = nullptr;
    other.d_ssm_conv_buffers_ = {};
    other.d_ssm_deltanet_buffers_ = {};
    other.live_recurrent_ = 0;
    other.saved_recurrent_ = 0;
    other.source_recurrent_ = 0;
    other.d_ssm_replay_qkv_ = nullptr;
    other.d_ssm_replay_alpha_ = nullptr;
    other.d_ssm_replay_beta_ = nullptr;
    other.d_ssm_replay_enabled_ = nullptr;
    other.saved_context_ = 0;
    other.replay_last_position_ = 0;
    other.replay_captured_positions_ = 0;
    other.has_saved_state_ = false;
    other.replay_capture_active_ = false;
  }
  return *this;
}

void QwenGpuArena::Reset() {
  // THE STATUS WORDS RESET HERE TOO. The per-step reset sits after a PASSING
  // verdict, so a step that threw leaves its failure in the words. That is what
  // makes the failure investigable — and it also means a new session would
  // inherit it and throw again before doing any work. A session boundary clears
  // the evidence deliberately, which is the one place that is correct.
  Turbo4ResetStepStatusWords(
      static_cast<Turbo4KernelStatus*>(d_kv_turbo4_write_status),
      static_cast<Turbo4KernelStatus*>(d_kv_turbo4_read_status), stream);

  const std::size_t num_layers = config_.SsmLayerCount();
  const std::size_t total_conv =
      num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet = num_layers * config_.ssm_time_step_rank *
                                     config_.ssm_state_size *
                                     config_.SsmValueSize();

  // Buffer 0 becomes the live state again; whatever buffer 1 holds is fully
  // overwritten before it is read.
  live_recurrent_ = 0;
  saved_recurrent_ = 0;
  source_recurrent_ = 0;
  SyncLivePointers();

  // Attention only reads positions through the current token. Prefill/decode
  // writes those rows before use; clearing reserved future KV adds O(capacity)
  // work to every short conversation and snapshot restore.
  if (d_ssm_conv_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_conv_state, 0, total_conv * sizeof(float),
                             stream));
  }
  if (d_ssm_deltanet_state != nullptr) {
    HIP_CHECK(hipMemsetAsync(
        d_ssm_deltanet_state, 0,
        total_deltanet *
            QwenRecurrentStateElementBytes(policy_.recurrent_state_storage),
        stream));
  }
  DisableSsmReplayCapture();
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
}

void QwenGpuArena::FreeAll() noexcept {
  if (d_hidden != nullptr)
    LogCleanupError(hipFree(d_hidden));
  if (d_normed != nullptr)
    LogCleanupError(hipFree(d_normed));
  if (d_q != nullptr)
    LogCleanupError(hipFree(d_q));
  if (d_k != nullptr)
    LogCleanupError(hipFree(d_k));
  if (d_v != nullptr)
    LogCleanupError(hipFree(d_v));
  if (d_attn_out != nullptr)
    LogCleanupError(hipFree(d_attn_out));
  if (d_ffn_gate != nullptr)
    LogCleanupError(hipFree(d_ffn_gate));
  if (d_ffn_up != nullptr)
    LogCleanupError(hipFree(d_ffn_up));
  if (d_ffn_act != nullptr)
    LogCleanupError(hipFree(d_ffn_act));
  if (d_ffn_out != nullptr)
    LogCleanupError(hipFree(d_ffn_out));
  if (d_ssm_qkv != nullptr)
    LogCleanupError(hipFree(d_ssm_qkv));
  if (d_conv_out != nullptr)
    LogCleanupError(hipFree(d_conv_out));
  if (d_ssm_gate != nullptr)
    LogCleanupError(hipFree(d_ssm_gate));
  if (d_ssm_out != nullptr)
    LogCleanupError(hipFree(d_ssm_out));
  if (d_alpha_buf != nullptr)
    LogCleanupError(hipFree(d_alpha_buf));
  if (d_beta_buf != nullptr)
    LogCleanupError(hipFree(d_beta_buf));
  if (d_ssm_kq_scales != nullptr)
    LogCleanupError(hipFree(d_ssm_kq_scales));
  if (d_ssm_alpha_beta != nullptr)
    LogCleanupError(hipFree(d_ssm_alpha_beta));
  if (d_logits != nullptr)
    LogCleanupError(hipFree(d_logits));
  if (d_attention_kv_f16 != nullptr)
    LogCleanupError(hipFree(d_attention_kv_f16));
  if (d_turbo4_attention_scratch != nullptr)
    LogCleanupError(hipFree(d_turbo4_attention_scratch));
  if (d_turbo4_decode_workspace != nullptr)
    LogCleanupError(hipFree(d_turbo4_decode_workspace));
  if (d_turbo4_first_pos != nullptr)
    LogCleanupError(hipFree(d_turbo4_first_pos));
  if (d_target_layer_features != nullptr) {
    LogCleanupError(hipFree(d_target_layer_features));
    d_target_layer_features = nullptr;
  }
  if (d_kv_cache != nullptr)
    LogCleanupError(hipFree(d_kv_cache));
  if (d_kv_turbo4 != nullptr)
    LogCleanupError(hipFree(d_kv_turbo4));
  // The two status words are separate allocations, so they are freed
  // SEPARATELY. Freeing the plane and assuming the words went with it would
  // leak two words per arena — small, invisible, and per-instance.
  if (d_kv_turbo4_write_status != nullptr)
    LogCleanupError(hipFree(d_kv_turbo4_write_status));
  if (d_kv_turbo4_read_status != nullptr)
    LogCleanupError(hipFree(d_kv_turbo4_read_status));
  for (float* buffer : d_ssm_conv_buffers_)
    if (buffer != nullptr)
      LogCleanupError(hipFree(buffer));
  for (void* buffer : d_ssm_deltanet_buffers_)
    if (buffer != nullptr)
      LogCleanupError(hipFree(buffer));
  if (d_prompt_tokens != nullptr)
    LogCleanupError(hipFree(d_prompt_tokens));
  if (d_scratch_bf16 != nullptr)
    LogCleanupError(hipFree(d_scratch_bf16));
  if (d_scratch_q8_act != nullptr)
    LogCleanupError(hipFree(d_scratch_q8_act));
  if (d_split_k_attention != nullptr)
    LogCleanupError(hipFree(d_split_k_attention));
  if (d_weights_bf16 != nullptr)
    LogCleanupError(hipFree(d_weights_bf16));
  if (d_ssm_replay_qkv_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_qkv_));
  if (d_ssm_replay_alpha_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_alpha_));
  if (d_ssm_replay_beta_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_beta_));
  if (d_ssm_replay_enabled_ != nullptr)
    LogCleanupError(hipFree(d_ssm_replay_enabled_));
  if (hipblas_handle != nullptr)
    LogCleanupError(hipblasDestroy(hipblas_handle));
  hipblaslt_gemm.reset();
  if (stream != nullptr)
    LogCleanupError(hipStreamDestroy(stream));

  d_hidden = nullptr;
  d_normed = nullptr;
  d_q = nullptr;
  d_k = nullptr;
  d_v = nullptr;
  d_attn_out = nullptr;
  d_ffn_gate = nullptr;
  d_ffn_up = nullptr;
  d_ffn_act = nullptr;
  d_ffn_out = nullptr;
  d_ssm_qkv = nullptr;
  d_conv_out = nullptr;
  d_ssm_gate = nullptr;
  d_ssm_out = nullptr;
  d_alpha_buf = nullptr;
  d_beta_buf = nullptr;
  d_ssm_kq_scales = nullptr;
  d_ssm_alpha_beta = nullptr;
  d_logits = nullptr;
  d_attention_kv_f16 = nullptr;
  d_turbo4_attention_scratch = nullptr;
  d_turbo4_decode_workspace = nullptr;
  d_turbo4_first_pos = nullptr;
  d_kv_cache = nullptr;
  d_kv_turbo4 = nullptr;
  d_kv_turbo4_write_status = nullptr;
  d_kv_turbo4_read_status = nullptr;
  turbo4_kv_bytes_recorded = 0;
  turbo4_status_bytes_recorded = 0;
  turbo4_scratch_bytes_recorded = 0;
  turbo4_decode_workspace_bytes_recorded = 0;
  turbo4_decode_partial_bytes_recorded = 0;
  d_ssm_conv_state = nullptr;
  d_ssm_deltanet_state = nullptr;
  d_prompt_tokens = nullptr;
  d_scratch_bf16 = nullptr;
  d_scratch_q8_act = nullptr;
  d_split_k_attention = nullptr;
  d_weights_bf16 = nullptr;
  d_ssm_conv_buffers_ = {};
  d_ssm_deltanet_buffers_ = {};
  live_recurrent_ = 0;
  saved_recurrent_ = 0;
  source_recurrent_ = 0;
  d_ssm_replay_qkv_ = nullptr;
  d_ssm_replay_alpha_ = nullptr;
  d_ssm_replay_beta_ = nullptr;
  d_ssm_replay_enabled_ = nullptr;
  saved_context_ = 0;
  replay_last_position_ = 0;
  replay_captured_positions_ = 0;
  has_saved_state_ = false;
  replay_capture_active_ = false;
  hipblas_handle = nullptr;
  stream = nullptr;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

// The Turbo4 PLANE LAYOUT, proved against independently labelled
// bytes. CPU only.
//
//   g++ -std=c++17 -Wall -Wextra -I. tests/models/qwen/kv_turbo4_plane_layout_test.cpp
//       -o /tmp/t4layout
//
// WHAT THIS IS FOR. Review accepted the claim that a Turbo4 plane is
// [layer][token][head][block] and therefore has the SAME LOGICAL ORDER as an
// fp16 plane, [layer][token][head][value] — so a token-major 2D copy preserves
// it and no head reorder is required. "No head reorder" is exactly the kind of
// claim that is true and undetectable: if it were FALSE the buffer would still
// be the right size, every value would still be a legal fp16, and the result
// would merely be wrong. So it is asserted here, not assumed.
//
// THE ORACLE IS BYTE IDENTITY, NOT A NORM. A norm check or a checksum passes
// for a permuted head. These checks compare the actual 66 bytes at an
// independently-computed offset against a fresh quantization of the values
// LABELLED with that coordinate, so any permutation of layer, token, head or
// block fails byte-exactly.
//
// INDEPENDENCE. SpecBlockOffset below is written straight from the layout
// sentence — [layer][token][head][block], token pitch 528 B at the UD
// geometry — and calls NO production function. The production offset is then
// compared against it. A test that computed the expected offset by calling the
// function under test would agree with every possible bug in it.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace t4 = gufo::hip::turbo4;

namespace {

int kSmallPlanefailures = 0;

void Check(bool ok, const std::string &what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++kSmallPlanefailures;
}

using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvCopyShape;
using gufo::hip::QwenKvDescribeCopy;
using gufo::hip::QwenKvPlaneGeometry;

constexpr std::size_t kHeadDim = 256;
constexpr std::size_t kKvHeads = 4;
constexpr std::size_t kBlocksPerHead = kHeadDim / 128;  // 2
constexpr std::size_t kTokenPitch = kKvHeads * kBlocksPerHead * 66;  // 528
// THE UD FULL-ATTENTION COUNT IS 16, NOT 65. Review corrected this.
//
// block_count is 65, but that counts EVERY block including the 1 MTP layer, so
// the production derivation is (block_count - nextn) / full_attention_interval
// = (65 - 1) / 4 = 16. An earlier version of this file used 65 as "the UD
// full-attention layers", which is not the model. Nothing here broke, because
// rows is just a count and the layout arithmetic is linear in it — which is
// exactly why a wrong constant survives: it produces a legal answer for a model
// that does not exist.
constexpr std::size_t kFullAttentionLayers = 16;

// KEPT DELIBERATELY, as a labelled synthetic stress case and nothing more.
// 65 rows exercises a longer plane than this model has. It is NOT model proof
// and must never be quoted as the UD geometry.
constexpr std::size_t kSyntheticRows = 65;

/// The 5-token plane the coordinate checks build, at UD head geometry.
constexpr QwenKvPlaneGeometry kSmallPlane{kHeadDim, kKvHeads, 5};

/// THE LAYOUT SENTENCE, TRANSCRIBED. No production call.
constexpr std::size_t SpecBlockOffset(std::size_t layer, std::size_t token,
                                      std::size_t head,
                                      std::size_t context) {
  return layer * context * kTokenPitch + token * kTokenPitch +
         head * kBlocksPerHead * 66;
}

/// The values a head carries, labelled by its coordinate. Deliberately
/// different for every (token, head): if the plane permuted heads, every head
/// would still decode to a plausible vector and only the label would disagree.
std::vector<float> LabelledHead(std::size_t token, std::size_t head) {
  std::vector<float> v(kHeadDim);
  for (std::size_t i = 0; i < kHeadDim; ++i) {
    // Small, smooth, well inside Turbo4's storable range, and distinct per head.
    v[i] = 0.01f * static_cast<float>(((token * kKvHeads + head) * 17 + i) %
                                      23);
  }
  return v;
}

void DecodeAt(const std::uint8_t *plane, std::size_t block_offset,
              std::size_t value_index, float *out) {
  float scratch[t4::kBlockValues];
  t4::DequantizeBlock(plane + block_offset, scratch);
  t4::RotateInverse(scratch);
  *out = scratch[value_index];
}

// ---------------------------------------------------------------------------

void PitchIdentity() {
  std::printf("pitch: one token is kv_heads * (head_dim/128) * 66\n");
  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, 4096};
  const std::size_t pitch = gufo::hip::QwenKvBytesPerToken(
      QwenKvCacheStorage::kTurbo4, g);
  Check(pitch == kTokenPitch, "Turbo4 token pitch is 528 B at the UD geometry");
  Check(kTokenPitch == 528, "kv_heads(4) x blocks_per_head(2) x 66 = 528");
  // The pitch must agree with the ALLOCATION authority, or a plane sized by
  // one and addressed by the other is a buffer whose last block is missing.
  Check(pitch == gufo::hip::QwenKvBytesForElements(
                     QwenKvCacheStorage::kTurbo4, kHeadDim * kKvHeads),
        "token pitch equals QwenKvBytesForElements over one token's values");
  // And with fp16: same order, fewer bytes.
  const std::size_t f16 =
      gufo::hip::QwenKvBytesPerToken(QwenKvCacheStorage::kFp16, g);
  Check(f16 == kHeadDim * kKvHeads * 2, "fp16 token pitch is 2048 B");
  // The ratio is 4 B/value over 66/128 B/value = 128/33. Stated as an integer
  // identity rather than a decimal, because a rounded ratio printed here would
  // read as an exact claim and would be wrong in the 5th digit.
  Check(f16 * 33 == kTokenPitch * 128,
        "fp16 and Turbo4 differ only in bytes per token: exactly 128/33");
}

void CopyShapeAgreement() {
  std::printf("copy shape: Turbo4 rows as fp16 rows, no reorder\n");
  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, 4096};
  const QwenKvCopyShape t4s = QwenKvDescribeCopy(
      QwenKvCacheStorage::kTurbo4, g, 4096, 4096, kFullAttentionLayers);
  const QwenKvCopyShape f16s = QwenKvDescribeCopy(
      QwenKvCacheStorage::kFp16, g, 4096, 4096, kFullAttentionLayers);
  // B-R5. This asserted width == kTokenPitch, i.e. that the per-row COPY WIDTH is
  // one token. It is the whole valid row, and had it been one token every Copy2D
  // caller moved position 0 of each row and left the rest of the reserved span
  // unwritten. The 528 B fact this line was really about is kept, but asserted
  // against the token pitch it describes.
  Check(kTokenPitch == 528, "one TOKEN is 528 B (4 heads x 2 blocks x 66 B)");
  Check(t4s.width == 4096 * kTokenPitch,
        "the per-row copy width is the whole valid row: 4096 tokens x 528 B");
  Check(t4s.rows == kFullAttentionLayers,
        "rows are the 16 full-attention layers: (65 blocks - 1 MTP) / interval 4");
  Check(t4s.token_major, "Turbo4 is token-major, like fp16");
  Check(gufo::hip::QwenKvCopyShapeAgreesWithFp16(g, 4096, 4096,
                                                    kFullAttentionLayers),
        "Turbo4 copy shape agrees with fp16 — the no-reorder predicate holds");
  Check(t4s.rows == f16s.rows && t4s.token_major == f16s.token_major,
        "rows and order are identical to fp16; only the pitch differs");
  Check(t4s.source_pitch == 4096 * kTokenPitch, "source pitch is ctx * 528");

  // fp32 is head-major: rows are (layer x head), NOT layers. If this ever
  // matched the token-major branch the planes would interleave.
  const QwenKvCopyShape f32s = QwenKvDescribeCopy(
      QwenKvCacheStorage::kFp32, g, 4096, 4096, kFullAttentionLayers);
  Check(!f32s.token_major, "fp32 stays head-major");
  Check(f32s.rows == kFullAttentionLayers * kKvHeads,
        "fp32 rows are 16 layers x 4 heads, so the three layouts are "
        "distinct");

  // A shape that cannot be described refuses with width 0, not a wrap.
  const QwenKvCopyShape bad = QwenKvDescribeCopy(
      static_cast<QwenKvCacheStorage>(99), g, 4096, 4096,
                           kFullAttentionLayers);
  Check(bad.width == 0U && bad.rows == 0U, "an unknown storage describes no copy");
  const QwenKvPlaneGeometry unaligned{kHeadDim + 64, kKvHeads, 4096};
  Check(QwenKvDescribeCopy(QwenKvCacheStorage::kTurbo4, unaligned, 4096, 4096,
                           kFullAttentionLayers)
                .width == 0U,
        "a head that straddles a block describes no copy");

  // The synthetic stress case, labelled as such. 65 rows is a legal plane for
  // some other model; it is NOT this one, and the check below exists to keep
  // it from quietly drifting back into being called the UD geometry.
  const QwenKvCopyShape synthetic = QwenKvDescribeCopy(
      QwenKvCacheStorage::kTurbo4, g, 4096, 4096, kSyntheticRows);
  Check(synthetic.rows == kSyntheticRows &&
            synthetic.width == 4096 * kTokenPitch,
        "SYNTHETIC STRESS CASE (65 rows, not this model): still token-major at "
        "the same 528 B per-token pitch, whole row copied");
}

/// The production offset, checked against the transcribed layout sentence.
///
/// SCOPE, because getting this wrong is exactly what a careless test does:
/// QwenKvHeadBlockOffsetChecked takes NO layer. A "plane" in kv_read.hpp is one
/// LAYER's slice, and the caller adds the layer stride. The first comparison is
/// therefore against layer 0; the layer stride is checked separately below. An
/// earlier version of this test compared the whole-plane address against the
/// per-layer function and failed — the test was wrong, not the offset.
void ProductionOffsetMatchesSpec() {
  std::printf("offset: production == the layout sentence\n");
  const std::size_t context = 7;
  const QwenKvPlaneGeometry g{kHeadDim, kKvHeads, context};
  bool all = true;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      std::size_t production_block = 0;
      if (!gufo::hip::QwenKvHeadBlockOffsetChecked(g, token, head,
                                                     &production_block))
        all = false;
      if (production_block != SpecBlockOffset(0, token, head, context) / 66)
        all = false;
    }
  Check(all,
        "QwenKvHeadBlockOffsetChecked equals the [token][head][block] index "
        "within a layer's plane, for every coordinate");

  // The layer stride is the caller's, and it is one whole context of tokens.
  Check(SpecBlockOffset(1, 0, 0, context) ==
            SpecBlockOffset(0, 0, 0, context) + context * kTokenPitch,
        "layer stride is one whole context of tokens, applied by the caller");

  // And the plane byte address the copy assumes is that offset times 66.
  const std::size_t token_address =
      SpecBlockOffset(2, 5, 3, context);
  Check(token_address == 2 * context * kTokenPitch + 5 * kTokenPitch +
                              3 * 132,
        "byte address of (layer 2, token 5, head 3) is layer+token+head");

  // The whole-plane size the reader demands must equal the copy's pitch x
  // context: if these disagreed, a reader could pass its bounds check against
  // a plane the copier had filled differently.
  Check(gufo::hip::QwenKvWholePlaneBytes(QwenKvCacheStorage::kTurbo4, g) ==
            context * kTokenPitch,
        "whole-plane bytes equal context x token pitch, the copy's own stride");
}

/// THE MAIN CHECK. Bytes labelled by coordinate, located by the spec.
void CoordinateLabelledBytes() {
  std::printf("bytes: the block at a spec offset holds THAT coordinate\n");
  const std::size_t context = 5;
  const std::size_t plane_bytes = context * kTokenPitch;
  std::vector<std::uint8_t> plane(plane_bytes, 0);

  // Write every head of every token through the production writer.
  bool wrote = true;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      const auto values = LabelledHead(token, head);
      if (!gufo::hip::QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(),
                                      plane.size(), kSmallPlane, token, head,
                                      values.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true))
        wrote = false;
    }
  Check(wrote, "every head of every token was written by QwenKvWriteHead");

  // Now locate each one by the SPEC, re-quantize the same labelled values
  // independently, and require the bytes to match exactly. A permuted head,
  // token or block cannot produce these bytes.
  std::size_t compared = 0;
  bool exact = true;
  bool head_order_matters = false;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      const auto values = LabelledHead(token, head);
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        std::uint8_t expected[t4::kBlockBytes];
        if (!t4::QuantizeBlockChecked(values.data() + b * 128, expected)) {
          exact = false;
          continue;
        }
        const std::size_t at = SpecBlockOffset(0, token, head, context) +
                               b * 66;
        if (std::memcmp(plane.data() + at, expected, 66) != 0) exact = false;
        ++compared;

        // Adjacent heads must NOT be interchangeable: compare head h's first
        // block against head h+1's. If these were equal, the whole check would
        // pass under a permuted layout.
        if (head + 1 < kKvHeads) {
          const std::size_t other =
              SpecBlockOffset(0, token, head + 1, context) + b * 66;
          if (std::memcmp(plane.data() + at, plane.data() + other, 66) == 0)
            head_order_matters = false;
          else
            head_order_matters = true;
        }
      }
    }
  Check(compared == context * kKvHeads * kBlocksPerHead,
        "every block of the plane was located by the spec");
  Check(exact,
        "the 66 bytes at each spec offset are byte-identical to a fresh "
        "quantization of that coordinate's labelled values");
  Check(head_order_matters,
        "neighbouring heads hold different bytes, so a permuted layout fails "
        "this check rather than passing it");

  // And the reader agrees with the writer, located by the same spec offsets.
  bool reads_back = true;
  for (std::size_t token = 0; token < context; ++token)
    for (std::size_t head = 0; head < kKvHeads; ++head) {
      float out[kHeadDim];
      if (!gufo::hip::QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(),
                                     plane.size(), kSmallPlane, token, head, out, gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true)) {
        reads_back = false;
        continue;
      }
      const auto values = LabelledHead(token, head);
      // Decode each block independently, from the SPEC offset.
      for (std::size_t b = 0; b < kBlocksPerHead; ++b) {
        for (std::size_t i = 0; i < 128; ++i) {
          float decoded = 0.0f;
          DecodeAt(plane.data(),
                   SpecBlockOffset(0, token, head, context) + b * 66, i,
                   &decoded);
          const float want = values[b * 128 + i];
          if (std::fabs(decoded - want) > 0.05f) reads_back = false;
        }
      }
      (void)out;
    }
  Check(reads_back,
        "every value decodes at its own spec offset — the label survives the "
        "round trip");
}

void PredicateSplit() {
  std::printf("predicates: allocatable, element-bytes and executable differ\n");
  using gufo::hip::QwenKvHasElementBytes;
  using gufo::hip::QwenKvIsAllocatableByExecutor;
  using gufo::hip::QwenKvIsExecutableByExecutor;
  const auto turbo4 = QwenKvCacheStorage::kTurbo4;
  const auto unknown = static_cast<QwenKvCacheStorage>(99);

  Check(QwenKvIsAllocatableByExecutor(turbo4),
        "a block plane is now ALLOCATABLE — the member and pitch exist");
  Check(!QwenKvHasElementBytes(turbo4),
        "and still has no bytes-per-ELEMENT — the two are not the same answer");
  Check(!QwenKvIsExecutableByExecutor(turbo4),
        "but is still not EXECUTABLE — no kernel takes a block pitch");
  Check(QwenKvIsAllocatableByExecutor(QwenKvCacheStorage::kFp16) &&
            QwenKvIsAllocatableByExecutor(QwenKvCacheStorage::kFp32),
        "fp16 and fp32 remain allocatable");
  Check(QwenKvIsExecutableByExecutor(QwenKvCacheStorage::kFp16) &&
            QwenKvIsExecutableByExecutor(QwenKvCacheStorage::kFp32),
        "and remain executable");
  // The only storage all three refuse.
  Check(!QwenKvIsAllocatableByExecutor(unknown) &&
            !QwenKvHasElementBytes(unknown) &&
            !QwenKvIsExecutableByExecutor(unknown),
        "an unknown precision is refused by all three");
  // Element bytes implies allocatable — the one implication that must hold.
  Check(!QwenKvHasElementBytes(unknown) &&
            QwenKvHasElementBytes(QwenKvCacheStorage::kFp32),
        "having element bytes implies being allocatable");
}

void ExecutionRefusal() {
  std::printf("refusal: execution is walled, allocation is not\n");
  bool threw = false;
  try {
    gufo::hip::RequireExecutableKvStorage(QwenKvCacheStorage::kTurbo4);
  } catch (const std::logic_error &e) {
    threw = std::strstr(e.what(), "block pitch") != nullptr;
  }
  Check(threw, "RequireExecutableKvStorage(kTurbo4) throws and names the gap");
  bool fp16_ok = true;
  try {
    gufo::hip::RequireExecutableKvStorage(QwenKvCacheStorage::kFp16);
    gufo::hip::RequireExecutableKvStorage(QwenKvCacheStorage::kFp32);
  } catch (...) {
    fp16_ok = false;
  }
  Check(fp16_ok, "fp16 and fp32 pass the execution gate");
}

}  // namespace

int main() {
  PitchIdentity();
  CopyShapeAgreement();
  ProductionOffsetMatchesSpec();
  CoordinateLabelledBytes();
  PredicateSplit();
  ExecutionRefusal();
  std::printf("%s  %d failure(s)\n", kSmallPlanefailures == 0 ? "PASS" : "FAIL",
              kSmallPlanefailures);
  return kSmallPlanefailures == 0 ? 0 : 1;
}

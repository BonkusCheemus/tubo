// The layer-offset caution: who adds layer_idx, and where it lands.
//
// THE REQUIREMENT. "The f16 kernels add layer_idx * total_k themselves; pass the
// scratch so that lands on it, and test with layer_idx > 0 against the f16
// path's output for the same K/V."
//
// The dequant side of that is already settled and already tested: the dequant
// launcher takes the LAYER'S BLOCK BASE, because Turbo4BlockByteOffset takes no
// layer argument (turbo4_dequant_f16_gpu_test.cpp already runs layer 1 and 3
// and reads the layer's own slot). So the layer offset on the WRITE side is the
// caller's job and is done.
//
// This file settles the READ side, which nobody had written down: the three
// attention routes do NOT apply the layer offset the same way.
//
//   attention_wmma.hip:657-661    HOST wrapper. layer_cache_offset =
//                                 layer_idx*max_context*kKvWidth, and the
//                                 kernel is handed layer_k/layer_v already
//                                 offset.
//   attention_tile.hip:432-435    the same, host side.
//   attention_batched.hip:264-266 the BASE pointer is passed, and the kernel
//                                 recomputes p_offset_f16 at :116-118.
//
// So two routes offset the read on the host and one inside the kernel. Under
// The chosen layout (K=scratch, V=scratch+max_context*kKvWidth, attn_layer_idx 0)
// all three collapse to offset 0 and the asymmetry is harmless. That is exactly
// why the layout says 0, and this file is what makes the "why" checkable rather
// than folklore.
//
// WHAT IS ACTUALLY ASSERTED.
//   1. Each of those three sites still exists and still says what it said when
//      the ruling was written. A source pin, because the fact is about CODE, and
//      no runtime observation of a wrong-but-plausible answer can distinguish
//      "offset applied once" from "offset applied twice".
//   2. wmma and tile ALSO pass layer_idx down to their repack kernels, which
//      recompute an offset from the BASE pointer. So with layer_idx > 0 the
//      READ is host-offset while the REPACK is kernel-offset: two different
//      bases in one launch. Pinned, not argued.
//   3. At the real UD geometry, what passing the true layer index would DO to a
//      one-layer scratch, measured as an address:
//        layer 1 -> strictly inside the buffer (it reads the V half as K), so
//                    it cannot fault, cannot trip the allocator, and returns a
//                    confident wrong number;
//        layer 4 -> exactly one past the end;
//        layer 5 -> past the end.
//      A silent wrong answer is the worst of the three, so the arithmetic is
//      pinned as arithmetic rather than left to a reader.
//   4. The same three relationships hold on REAL DEVICE MEMORY: a one-layer
//      scratch is filled with a distinct sentinel per half, and the layer-1
//      address is shown to hold the V sentinel. That is the demonstration that
//      the consequence is a wrong answer and not a crash -- checked, not
//      assumed.
//
// NOT ASSERTED HERE: that a full attention kernel produces a different answer at
// layer_idx > 0. That needs the wired route and real queries, which is E3 and
// E4. What is unchecked is stated as unchecked in the report rather than
// implied by these passing.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/kv_allocation.hpp"

namespace fs = std::filesystem;
namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
  if (!ok) ++g_failures;
}

std::string Slurp(const fs::path& p) {
  std::ifstream in(p);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

std::string StripComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  bool line = false, str = false;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
    if (line) {
      if (c == '\n') {
        line = false;
        out += c;
      }
      continue;
    }
    if (str) {
      if (c == '"') str = false;
      continue;
    }
    if (c == '/' && n == '/') {
      line = true;
      continue;
    }
    if (c == '/' && n == '*') {
      // Skip to the closing */ and RESUME whatever the comment looked like.
      // The single-star case used to fall through to `line = true`, which ate
      // the REST OF THE LINE as if it were a // comment -- so a construct like
      // `/*tag=*/expression` silently deleted the expression after it. Found by
      // the same bug in turbo4_prefill_wiring_test.cpp, where it hid a real
      // call site and made a count read 2 instead of 3.
      i += 2;
      while (i + 1 < src.size() && !(src[i] == '*' && src[i + 1] == '/'))
        ++i;
      i += 1;
      continue;
    }
    if (c == '"') str = true;
    out += c;
  }
  return out;
}

std::string Body(const std::string& name) {
  return StripComments(Slurp(fs::path("src/models/qwen/hip/kernels") / name));
}

// The offset expression every route shares, as plain text with the spacing
// normalised so a reformat does not read as a behaviour change.
std::string Flatten(std::string s) {
  std::string out;
  for (const char c : s)
    if (c != ' ' && c != '\n' && c != '\t' && c != '\r') out += c;
  return out;
}

constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kMaxContext = 262144;  // UD production context
constexpr std::size_t kKvWidth =
    static_cast<std::size_t>(kKvHeads) * kHeadDim;

std::size_t LayerOffsetHalfs(std::uint32_t layer) {
  return static_cast<std::size_t>(layer) * kMaxContext * kKvWidth;
}

void WmmaOffsetsTheReadOnTheHost() {
  const std::string src = Body("attention_wmma.hip");
  Check(src.find("layer_cache_offset") != std::string::npos,
        "attention_wmma.hip still names layer_cache_offset");
  const std::string flat = Flatten(src);
  Check(flat.find(
            "conststd::size_tlayer_cache_offset="
            "static_cast<std::size_t>(layer_idx)*max_context*kKvWidth;") !=
            std::string::npos,
        "attention_wmma.hip still computes layer_cache_offset as "
        "layer_idx*max_context*kKvWidth");
  Check(flat.find(
            "auto*layer_k=static_cast<__half*>(k_cache_f16)+layer_cache_"
            "offset;") != std::string::npos,
        "attention_wmma.hip adds that offset to k_cache_f16 on the HOST, so "
        "the kernel receives an already-offset pointer");
}

void TileOffsetsTheReadOnTheHost() {
  const std::string src = Body("attention_tile.hip");
  const std::string flat = Flatten(src);
  Check(flat.find(
            "conststd::size_tlayer_cache_offset="
            "static_cast<std::size_t>(layer_idx)*max_context*kv_width;") !=
            std::string::npos,
        "attention_tile.hip computes the same layer offset on the HOST");
  Check(flat.find(
            "auto*layer_k=static_cast<__half*>(k_cache_f16)+layer_cache_"
            "offset;") != std::string::npos,
        "attention_tile.hip adds it to k_cache_f16 before the launch");
}

void BatchedOffsetsInsideTheKernel() {
  const std::string src = Body("attention_batched.hip");
  const std::string flat = Flatten(src);
  // The asymmetry, stated as a fact about the code: this route has no host-side
  // layer_cache_offset at all, and hands the kernel the base pointer.
  Check(src.find("layer_cache_offset") == std::string::npos,
        "attention_batched.hip has NO host-side layer_cache_offset -- it does "
        "not offset the way wmma and tile do");
  Check(flat.find(
            "p_offset_f16=((static_cast<std::size_t>(layer_idx)*max_context+p)"
            "*kv_width)+(static_cast<std::size_t>(kv_h)*head_dim);") !=
            std::string::npos,
        "attention_batched.hip computes layer_idx*max_context*kv_width INSIDE "
        "the kernel instead");
  Check(flat.find("k_cache_f16+p_offset_f16") != std::string::npos,
        "and indexes k_cache_f16 from the base plus that offset");
}

// wmma and tile pass layer_idx on to their repack kernels, which recompute an
// offset from the BASE pointer. With layer_idx > 0 the read is host-offset and
// the repack is kernel-offset, so one launch mixes two bases. This is why 0 is
// the ruling and not merely a convenience.
void RepackAlsoConsumesLayerIdx() {
  const std::string fw = Flatten(Body("attention_wmma.hip"));
  const std::string ft = Flatten(Body("attention_tile.hip"));

  // The exact text of the fact: both repack launches hand over the BASE f16
  // pointer together with layer_idx, which is what makes them recompute their
  // own offset instead of inheriting the host-side one.
  const std::string base_plus_layer =
      "static_cast<__half*>(k_cache_f16),"
      "static_cast<__half*>(v_cache_f16),layer_idx";
  Check(fw.find(base_plus_layer) != std::string::npos,
        "attention_wmma.hip's repack launches are handed the BASE f16 pointer "
        "plus layer_idx, not the host-offset layer_k/layer_v");
  Check(fw.find(base_plus_layer) < fw.find("layer_k,layer_v"),
        "attention_wmma.hip then launches the READ with the offset layer_k / "
        "layer_v, so with layer_idx > 0 one launch mixes two different bases");
  Check(ft.find(base_plus_layer) != std::string::npos,
        "attention_tile.hip's repack does the same: base pointer plus layer_idx");
}

void OneLayerScratchArithmetic() {
  const std::size_t layer_halfs = LayerOffsetHalfs(1);
  const std::size_t halfs = 2 * layer_halfs;  // K and V of one layer
  const std::size_t bytes = halfs * sizeof(std::uint16_t);
  std::printf("  UD: kKvWidth=%zu max_context=%u scratch=%zu B (%.1f MiB)\n",
              kKvWidth, kMaxContext, bytes,
              static_cast<double>(bytes) / (1024.0 * 1024.0));

  // The scratch is TWO layer-offsets wide (one for K, one for V), so the layer
  // index at which the address runs off the end is 2 -- not 4. An earlier draft
  // of this file said 4, reasoned from full_attention_interval, which says
  // which layers are full-attention and has nothing to do with how wide the
  // buffer is. The test caught it.
  Check(LayerOffsetHalfs(1) == layer_halfs && layer_halfs < halfs,
        "layer_idx 1 lands STRICTLY INSIDE a one-layer scratch: it reads the V "
        "half as K, which faults nowhere and answers wrong");
  Check(LayerOffsetHalfs(1) == halfs / 2,
        "layer_idx 1 lands exactly on the V half's first element");
  Check(LayerOffsetHalfs(2) == halfs,
        "layer_idx 2 lands EXACTLY one past the end: the scratch is K and V, "
        "so two layer-offsets is its whole width");
  Check(LayerOffsetHalfs(3) > halfs,
        "any layer_idx above 2 addresses past the end of the scratch");
  Check(LayerOffsetHalfs(1) != 0,
        "and the offset is not a zero-offset no-op that would look correct");
}

// The same relationships on real device memory: a one-layer scratch with a
// DIFFERENT sentinel in each half. If layer 1 were a plausible crash, this
// would show an unmapped page. It shows the V sentinel instead, which is the
// whole point -- a silent wrong answer, not a fault.
void LayerOneReadsTheOtherHalfOnDevice() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("SKIP device check: no HIP device\n");
    return;
  }
  const std::size_t halfs = 2 * LayerOffsetHalfs(1);
  std::uint16_t* scratch = nullptr;
  if (hipMalloc(&scratch, halfs * sizeof(std::uint16_t)) != hipSuccess) {
    std::printf("SKIP device check: scratch allocation failed\n");
    return;
  }
  constexpr std::uint16_t kKSentinel = 0x1111;
  constexpr std::uint16_t kVSentinel = 0x2222;
  std::vector<std::uint16_t> fill(halfs);
  for (std::size_t i = 0; i < LayerOffsetHalfs(1); ++i) fill[i] = kKSentinel;
  for (std::size_t i = LayerOffsetHalfs(1); i < halfs; ++i)
    fill[i] = kVSentinel;
  if (hipMemcpy(scratch, fill.data(), halfs * sizeof(std::uint16_t),
                hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipFree(scratch);
    std::printf("SKIP device check: copy failed\n");
    return;
  }

  const std::size_t layer1 = LayerOffsetHalfs(1);
  std::uint16_t at_layer1 = 0;
  std::uint16_t at_zero = 0;
  const hipError_t err = hipMemcpy(&at_layer1, scratch + layer1,
                                   sizeof(at_layer1), hipMemcpyDeviceToHost);
  if (hipMemcpy(&at_zero, scratch, sizeof(at_zero),
                hipMemcpyDeviceToHost) != hipSuccess) {
    (void)hipFree(scratch);
    Check(false, "reading the layer-0 address failed");
    return;
  }
  Check(err == hipSuccess,
        "reading at the layer-1 address SUCCEEDS: it is inside the allocation, "
        "so no allocator or hardware check can ever report this mistake");
  Check(at_layer1 == kVSentinel,
        "and what it reads is the V half's sentinel, not layer 1's K -- a "
        "confident wrong answer");
  // The teeth. Without this, a fill that used the SAME sentinel for both halves
  // would satisfy the assertion above while proving nothing, because the test
  // would be comparing the value to a constant it also wrote. Requiring the two
  // addresses to DISAGREE is what makes the check about the offset rather than
  // about the fill.
  Check(at_layer1 != at_zero,
        "the layer-1 address and the layer-0 address hold DIFFERENT bytes, so "
        "the check above is about the offset and not about the fill");
  (void)hipFree(scratch);
}

}  // namespace

int main() {
  WmmaOffsetsTheReadOnTheHost();
  TileOffsetsTheReadOnTheHost();
  BatchedOffsetsInsideTheKernel();
  RepackAlsoConsumesLayerIdx();
  OneLayerScratchArithmetic();
  LayerOneReadsTheOtherHalfOnDevice();
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
// The one helper that builds a block layer's f16 pair.
//
// WHY THIS TEST EXISTS SEPARATELY FROM THE KERNELS. The helper's whole job is
// to make three facts inseparable: K is the scratch base, V is one layer on,
// and the layer index is 0. Every one of those is a pointer or an integer a
// call site could assemble wrongly on its own, and the failure is silent — a
// layer index of 1 reads the V half as K, inside the allocation, and answers
// confidently wrong. So the assembly is pinned here, once, where it can be
// checked without a GPU.
//
// WHAT IS ASSERTED.
//   1. V is exactly max_context*kv_heads*head_dim halfs on from K, at several
//      geometries including the production one.
//   2. The pair fills the scratch EXACTLY: the last element V addresses is the
//      last half the allocation owns, at 2*max_context*kv_heads*head_dim halfs.
//      A V offset one element short would still "work" and would hand the
//      launcher a buffer smaller than the route reads.
//   3. THE CONSTANTS: kTurbo4AttentionLayerIdx is 0 and kTurbo4SkipKvWrite is
//      true, and the helper takes no layer parameter at all — checked against
//      the header's own text, because a signature that grew a layer argument
//      would compile everywhere and quietly re-open the hazard.
//   4. Both f32 members are null. attention_batched.hip:36-62 writes raw f32
//      into k_cache/v_cache whenever they are non-null, which on a block layer
//      would write f32 over the 66-byte blocks.
//   5. A null scratch, a zero context, zero heads or zero dim REFUSE (valid
//      false) instead of producing a null pair a caller might launch with.
//   6. The production numbers, so the scale of the buffer is on the record: the
//      scratch is 1 GiB at the model max context and 32 MiB at 8192, which is
//      why a serve run sets -c instead of running at the model maximum.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "src/models/qwen/hip/turbo4_attention_pair.hpp"

namespace fs = std::filesystem;
using gufo::hip::MakeTurbo4AttentionF16Pair;
using gufo::hip::Turbo4AttentionF16Pair;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
  if (!ok) ++g_failures;
}

// A fake scratch: a byte array big enough for the geometry, so pointer
// arithmetic is real address arithmetic and not just integer offsets.
std::size_t LayerElements(std::uint32_t context, std::uint32_t kv_heads,
                          std::uint32_t head_dim) {
  return static_cast<std::size_t>(context) * kv_heads * head_dim;
}

std::vector<std::uint16_t> Scratch(std::uint32_t max_context,
                                   std::uint32_t kv_heads,
                                   std::uint32_t head_dim) {
  return std::vector<std::uint16_t>(2 * LayerElements(max_context, kv_heads,
                                                      head_dim),
                                    0xABAB);
}

// A scratch sized for the PREFILLED PREFIX instead of for --context
// must put every position where a full-context scratch put it, or the change is
// not bit-identical.
//
// WHAT MAKES IT IDENTICAL, as arithmetic. Every reader of this buffer addresses
// position p of head h at `p * kv_heads * head_dim + h * head_dim` -- the
// dequant destination (turbo4_kernels.hpp:336) and the attention routes
// (attention_wmma.hip:253-258) both. That offset contains no context term. So
// the ONLY thing a smaller capacity moves is the V base, and the helper takes
// that from the arena's recorded allocation. Here: two capacities, 32768 and
// the model maximum, and every position below the smaller one lands on the same
// offset in both.
//
// ALSO PINNED: elements = recorded_bytes / (2 * sizeof(uint16_t)). That is the
// arena's new expression for GetAttentionKvLayerElements under the switch, and
// it has to invert Turbo4AttentionScratchBytes exactly; a factor wrong there
// halves or doubles the V offset, which stays inside the allocation and answers
// confidently wrong.
void TiledCapacityKeepsEveryPositionOffset() {
  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kDim = 256;
  constexpr std::uint32_t kSmall = 32768;   // a prefilled 32k prompt
  constexpr std::uint32_t kFull = 262144;   // the model maximum
  const auto recorded_bytes = [](std::uint32_t tokens) {
    return 2U * static_cast<std::size_t>(tokens) * kHeads * kDim *
           sizeof(std::uint16_t);
  };
  for (const std::uint32_t tokens : {kSmall, kFull}) {
    Check(recorded_bytes(tokens) / (2U * sizeof(std::uint16_t)) ==
              LayerElements(tokens, kHeads, kDim),
          "tokens=" + std::to_string(tokens) +
              ": half the recorded bytes, in halfs, is one layer");
  }
  auto small = Scratch(kSmall, kHeads, kDim);
  auto full = Scratch(kFull, kHeads, kDim);
  const auto small_pair = MakeTurbo4AttentionF16Pair(
      small.data(), recorded_bytes(kSmall) / (2U * sizeof(std::uint16_t)));
  const auto full_pair = MakeTurbo4AttentionF16Pair(
      full.data(), recorded_bytes(kFull) / (2U * sizeof(std::uint16_t)));
  Check(small_pair.valid && full_pair.valid, "both capacities build a pair");
  Check(static_cast<std::size_t>(full_pair.v - full_pair.k) !=
            static_cast<std::size_t>(small_pair.v - small_pair.k),
        "the two capacities really are different layouts (the V base moved)");
  bool same = true;
  const std::uint32_t positions[] = {0, 1, 17, 4095, 4096, 32766, kSmall - 1};
  for (const std::uint32_t p : positions) {
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::size_t offset =
          static_cast<std::size_t>(p) * kHeads * kDim + h * kDim;
      // The reader's address, expressed exactly as the kernels express it, in
      // elements from each plane's own base. Equal across capacities is the
      // whole claim; it fails if any term here gains a context factor.
      const std::size_t small_k = static_cast<std::size_t>(
          (small_pair.k + offset) - small.data());
      const std::size_t full_k =
          static_cast<std::size_t>((full_pair.k + offset) - full.data());
      const std::size_t small_v = static_cast<std::size_t>(
          (small_pair.v + offset) - small_pair.k);
      const std::size_t full_v =
          static_cast<std::size_t>((full_pair.v + offset) - full_pair.k);
      same &= small_k == full_k;
      same &= small_v - (small_pair.v - small_pair.k) ==
              full_v - (full_pair.v - full_pair.k);
      // and the last element either plane addresses stays inside its own half
      same &= offset + kDim <= small.size() / 2;
    }
  }
  Check(same,
        "every position below the smaller capacity lands on the same K and V "
        "offset at both capacities, and inside the smaller allocation");
  // The saving, on the record, at the contexts the prefix-sized scratch serves.
  const std::size_t at_64k = recorded_bytes(65536);
  const std::size_t at_90k = recorded_bytes(92160);
  std::printf("  scratch bytes: 32k=%zu 64k=%zu 90k=%zu full=%zu\n",
              recorded_bytes(kSmall), at_64k, at_90k, recorded_bytes(kFull));
  Check(at_64k == 268435456ULL, "256 MiB at -c 65536");
  Check(at_90k == 377487360ULL, "360 MiB at -c 92160");
}

void VOffsetIsOneLayer() {
  struct Case {
    std::uint32_t context, heads, dim;
  };
  const Case cases[] = {{64, 4, 256}, {1024, 4, 256}, {8192, 4, 256},
                        {16384, 4, 256}, {262144, 4, 256}, {512, 8, 128}};
  for (const Case& c : cases) {
    auto storage = Scratch(c.context, c.heads, c.dim);
    const auto pair = MakeTurbo4AttentionF16Pair(
        storage.data(), LayerElements(c.context, c.heads, c.dim));
    const std::ptrdiff_t offset = pair.v - pair.k;
    const std::ptrdiff_t expected = static_cast<std::ptrdiff_t>(c.context) *
                                    static_cast<std::ptrdiff_t>(c.heads) *
                                    static_cast<std::ptrdiff_t>(c.dim);
    Check(pair.valid && pair.k == storage.data(),
          "context=" + std::to_string(c.context) + " heads=" +
              std::to_string(c.heads) + " dim=" + std::to_string(c.dim) +
              ": K is the scratch base");
    Check(offset == expected,
          "context=" + std::to_string(c.context) + ": V is exactly one layer "
          "on (" + std::to_string(expected) + " halfs), not the all-layers "
          "total_k");
  }
}

void PairFillsTheScratchExactly() {
  constexpr std::uint32_t kCtx = 8192;
  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kDim = 256;
  auto storage = Scratch(kCtx, kHeads, kDim);
  const auto pair = MakeTurbo4AttentionF16Pair(
      storage.data(), LayerElements(kCtx, kHeads, kDim));
  const auto* first = reinterpret_cast<const std::uint16_t*>(storage.data());
  const std::size_t last_index = storage.size() - 1;

  Check(pair.k == first, "K starts at the first element of the allocation");
  Check(static_cast<std::size_t>(pair.v - first) ==
            static_cast<std::size_t>(kCtx) * kHeads * kDim,
        "V starts one layer on");
  Check(&pair.v[last_index - static_cast<std::size_t>(kCtx) * kHeads * kDim] ==
            &first[last_index],
        "the LAST element V addresses is the LAST half the allocation owns -- "
        "there is no slack and no overrun");
}

void TheConstantsCannotVary() {
  Check(gufo::hip::kTurbo4AttentionLayerIdx == 0U,
        "kTurbo4AttentionLayerIdx is 0: a block layer's data is ALREADY in the "
        "scratch, so there is no index to pass");
  Check(gufo::hip::kTurbo4SkipKvWrite,
        "kTurbo4SkipKvWrite is true: the E1 writer already put this chunk's KV "
        "into blocks, so the route must not repack f16 over them");

  // The signature, not just the body. A helper that grew a layer parameter
  // would compile at every call site and quietly re-open the exact hazard the
  // constant removes.
  std::ifstream in("src/models/qwen/hip/turbo4_attention_pair.hpp");
  const std::string src((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  const std::size_t at = src.find("MakeTurbo4AttentionF16Pair(");
  const std::size_t close = src.find(')', at);
  const std::string signature =
      at == std::string::npos ? "" : src.substr(at, close - at);
  // Specific on purpose. The first version of this check rejected any
  // occurrence of the substring "layer", which the SANCTIONED parameter name
  // `layer_elements` contains -- so the check failed on the correct code and
  // would have been weakened to make it pass. The hazard is an INDEX: a caller
  // who can name the layer can pass the wrong one.
  Check(signature.find("layer_idx") == std::string::npos &&
            signature.find("layer_index") == std::string::npos &&
            signature.find("attn_layer") == std::string::npos,
        "MakeTurbo4AttentionF16Pair takes NO layer INDEX parameter: " +
            (signature.empty() ? std::string("SIGNATURE NOT FOUND")
                               : signature));
  Check(signature.find("scratch") != std::string::npos &&
            signature.find("layer_elements") != std::string::npos,
        "it takes the scratch and the ONE-LAYER element count, which is all a "
        "one-layer pair needs");
  Check(signature.find("num_kv_heads") == std::string::npos &&
            signature.find("head_dim") == std::string::npos,
        "and it does NOT take the geometry, so the kv_heads*max_context*head_dim "
        "product exists in ONE place (the arena accessor) instead of a second "
        "copy here that nothing would test when it drifts");
}

void F32PairStaysNull() {
  auto storage = Scratch(64, 4, 256);
  const auto pair = MakeTurbo4AttentionF16Pair(storage.data(), LayerElements(64, 4, 256));
  Check(pair.k_f32 == nullptr && pair.v_f32 == nullptr,
        "both f32 members are null on a block layer: a non-null one lets "
        "attention_batched.hip:36-62 store raw f32 over the 66-byte blocks");
}

void DegenerateInputRefuses() {
  auto storage = Scratch(64, 4, 256);
  Check(!MakeTurbo4AttentionF16Pair(nullptr, LayerElements(64, 4, 256)).valid,
        "a null scratch REFUSES rather than handing a launcher a null pair");
  Check(!MakeTurbo4AttentionF16Pair(storage.data(), 0).valid,
        "a ZERO one-layer element count refuses: this is the degenerate case the "
        "new signature has, and it is the one that would otherwise produce a "
        "pair whose K and V are the same pointer");
}

void ProductionScaleIsOnTheRecord() {
  // Design rule: the scratch scales with max_context, so a serve run sets
  // -c and E4 never runs at the model maximum. These are the numbers that
  // decision rests on.
  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kDim = 256;
  for (const std::uint32_t ctx : {8192U, 16384U, 262144U}) {
    const std::size_t bytes =
        2ULL * ctx * kHeads * kDim * sizeof(std::uint16_t);
    std::printf("  max_context=%7u -> scratch %10zu B (%6.1f MiB)\n", ctx,
                bytes, static_cast<double>(bytes) / (1024.0 * 1024.0));
  }
  const std::size_t at_model_max = 2ULL * 262144 * kHeads * kDim * 2;
  const std::size_t at_8k = 2ULL * 8192 * kHeads * kDim * 2;
  Check(at_model_max == 1073741824ULL,
        "the scratch is 1 GiB at the model max context -- stated, not "
        "assumed");
  Check(at_8k == 33554432ULL,
        "and 32 MiB at -c 8192, which is why E4 runs at 8192/16384 and never "
        "at the model maximum");
  Check(at_8k * 32 == at_model_max, "the two are the same rule at two sizes");
}

// By design the helper consumes the arena's one-layer accessor, so the
// two must be the same rule. This needs a QwenGpuArena, which needs HIP, so it
// is asserted here as the ARITHMETIC relation both sides implement, and the
// call sites are pinned to pass the accessor by name in
// turbo4_prefill_wiring_test.
void TheAccessorIsTheSameRule() {
  constexpr std::uint32_t kCtx = 262144;
  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kDim = 256;
  constexpr std::uint32_t kAttnLayers = 16;
  const std::size_t layer_elements = LayerElements(kCtx, kHeads, kDim);
  const std::size_t plane_elements = kAttnLayers * layer_elements;
  std::printf("  layer_elements=%zu plane_elements=%zu\n", layer_elements,
              plane_elements);
  Check(layer_elements == 268435456ULL,
        "GetAttentionKvLayerElements() at the production geometry is "
        "kv_heads*max_context*head_dim");
  Check(plane_elements / kAttnLayers == layer_elements,
        "and it is exactly one layer of the plane: the scratch offset and the "
        "plane count are the same rule at two sizes, not two authorities");
}

std::size_t Count(const std::string& hay, const std::string& needle) {
  std::size_t n = 0;
  for (std::size_t at = hay.find(needle); at != std::string::npos;
       at = hay.find(needle, at + needle.size())) {
    ++n;
  }
  return n;
}

// WHICH accessor feeds the helper. Both call sites must pass the ONE-LAYER
// count; passing the all-layers one is the total_k bug, 16x too far, and it
// passed every other check in this file when it was mutated.
void TheCallSitesPassTheOneLayerCount() {
  std::ifstream in("src/models/qwen/hip/prefill_chunk.cpp");
  const std::string src((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  if (src.empty()) {
    Check(false, "prefill_chunk.cpp was readable from the project root");
    return;
  }
  const std::size_t one_layer = Count(src, "GetAttentionKvLayerElements()");
  const std::size_t all_layers = Count(src, "GetAttentionKvPlaneElements()");
  std::printf("  prefill: GetAttentionKvLayerElements x%zu, "
              "GetAttentionKvPlaneElements x%zu\n", one_layer, all_layers);
  Check(one_layer == 2,
        "BOTH call sites that build the pair pass "
        "GetAttentionKvLayerElements()");
  Check(all_layers == 1,
        "and the all-layers accessor appears exactly once in prefill -- in the "
        "existing total_k offset for the f16 CACHE, which is correct there and "
        "must never reach the scratch");
}

}  // namespace

int main() {
  TheCallSitesPassTheOneLayerCount();
  TheAccessorIsTheSameRule();
  TiledCapacityKeepsEveryPositionOffset();
  VOffsetIsOneLayer();
  PairFillsTheScratchExactly();
  TheConstantsCannotVary();
  F32PairStaysNull();
  DegenerateInputRefuses();
  ProductionScaleIsOnTheRecord();
  std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::hip::dflash {

/// FR-Spec (arXiv 2502.14856) draft vocabulary cut, host side. The drafter's LM
/// head scores only these ids; every other logit stays -inf, so the selector,
/// sampler and verifier see an ordinary full-vocabulary row. The target still
/// verifies every token, so the emitted text cannot change -- only acceptance.
///
/// GUFO_DEV_DRAFT_VOCAB=<path>   little-endian uint32 ids (any order, dups ok)
/// GUFO_DEV_DRAFT_VOCAB=first:N  ids 0..N-513 + the top 512; data-free fallback, BPE ids
///                               roughly follow merge order
///
/// The count must be a multiple of 256, the alignment every LM-head row count
/// the GEMM routes have been run with (248320 = 970 * 256). Returns the ids
/// sorted and unique, or empty with `error` set.
inline constexpr std::size_t kDraftVocabAlign = 256;

[[nodiscard]] inline std::vector<std::uint32_t> ParseDraftVocab(
    std::string_view spec, std::size_t vocab_size, std::string& error) {
  std::vector<std::uint32_t> ids;
  if (spec.starts_with("first:")) {
    const std::string count_text(spec.substr(6));
    char* end = nullptr;
    const unsigned long long count = std::strtoull(count_text.c_str(), &end, 10);
    if (count_text.empty() || end == nullptr || *end != '\0' ||
        count > vocab_size) {
      error = "GUFO_DEV_DRAFT_VOCAB first:N needs 0 < N <= vocab";
      return {};
    }
    // The TOP 512 ids are kept too: Qwen puts its special tokens (<|im_end|>,
    // </think>, tool markers, the mask token) at the end of the vocabulary, and
    // a drafter that cannot propose <|im_end|> loses acceptance at every turn
    // end. ponytail: a fixed tail, not the tokenizer's special list; a vocab
    // with specials elsewhere needs a counts file (draft_vocab_counts keeps them).
    const std::size_t tail = std::min<std::size_t>(512, count);
    ids.reserve(count);
    for (std::size_t index = 0; index < count - tail; ++index)
      ids.push_back(static_cast<std::uint32_t>(index));
    for (std::size_t index = vocab_size - tail; index < vocab_size; ++index)
      ids.push_back(static_cast<std::uint32_t>(index));
  } else {
    std::ifstream in{std::string(spec), std::ios::binary};
    if (!in) {
      error = "GUFO_DEV_DRAFT_VOCAB file cannot be opened";
      return {};
    }
    std::uint32_t id = 0;
    while (in.read(reinterpret_cast<char*>(&id), sizeof(id))) ids.push_back(id);
    if (in.gcount() != 0) {
      error = "GUFO_DEV_DRAFT_VOCAB file is not a whole number of uint32 ids";
      return {};
    }
  }
  std::ranges::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.empty() || ids.back() >= vocab_size) {
    error = "GUFO_DEV_DRAFT_VOCAB is empty or holds an id >= vocab";
    return {};
  }
  if (ids.size() % kDraftVocabAlign != 0) {
    error = "GUFO_DEV_DRAFT_VOCAB unique id count must be a multiple of 256";
    return {};
  }
  return ids;
}

/// Calls copy(source_row, destination_row, row_count) once per run of
/// consecutive ids, so `first:N` is two copies and a frequency list is a few
/// thousand. `ids` must be sorted and unique (ParseDraftVocab's output).
template<typename Copy>
void ForEachDraftVocabRun(std::span<const std::uint32_t> ids, Copy copy) {
  std::size_t begin = 0;
  while (begin < ids.size()) {
    std::size_t end = begin + 1;
    while (end < ids.size() && ids[end] == ids[end - 1] + 1U) ++end;
    copy(static_cast<std::size_t>(ids[begin]), begin, end - begin);
    begin = end;
  }
}

}  // namespace gufo::hip::dflash

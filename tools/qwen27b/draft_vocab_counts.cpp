// FR-Spec frequency list for GUFO_DEV_DRAFT_VOCAB (draft_vocab.hpp).
//
//   draft_vocab_counts <target.gguf> <N> <out.bin> <text file>...
//
// Tokenizes the text files with the target's OWN tokenizer (gufo's
// QwenTokenizer from the GGUF metadata, so the ids are exactly the engine's)
// and writes the N most frequent ids as little-endian uint32. Ties and unseen
// ids fall back to id order (BPE ids roughly follow merge order). Every
// special token (<|im_end|>, </think>, tool-call markers) is kept first
// whatever its count: corpus text never contains them, and a drafter that
// cannot propose <|im_end|> loses acceptance at every turn end. Prints the
// share of corpus tokens the list covers: a draft whose true next token lies
// outside the list cannot be accepted, so coverage bounds acceptance.
//
// Only GGUF metadata is read; the tensor pages stay unmapped-in. Text is
// encoded in ~1 MiB slices cut at newlines, so memory stays flat.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <target.gguf> <N> <out.bin> <text file>...\n",
                 argv[0]);
    return 2;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "gguf: %s\n", error.c_str());
    return 1;
  }
  const auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    std::fprintf(stderr, "tokenizer: %s\n", error.c_str());
    return 1;
  }
  const std::size_t vocab = tokenizer->GetVocabSize();
  const std::size_t keep = std::strtoull(argv[2], nullptr, 10);
  if (keep == 0 || keep > vocab) {
    std::fprintf(stderr, "N must be in 1..%zu\n", vocab);
    return 2;
  }
  if (keep % 256 != 0)
    std::fprintf(stderr, "warning: N=%zu is not a multiple of 256; "
                         "GUFO_DEV_DRAFT_VOCAB will refuse it\n", keep);

  std::vector<std::uint64_t> counts(vocab, 0);
  gufo::tokenization::TokenizerOptions options;
  options.parse_special_tokens = false;  // corpus text, not chat markup
  constexpr std::size_t kSlice = 1U << 20;
  for (int arg = 4; arg < argc; ++arg) {
    std::ifstream in(argv[arg], std::ios::binary);
    if (!in) {
      std::fprintf(stderr, "skip unreadable %s\n", argv[arg]);
      continue;
    }
    std::string carry, buffer(kSlice, '\0');
    const auto encode = [&](std::size_t len) {
      for (const auto id :
           tokenizer->Encode(std::string_view(carry).substr(0, len), options))
        if (id < vocab) ++counts[id];
      carry.erase(0, len);
    };
    while (in.read(buffer.data(), kSlice) || in.gcount() > 0) {
      carry.append(buffer.data(), static_cast<std::size_t>(in.gcount()));
      const std::size_t cut = carry.rfind('\n');
      if (cut != std::string::npos) encode(cut + 1);
      // ponytail: a line longer than 8 MiB is split mid-line (one BPE merge
      // lost at the cut) so memory stays bounded.
      else if (carry.size() > 8 * kSlice) encode(carry.size());
    }
    if (!carry.empty()) encode(carry.size());  // last line without a newline
  }

  std::vector<std::uint32_t> order(vocab);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
    return counts[a] > counts[b];
  });
  std::vector<std::uint32_t> ids;
  std::vector<bool> taken(vocab, false);
  for (const auto& [text, id] : tokenizer->SpecialTokens())
    if (id < vocab && !taken[id] && ids.size() < keep) { taken[id] = true; ids.push_back(id); }
  const std::size_t specials = ids.size();
  for (const auto id : order) {
    if (ids.size() == keep) break;
    if (!taken[id]) { taken[id] = true; ids.push_back(id); }
  }
  const std::uint64_t total = std::accumulate(counts.begin(), counts.end(), 0ULL);
  std::uint64_t covered = 0;
  for (const auto id : ids) covered += counts[id];

  std::ofstream out(argv[3], std::ios::binary);
  out.write(reinterpret_cast<const char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(std::uint32_t)));
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  std::printf("tokens=%llu kept=%zu of %zu (specials %zu) coverage=%.4f out=%s\n",
              static_cast<unsigned long long>(total), keep, vocab, specials,
              total == 0 ? 0.0 : static_cast<double>(covered) / total, argv[3]);
  return 0;
}

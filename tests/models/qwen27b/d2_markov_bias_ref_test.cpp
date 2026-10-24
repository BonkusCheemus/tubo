/// The HOST Markov reference, checked against the GPU dump.
///
/// Requirement: "the reference is a HOST function in the test that recomputes
/// the Markov bias from the GGUF's dequantized markov_w1/w2 for the dumped prev
/// tokens, exactly as dflash.cpp does it (cite lines)."
///
/// This IS that function's caller. It does not reimplement the bias: it calls the
/// shipped `ComputeMarkovBiasRow` (dflash_weights.cpp:890-923), on weights loaded
/// by the shipped loader, so the dequantisation is the engine's own. A Python
/// reimplementation was tried first and was WRONG in a way worth recording: the
/// markov tensors are stored Q4_K (ggml type 12), not BF16, and reinterpreting
/// their bytes as BF16 produced all-NaN. The loader converts on bind, which is
/// why the executor's `type == kBF16` check (dflash/executor.cpp:1180) passes
/// while the file on disk says 12. Hand-rolling a dequantiser would have been the
/// second way to get this wrong.
///
/// WHAT IS COMPARED. The dump holds the POST-Markov logit row (output GEMM plus
/// bias). The reference produces the BIAS alone. The only quantity both sides can
/// produce is the difference between the post-Markov row and the base row the GEMM
/// wrote first — which is why the engine writes both files.
///
/// THE PREV-TOKEN CHAIN, from dflash/executor.cpp:1192 and :1211-1213:
/// d_markov_prev_ starts as the anchor token and is replaced after every step by
/// the token drafted AT that step. So position p is biased by proposed_ids[p-1],
/// and position 0 by the anchor.
///
/// BOUND: |diff| <= 2^-7 * max(|ref|, 1), plus argmax(base + ref) == argmax(post).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/state.hpp"

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

constexpr const char* kDSparkPath =
    "models/"
    "RedHat-DSpark-Q4_K_M.gguf";

/// <path>.base.bin / .bias.bin: uint32 rows, uint32 vocab, then rows*vocab f32.
struct Dump {
  std::uint32_t rows{0};
  std::uint32_t vocab{0};
  std::vector<float> data;
};

bool ReadDump(const fs::path& path, Dump& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  in.read(reinterpret_cast<char*>(&out.rows), sizeof(out.rows));
  in.read(reinterpret_cast<char*>(&out.vocab), sizeof(out.vocab));
  const std::size_t n = static_cast<std::size_t>(out.rows) * out.vocab;
  out.data.resize(n);
  in.read(reinterpret_cast<char*>(out.data.data()),
          static_cast<std::streamsize>(n * sizeof(float)));
  return static_cast<bool>(in);
}

/// Pull one integer field out of the dump's first JSONL line without a JSON
/// dependency: the line is written by the engine, so its shape is fixed, but a
/// missing field must be a loud failure rather than a silent default.
std::size_t JsonInt(const std::string& line, const std::string& key,
                    std::size_t fallback) {
  const std::string needle = "\"" + key + "\":";
  const auto at = line.find(needle);
  if (at == std::string::npos) return fallback;
  std::size_t i = at + needle.size();
  bool neg = false;
  if (i < line.size() && line[i] == '-') {
    neg = true;
    ++i;
  }
  std::size_t value = 0;
  bool any = false;
  while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i]))) {
    value = value * 10U + static_cast<std::size_t>(line[i] - '0');
    ++i;
    any = true;
  }
  if (!any) return fallback;
  return neg ? static_cast<std::size_t>(-static_cast<long>(value)) : value;
}

/// The proposed ids, in order, from the first dump line.
std::vector<std::uint32_t> JsonIds(const std::string& line) {
  std::vector<std::uint32_t> ids;
  const auto at = line.find("\"proposed_ids\":[");
  if (at == std::string::npos) return ids;
  std::size_t i = at + std::string("\"proposed_ids\":[").size();
  while (i < line.size() && line[i] != ']') {
    if (std::isdigit(static_cast<unsigned char>(line[i]))) {
      std::size_t v = 0;
      while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i]))) {
        v = v * 10U + static_cast<std::size_t>(line[i] - '0');
        ++i;
      }
      ids.push_back(static_cast<std::uint32_t>(v));
    } else {
      ++i;
    }
  }
  return ids;
}

std::string EnvOr(const char* name, const char* fallback) {
  const char* const v = std::getenv(name);
  return (v != nullptr && v[0] != '\0') ? std::string(v) : std::string(fallback);
}

/// The OLD element-wise formula, written out here so
/// it is the ORACLE for the new block-cached one. It uses only
/// `TensorRef::Get` — the host path — and shares no code with the HIP draft
/// kernel, so agreeing with it is not the fast path agreeing with itself.
/// Bit-exact comparison is the point: the speed-up must change the work, not
/// the numbers.
std::vector<float> ElementWiseOracle(
    const gufo::speculative::QwenDFlashWeights& w,
                                     gufo::tokenization::TokenId previous,
                                     std::size_t slice) {
  const std::size_t rank = w.dflash_config.markov_rank;
  const std::size_t w1_offset = static_cast<std::size_t>(previous) * rank;
  std::vector<float> out(slice);
  for (std::size_t candidate = 0; candidate < slice; ++candidate) {
    const std::size_t w2_offset = candidate * rank;
    float partial = 0.0F;
    for (std::size_t component = 0; component < rank; ++component) {
      partial += w.markov_w2.Get(w2_offset + component) *
                 w.markov_w1.Get(w1_offset + component);
    }
    out[candidate] = partial;
  }
  return out;
}

void CompareAgainstHostReference(const fs::path& dspark_path,
                                 const fs::path& run_dir) {
  Dump post;
  Dump base;
  const fs::path bias_path = run_dir / "draft_dump.jsonl.bias.bin";
  const fs::path base_path = run_dir / "draft_dump.jsonl.base.bin";
  Check(ReadDump(bias_path, post), "read the post-Markov dump");
  Check(ReadDump(base_path, base), "read the base dump");
  if (post.rows == 0 || post.data.empty()) return;
  Check(post.rows == base.rows && post.vocab == base.vocab,
        "the two dumps describe the same block");
  if (post.rows != base.rows || post.vocab != base.vocab) return;

  std::ifstream jsonl(run_dir / "draft_dump.jsonl");
  std::string first;
  std::getline(jsonl, first);
  const std::vector<std::uint32_t> ids = JsonIds(first);
  Check(ids.size() == post.rows,
        "the JSONL block and the dumps agree on length (" +
            std::to_string(ids.size()) + " vs " + std::to_string(post.rows) +
            ")");
  if (ids.size() != post.rows) return;

  const std::uint32_t anchor =
      static_cast<std::uint32_t>(JsonInt(first, "anchor_token", 0));

  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(dspark_path, &error);
  Check(reader != nullptr, "reader opens the DSpark artifact: " + error);
  if (reader == nullptr) return;
  auto weights =
      gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error);
  Check(weights.has_value(), "DSpark weights load: " + error);
  if (!weights.has_value()) return;

  const std::size_t vocab = post.vocab;

  // Bit-exactness of the fast path against the element-wise oracle, on the
  // REAL drafter, for the dump's anchor, over a 4096-candidate slice.
  {
    const std::size_t kSlice = 4096;
    const std::uint32_t prev = anchor;
    const auto ref_opt =
        gufo::speculative::ComputeMarkovBiasRow(*weights, prev);
    if (!ref_opt.has_value() || ref_opt->size() < kSlice) {
      Check(false, "fast path returned a row too small for the bit-exact check");
    } else {
      const std::vector<float> oracle =
          ElementWiseOracle(*weights, prev, kSlice);
      std::size_t mismatches = 0;
      std::size_t first_bad = 0;
      for (std::size_t i = 0; i < kSlice; ++i) {
        // Compare the bit patterns, not the values: +0.0/-0.0 and any NaN
        // payload must be caught, which == would let through.
        std::uint32_t a, b;
        std::memcpy(&a, &(*ref_opt)[i], sizeof(a));
        std::memcpy(&b, &oracle[i], sizeof(b));
        if (a != b) {
          if (mismatches == 0) first_bad = i;
          ++mismatches;
        }
      }
      Check(mismatches == 0,
            "fast path is BIT-EXACT vs the element-wise oracle over " +
                std::to_string(kSlice) + " candidates (prev=" +
                std::to_string(prev) + ", " + std::to_string(mismatches) +
                " mismatches, first at " + std::to_string(first_bad) + ")");
    }
  }

  // RESTATED BOUND. The earlier 2^-7 form was too tight
  // for the chosen storage: BF16 w1 AND w2, so two roundings over a rank-256
  // sum. The bound must scale with the magnitude actually accumulated, which
  // is sum_r |w1_r * w2_r| per candidate, not with |ref| alone.
  //
  // Computing that sum through TensorRef::Get would repeat the original defect
  // (63.6M Gets), so w2 is dequantised once here with the same canonical
  // DequantizeQ4_K, block by block. The bound is a property of the weights, not
  // of the path that reads them.
  const std::size_t rank = weights->dflash_config.markov_rank;
  std::vector<float> w1_row(rank);
  for (std::size_t c = 0; c < rank; ++c) {
    w1_row[c] = weights->markov_w1.Get(
        static_cast<std::size_t>(anchor) * rank + c);
  }
  std::vector<float> w2_all(vocab * rank);
  {
    constexpr std::size_t kBlk = 256;
    const auto* bytes =
        static_cast<const std::uint8_t*>(weights->markov_w2.data);
    for (std::size_t b = 0; b * kBlk < w2_all.size(); ++b) {
      gufo::quant::DequantizeQ4_K(
          bytes + b * sizeof(gufo::quant::block_q4_K), w2_all.data() + b * kBlk,
          kBlk);
    }
  }
  std::vector<double> absum(vocab);
  for (std::size_t cand = 0; cand < vocab; ++cand) {
    double s = 0.0;
    const float* w2r = w2_all.data() + cand * rank;
    for (std::size_t c = 0; c < rank; ++c) {
      s += std::abs(static_cast<double>(w1_row[c]) * w2r[c]);
    }
    absum[cand] = s;
  }

  double worst_excess = -1e30;
  std::size_t worst_pos = 0;
  std::size_t exceed_count = 0;
  double max_ratio = 0.0;
  bool argmax_ok = true;

  for (std::size_t p = 0; p < post.rows; ++p) {
    // Chain: position 0 is biased by the anchor, position p>0 by ids[p-1]
    // (executor.cpp:1192 seeds d_markov_prev_ from d_out_token_, and :1211
    // replaces it with the token drafted at the step just finished).
    const std::uint32_t prev = (p == 0) ? anchor : ids[p - 1];
    const auto ref_opt =
        gufo::speculative::ComputeMarkovBiasRow(*weights, prev);
    Check(ref_opt.has_value(),
          "position " + std::to_string(p) + ": host reference for prev=" +
              std::to_string(prev));
    if (!ref_opt.has_value()) {
      argmax_ok = false;
      continue;
    }
    const std::vector<float>& ref = *ref_opt;
    Check(ref.size() == vocab,
          "position " + std::to_string(p) + ": reference is vocab-wide");

    const float* post_row = post.data.data() + p * vocab;
    const float* base_row = base.data.data() + p * vocab;
    double pos_worst = -1e30;
    std::size_t argmax_post = 0;
    std::size_t argmax_ref = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      // bias_gpu = post - base, since post is base + bias
      const double got = static_cast<double>(post_row[i]) -
                         static_cast<double>(base_row[i]);
      const double diff = std::abs(got - static_cast<double>(ref[i]));
      const double bound = std::pow(2.0, -8.0) * absum[i] * 2.0 + 1e-6;
      const double excess = diff - bound;
      if (excess > 0.0) ++exceed_count;
      const double ratio = diff / bound;
      if (ratio > max_ratio) max_ratio = ratio;
      if (excess > pos_worst) pos_worst = excess;
      if (post_row[i] > post_row[argmax_post]) argmax_post = i;
      const double recombined = static_cast<double>(base_row[i]) +
                                static_cast<double>(ref[i]);
      if (recombined > static_cast<double>(base_row[argmax_ref]) +
                           static_cast<double>(ref[argmax_ref])) {
        argmax_ref = i;
      }
    }
    if (pos_worst > worst_excess) {
      worst_excess = pos_worst;
      worst_pos = p;
    }
    const bool same = argmax_post == argmax_ref;
    argmax_ok = argmax_ok && same;
    std::printf("  pos %zu prev=%u max_excess=%+.6f argmax post=%zu ref=%zu %s\n",
                p, prev, pos_worst, argmax_post, argmax_ref,
                same ? "OK" : "MISMATCH");
  }

  Check(exceed_count == 0,
        "every element is within 2^-8*sum|w1*w2|*2 + 1e-6 (" +
            std::to_string(exceed_count) + " exceed, max ratio " +
            std::to_string(max_ratio) + ", worst excess " +
            std::to_string(worst_excess) + " at pos " +
            std::to_string(worst_pos) + ")");
  std::printf("  exceed=%zu max_ratio=%.6f\n", exceed_count, max_ratio);
  Check(argmax_ok,
        "argmax(base + host bias) equals argmax(post) on every position");
}

}  // namespace

int main() {
  const fs::path run_dir(EnvOr("GUFO_D2_DUMP_DIR", ""));
  if (run_dir.empty()) {
    std::printf("SKIP: set GUFO_D2_DUMP_DIR to a d2_32step_* run directory\n");
    return 0;
  }
  const fs::path dspark_path(EnvOr("GUFO_DSPARK_GGUF", kDSparkPath));
  if (!fs::exists(dspark_path)) {
    std::printf("SKIP: real DSpark artifact not found: %s\n",
                dspark_path.c_str());
    return 0;
  }
  CompareAgainstHostReference(dspark_path, run_dir);
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;
constexpr std::array<std::uint32_t, 5> kContexts{1, 4095, 4096, 4097, 8192};
constexpr std::size_t kMaxK = 8;
constexpr const char* kSeedText =
    "A deterministic verification oracle compares the same token history "
    "across sequential decoding and causal speculative verification.";

void Require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}

std::vector<float> CopyLogits(Executor& executor) {
  const auto logits = executor.CopyLastLogits();
  Require(!logits.empty() &&
              std::ranges::all_of(logits, [](float v) { return std::isfinite(v); }),
          "decode logits are empty or non-finite");
  return {logits.begin(), logits.end()};
}

std::vector<float> CopyVerification(Executor& executor, std::size_t row) {
  const auto logits = executor.CopyVerificationLogits(row);
  Require(!logits.empty() &&
              std::ranges::all_of(logits, [](float v) { return std::isfinite(v); }),
          "verification logits are empty or non-finite");
  return {logits.begin(), logits.end()};
}

std::size_t Argmax(std::span<const float> logits) {
  return static_cast<std::size_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

double MaxAbsDiff(std::span<const float> a, std::span<const float> b) {
  Require(a.size() == b.size(), "logit vector sizes differ");
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    result = std::max(result, std::abs(static_cast<double>(a[i]) - b[i]));
  return result;
}

int Run(const char* model_path) {
  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(model_path, &error);
  Require(owner != nullptr, "open model: " + error);
  auto reader = std::shared_ptr<const gufo::core::GgufReader>(std::move(owner));
  auto executor = Executor::CreateFromGguf(reader, &error, 8200);
  Require(executor != nullptr, "load model: " + error);
  Require(gufo::hip::QwenKvIsBlockAddressed(
              executor->GetExecutionPolicy().kv_cache_storage),
          "runtime policy did not select Turbo4 KV");
  std::cout << "model=" << model_path << " max_context=8200"
            << " kv_storage=Turbo4\n";

  const auto seed = executor->GetTokenizer().Encode(kSeedText);
  Require(!seed.empty(), "seed text tokenized to empty");
  auto token_at = [&seed](std::size_t index) { return seed[index % seed.size()]; };
  std::vector<double> spread(kContexts.size(), -1.0);

  // Gate 2a: measure same-input decode-vs-decode spread BEFORE verification.
  for (std::size_t ci = 0; ci < kContexts.size(); ++ci) {
    const std::uint32_t context = kContexts[ci];
    std::vector<Token> prefix(context);
    for (std::size_t i = 0; i < prefix.size(); ++i) prefix[i] = token_at(i);
    std::array<std::vector<std::vector<float>>, 2> passes;
    for (std::size_t pass = 0; pass < passes.size(); ++pass) {
      executor->Reset();
      (void)executor->ForwardPromptBatch(prefix, 0, false);
      for (std::size_t row = 0; row < kMaxK; ++row) {
        (void)executor->ForwardToken(token_at(context + row), context + row, true);
        passes[pass].push_back(CopyLogits(*executor));
      }
      executor->Reset();
    }
    spread[ci] = 0.0;
    for (std::size_t row = 0; row < kMaxK; ++row)
      spread[ci] = std::max(spread[ci],
                            MaxAbsDiff(passes[0][row], passes[1][row]));
    std::cout << "DECODE_SPREAD context=" << context
              << " max_abs=" << spread[ci] << " compared_rows=" << kMaxK
              << " argmax_equal=";
    bool same_argmax = true;
    for (std::size_t row = 0; row < kMaxK; ++row)
      same_argmax &= Argmax(passes[0][row]) == Argmax(passes[1][row]);
    std::cout << same_argmax << '\n';
  }

  // Gate 2b: sequential Turbo4 decode is the oracle for each verify chunk.
  bool pass = true;
  for (std::size_t ci = 0; ci < kContexts.size(); ++ci) {
    const std::uint32_t context = kContexts[ci];
    std::vector<Token> prefix(context);
    for (std::size_t i = 0; i < prefix.size(); ++i) prefix[i] = token_at(i);
    for (std::size_t k = 1; k <= kMaxK; ++k) {
      executor->Reset();
      (void)executor->ForwardPromptBatch(prefix, 0, false);
      std::vector<std::vector<float>> sequential_logits;
      std::vector<Token> sequential_argmax;
      for (std::size_t row = 0; row < k; ++row) {
        sequential_argmax.push_back(
            executor->ForwardToken(token_at(context + row), context + row, true));
        sequential_logits.push_back(CopyLogits(*executor));
      }
      // The verify inputs are the same teacher-forced candidates as sequential decode.
      std::vector<Token> candidates(k);
      for (std::size_t row = 0; row < k; ++row)
        candidates[row] = token_at(context + row);
      executor->Reset();
      (void)executor->ForwardPromptBatch(prefix, 0, false);
      const auto verified = executor->ForwardVerificationChunk(candidates, context, true);
      Require(verified.size() == k, "verification returned the wrong row count");
      double max_abs = 0.0;
      std::size_t argmax_mismatch = 0;
      std::size_t prediction_mismatch = 0;
      for (std::size_t row = 0; row < k; ++row) {
        const auto logits = CopyVerification(*executor, row);
        max_abs = std::max(max_abs,
                           MaxAbsDiff(sequential_logits[row], logits));
        argmax_mismatch += Argmax(sequential_logits[row]) != Argmax(logits);
        prediction_mismatch += sequential_argmax[row] != verified[row];
      }
      const bool within_spread = max_abs <= spread[ci];
      const bool row_pass = within_spread && argmax_mismatch == 0 &&
                            prediction_mismatch == 0;
      pass &= row_pass;
      std::cout << "VERIFY context=" << context << " k=" << k
                << " max_abs=" << max_abs << " spread=" << spread[ci]
                << " argmax_mismatch=" << argmax_mismatch
                << " prediction_mismatch=" << prediction_mismatch
                << " pass=" << row_pass << '\n';
      executor->Reset();
    }
  }
  std::cout << "E6_MODEL_ORACLE=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: qwen27b_e6_verify_oracle_test MODEL.gguf\n";
      return 2;
    }
    return Run(argv[1]);
  } catch (const std::exception& e) {
    std::cerr << "E6_MODEL_ORACLE=ERROR " << e.what() << '\n';
    return 1;
  }
}

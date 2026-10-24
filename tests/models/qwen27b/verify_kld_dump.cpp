// Verify-pass KLD dump: writes the logits of the batched verify pass for offline KLD/NLL scoring.
// Modelled on e6_verify_oracle_test.cpp
// (same load / prefill / ForwardVerificationChunk calls) and speculative_verifier.cpp:648,723 (served call order).
//
// WHAT IT DOES. Teacher-forced next-token logits through the batched VERIFY pass, the only pass that can reach
// LaunchTurbo4AttentionVerifyWmma (batched_decode.cpp:823-865: sequence_size > 1, <= 8). Per (text, context) cell:
//   prefill tokens[0, ctx) in 2048-token chunks (no logits), then CHUNKS rounds of
//   SaveState(pos, true) -> ForwardVerificationChunk(tokens[pos, pos+8), pos, capture_logits=true) -> FinishVerification(true)
//   = the served all-accepted round. Row i of a round predicts tokens[pos+i+1]. The input tokens are the TEXT, never the
//   model's own output, so every arm sees identical inputs (teacher forcing). Rows are dumped as raw f32 (vocab per row).
// The reader is chosen by the PROCESS ENVIRONMENT (the switches are cached once per process): run once per arm, then
// compare with kld_compare.py. Arms: see q285_kld.sh.
//
// usage: verify_kld_dump MODEL.gguf OUTDIR TAG label=textfile[,label=textfile] CTX[,CTX] CHUNKS
//   writes OUTDIR/TAG_<label>_<ctx>.f32  (CHUNKS*8 rows x vocab, native f32)
//          OUTDIR/TAG_<label>_<ctx>.tgt  (CHUNKS*8 u32: the true next token of each row, for NLL)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/iq3_layout.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;
// served DFlash2 draft 7 + current token; GUFO_KLD_ROWS=7 measures the draft-6 verify width
static const std::size_t kRows = [] { const char* v = std::getenv("GUFO_KLD_ROWS"); return v ? std::size_t(std::strtoul(v, nullptr, 10)) : std::size_t{8}; }();
constexpr std::uint32_t kPrefillChunk = 2048;  // served prefill chunk

void Require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}

std::vector<std::string> Split(const std::string& text, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  for (std::string part; std::getline(ss, part, sep);)
    if (!part.empty()) out.push_back(part);
  return out;
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  Require(in.good(), "cannot read " + path);
  return {std::istreambuf_iterator<char>(in), {}};
}

int Run(int argc, char** argv) {
  const std::string model_path = argv[1], outdir = argv[2], tag = argv[3];
  const auto texts = Split(argv[4], ',');
  std::vector<std::uint32_t> contexts;
  for (const auto& c : Split(argv[5], ',')) contexts.push_back(static_cast<std::uint32_t>(std::stoul(c)));
  const std::size_t chunks = std::stoul(argv[6]);
  Require(!texts.empty() && !contexts.empty() && chunks > 0, "empty texts/contexts/chunks");
  const std::uint32_t max_ctx = *std::max_element(contexts.begin(), contexts.end()) +
                                static_cast<std::uint32_t>(kRows * chunks) + 8;

  std::string error;
  auto owner = gufo::core::GgufReader::OpenFile(model_path, &error);
  Require(owner != nullptr, "open model: " + error);
  auto executor = Executor::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(owner)), &error, max_ctx);
  Require(executor != nullptr, "load model: " + error);
  const std::size_t vocab = executor->GetConfig().vocab_size;
  // Proof of which KV format this process really runs (fp16 arm X must say 1; turbo6 arms must say 4).
  std::cout << "tag=" << tag << " max_context=" << max_ctx << " vocab=" << vocab << " kv_storage_enum="
            << static_cast<int>(executor->GetExecutionPolicy().kv_cache_storage)
            << " block_addressed="
            << gufo::hip::QwenKvIsBlockAddressed(executor->GetExecutionPolicy().kv_cache_storage) << std::endl;
  // Proof line: how many FFN gate/up tensors this process really serves through the SN IQ3 layouts (0 = native path).
  {
    std::size_t tags = 0;
    for (const auto& l : executor->GetModel().GetWeights().layers)
      tags += gufo::models::IsIq3Layout(l.ffn_gate.layout) + gufo::models::IsIq3Layout(l.ffn_up.layout);
    std::cout << "sn_iq3_tags=" << tags << std::endl;
  }

  for (const auto& spec : texts) {
    const auto eq = spec.find('=');
    Require(eq != std::string::npos, "text spec must be label=path: " + spec);
    const std::string label = spec.substr(0, eq);
    const auto tokens = executor->GetTokenizer().Encode(ReadFile(spec.substr(eq + 1)));
    for (const std::uint32_t ctx : contexts) {
      const std::size_t need = ctx + kRows * chunks + 1;  // +1: the last row's target token
      Require(tokens.size() >= need, label + ": text has " + std::to_string(tokens.size()) + " tokens, need " +
                                         std::to_string(need) + " (use a longer file or a smaller CTX)");
      const auto t0 = std::chrono::steady_clock::now();
      executor->Reset();
      for (std::uint32_t off = 0; off < ctx; off += kPrefillChunk) {
        const std::uint32_t n = std::min(kPrefillChunk, ctx - off);
        (void)executor->ForwardPromptBatch(std::span(tokens).subspan(off, n), off, false);
      }
      const auto t1 = std::chrono::steady_clock::now();
      const std::string stem = outdir + "/" + tag + "_" + label + "_" + std::to_string(ctx);
      std::ofstream rows_out(stem + ".f32", std::ios::binary | std::ios::trunc);
      std::ofstream tgt_out(stem + ".tgt", std::ios::binary | std::ios::trunc);
      Require(rows_out.good() && tgt_out.good(), "cannot write under " + outdir);
      for (std::size_t c = 0; c < chunks; ++c) {
        const std::uint32_t pos = ctx + static_cast<std::uint32_t>(kRows * c);
        const auto chunk = std::span(tokens).subspan(pos, kRows);
        executor->SaveState(pos, /*allow_divert=*/true);
        const auto predictions = executor->ForwardVerificationChunk(chunk, pos, /*capture_logits=*/true);
        Require(predictions.size() == kRows, "verify returned the wrong row count");
        for (std::size_t r = 0; r < kRows; ++r) {
          const auto logits = executor->CopyVerificationLogits(r);
          Require(logits.size() == vocab, "verification logits row has the wrong width");
          Require(std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); }),
                  "non-finite verification logits at pos " + std::to_string(pos + r));
          rows_out.write(reinterpret_cast<const char*>(logits.data()),
                         static_cast<std::streamsize>(logits.size() * sizeof(float)));
          const std::uint32_t target = static_cast<std::uint32_t>(tokens[pos + r + 1]);
          tgt_out.write(reinterpret_cast<const char*>(&target), sizeof(target));
        }
        executor->FinishVerification(/*commit_state=*/true);  // all rows accepted, as in the served all-accept round
      }
      rows_out.close();
      tgt_out.close();
      Require(!rows_out.fail() && !tgt_out.fail(), "write failed (disk full?) for " + stem);
      const auto t2 = std::chrono::steady_clock::now();
      std::cout << "cell " << label << " ctx=" << ctx << " rows=" << kRows * chunks << " prefill_s="
                << std::chrono::duration<double>(t1 - t0).count()
                << " verify_s=" << std::chrono::duration<double>(t2 - t1).count() << " -> " << stem << ".f32"
                << std::endl;
    }
  }
  std::cout << "VERIFY_KLD_DUMP=DONE" << std::endl;
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      std::cerr << "usage: verify_kld_dump MODEL.gguf OUTDIR TAG label=text[,..] CTX[,..] CHUNKS\n";
      return 77;  // CTest skip: this is a manual tool, not an argument-free test
    }
    return Run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "VERIFY_KLD_DUMP=ERROR " << e.what() << '\n';
    return 1;
  }
}

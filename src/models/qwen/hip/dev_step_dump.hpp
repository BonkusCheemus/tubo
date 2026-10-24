// The GUFO_DEV_STEP_DUMP per-step dump. Host code.
//
// WHY A HEADER AND NOT A .cpp. The measurement this serves has to be exact: one
// JSONL line per step, carrying the argmax, the top 5 (id, logit) pairs and BOTH
// Turbo4 status words. A test that could not call the formatter could only grep
// this project's source for its own strings, which is the "assert the
// implementation" trap. Here the formatter is callable, so a test can assert the
// literal bytes.
//
// WHY IT EXISTS AT ALL: `gufo serve` refuses logprobs (openai_chat.cpp), so no
// existing path hands a caller the per-step logits the parity arms need.
//
// DEFAULT OFF, AND "OFF" IS NOT "EQUIVALENT": with the variable unset
// ReadStepDumpConfig returns enabled == false, and every caller in decode.cpp
// branches on that before touching a buffer, a stream or a file. There is no
// copy, no allocation and no open. That is what keeps the unset case
// byte-identical to C1 and free at run time.
//
// GUFO_DEV_STEP_DUMP is read on every step, never cached, for the same reason as
// the C1 dev switch: a cached read cannot be turned off again inside one
// process, which is how a dev switch outlives its dev window.

#ifndef GUFO_MODELS_QWEN_HIP_DEV_STEP_DUMP_HPP_
#define GUFO_MODELS_QWEN_HIP_DEV_STEP_DUMP_HPP_

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace gufo::hip {

struct DevStepDumpConfig {
  bool enabled{false};
  std::string path;
};

/// Unset, empty, or anything without a value in it: OFF.
[[nodiscard]] inline DevStepDumpConfig ReadDevStepDumpConfig() {
  DevStepDumpConfig config;
  const char* const raw = std::getenv("GUFO_DEV_STEP_DUMP");
  if (raw == nullptr || raw[0] == '\0') return config;
  config.enabled = true;
  config.path = raw;
  return config;
}

inline std::string DevStepDumpJsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (const char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

/// Shortest representation that ROUND-TRIPS back to the same float.
///
/// NOT std::to_string: that is "%f" with six decimals, so it prints 2.0f as
/// "2.000000" -- noise digits on every line -- while a genuine 12.3456789f is
/// written as "12.345679", silently dropping digits. A parity run compares
/// logits between two arms, and a logit that lost precision in the artifact
/// makes two arms look closer or further apart than they are. to_chars shortest
/// gives "2", "1.5" and every digit the float actually carries.
inline std::string DevStepDumpFloat(float value) {
  char buffer[64];
  const std::to_chars_result result =
      std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec != std::errc{}) return "null";
  return std::string(buffer, result.ptr);
}

inline std::string DevStepDumpDouble(double value) {
  if (!std::isfinite(value)) return "null";
  char buffer[128];
  const std::to_chars_result result =
      std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec != std::errc{}) return "null";
  return std::string(buffer, result.ptr);
}

/// Builds the exact JSONL line, WITHOUT the trailing newline.
///
/// ARGMAX AND TOP-5 ARE TAKEN FROM THE HOST ROW, never from d_out_token:
/// d_out_token carries the SAMPLER, and two arms that differ in temperature or
/// penalties would then not be comparable at all. Ties in the top-5 are broken by
/// lower token id, so the ordering is total and two runs cannot disagree on
/// ordering alone.
[[nodiscard]] inline std::string DevStepDumpLine(
    const DevStepDumpConfig& config, std::uint32_t step,
    const std::vector<float>& logits, std::size_t vocab_size,
    std::uint32_t write_word, std::uint32_t read_word,
    std::uint32_t emitted_token, bool frontier = false) {
  std::size_t argmax_index = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[argmax_index]) argmax_index = i;
  }

  std::vector<std::size_t> order;
  order.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) order.push_back(i);
  const std::size_t want = std::min<std::size_t>(5, logits.size());
  std::partial_sort(order.begin(), order.begin() + static_cast<long>(want),
                    order.end(), [&logits](std::size_t a, std::size_t b) {
                      if (logits[a] != logits[b]) return logits[a] > logits[b];
                      return a < b;
                    });

  // STEP_DUMP is also the teacher-forced quality oracle. Keep its historic
  // top5 logits unchanged, and add normalized log-probabilities for consumers
  // comparing across rows whose logit offsets differ.
  double max_logit = -std::numeric_limits<double>::infinity();
  for (const float logit : logits)
    max_logit = std::max(max_logit, static_cast<double>(logit));
  double log_normalizer = std::numeric_limits<double>::quiet_NaN();
  if (std::isfinite(max_logit)) {
    double exp_sum = 0.0;
    for (const float logit : logits)
      exp_sum += std::exp(static_cast<double>(logit) - max_logit);
    log_normalizer = max_logit + std::log(exp_sum);
  }

  std::string line;
  line.reserve(160);
  line += "{\"step\":" + std::to_string(step);
  line += ",\"argmax\":" + std::to_string(argmax_index);
  // The token this step EMITTED, taken from the device's own decision, never
  // recomputed here. It is a separate D2H read of a separate device buffer:
  // argmax_index comes from reducing the host copy of the logits row on the CPU,
  // emitted_token comes from scratch.decode.sampled_token after the GPU argmax
  // kernel wrote it. Two reads, two buffers, two implementations -- so their
  // agreement is evidence rather than a tautology.
  //
  // SCOPE, STATED PLAINLY SO NOBODY OVER-READS IT. On the ForwardToken path that
  // buffer is written by LaunchBatchedGPUArgmax and nothing else, so emitted is
  // the DEVICE argmax, not a temperature sample. This therefore cross-checks a
  // device reduction against a host reduction over the same row -- it can catch a
  // raced or truncated copy, a NaN, a tie broken differently, or the kernel
  // reading a different row. It CANNOT catch a sampler preferring a token the
  // host row ranks lower, because no sampler runs on this path.
  line += ",\"emitted\":" + std::to_string(emitted_token);
  if (frontier) line += ",\"frontier\":true";
  line += ",\"top5\":[";
  for (std::size_t i = 0; i < want; ++i) {
    if (i != 0) line += ',';
    line += '[' + std::to_string(order[i]) + ',' +
            DevStepDumpFloat(logits[order[i]]) + ']';
  }
  line += "]";
  line += ",\"top5_logprobs\":[";
  for (std::size_t i = 0; i < want; ++i) {
    if (i != 0) line += ',';
    const double logprob =
        static_cast<double>(logits[order[i]]) - log_normalizer;
    line += '[' + std::to_string(order[i]) + ',' +
            DevStepDumpDouble(logprob) + ']';
  }
  line += "]";
  line += ",\"write_status\":" + std::to_string(write_word);
  line += ",\"read_status\":" + std::to_string(read_word);
  line += ",\"vocab\":" + std::to_string(vocab_size);
  line += ",\"dump_path\":\"" + DevStepDumpJsonEscape(config.path) + "\"}";
  return line;
}

/// Appends one line and flushes it, so a run that dies mid-generation still
/// leaves every completed step on disk.
///
/// Returns false when disabled or when the path cannot be opened. A broken dump
/// path MUST NOT kill generation: this is a measurement accessory, and the
/// measurement's own pass bar is about the decode step, not about its logger.
inline bool AppendDevStepDump(const DevStepDumpConfig& config,
                              std::uint32_t step,
                              const std::vector<float>& logits,
                              std::size_t vocab_size,
                              std::uint32_t write_word,
                              std::uint32_t read_word,
                              std::uint32_t emitted_token,
                              bool frontier = false) {
  if (!config.enabled || logits.empty()) return false;
  std::FILE* const file = std::fopen(config.path.c_str(), "a");
  if (file == nullptr) return false;
  const std::string line = DevStepDumpLine(config, step, logits, vocab_size,
                                           write_word, read_word,
                                           emitted_token, frontier);
  std::fwrite(line.data(), 1, line.size(), file);
  std::fputc('\n', file);
  std::fclose(file);
  // GUFO_DEV_STEP_DUMP_RAW=1 also appends the whole row as
  // vocab_size native-endian f32 to <path>.f32, row i = JSONL line i, so a
  // reader can compute exact KLD instead of a top-5 bound. Read every call,
  // like GUFO_DEV_STEP_DUMP.
  const char* const raw = std::getenv("GUFO_DEV_STEP_DUMP_RAW");
  if (raw != nullptr && raw[0] == '1') {
    std::FILE* const rows = std::fopen((config.path + ".f32").c_str(), "ab");
    if (rows == nullptr) return false;
    std::vector<float> row(vocab_size, -std::numeric_limits<float>::infinity());
    std::copy_n(logits.begin(), std::min(vocab_size, logits.size()),
                row.begin());
    std::fwrite(row.data(), sizeof(float), row.size(), rows);
    std::fclose(rows);
  }
  return true;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_DEV_STEP_DUMP_HPP_

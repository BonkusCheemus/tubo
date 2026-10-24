// The GUFO_DEV_STEP_DUMP per-step dump.
//
// WHAT IS ASSERTED HERE, AND WHAT CANNOT BE.
//
// Asserted on CPU, byte-exactly:
//   - the variable is OFF when unset, and OFF when set to the empty string;
//   - a line is NOT written, and NOT EVEN OPENED, when disabled;
//   - the JSONL line is the exact string the brief specifies, for a row whose
//     argmax and top-5 are known by hand.
//
// NOT assertable on CPU: "with the env unset, output is byte-identical to C1 on
// one prompt". That comparison needs a real prompt through a real model, which
// needs the GPU. What IS proven here is the mechanism that makes it hold: with
// the variable unset, ReadDevStepDumpConfig returns enabled == false, the call
// site in decode.cpp branches on that BEFORE queueing the logits copy, and
// AppendDevStepDump returns before opening any path. No allocation, no copy, no
// file. The end-to-end confirmation belongs to C2 and is named as owed.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen/hip/dev_step_dump.hpp"

namespace fs = std::filesystem;
using gufo::hip::AppendDevStepDump;
using gufo::hip::DevStepDumpConfig;
using gufo::hip::DevStepDumpLine;
using gufo::hip::ReadDevStepDumpConfig;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::string Slurp(const fs::path& path) {
  std::ifstream in(path);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

std::string RepoRoot() {
  const std::string self = __FILE__;
  const std::string tail = "tests/models/qwen/dev_step_dump_test.cpp";
  const auto pos = self.rfind(tail);
  if (pos == std::string::npos) return {};
  return self.substr(0, pos);
}

// A row whose argmax and ordering are obvious by inspection, so a change in the
// formatter cannot be excused as a change in the data.
std::vector<float> SampleRow() {
  //              0      1      2      3      4      5
  return {0.5F, -1.0F, 2.0F, 0.25F, 1.5F, -0.5F};
}

void UnsetIsOff() {
  ::unsetenv("GUFO_DEV_STEP_DUMP");
  const DevStepDumpConfig off = ReadDevStepDumpConfig();
  Check(!off.enabled, "unset: the dump is disabled");
  Check(off.path.empty(), "unset: no path is chosen");

  ::setenv("GUFO_DEV_STEP_DUMP", "", 1);
  const DevStepDumpConfig empty = ReadDevStepDumpConfig();
  Check(!empty.enabled, "empty string: the dump is disabled");

  ::setenv("GUFO_DEV_STEP_DUMP", "/tmp/gufo_c1b_should_not_appear.jsonl", 1);
  const DevStepDumpConfig on = ReadDevStepDumpConfig();
  Check(on.enabled, "set: the dump is enabled");
  Check(on.path == "/tmp/gufo_c1b_should_not_appear.jsonl",
        "set: the path is the one given");
  ::unsetenv("GUFO_DEV_STEP_DUMP");
}

void DisabledWritesNothing() {
  ::fs::remove("/tmp/gufo_c1b_disabled_probe.jsonl");
  DevStepDumpConfig off;
  off.enabled = false;
  off.path = "/tmp/gufo_c1b_disabled_probe.jsonl";
  const bool wrote =
      AppendDevStepDump(off, 0, SampleRow(), 6, 0, 0, /*emitted=*/2);
  Check(!wrote, "disabled: AppendDevStepDump reports it wrote nothing");
  Check(!fs::exists("/tmp/gufo_c1b_disabled_probe.jsonl"),
        "disabled: no file was even created");

  // And the enabled call with an EMPTY row writes nothing either: a step that
  // computed no logits has no observation to record.
  DevStepDumpConfig on;
  on.enabled = true;
  on.path = "/tmp/gufo_c1b_disabled_probe.jsonl";
  const bool empty_row = AppendDevStepDump(on, 0, {}, 6, 0, 0, /*emitted=*/2);
  Check(!empty_row && !fs::exists("/tmp/gufo_c1b_disabled_probe.jsonl"),
        "disabled: an empty logits row produces no file either");
}

void SchemaIsExact() {
  DevStepDumpConfig config;
  config.enabled = true;
  config.path = "/tmp/x.jsonl";
  const std::string line =
      DevStepDumpLine(config, 7, SampleRow(), 6, 2, 3, /*emitted=*/2);
  const std::string expected =
      "{\"step\":7,\"argmax\":2,\"emitted\":2,\"top5\":[[2,2],[4,1.5],[0,0.5],[3,0.25],"
      "[5,-0.5]],\"top5_logprobs\":[[2,-0.7586103507824205],"
      "[4,-1.2586103507824205],[0,-2.2586103507824205],"
      "[3,-2.5086103507824205],[5,-3.2586103507824205]],"
      "\"write_status\":2,\"read_status\":3,\"vocab\":6,"
      "\"dump_path\":\"/tmp/x.jsonl\"}";
  if (line != expected) {
    std::printf("       got:      %s\n", line.c_str());
    std::printf("       expected: %s\n", expected.c_str());
  }
  Check(line == expected, "schema: the JSONL line is byte-exact");
  // The four fields the brief names, all present.
  for (const char* field : {"\"step\":", "\"argmax\":", "\"emitted\":",
                            "\"top5\":", "\"top5_logprobs\":",
                            "\"write_status\":", "\"read_status\":"}) {
    Check(line.find(field) != std::string::npos,
          std::string("schema: carries ") + field);
  }
}

void TopLogProbsAreNormalizedAndKeepRawOrder() {
  DevStepDumpConfig config;
  config.enabled = true;
  config.path = "/tmp/x.jsonl";
  const auto hand_built = gufo::json::parse(DevStepDumpLine(
      config, 0, std::vector<float>{0.0F, 0.0F}, 2, 0, 0, 0));
  const auto* raw = hand_built.find("top5");
  const auto* logprobs = hand_built.find("top5_logprobs");
  Check(raw != nullptr && logprobs != nullptr,
        "logprobs: dump keeps raw top5 and adds normalized top5_logprobs");
  if (raw == nullptr || logprobs == nullptr) return;
  Check(raw->items().size() == 2 && logprobs->items().size() == 2,
        "logprobs: hand-built two-token row keeps both candidates");
  if (raw->items().size() != 2 || logprobs->items().size() != 2) return;
  const double expected = -std::log(2.0);
  Check(std::abs(logprobs->items()[0].items()[1].as_double() - expected) < 1e-12 &&
            std::abs(logprobs->items()[1].items()[1].as_double() - expected) <
                1e-12,
        "logprobs: zero logits normalize to -log(2) using the full row");

  const auto row = gufo::json::parse(
      DevStepDumpLine(config, 0, SampleRow(), 6, 0, 0, 2));
  const auto& raw_top = row.find("top5")->items();
  const auto& normalized_top = row.find("top5_logprobs")->items();
  double top5_mass = 0;
  bool same_order = raw_top.size() == normalized_top.size();
  for (std::size_t i = 0; i < raw_top.size() && i < normalized_top.size(); ++i) {
    same_order &= raw_top[i].items()[0].as_size() ==
                  normalized_top[i].items()[0].as_size();
    top5_mass += std::exp(normalized_top[i].items()[1].as_double());
  }
  Check(same_order, "logprobs: normalized top5 retains the raw-logit order");
  Check(top5_mass <= 1.0,
        "logprobs: the selected candidates carry no more than unit mass");
}

void FrontierTagIsOptIn() {
  DevStepDumpConfig config;
  config.enabled = true;
  config.path = "/tmp/x.jsonl";
  const auto logits = SampleRow();
  const std::string decode =
      DevStepDumpLine(config, 7, logits, 6, 0, 0, 2);
  const std::string frontier =
      DevStepDumpLine(config, 7, logits, 6, 0, 0, 2, true);
  Check(decode.find("\"frontier\"") == std::string::npos,
        "frontier tag: normal decode rows keep the old schema");
  Check(frontier.find("\"frontier\":true") != std::string::npos,
        "frontier tag: prefill rows are explicitly distinguishable");
}

/// THE CHECK ADDED IS "argmax == emitted, MISMATCH FAILS THE RUN". That
/// is only a real gate if the two fields CAN disagree. So this asserts they are
/// not welded together: feed an emitted id that is deliberately NOT the argmax
/// and require the formatter to print it verbatim.
///
/// Without this, a formatter that quietly recomputed emitted from the host row
/// would pass every other test in this file AND make the parity check unfailable.
void EmittedIsIndependentOfArgmax() {
  DevStepDumpConfig config;
  config.enabled = true;
  config.path = "/tmp/x.jsonl";
  const std::size_t host_argmax = 2;  // SampleRow()'s true argmax

  const std::string disagreeing =
      DevStepDumpLine(config, 0, SampleRow(), 6, 0, 0, /*emitted=*/99);
  Check(disagreeing.find("\"argmax\":2") != std::string::npos,
        "independence: the host argmax is still 2");
  Check(disagreeing.find("\"emitted\":99") != std::string::npos,
        "independence: a DISAGREEING emitted id is printed verbatim, not "
        "overwritten by the argmax");

  const std::string agreeing =
      DevStepDumpLine(config, 0, SampleRow(), 6, 0, 0,
                      /*emitted=*/static_cast<std::uint32_t>(host_argmax));
  Check(agreeing.find("\"argmax\":2,\"emitted\":2") != std::string::npos,
        "independence: an agreeing pair is recorded as two equal fields");

  // And the disagreement is detectable from the text alone, which is what C2
  // will do on each line.
  Check(disagreeing != agreeing,
        "independence: a mismatch is visible in the artifact, so C2 can fail "
        "the run on it");
}

/// The emitted id must cost nothing when the dump is off: same early return,
/// same absence of a file.
void UnsetCostsNothingForTheEmittedField() {
  ::fs::remove("/tmp/gufo_c1b_emitted_unset.jsonl");
  DevStepDumpConfig off;
  off.enabled = false;
  off.path = "/tmp/gufo_c1b_emitted_unset.jsonl";
  const bool wrote =
      AppendDevStepDump(off, 3, SampleRow(), 6, 0, 0, /*emitted=*/99);
  Check(!wrote && !fs::exists("/tmp/gufo_c1b_emitted_unset.jsonl"),
        "unset: carrying an emitted id still writes nothing and opens no file");
}

void AppendedLineIsReadableJsonl() {
  const fs::path path = "/tmp/gufo_c1b_append_probe.jsonl";
  fs::remove(path);
  DevStepDumpConfig config;
  config.enabled = true;
  config.path = path.string();
  AppendDevStepDump(config, 0, SampleRow(), 6, 0, 0, /*emitted=*/2);
  AppendDevStepDump(config, 1, SampleRow(), 6, 0, 0, /*emitted=*/2);

  const std::string body = Slurp(path);
  std::istringstream lines(body);
  std::string line;
  int count = 0;
  bool both_steps_present = true;
  while (std::getline(lines, line)) {
    if (line.find("\"step\":" + std::to_string(count)) == std::string::npos) {
      both_steps_present = false;
    }
    ++count;
  }
  Check(count == 2, "append: two steps produce two lines (" +
                        std::to_string(count) + ")");
  Check(both_steps_present, "append: each line carries its own step index");
  Check(body.find('\n') != std::string::npos,
        "append: lines are newline terminated");
  fs::remove(path);
}

/// The call site must branch on the switch BEFORE the copy, otherwise "unset
/// costs nothing" is an intention rather than a fact. This is a source check on
/// purpose: it is the only thing standing between the claim and the code.
void CallSiteSkipsTheCopyWhenUnset() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "call site: located the repository root");
    return;
  }
  const std::string decode =
      Slurp(fs::path(root) / "src/models/qwen/hip/decode.cpp");
  Check(!decode.empty(), "call site: decode.cpp is readable");
  if (decode.empty()) return;

  const auto read = decode.find("ReadDevStepDumpConfig()");
  const auto guard = decode.find("if (step_dump.enabled)", read);
  const auto copy = decode.find("hipMemcpyAsync(dump_logits.data()", guard);
  Check(read != std::string::npos && guard != std::string::npos &&
            copy != std::string::npos && read < guard && guard < copy,
        "call site: the logits copy sits INSIDE the enabled guard");

  // R3: the emitted id must arrive from its OWN device read, not be recomputed
  // from the logits row. next_token_id is filled by the d_out_token copy, which
  // is a different buffer from the logits row; assert both copies exist and that
  // the emitted value passed on is that one, not a host-side argmax.
  const auto token_copy =
      decode.find("hipMemcpyAsync(&next_token_id, d_out_token");
  const auto emitted_pass =
      decode.find("read_word, next_token_id)", guard);
  Check(token_copy != std::string::npos,
        "call site: the emitted id is read from d_out_token, its own device "
        "buffer");
  Check(emitted_pass != std::string::npos,
        "call site: next_token_id is what the dump receives");
}

void PrefillEmitsTheFrontierRow() {
  const std::string root = RepoRoot();
  if (root.empty()) {
    Check(false, "prefill frontier: located the repository root");
    return;
  }
  const std::string prefill =
      Slurp(fs::path(root) / "src/models/qwen/hip/prefill_chunk.cpp");
  Check(!prefill.empty(), "prefill frontier: prefill_chunk.cpp is readable");
  if (prefill.empty()) return;
  Check(prefill.find("ReadDevStepDumpConfig()") != std::string::npos,
        "prefill frontier: reads the independent step-dump switch");
  Check(prefill.find("AppendDevStepDump") != std::string::npos,
        "prefill frontier: records the prefill logits before decode");
  Check(prefill.find("/*frontier=*/true") != std::string::npos,
        "prefill frontier: tags the row so the consumer can reject decode rows");
}

}  // namespace

int main() {
  UnsetIsOff();
  DisabledWritesNothing();
  SchemaIsExact();
  TopLogProbsAreNormalizedAndKeepRawOrder();
  FrontierTagIsOptIn();
  EmittedIsIndependentOfArgmax();
  UnsetCostsNothingForTheEmittedField();
  AppendedLineIsReadableJsonl();
  CallSiteSkipsTheCopyWhenUnset();
  PrefillEmitsTheFrontierRow();
  std::printf("%s  %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL",
              g_failures);
  return g_failures == 0 ? 0 : 1;
}

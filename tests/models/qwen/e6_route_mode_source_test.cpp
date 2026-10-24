// Route telemetry must identify each executor entry.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

namespace {
int failures = 0;
void Check(bool condition, const char* message) {
  std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", message);
  if (!condition) ++failures;
}

std::string Read(const char* path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void HasCall(const std::string& source, const char* mode) {
  const std::string expected =
      std::string("EmitDecodeRouteTelemetry(") +
      (std::string(mode) == "decode" ? "weights_, policy_, \"decode\"" :
       std::string(mode) == "batched_token"
           ? "weights, coordinator->policy_, \"batched_token\""
           : "weights_, coordinator.policy_, \"verify\"");
  Check(source.find(expected) != std::string::npos, mode);
}
}  // namespace

int main() {
  const auto decode = Read("src/models/qwen/hip/decode.cpp");
  const auto batched = Read("src/models/qwen/hip/batched_decode.cpp");
  Check(!decode.empty() && !batched.empty(), "source files exist");
  HasCall(decode, "decode");
  HasCall(batched, "batched_token");
  HasCall(batched, "verify");
  Check(batched.find("LaunchTurbo4KvWriteBatched(") != std::string::npos,
        "batched verification writes Turbo4 K/V through the tested writer");
  Check(batched.find("LaunchTurbo4AttentionSplit(") != std::string::npos,
        "batched verification reads each row through the split reader");
  Check(batched.find("start_pos + static_cast<std::uint32_t>(row) + 1U") !=
            std::string::npos,
        "verification reader context is causal per candidate row");
  if (failures != 0) return 1;
  std::puts("ALL CHECKS PASSED");
  return 0;
}

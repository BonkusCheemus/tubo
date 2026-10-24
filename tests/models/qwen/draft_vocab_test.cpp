#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/qwen/hip/dflash/draft_vocab.hpp"

namespace {

namespace dflash = gufo::hip::dflash;

int failures = 0;

void Check(bool ok, const char* what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) {
    ++failures;
  }
}

// The host twin of BuildDraftVocabCut's device gather: same run walk, memcpy
// instead of hipMemcpy. Row r of the source holds bytes r*10 .. r*10+3.
std::vector<std::uint8_t> Gather(const std::vector<std::uint32_t>& ids,
                                 std::size_t vocab, std::size_t row_bytes,
                                 std::size_t& copies) {
  std::vector<std::uint8_t> source(vocab * row_bytes);
  for (std::size_t row = 0; row < vocab; ++row)
    for (std::size_t byte = 0; byte < row_bytes; ++byte)
      source[row * row_bytes + byte] =
          static_cast<std::uint8_t>(row * 10 + byte);
  std::vector<std::uint8_t> out(ids.size() * row_bytes, 0xEE);
  copies = 0;
  dflash::ForEachDraftVocabRun(
      ids, [&](std::size_t src, std::size_t dst, std::size_t count) {
        std::memcpy(out.data() + dst * row_bytes,
                    source.data() + src * row_bytes, count * row_bytes);
        ++copies;
      });
  return out;
}

void FirstNKeepsLowIdsAndTopTail() {
  std::string error;
  const auto ids = dflash::ParseDraftVocab("first:1024", 4096, error);
  Check(ids.size() == 1024 && ids.front() == 0 && ids[511] == 511 &&
            ids[512] == 3584 && ids.back() == 4095,
        "first:N yields ids 0..N-513 plus the top 512 (special tokens)");
  std::size_t copies = 0;
  const auto rows = Gather(ids, 4096, 4, copies);
  Check(copies == 2 && rows[511 * 4] == static_cast<std::uint8_t>(5110) &&
            rows[512 * 4] == static_cast<std::uint8_t>(35840),
        "first:N gathers in two contiguous copies");
  // N below the tail keeps only the tail.
  const auto small = dflash::ParseDraftVocab("first:256", 4096, error);
  Check(small.size() == 256 && small.front() == 3840 && small.back() == 4095,
        "first:N with N <= 512 keeps the top N ids");
}

void FileIdsSortDedupAndGather() {
  // 256 unique ids, written shuffled with two duplicates.
  std::vector<std::uint32_t> written;
  for (std::uint32_t id = 0; id < 128; ++id) written.push_back(1000 + id);
  for (std::uint32_t id = 0; id < 128; ++id) written.push_back(5 * id);
  written.push_back(1000);
  written.push_back(0);
  std::vector<std::uint32_t> shuffled(written.rbegin(), written.rend());
  const std::string path = "draft_vocab_test.bin";
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(shuffled.data()),
              static_cast<std::streamsize>(shuffled.size() * 4));
  }
  std::string error;
  const auto ids = dflash::ParseDraftVocab(path, 2000, error);
  Check(ids.size() == 256 && error.empty(), "file: duplicates dropped");
  Check(std::is_sorted(ids.begin(), ids.end()), "file: ids sorted");
  std::size_t copies = 0;
  const auto rows = Gather(ids, 2000, 4, copies);
  bool exact = true;
  for (std::size_t index = 0; index < ids.size(); ++index)
    for (std::size_t byte = 0; byte < 4; ++byte)
      exact &= rows[index * 4 + byte] ==
               static_cast<std::uint8_t>(ids[index] * 10 + byte);
  Check(exact, "file: row j of the cut head is source row ids[j]");
  // 0, 5, ..., 635 are 128 isolated ids; 1000..1127 is one run.
  Check(copies == 129, "file: one copy per run of consecutive ids");
  std::remove(path.c_str());
}

void BadSpecsRefused() {
  std::string error;
  Check(dflash::ParseDraftVocab("first:300", 1024, error).empty() &&
            !error.empty(),
        "count not a multiple of 256 is refused");
  error.clear();
  Check(dflash::ParseDraftVocab("first:2048", 1024, error).empty() &&
            !error.empty(),
        "N > vocab is refused");
  error.clear();
  Check(dflash::ParseDraftVocab("first:", 1024, error).empty() &&
            !error.empty(),
        "empty count is refused");
  error.clear();
  Check(dflash::ParseDraftVocab("/nonexistent/vocab.bin", 1024, error)
                .empty() &&
            !error.empty(),
        "missing file is refused");
  error.clear();
  const std::string path = "draft_vocab_test_odd.bin";
  {
    std::ofstream out(path, std::ios::binary);
    out.write("abcde", 5);
  }
  Check(dflash::ParseDraftVocab(path, 1024, error).empty() && !error.empty(),
        "partial uint32 is refused");
  std::remove(path.c_str());
}

}  // namespace

int main() {
  FirstNKeepsLowIdsAndTopTail();
  FileIdsSortDedupAndGather();
  BadSpecsRefused();
  std::printf("%s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}

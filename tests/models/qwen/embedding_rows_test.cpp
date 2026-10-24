#include "src/models/qwen/embedding_rows.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace fs = std::filesystem;
using gufo::models::qwen::EmbeddingResidentSpans;
using gufo::models::qwen::EmbeddingRowStore;
constexpr std::size_t kHidden = 5120, kRows = 97, kRowBytes = 2880,
                      kPage = 4096;

void Expect(bool value, const char* what) {
  if (!value) {
    std::cerr << "FAIL: " << what << '\n';
    std::exit(1);
  }
}

template<class F>
void Throws(F function, const char* what) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception&) {
    threw = true;
  }
  Expect(threw, what);
}

template<class T>
void Pod(std::vector<std::uint8_t>& bytes, T value) {
  const auto offset = bytes.size();
  bytes.resize(offset + sizeof(T));
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void String(std::vector<std::uint8_t>& bytes, const std::string& value) {
  Pod(bytes, std::uint64_t{value.size()});
  bytes.insert(bytes.end(), value.begin(), value.end());
}

std::vector<std::uint8_t> Fixture(
    bool has_output = true, bool aliased_output = false,
    gufo::core::GgmlType embedding_type = gufo::core::GgmlType::kQ4_K) {
  std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
  Pod(bytes, std::uint32_t{3});
  Pod(bytes, std::uint64_t{has_output ? 3U : 2U});
  Pod(bytes, std::uint64_t{0});
  const auto tensor = [&](const std::string& name, std::uint64_t dim0,
                          std::uint64_t dim1, gufo::core::GgmlType type,
                          std::uint64_t offset) {
    String(bytes, name);
    Pod(bytes, std::uint32_t{2});
    Pod(bytes, dim0);
    Pod(bytes, dim1);
    Pod(bytes, static_cast<std::uint32_t>(type));
    Pod(bytes, offset);
  };
  tensor("before.weight", 16, 1, gufo::core::GgmlType::kF32, 0);
  tensor("token_embd.weight", kHidden, kRows, embedding_type, kPage);
  const auto row_bytes =
      embedding_type == gufo::core::GgmlType::kBF16 ? kHidden * 2 : kRowBytes;
  const auto after = (kPage + kRows * row_bytes + 31) / 32 * 32;
  if (has_output) {
    if (aliased_output)
      tensor("output.weight", kHidden, kRows, embedding_type, kPage);
    else
      tensor("output.weight", kHidden, 1, gufo::core::GgmlType::kF32, after);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32, 0);
  const auto data = bytes.size();
  bytes.resize(data + after + kHidden * sizeof(float), 0x5a);
  for (std::size_t i = 0; i < kRows * row_bytes; ++i)
    bytes[data + kPage + i] = static_cast<std::uint8_t>((i * 17) ^ (i >> 8));
  return bytes;
}

void Write(const fs::path& path, std::span<const std::uint8_t> bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  Expect(out.good(), "fixture write");
}

void Flip(const fs::path& path, std::size_t offset) {
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  file.seekg(static_cast<std::streamoff>(offset));
  char byte = 0;
  file.read(&byte, 1);
  byte ^= 1;
  file.seekp(static_cast<std::streamoff>(offset));
  file.write(&byte, 1);
  file.flush();
  Expect(file.good(), "corruption write");
}

void CheckRows(EmbeddingRowStore& store, const gufo::core::GgufReader& reader,
               std::span<const std::uint32_t> ids, bool prefill) {
  std::vector<std::uint8_t> output(ids.size() * kRowBytes, 0xff);
  store.Fetch(ids, output, prefill);
  const auto* source = static_cast<const std::uint8_t*>(
      reader.FindTensor("token_embd.weight")->data);
  for (std::size_t i = 0; i < ids.size(); ++i)
    Expect(std::memcmp(source + ids[i] * kRowBytes,
                       output.data() + i * kRowBytes, kRowBytes) == 0,
           "fetched bytes equal GGUF row");
}

}  // namespace

int main() {
  using gufo::models::qwen::EmbeddingStageCapacity;
  std::size_t capacity = 0;
  const std::array<std::size_t, 8> ramp{1, 2, 4, 4, 8, 8, 8, 8};
  for (std::size_t width = 1; width <= ramp.size(); ++width) {
    capacity = EmbeddingStageCapacity(width, capacity, kRowBytes);
    Expect(capacity == ramp[width - 1],
           "width ramp uses geometric staging capacity");
  }
  Expect(EmbeddingStageCapacity(256, 0, kRowBytes) == 256 &&
             EmbeddingStageCapacity(300, 256, kRowBytes) == 512 &&
             EmbeddingStageCapacity(400, 512, kRowBytes) == 512 &&
             EmbeddingStageCapacity(800, 512, kRowBytes) == 1024,
         "growing prefill chunks reuse capacity");
  Throws([] { (void)EmbeddingStageCapacity(0, 0, kRowBytes); },
         "zero stage rows refused");
  Throws([] { (void)EmbeddingStageCapacity(1, 0, 0); },
         "zero stage row bytes refused");
  Throws(
      [] {
        (void)EmbeddingStageCapacity(2, 0,
                                     std::numeric_limits<std::size_t>::max());
      },
      "stage byte overflow refused");
  const auto maximum =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  Expect(EmbeddingStageCapacity(maximum - 1, maximum / 2 + 1, 1) == maximum,
         "geometric growth saturates at checked device-index bound");
  Throws([&] { (void)EmbeddingStageCapacity(maximum + 1, 0, 1); },
         "stage index overflow refused");

  const std::string known = "123456789";
  Expect(gufo::models::qwen::EmbeddingCrc32c(
             {reinterpret_cast<const std::uint8_t*>(known.data()),
              known.size()}) == 0xe3069283U,
         "CRC32C published check vector");
  char temporary[] = "/tmp/gufo-embrows-XXXXXX";
  const auto* made = ::mkdtemp(temporary);
  Expect(made != nullptr, "create temporary directory");
  const fs::path directory(made);
  struct Cleanup {
    fs::path directory;
    ~Cleanup() { fs::remove_all(directory); }
  } cleanup{directory};
  // Keep identity-cache writes within the fixture, not the real user's cache.
  ::setenv("XDG_CACHE_HOME", directory.c_str(), 1);
  const auto image = Fixture();
  const auto source = directory / "synthetic.gguf";
  const auto rows = directory / "synthetic.embrows";
  Write(source, image);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(source, &error);
  Expect(reader != nullptr, error.c_str());
  Expect(EmbeddingRowStore::Export(*reader, rows, &error), error.c_str());
  Expect(!EmbeddingRowStore::Export(*reader, rows, &error),
         "export never overwrites existing file");
  Expect(fs::file_size(rows) == (kRows + 1) * kPage, "padded file extent");
  auto store = EmbeddingRowStore::Open(*reader, rows, 2 * kPage, &error);
  if (!store) {
    std::cerr << "Direct-I/O fixture must work on this filesystem: " << error
              << '\n';
    return 1;  // Never label unsupported O_DIRECT a passing test.
  }
  Expect(store->RowBytes() == kRowBytes &&
             store->TensorBytes() == kRows * kRowBytes,
         "bound geometry");
  CheckRows(*store, *reader, std::array<std::uint32_t, 2>{1, 2}, false);
  Expect(store->Reads() == 2, "two initial misses");
  CheckRows(*store, *reader, std::array<std::uint32_t, 1>{1}, false);
  Expect(store->Hits() == 1 && store->Reads() == 2,
         "cache hit avoids direct read");
  CheckRows(*store, *reader, std::array<std::uint32_t, 1>{3}, false);
  CheckRows(*store, *reader, std::array<std::uint32_t, 1>{2}, false);
  Expect(store->Reads() == 4 && store->CachedBytes() == 2 * kPage,
         "LRU eviction and bound");

  std::mt19937 random(125);
  for (int i = 0; i < 500; ++i) {
    const std::array<std::uint32_t, 1> id{
        static_cast<std::uint32_t>(random() % kRows)};
    CheckRows(*store, *reader, id, false);
  }
  CheckRows(*store, *reader, std::array<std::uint32_t, 2>{0, kRows - 1}, false);
  auto batch_store = EmbeddingRowStore::Open(*reader, rows, 2 * kPage, &error);
  Expect(batch_store != nullptr, error.c_str());
  std::vector<std::uint32_t> batch;
  for (std::uint32_t i = 0; i < kRows; ++i) {
    batch.push_back(i);
    batch.push_back(i);
    batch.push_back(i);
  }
  CheckRows(*batch_store, *reader, batch, true);
  Expect(batch_store->Reads() == kRows,
         "prefill deduplicates IDs even beyond cache capacity");
  Expect(batch_store->CachedBytes() == 2 * kPage,
         "parallel prefill stays within cache bound");
  {
    std::vector<std::jthread> clients;
    for (std::uint32_t i = 0; i < 4; ++i)
      clients.emplace_back([&, i] {
        CheckRows(*batch_store, *reader,
                  std::array<std::uint32_t, 3>{i, i + 1, i}, true);
      });
  }
  std::vector<std::uint8_t> one(kRowBytes);
  Throws([&] { store->Fetch(std::array<std::uint32_t, 1>{kRows}, one, false); },
         "invalid token rejected");
  Throws([&] { store->Fetch(std::array<std::uint32_t, 1>{0}, {}, false); },
         "wrong output extent rejected");
  store->Fetch({}, {}, true);
  Expect(
      !EmbeddingRowStore::Open(*reader, directory / "missing", kPage, &error),
      "missing row file declines");
  Expect(!EmbeddingRowStore::Open(*reader, rows, 0, &error),
         "invalid cache capacity declines");

  for (int variant = 0; variant < 3; ++variant) {
    const auto fixture = Fixture(variant != 0, variant == 1,
                                 variant == 2 ? gufo::core::GgmlType::kBF16
                                              : gufo::core::GgmlType::kQ4_K);
    const auto path =
        directory / ("decline" + std::to_string(variant) + ".gguf");
    Write(path, fixture);
    auto unsupported = gufo::core::GgufReader::OpenFile(path, &error);
    Expect(unsupported != nullptr, "unsupported-mode fixture is a valid GGUF");
    Expect(
        !EmbeddingRowStore::Open(*unsupported, rows, kPage, &error),
        "tied/aliased output and unsupported quant decline before activation");
    Expect(error.find(variant == 2 ? "Q4_K" : "tied") != std::string::npos,
           "decline reason comes from the eligibility gate");
  }
  auto memory =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(memory != nullptr, "memory image parses");
  Expect(!EmbeddingRowStore::Open(*memory, rows, kPage, &error) &&
             error.find("file-backed") != std::string::npos,
         "anonymous image declines the file-backed exclusion path");

  auto changed = image;
  changed.back() ^= 1;
  Write(directory / "other.gguf", changed);
  auto other =
      gufo::core::GgufReader::OpenFile(directory / "other.gguf", &error);
  Expect(other != nullptr, "second source readable");
  Expect(!EmbeddingRowStore::Open(*other, rows, kPage, &error),
         "same-shaped stale source identity declines");
  const auto bad = directory / "bad.embrows";
  fs::copy_file(rows, bad);
  Flip(bad, 0);
  Expect(!EmbeddingRowStore::Open(*reader, bad, kPage, &error),
         "bad header declines");
  fs::copy_file(rows, bad, fs::copy_options::overwrite_existing);
  fs::resize_file(bad, fs::file_size(bad) - 1);
  Expect(!EmbeddingRowStore::Open(*reader, bad, kPage, &error),
         "bad file extent declines");
  fs::copy_file(rows, bad, fs::copy_options::overwrite_existing);
  Flip(bad, kPage + 16);
  auto corrupt = EmbeddingRowStore::Open(*reader, bad, kPage, &error);
  Expect(corrupt != nullptr,
         "row corruption tested lazily, not whole-file load");
  Throws([&] { corrupt->Fetch(std::array<std::uint32_t, 1>{0}, one, false); },
         "CRC32C catches row corruption");
  Expect(corrupt->CachedBytes() == 0, "bad bytes never cached");
  std::vector<std::uint32_t> failing_prefill;
  for (std::uint32_t row = 0; row < kRows; ++row)
    failing_prefill.push_back(row);
  std::vector<std::uint8_t> failed_output(kRows * kRowBytes);
  Throws([&] { corrupt->Fetch(failing_prefill, failed_output, true); },
         "prefill worker error joins pool and propagates first exception");
  Expect(corrupt->CachedBytes() <= kPage,
         "failing worker pool preserves cache bound");
  corrupt.reset();
  fs::copy_file(rows, bad, fs::copy_options::overwrite_existing);
  {
    std::fstream file(bad, std::ios::in | std::ios::out | std::ios::binary);
    std::array<char, kPage> record{};
    file.seekg(kPage);
    file.read(record.data(), record.size());
    file.seekp(2 * kPage);
    file.write(record.data(), record.size());
    file.flush();
    Expect(file.good(), "replace row with another valid-CRC record");
  }
  auto misplaced = EmbeddingRowStore::Open(*reader, bad, kPage, &error);
  Expect(misplaced != nullptr, "misplaced row setup");
  Throws([&] { misplaced->Fetch(std::array<std::uint32_t, 1>{1}, one, false); },
         "row ID check rejects a valid-CRC record at the wrong offset");
  misplaced.reset();
  fs::copy_file(rows, bad, fs::copy_options::overwrite_existing);
  auto truncated = EmbeddingRowStore::Open(*reader, bad, kPage, &error);
  Expect(truncated != nullptr, "runtime-truncation setup");
  fs::resize_file(bad, kPage + 17);
  Throws([&] { truncated->Fetch(std::array<std::uint32_t, 1>{0}, one, false); },
         "runtime truncation is a request error");
  Throws(
      [&] {
        truncated->Fetch(std::array<std::uint32_t, 1>{kRows - 1}, one, false);
      },
      "missing final record is a request error");

  ::unsetenv("GUFO_DEV_EMBED_SSD_CACHE_MB");
  Expect(EmbeddingRowStore::CacheBytesFromEnvironment() == 16ULL << 20,
         "approved default 16 MiB");
  ::setenv("GUFO_DEV_EMBED_SSD_CACHE_MB", "3", 1);
  Expect(EmbeddingRowStore::CacheBytesFromEnvironment() == 3ULL << 20,
         "cache environment parsed");
  for (const auto* value :
       {"", "0", "-1", "1x", "999999999999999999999999999999"}) {
    ::setenv("GUFO_DEV_EMBED_SSD_CACHE_MB", value, 1);
    Throws([] { (void)EmbeddingRowStore::CacheBytesFromEnvironment(); },
           "invalid cache environment refused");
  }
  ::unsetenv("GUFO_DEV_EMBED_SSD_CACHE_MB");

  const auto spans =
      EmbeddingResidentSpans(10 * kPage, kPage + 32, 5 * kPage, kPage);
  Expect(spans.size() == 2 && spans[0].offset == 0 &&
             spans[0].size == 2 * kPage && spans[1].offset == 6 * kPage &&
             spans[1].size == 4 * kPage,
         "only interior pages excluded, suffix retains ORIGINAL file offset");
  const auto tiny = EmbeddingResidentSpans(2 * kPage, 32, 64, kPage);
  Expect(tiny.size() == 1 && tiny[0].size == 2 * kPage,
         "shared boundary pages retained");
  Expect(EmbeddingResidentSpans(kPage, 0, kPage, kPage).empty(),
         "whole isolated tensor excluded");
  Throws([] { (void)EmbeddingResidentSpans(kPage, kPage + 1, 0, kPage); },
         "bad interval refused");
  Throws(
      [] {
        (void)EmbeddingResidentSpans(
            kPage, 1, std::numeric_limits<std::size_t>::max(), kPage);
      },
      "interval overflow refused");
  Throws([] { (void)EmbeddingResidentSpans(kPage, 0, 1, 0); },
         "zero page size refused");
  std::cout << "PASS: synthetic GGUF row identity; random500; prefill97 "
               "unique/291 rows; LRU; shared clients; CRC32C; "
               "identity/extent/read failures; spans; environment\n";
}

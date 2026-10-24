#include <iostream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/embedding_rows.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: export_embedding_rows MODEL.gguf NEW.embrows\n"
                 "Exports Q4_K token rows only; destination must not exist.\n";
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader || !gufo::models::qwen::EmbeddingRowStore::Export(
                     *reader, argv[2], &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << "Exported " << argv[2] << '\n';
  return 0;
}

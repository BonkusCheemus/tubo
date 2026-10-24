// NEGATIVE CONTROL probe.
//
// WHAT THIS IS FOR. The unit C1 dev switch (GUFO_DEV_TURBO4_DECODE) must be OFF
// by default, and when it is OFF every refusal RequireExecutableKvStorage
// produces must be BYTE-IDENTICAL to what the tree produced before the switch
// existed. Asserting that inside a normal test would only re-state the source:
// a test written after the change cannot prove the change changed nothing.
//
// So this file does not assert. It PRINTS. It captures the exact refusal text
// for every storage value, and the stdout of this binary is the artifact that
// gets diffed across the change. The diff is the evidence; a green test is not.
//
// It deliberately uses the ONE-ARGUMENT form of RequireExecutableKvStorage,
// which is the form every prefill path calls. If the switch had been
// implemented inside the guard itself, this probe would have gone quiet and the
// diff would have caught it — which is the property worth having.
//
// DETERMINISM. No pointers, no addresses, no iteration over a hash container.
// The same bytes must come out on every run or the diff is noise.
// The switch is unset at the top of main() so an ambient GUFO_DEV_TURBO4_DECODE
// in the shell cannot make the control lie.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "src/models/qwen/hip/kv_allocation.hpp"

namespace {

const char *StorageName(gufo::hip::QwenKvCacheStorage storage) {
  using gufo::hip::QwenKvCacheStorage;
  switch (storage) {
    case QwenKvCacheStorage::kFp32:
      return "kFp32";
    case QwenKvCacheStorage::kFp16:
      return "kFp16";
    case QwenKvCacheStorage::kTurbo4:
      return "kTurbo4";
  }
  return "kUnknown";
}

void Report(gufo::hip::QwenKvCacheStorage storage) {
  std::printf("--- storage=%s\n", StorageName(storage));
  try {
    gufo::hip::RequireExecutableKvStorage(storage);
    std::printf("outcome=ACCEPTED\n");
  } catch (const std::invalid_argument &e) {
    std::printf("outcome=REFUSED\n");
    std::printf("exception=std::invalid_argument\n");
    std::printf("message_bytes=%zu\n", std::string(e.what()).size());
    std::printf("message=[%s]\n", e.what());
  } catch (const std::logic_error &e) {
    std::printf("outcome=REFUSED\n");
    std::printf("exception=std::logic_error\n");
    std::printf("message_bytes=%zu\n", std::string(e.what()).size());
    std::printf("message=[%s]\n", e.what());
  } catch (const std::exception &e) {
    std::printf("outcome=REFUSED\n");
    std::printf("exception=std::exception\n");
    std::printf("message=[%s]\n", e.what());
  }
}

}  // namespace

int main() {
  // The control is the UNSET state. Make that true regardless of the shell.
  ::unsetenv("GUFO_DEV_TURBO4_DECODE");
  const char *after = std::getenv("GUFO_DEV_TURBO4_DECODE");
  std::printf("probe=kv_refusal_baseline\n");
  std::printf("GUFO_DEV_TURBO4_DECODE=%s\n", after == nullptr ? "<unset>" : after);
  std::printf("sizeof_QwenKvCacheStorage=%zu\n",
              sizeof(gufo::hip::QwenKvCacheStorage));

  using gufo::hip::QwenKvCacheStorage;
  Report(QwenKvCacheStorage::kFp32);
  Report(QwenKvCacheStorage::kFp16);
  Report(QwenKvCacheStorage::kTurbo4);
  // The unknown enumerator: the second, distinct refusal. Both must be stable.
  Report(static_cast<QwenKvCacheStorage>(200));
  std::printf("end\n");
  return 0;
}
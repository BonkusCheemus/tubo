// Tagged IQ3 serving fences, CPU only. GPU side: the GPU executor fence profile admits
// tagged IQ3 gate/up ONLY with GUFO_DEV_SN_IQ3_SERVE=1 and a covered profile;
// default (switch unset) refuses. C8: the CPU generator refuses tagged weights
// at load, and TensorGEMV aborts instead of misreading a tagged weight.
// IQ3_PROFILE_HEADER_ONLY builds just the C9 half (iq3_serve_fence_guard.py
// compiles it standalone against mutated headers).
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "src/models/qwen/iq3_pair_view.hpp"
#ifndef IQ3_PROFILE_HEADER_ONLY
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <vector>

#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/generator.hpp"
#endif

using namespace gufo::models;
namespace {
int g_failures = 0;
void Expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  g_failures += !ok;
}

QwenModelWeights TaggedIq3(gufo::core::GgmlType type) {
  QwenModelWeights w;
  w.layers.resize(2);
  // Numeric root only; the profile never dereferences it.
  const auto views = MakeIq3PairViews(
      reinterpret_cast<const void*>(std::uintptr_t{0x1000}), type, 34816, 5120,
      std::size_t{1} << 30);
  w.layers[1].ffn_gate = views.gate;
  w.layers[1].ffn_up = views.up;
  return w;
}

void ProfileCases() {
  const QwenModelWeights native;
  Expect(!TaggedGpuServeRefusal(native, false, false), "native, switch off");
  Expect(!TaggedGpuServeRefusal(native, false, true), "native ignores B-cache");
  for (const auto type :
       {gufo::core::GgmlType::kIQ3_S, gufo::core::GgmlType::kIQ3_XXS}) {
    const auto iq3 = TaggedIq3(type);
    Expect(TaggedGpuServeRefusal(iq3, false, false), "IQ3 default refuses");
    Expect(!TaggedGpuServeRefusal(iq3, true, false), "IQ3 switch+covered");
    Expect(TaggedGpuServeRefusal(iq3, true, true), "IQ3 switch+B-cache");
    auto rows32 = iq3;
    rows32.layers[0].ssm_out.layout = QwenWeightLayout::kRows32;
    Expect(TaggedGpuServeRefusal(rows32, true, false), "IQ3+Rows32 refuses");
    auto half = iq3;
    half.layers[1].ffn_up.layout = QwenWeightLayout::kNative;
    Expect(TaggedGpuServeRefusal(half, true, false), "gate-only tag refuses");
    auto swapped = iq3;
    std::swap(swapped.layers[1].ffn_gate, swapped.layers[1].ffn_up);
    Expect(TaggedGpuServeRefusal(swapped, true, false), "swapped views refuse");
    auto small = iq3;
    const auto v = MakeIq3PairViews(small.layers[1].ffn_gate.data, type, 64,
                                    512, std::size_t{1} << 30);
    small.layers[1].ffn_gate = v.gate;
    small.layers[1].ffn_up = v.up;
    Expect(TaggedGpuServeRefusal(small, true, false), "unserved shape refuses");
  }
  // The fences call the env overload: only exactly "1" lifts, B-cache refuses.
  const auto iq3 = TaggedIq3(gufo::core::GgmlType::kIQ3_S);
  unsetenv("GUFO_DEV_SN_IQ3_SERVE");
  unsetenv("GUFO_NPU_BCACHE");
  unsetenv("GUFO_NPU_BCACHE_TAGGED");
  Expect(TaggedGpuServeRefusal(iq3), "env unset refuses");
  for (const char* off : {"0", "", "true", "11"}) {
    setenv("GUFO_DEV_SN_IQ3_SERVE", off, 1);
    Expect(TaggedGpuServeRefusal(iq3), "env switch not exactly 1 refuses");
  }
  setenv("GUFO_DEV_SN_IQ3_SERVE", "1", 1);
  Expect(!TaggedGpuServeRefusal(iq3), "env switch=1 admits");
  setenv("GUFO_NPU_BCACHE", "", 1);
  Expect(!TaggedGpuServeRefusal(iq3), "env empty B-cache admits");
  setenv("GUFO_NPU_BCACHE", "/tmp/bcache", 1);
  Expect(TaggedGpuServeRefusal(iq3), "env B-cache refuses");
  for (const char* off : {"0", "", "true", "11"}) {
    setenv("GUFO_NPU_BCACHE_TAGGED", off, 1);
    Expect(TaggedGpuServeRefusal(iq3), "B-cache, tagged key not exactly 1 refuses");
  }
  setenv("GUFO_NPU_BCACHE_TAGGED", "1", 1);
  Expect(!TaggedGpuServeRefusal(iq3), "B-cache + tagged key=1 admits");
  Expect(TaggedGpuServeRefusal(iq3, true, true), "core profile still refuses B-cache");
  unsetenv("GUFO_DEV_SN_IQ3_SERVE");
  unsetenv("GUFO_NPU_BCACHE");
  unsetenv("GUFO_NPU_BCACHE_TAGGED");
}

#ifndef IQ3_PROFILE_HEADER_ONLY
QwenModelWeights SmallModel() {
  QwenModelWeights w;
  auto& c = w.config;
  c.architecture = "qwen35";
  c.num_layers = 1;
  c.hidden_size = 4;
  c.intermediate_size = 8;
  c.num_attention_heads = 2;
  c.num_key_value_heads = 2;
  c.head_dim = 2;
  c.vocab_size = 8;
  c.context_length = 32;
  c.full_attention_interval = 1;
  c.is_text_only = true;
  w.layers.resize(1);
  return w;
}

void CpuCases() {
  bool built = false;
  try {
    QwenGenerator native(SmallModel(), nullptr, 32);
    built = true;
  } catch (const std::exception&) {
  }
  Expect(built, "CPU generator loads native weights");
  for (const auto layout :
       {QwenWeightLayout::kIq3SRows8, QwenWeightLayout::kRows32}) {
    auto tagged = SmallModel();
    tagged.layers[0].ffn_gate.layout = layout;
    bool refused = false;
    try {
      QwenGenerator generator(std::move(tagged), nullptr, 32);
    } catch (const std::invalid_argument& e) {
      refused = true;
      std::printf("  refusal: %s\n", e.what());
    }
    Expect(refused, "CPU generator refuses tagged weights at load");
  }
  // TensorGEMV: the native F32 control reads; a tagged ref must abort (in a
  // child) instead of running the native-row quant dot over SN bytes.
  std::vector<float> a(4 * 256, 1.0F), x(256, 1.0F), y(4, 0.0F);
  QwenTensorRef ref{.data = a.data(),
                    .type = gufo::core::GgmlType::kF32,
                    .num_elements = a.size()};
  TensorGEMV(ref, x, 4, 256, y);
  Expect(y[0] == 256.0F && y[3] == 256.0F, "TensorGEMV native control");
  for (const auto layout :
       {QwenWeightLayout::kIq3SRows8, QwenWeightLayout::kRows32}) {
    ref.layout = layout;
    std::fflush(stdout);
    const pid_t child = fork();
    if (child == 0) {
      TensorGEMV(ref, x, 4, 256, y);
      _exit(0);
    }
    int status = 0;
    waitpid(child, &status, 0);
    Expect(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
           "TensorGEMV aborts on a tagged weight");
  }
}
#endif
}  // namespace

int main() {
  try {
    ProfileCases();
#ifndef IQ3_PROFILE_HEADER_ONLY
    CpuCases();
#endif
  } catch (const std::exception& e) {
    std::printf("FAIL exception %s\n", e.what());
    return 1;
  }
  std::printf("%s  %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
  return g_failures ? 1 : 0;
}

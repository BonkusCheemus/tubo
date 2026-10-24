#!/usr/bin/env python3
"""CPU runs of the ACTUAL B1 FFN routing and C9 profile.

Not GPU execution. (1) The real modules/ffn.cpp HIP FfnForward is compiled
against recording stubs of its launchers: tagged IQ3 gate/up must take
LaunchSnIq3Pair at B1 into distinct gate/up scratch, then the pointwise SwiGLU,
then down; native keeps the fused kernel; aliased scratch and launcher refusals
must throw out of FfnForward (not std::terminate). (2) iq3_serve_profile_test.cpp
is compiled header-only against the real iq3_pair_view.hpp: default refuses,
switch on + covered admits, uncovered refuses. (3) Both executor fences call the
profile. Mutants of each part must fail.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]

GEMM_STUB = r'''
#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "src/models/qwen/state.hpp"
struct ihipStream_t;
using hipStream_t = ihipStream_t*;
namespace gufo::hip {
inline std::vector<std::string> calls;
inline const void* last[4];
inline std::size_t last_batch = 0;
inline bool refuse_pair = false;
inline void LaunchSnIq3Pair(const models::QwenTensorRef& g,
                            const models::QwenTensorRef& u, const float* x,
                            float* go, float* uo, float* packed,
                            std::size_t batch, hipStream_t) {
  if (refuse_pair) throw std::runtime_error("stub refusal");
  calls.push_back("pair");
  last[0] = x; last[1] = go; last[2] = uo; last[3] = packed;
  last_batch = batch;
  (void)g; (void)u;
}
inline void LaunchGEMV(const void*, core::GgmlType, const float*, float* y,
                       std::size_t, std::size_t, hipStream_t = nullptr) {
  calls.push_back("down"); last[3] = y;
}
}  // namespace gufo::hip
'''

SWIGLU_STUB = r'''
#pragma once
namespace gufo::hip {
inline void LaunchFusedSwiGLUGEMV(const void*, core::GgmlType, const void*,
                                  core::GgmlType, const float*, float*,
                                  std::size_t, std::size_t, hipStream_t) {
  calls.push_back("fused");
}
inline void LaunchBatchedSwiGLUActivation(const float* g, const float* u,
                                          float* out, void* bf16,
                                          std::size_t n, hipStream_t) {
  calls.push_back("swiglu");
  if (g != last[1] || u != last[2] || bf16 || n != 17408 || !out)
    calls.push_back("swiglu_bad_args");
}
}  // namespace gufo::hip
'''

FFN_MAIN = r'''
#include <cstdio>
#include "src/models/qwen/iq3_pair_view.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/modules/ffn.hpp"
namespace gufo::models {
void TensorGEMV(const QwenTensorRef&, std::span<const float>, std::size_t,
                std::size_t, std::span<float>) noexcept {}
}
using namespace gufo::models;
using gufo::hip::calls;
int main() {
  static float x[5120], g[17408], u[17408], act[17408], out[5120];
  const auto v = MakeIq3PairViews(
      reinterpret_cast<const void*>(std::uintptr_t{0x1000}),
      gufo::core::GgmlType::kIQ3_S, 34816, 5120, std::size_t{1} << 30);
  const QwenTensorRef down{.data = out, .type = gufo::core::GgmlType::kQ8_0,
                           .num_elements = 5120 * 17408};
  const qwen::HipModuleContext ctx;
  qwen::FfnLayerView tagged{v.gate, v.up, down, 5120, 17408};
  qwen::FfnForward(ctx, tagged, x, g, u, act, out);
  if (calls != std::vector<std::string>{"pair", "swiglu", "down"}) return 1;
  if (gufo::hip::last_batch != 1 || gufo::hip::last[0] != x ||
      gufo::hip::last[1] != g || gufo::hip::last[2] != u) return 2;
  calls.clear();
  auto native = tagged;
  native.gate = native.up = down;
  qwen::FfnForward(ctx, native, x, act, act, act, out);
  if (calls != std::vector<std::string>{"fused", "down"}) return 3;
  calls.clear();
  bool threw = false;
  try { qwen::FfnForward(ctx, tagged, x, act, act, act, out); }
  catch (const std::invalid_argument&) { threw = true; }
  if (!threw || !calls.empty()) return 4;  // decode_step's old aliased call
  gufo::hip::refuse_pair = true;
  threw = false;
  try { qwen::FfnForward(ctx, tagged, x, g, u, act, out); }
  catch (const std::runtime_error&) { threw = true; }
  if (!threw) return 5;
  return 0;
}
'''


def compile_run(compiler, sources, includes, defines, directory, name):
    binary = Path(directory)/name
    cmd = [compiler, '-std=c++20', '-O1', *[f'-D{d}' for d in defines],
           *[f'-I{i}' for i in includes], *map(str, sources), '-o', str(binary)]
    if subprocess.run(cmd).returncode:
        return 'compile_error'
    return subprocess.run([str(binary)], capture_output=True).returncode


def check_ffn(compiler, directory, ffn_source, tag, ffn_header=None):
    stub = Path(directory)/tag
    ops = stub/'src/models/qwen/hip/ops'
    ops.mkdir(parents=True)
    if ffn_header:
        (stub/'src/models/qwen/modules').mkdir(parents=True)
        (stub/'src/models/qwen/modules/ffn.hpp').write_text(ffn_header)
    (ops/'gemm.hpp').write_text(GEMM_STUB)
    (ops/'swiglu.hpp').write_text(SWIGLU_STUB)
    (stub/'ffn.cpp').write_text(ffn_source)
    (stub/'main.cpp').write_text(FFN_MAIN)
    return compile_run(compiler, [stub/'ffn.cpp', stub/'main.cpp',
                                  ROOT/'src/core/quant/ggml_dequant.cpp'],
                       [stub, ROOT], ['ENGINE_ENABLE_HIP=1'], directory,
                       tag + '.bin')


def check_profile(compiler, directory, header, tag):
    inc = Path(directory)/tag
    (inc/'src/models/qwen').mkdir(parents=True)
    (inc/'src/models/qwen/iq3_pair_view.hpp').write_text(header)
    # -I order: the mutated copy shadows the real header for every include.
    return compile_run(compiler,
                       [ROOT/'tests/models/qwen/iq3_serve_profile_test.cpp',
                        ROOT/'src/core/quant/ggml_dequant.cpp'],
                       [inc, ROOT], ['IQ3_PROFILE_HEADER_ONLY'], directory,
                       tag + '.bin')


def check_fences(executor, loader, decode_step, gemm, batched):
    call = 'models::TaggedGpuServeRefusal(model->GetWeights())'
    req = executor[executor.index('const QwenGpuModel& RequireModel('):]
    req = req[:req.index('return *model;')]
    assert call in req and 'throw std::invalid_argument(why);' in req, \
        'executor ctor fence'
    assert 'HasNonNativeWeights' not in req, 'executor fence bypass'
    create = loader[loader.index('QwenGpuExecutor::Create('):]
    create = create[:create.index('return std::make_unique<QwenGpuExecutor>')]
    assert call in create and '*error_msg = why;' in create, 'loader fence'
    assert 'HasNonNativeWeights' not in create, 'loader fence bypass'
    users = subprocess.run(['grep', '-rln', 'GUFO_DEV_SN_IQ3_SERVE', 'src'],
                           cwd=ROOT, capture_output=True, text=True).stdout
    assert users.split() == ['src/models/qwen/iq3_pair_view.hpp'], users
    # B1 reaches the tagged routes; decode_step hands FfnForward real scratch.
    ffn = re.sub(r'\s+', ' ', decode_step)
    assert ('ffn_view, decode_scratch.normed, ffn_scratch.gate, '
            'ffn_scratch.up, ffn_scratch.activation, ffn_scratch.out)') in ffn
    pair = gemm[gemm.index('void LaunchSnIq3Pair('):]
    pair = pair[:pair.index('\n}\n')]
    assert 'batch < 1 || batch > 8' in pair, 'pair launcher B1'
    assert 'case 1: launch.template operator()<1>(); break;' in pair, 'B1 case'
    act = batched[batched.index('void LaunchFfnActivation('):]
    assert 'batch_size < 1 || batch_size > 8' in act[:800], 'verify B1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    q = ROOT/'src/models/qwen'
    texts = {k: (q/p).read_text() for k, p in {
        'executor': 'hip/executor.cpp', 'loader': 'hip/model_loader.cpp',
        'decode_step': 'hip/decode_step.cpp',
        'gemm': 'hip/kernels/prefill_quant_gemm.hip',
        'batched': 'hip/batched_decode.cpp'}.items()}
    check_fences(**texts)
    fence_mutants = {
        'loader_fence_dropped': dict(texts, loader=texts['loader'].replace(
            'models::TaggedGpuServeRefusal(model->GetWeights())', 'nullptr', 1)),
        'decode_aliased_scratch': dict(texts, decode_step=texts[
            'decode_step'].replace('ffn_scratch.gate,', 'ffn_scratch.activation,', 1)),
        'no_b1_case': dict(texts, gemm=texts['gemm'].replace(
            'case 1: launch.template operator()<1>(); break;', '', 1)),
    }
    for name, mutant in fence_mutants.items():
        try:
            check_fences(**mutant)
        except (AssertionError, ValueError):
            print(f'source mutant {name}: rejected')
            continue
        raise AssertionError(f'source mutant {name} passed')

    ffn = (q/'modules/ffn.cpp').read_text()
    header = (q/'iq3_pair_view.hpp').read_text()
    ffn_mutants = {
        'tagged_takes_fused': ffn.replace(
            'if (view.gate.layout != QwenWeightLayout::kNative ||\n'
            '      view.up.layout != QwenWeightLayout::kNative) {',
            'if (false) {', 1),
        'b2_not_b1': ffn.replace('nullptr, 1, stream);', 'nullptr, 2, stream);', 1),
        'noexcept_back': ffn.replace('std::span<float> out) {\n  const auto stream',
                                     'std::span<float> out) noexcept {\n  const auto stream', 1),
        'no_alias_check': ffn.replace(
            '|| gate_scratch.data() == up_scratch.data())', ')', 1),
    }
    header_mutants = {
        'switch_default_on': header.replace(
            '  if (!serve_switch)\n    return "Rows32 Qwen GPU readers are not '
            'integrated yet";\n', '', 1),
        'env_any_value': header.replace('std::string_view(serve) == "1"',
                                        'serve[0] != \'\\0\'', 1),
        'bcache_ignored': header.replace('  if (npu_bcache)\n', '  if (false)\n', 1),
        'pair_unchecked': header.replace(
            'if (!IsSnIq3VerifyPair(layer.ffn_gate, layer.ffn_up))', 'if (false)', 1),
        'other_tags_ignored': header.replace('return rest.HasNonNativeWeights()',
                                             'return false', 1),
    }
    with tempfile.TemporaryDirectory() as directory:
        rc = check_ffn(args.compiler, directory, ffn, 'ffn_actual')
        print(f'ffn actual: rc={rc}')
        assert rc == 0, 'FfnForward B1 routing'
        ffn_hpp = (q/'modules/ffn.hpp').read_text()
        noexcept_hpp = ffn_hpp.replace(
            ');  // throws: tagged IQ3 refusals', ') noexcept;', 1)
        assert noexcept_hpp != ffn_hpp
        for name, mutant in ffn_mutants.items():
            assert mutant != ffn, name
            rc = check_ffn(args.compiler, directory, mutant, 'ffn_' + name,
                           noexcept_hpp if name == 'noexcept_back' else None)
            print(f'ffn mutant {name}: rc={rc}')
            assert rc != 0, name
        rc = check_profile(args.compiler, directory, header, 'profile_actual')
        print(f'profile actual: rc={rc}')
        assert rc == 0, 'C9 profile'
        for name, mutant in header_mutants.items():
            assert mutant != header, name
            rc = check_profile(args.compiler, directory, mutant, 'p_' + name)
            print(f'profile mutant {name}: rc={rc}')
            assert rc != 0, name
    print('PASS IQ3 B1 FFN route, C9 profile/fence, both fences wired')


if __name__ == '__main__':
    main()

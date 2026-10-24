#!/usr/bin/env python3
"""CPU run of the ACTUAL Dot4 scratch manager with HIP stubs.

Not GPU execution: hipMalloc/hipFree/hipStreamSynchronize/hipStreamIsCapturing
are counting stubs. Checks: the serving reserve covers every B<=8 consumer
without HIP calls during capture; capture-time growth throws (no grow, free,
sync or nullptr fallback); the old `return nullptr` must fail this check; the
executor constructor reserves before the only serving capture site.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]


def block(source, start):
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


SHIM = r'''
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
struct float2 { float x, y; };
struct Stream;
using hipStream_t = Stream*;
enum hipError_t { hipSuccess = 0 };
enum hipStreamCaptureStatus { hipStreamCaptureStatusNone = 0,
                              hipStreamCaptureStatusActive = 1 };
int capturing = 0, mallocs = 0, frees = 0, syncs = 0;
hipError_t hipStreamIsCapturing(hipStream_t, hipStreamCaptureStatus* s) {
  *s = capturing ? hipStreamCaptureStatusActive : hipStreamCaptureStatusNone;
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) { ++syncs; return hipSuccess; }
template<class T> hipError_t hipMalloc(T** p, std::size_t n) {
  ++mallocs; *p = static_cast<T*>(std::malloc(n)); return hipSuccess;
}
hipError_t hipFree(void* p) { ++frees; std::free(p); return hipSuccess; }
'''

MAIN = r'''
template<class F> bool Throws(F f) {
  try { f(); } catch (const std::runtime_error&) { return true; }
  return false;
}
int main() {
  auto* const serving = reinterpret_cast<hipStream_t>(0x10);
  auto* const unreserved = reinterpret_cast<hipStream_t>(0x20);
  ReserveDot4Scratch(8 * 17408, serving);  // executor ctor, 27B widths
  if (mallocs != 2 || syncs != 0 || frees != 0) return 1;
  auto* const q = GetDot4Scratch(serving, 1)->q;
  capturing = 1;
  // Consumers: B1 V3 cells, IQ3 pair 8*5120, native down B8 8*17408.
  for (std::size_t elems : {5120UL, 8UL * 5120, 17408UL, 8UL * 17408}) {
    auto* const s = GetDot4Scratch(serving, elems);
    if (s == nullptr || s->q != q) return 2;
  }
  if (mallocs != 2 || syncs != 0 || frees != 0) return 3;
  if (!Throws([&] { (void)GetDot4Scratch(serving, 8 * 17408 + 16); }))
    return 4;
  if (!Throws([&] { ReserveDot4Scratch(8 * 25600, serving); })) return 5;
  if (!Throws([&] { (void)GetDot4Scratch(unreserved, 16); })) return 6;
  if (mallocs != 2 || syncs != 0 || frees != 0) return 7;
  capturing = 0;
  if (GetDot4Scratch(serving, 8 * 17408)->q != q) return 8;
  return 0;
}
'''


def check_serving_order():
    executor = (ROOT/'src/models/qwen/hip/executor.cpp').read_text()
    ctor = block(executor, executor.index('QwenGpuExecutor::QwenGpuExecutor('))
    want = ('ReserveDot4Scratch(8U * std::max({c.hidden_size, '
            'c.intermediate_size,')
    assert want in re.sub(r'\s+', ' ', ctor), 'executor ctor must reserve'
    assert 'arena_.stream);' in ctor, 'reserve must target arena_.stream'
    # The ctor runs before any member function; the only Qwen capture is
    # TryCapture(arena_.stream, ...) in ForwardToken.
    captures = subprocess.run(
        ['grep', '-rl', 'hipStreamBeginCapture', 'src/models/qwen', 'src/core'],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert captures == ['src/core/hip/detail/hip_graph_decode_executor.hpp'], captures
    users = subprocess.run(['grep', '-rn', r'\.TryCapture(', 'src/models/qwen'],
                           cwd=ROOT, capture_output=True, text=True).stdout
    assert users.count('\n') == 1 and 'decode.cpp' in users, users
    assert 'TryCapture(arena_.stream,' in users, users


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    check_serving_order()
    gemm = (ROOT/'src/models/qwen/hip/kernels/prefill_quant_gemm.hip').read_text()
    start = gemm.index('struct Dot4Scratch {')
    end = gemm.index('void ReserveDot4Scratch(')
    manager = gemm[start:end] + block(gemm, end)
    mutant, count = re.subn(
        r'throw std::runtime_error\(\s*"Dot4 scratch growth during graph capture.*?"\);',
        'return nullptr;', manager)
    assert count == 1, 'cannot identify the capture refusal'
    with tempfile.TemporaryDirectory() as directory:
        cpp, binary = Path(directory)/'guard.cpp', Path(directory)/'guard'
        for name, code in [('actual', manager), ('return_nullptr', mutant)]:
            cpp.write_text(SHIM + code + MAIN)
            subprocess.run([args.compiler, '-std=c++20', '-O1', str(cpp), '-o',
                            str(binary)], check=True)
            rc = subprocess.run([str(binary)]).returncode
            print(f'{name}: rc={rc}')
            assert (rc == 0) == (name == 'actual'), name
    print('PASS Dot4 scratch: serving reserve before capture; capture growth throws')


if __name__ == '__main__':
    main()

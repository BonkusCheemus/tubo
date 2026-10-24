#!/usr/bin/env python3
"""Source contract for the tagged IQ3 prefill route.

ForwardPromptChunk must refuse the NPU B-cache (C7), unpack the tagged G/U pair
into the arena BF16 weight scratch, rebind gate/up to the native copies, and
only then reach the native readers C3-C6; no reader may still see the tagged
`layer.ffn_gate/up`. Mutants (stale tagged read, unpack after the readers,
dropped refusal) must fail. IQ3 must stay on the native WMMA route, so
gemm_weight never dequantizes the scratch into itself.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
READERS = ['TryLaunchBatchedDualQuantGEMMSwiGLUFp16(',
           'LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(',
           'LaunchBatchedDualQuantGEMMPreQuantized(',
           'gemm_weight(ffn_gate,', 'gemm_weight(ffn_up,',
           'LaunchBatchedQuantGEMMSwiGLUFp16(']


def check(source):
    start = source.index('dev_timing::Phase::kPrefillFfn')
    ffn = source[start:source.index('ffn_timer.Stop();', start)]
    refuse = ffn.index('throw std::runtime_error("tagged IQ3 prefill refuses '
                       'GUFO_NPU_BCACHE")')
    views = ffn.index('models::Iq3NativeScratchViews(')
    unpack = ffn.index('LaunchUnpackIq3View(native.pair,')
    rebind = ffn.index('ffn_gate = native.gate;')
    assert 'ffn_up = native.up;' in ffn[rebind:rebind + 80]
    assert refuse < views < unpack < rebind, 'refuse, views, unpack, rebind'
    readers = [ffn.index(r) for r in READERS]
    assert rebind < min(readers), 'a reader precedes the unpack'
    tail = ffn[rebind:]
    assert not re.search(r'layer\.ffn_(gate|up)\b', tail), 'stale tagged read'
    for r in READERS[:3]:
        call = tail[tail.index(r):]
        call = call[:call.index(');')]
        assert 'ffn_gate.data' in call and 'ffn_up.data' in call, r


def main():
    source = (ROOT/'src/models/qwen/hip/prefill_chunk.cpp').read_text()
    check(source)
    mutants = {
        'stale_tagged_read': source.replace(
            'ffn_gate.type, ffn_gate.data, ffn_up.data,',
            'ffn_gate.type, layer.ffn_gate.data, ffn_up.data,', 1),
        'no_bcache_refusal': source.replace(
            'refuses GUFO_NPU_BCACHE', 'ignores GUFO_NPU_BCACHE', 1),
        'unpack_after_readers': source.replace(
            'LaunchUnpackIq3View(native.pair,', 'Unused(native.pair,', 1).replace(
            '      // opt-c164-swiglu-quant:',
            '      LaunchUnpackIq3View(native.pair, 0, 0, 0);\n'
            '      // opt-c164-swiglu-quant:', 1),
    }
    for name, mutant in mutants.items():
        assert mutant != source, name
        try:
            check(mutant)
        except (AssertionError, ValueError):
            print(f'mutant {name}: rejected')
            continue
        raise AssertionError(f'mutant {name} passed')
    gemm = (ROOT/'src/models/qwen/hip/ops/gemm.hpp').read_text()
    wmma = gemm[gemm.index('constexpr bool IsNativeWmmaQuant('):]
    wmma = wmma[:wmma.index('return true;')]
    assert 'kIQ3_S:' in wmma and 'kIQ3_XXS:' in wmma, 'IQ3 left native WMMA'
    captures = subprocess.run(
        ['grep', '-rl', 'hipStreamBeginCapture', 'src/models/qwen', 'src/core'],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert captures == ['src/core/hip/detail/hip_graph_decode_executor.hpp'], captures
    users = subprocess.run(['grep', '-rln', r'\.TryCapture(', 'src/models/qwen'],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert users == ['src/models/qwen/hip/decode.cpp'], users
    print('PASS IQ3 prefill: refuse B-cache, unpack, rebind, then native readers')


if __name__ == '__main__':
    main()

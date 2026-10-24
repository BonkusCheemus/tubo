#!/usr/bin/env python3
"""Second oracle for tests/core/iquant_decode_test.cpp (extra llama.cpp quant types).

Usage:
    mkdir -p /tmp/iq && GUFO_IQUANT_DUMP_DIR=/tmp/iq build/<dir>/iquant_decode_test
    python3 tests/core/iquant_gguf_py_check.py /tmp/iq

Decodes the same random blocks with gguf-py (llama.cpp's python port, with its
own copy of the grid tables) and compares float bit patterns with gufo's CPU
decode. Needs numpy and gguf.
"""
import sys

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

CASES = {"Q2_K": T.Q2_K, "IQ2_XXS": T.IQ2_XXS, "IQ2_XS": T.IQ2_XS,
         "IQ2_S": T.IQ2_S, "IQ3_XXS": T.IQ3_XXS, "IQ1_M": T.IQ1_M}
BLOCK_BYTES = {"Q2_K": 84, "IQ2_XXS": 66, "IQ2_XS": 74, "IQ2_S": 82,
               "IQ3_XXS": 98, "IQ1_M": 56}


def main(directory):
    failed = False
    for name, qtype in CASES.items():
        raw = np.fromfile(f"{directory}/{name}.blocks", dtype=np.uint8)
        nb = raw.size // BLOCK_BYTES[name]
        data = raw.reshape(nb, BLOCK_BYTES[name])
        want = dequantize(data, qtype).astype(np.float32).reshape(-1)
        got = np.fromfile(f"{directory}/{name}.f32", dtype=np.float32)
        same = want.view(np.uint32) == got.view(np.uint32)
        print(f"{'OK  ' if same.all() else 'FAIL'} {name}: "
              f"{int(same.sum())}/{same.size} elements bit-identical to gguf-py")
        failed |= not same.all()
    return 1 if failed else 0


sys.exit(main(sys.argv[1]))

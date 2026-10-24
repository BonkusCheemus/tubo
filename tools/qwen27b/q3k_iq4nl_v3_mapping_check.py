#!/usr/bin/env python3
"""Host check of the v3 decode (Q3KHalfBlockDot, Iq4NlHalfBlockDot in small_batch_dot4.hpp).

Reference = gguf-py's dequantize (an independent port of ggml dequantize_row_q3_K / iq4_nl).
Mirror = the kernel's lane math, element by element: half h, slot s (0..7) covers
elements 128h + 16s .. +15 of a 256-element (virtual) super-block.
  Q3_K:   q = ((qs[32h + 16(s&1) + l] >> 2(s>>1)) & 3) | ((hmask[16(s&1) + l] >> (4h + (s>>1))) & 1) << 2,
          value = d * (sc[8h + s] - 32) * (q - 4), sc from ggml's 12-byte aux unpack (per-half words).
  IQ4_NL: block c = s>>1 of the half (bytes 18(4h + c)), nibble s&1, value = d_c * kvalues[nibble].
Random blocks (all byte values) + optional real blocks from a GGUF. Pure numpy, CPU.
  python3 q3k_iq4nl_v3_mapping_check.py [model.gguf]
"""
import sys
import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113])


def f16(b, off):
    return float(np.frombuffer(bytes(b[off:off + 2]), dtype=np.float16)[0])


def mirror_q3k(blk):
    hm, qs, sc12, d = blk[0:32], blk[32:96], blk[96:108], f16(blk, 108)
    aux = np.frombuffer(bytes(sc12), dtype='<u4').astype(np.uint64)
    tmp = int(aux[2])
    out = np.zeros(256)
    for h in range(2):
        if h == 0:
            w = [(int(aux[0]) & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4),
                 (int(aux[1]) & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4)]
        else:
            w = [((int(aux[0]) >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4),
                 ((int(aux[1]) >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4)]
        for s in range(8):
            sc = (w[s >> 2] >> (8 * (s & 3))) & 0xFF
            for l in range(16):
                low = (int(qs[32 * h + 16 * (s & 1) + l]) >> (2 * (s >> 1))) & 3
                hb = (int(hm[16 * (s & 1) + l]) >> (4 * h + (s >> 1))) & 1
                out[128 * h + 16 * s + l] = d * (sc - 32) * ((low | (hb << 2)) - 4)
    return out


def mirror_iq4nl(sb):  # sb = 144 bytes = 8 blocks
    out = np.zeros(256)
    for h in range(2):
        for s in range(8):
            c = s >> 1
            off = 18 * (4 * h + c)
            d = f16(sb, off)
            for l in range(16):
                nib = (int(sb[off + 2 + l]) >> (4 * (s & 1))) & 15
                out[128 * h + 16 * s + l] = d * KV[nib]
    return out


def check(name, typ, bsize, mirror, data):
    ref = dequantize(data, typ).astype(np.float64).reshape(-1, 256)
    sbs = data.reshape(-1, bsize)
    bad = 0
    for i in range(sbs.shape[0]):
        m = mirror(sbs[i])
        r = ref[i]
        ok = np.isfinite(r)
        if not np.allclose(m[ok], r[ok], rtol=1e-6, atol=1e-6 * (np.abs(r[ok]).max() + 1e-30)):
            bad += 1
    print(f"{name}: {sbs.shape[0]} super-blocks, mismatches {bad}")
    return bad


def main():
    rng = np.random.default_rng(1)
    bad = 0
    q3 = rng.integers(0, 256, (64, 110), dtype=np.uint8)
    q3[:, 108:110] = np.frombuffer(np.full(64, 0.01, np.float16).tobytes(), np.uint8).reshape(64, 2)
    bad += check("Q3_K random", Q.Q3_K, 110, mirror_q3k, q3.reshape(-1))
    nl = rng.integers(0, 256, (64, 8, 18), dtype=np.uint8)
    nl[:, :, 0:2] = np.frombuffer(rng.uniform(-0.02, 0.02, 64 * 8).astype(np.float16).tobytes(), np.uint8).reshape(64, 8, 2)
    bad += check("IQ4_NL random", Q.IQ4_NL, 144, mirror_iq4nl, nl.reshape(-1))
    if len(sys.argv) > 1:
        import gguf
        r = gguf.GGUFReader(sys.argv[1], "r")
        for t in r.tensors:
            if t.tensor_type == Q.Q3_K:
                bad += check(f"Q3_K {t.name}", Q.Q3_K, 110, mirror_q3k, np.asarray(t.data).reshape(-1)[:110 * 64])
                break
        for t in r.tensors:
            if t.tensor_type == Q.IQ4_NL:
                bad += check(f"IQ4_NL {t.name}", Q.IQ4_NL, 144, mirror_iq4nl, np.asarray(t.data).reshape(-1)[:144 * 64])
                break
    print("PASS" if bad == 0 else "FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

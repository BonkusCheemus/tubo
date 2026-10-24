#!/usr/bin/env python3
"""Host check of SmallBatchDot4KQuantV3Kernel's index/bit mapping (small_batch_dot4.hpp).

Reference = ggml dequantize_row_q5_K / q4_K (get_scale_min_k4). Mirror = the
kernel's lane math: half h owns bytes [16h, 16h+16) of each 32-byte qs group g,
nibble n gives sub-block 32-element index j = 2g+n, qh bit j, activation
sub16 = 4g + 2n + h, words little-endian. Checks that the mirror's per-element
weights equal the reference and that its affine dot (scale*xd*dot - offset*ds.y)
equals sum(w * dequantized x) for random blocks. Pure numpy, CPU.
  python3 q5k_v3_mapping_check.py
"""
import numpy as np


def scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def reference(d, dmin, scales, qh, qs, has_qh):
    out = np.zeros(256)
    for g in range(4):
        sc1, m1 = scale_min_k4(2 * g, scales)
        sc2, m2 = scale_min_k4(2 * g + 1, scales)
        for l in range(32):
            lo, hi = qs[32 * g + l] & 0xF, qs[32 * g + l] >> 4
            if has_qh:
                lo += 16 if qh[l] & (1 << (2 * g)) else 0
                hi += 16 if qh[l] & (1 << (2 * g + 1)) else 0
            out[64 * g + l] = d * sc1 * lo - dmin * m1
            out[64 * g + 32 + l] = d * sc2 * hi - dmin * m2
    return out


def words(b):  # 16 bytes -> 4 little-endian uint32
    return [int.from_bytes(bytes(int(v) for v in b[4 * i:4 * i + 4]), "little") for i in range(4)]


def mirror(d, dmin, scales, qh, qs, has_qh, xq, xd):
    """Returns per-element weights (index = sub16*16 + e) and the kernel's dot."""
    head_y, head_z, head_w = (int.from_bytes(bytes(int(v) for v in scales[4 * i:4 * i + 4]), "little") for i in range(3))
    w = np.zeros(256)
    total = 0.0
    for half in range(2):
        qhw = words(qh[16 * half:16 * half + 16]) if has_qh else [0] * 4
        for g in range(4):
            qsw = words(qs[32 * g + 16 * half:32 * g + 16 * half + 16])
            for n in range(2):
                j = 2 * g + n
                b = 8 * (j & 3)
                if j < 4:
                    sc, mn = (head_y >> b) & 63, (head_z >> b) & 63
                else:
                    sc = ((head_w >> b) & 15) | ((((head_y >> b) >> 6) & 3) << 4)
                    mn = (((head_w >> b) >> 4) & 15) | ((((head_z >> b) >> 6) & 3) << 4)
                q = []
                for i in range(4):
                    v = (qsw[i] >> (4 * n)) & 0x0F0F0F0F
                    if has_qh:
                        v |= ((qhw[i] >> j) & 0x01010101) << 4
                    q += [(v >> (8 * k)) & 0xFF for k in range(4)]
                sub = 4 * g + 2 * n + half
                for e in range(16):
                    w[sub * 16 + e] = d * sc * q[e] - dmin * mn
                dot = int(np.dot(q, xq[sub]))
                total += d * sc * (xd[sub] * dot) - dmin * mn * (xd[sub] * int(xq[sub].sum()))
    return w, total


def main():
    rng = np.random.default_rng(7)
    for has_qh in (True, False):
        for trial in range(200):
            d, dmin = rng.uniform(0.001, 0.05, 2)
            scales = rng.integers(0, 256, 12)
            qh = rng.integers(0, 256, 32)
            qs = rng.integers(0, 256, 128)
            ref = reference(d, dmin, scales, qh, qs, has_qh)
            # Activations: int8 per 16-element sub-block, as QuantizeActivationsDot4Kernel.
            x = rng.normal(size=256)
            xq = np.zeros((16, 16), dtype=np.int64)
            xd = np.zeros(16)
            for s in range(16):
                amax = np.abs(x[16 * s:16 * s + 16]).max()
                xd[s] = amax / 127
                xq[s] = np.clip(np.rint(x[16 * s:16 * s + 16] * 127 / amax), -127, 127)
            w, total = mirror(d, dmin, scales, qh, qs, has_qh, xq, xd)
            assert np.allclose(w, ref, rtol=0, atol=1e-12), (has_qh, trial)
            exact = float(np.dot(ref, (xq * xd[:, None]).ravel()))
            assert abs(total - exact) <= 1e-9 * max(1.0, abs(exact)), (has_qh, trial, total, exact)
    print("ok: v3 lane mapping == ggml dequant (Q5_K + Q4_K), affine dot exact, 400 random blocks")


if __name__ == "__main__":
    main()

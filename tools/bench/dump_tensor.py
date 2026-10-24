"""Dump one tensor's raw bytes from a GGUF (stdlib only) for verify_wmma_bench.

usage: python3 dump_tensor.py <gguf> <tensor name> <out file>
Prints: ggml_type_id m k (m = rows = ne[1], k = ne[0]).
"""
import struct
import sys

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLOCK = {8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 20: (32, 18), 23: (256, 136)}


def main(path, want, out):
    f = open(path, 'rb')
    rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
    rstr = lambda: f.read(rd('Q')).decode('utf-8', 'replace')

    def skip(t):
        if t == 8:
            rstr()
        elif t == 9:
            et = rd('I'); n = rd('Q')
            for _ in range(n):
                skip(et)
        else:
            f.read(SZ[t])
    assert f.read(4) == b'GGUF'
    rd('I'); nt = rd('Q'); nkv = rd('Q')
    align = 32
    for _ in range(nkv):
        key = rstr(); t = rd('I')
        if key == 'general.alignment':
            align = rd('I')
        else:
            skip(t)
    infos = []
    for _ in range(nt):
        name = rstr(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; t = rd('I'); off = rd('Q')
        infos.append((name, dims, t, off))
    base = (f.tell() + align - 1) // align * align
    for name, dims, t, off in infos:
        if name == want:
            k, m = dims[0], dims[1]
            e, b = BLOCK[t]
            n = m * (k // e) * b
            f.seek(base + off)
            open(out, 'wb').write(f.read(n))
            print(t, m, k)
            return
    raise SystemExit(f'no tensor {want}')


if __name__ == '__main__':
    main(*sys.argv[1:4])

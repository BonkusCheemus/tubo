"""Emit the Q4_K/Q5_K v3 dispatch table from a dflash_gemm_bench run.

Per cell (type, m, k, batch): median ms over rounds per geometry. A cell gets
v3 only when the best v3 geometry beats the best EXACT geometry (any name not
starting with dot4) by at least MIN_GAIN; every other cell keeps the exact
kernel. Exact-best >= production, so the margin is conservative. IQ4_XS
is different: production already runs dot4/dot4s for it at B4-8, so its
baseline is the best of EVERY non-v3 geometry, dot4 included. The
same baseline for Q4_K/Q5_K too -- at B8 production runs dot4s_r2w8 for them
(per a served-decode trace), so beating exact alone is not enough. Geometry suffix x2 / x4 =
RowsPerLane 2 / 4 (printed as a 7th field; x4 is IQ3_S/IQ3_XXS/Q4_K/Q5_K/IQ4_XS at Batch >= 6).

usage: python3 v3_table_from_bench.py <bench_dir> [min_gain=1.03]
Prints C++ initializer rows for kTable in FindKQuantV3Cell (prefill_quant_gemm.hip).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TYPES = {'q4': 'kQ4_K', 'q5': 'kQ5_K', 'iq4': 'kIQ4_XS', 'iq4-nl': 'kIQ4_NL', 'q3': 'kQ3_K', 'iq3-s': 'kIQ3_S', 'iq3-xxs': 'kIQ3_XXS', 'iq2-s': 'kIQ2_S'}  # iq4-nl, q3: added with the v3 Q3_K/IQ4_NL kernels
GEOMS = {'r2w4': (2, 4, 1), 'r4w2': (4, 2, 1), 'r4w4': (4, 4, 1), 'r4w8': (4, 8, 1),
         'r8w4': (8, 4, 1), 'r2w4x2': (2, 4, 2), 'r4w4x2': (4, 4, 2), 'r4w8x2': (4, 8, 2),
         'r8w4x2': (8, 4, 2), 'r2w4x4': (2, 4, 4), 'r4w4x4': (4, 4, 4), 'r4w8x4': (4, 8, 4),
         'r8w4x4': (8, 4, 4)}
# k suffix = ActK (q8_K activations): 4th field act_k = 1.
GEOMS.update({g + 'k': v + (1,) for g, v in list(GEOMS.items())})


def cells(bench_dir):
    for path in sorted(Path(bench_dir).glob('*_m*_k*_b*.csv')):
        rows = defaultdict(list)
        with path.open() as f:
            lines = [l for l in f if l.split(',', 1)[0] in TYPES]
        for r in csv.reader(lines):
            rows[(r[0], int(r[1]), int(r[2]), int(r[3]), r[5])].append(float(r[7]))
        yield rows


def pick(rows, min_gain):
    by_cell = defaultdict(dict)
    for (t, m, k, b, geom), ms in rows.items():
        by_cell[(t, m, k, b)][geom] = statistics.median(ms)
    out = []
    for cell, geoms in by_cell.items():
        exact = min((v, g) for g, v in geoms.items() if not g.startswith('dot4v3_'))
        v3 = [(v, g) for g, v in geoms.items() if g.startswith('dot4v3_')]
        if not v3:
            continue
        best_v3 = min(v3)
        gain = exact[0] / best_v3[0]
        if gain >= min_gain:
            out.append((cell, best_v3[1].removeprefix('dot4v3_'), gain, exact[1]))
    return out


def main():
    bench_dir = sys.argv[1]
    min_gain = float(sys.argv[2]) if len(sys.argv) > 2 else 1.03
    picked = []
    for rows in cells(bench_dir):
        picked += pick(rows, min_gain)
    picked.sort()
    for (t, m, k, b), geom, gain, exact in picked:
        rows_, waves, per_lane, *act = GEOMS[geom]
        ty = TYPES[t]
        tail = f', {per_lane}, 1' if act else (f', {per_lane}' if per_lane != 1 else '')
        print(f'    {{T::{ty}, {m}, {k}, {b}, {rows_}, {waves}{tail}}},  // {gain:.2f}x vs {exact}')
    print(f'// {len(picked)} cells, min_gain {min_gain}', file=sys.stderr)


def _selftest():
    rows = {('q5', 8, 256, 1, 'tile1', ): [2.0, 2.0], ('q5', 8, 256, 1, 'dot4v3_r4w4'): [1.0, 1.0],
            ('q5', 8, 256, 2, 'tile1'): [1.0], ('q5', 8, 256, 2, 'dot4v3_r4w4'): [1.0],
            ('q5', 8, 256, 3, 'tile1'): [1.0], ('q5', 8, 256, 3, 'dot4s_r2w8'): [0.1]}
    got = pick(rows, 1.03)
    assert [(c, g) for c, g, _, _ in got] == [(('q5', 8, 256, 1), 'r4w4')], got
    # iq4: a dot4 geometry counts as the baseline, so v3 must beat it too.
    rows = {('iq4', 8, 256, 8, 'typed'): [2.0], ('iq4', 8, 256, 8, 'dot4s_r2w8'): [1.0],
            ('iq4', 8, 256, 8, 'dot4v3_r8w4'): [0.99],
            ('iq4', 8, 256, 6, 'typed'): [2.0], ('iq4', 8, 256, 6, 'dot4_r2w8'): [1.0],
            ('iq4', 8, 256, 6, 'dot4v3_r4w8'): [0.5]}
    got = pick(rows, 1.03)
    assert [(c, g) for c, g, _, _ in got] == [(('iq4', 8, 256, 6), 'r4w8')], got
    # q5 now uses the same all-non-v3 baseline; an x2 geometry can win.
    rows = {('q5', 8, 256, 8, 'tile1'): [2.0], ('q5', 8, 256, 8, 'dot4s_r2w8'): [1.0],
            ('q5', 8, 256, 8, 'dot4v3_r4w8'): [0.99], ('q5', 8, 256, 8, 'dot4v3_r4w8x2'): [0.8]}
    got = pick(rows, 1.03)
    assert [(c, g) for c, g, _, _ in got] == [(('q5', 8, 256, 8), 'r4w8x2')], got
    # x4 geometries (IQ3 + Q4_K/Q5_K/IQ4_XS, Batch >= 6) must be present so an x4 cell
    # never KeyErrors at GEOMS[geom] in main; per_lane is the 7th table field.
    for g, want in (('r2w4x4', (2, 4, 4)), ('r4w4x4', (4, 4, 4)),
                    ('r4w8x4', (4, 8, 4)), ('r8w4x4', (8, 4, 4))):
        assert GEOMS[g] == want and GEOMS[g + 'k'] == want + (1,)


if __name__ == '__main__':
    _selftest()
    main()

#!/usr/bin/env python3
"""budget.py -- loss budget per weak kernel cell (analysis only, no GPU).

MEASURES: for each weight-matrix shape, where the time goes between "read the bytes at peak" and the measured
  batch-8 kernel time: launch ramp/tail/gap of one plain launch, kernel structure at batch 1, and what batch 8
  adds; plus the VALU cycles per wave-block against the SIMD budget at the target rate.
RUN:      python3 budget.py     (no arguments; asserts check that the budget closes)
INPUTS:   every number in the tables is a GB/s or an instruction count measured earlier on one machine
  (the pure-read column comes from the ceil.hip sweep). Replace them with your own measurements
  before drawing conclusions.
WRONG IF: an assert fails, or the pure-read column does not come from the same size, memory kind and clock as
  the kernel columns (the "fixed" and "B1-rd" terms then mix unlike things).
"""
import sys
if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(0)

PEAK, T95 = 253.4, 241.0  # GB/s; 241 = 95% of 253.4 (the target rate)

# bytes per launch = rows * (k/256) * block_bytes  (block_bytes: bytes per 256-weight block of the quantized format)
cells = {
    # name: (rows, k, block_B, B1 GB/s, B8 GB/s, plain-launch pure-read GB/s typical, source of read)
    "IQ3_XXS 34816x5120": (34816, 5120, 98, 222.3, 211.5, 245.5, "ceil 64 MB host typical (not 68 MB)"),
    "IQ3_S   34816x5120": (34816, 5120, 110, 226.8, 216.3, 245.5, "ceil 64 MB host typical (not 77 MB)"),
    "Q4_K    6144x5120 ": (6144, 5120, 144, 229.7, 213.8, 227.3, "ceil 17.7 MB host typical"),
    "Q5_K    5120x6144 ": (5120, 6144, 176, 228.7, 214.4, 231.9, "ceil 21.6 MB host typical"),
    "Q6_K    5120x6144 ": (5120, 6144, 210, 235.8, 214.6, 237.8, "ceil 25.8 MB host typical"),
}
IQ3_K4_STREAM = 234.6  # GB/s: stream-only kernel with the IQ3_S k4 load pattern, no math, in the GEMV benchmark

def us(nbytes, gbps):
    return nbytes / (gbps * 1e3)  # bytes / (GB/s) -> microseconds

print(f"{'cell':20} {'MB':>6} {'@253':>6} {'@241':>6} {'read':>6} {'B1':>6} {'B8':>6} "
      f"{'a:fix':>6} {'B1-rd':>6} {'B8-B1':>6} {'need':>6}")
for name, (rows, k, bb, b1, b8, rd, _src) in cells.items():
    nbytes = rows * (k // 256) * bb
    t253, t241, trd = us(nbytes, PEAK), us(nbytes, T95), us(nbytes, rd)
    t1, t8 = us(nbytes, b1), us(nbytes, b8)
    fixed = trd - t253          # launch ramp/tail/gap + DRAM effects of ONE plain launch
    b1_extra = t1 - trd         # kernel structure at B1 (load shape, decode, grid)
    grow = t8 - t1              # what B8 adds over B1
    need = t8 - t241            # microseconds to cut for 241
    # budget closes by construction: t253 + fixed + b1_extra + grow == t8
    assert abs(t253 + fixed + b1_extra + grow - t8) < 1e-9
    print(f"{name:20} {nbytes/1e6:6.2f} {t253:6.1f} {t241:6.1f} {trd:6.1f} {t1:6.1f} {t8:6.1f} "
          f"{fixed:6.1f} {b1_extra:6.1f} {grow:6.1f} {need:6.1f}"
          + ("   241 < plain pure read: NOT reachable in one plain launch" if rd < T95 else ""))

# VALU share of SIMD time at 241 GB/s (ISA counts; dot4 = 1.63 fmac slots; other VALU 1.0-1.42 slots)
print("\nVALU per wave-block at B8 vs SIMD budget at 241 GB/s (80 SIMD, 1.7 GHz)")
isa = {  # name: (bytes per wave-block, dot4, other VALU, KIND)
    "IQ3_XXS snx_k4ns": (32 * 98, 512, 302, "ISA measured (instruction census)"),
    "IQ3_S k4": (32 * 110, 512, 374, "ISA measured (instruction census)"),
    "Q4_K hp": (32 * 144, 512, 390, "INFERRED (no ISA dump; Q5 minus qh merge)"),
    "Q5_K hp": (32 * 176, 512, 466, "ISA measured (instruction census)"),
    "Q6_K hp": (32 * 210, 512, 560, "INFERRED (no ISA dump; 16 scales/256)"),
}
for name, (wb, dot4, other, kind) in isa.items():
    budget = 80 * wb * 1.7e9 / (T95 * 1e9)
    lo, hi = dot4 * 1.63 + other * 1.0, dot4 * 1.63 + other * 1.42
    print(f"{name:18} budget {budget:6.0f} cyc  need {lo:5.0f}-{hi:5.0f}  = {lo/budget:4.0%}-{hi/budget:4.0%}  [{kind}]")

# sanity: 241 is 95% of peak, and Q5 bytes match the logged 21,626,880
assert abs(T95 / PEAK - 0.951) < 0.001
assert 5120 * 24 * 176 == 21_626_880

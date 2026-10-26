"""period.py <log> <cfg> <bin_us> -- look for a periodic pattern in the dips of a trace log written with BINS=1.

MEASURES: the autocorrelation of the per-bin GB/s deviation from each round's mean, averaged over rounds; prints
  the top lags (in microseconds) with their correlation, the correlation at a few fixed lags, and the lengths of
  runs of bins more than 4 GB/s below the round median. A real period shows as a lag with r well above the rest.
RUN:      BINS=1 BIN_US=50 ./m99 trace <bytes> host <cfg> > trace.log; python3 period.py trace.log <cfg> 50
WRONG IF: <bin_us> differs from the BIN_US the trace was recorded with (lags are then mislabelled), or the best
  lag has r below about 0.1 (no period; the listed lags are noise).
"""
import sys, statistics as st
if len(sys.argv) < 4 or sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(0)
log, cfg, bu = sys.argv[1], sys.argv[2], float(sys.argv[3])
rounds = [[float(x) for x in ln.strip().split(',')[3:]] for ln in open(log) if ln.startswith('bins,' + cfg + ',')]
L = min(len(r) for r in rounds); maxlag = L // 2
acc = [0.0] * maxlag
for r in rounds:
    m = st.mean(r); d = [x - m for x in r[:L]]; v = sum(x * x for x in d) or 1
    for lag in range(1, maxlag): acc[lag] += sum(d[i] * d[i + lag] for i in range(L - lag)) / v
acc = [a / len(rounds) for a in acc]
top = sorted(range(2, maxlag), key=lambda k: -acc[k])[:8]
print(f"{cfg}: rounds={len(rounds)} bins/round={L} top lags (us, r):", ", ".join(f"{k*bu:.0f}:{acc[k]:.3f}" for k in sorted(top)))
for k in (120, 121, 122, 60, 61): 
    if k < maxlag: print(f"  lag {k*bu:.0f} us r={acc[k]:.3f}")
# dip run lengths
runs = []
for r in rounds:
    med = st.median(r); n = 0
    for x in r:
        if x < med - 4: n += 1
        elif n: runs.append(n); n = 0
import collections; c = collections.Counter(runs); print("  dip run lengths (bins):", sorted(c.items())[:12])

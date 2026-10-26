"""binstat.py <log> -- statistics of a trace log written with BINS=1.

MEASURES: per cfg, from the 'bins' lines (GB/s per bin over the steady middle of each round): the mean (what the
  round delivers), the median and mode (the undisturbed rate), and the cost of dips (bins more than 4 GB/s
  below the median, replaced by the median to show what the round would deliver without them).
RUN:      BINS=1 ./m99 trace <bytes> host <cfg> > trace.log; python3 binstat.py trace.log
WRONG IF: the number of bins is small, or the mode bucket differs from the median by more than the bucket width
  (the distribution is not one peak plus dips, so "dip cost" is not meaningful).
"""
import sys, statistics as st, collections
if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(0)
d = collections.defaultdict(list)
for ln in open(sys.argv[1]):
    if ln.startswith('bins,'):
        f = ln.strip().split(','); d[f[1]] += [float(x) for x in f[3:]]
for cfg, b in d.items():
    m, med = st.mean(b), st.median(b)
    floor = med - 4.0   # a bin more than 4 GB/s under the median counts as a dip
    clipped = [max(x, med) if x < floor else x for x in b]
    hist = collections.Counter(int(x // 2) * 2 for x in b)
    mode = max(hist, key=hist.get)
    print(f"{cfg}: bins={len(b)} mean={m:.2f} median={med:.2f} mode_bucket={mode}-{mode+2} dips(<{floor:.0f})={sum(x<floor for x in b)/len(b)*100:.1f}% "
          f"mean_if_dips_removed={st.mean(clipped):.2f} dip_cost={st.mean(clipped)-m:.2f} GB/s")

"""agg.py <logs...> -- aggregate m99 'ab' logs.

MEASURES: per cfg, the median and range of the per-run 'final' values (each = median of 3 in-run passes x 7 rounds),
  over all runs and over runs not hit by a desktop GPU burst (burst = any cfg in that run < 230 GB/s; the
  threshold is for a ~250 GB/s part, change it for other hardware). pct assumes a 256 GB/s theoretical peak.
RUN:      python3 agg.py run1.log run2.log ...
WRONG IF: the number of quiet runs is small (the median is then not stable), or min and max of the quiet runs
  differ by more than a few GB/s (the machine was not quiet even when no burst was detected).
"""
import sys, csv, statistics as st, collections
if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(0)
allr, quiet = collections.defaultdict(list), collections.defaultdict(list)
for f in sys.argv[1:]:
    fin = {r[1]: float(r[3]) for r in csv.reader(open(f)) if r and r[0] == 'final' and r[1] != 'cfg'}
    burst = min(v for k, v in fin.items() if 'l0' not in k) < 230
    for k, v in fin.items():
        allr[k].append(v)
        if not burst: quiet[k].append(v)
for k in allr:
    q = quiet[k]
    print(f"{k}: all n={len(allr[k])} med={st.median(allr[k]):.2f} | quiet n={len(q)} med={st.median(q):.2f} min={min(q):.2f} max={max(q):.2f} pct={st.median(q)/2.56:.2f}%")

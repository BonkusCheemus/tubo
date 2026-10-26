"""trsum.py <trace log> -- summarise a compact trace (output of: TRACE_COMPACT=1 magic trace ...).

MEASURES: per arm: median launch duration, median completion-to-completion period, median gap between a launch's
  start and the previous launch's end (negative = the two overlap), and event GB/s per round.
RUN:      TRACE_COMPACT=1 ./magic trace <bytes> <host|dev> <arm> > trace.log; python3 trsum.py trace.log
WRONG IF: the period is not about duration + gap (the trace dropped launches), or an arm labelled overlapping
  shows a positive median gap (it did not overlap, so its speed gain is not from overlap).
"""
import sys, statistics as st
if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(0)
arm = None; data = {}
for l in open(sys.argv[1]):
    if l.startswith('# arm '): arm = l.split()[2]; data[arm] = {'L': [], 'R': []}
    elif l.startswith('L,') and arm: r, i, t, d, g = l.strip().split(',')[1:]; data[arm]['L'].append((int(r), int(i), float(t), float(d)))
    elif l.startswith('R,') and arm: data[arm]['R'].append(float(l.strip().split(',')[2]))
for a, v in data.items():
    L = v['L']; per = []; gap = []
    for k in range(1, len(L)):
        if L[k][0] == L[k-1][0]:
            per.append((L[k][2] + L[k][3]) - (L[k-1][2] + L[k-1][3])); gap.append(L[k][2] - (L[k-1][2] + L[k-1][3]))
    print(f"{a}: dur_med={st.median(x[3] for x in L):.2f}us  end-to-end period_med={st.median(per):.2f}us "
          f"start-minus-prev-end_med={st.median(gap):.2f}us  event GBps median={st.median(v['R']):.1f} (traced kernel)")

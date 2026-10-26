"""merge.py <prefix> -- merge <prefix><pass>_<bytes>_<mem>.log files (prefix f for runall.sh, z for runfinal.sh).

MEASURES: per case x arm: median over passes of the call median (GB/s), best pass, and the median paired gain
  versus the plain baseline g_256_80_1 measured in the same call. Also the load average of each call.
RUN:      python3 merge.py f      (default prefix f; prints a markdown table)
WRONG IF: FAULT or INVALID lines are printed; the load average of any call is high; or the paired gain is
  smaller than the spread between passes.
"""
import glob, re, statistics as st, collections, sys
if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(0)
pat = sys.argv[1] if len(sys.argv) > 1 else 'f'
d = collections.defaultdict(lambda: collections.defaultdict(list)); gain = collections.defaultdict(lambda: collections.defaultdict(list))
loads = collections.defaultdict(list); bad = 0
for f in sorted(glob.glob(pat + '[0-9]*_*.log')):
    m = re.match(pat + r'(\d+)_(\d+)_(host|dev)\.log', f.split('/')[-1]);
    if not m: continue
    case = (int(m.group(2)), m.group(3)); rows = {}
    for l in open(f):
        if l.startswith('# ') and ' load ' in l: loads[case].append(float(l.split(' load ')[1].split()[0]))
        if 'kernel log since' in l: continue
        if re.search(r'ring|reset|page fault', l, re.I) and not l.startswith('#'): bad += 1; print('FAULT', f, l.strip())
        p = l.strip().split(',')
        if len(p) >= 6 and p[1] in ('0', '1') and re.match(r'^[gtspqrw]_', p[0]):
            if p[1] != '1': print('INVALID', f, p[0]); continue
            rows[p[0]] = float(p[2])
    for a, v in rows.items():
        d[case][a].append(v)
        if 'g_256_80_1' in rows: gain[case][a].append(v / rows['g_256_80_1'] - 1)
print('faults:', bad)
print('| size | mem | arm | median of pass medians GB/s | best pass | gain vs plain (median, same call) | passes |')
for case in sorted(d):
    for a in sorted(d[case], key=lambda a: -st.median(d[case][a])):
        v = d[case][a]
        print(f"| {case[0]} | {case[1]} | {a} | {st.median(v):.1f} | {max(v):.1f} | {100*st.median(gain[case][a]):+.1f}% | {' '.join('%.0f'%x for x in v)} |")
    print(f"| {case[0]} | {case[1]} | loadavg per call | {' '.join('%.1f'%x for x in loads[case])} |")

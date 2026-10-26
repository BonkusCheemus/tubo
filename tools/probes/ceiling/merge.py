"""merge.py -- summarise the ceil sweep logs (p<pass>_<bytes>_<mem>[_wg<N>|_copies1].log in the current directory).

MEASURES: per (size, memory kind, variant): for every launch config the median over passes of GB/s;
  the best config by that median, its per-pass values, and the top-5 config medians.
RUN:      python3 merge.py        (prints a markdown table)
WRONG IF: a BAD line is printed (a config failed its checksum), or the per-pass values of the best config
  spread by more than a few GB/s (the sweep ran on a busy machine).
"""
import glob,re,statistics as st,collections,sys
if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(0)
d=collections.defaultdict(lambda:collections.defaultdict(list))
for f in sorted(glob.glob('p[0-9]_*.log')):
    tag=re.sub(r'^p\d_','',f[:-4]); tag=re.sub(r'_wg\d+$','',tag)
    for l in open(f):
        if not l.startswith(('host,','dev,')): continue
        p=l.strip().split(',')
        if p[-1]!='1': print('BAD',f,l); continue
        key=tuple(p[3:8]); d[tag][key].append((float(p[9]),float(p[10]),float(p[8])))
print('| case | best cfg (wg,grid,unr,style,nt) | median-of-passes GB/s | passes (GB/s) | min-ms GB/s | all-config spread top5 |')
for tag in sorted(d):
    rows=[]
    for k,v in d[tag].items():
        rows.append((st.median(x[0] for x in v),k,[x[0] for x in v],[x[1] for x in v]))
    rows.sort(reverse=True)
    m,k,ps,mins=rows[0]
    bytes_=int(tag.split('_')[0]); mn=bytes_/(min(mins) if False else 1)
    print(f"| {tag} | {','.join(k)} | {m:.1f} | {', '.join('%.1f'%x for x in ps)} | n/a | {', '.join('%.0f'%r[0] for r in rows[:5])} |")

# Loop census of the OpusIq3sB8* kernels in an ISA file.
# usage: cen.py file.s [dump]   (dump -> loop_<label>.s with the loop body)
import re, collections, sys
txt = open(sys.argv[1]).read().split('\n')
starts = []
for i, l in enumerate(txt):
    m = re.match(r'^(_ZN4gufo3hip\d+OpusIq3sB8(Cut|Vopd)?Kernel(I\S*?)E[Ev]\S*):', l)
    if m:
        starts.append((i, m.group(1), (m.group(2) or "K4") + m.group(3)))
for i, n, targs in starts:
    j = i
    while not txt[j].startswith('.Lfunc_end'):
        j += 1
    body = txt[i:j]
    meta = {}
    for l in txt[j:j + 60]:
        for pat in ('NumSgprs', 'NumVgprs', 'ScratchSize', 'Occupancy'):
            if l.startswith('; ' + pat + ':') or l.startswith('; Total' + pat + ':'):
                meta[pat] = l.split(':')[1].strip()
    idx = []; labs = {}
    for l in body:
        s = l.split(';')[0].strip()
        if not s:
            continue
        if re.match(r'^\.LBB\S+:', s):
            labs[s.split(':')[0]] = len(idx); continue
        if s.startswith('.'):
            continue
        idx.append(s)
    best = None
    for p, s in enumerate(idx):
        m = re.match(r's_cbranch\w*\s+(\.LBB\S+)', s) or re.match(r's_branch\s+(\.LBB\S+)', s)
        if m and m.group(1) in labs and labs[m.group(1)] <= p:
            span = p - labs[m.group(1)]
            if best is None or span > best[0]:
                best = (span, labs[m.group(1)], p)
    sp, a, b = best
    loop = idx[a:b + 1]
    c = collections.Counter(s.split()[0] for s in loop)
    cnt = lambda pred: sum(v for k, v in c.items() if pred(k))
    label = re.sub(r'[^A-Za-z0-9]+', '_', targs)[:60]
    print(label, meta, 'total', len(idx), 'loop', len(loop), 'valu', cnt(lambda k: k.startswith('v_')),
          'dot4', cnt(lambda k: 'dot4' in k), 'dual', cnt(lambda k: k.startswith('v_dual')),
          'ds_load', cnt(lambda k: k.startswith('ds_load')), 'sload', cnt(lambda k: k.startswith('s_load')),
          'gload', cnt(lambda k: k.startswith('global_load')), 'scratch', cnt(lambda k: k.startswith('scratch')),
          'dswait', cnt(lambda k: k.startswith('s_waitcnt')))
    if len(sys.argv) > 2:
        open('loop_%s.s' % label, 'w').write('\n'.join(loop))

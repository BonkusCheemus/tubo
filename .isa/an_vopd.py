#!/usr/bin/env python3
"""Per-kernel ISA census of the OpusIq3sB8VopdKernel instances in an -S file.
usage: an_vopd.py file.s   -> main K loop (largest backward-branch region) counts."""
import re, sys, collections
txt = open(sys.argv[1]).read().split('\n')
funcs = {}
cur = None
for i, l in enumerate(txt):
    m = re.match(r'^(_Z\S+):\s*(;.*)?$', l)
    if m and ('Vopd' in m.group(1) or re.search(r'OpusIq3sB8KernelILm[148]ELj4ELj4ELb0ELb0ELb0ELb0ELb0E', m.group(1))):
        cur = m.group(1); funcs[cur] = []; continue
    if cur and l.startswith('.Lfunc_end'):
        cur = None; continue
    if cur: funcs[cur].append(l)
meta = {}
cur = None
pend = None
for l in txt:
    m = re.match(r'^\s+\.group_segment_fixed_size:\s+(\d+)', l)
    if m: pend = int(m.group(1))
    m = re.match(r'^\s+\.name:\s+(\S+)', l)
    if m:
        cur = m.group(1)
        if pend is not None: meta.setdefault(cur, {})['group_segment_fixed_size'] = pend; pend = None
    for key in ('sgpr_count','vgpr_count','sgpr_spill_count','vgpr_spill_count','private_segment_fixed_size'):
        m = re.match(r'^\s+\.%s:\s+(\d+)' % key, l)
        if m and cur: meta.setdefault(cur, {})[key] = int(m.group(1))
for name, body in funcs.items():
    lab = {}
    for i, l in enumerate(body):
        m = re.match(r'^(\.LBB\S+):', l)
        if m: lab[m.group(1)] = i
    best = None
    for i, l in enumerate(body):
        m = re.match(r'^\s+s_cbranch\w*\s+(\.LBB\S+)', l) or re.match(r'^\s+s_branch\s+(\.LBB\S+)', l)
        if m and m.group(1) in lab and lab[m.group(1)] < i:
            span = i - lab[m.group(1)]
            if best is None or span > best[1] - best[0]: best = (lab[m.group(1)], i)
    if not best: print(name, 'no loop'); continue
    lines = [l.strip() for l in body[best[0]:best[1]+1]]
    ins = [l for l in lines if l and not l.startswith(('.', ';')) and not re.match(r'^\.?LBB', l) and not l.endswith(':')]
    c = collections.Counter()
    for l in ins:
        op = l.split()[0]
        c[op] += 1
    dual = sum(v for k, v in c.items() if k.startswith('v_dual_'))
    dual_dot = sum(1 for l in ins if l.startswith('v_dual_dot2acc') and '::' in l and l.split('::')[1].strip().startswith('v_dual_dot2acc'))
    valu = sum(v for k, v in c.items() if k.startswith('v_'))
    d = lambda p: sum(v for k, v in c.items() if k.startswith(p))
    m = re.search(r'ILm(\d+)ELj4ELj4ELb(\d)(?:ELb(\d))?', name)
    k4 = 'Vopd' not in name
    mt = meta.get(name, {})
    # mangled name in meta ends with .kd? try prefix
    mt = meta.get(name, {})
    print(f"== B{m.group(1)} {'k4' if k4 else ('CV' if m.group(2)=='1' else 'T16')}{' Pad8' if (not k4) and m.group(3)=='1' else ''}  loop instr {len(ins)}")
    print(f"  VALU(all v_*) {valu}; v_dual_* {dual} (dot2acc+dot2acc pairs {dual_dot}); single dot2acc {d('v_dot2acc')}; other VALU {valu - dual - d('v_dot2acc')}")
    print(f"  ds_load {d('ds_load')}, s_load {d('s_load')}, global_load {d('global_load')}, s_delay_alu {d('s_delay_alu')}, s_waitcnt {d('s_waitcnt')}, v_mov {d('v_mov')}, v_pk {d('v_pk')}, v_perm {d('v_perm')}")
    print(f"  scratch ops {d('scratch_')}")
    print(f"  SALU(s_*, excl load/wait/delay/branch) {sum(v for k,v in c.items() if k.startswith('s_') and not k.startswith(('s_load','s_waitcnt','s_delay','s_cbranch','s_branch','s_clause','s_nop')))}")
    print(f"  meta {mt}")
    if '-v' in sys.argv:
        print('\n'.join(ins))

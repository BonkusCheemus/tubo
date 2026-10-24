# VALU census by purpose for loop dumps (cen.py dump). usage: purpose.py loopfile...
import re,sys,collections
def cls(l):
    op=l.split()[0]
    if not op.startswith('v_'): return None
    if 'dot4' in op: return '1 dot4 (MAC)'
    if op in('v_mad_i32_i24','v_mul_i32_i24_e32','v_mul_i32_i24'): return '2 scale fold (mad/mul24)'
    if op=='v_perm_b32':
        return '3 grid-address perm' if re.search(r'0xc0c0[0-9a-f]{3,4}\s*$',l) or re.search(r'0xc0c0[0-7][0-7]\s*$',l) else '4 tag interleave perm'
    if op.startswith('v_lshlrev') and re.search(r', 2, v\d+$',l): return '5 address <<2'
    if op=='v_and_b32_e32' and ('0xfcfcfcfc' in l): return '6 Idx1 A=qs&FC'
    if op=='v_and_or_b32' and '0x3030303' in l: return '7 Idx1 C=qs&3|tag'
    if op=='v_and_b32_e32' and '0x3030303' in l: return '7 Idx1 C=qs&3|tag'
    if op=='v_or3_b32': return '7 Idx1 C=qs&3|tag'
    if '0x1e1e1e1e' in l or '0x78787878' in l: return '8 sign even/odd mask'
    if op.startswith('v_lshlrev') and re.search(r', [13], v\d+$',l): return '8 sign even/odd shift'
    if op.startswith('v_lshrrev') and re.search(r', [13], v\d+$',l): return '8 sign even/odd shift'
    if op=='v_mul_u32_u24_e32' or '0x1010101' in l or '0x4040404' in l or op=='v_bfe_u32' and re.search(r'v\d+, 4$',l) is None and False: return '9 qh spread (bfe/and/mul/and_or)'
    if op=='v_bfe_u32': return '9 qh spread (bfe/and/mul/and_or)'
    if op=='v_and_b32_e32' and re.search(r', 15, v\d+$',l): return '9 qh spread (bfe/and/mul/and_or)'
    if (op.startswith('v_lshrrev') or op=='v_and_or_b32' or op=='v_or_b32_e32') : return 'a scale (shr/and_or)'
    if op in('v_cvt_f32_i32_e32','v_mul_f32_e32','v_fmac_f32_e32','v_dual_fmac_f32','v_dual_mul_f32','v_cvt_f32_f16_e32'): return 'b float fold (cvt/mul/fmac)'
    if op.startswith('v_add_co'): return 'c VMEM address (64-bit add)'
    if op in('v_dual_mov_b32','v_mov_b32_e32','v_mov_b16_e32'): return 'd prefetch copy (mov)'
    return 'e other:'+op
for f in sys.argv[1:]:
    c=collections.Counter()
    for l in open(f):
        l=l.split(';')[0].strip()
        if not l or l.startswith('s_delay'): continue
        k=cls(l)
        if k: c[k]+=1
    print(f, sum(c.values()))
    for k in sorted(c): print('   %-34s %4d'%(k,c[k]))

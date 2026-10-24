#!/usr/bin/env python3
"""Emits the inline-asm macros of OpusIq3sB8VopdKernel
(src/models/qwen/hip/kernels/sn_iq3.hip, between the GENERATED markers).
Run:  python3 .isa/gen_vopd_asm.py > /tmp/macros.h  and paste between the markers.

Register plan (pinned, so the VOPD legality rules hold by construction):
  accumulators  v0..v15   token t: lo = v(2t) (even, X half), hi = v(2t+1) (odd, Y half)
  weights       v16..v31  half-sub-block h (0/1): word i -> pair A = v(16+8h+2i)
                          (X vsrc1), pair B = v(17+8h+2i) (Y vsrc1): banks differ.
  src0          SGPR activation dwords (f16 pairs), distinct SGPRs for X and Y.
"""
out = []
def p(s=""): out.append(s)

def m_macro(q, n, h):
    name = f"SNB_VOPD_M_Q{q}_N{n}_H{h}"
    w0 = 16 + 8 * h
    lines = []
    for i in range(4):
        for j in range(n):
            lo = 2 * (4 * q + j); hi = lo + 1
            lines.append(
                f'      "v_dual_dot2acc_f32_f16 v{lo}, %[a{j}_{2*i}], v{w0+2*i} :: '
                f'v_dual_dot2acc_f32_f16 v{hi}, %[a{j}_{2*i+1}], v{w0+2*i+1}\\n"')
    outs = []
    for j in range(n):
        lo = 2 * (4 * q + j); hi = lo + 1
        outs.append(f'"+{{v{lo}}}"(acc[{lo}])')
        outs.append(f'"+{{v{hi}}}"(acc[{hi}])')
    ins = []
    for j in range(n):
        for c in range(8):
            ins.append(f'[a{j}_{c}] "s"(a[{j}][{8*h+c}])')
    for k in range(8):
        ins.append(f'"{{v{w0+k}}}"(wv[{8*h+k}])')
    p(f"#define {name}() \\")
    p("  asm volatile( \\")
    for l in lines:
        p(l + " \\")
    p("      : " + ", ".join(outs) + " \\")
    p("      : " + ", ".join(ins) + ")")
    p()

def d_macro(kind, h):
    w0 = 16 + 8 * h
    name = f"SNB_VOPD_D_{kind}_H{h}"
    lines = []
    for k in range(8):
        if kind == "T16":
            lines.append(f'      "v_pk_mul_f16 v{w0+k}, %[r{k}], %[sc]\\n"')
        else:
            lines.append(f'      "v_pk_fma_f16 v{w0+k}, %[r{k}], %[sc], %[cc]\\n"')
    outs = [f'"=&{{v{w0+k}}}"(wv[{8*h+k}])' for k in range(8)]
    ins = [f'[r{k}] "v"(r[{k}])' for k in range(8)] + ['[sc] "v"(sc)']
    if kind == "CV":
        ins.append('[cc] "v"(cc)')
    p(f"#define {name}() \\")
    p("  asm volatile( \\")
    for l in lines:
        p(l + " \\")
    p("      : " + ", ".join(outs) + " \\")
    p("      : " + ", ".join(ins) + ")")
    p()

for h in range(2):
    for q in range(2):
        for n in range(1, 5):
            m_macro(q, n, h)
for kind in ("T16", "CV"):
    for h in range(2):
        d_macro(kind, h)
print("\n".join(out))

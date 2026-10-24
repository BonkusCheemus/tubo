#!/usr/bin/env python3
"""Toy event sim: does k4's block-round structure (272 blocks in 80 slots,
5 K-blocks per wave, codebook staged per block) cost B8 more than B1, and what
would a persistent grid (80 blocks, work split evenly, prefetch across tiles)
give?  DRAM = one FIFO server (rate R) + fixed latency L.  SIMD = FIFO server
(compute c per wave-block).  Wave = k4 depth-1 prefetch.  Model only: no GPU.
usage: roundsim.py [R_GBps L_us]"""
import heapq, sys

R = float(sys.argv[1]) if len(sys.argv) > 1 else 232.0   # GB/s, DRAM stream
L = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5     # us, loaded latency
WB = 3520.0                                              # bytes per wave-block
WGP, SLOTS, SIMD_PER_WGP = 20, 4, 4
STAGE, EPI, F0 = 1.5, 0.5, 3.0                           # us (assumed)


def run(m, c_us, persistent, kblocks=20, kph=4):
    """Return kernel time in us."""
    t_dram = [0.0]
    simd_free = [0.0] * (WGP * SIMD_PER_WGP)
    ev = []  # (time, seq, kind, data)
    seq = [0]

    def push(t, kind, d):
        seq[0] += 1
        heapq.heappush(ev, (t, seq[0], kind, d))

    def dram(t):
        start = max(t, t_dram[0])
        t_dram[0] = start + WB / (R * 1e3)  # us
        return t_dram[0] + L

    rowwaves = m // 32
    if persistent:
        items = rowwaves * kblocks
        blocks = WGP * SLOTS
        per_wave = items / (blocks * 16)
        tasks = [int(per_wave)] * (blocks * 16)
        for i in range(items - int(per_wave) * blocks * 16):
            tasks[i] += 1
        block_tasks = [tasks[b * 16:(b + 1) * 16] for b in range(blocks)]
    else:
        blocks = rowwaves // 4
        block_tasks = [[kblocks // kph] * 16 for _ in range(blocks)]
    nxt = [0]
    slot_free = []
    for w in range(WGP):
        for s in range(SLOTS):
            slot_free.append(w)
    done = [0.0]
    pending = {}

    def start_block(t, wgp):
        if nxt[0] >= len(block_tasks):
            return
        b = nxt[0]
        nxt[0] += 1
        pending[b] = [16, wgp]
        t0 = t + STAGE
        for i, n in enumerate(block_tasks[b]):
            simd = wgp * SIMD_PER_WGP + (i % SIMD_PER_WGP)
            if n == 0:
                push(t0, "wavedone", (b,))
                continue
            push(t0, "issue", (b, simd, 0, n))

    # dispatch order: round-robin over WGPs
    for s in range(SLOTS):
        for w in range(WGP):
            start_block(0.0, w)
    while ev:
        t, _, kind, d = heapq.heappop(ev)
        if kind == "issue":           # wave issues load of item j, waits for it
            b, simd, j, n = d
            push(dram(t), "arrive", (b, simd, j, n))
        elif kind == "arrive":        # data of item j here: prefetch j+1, compute j
            b, simd, j, n = d
            if j + 1 < n:
                a = dram(t)
            else:
                a = None
            s0 = max(t, simd_free[simd])
            simd_free[simd] = s0 + c_us
            push(simd_free[simd], "computed", (b, simd, j, n, a))
        elif kind == "computed":
            b, simd, j, n, a = d
            if a is None:
                push(t, "wavedone", (b,))
            else:
                push(max(t, a), "arrive2", (b, simd, j + 1, n))
        elif kind == "arrive2":       # same as arrive but data already counted
            b, simd, j, n = d
            a = dram(t) if j + 1 < n else None
            s0 = max(t, simd_free[simd])
            simd_free[simd] = s0 + c_us
            push(simd_free[simd], "computed", (b, simd, j, n, a))
        elif kind == "wavedone":
            b = d[0]
            pending[b][0] -= 1
            if pending[b][0] == 0:
                te = t + EPI
                done[0] = max(done[0], te)
                start_block(te, pending[b][1])
    return F0 + done[0]


if __name__ == "__main__":
    for c_cyc_b1, c_cyc_b8 in ((450, 1550), (450, 1663)):
        print(f"R {R} L {L}  c(B1) {c_cyc_b1} c(B8) {c_cyc_b8} cycles")
        for m in (34816, 6144):
            mb = m * 5120 * 3.4375 / 8 / 1e6
            for name, per in (("k4", False), ("persist", True)):
                r = []
                for c in (c_cyc_b1, c_cyc_b8):
                    us = run(m, c / 1700.0, per)
                    r.append(f"{us:7.1f} us {mb * 1e3 / us:6.1f} GB/s")
                print(f"  m{m:5d} {name:8s} B1 {r[0]} | B8 {r[1]}")

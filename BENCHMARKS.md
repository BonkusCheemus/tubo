Written by Claude (an AI model by Anthropic) on behalf of the account owner.

# tubo benchmarks

Every number here was measured on one laptop, under the conditions written beside it. Status words: "measured" was run; "estimate" is arithmetic, never run; "quoted" is someone else's figure; "TBD" is not done. How to rerun each one: [REPRODUCE.md](REPRODUCE.md).

**Compare on the same machine only.** Our laptop is not the machine behind upstream's published numbers: a Ryzen AI MAX+ 392 (12 CPU cores) in a 32 GB laptop with a laptop power limit, against their Ryzen AI MAX+ 395 (16 cores) with 128 GB. The GPU is the same Radeon 8060S, yet vanilla gufo runs slower here than in their table (decode with the drafter 26-34% slower at 1700 MHz or the stock clock, N74; at the pinned 2000 MHz used below, prose decode 22.4 vs their 26.9 and prefill 429 vs 578), and we have not found why (N75). So every tubo-vs-vanilla number here was measured on this laptop, both engines, same settings. Do not compare our numbers with upstream's published table; to check our claims, run both engines on your own machine.

Machine: Ryzen AI MAX+ 392, Radeon 8060S (gfx1151, 40 CUs), 32 GB LPDDR5X-8000 (30 GiB visible, 24 GiB allocatable in all tests), ROCm 7.2.x, Linux. Theoretical DRAM peak: LPDDR5X-8000 x 256-bit = 256.0 GB/s.

## 1. Engine only: vanilla gufo vs tubo (headline)

The only difference between the two arms is the engine.

- vanilla = upstream gufo at commit `1b8ad798`, built on this laptop; tubo = this repository.
- Both: Qwen3.8-27B UD-Q4_K_XL (Unsloth) with the DFlash2 Q4_K_M drafter (z-lab), fp16 KV cache, the same prompts, temperature 0, GPU clock pinned at 2000 MHz.
- tubo runs [docs/tubo.env](docs/tubo.env) minus every key that contains `TURBO` (so the KV stays fp16), with `--draft-tokens 7 --min-draft-tokens 1 --draft-policy fixed`. Its default speed mode is exact.
- tubo can split part of prefill onto the NPU (the `GUFO_NPU_*` lines), but only for prompt chunks of 2048 tokens or more (`GUFO_NPU_MIN_M`, default 2048). The prompts in section 1 are shorter (597 and ~2040 tokens), so the NPU does not act in these numbers; the prefill difference is GPU-side.

### 1a. Our A/B (the video conditions)

597-token prompt (a nonce first line, so no request hits a prompt cache), 512 tokens out, context 4096, `--cache-ram-bytes 536870912` for both. Two loads per engine, one warm-up request per load dropped.

| metric | vanilla `1b8ad798` | tubo | change |
|---|--:|--:|--:|
| prefill (pp), tok/s | 268.5 ± 20.7 | 275.7 ± 7.4 | tie (within noise) |
| decode (tg), tok/s | 29.6 ± 2.7 | **37.8 ± 1.5** | **+28%** |
| time to first token, s | 2.24 ± 0.17 | 2.19 ± 0.06 | tie |
| wall time per request, s | 18.4 ± 1.7 | **14.6 ± 0.6** | **-20%** |
| runs | 9 | 10 | |

One vanilla request is excluded: it waited 436 s for its first token (a stall we could not explain; the other 9 took about 2.2 s). Including it only makes vanilla's means worse.

**This repository reproduces it.** A build of this repository with the REPRODUCE.md recipe against the binary that produced the table above, same conditions, 5 prompts x 2 loads each, loads interleaved (2026-10-27): temperature-0 output identical on 5 of 5 prompts in every load; decode 39.56 ± 1.18 vs 39.57 ± 1.10 tok/s, prefill 287.7 vs 286.1, time to first token 2.07 vs 2.08 s, wall 13.32 vs 13.32 s (n = 10 each).

### 1b. Upstream's own benchmark driver

`tools/bench/model-bench.py` and `docs/models/qwen3.8-27b/artifacts/bench.json` from upstream, unchanged, plus `--cache-ram-bytes 2147483648` for both engines. Context 4096, 2k-token prompt, 128 tokens out, one warm-up run, then 3 runs per cell (5 for the prose cells, rerun after one stalled vanilla request).

| cell (context 0) | vanilla | tubo | change |
|---|--:|--:|--:|
| prose, prefill tok/s | 429 ± 13 | 533 ± 13 | +24% |
| prose, decode tok/s | 22.4 ± 2.0 | 24.3 ± 1.8 | +9% |
| repetition, prefill tok/s | 433 | 550 | +27% |
| repetition, decode tok/s | 57.2 | 60.7 | +6% |

The decode gap is smaller here than in 1a: only 128 output tokens after a 2k prompt, and the drafter's guesses are accepted less often on this prose (38% vanilla, 26-27% tubo).

Longer context: vanilla gufo does not load past context 4096 on this 32 GB laptop (N76). tubo, same driver, prose cell, 2k prompt after the given amount of earlier context:

| config | 0 | 4k | 8k | 16k | 32k | 64k |
|---|--:|--:|--:|--:|--:|--:|
| apex + turbo6, prefill | 436-507 | 481-493 | 404 | 367 | 306 | 233 |
| apex + turbo6, decode | 28.9 | 26.9 | 27.1 | 25.1 | 27.2 | 21.8 |
| UD-Q4 + turbo6, prefill | 512 | 450 | 414 | — | 315 | — |
| UD-Q4 + turbo6, decode | 24.6 | 23.1 | 24.9 | — | 20.9 | — |

Decode moves ±10% between cells because acceptance depends on the text. Prefill at context 0 varied 436-507 across five loads on one day; cause not found.

## 2. KV format and model: quality with speed (tubo only)

These rows differ in KV format or model, so the speed here is not an engine comparison. Temperature 0, thinking off, GPU pinned at 2000 MHz.

| configuration | HumanEval (164) | MATH-500 | needles, 32k | needles, 64-128k | HumanEval tok/s (whole run) |
|---|--:|--:|--:|--:|--:|
| vanilla, UD-Q4, fp16 KV | 156 | — | does not load | — | 36.1 |
| tubo, UD-Q4, fp16 KV | 155 | 389 (77.8%) | 60/60 | does not load | 46.5 |
| tubo, UD-Q4, turbo6 KV | 155 | — | 60/60 | 40/40 at 64k | 46.7 |
| tubo, apex model, turbo6 KV | 154 | 383 (76.6%) | — | 40/40 at 128k | 53.3 |

- HumanEval: differences of 1-2 problems; exact McNemar test vs vanilla p = 1.00, 1.00, 0.79. tubo with fp16 KV already flips 3 problems vs vanilla (kernels add numbers in a different order); turbo6 flips none beyond those.
- MATH-500: exact match of the boxed answer, so equivalent forms count as wrong and absolute scores read low; 81 (UD-Q4) and 88 (apex) answers hit the token limit. 383 vs 389: exact McNemar p = 0.41, so "near lossless", not "equal". It changes two things at once (fine-tune and KV format). Apex: 17.8 s and 1005 tokens per problem; UD-Q4: 20.9 s and 1043.
- Needles: 20 hidden key-value pairs in one long prompt, all asked at once (RULER-style); 3 seeds at 32k, 2 at 64k and 128k.

Memory (RAM taken from the system after load and one request, in GiB; on this chip GPU memory is system RAM):

| configuration | RAM at context 4096 | longest context that loads on 32 GB |
|---|--:|--:|
| vanilla, UD-Q4, fp16 KV | 21.7 GiB | 4096 (8192 is refused, N76) |
| tubo, UD-Q4, fp16 KV | 21.3 GiB | 32k |
| tubo, UD-Q4, turbo6 KV | 20.8 GiB | 64k tested (needles 40/40); 128k loads only with the prompt cache off, right at the 24 GiB GPU-memory cap |
| tubo, apex model, turbo6 KV | 17.6 GiB | 128k tested (needles 40/40); 262k loads (23.4 GiB) with the prompt cache off |

## 3. Fast mode (opt-in)

Every number above is the default exact mode. `--speed-mode fast` is checked with one number: the total KL divergence of its output against exact mode. That number is not measured yet, so this file shows no speed for fast mode. The levers and what is known about each: [docs/SERVER.md](docs/SERVER.md#speed-mode-exact-fast-or-single-levers).

## 4. Kernel efficiency: % of theoretical DRAM bandwidth

Machine: Ryzen AI MAX+ 392 (Radeon 8060S iGPU, gfx1151), 32 GB LPDDR5X-8000, laptop. % = GB/s ÷ 256.0 GB/s
(LPDDR5X-8000 × 256-bit, theoretical). Each kernel alone, microbenchmark, GPU clock pinned to 1700 MHz, normal desktop.
Decode reads every weight once per step, so GB/s is the speed that matters. Width 1 = plain decode (one token);
width 8 = one speculative-decoding verify step (8 tokens).

Anything below 90% is still being improved.

### Weight kernels, by type (all real tensors of both models, weighted by bytes)

Every weight tensor of UD-Q4_K_XL and the apex model, each run the way the server runs it (same kernels, same
launch groups, weights rotated through 256 MiB so no cache helps). GB/s = weight bytes / weight time; the
activation-quantize step is timed and reported apart. GPU pinned 1700 MHz.

| weight type | GB per step | width 1 | width 8 |
|---|--:|--:|--:|
| Q6_K | 4.05 | 222 GB/s (87%) | 206 (81%) |
| IQ4_XS | 6.16 | 211 (83%) | 175 (68%) |
| IQ3_S | 2.49 | 210 (82%) | 203 (79%) |
| Q8_0 | 0.25 | 198 (77%) | 110 (43%) |
| Q5_K | 9.00 | 191 (74%) | 193 (76%) |
| IQ3_XXS | 2.18 | 190 (74%) | 196 (76%) |
| IQ4_NL | 0.28 | 187 (73%) | 174 (68%) |
| Q4_K | 5.13 | 160 (63%) | 192 (75%) |
| Q3_K | 0.11 | 135 (53%) | 92 (36%) |

Whole weight pass (all types together):

| model | width 1 | width 8 | vanilla gufo, width 8 |
|---|--:|--:|--:|
| UD-Q4_K_XL | 194 GB/s (76%) | 186 (73%) | 135 (53%) |
| apex | 191 (75%) | 192 (75%) | cannot load (no IQ3_XXS) |

At width 8 (one verify step) tubo moves the UD-Q4 weights 37% faster than vanilla gufo on the same laptop.
At width 1 the two are close (+3.6%): most width-1 kernels are the same code in both.
Low rows: Q4_K at width 1 runs inside fused attention/SSM input
kernels shared with upstream; Q3_K and Q8_0 are small tensors.

### Attention

| kernel | GB/s | % of 256 |
|---|--:|--:|
| verify attention over 6-bit KV, 32k context, 1 row | ~107-121 | ~42-47 |
| verify attention over 6-bit KV, 32k context, 8 rows | ~76 | ~30 |

Low percentage, still faster than fp16 KV: 6-bit KV is 2.6x fewer bytes. At 8 rows the upstream fp16 kernel also
drops to ~30% on this chip, so ours finishes the same attention ~2.5x sooner; at 1 row fp16 reaches ~92% and ours
is still ~1.2-1.3x sooner. Other engines: fp16-KV attention 83-95% at 1 row; compressed-KV kernels 27-65%,
BitDecoding 88% (arXiv 2503.18773). Our derived ceiling for bit-exact 6-bit attention at 8 rows: ~76% at 1700 MHz
(the math, not memory, is the limit there).

### Memory ceiling (pure read, no math)

250.8 GB/s (98.0%) on a normal desktop; 252.8 GB/s (98.8%) in a controlled quiet run (other GPU/CPU work paused,
screen off). Setup and steps to reproduce: [tools/probes/README.md](tools/probes/README.md), "Reproduce the pure-read numbers".

## 5. Full record

The rows below are older measurements from the development history; sections I and J name the tree and the binary each one used. N-numbers are cited in the README.

Theoretical DRAM peak: 256.0 GB/s. 253.4 GB/s is the 99% line (0.99 x 256), not a measured peak. Best single measured round: 253.82 GB/s. Both percentage columns are given: of 256.0 and of 253.4.
GB/s = bytes / time / 1e9 (decimal). "pin 1700" = GPU core clock pinned to 1700 MHz. "Watts" = GPU-only (gpu_metrics) unless it says PPT (whole-package power, includes CPU and desktop). n/a = not recorded.

### A. DRAM ceiling (pure read, no math)

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N01 | stable steady-state read, quiet run 2 | 252.83 GB/s | 98.8 | 99.8 | n/a (stream arms 5.5-5.9 W in the desktop run) | pin 1700; 21.6 MB host-mapped, 7 rotating copies; persistent kernel, static wave-interleave, 40 workgroups, 196 buffer reads per window; median of 9 runs, each a median of 3 passes x 7 rounds; all other processes paused, screen off, 38 C at start (dynamic 32 KB arm: 252.11) | measured |
| N02 | same, quiet run 1 | 252.81 GB/s | 98.8 | 99.8 | n/a | as N01, 59 C at start | measured |
| N03 | screen off only (processes running) | 252.17 GB/s | 98.5 | 99.5 | n/a | static arm as N01, median of 9, screen off, processes running | measured |
| N04 | normal desktop | 250.8 GB/s (249.6-251.8) | 98.0 | 99.0 | 5.5-5.9 GPU | dynamic 32 KB chunk arm, 80 workgroups; median of 9 runs, desktop on | measured |
| N05 | undisturbed 50 us bins, median | 252.97 GB/s | 98.8 | 99.8 | n/a | quiet runs, same value in 3 runs; 10.6-10.8% of bins are dips | measured |
| N06 | mean of bins with dips removed | 253.40 GB/s | 99.0 | 100.0 | n/a | quiet run 2 (run 1: 253.32) | measured |
| N07 | best single round | 253.82 GB/s | 99.1 | 100.2 | n/a | quiet run 1: 15 of 108 rounds >= 253.4; 0 of 108 pass medians >= 253.4 (max 253.35) | measured |
| N08 | read rate vs GPU core clock (21.6 MB, best of two kernels) | 1.0 GHz 249.5; 1.3 GHz 250.8; 1.7 GHz 250.1; 2.0 GHz 251.5 | 97.4-98.3 | 98.4-99.3 | PPT 20-23 W, flat | median of 3 runs per clock, desktop on, nothing paused | measured |
| N09 | read rate at 92 C | -3 to -4% vs 83 C | - | - | n/a | one sample per temperature | measured (weak) |
| N10 | one plain launch, pure read, by size (typical best config) | 17.7 MB 227.3; 21.6 MB 231.9; 25.8 MB 237.8; 64 MB 245.5; 256 MB 248.7 | 88.8-97.1 | 89.7-98.1 | n/a | host-mapped, 24 back-to-back launches per timed round, pin 1700, desktop load | measured |
| N11 | one launch at IQ3 sizes, best config | 68.2 MB 250.7; 76.6 MB 248.1 (median over configs 238.8 / 235.7) | 97.9 / 96.9 | 98.9 / 97.9 | n/a | host-mapped, clock about 1650 MHz (header reading), 1-minute load 1.3-1.8, one run per size | measured |

### B. Launch overhead and overlap (21.6 MB host-mapped, pin 1700)

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N12 | plain back-to-back launches | 230.1 GB/s | 89.9 | 90.8 | 6.3 GPU | per-launch loss about 11 us = gap 4.3 + ramp 6.0 + tail 6-8 | measured |
| N13 | independent launches, any-order flag (overlap) | 246.9 GB/s | 96.4 | 97.4 | 6.9-7.1 GPU | event pair around 24 launches | measured |
| N14 | dependent launches (done-counter protocol, serving-like) | 241.1 GB/s | 94.2 | 95.1 | 6.8-6.9 GPU | 0 wrong-order, 0 timeouts in about 50,000 launches | measured |
| N15 | dependent chain captured in a graph | 228.4 GB/s (plain graph 231.8) | 89.2 | 90.1 | n/a | capture drops the any-order flag on HIP 7.2.4 | measured |
| N16 | atomic 32 KB chunk counter inside ONE launch | 207.3 vs 219.1 static (event GB/s) | 81.0 | 81.8 | n/a | one run, box not quiet; about -5% | measured (n=1) |
| N17 | effect of ROCm 10.1 (keeps any-order flag in graphs) | +3-4% (the dependent arm) | - | - | n/a | upstream fix merged 2026-07-29; we run 7.2.4 and have not tried 10.1 | estimate |

### C. Load/compute decoupling probe (synthetic VALU load, 95% target = 241 GB/s)

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N18 | Q5-sized, 4 matrices per launch, 16 VALU units/block: decoupled vs coupled | 244.6 vs 240.2 GB/s | 95.5 vs 93.8 | 96.5 vs 94.8 | n/a | pin 1650-1725, clean run | measured (microbench) |
| N19 | same at 22 units/block (about B8 load) | 232.2 vs 215.2 GB/s (+8%) | 90.7 vs 84.1 | 91.6 vs 84.9 | n/a | 95% not reached | measured (microbench) |
| N20 | IQ3-sized (68 MB), no VALU: decoupled vs coupled | 247.3 vs 239.9 GB/s | 96.6 vs 93.7 | 97.6 vs 94.7 | n/a | - | measured (microbench) |
| N21 | coupling cost, 1 matrix, 22 units vs 0 | -5.3% | - | - | n/a | reproduced twice (-5.4% earlier) | measured |
| N22 | real decoupled Q5_K kernel, 4 matrices per launch (5120x6144) | best 171.3 GB/s vs 206.3 for the current coupled kernel (-17%); 1 matrix: 140.2 vs 200.1 | 66.9 vs 80.6 | 67.6 vs 81.4 | PPT 27.5 mean / 49.8 max (whole run) | pin 1700, host-mapped, 7 arm configs, one run; prediction was 232-238 and the bar 225: the design as written FAILED | measured (n=1) |
| N23 | quiet rerun: decoupled vs coupled at real VALU load | Q6-sized 24 units, 4 matrices: 235.3 vs 220.2 (+6.9%); IQ3-sized 20 units: 239.9 vs 238.1 and 237.6 vs 241.5 (no gain); IQ3-sized no VALU: 248.6 | 91.9 vs 86.0; 93.7 vs 93.0; 92.8 vs 94.3; 97.1 | 92.9 vs 86.9; 94.7 vs 94.0; 93.8 vs 95.3; 98.1 | n/a | pin 1700, load <= 1.45, kernel log clean | measured (microbench) |

### D. Decode kernels (microbench, 27B hybrid model shapes, pin 1700, kernel-only unless noted)

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N24 | IQ3_S fused gate+up 34816x5120, B1 / B4 / B8 | 226.8 / 230.2 / 216.3 GB/s | 88.6 / 89.9 / 84.5 | 89.5 / 90.8 / 85.4 | n/a | persistent kernel, split S=1, median n=12; activation quantize timed apart | measured |
| N25 | IQ3_XXS fused 34816x5120, B1 / B4 / B8 | 222.3 / 224.1 / 211.5 GB/s | 86.8 / 87.5 / 82.6 | 87.7 / 88.4 / 83.5 | n/a | n=12; activation quantize (about 7.5 us per launch) timed apart | measured |
| N26 | IQ4_XS 5120x17408, B1 / B4 / B8 | 231.6 / 224.8 / 212.1 GB/s | 90.5 / 87.8 / 82.9 | 91.4 / 88.7 / 83.7 | n/a | n=8, kernel-only | measured |
| N27 | band: 9 served shapes x 8 widths (B1-B8), best kernel per cell | 72 of 72 cells >= 210 GB/s (lowest 211.5) | >= 82.0 | >= 82.9 | n/a | not covered: small K/V 1024x5120 (147-181), Q8_0, F32; timing convention differs across rows | measured, re-run clean before posting |
| N28 | Q5_K 5120x6144 B8 | 177 -> 210-214 GB/s | 69 -> 82-84 | 70 -> 83-84 | n/a | interleaved layout + hoisted loads; n=16 | measured |
| N29 | served IQ3 gate/up before (two 17408x5120 launches), B8 | IQ3_S 104.4; IQ3_XXS 119.2 GB/s | 40.8 / 46.6 | 41.2 / 47.0 | n/a | served kernels, same pin | measured |
| N30 | new kernel, same 17408x5120 launch, B8 | IQ3_S 182.4; IQ3_XXS 192.5 GB/s | 71.2 / 75.2 | 72.0 / 76.0 | n/a | quantize timed apart | measured |
| N31 | new kernel on real model weights, gate+up pair, B2-B8 | 175-223 GB/s (IQ3_S 223/178/185/200/203/175/201; IQ3_XXS 215/180/214/208/212/209/207) | 68.4-87.1 | 69.1-88.0 | n/a | max rel. error 2.4e-7 vs reference; 14 widths x 2 types passed; weight-only timed views | measured |
| N32 | attention reader, verify, 32k context, 8 rows, turbo6 | 68.4 GB/s (wmma reader); 49 GB/s in served trace | 26.7 / 19.1 | 27.0 / 19.3 | n/a | memory-bound would be about 240; reader is issue/LDS bound | measured |
| N33 | attention reader after pipeline fix | 0.92-0.96 ms vs 1.06-1.25 ms per launch (-12%) | - | - | n/a | bitwise identical output | measured |

### E. Serving (decode, 32k context, speculative decoding with a parallel drafter, pin 1700)

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N34 | baseline decode | 22.1 tok/s mean, round 149.6 ms, 2.3 accepted tokens/round | - | - | PPT 54.8 mean / 55.7 max | prose prompt, greedy, n=3 | measured (superseded by N82) |
| N35 | IQ3 new kernels vs baseline, A/B 1 | 23.95 vs 22.08 tok/s (+8.5%); round 136.5 vs 149.6 ms (-8.8%); tok/s/W 0.440 vs 0.403 | - | - | PPT 54.5 / 54.8 | n=3 per arm, runs ON 21.80/24.75/25.32, baseline 23.10/20.64/22.51; noise floor 14.7%; NOT significant | measured (superseded by N38) |
| N36 | IQ3 new kernels vs same binary with the path off | +6.5% tok/s, -11.1% round time | - | - | PPT 54.5 vs 54.0 | same A/B 1; inside the noise floor | measured (superseded by N38) |
| N37 | A/B 2 (earlier tree, no other changes), round time | 140.2 vs 158.8 ms (-11.7%); tok/s 23.24 vs 21.74 | - | - | PPT 54-55 | n=3 per arm | measured (superseded by N38) |
| N38 | A/B 3, n=6 per arm: IQ3 new kernels vs served binary | 25.19 vs 23.10 tok/s (+9.05%); round 129.2 vs 142.7 ms (-9.5%); tok/s/W 0.467 vs 0.423 (+10.4%); same binary with the path off: 22.65 (ON vs off +11.2%) | - | - | PPT new path 54.0 / served 54.6 | 33,712-token prompt, 512 tokens out, greedy, pin 1700, order interleaved, warm-up run dropped; noise floor 2.56% (max within-arm spread): OUTSIDE noise; each arm gave identical text run to run; ON text differs from baseline (not bit-exact by design, see N40) | measured |
| N39 | attention pipeline + batched small GEMV, round time | -3.9 ms/round (-2.8%) on a quiet rerun; an earlier -8.7 ms was noise | - | - | PPT about 55 | n=3-4 per arm, temp-0 text identical | measured |
| N40 | KLD, new IQ3 path vs native path (verify, 256 rows/cell) | 6.3e-4 / 1.0e-3 (code 8k / 32k); 7.4e-4 / 1.1e-3 (wiki 8k / 32k) | - | - | - | top-1 same: 99.6 / 99.2 / 97.7 / 98.4 %; mean dNLL -0.0032..+0.0016; our own bar was 5e-4: FAIL | measured |
| N41 | KLD, repeat of the native path vs itself | 0.0 exactly | - | - | - | confirms the harness is deterministic | measured |
| N42 | prose check, new IQ3 path, temp 0 and 0.6 | same 7 facts correct, same plan, different wording | - | - | - | one prompt, spot read, not a benchmark | measured (weak) |

### F. KV cache (K+V = 32,768 values per token for this model: 16 full-attention layers x 4 KV heads x 256 dims x 2; 16,384 per plane)

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N44 | bits per value | turbo6 6.125; q8_0 8.5; fp16 16; turbo4 4.125 | - | - | - | slot sizes incl. scales | measured (turbo4, fp16, q8_0 byte counts logged); turbo6 = arithmetic |
| N45 | bytes per token | fp16 65,536; q8_0 34,816; turbo6 25,088; turbo4 16,896 | - | - | - | 32,768 values x bits / 8 | turbo4/fp16 measured; others estimate |
| N46 | turbo6 vs fp16 KV, mean KLD | 2.0e-4, 3.3e-4, 2.6e-4, 3.3e-4 (4 cells) | - | - | - | 256 rows per cell, top-1 98.8-100% | measured |
| N47 | third-party figure, same bit widths | turbo6/turbo6 5.0-5.6e-4 vs q8_0/q8_0 2.5-3.6e-4 | - | - | - | other author's GPU and fork; not our measurement | quoted |
| N48 | decode speed turbo6 / turbo4, 8k ctx | code 0.943, json 0.989, prose 0.976-1.003 | - | - | - | earlier reader; before the verify-reader fixes | measured (superseded) |
| N49 | fits in memory | 90,112-token window + drafter, peak 23.46 GiB | - | - | - | earlier binary; 24 GiB budget | measured |
| N50 | KV at 262,144 tokens (model native limit) | turbo6 6.58 GB; q8_0 9.13 GB; fp16 17.18 GB | - | - | - | arithmetic from N45 | estimate |
| N51 | KV at 1,048,576 tokens | turbo6 26.3 GB (24.5 GiB); q8_0 36.5 GB; fp16 68.7 GB | - | - | - | plus about 4.3 GB per-token scratch and 14.4 GB weights; model native context is 262,144, quality beyond it unmeasured | estimate |
| N52 | KV read time per verify round at 1M tokens | 0.39 s at 68 GB/s; 0.11 s at 240 GB/s | - | - | - | extrapolated from N32 | estimate |

### G. Prefill

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N56 | GPU+NPU prefill split, our implementation | x1.12-1.16 at 8k tokens | - | - | - | stacked, decode unchanged | measured |
| N57 | GPU+NPU prefill split, community fork's claim | +28% at 8k (629 -> 806 tok/s) | - | - | - | their README, not reproduced | quoted |

### H. Cold serve: energy per token vs GPU clock (served decode, 33,712-token prompt, 512 out, temp 0, n=3 per arm per clock, one session 10-22)

Watts = whole-package power (PPT) averaged over decode. J/token = W / tok/s. kWh per 1M output tokens = J/token / 3.6.
ON = IQ3 kernel path (N38), REF = the served binary before it. 2000 MHz was not run in this session; see N83 (later tree).

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N60 | ON at 1700 / 1300 / 1000 / 800 MHz | 25.76 / 22.73 / 19.16 / 15.77 tok/s; 2.12 / 2.29 / 2.79 / 3.26 J/token | - | - | 54.5 / 52.1 / 53.4 / 51.4 | n=3 each; spread within arm <= 1.9% | measured |
| N61 | REF at 1700 / 1300 / 1000 / 800 MHz | 24.44 / 20.60 / 16.53 / 13.65 tok/s; 2.25 / 2.65 / 3.24 / 3.95 J/token | - | - | 54.9 / 54.6 / 53.6 / 53.9 | n=3 each; spread <= 2.2% | measured |
| N62 | ON gain over REF by clock | +5.4% (1700), +10.3% (1300), +15.9% (1000), +15.5% (800) | - | - | - | the IQ3 kernel spends fewer GPU cycles, so it helps more when the clock is low | measured |
| N63 | package power vs GPU clock | 51-55 W at every clock from 800 to 1700; the GPU core is ~14 W of ~55 W at 1700 (gpu_metrics) | - | - | - | lowering the GPU clock saves ~1-3 W and costs up to 39% speed, so energy per token is LOWEST at 1700 | measured (superseded by N83: on a later tree 2000 MHz gives lower J/token than 1700) |
| N64 | energy per 1M output tokens, ON at 1700 | 0.59 kWh (~JPY 18 at an assumed JPY 31/kWh) | - | - | 54.5 | from N60; electricity price is an assumption | estimate |

WRONG IF: a rerun at n>=6 moves J/token at 1700 above 1300's, or the 2000 MHz arm gives lower J/token than 1700.

### I. Final benchmark (10-23/24): tubo vs vanilla gufo, same laptop

"vanilla" = upstream gufo at commit 53ebb51e, built on this laptop. "tubo" = the fork with its default settings (draft 7).
Harness for N65-N78: upstream's own benchmark driver (tools/bench/model-bench.py) and bench.json, with a cache-RAM limit flag added. Model Qwen3.8-27B UD-Q4_K_XL with the DFlash2 Q4_K_M drafter (DF); "no drafter" = plain autoregressive decode.
n=3 per cell after one discarded warm-up run. d0 = context 4096; d32768 = context 36864 with a ~32,764-token prompt. The driver cycles 3 different prompts across the 3 reps. tg / pp = server-reported decode / prefill tok/s.
PPT here = whole-package power averaged over the whole engine run (prefill and decode, both cells), so tok/s/W = tg / that mean.
"auto Balanced" = no clock pin, platform profile Balanced. tubo and vanilla cells ran in separate runs of the same session (pin 1700: 10-23 late and 10-24 early; auto: 10-24, about 45 min apart).
Upstream's published numbers come from a different machine (Ryzen AI MAX+ 395, 128 GiB) and are not compared with here; N74 only records how far our vanilla build is from them.

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N65 | prose tg, DF d0, pin 1700 | tubo 22.85 +- 2.73 vs vanilla 19.53 +- 1.84 tok/s (+17%) | - | - | PPT tubo 67.7 / vanilla 62.7 | n=3 | measured |
| N66 | repetition tg, DF d0, pin 1700 | tubo 54.75 vs vanilla 43.67 +- 0.87 tok/s (+25%) | - | - | as N65 | n=3 | measured |
| N67 | prefill pp, DF prose d0, pin 1700 | tubo 442 vs vanilla 375 tok/s (+18%) | - | - | as N65 | n=3 | measured |
| N68 | tok/s/W, DF d0, pin 1700 | prose tubo 0.338 vs vanilla 0.311 (+9%); repetition 0.809 vs 0.696 (+16%) | - | - | as N65 | tubo draws 5 W more over the run | measured (ratios of measured values) |
| N69 | prose tg and pp, DF d0, auto Balanced | tg tubo 24.21 +- 3.08 vs vanilla 18.63 +- 1.86 (+30%); pp 436 vs 348 (+25%) | - | - | PPT tubo 71.2 / vanilla 56.5 | n=3 | measured |
| N70 | repetition tg and pp, DF d0, auto Balanced | tg tubo 59.14 +- 4.07 vs vanilla 42.81 +- 0.55 (+38%); pp 454 vs 391 (+16%) | - | - | as N69 | n=3 | measured |
| N71 | tok/s/W, DF d0, auto Balanced | prose tubo 0.340 vs vanilla 0.330 (+3%); repetition 0.831 vs 0.758 (+10%) | - | - | as N69 | tubo draws 14.7 W more in this mode, so most of the speed gain is not an energy gain | measured (ratios of measured values) |
| N72 | Performance platform profile, auto clock | vanilla prose 19.81 +- 3.07 (+6% vs N69), repetition 46.11 +- 2.72 (+8% vs N70); tubo prose 24.47 +- 3.45 (+1%), repetition 62.24 +- 4.61 (+5%); tok/s/W prose tubo 0.261, vanilla 0.224 | - | - | PPT vanilla 88.5 (+32 vs Balanced), tubo 93.9 (+23) | n=3, one run, edge up to 86 C; more watts, little speed: lower tok/s/W than Balanced | measured |
| N73 | replication: vanilla temp-0 text | first-rep completion sha256 = upstream's published sha (prose 345b3854, repetition 82fd84fd) in every vanilla run (pin 1700, Balanced, Performance) | - | - | - | so the build and harness setup reproduce upstream's output. tubo repetition texts = vanilla's (all 3 reps); tubo prose texts differ (not bit-exact to vanilla on prose). Vanilla no-drafter d32768 text differs from upstream's (suspected cause: context 36864 vs 133,760; unchecked) | measured |
| N74 | replication: vanilla speed vs upstream's published row | DF prose d0: upstream 26.91 (pp 578) vs ours 19.53 pin 1700 (-27.4%), 18.63 Balanced (-30.8%), 19.81 Performance (-26.4%). DF repetition d0: 64.74 vs 43.67 (-32.5%), 42.81 (-33.9%), 46.11 (-28.8%). No drafter prose d0: 12.37 (pp 649) vs 10.86 (pp 404), 10.58, 10.82 | - | - | - | ours: N65-N72 conditions; upstream: their 128 GiB machine, n=1 | measured (ours) / quoted (upstream) |
| N75 | cause of the gap in N74 | hypothesis, not tested: this laptop's power and heat limit (Balanced runs peak near 80 W PPT, edge 78-88 C). Other untested differences: upstream context 133,760 vs 4096, unpinned n=1, a 395 vs our 392 CPU (16 vs 12 CPU cores; same Radeon 8060S, 40 CUs) | - | - | - | from run notes; no run isolates any one of these | hypothesis |
| N76 | vanilla DF at d32768 on this 32 GB laptop | does not load: 0 of 5 tries (pin 1700 and auto clock); the server exits before ready with bad_alloc while creating sessions. Context 8192 is refused the same way (2026-10-27, 8.2 GB free at that point). Where it throws (debugger, 2026-10-27): `ResourceExhausted` (a `std::bad_alloc`) from `ResourceLedger::ConstrainLimits` (src/cache/ledger.cpp) during the prompt-cache setup (src/cli/serve/component_text_cache.cpp): the cache limit is host RAM available minus a fixed 4 GiB reserve (`kHostSnapshotHeadroomBytes`) minus staging, and the context-sized session state already charged to it is larger. The prompt cache is always on, so no flag avoids it. Needs ~8.9 GiB free after the weights (state 2.68 GB + scratch 2.51 GB + a fixed 4 GiB host-RAM headroom the engine keeps); the most we freed was 8.57 GiB. No-drafter d32768 (about 1.1 GiB less) loads: tg 9.20 +- 0.62, pp 297 (Balanced) | - | - | PPT 59.4 (no-drafter cell) | other sessions closed for the best try | measured |
| N77 | tubo DF prose at d32768 | tg 19.25 / pp 220 (pin 1700); tg 20.27 +- 0.88 / pp 232 (auto Balanced). | - | - | PPT 57.1 (Balanced run) | n=3 | measured |
| N78 | tubo DF repetition at d32768, per prompt | prompt A 56.0 / 55.3 / 57.6 tok/s (drafter tokens accepted 109 of 120); B 22.9 / 22.4 (84 of 288); C 20.2 / 19.9 (78 of 332); mean of the 3 prompts about 33. | - | - | - | two runs on 10-24 (the first at auto Balanced); each prompt repeats to the token across them; the old +- 20.9 spread is the drafter predicting one prompt well and two poorly, not run noise | measured |

Rows N79-N85: tubo only, our own serving harness (temp 0, arms interleaved, warm-up run dropped, kernel log clean), on the 3-4 bit mixed "apex" GGUF (named in THIRD_PARTY_NOTICES.md, "Models used"), not UD-Q4_K_XL. PPT = mean over decode.

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N79 | draft tokens 7 vs 6 | draft 7 (default, bit-exact) 28.33 tok/s, 0.519 tok/s/W; draft 6 29.43 (+3.87%, noise 0.53%), 0.539 tok/s/W; draft 6 temp-0 text differs from draft 7 in 9 of 9 runs | - | - | PPT 54.6 both | n=9 per arm, pin 2000, Balanced. Earlier pin-1700 sessions: draft 6 +3.6% (n=10), +4.6% (n=3), +5.3% (n=9). Draft 6 = opt-in | measured |
| N80 | draft 6 KLD vs the default path (teacher-forced, 224 positions, code/wiki x 8192/31744 context) | mean KLD 5.3e-4 to 1.33e-3; same top-1 96.9-99.1% | - | - | - | fails our 5e-4 / 99% bar in 4 of 4 cells | measured |
| N81 | KQ_X switch, an attempt to make draft 6 bit-exact | draft 6 + KQ_X 29.53 tok/s (+4.23%), text differs from draft 7 and from plain draft 6 (9 of 9); draft 7 + KQ_X 28.16 (-0.61%, inside 0.92% noise), text = draft 7 (9 of 9) | - | - | PPT 55.1 / 54.8 | same session as N79 | measured: did not make draft 6 bit-exact |
| N82 | tubo served decode, current default (draft 7) | 26.85 tok/s at pin 1700 (n=9); 28.33 at pin 2000 (N79) | - | - | PPT 54.6 / 54.6 | replaces N34 as the current served speed | measured |
| N83 | GPU clock 2000 vs 1700, served | +5.8% tok/s at +0.6 W, so lower J/token at 2000 | - | - | PPT 53.4-54.5 both clocks | n=9 at 1700, n=5 at 2000; tree before N82 | measured |
| N84 | auto clock: Balanced vs quiet platform profile | Balanced 28.51 tok/s at 58.2 W (0.49 tok/s/W); quiet 26.68 at 45.1 W (0.59 tok/s/W, +21%); text identical | - | - | 58.2 / 45.1 | n=3 per arm, one session; quiet = pin-1700 speed (26.36 in the same session) at 9 W less | measured (n=3) |
| N85 | bit-exact decode changes after N38 | packed IQ3 path +2.50% (pin 1700, n=9) / +2.59% (pin 2000, n=5); IQ4_XS width-8 wave32 row +1.41% (n=9); logits dump identical in every run | - | - | PPT 53.4-54.6 | each vs the tree before it | measured |
| N86 | prefill GEMM opt-in (later post) | bitwise exact in 41 of 41 cells; speed mixed: Q6_K +2.5..+5.7% on 7 of 9 shapes, IQ4_XS -1.5..+7.8%, Q4_K and Q5_K mostly slower; best cell 26.8 TFLOPS = 82% of a 32.6 TFLOPS matrix peak | - | - | n/a | microbench, pin 1700, 6 AB/BA pairs per cell; not in serving | measured (microbench) |
| N87 | speculative vs plain decode, temp 0, 512 tokens, 34k prose + code, served config | text NOT identical (prose diverges at char 11, code at char 825); speed prose 27.88 vs 8.21 tok/s (3.4x), code 16.92 vs 7.74 (2.2x) | - | - | - | n=1 per arm; cause = 8-row verify vs 1-row kernel summation order | measured |

WRONG IF: a rerun with the same driver gives vanilla DF prose d0 >= 24 tok/s here, vanilla DF d32768 loads on this 32 GB laptop, or a third run of N78 gives different accepted-token counts for the same prompt.

### J. First-export tree vs the previous served binary (exact default)

Our serving harness, same rules as N79-N85 ("apex" GGUF), pin 1700, temp 0, draft 7, fixed draft policy, 9 runs per arm in an interleaved order (Williams square), warm-up run dropped. "previous" = the binary served before this tree, which ran with the lookup route ON. "current" = the tree of the first export (2026-10-25) with its served environment and the lookup route OFF (the exact default). PPT = mean over decode.

| ID | result | value | % of 256.0 | % of 253.4 | watts | conditions | status |
|---|---|---|---|---|---|---|---|
| N88 | decode tok/s, current vs previous, by text kind | prose 26.26 +- 0.15 vs 23.07 +- 0.13 (+13.8%); code 25.98 +- 0.22 vs 24.05 +- 0.35 (+8.0%); json 29.93 +- 0.04 vs 22.38 +- 0.08 (+33.7%). tok/s/W prose 0.480 vs 0.423; code 0.477 vs 0.439; json 0.536 vs 0.400. Lookup route on top of current: prose +15.9% (noise 1.5%), code and json 0% (no routed rounds) | - | - | PPT 54.5-55.8 current / 54.6-56.0 previous | one session; prose n=9 per arm, code and json n=2 per arm (spread 0.4-2.1%); temp-0 text differs from previous in every run, as expected: current runs the SN IQ3 path (N40), previous the native path, so "exact" here means exact to tubo's own default, not to the previous binary or to upstream; the run stopped at the fourth (copy-heavy) kind on a log-format check, and the temperature-0.6 read was not done | measured (one session, run incomplete) |
| N89 | prefill tok/s, current vs previous | prose 413.7 vs 361.3 (+14.5%); code 417.9 vs 362.0 (+15.4%); json 444.7 vs 379.6 (+17.2%) | - | - | as N88 | as N88 | measured (one session, run incomplete) |
| N90 | KLD, current vs the earlier tubo tree with the same IQ3 path | 0 exactly in 4 of 4 cells, same top-1 100% | - | - | - | teacher-forced, code/wiki x 8192/31744 context; the previous served binary's own KLD tool was not built, so current vs previous is not measured | measured |

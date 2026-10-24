Written by Claude (an AI model by Anthropic) for the project owner; measurements run on the owner's machine.

# tubo

tubo is a fork of [gufo](https://github.com/gufo-org/gufo), the open-source Strix Halo inference engine. It targets one chip (Ryzen AI MAX+ 392, Radeon 8060S, gfx1151) and one model family (Qwen3.8-27B: 64 layers, 48 Gated DeltaNet + 16 gated attention, plus an MTP block in the GGUF). It is based on gufo v0.19.0 plus one commit (`9a9717c1`); upstream history is kept unchanged below our commits, and the original README is in [README_gufo.md](README_gufo.md).

Everything here is measured on one machine, under one set of conditions, and marked as such. Read "Honest limits" before relying on any number. Measurements, conditions and status of every number are in [BENCHMARKS.md](BENCHMARKS.md) (row ids N01-N87 below point there). [TBD: method write-up]

## What tubo adds on top of gufo

| change | where it acts | status |
|---|---|---|
| Strix-native (SN) weight layout and kernels for IQ3_S / IQ3_XXS FFN gate+up | decode and verify, widths 1-8 | in serving, behind a switch (off by default) |
| Load-time repack to SN, byte-exact, at most +2% bytes; one native copy per layer is unpacked for prefill | model load | in serving with the switch |
| Verify-attention pipeline for turbo6 K=V (working prefetch, 32-bit addressing) | verify rounds | on by default (in builds with -DGUFO_T4_PREFETCH=1, as in the build recipe below); output bitwise identical to the old reader |
| One batched kernel replacing a loop of small GEMVs | verify rounds | on by default; bitwise equal |
| Prefill levers: rescheduled attention, wave32 KV dequant, commit-stage IQ3 gate/up, DeltaNet scan split | prefill | behind switches; bitwise identical by CPU proof (the chunked WMMA DeltaNet scan is not, see Speed mode) |
| SSD-backed embedding rows (`GUFO_DEV_EMBED_SSD`, file made by `tools/qwen27b/export_embedding_rows`) | model load, RAM | behind a switch; same bytes as the GGUF, CRC per row; saves about 1 GB RAM |
| `--speed-mode exact\|fast` and `--speed-levers` | server flags | default exact; see below |
| Turbo4/5/6 quantized KV cache for the 27B path: HIP writer, dequantizer and WMMA verify reader, ported from the TurboQuant-style block layouts of a community llama.cpp fork (upstream gufo stores fp16/fp32 KV only) | KV cache | selected with environment switches |
| Q4_K / Q5_K / Q6_K / IQ4_XS band kernels | benchmark harness only | not wired into serving yet |
| Memory-bound probes: pure-read (`tools/qwen27b/bw_probe.hip`); launch overlap and load/compute decoupling | standalone HIP files | pure-read in the tree; the other two [TBD: add to repository] |

The tree also carries an optional GPU+NPU prefill split of our own (needs an XDNA stack and a compiled kernel file; off unless GUFO_NPU_SPLIT is set). It is separate from the community fork's NPU work credited below.

## Measured effect (one machine, pinned 1700 MHz)

- IQ3 gate/up at verify width 8, same 17408 x 5120 launch: 104.4 / 119.2 GB/s (IQ3_S / IQ3_XXS) before, 182.4 / 192.5 after. Fused pair 34816 x 5120: 216.3 / 211.5 GB/s.
- Served decode, 33,712-token prompt, 512 tokens out, greedy, n=6 per arm, arms interleaved: 25.19 vs 23.10 tok/s (+9.05%), round 129.2 vs 142.7 ms (-9.5%), 0.467 vs 0.423 tok/s per watt of package power (+10.4%). The noise floor is 2.56%, so the gain is outside the noise (N38). The same binary with the path off gave 22.65 tok/s. Earlier n=3 runs (+8.5%, inside a 14.7% spread) are superseded.
- Quality: KLD of the SN IQ3 path against the native path is 6.3e-4 to 1.1e-3 over four cells (256 rows each), top-1 agreement 97.7-99.6%.
- Attention pipeline plus batched GEMV: -3.9 ms per round (-2.8%), bitwise identical output.

## Build

Requirements we used: Linux, ROCm 7.2.x with the AMD clang under the ROCm root, CMake and Ninja, a recent g++ (we used 15), a gfx1151 GPU. Build with the clang inside ROCm, not the `hipcc` wrapper, and export the ROCm root first; without it CMake misidentifies the HIP compiler and drops the HIP standard flags.

```sh
export ROCM_PATH=/opt/rocm
F="-O2 -g -DNDEBUG -std=c++20 -DGUFO_T4_PREFETCH=1"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 \
  -DGUFO_ENABLE_NPU=ON -DBUILD_TESTING=ON \
  "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=$F" "-DCMAKE_HIP_FLAGS_RELWITHDEBINFO=$F"
cmake --build build --target gufo --parallel 4
```

`-DGUFO_ENABLE_NPU=ON` is what our served build used; we also built with it OFF for a quality gate. Compiles use a lot of memory; on a 30 GiB machine keep the job count low.

Model loading, serving flags and the HTTP API are the same as upstream gufo; see its documentation. We do not distribute weights.

## Speed mode: exact by default, faster on request

The server runs the bit-exact path unless you ask otherwise. Speedups that change the output, even slightly, are opt-in:

- `--speed-mode exact` (default): nothing changes.
- `--speed-mode fast`: every lever marked "in fast" below.
- `--speed-levers a,b`: exactly the named levers (with `fast`, the union).

At startup each active lever prints one line with its measured gain and its measured output change; `GET /health` and `GET /v1/models` report `speed_mode` and `speed_levers`.

| lever | in fast | gain (27B, one machine) | output change |
|---|---|---|---|
| `gdn_chunk` | yes | prefill +2.42%, decode +2.7% | KLD vs exact 0.0006-0.0012, same top-1 96.9-99.1% |
| `draft6` | yes | decode +3.6-5.3% | KLD 0.0005-0.0013, same top-1 96.9-99.1% (N79-N80) |
| `lookup_route` | yes | prose decode +15.9% (one prompt); code/json 0% | KLD not measured; temperature-0 prose text changes (N87) |
| `vrd6` | no | verify -4.7% ms, decode -0.9% | KLD 0.0006-0.0015, same top-1 97.3-99.6% |

Full rules (env overrides, draft width reporting): [docs/SERVER.md, "Speed mode"](docs/SERVER.md#speed-mode-exact-fast-or-single-levers).

## Environment switches (names only)

Boolean switches below are set to `1` unless noted. They are developer switches; the names can change.

Served configuration used for the numbers above:

- GUFO_DEV_WEIGHTS_GTT, GUFO_DEV_SMALLBATCH_TUNED, GUFO_DEV_IQ_FAST, GUFO_DEV_KQUANT_V3, GUFO_DEV_IQ3_DUAL_GU
- GUFO_DEV_TURBO4_KV and GUFO_DEV_TURBO_KV_BITS (value 4, 5 or 6; 6 for the numbers above)
- GUFO_DEV_VERIFY_WMMA_WIDE, GUFO_DEV_BATCHED_VERIFY_ROPE, GUFO_DEV_SCRATCH_TILED, GUFO_DEV_NO_EXTRA_SYNC
- GUFO_NPU_SPLIT, GUFO_NPU_BO_SYNC (only if your NPU stack is set up)
- HSA_OVERRIDE_GFX_VERSION (see gufo's documentation for the value)

The complete served environment for the current numbers (BENCHMARKS.md row N87; paths are placeholders; HSA_OVERRIDE_GFX_VERSION as in gufo's documentation):

```sh
GUFO_DEV_ALLOW_UNPINNED_QWEN38_TEMPLATE=1 GUFO_DEV_NO_EXTRA_SYNC=1 GUFO_DEV_DFLASH_KEEP_QUANT=1
GUFO_DEV_TURBO4_KV=1 GUFO_DEV_TURBO4_DECODE=1 GUFO_DEV_TURBO4_PREFILL=1 GUFO_DEV_TURBO_KV_BITS=6
GUFO_DEV_SSM_NO_SNAPSHOT=1 GUFO_DEV_HOSTOPS=1 GUFO_DEV_HOSTOPS_FEAT=1 GUFO_DEV_HOSTOPS_FRONTIER=1
GUFO_DEV_SCRATCH_TILED=1 GUFO_DEV_DFLASH_DROP_PAGECACHE=1 GUFO_DEV_SMALLBATCH_TUNED=1 GUFO_DEV_IQ_FAST=1
GUFO_DEV_TURBO4_VERIFY_WMMA=1 GUFO_DEV_WEIGHTS_GTT=1 GUFO_DEV_KQUANT_V3=1 GUFO_DEV_VERIFY_WMMA_WIDE=1
GUFO_DEV_BATCHED_VERIFY_ROPE=1 GUFO_DEV_IQ3_DUAL_GU=1 GUFO_DEV_DRAFT_LEN_LOG=1
GUFO_NPU_SPLIT=/path/to/gemm.xclbin GUFO_NPU_BO_SYNC=0 GUFO_NPU_MAX_M=4096
GUFO_NPU_ROWS=1536 GUFO_NPU_GU_ROWS=7424 GUFO_NPU_STORE_PCT=35
GUFO_SN_K4=iq3 GUFO_DEV_SN_IQ3_SERVE=1 GUFO_DEV_ATTN6_V3=1 GUFO_DEV_ATTN6_SPLITS=0 GUFO_DEV_IQ3_PACKED=1
GUFO_DEV_DEQ_V2=1 GUFO_DEV_IQ3_CS=1 GUFO_DEV_EMBED_SSD=/path/to/model.embrows
GUFO_NPU_BCACHE=/path/to/bcache.gnpb GUFO_NPU_BCACHE_TAGGED=1
# server flags: --draft-tokens 7 (the default) with a fixed draft policy; [TBD: list the remaining server flags of the benchmark runner]
```

The NPU kernel file (`GUFO_NPU_SPLIT`) is built outside this repository. [TBD: publish its source and build steps, or drop the NPU lines from the public served config]

tubo-specific, all off unless set (the served configuration sets every one of them except GUFO_DEV_VERIFY_T6_PIPE and the per-plane KV widths):

- GUFO_DEV_ATTN6_V3, GUFO_DEV_ATTN6_SPLITS (value `0` in the served configuration), GUFO_DEV_IQ3_PACKED, GUFO_DEV_DEQ_V2, GUFO_DEV_IQ3_CS
- GUFO_DEV_EMBED_SSD (path to an `.embrows` file), GUFO_NPU_BCACHE (path to an NPU B-cache file) with GUFO_NPU_BCACHE_TAGGED

- GUFO_SN_K4 (takes the string `iq3`) together with GUFO_DEV_SN_IQ3_SERVE: the SN IQ3 serving path. A model loaded with SN tags refuses to run without both.
- GUFO_DEV_VERIFY_T6_PIPE: on by default; set to `0` to get the old reader.
- GUFO_DEV_TURBO_KV_K_BITS, GUFO_DEV_TURBO_KV_V_BITS: per-plane widths; they need GUFO_DEV_TURBO4_KV.

## Numerics changes

Anything that can change generated text. Temperature-0 text is not guaranteed to match upstream when these are on.

1. SN IQ3 path (off by default): activations are quantized with one scale per 256 values; the native IQ3 path uses one per 16. Measured KLD against native 6.3e-4 to 1.1e-3. Temperature-0 text differs from the native path. Weight repack itself is bit-exact.
2. turbo6 (and turbo5) KV: lossy by design. KLD against fp16 KV 2.0-3.3e-4 on our model.
3. IQ3_DUAL_GU (prefill): measured KLD 4.1e-4 against the native path on prose cuts (a different method from item 1); prefill is about 13% faster.
4. NPU split (prefill, optional): activations are bf16-rounded and results are added from a block-float path; KLD gate passed at 5.3e-4 to 9.2e-4 against GPU-only on three text kinds.
5. Attention pipeline, batched GEMV: bitwise identical output in our tests; no numeric change intended.
6. The merge with upstream replaced our recurrent-state rollback with upstream's. Upstream claims bit-exact results. We have not yet compared temperature-0 text against our previous binary on a GPU. [TBD: GPU check after the merge]

## Honest limits

- gfx1151 only. Not tested on other AMD GPUs, NVIDIA, or Windows.
- One model family at 27B. Other models are untested; the kernels are shape-specific (fused gate/up of 5120 x 17408).
- Speed numbers are pinned at 1700 MHz. Daily use at the default clock is faster; our A/B comparisons are pinned.
- The serving A/B for the SN IQ3 path (+9.05% tok/s, n=6, one prompt, one machine) is outside run noise (N38). It says nothing about quality; see the next item.
- The SN IQ3 path fails our own quality bar (KLD 5e-4). Enable it only if you accept the trade-off.
- Power numbers for serving are whole-package (CPU included), not GPU only.
- The band kernels for Q4_K, Q5_K, Q6_K and IQ4_XS are microbenchmark results and are not in the serving path.
- The merge with upstream has passed CPU tests (64 of 66; the two failures also fail before the merge). GPU tests have not run on the merged tree. [TBD]
- Memory budget: tested within 24 GiB allocatable. Longer contexts than 90,112 tokens with the drafter were not run.
- ROCm 10.1 (which keeps the any-order launch flag through graph capture) is untested.

## Credits

gufo and its contributors (base engine, MIT); llama.cpp and ggml (formats, reference kernels); the community gufo fork with a GPU+NPU prefill split; Strata and Maya (design notes read); AMD's ROCm graph-capture fix; the community llama.cpp fork whose TurboQuant-style KV layouts turbo5/turbo6 follow; hipfire. See THIRD_PARTY_NOTICES.md.

## License

MIT, inherited from gufo. Third-party code keeps its own license; see NOTICE and THIRD_PARTY_NOTICES.md. [TBD: confirm that the probe files added carry the same header.]

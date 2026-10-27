Written by Claude (an AI model by Anthropic) on behalf of the account owner.

# tubo

tubo is a faster fork of [gufo](https://github.com/gufo-org/gufo) for one chip: AMD Strix Halo (Ryzen AI MAX+ 392 / Radeon 8060S, gfx1151), running Qwen3.8-27B. By default the output is the same text our reference build produces; nothing is traded for speed unless you ask for it.

## Highlights

Same laptop (32 GB), same model (Qwen3.8-27B UD-Q4_K_XL + DFlash2 drafter), same fp16 KV cache, same prompt. Only the engine differs.

| | gufo | tubo |
|---|--:|--:|
| decode speed | 29.6 tok/s | **37.8 tok/s (+28%)** |
| time for a 512-token answer | 18.4 s | **14.6 s (-20%)** |
| prefill / time to first token | 268 tok/s / 2.24 s | 276 tok/s / 2.19 s (same) |
| longest context that loads on 32 GB | 4k | **32k** |
| HumanEval (164 problems) | 156 | 155 (difference is noise) |

With the smaller **apex** model and the 6-bit **turbo6** KV cache (both explained below): 53 tok/s over the whole HumanEval run, about 17.6 GiB of RAM at 4k context, and a 128k-token prompt with all 40 hidden facts found. This build of the repo gives the same output text and the same speed as the binary behind these numbers.

Full tables, conditions and caveats: [BENCHMARKS.md](BENCHMARKS.md). Rerun everything on your machine: [REPRODUCE.md](REPRODUCE.md).

## Words used here

- **UD-Q4**: Unsloth's 4-bit GGUF of Qwen3.8-27B (17.6 GB). Both engines load it, so every engine-vs-engine number uses it.
- **apex**: a smaller (14.2 GB) mixed 3-4-bit GGUF of a fine-tuned Qwen3.8-27B, by IsValorum (credits below). It is our daily model. Vanilla gufo cannot load it (its loader has no IQ3_XXS), and it is a different model, so it is judged on quality scores, never as an engine comparison.
- **turbo6**: a 6-bit KV cache format (TheTom's llama-cpp-turboquant, turbo5/turbo6 by JakeATX), 2.6x smaller than fp16. It changes the output slightly, so it is tested for quality (BENCHMARKS.md section 2).
- **DFlash2 drafter**: a small model that guesses the next tokens so the big one can check 8 at a time (speculative decoding). Both engines use it.
- **exact mode** (default): output identical to our reference build. **fast mode** (`--speed-mode fast`, opt-in): faster but changes the text slightly; its quality cost is not measured yet.

**Compare on the same machine only.** Our laptop is not the machine behind upstream's published numbers: a Ryzen AI MAX+ 392 (12 CPU cores) in a 32 GB laptop with a laptop power limit, against their Ryzen AI MAX+ 395 (16 cores) with 128 GB. The GPU is the same Radeon 8060S, yet vanilla gufo runs slower here than in their table (decode with the drafter 26-34% slower at 1700 MHz or the stock clock, N74), and we have not found why (N75). So every tubo-vs-vanilla number here was measured on this laptop, both engines, same settings. Do not compare our numbers with upstream's published table; to check our claims, run both engines on your own machine.

## Details

Base: gufo v0.19.0 plus one commit (`9a9717c1`); upstream history is kept unchanged below our commits, and the original README is in [README_gufo.md](README_gufo.md). Model family: Qwen3.8-27B (64 layers, 48 Gated DeltaNet + 16 gated attention, plus an MTP block in the GGUF). Row ids N01-N90 point into [BENCHMARKS.md](BENCHMARKS.md). Read "Honest limits" below before relying on any number.

## What tubo adds on top of gufo

| change | where it acts | status |
|---|---|---|
| Strix-native (SN) weight layout and kernels for IQ3_S / IQ3_XXS FFN gate+up | decode and verify, widths 1-8 | in serving, behind a switch (off by default) |
| Load-time repack to SN, byte-exact, at most +2% bytes; one native copy per layer is unpacked for prefill | model load | in serving with the switch |
| Verify-attention pipeline for turbo6 K=V (working prefetch, 32-bit addressing) | verify rounds | on by default (the CMake option `GUFO_T4_PREFETCH`, ON by default); output bitwise identical to the old reader |
| Turbo6 verify attention: Q-in-register reader, fused rotate + reader + reduce, reader layout changes | verify rounds | behind switches; bitwise equal to the reader they replace |
| One batched kernel replacing a loop of small GEMVs; fused tiny verify projections (K+V, SSM alpha+beta, shared activation quantize) | verify rounds | on by default / behind a switch; bitwise equal |
| DFlash2 drafter: split attention route, compile-time top-k selector | draft rounds | behind switches; bit-exact |
| Prefill levers: rescheduled attention, wave32 KV dequant, commit-stage IQ3 gate/up, DeltaNet scan split | prefill | behind switches; bitwise identical by CPU proof (the chunked WMMA DeltaNet scan is not, see Speed mode) |
| SSD-backed embedding rows (`GUFO_DEV_EMBED_SSD`, file made by `tools/qwen27b/export_embedding_rows`) | model load, RAM | behind a switch; same bytes as the GGUF, CRC per row; saves about 1 GB RAM |
| `--speed-mode exact\|fast` and `--speed-levers` | server flags | default exact; see below |
| Turbo4/5/6 quantized KV cache for the 27B path: HIP writer, dequantizer and WMMA verify reader, ported from TheTom/llama-cpp-turboquant (turbo4) and JakeATX's turbo5/turbo6 types (upstream gufo stores fp16/fp32 KV only) | KV cache | selected with environment switches |
| Q4_K / Q5_K / Q6_K / IQ4_XS band kernels | benchmark harness only | not wired into serving yet |
| Memory-bound probes: pure read, launch overlap, load/compute decoupling | [tools/probes/](tools/probes/README.md) | standalone HIP files |

The tree also carries an optional GPU+NPU prefill split of our own (needs an XDNA stack and a compiled kernel file; off unless GUFO_NPU_SPLIT is set). It is separate from the community fork's NPU work credited below.

## Measured effect (one machine)

- Engine only, same model (UD-Q4_K_XL + DFlash2), same fp16 KV, pinned 2000 MHz: decode 37.8 vs 29.6 tok/s (+28%), wall time per request 14.6 vs 18.4 s (-20%), prefill and time to first token equal within noise. Upstream's own benchmark driver: prose prefill +24%, decode +9% (BENCHMARKS.md section 1). A build of this repository gives the same output text and the same speed as the binary behind these numbers.
- IQ3 gate/up at verify width 8, same 17408 x 5120 launch, pinned 1700 MHz: 104.4 / 119.2 GB/s (IQ3_S / IQ3_XXS) before, 182.4 / 192.5 after. Fused pair 34816 x 5120: 216.3 / 211.5 GB/s.
- Served decode, 33,712-token prompt, 512 tokens out, greedy, pinned 1700 MHz, n=6 per arm, arms interleaved: 25.19 vs 23.10 tok/s (+9.05%), round 129.2 vs 142.7 ms (-9.5%), 0.467 vs 0.423 tok/s per watt of package power (+10.4%). The noise floor is 2.56%, so the gain is outside the noise (N38). The same binary with the path off gave 22.65 tok/s.
- Quality: KLD of the SN IQ3 path against the native path is 6.3e-4 to 1.1e-3 over four cells (256 rows each), top-1 agreement 97.7-99.6% (N40).
- Attention pipeline plus batched GEMV: -3.9 ms per round (-2.8%), bitwise identical output (N39).

## Build

Requirements we used: Linux, ROCm 7.2.x with the AMD clang under the ROCm root, CMake and Ninja, a recent g++ (we used 15), a gfx1151 GPU. The exact build line of the served binary is in [REPRODUCE.md](REPRODUCE.md). In short:

```sh
export ROCM_PATH=/opt/rocm
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=g++-15 -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 -DGUFO_ENABLE_NPU=ON
cmake --build build --target gufo --parallel 4
```

Model loading, serving flags and the HTTP API are the same as upstream gufo; see its documentation. We do not distribute weights.

## Speed mode: exact by default, faster on request

The server runs the bit-exact path unless you ask otherwise. Speedups that change the output, even slightly, are opt-in:

- `--speed-mode exact` (default): nothing changes.
- `--speed-mode fast`: every lever marked "in fast" below (today only `lookup_route`).
- `--speed-levers a,b`: exactly the named levers (with `fast`, the union).

At startup each active lever prints one line with its measured gain and its measured output change; `GET /health` and `GET /v1/models` report `speed_mode` and `speed_levers`. Fast mode as a whole: its KLD against exact is not measured yet (BENCHMARKS.md section 3), so treat the gains below as unverified for quality.

Numbers: 27B apex model, GPU clock pinned at 1700 MHz, decode tok/s against exact on one prose, one code and one json prompt.

| lever | in fast | gain | output change |
|---|---|---|---|
| `lookup_route` | yes | decode prose +15.7%, code -0.2%, json 0.0% | KLD not measured; temperature-0 prose text changes; code/json text identical to exact |
| `gdn_chunk` | no | prefill +2.42%; decode prose +3.1%, code -1.0%, json -17.7% | KLD vs exact 0.00103 |
| `draft6` | no | decode prose +4.1%, code -10.1%, json -17.2% | KLD 0.0005-0.0013, same top-1 96.9-99.1% (N79, N80) |
| `vrd6` | no | verify -4.7% ms, decode -0.9% | KLD 0.0006-0.0015, same top-1 97.3-99.6% |

Full rules (env overrides, draft width reporting): [docs/SERVER.md, "Speed mode"](docs/SERVER.md#speed-mode-exact-fast-or-single-levers).

## Environment switches

The served configuration is [docs/tubo.env](docs/tubo.env): every key with one line on what it does, values as served, paths as placeholders. They are developer switches; the names can change.

tubo-specific switches that are off unless set:

- GUFO_SN_K4 (takes the string `iq3`) together with GUFO_DEV_SN_IQ3_SERVE: the SN IQ3 serving path. A model loaded with SN tags refuses to run without both.
- GUFO_DEV_VERIFY_T6_PIPE: on by default; set to `0` to get the old reader.
- GUFO_DEV_TURBO_KV_K_BITS, GUFO_DEV_TURBO_KV_V_BITS: per-plane widths; they need GUFO_DEV_TURBO4_KV.
- GUFO_DEV_NORMQ (fused RMSNorm + activation quantize; GPU op test `qwen_normq_op_test`) and GUFO_DEV_HEADQ6 (Q6_K output head row groups): written to be bitwise equal to the path they replace; not in the served configuration.

## Numerics changes

Anything that can change generated text. Temperature-0 text is not guaranteed to match upstream when these are on.

1. SN IQ3 path (off by default): activations are quantized with one scale per 256 values; the native IQ3 path uses one per 16. Measured KLD against native 6.3e-4 to 1.1e-3. Temperature-0 text differs from the native path. Weight repack itself is bit-exact.
2. turbo6 (and turbo5) KV: lossy by design. KLD against fp16 KV 2.0-3.3e-4 on our model (N46).
3. IQ3_DUAL_GU (prefill): measured KLD 4.1e-4 against the native path on prose cuts (a different method from item 1); prefill is about 13% faster.
4. NPU split (prefill, optional): activations are bf16-rounded and results are added from a block-float path; KLD gate passed at 5.3e-4 to 9.2e-4 against GPU-only on three text kinds.
5. Attention pipeline, batched GEMV, fused verify projections, drafter and selector routes: bitwise identical output in our tests; no numeric change intended.
6. Speculative decoding is not token-identical to one-token-at-a-time decoding: the 8-row verify pass adds floats in a different order than the 1-row path, so a near-tie can resolve differently (N87).
7. The merge with upstream replaced our recurrent-state rollback with upstream's. Upstream claims bit-exact results. On the merged tree, HumanEval (155 of 164 vs vanilla 156) and the long-context needle test (60/60 at 32k) pass (BENCHMARKS.md section 2). Temperature-0 text against our pre-merge binary was not compared.

## Honest limits

- gfx1151 only. Not tested on other AMD GPUs, NVIDIA, or Windows.
- One model family at 27B. Other models are untested; the kernels are shape-specific (fused gate/up of 5120 x 17408).
- A/B speed numbers are taken at a pinned GPU clock (1700 or 2000 MHz, written beside each). Daily use at the default clock differs.
- The serving A/B for the SN IQ3 path (+9.05% tok/s, n=6, one prompt, one machine) is outside run noise (N38). It says nothing about quality; see the next item.
- The SN IQ3 path fails our own quality bar (KLD 5e-4). Enable it only if you accept the trade-off.
- Power numbers for serving are whole-package (CPU included), not GPU only.
- The band kernels for Q4_K, Q5_K, Q6_K and IQ4_XS are microbenchmark results and are not in the serving path.
- The merge with upstream has passed CPU tests (64 of 66; the two failures also fail before the merge). GPU tests on this tree (2026-10-27): graph ops, Turbo verify attention (0 failures), fused norm+quantize (bit-identical) and the SSM/DeltaNet ops pass. Four arena sub-tests of the SSM test abort ("KV plane size computed as zero": the test builds a model with no attention layers, which the KV allocator rejects); `GUFO_SSM_TEST_GDN_ONLY=1` skips them. Not fixed yet.
- Memory budget: tested within 24 GiB allocatable. Longest prompts run with the drafter: 128k tokens (apex + turbo6); UD-Q4 + turbo6 was tested to 64k.
- ROCm 10.1 (which keeps the any-order launch flag through graph capture) is untested.

## Credits

We built on other people's work and took ideas from more. Thank you to:

**Code and files we use** (their licenses apply; see [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)):

- [gufo](https://github.com/gufo-org/gufo) and its contributors: the base engine (MIT).
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp): the weight block formats and reference kernels (K-quants and IQ formats by ikawrakow) (MIT).
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant): the turbo4 KV cache type and pipeline, whose block layout and constants our HIP code follows (MIT).
- [JakeATX](https://github.com/JakeATX): the turbo5/turbo6 KV cache types (bit layouts, codebooks, reference codec) we ported, written in [PR #381](https://github.com/TheTom/llama-cpp-turboquant/pull/381) to TheTom/llama-cpp-turboquant (MIT).
- [amd/IRON](https://github.com/amd/IRON): the toolkit our NPU matmul program is built with, and the packing routine our NPU weight packer ports (Apache-2.0).
- [z-lab/dflash](https://github.com/z-lab/dflash): the DFlash2 draft model used for speculative decoding.

**Models** (not distributed; their own licenses apply):

- The Qwen team (Qwen3.8-27B) and Unsloth (the UD-Q4_K_XL quantization).
- For the apex model (Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated, VAL-APEX-I-MiniPlus-V3 GGUF): quantization by [IsValorum](https://huggingface.co/IsValorum/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-VAL-APEX-I-MiniPlus-V3-GGUF), abliteration by [huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated), fine-tune by [nerkyor](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2), base model Qwen3.8-27B by the Qwen team.

**Ideas we borrowed** (no code copied):

- [mikealanni/gufo](https://github.com/mikealanni/gufo): splitting prefill between the GPU and the NPU. (Its Flash-Next row-reuse change, which upstream gufo adapted, is credited in upstream's NOTICE.)
- [Strata](https://github.com/Niko1221/Strata), [Project Maya](https://github.com/mw00/project-maya) and [hipfire](https://github.com/warpfront/hipfire): design notes and kernel ideas we read.
- AMD ROCm (rocm-systems #9421): the graph-capture fix we waited for.

**Papers:**

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023, [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).
- Chen, Liang, Liu, "DFlash: Block Diffusion for Flash Speculative Decoding", [arXiv:2602.06036](https://arxiv.org/abs/2602.06036).
- Zandieh, Daliri, Hadian, Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 52(4), 2009 (the "% of theory" bound).

If we left out someone whose work we used, that is a mistake, not an intentional omission. If you think credit is due, please open an issue and we will add it.

## License

MIT, inherited from gufo. Third-party code keeps its own license; see [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Files we added fall under the repository's MIT license, except the ported code that names its source in its header: `npu_bfp16_pack.hpp` (Apache-2.0, AMD IRON) and the Turbo KV codec files (MIT, ggml authors); license texts in [licenses/](licenses/).

Written by Claude (an AI model by Anthropic) on behalf of the account owner.

# Reproducing tubo's numbers

Every comparison in [BENCHMARKS.md](BENCHMARKS.md) runs both engines on the same machine. Run them on yours; do not compare your numbers with ours or with upstream's published table (different machines, see the "Compare on the same machine only" paragraph there).

Placeholders: `<tubo>` is a checkout of this repository, `<vanilla>` a checkout of upstream gufo, `<models>` the directory holding the GGUF files, `<out>` any scratch directory.

## 1. Build tubo (the served build)

Linux, ROCm 7.2.x, CMake, Ninja, g++ 15, a gfx1151 GPU. Use the clang inside ROCm, not the `hipcc` wrapper, and export the ROCm root first: without it CMake does not identify the HIP compiler and drops the HIP standard flags.

```sh
cd <tubo>
export ROCM_PATH=/opt/rocm
F="-O2 -g -DNDEBUG -std=c++20"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 \
  -DGUFO_ENABLE_NPU=ON -DBUILD_TESTING=ON \
  "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=$F" "-DCMAKE_HIP_FLAGS_RELWITHDEBINFO=$F"
cmake --build build --target gufo --parallel 4
```

- `GUFO_T4_PREFETCH` (the pipelined Turbo6 verify reader) is a CMake option, ON by default; the served binary has it on. `-DGUFO_T4_PREFETCH=OFF` builds the plain reader.
- `-DGUFO_ENABLE_NPU=ON` builds the optional GPU+NPU prefill split and needs the XRT userspace library. Without an NPU stack, build with `OFF` and drop the `GUFO_NPU_*` lines from the environment.
- Compiles take a lot of memory; on a 32 GB machine keep `--parallel` low.

## 2. Build vanilla gufo

```sh
git clone https://github.com/gufo-org/gufo <vanilla>
cd <vanilla> && git checkout 1b8ad798
export ROCM_PATH=/opt/rocm
cmake --preset release -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 -DGUFO_REVISION=1b8ad798
cmake --build --preset release --parallel 4      # binary: build/release/gufo
```

## 3. Models

The same files for both engines (not distributed here; see each model card for its license):

- target: Qwen3.8-27B UD-Q4_K_XL GGUF (Unsloth), `<models>/Qwen3.8-27B-UD-Q4_K_XL.gguf`
- drafter: Qwen3.8-27B DFlash2 Q4_K_M GGUF (z-lab), `<models>/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`

The exact upstream revisions are linked in upstream's README (`README_gufo.md`, model table).

## 4. Environment for tubo

Copy [docs/tubo.env](docs/tubo.env), fill in the `<...>` paths, then for the engine-only comparison drop every line whose key contains `TURBO` (that keeps the KV cache fp16, like vanilla):

```sh
grep -v TURBO docs/tubo.env | grep -E '^[A-Z]' > <out>/tubo-fp16kv.env   # edit the <...> paths in it
```

`GUFO_DEV_EMBED_SSD` needs an `.embrows` file made once from the target GGUF (`tools/qwen27b/README.embedding-ssd.md`); remove the line to keep embeddings in RAM. `GUFO_NPU_SPLIT` needs the compiled NPU GEMM kernel; remove the `GUFO_NPU_*` lines to run GPU only.

The NPU split needs locked memory. An interactive shell often has a small `ulimit -l`, and then the server logs `npu split: init failed` and silently runs GPU only. Run the server where the memlock limit is unlimited (for example a systemd user unit), and check the log for that line: a run that printed it is a GPU-only run.

## 5. Engine-only A/B with upstream's own driver

The driver is upstream's `tools/bench/model-bench.py` with upstream's `docs/models/qwen3.8-27b/artifacts/bench.json`; use the copies from `<vanilla>` for both engines, so both are measured by the same code. The driver starts the server itself on a private port.

Two config copies add the server flags we used (`--cache-ram-bytes` for both; tubo also runs the default bit-exact draft width with a fixed draft policy):

```sh
cd <vanilla>
python3 - <<'PY'
import json
b = json.load(open("docs/models/qwen3.8-27b/artifacts/bench.json"))
for name, extra in (("vanilla", ["--cache-ram-bytes", "2147483648"]),
                    ("tubo", ["--cache-ram-bytes", "2147483648", "--draft-tokens", "7",
                              "--min-draft-tokens", "1", "--draft-policy", "fixed"])):
    d = json.loads(json.dumps(b)); d["gufo"]["llm"] += extra
    json.dump(d, open(f"<out>/bench_{name}.json", "w"), indent=1)
PY
```

Then, with the GPU clock pinned (we used 2000 MHz; see "Clock" below), AC power on and nothing else using the GPU, run one warm-up cell and throw it away, then each arm with 3 repetitions. Interleave the arms if you repeat the whole run (vanilla, tubo, tubo, vanilla, ...).

```sh
cd <vanilla>
G="--model qwen3.8-27b --gguf q4=<models>/Qwen3.8-27B-UD-Q4_K_XL.gguf --draft <models>/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
RUN="run --target gufo --table single-dflash2-q4 --depths 0 --mode dflash2 --context 4096"

# warm-up (discard)
python3 tools/bench/model-bench.py $G --gufo build/release/gufo --config <out>/bench_vanilla.json \
  --artifacts-dir <out>/warm --log-dir <out>/warm-logs $RUN --repetitions 1

# vanilla
python3 tools/bench/model-bench.py $G --gufo build/release/gufo --config <out>/bench_vanilla.json \
  --artifacts-dir <out>/vanilla --log-dir <out>/vanilla-logs $RUN --repetitions 3

# tubo
env $(cat <out>/tubo-fp16kv.env) \
python3 tools/bench/model-bench.py $G --gufo <tubo>/build/gufo --config <out>/bench_tubo.json \
  --artifacts-dir <out>/tubo --log-dir <out>/tubo-logs $RUN --repetitions 3
```

- `--artifacts-dir` keeps the results out of upstream's `docs/models/` tree. Do not run the driver's `render` command against upstream's `BENCHMARKS.md`: that file holds upstream's own machine's numbers.
- Each artifact JSON holds prefill (pp) and decode (tg) tok/s per prompt kind, and the exact server command line of every row. The server logs are in `--log-dir`.
- `--depths 0 --context 4096` is the cell both engines load on a 32 GB machine. On a larger machine add depths (for example `--depths 0,32768 --context 36864`).
- Autoregressive (no drafter) cells: `--table single-ar-q4 --mode ar`; for tubo use the vanilla config there, because `--draft-policy` is refused without DFlash2.

Same rules for any A/B we report: same binary flags except the engine, same prompt, the first run of a series dropped as warm-up, and at least 9 runs per arm when the gain is small (a few percent); report the spread inside each arm next to the difference between arms.

### Clock

We pin the GPU core clock for every A/B so runs compare: 1700 MHz in the older rows, 2000 MHz in the current ones (written beside each number). Pinning uses the amdgpu sysfs interface and needs root:

```sh
D=$(dirname /sys/bus/pci/drivers/amdgpu/0000:*/power_dpm_force_performance_level)
echo manual   > $D/power_dpm_force_performance_level
echo "s 0 2000" > $D/pp_od_clk_voltage      # minimum = maximum = 2000 MHz
echo "s 1 2000" > $D/pp_od_clk_voltage
echo c        > $D/pp_od_clk_voltage
# release afterwards:
echo r > $D/pp_od_clk_voltage; echo c > $D/pp_od_clk_voltage; echo auto > $D/power_dpm_force_performance_level
```

This is what we ran on our kernel; check the interface on yours first. Keep the platform power profile the same for both arms.

## 6. Quality table (MATH-500)

BENCHMARKS.md section 2 runs one problem set against three tubo configurations (UD-Q4_K_XL with fp16 KV, UD-Q4_K_XL with turbo6 KV, the apex model with turbo6 KV), same server flags, temperature 0. Grading script and prompt format: <TBD-math500>.

## 7. Pure-read DRAM probe

The memory ceiling numbers (BENCHMARKS.md section A, N01-N07) come from `tools/probes/dram_stream/m99.hip`. Full description, configs and the three measurement settings: [tools/probes/README.md](tools/probes/README.md), section "Reproduce the pure-read numbers". In short:

```sh
cd <tubo>/tools/probes
hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o m99 dram_stream/m99.hip
# pin the GPU core clock to 1700 MHz, AC power on
for i in 1 2 3 4 5 6 7 8 9; do
  ./m99 ab 21626880 host bu8_u8_l1_c0_p0_g40_t256_r28,bu16_u1_l4_c0_p0_g80_t256_k32_r28 > R$i.log
done
python3 dram_stream/agg.py R1.log R2.log R3.log R4.log R5.log R6.log R7.log R8.log R9.log
```

Run it once on your normal desktop, then again with the screen off and other GPU/CPU work stopped. The percentages printed divide by 256 GB/s (LPDDR5X-8000 on a 256-bit bus); change the divisor if your memory differs.

## 8. Tests

CPU tests run anywhere: `ctest --test-dir build -L cpu`. GPU tests (label `gpu`) need a gfx1151 and take the GPU for a while.

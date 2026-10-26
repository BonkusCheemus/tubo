# Measurement scripts

Probes used to measure how close a decode kernel gets to the DRAM limit of one AMD APU (gfx1151, LPDDR5X-8000 on a
256-bit bus, theoretical 256 GB/s). Each probe is a small HIP program plus the shell and Python glue that ran it.
Every file starts with a header: what it measures, how to run it, and a "wrong if" line (the result that would
show the measurement is wrong). Shell scripts take `--help`.

Nothing here is built or run for you. Build with `hipcc` (ROCm), run on a quiet machine.

| directory | files | question |
| :-- | :-- | :-- |
| `ceiling/` | `ceil.hip`, `runall.sh`, `merge.py` | What is the best pure-read rate at the size of one weight matrix, over many launch shapes? |
| `overlap/` | `magic.hip`, `runall.sh`, `runfinal.sh`, `merge.py`, `trsum.py` | Does letting consecutive launches overlap (any-order launch, dependent windows) lift that rate? |
| `dram_stream/` | `m99.hip`, `m99one.hip`, `agg.py`, `binstat.py`, `period.py`, `clocks.sh` | What is the best sustained stream rate over a window of 1 GB or more, and where do the dips come from? |
| `decoupling/` | `dcp.hip`, `probes.sh`, `budget.py` | Is the loss of a batch-8 kernel caused by load/compute coupling, and how big is each loss term? |
| `decoupled_q5k/` | `dq5.hip`, `run.sh` | A real Q5_K kernel built on the decoupled design. **It failed**: 171 GB/s against 206 GB/s for the coupled kernel (R=4); read the header of `dq5.hip`. |

## Environment variables

No script contains a device path. They read these and stop with a clear error if one is missing.

| variable | meaning | used by |
| :-- | :-- | :-- |
| `GPURUN` | the pinned-clock wrapper (see below), called as `$GPURUN <log file> <command...>` | all `.sh` runners |
| `CEIL_BIN`, `MAGIC_BIN`, `M99_BIN`, `DCP_BIN`, `DQ5_BIN` | path to the built binary | the runner of that probe |
| `GPU_SYSFS_DIR` | the amdgpu device directory in sysfs (the one that holds `gpu_metrics`) | `magic` (METRICS, pwr), `m99`/`m99one` (ab, pwr), `clocks.sh` |
| `GPU_POWER_FILE` | the hwmon `power1_input` file of that device (microwatts) | `magic pwr`, `m99 pwr` |
| `GPURUN_MHZ` | set by `clocks.sh`; the wrapper must pin the core clock to this value when it is set | `clocks.sh` |

## The pinned-clock wrapper (not included)

The wrapper is a short shell script tied to one machine's device paths and service manager, so it is described here
and not copied. It does four things around one command:

1. Take a lock, so only one benchmark runs on the GPU at a time (`flock` on a lock file).
2. Pin the GPU core clock to a fixed value (we used 1700 MHz as the test clock) through the amdgpu sysfs interface.
   Writing those files needs root.
3. Wait for AC power before starting (on battery the clocks and memory behave differently), and check again at the end.
4. Log power during the run (sample the hwmon `power1_input` file about every 0.2 s and write mean and maximum
   into the log), log the core clock before and after, and append any GPU fault lines found in the kernel log
   since the start (ring timeout, GPU reset, page fault).

Then it runs `<command...>` with a hard timeout, appending the command's output to the log file.

## Quiet runs

For the quiet runs we paused background work and turned the screen off.

## Build lines

```
hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o ceil  ceiling/ceil.hip
hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o magic overlap/magic.hip
hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o m99   dram_stream/m99.hip
hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o dcp   decoupling/dcp.hip
```

`dq5.hip` does not build alone: it includes kernel headers from the tubo source tree (see its header).

## Conventions

- Timing follows the repository's GEMV benchmark: one event pair around 24 back-to-back launches, divided by 24.
  The buffer is copied `ceil(128 MiB / bytes)` times and launch `i` reads copy `i % copies`, so data is not
  served from cache.
- Memory kinds: `host` = anonymous hugepage mmap, registered read-only and mapped into the GPU (how the served
  weights are held); `dev` = ordinary device memory.
- Percent-of-peak figures printed by `m99` and `agg.py` divide by 256 GB/s. Change the divisor if your memory differs.
- The constants in `decoupling/budget.py` (GB/s per cell, instruction counts) are our measurements; replace them
  with yours.

## Reproduce the pure-read numbers (BENCHMARKS.md section A)

The ~99% pure-read figure comes from a **controlled** run, not from normal use. Three settings, same probe,
same config, numbers from our machine:

| setting | what we did | median of 9 runs | % of 256 GB/s |
| :-- | :-- | --: | --: |
| normal desktop | nothing paused, screen on (N04, 80-workgroup 32 KB-chunk config) | 250.8 GB/s (249.6-251.8) | 98.0 |
| screen off | screen off, background processes still running (N03) | 252.2 GB/s | 98.5 |
| quiet | every other GPU/CPU user paused (agents, terminal UI), screen off, start below 60 C (N01) | 252.8 GB/s | 98.8 |

The single best 50 µs window (253.8 GB/s) is a peak, not a rate; quote the medians.

What the probe does: one persistent kernel reads a 21.6 MB weight-sized buffer, held as host-mapped read-only
memory in 7 rotating copies, 28 times per timed window (over 1 GB per window), timed with an event pair and the
GPU's own wall clock. No math, no writes: it measures the memory system, not a model.

Steps:

```
hipcc -O3 --offload-arch=gfx1151 -std=c++20 -o m99 dram_stream/m99.hip
# pin the GPU core clock to 1700 MHz (root, amdgpu sysfs), AC power on
for i in 1 2 3 4 5 6 7 8 9; do
  ./m99 ab 21626880 host bu8_u8_l1_c0_p0_g40_t256_r28,bu16_u1_l4_c0_p0_g80_t256_k32_r28 > R$i.log
done
python3 dram_stream/agg.py R1.log R2.log R3.log R4.log R5.log R6.log R7.log R8.log R9.log
```

Run it once on your normal desktop, then again with the screen off and other work stopped. The gap between the
two is the share your desktop takes. Configs: `bu8 ... g40` is the 40-workgroup static arm (quiet rows),
`bu16 ... g80 ... k32` the 80-workgroup 32 KB-chunk arm (desktop row); the cfg grammar is in the header of `m99.hip`.

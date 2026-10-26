#!/bin/bash
# clocks.sh -- does the GPU core clock change the DRAM stream rate?
#
# MEASURES: the two best m99 configurations (static chunks bu8_u8_l1_g40 and dynamic 32 KB chunks bu16_u1_l4_g80_k32)
#   at four pinned core clocks (1000, 1300, 1700, 2000 MHz), 3 'ab' runs per clock, 21626880-byte host-mapped buffer.
#
# RUN:   GPURUN=<wrapper> M99_BIN=<path to built m99> GPU_SYSFS_DIR=<dir> ./clocks.sh   (logs K<mhz>_<i>.log in the cwd)
#        GPURUN is called as:  $GPURUN <log file> <command...>  and must pin the core clock to $GPURUN_MHZ, which
#        this script sets for each step (see README.md). Summarise with:  python3 agg.py K1700_*.log
#
# WRONG IF: the sclk line the wrapper writes into each log differs from the requested clock by more than a few
#   percent (the pin did not take effect); or the three runs at one clock disagree by more than the difference
#   between clocks (the machine was not quiet).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

: "${GPURUN:?set GPURUN to the pinned-clock wrapper (see README.md)}"
: "${M99_BIN:?set M99_BIN to the built m99 binary}"
: "${GPU_SYSFS_DIR:?set GPU_SYSFS_DIR to the amdgpu device directory in sysfs (see README.md)}"
G=$GPURUN
C=bu8_u8_l1_c0_p0_g40_t256_r28,bu16_u1_l4_c0_p0_g80_t256_k32_r28
for mhz in 1000 1300 1700 2000; do for i in 1 2 3; do
  GPURUN_MHZ=$mhz $G K${mhz}_$i.log "$M99_BIN" ab 21626880 host $C; done; done

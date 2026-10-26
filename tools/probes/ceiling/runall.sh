#!/bin/bash
# runall.sh -- run the full ceil sweep: 3 independent passes (p1..p3) over every size x memory kind.
#
# MEASURES: pure-read GB/s ceiling at the buffer sizes of the weight matrices (17.7, 21.6, 25.8, 35.4, 67.1 MB
#   and 256 MB with three workgroup sizes), host-mapped and device memory, plus a one-copy variant. Each
#   invocation = 3 reps x 3 rounds x 24 launches per config, run through the pinned-clock wrapper.
#
# RUN:   GPURUN=<wrapper> CEIL_BIN=<path to built ceil> ./runall.sh        (logs go to the current directory)
#        GPURUN is called as:  $GPURUN <log file> <command...>   (see README.md: lock, pin clock, AC, power log)
#        Then:  python3 merge.py   in the same directory.
#
# WRONG IF: any log has ok=0 rows or a kernel-log fault line; or the three passes of one case disagree by more
#   than a few GB/s (merge.py prints the per-pass values); or the machine was loaded (check the load line the
#   wrapper writes into each log).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

: "${GPURUN:?set GPURUN to the pinned-clock wrapper (see README.md)}"
: "${CEIL_BIN:?set CEIL_BIN to the built ceil binary}"
G=$GPURUN
export REPS=3
for p in 1 2 3; do
 for mem in host dev; do
  for b in 17694720 21626880 25804800 35389440 67108864; do
   $G $PWD/p${p}_${b}_${mem}.log "$CEIL_BIN" $b $mem
  done
  for wg in 256 512 1024; do $G $PWD/p${p}_268435456_${mem}_wg$wg.log "$CEIL_BIN" 268435456 $mem 0 $wg; done
 done
 for b in 17694720 25804800; do $G $PWD/p${p}_${b}_host_copies1.log "$CEIL_BIN" $b host 1; done
done
echo ALLDONE > $PWD/alldone.flag

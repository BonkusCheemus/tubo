#!/bin/bash
# runall.sh -- launch-overlap sweep: 5 passes, 45 s apart, over 3 buffer sizes x host/dev memory.
#
# MEASURES: GB/s of nine launch styles (plain, tiled, any-order, dependent) with all arms interleaved inside
#   one call; each call = REPS (15) reps x 3 rounds of 24 launches. Logs f<pass>_<bytes>_<mem>.log.
#
# RUN:   GPURUN=<wrapper> MAGIC_BIN=<path to built magic> ./runall.sh      (logs go to the current directory)
#        GPURUN is called as:  $GPURUN <log file> <command...>   (see README.md)
#        Then:  python3 merge.py f   in the same directory.
#
# WRONG IF: any log has a FAULT line or "wrong_order"/"spin_timeout" > 0; the plain arm g_256_80_1 moves by more
#   than the claimed gain between passes; or loadavg per call (printed by merge.py) is above ~1.5.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

: "${GPURUN:?set GPURUN to the pinned-clock wrapper (see README.md)}"
: "${MAGIC_BIN:?set MAGIC_BIN to the built magic binary}"
G=$GPURUN
A=g_256_80_1,t_1024_640_2,g_256_80_1_ao,t_256_1280_1_ao,t_1024_320_2_ao,t_512_640_1_ao,p_256_80_1_ao,q_256_80_2_ao,q_256_80_8_ao
export REPS=15
for p in 1 2 3 4 5; do
  for mem in host dev; do for b in 21626880 17694720 25804800; do
    $G $PWD/f${p}_${b}_${mem}.log "$MAGIC_BIN" ab $b $mem $A
  done; done
  sleep 45
done
echo DONE > runall.flag

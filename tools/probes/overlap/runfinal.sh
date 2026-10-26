#!/bin/bash
# runfinal.sh -- dependent-launch windows against the plain launch and the best independent launch.
#
# MEASURES: 3 passes; per pass, GB/s of g_256_80_1 (plain), t_256_1280_1_ao (independent, any order),
#   p_256_80_1_ao and w_256_80_{2,4,8}_ao (dependent windows of 1, 2, 4, 8 launches) at 3 sizes, host and dev
#   memory; then GB/s and watts per arm (pwr mode). Logs z<pass>_<bytes>_<mem>.log and pwr<pass>.log.
#
# RUN:   GPURUN=<wrapper> MAGIC_BIN=<path to built magic> ./runfinal.sh    (logs go to the current directory)
#        GPURUN is called as:  $GPURUN <log file> <command...>   (see README.md)
#        Also needs GPU_SYSFS_DIR and GPU_POWER_FILE for the pwr step.
#        Then:  python3 merge.py z   in the same directory.
#
# WRONG IF: wrong_order or spin_timeout counts are non-zero in any log (the dependent arm did not keep order, so
#   its GB/s is not valid); or the dependent arm beats the independent arm by less than the pass-to-pass spread.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

: "${GPURUN:?set GPURUN to the pinned-clock wrapper (see README.md)}"
: "${MAGIC_BIN:?set MAGIC_BIN to the built magic binary}"
G=$GPURUN
A=g_256_80_1,t_256_1280_1_ao,p_256_80_1_ao,w_256_80_2_ao,w_256_80_4_ao,w_256_80_8_ao
export REPS=15
for p in 1 2 3; do
  for mem in host dev; do for b in 21626880 17694720 25804800; do
    $G $PWD/z${p}_${b}_${mem}.log "$MAGIC_BIN" ab $b $mem $A
  done; done
  $G $PWD/pwr${p}.log "$MAGIC_BIN" pwr 21626880 host g_256_80_1,t_256_1280_1_ao,p_256_80_1_ao,w_256_80_4_ao
  sleep 30
done
echo DONE > runfinal.flag

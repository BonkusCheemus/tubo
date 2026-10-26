#!/bin/bash
# run.sh -- first GPU run of dq5 (decoupled Q5_K kernel; see dq5.hip: this design FAILED, 171 vs 206 GB/s).
#
# MEASURES: R=1 (one Q5 matrix 5120x6144 per round) then R=4 (4 matrices per persistent launch). The binary
#   validates every arm on every copy before timing it and exits 3 on any mismatch or spin timeout. This script
#   stops at the first non-zero exit code, or at a ring timeout / GPU reset in the kernel log.
#
# RUN:   DQ5_BIN=<path to built dq5> $GPURUN <log file> ./run.sh
#        The wrapper (see README.md) already holds the benchmark lock, pins the clock and logs power, so this
#        script must NOT take the lock itself (the lock is not re-entrant).
#
# WRONG IF: "# R=... rc=" is not 0; the kernel-log check stops the run; or arm a (the coupled baseline) is slower
#   than it is in the GEMV benchmark (machine loaded or clock not pinned).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

: "${DQ5_BIN:?set DQ5_BIN to the built dq5 binary}"
T0=$(date '+%Y-%m-%d %H:%M:%S')
ringok() { ! journalctl -k --since "$T0" --no-pager | grep -qiE "ring .* timeout|GPU reset" || { echo "STOP ring timeout before $1"; exit 3; }; }
for R in 1 4; do
  ringok "R=$R"
  "$DQ5_BIN" 5120 6144 "$R" host; rc=$?
  echo "# R=$R rc=$rc"
  [ "$rc" = 0 ] || exit "$rc"
done
ringok end

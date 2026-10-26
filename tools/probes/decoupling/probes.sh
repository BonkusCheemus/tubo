#!/bin/bash
# probes.sh -- the probe series that decides whether decoupling load from compute is worth building.
#
# MEASURES: P1: single-launch pure-read ceiling (ceil) at the IQ3 matrix sizes and the fused Q4 size.
#   P2: coupled versus decoupled kernels (dcp) at the Q5 size (coupled/decoupled arm output at N64 = 0, 16, 22 and
#   R = 1, 4), the IQ3 sizes (CH=6, 7, with and without arithmetic) and the Q6 size. All through the wrapper.
#
# RUN:   GPURUN=<wrapper> CEIL_BIN=<built ceil> DCP_BIN=<built dcp> ./probes.sh   (logs go to the current directory)
#        GPURUN is called as:  $GPURUN <log file> <command...>   (see README.md). Build the binaries first
#        (see the headers of ../ceiling/ceil.hip and dcp.hip). Read the results with the "READ IT AS" list in dcp.hip.
#
# WRONG IF: a log contains FAIL or a fault line; the P1 pure-read rate at a size is below the same size in the
#   ceiling sweep (the machine was loaded); or N64=0 coupled and decoupled runs disagree with the pure-read ceiling.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case "${1:-}" in -h|--help) usage; exit 0;; esac

: "${GPURUN:?set GPURUN to the pinned-clock wrapper (see README.md)}"
: "${CEIL_BIN:?set CEIL_BIN to the built ceil binary}"
: "${DCP_BIN:?set DCP_BIN to the built dcp binary}"
G=$GPURUN; C=$CEIL_BIN; D=$DCP_BIN

# P1 (~2 min): single-launch pure-read ceiling AT the IQ3 sizes + the fused Q4 size.
for b in 68239360 76595200 47185920; do REPS=3 $G $PWD/p1_${b}.log $C $b host; done

# P2: coupling vs decoupling. Q5 size first.
for n in 0 16 22; do for r in 1 4; do CH=11 $G $PWD/p2_q5_n${n}_r${r}.log $D 21626880 $r $n host; done; done
for ch in 6 7; do CH=$ch $G $PWD/p2_iq3_ch${ch}_n0.log $D 68239360 1 0 host; CH=$ch $G $PWD/p2_iq3_ch${ch}_n20.log $D 68239360 1 20 host; done
CH=13 $G $PWD/p2_q6_n24.log $D 25804800 1 24 host

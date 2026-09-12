#!/bin/bash
# s25 targeted scope: Timer-tree A/B of the SNH G/H scope via UIPC_SNK1_STENCIL2
set -uo pipefail
PY=/root/work/venv/bin/python
OUT=/root/work/timers_s25c; mkdir -p $OUT
FRAMES=${FRAMES:-60}
REPS=${REPS:-"1 2"}
declare -A DIR=( [stiff-gipc-case2]=88_stiff_gipc_benchmark [mas-bunny]=89_mas_bunny )
for rep in $REPS; do
for scene in stiff-gipc-case2 mas-bunny; do
  cd /root/work/src/libuipc-samples/examples/${DIR[$scene]}
  for mode in new old; do
    if [ "$mode" = old ]; then export UIPC_SNK1_STENCIL2=0; else unset UIPC_SNK1_STENCIL2 || true; fi
    echo "### timer $scene $mode rep$rep (frames=$FRAMES) $(date -u +%H:%M:%S)"
    WB_LOG=Warn WB_TIMER=1 UIPC_BENCHMARK_TIMERS=1 timeout 1200 $PY main.py --headless $FRAMES \
      > $OUT/$scene.$mode.$rep.txt 2>&1
    grep -E "G/H uipc::backend::cuda::StableNeoHookean3D|E uipc::backend::cuda::StableNeoHookean3D|\*Newton Iteration|\*FEM Reporters G/H|\*Pipeline " \
      $OUT/$scene.$mode.$rep.txt | sed 's/  */ /g'
  done
done
done

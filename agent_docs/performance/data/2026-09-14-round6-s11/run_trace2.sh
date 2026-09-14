#!/bin/bash
set -u
cd /workspace/output/round6/s11
for r in 1 2 3; do
  for a in 0 1 3 2; do
    bash tracerun.sh rwbn_a${a}_r${r} 6_wrecking_balls 120 UIPC_ABD_GH_PREPASS=$a || exit 3
  done
done
echo TRACE2_DONE

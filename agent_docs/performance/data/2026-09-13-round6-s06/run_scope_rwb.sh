#!/bin/bash
set -u
cd /workspace/output/round6/s06
for r in 1 2 3; do
  for arm in c0 c1; do
    v=${arm#c}
    ./nsysrun.sh rwb_${arm}_r${r} 6_wrecking_balls 120 UIPC_CCD_COMPACT=$v || echo "FAIL rwb_${arm}_r${r}"
  done
done
echo SCOPE_RWB_DONE

#!/bin/bash
set -u
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
for sc in cube-wall-cloth tumbler-garments mas-bunny; do
  $UIPC_PERF_PY /workspace/output/round6/ab.py --scene $sc --n 3 --label s05 \
    --out /workspace/output/round6/s05/ab \
    --arm old UIPC_CCD_CULL=0 --arm new UIPC_CCD_CULL=1 \
    > /workspace/output/round6/s05/ab/ab_${sc}.txt 2>&1 || echo "FAIL ab $sc"
  echo "AB_DONE $sc"
done
bash /workspace/output/round6/s05/run_verify.sh

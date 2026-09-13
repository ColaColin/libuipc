#!/bin/bash
set -u
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
for scene in "$@"; do
  $UIPC_PERF_PY /workspace/output/round6/ab.py --scene "$scene" --n 5 \
    --label s04 --out /workspace/output/round6/s04/ab_$scene \
    --arm old UIPC_CCD_EARLY_OUT=0 --arm new UIPC_CCD_EARLY_OUT=1 \
    2>&1 | tee /workspace/output/round6/s04/ab_$scene.txt
done
echo AB_DONE

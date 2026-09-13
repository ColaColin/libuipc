#!/bin/bash
set -u
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
AB=/workspace/output/round6/ab.py
run() {  # scene n
  $UIPC_PERF_PY $AB --scene $1 --n $2 --label s06 --out /workspace/output/round6/s06/ab \
    --arm old UIPC_CCD_COMPACT=0 --arm new UIPC_CCD_COMPACT=1 \
    > /workspace/output/round6/s06/ab/ab_$1.txt 2>&1 || echo "AB FAIL $1"
  tail -14 /workspace/output/round6/s06/ab/ab_$1.txt
}
run mas-bunny 5
run stiff-gipc-case2 5
run cube-wall-cloth 5
run tumbler-garments 5
run rigid-wrecking-balls 3
echo AB_DONE

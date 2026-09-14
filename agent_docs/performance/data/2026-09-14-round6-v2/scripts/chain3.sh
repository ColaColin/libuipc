#!/bin/bash
set -u
cd /workspace/output/round6/v2
source /workspace/deps/libuipc-src/env_perf.sh
until grep -q "CHAIN2 DONE" chain2.log; do sleep 15; done
echo "=== cloth_min_y follow-up: p0 vs p1 vs pnull, n=40 $(date +%H:%M:%S)"
$UIPC_PERF_PY abn.py --scene cube-wall-cloth --n 40 --label cminy --out /workspace/output/round6/v2/cost \
  --baseline p0 \
  --arm p0 UIPC_CONTACT_RANK1=0 \
  --arm pnull UIPC_CONTACT_RANK1=00 \
  --arm p1 UIPC_CONTACT_RANK1=1 > cost_cminy.txt 2>&1
echo "CHAIN3 DONE rc=$? $(date +%H:%M:%S)"

#!/bin/bash
set -u
cd /workspace/output/round6/v2
source /workspace/deps/libuipc-src/env_perf.sh
$UIPC_PERF_PY abn.py --scene cube-wall-cloth --n 30 --label trunc100 --out /workspace/output/round6/v2/cost \
  --baseline p0c \
  --arm p0c UIPC_CONTACT_RANK1=0 UIPC_CWC_TIGHT=100 \
  --arm p1c UIPC_CONTACT_RANK1=1 UIPC_CWC_TIGHT=100 > cost_trunc100.txt 2>&1
echo "CHAIN5 DONE rc=$? $(date +%H:%M:%S)"

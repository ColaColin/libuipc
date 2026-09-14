#!/bin/bash
set -u
cd /workspace/output/round6/v2
source /workspace/deps/libuipc-src/env_perf.sh
$UIPC_PERF_PY abn.py --scene cube-wall-cloth --n 25 --label trunc --out /workspace/output/round6/v2/cost \
  --baseline p0 \
  --arm p0  UIPC_CONTACT_RANK1=0 UIPC_CWC_TIGHT=1 \
  --arm p1  UIPC_CONTACT_RANK1=1 UIPC_CWC_TIGHT=1 \
  --arm p0t UIPC_CONTACT_RANK1=0 UIPC_CWC_TIGHT=10 \
  --arm p1t UIPC_CONTACT_RANK1=1 UIPC_CWC_TIGHT=10 > cost_trunc.txt 2>&1
echo "CHAIN4 DONE rc=$? $(date +%H:%M:%S)"

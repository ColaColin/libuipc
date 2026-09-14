#!/bin/bash
set -u
cd /workspace/output/round6/v2
source /workspace/deps/libuipc-src/env_perf.sh
echo "=== extra micro configs $(date +%H:%M:%S)"
NREP=5 CONFIGS="fastslide tight" bash run_micro.sh >> micro_sweep.log 2>&1
echo "=== micro2 done $(date +%H:%M:%S)"
echo "=== cost: cube-wall-cloth Newton drift at this head, n=20 $(date +%H:%M:%S)"
$UIPC_PERF_PY abn.py --scene cube-wall-cloth --n 20 --label cost_cwc --out /workspace/output/round6/v2/cost \
  --baseline p0 \
  --arm p0 UIPC_CONTACT_RANK1=0 \
  --arm pnull UIPC_CONTACT_RANK1=00 \
  --arm p5 UIPC_CONTACT_RANK1=5 \
  --arm p1 UIPC_CONTACT_RANK1=1 > cost_cwc.txt 2>&1
echo "=== cost done rc=$? $(date +%H:%M:%S)"
for a in 0 1; do
  echo "=== gate UIPC_CONTACT_RANK1=$a $(date +%H:%M:%S)"
  env UIPC_CONTACT_RANK1=$a bash /workspace/output/round6/gate.sh > gate_rank1_$a.txt 2>&1
done
echo "=== gate default $(date +%H:%M:%S)"
bash /workspace/output/round6/gate.sh > gate_default.txt 2>&1
echo "CHAIN2 DONE $(date +%H:%M:%S)"

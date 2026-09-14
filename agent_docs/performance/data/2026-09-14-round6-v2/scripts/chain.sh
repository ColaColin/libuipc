#!/bin/bash
set -u
cd /workspace/output/round6/v2
until grep -q "SWEEP DONE" sweep.log; do sleep 20; done
echo "=== sweep done, analysing at n=20 $(date +%H:%M:%S)"
source /workspace/deps/libuipc-src/env_perf.sh
$UIPC_PERF_PY analyze.py > analysis_n20.txt 2>&1
echo "=== analysis_n20 written $(date +%H:%M:%S)"
bash run_micro.sh  > micro_sweep.log 2>&1
echo "=== micro done $(date +%H:%M:%S)"
bash run_crease.sh > crease_sweep.log 2>&1
echo "=== crease done $(date +%H:%M:%S)"
bash envaudit.sh   > envaudit.txt 2>&1
echo "=== envaudit done $(date +%H:%M:%S)"
echo "CHAIN DONE $(date +%H:%M:%S)"

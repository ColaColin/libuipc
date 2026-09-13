#!/bin/bash
# Truncation-bias test: the same A/B at a 10x tighter Newton stopping tolerance.
# A truncation bias must shrink; chaotic divergence must not.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
SC=/workspace/deps/libuipc-src/libuipc-samples/examples/95_tumbler_garments
OUT=/workspace/output/round6/v1/toltest
mkdir -p "$OUT"; cd "$SC"
for r in 1 2 3 4 5 6 7 8 9 10; do
  for arm in A B; do
    f="$OUT/${arm}_${r}.log"
    [ -s "$f" ] && grep -q VERIFY_RESULT "$f" && { echo "skip $arm$r"; continue; }
    echo "=== tol $arm$r $(date +%H:%M:%S)"
    if [ "$arm" = A ]; then
      env WB_LOG=Info UIPC_DSB_GAUSS_NEWTON=0 timeout 1800 "$UIPC_PERF_PY" main.py \
          --headless 180 --verify --vel-tol=0.005 > "$f" 2>&1
    else
      env WB_LOG=Info timeout 1800 "$UIPC_PERF_PY" main.py \
          --headless 180 --verify --vel-tol=0.005 > "$f" 2>&1
    fi
    echo "   rc=$?"
  done
done

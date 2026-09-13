#!/bin/bash
set -u
source /workspace/deps/libuipc-src/env_perf.sh
S=/workspace/deps/libuipc-src/agent_docs/performance/data/2026-09-13-round6-v1/crease_micro.py
OUT=/workspace/output/round6/v1/crease
mkdir -p "$OUT"
ARGS="--frames=200 --phi-max=145 --vel-tol=0.002 --tol-rate=1e-4"
for r in 1 2 3 4 5 6 7 8; do
  for arm in A B; do
    f="$OUT/${arm}_${r}.txt"
    [ -s "$f" ] && grep -q "frame=200" "$f" && { echo "skip $arm$r"; continue; }
    if [ "$arm" = A ]; then
      env UIPC_DSB_GAUSS_NEWTON=0 timeout 900 "$UIPC_PERF_PY" "$S" $ARGS --tag="${arm}_${r}" > "$f" 2>&1
    else
      timeout 900 "$UIPC_PERF_PY" "$S" $ARGS --tag="${arm}_${r}" > "$f" 2>&1
    fi
    echo "$arm$r rc=$? $(grep -c CREASE\  "$f") frames"
  done
done
# a second, sharper configuration: fewer hinge rows -> more angle per hinge
ARGS2="--frames=200 --phi-max=145 --cells=3 --wide=24 --edge=0.02 --vel-tol=0.002 --tol-rate=1e-4"
for r in 1 2 3 4; do
  for arm in A B; do
    f="$OUT/sharp_${arm}_${r}.txt"
    [ -s "$f" ] && grep -q "frame=200" "$f" && { echo "skip sharp $arm$r"; continue; }
    if [ "$arm" = A ]; then
      env UIPC_DSB_GAUSS_NEWTON=0 timeout 900 "$UIPC_PERF_PY" "$S" $ARGS2 --tag="s${arm}_${r}" > "$f" 2>&1
    else
      timeout 900 "$UIPC_PERF_PY" "$S" $ARGS2 --tag="s${arm}_${r}" > "$f" 2>&1
    fi
    echo "sharp $arm$r rc=$? $(grep -c CREASE\  "$f") frames"
  done
done

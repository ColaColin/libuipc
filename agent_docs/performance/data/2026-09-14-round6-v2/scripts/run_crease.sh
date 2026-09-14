#!/bin/bash
# Round-6 V2: V1's crease-severity micro-test, run against *this* change.
# Contact is disabled in that scene, so the contact Hessian is never assembled:
# arm 0 and arm 1 must be indistinguishable.  It is a NULL test here -- if the
# two arms differ, the env switch or the build is wrong, not the physics.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
S=/workspace/output/round6/v2/crease_micro.py
OUT=/workspace/output/round6/v2/crease
mkdir -p "$OUT"
ARGS="--frames=200 --phi-max=145 --vel-tol=0.002 --tol-rate=1e-4"
for r in 1 2 3 4 5 6; do
  for arm in 0 1; do
    f="$OUT/${arm}_${r}.txt"
    [ -s "$f" ] && grep -q "frame=200 " "$f" && { echo "skip $arm$r"; continue; }
    env UIPC_CONTACT_RANK1="$arm" UIPC_CREASE_WS="$OUT/ws_${arm}_${r}/" \
        timeout 900 "$UIPC_PERF_PY" "$S" $ARGS --tag="${arm}_${r}" > "$f" 2>&1
    echo "$arm$r rc=$? $(grep -c 'CREASE ' "$f") frames"
  done
done
echo "CREASE DONE $(date +%H:%M:%S)"

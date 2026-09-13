#!/bin/bash
# A'' : exact path with a perturbation sized to MATCH the frame-1 seed that the
# Gauss-Newton search direction itself injects (1.83e-7 m rms).  This closes the
# only gap in the divergence envelope test: A' 's 1e-9 m / 1e-6 rad seeds are
# smaller than B's, so the first five frames of the A-vs-B curve sat above the
# physically-equivalent envelope purely because of seed size.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
SC=/workspace/deps/libuipc-src/libuipc-samples/examples/95_tumbler_garments
OUT=/workspace/output/round6/v1/runs
cd "$SC"
declare -a PERT=(
  "--perturb-vertex=1e-6"
  "--perturb-vertex=-1e-6"
  "--perturb-vertex=1e-6 --perturb-garment=1"
  "--perturb-yaw=1e-5"
  "--perturb-yaw=-1e-5 --perturb-garment=2"
)
for r in 1 2 3 4 5; do
  f="$OUT/App_${r}.log"
  [ -s "$f" ] && grep -q VERIFY_RESULT "$f" && { echo "skip App_$r"; continue; }
  echo "=== App_$r ${PERT[$((r-1))]} $(date +%H:%M:%S)"
  env WB_LOG=Info UIPC_DSB_GAUSS_NEWTON=0 timeout 1200 "$UIPC_PERF_PY" main.py \
      --headless 180 --verify --dump-positions="$OUT/App_${r}.npy" ${PERT[$((r-1))]} > "$f" 2>&1
  echo "   rc=$?"
done

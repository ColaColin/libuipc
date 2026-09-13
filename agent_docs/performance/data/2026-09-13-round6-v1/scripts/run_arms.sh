#!/bin/bash
# Round-6 V1: the three-arm experiment.  One build, one GPU, interleaved A / A' / B.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
SC=/workspace/deps/libuipc-src/libuipc-samples/examples/95_tumbler_garments
OUT=/workspace/output/round6/v1/runs
cd "$SC"

# arm-name  env-value            extra flags
declare -a REP=(1 2 3 4 5 6 7 8 9 10)
declare -a PERT=(
  "--perturb-yaw=1e-6"
  "--perturb-yaw=-1e-6"
  "--perturb-yaw=1e-6 --perturb-garment=1"
  "--perturb-vertex=1e-9"
  "--perturb-vertex=1e-9 --perturb-garment=3"
  "--perturb-yaw=2e-6"
  "--perturb-yaw=-2e-6 --perturb-garment=2"
  "--perturb-vertex=-1e-9"
  "--perturb-vertex=1e-9 --perturb-garment=1"
  "--perturb-yaw=1e-6 --perturb-garment=3"
)

one () {  # $1 arm  $2 rep  $3 gnenv  $4... flags
  local arm=$1 rep=$2 gn=$3; shift 3
  local tag="${arm}_${rep}"
  if [ -s "$OUT/$tag.log" ] && grep -q "^VERIFY_RESULT" "$OUT/$tag.log"; then
    echo "skip $tag"; return
  fi
  echo "=== $tag  GN=$gn  flags: $*  $(date +%H:%M:%S)"
  if [ -n "$gn" ]; then
    env WB_LOG=Info UIPC_DSB_GAUSS_NEWTON="$gn" timeout 1200 \
        "$UIPC_PERF_PY" main.py --headless 180 --verify \
        --dump-positions="$OUT/$tag.npy" "$@" > "$OUT/$tag.log" 2>&1
  else
    env WB_LOG=Info timeout 1200 \
        "$UIPC_PERF_PY" main.py --headless 180 --verify \
        --dump-positions="$OUT/$tag.npy" "$@" > "$OUT/$tag.log" 2>&1
  fi
  local rc=$?
  echo "    rc=$rc  $(grep -c 'SimplexTrajectoryFilter PTs' "$OUT/$tag.log") pair lines"
}

for r in "${REP[@]}"; do
  one A  "$r" 0
  one Ap "$r" 0 ${PERT[$((r-1))]}
  one B  "$r" ""
done

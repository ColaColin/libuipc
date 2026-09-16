#!/bin/bash
# Round-7 s05 divergence arms: A exact (n=10) interleaved with B GN (n=10),
# then A'' matched-perturbation exact controls (n=5). Full 130-frame --verify
# runs with position dumps. Resumable: skips runs whose .json already exists.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src/libuipc-samples/examples/103_crease_press
OUT=/workspace/output/round7/s05/runs
mkdir -p "$OUT"

run() {  # name mode [extra scene flags...]
  local name=$1 mode=$2; shift 2
  local log="$OUT/$name.log"
  if [ -f "$OUT/$name.json" ]; then echo "=== $name skipped (exists)"; return; fi
  echo "=== $name ($mode) start $(date +%T)"
  if [ "$mode" = "exact" ]; then
    env -u CP_DT -u CP_PDEPTH -u CP_DIE -u CP_YSTRAIN -u CP_YSTRESS \
      UIPC_DAHL_GAUSS_NEWTON=0 \
      timeout 900 $UIPC_PERF_PY main.py 130 --verify \
      --result="$OUT/$name.json" --dump-positions="$OUT/$name.npy" "$@" > "$log" 2>&1
  else
    env -u CP_DT -u CP_PDEPTH -u CP_DIE -u CP_YSTRAIN -u CP_YSTRESS -u UIPC_DAHL_GAUSS_NEWTON \
      timeout 900 $UIPC_PERF_PY main.py 130 --verify \
      --result="$OUT/$name.json" --dump-positions="$OUT/$name.npy" "$@" > "$log" 2>&1
  fi
  local rc=$?
  echo "=== $name rc=$rc end $(date +%T) $(grep -o 'TOTAL frames=130.*' "$log" | tail -1)"
}

for k in 01 02 03 04 05 06 07 08 09 10; do
  run "a$k" exact
  run "b$k" gn
done
run ap1 exact --perturb-vertex=1e-6  --perturb-sheet=0
run ap2 exact --perturb-vertex=-1e-6 --perturb-sheet=0
run ap3 exact --perturb-vertex=1e-5  --perturb-sheet=0
run ap4 exact --perturb-yaw=1e-5     --perturb-sheet=0
run ap5 exact --perturb-yaw=1e-5     --perturb-sheet=3
echo SWEEP_DONE

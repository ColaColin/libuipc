#!/bin/bash
# s25 end-to-end A/B via UIPC_SNK1_STENCIL2, interleaved.
set -uo pipefail
cd /root/work/src
PY=/root/work/venv/bin/python
OUT=${OUT:-/root/work/ab_s25}
mkdir -p $OUT
SCENES="${SCENES:-stiff-gipc-case2 mas-bunny}"
REPS="${REPS:-1 2 3}"
run () {
  local scene=$1 mode=$2 rep=$3
  if [ "$mode" = old ]; then export UIPC_SNK1_STENCIL2=0; else unset UIPC_SNK1_STENCIL2 || true; fi
  echo "### $scene $mode rep$rep  SS=${UIPC_SNK1_STENCIL2:-unset}  $(date -u +%H:%M:%S)"
  $PY scripts/run_benchmark.py run "$scene" --python $PY > $OUT/${scene}.${mode}.${rep}.log 2>&1 \
    || { echo "FAILED $scene $mode $rep"; tail -20 $OUT/${scene}.${mode}.${rep}.log; }
  cp output/benchmark-runs/${scene}.json $OUT/${scene}.${mode}.${rep}.json 2>/dev/null
  grep -o 'TOTAL frames=[0-9]* mean=[0-9.]*ms median=[0-9.]*ms p95=[0-9.]*ms' $OUT/${scene}.${mode}.${rep}.log | tail -1
}
for rep in $REPS; do
  for scene in $SCENES; do
    for mode in new old; do run $scene $mode $rep; done
  done
done
echo "=== SUMMARY ==="
$PY /root/work/summarize.py "$OUT/*.json"

#!/bin/bash
# One UIPC_BENCHMARK_TIMERS=1 stage-timing diagnostic per scene per tree, frame counts as in the 2026-09-01 doc (100/60/60/70).
set -uo pipefail
OUT=/work/results/timers; mkdir -p $OUT
for side in head base dahl; do
  PY=/work/venv-src-$side/bin/python; R=/work/libuipc-$side
  for pair in rigid-wrecking-balls:100 stiff-gipc-case2:60 mas-bunny:60 cube-wall-cloth:70; do
    s=${pair%%:*}; f=${pair##*:}
    echo "== timers $side $s $f $(date -u +%T)"
    (cd $R && $PY scripts/run_benchmark.py run $s --frames $f --python $PY --env UIPC_BENCHMARK_TIMERS=1) > $OUT/${side}_${s}_timers.stdout 2>&1
    echo "rc=$? $(grep -E '^TOTAL' $OUT/${side}_${s}_timers.stdout | head -1)"
  done
done
echo TIMERS_DONE

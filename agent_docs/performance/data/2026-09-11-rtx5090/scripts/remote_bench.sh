#!/bin/bash
# Run the canonical suite alternating A/B. Usage: remote_bench.sh <tag> <runs> <pyA> <pyB> [extra run_benchmark args]
# tag names the results subdir; A = head tree, B = base tree.
set -uo pipefail
TAG=$1; RUNS=$2; PYA=$3; PYB=$4; shift 4; EXTRA="$@"
OUT=/work/results/$TAG; mkdir -p $OUT
SCENES="rigid-wrecking-balls stiff-gipc-case2 mas-bunny cube-wall-cloth"
export UIPC_BENCHMARK_TIMERS=0
for r in $(seq 1 $RUNS); do
  if [ $((r % 2)) -eq 1 ]; then ORDER="head base"; else ORDER="base head"; fi
  for side in $ORDER; do
    PY=$( [ $side = head ] && echo $PYA || echo $PYB ); R=/work/libuipc-$side
    for s in $SCENES; do
      echo "== run $r $side $s $(date -u +%T)"
      (cd $R && $PY scripts/run_benchmark.py run $s --python $PY $EXTRA) > $OUT/${side}_${s}_r$r.stdout 2>&1
      rc=$?; grep -E "^TOTAL|^metadata:|^error" $OUT/${side}_${s}_r$r.stdout | head -3; echo "rc=$rc"
    done
  done
done
echo BENCH_DONE $TAG

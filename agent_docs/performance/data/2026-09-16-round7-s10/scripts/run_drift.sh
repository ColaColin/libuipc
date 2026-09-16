#!/bin/bash
# s10 drift: n default crease-press runs at head, json copied per run
set -u
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
N=${1:-8}
for i in $(seq 1 $N); do
  $UIPC_PERF_PY scripts/run_benchmark.py run crease-press --python $UIPC_PERF_PY > /workspace/output/round7/s10/drift/run_$i.log 2>&1
  rc=$?
  cp output/benchmark-runs/crease-press.json /workspace/output/round7/s10/drift/drift_head_$i.json || rc=$?
  m=$(python3 -c "import json;d=json.load(open('/workspace/output/round7/s10/drift/drift_head_$i.json'));print('%.2f'%d['reportedFrameTiming']['meanFrameMs'])" 2>/dev/null)
  nt=$(python3 -c "
import json;d=json.load(open('/workspace/output/round7/s10/drift/drift_head_$i.json'));fs=d['reportedBenchmark']['frame_stats']
print(sum(f['newton_iterations'] for f in fs), sum(f['linear_solver_iterations'] for f in fs), sum(f['line_search_trials'] for f in fs))" 2>/dev/null)
  echo "run $i rc=$rc meanFrameMs=$m newton/pcg/ls=$nt"
  [ $rc -ne 0 ] && exit $rc
done
echo DRIFT-DONE

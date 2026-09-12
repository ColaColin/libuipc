#!/bin/bash
# Composed correctness gate on the head build (build-perf).
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
for t in common core geometry sanity_check regression backend_cuda sim_case; do
  echo "=== $t"
  ./build-perf/Release/bin/uipc_test_$t 2>&1 | grep -E "All tests passed|FAILED|failed|assertions in" | tail -3
done
echo "=== pytest"
$UIPC_PERF_PY -m pytest -q -m "cuda and not example" python/tests 2>&1 | tail -2

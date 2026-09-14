#!/bin/bash
# s09: cuda_gpu_trace for the per-launch timeline (overlap measurement).
# Full run on rigid-wrecking-balls (120 frames) -- the csv is ~40 MB there, not the
# 400 MB a tumbler run costs, so no window selection is needed and s03's
# "a short window contains zero contact-assembly launches" trap does not apply.
set -u
PFX=/workspace/output/round6/s09/trace/$1; DIR=$2; FRAMES=$3; shift 3
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv"
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
for kv in "$@"; do export "$kv"; done
cd "$TREE/libuipc-samples/examples/$DIR" || exit 2
$NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
   $PY main.py --headless $FRAMES > "$PFX.run.log" 2>&1
rc=$?
$NSYS stats --report cuda_gpu_trace --format csv --force-export true \
   --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
if [ ! -s "${PFX}_cuda_gpu_trace.csv" ]; then
  echo "NSYS-FAIL no gpu_trace csv for $PFX (rc=$rc)"; tail -5 "$PFX.stats.log"; exit 3
fi
echo "tracerun ok $PFX rc=$rc size=$(du -h ${PFX}_cuda_gpu_trace.csv | cut -f1)"

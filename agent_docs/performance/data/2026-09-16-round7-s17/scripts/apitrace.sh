#!/bin/bash
# s17: cuda API + GPU trace of a crease-press window, for the PCG convergence
# readback mechanism: the per-block D2H round trip and the GPU-idle gap
# between consecutive PCG graph replays. Same capture shape as s09/s10.
set -u
PFX=$1; DIR=$2; FRAMES=$3; shift 3
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv" "${PFX}_cuda_api_trace.csv" \
      "${PFX}_cuda_gpu_kern_sum.csv" "${PFX}_cuda_api_sum.csv"
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
for kv in "$@"; do export "$kv"; done
cd "$DIR" || exit 2
$NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
   $PY main.py --headless $FRAMES > "$PFX.run.log" 2>&1
rc=$?
$NSYS stats --report cuda_gpu_trace --report cuda_api_trace --report cuda_gpu_kern_sum --report cuda_api_sum \
   --format csv --force-export true --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
ok=1
[ -s "${PFX}_cuda_gpu_trace.csv" ] || ok=0
[ -s "${PFX}_cuda_api_trace.csv" ] || ok=0
if [ $ok -eq 0 ]; then echo "NSYS-FAIL missing csvs for $PFX (rc=$rc)"; tail -5 "$PFX.stats.log"; exit 3; fi
echo "apitrace ok $PFX rc=$rc gpu=$(du -h ${PFX}_cuda_gpu_trace.csv | cut -f1) api=$(du -h ${PFX}_cuda_api_trace.csv | cut -f1)"

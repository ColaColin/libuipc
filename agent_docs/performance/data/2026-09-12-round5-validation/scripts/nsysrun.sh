#!/bin/bash
# usage: nsysrun.sh <prefix> <tree> <py> <scenedir> <frames> [ENV=V ...]
# Fresh-prefix discipline: delete .nsys-rep/.sqlite/csv first, fail loudly if no csv appears.
set -u
PFX=$1; TREE=$2; PY=$3; DIR=$4; FRAMES=$5; shift 5
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv" "${PFX}_cuda_gpu_kern_sum.csv" "${PFX}_cuda_gpu_mem_time_sum.csv"
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
for kv in "$@"; do export "$kv"; done
cd "$TREE/libuipc-samples/examples/$DIR" || exit 2
$NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
   $PY main.py --headless $FRAMES > "$PFX.run.log" 2>&1
rc=$?
$NSYS stats --report cuda_gpu_trace --report cuda_gpu_kern_sum --format csv --force-export true \
   --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
if [ ! -s "${PFX}_cuda_gpu_trace.csv" ]; then
  echo "NSYS-FAIL no trace csv for $PFX (rc=$rc)"; tail -5 "$PFX.stats.log"; exit 3
fi
echo "nsysrun ok $PFX rc=$rc rows=$(wc -l < ${PFX}_cuda_gpu_trace.csv)"

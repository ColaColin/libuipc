#!/bin/bash
# usage: nsysrun.sh <prefix> <frames> [ENV=V ...] -- extra main.py args...
set -u
PFX=$1; FRAMES=$2; shift 2
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
EXTRA=""
for kv in "$@"; do
  if [[ "$kv" == *=* && "$kv" != -* ]]; then export "$kv"; else EXTRA="$EXTRA $kv"; fi
done
rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv" "${PFX}_cuda_gpu_kern_sum.csv"
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
cd $TREE/libuipc-samples/examples/103_crease_press || exit 2
$NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
   $PY main.py $FRAMES $EXTRA > "$PFX.run.log" 2>&1
rc=$?
$NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
   --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
[ -s "${PFX}_cuda_gpu_kern_sum.csv" ] || { echo "NSYS-FAIL no kern_sum csv $PFX (rc=$rc)"; tail -5 "$PFX.stats.log"; exit 3; }
echo "nsysrun ok $PFX rc=$rc"

#!/bin/bash
# s06 variant of round5/validation/nsysrun.sh: kern_sum only (the tumbler's
# cuda_gpu_trace csv is 400 MB and this step only needs per-kernel totals).
# Fresh prefix per run; fails loudly if the csv does not appear.
set -u
PFX=/workspace/output/round6/s08/nsys/$1; DIR=$2; FRAMES=$3; shift 3
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_kern_sum.csv"
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
$NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
   --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
if [ ! -s "${PFX}_cuda_gpu_kern_sum.csv" ]; then
  echo "NSYS-FAIL no kern_sum csv for $PFX (rc=$rc)"; tail -5 "$PFX.stats.log"; exit 3
fi
echo "nsysrun ok $PFX rc=$rc rows=$(wc -l < ${PFX}_cuda_gpu_kern_sum.csv)"

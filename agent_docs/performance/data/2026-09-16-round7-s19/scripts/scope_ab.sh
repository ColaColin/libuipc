#!/bin/bash
# s19: full-run crease-press nsys A/B of UIPC_PDSB_OCC (pair vs single), ABBA,
# fresh prefixes, graph-node tracing. usage: scope_ab.sh <outdir> [rounds]
set -u
OUT=${1:-/workspace/output/round7/s19}; ROUNDS=${2:-2}
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
cd $TREE/libuipc-samples/examples/103_crease_press || exit 2
seqs=("new old" "old new")
for r in $(seq 1 $ROUNDS); do
  for arm in ${seqs[$(( (r-1) % 2 ))]}; do
    PFX=$OUT/scope_r${r}_${arm}
    rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_kern_sum.csv" "${PFX}_cuda_gpu_trace.csv"
    if [ "$arm" = new ]; then unset UIPC_PDSB_OCC; else export UIPC_PDSB_OCC=0; fi
    $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
       $PY main.py > "$PFX.run.log" 2>&1
    rc=$?
    $NSYS stats --report cuda_gpu_kern_sum --report cuda_gpu_trace --format csv --force-export true \
       --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
    [ -s "${PFX}_cuda_gpu_kern_sum.csv" ] || { echo "NSYS-FAIL $PFX rc=$rc"; exit 3; }
    echo "ok r$r $arm rc=$rc"
  done
done

#!/bin/bash
# s06: repeats for the def(256) vs bd128 (spread pick) per-launch comparison.
# Interleaved D,1,D,1,D,1 full runs.
set -u
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
S=/workspace/output/round7/s06
SCENE=$TREE/libuipc-samples/examples/103_crease_press
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH

one() {  # prefix envval
  local PFX=$1 ENVV=$2
  rm -f "$PFX.nsys-rep" "${PFX}_cuda_gpu_kern_sum.csv" "$PFX.stats.log" "$PFX.run.log"
  (cd "$SCENE" || exit 2
   UIPC_PCG_BLOCK_DIM=$ENVV $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
     $PY main.py 130 > "$PFX.run.log" 2>&1)
  $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
    --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
  [ -s "${PFX}_cuda_gpu_kern_sum.csv" ] || { echo "NSYS-FAIL $PFX"; return 3; }
  echo "cap ok $PFX"
}

one "$S/rep_d1" 256
one "$S/rep_1a" 128
one "$S/rep_d2" 256
one "$S/rep_1b" 128
one "$S/rep_d3" 256
one "$S/rep_1c" 128
echo REPEATS_DONE

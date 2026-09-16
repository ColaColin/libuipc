#!/bin/bash
# s06: price launch_spread on the fused PCG vector kernels WITHOUT wiring it.
# spread_block_dim_from(n=50334, bd=256) -> 128 (grid 197 -> 394, want 8*40=320),
# which is exactly UIPC_PCG_BLOCK_DIM=128. Sweep the whole block-dim axis with
# full-run kern_sum captures; fused_dot / SpMV / MAS local solve are in-run
# controls (their geometry does not route through the knob).
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

kernsum() {  # prefix frames envsetting
  local PFX=$1 FRAMES=$2 ENVSET=$3
  rm -f "$PFX.nsys-rep" "${PFX}_cuda_gpu_kern_sum.csv" "$PFX.stats.log" "$PFX.run.log"
  (cd "$SCENE" || exit 2
   env $ENVSET $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
     $PY main.py $FRAMES > "$PFX.run.log" 2>&1)
  local rc=$?
  $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
    --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
  [ -s "${PFX}_cuda_gpu_kern_sum.csv" ] || { echo "NSYS-FAIL no kern_sum csv $PFX rc=$rc"; tail -3 "$PFX.run.log"; return 3; }
  echo "cap ok $PFX (rc=$rc)"
}

trace() {  # prefix frames envsetting  (short window, geometry + node gaps)
  local PFX=$1 FRAMES=$2 ENVSET=$3
  rm -f "$PFX.nsys-rep" "${PFX}_cuda_gpu_trace.csv" "$PFX.stats.log" "$PFX.run.log"
  (cd "$SCENE" || exit 2
   env $ENVSET $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
     $PY main.py $FRAMES > "$PFX.run.log" 2>&1)
  $NSYS stats --report cuda_gpu_trace --format csv --force-export true \
    --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
  [ -s "${PFX}_cuda_gpu_trace.csv" ] || { echo "TRACE-FAIL $PFX"; return 3; }
  echo "trace ok $PFX"
}

# full runs for per-launch pricing
kernsum "$S/sweep_bd64"  130 "UIPC_PCG_BLOCK_DIM=64"
kernsum "$S/sweep_bd128" 130 "UIPC_PCG_BLOCK_DIM=128"
kernsum "$S/sweep_def"   130 "UIPC_PCG_BLOCK_DIM=256"
kernsum "$S/sweep_bd512" 130 "UIPC_PCG_BLOCK_DIM=512"
kernsum "$S/sweep_bd1024" 130 "UIPC_PCG_BLOCK_DIM=1024"

# short traces for geometry + inter-node gaps (default + one spread arm)
trace "$S/geom_def"  3 "UIPC_PCG_BLOCK_DIM=256"
trace "$S/geom_bd128" 3 "UIPC_PCG_BLOCK_DIM=128"
echo SWEEP_DONE

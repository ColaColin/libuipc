#!/bin/bash
# Round-7 s05 nsys captures, v2: the round's own invocation (s00 nsysrun.sh /
# s03 nsysrun_scene.sh) -- nsight-systems 2024.6.2 install AND
# --cuda-graph-trace=node, without which the CUDA-graph-captured PCG/MAS
# kernels are invisible (the v1 capture lost ~950k launches that way).
set -u
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
S=/workspace/output/round7/s05
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH

cap() {  # prefix dir frames
  local PFX=$1 DIR=$2 FRAMES=$3
  rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv" "${PFX}_cuda_gpu_kern_sum.csv"
  (cd "$DIR" || exit 2
   $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
     $PY main.py $FRAMES > "$PFX.run2.log" 2>&1)
  local rc=$?
  $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
    --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats2.log" 2>&1
  [ -s "${PFX}_cuda_gpu_kern_sum.csv" ] || { echo "NSYS-FAIL no kern_sum csv $PFX rc=$rc"; tail -5 "$PFX.stats2.log"; return 3; }
  echo "cap ok $PFX rc=$rc"
}

cap "$S/nsys2_cp" /workspace/deps/libuipc-src/libuipc-samples/examples/103_crease_press 130
cap "$S/nsys2_cwc" /workspace/deps/libuipc-src/libuipc-samples/examples/93_cube_wall_cloth "--headless 100"
echo NSYS2_DONE

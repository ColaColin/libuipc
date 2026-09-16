#!/bin/bash
# s08 scope A/B: full 130-frame crease-press under nsys, both arms, fresh
# prefixes, graph-node tracing (s05 instrument lesson).
# new arm = default (SymAsm=2); old arm = UIPC_MAKE_SPD_BLOCKED_HALF=0.
set -u
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
S=/workspace/output/round7/s08
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH

cap() {  # prefix envstr
  local PFX=$1 ENVSTR=$2
  rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv" "${PFX}_cuda_gpu_kern_sum.csv"
  (cd /workspace/deps/libuipc-src/libuipc-samples/examples/103_crease_press || exit 2
   eval "export $ENVSTR"
   $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
     $PY main.py 130 > "$PFX.run.log" 2>&1)
  local rc=$?
  $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
    --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
  [ -s "${PFX}_cuda_gpu_kern_sum.csv" ] || { echo "NSYS-FAIL no kern_sum csv $PFX rc=$rc"; tail -5 "$PFX.stats.log"; return 3; }
  echo "cap ok $PFX ($ENVSTR) rc=$rc"
}
# ABBA order, two rounds for drift control
cap "$S/nsys_new_r1"  "UIPC_MAKE_SPD_BLOCKED_HALF="
cap "$S/nsys_old_r1"  "UIPC_MAKE_SPD_BLOCKED_HALF=0"
cap "$S/nsys_old_r2"  "UIPC_MAKE_SPD_BLOCKED_HALF=0"
cap "$S/nsys_new_r2"  "UIPC_MAKE_SPD_BLOCKED_HALF="
echo SCOPE_AB_DONE

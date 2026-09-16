#!/bin/bash
# s08 contact-context check: full cube-wall-cloth run under nsys on the branch
# build (default env). The contact EE arm embeds the new 4x3 assembly (its
# stack grew 80-184 B); compare do_assemble per-launch vs s05's main-side
# reference (1978.0 us/launch for <...,8>, cross-session envelope +-1.2-2 %).
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
PFX=$S/nsys_cwc
rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv" "${PFX}_cuda_gpu_kern_sum.csv"
(cd $TREE/libuipc-samples/examples/93_cube_wall_cloth || exit 2
 $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
   $PY main.py > "$PFX.run.log" 2>&1)
rc=$?
$NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
  --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
[ -s "${PFX}_cuda_gpu_kern_sum.csv" ] && echo "cap ok $PFX rc=$rc" || { echo "NSYS-FAIL rc=$rc"; tail -3 "$PFX.stats.log"; }

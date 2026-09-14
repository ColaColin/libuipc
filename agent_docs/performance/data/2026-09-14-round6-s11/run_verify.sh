#!/bin/bash
# s11: bit-identity of the two new call sites (=3, =4) against the in-place path.
set -u
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
export UIPC_ABD_GH_PREPASS_VERIFY=1
OUT=/workspace/output/round6/s11/verify
mkdir -p "$OUT"
for m in 3 4; do
  for sc in "6_wrecking_balls 40" "93_cube_wall_cloth 40"; do
    set -- $sc
    ( cd "$TREE/libuipc-samples/examples/$1" && UIPC_ABD_GH_PREPASS=$m $PY main.py --headless $2 ) \
      > "$OUT/v_m${m}_$1.log" 2>&1
    echo "== mode $m  $1"; grep -iE "ABDGHPrepassVerify|mismatch|TOTAL frames" "$OUT/v_m${m}_$1.log" | tail -4
  done
done
echo VERIFY_DONE

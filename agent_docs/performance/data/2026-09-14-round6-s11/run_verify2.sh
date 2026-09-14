#!/bin/bash
set -u
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH UIPC_ABD_GH_PREPASS
export UIPC_ABD_GH_PREPASS_VERIFY=1
OUT=/workspace/output/round6/s11/verify; mkdir -p "$OUT"
for sc in "6_wrecking_balls 40" "93_cube_wall_cloth 40"; do
  set -- $sc
  ( cd "$TREE/libuipc-samples/examples/$1" && $PY main.py --headless $2 ) > "$OUT/v_default_$1.log" 2>&1
  echo "== shipped default (no UIPC_ABD_GH_PREPASS set => mode 4)  $1"
  grep -iE "ABDGHPrepassVerify|TOTAL frames" "$OUT/v_default_$1.log" | tail -3
done
echo VERIFY2_DONE

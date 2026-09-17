#!/bin/bash
# Round-7 s21: remaining five scenes + one --verify per arm (regime) + the
# closing kernel-ranking snapshot (nsys full run, head).
# crease-press (n=20) runs separately first; this fires after it.
set -eu
source /workspace/deps/libuipc-src/env_perf.sh
source /workspace/output/round7/base/env_base.sh
cd /workspace/deps/libuipc-src
OUT=/workspace/output/round7/s21
AB2=/workspace/output/round7/ab2.py

$UIPC_PERF_PY $AB2 --scene tumbler-garments --n 10 --label s21 --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY $AB2 --scene cube-wall-cloth  --n 10 --label s21 --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY $AB2 --scene stiff-gipc-case2 --n 5  --label s21 --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY $AB2 --scene mas-bunny        --n 5  --label s21 --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY $AB2 --scene rigid-wrecking-balls --n 5 --label s21 --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY

# One full --verify run per arm at the final binaries (regime observables).
for arm in base head; do
  if [ $arm = base ]; then PY=$UIPC_BASE_PY; else PY=$UIPC_PERF_PY; fi
  ( cd libuipc-samples/examples/103_crease_press && \
    $PY main.py --headless 130 --verify --result $OUT/verify_${arm}.json ) \
    > $OUT/verify_${arm}.log 2>&1
  echo "verify $arm done: $(tail -1 $OUT/verify_${arm}.log)"
done

# Closing kernel-ranking snapshot: one full-run nsys at final head.
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
PFX=$OUT/kernelsnap
rm -f "$PFX.nsys-rep" "$PFX.sqlite" "${PFX}_cuda_gpu_trace.csv" "${PFX}_cuda_gpu_kern_sum.csv"
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
( cd libuipc-samples/examples/103_crease_press && \
  $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$PFX" \
    $UIPC_PERF_PY main.py 130 --headless > "$PFX.run.log" 2>&1 )
$NSYS stats --report cuda_gpu_kern_sum --format csv --force-export true \
  --output "$PFX" "$PFX.nsys-rep" > "$PFX.stats.log" 2>&1
[ -s "${PFX}_cuda_gpu_kern_sum.csv" ] || { echo "NSYS-FAIL no kern_sum csv"; exit 3; }
echo "S21_REST_DONE"

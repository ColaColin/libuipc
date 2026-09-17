#!/bin/bash
# Round-7 s20 Part B: accumulated drift, base (fce57589) vs head (69c2af51),
# two builds, interleaved ABBA with one discarded warm-up per arm (ab2.py).
# crease-press n=10/arm; the five regression scenes n=5/arm.
set -eu
source /workspace/deps/libuipc-src/env_perf.sh
source /workspace/output/round7/base/env_base.sh
cd /workspace/deps/libuipc-src

$UIPC_PERF_PY /workspace/output/round7/s20/ab2.py --scene crease-press     --n 10 --label drift_cp    --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY /workspace/output/round7/s20/ab2.py --scene cube-wall-cloth  --n 5  --label drift_cwc   --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY /workspace/output/round7/s20/ab2.py --scene tumbler-garments --n 5  --label drift_tumb  --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY /workspace/output/round7/s20/ab2.py --scene stiff-gipc-case2 --n 5  --label drift_case2 --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY /workspace/output/round7/s20/ab2.py --scene mas-bunny        --n 5  --label drift_bunny --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY
$UIPC_PERF_PY /workspace/output/round7/s20/ab2.py --scene rigid-wrecking-balls --n 5 --label drift_rwb --arm base $UIPC_BASE_PY --arm head $UIPC_PERF_PY

# One full --verify run per arm (regime observables), outside the timed section.
for arm in base head; do
  if [ $arm = base ]; then PY=$UIPC_BASE_PY; else PY=$UIPC_PERF_PY; fi
  ( cd libuipc-samples/examples/103_crease_press && \
    $PY main.py --headless 130 --verify --result /workspace/output/round7/s20/verify_${arm}.json ) \
    > /workspace/output/round7/s20/verify_${arm}.log 2>&1
  echo "verify $arm done: $(tail -1 /workspace/output/round7/s20/verify_${arm}.log)"
done
echo DRIFT_DONE

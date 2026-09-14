#!/bin/bash
# s13 end-to-end sweeps. One build, env-switch A/B through ab.py (ABBA + one
# discarded warm-up per arm). Serial: nothing else may touch the GPU.
set -u
cd /workspace/deps/libuipc-src
source /workspace/deps/libuipc-src/env_perf.sh
AB="$UIPC_PERF_PY /workspace/output/round6/ab.py"
O=/workspace/output/round6/s13
run() { echo "##### $*"; $AB "$@" 2>&1; echo; }

run --scene mas-bunny --n 10 --label s13mb --out $O/ab \
    --arm off UIPC_SPMV_GRID_FIT=0 --arm fit UIPC_SPMV_GRID_FIT=1
run --scene stiff-gipc-case2 --n 8 --label s13c2 --out $O/ab \
    --arm off UIPC_SPMV_GRID_FIT=0 --arm fit UIPC_SPMV_GRID_FIT=1
run --scene mas-bunny --n 6 --label s13null --out $O/ab \
    --arm nullA UIPC_SPMV_GRID_FIT=1 --arm nullB UIPC_SPMV_GRID_FIT=01
run --scene cube-wall-cloth --n 6 --label s13cwc --out $O/ab \
    --arm off UIPC_SPMV_GRID_FIT=0 --arm fit UIPC_SPMV_GRID_FIT=1
run --scene rigid-wrecking-balls --n 6 --label s13rwb --out $O/ab \
    --arm off UIPC_SPMV_GRID_FIT=0 --arm fit UIPC_SPMV_GRID_FIT=1

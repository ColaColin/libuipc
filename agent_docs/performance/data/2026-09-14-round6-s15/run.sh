#!/bin/bash
# s15: plain (un-profiled) run of one example: run.sh <tag> <example dir> <frames> [KEY=VAL ...]
set -u
LOG=/workspace/output/round6/s15/runs/$1.log; DIR=$2; FRAMES=$3; shift 3
mkdir -p /workspace/output/round6/s15/runs
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Info UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
for kv in "$@"; do export "$kv"; done
cd "$TREE/libuipc-samples/examples/$DIR" || exit 2
$PY main.py --headless $FRAMES > "$LOG" 2>&1
echo "run rc=$? $LOG"

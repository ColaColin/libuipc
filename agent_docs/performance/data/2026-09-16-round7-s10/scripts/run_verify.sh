#!/bin/bash
# s10: one full --verify default run at head (regime observables)
set -u
source /workspace/deps/libuipc-src/env_perf.sh
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
unset WB_TIMER NO_MAS NO_GRAPH UIPC_D2H_PROFILE
cd /workspace/deps/libuipc-src/libuipc-samples/examples/103_crease_press || exit 2
$UIPC_PERF_PY main.py 130 --verify --result /workspace/output/round7/s10/verify_head.json > /workspace/output/round7/s10/verify_head.log 2>&1
echo "verify rc=$?"
tail -5 /workspace/output/round7/s10/verify_head.log

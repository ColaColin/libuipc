#!/bin/bash
# s10: full default-frame crease-press run with the round-4 host_read funnel profiler
set -u
source /workspace/deps/libuipc-src/env_perf.sh
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_D2H_PROFILE=2
unset WB_TIMER NO_MAS NO_GRAPH
cd /workspace/deps/libuipc-src/libuipc-samples/examples/103_crease_press || exit 2
$UIPC_PERF_PY main.py 130 > /workspace/output/round7/s10/stall/cp_d2hprof.log 2>&1
echo "rc=$?"
grep -c "d2h-site" /workspace/output/round7/s10/stall/cp_d2hprof.log

#!/bin/bash
set -e
source /workspace/deps/libuipc-src/build_env.sh
export PATH=/workspace/deps/localbin:/workspace/deps/uipc-perf-env/bin:$CUDA_HOME/bin:$PATH
cmake --build /workspace/output/round6/base/wt/build-base -j 20

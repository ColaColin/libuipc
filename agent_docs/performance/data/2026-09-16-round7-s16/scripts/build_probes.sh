#!/bin/bash
# s16 standalone diagnosis probes (no engine build; same include set/flags as
# the backend TU, minus -rdc). Run: ./build_probes.sh && ./ql_probe 65536 100
set -e
REPO=/workspace/deps/libuipc-src
CUDA=/workspace/deps/cuda-12.8
HERE="$(cd "$(dirname "$0")" && pwd)"
for P in ql_probe mc_probe levers_probe; do
  $CUDA/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ \
    -DUIPC_PROJECT_DIR="R\"($REPO)\"" -DUIPC_RUNTIME_CHECK=1 \
    -I$REPO/src -I$REPO/src/backends/cuda -I$REPO/src/backends/cuda/cuda_tool \
    -I$CUDA/include -I$REPO/include \
    -isystem $REPO/build-perf/vcpkg_installed/x64-linux/include/eigen3 \
    -isystem $REPO/build-perf/vcpkg_installed/x64-linux/include \
    -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_75,code=sm_75" \
    --extended-lambda --expt-relaxed-constexpr -diag-suppress=554 \
    -o "$HERE/$P" "$HERE/$P.cu"
done

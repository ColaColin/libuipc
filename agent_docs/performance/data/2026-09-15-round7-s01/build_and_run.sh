#!/bin/bash
# Round-7 s01 numerics verifier: build + run.
# Uses the same include set and flags as the backend TU (from
# build-perf/compile_commands.json), minus -rdc (all-inline device code).
set -e
REPO=/workspace/deps/libuipc-src
CUDA=/workspace/deps/cuda-12.8
HERE="$(cd "$(dirname "$0")" && pwd)"

$CUDA/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ \
  -DUIPC_PROJECT_DIR="R\"($REPO)\"" -DUIPC_RUNTIME_CHECK=1 \
  -I$REPO/src -I$REPO/src/backends/cuda -I$REPO/src/backends/cuda/cuda_tool \
  -I$CUDA/include -I$REPO/include \
  -isystem $REPO/build-perf/vcpkg_installed/x64-linux/include/eigen3 \
  -isystem $REPO/build-perf/vcpkg_installed/x64-linux/include \
  -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_75,code=sm_75" \
  --extended-lambda --expt-relaxed-constexpr \
  -o "$HERE/verify_proj" "$HERE/verify_proj.cu"

"$HERE/verify_proj" "${1:-200000}" "${2:-7}"

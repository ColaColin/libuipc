#!/bin/bash
set -eu
SRC=/workspace/output/round6/s07/gn_contact_probe.cu
OUT=/workspace/output/round6/s07/gn_contact_probe
/workspace/deps/cuda-12.8/bin/nvcc -O2 -std=c++20 -arch=sm_75 "$SRC" -o "$OUT" \
  -ccbin=/usr/bin/g++ --expt-relaxed-constexpr -diag-suppress 20012 \
  -I/workspace/deps/libuipc-src/src \
  -I/workspace/deps/libuipc-src/src/backends/cuda \
  -I/workspace/deps/libuipc-src/src/backends/cuda/cuda_tool \
  -I/workspace/deps/cuda-12.8/include \
  -I/workspace/deps/libuipc-src/include \
  -isystem /workspace/deps/libuipc-src/build-perf/vcpkg_installed/x64-linux/include/eigen3 \
  -isystem /workspace/deps/libuipc-src/build-perf/vcpkg_installed/x64-linux/include \
  -DNDEBUG -DUIPC_RUNTIME_CHECK=1 \
  -DUIPC_BACKEND_DIR='R"(/workspace/deps/libuipc-src/src/backends)"' \
  -DUIPC_BACKEND_NAME='R"(cuda)"' \
  -DUIPC_PROJECT_DIR='R"(/workspace/deps/libuipc-src)"' \
  -DUIPC_RELATIVE_SOURCE_FILE='R"(probe)"'
echo "built $OUT"

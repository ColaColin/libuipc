#!/bin/bash
# s25 static probe: compile one SNH TU variant with the real build flags, sm_86
set -e
SRC=$1; OUT=$2
S=/root/work/src
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler \
 -DCPPTRACE_STATIC_DEFINE -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL \
 -DUIPC_BACKEND_DIR="R\"($S/src/backends)\"" -DUIPC_BACKEND_EXPORT_DLL=1 \
 -DUIPC_BACKEND_NAME="R\"(cuda)\"" -DUIPC_PROJECT_DIR="R\"($S)\"" -DUIPC_RUNTIME_CHECK=1 \
 -DUIPC_VERSION_MAJOR=0 -DUIPC_VERSION_MINOR=9 -DUIPC_VERSION_PATCH=0 \
 -DUIPC_RELATIVE_SOURCE_FILE="R\"(src/backends/cuda/finite_element/constitutions/stable_neo_hookean_3d.cu)\"" \
 -I$S/src -I$S/src/backends/cuda -I$S/src/backends/cuda/cuda_tool \
 -I/usr/local/cuda/targets/x86_64-linux/include -I$S/include \
 -isystem $S/build/vcpkg_installed/x64-linux/include/eigen3 \
 -isystem $S/build/vcpkg_installed/x64-linux/include \
 -O3 -DNDEBUG -std=c++20 -arch=native -Xcompiler=-fPIC --extended-lambda --expt-relaxed-constexpr \
 --diag-suppress=20012,1388,27,174,1394,997,1866,69,177,554,20014,2361,20011,940,55,221,1028 \
 -Xptxas -v -x cu -rdc=true -c "$SRC" -o "$OUT.o" 2> "$OUT.ptxas"

#!/bin/bash
# static stage-stubbing probe for StableNeoHookean3D G/H (CPU only, no GPU)
set -e
S=/workspace/deps/libuipc-src
OUT=$1; shift
/workspace/deps/cuda-12.8/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ \
 -DCPPTRACE_STATIC_DEFINE -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL \
 -DUIPC_BACKEND_DIR="R\"($S/src/backends)\"" -DUIPC_BACKEND_EXPORT_DLL=1 \
 -DUIPC_BACKEND_NAME="R\"(cuda)\"" -DUIPC_PROJECT_DIR="R\"($S)\"" -DUIPC_RUNTIME_CHECK=1 \
 -DUIPC_VERSION_MAJOR=0 -DUIPC_VERSION_MINOR=9 -DUIPC_VERSION_PATCH=0 \
 -DUIPC_RELATIVE_SOURCE_FILE="R\"(probe.cu)\"" \
 -I$(pwd)/inc -I$S/src -I$S/src/backends/cuda -I$S/src/backends/cuda/cuda_tool -I/workspace/deps/cuda-12.8/include -I$S/include \
 -isystem $S/build-perf/vcpkg_installed/x64-linux/include/eigen3 \
 -isystem $S/build-perf/vcpkg_installed/x64-linux/include \
 -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_75,code=[compute_75,sm_75]" \
 -Xcompiler=-fPIC --extended-lambda --expt-relaxed-constexpr \
 --diag-suppress=20012,1388,27,174,1394,997,1866,69,177,554,20014,2361,20011,940,55,221,1028 \
 -Xptxas -v "$@" -x cu -rdc=true -c probe.cu -o $OUT.o 2> $OUT.ptxas

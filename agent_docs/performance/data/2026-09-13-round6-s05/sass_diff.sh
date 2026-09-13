#!/bin/bash
# Compile the trajectory-filter TU at two revisions with the build's exact
# flags and diff the SASS per instantiation.  Fails loudly if either compile
# or either cuobjdump produces nothing.
set -eu
TREE=/workspace/deps/libuipc-src
OUT=/workspace/output/round6/s05/sass
mkdir -p "$OUT"
SRC=src/backends/cuda/collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.cu
compile() {  # $1 = tag ; compiles whatever is currently in the tree
  /workspace/deps/cuda-12.8/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ \
    -DCPPTRACE_STATIC_DEFINE -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL \
    -DUIPC_BACKEND_DIR="R\"($TREE/src/backends)\"" -DUIPC_BACKEND_EXPORT_DLL=1 \
    -DUIPC_BACKEND_NAME="R\"(cuda)\"" -DUIPC_PROJECT_DIR="R\"($TREE)\"" \
    -DUIPC_RUNTIME_CHECK=1 -DUIPC_VERSION_MAJOR=0 -DUIPC_VERSION_MINOR=9 -DUIPC_VERSION_PATCH=0 \
    -DUIPC_RELATIVE_SOURCE_FILE="R\"($SRC)\"" \
    -I$TREE/src -I$TREE/src/backends/cuda -I$TREE/src/backends/cuda/cuda_tool \
    -I/workspace/deps/cuda-12.8/include -I$TREE/include \
    -isystem $TREE/build-perf/vcpkg_installed/x64-linux/include/eigen3 \
    -isystem $TREE/build-perf/vcpkg_installed/x64-linux/include \
    -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_75,code=[compute_75,sm_75]" \
    -Xcompiler=-fPIC --extended-lambda --expt-relaxed-constexpr \
    --diag-suppress=20012,1388,27,174,1394,997,1866,69,177,554,20014,2361,20011,940,55,221,1028 \
    -x cu -rdc=true -c "$TREE/$SRC" -o "$OUT/$1.o"
  /workspace/deps/cuda-12.8/bin/cuobjdump -sass "$OUT/$1.o" > "$OUT/$1.sass"
  [ -s "$OUT/$1.sass" ] || { echo "SASS-FAIL empty dump for $1"; exit 3; }
  /workspace/deps/cuda-12.8/bin/cuobjdump -res-usage "$OUT/$1.o" > "$OUT/$1.res"
}
compile "$1"
echo "compiled $1 -> $OUT/$1.sass ($(wc -l < "$OUT/$1.sass") lines)"

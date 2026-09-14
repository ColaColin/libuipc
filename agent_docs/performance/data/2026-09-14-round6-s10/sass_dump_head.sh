#!/bin/bash
# s10: compile the two touched device TUs at HEAD and at the branch base with
# the exact build-perf command line, and diff the SASS of every kernel in them.
# The step's claim is that only the *launch stream* moved: the device code must
# be byte-identical.
set -eu
OUT=/workspace/output/round6/s10/sass
mkdir -p "$OUT"
CUDA=/workspace/deps/cuda-12.8
BP=/workspace/deps/libuipc-src/build-perf
compile() { # $1 = tree root, $2 = rel .cu path, $3 = out .o
  $CUDA/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ \
    -DCPPTRACE_STATIC_DEFINE -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL \
    -DUIPC_BACKEND_DIR="R\"($1/src/backends)\"" -DUIPC_BACKEND_EXPORT_DLL=1 \
    -DUIPC_BACKEND_NAME="R\"(cuda)\"" -DUIPC_PROJECT_DIR="R\"($1)\"" \
    -DUIPC_RUNTIME_CHECK=1 -DUIPC_VERSION_MAJOR=0 -DUIPC_VERSION_MINOR=9 -DUIPC_VERSION_PATCH=0 \
    -DUIPC_RELATIVE_SOURCE_FILE="R\"($2)\"" \
    -I$1/src -I$1/src/backends/cuda -I$1/src/backends/cuda/cuda_tool -I$CUDA/include -I$1/include \
    -isystem $BP/vcpkg_installed/x64-linux/include/eigen3 -isystem $BP/vcpkg_installed/x64-linux/include \
    -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_75,code=[compute_75,sm_75]" \
    -Xcompiler=-fPIC --extended-lambda --expt-relaxed-constexpr \
    --diag-suppress=20012,1388,27,174,1394,997,1866,69,177,554,20014,2361,20011,940,55,221,1028 \
    -x cu -rdc=true -c "$1/$2" -o "$3"
}
for f in src/backends/cuda/affine_body/constitutions/ortho_potential.cu \
         src/backends/cuda/affine_body/bdf/affine_body_bdf1_kinetic.cu \
         src/backends/cuda/affine_body/constitutions/arap.cu \
         src/backends/cuda/affine_body/abd_linear_subsystem.cu ; do
  b=$(basename "$f" .cu)
  compile /workspace/deps/libuipc-src "$f" "$OUT/$b.head.o"
  [ -s "$OUT/$b.base.o" ] || compile /workspace/output/round6/s10/base-tree "$f" "$OUT/$b.base.o"
  $CUDA/bin/cuobjdump -sass "$OUT/$b.head.o" > "$OUT/$b.head.sass"
  [ -s "$OUT/$b.base.sass" ] || $CUDA/bin/cuobjdump -sass "$OUT/$b.base.o" > "$OUT/$b.base.sass"
  python3 /workspace/output/round6/s10/sass_fn_diff.py "$OUT/$b.base.sass" "$OUT/$b.head.sass" || true
done

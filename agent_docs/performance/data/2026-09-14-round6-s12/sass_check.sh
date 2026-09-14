#!/bin/bash
# s12: the two .cu files this step touches carry only HOST changes (an env parse
# moved into a shared header, two call sites, a default). Prove the device code
# is byte-identical by compiling both TUs at HEAD and at the branch base with the
# build-perf command line and diffing the SASS per function, after normalising
# the anonymous-namespace mangling (nvcc encodes the source path and the temp
# file's PID into every such symbol).
set -u
OUT=/workspace/output/round6/s12/sass
mkdir -p "$OUT"
CUDA=/workspace/deps/cuda-12.8
BP=/workspace/deps/libuipc-src/build-perf
compile() {
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
norm() { sed -E 's/_GLOBAL__N__[0-9a-f]+_/_GLOBAL__N__X_/g; s/__nv_static_[0-9]+__[0-9a-f]+_/__nv_static_X__X_/g; s/_cu_[0-9a-f]+_[0-9]+/_cu_X_X/g' "$1"; }
for f in src/backends/cuda/finite_element/fem_linear_subsystem.cu ; do
  b=$(basename "$f" .cu)
  compile /workspace/deps/libuipc-src "$f" "$OUT/$b.head.o" || { echo "COMPILE FAIL head $b"; continue; }
  compile /workspace/output/round6/s12/base-tree "$f" "$OUT/$b.base.o" || { echo "COMPILE FAIL base $b"; continue; }
  $CUDA/bin/cuobjdump -sass "$OUT/$b.head.o" | grep -v "^\s*//" > "$OUT/$b.head.sass"
  $CUDA/bin/cuobjdump -sass "$OUT/$b.base.o" | grep -v "^\s*//" > "$OUT/$b.base.sass"
  norm "$OUT/$b.base.sass" > "$OUT/$b.base.norm"; norm "$OUT/$b.head.sass" > "$OUT/$b.head.norm"
  nf=$(grep -c 'Function : ' "$OUT/$b.head.norm"); ni=$(grep -cE '^\s+/\*[0-9a-f]+\*/' "$OUT/$b.head.norm")
  if diff -q "$OUT/$b.base.norm" "$OUT/$b.head.norm" >/dev/null; then
    echo "SASS IDENTICAL  $b   ($nf functions, $ni SASS instructions)"
  else
    echo "SASS DIFFERS    $b   ($nf functions)"; diff "$OUT/$b.base.norm" "$OUT/$b.head.norm" | head -30
  fi
done

#!/bin/bash
# Build one libuipc tree from source (Release, native sm_120, pybind into its own venv). Usage: remote_build.sh head|base
set -euo pipefail
T=$1; R=/work/libuipc-$T; V=/work/venv-src-$T; J=${J:-$(nproc)}
export VCPKG_ROOT=/work/vcpkg
[ -d $VCPKG_ROOT ] || { git clone -q https://github.com/microsoft/vcpkg $VCPKG_ROOT && $VCPKG_ROOT/bootstrap-vcpkg.sh -disableMetrics >/dev/null; }
[ -d $V ] || { python3 -m venv $V; $V/bin/pip install -q --upgrade pip; $V/bin/pip install -q numpy pybind11 pybind11-stubgen; }
cd $R; mkdir -p build
# pre-install vcpkg deps (same manifest CMake would generate); shared binary cache across the two trees
$V/bin/python scripts/gen_vcpkg_json.py build --dev_mode=OFF --with_usd_support=OFF --with_vdb_support=OFF --with_cuda_backend=ON >/dev/null
$VCPKG_ROOT/vcpkg install --x-manifest-root=build --x-install-root=build/vcpkg_installed --triplet x64-linux --disable-metrics > build/vcpkg_install.log 2>&1 || { tail -30 build/vcpkg_install.log; exit 1; }
echo "vcpkg done $(date -u +%T)"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DUIPC_CUDA_ARCHITECTURES=native -DUIPC_BUILD_PYBIND=ON -DUIPC_PYTHON_EXECUTABLE_PATH=$V/bin/python \
  -DUIPC_BUILD_EXAMPLES=OFF -DUIPC_BUILD_TESTS=OFF -DUIPC_BUILD_BENCHMARKS=OFF -DUIPC_DEV_MODE=ON > build/configure.log 2>&1 || { tail -40 build/configure.log; exit 1; }
grep -E "CUDA_ARCHITECTURES|CUDA compiler identification" build/configure.log | head -3
echo "configure done $(date -u +%T)"
cmake --build build -j$J > build/build.log 2>&1 || { grep -nE "error|Error" build/build.log | head -30; tail -20 build/build.log; exit 1; }
echo "build done $(date -u +%T)"
$V/bin/python -c "import uipc; print('$T src build', uipc.__version__)"
sha256sum $V/lib/python3*/site-packages/uipc/_native/libuipc_backend_cuda.so build/Release/bin/libuipc_backend_cuda.so
echo BUILD_OK $T

#!/bin/bash
# Configure the isolated Dahl-constitution build (Release, python bindings into
# /workspace/deps/uipc-dahl-env). Never touches the active perf/newton-cost
# install in /workspace/deps/uipc-src-env. Usage: ./configure_dahl.sh [extra cmake args]
set -e
source "$(dirname "$0")/build_env.sh"
# Override the build python with the dedicated isolated venv
export UIPC_SRC_PY=/workspace/deps/uipc-dahl-env/bin/python
export PATH=/workspace/deps/localbin:/workspace/deps/uipc-dahl-env/bin:$CUDA_HOME/bin:$PATH
cd "$(dirname "$0")"
cmake -S . -B build-dahl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DCMAKE_CUDA_COMPILER=$CUDACXX \
  -DCMAKE_CUDA_HOST_COMPILER=$CUDAHOSTCXX \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DUIPC_CUDA_ARCHITECTURES="75" \
  -DUIPC_BUILD_PYBIND=ON \
  -DUIPC_PYTHON_EXECUTABLE_PATH=$UIPC_SRC_PY \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_BENCHMARKS=OFF \
  -DUIPC_DEV_MODE=ON \
  -DVCPKG_MANIFEST_INSTALL=OFF \
  -DVCPKG_INSTALLED_DIR=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed \
  "$@"

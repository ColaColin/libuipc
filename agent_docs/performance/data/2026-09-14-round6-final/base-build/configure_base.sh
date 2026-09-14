#!/bin/bash
# Round-6 V3: configure the perf-round6-base (60af65e6) worktree exactly as configure_perf.sh
# configures build-perf: same toolchain, same flags, same vcpkg tree, tests ON, sm_75 --
# only the source tree, the build directory and the venv differ.
set -e
source /workspace/deps/libuipc-src/build_env.sh
export UIPC_SRC_PY=/workspace/output/round6/base/venv/bin/python
export PATH=/workspace/deps/localbin:/workspace/deps/uipc-perf-env/bin:$CUDA_HOME/bin:$PATH
cd /workspace/output/round6/base/wt
cmake -S . -B build-base -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
  -DCMAKE_CUDA_COMPILER=$CUDACXX \
  -DCMAKE_CUDA_HOST_COMPILER=$CUDAHOSTCXX \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DUIPC_CUDA_ARCHITECTURES="75" \
  -DUIPC_BUILD_PYBIND=ON \
  -DUIPC_PYTHON_EXECUTABLE_PATH=$UIPC_SRC_PY \
  -DUIPC_BUILD_EXAMPLES=OFF \
  -DUIPC_BUILD_TESTS=ON \
  -DUIPC_BUILD_BENCHMARKS=OFF \
  -DUIPC_DEV_MODE=ON \
  -DVCPKG_MANIFEST_INSTALL=OFF \
  -DVCPKG_INSTALLED_DIR=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed \
  -DCMAKE_MAKE_PROGRAM=/workspace/deps/uipc-perf-env/bin/ninja \
  "$@"

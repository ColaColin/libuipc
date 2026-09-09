# Source this before configuring/building libuipc from source on this box.
#   source /workspace/deps/libuipc-src/build_env.sh
export CUDA_HOME=/workspace/deps/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDAToolkit_ROOT=$CUDA_HOME
export CUDAHOSTCXX=/usr/bin/g++
export VCPKG_ROOT=/workspace/deps/vcpkg
export CMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
export UIPC_SRC_PY=/workspace/deps/uipc-src-env/bin/python
# localbin: zip/unzip python shims (vcpkg); venv bin: cmake + ninja
export PATH=/workspace/deps/localbin:/workspace/deps/uipc-src-env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

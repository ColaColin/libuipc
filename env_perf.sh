# Runtime environment for the isolated perf/kernels-branch pyuipc in /workspace/deps/uipc-perf-env.
#   source /workspace/deps/libuipc-src/env_perf.sh
export UIPC_PERF_PY=/workspace/deps/uipc-perf-env/bin/python
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$CUDA_HOME/bin:/workspace/deps/uipc-perf-env/bin:$PATH
# Headless polyscope shim from libuipc-samples (replaces the GL/X11 polyscope)
export PYTHONPATH=/workspace/archive/libuipc-samples/shim${PYTHONPATH:+:$PYTHONPATH}

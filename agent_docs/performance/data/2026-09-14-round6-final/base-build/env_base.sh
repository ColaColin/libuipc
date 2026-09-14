# Runtime environment for the perf-round6-base (60af65e6) pyuipc build (round-6 V3).
export UIPC_BASE_PY=/workspace/output/round6/base/venv/bin/python
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$CUDA_HOME/bin:/workspace/output/round6/base/venv/bin:$PATH
export PYTHONPATH=/workspace/archive/libuipc-samples/shim${PYTHONPATH:+:$PYTHONPATH}

#!/bin/bash
set -u
TREE=/workspace/deps/libuipc-src
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
export UIPC_CCD_COMPACT=1 UIPC_CCD_COMPACT_VERIFY=1
cd $TREE/libuipc-samples/examples/$1
/workspace/deps/uipc-perf-env/bin/python main.py --headless ${2:-180} 2>&1 | grep -E 'ccd_compact_verify|Error|error' | tail -40

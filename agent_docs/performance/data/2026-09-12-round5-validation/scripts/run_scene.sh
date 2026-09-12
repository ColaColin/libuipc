#!/bin/bash
# usage: run_scene.sh <tree> <pyexe> <scene-dir> <frames> [extra env...]
TREE=$1; PY=$2; DIR=$3; FRAMES=$4
export PYTHONPATH=/workspace/archive/libuipc-samples/shim${PYTHONPATH:+:$PYTHONPATH}
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
cd $TREE/libuipc-samples/examples/$DIR && exec $PY main.py --headless $FRAMES

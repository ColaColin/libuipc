#!/bin/bash
# s08 diagnosis: flag-dimension census of the contact pair population.
set -u
SC=$1
case $SC in
  cwc) DIR=93_cube_wall_cloth;      F=100 ;;
  c2)  DIR=88_stiff_gipc_benchmark; F=250 ;;
  tum) DIR=95_tumbler_garments;     F=180 ;;
  rwb) DIR=6_wrecking_balls;        F=120 ;;
  mas) DIR=94_mas_bunny;            F=100 ;;
  *) echo "unknown scene $SC"; exit 2 ;;
esac
TREE=/workspace/deps/libuipc-src
PY=/workspace/deps/uipc-perf-env/bin/python
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0 UIPC_CONTACT_FLAGSTATS=1
cd "$TREE/libuipc-samples/examples/$DIR" || exit 2
$PY main.py --headless $F 2>&1 | grep UIPC_FLAGSTATS | tail -3

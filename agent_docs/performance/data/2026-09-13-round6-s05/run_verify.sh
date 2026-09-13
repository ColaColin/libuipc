#!/bin/bash
set -u
TREE=/workspace/deps/libuipc-src
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
cd $TREE/libuipc-samples/examples/95_tumbler_garments
for r in 1 2 3 4 5 6; do
  for v in 0 1; do
    UIPC_CCD_CULL=$v /workspace/deps/uipc-perf-env/bin/python main.py --headless --verify 180 \
      > /workspace/output/round6/s05/verify/verify_c${v}_r${r}.txt 2>&1 \
      || echo "FAIL verify_c${v}_r${r}"
  done
done
echo VERIFY_DONE

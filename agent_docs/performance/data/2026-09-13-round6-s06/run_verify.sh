#!/bin/bash
# s06: the tumbler --verify audit AND the n>=40/arm re-characterisation of
# s05's 1-in-14 preconditioner-NaN abort, in one job.
set -u
TREE=/workspace/deps/libuipc-src
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
cd $TREE/libuipc-samples/examples/95_tumbler_garments
N=${1:-40}
for r in $(seq 1 $N); do
  for v in 0 1; do
    UIPC_CCD_COMPACT=$v /workspace/deps/uipc-perf-env/bin/python main.py --headless --verify 180 \
      > /workspace/output/round6/s06/verify/verify_c${v}_r${r}.txt 2>&1 \
      || echo "FAIL verify_c${v}_r${r}"
  done
  echo "verify round $r done"
done
echo VERIFY_DONE

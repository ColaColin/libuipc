#!/bin/bash
# s07: tumbler --verify audit, three Proj arms (0 = exact default, 1 = PE+PP rank-1,
# 5 = PP-only rank-1).  Non-penetration observables, not just convergence.
set -u
TREE=/workspace/deps/libuipc-src
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
cd $TREE/libuipc-samples/examples/95_tumbler_garments
N=${1:-8}
for r in $(seq 1 $N); do
  for v in 0 1 5; do
    UIPC_CONTACT_RANK1=$v /workspace/deps/uipc-perf-env/bin/python main.py --headless --verify 180 \
      > /workspace/output/round6/s07/verify/verify_p${v}_r${r}.txt 2>&1 \
      || echo "FAIL verify_p${v}_r${r}"
  done
  echo "verify round $r done"
done
echo VERIFY_DONE

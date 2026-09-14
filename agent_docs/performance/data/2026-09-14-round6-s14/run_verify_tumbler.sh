#!/bin/bash
# s14: the tumbler --verify audit, n per arm = $1 (default 6), arms off / mode $2 (default 1).
set -u
TREE=/workspace/deps/libuipc-src
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
O=/workspace/output/round6/s14/tverify; mkdir -p $O
cd $TREE/libuipc-samples/examples/95_tumbler_garments
N=${1:-6}; M=${2:-1}
for r in $(seq 1 $N); do
  for v in 0 $M; do
    UIPC_CONTACT_DEFERRED_JOIN=$v /workspace/deps/uipc-perf-env/bin/python main.py --headless --verify 180 \
      > $O/verify_m${v}_r${r}.txt 2>&1 || echo "FAIL verify_m${v}_r${r}"
  done
  echo "verify round $r done"
done
echo TVERIFY_DONE

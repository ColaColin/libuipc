#!/bin/bash
# s09: tumbler --verify audit, three UIPC_CONTACT_SPLIT arms.  The split changes
# execution order (and therefore atomic accumulation order), so the arms are audited
# on the non-penetration / containment observables, not just on convergence.
set -u
TREE=/workspace/deps/libuipc-src
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
cd $TREE/libuipc-samples/examples/95_tumbler_garments
N=${1:-6}
mkdir -p /workspace/output/round6/s09/verify
for r in $(seq 1 $N); do
  for v in 2 1 0; do
    UIPC_CONTACT_SPLIT=$v /workspace/deps/uipc-perf-env/bin/python main.py --headless --verify 180 \
      > /workspace/output/round6/s09/verify/verify_a${v}_r${r}.txt 2>&1 \
      || echo "FAIL verify_a${v}_r${r}"
  done
  echo "verify round $r done"
done
echo VERIFY_DONE

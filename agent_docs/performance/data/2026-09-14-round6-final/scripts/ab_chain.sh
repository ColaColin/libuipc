#!/bin/bash
# Round-6 V3: the base-vs-head sweep, interleaved (ABBA + one discarded warm-up per arm), one scene after another.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
BASE_PY=/workspace/output/round6/base/venv/bin/python
HEAD_PY=/workspace/deps/uipc-perf-env/bin/python
O=/workspace/output/round6/v3/ab
run () { # scene n
  $UIPC_PERF_PY /workspace/output/round6/v3/ab2.py --scene $1 --n $2 --label v3 --out $O \
     --arm base $BASE_PY --arm head $HEAD_PY > $O/ab_$1.txt 2>&1 || echo "FAILED $1"
  echo "done $1 $(date +%H:%M:%S)"
}
run tumbler-garments 20
run rigid-wrecking-balls 5
run cube-wall-cloth 5
run stiff-gipc-case2 5
run mas-bunny 5
echo CHAIN_DONE

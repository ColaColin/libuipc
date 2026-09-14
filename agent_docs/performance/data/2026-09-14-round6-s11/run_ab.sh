#!/bin/bash
# s11 end-to-end, one build, interleaved (abn.py: rotated arm order, 1 discarded warm-up/arm)
set -u
cd /workspace/output/round6/s11
PY=${UIPC_PERF_PY:-/workspace/deps/uipc-perf-env/bin/python}
mkdir -p ab
run() { local sc=$1 n=$2 lab=$3; shift 3
  $PY abn.py --scene "$sc" --n "$n" --label "$lab" --out ab/$lab --baseline off "$@" > ab/$lab.txt 2>&1 \
     || echo "FAIL $lab"
  echo "--- $lab"; tail -40 ab/$lab.txt; }
run rigid-wrecking-balls 10 rwb \
  --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 \
  --arm before UIPC_ABD_GH_PREPASS=2 --arm fork3 UIPC_ABD_GH_PREPASS=3 \
  --arm fork4 UIPC_ABD_GH_PREPASS=4 --arm null UIPC_ABD_GH_PREPASS=00
echo AB_RWB_DONE

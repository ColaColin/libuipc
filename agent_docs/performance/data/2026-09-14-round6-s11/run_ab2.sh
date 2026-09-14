#!/bin/bash
set -u
cd /workspace/output/round6/s11
PY=${UIPC_PERF_PY:-/workspace/deps/uipc-perf-env/bin/python}
$PY abn.py --scene rigid-wrecking-balls --n 30 --label rwb30 --out ab/rwb30 --baseline off \
  --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 \
  --arm before UIPC_ABD_GH_PREPASS=2 --arm fork4 UIPC_ABD_GH_PREPASS=4 \
  --arm null UIPC_ABD_GH_PREPASS=00 > ab/rwb30.txt 2>&1 || echo FAIL
echo AB_RWB30_DONE

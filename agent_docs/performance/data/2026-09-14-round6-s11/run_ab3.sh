#!/bin/bash
set -u
cd /workspace/output/round6/s11
PY=${UIPC_PERF_PY:-/workspace/deps/uipc-perf-env/bin/python}
$PY abn.py --scene rigid-wrecking-balls --n 20 --label rwb_f3 --out ab/rwb_f3 --baseline off \
  --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 \
  --arm fork3 UIPC_ABD_GH_PREPASS=3 --arm null UIPC_ABD_GH_PREPASS=00 > ab/rwb_f3.txt 2>&1 || echo FAIL
echo AB_F3_DONE

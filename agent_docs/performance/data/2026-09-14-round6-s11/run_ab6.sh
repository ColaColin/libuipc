#!/bin/bash
set -u
cd /workspace/output/round6/s11
PY=${UIPC_PERF_PY:-/workspace/deps/uipc-perf-env/bin/python}
run() { local sc=$1 n=$2 lab=$3; shift 3
  $PY abn.py --scene "$sc" --n "$n" --label "$lab" --out ab/$lab --baseline off "$@" > ab/$lab.txt 2>&1 || echo "FAIL $lab"; }
# confirmation on the SHIPPED binary (default flipped to 4, env parse moved to a shared header).
# `dflt` sets a harmless variable so the arm is recorded in the metadata while leaving
# UIPC_ABD_GH_PREPASS unset -- that arm exercises the actual default path.
run rigid-wrecking-balls 12 rwb_final --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 \
   --arm fork4 UIPC_ABD_GH_PREPASS=4 --arm dflt UIPC_ABD_GH_PREPASS_DEFAULT_ARM=1
echo RWB_FINAL_DONE
run cube-wall-cloth 12 cwc_final --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 \
   --arm fork4 UIPC_ABD_GH_PREPASS=4 --arm dflt UIPC_ABD_GH_PREPASS_DEFAULT_ARM=1
echo CWC_FINAL_DONE

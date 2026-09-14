#!/bin/bash
set -u
cd /workspace/output/round6/s11
PY=${UIPC_PERF_PY:-/workspace/deps/uipc-perf-env/bin/python}
run() { local sc=$1 n=$2 lab=$3; shift 3
  $PY abn.py --scene "$sc" --n "$n" --label "$lab" --out ab/$lab --baseline off "$@" > ab/$lab.txt 2>&1 || echo "FAIL $lab"; }
run cube-wall-cloth 10 cwc --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 \
   --arm before UIPC_ABD_GH_PREPASS=2 --arm fork3 UIPC_ABD_GH_PREPASS=3 --arm fork4 UIPC_ABD_GH_PREPASS=4
echo CWC_DONE
run mas-bunny 6 mb --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 --arm fork4 UIPC_ABD_GH_PREPASS=4
echo MB_DONE
run stiff-gipc-case2 4 c2 --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 --arm fork4 UIPC_ABD_GH_PREPASS=4
echo C2_DONE

#!/bin/bash
set -u
cd /workspace/output/round6/s11
PY=${UIPC_PERF_PY:-/workspace/deps/uipc-perf-env/bin/python}
run() { local sc=$1 n=$2 lab=$3; shift 3
  $PY abn.py --scene "$sc" --n "$n" --label "$lab" --out ab/$lab --baseline off "$@" > ab/$lab.txt 2>&1 || echo "FAIL $lab"; }
run cube-wall-cloth 12 cwc2 --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 \
   --arm fork4 UIPC_ABD_GH_PREPASS=4 --arm null UIPC_ABD_GH_PREPASS=00
echo CWC2_DONE
run tumbler-garments 5 tum --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 --arm fork4 UIPC_ABD_GH_PREPASS=4
echo TUM_DONE
run stiff-gipc-case2 6 c2b --arm off UIPC_ABD_GH_PREPASS=0 --arm fork4 UIPC_ABD_GH_PREPASS=4 --arm null UIPC_ABD_GH_PREPASS=00
echo C2B_DONE

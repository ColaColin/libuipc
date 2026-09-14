#!/bin/bash
# s10 end-to-end, one build, interleaved (abn.py: rotated arm order, one discarded warm-up per arm)
set -u
cd /workspace/output/round6/s10
PY=${UIPC_PERF_PY:-/workspace/deps/uipc-perf-env/bin/python}
mkdir -p ab
run() { # scene n label arms...
  local sc=$1 n=$2 lab=$3; shift 3
  $PY abn.py --scene "$sc" --n "$n" --label "$lab" --out ab/$lab --baseline off "$@" > ab/$lab.txt 2>&1 \
     || echo "FAIL $lab"
  echo "--- $lab"; tail -25 ab/$lab.txt
}
run rigid-wrecking-balls 8 rwb --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 --arm before UIPC_ABD_GH_PREPASS=2 --arm null UIPC_ABD_GH_PREPASS=00
run cube-wall-cloth 8 cwc --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1 --arm before UIPC_ABD_GH_PREPASS=2
run tumbler-garments 5 tum --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1
run mas-bunny 4 mb --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1
run stiff-gipc-case2 4 c2 --arm off UIPC_ABD_GH_PREPASS=0 --arm after UIPC_ABD_GH_PREPASS=1
echo AB_DONE

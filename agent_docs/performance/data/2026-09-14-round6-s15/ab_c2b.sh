#!/bin/bash
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
$UIPC_PERF_PY /workspace/output/round6/ab.py --scene stiff-gipc-case2 --n 8 --label s15_c2b --out /workspace/output/round6/s15/ab/c2b --arm old UIPC_MAS_ROWDOT2=0 --arm u16 UIPC_MAS_ROWDOT2=16 > /workspace/output/round6/s15/ab_c2b.txt 2>&1
echo DONE > /workspace/output/round6/s15/ab_c2b.done

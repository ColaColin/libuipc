#!/bin/bash
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
P=$UIPC_PERF_PY
$P /workspace/output/round6/ab.py --scene mas-bunny --n 10 --label s15_mb --out /workspace/output/round6/s15/ab/mb --arm old UIPC_MAS_ROWDOT2=0 --arm u16 UIPC_MAS_ROWDOT2=16 > /workspace/output/round6/s15/ab_mb.txt 2>&1
$P /workspace/output/round6/ab.py --scene stiff-gipc-case2 --n 8 --label s15_c2 --out /workspace/output/round6/s15/ab/c2 --arm old UIPC_MAS_ROWDOT2=0 --arm u16 UIPC_MAS_ROWDOT2=16 > /workspace/output/round6/s15/ab_c2.txt 2>&1
$P /workspace/output/round6/ab.py --scene cube-wall-cloth --n 6 --label s15_cwc --out /workspace/output/round6/s15/ab/cwc --arm old UIPC_MAS_ROWDOT2=0 --arm u16 UIPC_MAS_ROWDOT2=16 > /workspace/output/round6/s15/ab_cwc.txt 2>&1
echo CHAIN-DONE > /workspace/output/round6/s15/ab_chain.done

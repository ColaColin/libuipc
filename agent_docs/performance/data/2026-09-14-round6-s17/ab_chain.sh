#!/bin/bash
# s17 A/B chain: UIPC_SEG_REDUCE2=0 (cub tree, main's kernel) vs =1 (level-skipping tree). ab.py: ABBA, one discarded warm-up per arm.
TAG=${1:-s17}
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
P=$UIPC_PERF_PY
O=/workspace/output/round6/s17
$P /workspace/output/round6/ab.py --scene mas-bunny --n 10 --label ${TAG}_mb --out $O/ab/mb --arm old UIPC_SEG_REDUCE2=0 --arm new UIPC_SEG_REDUCE2=1 > $O/ab_mb.txt 2>&1
$P /workspace/output/round6/ab.py --scene stiff-gipc-case2 --n 8 --label ${TAG}_c2 --out $O/ab/c2 --arm old UIPC_SEG_REDUCE2=0 --arm new UIPC_SEG_REDUCE2=1 > $O/ab_c2.txt 2>&1
$P /workspace/output/round6/ab.py --scene cube-wall-cloth --n 6 --label ${TAG}_cwc --out $O/ab/cwc --arm old UIPC_SEG_REDUCE2=0 --arm new UIPC_SEG_REDUCE2=1 > $O/ab_cwc.txt 2>&1
$P /workspace/output/round6/ab.py --scene rigid-wrecking-balls --n 6 --label ${TAG}_rwb --out $O/ab/rwb --arm old UIPC_SEG_REDUCE2=0 --arm new UIPC_SEG_REDUCE2=1 > $O/ab_rwb.txt 2>&1
echo CHAIN-DONE > $O/ab_chain.done

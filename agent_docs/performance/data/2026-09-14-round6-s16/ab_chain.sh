#!/bin/bash
# s16 A/B chain: UIPC_MAS_FUSED_R=0 (old two-kernel path) vs =$MODE (fused). ab.py: ABBA, one discarded warm-up per arm.
MODE=${1:?mode}; TAG=${2:-s16}
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
P=$UIPC_PERF_PY
O=/workspace/output/round6/s16
$P /workspace/output/round6/ab.py --scene mas-bunny --n 10 --label ${TAG}_mb --out $O/ab/mb --arm old UIPC_MAS_FUSED_R=0 --arm fused UIPC_MAS_FUSED_R=$MODE > $O/ab_mb.txt 2>&1
$P /workspace/output/round6/ab.py --scene stiff-gipc-case2 --n 8 --label ${TAG}_c2 --out $O/ab/c2 --arm old UIPC_MAS_FUSED_R=0 --arm fused UIPC_MAS_FUSED_R=$MODE > $O/ab_c2.txt 2>&1
$P /workspace/output/round6/ab.py --scene cube-wall-cloth --n 6 --label ${TAG}_cwc --out $O/ab/cwc --arm old UIPC_MAS_FUSED_R=0 --arm fused UIPC_MAS_FUSED_R=$MODE > $O/ab_cwc.txt 2>&1
echo CHAIN-DONE > $O/ab_chain.done

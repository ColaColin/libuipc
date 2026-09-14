#!/bin/bash
# s13 second sweep: the two round-5 switches that shipped default-off with an
# untested prediction attached, re-measured at this head against the fitted grid.
set -u
cd /workspace/deps/libuipc-src
source /workspace/deps/libuipc-src/env_perf.sh
AB="$UIPC_PERF_PY /workspace/output/round6/ab.py"
O=/workspace/output/round6/s13
run() { echo "##### $*"; $AB "$@" 2>&1; echo; }

# R6: the persistent grid-stride rewrite, against the capacity grid it was
# rejected against (FIT off in both arms so this is one variable).
run --scene mas-bunny --n 8 --label s13r6mb --out $O/ab \
    --arm cap UIPC_SPMV_GRID_FIT=0 UIPC_SPMV_GRID_STRIDE=0 \
    --arm stride UIPC_SPMV_GRID_FIT=0 UIPC_SPMV_GRID_STRIDE=1
# R7: the PCG scalar fold, ungated (MAXGRID=0), against the shipped chain.
run --scene mas-bunny --n 8 --label s13r7mb --out $O/ab \
    --arm nofold UIPC_PCG_FOLD=0 --arm fold1 UIPC_PCG_FOLD=1 UIPC_PCG_FOLD_MAXGRID=0

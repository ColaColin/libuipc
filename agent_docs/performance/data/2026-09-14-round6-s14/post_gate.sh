#!/bin/bash
# s14: after the gate -- (1) tumbler --verify n=6/arm, (2) rwb traces off/m1 (mechanism: what
# lands in part 1's shadow on an ABD scene), (3) rwb re-measured at n=16 (the n=8 read -1.54 %
# with the count guard firing), (4) UIPC_BCOO_HASH first-conversion hash across arms on mas-bunny.
set -u
S=/workspace/output/round6/s14
source /workspace/deps/libuipc-src/env_perf.sh
bash $S/run_verify_tumbler.sh 6 1 > $S/tverify.log 2>&1; echo "TVERIFY rc=$?"
cd $S && ./tracerun.sh rwb_off_full 6_wrecking_balls 120 UIPC_CONTACT_DEFERRED_JOIN=0 && ./tracerun.sh rwb_m1_full 6_wrecking_balls 120 UIPC_CONTACT_DEFERRED_JOIN=1; echo "RWB_TRACES rc=$?"
cd /workspace/deps/libuipc-src
$UIPC_PERF_PY $S/../ab.py --scene rigid-wrecking-balls --n 16 --label s14rwb16 --out $S/ab \
  --arm off UIPC_CONTACT_DEFERRED_JOIN=0 --arm on UIPC_CONTACT_DEFERRED_JOIN=1 > $S/ab_rwb16.log 2>&1; echo "RWB16 rc=$?"
export PYTHONPATH=/workspace/archive/libuipc-samples/shim WB_LOG=Warn
cd /workspace/deps/libuipc-src/libuipc-samples/examples/89_mas_bunny
for v in 0 1 2; do UIPC_BCOO_HASH=3 UIPC_CONTACT_DEFERRED_JOIN=$v $UIPC_PERF_PY main.py --headless 3 2>&1 | grep 'bcoo-hash' | sed "s/^/mode=$v /"; done > $S/bcoo_hash_mb.txt
cd /workspace/deps/libuipc-src/libuipc-samples/examples/88_stiff_gipc_benchmark
for v in 0 1 2; do UIPC_BCOO_HASH=3 UIPC_CONTACT_DEFERRED_JOIN=$v $UIPC_PERF_PY main.py --headless 3 2>&1 | grep 'bcoo-hash' | sed "s/^/mode=$v /"; done > $S/bcoo_hash_c2.txt
echo POST_GATE_DONE

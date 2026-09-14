#!/bin/bash
# s14: the untargeted scenes. mode = the candidate default passed as $1.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
S=/workspace/output/round6/s14
M=${1:-1}
$UIPC_PERF_PY $S/../ab.py --scene mas-bunny --n 8 --label s14mb --out $S/ab \
  --arm off UIPC_CONTACT_DEFERRED_JOIN=0 --arm on UIPC_CONTACT_DEFERRED_JOIN=$M > $S/ab_mb.log 2>&1
echo "MB_DONE rc=$?"
$UIPC_PERF_PY $S/../ab.py --scene cube-wall-cloth --n 8 --label s14cwc --out $S/ab \
  --arm off UIPC_CONTACT_DEFERRED_JOIN=0 --arm on UIPC_CONTACT_DEFERRED_JOIN=$M > $S/ab_cwc.log 2>&1
echo "CWC_DONE rc=$?"
$UIPC_PERF_PY $S/../ab.py --scene rigid-wrecking-balls --n 8 --label s14rwb --out $S/ab \
  --arm off UIPC_CONTACT_DEFERRED_JOIN=0 --arm on UIPC_CONTACT_DEFERRED_JOIN=$M > $S/ab_rwb.log 2>&1
echo "RWB_DONE rc=$?"
$UIPC_PERF_PY $S/../ab.py --scene tumbler-garments --n 5 --label s14tum --out $S/ab \
  --arm off UIPC_CONTACT_DEFERRED_JOIN=0 --arm on UIPC_CONTACT_DEFERRED_JOIN=$M > $S/ab_tum.log 2>&1
echo "TUM_DONE rc=$?"

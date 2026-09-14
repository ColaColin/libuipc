#!/bin/bash
# s14: the end-to-end verdict on stiff-gipc-case2 -- four arms interleaved in one build
# (s11's abn.py: rotated + reversed order, one discarded warm-up per arm):
# off (=0, today's join), mode 1, mode 2, and a bit-identical null (=00 -> atoi 0).
set -u
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/output/round6/s14
N=${1:-12}
$UIPC_PERF_PY abn.py --scene stiff-gipc-case2 --n $N --label s14c2 --out ab/s14c2 --baseline off \
  --arm off UIPC_CONTACT_DEFERRED_JOIN=0 --arm m1 UIPC_CONTACT_DEFERRED_JOIN=1 \
  --arm m2 UIPC_CONTACT_DEFERRED_JOIN=2 --arm null UIPC_CONTACT_DEFERRED_JOIN=00 > ab/s14c2.txt 2>&1
echo "C2_DONE rc=$?"

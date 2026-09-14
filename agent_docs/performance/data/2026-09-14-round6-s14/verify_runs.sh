#!/bin/bash
# s14: UIPC_CONTACT_DEFERRED_JOIN_VERIFY=1 -- device-side word comparison after the deferred
# join, plus the "would-have-been-stale" measurement. case2 (fast path, modes 1 and 2), cwc
# (multi-receiver fallback: the join must be taken by the manager, pending_at_join == 0),
# rwb (ABD fast path: read in report_extent) and the tumbler (fallback).
set -u
TREE=/workspace/deps/libuipc-src
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=$TREE/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn
PY=/workspace/deps/uipc-perf-env/bin/python
O=/workspace/output/round6/s14/verify; mkdir -p $O
run() { # tag dir frames mode
  ( cd $TREE/libuipc-samples/examples/$2 && UIPC_CONTACT_DEFERRED_JOIN=$4 UIPC_CONTACT_DEFERRED_JOIN_VERIFY=1 \
      $PY main.py --headless $3 > $O/$1.log 2>&1; echo "$1 rc=$?" )
  grep -h 'deferred-join-verify' $O/$1.log | tail -1
}
run c2_m1  88_stiff_gipc_benchmark 40 1
run c2_m2  88_stiff_gipc_benchmark 40 2
run cwc_m1 93_cube_wall_cloth      40 1
run rwb_m1 6_wrecking_balls        40 1
run mb_m1  89_mas_bunny            40 1
run tum_m2 95_tumbler_garments     30 2
echo VERIFY_RUNS_DONE

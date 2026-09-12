#!/bin/bash
V=/workspace/output/round5/validation
P=$V/probes; mkdir -p $P
L=$V/audit.log
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0
unset WB_TIMER NO_MAS NO_GRAPH
export UIPC_GRID_SPREAD_VERIFY=1 UIPC_BUFFER_FILL_VERIFY=1 UIPC_BVH_BATCH_COUNTS_VERIFY=1 \
  UIPC_BVH_REFIT_VERIFY=1 UIPC_BVH_SELF_CULL_VERIFY=1 UIPC_BVH_TWO_PHASE_VERIFY=1 \
  UIPC_CONVERT_VERIFY=1 UIPC_HOST_SYNC_VERIFY=1 UIPC_MAS_APPLY_VERIFY=1 \
  UIPC_MAS_INVERT_VERIFY=1 UIPC_MAS_R_TAIL_VERIFY=1 UIPC_MAS_SCATTER_VERIFY=1 \
  UIPC_ABD_DIAG_APPLY_VERIFY=1 UIPC_PCG_AP_ZERO_VERIFY=1 UIPC_PCG_FUSE_DOT_VERIFY=1 \
  UIPC_SPMV_VERIFY=1 UIPC_SPMV_PROBE=1 UIPC_SEG_FILL_POISON=2 \
  UIPC_BCOO_HASH=1 UIPC_FILL_PROBE=1 UIPC_TRIPLET_PATTERN_PROBE=1 UIPC_D2H_PROFILE=1
echo "=== PHASE H: composed probe run, full stderr ($(date -u +%H:%M:%S))" >> $L
for s in 6_wrecking_balls:wb 93_cube_wall_cloth:cwc 88_stiff_gipc_benchmark:c2 89_mas_bunny:mb; do
  d=${s%%:*}; n=${s##*:}
  ( cd /workspace/deps/libuipc-src/libuipc-samples/examples/$d && \
    timeout 3000 /workspace/deps/uipc-perf-env/bin/python main.py --headless 12 ) \
    > $P/$n.out 2> $P/$n.err
  echo "probe $n rc=$? stderr_lines=$(wc -l < $P/$n.err)" >> $L
done
# same again with R7's fold enabled, so PCG_FOLD_VERIFY has something to verify
export UIPC_PCG_FOLD=1 UIPC_PCG_FOLD_VERIFY=1
( cd /workspace/deps/libuipc-src/libuipc-samples/examples/6_wrecking_balls && \
  timeout 3000 /workspace/deps/uipc-perf-env/bin/python main.py --headless 12 ) \
  > $P/wb_fold.out 2> $P/wb_fold.err
echo "probe wb_fold rc=$? stderr_lines=$(wc -l < $P/wb_fold.err)" >> $L
echo "=== PHASE H DONE ($(date -u +%H:%M:%S))" >> $L

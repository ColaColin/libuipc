#!/bin/bash
# Round-6 V3 env-switch audit (PERF_METHOD §5): kernel-level evidence, not wall time.
# For every switch the round added, profile a short run with the switch at its default
# and at its rollback value, and read the *kernels actually launched* (nsys kern_sum:
# demangled template arguments and instance counts) or, where the switch changes launch
# geometry / stream placement rather than the kernel, the sqlite trace (gridX, streamId,
# start/end). Fresh prefix per run; fails loudly if no csv appears (brief rule 8).
set -u
source /workspace/deps/libuipc-src/env_perf.sh
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
OUT=/workspace/output/round6/v3/envaudit
REPO=/workspace/deps/libuipc-src
mkdir -p "$OUT"

prof () {  # $1 tag  $2 scene-dir  $3 frames  rest: env assignments
  local tag=$1 dir=$2 frames=$3; shift 3
  local p="$OUT/$tag"
  rm -f "$p".nsys-rep "$p".sqlite "$p"_cuda_gpu_kern_sum.csv
  ( cd "$REPO/libuipc-samples/examples/$dir" && \
    env WB_LOG=Warn UIPC_BENCHMARK_TIMERS=0 "$@" $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$p" \
      "$UIPC_PERF_PY" main.py --headless "$frames" > "$OUT/$tag.stdout" 2>&1 )
  $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      -o "$p" "$p.nsys-rep" > "$OUT/$tag.stats.log" 2>&1
  if [ ! -s "$p"_cuda_gpu_kern_sum.csv ]; then echo "FAIL: no kern_sum csv for $tag"; return 1; fi
  if [ ! -s "$p".sqlite ]; then echo "FAIL: no sqlite for $tag"; return 1; fi
  echo "ok $tag ($(wc -l < "$p"_cuda_gpu_kern_sum.csv) kernel rows) env: $*"
}

T=95_tumbler_garments; R=6_wrecking_balls; M=89_mas_bunny; C=88_stiff_gipc_benchmark

# --- tumbler: hinge, contact part 1 / part 2 projections, CCD, segmented reduce -----------
prof t_default $T 12
prof t_gn0     $T 12 UIPC_DSB_GAUSS_NEWTON=0
prof t_rank1_0 $T 12 UIPC_CONTACT_RANK1=0
prof t_spd1b0  $T 12 UIPC_CONTACT_SPD1_BASIS=0
prof t_ccdeo0  $T 12 UIPC_CCD_EARLY_OUT=0
prof t_ccdcp0  $T 12 UIPC_CCD_COMPACT=0
prof t_seg0    $T 12 UIPC_SEG_REDUCE2=0

# --- rwb: the K9 split (prepass pinned to 1 in every arm, per s11) and the prepass placement
prof r_default    $R 12
prof r_pre1       $R 12 UIPC_ABD_GH_PREPASS=1
prof r_pre0       $R 12 UIPC_ABD_GH_PREPASS=0
prof r_split0_pre1 $R 12 UIPC_CONTACT_SPLIT=0 UIPC_ABD_GH_PREPASS=1
prof r_split1_pre1 $R 12 UIPC_CONTACT_SPLIT=1 UIPC_ABD_GH_PREPASS=1

# --- mas-bunny: SpMV grid fit, MAS row-dot, MAS fused restriction ------------------------
prof m_default $M 12
prof m_fit0    $M 12 UIPC_SPMV_GRID_FIT=0
prof m_rd0     $M 12 UIPC_MAS_ROWDOT2=0
prof m_fr0     $M 12 UIPC_MAS_FUSED_R=0

# --- case2: the deferred contact join --------------------------------------------------
prof c_default $C 12
prof c_dj0     $C 12 UIPC_CONTACT_DEFERRED_JOIN=0

echo AUDIT_DONE

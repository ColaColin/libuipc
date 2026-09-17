#!/bin/bash
# Round-7 s20 env-switch audit (PERF_METHOD §5): kernel-level evidence that every
# knob this round introduced or touched still selects its OLD path at the head
# (main 69c2af51). For each knob: a short crease-press nsys capture with the knob
# at its rollback value, then read the kernels actually launched (kern_sum:
# demangled template arguments / instance counts). Fresh prefix per run; fails
# loudly if no csv appears (brief rule 8). Round-6 V3's envaudit pattern.
#
# PCG_POLL is the one switch that changes host/API behaviour rather than kernel
# identity: its evidence is the cuda-api trace (8-byte D2H memcpyAsync count at
# the solver site, s17's instrument), read from the same captures' sqlite.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
OUT=/workspace/output/round7/s20/envaudit
REPO=/workspace/deps/libuipc-src
mkdir -p "$OUT"

prof () {  # $1 tag  $2 frames  rest: env assignments
  local tag=$1 frames=$2; shift 2
  local p="$OUT/$tag"
  rm -f "$p".nsys-rep "$p".sqlite "$p"_cuda_gpu_kern_sum.csv "$p"_cuda_api_sum.csv
  ( cd "$REPO/libuipc-samples/examples/103_crease_press" && \
    env WB_LOG=Error "$@" $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$p" \
      "$UIPC_PERF_PY" main.py --headless "$frames" > "$OUT/$tag.stdout" 2>&1 )
  $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      -o "$p" "$p.nsys-rep" > "$OUT/$tag.stats.log" 2>&1
  if [ ! -s "$p"_cuda_gpu_kern_sum.csv ]; then echo "FAIL: no kern_sum csv for $tag"; return 1; fi
  if [ ! -s "$p".sqlite ]; then echo "FAIL: no sqlite for $tag"; return 1; fi
  echo "ok $tag ($(wc -l < "$p"_cuda_gpu_kern_sum.csv) kernel rows) env: $*"
}

F=16

# --- the default reference arm -----------------------------------------------------------
prof k_default $F

# --- dahl: GAUSS_NEWTON off restores the exact Hessian + s02 knob tree ------------------
prof k_dahl_gn0     $F UIPC_DAHL_GAUSS_NEWTON=0
prof k_dahl_rspd0   $F UIPC_DAHL_GAUSS_NEWTON=0 UIPC_DAHL_REDUCED_SPD=0
prof k_dahl_blk0    $F UIPC_DAHL_GAUSS_NEWTON=0 UIPC_DAHL_BLOCKED_PROJ=0
prof k_dahl_tql0    $F UIPC_DAHL_GAUSS_NEWTON=0 UIPC_DAHL_TQL2=0

# --- the two plastic kernels share the PDSB knob family ----------------------------------
prof k_pdsb_rspd0   $F UIPC_PDSB_REDUCED_SPD=0
prof k_pdsb_blk0    $F UIPC_PDSB_BLOCKED_PROJ=0
prof k_pdsb_tql0    $F UIPC_PDSB_TQL2=0

# --- NeoHookeanShell2D membrane ----------------------------------------------------------
prof k_nhs2d_rspd0  $F UIPC_NHS2D_REDUCED_SPD=0
prof k_nhs2d_blk0   $F UIPC_NHS2D_BLOCKED_PROJ=0
prof k_nhs2d_tql0   $F UIPC_NHS2D_TQL2=0

# --- the s08 half-assembly (helper-level, all four 4x3 families) -------------------------
prof k_half0        $F UIPC_MAKE_SPD_BLOCKED_HALF=0

# --- converter: the s13 fold and the s15 K-serial tree ------------------------------------
prof k_segred_uns0  $F UIPC_SEGRED_UNSTAGE=0
prof k_segred_tree0 $F UIPC_SEGRED_TREE=0

# --- the s14 doublet fold ------------------------------------------------------------------
prof k_doublet_uns0 $F UIPC_DOUBLET_UNSTAGE=0

# --- the s17 doorbell (host-path switch; api-trace evidence) -------------------------------
prof k_pcg_poll0    12 UIPC_PCG_POLL=0
prof k_pcg_polldef  12

echo AUDIT_DONE

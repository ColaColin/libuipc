#!/bin/bash
V=/workspace/output/round5/validation
PY=/workspace/deps/uipc-perf-env/bin/python
L=$V/audit.log
fp() { timeout 5400 $PY $V/fp.py "$@" >> $L 2>&1; }
echo "=== PHASE B: switch audit, 60 frames, n=2 ($(date -u +%H:%M:%S))" >> $L
# reference arms (n=5) at the phase-B frame count
fp wb  --tree head --frames 60 --reps 5 --tag Bref
fp c2  --tree head --frames 60 --reps 5 --tag Bref
fp cwc --tree head --frames 60 --reps 3 --tag Bref
fp mb  --tree head --frames 60 --reps 3 --tag Bref
# --- wrecking-balls arms (ABD + contact) ---
for a in "GRID_SPREAD=0" "GRID_SPREAD_BLOCK=32" "GRID_SPREAD_BPSM=1" "GRID_SPREAD_ONLY=ZZNOSUCHTAG" \
         "HOST_SYNC_FAST=0" "HOST_SYNC_SPIN=1" "BVH_BATCH_COUNTS=0" "CONTACT_SPD_TQL=0" "CONTACT_SPD2=0" \
         "PCG_FOLD=1" "PCG_FOLD=2" "PCG_DOT_FENCE=1" "MAKE_SPD_JACOBI=0" "BUFFER_FILL_SPREAD=0"; do
  fp wb --tree head --frames 60 --reps 2 --tag "B_${a}" --env "UIPC_${a}"
done
# --- case2 arms (FEM/SNH/SpMV/segmental) ---
for a in "QR_SVD_FIXED=0" "SNK1_STENCIL2=0" "SNK1_OCC=0" "SKIP_DEAD_FILL=0" "SEG_NARROW_FILL=0" \
         "BUFFER_FILL_SPREAD=0" "BUFFER_FILL_BPSM=8" "GRID_SPREAD=0" "SPMV_GRID_STRIDE=1"; do
  fp c2 --tree head --frames 60 --reps 2 --tag "B_${a}" --env "UIPC_${a}"
done
fp c2 --tree head --frames 60 --reps 2 --tag "B_SPMV_GRID_STRIDE=1+WAVES=4" --env UIPC_SPMV_GRID_STRIDE=1 --env UIPC_SPMV_GRID_WAVES=4
# hinge lives in cube-wall-cloth
fp cwc --tree head --frames 60 --reps 2 --tag "B_MAKE_SPD_JACOBI=0" --env UIPC_MAKE_SPD_JACOBI=0
fp cwc --tree head --frames 60 --reps 2 --tag "B_GRID_SPREAD=0" --env UIPC_GRID_SPREAD=0
fp mb  --tree head --frames 60 --reps 2 --tag "B_QR_SVD_FIXED=0" --env UIPC_QR_SVD_FIXED=0
fp mb  --tree head --frames 60 --reps 2 --tag "B_SKIP_DEAD_FILL=0" --env UIPC_SKIP_DEAD_FILL=0
echo "=== PHASE B DONE ($(date -u +%H:%M:%S))" >> $L
# --- PHASE C: every step switch off at once, vs base ---
ALLOFF="--env UIPC_MAKE_SPD_JACOBI=0 --env UIPC_GRID_SPREAD=0 --env UIPC_BUFFER_FILL_SPREAD=0 \
--env UIPC_HOST_SYNC_FAST=0 --env UIPC_BVH_BATCH_COUNTS=0 --env UIPC_QR_SVD_FIXED=0 \
--env UIPC_CONTACT_SPD_TQL=0 --env UIPC_CONTACT_SPD2=0 --env UIPC_SKIP_DEAD_FILL=0 \
--env UIPC_SNK1_STENCIL2=0 --env UIPC_SNK1_OCC=0 --env UIPC_SEG_NARROW_FILL=0"
echo "=== PHASE C: all step switches off ($(date -u +%H:%M:%S))" >> $L
for s in wb cwc c2 mb; do
  case $s in wb) f=120;; cwc) f=120;; c2) f=250;; mb) f=100;; esac
  fp $s --tree head --frames $f --reps 3 --tag C_alloff $ALLOFF
done
echo "=== PHASE C DONE ($(date -u +%H:%M:%S))" >> $L
# --- PHASE D: all verification probes on, composed ---
PROBES="--env UIPC_GRID_SPREAD_VERIFY=1 --env UIPC_BUFFER_FILL_VERIFY=1 --env UIPC_BVH_BATCH_COUNTS_VERIFY=1 \
--env UIPC_BVH_REFIT_VERIFY=1 --env UIPC_BVH_SELF_CULL_VERIFY=1 --env UIPC_BVH_TWO_PHASE_VERIFY=1 \
--env UIPC_CONVERT_VERIFY=1 --env UIPC_HOST_SYNC_VERIFY=1 --env UIPC_MAS_APPLY_VERIFY=1 \
--env UIPC_MAS_INVERT_VERIFY=1 --env UIPC_MAS_R_TAIL_VERIFY=1 --env UIPC_MAS_SCATTER_VERIFY=1 \
--env UIPC_ABD_DIAG_APPLY_VERIFY=1 --env UIPC_PCG_AP_ZERO_VERIFY=1 --env UIPC_PCG_FUSE_DOT_VERIFY=1 \
--env UIPC_SPMV_VERIFY=1 --env UIPC_SPMV_PROBE=1 --env UIPC_SEG_FILL_POISON=2 \
--env UIPC_BCOO_HASH=1 --env UIPC_FILL_PROBE=1 --env UIPC_TRIPLET_PATTERN_PROBE=1 --env UIPC_D2H_PROFILE=1"
echo "=== PHASE D: all probes composed ($(date -u +%H:%M:%S))" >> $L
for s in wb cwc c2 mb; do
  fp $s --tree head --frames 12 --reps 1 --tag D_probes_$s --show-stderr $PROBES
done
echo "=== PHASE D DONE ($(date -u +%H:%M:%S))" >> $L

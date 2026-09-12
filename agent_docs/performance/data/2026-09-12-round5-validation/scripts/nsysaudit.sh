#!/bin/bash
V=/workspace/output/round5/validation
N=$V/nsys; mkdir -p $N
L=$V/audit.log
HT=/workspace/deps/libuipc-src;  HP=/workspace/deps/uipc-perf-env/bin/python
BT=/workspace/deps/libuipc-valbase; BP=/workspace/deps/uipc-valbase-env/bin/python
run() { # name tree py dir frames env...
  local nm=$1 tr=$2 py=$3 d=$4 f=$5; shift 5
  bash $V/nsysrun.sh $N/$nm $tr $py $d $f "$@" >> $L 2>&1
}
echo "=== PHASE G: kernel-level switch audit (nsys) ($(date -u +%H:%M:%S))" >> $L
# wrecking balls, 15 frames
run wb_head   $HT $HP 6_wrecking_balls 15
run wb_base   $BT $BP 6_wrecking_balls 15
run wb_spd2off   $HT $HP 6_wrecking_balls 15 UIPC_CONTACT_SPD2=0
run wb_tqloff    $HT $HP 6_wrecking_balls 15 UIPC_CONTACT_SPD_TQL=0
run wb_gsoff     $HT $HP 6_wrecking_balls 15 UIPC_GRID_SPREAD=0
run wb_gsonly    $HT $HP 6_wrecking_balls 15 UIPC_GRID_SPREAD_ONLY=ZZNOSUCHTAG
run wb_gsblock32 $HT $HP 6_wrecking_balls 15 UIPC_GRID_SPREAD_BLOCK=32
run wb_gsbpsm1   $HT $HP 6_wrecking_balls 15 UIPC_GRID_SPREAD_BPSM=1
run wb_bvhoff    $HT $HP 6_wrecking_balls 15 UIPC_BVH_BATCH_COUNTS=0
run wb_hsfastoff $HT $HP 6_wrecking_balls 15 UIPC_HOST_SYNC_FAST=0
run wb_hsspin    $HT $HP 6_wrecking_balls 15 UIPC_HOST_SYNC_SPIN=1
run wb_fold1     $HT $HP 6_wrecking_balls 15 UIPC_PCG_FOLD=1
run wb_fold2     $HT $HP 6_wrecking_balls 15 UIPC_PCG_FOLD=2
run wb_mspdoff   $HT $HP 6_wrecking_balls 15 UIPC_MAKE_SPD_JACOBI=0
run wb_bfsoff    $HT $HP 6_wrecking_balls 15 UIPC_BUFFER_FILL_SPREAD=0
# case2, 12 frames
run c2_head    $HT $HP 88_stiff_gipc_benchmark 12
run c2_base    $BT $BP 88_stiff_gipc_benchmark 12
run c2_svdoff  $HT $HP 88_stiff_gipc_benchmark 12 UIPC_QR_SVD_FIXED=0
run c2_st2off  $HT $HP 88_stiff_gipc_benchmark 12 UIPC_SNK1_STENCIL2=0
run c2_occoff  $HT $HP 88_stiff_gipc_benchmark 12 UIPC_SNK1_OCC=0
run c2_dfoff   $HT $HP 88_stiff_gipc_benchmark 12 UIPC_SKIP_DEAD_FILL=0
run c2_segoff  $HT $HP 88_stiff_gipc_benchmark 12 UIPC_SEG_NARROW_FILL=0
run c2_bfsoff  $HT $HP 88_stiff_gipc_benchmark 12 UIPC_BUFFER_FILL_SPREAD=0
run c2_bfbpsm8 $HT $HP 88_stiff_gipc_benchmark 12 UIPC_BUFFER_FILL_BPSM=8
run c2_gsoff   $HT $HP 88_stiff_gipc_benchmark 12 UIPC_GRID_SPREAD=0
run c2_spmvgs  $HT $HP 88_stiff_gipc_benchmark 12 UIPC_SPMV_GRID_STRIDE=1
# cube-wall (hinge)
run cwc_head    $HT $HP 93_cube_wall_cloth 12
run cwc_mspdoff $HT $HP 93_cube_wall_cloth 12 UIPC_MAKE_SPD_JACOBI=0
echo "=== PHASE G DONE ($(date -u +%H:%M:%S))" >> $L

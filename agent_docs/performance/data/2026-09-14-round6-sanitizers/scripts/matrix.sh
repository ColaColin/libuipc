#!/bin/bash
source /workspace/output/round6/sanitizers/san.sh
HEADREPO=/workspace/deps/libuipc-src
BASEREPO=/workspace/output/round6/base/wt
SCENES="6_wrecking_balls 93_cube_wall_cloth 88_stiff_gipc_benchmark 89_mas_bunny 95_tumbler_garments"
CONTACT="6_wrecking_balls 93_cube_wall_cloth 95_tumbler_garments"
MF=12; IF=8; RF=6; SF=12
RB="UIPC_DSB_GAUSS_NEWTON=0 UIPC_CONTACT_RANK1=0 UIPC_CCD_EARLY_OUT=0 UIPC_CCD_COMPACT=0 UIPC_CONTACT_SPD1_BASIS=0 UIPC_ABD_GH_PREPASS=0 UIPC_SPMV_GRID_FIT=0 UIPC_CONTACT_DEFERRED_JOIN=0 UIPC_MAS_ROWDOT2=0 UIPC_MAS_FUSED_R=0 UIPC_SEG_REDUCE2=0"

phase () { echo "### $1  $(date '+%F %T')" | tee -a $OUT/driver.log; }
all_tools_scenes () { # arm py [env]
  local arm=$1 py=$2; shift 2
  for d in $SCENES; do
    san_scene $arm $py memcheck  $d $MF "$@"
    san_scene $arm $py initcheck $d $IF "$@"
    san_scene $arm $py racecheck $d $RF "$@"
    san_scene $arm $py synccheck $d $SF "$@"
  done
}
phase "A head default (main 67062f5f, build-perf)"
all_tools_scenes head $HEADPY
phase "B base default (perf-round6-base 60af65e6, build-base)"
all_tools_scenes base $BASEPY
phase "C head all-switches-rolled-back: $RB"
all_tools_scenes rb $HEADPY $RB
phase "E tests: head default"
for tb in backend_cuda sim_case; do for t in memcheck initcheck racecheck synccheck; do
  san_test head $HEADREPO $HEADREPO/build-perf $t $tb; done; done
phase "E tests: base"
for tb in backend_cuda sim_case; do for t in memcheck initcheck racecheck synccheck; do
  san_test base $BASEREPO $BASEREPO/build-base $t $tb; done; done
phase "E tests: head rolled back (memcheck, racecheck)"
for tb in backend_cuda sim_case; do for t in memcheck racecheck; do
  san_test rb $HEADREPO $HEADREPO/build-perf $t $tb $RB; done; done
phase "D concurrency variants on the contact scenes (memcheck $MF f, racecheck $RF f, synccheck $SF f)"
for v in "pp0 UIPC_ABD_GH_PREPASS=0" "pp1 UIPC_ABD_GH_PREPASS=1" "pp2 UIPC_ABD_GH_PREPASS=2" "pp3 UIPC_ABD_GH_PREPASS=3" \
         "dj0 UIPC_CONTACT_DEFERRED_JOIN=0" "dj2 UIPC_CONTACT_DEFERRED_JOIN=2" "sp0 UIPC_CONTACT_SPLIT=0" "sp1 UIPC_CONTACT_SPLIT=1"; do
  set -- $v; arm=$1; shift
  scenes="$CONTACT"; case $arm in dj*|sp*) scenes="$CONTACT 88_stiff_gipc_benchmark";; esac
  for d in $scenes; do
    san_scene $arm $HEADPY memcheck  $d $MF "$@"
    san_scene $arm $HEADPY racecheck $d $RF "$@"
    san_scene $arm $HEADPY synccheck $d $SF "$@"
  done
done
phase "D2 MAS / reduce variants on the FEM scenes (racecheck $RF f, memcheck $MF f)"
for v in "fr0 UIPC_MAS_FUSED_R=0" "fr1 UIPC_MAS_FUSED_R=1" "rd0 UIPC_MAS_ROWDOT2=0" "sr0 UIPC_SEG_REDUCE2=0"; do
  set -- $v; arm=$1; shift
  for d in 89_mas_bunny 88_stiff_gipc_benchmark; do
    san_scene $arm $HEADPY racecheck $d $RF "$@"
    san_scene $arm $HEADPY memcheck  $d $MF "$@"
  done
done
phase "F long tumbler windows into the dense-pile regime"
san_scene head $HEADPY racecheck 95_tumbler_garments 24
san_scene base $BASEPY racecheck 95_tumbler_garments 24
san_scene head $HEADPY memcheck  95_tumbler_garments 30
san_scene base $BASEPY memcheck  95_tumbler_garments 30
san_scene head $HEADPY initcheck 95_tumbler_garments 24
san_scene base $BASEPY initcheck 95_tumbler_garments 24
phase "DONE"
touch $OUT/DONE

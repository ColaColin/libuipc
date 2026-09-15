#!/bin/bash
# Replacement for matrix.sh's tail: sim_case trimmed (racecheck on a subset), then D, D2, F, G.
source /workspace/output/round6/sanitizers/san.sh
HEADREPO=/workspace/deps/libuipc-src
BASEREPO=/workspace/output/round6/base/wt
CONTACT="6_wrecking_balls 93_cube_wall_cloth 95_tumbler_garments"
MF=12; IF=8; RF=6; SF=12
RB="UIPC_DSB_GAUSS_NEWTON=0 UIPC_CONTACT_RANK1=0 UIPC_CCD_EARLY_OUT=0 UIPC_CCD_COMPACT=0 UIPC_CONTACT_SPD1_BASIS=0 UIPC_ABD_GH_PREPASS=0 UIPC_SPMV_GRID_FIT=0 UIPC_CONTACT_DEFERRED_JOIN=0 UIPC_MAS_ROWDOT2=0 UIPC_MAS_FUSED_R=0 UIPC_SEG_REDUCE2=0"
# sim_case subset for racecheck: the subsystems round 6 touched -- MAS (s15/s16/s17, the
# converter's segmented reduce runs in every FEM case), contact assembly (s08/s14/V2, the
# K9 split), ABD (s10/s11), discrete-shell bending (s02), CCD (s04/s06 run in every contact case).
SUBSET='53_*,56_*,60_*,1_abd*,2_abd*,3_abd*,4_abd*,18_*,35_*,33_*,6_abd*'
phase () { echo "### $1  $(date '+%F %T')" | tee -a $OUT/driver.log; }
san_test_sub () { # arm root bdir tool subset [env]
  local arm=$1 root=$2 bdir=$3 tool=$4 sub=$5; shift 5
  local tag="${arm}__${tool}__sim_case_subset"; local log="$OUT/logs/$tag.txt"; local t0=$(date +%s)
  ( cd "$root" && env "$@" timeout 7200 "$CS" --tool "$tool" $CSFLAGS "$bdir"/Release/bin/uipc_test_sim_case "$sub" ) > "$log" 2>&1
  local rc=$?
  printf '%s rc=%d %4ds %s | %s | %s\n' "$(date +%H:%M:%S)" "$rc" "$(( $(date +%s) - t0 ))" "$tag" "$(_summ "$log")" "$(grep -E 'All tests passed|test cases:' "$log" | tail -1)" | tee -a "$OUT/driver.log"
}
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
phase "G replicate tumbler racecheck 6f (run-to-run scatter of the count on a non-deterministic scene)"
san_scene base2 $BASEPY racecheck 95_tumbler_garments 6
san_scene head2 $HEADPY racecheck 95_tumbler_garments 6
san_scene rb2 $HEADPY racecheck 95_tumbler_garments 6 $RB
phase "E2 tests: backend_cuda base x4, rb memcheck+racecheck"
for t in memcheck initcheck racecheck synccheck; do san_test base $BASEREPO $BASEREPO/build-base $t backend_cuda; done
san_test rb $HEADREPO $HEADREPO/build-perf memcheck backend_cuda $RB
san_test rb $HEADREPO $HEADREPO/build-perf racecheck backend_cuda $RB
phase "E2 tests: sim_case subset ($SUBSET), head x4; base / rb only where the head reports something"
for t in synccheck memcheck initcheck racecheck; do san_test_sub head $HEADREPO $HEADREPO/build-perf $t "$SUBSET"; done
for t in initcheck racecheck; do
  if grep -qE "SUMMARY: [1-9]" $OUT/logs/head__${t}__sim_case_subset.txt; then
    san_test_sub base $BASEREPO $BASEREPO/build-base $t "$SUBSET"
    san_test_sub rb $HEADREPO $HEADREPO/build-perf $t "$SUBSET" $RB
  else echo "$(date +%T) skip base/rb $t sim_case subset: head clean" | tee -a $OUT/driver.log; fi
done
phase "DONE2"; touch $OUT/DONE2

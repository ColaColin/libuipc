#!/bin/bash
# After the variant stage: the owner-authorized full sim_case memcheck (8 h timeout), the small
# replicate/base runs the report needs, then racecheck over sim_case sized to the clock, then the
# long tumbler windows.
source /workspace/output/round6/sanitizers/san.sh
HEADREPO=/workspace/deps/libuipc-src
BASEREPO=/workspace/output/round6/base/wt
RB="UIPC_DSB_GAUSS_NEWTON=0 UIPC_CONTACT_RANK1=0 UIPC_CCD_EARLY_OUT=0 UIPC_CCD_COMPACT=0 UIPC_CONTACT_SPD1_BASIS=0 UIPC_ABD_GH_PREPASS=0 UIPC_SPMV_GRID_FIT=0 UIPC_CONTACT_DEFERRED_JOIN=0 UIPC_MAS_ROWDOT2=0 UIPC_MAS_FUSED_R=0 UIPC_SEG_REDUCE2=0"
phase () { echo "### $1  $(date '+%F %T')" | tee -a $OUT/driver.log; }
san_test_sel () { # arm root bdir tool tag selector [env]   (Catch2 selector; -d yes prints per-case durations)
  local arm=$1 root=$2 bdir=$3 tool=$4 tg=$5 sel=$6; shift 6
  local tag="${arm}__${tool}__sim_case_${tg}"; local log="$OUT/logs/$tag.txt"; local t0=$(date +%s)
  ( cd "$root" && env "$@" timeout ${TMO:-7200} "$CS" --tool "$tool" $CSFLAGS "$bdir"/Release/bin/uipc_test_sim_case "$sel" -d yes ) > "$log" 2>&1
  local rc=$?
  printf '%s rc=%d %5ds %s | %s | %s\n' "$(date +%H:%M:%S)" "$rc" "$(( $(date +%s) - t0 ))" "$tag" "$(_summ "$log")" "$(grep -E 'All tests passed|test cases:' "$log" | tail -1)" | tee -a "$OUT/driver.log"
}
phase "H full sim_case memcheck on head, all 95 cases, timeout 8 h (owner-authorized overnight run)"
TMO=28800 san_test_sel head $HEADREPO $HEADREPO/build-perf memcheck full '*'
phase "E2 backend_cuda: base x4, rb memcheck+racecheck"
for t in memcheck initcheck racecheck synccheck; do san_test base $BASEREPO $BASEREPO/build-base $t backend_cuda; done
san_test rb $HEADREPO $HEADREPO/build-perf memcheck backend_cuda $RB
san_test rb $HEADREPO $HEADREPO/build-perf racecheck backend_cuda $RB
phase "G replicates: tumbler racecheck 6f x (base2, head2, rb2); mas-bunny racecheck 6f x (head2, base2, head3)"
san_scene base2 $BASEPY racecheck 95_tumbler_garments 6
san_scene head2 $HEADPY racecheck 95_tumbler_garments 6
san_scene rb2 $HEADPY racecheck 95_tumbler_garments 6 $RB
san_scene head2 $HEADPY racecheck 89_mas_bunny 6
san_scene base2 $BASEPY racecheck 89_mas_bunny 6
san_scene head3 $HEADPY racecheck 89_mas_bunny 6
phase "I racecheck over sim_case, head: the MAS / shell-bending / contact cases first (subset A), timeout 5 h; then base on the same subset only if the head reports anything outside the known MAS family"
SUBA='53_*,56_*,60_*,33_*,1_abd*,18_*'
TMO=18000 san_test_sel head $HEADREPO $HEADREPO/build-perf racecheck subsetA "$SUBA"
if grep -qE "SUMMARY: [1-9]" $OUT/logs/head__racecheck__sim_case_subsetA.txt; then
  TMO=18000 san_test_sel base $BASEREPO $BASEREPO/build-base racecheck subsetA "$SUBA"
else echo "$(date +%T) skip base racecheck sim_case subsetA: head clean" | tee -a $OUT/driver.log; fi
phase "F long tumbler windows into the dense-pile regime"
san_scene head $HEADPY racecheck 95_tumbler_garments 24
san_scene base $BASEPY racecheck 95_tumbler_garments 24
san_scene head $HEADPY memcheck  95_tumbler_garments 30
san_scene head $HEADPY initcheck 95_tumbler_garments 24
san_scene base $BASEPY initcheck 95_tumbler_garments 24
phase "DONE3"; touch $OUT/DONE3

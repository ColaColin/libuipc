#!/bin/bash
# After matrix4: racecheck over the FULL sim_case on head (racecheck cost ~= memcheck cost on this
# suite, measured on subset A), base on exactly the cases that reported anything, then synccheck and
# initcheck over the full suite the same way.
source /workspace/output/round6/sanitizers/san.sh
HEADREPO=/workspace/deps/libuipc-src
BASEREPO=/workspace/output/round6/base/wt
phase () { echo "### $1  $(date '+%F %T')" | tee -a $OUT/driver.log; }
san_test_sel () { local arm=$1 root=$2 bdir=$3 tool=$4 tg=$5 sel=$6; shift 6
  local tag="${arm}__${tool}__sim_case_${tg}"; local log="$OUT/logs/$tag.txt"; local t0=$(date +%s)
  ( cd "$root" && env "$@" timeout ${TMO:-7200} "$CS" --tool "$tool" $CSFLAGS "$bdir"/Release/bin/uipc_test_sim_case "$sel" -d yes ) > "$log" 2>&1
  local rc=$?
  printf '%s rc=%d %5ds %s | %s | %s\n' "$(date +%H:%M:%S)" "$rc" "$(( $(date +%s) - t0 ))" "$tag" "$(_summ "$log")" "$(grep -E 'All tests passed|test cases:' "$log" | tail -1)" | tee -a "$OUT/driver.log"
}
base_on_reporting () { # tool
  local sel=$(python3 $OUT/assign_records.py $OUT/logs/head__$1__sim_case_full.txt --selector)
  if [ "$sel" = "NONE" ]; then echo "$(date +%T) $1: head reported nothing on sim_case -- no base run needed" | tee -a $OUT/driver.log; return; fi
  echo "$(date +%T) $1: base re-run on the reporting cases: $sel" | tee -a $OUT/driver.log
  TMO=18000 san_test_sel base $BASEREPO $BASEREPO/build-base $1 reporting "$sel"
}
phase "J racecheck over the FULL sim_case on head (95 cases), timeout 5 h"
TMO=18000 san_test_sel head $HEADREPO $HEADREPO/build-perf racecheck full '*'
base_on_reporting racecheck
phase "K synccheck over the full sim_case on head"
TMO=18000 san_test_sel head $HEADREPO $HEADREPO/build-perf synccheck full '*'
base_on_reporting synccheck
phase "L initcheck over the full sim_case on head, timeout 5 h"
TMO=18000 san_test_sel head $HEADREPO $HEADREPO/build-perf initcheck full '*'
base_on_reporting initcheck
phase "DONE5"; touch $OUT/DONE5

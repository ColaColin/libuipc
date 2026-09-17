#!/bin/bash
# Phase A4: memcheck coverage repair on crease-press. The default memcheck arm
# exhausted the sanitizer's device-shadow memory on this 8 GB card (101 177
# "Internal Sanitizer Error ... Unable to allocate enough memory" records,
# first at the very first MAS launch) -- those records are instrument failures,
# not application defects, and the untracked set covers part of every family
# (worst: the whole PCG/MAS/SpMV graph family, ~10.6k launches each).
# --force-synchronization-limit 1 bounds the sanitizer's device memory at
# runtime cost; this arm re-runs head AND base to restore coverage.
source /workspace/output/round7/s20/san.sh
FSL="--force-synchronization-limit 1"

phase () { echo "### $1  $(date '+%F %T')" | tee -a $OUT/driver.log; }

CP=103_crease_press

phase "A4a redo: head memcheck 6_wrecking_balls 12f (the 21:47:34 cell was the cudaErrorDevicesUnavailable victim of the dying cp racecheck process: 0 s, 2 x error 46)"
san_scene head $HEADPY memcheck 6_wrecking_balls 12

phase "A4b redo: base memcheck 103_crease_press 30f (the 23:48:20 cell was the same victim of the dying pcgpollver racecheck: 1 s, 2 x error 46)"
san_scene base $BASEPY memcheck 103_crease_press 30

phase "A4c redo: base memcheck 6_wrecking_balls 12f (the 00:35:21 cell was the same victim of the dying base cp racecheck: 1 s, 2 x error 46)"
san_scene base $BASEPY memcheck 6_wrecking_balls 12

phase "A4 memcheck repair: crease-press, force-sync 1, 20f (press1+4 hold1 frames), head+base"
( cd "$REPO/libuipc-samples/examples/$CP" && \
  env timeout 14400 "$CS" --tool memcheck --target-processes all --launch-timeout 300 \
  --print-limit 100000 $FSL "$HEADPY" main.py --headless 20 ) > "$OUT/logs/repair_head__memcheck__${CP}_20f.txt" 2>&1
echo "$(date +%H:%M:%S) repair_head rc=$? | $(grep -E 'ERROR SUMMARY' "$OUT/logs/repair_head__memcheck__${CP}_20f.txt" | grep -v printed | tail -1) | internal=$(grep -c 'Internal Sanitizer Error' "$OUT/logs/repair_head__memcheck__${CP}_20f.txt")" | tee -a $OUT/driver.log

( cd "$REPO/libuipc-samples/examples/$CP" && \
  env timeout 14400 "$CS" --tool memcheck --target-processes all --launch-timeout 300 \
  --print-limit 100000 $FSL "$BASEPY" main.py --headless 20 ) > "$OUT/logs/repair_base__memcheck__${CP}_20f.txt" 2>&1
echo "$(date +%H:%M:%S) repair_base rc=$? | $(grep -E 'ERROR SUMMARY' "$OUT/logs/repair_base__memcheck__${CP}_20f.txt" | grep -v printed | tail -1) | internal=$(grep -c 'Internal Sanitizer Error' "$OUT/logs/repair_base__memcheck__${CP}_20f.txt")" | tee -a $OUT/driver.log

phase "A4 DONE"

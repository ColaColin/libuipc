#!/bin/bash
# Phase A1: HEAD arm (main 69c2af51, build-perf / uipc-perf-env).
# crease-press FIRST (it exercises everything the round touched), then the five
# regression scenes, then the round's verify probes composed on crease-press.
source /workspace/output/round7/s20/san.sh

phase () { echo "### $1  $(date '+%F %T')" | tee -a $OUT/driver.log; }

CP=103_crease_press
SCENES="6_wrecking_balls 93_cube_wall_cloth 88_stiff_gipc_benchmark 89_mas_bunny 95_tumbler_garments"

phase "A1 head: crease-press, three tools, 30f (press1+hold1+lift1-start)"
san_scene head $HEADPY memcheck  $CP 30
san_scene head $HEADPY initcheck $CP 30
san_scene head $HEADPY racecheck $CP 30

phase "A2 head: five regression scenes, round-6 frame counts"
for d in $SCENES; do
  san_scene head $HEADPY memcheck  $d 12
  san_scene head $HEADPY initcheck $d 8
  san_scene head $HEADPY racecheck $d 6
done

phase "A3 head: the round's verify probes composed on crease-press (memcheck 30f each)"
san_scene head_gnverify    $HEADPY memcheck $CP 30 UIPC_DAHL_GN_VERIFY=1
san_scene head_segredver   $HEADPY memcheck $CP 30 UIPC_SEGRED_VERIFY=1
san_scene head_doubletver  $HEADPY memcheck $CP 30 UIPC_DOUBLET_VERIFY=1
san_scene head_pcgpollver  $HEADPY memcheck $CP 30 UIPC_PCG_POLL_VERIFY=1
san_scene head_pcgpollver  $HEADPY racecheck $CP 12 UIPC_PCG_POLL_VERIFY=1

phase "A1-A3 DONE"

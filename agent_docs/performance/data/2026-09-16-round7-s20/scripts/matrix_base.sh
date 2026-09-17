#!/bin/bash
# Phase B1: BASE arm (perf-round7-base fce57589, build-base / round7 base venv).
# Same scenes, same frame counts as the head arm -- any head finding must
# reproduce here count-for-count to be pre-existing.
source /workspace/output/round7/s20/san.sh

phase () { echo "### $1  $(date '+%F %T')" | tee -a $OUT/driver.log; }

CP=103_crease_press
SCENES="6_wrecking_balls 93_cube_wall_cloth 88_stiff_gipc_benchmark 89_mas_bunny 95_tumbler_garments"

phase "B1 base: crease-press, three tools, 30f"
san_scene base $BASEPY memcheck  $CP 30
san_scene base $BASEPY initcheck $CP 30
san_scene base $BASEPY racecheck $CP 30

phase "B2 base: five regression scenes"
for d in $SCENES; do
  san_scene base $BASEPY memcheck  $d 12
  san_scene base $BASEPY initcheck $d 8
  san_scene base $BASEPY racecheck $d 6
done

phase "B1-B2 DONE"

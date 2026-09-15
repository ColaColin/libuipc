#!/bin/bash
source /workspace/output/round6/sanitizers/san.sh
echo "### calibration $(date)" >> $OUT/driver.log
for d in 95_tumbler_garments 88_stiff_gipc_benchmark 89_mas_bunny; do
  san_scene calib $HEADPY racecheck $d 2
done
san_scene calib $HEADPY memcheck 88_stiff_gipc_benchmark 6
san_scene calib $HEADPY synccheck 95_tumbler_garments 6
san_scene calib $HEADPY initcheck 88_stiff_gipc_benchmark 4

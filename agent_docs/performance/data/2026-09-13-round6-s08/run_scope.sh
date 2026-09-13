#!/bin/bash
# s08 scope sweep: contact part 1 (PT+EE) per launch, three arms.
# b0 = shipped dense-Q reduced projection, b1 = s31 basis-free extended to M=4,
# b2 = diagnosis stub (PT + un-mollified-EE projections skipped entirely).
set -u
cd /workspace/output/round6/s08
SC=${1:-cwc}; REPS=${2:-3}
case $SC in
  cwc) DIR=93_cube_wall_cloth;      F=100 ;;
  c2)  DIR=88_stiff_gipc_benchmark; F=250 ;;
  tum) DIR=95_tumbler_garments;     F=180 ;;
  rwb) DIR=6_wrecking_balls;        F=120 ;;
  *) echo "unknown scene $SC"; exit 2 ;;
esac
for r in $(seq 1 $REPS); do
  for arm in b0 b1 b2; do
    v=${arm#b}
    ./nsysrun.sh ${SC}_${arm}_r${r} $DIR $F UIPC_CONTACT_SPD1_BASIS=$v || echo "FAIL ${SC}_${arm}_r${r}"
  done
done
echo SCOPE_DONE_$SC

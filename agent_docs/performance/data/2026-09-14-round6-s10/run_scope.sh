#!/bin/bash
# s10 scope sweep: the ABD body-local G/H prepass, two arms.
#   b0 = UIPC_ABD_GH_PREPASS=0  in-place on the default stream (pre-s10)
#   b1 = UIPC_ABD_GH_PREPASS=1  forked before the contact phase (s10, default)
set -u
cd /workspace/output/round6/s10
SC=${1:-rwb}; REPS=${2:-3}
case $SC in
  cwc) DIR=93_cube_wall_cloth;      F=100 ;;
  c2)  DIR=88_stiff_gipc_benchmark; F=250 ;;
  tum) DIR=95_tumbler_garments;     F=180 ;;
  rwb) DIR=6_wrecking_balls;        F=120 ;;
  mb)  DIR=89_mas_bunny;            F=100 ;;
  *) echo "unknown scene $SC"; exit 2 ;;
esac
for r in $(seq 1 $REPS); do
  for arm in b1 b0; do
    v=${arm#b}
    ./nsysrun.sh ${SC}_${arm}_r${r} $DIR $F UIPC_ABD_GH_PREPASS=$v || echo "FAIL ${SC}_${arm}_r${r}"
  done
done
echo SCOPE_DONE_$SC

#!/bin/bash
# s09 scope sweep: the K9 contact stream split, three arms.
#   a0 = UIPC_CONTACT_SPLIT=0  fused single launch (pre-K9)
#   a1 = UIPC_CONTACT_SPLIT=1  two launches, both on the default stream (serial)
#   a2 = UIPC_CONTACT_SPLIT=2  two launches, part 1 on a side stream (shipped)
set -u
cd /workspace/output/round6/s09
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
  for arm in a2 a1 a0; do
    v=${arm#a}
    ./nsysrun.sh ${SC}_${arm}_r${r} $DIR $F UIPC_CONTACT_SPLIT=$v || echo "FAIL ${SC}_${arm}_r${r}"
  done
done
echo SCOPE_DONE_$SC

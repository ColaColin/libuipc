#!/bin/bash
# s07 scope sweep: part-2 (PE+PP) contact assembly per launch, four Proj arms.
# p0 = exact (default), p1 = PE+PP rank-1 (s03 mode 1), p5 = PP only, p6 = PE only.
set -u
cd /workspace/output/round6/s07
SC=${1:-cwc}
case $SC in
  cwc) DIR=93_cube_wall_cloth;      F=100 ;;
  c2)  DIR=88_stiff_gipc_benchmark; F=250 ;;
  tum) DIR=95_tumbler_garments;     F=180 ;;
  rwb) DIR=6_wrecking_balls;        F=120 ;;
  *) echo "unknown scene $SC"; exit 2 ;;
esac
for r in 1 2 3; do
  for arm in p0 p1 p5 p6; do
    v=${arm#p}
    ./nsysrun.sh ${SC}_${arm}_r${r} $DIR $F UIPC_CONTACT_RANK1=$v || echo "FAIL ${SC}_${arm}_r${r}"
  done
done
echo SCOPE_DONE_$SC

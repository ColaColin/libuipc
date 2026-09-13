#!/bin/bash
set -u
cd /workspace/output/round6/s06
for r in 1 2 3; do
  for arm in c0 c1; do
    v=${arm#c}
    ./nsysrun.sh tum_${arm}_r${r} 95_tumbler_garments 180 UIPC_CCD_COMPACT=$v || echo "FAIL tum_${arm}_r${r}"
    ./nsysrun.sh c2_${arm}_r${r}  88_stiff_gipc_benchmark 250 UIPC_CCD_COMPACT=$v || echo "FAIL c2_${arm}_r${r}"
    ./nsysrun.sh cwc_${arm}_r${r} 93_cube_wall_cloth 100 UIPC_CCD_COMPACT=$v || echo "FAIL cwc_${arm}_r${r}"
  done
done
for r in 1 2; do
  for arm in c0 c1; do
    v=${arm#c}
    ./nsysrun.sh mb_${arm}_r${r} 89_mas_bunny 100 UIPC_CCD_COMPACT=$v || echo "FAIL mb_${arm}_r${r}"
  done
done
echo SCOPE_DONE

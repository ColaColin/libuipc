#!/bin/bash
set -u
cd /workspace/output/round6/s04
# control validity first (PERF_METHOD 2.2): does the scene launch filter_toi at all?
./nsysrun.sh ctl_bunny 89_mas_bunny 100 UIPC_CCD_EARLY_OUT=0 || echo "FAIL ctl_bunny"
for r in 1 2 3; do
  for arm in eo0 eo1; do
    v=${arm#eo}
    ./nsysrun.sh cwc_${arm}_r${r} 93_cube_wall_cloth 100 UIPC_CCD_EARLY_OUT=$v || echo "FAIL cwc_${arm}_r${r}"
    ./nsysrun.sh rwb_${arm}_r${r} 6_wrecking_balls  120 UIPC_CCD_EARLY_OUT=$v || echo "FAIL rwb_${arm}_r${r}"
    ./nsysrun.sh c2_${arm}_r${r}  88_stiff_gipc_benchmark 250 UIPC_CCD_EARLY_OUT=$v || echo "FAIL c2_${arm}_r${r}"
  done
done
echo SCOPE2_DONE

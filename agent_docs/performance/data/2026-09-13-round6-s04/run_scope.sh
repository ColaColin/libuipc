#!/bin/bash
set -u
cd /workspace/output/round6/s04
for r in 1 2 3; do
  for arm in eo0 eo1; do
    v=${arm#eo}
    ./nsysrun.sh tum_${arm}_r${r} 95_tumbler_garments 180 UIPC_CCD_EARLY_OUT=$v || echo "FAIL tum_${arm}_r${r}"
  done
done
echo SCOPE_DONE

#!/bin/bash
set -u
cd /workspace/output/round6/s05
SCENE=$1; DIR=$2; FRAMES=$3; N=${4:-3}
for r in $(seq 1 $N); do
  for arm in c0 c1; do
    v=${arm#c}
    ./nsysrun.sh ${SCENE}_${arm}_r${r} $DIR $FRAMES UIPC_CCD_CULL=$v || echo "FAIL ${SCENE}_${arm}_r${r}"
  done
done
echo SCOPE_DONE_$SCENE

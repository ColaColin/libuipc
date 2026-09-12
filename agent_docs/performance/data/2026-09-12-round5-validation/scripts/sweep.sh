#!/bin/bash
# usage: sweep.sh <label> <tree> <pyexe> <frames> <tools...>
LABEL=$1; TREE=$2; PY=$3; FRAMES=$4; shift 4
CS=/workspace/deps/cuda-12.8/bin/compute-sanitizer
OUT=/workspace/output/round5/validation/sanitizer
for tool in "$@"; do
  for s in 6_wrecking_balls:wb 93_cube_wall_cloth:cwc 88_stiff_gipc_benchmark:c2 89_mas_bunny:mb; do
    d=${s%%:*}; n=${s##*:}
    f=$OUT/${LABEL}_${tool}_${n}.log
    st=$(date +%s)
    timeout 5400 $CS --tool $tool --print-limit 20 bash $OUT/run_scene.sh $TREE $PY $d $FRAMES > $f 2>&1
    rc=$?
    en=$(date +%s)
    sum=$(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $f | tail -1)
    echo "$LABEL $tool $n frames=$FRAMES rc=$rc $((en-st))s :: $sum"
  done
done

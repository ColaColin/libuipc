#!/bin/bash
# Serial sanitizer queue over both revisions. Progress -> queue_progress.log
OUT=/workspace/output/round5/validation/sanitizer
CS=/workspace/deps/cuda-12.8/bin/compute-sanitizer
HEADTREE=/workspace/deps/libuipc-src;  HEADPY=/workspace/deps/uipc-perf-env/bin/python
BASETREE=/workspace/deps/libuipc-valbase; BASEPY=/workspace/deps/uipc-valbase-env/bin/python
P=$OUT/queue_progress.log
run() { # label tree py tool scenedir short frames
  local label=$1 tree=$2 py=$3 tool=$4 d=$5 n=$6 f=$7
  local lf=$OUT/${label}_${tool}_${n}.log
  [ -s "$lf" ] && grep -q 'SUMMARY' "$lf" && { echo "SKIP $label $tool $n (cached)" >> $P; return; }
  local st=$(date +%s)
  timeout 5400 $CS --tool $tool --print-limit 20 bash $OUT/run_scene.sh $tree $py $d $f > $lf 2>&1
  local rc=$?; local en=$(date +%s)
  echo "$label $tool $n frames=$f rc=$rc $((en-st))s :: $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $lf | tail -1)" >> $P
}
runbin() { # label builddir tool binary
  local label=$1 bd=$2 tool=$3 b=$4
  local lf=$OUT/${label}_${tool}_bin_${b}.log
  [ -s "$lf" ] && grep -q 'SUMMARY' "$lf" && { echo "SKIP $label $tool bin_$b (cached)" >> $P; return; }
  local st=$(date +%s)
  ( export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:/workspace/deps/cuda-12.8/lib64
    cd $(dirname $bd) && timeout 5400 $CS --tool $tool --print-limit 20 $bd/Release/bin/uipc_test_$b ) > $lf 2>&1
  local rc=$?; local en=$(date +%s)
  echo "$label $tool bin_$b rc=$rc $((en-st))s :: $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $lf | tail -1)" >> $P
}
SCENES="6_wrecking_balls:wb 93_cube_wall_cloth:cwc 88_stiff_gipc_benchmark:c2 89_mas_bunny:mb"
for tool in racecheck initcheck memcheck; do
  for s in $SCENES; do run head $HEADTREE $HEADPY $tool ${s%%:*} ${s##*:} 20; done
done
for tool in memcheck racecheck initcheck; do
  for s in $SCENES; do run base $BASETREE $BASEPY $tool ${s%%:*} ${s##*:} 20; done
done
for tool in memcheck racecheck initcheck; do
  for b in backend_cuda sim_case sanity_check regression; do
    runbin head /workspace/deps/libuipc-src/build-perf $tool $b
    runbin base /workspace/deps/libuipc-valbase/build-base $tool $b
  done
done
echo "QUEUE DONE" >> $P

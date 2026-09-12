#!/bin/bash
OUT=/workspace/output/round5/validation/sanitizer
CS=/workspace/deps/cuda-12.8/bin/compute-sanitizer
P=$OUT/queue_progress.log
run() { local label=$1 tree=$2 py=$3 tool=$4 d=$5 n=$6 f=$7
  local lf=$OUT/${label}_${tool}_${n}.log
  local st=$(date +%s)
  timeout 5400 $CS --tool $tool --print-limit 20 bash $OUT/run_scene.sh $tree $py $d $f > $lf 2>&1
  local rc=$?; echo "$label $tool $n frames=$f rc=$rc $(( $(date +%s)-st ))s :: $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $lf | tail -1)" >> $P; }
runbin() { local label=$1 bd=$2 tool=$3 b=$4
  local lf=$OUT/${label}_${tool}_bin_${b}.log
  [ -s "$lf" ] && grep -q 'SUMMARY' "$lf" && { echo "SKIP $label $tool bin_$b (cached)" >> $P; return; }
  local st=$(date +%s)
  ( export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:/workspace/deps/cuda-12.8/lib64
    timeout 5400 $CS --tool $tool --print-limit 20 $bd/Release/bin/uipc_test_$b ) > $lf 2>&1
  local rc=$?
  echo "$label $tool bin_$b rc=$rc $(( $(date +%s)-st ))s :: $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $lf | tail -1) | $(grep -E 'All tests passed|assertions' $lf | tail -1)" >> $P; }
HB=/workspace/deps/libuipc-src/build-perf; BB=/workspace/deps/libuipc-valbase/build-base
run head /workspace/deps/libuipc-src /workspace/deps/uipc-perf-env/bin/python racecheck 88_stiff_gipc_benchmark c2 20
for tool in memcheck initcheck racecheck; do
  for b in backend_cuda sanity_check regression; do
    runbin head $HB $tool $b; runbin base $BB $tool $b
  done
done
for tool in memcheck initcheck; do runbin head $HB $tool sim_case; runbin base $BB $tool sim_case; done
echo "QUEUE2 DONE" >> $P

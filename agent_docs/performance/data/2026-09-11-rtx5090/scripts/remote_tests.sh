#!/bin/bash
# Add-on: enable tests (incremental reconfigure) on head and base, run sim_case 74/80 and backend_cuda "lbvh" x3.
set -uo pipefail
OUT=/work/results/tests; mkdir -p $OUT
for side in head base; do
  R=/work/libuipc-$side; V=/work/venv-src-$side
  echo "== tests build $side $(date -u +%T)"
  (cd $R && cmake -S . -B build -DUIPC_BUILD_TESTS=ON > build/configure_tests.log 2>&1 && cmake --build build -j24 --target sim_case backend_cuda > build/build_tests.log 2>&1); echo "build rc=$? $(date -u +%T)"; tail -2 $R/build/build_tests.log
  ls $R/build/Release/bin/ | grep uipc_test
  cd $R/build/Release/bin || continue
  for c in 74_abd_revolute_joint_external_force 80_abd_revolute_joint_driving_and_external_torque; do
    echo "-- sim_case $c $side"; timeout 900 ./uipc_test_sim_case "$c" > $OUT/${side}_simcase_$c.log 2>&1; echo "rc=$?"; tail -3 $OUT/${side}_simcase_$c.log
  done
  for i in 1 2 3; do echo "-- lbvh run $i $side"; timeout 600 ./uipc_test_backend_cuda "lbvh" > $OUT/${side}_lbvh_$i.log 2>&1; echo "rc=$?"; tail -2 $OUT/${side}_lbvh_$i.log; done
done
nvcc --version | tail -1
echo TESTS_DONE

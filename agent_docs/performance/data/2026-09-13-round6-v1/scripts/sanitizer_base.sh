#!/bin/bash
# Reproduce the head's sanitizer findings on a *baseline binary* rather than arguing
# they are pre-existing.  The baseline is the round-5 validation pass's own base
# install, /workspace/deps/uipc-valbase-env, built from `perf-round5-base` = 890482c2
# -- i.e. a binary older than perf-round6-base.  Its source tree is gone; the
# installed backend libraries are what is run here.
set -u
OUT=/workspace/output/round6/v1/sanitizer
REPO=/workspace/deps/libuipc-src
CS=/workspace/deps/cuda-12.8/bin/compute-sanitizer
BASEPY=/workspace/deps/uipc-valbase-env/bin/python
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
export LD_LIBRARY_PATH=$REPO/build-dahl/vcpkg_installed/x64-linux/lib:/workspace/deps/cuda-12.8/lib64
run () { # tool frames tag
  echo "=== BASE $3 $1 $2f  $(date +%H:%M:%S)"
  ( cd "$REPO/libuipc-samples/examples/93_cube_wall_cloth" && \
    timeout 5400 "$CS" --tool "$1" --target-processes all "$BASEPY" main.py --headless "$2" ) \
    > "$OUT/base_$3.txt" 2>&1
  echo "   rc=$?"; grep -E "ERROR SUMMARY|RACECHECK SUMMARY" "$OUT/base_$3.txt" | tail -2
}
run memcheck  6 memcheck_cwc
run initcheck 4 initcheck_cwc
run racecheck 2 racecheck_cwc

#!/bin/bash
# Round-6 sanitizer sweep on the composed head (main = 67062f5f) and on the
# perf-round6-base build.  One function, called by the matrix scripts.
#   san_scene <arm> <py> <tool> <scene-dir> <frames> [ENV=VAL ...]
#   san_test  <arm> <repo-root> <build-dir> <tool> <test-name> [ENV=VAL ...]
# Every run writes logs/<arm>__<tool>__<subject>.txt and one line to driver.log.
set -u
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
OUT=/workspace/output/round6/sanitizers
REPO=/workspace/deps/libuipc-src
CS=$CUDA_HOME/bin/compute-sanitizer
HEADPY=/workspace/deps/uipc-perf-env/bin/python
BASEPY=/workspace/output/round6/base/venv/bin/python
CSFLAGS="--target-processes all --launch-timeout 300 --print-limit 100000"

_summ () { grep -E "ERROR SUMMARY|RACECHECK SUMMARY|SYNCCHECK SUMMARY" "$1" | grep -v "were not printed" | tail -1; }

san_scene () {
  local arm=$1 py=$2 tool=$3 dir=$4 frames=$5; shift 5
  local tag="${arm}__${tool}__${dir}_${frames}f"
  local log="$OUT/logs/$tag.txt"
  local t0=$(date +%s)
  ( cd "$REPO/libuipc-samples/examples/$dir" && \
    env "$@" timeout ${TMO:-7200} "$CS" --tool "$tool" $CSFLAGS "$py" main.py --headless "$frames" ) > "$log" 2>&1
  local rc=$?
  printf '%s rc=%d %4ds %s | %s\n' "$(date +%H:%M:%S)" "$rc" "$(( $(date +%s) - t0 ))" "$tag" "$(_summ "$log")" | tee -a "$OUT/driver.log"
}

san_test () {  # san_test <arm> <repo-root> <build-dir> <tool> <test-name> [ENV=VAL ...]
  local arm=$1 root=$2 bdir=$3 tool=$4 tb=$5; shift 5
  local tag="${arm}__${tool}__${tb}"
  local log="$OUT/logs/$tag.txt"
  local t0=$(date +%s)
  ( cd "$root" && env "$@" timeout ${TMO:-7200} "$CS" --tool "$tool" $CSFLAGS "$bdir"/Release/bin/uipc_test_$tb ) > "$log" 2>&1
  local rc=$?
  printf '%s rc=%d %4ds %s | %s | %s\n' "$(date +%H:%M:%S)" "$rc" "$(( $(date +%s) - t0 ))" "$tag" "$(_summ "$log")" "$(grep -E 'All tests passed|test cases:' "$log" | tail -1)" | tee -a "$OUT/driver.log"
}

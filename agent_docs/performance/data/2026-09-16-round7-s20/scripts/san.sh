#!/bin/bash
# Round-7 s20 sanitizer sweep on the composed head (main = 69c2af51) and on the
# perf-round7-base build (fce57589).  One function, called by the matrix scripts.
#   san_scene <arm> <py> <tool> <scene-dir> <frames> [ENV=VAL ...]
# Every run writes logs/<arm>__<tool>__<subject>.txt and one line to driver.log.
# Frame windows: crease-press 30f = press1 (0-15) + hold1 (16-21) + 8 lift1 frames
# (23 % of the 130-frame run) -- the yield/dahl-commit paths execute; the five
# round-6 scenes keep round-6's counts (memcheck 12 / initcheck 8 / racecheck 6).
set -u
export CUDA_HOME=/workspace/deps/cuda-12.8
export LD_LIBRARY_PATH=/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/workspace/archive/libuipc-samples/shim
OUT=/workspace/output/round7/s20
REPO=/workspace/deps/libuipc-src
CS=$CUDA_HOME/bin/compute-sanitizer
HEADPY=/workspace/deps/uipc-perf-env/bin/python
BASEPY=/workspace/output/round7/base/venv/bin/python
CSFLAGS="--target-processes all --launch-timeout 300 --print-limit 100000"
mkdir -p "$OUT/logs"

_summ () { grep -E "ERROR SUMMARY|RACECHECK SUMMARY|SYNCCHECK SUMMARY" "$1" | grep -v "were not printed" | tail -1; }

san_scene () {
  local arm=$1 py=$2 tool=$3 dir=$4 frames=$5; shift 5
  local tag="${arm}__${tool}__${dir}_${frames}f"
  local log="$OUT/logs/$tag.txt"
  local t0=$(date +%s)
  ( cd "$REPO/libuipc-samples/examples/$dir" && \
    env "$@" timeout ${TMO:-7200} "$CS" --tool "$tool" $CSFLAGS "$py" main.py --headless "$frames" ) > "$log" 2>&1
  local rc=$?
  printf '%s rc=%d %5ds %s | %s\n' "$(date +%H:%M:%S)" "$rc" "$(( $(date +%s) - t0 ))" "$tag" "$(_summ "$log")" | tee -a "$OUT/driver.log"
}

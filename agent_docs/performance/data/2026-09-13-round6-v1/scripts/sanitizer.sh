#!/bin/bash
# Round-6 V1: compute-sanitizer over short runs of the tumbler and cube-wall-cloth.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
OUT=/workspace/output/round6/v1/sanitizer
REPO=/workspace/deps/libuipc-src
mkdir -p "$OUT"
CS=/workspace/deps/cuda-12.8/bin/compute-sanitizer

run () { # $1 tool  $2 scene-dir  $3 frames  $4 tag  rest: env
  local tool=$1 dir=$2 frames=$3 tag=$4; shift 4
  echo "=== $tag $tool $dir ${frames}f  $(date +%H:%M:%S)"
  ( cd "$REPO/libuipc-samples/examples/$dir" && \
    env "$@" timeout 5400 "$CS" --tool "$tool" --target-processes all \
        --launch-timeout 300 \
        "$UIPC_PERF_PY" main.py --headless "$frames" ) > "$OUT/$tag.txt" 2>&1
  echo "   rc=$? : $(grep -cE '^====== ERROR|error detected|Invalid|race|uninitialized' "$OUT/$tag.txt" 2>/dev/null) hits; summary:"
  grep -E "ERROR SUMMARY|RACECHECK SUMMARY|error detected" "$OUT/$tag.txt" | tail -3
}

run memcheck  95_tumbler_garments 6 memcheck_tumbler
run memcheck  93_cube_wall_cloth  6 memcheck_cwc
run initcheck 95_tumbler_garments 4 initcheck_tumbler
run initcheck 93_cube_wall_cloth  4 initcheck_cwc
run racecheck 95_tumbler_garments 2 racecheck_tumbler
run racecheck 93_cube_wall_cloth  2 racecheck_cwc

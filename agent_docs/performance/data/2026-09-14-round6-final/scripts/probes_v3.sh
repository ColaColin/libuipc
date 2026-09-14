#!/bin/bash
# Round-6 V3: compose the round's own verification probes and run them together on the head
# build (PERF_METHOD §5). Each probe re-runs the old path beside the new one on device and
# reports a mismatch count / max relative difference. WB_LOG=Info because two of them report
# at info level. Where a probe reports only on failure, the same run is profiled so the
# verify kernel's launch count proves it executed.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
NSYS=/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys
OUT=/workspace/output/round6/v3/probes
REPO=/workspace/deps/libuipc-src
mkdir -p "$OUT"
T=95_tumbler_garments; R=6_wrecking_balls; M=89_mas_bunny; C=88_stiff_gipc_benchmark

run () { # tag scene-dir frames env...
  local tag=$1 dir=$2 frames=$3; shift 3
  ( cd "$REPO/libuipc-samples/examples/$dir" && env WB_LOG=Info UIPC_BENCHMARK_TIMERS=0 "$@" \
      "$UIPC_PERF_PY" main.py --headless "$frames" > "$OUT/$tag.txt" 2>&1 ); echo "rc=$? $tag ($*)"
}
runprof () { # tag scene-dir frames env...   (also count the verify kernel launches)
  local tag=$1 dir=$2 frames=$3; shift 3
  local p="$OUT/$tag"; rm -f "$p".nsys-rep "$p".sqlite "$p"_cuda_gpu_kern_sum.csv
  ( cd "$REPO/libuipc-samples/examples/$dir" && env WB_LOG=Info UIPC_BENCHMARK_TIMERS=0 "$@" \
      $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$p" "$UIPC_PERF_PY" main.py --headless "$frames" > "$OUT/$tag.txt" 2>&1 ); echo "rc=$? $tag ($*)"
  $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export=true -o "$p" "$p.nsys-rep" > "$OUT/$tag.stats.log" 2>&1
  [ -s "$p"_cuda_gpu_kern_sum.csv ] || echo "FAIL: no kern_sum csv for $tag"
}

run     gn_verify_tum   $T 12 UIPC_DSB_GN_VERIFY=1
run     gn_verify_cwc   93_cube_wall_cloth 12 UIPC_DSB_GN_VERIFY=1
run     ccd_compact_verify_tum $T 40 UIPC_CCD_COMPACT_VERIFY=1
runprof abd_prepass_verify_rwb $R 12 UIPC_ABD_GH_PREPASS_VERIFY=1
runprof abd_prepass_verify_cwc 93_cube_wall_cloth 12 UIPC_ABD_GH_PREPASS_VERIFY=1
run     dj_verify_c2    $C 40 UIPC_CONTACT_DEFERRED_JOIN_VERIFY=1
run     dj_verify_rwb   $R 40 UIPC_CONTACT_DEFERRED_JOIN_VERIFY=1
run     mas_apply_verify_mb $M 12 UIPC_MAS_APPLY_VERIFY=5
run     mas_apply_verify_c2 $C 12 UIPC_MAS_APPLY_VERIFY=5
runprof mas_rtail_verify_mb $M 12 UIPC_MAS_R_TAIL_VERIFY=1
run     seg_verify_mb   $M 12 UIPC_SEG_VERIFY=1 UIPC_SEG_HIST=1
run     seg_verify_c2   $C 12 UIPC_SEG_VERIFY=1 UIPC_SEG_HIST=1
run     seg_verify_tum  $T 12 UIPC_SEG_VERIFY=1 UIPC_SEG_HIST=1
run     spmv_verify_mb  $M 12 UIPC_SPMV_VERIFY=1
run     spmv_verify_c2  $C 12 UIPC_SPMV_VERIFY=1
echo PROBES_DONE

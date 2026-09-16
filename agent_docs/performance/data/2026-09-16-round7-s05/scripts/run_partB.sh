#!/bin/bash
# Round-7 s05 Part B: probe composition + drift runs + fresh nsys rankings.
# Serial on the GPU; run after run_arms.sh completes.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
S=/workspace/output/round7/s05
CP=/workspace/deps/libuipc-src/libuipc-samples/examples/103_crease_press
cd "$CP"

# 1) GN_VERIFY probe at default (GN) -- full 130-frame run, the round's probe
#    composition check (s03's probe must still work at this head).
if [ ! -f "$S/gnverify_fullrun.txt" ]; then
  echo "=== gnverify full run start $(date +%T)"
  env -u CP_DT -u CP_PDEPTH -u CP_DIE -u CP_YSTRAIN -u CP_YSTRESS \
    -u UIPC_DAHL_GAUSS_NEWTON UIPC_DAHL_GN_VERIFY=1 \
    timeout 900 $UIPC_PERF_PY main.py 130 > "$S/gnverify_fullrun.txt" 2>&1
  echo "=== gnverify rc=$? $(grep -c SpreadVerify "$S/gnverify_fullrun.txt") spreadverify lines"
fi

# 2) env audit: 12-frame nsys of the EXACT arm -- the launched dahl
#    instantiation must be <1,1> (the switch really selects the exact path).
if [ ! -f "$S/envaudit_exact_kern.csv" ]; then
  echo "=== envaudit exact-arm nsys start $(date +%T)"
  env -u CP_DT -u CP_PDEPTH -u CP_DIE -u CP_YSTRAIN -u CP_YSTRESS \
    UIPC_DAHL_GAUSS_NEWTON=0 \
    timeout 900 nsys profile -o "$S/envaudit_exact" -f true \
    --env UIPC_DAHL_GAUSS_NEWTON=0 -- $UIPC_PERF_PY main.py 12 \
    > "$S/envaudit_exact_run.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv \
    -o "$S/envaudit_exact_kern" "$S/envaudit_exact.nsys-rep" > /dev/null 2>&1
  [ -f "$S/envaudit_exact_kern_cuda_gpu_kern_sum.csv" ] \
    && echo "envaudit csv OK" || echo "ENVaudit CSV MISSING -- FAIL LOUDLY"
fi

# 3) drift: n=6 plain default runs through the benchmark harness
for k in 1 2 3 4 5 6; do
  if [ ! -f "$S/drift_default_$k.json" ]; then
    echo "=== drift run $k start $(date +%T)"
    (cd /workspace/deps/libuipc-src && timeout 900 $UIPC_PERF_PY scripts/run_benchmark.py \
      run crease-press --python $UIPC_PERF_PY > "$S/drift_default_$k.log" 2>&1)
    cp /workspace/deps/libuipc-src/output/benchmark-runs/crease-press.json "$S/drift_default_$k.json"
    echo "=== drift $k rc=$? $(grep -o 'meanFrameMs.*' "$S/drift_default_$k.log" | head -1)"
  fi
done

# 4) fresh full-run kernel ranking, crease-press at default (GN) head
if [ ! -f "$S/nsys_cp_default_kern.csv" ]; then
  echo "=== nsys crease-press default full run start $(date +%T)"
  env -u CP_DT -u CP_PDEPTH -u CP_DIE -u CP_YSTRAIN -u CP_YSTRESS \
    -u UIPC_DAHL_GAUSS_NEWTON \
    timeout 900 nsys profile -o "$S/nsys_cp_default" -f true \
    -- $UIPC_PERF_PY main.py 130 > "$S/nsys_cp_default_run.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv \
    -o "$S/nsys_cp_default_kern" "$S/nsys_cp_default.nsys-rep" > /dev/null 2>&1
  [ -f "$S/nsys_cp_default_kern_cuda_gpu_kern_sum.csv" ] \
    && echo "nsys cp csv OK" || echo "NSYS CP CSV MISSING -- FAIL LOUDLY"
fi

# 5) cross-scene reference ranking: cube-wall-cloth full run
if [ ! -f "$S/nsys_cwc_kern.csv" ]; then
  echo "=== nsys cube-wall-cloth full run start $(date +%T)"
  (cd /workspace/deps/libuipc-src/libuipc-samples/examples/93_cube_wall_cloth \
    && timeout 900 nsys profile -o "$S/nsys_cwc" -f true \
    -- $UIPC_PERF_PY main.py --headless 100 > "$S/nsys_cwc_run.log" 2>&1)
  nsys stats --report cuda_gpu_kern_sum --format csv \
    -o "$S/nsys_cwc_kern" "$S/nsys_cwc.nsys-rep" > /dev/null 2>&1
  [ -f "$S/nsys_cwc_kern_cuda_gpu_kern_sum.csv" ] \
    && echo "nsys cwc csv OK" || echo "NSYS CWC CSV MISSING -- FAIL LOUDLY"
fi
echo PARTB_DONE

#!/bin/bash
# Round-7 s05 nsys captures with the correct nsys path (env_perf.sh does not
# put /workspace/deps/nsight/bin on PATH -- the first attempt failed loudly).
set -u
source /workspace/deps/libuipc-src/env_perf.sh
NSYS=/workspace/deps/cuda-12.8/host/target-linux-x64/nsys
S=/workspace/output/round7/s05
CP=/workspace/deps/libuipc-src/libuipc-samples/examples/103_crease_press
cd "$CP"
"$NSYS" --version || { echo "NSYS BROKEN"; exit 1; }

# 1) env audit: 12-frame capture of the EXACT arm -- the launched dahl
#    instantiation must be <1,1> (UIPC_DAHL_GAUSS_NEWTON=0 selects the exact path)
if [ ! -f "$S/envaudit_exact_kern_cuda_gpu_kern_sum.csv" ]; then
  echo "=== envaudit exact-arm nsys start $(date +%T)"
  env -u CP_DT -u CP_PDEPTH -u CP_DIE -u CP_YSTRAIN -u CP_YSTRESS \
    UIPC_DAHL_GAUSS_NEWTON=0 \
    timeout 900 "$NSYS" profile -o "$S/envaudit_exact" -f true \
    -- $UIPC_PERF_PY main.py 12 > "$S/envaudit_exact_run.log" 2>&1
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
    -o "$S/envaudit_exact_kern" "$S/envaudit_exact.nsys-rep" > /dev/null 2>&1
  [ -f "$S/envaudit_exact_kern_cuda_gpu_kern_sum.csv" ] \
    && echo "envaudit csv OK" || echo "ENVAUDIT CSV MISSING -- FAIL LOUDLY"
fi

# 2) fresh full-run kernel ranking, crease-press at default (GN) head
if [ ! -f "$S/nsys_cp_default_kern_cuda_gpu_kern_sum.csv" ]; then
  echo "=== nsys crease-press default full run start $(date +%T)"
  env -u CP_DT -u CP_PDEPTH -u CP_DIE -u CP_YSTRAIN -u CP_YSTRESS \
    -u UIPC_DAHL_GAUSS_NEWTON \
    timeout 900 "$NSYS" profile -o "$S/nsys_cp_default" -f true \
    -- $UIPC_PERF_PY main.py 130 > "$S/nsys_cp_default_run.log" 2>&1
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
    -o "$S/nsys_cp_default_kern" "$S/nsys_cp_default.nsys-rep" > /dev/null 2>&1
  [ -f "$S/nsys_cp_default_kern_cuda_gpu_kern_sum.csv" ] \
    && echo "nsys cp csv OK" || echo "NSYS CP CSV MISSING -- FAIL LOUDLY"
fi

# 3) cross-scene reference ranking: cube-wall-cloth full run
if [ ! -f "$S/nsys_cwc_kern_cuda_gpu_kern_sum.csv" ]; then
  echo "=== nsys cube-wall-cloth full run start $(date +%T)"
  (cd /workspace/deps/libuipc-src/libuipc-samples/examples/93_cube_wall_cloth \
    && timeout 900 "$NSYS" profile -o "$S/nsys_cwc" -f true \
    -- $UIPC_PERF_PY main.py --headless 100 > "$S/nsys_cwc_run.log" 2>&1)
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
    -o "$S/nsys_cwc_kern" "$S/nsys_cwc.nsys-rep" > /dev/null 2>&1
  [ -f "$S/nsys_cwc_kern_cuda_gpu_kern_sum.csv" ] \
    && echo "nsys cwc csv OK" || echo "NSYS CWC CSV MISSING -- FAIL LOUDLY"
fi
echo NSYS_DONE

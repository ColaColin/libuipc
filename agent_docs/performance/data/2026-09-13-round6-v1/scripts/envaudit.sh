#!/bin/bash
# Round-6 V1 env-switch audit: kernel-level evidence, not wall time.
# For each switch, profile a short run and list the *mangled/demangled kernel
# names actually launched*.  A switch that selects a different template
# instantiation must change the name list; one that is inert must not.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
OUT=/workspace/output/round6/v1/envaudit
REPO=/workspace/deps/libuipc-src
mkdir -p "$OUT"

prof () {  # $1 tag  $2 scene-dir  $3 frames  rest: env assignments
  local tag=$1 dir=$2 frames=$3; shift 3
  local p="$OUT/$tag"
  rm -f "$p".nsys-rep "$p"_cuda_gpu_kern_sum.csv
  ( cd "$REPO/libuipc-samples/examples/$dir" && \
    env "$@" /workspace/deps/nsight/bin/nsys profile -t cuda --cuda-graph-trace=node -f true -o "$p" \
      "$UIPC_PERF_PY" main.py --headless "$frames" > "$OUT/$tag.stdout" 2>&1 )
  /workspace/deps/nsight/bin/nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      -o "$p" "$p.nsys-rep" > "$OUT/$tag.stats.log" 2>&1
  if [ ! -s "$p"_cuda_gpu_kern_sum.csv ]; then
    echo "FAIL: no kern_sum csv for $tag"; return 1
  fi
  echo "ok $tag ($(wc -l < "$p"_cuda_gpu_kern_sum.csv) kernel rows)"
}

names () { # print the kernels of interest from a kern_sum csv
  local p=$1 pat=$2
  python3 - "$p" "$pat" <<'PY'
import csv, sys, re
rows = list(csv.DictReader(open(sys.argv[1])))
pat = re.compile(sys.argv[2])
for r in rows:
    n = r.get("Name") or r.get("Kernel Name") or ""
    if pat.search(n):
        print(f"{float(r['Instances']):9.0f}  {n}")
PY
}

echo "### s02  UIPC_DSB_GAUSS_NEWTON"
prof gn_default 95_tumbler_garments 12
prof gn_off     95_tumbler_garments 12 UIPC_DSB_GAUSS_NEWTON=0
echo "--- default (switch unset):"; names "$OUT/gn_default_cuda_gpu_kern_sum.csv" "DiscreteShellBending"
echo "--- UIPC_DSB_GAUSS_NEWTON=0:"; names "$OUT/gn_off_cuda_gpu_kern_sum.csv" "DiscreteShellBending"

echo "### s03  UIPC_CONTACT_RANK1  (must be INERT by default)"
prof c_default 95_tumbler_garments 12
prof c_on      95_tumbler_garments 12 UIPC_CONTACT_RANK1=1
echo "--- default (switch unset):"; names "$OUT/c_default_cuda_gpu_kern_sum.csv" "do_assemble_kernel"
echo "--- UIPC_CONTACT_RANK1=1:";   names "$OUT/c_on_cuda_gpu_kern_sum.csv" "do_assemble_kernel"

echo "### s01  UIPC_DSB_OCC  (must be INERT by default)"
prof occ_on 93_cube_wall_cloth 12 UIPC_DSB_OCC=1
prof occ_def 93_cube_wall_cloth 12
echo "--- default:";        names "$OUT/occ_def_cuda_gpu_kern_sum.csv" "DiscreteShellBending"
echo "--- UIPC_DSB_OCC=1:"; names "$OUT/occ_on_cuda_gpu_kern_sum.csv" "DiscreteShellBending"

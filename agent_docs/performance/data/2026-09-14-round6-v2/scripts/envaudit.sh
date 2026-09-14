#!/bin/bash
# Round-6 V2 env-switch audit: kernel-level evidence that the three arms really
# run the three code paths they claim to (not wall time).
set -u
source /workspace/deps/libuipc-src/env_perf.sh
OUT=/workspace/output/round6/v2/envaudit
REPO=/workspace/deps/libuipc-src
mkdir -p "$OUT"

prof () {  # $1 tag  $2 scene-dir  $3 frames  rest: env assignments
  local tag=$1 dir=$2 frames=$3; shift 3
  local p="$OUT/$tag"
  rm -f "$p".nsys-rep "$p"_cuda_gpu_kern_sum.csv
  ( cd "$REPO/libuipc-samples/examples/$dir" && \
    env "$@" /workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys profile -t cuda --cuda-graph-trace=node -f true -o "$p" \
      "$UIPC_PERF_PY" main.py --headless "$frames" > "$OUT/$tag.stdout" 2>&1 )
  /workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      -o "$p" "$p.nsys-rep" > "$OUT/$tag.stats.log" 2>&1
  if [ ! -s "$p"_cuda_gpu_kern_sum.csv ]; then
    echo "FAIL: no kern_sum csv for $tag"; return 1
  fi
  echo "ok $tag ($(wc -l < "$p"_cuda_gpu_kern_sum.csv) kernel rows)"
}

names () {
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

echo "### UIPC_CONTACT_RANK1 -- the three arms, on the tumbler"
prof rank1_0 95_tumbler_garments 12 UIPC_CONTACT_RANK1=0
prof rank1_d 95_tumbler_garments 12
prof rank1_1 95_tumbler_garments 12 UIPC_CONTACT_RANK1=1
for t in rank1_0 rank1_d rank1_1; do
  echo "--- $t:"; names "$OUT/${t}_cuda_gpu_kern_sum.csv" "do_assemble_kernel"
done
echo
echo "### the hinge kernel must be identical in all three (s02's default, untouched)"
for t in rank1_0 rank1_d rank1_1; do
  echo "--- $t:"; names "$OUT/${t}_cuda_gpu_kern_sum.csv" "DiscreteShellBending.*gradient_hessian"
done

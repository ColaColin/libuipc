#!/bin/bash
# Round-6 V2: the contact-severity micro-test sweep.
# Two arms (UIPC_CONTACT_RANK1=0 exact, =1 the change), interleaved, over a
# sweep of contact severity (press depth) and, separately, of sliding.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
SRC=/workspace/output/round6/v2/contact_micro.py
OUT=/workspace/output/round6/v2/micro
mkdir -p "$OUT"

NREP=${NREP:-5}
CONFIGS=${CONFIGS:-"base deep slide fastslide tight"}

# Tuned on smoke runs before the sweep: 6 layers x 441 vertices, the barrier the
# only load, vel-tol/tol-rate tight enough that Newton counts are a live signal
# (2-5 per frame, PCG 10-200) rather than pinned at the increment test's floor.
# The measured pair population in this scene is **PE 16 893 / PP 0** -- 100 % of
# contact part 2's pairs take the branch `UIPC_CONTACT_RANK1=1` approximates,
# which the tumbler (PE:PP about 3:1) does not achieve.
TOL="--vel-tol=0.0002 --tol-rate=1e-5"
cfg_flags () {
  case "$1" in
    base)  echo "--frames=80 --press=0.0090 --drag=0.000 --layers=6 --cells=20 $TOL" ;;
    deep)  echo "--frames=80 --press=0.0115 --drag=0.000 --layers=6 --cells=20 $TOL" ;;
    slide) echo "--frames=80 --press=0.0090 --drag=0.030 --layers=6 --cells=20 $TOL" ;;
    # the two axes the device probe says matter: sliding (the dropped
    # eigenvector is 96.6 % along the edge) and a tolerance tight enough that
    # the increment test stops masking a Newton-count difference.
    fastslide) echo "--frames=80 --press=0.0115 --drag=0.070 --layers=6 --cells=20 $TOL" ;;
    tight) echo "--frames=80 --press=0.0115 --drag=0.030 --layers=6 --cells=20 --vel-tol=0.00002 --tol-rate=1e-6" ;;
  esac
}

one () {  # $1 cfg  $2 arm  $3 rep
  local cfg=$1 arm=$2 rep=$3
  local tag="${cfg}_${arm}_${rep}"
  local log="$OUT/$tag.log"
  if [ -s "$log" ] && grep -q "^CONTACT frame=80 " "$log"; then echo "skip $tag"; return; fi
  echo "=== $tag  $(date +%H:%M:%S)"
  env UIPC_CONTACT_RANK1="$arm" UIPC_MICRO_WS="$OUT/ws_$tag/" timeout 1800 \
      "$UIPC_PERF_PY" "$SRC" $(cfg_flags "$cfg") --tag="$tag" > "$log" 2>&1
  echo "    rc=$? $(grep -c '^CONTACT ' "$log") frames"
}

for c in $CONFIGS; do
  for r in $(seq 1 "$NREP"); do
    if [ $((r % 2)) -eq 0 ]; then ORDER="1 0"; else ORDER="0 1"; fi
    for a in $ORDER; do one "$c" "$a" "$r"; done
  done
done
echo "MICRO DONE $(date +%H:%M:%S)"

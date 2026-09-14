#!/bin/bash
# Round-6 V2: the four-arm experiment.  One build at 83ea553c, one GPU.
# Arms interleaved within each rep, with the arm order rotated per rep, so the
# null (A vs A') is drawn from the same scene states as the effect (B) and no
# arm collects a systematic share of any drift in the box.
set -u
source /workspace/deps/libuipc-src/env_perf.sh
SC=/workspace/deps/libuipc-src/libuipc-samples/examples/95_tumbler_garments
OUT=/workspace/output/round6/v2/runs
mkdir -p "$OUT"
cd "$SC"

NREP=${NREP:-20}
DUMP_UPTO=${DUMP_UPTO:-6}    # dump positions for the first N reps of A/Ap/B

declare -a PERT=(
  "--perturb-yaw=1e-6"
  "--perturb-yaw=-1e-6"
  "--perturb-yaw=1e-6 --perturb-garment=1"
  "--perturb-vertex=1e-9"
  "--perturb-vertex=1e-9 --perturb-garment=3"
  "--perturb-yaw=2e-6"
  "--perturb-yaw=-2e-6 --perturb-garment=2"
  "--perturb-vertex=-1e-9"
  "--perturb-vertex=1e-9 --perturb-garment=1"
  "--perturb-yaw=1e-6 --perturb-garment=3"
)

one () {  # $1 arm  $2 rep
  local arm=$1 rep=$2
  local tag="${arm}_${rep}"
  local log="$OUT/$tag.log"
  if [ -s "$log" ] && grep -q "^VERIFY_RESULT" "$log"; then echo "skip $tag"; return; fi
  local -a flags=()
  local -a envs=()
  case "$arm" in
    A)  envs=(UIPC_CONTACT_RANK1=0) ;;
    Ap) envs=(UIPC_CONTACT_RANK1=0); read -r -a flags <<< "${PERT[$(( (rep-1) % 10 ))]}" ;;
    B)  envs=(UIPC_CONTACT_RANK1=1) ;;
    S)  envs=() ;;                       # the shipped default (Proj = 5)
  esac
  if [ "$arm" != "S" ] && [ "$rep" -le "$DUMP_UPTO" ]; then
    flags+=("--dump-positions=$OUT/$tag.npy")
  fi
  echo "=== $tag  env:${envs[*]-none}  flags:${flags[*]-none}  $(date +%H:%M:%S)"
  env "${envs[@]}" timeout 1800 "$UIPC_PERF_PY" main.py --headless 180 --verify \
      "${flags[@]}" > "$log" 2>&1
  echo "    rc=$? $(grep -c VERIFY_RESULT "$log") result lines"
}

for r in $(seq 1 "$NREP"); do
  case $(( (r-1) % 4 )) in
    0) ORDER="A Ap B S" ;;
    1) ORDER="Ap B S A" ;;
    2) ORDER="B S A Ap" ;;
    3) ORDER="S A Ap B" ;;
  esac
  for a in $ORDER; do one "$a" "$r"; done
done
echo "SWEEP DONE $(date +%H:%M:%S)"

#!/bin/bash
V=/workspace/output/round5/validation
PY=/workspace/deps/uipc-perf-env/bin/python
L=$V/audit.log
declare -A F=( [wb]=120 [cwc]=120 [c2]=250 [mb]=100 )
fp() { timeout 5400 $PY $V/fp.py "$@" >> $L 2>&1; }
echo "=== PHASE A: default arms n=5 ($(date -u +%H:%M:%S))" >> $L
for s in wb cwc c2 mb; do fp $s --tree head --frames ${F[$s]} --reps 5 --tag Ahead; done
for s in wb cwc c2 mb; do fp $s --tree base --frames ${F[$s]} --reps 5 --tag Abase; done
echo "=== PHASE A DONE ($(date -u +%H:%M:%S))" >> $L

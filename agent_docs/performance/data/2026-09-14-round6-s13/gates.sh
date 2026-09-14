#!/bin/bash
# s13 correctness gate in both arms (default = fitted grid, and the rollback).
set -u
O=/workspace/output/round6/s13
UIPC_SPMV_GRID_FIT=1 bash /workspace/output/round6/gate.sh > $O/gate_fit.txt 2>&1
UIPC_SPMV_GRID_FIT=0 bash /workspace/output/round6/gate.sh > $O/gate_off.txt 2>&1
echo "=== fit"; cat $O/gate_fit.txt
echo "=== off"; cat $O/gate_off.txt
echo "=== baseline"; cat /workspace/output/round6/baseline_tests.txt

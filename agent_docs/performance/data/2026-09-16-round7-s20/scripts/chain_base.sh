#!/bin/bash
# Chain: wait for matrix_head.sh (PID in arg) to exit, then run matrix_base.sh.
set -u
HPID=$1
while kill -0 "$HPID" 2>/dev/null; do sleep 20; done
echo "matrix_head exited $(date '+%F %T'); starting base arm" >> /workspace/output/round7/s20/driver.log
bash /workspace/output/round7/s20/matrix_base.sh >> /workspace/output/round7/s20/matrix_base.log 2>&1
echo "matrix_base done $(date '+%F %T')" >> /workspace/output/round7/s20/driver.log

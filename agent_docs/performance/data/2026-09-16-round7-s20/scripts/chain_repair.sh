#!/bin/bash
set -u
while kill -0 "$1" 2>/dev/null; do sleep 20; done
sleep 90   # let any dying-sanitizer GPU unavailability window clear
echo "### chain: base arm done, starting repair $(date '+%F %T')" >> /workspace/output/round7/s20/driver.log
bash /workspace/output/round7/s20/matrix_repair.sh >> /workspace/output/round7/s20/matrix_repair.log 2>&1
echo "### chain: repair done $(date '+%F %T')" >> /workspace/output/round7/s20/driver.log

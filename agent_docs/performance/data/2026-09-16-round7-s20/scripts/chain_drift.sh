#!/bin/bash
set -u
# wait for the repair chain to finish (marker line in driver.log), then drift, then envaudit
while ! grep -q "chain: repair done" /workspace/output/round7/s20/driver.log; do sleep 30; done
sleep 60
echo "### chain: starting drift B $(date '+%F %T')" >> /workspace/output/round7/s20/driver.log
bash /workspace/output/round7/s20/drift.sh > /workspace/output/round7/s20/drift.log 2>&1
echo "### chain: drift B done rc=$? $(date '+%F %T')" >> /workspace/output/round7/s20/driver.log
sleep 30
echo "### chain: starting envaudit C $(date '+%F %T')" >> /workspace/output/round7/s20/driver.log
bash /workspace/output/round7/s20/envaudit_s20.sh > /workspace/output/round7/s20/envaudit.log 2>&1
echo "### chain: envaudit C done rc=$? $(date '+%F %T')" >> /workspace/output/round7/s20/driver.log

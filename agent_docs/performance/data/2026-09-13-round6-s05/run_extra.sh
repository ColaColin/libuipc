#!/bin/bash
set -u
# wait for the verify chain to finish
while ! grep -q VERIFY_DONE /workspace/output/round6/s05/run_ab.log 2>/dev/null; do sleep 10; done
source /workspace/deps/libuipc-src/env_perf.sh
cd /workspace/deps/libuipc-src
# mas-bunny at n=5: the harness flagged n=3 for a ~1 % effect (PERF_METHOD 2.4)
$UIPC_PERF_PY /workspace/output/round6/ab.py --scene mas-bunny --n 5 --label s05b \
  --out /workspace/output/round6/s05/ab \
  --arm old UIPC_CCD_CULL=0 --arm new UIPC_CCD_CULL=1 \
  > /workspace/output/round6/s05/ab/ab_mas-bunny_n5.txt 2>&1 || echo "FAIL ab mas-bunny n5"
echo AB5_DONE
# and one nsys pair so mas-bunny's prediction is computable from its own kernel shares
cd /workspace/output/round6/s05
./nsysrun.sh mb_c0_r1 89_mas_bunny 100 UIPC_CCD_CULL=0 || echo "FAIL mb_c0_r1"
./nsysrun.sh mb_c1_r1 89_mas_bunny 100 UIPC_CCD_CULL=1 || echo "FAIL mb_c1_r1"
echo EXTRA_DONE

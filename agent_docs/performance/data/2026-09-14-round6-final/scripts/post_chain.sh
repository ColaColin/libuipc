#!/bin/bash
# V3: after the sweep -- the correctness gate on the head build, the env-switch audit, the verify probes. Serial, one GPU.
until grep -q CHAIN_DONE /workspace/output/round6/v3/ab_chain.log; do sleep 15; done
# (pgrep guard removed: it self-matched and hung -- the sweep is finished before this runs)
echo "gate start $(date +%H:%M:%S)"
bash /workspace/output/round6/gate.sh > /workspace/output/round6/v3/gate_head.txt 2>&1
echo "gate done $(date +%H:%M:%S)"
bash /workspace/output/round6/v3/envaudit_v3.sh > /workspace/output/round6/v3/envaudit.log 2>&1
echo "audit done $(date +%H:%M:%S)"
bash /workspace/output/round6/v3/probes_v3.sh > /workspace/output/round6/v3/probes.log 2>&1
echo "probes done $(date +%H:%M:%S)"
echo POST_DONE

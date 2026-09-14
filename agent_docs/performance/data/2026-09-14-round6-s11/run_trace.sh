#!/bin/bash
# s11: one full rwb run per arm under nsys, cuda_gpu_trace -> per-launch timeline
set -u
cd /workspace/output/round6/s11
for a in 0 1 2 3 4; do
  bash tracerun.sh rwb_a$a 6_wrecking_balls 120 UIPC_ABD_GH_PREPASS=$a || exit 3
done
echo TRACE_DONE

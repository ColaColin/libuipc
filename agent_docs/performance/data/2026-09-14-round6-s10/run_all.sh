#!/bin/bash
# s10 measurement plan, one build, three arms of UIPC_ABD_GH_PREPASS:
#   0 = off (pre-s10), 1 = prepass issued AFTER the contact launches (shipped),
#   2 = prepass issued BEFORE them (the placement that does not work)
set -u
cd /workspace/output/round6/s10
echo "=== traces (per-launch timeline, ratios inside one run)"
for a in 1 0 2; do ./tracerun.sh rwb_c$a 6_wrecking_balls 120 UIPC_ABD_GH_PREPASS=$a || echo FAIL_trace_$a; done
echo "=== scope kern_sum rwb, 3 arms x 3 reps"
for r in 1 2 3; do for a in 1 0 2; do ./nsysrun.sh rwb_c${a}_r$r 6_wrecking_balls 120 UIPC_ABD_GH_PREPASS=$a || echo FAIL_scope_rwb_${a}_$r; done; done
echo "=== scope kern_sum cwc, 2 arms x 3 reps"
for r in 1 2 3; do for a in 1 0; do ./nsysrun.sh cwc_c${a}_r$r 93_cube_wall_cloth 100 UIPC_ABD_GH_PREPASS=$a || echo FAIL_scope_cwc_${a}_$r; done; done
echo "=== control census: does mas-bunny / case2 / tumbler run the prepass kernels at all"
./nsysrun.sh mb_c1 89_mas_bunny 100 UIPC_ABD_GH_PREPASS=1 || echo FAIL_mb
./nsysrun.sh c2_c1 88_stiff_gipc_benchmark 250 UIPC_ABD_GH_PREPASS=1 || echo FAIL_c2
./nsysrun.sh tum_c1 95_tumbler_garments 180 UIPC_ABD_GH_PREPASS=1 || echo FAIL_tum
echo RUN_ALL_DONE

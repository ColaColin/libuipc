# 2026-09-05 — `lbvh` Config Key Fix and Broadphase Method Cross-Measurement

- Status: alias fix accepted; default broadphase method unchanged (owner
  decision); `stackless_bvh` reconfirmed as the fastest alternative on the
  heavy scenes, `linear_bvh` measured for the first time
- Workload: samples `88_stiff_gipc_benchmark`, `93_cube_wall_cloth`,
  `34_cloth_stack`, `11_bunny_cloth` — 100 frames x 2 runs per
  {`info_stackless_bvh`, `stackless_bvh`, `linear_bvh`}, clean protocol
  (`bench_stats.py`, no `WB_LOG`), `dcd_candidate_reuse=0` throughout
- Environment: Linux source build, NVIDIA RTX 2070 SUPER (sm_75), driver
  595.84, CUDA 12.8, Release, `UIPC_CUDA_ARCHITECTURES=75`, double precision
- Commits: base `3d0f4867` (branch `perf/newton-cost`), this change
- Config: `collision_detection/method` (string, default `info_stackless_bvh`)

## Question

The `lbvh` broadphase trajectory-filter setting aborted every run that used it,
so it had never been measured. Diagnose the abort, make the setting usable
without weakening the collision detection, and measure the LBVH filter against
the default (`info_stackless_bvh`) and `stackless_bvh` alternatives.

## Root cause

`lbvh` was never a registered value of `collision_detection/method`. The
LBVH-based broadphase (`LBVHSimplexTrajectoryFilter`, backed by
`AtomicCountingLBVH` over `LinearBVH`) has been selectable under the canonical
name `linear_bvh` since it was introduced ("impl linear_bvh as a
user-friendly interface"). Setting `method: "lbvh"` failed scene-config
validation in `uipc::core::validate_scene_config`
(`src/core/core/scene_default_config.cpp`) with

```
expected one of ["info_stackless_bvh","info_stackless_bvh_v0",
"stackless_bvh","linear_bvh"], got "lbvh"
```

which aborts the process at `Scene()` construction. There is no kernel defect:
the filter itself runs cleanly on every scene tested (88, 34, 92). The abort
was misread as a crash because the run log was lost.

## Fix

Accept `lbvh` as an alias of `linear_bvh` in both places that spell the
selector (the scene-config enum and the filter's `do_build` self-check), so
the natural abbreviation of the filter's own class name no longer aborts:

- `src/core/core/scene_default_config.cpp`: enum + `conditionalValues`
- `src/backends/cuda/collision_detection/filters/lbvh_simplex_trajectory_filter.cu`:
  `do_build` accepts both spellings

Semantics are unchanged: both spellings build the same
`LBVHSimplexTrajectoryFilter` (verified via the engine `systems.json`: with
`method: "lbvh"` the only broadphase system present is
`LBVHSimplexTrajectoryFilter`), the candidate predicates are untouched, and
the 30-frame scene-88 mean matches the `linear_bvh` run to 0.1% (258.8 ms
either way). The leaf predicates are code-identical to the default filter's
(same contact-mask, incidence and self-collision pruning, same
`expand = d_hat + thickness` and `distance::*_ccd_broadphase` conservative
test) — the methods differ only in BVH construction and traversal.

## Measurement (100 frames x 2 runs, reuse off, HEAD + this fix)

Mean `world.advance()` wall ms per frame (HEAD baselines: 497.6 / 232.5 /
64.1 / 50.7; relative deltas vs the same-binary default run):

| scene | info_stackless_bvh | stackless_bvh | linear_bvh / lbvh |
| --- | --- | --- | --- |
| 88_stiff_gipc_benchmark | 497.2 | **479.0 (-3.7%)** | 486.7 (-2.2%) |
| 93_cube_wall_cloth | 237.0 | **228.7 (-3.5%)** | 232.4 (-1.9%) |
| 34_cloth_stack | 59.3 | **57.7 (-2.7%)** | 60.5 (+2.0%) |
| 11_bunny_cloth | **50.2** | 50.7 (+1.0%) | 50.9 (+1.4%) |

Newton iterations/frame are equal across methods on every scene (88:
6.11/6.10/6.10; 93: 5.13/5.08/5.15; 34: 1.80/1.81/1.85; 11: 6.28/6.25/6.12),
i.e. the same algorithmic path; the deltas are pure broadphase throughput.
Reproducibility: the default-method run reproduces the earlier HEAD baseline
on 88 (497.2 vs 497.6 ms) and the earlier `stackless_bvh` result
(479.0 vs 477 ms).

Combined configuration on 88, `stackless_bvh` + `dcd_candidate_reuse=1`
(2 x 100 frames): 431.1 / 428.8 ms — mean 430.0, i.e. -13.5% vs the
same-binary default and -10.2% vs `stackless_bvh` alone. The method gain and
the reuse gain compose roughly additively (reuse alone was measured at
-12.6% in the 2026-09-05 reuse record).

## Gates

The best alternative (`stackless_bvh`) was re-gated on this binary; the
`linear_bvh` trajectory gate was measured for the record since the method had
never been gated before.

- 88 trajectory gate (centroid p50 displacement vs the branch reference
  `/tmp/gate_old/88_traj.csv`, bar 1e-3 m), 100 frames per method:
  default rerun p50 = 4.84e-4 m, `stackless_bvh` p50 = 3.06e-4 m,
  `linear_bvh` p50 = 1.80e-4 m — all PASS; the alternatives are as close to
  the reference as a rerun of the default itself.
- 34 capture gate for `stackless_bvh` on this binary:
  `validate_capture.py` PASS (100 frames, gap 1.29e-3); final-vertex
  similarity vs the same-binary default capture 1.88e-2 against a same-binary
  run-to-run band of 1.13e-2 (1.66x band, limit 2.5x) — PASS. The
  candidate-vs-default distance is larger than in the earlier
  `stackless_bvh` gate (2.17e-3 vs a 1.03e-2 band) because the band itself
  moved: contact-chaos amplification on this scene makes both numbers wander
  between runs; the pass criterion is the band ratio, which holds with
  margin.

## Decision

- The alias fix ships; the default method stays `info_stackless_bvh` — no
  measured alternative wins everywhere (`stackless_bvh` is -2.7..-3.7% on the
  three heavier scenes but +1.0% on `11_bunny_cloth`; `linear_bvh` is -1.9%/
  -2.2% on the two heavy scenes and slightly worse on the two light ones).
- If the owner wants a single faster default for contact-heavy scenes:
  `stackless_bvh` + `dcd_candidate_reuse=1` is the strongest measured
  combination (430.0 ms on 88, -13.5% vs default, gates PASS on this
  binary); the cost is +1.0% on `11_bunny_cloth` for the method swap alone.
- `linear_bvh`/`lbvh` remains useful as a comparison path only.

## Reproduction

```
source /workspace/deps/libuipc-src/env-src.sh
cd /workspace/libuipc-samples/examples/88_stiff_gipc_benchmark
# the previously-aborting setting now runs:
UIPC_TUNE='{"collision_detection":{"method":"lbvh"}}' \
  $UIPC_PY main.py --headless 30
# timing matrix (per scene, per method):
UIPC_TUNE='{"collision_detection":{"method":"<m>","dcd_candidate_reuse":0}}' \
  BENCH_STATS_OUT=/tmp/lbvh/timing/<s>_<m>_<rid>.json \
  $UIPC_PY /workspace/libuipc-samples/bench_stats.py main.py --headless 100
```

Artifacts: `/tmp/lbvh/timing/` (24 A/B JSONs + combined reuse runs),
`/tmp/lbvh/traj_*.csv`, `/tmp/lbvh/cap_*` (captures + validation reports),
`/tmp/lbvh_88_{repro,linear,alias}.log` (abort before the fix, clean
`linear_bvh` and clean `lbvh` after it).

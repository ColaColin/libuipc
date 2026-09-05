# 2026-09-05 — Certified DCD Broadphase Candidate Reuse Across Newton Iterations

- Status: Accepted behind `collision_detection/dcd_candidate_reuse` (default off)
- Workload: samples `88_stiff_gipc_benchmark`, `93_cube_wall_cloth`,
  `34_cloth_stack`, `11_bunny_cloth` (100-frame x 2 end-to-end A/B, clean
  protocol, no `WB_LOG`); 30/30/100-frame certification-verification runs with
  `WB_LOG=Info`
- Environment: Linux source build, NVIDIA RTX 2070 SUPER (sm_75), CUDA 12.8,
  double precision, default config otherwise
- Commits: base `82621df3` (branch `perf/newton-cost`), this change
- Config: `collision_detection/dcd_candidate_reuse` (flag, default 0);
  verification instrument `extras/debug/dcd_candidate_reuse_verify` (flag,
  default 0), implemented in
  `src/backends/cuda/collision_detection/candidate_reuse_verify.cu`

## Question

The DCD broadphase detection (AABB build + 3 BVH builds + 4 traversals with
exact-distance leaf tests, ~10.5 ms per Newton iteration on scene 88) re-runs
at the start of every Newton iteration past the first. Can iteration k reuse a
candidate set computed earlier in the same frame, with a *proof* that the
reused set contains everything a fresh detection at k would report? The
research-diagnostic oracle (`extras/debug/candidate_reuse_oracle`, commit
`196a549b`) measured the wall-time ceiling of such reuse at -13.2% on scene 88
/-10.6% on 93 but was flagged "not a certified superset".

## Method

The engine already runs a second, swept broadphase every Newton iteration:
`detect_trajectory_candidates(alpha)` in the line search sweeps every
primitive over `[x0, x0 + alpha_detect * dx]` (mandatory for the CCD TOI
filter, unchanged by this feature) and writes its result into the *same*
candidate buffers the DCD pass uses. The feature keeps that set for iteration
k instead of re-running the DCD detection, and lets `compute_energy`'s
`filter_active` inside the line search refresh the active set at the stepped
positions, exactly as before.

### Certification argument (why the reused set is a superset, unconditionally)

1. The trajectory leaf predicates accept a pair iff the two primitives' swept
   boxes, inflated per axis by the pair's `expand = d_hat + thickness`
   (`PP/PE/EE/PT_d_hat` + `*_thickness`), overlap on every axis — a
   per-axis relaxation of the exact distance test.
2. The line search only ever applies steps `alpha <= alpha_detect` (the TOI
   filter, the CFL filter, and the halvings all shrink it; `step_forward`
   computes `pos = safe_pos + alpha * disp`), and no other code moves
   positions inside a frame. Hence every vertex's position at iteration k is
   `x0 + alpha * dx` with `alpha in [0, alpha_detect]` — an interior point of
   the swept segment the trajectory detection used.
3. If a pair's exact distance at the current positions is `< expand` (the
   fresh DCD pass would report it), its closest points witness a per-axis gap
   `< expand` between the current boxes (Euclidean distance bounds every
   axis), and the swept boxes contain the current boxes, so their per-axis
   gap is also `< expand`: the pair passed the trajectory leaf predicate. The
   traversal visits it because each side's box inflation
   (`point/edge/triangle_dcd_expansion + thickness`) is arithmetically `>=`
   every pair's `expand` for all four pair types. Therefore
   **fresh set F_k is contained in reused set R_{k-1}** — no slack inflation,
   no motion budget, no re-detection trigger; the invariant holds by
   construction, not "while a budget lasts".
4. Active-set equality: every extra pair of R is `>= expand = thickness +
   d_hat` away, outside the active window `D_range = (thickness, thickness +
   d_hat]`, so `filter_active` drops it. `filter_active` is monotone in the
   raw set, so the active set handed to contact assembly at iteration k is
   *identical* to a fresh detection's (same pairs; only the buffer order can
   differ, which permutes floating-point reduction order — the same class of
   nondeterminism the engine already exhibits across runs).

The motion-budget/slack design from the research phase (inflate queries by
`m * d_hat`, re-detect when a vertex leaves its stored box) turned out to be
unnecessary: the mandatory CCD sweep *is* the slack, and it is already paid
for. The reused set is the swept box-overlap set, which on these scenes is
nearly iteration-invariant (it changes only when a vertex moves by a
nontrivial fraction of its inflation).

The engine-level gate also covers the legacy broadphase methods
(`info_stackless_bvh_v0`, `stackless_bvh`, `linear_bvh`): they run the same
inflated-box test at every alpha (no exact-distance branch), so the reused
swept-box set contains their fresh alpha=0 set a fortiori, and a pair only in
the reused set has a per-axis gap above `expand`, outside the active window —
the same active-set equality. The verification instrument, however, only
exposes the default method's raw candidate sets.

### Cost model

Both engine variants run `filter_active` over the large swept set inside the
line search every iteration; the feature only removes the per-iteration DCD
detection (AABB + BVH + traversal + exact-distance tests) and the small
fresh-set `filter_active` at the iteration head. The reused set being ~5-65x
larger than the fresh DCD set therefore adds no downstream work relative to
the default engine.

## Results

### Certification verification (extras/debug/dcd_candidate_reuse_verify)

Per reused iteration the instrument snapshots the reused set, re-runs the
fresh DCD detection, and checks containment on the host:

| scene | frames | reused iterations | with fresh > 0 | containment violations |
|---|---|---|---|---|
| 88_stiff_gipc_benchmark | 30 | 89 | 89 | **0** |
| 34_cloth_stack | 30 | 79 | 72 | **0** |
| 93_cube_wall_cloth | 100 | 410 | 309 | **0** |

Mean sizes on contact iterations (reused / fresh / ratio): 88 = 426,985 /
6,562 (65x), 34 = 155,497 / 3,795 (41x), 93 = 235,859 / 47,488 (5x). This is
the count-growth equivalent of the slack knob: the certified superset grows
5-65x over the fresh set, at zero extra downstream cost (see cost model).

### End-to-end wall time (100 frames x 2 runs, mean advance ms/frame)

| scene | default | reuse on | delta | branch HEAD ref |
|---|---|---|---|---|
| 88_stiff_gipc_benchmark | 489.9 / 497.8 | 434.3 / 428.9 | **-12.6%** (493.9 -> 431.6) | 497.6 |
| 93_cube_wall_cloth | 232.1 / 231.2 | 209.3 / 211.3 | **-9.2%** (231.7 -> 210.3) | 232.5 |
| 34_cloth_stack | 58.6 / 61.3 | 60.1 / 55.1 | -4.0% (60.0 -> 57.6, high variance) | 64.1 |
| 11_bunny_cloth | 49.8 / 51.0 | 45.3 / 45.3 | **-10.1%** (50.4 -> 45.3) | 50.7 |

Scene 88 lands on the oracle ceiling (434.3 ms in commit `196a549b`; 431.6
here). 34's single-run spread straddles zero (run 1: +2.6%, run 2: -10.1%);
its frames sit at 1-2 Newton iterations where the reuse window is small. No
scene regresses on the two-run mean.

### Mechanics (unchanged within run-to-run variance)

newton iters/frame and PCG iters/newton, default -> reuse:
88 = 6.08/6.11 -> 6.12/6.10 and 40.0 -> 40.3; 93 = 5.07/5.01 -> 5.06/5.08 and
39.1 -> 38.9; 34 = 1.79/1.86 -> 1.94/1.82 and 42.8 -> 42.5; 11 = 6.16/6.23 ->
6.25/6.20 and 25.9 -> 23.9. Same algorithmic path.

### Accuracy gates

- `validate_capture.py` on 100-frame reuse captures: `34_cloth_stack` PASS
  (gap 1.15e-3), `11_bunny_cloth` PASS (gap 1.42e-3).
- Similarity (final vertex distance) vs same-binary default runs, with the
  same-binary run-to-run band as reference: 34 = 7.54e-3 vs band 8.35e-3
  (0.90x, limit 2.5x); 11 = 2.54e-3 vs band 2.59e-3 (0.98x). Against the
  HEAD pristine captures: 34 = 7.00e-3 (this binary's default run: 9.71e-3),
  11 = 3.26e-3 (default: 2.99e-3) — the reuse run is as close to the
  reference as a rerun of the default binary.
- `88_stiff_gipc_benchmark` centroid trajectories vs the branch gate
  reference (`/tmp/gate_old/88_traj.csv`): reuse p50 = 2.16e-4 m, p90 =
  2.48e-3 m (bar ~1e-3 m on p50); the same-binary default run's own diff
  against that reference is p50 = 2.57e-4 m — the reuse run is
  indistinguishable from a rerun.

## Decision

Ship behind `collision_detection/dcd_candidate_reuse`, default off. The
certification is structural (holds for every frame and iteration, verified
470/470 contact-bearing iterations with zero violations), the active set is
provably identical to a fresh detection's, and the wall-time gain is at the
oracle ceiling on the primary scene. The default stays off for the owner to
decide; the measured case for flipping it: -12.6% / -9.2% / -10.1% /
-4.0% on 88 / 93 / 11 / 34 at unchanged iteration mechanics and
in-band accuracy. The slack/motion-budget variant was not implemented: the
sweep-based certificate makes it redundant (no budget to exhaust, no growth
knob to tune), which is also why there is no `m * d_hat` parameter.

The oracle flag `extras/debug/candidate_reuse_oracle` is superseded by this
feature (same hook, now with a proof); it remains for continuity of the
research record in `beam-physics/docs/`.

## Reproduction

```
source /workspace/deps/libuipc-src/env-src.sh
cd /workspace/libuipc-samples/examples/88_stiff_gipc_benchmark
# certification check (reuse + verify + Info log):
UIPC_TUNE='{"collision_detection":{"dcd_candidate_reuse":1},
            "extras":{"debug":{"dcd_candidate_reuse_verify":1}}}' \
  WB_LOG=Info $UIPC_PY /workspace/libuipc-samples/bench_stats.py main.py --headless 30
# end-to-end A/B (no WB_LOG):
UIPC_TUNE='{"collision_detection":{"dcd_candidate_reuse":0|1}}' \
  $UIPC_PY /workspace/libuipc-samples/bench_stats.py main.py --headless 100
```

Verification logs: `/tmp/reuse/verify{88,34,93b}.log`; timing JSONs:
`/tmp/reuse/timing/`; captures: `/tmp/reuse/cap/`; 88 trajectories:
`/tmp/reuse/88_{on,off}_traj.csv`.

# round 6 / s08 — contact part 1's reduced PSD projection: s31's basis-free form, extended to M = 4

Every file here backs one claim in the round record's s08 section.

## the pick — why the brief's two candidates were both set aside

| file | claim |
|---|---|
| `flagstats_all.txt` | **the census that decided the step.** Device-side count of the distance-flag dimension of every contact pair, four scenes, full runs. `PT[d2,d3,d4]` and `EEu[d2,d3,d4]` are **0, 0, all** in every scene — contact part 1 takes the `M = 4` branch of `make_spd_contact` in 100 % of its pairs, so s31's `Basis = 1` reduction (implemented for `M <= 3` only) had **no caller at all** there. It also carries `minD/dhat2`, the closest the scenes ever come to the barrier's singularity |
| `flagstats_kernel.cu.txt`, `flagstats.sh` | the census kernel (diagnosis only, `UIPC_CONTACT_FLAGSTATS=1`) and its runner. It is **not** in the shipped tree: it was built, run, and removed, so the committed translation unit carries no diagnostic code |

## numerics — the change is exact up to rounding

| file | claim |
|---|---|
| `basis4_probe.cu`, `build_probe.sh` | the probe: 1 000 000 randomised draws per pair type on the device, against the real `__device__` functions, in five families — generic, near-parallel edges / degenerate closest-point weights, pairs exactly at `d_hat`, gap driven to 1e-14 of `d_hat`, and primitives of 1e-7 edge length |
| `basis4_probe.txt` | the result. 1.14 M usable dim-4 samples. New vs shipped: **1.3e-15 mean / 6.3e-15 max** relative Frobenius error in four of the five families, exact symmetry (3.1e-16), **0 non-PSD samples**. The fifth family (`D/dHat^2` between 1e-36 and 1e-16) is where **both** reduced forms lose to a full `make_spd<12>` of the same Hessian — 3.9e-2 mean for the shipped one, 4.0e-2 for the new one — i.e. pre-existing, not introduced. `flagstats_all.txt` measures the closest any scene gets: `D/dHat^2 = 2.6e-3`, thirteen orders of magnitude above it, and the probe's clean families cover down to 1e-10 |

## the old arm reproduces main's binary

| file | claim |
|---|---|
| `sass_identity_summary.txt` | **71 / 71** SASS streams of the translation unit byte-identical to main `4cc97465`, including the part-1 instantiation the benchmarks launch (`<0,1,1,1,0,0>`, 30 954 instructions). Read this one first — it explains why the shipped source needs the body-hash match |
| `sass_identity_prefinal.txt` | the same comparison before the default flip, matched by name: 71/71, 3 new functions |
| `sass_identity.txt` | the shipped source, matched by SASS body hash (the anonymous-namespace mangling embeds a source hash, so names move): 21 by name + 50 by body = 71/71 |
| `sass_diff.sh`, `sass_cmp.py` | how both were produced |
| `res_usage_part1.txt` | registers **255 in every arm** — the cc 7.5 ceiling, so both arms launch at the same geometry and nothing here is an occupancy artefact. Stack 10 456 -> 9 496 B |
| `part1_static.txt` | static instruction / FP64 / local-memory counts per instantiation: 30 954 -> 28 274 (-8.7 %), FP64 1908 -> 1640 (-14.0 %), LDL 329 -> 92 (-72 %) |

## performance

| file | claim |
|---|---|
| `predictions.txt` | **written before any end-to-end run** (PERF_METHOD 2.9). Predicted cube-wall-cloth -0.72 to -0.88 % ms/Newton, case2 -0.35 to -0.43 %, and rigid-wrecking-balls *unresolvable* |
| `scope_all.txt` | per-launch µs of part 1 and part 2, three arms, four scenes, n=3 full runs each; plus the critical-path table (GPU-kernel-sum / wall and which of K9's two streams is longer — **part 1 in all four scenes**) |
| `scope_norm_{cwc,c2,rwb,tum}.txt` | the same, with part 1 normalised by untouched kernels of other families measured in the same run (the round's tumbler lesson). cube-wall-cloth and case2 are clean instruments (untouched kernels move <= +1.2 %); rigid-wrecking-balls and the tumbler are not |
| `nsys/*.csv`, `nsys/wall_per_run.txt` | the raw `cuda_gpu_kern_sum` of all 42 profiled full runs, and the wall time of each |
| `ab/ab_*.log`, `ab/summary_*.json`, `ab/raw_runs_*.json` | end-to-end A/B, one build, ABBA + discarded warm-up: cube-wall-cloth n=40, case2 n=10, rigid-wrecking-balls n=8 |
| `scope.py`, `norm.py`, `critpath.py`, `run_scope.sh`, `nsysrun.sh` | the analysis and the runners |

## gates and observables

| file | claim |
|---|---|
| `gate_b0.txt`, `gate_b1.txt`, `gate_default_on.txt` | `gate.sh` in all three arms — rollback, explicit opt-in, and the shipped default — assertion counts identical to `baseline_tests.txt`: 11/3, 1112/36, 2730/46, 100/3, 4/1, 448/23, **14213/95**, pytest 48 passed 1 skipped |
| `default_selects.txt` | env-switch audit at **kernel level**: with no env var part 1 launches `Proj = 8` and part 2 keeps s07's `Proj = 5`; with `UIPC_CONTACT_SPD1_BASIS=0` part 1 is back on `Proj = 0` |
| `verify_stats.txt`, `verify/*.json` | tumbler `--verify`, 180 frames, n=8 per arm. `verify_ok`, `all_finite`, radial and axial containment true in all 16 runs; 0 non-converged, 0 `hit_newton_limit`, 0 `hit_line_search_limit` frames; every non-penetration observable overlapping. **`verify_area_ratio_max`, the round's watch statistic, moves DOWN** (1.383 -> 1.288) — the opposite direction to the three Hessian approximations that moved it |

# round 6 / s13 — the SpMV+dot grid was sized from the raw triplet count

Every file here backs a claim in `agent_docs/performance/2026-09-13-perf-round6.md`, section **s13**.

| file | claim it backs |
|---|---|
| `predictions.txt` | P1-P5, written before any A/B arm ran. P1 (scope) held; **P2 (end to end) did not** -- it predicted -1.1 to -1.3 % on case2 and -0.5 to -0.6 % on `mas-bunny` and got -0.97 % and -0.05 %; **P3 was refuted** (rwb is not over-launched at this head); P4 held (3 distinct grid values on case2, 1 on `mas-bunny`) |
| `kernsum_mb_head.txt`, `kernsum_c2_head.txt` | the family re-ranked at this head with `UIPC_SPMV_GRID_FIT=0` (= main's geometry). SpMV is **28.76 % of `mas-bunny`** and **18.20 % of case2**; the three MAS kernels are 26 % / 13.7 % |
| `scope_gridfit.txt` | per-launch SpMV cost and the per-session grid shape, four scenes x {capacity grid, fitted grid} plus round 5's grid-stride, one build, full runs, nothing else on the box |
| `results_c2.txt`, `results_c2b.txt`, `pooled_c2.txt` | the two independent `stiff-gipc-case2` A/Bs and their pooled n=16 statistics: **-0.97 % mean (p=3.0e-06), -0.93 % ms/Newton (p=2.9e-07)**, counts flat to 0.05 % |
| `results_mb.txt`, `null_mb.txt` | `mas-bunny` at n=10 (**-0.05 %, p=0.42**) against the null arm's +0.10 % envelope at n=6 -- the fitted grid is worth nothing on that scene |
| `results_cwc.txt`, `results_rwb.txt` | the other two scenes: flat, and rwb's wall is unreadable by the harness's own iteration guard |
| **`instrument_check.txt`** | the step's second result: **nsys inflates the capacity-grid arm by 1.29 pp (`mas-bunny`) and 0.57 pp (case2) more than the fitted arm**, against a scope-to-wall shortfall of 0.78 / 0.65 pp. `kern_sum` per kernel is a ranking instrument, not a wall forecast for a launch-geometry change |
| `r6_stride_c2.txt` | round 5's R6 `UIPC_SPMV_GRID_STRIDE` re-measured at this head: -0.32 % mean, **-0.55 % ms/Newton (p=0.0089)** -- real but half the fitted grid, and not order-preserving |
| `r7_fold_mb.txt` | round 5's R7 `UIPC_PCG_FOLD=1 UIPC_PCG_FOLD_MAXGRID=0`: **+0.14 % mean (p=0.046), +0.18 % ms/PCG (p=0.0087)** -- a resolvable regression; rejection confirmed |
| `sass_identity.txt` | all three touched TUs compiled at head and at `0f276920` with the identical command line and diffed per function: **0 changed SASS instructions** (22/8248, 43/18528, 57/20664 functions/instructions) |
| `gate_fit.txt`, `gate_off.txt` | `gate.sh` vs `baseline_tests.txt`, identical counts in both arms |
| `ab/*_summary.json` | the `ab.py` summaries for every sweep (per-run raw json not archived: 124 files, 9 MB) |

Scripts: `nsysrun.sh` (kern_sum, fresh prefix per run, fails loudly), `scope.py` (per-launch cost +
grid shape read from the same session), `sass_check.sh`, `sweep.sh`, `sweep2.sh`.

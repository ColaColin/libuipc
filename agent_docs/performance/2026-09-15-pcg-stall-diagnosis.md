# The one-frame PCG stall on `stiff-gipc-case2` — reproduced and classified

**Date:** 2026-09-15 · **Head:** `diag/pcg-stall-trace` (main + two default-off probes) ·
**Box:** local RTX 2070 SUPER · **Evidence:** `/workspace/output/pcg-stall/` (curated: `event1/`)

Round 6's record carried, under "found outside", a one-frame linear-solver stall on
`stiff-gipc-case2` (~1 run in 48; frame 150 taking 36 640 PCG iterations over 7 Newton
iterations against ~300 in siblings). TODO.md asked for a reproduction and a classification:
preconditioner breakdown, line-search interaction, or genuinely stiff configuration.

**Answer: preconditioner breakdown.** The system matrix at the broken solve is SPD and easy —
plain unpreconditioned CG solves it in 809 iterations — while the MAS-preconditioned solver
oscillates on it. Details and the excluded alternatives below.

## The event

Three sightings exist in the historical record, all on round-6-era trees:

| run | frame | frame PCG | flavour |
|---|---|---|---|
| s14 null_r11 (2026-09-14) | 150 | 36 640 | converged ~100× slow |
| s15 c2 u16_r1 (2026-09-14) | 172 | 12 235 | converged slow |
| this investigation, run 80 (2026-09-15) | 79 | 255 941 | **hit the iteration cap, unconverged** |

The new sighting was caught by a 140-run sweep with two default-off probes
(`UIPC_PCG_TRACE`, `UIPC_PCG_STALL_DUMP`, see commits `abd3d66c`/`2ac0538f`): a per-solve /
per-host-check trace of the block-replay loop, and an automatic A/b dump when a solve
exceeds an iteration threshold. Rates: 1 event in ~140 runs today; 3 in ~278 post-s13 full
runs; **0 in ~350 pre-s13 full runs** — see "s13" below.

Per-solve view of the event frame (from the trace):

```
frame 79, newton 0..6:  rz0=+1.56e-2 (35 it)  +2.96e-3 (40)  -1.27e-2 (25)
                       -1.44e-6 (255666, UNCONVERGED)
                       +1.90e-3 (40)  +1.13e-3 (20)  +2.35e-4 (115)
```

Exactly one solve breaks; the neighbouring frames and the frame's other Newton iterations
are normal; Newton counts (7), line-search trials (9) and convergence are unchanged from
siblings. The run-level effect is one frame at 8.9 s against ~0.18 s.

## What it is not

- **Not the iteration cap being "reached by a stiff system".** The cap is
  `max_iter = 2n = 255 666` at n = 127 833 DOF (2×19 193 bunny nodes + 4 225 cloth
  vertices × 3). The s14/s15 sightings (36 640 / 12 235) are ~14 % / ~5 % of one cap:
  those solves *converged*, slowly. My sighting did hit the cap — but see below, the
  system itself was not the problem.
- **Not a line-search interaction.** Line-search trials and Newton counts are identical
  to sibling runs at the event frames; the event lives entirely inside one linear solve.
- **Not an unsolvable/stiff system.** The dumped matrix of the broken solve
  (`A.79.3.mtx`, auto-captured by the probe): SPD, symmetry defect 7e-18, λ_min
  ≈ +2.6e-4, λ_max ≈ 1195 against a normal solve's ~4.1 — κ(A) ≈ 4.6e6 (normal: 4e4).
  **Plain double-precision CG, no preconditioner, solves this exact system in 809
  iterations** (normal system: 484). The stiff ladder (1195, 244, 127, 107, 85, 83 …)
  is localized in the upper bunny (6 hot vertices for λ_max) — a transient local
  compression/contact configuration. The preconditioner made a solvable system ~300×
  worse than no preconditioner at all.
- **Not the level-0 float Gauss-Jordan inversion losing PSD.** The cluster inversion is
  float32, fixed pivot order, no pivoting (`invert_cluster_matrices_sweep_kernel`) — but
  replicating that inversion offline on 4 168 surrogate 16-vertex clusters built from the
  dumped matrices leaves **every** inverse PSD, on the stalled matrix as well (worst
  cluster κ = 7.7e5, an order of magnitude short of where float GJ flips).
- **Not a dump/reconstruction artifact.** `bcoo_A` is block-upper-triangle at 3×3
  granularity (the SpMV is `rbk_sym_spmv_dot`); a scalar-level `A+Aᵀ−D` doubles each
  diagonal block's own off-diagonal entries and *manufactures* negative eigenvalues.
  The block-aware reconstruction is what the GPU computes.

## What it is

The trace of the broken solve shows the textbook signature of PCG on an **indefinite
preconditioned operator**: `rz0 = r0ᵀz0 < 0` (an SPD P⁻¹ makes this strictly positive;
here the *applied* preconditioner is indefinite on r0), and `rz` cycles with period
~30–60 iterations, sign negative on 50.8 % of 51 133 samples, no monotone trend.

Two coupled pathologies, both visible in the frame-79 table above:

1. **The MAS apply is indefinite on r0 at this frame** — at newton 2 (`rz0 = −1.27e-2`)
   *and* newton 3. At newton 2 PCG coped (25 iterations). Where the indefiniteness
   nearly annihilates r0 (newton 3: |rz0| ≈ 1.4e-6 against ~1e-3 typical) …
2. **… the relative tolerance becomes a trap**: `rz_tol = 1e-4·|rz0| ≈ 1.4e-10` is a
   near-absolute target. The iteration's best was |rz| = 2.14e-10 — within 1.5× of the
   tolerance — then it wandered off and cycled to the cap.

So the s14/s15 "slow but converged" flavour and my "cap" flavour are the same event at
different severity: a transiently-stiff-but-benign SPD system on which the preconditioner's
float multi-level apply loses PSD, with the damage sometimes small enough to still converge.

## Where the breakdown lives (open)

Level-0 cluster inversion and float storage of the inverses are exonerated (above). The
indefiniteness enters through the **multi-level float pipeline** (restriction /
coarse-level accumulations / prolongation) or its interaction with the coarse Galerkin
clusters, whose conditioning is not covered by the level-0 surrogate test. The engine's
own `UIPC_MAS_APPLY_VERIFY` modes cannot see it: they compare the pipeline against itself
and never run inside captured graph replays, which is where the PCG iterations execute.

Fix directions were ranked; **the detect-and-re-solve guard shipped the same day**
(commit `8849bfa6`, see below). Localizing the multi-level stage remains optional
hardening (retro-capture cluster dump on a fresh event, or offline replication of the
coarsening).

## The shipped guard (2026-09-15, commit 8849bfa6)

`LinearFusedPCG::fused_pcg` now checks the solve-start `rz0`: **`rz0 < 0`, or `rz0`
non-finite with a finite `r`** (the case `check_init_rz_nan_inf` used to abort on — the
round-6 tumbler "preconditioner NaN" abort flavour) → redo the solve's initialization
with the preconditioner bypassed (`z = r`, plain CG; `x` is still the all-zero start, so
only `z`, `p` and the rz scalars are redone), run the plain-launch path (no graph replay,
the captured graph embeds the preconditioner launches), warn once with frame/newton. A
non-finite `r` (assembly NaN) still aborts as before. `UIPC_PCG_PSD_FALLBACK=0` is the
rollback; `UIPC_PCG_PSD_FALLBACK_TEST=k` forces the bypass every k-th solve (test arm);
the trace probe's S/E lines gained a `bypass=` field. The full-GPU while-loop mode
(graph_mode 2) has no host read of the initial rz and is not covered.

Validation:

- **Gate identical to `baseline_tests.txt` in all suites** (11/3, 1112/36, 2730/46,
  100/3, 4/1, 448/23, 14213/95, pytest 48+1).
- **Forced arm** (`UIPC_PCG_PSD_FALLBACK_TEST=37`, 12-frame case2 run): 2/39 solves
  bypassed, both converged (90 / 200 iterations against a median 45 for MAS solves),
  0 unconverged, all frames converged; the one Newton-count shift on a bypassed frame is
  the same rounding-level trajectory variation the scene shows between any two runs.
- **A/B, all five benchmark scenes, n=6/arm, ABBA + warm-up** (`off` =
  `UIPC_PCG_PSD_FALLBACK=0` vs `on` = default; raw runs in
  `/workspace/output/round6/psdfb/`): every statistic overlapping, no regression beyond
  any scene's own envelope. `mas-bunny` — the deterministic control — at **Newton 465 /
  line search 465 in all 12 runs of both arms** (this repo's standard proof that a change
  computes nothing differently), PCG totals ±0.03 %. ms/PCG (trajectory-insensitive):
  mb +0.65 %, cwc −0.64 %, case2 −0.63 %, rwb −1.06 % (p=0.10), tumbler −0.54 % — all
  overlapping. ms/Newton: mb +0.68 % (p=0.53), cwc −0.14 % (p=0.93), case2 +0.16 %
  (p=0.80), rwb −0.59 % (p=0.33), tumbler −0.64 % (p=0.81). The count guards fired on
  cwc/rwb/tumbler with movements (+0.5…+2.4 %) inside those scenes' recorded run-to-run
  count envelopes (round 6: tumbler PCG ±4 %, rwb counts ±1 %+, single-run wall
  134–169 ms); the healthy path adds no device work and no computed value differs — the
  drift is the scenes' own non-determinism. No fallback triggered naturally in the 60
  measured runs (case2 PCG totals all within the normal 63–65 k band).

## Measurement hazard (why round 6 noticed it)

1 event in ~40–140 runs inflates a pooled mean enough to require exclusion (s14's null arm
was reported as +0.11 % ms/Newton *excluding* the stall run; including it, the arm's mean
moves by the event's ~8 s against a ~50 s run). **With the shipped guard the damage is
bounded** — a triggered event costs one ~800-iteration plain-CG solve (~+0.8 % of a case2
run's PCG total) instead of 12 k–256 k iterations. For future sweeps: a per-frame
`linear_solver_iterations > 2000` check on case2 still flags event runs cheaply, and the
solver's own warning line (`preconditioner action not SPD/finite`) names them exactly.

## Relation to s13 (unresolved)

All three events are on trees containing s13; zero in ~350 pre-s13 full runs; but s13's
own ~54-run case2 sweeps were clean, as were s16/s17/v3 (~65 runs). 3/278 ≈ 1.1 % post vs
<0.85 % (95 % bound) pre — suggestive, not conclusive. s13's behavioural changes inside
the MAS loop (coarse-tail seeding/re-zeroing) were code-reviewed against the stale-state
theory and look covered (`set_preconditioner` re-zeroes per Newton; `collect_final_Z`'s
tail zeroing precedes its `converged` gate), but nothing is proved.

## Instruments landed (default off)

- `UIPC_PCG_TRACE=<path>` — per-solve S/E lines and per-host-check C lines (frame,
  newton, iter, rz) from the block-replay loop. Host reads only; numerics unchanged.
- `UIPC_PCG_STALL_DUMP=<iters>` — dump that solve's A and b (`_dump_A_b`'s writer) when a
  solve exceeds the threshold. `bcoo_A`/`b` persist until the next Newton assembly, so a
  post-solve dump captures the stalled system exactly.

Analysis scripts (trace parser, mtx analyzer, float-GJ replication):
`/workspace/output/pcg-stall/{parse_trace.py,analyze_dump.py,test_float_gj.py}`.
Curated evidence for the event: `/workspace/output/pcg-stall/event1/` (trace, bench JSON,
A/b dump of the broken solve); normal reference: the `A.6.2`/`b.6.2` pair in the scene's
workspace `debug/` folder.

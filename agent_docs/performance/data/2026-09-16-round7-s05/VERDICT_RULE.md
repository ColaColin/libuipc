# Round-7 s05 validation — the verdict rule, fixed before any arm was run

Committed before the first arm of the three-arm experiment executes, so the analysis cannot be
fitted to the answer (round-6 V1 protocol, PERF_METHOD §2). Subject: **s03's dahl Gauss-Newton
Hessian** (`UIPC_DAHL_GAUSS_NEWTON`, default on = `<3,1>`), the only search-direction change that
shipped this round. Its dropped-term relFro (median 0.857 vs the shipped exact arm) is far larger
than the plain hinge's was when round-6 V1 validated that one — which is exactly why this pass
exists.

## The arms

One build: engine `0545931f` (= main head, s04 merged; engine tree identical to pre-s04 main),
binary `libuipc_backend_cuda.so` sha256 `c2a104b6…` recorded in `binary_hashes_start.txt`.
Scene: `103_crease_press` at samples `a47ba7d` (the s05 opt-in commit; default path proved
byte-stable, see `byte_stability.txt`). Every run is the full default 130 frames with
`--verify --dump-positions --result`.

| arm | env | scene | n |
|---|---|---|---|
| **A** | `UIPC_DAHL_GAUSS_NEWTON=0` (exact dahl Hessian) | default | 10 |
| **A''** | `UIPC_DAHL_GAUSS_NEWTON=0` | **perturbed initial state** (5 distinct seeds, below) | 5 |
| **B** | unset (Gauss-Newton, the shipped default) | default | 10 |

`s01`/`s02` knobs are left at their defaults in every arm (both shipped default-on and both are
search-direction-neutral projections; the A arm = `<1,1>` + exact Hessian is the exact physics).

The five A'' seeds (all on the exact path; verified to touch exactly the intended sheet/vertex,
`byte_stability.txt`):

| run | flag | initial footprint |
|---|---|---|
| A''1 | `--perturb-vertex=1e-6 --perturb-sheet=0` | one free denim vertex, +1e-6 m in y |
| A''2 | `--perturb-vertex=-1e-6 --perturb-sheet=0` | same, −1e-6 m |
| A''3 | `--perturb-vertex=1e-5 --perturb-sheet=0` | one free denim vertex, +1e-5 m |
| A''4 | `--perturb-yaw=1e-5 --perturb-sheet=0` | denim sheet yawed 1e-5 rad (corner ≈ 4e-6 m) |
| A''5 | `--perturb-yaw=1e-5 --perturb-sheet=3` | a sheet-metal sheet yawed 1e-5 rad |

d_hat is 2.0 mm, the denim edge 11 mm, the layer gap 7.3 mm; every seed is ≤ 5e-3 of the smallest
physical length in the problem. Two seeds sit on dahl sheets deliberately, so the control's seed
enters through the same constitution the change under test modifies.

**R** = pooled A ∪ A'' (n = 15): physically equivalent runs — same physics, same solver, same code
path — differing only by perturbations far below any physical scale, plus the GPU-reduction-order
seed the scene supplies for free (its PCG count is non-deterministic from frame 0, s00). **T** = B
(n = 10).

**Widen-only rule (pre-registered):** B's seed size is unknown before the first measurement. If
the median CROSS frame-1 rms divergence exceeds 3× the largest EQUIV frame-1 pair rms, up to five
further A'' runs are added with initial perturbations sized to bracket B's seed — seeds are only
ever added, never removed, and every addition is stated in the results. (Round-6 V1 had to do this
after the fact as A''; here it is in the rule.)

## Safety items — pass/fail, not statistical

A **fail** is anything that occurs in B and not in R:

1. any non-finite position (`verify_all_finite`);
2. containment broken: `contained_x` / `contained_z` false, or `verify_y_min` below the die plane;
3. triangle inversion: `verify_area_ratio_min ≤ 0`, or sustained collapse
   (`tri_height_no_sustained_collapse` false);
4. any frame with `converged == 0`, `hit_newton_limit`, or `hit_line_search_limit`
   (`verify_not_converged_frames`, `verify_hit_newton_limit_frames`, `verify_hit_ls_limit_frames`);
5. `verify_max_speed` above the scene's 50 m/s audit bound;
6. Newton divergence over the run (`no_newton_divergence` false).

**Regime checks** (`dahl_state_evolved`, `plastic_yielded`, `residual_creases`, and the composite
`verify_ok`) are reported but pre-declared **knife-edge**: s04 documented the exact arms flipping
`plastic_yielded` (strain 0.0 % one draw) and `no_inversion_or_collapse` (area-min 0.252 another)
run to run. The fail condition for a knife-edge check is: false in ≥ half of B's runs while true in
every run of R.

## Statistical observables (per run; distributions, not single runs)

Geometry/energy: `area_ratio_min`, `area_ratio_max`, `tri_height_min`, `max_speed`,
`mean_disp_mm`, `mean_disp_mm_last_quarter`, `ke_last_quarter`, `max_defl_mm`,
`abs_x_max`, `abs_z_max`, `y_min`, residual crease depth per sheet (7 values; also their min).
Constitution: `dahl_active_frac_final`, `dahl_mean_absF_over_M_final`,
`dahl_crease_absF_over_M_final`, `strain_yield_frac_final`, `stress_yield_frac_final`,
`plastic_mean_abs_theta_bar_final`, `crease_theta_max`.
Solver: `newton_total`, `pcg_total`, `line_search_total`, `newton_first_quarter`,
`newton_last_quarter`, `ccd_toi_min`, frames with `last_line_search_alpha < 1`,
frames with `last_ccd_toi < 1`.

## The rule

1. **Envelope**: `[min(B), max(B)]` must intersect `[min(R), max(R)]` and `median(B)` must lie
   inside R's range.
2. **Shift**: two-sided Mann–Whitney of B vs R is called **systematic** only if **both**
   `p < 0.05` (raw) and `|median(B) − median(R)| > 0.5 × range(R)`. Holm-adjusted p reported
   alongside. With ~30 observables, p < 0.05 alone happens by chance; a move smaller than half
   the null's own range is not separable from chaotic wander.
3. **Spread**: `range(B)/range(R)` reported; a widened spread alone is **not** a bias (chaotic
   divergence widens spread symmetrically). It escalates only with a rule-2 median shift.
4. **Trajectory divergence** — the decisive instrument. Per frame `f = 0..130`,
   `rms(P_i − P_j)` over all 16 774 sheet vertices for every run pair. Families:
   within-A (45 pairs), within-A'' (10), A-vs-A'' (50) — pooled as **EQUIV** (105);
   **CROSS** = A-vs-B (100) + A''-vs-B (50); within-B (45) reported separately.
   B passes iff all four hold:
   - **(i) seed**: median CROSS rms at frame 1 ≤ 3 × max EQUIV frame-1 rms (else the widen rule
     fires before anything else is read);
   - **(ii) growth rate**: e-folding per frame over the log-linear growth window (fitted on
     frames 3–15 of each family's median curve), ratio CROSS/EQUIV within [0.5, 2.0];
   - **(iii) saturation**: mean rms over the last 40 frames, CROSS/EQUIV within [0.5, 2.0];
   - **(iv) frame-wise containment**: from frame 8 to 130 the CROSS median curve stays ≤ 2 × the
     EQUIV p95 curve at every frame (frames 1–7 are the seed-entry window, where round-6 V1's
     larger seed also sat above the envelope for 2 frames by seed size alone).
5. **Direction, pre-declared**: "worse" has a known sign for `area_ratio_max` ↑, `tri_height_min` ↓,
   `max_speed` ↑, `ke_last_quarter` ↑, `strain/stress_yield_frac_final` ↑ (a degraded solve presses
   deeper creases — s04's signature), `newton_total` ↑. One-sided tests reported for these with the
   same 0.5·range effect-size floor.

## Verdict

- **physics unchanged** — no safety fail, no observable meets rule 2, and all four divergence
  conditions hold.
- **systematic bias found** — any safety fail, or any rule-2 failure, or any divergence condition
  broken. Reported with the numbers either way.

If a verdict is marginal at the planned n, n is raised rather than the conclusion rounded, and the
raise is stated (round-6 V1's own rule).

## Scope decisions pre-declared

- **No separate crease-severity micro-test** (round-6 V1's `crease_micro.py`): the crease-press
  scene itself drives hinges through |δ| > 1 states in every press frame (s04's own account of
  where the plastic GN failed), and `crease_theta_max` is recorded per run — the severity sweep is
  the scene, not a side case.
- **Count honesty**: s03 claimed Newton +0.17 % at n=16. This pass re-measures newton/pcg/LS totals
  at n=10 vs the R envelope; any replicated shift is reported as the correction, round-6 style,
  whatever its sign.
- **Probe composition** (instrument checks, not part of the verdict): one full run with
  `UIPC_DAHL_GN_VERIFY=1` at the default (GN) setting — the probe must still work at this head and
  the gradient must still be bit-identical (0 mismatching words); one 12-frame nsys capture of the
  A arm proving the launched dahl instantiation is `<1,1>` (the switch really selects the exact
  path), mirroring round-6 V1's `envaudit.txt`.

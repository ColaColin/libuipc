# Round-6 V1 validation — the verdict rule, fixed before any arm was run

Committed before the first arm of the three-arm experiment executed, so the analysis
cannot be fitted to the answer (PERF_METHOD §2, VALIDATION_BRIEF "state the rule first").

## The three arms

| arm | build | scene | n |
|---|---|---|---|
| **A** | one build at `7316bbc5` | `tumbler-garments`, default, `UIPC_DSB_GAUSS_NEWTON=0` | 5 |
| **A'** | same build | same scene, `UIPC_DSB_GAUSS_NEWTON=0`, **perturbed initial state** (5 distinct, physically meaningless perturbations) | 5 |
| **B** | same build | same scene, default (`UIPC_DSB_GAUSS_NEWTON` unset = Gauss-Newton) | 5 |

`UIPC_CONTACT_RANK1` is left **unset** in every arm (s03's default-off exact contact path).

The five A' perturbations (`main.py` opt-in flags, default path byte-stable — proved):

| run | flag |
|---|---|
| A'1 | `--perturb-yaw=1e-6` (towel, +1e-6 rad ≈ +3e-7 m at the corner) |
| A'2 | `--perturb-yaw=-1e-6` (towel) |
| A'3 | `--perturb-yaw=1e-6 --perturb-garment=1` (pillowcase) |
| A'4 | `--perturb-vertex=1e-9` (towel vertex 0, +1 nm in x) |
| A'5 | `--perturb-vertex=1e-9 --perturb-garment=3` (washcloth vertex 0) |

## Reference distribution and test set

**R** = the pooled per-run observable values of **A ∪ A'** (n=10). Every member of R is a
physically equivalent run of the same system: same physics, same solver, same code path,
differing only by a perturbation far below any physical scale in the scene (the smallest,
1 nm, is 1.3e-6 of `d_hat`; the largest, 3e-7 m, is 5e-5 of an element edge).

**T** = arm B (n=5).

Pooling A and A' is deliberate: the scene is *already* not bit-reproducible run to run
(GPU reduction order), so within-A scatter is itself a perturbation experiment with a
~1 ulp seed. A' adds a second, deliberately larger seed. R therefore spans the seed range
1e-16 … 3e-7 m, and if B's divergence is chaotic it must fall inside what that range produces.

## Safety items — pass/fail, not statistical

A **fail** is anything in B that does not also occur somewhere in R:

1. any non-finite position;
2. `verify_contained_radial` or `verify_contained_axial` false (a vertex leaves the bore);
3. `verify_lifter_depth_max > 0` (a vertex inside a lifter box);
4. `verify_area_ratio_min <= 0` (triangle inversion) or `verify_tri_height_min` below the
   `2r + d_hat = 3.27 mm` floor;
5. `verify_cc_min_dist <= 0` or `verify_bore_gap_min <= 0` (interpenetration);
6. any frame with `converged == 0`, `hit_newton_limit`, or `hit_line_search_limit`;
7. `verify_max_speed` above the scene's 50 m/s audit bound.

## Statistical observables (per run; distributions, not single runs)

`area_ratio_max`, `area_ratio_min`, `tri_height_min`, `cc_min_dist`, `bore_gap_min`,
`r_max`, `abs_z_max`, `lifter_depth_max`, `max_speed`, `ke_mean`, `e_tot_last`,
`mean_disp_mm`, `mean_disp_mm_last_quarter`, `centroid_turn_deg` (4 garments),
`drum_track_err_deg_max`, `drum_track_err_deg_final`, `newton_total`, `pcg_total`,
`line_search_total`, `ccd_toi_min`, `ccd_toi_clamped_frames`, `ls_alpha_cut_frames`,
and the per-frame **active contact-pair count** (PT+EE+PE+PP) summed and averaged per run.

## The rule

For each observable X:

1. **Envelope**: B's `[min, max]` must intersect R's `[min, max]`, **and** `median(B)` must
   lie inside R's `[min, max]`.
2. **Shift**: two-sided Mann–Whitney U of B vs R. A shift is called **systematic** only if
   **both**
   - `p < 0.05` (raw), **and**
   - `|median(B) − median(R)| > 0.5 × range(R)`.

   Both are required. With n=5 vs n=10 over ~25 observables, p<0.05 alone is expected about
   once by chance; and a location move smaller than half the null range is not separable
   from chaotic wander. Holm-adjusted p is reported alongside the raw p.
3. **Spread**: `range(B) / range(R)` and a Brown–Forsythe test. **A widened spread alone is
   not a bias** — chaotic divergence widens spread symmetrically. It is reported, and it is
   escalated only if it is accompanied by a one-directional median shift (rule 2).
4. **Trajectory divergence**: per frame, `rms(P_i − P_j)` over cloth vertices for every run
   pair. Three families: within-A, A-vs-A' (both physically equivalent → the envelope) and
   A-vs-B. **B passes if the A-vs-B curve lies inside the envelope of the physically
   equivalent families over the whole run**, in particular if it saturates at the same level.
5. **Direction, pre-declared**: for six observables "worse" has a known sign —
   `area_ratio_max` ↑, `tri_height_min` ↓, `cc_min_dist` ↓, `bore_gap_min` ↓,
   `drum_track_err_deg_max` ↑, `ke_mean` ↑ (spurious energy). These also get a one-sided
   test in the bad direction, reported separately, and the same 0.5·range effect-size floor.

## Verdict

- **physics unchanged** — no safety fail, every observable passes rules 1–4, and the
  divergence envelope of rule 4 contains A-vs-B.
- **systematic bias found** — any safety fail, or any observable that fails rule 2 (both
  conditions), or an A-vs-B divergence curve outside the physically-equivalent envelope.

If any observable's verdict is marginal at n=5/arm, n is raised rather than the conclusion
rounded, and the raise is stated.

## Convergence-quality micro-test (separate, pre-declared)

The dropped term is `E'(θ)·∇²θ`, which scales with `|θ − θ̄|`. A strip of cloth is folded to a
progressively sharper crease over a sweep of imposed crease severity, and Newton counts,
line-search trials and convergence are recorded for A vs B **as a function of severity**.
Pre-declared reading: if Gauss-Newton degrades, Newton counts must rise with severity in B
relative to A, with the gap growing monotonically. If no such regime is reachable, that is
stated as a negative result with the range of `|θ − θ̄|` covered.

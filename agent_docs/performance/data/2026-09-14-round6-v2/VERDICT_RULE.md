# Round-6 V2 validation — the verdict rule, fixed before any arm was run

Committed **before the first arm of the four-arm experiment executed**, so the analysis cannot be
fitted to the answer (PERF_METHOD §2, V1's `VERDICT_RULE.md` at `ab5adcf6` is the model).

**What is being decided**: whether `UIPC_CONTACT_RANK1=1` (contact part 2, the closed-form rank-1
Hessian on **both** the PE and PP branches) may become the default. It is merged, default off, and
worth −3.23 % ms/Newton on `cube-wall-cloth` (s07, n=40, predicted then measured). The shipped
default is `Proj = 5` (PP only), which is exact up to rounding; `Proj = 1` adds the **PE** branch,
which keeps rank 1 of the exact projection's rank 2 and is therefore an approximation
(‖H_proj − H_rank1‖/‖H_proj‖ = 9.78e-05 mean, **1.900e-02 max**, s07's device probe).

**The blocker** is one statistic: `verify_area_ratio_max` on `tumbler-garments`. It has moved
**up** under three independent Hessian approximations (V1/s02's Gauss-Newton hinge, s03's contact
modes 1 and 2, s07's mode 1: 1.324 → 1.491, max 2.029, p = 0.17 at n = 8) and **down** under one
exact change (s08, 1.383 → 1.288, same n = 8). n = 8 cannot read it.

**Standing instruction from the owner**: the change is acceptable *only if the simulation stays
stable and correct and behaves as physically as before*. Throughput does not buy fidelity. A
recommendation **not** to ship is a successful outcome of this pass.

---

## 1. What `area_ratio_max` actually measures, and why the instrument had to change first

`verify_area_ratio_max` is `max over 181 frames of (max over ~19 900 garment triangles of
A_tri / A_tri_rest)`. It is a **double extreme value**: one triangle, in one frame, of one run. Its
run-to-run scatter is therefore governed by the tail of a chaotic per-frame series, not by the
membrane's state, and a Welch or Mann–Whitney test on run maxima is a weak instrument for it — which
is exactly how the round got to three sightings without a decision.

The physically meaningful question is **whether the membrane is being stretched further**, not
whether one triangle in one frame hit a transient. So `verify.py` was extended (audit only — nothing
it computes is read by the simulation) to record, **per frame**, the whole distribution behind the
max:

| per-frame field | what it is |
|---|---|
| `ar_q50`, `ar_q99`, `ar_q999` | quantiles of `A/A0` over all ~19 900 triangles |
| `ar_mean` | mean of `A/A0` |
| `ar_total` | `sum(A) / sum(A0)` — the **bulk** areal stretch of the whole membrane |
| `ar_n_gt_1p2 / 1p4 / 1p6` | number of triangles above each threshold |
| `ar_argmax_garment`, `ar_argmax_tri` | which garment and which triangle attains the frame max |
| `area_ratio_max` (already present) | the frame max |

and, per run, `verify_ar_q999_mean/max`, `verify_ar_q99_mean/max`, `verify_ar_mean_mean`,
`verify_ar_total_mean/max`, `verify_ar_n_gt_1p{2,4,6}_total`, `verify_ar_frames_gt_1p4`,
`verify_ar_argmax_frame`, `verify_ar_argmax_garment`.

**The verdict is judged primarily on the membrane statistics, not on the run maximum.** That choice
is made here, in advance, for the stated reason, and the run maximum is still reported and still
tested — see §5.

## 2. The arms

One build at `83ea553c` (= `main`), one GPU, all arms interleaved in one sweep so the null is
measured beside the effect (s07's lesson). Scene `tumbler-garments`, `--headless 180 --verify`.
`UIPC_DSB_GAUSS_NEWTON` unset in every arm (s02's Gauss-Newton hinge is the shipped default and
V1 cleared it).

| arm | env | scene | n |
|---|---|---|---|
| **A** | `UIPC_CONTACT_RANK1=0` | default | 20 |
| **A′** | `UIPC_CONTACT_RANK1=0` | **perturbed initial state**, 10 distinct physically meaningless perturbations × 2 | 20 |
| **B** | `UIPC_CONTACT_RANK1=1` (**the change under test**) | default | 20 |
| **S** | unset (= `Proj 5`, **the shipped default**) | default | 20 |

**R** (the reference) = A ∪ A′ pooled, n = 40. Every member of R is the same physics, the same
solver and the same code path, differing only by a perturbation of 1e-9 m … 3e-7 m — 1e-6 to 1e-4 of
`d_hat` (1.27 mm) and of the element edge (6.2 mm). Within-A scatter is itself a perturbation
experiment with a ~1 ulp seed (V1's correction to s00), so R spans the seed range 1e-16 … 3e-7 m.

**T** = B, n = 20. **S is a positive control**: it is exact up to rounding (s07 proved the PP closed
form analytically and at 4.75e-16 mean relative Frobenius error over 140 470 device samples), so it
*must* pass. **Pre-registered: if S trips any rule that B trips, the rule is over-sensitive and B's
trip is not evidence of a physics change** — that is the instrument check, and it is declared now
rather than invoked later.

A′'s perturbations (`main.py` opt-in flags added by V1, default path byte-stable):
`--perturb-yaw=±1e-6`, `±2e-6`, `--perturb-vertex=±1e-9`, each on one of the four garments; the same
ten V1 used, each run twice.

## 3. Safety items — pass/fail, not statistical

A **fail** is anything in B that does not also occur somewhere in R:

1. any non-finite position;
2. `verify_contained_radial` or `verify_contained_axial` false;
3. `verify_lifter_depth_max > 0` (a vertex inside a lifter box);
4. `verify_area_ratio_min <= 0` (triangle inversion), or `verify_tri_height_min` below the
   `2r + d_hat = 3.27 mm` physical reference **and** below R's own minimum;
5. `verify_cc_min_dist <= 0` or `verify_bore_gap_min <= 0` (interpenetration);
6. any frame with `converged == 0`, `hit_newton_limit`, or `hit_line_search_limit`;
7. `verify_max_speed` above the scene's 50 m/s audit bound;
8. `verify_ok` false in any run.

Plus, outside the tumbler: `gate.sh` identical to `baseline_tests.txt` with `UIPC_CONTACT_RANK1=1`,
including `sim_case`'s **95** contact-covering cases.

## 4. The primary test — the membrane family

Four statistics, declared as **the family the verdict turns on**:

`verify_ar_q999_mean`, `verify_ar_q99_mean`, `verify_ar_mean_mean`, `verify_ar_total_mean`.

These are means over 181 frames of a quantile or a bulk sum over ~19 900 triangles — location
statistics, not extreme values, so they can be read at n = 20.

Each is converted to **areal strain** `ε = X − 1` (0 = unstretched) before the effect size is taken,
because a ratio of ratios near 1 hides the size of the physical effect.

**Systematic (→ do not ship)** if, for any member of the family:

- two-sided Mann–Whitney U of B vs R has **Holm-adjusted p < 0.05** within this family of four, **and**
- `median(ε_B) / median(ε_R) − 1 > +0.10` — the median areal strain is **more than 10 % larger** in
  the stretching direction.

Both are required. The 10 % relative-strain floor is set now, from the physics rather than from the
data: the garments are strain-limited (`StrainLimitingBaraffWitkinShell`, `strain_rate = 100`), so a
tenth of the membrane's own working strain is the smallest change that could plausibly matter to the
material model, and anything below it is not separable from chaotic wander at any affordable n.

A **one-sided** version of the same test (stretching direction only) is reported alongside, since
"worse" has a known sign here.

## 5. The secondary test — the run maximum itself, three ways

`verify_area_ratio_max` is still tested, with its extreme-value nature handled explicitly:

1. **Run maxima**: two-sided Mann–Whitney B vs R, plus the envelope rule (median(B) inside R's
   [min, max]; ranges intersect). Reported with the acknowledgement that this is the weak instrument.
2. **Pooled per-frame frame-max distribution** (20 runs × 181 frames = 3 620 frames per arm):
   compare q50 / q95 / q99 / q99.9 / max, and Mann–Whitney on the pooled per-frame values.
   A membrane shift must show at the **median and upper quantiles together**; a transient shows only
   at the extreme tail.
3. **Tail rate**: the fraction of frames with frame-max above **1.4** (fixed now, before looking:
   V1 measured 0.17–0.22 % of frames above 1.4 on all three of its arms, so it is a live threshold
   with counts large enough to test). Two-sided Fisher exact / binomial on B vs R frame counts.

**The run maximum alone cannot fail the change.** It escalates only if condition 2 or 3 is
significant **and** the primary membrane family (§4) moves in the same direction: i.e. the run
maximum is treated as confirmatory evidence for a membrane shift, never as a verdict on its own.
This is declared now because the opposite — letting a single 2.029 decide — is what has held the
change for three steps without resolving it.

## 6. Confirmatory observables (reported, and part of the joint signature)

`verify_mean_disp_mm_last_quarter` (s07 saw it move with the area ratio: 15.54 → 17.04, p = 0.10),
`verify_area_ratio_min`, `verify_tri_height_min`, `verify_cc_min_dist`, `verify_bore_gap_min`,
`verify_r_max`, `verify_max_speed`, `verify_ke_mean`, `verify_e_tot_last`, `verify_mean_disp_mm`,
`verify_centroid_turn_deg` (4), `verify_drum_track_err_deg_max`, `verify_newton_total`,
`verify_pcg_total`, `verify_line_search_total`, `verify_ccd_toi_min`,
`verify_ccd_toi_clamped_frames`, `verify_ls_alpha_cut_frames`.

Screened with V1's rule for comparability — **systematic** only if raw p < 0.05 **and**
`|median(B) − median(R)| > 0.5 × range(R)` — with Holm-adjusted p reported across the whole screen.
A widened spread alone is **not** a bias (chaotic divergence widens spread); the bias signature is a
one-directional median shift.

**Joint signature (→ do not ship)**: `verify_mean_disp_mm_last_quarter` up **and** any membrane
statistic up, both with raw p < 0.05, even if neither alone clears its own threshold. That pairing
is what s07 flagged, and it is the one combination this pass was called to resolve.

## 7. Cost side (reported, not part of the pass/fail)

Newton / PCG / line-search totals per arm on the tumbler, and a re-measurement of the
`cube-wall-cloth` Newton-count drift at this head (s08 landed in the same kernel family since s07's
n = 50 sweep). Pre-registered reading: s07 bounded the drift to [−0.135, +0.207] % on case2; if
`cube-wall-cloth` Newton counts at this head move more than **±2 %** between `=0` and `=1` at n ≥ 20,
the cost claim is no longer the one s07 recorded and must be re-stated.

## 8. The contact-severity micro-test (the part that generalises past our scenes)

V1's `crease_micro.py` isolates the *bending* Hessian and is run against this change too, unchanged,
as a null: the contact Hessian is not exercised there (contact is disabled), so **B must be
indistinguishable from A on it**. If it is not, the env switch or the build is wrong.

The analogous targeted case for a **contact** Hessian needs contact severity, not bending severity.
The dropped term is `B'(D) · hess(D)` on the **PE** branch, and s07's probe says the rank-1 form's
error is governed by the second eigenvalue the exact projection keeps: the error is
9.78e-05 on average but reaches **1.9e-02**. A device sweep locates which PE geometry maximises it
(a function of `D/d_hat` and of where the point's foot falls on the edge), and a scene micro-test is
then built to drive contact into that regime, sweeping severity and recording Newton counts,
line-search trials and convergence for A vs B **as a function of severity**.

**Pre-declared reading**: if the rank-1 PE Hessian degrades, Newton counts must rise in B relative to
A as contact severity rises, with the gap growing monotonically. If no such regime is reachable, that
is stated as a negative result with the range of severity covered. A degradation regime that is
*reachable in a physically sensible configuration* is on its own grounds to recommend against
shipping, independently of the tumbler.

## 9. Verdict and sample size

- **SHIP** — no safety fail; the membrane family shows no systematic upward shift (§4); the joint
  signature (§6) is absent; the run-maximum evidence (§5) does not escalate; the micro-test finds no
  reachable degradation regime; the gate is identical; and the cost claim holds (§7).
- **DO NOT SHIP** — any safety fail in B not in R; or any membrane statistic meets §4; or the joint
  signature of §6; or a reachable degradation regime in §8; or a gate difference.
- **MARGINAL** — if any §4 statistic has Holm-adjusted p < 0.05 with a strain shift between +5 % and
  +10 %, or p between 0.05 and 0.15 with a shift above +10 %, **n is raised to 30 and then 40 per
  arm** rather than the conclusion being rounded, and the raise is stated in the report.

Because PE carries 93 % of mode 1's win and PP (already shipping) carries 7 %, a verdict of
**"PE-only is not safe, PP-only already ships, stop here"** is an available and complete outcome.

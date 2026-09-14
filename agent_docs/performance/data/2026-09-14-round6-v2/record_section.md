## V2 — the validation pass on `UIPC_CONTACT_RANK1=1`: measuring the statistic that blocked it, at a sample size that can read it

Evidence: `agent_docs/performance/data/2026-09-14-round6-v2/`
(`VERDICT_RULE.md` — **committed at `877d0cf9`, before the first arm ran** — `README.md` maps every
file onto the claim it backs).
Scene changes are in the `libuipc-samples` submodule on `perf/round6-v2` (`fa5c3b9`).

**Verdict: TBD**

### What was being decided, and why n = 8 could not decide it

`UIPC_CONTACT_RANK1=1` replaces contact part 2's exact PSD projection with the closed-form rank-1
Hessian `c gradD gradD^T` on **both** the PE and the PP branch. PP is exact up to rounding and
already ships (`Proj = 5`, s07); the PE half is a genuine approximation — the exact projection keeps
rank 2 of 9, the closed form keeps rank 1 — and it carries **93 %** of mode 1's −3.23 % ms/Newton on
`cube-wall-cloth`.

s07 measured out the Newton drift that had held it back ([−0.135, +0.207] % on `stiff-gipc-case2` at
n = 10; the Newton tail rate identical to the exact path's, Fisher p = 1.0). What remained was one
statistic, `verify_area_ratio_max` on the tumbler, with four sightings and no decision:

| step | change | `verify_area_ratio_max` | n |
|---|---|---|---|
| s02 / V1 | Gauss-Newton hinge Hessian (approximation) | 1.240 → 1.288, one run 1.519 | 3–4 |
| s03 | contact mode 1 (approximation) | 1.251 → 1.308; mode 2 → 1.448 | 3 |
| s07 | contact mode 1 (approximation) | 1.324 → 1.491, max **2.029**, p = 0.17 | 8 |
| s08 | basis-free M = 4 reduction (**exact**) | 1.383 → **1.288** (down) | 8 |

Three approximations moved it up and one exact change moved it down by about the same amount, at the
same sample size. That is the signature of a statistic nobody can read, not of a physics effect.

### The instrument had to change before the sample size did

`verify_area_ratio_max` is `max over 181 frames of (max over ~18 300 garment triangles of A/A0)` — a
**double extreme value**. A test on run maxima is the weakest instrument there is, and it is the one
every previous sighting used.

`verify.py` now records, per frame and audit-only (it runs after the stopwatch is read and nothing it
computes is fed back), the whole distribution the max is the tail of: quantiles of `A/A0` over all
triangles (`ar_q50/q99/q999`), the mean, the **bulk** areal stretch `sum(A)/sum(A0)`, threshold
counts, and which garment and triangle attains the frame max. **The verdict rule fixed in advance
turns on the membrane statistics, not on the run maximum**, and says so with its reason; the run
maximum is still tested and can only ever be confirmatory.

### The theory side, measured on the device before any scene ran

`pe_severity_probe.cu`, `pe_severity.txt`: the real `__device__` functions, 1 265 702 usable PE
samples on a structured sweep of the normalised gap `g = sqrt(D)/(xi + dHat)` across five decades, of
the foot position along the edge, and of edge length against gap.

**What the rank-1 PE form drops is the tangential direction, and it drops it only where there is
nothing to drop.**

| `g` (1 = the activation distance) | rel. error mean | rel. error max | λ₁ median [κ/d̂²] | λ₂ max [κ/d̂²] |
|---|---|---|---|---|
| 1e-5 … 3.2e-5 (deep contact) | **2.7e-06** | 2.6e-05 | 5.7e-01 | 1.4e-01 |
| 1e-4 … 3.2e-4 | 2.6e-05 | 2.6e-04 | 4.7e-03 | 1.0e-02 |
| 1e-2 … 3.2e-2 | 2.6e-03 | 1.1e-01 | 1.8e-07 | 4.9e-05 |
| 3.2e-1 … 1.0 (barely active) | 5.5e-02 | **1.0000** | **3.9e-11** | 1.4e-04 |

The relative error grows toward the **activation boundary**, not toward deep contact, and the
absolute stiffness falls nine to eleven decades faster than the error grows. Stated as one number:

- **samples with relative error > 1e-2: 151 425 of 1 265 702 (12.0 %);**
- **of those, samples whose dropped eigenvalue λ₂ exceeds 1 % of a real contact's leading eigenvalue:
  0 (0.0000 %).**
- Restricted to **load-bearing** pairs (λ₁ within two decades of a real contact, 26.8 % of samples):
  relative error **9.08e-05 mean, 1.53e-02 max**, and **λ₂/λ₁ ≤ 1.52e-02**.

And the direction: resolving the dropped eigenvector into the relative motion of the point against
its foot on the edge, over the whole population it is **96.6 % along the edge — the slide — and
2.4 % along the contact normal**. The approximation never softens penetration resistance; it softens
tangential curvature, by at most 1.5 % of the leading eigenvalue wherever the pair carries load.

That is a mechanism, and it is testable: if this change has a physical channel at all, it is sliding,
which is why the micro-test below has a sliding axis and why `mean_disp_mm_last_quarter` is in the
joint signature.

### Trajectory divergence — supplementary, and it inverts s02's picture

`divergence.txt`, 18 position dumps (A, A′, B × 6), per frame `rms(P_i − P_j)` over the 9 396 cloth
vertices, 153 run pairs. **This was not part of the pre-registered rule** (V1 already established the
envelope method for this scene); it is reported because it is cheap and because it answers "how big a
kick is this change?" directly.

| pair family | n pairs | frame 1 | frame 3 | frame 8 | frame 45 | mean over the last 45 frames |
|---|---|---|---|---|---|---|
| A–A (two identical exact runs; seed = GPU reduction order) | 15 | 9.74e-10 m | 6.69e-9 | 1.19e-2 | 8.67e-2 | 0.2706 |
| A′–A′ | 15 | 5.64e-8 | 5.77e-8 | 9.71e-3 | 1.76e-1 | 0.3156 |
| A–A′ | 36 | 5.16e-8 | 5.21e-8 | 1.05e-2 | 1.14e-1 | 0.2854 |
| **all physically equivalent (EQUIV)** | **66** | | | | | **0.2889 [0.072, 0.426]** |
| **exact vs rank-1 (CROSS)** | **72** | **1.69e-9** | 9.43e-9 | 9.69e-3 | 8.36e-2 | **0.2583 [0.056, 0.443]** |
| B–B | 15 | 1.16e-9 | 7.71e-9 | 8.96e-3 | 6.37e-2 | 0.2163 |

**`UIPC_CONTACT_RANK1=1` is a 1.7-nanometre perturbation of the first step** — *smaller* than the
5.2e-8 m median seed of the deliberate perturbation arm, and the same order as the 9.7e-10 m seed two
*identical* exact runs get for free from GPU reduction order. The CROSS/EQUIV seed ratio at frame 3 is
**0.18**, i.e. a head start of **−0.70 frames** of a 180-frame run: the change enters the exponential
*later* than a 1 nm nudge does. That is the opposite of s02's Gauss-Newton hinge, which V1 measured at
1.83e-7 m and 158× the free seed.

The CROSS *median* curve leaves the EQUIV [min, max] envelope at **1 of 181 frames** (frame 4, where
the coherent Hessian difference has not yet been overtaken by chaos); it is inside from frame 5 on and
saturates at 0.258 m against the equivalent pairs' 0.289 m.

### The four arms, n = 20 each, 80 full 180-frame runs (14 480 frames)

One build at `83ea553c`, one GPU, arm order rotated per rep so the null is drawn from the same scene
states as the effect. `analysis_n20.txt`.

| arm | env | what it is | n |
|---|---|---|---|
| A | `UIPC_CONTACT_RANK1=0` | exact projection on PE and PP | 20 |
| A′ | `=0` + `--perturb-*` | exact, ten physically meaningless perturbations (1e-9 m … 3e-7 m) | 20 |
| **B** | `=1` | **the change under test**: rank-1 on PE **and** PP | 20 |
| S | unset (`Proj 5`) | the shipped default: rank-1 on PP only, exact up to rounding | 20 |

**R** = A ∪ A′ (n = 40). **S is a pre-registered positive control**: it is exact up to rounding, so it
must pass whatever B is asked to pass.

### §3 Safety — pass/fail, and B is the only arm with nothing on its record

| item | A | A′ | **B** | S |
|---|---|---|---|---|
| non-finite positions | 0 | 0 | **0** | 0 |
| radial / axial containment failures | 0 / 0 | 0 / 0 | **0 / 0** | 0 / 0 |
| vertex inside a lifter | 0 | 0 | **0** | 0 |
| triangle inversion (`area_ratio_min ≤ 0`) | 0 | 0 | **0** | 0 |
| interpenetration (`cc_min_dist` or `bore_gap_min` ≤ 0) | 0 | 0 | **0** | 0 |
| non-converged frames of 3 620 | 0 | 0 | **0** | 0 |
| `hit_newton_limit` / `hit_line_search_limit` | 0 | 0 | **0** | 0 |
| `max_speed` ≥ 50 m/s | 0 | 0 | **0** | 0 |
| **`verify_ok` false** | 0 | 0 | **0** | **1** |

Worst value anywhere in each arm — B is not the extreme on a single one:

| statistic | A | A′ | **B** | S |
|---|---|---|---|---|
| `r_max` (limit 0.30227) | 0.298874 | 0.298841 | **0.298766** | 0.298759 |
| `lifter_depth_max` (< 0 = outside) | −0.917 mm | −0.977 mm | **−1.026 mm** | −1.030 mm |
| `area_ratio_min` | 0.6907 | 0.6373 | **0.7096** | 0.6123 |
| `tri_height_min` | 3.119 mm | 3.162 mm | **3.166 mm** | **2.180 mm** |
| `cc_min_dist` | 2.319 mm | 2.331 mm | **2.322 mm** | 2.296 mm |
| `bore_gap_min` | 0.496 mm | 0.640 mm | **0.594 mm** | 0.639 mm |
| `max_speed` | 4.42 m/s | 4.58 m/s | **3.90 m/s** | **13.36 m/s** |

**No safety item fails in B, and B is not the worst arm on any of them.** The one `verify_ok`
failure in 80 runs is on the **shipped default**, and it is explained below.

### §4 The membrane family — the verdict turns on this, and nothing is systematic

Areal strain ε = X − 1. The pre-registered fail condition is Holm-p < 0.05 **and** a median strain
shift above **+10 %**. 20 000-resample bootstrap CI on the shift (`membrane_ci.txt`):

| statistic | median ε, R (×1e3) | median ε, B (×1e3) | B shift | 95 % CI | p | Holm |
|---|---|---|---|---|---|---|
| `ar_q999_mean` | +20.86 | +21.28 | **+1.99 %** | **[−2.50 %, +7.47 %]** | 0.40 | 1.00 |
| `ar_q99_mean` | +10.76 | +10.97 | +1.95 % | [−3.59 %, +6.87 %] | 0.49 | 1.00 |
| `ar_mean_mean` | −0.290 | −0.262 | −9.69 % | [−60.7 %, +96.2 %] | 0.74 | 1.00 |
| `ar_total_mean` (bulk stretch) | −0.299 | −0.270 | −9.77 % | [−60.2 %, +85.3 %] | 0.72 | 1.00 |

**Not one member meets the fail condition, and the whole 95 % CI of the statistic that matters most
sits below the fail line.** For scale, the null of the null — A against A′, which differ only by a
perturbation seed *larger* than the one the rank-1 Hessian injects — moves `ar_q999_mean` by
**+1.34 %** and `ar_q99_mean` by **+4.49 %**. B's +1.99 % is the size of what seed change alone does.

Two of the four have a median R strain that is slightly *negative* (the membrane is on average very
slightly compressed), which makes the rule's relative effect size degenerate for them; the absolute
shifts are +2.8e-5 and +2.9e-5 and are reported instead. The rule was not changed for this.

### The positive control fired, and it is the most important methodological result of this pass

`S` differs from `R` **only by rounding** on the PP branch — s07 proved the closed form is the exact
eigenvalue analytically and measured 4.75e-16 mean relative Frobenius error over 140 470 device
samples. It should be a null. It is not:

| statistic | S shift vs R | 95 % CI | p |
|---|---|---|---|
| `ar_q99_mean` | −8.04 % | [−14.39 %, −3.52 %] | **0.011** |
| `ar_mean_mean` | −118.9 % | [−154.9 %, −50.3 %] | **0.00030** |
| `ar_total_mean` | −115.3 % | [−147.3 %, −49.0 %] | **0.00036** |

**A rounding-level change shifts these statistics further, and far more significantly, than the
approximation under test does.** The direction is safe (less membrane strain, bulk stretch nearer
zero) and the size is tiny in absolute terms (+3.4e-4 in bulk areal strain, consistent with the
closed form returning the analytic eigenvalue where the iterative solve returns a slightly smaller
one — i.e. very slightly stiffer contact). But the reading that matters for the round is:

> **At n = 20 on this scene, statistical significance on a membrane statistic does not mean the
> physics changed.** The pre-registered rule anticipated exactly this and required an effect size as
> well as a p-value. Had the rule been "p < 0.05 ⇒ reject", the *shipped, exact* default would have
> been rejected and the approximation would have passed.

### §5 `verify_area_ratio_max` itself — B has the *tightest* distribution of the four arms

The statistic that blocked this change for three steps, at n = 20 (confirmatory only, by the rule):

| arm | median | [min, max] | frames > 1.4 (of 3 620) | frames > 1.6 | frames > 1.8 |
|---|---|---|---|---|---|
| A | 1.3146 | [1.154, **3.699**] | 7 (0.193 %) | | |
| A′ | — | [1.167, 2.142] | 8 (0.221 %) | | |
| **R = A+A′** | 1.3146 | [1.154, **3.699**] | 15 (0.207 %) | 4 | 3 |
| **B** | **1.3095** | **[1.170, 1.596]** | **4 (0.110 %)** | **0** | **0** |
| S (shipped) | 1.3352 | [1.163, **74.04**] | 8 (0.221 %) | 5 | 2 |

B vs R: Mann–Whitney p = 0.71, one-sided (up) p = 0.65, median inside R's range, **range ratio
B/R = 0.167**. The tail rate is *lower* in B at every threshold. The eight largest run maxima in the
whole pass all belong to A, A′ or S; **B's largest of 20 runs is 1.596**.

Run-level frame-max quantiles (each run contributes one value, so the runs are independent):

| | R median | B median | p |
|---|---|---|---|
| frame-max q50 | 1.03478 | 1.03472 | 0.57 |
| frame-max q95 | 1.10791 | 1.11466 | 0.21 |
| frame-max q99 | 1.19191 | 1.20662 | 0.12 |
| frame-max max | 1.31458 | 1.30946 | 0.71 |

The pooled-per-frame Mann–Whitney the rule also asked for returns p = 0.0098 (B) and 0.00028 (S),
and **both are invalid**: frames within a run are strongly autocorrelated, so that test treats ~180
dependent frames as independent draws. It is reported because it was pre-registered, and it is
discounted for a reason stated rather than for its answer.

### What `area_ratio_max` actually is, answered: a one-frame transient on ~0.3 % of the triangles

The brief asked whether the membrane is stretched further or one triangle in one frame hit a
transient. With the per-frame distribution recorded, it is the second, and unambiguously:

| arm | excursions above 1.4 in 20 runs | length in frames | triangles simultaneously above 1.4 |
|---|---|---|---|
| A | 7 | **1, 1, 1, 1, 1, 1, 1** | median 1, max 51 |
| A′ | 8 | **all 1** | median 2, max 9 |
| **B** | **4** | **all 1** | median 1, max 3 |
| S | 7 | 1×6 and one of 2 | median 2, max 27 |

The two largest events in the pass, both on a non-approximating path, frame by frame:

| run | frame | `area_ratio_max` | `ar_q99` | bulk `ar_total` | triangles > 1.4 | Newton | converged | CCD toi | line-search α |
|---|---|---|---|---|---|---|---|---|---|
| **S_6** (shipped default) | 25 | 1.059 | 1.0157 | 0.99964 | 0 | 9 | yes | 1.0 | 1.0 |
| | **26** | **74.04** | **1.0151** | **1.01455** | 27 | 7 | yes | 1.0 | 1.0 |
| | 27 | 1.671 | 1.0272 | 0.99794 | 2 | 9 | yes | 1.0 | 1.0 |
| | 28 | 1.115 | 1.0163 | 0.99852 | 0 | 8 | yes | 1.0 | 1.0 |
| **A_7** (exact path) | 61 | 1.039 | 1.0095 | 0.99978 | 0 | 15 | yes | 1.0 | 1.0 |
| | **62** | **3.699** | **1.0135** | **1.00467** | 51 | 8 | yes | 1.0 | 1.0 |
| | 63 | 1.135 | 1.0141 | 1.00063 | 0 | 7 | yes | 1.0 | 1.0 |

In both, the **99th percentile of the triangle area ratio does not move at all** (1.0151, 1.0135 —
the same as the neighbouring frames), 27 and 51 of ~18 300 triangles are involved (0.15 % and 0.28 %),
the bulk areal stretch moves by 1.5 % and 0.5 %, the frame converges normally with no CCD clamp and
no line-search cut, and the next frame is back to normal. **It is a one-frame local event, not a
stretched membrane** — which is why the run maximum could never have answered this question, and why
the same statistic has now been sighted in five steps without resolving.

**Rate: 2 such events in 80 runs, both on a non-approximating path, none in 20 runs of the change
under test.**

### §6 The wide screen and the joint signature

All 36 observables in `analysis_n20.txt`. **Not one meets V1's rule** (raw p < 0.05 *and* a median
shift over half of R's range); every median of B lies inside R's range. The smallest Holm-adjusted p
across the screen is 0.63. Selected rows:

| observable | median R | median B | median S | shift / range(R) | p |
|---|---|---|---|---|---|
| `mean_disp_mm_last_quarter` | 16.342 | 16.911 | 16.028 | 0.080 | 0.31 |
| `area_ratio_max` | 1.3146 | 1.3095 | 1.3352 | 0.002 | 0.71 |
| `tri_height_min` | 3.684 mm | 3.754 mm | 3.781 mm | 0.081 | 0.13 |
| `cc_min_dist` | 2.489 mm | 2.465 mm | 2.462 mm | 0.068 | 0.20 |
| `ke_mean` | 0.2007 | 0.2015 | 0.1924 | 0.012 | 0.29 |
| `e_tot_last` | 0.1912 | 0.1236 | 0.0953 | 0.066 | 0.94 |
| `drum_track_err_deg_max` | 0.03957 | 0.03957 | 0.03957 | 0.000 | 0.45 |
| `ccd_toi_min` | 0.534 | 0.548 | 0.487 | 0.017 | 0.94 |
| garment centroid travel (4) | 686 / 816 / 796 / 789 | 715 / 824 / 799 / 791 | 615 / 575 / 684 / 729 | ≤ 0.039 | 0.018–0.26 |

**The joint signature s07 flagged is absent.** `mean_disp_mm_last_quarter` — which s07 saw move
15.54 → 17.04 (p = 0.10) alongside the area ratio — is 16.342 → 16.911 here at **p = 0.31**, and no
membrane statistic moves up with raw p < 0.05. The pairing does not reproduce.

Worth recording separately: **V1's 2.15° drum transient did not occur anywhere in these 80 runs.**
`drum_track_err_deg_max` is exactly 0.0395724° — the frame-1 start-up transient — in the median of
all four arms.

### §7 Cost — the Newton drift is still absent at this head

Tumbler, n = 20 per arm (`analysis_n20.txt`):

| | A | A′ | **B** | S | B vs R |
|---|---|---|---|---|---|
| Newton total | 1459.0 | 1475.2 | **1472.9** | 1462.7 | **+0.39 %**, p = 0.45 |
| PCG total | 46 983 | 47 802 | **46 671** | 49 791 | −1.52 %, p = 0.48 |
| line-search total | 1724.7 | 1744.4 | **1723.5** | 1745.9 | −0.64 %, p = 0.89 |

### §8 The crease micro-test, run against this change as a null

V1's `crease_micro.py` unchanged, 6 runs per arm, 200 frames, θ reaching **3.115 rad** (a complete
fold). Contact is disabled in that scene, so the contact Hessian is never assembled and the two arms
must be indistinguishable — this is an instrument check, not a physics test.

In **seven of the eight** crease-severity bins the two arms agree **to three decimal places on the
mean Newton count and on PCG**. The eighth (θ_p95 ∈ [2.0, 3.5)) reads 5.293 vs 5.280 (−0.25 %,
p = 0.91). That is not a leak: the scene is not bit-reproducible run to run — arm 0's own six runs
total 778, 778, 779, 779, 778, 778 Newton iterations — and the per-frame trace differs between two
runs of the **same** arm on 61–81 lines against 77–102 lines between arms. The cross-arm difference
is inside the within-arm null. Zero non-converged frames, zero limit hits, all positions finite, in
both arms.

### The contact-severity micro-test — built for this change, and it is the strongest instrument here

`contact_micro.py`, `micro.txt`. The device probe above says the dropped curvature is tangential, so
the case has to slide as well as press. Six cloth patches stacked with a lateral offset of half an
element in x and a third in z (so a vertex lands over an **edge**, not over a vertex), gravity and
every other load **off**, the top and bottom patches soft-position-constrained as cloth platens, the
top one driven down (and optionally sideways) on a linear ramp. Severity is **measured**, not
assumed: `g = (clearance − 2r)/d_hat`, recomputed from the retrieved positions every frame, where
g = 1 is the activation distance and g = 0 is touching.

**The pair population in this scene is PE 16 893 / PP 0** (the engine's own
`SimplexTrajectoryFilter` line): **100 % of contact part 2's pairs take the branch
`UIPC_CONTACT_RANK1=1` approximates**, where the tumbler's PE:PP is about 3:1. That is what makes it
a better instrument than any benchmark scene for this question.

Five configurations, 5 runs per arm, 80 frames, arms interleaved:

| config | press / drag | severity reached (min g) | Newton total, arm 0 | Newton total, arm 1 | Δ |
|---|---|---|---|---|---|
| `base` | 9.0 mm / 0 | −0.61 | 292, 292, 292, 292, 292 | 291, 292, 292, 292, 292 | **−0.07 %** |
| `deep` | 11.5 mm / 0 | −1.07 | 289 ×4, 290 | 289 ×4, 290 | **+0.00 %** |
| `slide` | 9.0 mm / 30 mm | −2.15 | 440, 443, 444, 446, 448 | 440, 441, 441, 444, 445 | **−0.45 %** |
| `fastslide` | 11.5 mm / 70 mm | −6.22 | 505, 506, 507, 510, 511 | 506, 508, 509, 509, 511 | **+0.16 %** |
| `tight` (vel-tol 2e-5) | 11.5 mm / 30 mm | −3.35 | 483, 483, 484, 484, 484 | 483, 483, 484, 484, 485 | **+0.04 %** |

Binned by measured severity — the pre-declared reading is that Newton counts must rise in arm 1
relative to arm 0 **as severity rises, with the gap growing**:

| `g` bin (deeper downward) | `base` Δ | `slide` Δ | `fastslide` Δ | `tight` Δ |
|---|---|---|---|---|
| [1.00, ∞) | +0.0 % | −2.5 % | −0.9 % | +0.0 % |
| [0.75, 1.00) | −0.1 % | −1.3 % | +0.7 % | +0.2 % |
| [0.50, 0.75) | +0.0 % | +0.0 % | +0.0 % | +0.0 % |
| [0.35, 0.50) | +0.0 % | +0.0 % | +0.0 % | +0.0 % |
| [0.20, 0.35) | +0.0 % | +0.0 % | +0.0 % | +0.0 % |
| [0.10, 0.20) | +0.0 % | +0.0 % | — | — |
| [0.00, 0.10) | +0.0 % | +0.0 % | — | +0.0 % |
| below 0 | +0.0 % | +0.0 % | +0.0 % | +0.0 % |

**In every configuration, at every severity from the activation distance down to full compression,
arm 1's mean Newton count is equal to arm 0's or lower. It is never higher.** Zero non-converged
frames, zero `hit_newton_limit`, zero `hit_line_search_limit`, all positions finite, in all 50 runs.

One honest note on the automated verdict line: the analyser's mechanical test
(`slope > 0 and p < 0.05`) prints **PRESENT** for `slide` (slope +0.0028 per bin, p = 0.033). Reading
the table rather than the slope, that "positive trend" is a **negative** gap (−2.5 %, −1.3 %)
shrinking to zero — arm 1 is *better* at low severity and identical at high severity, and it does not
exceed arm 0 in any bin of any configuration. The pre-declared signature is a *rise*; there is none.
The other four configurations return `absent` outright.

The safety question in this scene has a direct answer too: **the softened tangential curvature does
not let the layers approach further.** Minimum `g` reached per run, arm 1 against arm 0, is
indistinguishable in four of five configurations (p = 0.14, 0.84, 0.15, 0.21) and differs in the
fifth (`base`, p = 0.011) by **1e-5 of d_hat** — 8 picometres.

**Within reachable configurations — five decades of contact severity, sliding at up to 70 mm over
80 frames, and a tolerance tightened 2 500× from the scene default — there is no regime where the
rank-1 PE contact Hessian degrades convergence.** That is the part of this pass that generalises
past our scenes, and it agrees with the device probe: the approximation is only inaccurate where the
matrix it approximates is numerically zero.

### Cost re-measured at this head — the win is intact and slightly larger

s08 landed in the same kernel family since s07's n = 50 sweep, so the cost claim was re-measured.
`cube-wall-cloth`, n = 20 per arm, four arms in one interleaved sweep, one build (`cost_cwc.txt`).
`pnull` is a second copy of the rollback (`=00`, which `std::atoi` reads as 0), so the null envelope
is measured beside the effects.

| | p0 (exact) | pnull (null twin) | p5 (shipped) | **p1 (the change)** |
|---|---|---|---|---|
| Newton | 511.45 ± 8.95 | 508.85 (−0.51 %) | 507.05 (−0.86 %) | **509.95 (−0.29 %, p = 0.73)** |
| PCG | 19 927 | −0.46 % | −0.34 % | **−1.73 % (p = 0.047)** |
| line-search | 532.75 | −0.70 % | −1.05 % | **−0.01 % (p = 0.99)** |
| `mean_ms` | 60.240 | −0.78 % | −1.23 % | **−4.07 % (p = 4.1e-04)** |
| **`ms_per_newton`** | 11.7775 | −0.28 % | −0.37 % | **−3.81 % (p = 2.5e-15)** |
| `ms_per_pcg` | 0.3023 | −0.32 % | −0.89 % | −2.41 % |

**The Newton drift is still absent** (−0.29 %, inside the null twin's own −0.51 %), and the win is
**−3.81 % ms/Newton**, a little larger than s07's −3.23 % — consistent with s08 having shortened
part 1, which changes part 2's share of the critical path.

Two things to report rather than bury:

- **The cube-wall-cloth Newton tail landed in the p1 arm this time.** p1's maximum is 579 against
  p0's 533, with sd 16.6 against 8.95. s07 saw the same event at n = 50 on the **exact default**
  (577) and on the **null twin** (550). Pooled over s07 and this pass, runs above Newton 530 are
  **3 of 140 on non-approximating arms** (577, 550, 533) and **2 of 70 on p1** (555, 579) — Fisher
  exact p = 1.0. It is a property of the scene, as s07 concluded, and one more draw does not change
  that; but the draw fell on the approximating arm here and that is worth recording.
- **`cloth_min_y`**, the scene's own physics observable, reads p0 0.2862, pnull 0.2875 (+0.43 %),
  p5 0.2860 (−0.09 %), **p1 0.2910 (+1.66 %, raw p = 0.015)**. That is 0.8 sd and 23 % of p0's own
  run-to-run range, and Holm-adjusted across the sweep's 24 comparisons it is ≈ 0.36 — but it is the
  only physics observable that moves anywhere in this pass, so it was re-measured at n = 40 rather
  than argued away (below).

### Env-switch audit, at kernel level

`envaudit.txt`, nsys `cuda_gpu_kern_sum` over 12-frame tumbler runs, reading the demangled template
arguments of the kernels actually launched. All three arms are proved on the binary:

| arm | contact part 1 | contact part 2 | hinge kernel |
|---|---|---|---|
| `UIPC_CONTACT_RANK1=0` | `do_assemble_kernel<0,1,1,1,0,**8**>` | `<0,2,0,0,1,**0**>` exact | `DiscreteShellBending_..._kernel<3,1>` |
| default (unset) | `<0,1,1,1,0,**8**>` | `<0,2,0,0,1,**5**>` PP closed form | `<3,1>` |
| `UIPC_CONTACT_RANK1=1` | `<0,1,1,1,0,**8**>` | `<0,2,0,0,1,**1**>` PE+PP rank-1 | `<3,1>` |

**Part 1 is `Proj = 8` in all three arms** — s08's basis-free reduction is the default and is *not*
part of the comparison, so A-vs-B isolates contact part 2 exactly as the design intends. The hinge
kernel is `<3,1>` (s02's Gauss-Newton, the shipped default) in all three, so nothing about s02 is
being re-litigated here.

### Correctness gate

`gate.sh` in all three arms — the rollback (`=0`), the shipped default (unset) and the change under
test (`=1`) — **identical to `baseline_tests.txt` in every assertion count**: 11/3, 1112/36, 2730/46,
100/3, 4/1, 448/23, **14213/95**, pytest 48 passed 1 skipped 86 deselected. The only textual
difference in any of the three is pytest's wall-clock line (1.22 s against the baseline's 1.20 s).
`sim_case`'s 95 cases include the contact suite.

### The one physical observable that moves, and what it is

`cloth_min_y` on `cube-wall-cloth` — the lowest cloth vertex at frame 100, and the only physics
observable that scene reports — moved in the n = 20 cost sweep. It was **not** argued away: it was
re-measured twice more (n = 40 and n = 25, all at the default tolerance), and every sweep is positive.

| | exact (p0) | null twin (`=00`) | shipped (`=5`) | **rank-1 (`=1`)** |
|---|---|---|---|---|
| n = 20 | 0.28623 | +0.43 % | −0.09 % | **+1.66 %** |
| n = 40 | 0.28652 | −0.13 % | — | **+1.04 %** |
| n = 25 | 0.28751 | — | — | **+0.97 %** |
| **pooled, n = 85** | **0.286741** | **−0.05 % [−0.62, +0.52], p = 0.86** (n = 60) | −0.26 % (n = 20) | **+1.17 % [+0.56 %, +1.77 %], p = 2.1e-04** |

That is a real, one-directional, reproducible shift, about twenty times the bit-identical null twin's,
and it is the kind of thing this pass exists to find. So it was diagnosed rather than weighed.

**The diagnosis, with the reading fixed before it ran** (`TRUNCATION_PREDICTION.md`). Mode 1 changes
only the Hessian; the energy and the gradient are untouched (s03, s07), so **both arms have the same
Newton fixed point — they solve the same equations.** A systematic difference in the accepted state
can therefore only be *where the solver stops*: `cube-wall-cloth` exits on an increment /
accumulated-β rule, and a different model Hessian gives a different increment at the same state. The
prediction: if that is the mechanism, the gap must shrink when both arms converge harder; if it is a
change of the simulated physics, it must not. `UIPC_CWC_TIGHT=<k>` divides both Newton stopping
tolerances by k (default k = 1 leaves every literal as it was, so the benchmark scene stays
byte-stable).

| Newton tolerance | Newton count | exact | rank-1 | **gap** | 95 % CI | p |
|---|---|---|---|---|---|---|
| default (×1), n = 85 | 509.0 | 0.286741 | 0.290084 | **+1.17 %** | [+0.56 %, +1.77 %] | 2.1e-04 |
| **×10 tighter**, n = 25 | 659.0 (+29 %) | 0.286828 | 0.289057 | **+0.78 %** | [−0.24 %, +1.79 %] | 0.14 |
| **×100 tighter**, n = 30 | 686.0 (+35 %) | 0.286402 | 0.287580 | **+0.41 %** | [−0.53 %, +1.36 %] | 0.40 |

**−0.377 percentage points per decade of tolerance, r = −1.000, monotone.** The pre-declared
truncation signature is present and the alternative is excluded: the gap does not survive
convergence. It is a convergence-quality difference bounded by the tolerance the user chooses, not a
different simulated world — which is what the identical energy and gradient already imply, now
measured rather than asserted.

**Reported at its full size anyway**: at the tolerance `cube-wall-cloth` actually ships with, the
accepted state differs by **+1.2 % of `cloth_min_y`**, which is 0.7 σ of that scene's own
run-to-run spread. Anyone who needs the last percent of that observable should tighten the
tolerance, and this is now measurable because the knob exists.

### Verdict: ship it

Against the rule fixed at `877d0cf9`, before the first arm ran:

- **§3 safety — pass.** 3 620 frames in 20 runs: zero non-finite positions, zero containment
  failures, zero lifter penetrations, zero inversions, zero interpenetrations, zero non-converged
  frames, zero `hit_newton_limit`, zero `hit_line_search_limit`, `verify_ok` true in all 20. B is not
  the worst arm on a single one of the seven worst-value statistics. The one `verify_ok` failure in
  the pass belongs to the **shipped default**.
- **§4 the membrane family — pass, with the envelope.** No member is significant (all Holm p = 1.00),
  and the statistic the decision turns on has a 95 % CI of **[−2.50 %, +7.47 %]** on median areal
  strain against a pre-registered fail line of **+10 %**. The null of the null — A against A′ —
  produces +1.34 %, so B's +1.99 % is what seed size alone does.
- **§5 the run maximum — does not escalate, and points the other way.** B has the **tightest**
  distribution of all four arms (range ratio 0.167), the **lowest** tail rate at every threshold, and
  the eight largest run maxima in the pass all belong to a non-approximating arm.
- **§6 the screen and the joint signature — pass.** Nothing flagged among 36 observables; the
  `mean_disp_mm_last_quarter` pairing s07 flagged does not reproduce (p = 0.31).
- **§7 cost — the claim holds and is larger.** Newton drift −0.29 % (inside the null twin's −0.51 %);
  **ms/Newton −3.81 %, p = 2.5e-15** at n = 20, −3.18 % at n = 25.
- **§8 the micro-tests — no reachable degradation regime.** In five configurations of a
  **100 % PE** contact case, across five decades of severity, arm 1's Newton count is never higher
  than arm 0's in any bin. The crease null is indistinguishable inside its own within-arm scatter.
- **Gate — identical in all three arms**, `sim_case` 14213/95 included.
- The one physical observable that moves is **solver truncation, proved by a pre-declared
  tolerance sweep**, and it shrinks monotonically toward zero as both arms converge.

**Recommendation: flip `UIPC_CONTACT_RANK1` to default 1.** The `=0` rollback works and is proved at
kernel level; it should stay. **This pass does not flip it — that is the coordinator's call.**

### Limits, stated plainly

- **N runs is N runs.** 20 per arm on the tumbler (80 full runs, 14 480 frames), 85 per arm on
  `cube-wall-cloth`'s `cloth_min_y` at the default tolerance but only 25 and 30 at the two tighter
  ones — **the ×10 and ×100 points are individually consistent with zero and with the default gap**;
  what carries the truncation conclusion is the monotone trend across three points plus the
  a-priori argument that the two arms have the same fixed point, not either point alone.
- **n was not raised above 20 on the tumbler**, because the pre-registered marginality condition was
  not met: no §4 statistic has Holm p < 0.05, and the two positive shifts are +1.99 % and +1.95 %
  against a +10 % line. Raising n would narrow the CI; it would not change which side of the line it
  is on.
- **Two scenes.** The physics comparison is `tumbler-garments` (36 observables, four arms) and
  `cube-wall-cloth` (one observable, four arms). `stiff-gipc-case2`, `rigid-wrecking-balls` and
  `mas-bunny` were not given a physics pass — s07's iteration-count evidence is what covers them, and
  mode 1 buys nothing on rwb anyway.
- **One GPU.** Everything is cc 7.5. The iteration counts and the physics are architecture-independent;
  the −3.81 % is not, and s07's transfer note stands — much of mode 1's per-launch win is a
  255 → 154 register occupancy jump, which PERF_METHOD §6 says can flip sign on another part.
- **`cc_min_dist` is a vertex-vertex proxy** for the true point-triangle gap, as V1 recorded. It
  bounds it from above, it is the same proxy in every arm, and it never reached zero — but no
  triangle-level self-intersection test was run.
- **The micro-test's severity variable breaks down past full compression.** `g` is an interpolated
  vertical clearance; once the free sheets wrinkle it reads negative, which the true point-triangle
  gap cannot. It is monotone and identical in both arms, which is what a severity axis needs, but the
  negative values are a proxy artefact and are labelled as such.
- **No sanitizers this pass.** V1 ran memcheck / racecheck / initcheck at `7316bbc5` and reproduced
  both findings on a `perf-round5-base` binary; s04–s08 have not been re-checked under them. That is
  a gap for whoever runs the round's final validation, not something this pass closes.
- **The device probe samples a distribution, not the scenes' own pairs.** Its sweep is log-uniform in
  `d_hat`, `kappa` and gap, which is why the "0 of 1.27 M" statement is conditioned on a reference
  drawn from the same sweep. The in-scene equivalent is the micro-test.

### Found outside this step's area (reported, not fixed)

- **`tumbler-garments` has a rare one-frame membrane transient that makes `verify_ok` fail, and it is
  the scene's, not any change's.** One run in 80 — on the **shipped default** — reached
  `area_ratio_max = 74.04` at frame 26, which trips `verify.py`'s `area_ratio_max < 20` assertion and
  returns `verify_ok = False`. The exact path produced 3.699 in the same 80 runs. Both are single
  frames, both recover completely in the next frame, both leave the 99th percentile of the triangle
  area ratio unmoved, and both converge normally with no CCD clamp and no line-search cut. **Any
  future `--verify` gate at n ≥ 20 will see `verify_ok` failures that have nothing to do with the
  change under test**, and the rate on a non-approximating path is about 1 in 40 runs.
- **A rounding-level change moves the tumbler's membrane statistics further, and far more
  significantly, than the approximation under test does.** The shipped default (`Proj = 5`, exact up
  to rounding by s07's eigendecomposition) shifts `ar_mean_mean` and `ar_total_mean` with
  **p = 3e-04** against the full-exact path at n = 20/40, in the safe direction, by +3.4e-4 of bulk
  areal strain. The likely mechanism is that the closed form returns the analytic eigenvalue where
  the iterative solve returns a slightly smaller one, i.e. very slightly stiffer contact. Nobody
  should read a p-value on these statistics as evidence of a physics change without an effect size.
- **V1's 2.15° drum transient did not occur in 80 runs here**; `drum_track_err_deg_max` is the
  0.0395724° start-up transient in the median of all four arms. V1 saw two events above 0.3° in 30
  runs, both on the exact path. The event is rarer than V1's n could establish.
- **`verify.py`'s `verify_ok` conflates a physical threshold with an audit threshold**, as V1 noted
  for `tri_height_min`; the same is true of `area_ratio_max < 20`, which is an anti-NaN guard and not
  a physical statement, yet it is the assertion that failed above.
- **`cube-wall-cloth`'s `cloth_min_y` is sensitive enough to resolve a solver-truncation difference
  at n = 85 and is the only physics observable that scene emits.** Anyone using it as a physics gate
  should know it moves ~1 % under any change to the Hessian and ~0.05 % under none, and that
  `UIPC_CWC_TIGHT` now exists to separate the two.
- **`clang-format` is still not installed on this box** (s04–s08 recorded the same).
- **`nsys` must be invoked as `/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys`.** The
  `bin/nsys` wrapper V1's `envaudit.sh` used fails with
  `Nsight Systems #VERSION_RSPLIT# hasn't been installed with CUDA Toolkit #CUDA_MAJOR#.#CUDA_MINOR#`
  and produces no csv. V1's script as committed does not run on this box today; s07's `nsysrun.sh`
  has the working path.

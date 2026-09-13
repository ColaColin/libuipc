
## s07 — the Newton drift that held s03's closed form back is not real, and the branch that carries the win is the one that is *not* exact

Evidence: `agent_docs/performance/data/2026-09-13-round6-s07/` (`README.md` maps every file onto
the claim it backs; `predictions.txt`, `sass_identity.txt`, `sass_resusage.txt`, `part2_static.txt`,
`gn_contact_probe.txt`, `scope_*.txt`, `drift_ci.txt`, `tail_pooled.txt`, `verify_*`, `gate*.txt`,
`default_selects_mode5.txt`, `ab/`, `nsys/`, `verify/`).

**Verdict: the PP branch alone ships, default on; PE+PP stays opt-in for a reason that is no longer
the Newton drift.** s03 held `UIPC_CONTACT_RANK1=1` back because Newton counts drifted +1–2 % in six
of seven measurements and one cube-wall-cloth run in fifteen needed Newton 576. At n=10 on the flat
instrument and n=50 on cube-wall-cloth, **neither reproduces**: the drift is bounded to
**[−0.135 %, +0.207 %]** on `stiff-gipc-case2`, and the tail event happens on the **exact default
path at the same rate** — the largest Newton count in 250 full runs (**577**) is an
`UIPC_CONTACT_RANK1=0` run. What the split also shows is that **mode 1's win is 93 % PE and 7 % PP**,
and PE is the half that is an approximation.

### The brief's premise, checked first — and the split inverts the step's expected outcome

The brief asked for the PE:PP split because PP is exact and therefore "sets the ceiling for a variant
that has no drift at all". The split was measured first, and it says the ceiling is almost nothing.
Full-run nsys, 3 runs per arm, **four** scenes, contact part 2 (`do_assemble_kernel<0,2,0,0,1,Proj>`)
µs per launch:

| µs/launch, part 2 | p0 exact | **p1** PE+PP | **p5** PP only | **p6** PE only |
|---|---:|---:|---:|---:|
| cube-wall-cloth | 1404.2 | 782.3 **−44.29 %** | 1363.4 **−2.90 %** | 838.2 **−40.31 %** |
| stiff-gipc-case2 | 810.7 | 435.0 **−46.34 %** | 806.5 **−0.51 %** | 467.6 **−42.32 %** |
| rigid-wrecking-balls | 985.1 | 537.0 **−45.49 %** | 946.0 **−3.96 %** | 660.5 **−32.95 %** |
| tumbler-garments | 887.6 | 496.6 **−44.05 %** | 864.0 **−2.66 %** | 565.6 **−36.28 %** |

The branch costs decompose additively (residual 1.9 % on cube-wall-cloth): the **exact PE**
projection is ~566 µs/launch and the **exact PP** projection ~41 µs/launch, a factor of **14**. Two
independent causes, both measurable: PE pairs outnumber PP pairs about **3:1** in this scene (the
`do_compute_energy_k3`/`k4` kernels, which are one thread per pair and otherwise identical in shape,
run at 138.4+110.5 µs against 51.2+30.4 µs per launch), and PE's exact path costs about **4–5x** more
per pair — `point_edge_distance2_hessian` plus s31's flagged 9x9 reduction against
`point_point_distance2_hessian` plus a 6x6. **`Proj = 5` therefore removes 7 % of what mode 1
removes, and that is the whole ceiling of the strictly-safe variant.**

### Why the PP closed form is exact, argued as well as measured

For point-point, `hess(D) = 2K` with `K = [[I,-I],[-I,I]]` and `grad(D) = 2[u;-u]`, `u = P0 - P1`.
`K` has eigenvalue 2 on the 3-dimensional subspace `{[v;-v]}` and 0 on its complement, and
`grad(D) grad(D)^T` annihilates `ker(K)` as well. On `range(K)` the exact Hessian acts as
`8 B'' u u^T + 4 B' I`: one eigenvalue `8 B'' D + 4 B'` along `u`, and two equal to `4 B' <= 0`,
which the PSD projection sets to zero. Since `|grad D|^2 = 8D`, the closed form's single eigenvalue
`c |grad D|^2 = (B'' + B'/(2D)) 8D = 8 B'' D + 4 B'` **is that eigenvalue exactly** — there is no
approximation to bound. s03's device probe, re-run at this head against the real `__device__`
functions over 140 470 samples, reproduces it: `‖H_proj − c gg^T‖/‖H_proj‖` **4.754e-16 mean,
1.451e-15 max**, `lambda_max` ratio 1.000000 exactly, 0 samples with a negative eigenvalue. PE, by
contrast, keeps rank 2 of 9 and reads 9.78e-05 mean / **1.900e-02 max**.

### The drift question, settled on two scenes

Every arm ran in **one interleaved sweep per scene** (`abn.py`: arm order rotated per rep and
reversed on even reps — the N-arm generalisation of `ab.py`'s ABBA — one discarded warm-up per arm),
so the arms are draws from the same scene state rather than from separate A/Bs. The null arm is a
second copy of the default (`UIPC_CONTACT_RANK1=00`, which `std::atoi` reads as 0), i.e. **the null
envelope is measured in the same sweep as the effects**.

**`stiff-gipc-case2`, n=10 per arm — Newton counts, with the 95 % Welch CI on the difference:**

| arm | Newton mean | sd | range | Δ vs p0 | 95 % CI |
|---|---:|---:|---|---:|---|
| p0 exact | 1660.8 | 2.4 | [1657, 1664] | — | — |
| **pnull** (bit-identical to p0) | 1659.5 | 2.6 | [1655, 1664] | −0.08 % | [−0.227, +0.070] % |
| **p1** PE+PP | 1661.4 | 3.2 | [1654, 1667] | **+0.04 %** | **[−0.135, +0.207] %** |
| p5 PP only | 1660.4 | 2.9 | [1655, 1664] | −0.02 % | [−0.183, +0.135] % |
| p6 PE only | 1659.4 | 5.2 | [1652, 1666] | −0.08 % | [−0.334, +0.166] % |

**s03's +1–2 % is excluded by an order of magnitude.** PCG and line-search counts move −0.3 to +0.2 %,
every arm overlapping (`p1` PCG −0.36 %, CI [−1.29, +0.57] %).

**`cube-wall-cloth`, n=50 per arm (an n=10 sweep plus an n=40 sweep, 250 full runs):**

| arm | Newton mean | sd | median | max | runs > 530 | Δ vs p0 | Welch p |
|---|---:|---:|---:|---:|---:|---:|---:|
| p0 exact | 508.90 | 12.17 | 508 | **577** | **1 / 50** | — | — |
| pnull | 510.88 | 8.21 | 510 | **550** | **1 / 50** | +0.39 % | 0.35 |
| p1 PE+PP | 508.08 | 9.44 | 507 | 555 | 1 / 50 | −0.16 % | 0.71 |
| p5 PP only | 507.74 | 5.20 | 508 | 519 | 0 / 50 | −0.23 % | 0.54 |
| p6 PE only | 506.96 | 7.11 | 505 | 525 | 0 / 50 | −0.38 % | 0.34 |

**The tail is a property of the scene, not of the approximation.** s03's "one cube-wall run in fifteen
needed Newton 576 against a 496–526 band" is the same event: it occurs here once in fifty on the
*shipped exact default* (577) and once in fifty on the null twin (550), against once in fifty under
mode 1 (555). Fisher exact on exact-path (2/100) versus approximating-path (1/100) tail rate:
**p = 1.0**. Levene on the Newton spread: every arm p >= 0.06. Inspected frame by frame, the 555 run
is not a solver failure but a trajectory that diverges around frame 68 and needs +2 to +3 Newton
iterations per frame for the last thirty frames — which is exactly what the exact path's 577 run does
too. **What s03 was missing was an exact-path control of the same size**, and the round record's own
warning that cube-wall-cloth's counts move ±2 % on their own is the reading that survives.

`rigid-wrecking-balls`, n=15 per arm, is the third reading: Newton p1 −1.09 % (p=0.046), p5 −0.48 %,
pnull +0.10 %.

### Performance — predicted first, then measured

`predictions.txt` was written from the cube-wall-cloth scope before any end-to-end run:
`0.873 x 8.56 % x per-launch delta` (0.873 = that scene's GPU-kernel-time / wall ratio; 8.56 % = part
2's share of GPU kernel time).

| cube-wall-cloth `ms_per_newton` | predicted | measured (n=40) | p |
|---|---:|---:|---:|
| p1 PE+PP | **−3.31 %** | **−3.23 %** | 4e-25 |
| p5 PP only | **−0.22 %** | **−0.17 %** | 0.42 |
| p6 PE only | **−3.01 %** | **−3.09 %** | 3.2e-24 |

All three land within 0.1 pp — but **the prediction was right for a partly wrong reason, and that is
worth recording**. The 0.873 factor came from `baseline-runs/`'s 69.7 ms/frame, which is stale: at
this head cube-wall-cloth runs at 60.6 ms/frame and its GPU-kernel-time / wall ratio is **1.005**, so
the arithmetic "should" have predicted −3.81 %. The measured conversion from the family total
(−3.95 % of GPU kernel time) to the wall (−3.23 % of ms/Newton) is **0.82**, which is what the stale
factor accidentally reproduced. The conversion is a real quantity — the share of the removed kernel
that sits on the critical path — and the next section is what it depends on.

End to end, across the suite:

| scene | n | p1 PE+PP | p5 PP only (shipped) |
|---|---|---|---|
| cube-wall-cloth | 40 | `meanFrameMs` −3.49 % (p=3.5e-07), ms/Newton **−3.23 %** | −0.44 % / **−0.17 %**, both overlapping |
| stiff-gipc-case2 | 10 | −0.56 % / **−0.60 %** (p=0.005), Newton +0.04 % — guard silent | −0.14 % / −0.12 %, overlapping |
| rigid-wrecking-balls | 15 | **+0.04 % mean, ms/Newton +1.14 % (p=0.010), median +2.20 % (p=0.0008)** at Newton −1.09 % — the guard fires; no win here | −1.01 % / −0.54 %, overlapping |
| tumbler-garments | 8 (`--verify`) | Newton 1460.5 vs 1476.4, PCG −3.3 %, all overlapping | Newton 1466.2, all overlapping |

**Mode 1 does not convert on rigid-wrecking-balls, and the scope says why it should have**: contact
assembly there is −19.7 % and the scene's GPU kernel time −8.8 %, yet the wall does not move. That
scene's kernel-time sum is **7 % larger than its wall** — part 1 runs on the side stream and part 2 on
the default stream (K9's split), and on rwb part 1 (1499.6 µs) is longer than part 2 (985.1 µs), so
shrinking part 2 shortens a stream that was never the critical path. The same ratio is 1.005 on
cube-wall-cloth (where the full win converts) and 1.017 on `stiff-gipc-case2` (where 0.4 of it does).
**Part 2's share of GPU kernel time is not the right predictor of its end-to-end value; its share of
the critical path is, and the two differ by scene.**

### What the shipped default costs and what it does not

`Proj = 5` is **252 registers against the exact path's 255** — the same cc 7.5 occupancy granule, so
the two arms launch at the same geometry and nothing about the measurement is an occupancy artefact.
`Proj = 1` and `= 6` drop to **154**, a large occupancy jump; a good share of mode 1's −44 % per launch
is therefore occupancy, not arithmetic, and PERF_METHOD §6 says that category can flip sign on
another part. The shipped mode carries none of that: it deletes `point_point_distance2_hessian` and
one 6x6 reduced eigen-solve per PP pair, and nothing else.

### Why PE+PP is still not the default, now that the drift is gone

The tumbler `--verify` audit, 180 frames, **8 runs per arm**, three arms, one build: `verify_ok`,
`all_finite`, radial and axial containment **true in all 24 runs**; 0 non-converged, 0
`hit_newton_limit`, 0 `hit_line_search_limit` frames in every run of every arm; `cc_min_dist`,
`bore_gap_min`, `tri_height_min`, `lifter_depth_max`, `max_speed` all overlapping. One statistic does
not overlap comfortably:

| `verify_area_ratio_max` | sorted, 8 runs |
|---|---|
| p0 exact | 1.197 1.203 1.281 1.305 1.317 1.329 1.398 **1.563** |
| **p1 PE+PP** | 1.197 1.202 1.270 1.391 1.433 1.657 1.751 **2.029** |
| p5 PP only | 1.175 1.209 1.238 1.243 1.289 1.340 1.348 **1.367** |

Welch p = 0.172, Mann-Whitney p = 0.44 — **not resolvable at n=8**, and it is an extreme-value
statistic, which V1 already flagged as the weakest kind of evidence on a handful of runs. But it is
the *same* membrane-stretch statistic that moved under s02's Gauss-Newton hinge (V1) and under s03's
modes 1 and 2, it moves in the same direction for the third time, `mean_disp_mm_last_quarter` moves
with it (15.54 -> 17.04, p = 0.10), and the p1 maximum is 30 % beyond anything the exact path produced.
**p5 sits inside p0's envelope on every observable, including this one.** So the blocker on mode 1 is
no longer the Newton count — it is one physics observable that needs a V1-scale pass, and this step
does not have the standing to clear it with 8 runs.

### Gates

- **Correctness**: `gate.sh` identical to `baseline_tests.txt` in **all three arms** — the shipped
  default (`=5`), the rollback (`=0`) and the opt-in (`=1`): 11/3, 1112/36, 2730/46, 100/3, 4/1,
  448/23, **14213/95**, pytest 48 passed 1 skipped. `sim_case`'s 95 cases include the contact suite.
- **Old arm reproduces main's binary**: standalone compile of the TU with the build's exact flags at
  `a3061ef5` and at this head, SASS-diffed: **67 / 67 common streams byte-identical**, including both
  instantiations the benchmark scenes launch (`<0,1,1,1,0,0>` 30 954 instr, `<0,2,0,0,1,0>` 12 306
  instr). The only new functions are the four `Proj = 5 / 6` instantiations. This is stricter than
  s03's result on the same file (52 / 53), because s07 adds template values rather than editing the
  shared body.
- **Numerics**: rounding-level, not bit-identical, and proved at that level — the analytic
  eigendecomposition above plus 140 470 device samples at 4.754e-16 mean / 1.451e-15 max relative
  Frobenius error, `lambda_max` ratio exactly 1.000000, 0 samples non-PSD, exact symmetry. Compared
  against the old path's own run-to-run noise (PERF_METHOD §2.3): the scene's Newton counts move
  ±1.6 % between identical exact-path runs, and p5's Newton delta is −0.23 % (p = 0.54).
- **Env-switch audit at kernel level**: with no env var the part-2 launch is
  `do_assemble_kernel<(bool)0,(int)2,(bool)0,(bool)0,(bool)1,(int)5>`; with `UIPC_CONTACT_RANK1=0` it
  is `(int)0`; part 1 is `(int)0` in both.

### Transfer prediction

`Proj = 5` is **algorithmic — it transfers**: it deletes a 6x6 Hessian assembly and a reduced
eigen-solve per PP pair at an unchanged launch geometry (252 vs 255 registers, same granule). What
will not transfer is its *size*, which is already below noise here: this box's 1/32-rate FP64
over-rewards the removed arithmetic, so −0.5 to −4.0 % per launch of part 2 is an **upper** estimate.
The honest statement is that this change is adopted because it is **exact and free**, not because it
is fast — its end-to-end effect is below every scene's noise floor on this box and will be smaller on
a modern part.

`Proj = 1` is a **mixture**: the arithmetic half transfers, but 255 -> 154 registers is an occupancy
change, and §6 records that occupancy tuning can flip sign between architectures. Its −44 % per launch
should be treated as this box's number, not the change's.

### Candidates for the next step

| candidate | measured evidence | where to gate it |
|---|---|---|
| **`UIPC_CONTACT_RANK1=1` is unblocked on iteration counts and needs one V1-scale physics pass to ship** | the drift that held it back is bounded to **[−0.135 %, +0.207 %]** Newton on case2 at n=10 and its Newton tail rate is identical to the exact path's (2/100 vs 1/100, Fisher p=1.0). It is worth **−3.23 % ms/Newton on cube-wall-cloth (n=40, p=4e-25)** and −0.60 % on case2. The one thing standing between it and default-on is `verify_area_ratio_max` 1.324 -> 1.491 (max 2.029) at n=8, p=0.17 — the third sighting of the same statistic (V1/s02, s03, s07) | tumbler `--verify` at **n >= 20 per arm** on `area_ratio_max` and `mean_disp_mm_last_quarter`, plus the crease micro-test V1 built for exactly this question; and re-check rigid-wrecking-balls, where mode 1 buys nothing |
| **Contact part 1 (PT+EE) is still the largest kernel in the suite and its projection is still ~74 % of it** | unchanged from s03's stage stub (rwb 1512 -> 389 µs/launch) and re-measured here: part 1 is **2123.6 µs/launch on cube-wall-cloth** and **1945.4 on the tumbler**, against part 2's 1404.2 / 887.6. s03's mode 2 is rejected (rwb +4.75 %), but the **mollified EE branch is a two-scalar function nobody has exploited** — `H = J^T M_2 J + ...` with `J` 2x12 and `M_2` a **2x2**, the same shape `friction_make_spd` already uses to collapse 12x12 to 2x2 | rwb (part 1 is the largest single kernel there) + cube-wall-cloth; needs its own numerics probe for the two dropped `hess` terms |
| **The K9 stream split makes part 2's GPU-time share a bad predictor of its wall value, and nobody has measured the split itself** | GPU-kernel-sum / wall is **1.005** (cube-wall-cloth), **1.017** (case2), **1.071** (rwb) — so on rwb ~7 % of all kernel time is hidden by concurrency, part 1 (1499.6 µs) is longer than part 2 (985.1 µs), and a −45 % cut to part 2 buys **0.00 %** of wall. The same switch (`UIPC_CONTACT_SPLIT`) that creates the overlap has never been re-measured since K9 | `UIPC_CONTACT_SPLIT=0/1/2` per scene, per launch and end to end; rwb and the tumbler are where the overlap is largest |
| **`InfoStacklessBVH_pairFilter` and the `stacklessSelf` overflow fallback** | unchanged from s06's list, both still untried (s05's candidates 2 and 3): 812 342 staged / 387 081 kept (47.6 %) in the traversal, and 98 registers paid on every thread for a capacity-overflow path | `pairFilter` + `stacklessSelf` per launch, five scenes |

### Found outside this step's area (reported, not fixed)

- **`do_assemble_kernel` part 2 at `Proj = 0` compiles to 255 registers** — the cc 7.5 ceiling, one
  256-thread block per SM. Both rank-1 modes land at 154. Nobody has asked what the *exact* path
  costs in occupancy: a `__launch_bounds__` or a split of the PE branch's flagged cases might buy
  part of mode 1's win **without** its approximation. s01's lesson (occupancy flips sign between
  scenes on the same GPU) applies, but the register number is large enough to be worth one
  measurement.
- **`cube-wall-cloth` is a far better instrument than the round record credits it with, for counts
  as well as for wall.** At **7.3 s per full run** it supports n=40 per arm in 25 minutes; its Newton
  distribution is 508.9 ± 12.2 with a ~2 % upper tail rate, so a **1 % count effect needs n ≈ 40**,
  not the n=5 that round 6 has been using on it. Two of this round's count claims were made at n=5-8.
- **The round has no recorded null envelope for Newton *counts*, only for wall time.** Measured here,
  in the same sweeps: cube-wall-cloth **sd 8-12 counts = 1.6-2.4 % on Newton, with a ~2 %/run tail
  above +4 %**,
  `stiff-gipc-case2` **±0.15 %**, `rigid-wrecking-balls` **±1.5 %**. A count claim below those is not
  readable, and s03's was.
- **`verify_area_ratio_max` has now moved in the same direction under three independent Hessian
  approximations** (s02's Gauss-Newton hinge, s03's contact modes 1 and 2, s07's mode 1). It is the
  round's most sensitive physics observable and the cheapest one to over-read — it is an extreme value
  over 180 frames, so it needs n >= 20 before anyone treats a shift as real *or* as absent.
- **`clang-format` is still not installed on this box** (s04, s05 and s06 recorded the same).

# Round 5, s31 (w2-ada, RTX 4060 Ti, cc 8.9) — contact G+H part 2: the range basis, not the solver

Worker w2-ada's third step, and the follow-on to s25. Target: **contact G+H part 2**,
`do_assemble_kernel<false, 2, false, *>` — the PE + PP branch, which s25 made the binding half of
the contact fork.

Measured on `perf/round5` at **`72de9e5c`** (s19, s22, s23, s24 and s25 composed, i.e. w2's part-1
solver merged with w3's s24 launch-geometry sweep), one build, env A/B, nsys 2024.6.2 60-frame
windows. Branch `perf/round5-w2-ada`.

> **Merge sanity check first, as the brief asked.** `git diff 5b8766ec 29aef03c` over
> `ipc_simplex_normal_contact.cu` is exactly two hunks: the `<cuda_tool/spread_launch.h>` include and
> the `do_compute_energy` launches (w3's s24). It touches no line containing `SpdTql`,
> `do_assemble`, `make_spd`, `Solver` or `blocked`. s25 survived the merge intact.

## 1. Where part 2's time actually goes — and it is *not* where part 1's was

### Baseline on this box after the merge (rigid-wrecking-balls, 60-frame nsys window)

186 Newton / 3 920 PCG, **5.9486 ms/it** of kernel time (w3's s24 took the scene from 6.80 to 5.95):

| rank | ms/it | share | kernel |
|---|---:|---:|---|
| **1** | **1.0116** | **17.00 %** | **contact G+H part 2, `<false,2,false,false>` (PE+PP)** |
| **2** | **1.0114** | **17.00 %** | contact G+H part 1, `<false,1,true,true>` (PT+EE) |
| 3 | 0.5332 | 8.96 % | `abd_..._dytopo_effect_pair_warp` |
| 4 | 0.3523 | 5.92 % | `InfoStacklessBVH_stacklessSelf` |
| 5 | 0.2835 | 4.77 % | IPC simplex **frictional** contact `do_assemble_kernel<false>` |

Union of the two forked launches: 1.6639 ms/it. **s25 did what it said it would: part 2 is now the
joint-largest kernel of the suite on this box**, tied with part 1 to three digits.

### Stage-stubbing probe — with the launch geometry pinned

The first probe pass was **contaminated and I threw it away**: stubbing part 2's projection drops its
register count 255 -> 168, and w3's s24 `best_block_dim` then hands the kernel a 384-thread block
instead of 256, i.e. 12 resident warps per SM instead of 8. The measured "-33.5 %" was one third
occupancy. Every number below comes from a build with a probe-only pin of part 1's and part 2's block
dim to 256 (in my own TU, reverted before commit); at 256 threads both 255 and 168 registers give one
block and 8 warps per SM, so the comparison is work-only.

rigid-wrecking-balls, ms per Newton iteration. Stub builds change the trajectory (Newton 188-220), so
these are attribution numbers, not A/B numbers:

| build | part 1 | **part 2** | part 2 regs | share of part 2 |
|---|---:|---:|---:|---:|
| control (all stubs off, pinned) | 1.0207 | **1.0185** | 255 | — |
| PE **and** PP projections stubbed | 0.9887 | 0.5917 | 168 | **41.9 %** |
| PE projection stubbed | 1.0015 | 0.5898 | 172 | **42.1 %** |
| PP projection stubbed | 0.9846 | 0.9480 | 255 | 6.9 % |
| **PE dim-3 projection only** (`make_spd_contact<9,3>`) | 0.9895 | 0.6085 | 176 | **40.3 %** |
| all M = 2 projections (`<9,2>` + `<6,2>`) | 1.0040 | 1.0040 | 255 | **1.4 %** |
| **PE dim-3 `make_spd<4>` only** (basis algebra kept) | 1.0064 | 0.7611 | **254** | **25.3 %** |

| component of part 2 | ms/it | share of part 2 |
|---|---:|---:|
| **`make_spd_contact<9,3>` — the PE dim-3 reduced projection** | **0.410** | **40.3 %** |
| &nbsp;&nbsp;of which the 4x4 eigen-solve `make_spd<4>` | 0.257 | 25.3 % |
| &nbsp;&nbsp;of which the basis algebra (Q, `Q^T Hs Q`, `Q Hred Q^T`, Hs/Hspd) | 0.153 | 15.0 % |
| every M = 2 projection (PE dim 2 **and** all of PP) | 0.015 | 1.4 % |
| everything else: distance flags, both symbolic Hessians, the gradient/Hessian writes | ~0.59 | 58 % |

**The brief's hypothesis was half right and the interesting half was wrong.**

- The projection *is* a cost centre, but it is **41.9 %** of part 2, not part 1's 59.5 %. I ran the
  probe rather than inheriting part 1's answer, as the brief asked; the answer is different.
- The brief's candidate (a) — forward TQL, "close to a one-token change" — reaches **only the PE
  dim-3 branch**, because `make_spd_contact<9,2>` and `<6,2>` end in `make_spd<3>`, which takes
  Eigen's closed form whatever `Solver` says. That branch is worth 25.3 % of part 2, so the
  one-token change had a real prize behind it. But
- **w0's R1 note in the brief — "Jacobi wins at N = 4, and TQL at N = 4 was never measured against
  it" — points at the wrong 1.4 %.** The M = 2 projections, where N = 3 and no solver choice exists
  at all, are **1.4 % of part 2**; the whole prize is at N = 4, where w0 measured TQL and Jacobi at
  the *same* 1.20x. There was no unmeasured solver question here.
- The genuinely unexploited half was the one nobody listed: **the range basis itself**, 15.0 % of
  part 2, with no blocked form in the tree to reuse. Part 1's analogue (K16's blocked basis) existed
  and only needed wiring. Part 2's did not exist and had to be derived.

## 2. The change

`UIPC_CONTACT_SPD2=0` restores the old path. Two mechanisms, both behind one switch:

### (1) s19's tridiagonal QL at N = 4

`PE_barrier_make_spd<Solver, Basis>` and `PP_barrier_make_spd<Solver, Basis>` become templates (they
were the two `make_spd_contact` wrappers s25 left untemplated) and forward `Solver` into
`make_spd<4>` for the PE dim-3 branch.

### (2) A basis-free form of the same reduced projection, for M <= 3

`make_spd_contact<N, M, Solver, Basis>` gains `Basis = 1`. The old path builds the explicit
`3M x (M+1)` matrix `Q` of `barrier_range_basis` and forms `Q^T Hs Q` and `Q Hred Q^T`. The new one
never builds it.

`Q`'s columns are `(s^ (x) t1, s^ (x) t2, u_1 (x) n^, ..., u_{M-1} (x) n^)`. The `u_j` span the
mean-free weights of `R^M`, and `s^` is itself mean-free (`sum_j s_j = 0` holds exactly for every
caller), so `span{u_j} = span{s^, v_1, ..., v_{M-2}}` with `v_i perp s^`, and

```
range(Q) = s^ (x) span{t1, t2, n^}  (+)  v_1 (x) n^  (+) ...
         = s^ (x) R^3               (+)  v_1 (x) n^  (+) ...
```

because `(t1, t2, n^)` is an orthonormal basis of `R^3`. **The tangent frame drops out of the
subspace entirely** — and with it the `|n^_0| >= 0.9` branch, the two `normalized()` calls and the
cross product that build it, and the Gram-Schmidt loop for `U`. Since
`Q' make_spd(Q'^T H Q') Q'^T = Q make_spd(Q^T H Q) Q^T` for any two orthonormal bases `Q' = Q R` of
the same subspace (`make_spd` clamps eigenvalues, so it commutes with orthogonal conjugation), the
projection is unchanged.

- **M = 2** (PP always, PE dim 2, and part 1's PT/EE dim 2): `range(Q) = s^ (x) R^3`. The whole
  reduction collapses to `Hss = sum_ab s^_a s^_b H_ab`, one 3x3 `make_spd`, and
  `H_ab <- s^_a s^_b Hss`. **No `Q`, no `n^`, no tangent frame, no 6x6 `Hs`, no 6x6 `Hspd`.**
- **M = 3** (PE dim 3, and part 1's PT/EE dim 3): one extra direction
  `v = ((1,1,1) x s^)/|.|`, the unit mean-free vector orthogonal to `s^` — three subtractions and one
  `normalize()` against the old path's Gram-Schmidt. The reduced matrix is the bordered 4x4
  `[[Hss, m], [m^T, c]]` with `m = (sum_ab s^_a v_b H_ab) n^` and `c = n^^T (sum_ab v_a v_b H_ab) n^`,
  accumulated in one pass over the nine 3x3 blocks of `H` with two 3x3 and two 3-vector accumulators.
  The map back is `H_ab = s^_a P_b + v_a R_b` from three precomputed 3x3 pairs.
- **M = 4** falls through to the dense-Q path unchanged. Deriving `v_1, v_2` there is a 2-dim
  Gram-Schmidt and the reconstruction has six 3x3 terms per block; it is real work for a branch the
  probe says is not in part 2 at all, so it is left for a later step (see candidates).

`Spd2` is a template parameter of `do_assemble_kernel`, not a runtime flag (the s14 lesson). Part 1
has no PE/PP branch and is only ever instantiated with `Spd2 = false`, so **s25's part-1 code is not
touched**; part 0 (the fused fallback) and part 2 carry both instantiations.

### Static shape (ptxas `-v`, sm_89, the same object, only the template argument differs)

| instantiation | registers | stack frame | spill stores / loads |
|---|---:|---:|---:|
| part 2, `Spd2 = false` | 255 | 2 064 B | 88 / 88 B |
| part 2, **`Spd2 = true`** | 255 | **2 016 B** | **8 / 8 B (-91 %)** |
| part 1, `EEReducedRange=true, SpdTql=true` (untouched) | 255 | 6 272 B | 4 236 / 4 352 B |

**Registers are unchanged at 255, so occupancy is unchanged** — one 256-thread block per SM either
way, and `best_block_dim` returns 256 in both arms (confirmed in every nsys window: `grid=Nx256`,
`r=255`). The frame barely moves: part 2's 2 KB is the *symbolic* Hessian, not the projection. **The
gain is instruction count**, which matters for §6.

## 3. Correctness

`bash /root/work/run_tests.sh /root/work/tests_s31`, against this box's `baseline_tests.txt`:

| binary | baseline | s31 |
|---|---|---|
| common | 11 assertions / 3 cases | **same** |
| core | 1 112 / 36 | **same** |
| geometry | 2 730 / 46 | **same** |
| sanity_check | 100 / 3 | **same** |
| regression | 4 / 1 | **same** |
| backend_cuda | 352 / 22 | **448 / 23** — see below |
| sim_case | 14 213 / 95 | **same** |
| pytest | (baseline capture broken: rc=2, 7 collection errors) | 48 passed, 1 skipped |

`backend_cuda` is **+96 assertions / +1 case, and none of it is mine**: the box's
`baseline_tests.txt` was captured at `perf-round5-base`, and the head adds
`apps/tests/backends/cuda/qr_svd.cu` (w3's s24, commit `d8a2e91c`). `--list-tests` names the extra
case, `fixed-sweep 3x3 SVD matches the iterative QR-SVD on GPU`, tagged `[cuda][qr_svd]`. My diff
touches two files under `src/backends/cuda/contact_system/` and adds no test case, so the counts at
the head with and without `Spd2` are identical by construction. **Everything else matches exactly; no
new failures, no flakes.**

## 4. Numerics — rounding level for the algebra, proved; and the clamp boundary, measured

Standalone verifier (`verify_s26.cu`, the s14/s16/s25 pattern: the repo's own device functions, on
the GPU, both paths on the same input). 3 seeds x 300 000 PE pairs and 3 x 300 000 PP pairs per arm.
PE pairs are sampled with the closest-point parameter over `[-0.3, 1.3]` so both flag branches occur
(measured 37.4 % dim 2 / 62.6 % dim 3); gaps 1e-4..5e-2, `kappa*dt^2` up to 1e8, thickness 0 and
0.1 d, the whole pair translated up to 10 length units from the origin.

### The algebra is exact to rounding

The decisive arm. Both reductions with the **eigenvalue clamp removed**, i.e. the plain orthogonal
projector `Q (Q^T H Q) Q^T` onto `range(Q)` computed the two ways:

| arm | n | p50 | p90 | p99 | p99.9 | **max** |
|---|---:|---|---|---|---|---|
| **PE, clamp-free, old basis vs new basis** | 3 x 300 k | 4.59e-16 | 9.02e-16 | 1.36e-15 | 1.74e-15 | **2.54e-15** |

**2.5e-15 over 900 000 pairs.** That proves both claims at once: the new basis spans exactly the same
subspace, and the one-pass accumulation is arithmetically equivalent to `Q^T Hs Q` / `Q Hred Q^T`.

### With the clamp on, the difference is the clamp boundary — and the old path has the same one

`make_spd` is discontinuous at `lambda = 0`: an eigenvalue within ~1e-9 of zero is kept by one
basis's eigen-solve and zeroed by the other's. That is not a property of my change, so I measured the
old path's own version of it. The **tangent frame is already an arbitrary choice** — the shipped code
seeds it from `(1,0,0)` unless `|n^_0| >= 0.9` — so I built a second, equally valid instance of the
*old* path seeded from `(0,0,1)` and ran old-vs-old-alt as the reference noise floor (PERF_METHOD 2.3):

| arm | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| **PE old vs old, alternative tangent frame** (reference) | 7.70e-16 | **2.357e-9** | 4.65e-9 | 7.10e-9 | **1.10e-8** |
| PE old vs **new (shipped)** | 1.02e-15 | **2.348e-9** | 5.03e-9 | 7.83e-9 | 1.10e-7 |
| PP old vs old, alternative tangent frame (reference) | 1.91e-16 | 5.89e-16 | 4.52e-9 | 7.93e-9 | 1.12e-8 |
| PP old vs **new (shipped)** | 3.87e-16 | 9.91e-16 | 5.03e-9 | 8.49e-9 | 2.78e-8 |

`p90` agrees to **three digits** (2.357e-9 vs 2.348e-9), `p99` within 8 %, `p99.9` within 10 %. The
distribution is bimodal — ~85 % of pairs agree to 1e-15, ~15 % sit at 1e-9 — and it is **the same
bimodal distribution for a change of tangent frame the shipped code already makes arbitrarily**.

Against the *exact* full-size projection (`make_spd<9,1>` / `make_spd<6,1>`), which is s16's standard:

| arm | p50 | p99 | p99.9 | max |
|---|---|---|---|---|
| PE old reduced vs exact 9x9 | 1.08e-15 | 3.39e-9 | 4.42e-9 | 5.70e-9 |
| PE new reduced vs exact 9x9 | 1.12e-15 | 4.31e-9 | 6.21e-9 | 1.12e-7 |
| PP old reduced vs exact 6x6 | 9.72e-16 | 3.79e-9 | 4.93e-9 | 5.69e-9 |
| PP new reduced vs exact 6x6 | 9.85e-16 | 4.53e-9 | 6.96e-9 | 4.53e-8 |

**A hypothesis I had and refuted.** I first assumed the 1e-9 tail was `eps * cond(H)` and that the
barrier's `kappa*dt^2` drove it. Re-running the whole verifier with `kappa*dt^2` sampled from
`[1e2,1e4]`, `[1e4,1e6]` and `[1e6,1e8]` gives p50/p90/p99/p99.9 **identical to three digits across
four decades of kappa**. Conditioning is not the mechanism; the clamp boundary is, and the clamp-free
arm is what proves it.

**Honest limits.** This is *not* as tight as s25's 3.1e-14, and the brief asked me to hold to that
standard. It cannot be met here and I do not claim it: s25 changed the *summation order* within one
basis, while s26 changes the basis, and any basis change of a clamped projection costs ~1e-9 on the
pairs with a near-zero eigenvalue — as the old path's own alt-frame control shows at 1.10e-8. At
every percentile through p99.9 this change sits inside that control; at the single worst sample in
300 000 it is ~10x wider (1.1e-7 against 1.1e-8). For scale, s16's accepted reduced-range EE
projection was 4.4e-9 max on the same kind of sample. Zero NaN, zero non-finite values and zero
negative diagonals in all 1.8 million clamped samples, in both paths.

## 5. Performance

### Targeted scope, A/B in one build via `UIPC_CONTACT_SPD2` (nsys, 60 frames, 2 runs each way)

ms per Newton iteration. **Part 1, part 2 and their union**, as the brief requires:

| scene | | part 1 (untouched) | **part 2 (mine)** | **union** | whole-scene kernel |
|---|---|---:|---:|---:|---:|
| rigid-wrecking-balls | SPD2=0 | 1.0168 / 1.0304 | 1.0069 / 0.9829 | 1.7046 / 1.6993 | 5.9850 / 5.9613 |
| | SPD2=1 | 1.0013 / 1.0095 (**-1.8 %**) | **0.7936 / 0.7972 (-20.1 %)** | **1.5323 / 1.4940 (-11.1 %)** | 5.7461 / 5.7446 (-3.8 %) |
| cube-wall-cloth | SPD2=0 | 0.6282 / 0.6327 | 0.4286 / 0.4307 | 0.9159 / 0.9238 | 6.0460 / 6.0484 |
| | SPD2=1 | 0.6242 / 0.6227 (**-1.2 %**) | **0.3449 / 0.3389 (-20.4 %)** | **0.8592 / 0.8468 (-7.3 %)** | 5.9610 / 5.9259 (-1.9 %) |

Part 2 is the stable quantity: **-20.1 % and -20.4 % on two different scenes**, with the two repeats
agreeing to 0.5 pp on the wrecking balls and 1.7 pp on the cube wall. Newton counts across the eight
windows: 190/188 vs 189/180, and 319/318 vs 318/318.

**Part 1 does not regress on either scene — it gets slightly faster** (-1.8 % / -1.2 %) without being
touched, the same K9 SM-sharing mirror s25 saw in the opposite direction. This is the check that
flipped round-4's s17 and w3's R3; it passes in both directions.

### Which half of the gain comes from which mechanism

Two extra builds with `Spd2Solver` / `Spd2Basis` forced, measured on **cube-wall-cloth**, whose
trajectory is reproducible (Newton 318-319 in every run) — the wrecking balls' pair population varies
enough between runs (grid 55-85 blocks) that the split is noise-dominated there and I do not quote it:

| build | part 2, 2 runs | mean | vs base |
|---|---|---:|---:|
| base: Eigen + explicit basis Q | 0.4286 / 0.4307 | 0.42965 | — |
| **TQL solver only**, explicit Q | 0.3791 / 0.3729 | 0.3760 | **-12.5 %** |
| Eigen solver, **basis-free only** | 0.3625 / 0.3700 | 0.36625 | **-14.8 %** |
| **both (shipped)** | 0.3449 / 0.3389 | **0.3419** | **-20.4 %** |

Close to additive, and the two halves are **comparable in size** — unlike part 1, where the basis was
the larger half by 1.5x. Both arms of the split are disjoint from the shipped arm, so neither
mechanism is redundant.

### End-to-end, default frame counts, one build

| scene | arm | mean ms/frame | **ms per Newton iteration** | Newton | PCG |
|---|---|---:|---:|---:|---:|
| rigid-wrecking-balls, **6 runs each way** | SPD2=0 | 23.056 / 22.711 / 22.613 / 22.541 / 22.009 / 23.210 | 5.7519 / 5.7014 / 5.7490 / 5.7065 / 5.6676 / 5.7785 | 481 478 472 474 466 482 | 13 075 / 13 155 / 12 855 / 12 550 / 12 920 / 13 250 |
| | **SPD2=1** | 21.993 / 22.496 / 23.148 / 22.082 / 22.717 / 22.627 | **5.6392 / 5.5776 / 5.7391 / 5.6500 / 5.6323 / 5.6450** | 468 484 484 469 484 481 | 12 605 / 12 970 / 12 900 / 12 655 / 13 110 / 12 930 |
| | | -0.79 % mean | **-1.37 % per iteration**, 33 of 36 pairwise comparisons favour B, Mann-Whitney U = 33 (p < 0.05) | | |
| cube-wall-cloth, 3 runs each way | SPD2=0 | 56.581 / 56.253 / 56.823 | 10.9653 / 10.9868 / 10.9065 | 516 512 521 | 20 195 / 19 800 / 20 165 |
| | **SPD2=1** | 53.439 / 55.722 / 54.497 | **10.7093 / 10.7159 / 10.7278** | 499 520 508 | 19 725 / 20 235 / 19 725 |
| | | **-3.54 % mean** | **-2.15 % per iteration, distributions disjoint** (A min 10.9065 > B max 10.7278) | | |
| **mas-bunny** (control), 3 each way | SPD2=0 | 51.736 / 51.843 / 51.720 | 11.1259 / 11.1490 / 11.1225 | **465 / 465 / 465** | 35 210 / 35 215 / 35 235 |
| | SPD2=1 | 51.715 / 51.703 / 51.644 | 11.1215 / 11.1188 / 11.1062 | **465 / 465 / 465** | 35 225 / 35 210 / 35 200 |
| | | -0.20 % | **-0.15 %**, Newton 465 in all six runs | | |
| **stiff-gipc-case2** (control), 3 each way | SPD2=0 | 170.987 / 170.440 / 171.667 | 25.7666 / 25.6842 / 25.8069 | 1 659 / 1 659 / 1 663 | 64 105 / 64 035 / 64 935 |
| | SPD2=1 | 169.634 / 170.943 / 170.222 | 25.7490 / 25.7599 / 25.6358 | 1 647 / 1 659 / 1 660 | 64 205 / 64 530 / 64 415 |
| | | -0.35 % | **-0.15 %, inside scatter** (arms interleave) | | |

**No scene regresses.** Newton and PCG counts overlap between the arms on every scene, and mas-bunny
is deterministic at 465 both ways, so nothing here is an algorithmic-path change.

**Why the end-to-end gain is smaller than s25's -4.14 % for a bigger kernel win.** s25 measured on a
tree without w3's s24; that step took the wrecking balls' kernel time from 6.80 to 5.95 ms/it without
changing frame time much, so the frame is now materially more host-bound and a fixed kernel saving
buys proportionally less wall time. -20 % of a 17 % kernel is -3.8 % of kernel time (measured) but
only -1.4 % of frame time per iteration. That is a property of the composed tree, not of this step,
and it is a standing argument for the `event_write_scene` readback still sitting on the pick list.

## 6. Cross-architecture expectation

**Part 2 shows the same latency-bound signature as part 1, and this gain is code-shape, not
occupancy.**

- `best_block_dim` returns **256 in both arms** and ptxas reports **255 registers in both arms**, so
  the kernel runs one 256-thread block (8 warps) per SM either way. The occupancy component is
  *measured to be zero*, not argued. This matters because w3's R3 established that occupancy-shaped
  changes to *this exact kernel* flip sign between cc 7.5 and cc 12.0; none of that applies here.
  The probe pass I threw away (§1) is the concrete demonstration of how easily the two get confused
  on this kernel — the register drop of a stub moved `best_block_dim` from 256 to 384 and inflated a
  -20 % into a -33 %.
- **The basis half (-14.8 %) removes instructions and temporaries**: a 9x4 `Q`, two 9x4 matrix
  products, a 9x9 `Hs` and a 9x9 `Hspd`, three `normalized()` calls, a cross product, the
  `|n^_0| >= 0.9` branch and a Gram-Schmidt loop. Fewer instructions at equal occupancy transfers
  everywhere. Spill traffic also drops 88 -> 8 bytes, which is a *larger* share of a faster SM's
  time, so if anything this component grows on the 5090.
- **The solver half (-12.5 %)** is s19's mechanism at a new call site and a new size (N = 4). w0
  measured it at 1.20x standalone on cc 7.5, and Jacobi at the *same* 1.20x, so N = 4 is the one size
  where the algorithm choice does not matter and only the implementation does. Neither depends on
  FP64 rate: the saving is instruction count, and the 5090's 1/64 FP64 makes an FP64-heavy kernel a
  *larger* share of its frame, as p01 and s15 both observed.

Part 2 costs 1.0116 ms/it on cc 8.9 against part 1's 1.0114 — the two branches, with completely
different arithmetic and a 5x different grid, land within 0.02 % of each other. That is the signature
of a kernel bound by one-block-per-SM latency rather than by its own work, which is exactly what s25
found for part 1 across cc 7.5 / 8.9 / 12.0. **I expect part 2 on the 5090 to sit at a similar share
and to take a similar -18 to -21 %.** The union should move by roughly half of that, because part 1
becomes binding again the moment part 2 drops below it — which is precisely what happened here in
reverse, and what this step's union numbers show (-11.1 % against part 2's -20.1 %).

## 7. Candidates for the next step

1. **`make_spd_contact<12,4>` — the M = 4 case — is the last piece of the same derivation and it is
   part 1's, not part 2's.** The basis-free form generalises: `range(Q) = s^ (x) R^3 (+) v_1 (x) n^
   (+) v_2 (x) n^`, with `v_1, v_2` an orthonormal basis of `{x in R^4 : sum x = 0, x . s^ = 0}`.
   It would remove a **12x5 `Q`, a 12x12 `Hs` and a 12x12 `Hspd`** from the PT dim-4 and EE dim-4
   branches, which is a far bigger temporary than anything M <= 3 saves. s25 measured
   `EE_barrier_make_spd` at 8.9 % of part 1 and the PT branch inside its 40.5 % remainder, so the
   prize is smaller in share but the per-call saving is larger. The `Basis` template axis is already
   in place and part 1 already has a spare `false` instantiation.
2. **Part 2's remaining 58 % is the two symbolic Hessians and the writes, and nobody has measured
   them apart.** `PE_barrier_gradient_hessian` (9x9) and `PP_barrier_gradient_hessian` (6x6) plus the
   `TripletMatrixAssembler` half-block writes are now the dominant half of a 17 %-of-suite kernel.
   The stub scaffolding for this (`UIPC_STUB_PE_GH`, `PP_GH`, `PE_WRITE`, `PP_WRITE`) is written and
   proven against the control in this step's probe but was not spent — the projection answer arrived
   first. One pinned build per stub would settle it.
3. **Part 2's 2 064 B stack frame is *not* the projection** (it only moved 48 B when the projection
   was rewritten, while spill traffic fell 88 -> 8 B). Whatever holds 2 KB live is in the symbolic
   Hessian path, and at 255 registers it is what pins the kernel to 8 warps per SM. This is the
   register-pressure lever w3's R3 could not find a way into from the launch side.
4. **The `event_write_scene` 460 KB readback** (388 us/frame, 53 % of remaining host stall) —
   carried over from s22 and now *more* valuable, because §5 shows the composed tree is materially
   host-bound: a -20 % on the suite's largest kernel bought only -1.4 % of frame time on the
   wrecking balls. Still needs an application-semantics decision.

## 8. Found in another worker's area — and one incident

### The coordinator's `perf/round5` already contains this step's code, unlabelled

**Commit `4012998d` ("round-5 s26 accepted -- the zero-fill was dead") carries my two contact files
along with w0's zero-fill change.** `git log -S"Spd2Basis"` names it as the single commit that
introduced this step's code to `perf/round5`, and `git show 4012998d --name-only` lists
`codim_ipc_simplex_normal_contact_function.h` and `ipc_simplex_normal_contact.cu` next to the
round record. Its message does not mention them.

Cause: `/workspace/deps/libuipc-r5-coord` is the coordinator's shared worktree and also this
session's working directory, so my in-progress edits were sitting in it uncommitted when w0's step
was committed with a path-inclusive `git add`. The content is byte-identical to what this branch
ships (`git diff HEAD` over both files is empty at `8a7e0ea0`), so `perf/round5` is not *broken* — it
is carrying an ungated, unrecorded change under another step's number, with no ledger row and no
env-switch entry. **The coordinator should reconcile: either take this branch's commit as the
provenance for those two files, or split them out of `4012998d`.** I have not touched the coordinator
branch; this work is committed on `perf/round5-w2-ada` from a clean `72de9e5c`.

Two process rules follow, and both belong in the brief:
- **A worker must not edit files inside the coordinator's worktree.** Edit on the box, or in a
  private clone. This is the second time round 5 has lost track of who owns a change.
- **The source comments in this commit say `round-5 (s26)`, deliberately.** Those exact bytes are
  already on `perf/round5` via `4012998d`, so leaving them makes this branch's two source files
  **byte-identical** to what the coordinator branch already carries -- which is the property that
  makes reconciliation a no-op rather than a diff to review. The step is filed as s31 everywhere
  else.
- **Step numbers are being allocated twice.** This step was briefed as "part 2" with no number; by
  the time it finished, `s26` through `s30` were taken and `s26` in particular was already used for
  an unrelated w0 step. It is filed as **s31**.

### Other findings

- **`best_block_dim` has no override, and that makes every stage-stubbing probe on a 255-register
  kernel unsound by default.** Removing work drops registers, which changes the auto-selected block
  size, which changes occupancy — silently, with no profiler warning. My first probe pass reported
  -33.5 % where the truth was -20 %. `launch.h`'s `best_block_dim` should take an env override
  (`UIPC_BLOCK_DIM_PIN`) purely so probes can pin it; that is w3's file and w3's call.
- **The IPC simplex *frictional* contact `do_assemble_kernel` is now 4.77 % of the wrecking balls**
  (0.2835 ms/it, grid 383x64, 186 registers) and is still on nobody's list, for the second step
  running. It is a separate TU with its own `make_spd_contact` call sites, which means **it inherits
  the `Basis` axis for free** — it would need a launch-side flag and nothing else.
- **This box's `baseline_tests.txt` is stale for `backend_cuda`** (352/22 at `perf-round5-base`
  against 448/23 at the head, from w3's s24 `qr_svd.cu`). Every later worker on this box will hit the
  same mismatch and have to re-derive that it is benign. It should be re-captured at the head.

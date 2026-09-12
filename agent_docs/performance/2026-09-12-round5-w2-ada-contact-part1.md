# Round 5, s24 (w2-ada, RTX 4060 Ti, cc 8.9) — contact G+H part 1: half of it is one 9x9 eigen-solve

Worker w2-ada's second step. Rotated off the D2H pipeline (s22 measured its remaining ceiling at
~1.5 % of frame) onto **contact G+H part 1**, `do_assemble_kernel<false, 1, *>` — the PT + EE branch,
the largest single kernel of the suite and the only top-ranked one no round-5 worker had touched.

Measured on `perf/round5` at **`d4eff24e`** (i.e. *with* w0's s19, w3's s20 and w2's s22), one build,
env A/B, nsys 2024.6.2 60-frame windows.

## 1. Where part 1's time actually goes

### Baseline ranking on this box (ms per Newton iteration, 60-frame nsys windows)

**rigid-wrecking-balls** — 188 Newton / 3 935 PCG, 7.069 ms/it of kernel time:

| rank | ms/it | share | kernel |
|---|---:|---:|---|
| **1** | **1.2227** | **17.30 %** | **contact G+H part 1, `do_assemble_kernel<false,1,true>` (PT+EE)** |
| 2 | 1.0399 | 14.71 % | contact G+H part 2, `<false,2,false>` (PE+PP) |
| 3 | 0.5361 | 7.58 % | `abd_..._dytopo_effect_pair_warp` |
| 4 | 0.3484 | 4.93 % | `InfoStacklessBVH_stacklessSelf` |
| 5 | 0.3186 | 4.51 % | IPC simplex **frictional** contact `do_assemble_kernel<false>` |
| 6 | 0.2605 | 3.68 % | `InfoStacklessBVH_pairFilter<true>` |
| 7 | 0.2384 | 3.37 % | `ortho_potential` G/H (s19+s20 already applied) |
| 8 | 0.2345 | 3.32 % | `abd_diag_preconditioner_do_assemble` |

Union of the contact G+H launches (the two parts run concurrently across the K9 fork): **1.930 ms/it**.

**cube-wall-cloth** — 319 Newton, 7.098 ms/it: hinge G/H 1.5556 (21.92 %), **part 1 0.7221 (10.17 %)**,
part 2 0.4397 (6.19 %); union 1.0051.

So part 1 is rank 1 on cc 8.9 exactly as it is on cc 7.5 (15.9 %) and cc 12.0 (15.7 %). **Three
architectures, three independent measurements, same rank.**

### Stage-stubbing probe (round-4 s17 probe (c) method: comment the call out, rebuild one TU, nsys)

rigid-wrecking-balls, ms per Newton iteration. The stub builds change the trajectory (the Hessian is
no longer projected), so these are attribution numbers, not A/B numbers:

| build | part 1 | part 2 | union |
|---|---:|---:|---:|
| control (all stubs off) | 1.2165 | 1.0236 | 1.9359 |
| the mollified-EE 9x9 projection stubbed | 0.6008 | 0.8882 | 1.5427 |
| **both** EE projections stubbed | 0.4919 / 0.4935 | 0.7640 / 0.7726 | 1.3826 / 1.3709 |

(the last row was measured twice, in two independently patched builds, and agrees to 0.3 %.)

| component of part 1 | ms/it | share of part 1 |
|---|---:|---:|
| **`make_spd_translation_free_4x3` — the 9x9 eigen-solve on the mollified EE Hessian** | **0.616** | **50.6 %** |
| `EE_barrier_make_spd` — the reduced 5x5/4x4/3x3 projection of the un-mollified partition | 0.108 | 8.9 % |
| **EE projection, total** | **0.724** | **59.5 %** |
| everything else: the distance flags, *both* 12x12 symbolic Hessians (`edge_edge_distance2_hessian`, `edge_edge_mollifier_hessian`), the mollifier combination, the gradient/Hessian writes, and the entire PT branch | 0.493 | 40.5 % |

**This was not predicted.** Round 4's pick list expected part 1's remaining cost to be spread over the
mollified Hessian construction and the ABI-call overhead; s17 had already shown the ABI call is not
what the family pays for. It is one function, and it is the one w0's s19 replaced *everywhere except
here* — s19's own comment says so: "the contact branches and the cold call sites keep the old solver
until a follow-up step measures them." This step is that follow-up.

Second-order but worth recording: **stubbing part 1's projection also speeds up part 2 by 7 %**
(1.0236 -> 0.7640) although part 2's own code is untouched. At 255 registers only one 256-thread block
fits per SM, so the two forked parts genuinely compete for SMs — the K9 coupling round-4's s17 and
w3's R3 both ran into, here acting in the *helpful* direction.

## 2. The change

`UIPC_CONTACT_SPD_TQL=0` restores the old path.

Part 1's PSD projection moves onto two mechanisms that already exist in the tree and were never wired
to this call site:

1. **s19's fixed-size tridiagonal-QL eigen-solve.** `make_spd_contact<N, M>` gains a third template
   parameter `Solver`, forwarded from `PT_barrier_make_spd<Solver>` / `EE_barrier_make_spd<Solver>`
   into `make_spd<M+1, Solver>`. `Solver = 1` is `cuda_tool::eigen::evd_tridiag_ql` — the same
   algorithm Eigen runs, without its dynamic-size block expressions and its out-of-line
   `selfadjoint_matrix_vector_product<double,long>`.
2. **K16's blocked translation-free basis.** The mollified EE path calls
   `make_spd_translation_free_4x3_blocked<1>` instead of `make_spd_translation_free_4x3<0>` — the
   same 9x9 projection assembled from 3x3 blocks with the constant Helmert weights, with no 12x9
   basis matrix and none of its `Q^T H Q` / `Q Hr Q^T` temporaries. Round 3 introduced this form and
   round 4 wired it into the discrete-shell hinge; **it had no caller in the contact family.**

`SpdTql` is a template parameter of `do_assemble_kernel`, not a runtime flag, so each instantiation
carries one code path's stack frame (the s14 lesson). Gradient-only launches and part 2 never reach a
12x12 projection, so they are only instantiated with `SpdTql = false`.

**Part 2 (PE+PP) is deliberately left on the old solver** — it is w3's area this round.

### Which half of the gain comes from which mechanism (three separate builds, wrecking balls)

| build | part 1 | vs base | part 2 | union |
|---|---:|---:|---:|---:|
| base: Eigen `SelfAdjointEigenSolver` + dense 12x9 Q | 1.2196 | — | 1.032 | 1.933 |
| TQL solver, dense Q | 1.1174 | −8.4 % | 1.0213 | 1.8420 |
| Eigen solver, blocked Q | 1.0621 | −12.9 % | 0.9810 | 1.8613 |
| **TQL solver + blocked Q (shipped)** | **1.0324** | **−15.4 %** | 0.9764 | 1.8370 |

The two mechanisms are close to additive and **the blocked basis is the larger half** — which is a
result in its own right, because the hinge (where the blocked form was adopted in round 3) has a much
smaller frame than part 1's 13 KB.

### Static shape (`cuobjdump -res-usage`, sm_89, one object file so only the template argument differs)

| instantiation | registers | stack frame |
|---|---:|---:|
| part 1, `EEReducedRange=true`, `SpdTql=false` | 255 | 13 320 B |
| part 1, `EEReducedRange=true`, **`SpdTql=true`** | 255 | **10 488 B (−21.3 %)** |
| part 1, `EEReducedRange=false`, `SpdTql=false` | 255 | 13 272 B |
| part 1, `EEReducedRange=false`, **`SpdTql=true`** | 255 | **9 040 B (−31.9 %)** |
| part 2 (untouched) | 255 | 2 104 B |

**Registers are unchanged at 255, so occupancy is unchanged** — one 256-thread block per SM either
way. The gain is instruction count and local-memory traffic, with a measured-zero occupancy
component. That matters for the transfer question (§6).

## 3. Correctness

`bash /root/work/run_tests.sh /root/work/tests_s24`, against this box's `baseline_tests.txt`:

| binary | baseline | s24 |
|---|---|---|
| common | 11 assertions / 3 cases | **same** |
| core | 1 112 / 36 | **same** |
| geometry | 2 730 / 46 | **same** |
| sanity_check | 100 / 3 | **same** |
| regression | 4 / 1 | **same** |
| backend_cuda | 352 / 22 | **same** |
| sim_case | 14 213 / 95 | **same** |
| pytest | (baseline capture was broken: rc=2, 7 collection errors) | 48 passed, 1 skipped |

Identical counts, no new failures, no flakes. The baseline was captured at `perf-round5-base`, so this
also confirms **s19, s20 and s22 moved zero assertions** on this box.

## 4. Numerics — rounding level, proved on 1.8 million randomised pairs

Standalone verifier (`verify_s24.cu`, the s14/s16 pattern: the repo's own device functions, on the
GPU, both projection paths on the same input). 3 seeds x 300 000 EE pairs and 3 x 300 000 PT pairs.
EE pairs: translated up to 10 length units from the origin, one in eight built nearly parallel so
12.5 % land in the mollifier-active regime (37.6 k of 300 k measured), gaps 1e-4..5e-2, kappa*dt^2 up
to 1e8, thickness 0 and 0.1 d, flag dims 2/3/4 = 68.7 k / 164.1 k / 67.1 k. Maximum relative
difference over the whole 12x12:

| set | p50 | p90 | p99 | p99.9 | max | negative diagonal |
|---|---|---|---|---|---|---|
| EE seed 0 / 1 / 2 | 7.40e-16 | 1.99e-15 | 3.54e-15 | 5.22e-15 | **3.11e-14 / 2.04e-14 / 2.92e-14** | 0 |
| PT seed 0 / 1 / 2 | 7.22e-16 | 1.91e-15 | 3.19e-15 | 4.18e-15 | **1.50e-14 / 7.58e-15 / 6.55e-15** | 0 |

For scale, s16's verifier of the *reduced-range* EE projection against `make_spd<12>` — an accepted
round-4 change — reported max 4.4e-9 on the same kind of sample. This change is **five orders of
magnitude tighter** than one the project already accepted: it is the same algorithm on the same
subspace, only a different implementation of the eigen-solve and a different summation order in the
basis products.

0.44 % of the EE samples (1 239-1 315 per seed) produce a NaN — **in both paths, with identical
counts**. They are degenerate configurations the random sampler emits (a vanishing flagged distance),
not something the new path introduces; they are excluded from the percentiles above.

## 5. Performance

### Targeted scope, A/B in one build via `UIPC_CONTACT_SPD_TQL` (nsys, 60 frames)

ms per Newton iteration, **two runs each way**. **Part 1, part 2 and their union**, as the brief
requires:

| scene | | part 1 (mine) | part 2 (untouched) | **union** | whole-scene kernel |
|---|---|---:|---:|---:|---:|
| rigid-wrecking-balls | TQL=0 | 1.2233 / 1.2228 | 1.0495 / 1.0490 | 1.9659 / 1.9310 | 7.1038 / 7.0965 |
| | TQL=1 | **1.0293 / 1.0302 (−15.8 %)** | 0.9762 / 0.9854 (−6.6 %) | **1.7848 / 1.8201 (−7.5 %)** | 6.8002 / 6.8341 (−3.8 %) |
| cube-wall-cloth | TQL=0 | 0.7221 / 0.7194 | 0.4397 / 0.4252 | 1.0051 / 1.0247 | 7.0980 / 7.0724 |
| | TQL=1 | **0.6266 / 0.6248 (−13.2 %)** | 0.4398 / 0.4188 (−2.4 %) | **0.9129 / 0.9202 (−9.7 %)** | 6.9961 / 6.9559 (−1.7 %) |

Part 1 is the stable quantity: its two repeats agree to 0.1 pp on both scenes (−15.9 / −15.8 %,
−13.2 / −13.2 %). The **union** carries more scatter (−9.2 / −5.7 % on the wrecking balls) because it
also depends on how the two forked launches happen to overlap in a given run; the pair-averaged figure
is the one to quote, and it is negative on both scenes in all four runs.

**The union moves, and part 2 does not regress on either scene.** On the wrecking balls part 2 gets
*faster* (−6.6 %) without being touched — part 1 releases its SMs earlier. On the cube wall, where
part 2 is a quarter of the grid, it is flat. This is the check round-4's s17 and w3's R3 both failed;
it passes here in both directions.

Newton counts across the eight nsys windows: 189/188 and 188/188 (wrecking balls), 319/318 and
318/317 (cube wall) — matched within one iteration in every pair.

### End-to-end, default frame counts, 3 runs each way, one build

| scene | arm | mean ms/frame | median | **ms per Newton iteration** | Newton | PCG |
|---|---|---:|---:|---:|---:|---:|
| rigid-wrecking-balls | TQL=0 | 27.271 / 28.049 / 28.100 | 23.299 / 23.434 / 23.903 | 6.9333 / 6.9399 / 6.9959 | 472 / 485 / 482 | 12 580 / 13 150 / 13 420 |
| | **TQL=1** | 25.655 / 26.479 / 27.153 | 21.708 / 22.550 / 22.591 | **6.4675 / 6.5786 / 6.7601** | 476 / 483 / 482 | 13 060 / 13 245 / 13 145 |
| | | **−4.96 % mean** | −5.2 % | **−4.14 % per iteration, distributions disjoint** | | |
| cube-wall-cloth | TQL=0 | 61.796 / 60.624 / 61.259 | 53.878 / 55.914 / 54.894 | 12.1169 / 11.9575 / 12.0351 | 510 / 507 / 509 | 20 195 / 19 480 / 19 935 |
| | **TQL=1** | 65.505 / 60.326 / 60.193 | 64.334 / 54.745 / 54.228 | **11.9754 / 11.8752 / 11.8489** | 547 / 508 / 508 | 20 650 / 19 735 / 20 020 |
| | | (rep 1 of arm B is a Newton outlier, 547) | | **−1.14 % per iteration** | | |
| **mas-bunny** (control) | TQL=0 | 53.623 / 53.560 / 53.580 | 59.647 / 59.583 / 59.764 | 11.5318 / 11.5182 / 11.5226 | **465 / 465 / 465** | 35 200 / 35 235 / 35 205 |
| | TQL=1 | 53.606 / 53.583 / 53.573 | 59.631 / 59.646 / 59.849 | 11.5282 / 11.5232 / 11.5210 | **465 / 465 / 465** | 35 225 / 35 220 / 35 195 |
| | | **−0.01 %, flat** | | **−0.00 %** | Newton 465 in all six runs | |
| **stiff-gipc-case2** (control) | TQL=0 | 186.906 / 186.946 / 185.976 | | 28.1654 / 28.1376 / 28.1953 | 1 659 / 1 661 / 1 649 | 64 130 / 64 025 / 63 435 |
| | TQL=1 | 186.418 / 187.800 / 186.579 | | 28.2795 / 28.2322 / 28.1162 | 1 648 / 1 663 / 1 659 | 64 570 / 64 550 / 64 045 |
| | | +0.17 % | | +0.15 %, **inside scatter** (arms overlap) | | |

Read the per-Newton-iteration column on the two contact scenes: both have trajectory scatter in the
mean. On the wrecking balls the three values of the two settings **do not overlap**
(6.9333-6.9959 against 6.4675-6.7601). On the cube wall they nearly do not (11.9575-12.1169 against
11.8489-11.9754; one pair overlaps by 0.02).

case2 is genuinely flat rather than slightly positive: its EE population is ~250 pairs per launch and
mollifier-free, so part 1 is ~4 % of its kernel time and a −13 % there is −0.5 % of kernel time, below
this scene's run-to-run scatter. mas-bunny is deterministic (Newton 465 in all six runs) and did not
move by 0.01 %; it has almost no EE pairs at all.

**No scene regresses.**

## 6. Cross-architecture expectation

Two components, and they transfer differently:

- **The blocked basis (the larger half, −12.9 % of part 1 on its own)** removes a 12x9 matrix and two
  12x9 products from a kernel with a 13 KB stack frame. It is fewer instructions and less
  local-memory traffic at **unchanged registers and unchanged occupancy** (255 -> 255, one block per
  SM, measured). That is algorithmic/code-shape and **should transfer, at equal or larger relative
  size on the 5090**: local memory is backed by L2/DRAM everywhere, and a faster SM makes the same
  spill traffic a larger share of the kernel.
- **The TQL solver (−8.4 % on its own)** is exactly s19's mechanism at a new call site. s19 measured
  it as −15.0 % on the hinge on cc 7.5 and the coordinator's note already classifies it as a
  code-shape win with an uneven register-pressure component. Here the register component is
  *measured to be zero* (255 both ways), which removes the part that would not have transferred.

Neither component is occupancy tuning, and neither depends on FP64 rate — the eigen-solve is FP64 but
the *saving* is instruction count and local-memory traffic, not FP64 throughput. I expect the 5090 to
show a part-1 gain in the same −13 to −16 % band.

### A number against the other boxes

The same kernel, ms per launch, all from 60-frame nsys windows on rigid-wrecking-balls:

| box | cc | part 1 ms/launch, before | share of scene kernel time |
|---|---|---:|---|
| RTX 2070 SUPER (round 4, after s17) | 7.5 | 1.802 ms/it | 15.9 % |
| RTX 4060 Ti (this step, before) | 8.9 | 1.223 ms/it (1.284 ms/launch) | 17.3 % |
| RTX 5060 Ti (w3, round 5) | 12.0 | 1.197 ms/it | 15.7 % |

The kernel costs almost the same on cc 8.9 and cc 12.0 and only ~50 % more on cc 7.5, while the rest
of the frame scales much better — which is the signature of a latency-bound, one-block-per-SM kernel
rather than a throughput-bound one, and is consistent with s16's original reading. Its *share* is
therefore stable across all three architectures, and so a −15 % on it is a −2 to −3 % of scene kernel
time everywhere.

## 7. Candidates for the next step

1. **Part 2 (PE+PP) has the identical opportunity and is 14.7 % of the wrecking balls.** It is w3's
   area, so this step left it alone, but the evidence is direct: its `PE_barrier_make_spd` /
   `PP_barrier_make_spd` go through the same `make_spd_contact` (now `Solver`-templated, so it is a
   one-token change) into `make_spd<4>` and `make_spd<3>`. `make_spd<3>` takes Eigen's closed form
   either way, so only the PE dim-3 branch (`make_spd<4>`) would move — a smaller prize than part 1's
   9x9, but free to try. Note w0's R1 measured Jacobi *winning* at N = 4 (1.20x), the one size where
   it does; TQL at N = 4 was never measured.
2. **After this step the contact G+H union is limited by part 2, not part 1** (wrecking balls:
   part 1 1.030, part 2 0.985, union 1.820). Whoever takes part 2 next should expect the union to
   move by roughly half of whatever part 2 moves, because part 1 then becomes binding again.
3. **The remaining 40 % of part 1 is two 12x12 symbolic Hessians.** With the projection now at ~44 %
   of the branch, `edge_edge_distance2_hessian` + `edge_edge_mollifier_hessian` are the next target,
   and neither has ever been measured separately. A stub of the mollifier arithmetic alone suggested
   it is small, but that probe was contaminated (see §8) and should be redone cleanly.
4. **The `event_write_scene` 460 KB readback** (388 us/frame, 53 % of all remaining host stall) —
   carried over from s22, still needs an application-semantics decision about whether vertex state may
   lag a frame. Not a kernel item.

## 8. Found in another worker's area

- **Part 1's launch geometry is under-occupied and it is `best_block_dim` again.** On this 34-SM part,
  part 1 runs a median **28 blocks x 256 threads** on the wrecking balls and **19 x 256** on the cube
  wall — 82 % and 56 % of the SMs hold one block each, and the rest hold nothing. At 255 registers a
  256-thread block is the whole SM's register file, so shrinking the block to 32 threads would keep
  the same 8 warps per SM while spreading them over *all* 34 SMs instead of 19-28. This is exactly the
  pathology w0 and w3 converged on, at a launch site neither of them listed. **Reported, not fixed** —
  w3 owns the consolidation of `spread_block_dim` / `fitted_block_dim` and should add this site to the
  sweep. It is orthogonal to this step (which changes no launch parameter) and the two should compose.
- **The IPC simplex *frictional* contact `do_assemble_kernel` is 4.5 % of the wrecking balls**
  (0.3186 ms/it, 100 blocks, 186 registers) and appears on nobody's list. It is a separate TU
  (`ipc_simplex_frictional_contact.cu`) with its own projection call sites.
- **The round-4 post-s17 pick-list item "`make_spd_contact`'s two NxN temporaries" is already
  refuted** and should be struck: round 4's own s17 implemented the `Identity` flag, measured ptxas
  reporting *identical* stack frames with and without it, and reverted. The same toolchain (CUDA
  12.8) is on this box. The brief passed it on as "never attempted"; it was attempted.
- **Probe hygiene, for the record.** Two of this step's stub probes were run with a broken reset
  (`sed -E 's/^#define (UIPC_STUB_[A-Z_]+) 1$/.../'` does not match a macro name ending in a digit),
  so the "EE_PROJ5 alone" and "MOLL_MATH alone" arms actually ran with the other stubs still on. The
  accidental duplicate turned into a useful control — it reproduced the both-projections-stubbed
  number to 0.3 % in an independently patched build — but the mollifier-arithmetic attribution in §7.3
  is not trustworthy and is flagged as such.

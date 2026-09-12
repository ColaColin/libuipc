# Round 5 / w0-turing — `make_spd<N>`, the PSD projection eigen-solve (RTX 2070 SUPER, cc 7.5)

Worker record for perf round 5, step **s19**, base `perf-round5-base` (`890482c2`), branch
`perf/round5-w0-turing`, commit `7abdf0ad`. Box: local RTX 2070 SUPER, CUDA 12.8, sm_75,
`build-perf` (Release, tests ON). Raw artefacts (microbenchmark sources, nsys reports, kernel
summaries, dumps, logs) in `/workspace/output/round5/w0-turing/`.

**The coordinator should paste the ledger row, the two rejections and the cross-architecture note
below into `agent_docs/performance/2026-09-12-perf-round5.md` on `perf/round5`** — that file lives
on the coordinator branch, so this worker branch carries its own copy to avoid a merge conflict.

## 1. The assignment's hypothesis, and why it is wrong

Round 4 measured `make_spd<N>` as its biggest single cost centre (s17 probe (c)): 82 % of the
discrete-shell hinge kernel and 69 % of the ABD ortho potential, both ending in Eigen's
`SelfAdjointEigenSolver::compute()`. The round-5 pick list proposed replacing it with a fixed-sweep
cyclic Jacobi with a *relative* off-diagonal threshold, on the theory that the kernels pay for
Eigen's dynamic-size internals and out-of-line calls rather than for arithmetic, so 4x the flops
would still win if they are branch-free and unrolled.

The standalone A/B was done first, as the assignment asked. **Jacobi loses.** See rejection R1.

What the probe *did* confirm is that the implementation, not the algorithm, is where the slack is:
the same Householder + implicit-shift QL that Eigen runs, written as fixed-size loops
(`cuda_tool::eigen::evd_tridiag_ql`), is 1.15-1.39x faster than Eigen's at N = 9 standalone and
-15 % / -8 % per launch inside the two real kernels.

## 2. Standalone microbenchmark (sm_75, `spd_bench.cu` / `spd_bench2.cu`)

Both paths compiled into one kernel with the backend's own flags (`-O3 -std=c++20 -dc
--expt-relaxed-constexpr`, `--generate-code=arch=compute_75,code=[compute_75,sm_75]`), 4 projections
per thread, best of 5. Speedup vs Eigen's `SelfAdjointEigenSolver` (>1 = faster):

| input | threads | N | jacobi | jacobi (packed upper) | **tridiag QL** |
|---|---:|---:|---:|---:|---:|
| random | 512 | 4 | 1.20x | 1.22x | **1.20x** |
| random | 512 | 5 | 0.91x | 0.93x | **1.19x** |
| random | 512 | 9 | 0.49x | 0.62x | **1.22x** |
| random | 512 | 12 | 0.38x | 0.39x | 0.96x |
| random | 2 048 | 9 | 0.49x | 0.50x | **1.00x** |
| random | 16 384 | 9 | 0.59x | 0.64x | **1.22x** |
| random | 16 384 | 12 | 0.29x | 0.57x | **1.17x** |
| real hinge 9x9 | 512 / 2 048 / 16 384 | 9 | 0.49 / 0.49 / 0.57x | 0.62 / 0.51 / 0.62x | **1.39 / 1.15 / 1.30x** |
| real ortho 9x9 | 512 / 2 048 / 16 384 | 9 | 0.67 / 0.67 / 0.76x | 0.69 / 0.70 / 0.83x | **1.07 / 1.06 / 1.18x** |

"real" = 251 hinge and 107 ortho 9x9 matrices dumped out of running scenes by an in-kernel probe
(sampled every 1 021st call so they span the whole run, not just the rest state).

ptxas frames for the microbenchmark kernels (N = 9): Eigen 3 488 B / 0 spills, tridiag QL
**1 440 B / 0 spills**, Jacobi 3 704 B / 4 296 B spills, packed Jacobi 1 432 B / 1 736 B spills.

## 3. The change (s19)

- `cuda_tool::eigen::evd_tridiag_ql<T, N>` — EISPACK `tred2` + `tql2` as fixed-size loops.
  Eigenvalues are not sorted (`make_spd` does not need them sorted).
- `make_spd<N, Solver>` — `Solver` is a **template parameter**, so each instantiation carries one
  code path's stack frame (the s14 lesson; a runtime flag would make every frame the union).
  `Solver = 0` is Eigen's solver and the exact pre-round-5 code, `Solver = 1` the new one, whose
  reconstruction sums only the upper triangle of `V diag(w) V^T` and mirrors it. `N <= 3` always
  takes Eigen's closed-form `computeDirect`, which beats both.
- The default is **0**, so this step changes exactly the two call sites it measures: the
  discrete-shell hinge (`discrete_shell_bending.cu`, via `make_spd_translation_free_4x3[_blocked]<Solver>`)
  and the ABD ortho potential (`ortho_potential.cu`). Both select the solver at launch from the env
  switch **`UIPC_MAKE_SPD_JACOBI`** (`=0` restores Eigen — the rollback and the A/B reference).
  The contact branches (`make_spd_contact` -> `make_spd<3|4|5>`) and the cold call sites (joints,
  stitches, ARAP) are deliberately untouched; see "next steps".

ptxas, sm_75, the instantiations the benchmarks launch:

| kernel instantiation | stack frame | spill stores / loads | registers |
|---|---:|---:|---:|
| hinge `<Proj=1, Solver=0>` (old) | 3 472 B | 2 456 / 2 456 | 255 |
| hinge `<Proj=1, Solver=1>` (new) | **3 104 B** | **1 576 / 1 576** | 255 |
| ortho `<Solver=0>` (old) | 2 240 B | 0 / 0 | 254 |
| ortho `<Solver=1>` (new) | **792 B** | 0 / 0 | **192** |

## 4. Numerics — rounding-level, proved

Device verifier: both projections run on the same input in one kernel, max relative Frobenius
difference `|make_spd_eigen(H) - make_spd_tql2(H)|_F / |H|_F`.

| population | N | samples | max rel. diff | mean | non-finite |
|---|---:|---:|---:|---:|---:|
| wide spectra, ~30 % negative, kappa up to 1e10 | 4 / 5 / 9 / 12 | 4 x 200 k | 4.3e-15 / 5.1e-15 / 6.7e-15 / 7.1e-15 | ~1e-15 | 0 |
| near-degenerate clusters (1e-13 apart) | 4 / 5 / 9 / 12 | 4 x 200 k | 2.6e-15 / 3.1e-15 / 3.8e-15 / 4.4e-15 | ~1e-15 | 0 |
| rank-deficient (half the spectrum exactly 0) | 4 / 5 / 9 / 12 | 4 x 200 k | 3.6e-15 / 4.5e-15 / 5.7e-15 / 6.3e-15 | ~1e-15 | 0 |
| already PSD | 4 / 5 / 9 / 12 | 4 x 200 k | 4.3e-15 / 5.1e-15 / 6.7e-15 / 7.1e-15 | ~1e-15 | 0 |
| **real hinge Hessians** (dumped from cube-wall-cloth) | 9 | 200 k | **3.8e-15** | 1.1e-15 | 0 |
| **real ortho Hessians** (dumped from wrecking balls) | 9 | 200 k | **2.8e-15** | 1.5e-15 | 0 |

That is an order of magnitude tighter than the 1.28e-13 that round 4's s14 accepted for the same
projection. Not bit-identical, and not claimed to be.

End-to-end evidence that the trajectory does not change beyond the existing noise: mas-bunny (the
control scene, which has neither constitution) gives Newton 465 in all four runs and a centroid
that differs in the 8th digit between *any* two runs, switch or no switch; cube-wall-cloth and
case2 stay inside their documented Newton/PCG scatter (see the ledger row).

## 5. Ledger row (for the shared record)

| Step | Worker | Commit | Change | Scope measurement | End-to-end effect | Tests | Verdict |
|---|---|---|---|---|---|---|---|
| s19 | w0-turing | `7abdf0ad` | **Fixed-size tridiagonal-QL eigen-solve for the PSD projection** (`cuda_tool::eigen::evd_tridiag_ql`, `make_spd<N, Solver>`): the same Householder + implicit-shift QL algorithm Eigen's `SelfAdjointEigenSolver::compute()` runs, written as fixed-size loops without Eigen's dynamic-size block expressions and its out-of-line `selfadjoint_matrix_vector_product<double, long>`, plus an upper-triangle-only `V diag(w) V^T` reconstruction. Solver is a template parameter (one stack frame per instantiation, the s14 lesson); wired into the discrete-shell hinge and the ABD ortho potential, `UIPC_MAKE_SPD_JACOBI=0` = Eigen. `N <= 3` keeps `computeDirect`. ptxas sm_75: hinge 3 472 -> 3 104 B frame, 2 456 -> 1 576 B spill stores; ortho 2 240 -> 792 B frame, 254 -> 192 registers | nsys 60-frame A/B in one build, ms per launch (identical launch counts): `DiscreteShellBending_do_compute_gradient_hessian` **2.6189 -> 2.2271 on cube-wall-cloth (-15.0 %)** and **2.5912 -> 2.2838 on stiff-gipc-case2 (-11.9 %)**; `ortho_potential_compute_gradient_hessian` **0.9150 -> 0.8392 on cube-wall (-8.3 %)** and **1.1371 -> 1.0850 on wrecking balls (-4.6 %)**. Net of the s17 stub probe's non-projection remainder, the projection itself is **-18.3 %** (hinge) and **-13.4 %** (ortho), i.e. 1.22x / 1.16x, matching the standalone microbenchmark. Scene kernel total: cube-wall 4 305.0 -> 4 163.6 ms (-3.3 %), case2 10 786.5 -> 10 527.9 (-2.4 %), wrecking balls 2 181.7 -> 2 177.2 (-0.2 %) | Official benchmarks, default frames, env A/B in one build, 2 runs each way (mean / median ms per frame; Newton; PCG): cube-wall-cloth 91.86 / 83.18 (517, 20 360) and 92.11 / 85.66 (516, 20 335) -> **86.33 / 81.30** (507, 19 490) and **87.78 / 82.01** (512, 19 810) = **-5.4 % mean**, disjoint distributions, **-4.1 % per Newton iteration** (17.81 -> 17.09 ms); stiff-gipc-case2 223.67 / 229.34 (1 661, 64 190) and 224.68 / 229.77 (1 663, 65 845) -> **219.70 / 227.01** (1 659, 63 655) and **220.63 / 227.10** (1 653, 64 765) = **-1.8 % mean**, disjoint, -1.5 % per Newton iteration; rigid-wrecking-balls 39.93 (474) / 40.86 (478) -> 40.51 (475) / 38.90 (467) = -1.7 % mean but the distributions overlap, -0.7 % per Newton iteration, i.e. inside scatter as predicted (ortho is 9.8 % of its kernel time); mas-bunny (control, neither constitution) 74.06 / 73.93 -> 74.12 / 74.06, Newton 465 in all four runs | common 11/3, core 1112/36, geometry 2730/46, sanity_check 100/3, regression 4/1, backend_cuda 352/22, sim_case 14213/95, pytest 48 passed / 1 skipped — identical to the round-4 baseline table, no flakiness in this pass | accepted |

## 6. Rejections (for the shared record)

### R1 — the fixed-sweep cyclic Jacobi of the pick list is **slower**, not faster

Numbers in §2. Jacobi wins only at N = 4 (1.20x), where its whole working set (32 doubles) fits in
registers; from N = 9 on, `A` and `V` together are 162 doubles and spill whatever the layout, so the
~4x flops are paid in full (0.49-0.64x at N = 9, 0.29-0.57x at N = 12). Two variants were measured:
the naive full-symmetric update, and a packed upper-triangle version that halves the rotation's
stores. The relative threshold the pick list asked for was implemented
(`off(A)^2 <= 1e-30 |A|_F^2`, per-rotation skip at 1e-34) and is not the problem: convergence is
6.14 sweeps mean / 8 max on random matrices and 5.55 / 7 on real hinge Hessians, i.e. as good as
Jacobi gets. **Hypothesis refuted.** The accepted s19 keeps Eigen's *algorithm* and replaces its
*implementation*.

### R2 — "skip the projection when the Hessian is already PSD" is dead on the real data

An already-PSD Hessian makes `make_spd` an expensive identity, and an LDL^T test costs ~2 % of the
eigen-solve (N^3/6 ~ 120 flops at N = 9). An in-kernel probe (the eigen path's own eigenvalues as
ground truth, an LDL^T test with a relative pivot tolerance, and a `__ballot_sync` per warp):

| call site | scene | calls | min eig >= 0 | LDL test says PSD | warps *uniformly* PSD |
|---|---|---:|---:|---:|---:|
| discrete-shell hinge | cube-wall-cloth | 200 000 | **0** | 18 315 (9.2 %) | 380 / 6 250, and not growing |
| ABD ortho potential | rigid-wrecking-balls | 100 000 | 9 291 (9.3 %) | 10 335 (10.3 %) | 149 / 3 136 = **4.8 %** |

Not one hinge Hessian in 200 000 is PSD, and a lane that skips the solve saves nothing unless its
whole warp skips it — so the ceiling is ~5 % of the projection on the ABD side and zero on the
hinge. (The LDL^T test also showed a 33 % false-positive rate at the 1e-13 level on the hinge: its
null-pivot column tolerance is `sqrt(eps_pivot) * scale`, which is loose by construction.)
**Rejected, not implemented.**

Sampled spectra of the real 9x9 Hessians behind the two call sites (251 hinge / 107 ortho matrices,
sampled across a whole run) explain both results and bound the other cheap-looking shortcut: the
hinge has **4 negative eigenvalues out of 9** in 228 of 251 samples (numerical rank 8, min eig /
scale median -1.2e-4), the ortho potential 2-3 negative (rank 9, or 6 at rest, where it is exactly
PSD). So "compute only the negative eigenpairs by inverse iteration and do a rank-k update" would
have to find 4 of 9 eigenpairs and cannot save more than about half — with all of inverse
iteration's clustering hazards.

## 7. Cross-architecture expectation (for the shared record)

Two components, with different transfer expectations:

- *Algorithmic / code-shape* — the fixed-size QL executes the same operation count as Eigen's, but
  without the out-of-line `selfadjoint_matrix_vector_product<double, long>` call per
  tridiagonalisation step and without dynamic-size index arithmetic: fewer instructions for the same
  FLOPs. **This part should transfer to every architecture.**
- *Spill / register-pressure driven* — the hinge instantiation's frame drops 3 472 -> 3 104 B and its
  spill stores 2 456 -> 1 576 B, and the ortho kernel falls from 254 to 192 registers. A 2070S, with
  64 KB of registers per SM and 1/32-rate FP64, over-rewards exactly this, so **the 5090 should see
  less than the -15 % / -8 % per launch measured here.** The ortho kernel's drop to 192 registers may
  additionally buy occupancy on some architectures and nothing on others; that part is explicitly not
  portable.

No launch configuration, no algorithm and no scene changed, so nothing should regress anywhere.
Expect the cube-wall end-to-end number (-5.4 % here) to land smaller on a 5090.

## 8. Next steps this step measured but did not take

1. **Extend the solver to the contact branches.** `make_spd_contact` ends in `make_spd<4>` and
   `make_spd<5>` (PT/EE dim 4 and PE dim 3 reduced projections; `make_spd<3>` keeps `computeDirect`),
   the tail of every contact branch, and the microbenchmark gives 1.15-1.22x there. It needs a
   `Solver` template parameter threaded through `do_assemble_kernel` — the same file whose PE+PP
   launch config w3 owns this round, which is why this step stayed out of it. Contact G+H is
   1.86 / 1.10 / 1.25 ms per Newton iteration on wrecking balls / cube-wall / case2 after s17, so the
   prize is roughly 0.5-1 % of two scenes.
2. **The 12x12 path.** `make_spd<12>` is still the fallback for the joints and for
   `UIPC_DSB_REDUCED_SPD=0`; QL is 0.96-1.17x there standalone, so it is worth switching only
   together with (1).
3. **The hinge kernel is latency-bound, not throughput-bound.** At 318 launches over 60 frames the
   grid is a few thousand threads for a serial per-thread eigen-solve; the 2070S is ~1.5 warps per SM
   at that point. A *lane-parallel* projection (one warp or sub-warp group per Hessian, parallel
   Jacobi with round-robin pairing — 5 independent rotations per round at N = 9) trades Jacobi's 4x
   flops for a 4-5x shorter critical path and would finally use the idle SMs. It is a restructuring of
   the call sites, not of `make_spd`, and it is the largest remaining lever on this cost centre.
4. **`abd_diag_preconditioner_do_assemble`** is now the 4th kernel of cube-wall-cloth (0.77-0.78 ms
   per launch, 5.7-6.0 % of its kernel time) and is one 12x12 `cuda_tool::eigen::inverse` per thread —
   the same "serial dense linear algebra per thread in a small grid" shape as this step's target, and
   untouched since round 3.

# Round 5 / w0-turing — step s20: the warp-cooperative PSD projection (rejected),
# and the launch geometry the profile pointed at instead (accepted)

Worker record for perf round 5, step **s20**, on top of s19 (`7abdf0ad`), branch
`perf/round5-w0-turing`. Box: local RTX 2070 SUPER, cc 7.5, **40 SMs**, CUDA 12.8, sm_75,
`build-perf` (Release, tests ON). Raw artefacts — microbenchmark sources and output, nsys reports
and traces, kernel summaries, benchmark JSONs, test log — in `/workspace/output/round5/w0-turing/`.

**The coordinator should paste §6 (ledger row), §7 (rejection R3) and §8 (cross-architecture) into
`agent_docs/performance/2026-09-12-perf-round5.md` on `perf/round5`.**

## 1. The assignment, and how it differs from R1

The step assigned was candidate 2 of s19's next-step list: a **warp-cooperative / lane-parallel PSD
projection**. Its premise, in my own words from s19: the hinge kernel is latency-bound, a few
thousand threads each running a *serial* eigen-solve, leaving SMs idle; a warp-cooperative Jacobi
does ~5 independent rotations per round at N = 9, trading ~4x the flops for a 4-5x shorter critical
path.

**Why this is not a re-run of R1.** R1 rejected *serial* cyclic Jacobi at N = 9 because `A` + `V`
are 162 doubles and spill whatever the layout — the working set, not the flop count, decided
(0.49-0.64x of Eigen at N = 9, 0.29-0.40x at N = 12). A warp-cooperative formulation spreads that
working set across the lanes' registers, which is precisely the premise R1 falsified for the serial
case. That premise is **confirmed**: ptxas, sm_75, the warp kernel written for this step uses
**168 registers, 72 B stack, 0 B spills**, against serial tql2's 255 registers / 1 440 B frame.
The spilling does not move — it disappears.

It loses anyway, for a reason R1 never tested: **lane efficiency**. See R3 (§7).

## 2. Profile first (nsys `cuda_gpu_trace`, 60-frame runs at s19 head)

The grid geometry of the two projection call sites, which no round had looked at:

| kernel | scene | grid x block | registers | SMs reached (of 40) |
|---|---|---|---:|---|
| `ortho_potential_compute_gradient_hessian` | rigid-wrecking-balls | **`<<<3, 256>>>`** | 192 | **3** |
| `ortho_potential_compute_gradient_hessian` | cube-wall-cloth | **`<<<8, 256>>>`** | 192 | **8** |
| `DiscreteShellBending_do_compute_gradient_hessian` | cube-wall-cloth / case2 | **`<<<48, 256>>>`** | 255 | 40, **but two waves** |
| `abd_diag_preconditioner_do_assemble` (not mine) | cube-wall / wrecking balls | `<<<8, 256>>>` / `<<<3, 256>>>` | 255 | 8 / 3 |

The block size comes from `cuda_tool::best_block_dim` =
`cudaOccupancyMaxPotentialBlockSize`, which maximises *resident warps per SM* and is blind to how
many SMs the resulting grid reaches. At 255 registers a 256-thread block is 65 280 of the SM's
65 536 registers, so exactly **one block is resident per SM**: the hinge's 48 blocks are 40 in a
first wave and 8 in a second, i.e. two waves are run for 1.2 waves of work; and the ABD ortho
potential simply never leaves 3 or 8 SMs of 40.

So the assignment's premise — "the SMs are idle" — is right, and understated. But the idle SMs are
not idle for want of intra-matrix parallelism. They are idle because the *blocks* are too big.

## 3. What was measured (standalone, sm_75, `spd_warp.cu`)

Both questions in one harness, benchmarked **by matrix count** so that a serial path (1 matrix per
thread) and a cooperative path (1 matrix per 9 lanes) are directly comparable. Inputs: randomised
9x9 spectra and the **251 real hinge Hessians** dumped from cube-wall-cloth by s19's in-kernel probe.
Compiled with the backend's own flags. Best of 5, 4 projections per thread.

Warp path: 9 lanes per matrix, lane `li` owning column `li` of `A` and of `V`; circle-method
round-robin pairing over 10 slots (slot 9 = dummy), so a round is 4 independent rotations and a
sweep is 9 rounds = 36 = C(9,2). The column update is one shuffle exchange with the partner lane;
the row update is four local 2-element updates, which is why every lane needs all four `(c, s)` of
the round. `GPM` = matrices per warp: 1 (9 of 32 lanes busy) or 3 (27 of 32). The pairing formula is
checked against the generated circle-method table at startup.

Real hinge Hessians, ms per projection pass (`>1x` = faster than serial tql2 at block 256):

| path | block | grid | M = 2 048 (ortho grid) | M = 12 288 (hinge grid) | M = 65 536 |
|---|---:|---:|---:|---:|---:|
| **serial tql2 (s19), block 256** | 256 | 8 / 48 / 256 | 0.8002 (1.00x) | 1.7041 (1.00x) | 7.3513 (1.00x) |
| serial tql2, block 128 | 128 | 16 / 96 / 512 | 0.4286 (1.87x) | 1.3409 (1.27x) | 7.2807 (1.01x) |
| serial tql2, block 64 | 64 | 32 / 192 / 1 024 | 0.2939 (2.72x) | 1.3431 (1.27x) | 7.2082 (1.02x) |
| **serial tql2, block 32** | 32 | 64 / 384 / 2 048 | **0.2897 (2.76x)** | **1.3355 (1.28x)** | 6.9496 (1.06x) |
| serial eigen (pre-s19), block 256 | 256 | | 0.9177 (0.87x) | 2.1339 (0.80x) | 9.5462 (0.77x) |
| **warp jacobi GPM=3, 6 sweeps** | 64 | 342 / 2 048 / 10 923 | 1.3119 (**0.61x**) | 6.8767 (**0.25x**) | 36.58 (0.20x) |
| **warp jacobi GPM=3, 8 sweeps** | 64 | | 1.7426 (**0.46x**) | 9.1377 (**0.19x**) | 48.60 (0.15x) |
| warp jacobi GPM=1, 6 sweeps | 64 | | 3.4488 (0.23x) | 20.33 (0.08x) | 109.47 (0.07x) |

(Randomised inputs give the same picture: 0.77x / 0.30x / 0.20x for GPM=3 at 6 sweeps.)

Two results, and they point in opposite directions.

## 4. The change (s20) — grid-fitted block size for the two projection call sites

`uipc::backend::cuda::fitted_block_dim(kernel, n, blocks_per_sm = 8)` in
`src/backends/cuda/utils/proj_launch.h`: take `best_block_dim`'s answer and halve it (down to one
warp) while the resulting grid is smaller than `blocks_per_sm` blocks per SM. When `n` is large
enough that the grid already covers the device the occupancy-optimal answer is returned unchanged,
so the helper only ever acts in the small-grid regime. `blocks_per_sm = 8` is the measured optimum,
not a guess: block 32 is the fastest column of the table above at every matrix count.

Wired into the two call sites s19 owns — the discrete-shell hinge
(`discrete_shell_bending.cu`) and the ABD ortho potential (`ortho_potential.cu`) — behind
**`UIPC_PROJ_BLOCK_FIT`** (`=0` restores `best_block_dim`, i.e. the pre-s20 geometry; the rollback
and the A/B reference). It is a *new, separate* header so that the two `.cu` files are the only
translation units that recompile, and so that nothing in `cuda_tool/launch.h` changes for the other
three workers.

Measured geometry change, cube-wall-cloth: hinge `<<<48, 256>>>` -> `<<<380, 32>>>`,
ortho `<<<8, 256>>>` -> `<<<60, 32>>>`; wrecking balls ortho `<<<3, 256>>>` -> `<<<18, 32>>>`.

`UIPC_MAKE_SPD_JACOBI` is untouched and orthogonal — audited at the end of this step (§5).

## 5. Correctness and numerics

**Gate.** `run_tests.sh`, identical counts to the round-4 baseline table and to s19's pass:
common 11/3, core 1112/36, geometry 2730/46, sanity_check 100/3, regression 4/1,
backend_cuda 352/22, sim_case 14213/95, pytest 48 passed / 1 skipped. No new failures, no flakes.
(`tests_s20.log`.)

**Numerics: bit-identical, and not by assertion.** The change alters no arithmetic — the launched
`__global__` function is the same compiled instantiation either way (confirmed by the mangled names
in the nsys trace, §5 audit below); only the grid/block pair differs. Both kernels derive everything
from the global thread index and write to index-addressed destinations
(`DoubletVectorAssembler::write`, `TripletMatrixAssembler::write`, `body_hessian(i)`), with **no
atomics, no shared memory, no `__syncthreads` and no warp cooperation** — checked by reading both
kernels and `utils/matrix_assembler.h` (the only `atomic_add` there is `DenseVectorAssembler`'s,
which neither kernel uses). So each output location and its value depend on the thread index alone,
and the result cannot depend on how indices are grouped into blocks.

Demonstrated rather than argued (`block_invariance.cu`, the real `make_spd<9>` tridiagonal-QL
projection, backend flags, sm_75):

| n matrices | block 256 vs 128 | vs 64 | vs 32 |
|---|---|---|---|
| 12 160 (hinge grid) | 0 / 984 960 doubles differ in any bit | 0 | 0 |
| 1 920 (ortho grid) | 0 / 155 520 doubles differ in any bit | 0 | 0 |

This is a *stronger* numerics result than s19's (which was rounding-level at 3.8e-15): s20 changes
nothing an FPU sees. End-to-end the scenes still move within their documented run-to-run scatter
because the rest of the solver has atomics — mas-bunny, the control scene that has neither
constitution, gives Newton 465 in all four runs and a centroid agreeing to the 7th digit.

**Rollback audit** (nsys trace, 12-frame cube-wall-cloth runs, kernel instantiation from the
mangled name):

| env | hinge | ortho |
|---|---|---|
| default | `<<<380, 32>>>` `kernel<1, 1>` (tql2) | `<<<60, 32>>>` `kernel<1>` (tql2) |
| `UIPC_MAKE_SPD_JACOBI=0` | `<<<380, 32>>>` `kernel<1, 0>` (**Eigen**) | `<<<60, 32>>>` `kernel<0>` (**Eigen**) |
| both `=0` | `<<<48, 256>>>` `kernel<1, 0>` | `<<<8, 256>>>` `kernel<0>` |

s19's switch still selects the old Eigen path, and the two switches are independent.

## 6. Performance

### Targeted scope — nsys, 60 frames, one build, A/B on `UIPC_PROJ_BLOCK_FIT`

ms per launch (launch counts differ by one because the run lengths differ by one Newton iteration;
per-launch is the comparable quantity):

| kernel | scene | off | on | delta |
|---|---|---:|---:|---:|
| `DiscreteShellBending_do_compute_gradient_hessian` | cube-wall-cloth | 2.2393 | **1.7182** | **−23.3 %** |
| `DiscreteShellBending_do_compute_gradient_hessian` | stiff-gipc-case2 | 2.2965 | **1.7679** | **−23.0 %** |
| `ortho_potential_compute_gradient_hessian` | cube-wall-cloth | 0.8419 | **0.2958** | **−64.9 %** |
| `ortho_potential_compute_gradient_hessian` | rigid-wrecking-balls | 1.0567 | **0.3258** | **−69.2 %** |
| `abd_diag_preconditioner_do_assemble` (untouched control) | cube-wall / wb | 0.7879 / 0.6481 | 0.7836 / 0.6483 | −0.5 % / +0.0 % |
| scene kernel total | cube-wall-cloth | 4 164.7 | 3 838.2 | −7.8 % |
| scene kernel total | rigid-wrecking-balls | 2 099.9 | 1 988.7 | −5.3 % |
| scene kernel total | stiff-gipc-case2 | 10 656.6 | 10 511.5 | −1.4 % |

The standalone harness predicted −21.6 % (hinge grid) and −63.8 % (ortho grid) on the same real
matrices; the kernels moved −23.3 % and −64.9 %. The model and the machine agree.

### End-to-end — official benchmarks, default frames, one build, env A/B, 2 runs each way

mean / median ms per frame (Newton; PCG):

| scene | `FIT=0` | `FIT=1` | mean | per Newton it. |
|---|---|---|---:|---:|
| **cube-wall-cloth** | 88.54 / 80.53 (508; 20 285), 84.71 / 81.04 (497; 19 030) | **80.34 / 76.09** (505; 19 635), **82.39 / 74.27** (509; 20 190) | **−6.1 %** | 17.24 -> 16.05 = **−6.9 %** |
| **rigid-wrecking-balls** | 39.29 / 33.62 (467; 12 905), 38.66 / 33.57 (461; 12 680) | **37.53 / 31.22** (486; 13 090), **35.71 / 31.03** (467; 12 975) | **−6.1 %** | 10.08 -> 9.22 = **−8.5 %** |
| **stiff-gipc-case2** | 221.60 / 224.08 (1 664; 64 390), 219.85 / 225.28 (1 655; 63 985) | **219.22 / 226.35** (1 661; 65 035), **215.67 / 222.43** (1 654; 63 890) | **−1.5 %** | 33.25 -> 32.80 = −1.4 % |
| **mas-bunny** (control) | 74.08 (465; 35 210), 73.99 (465; 35 205) | 74.20 (465; 35 235), 73.96 (465; 35 210) | +0.1 % | unchanged |

The three targeted distributions are disjoint run-to-run (cube-wall: {84.71, 88.54} vs
{80.34, 82.39}; wrecking balls: {38.66, 39.29} vs {35.71, 37.53}; case2: {219.85, 221.60} vs
{215.67, 219.22}, the last only just). mas-bunny drives neither constitution and does not move —
Newton 465 in all four runs, as in s19. Every run converged, no Newton or line-search limit hit.

case2 is the smallest win because the ABD ortho potential is absent there (no ABD) and the hinge is
a smaller share of a much heavier FEM frame; its scope number (−23.0 % on the hinge) is the same as
cube-wall's.

### Ledger row (for the shared record)

| Step | Worker | Commit | Change | Scope measurement | End-to-end effect | Tests | Verdict |
|---|---|---|---|---|---|---|---|
| s20 | w0-turing | (this commit) | **Grid-fitted block size for the two PSD-projection call sites.** `fitted_block_dim` (`utils/proj_launch.h`) halves `cudaOccupancyMaxPotentialBlockSize`'s answer, down to one warp, while the grid is below 8 blocks per SM; a large `n` returns the occupancy-optimal answer unchanged. `best_block_dim` maximises warps per SM and is blind to how many SMs the grid reaches: at 255 registers a 256-thread block is one resident block per SM, so the discrete-shell hinge ran `<<<48, 256>>>` = two waves for 1.2 waves of work, and the ABD ortho potential ran `<<<3, 256>>>` on wrecking balls = **3 of 40 SMs**. Both kernels are register-bound with no shared memory, no `__syncthreads` and no atomics, so a smaller block costs nothing. `UIPC_PROJ_BLOCK_FIT=0` = old geometry. Measured: hinge `<<<48,256>>>` -> `<<<380,32>>>`, ortho `<<<8,256>>>` -> `<<<60,32>>>` / `<<<3,256>>>` -> `<<<18,32>>>` | nsys, 60 frames, one build: hinge G/H **2.2393 -> 1.7182 ms/launch on cube-wall (−23.3 %)** and 2.2965 -> 1.7679 on case2 (−23.0 %); ortho G/H **0.8419 -> 0.2958 on cube-wall (−64.9 %)** and **1.0567 -> 0.3258 on wrecking balls (−69.2 %)**; untouched `abd_diag_preconditioner` −0.5 % / +0.0 % as a control. Scene kernel total: cube-wall 4 164.7 -> 3 838.2 (−7.8 %), wrecking balls 2 099.9 -> 1 988.7 (−5.3 %), case2 10 656.6 -> 10 511.5 (−1.4 %) | env A/B, 2 runs each way, mean ms/frame (Newton; PCG): **cube-wall-cloth** 88.54 (508; 20 285) / 84.71 (497; 19 030) -> **80.34** (505; 19 635) / **82.39** (509; 20 190) = **−6.1 %**, disjoint, **−6.9 % per Newton iteration**; **rigid-wrecking-balls** 39.29 (467) / 38.66 (461) -> **37.53** (486) / **35.71** (467) = **−6.1 %**, disjoint, **−8.5 %/it**; **stiff-gipc-case2** 221.60 (1 664) / 219.85 (1 655) -> 219.22 (1 661) / 215.67 (1 654) = **−1.5 %**, disjoint, −1.4 %/it; mas-bunny control unchanged (Newton 465 in all four runs) | identical counts to the round-4 baseline: common 11/3, core 1112/36, geometry 2730/46, sanity 100/3, regression 4/1, backend_cuda 352/22, sim_case 14213/95, pytest 48 passed 1 skipped. No flakes | **accepted** |

## 7. Rejection R3 — the warp-cooperative Jacobi, for a reason R1 did not test

Numbers in §3. At the hinge's own matrix count the warp-cooperative path is **0.25x** of serial
tql2 at 6 sweeps and **0.19x** at 8; at the ortho potential's, where 32 of 40 SMs really are idle,
it is **0.61x / 0.46x** — and the fair comparison there is not tql2 at block 256 but tql2 at
block 32, which is 2.76x faster for free, making the cooperative path **0.22x / 0.17x** of the
available baseline. There is no matrix count at which it wins.

**Why, precisely.** R1's objection is answered: ptxas gives the warp kernel 168 registers, 72 B
stack and **zero spills** against serial tql2's 255 / 1 440 B, so the working set genuinely moves
into the lanes' registers. What kills it is a cost R1 never had to price — **lane efficiency and
shuffle traffic**:

- Parallel Jacobi has 4 independent rotations per round at N = 9, so the *critical path* is indeed
  9 rounds per sweep instead of 36 rotations, a 4x shortening, exactly as predicted.
- But the *work* is unchanged from serial Jacobi, and R1 already measured serial Jacobi at 0.49-0.64x
  of Eigen where tql2 is 1.22x — a ~2.2x arithmetic handicap before any parallelism.
- Spreading one matrix over 9 lanes leaves 23 of 32 lanes idle (GPM = 1) or 5 of 32 (GPM = 3). The
  GPM = 1 -> GPM = 3 measurement isolates this exactly: **0.08x -> 0.25x**, a 3.1x recovery, i.e.
  lane waste alone was the dominant term and packing three matrices per warp recovers nearly all of
  what is recoverable. It is still 4x short.
- On top of that the distributed form pays ~60 shuffle instructions per round (partner column
  exchange for `A` and `V`, the four `(c, s)` broadcasts) that the serial form does not pay at all.

The net is ~4x the warp-instructions per matrix of s19's serial QL, against an idle-SM headroom
that the profile bounds at 1.7x on the hinge (two waves for 1.2 waves of work) and ~5x on the ABD
ortho potential (8 of 40 SMs) — and on the ortho side that same headroom is available for free from
the block size. **The idle SMs were a launch-geometry problem, not a parallelism-granularity
problem**, and buying them with 4x the flops is the expensive way to fix it.

Sweep count, for the record (the same fixed-count question w1's p02 asked of `qr_svd`): the warp
path needs **8 sweeps**, not 6. Verified against `make_spd_eigen` on 2e4 matrices per family, max
relative Frobenius difference: 6 sweeps 3.1e-6 (unusable), **8 sweeps 2.9e-14**, 10 sweeps 8.2e-15;
near-degenerate clusters 2.0e-9 and rank-deficient 6.8e-12 at 8 sweeps; real hinge Hessians 8.8e-15
at 8. The 6-sweep timings above are therefore **optimistic by 33 %** and are quoted only to show
that even the unusable setting loses.

## 8. Cross-architecture expectation

This step is the one the brief warns about — *"occupancy/launch-bounds tuning: often does not
transfer"* — and it is worth being precise about why the usual warning is **inverted** here.

- The gain is **not** per-SM occupancy tuning. Resident warps per SM are unchanged: at 255 registers
  an SM holds 8 warps whether they arrive as one 256-thread block or eight 32-thread blocks. What
  changes is how many **SMs** the grid reaches, and the tail-wave quantisation.
- That is a **grid-size-versus-machine-size** property, and it gets **worse on a bigger GPU**. The
  acceptance box is an RTX 5090 with ~170 SMs: `<<<3, 256>>>` reaches 3 of 170 and `<<<48, 256>>>`
  reaches 48 of 170, so on the 5090 the *unfixed* kernels waste 98 % and 72 % of the machine rather
  than 92 % and (two waves of) 40 %. `fitted_block_dim` returns 32 on both boxes for these `n`
  (`want = 8 x 170 = 1 360` blocks, never reached), so the same 380- and 60-block grids are
  launched, now over 170 SMs. **Expect the 5090 to show more than −23 % / −65 % per launch, not
  less.**
- The one component that does *not* transfer cleanly is the exact crossover: `blocks_per_sm = 8`
  and the resident-warp count behind it depend on the register file and the per-SM block limit,
  which differ across cc 7.5 / 8.6 / 8.9 / 12.0. The helper is written so that a wrong guess is
  bounded — it can only ever return something between one warp and `best_block_dim`'s answer, and
  above a threshold `n` it returns `best_block_dim` unchanged.
- This is **the opposite of s19's transfer profile**, which was half spill/register relief that a
  2070S over-rewards. s20 has **no** arithmetic, algorithm or register component at all; it is pure
  machine coverage.

Nothing should regress anywhere: no arithmetic changed, the kernel binaries are identical, and the
untargeted control scene (mas-bunny) and the untargeted control kernel
(`abd_diag_preconditioner_do_assemble`, −0.5 % / +0.0 %) both sit still.

## 9. Found in another worker's area

- **`abd_diag_preconditioner_do_assemble` has exactly the same bug and nobody owns it.** It launches
  `<<<8, 256>>>` on cube-wall-cloth and `<<<3, 256>>>` on rigid-wrecking-balls at **255 registers**,
  i.e. 8 and 3 of 40 SMs, and it is one 12x12 `cuda_tool::eigen::inverse` per thread — the same
  "serial dense linear algebra per thread in a tiny grid" shape as the ABD ortho potential, which
  responded to the identical treatment with **−69 %**. It is now cube-wall-cloth's **third** kernel
  (0.784 ms per launch, 250 ms of 3 838 ms = 6.5 %) and wrecking balls' third (0.648 ms, 123 ms of
  1 989 = 6.2 %). `fitted_block_dim` is a one-line call at its launch site. It sits in w3's
  `ortho_potential` + `abd_diag_preconditioner` stream-overlap area, so this step did not touch it.
  **This is the single cheapest item left in the round that I can see.**
- The same question should be asked of **every** register-heavy kernel with a small `n`. The two
  `do_assemble_kernel<false, ...>` contact kernels are cube-wall's 2nd and 5th (1.72 / 1.10 ms per
  launch) and are w3's; I did not profile their grids, but `best_block_dim` is used at every named
  launch site in the backend, so the pathology is structural, not local. A sweep of
  `cuda_gpu_trace` for `grid < SM_count` is a 10-minute coordinator-level probe.
- s19's item 1 stands unchanged: extending the s19 solver to `make_spd<4>` / `make_spd<5>` in the
  contact branches is still w3's file, and the microbenchmark still gives 1.15-1.22x there.

## 10. Candidates after s20, with the measured evidence

1. **`fitted_block_dim` for `abd_diag_preconditioner_do_assemble`** — see §9. Evidence: the same
   kernel shape at `<<<3, 256>>>` / `<<<8, 256>>>`, and a −69 % measurement on its twin. ~6 % of two
   scenes' kernel time is in play. Needs a w3 hand-off or a reassignment.
2. **A grid-coverage sweep of the whole backend.** `best_block_dim` is
   `cudaOccupancyMaxPotentialBlockSize` everywhere, and this step shows it is the wrong objective
   whenever `n` is smaller than `SMs x blocks_per_SM x block`. One `nsys stats --report
   cuda_gpu_trace` pass per scene, filtered to `GrdX < 40` (or `< 170` on the 5090), ranks every
   remaining instance of the pathology. This is the highest-yield-per-minute item I found all round.
3. **The hinge kernel is now 1.72 ms per launch and still the largest single kernel on
   cube-wall-cloth** (548 ms of 3 838 = 14.3 %). After s19 and s20 the remaining lever inside it is
   the register count itself: 255 registers with 1 576 B of spill stores caps it at 8 resident warps
   per SM. Anything that gets it under ~212 registers doubles the resident warps. s19's ptxas table
   says the projection is no longer the dominant contributor to that frame — the stencil derivative
   `DSB::ddEddx` is — which is inside my subsystem's call site but outside `make_spd` proper.
4. **`make_spd<12>` and the cold call sites** (joints, stitches, ARAP) still use Solver = 0 and the
   unfitted block size. Cheap to fold in, but they are cold: no measurement yet says they matter.

# Round 5 s21 (w3-blackwell) — sweeping the whole suite for launches that never reach the SMs

RTX 5060 Ti (cc 12.0, **36 SMs**), instance 50737979, branch `perf/round5-w3-blackwell` on top of
`perf/round5` = `c5280c09` (s19 + s20 merged). Follows s20, which fixed three ABD per-body launches;
this step sweeps every launch in the four benchmark scenes, fixes 24 more call sites, and
**consolidates w3's `cuda_tool/spread_launch.h` with w0's `utils/proj_launch.h` into one helper with
one heuristic and one env switch**.

## 1. The audit — how the sweep was done

`audit.py` profiles a scene with `nsys --trace=cuda`, groups the `cuda_gpu_trace` rows by
`(kernel, GrdX, BlkX, Reg/Trd)` and ranks by total GPU time per Newton iteration. That is the
instrument the round was missing: `cuda_gpu_kern_sum` gives times but not geometry, and the geometry
is the whole story here.

**rigid-wrecking-balls, post-s20, 60 frames / 188 Newton iterations, 7.1694 ms/it of kernel time:**
every launch whose grid was below the 36-SM count accounted for 3.85 ms/it, and the launches at
grid 1–5 — the ones that cannot use more than 14 % of the machine no matter how well they are
scheduled — accounted for **~0.76 ms/it, 10.6 % of the scene**:

| ms/it | grid | block | regs | kernel | fixed? |
|---|---|---|---|---|---|
| 0.0900 | 3 | 256 | 254 | `abd_linear_subsystem_assemble_kinetic_shape_k2` | yes |
| 0.0830 | 1 | 640 | 84 | `do_assemble_kernel` (**vertex–half-plane friction**) | yes |
| 0.0726 | 1 | 768 | 80 | `do_assemble_kernel` (**vertex–half-plane normal**) | yes |
| 0.0713 | 2 | 512 | 102 | `do_compute_energy_k1` (contact line search) | yes |
| 0.0525 | 3 | 768 | 80 | `do_compute_energy_k4` | yes |
| 0.0516 | 5 | 256 | 255 | `assemble_dytopo_effect_pair_k3` | yes |
| 0.0496 | 1 | 896 | 68 | `affine_body_bdf1_kinetic_compute_gradient_hessian` | yes |
| 0.0487 | 1 | 896 | 72 | `do_compute_energy_kernel` (vertex–half-plane) | yes |
| 0.0462 | 2 | 512 | 126 | `compute_feasible_step_PT` (CCD) | yes |
| 0.0207 | 5 | 256 | 38 | `buffer_fill_kernel` (`cuda_tool` infrastructure) | no — shared utility |
| 0.0199 | 1 | **1** | 32 | `fused_pcg_scalar_kernel` | no — one thread of work |

**Correction to s20's own candidate list**: s20 reported the two grid-1 `do_assemble_kernel`
launches as "the friction `do_assemble`". They are not — the templated simplex friction assemble
runs at grid 26–31. They are the **vertex–half-plane** normal and frictional contact assemblies
(`ipc_vertex_half_plane_{normal,frictional}_contact.cu`), i.e. the ground plane. Fixed here.

## 2. Consolidating the two helpers

w0 found the same pathology independently on the kernels it owned and wrote
`utils/proj_launch.h::fitted_block_dim`. Two helpers with two heuristics and two env switches is
worse than either, so s21 keeps one:

| | w3 s20 `spread_block_dim` | w0 s20 `fitted_block_dim` | s21 (shipped) |
|---|---|---|---|
| criterion | binary: is the natural grid smaller than the SM count? | ramp: halve while the grid is below 8 blocks/SM | **w0's ramp** |
| catches grid 1–3 | yes | yes | yes |
| catches **wave quantisation** (grid 48 on 40 SMs at 255 regs = two waves for 1.2 waves of work) | **no** | yes | yes |
| step | one warp per block | halve the block | **halve the warp count** |
| env switch | `UIPC_GRID_SPREAD` | `UIPC_PROJ_BLOCK_FIT` | `UIPC_GRID_SPREAD` |

**The ramp is the better-justified criterion and it subsumes the binary one**, so the ramp ships.
Both heuristics were already known to agree on the *answer* at the small-grid sites (w3 measured a
block-size sweep: 32 → 0.8500, 64 → 0.9205, 96 → 0.9882 ms/it; w0's fit lands on 32 at every site
it touched), so consolidating costs nothing measured and removes a silent disagreement.

**One correction to w0's version, and it matters for the bit-identity argument.** Repeated halving
of a block size that is a multiple of 32 but not a power of two produces block sizes that are *not*
multiples of 32, and `best_block_dim` really does return 640, 768 and 896 at sites in this backend
(the table above). `640 -> 320 -> 160 -> 80 -> 40`: a 40-thread block straddles warp boundaries, so
warp composition — and therefore `__activemask()` — would change with the geometry. s21 halves the
**warp count** instead (`20 -> 10 -> 5 -> 2 -> 1`), so every block is a whole number of warps and
warp *k* always covers items `[32k, 32k+32)` whatever the block size. That is what makes the
bit-identity argument survive contact with kernels that use warp-level primitives.

Shipped helper (`cuda_tool/spread_launch.h`):

```cpp
int bd = best_block_dim(kernel);              // cudaOccupancyMaxPotentialBlockSize
int warps = bd / 32;
const int want = blocks_per_sm * sm_count;    // blocks_per_sm = 8
while(warps > 1 && grid(n, warps * 32) < want) warps >>= 1;
return warps * 32;
```

Env: `UIPC_GRID_SPREAD=0` (pre-round-5 geometry everywhere, the rollback),
`UIPC_GRID_SPREAD_BPSM=<n>`, `UIPC_GRID_SPREAD_BLOCK=<n>`, `UIPC_GRID_SPREAD_ONLY=<substring>`
(spread only at call sites whose tag matches — the per-site A/B instrument),
`UIPC_GRID_SPREAD_VERIFY=1`.

`utils/proj_launch.h` should be **dropped** when w0's `ec79c7c2` is merged: its `ortho_potential`
hunk is already done on `perf/round5` by s20 and its `discrete_shell_bending` hunk is reimplemented
here on the consolidated helper.

## 3. Sites changed (24 launch sites, 8 files)

Geometry only — one call per site, no kernel body, no algorithm, no host-side sync.

| file | sites |
|---|---|
| `affine_body/abd_linear_subsystem.cu` | `assemble_kinetic_shape_k1`, `k2`, `assemble_dytopo_effect_pair_k3` |
| `affine_body/bdf/affine_body_bdf1_kinetic.cu` | energy, gradient+Hessian |
| `contact_system/contact_models/ipc_simplex_normal_contact.cu` | line-search energies `k1..k4` |
| `contact_system/contact_models/ipc_simplex_frictional_contact.cu` | energies `k1..k4`, `do_assemble` |
| `contact_system/contact_models/ipc_vertex_half_plane_normal_contact.cu` | energy, `do_assemble` |
| `contact_system/contact_models/ipc_vertex_half_plane_frictional_contact.cu` | energy, `do_assemble` |
| `contact_system/global_contact_manager.cu` | CCD `compute_feasible_step_{PT,EE,PE,PP}` |
| `finite_element/constitutions/discrete_shell_bending.cu` | energy, gradient+Hessian (**w0's hinge**) |

### Sites deliberately **not** changed, and why

- **`ipc_simplex_normal_contact::do_assemble` (contact G+H parts 0/1/2).** Contains
  `__any_sync(__activemask(), mollified)` (s16's per-warp reduced projection). Warp composition is
  in fact preserved by a warp-quantised block size, so the geometry change would be safe — but it
  already runs at grid 26–31 of 36 and the ramp would take it to grid ~219 at block 32, which is a
  *throughput* change to the round's single largest kernel and needs its own measurement. Left for
  a follow-up, not smuggled into a geometry sweep.
- **`ee_mollifier_partition_kernel`.** `atomicAdd` allocates the output slot, so the permutation it
  writes depends on execution order. Not bit-identical under any geometry change. Left alone.
- **`abd_linear_subsystem` dytopo `pair_warp` kernel (s05).** `__shfl_sync` across a warp *and*
  `atomicAdd` accumulation. Left alone.
- **`cuda_tool::details::buffer_fill_kernel`** (0.0207 ms/it at grid 5). Shared infrastructure used
  by every subsystem; a helper-level change there is a different step.
- **`fused_pcg_scalar_kernel`**: `<<<1, 1>>>`, 4 015 launches, **0.0199 ms/it of pure launch
  overhead** for a scalar update. There is no parallelism to spread. Reported to w2 (below).

### The cooperation audit (what makes the bit-identity claim valid)

Every file above was grepped for `__shared__`, `__syncthreads`, `__shfl`, `*_sync`, `__activemask`,
`atomic*` and `+=` into an output. Results: `ipc_simplex_frictional_contact.cu`,
`ipc_vertex_half_plane_{normal,frictional}_contact.cu`, `affine_body_bdf1_kinetic.cu`,
`global_contact_manager.cu` and `discrete_shell_bending.cu` are **clean** — no cooperation of any
kind. `ipc_simplex_normal_contact.cu` has two hits, both in kernels that were **not** touched (see
above). `abd_linear_subsystem.cu` has hits only in the `pair_warp` kernel, which was not touched.

One non-idempotent kernel was found and handled rather than skipped:
`assemble_dytopo_effect_pair_k3` does `diag_hessian(L) += H12x12` (safe — `pair_key` is unique, so
at most one thread per body hits `L == R`, and it is a plain read-modify-write, not an atomic — but
**not** idempotent). Running it twice for the verification would double-add, so the verifier gained
a save/restore phase for declared in/out buffers (`launch_spread_io`).

## 4. Numerics — bit-identical, proved on device

`UIPC_GRID_SPREAD_VERIFY=1` makes every instrumented site run the kernel **twice per invocation**:
first with the geometry `best_*_dim` would have picked (the reference), then — after a device-to-device
snapshot of every output region — with the spread geometry that ships. All output words are then
compared on device. Comparison is in **32-bit** words, not 64: a `TripletMatrixView`'s row/column
index arrays are `int*` at an arbitrary triplet offset, so an 8-byte load off them can be
misaligned (this was found the hard way — a sticky `misaligned address` from the first version).

**rigid-wrecking-balls, 60 frames, every instrumented site, 0 mismatching words:**

| site | output words (32-bit) | mismatching |
|---|---|---|
| `IPCSimplexFrictionalContact::assemble` | 744 362 651 | **0** |
| `ABDLinearSubsystem::dytopo_pair_k3` | 94 464 064 | **0** |
| `ABDLinearSubsystem::kinetic_shape_k2` | 64 912 512 | **0** |
| `AffineBodyBDF1Kinetic::gradient_hessian` | 33 310 368 | **0** |
| `IPCSimplexNormalContact::energy` | 14 139 726 | **0** |
| `IPCSimplexFrictionalContact::energy` | 13 993 818 | **0** |
| `GlobalContactManager::ccd_PE` | 5 363 360 | **0** |
| `GlobalContactManager::ccd_PP` | 2 635 556 | **0** |
| `ABDLinearSubsystem::kinetic_shape_k1` | 2 562 336 | **0** |
| `GlobalContactManager::ccd_EE` | 2 112 752 | **0** |
| `IPCVertexHalfPlaneNormalContact::assemble` | 1 572 480 | **0** |
| `IPCVertexHalfPlaneFrictionalContact::assemble` | 1 563 840 | **0** |
| `AffineBodyBDF1Kinetic::energy` | 291 592 | **0** |
| `GlobalContactManager::ccd_PT` | 262 404 | **0** |
| `IPCVertexHalfPlaneNormalContact::energy` | 158 720 | **0** |
| `IPCVertexHalfPlaneFrictionalContact::energy` | 157 440 | **0** |
| **total** | **981 903 659** | **0** |

**cube-wall-cloth, 40 frames** (adds the hinge), 0 mismatching:
`DiscreteShellBending::gradient_hessian` **490 728 960** words, `DiscreteShellBending::energy`
5 642 240, `ABDLinearSubsystem::kinetic_shape_k2` 206 622 720,
`AffineBodyBDF1Kinetic::gradient_hessian` 106 030 080, `dytopo_pair_k3` 23 525 440,
`IPCSimplexFrictionalContact::assemble` 16 771 671, `kinetic_shape_k1` 8 156 160,
`AffineBodyBDF1Kinetic::energy` 890 880, contact energies and CCD 649 360. **Total 859 017 511
words, 0 mismatching.**

s20's own verifiers, re-run in the composed build, also stay at zero: `OrthoSpreadVerify`
16 655 184 / 53 015 040 words and `ABDDiagSpreadVerify` 15 374 016 / 48 936 960 words, 0 mismatching
on both scenes — which is the check the coordinator asked for on the s19/s20 merge resolution.

**1.84e9 output words compared across two scenes, zero mismatching bits.**

## 5. Performance — three arms in one build

- **A** `UIPC_GRID_SPREAD=0` — pre-round-5 geometry everywhere.
- **B** `UIPC_GRID_SPREAD_ONLY=__none__` — s20's three sites only.
- **C** default — s20 + s21.

### Per-kernel, nsys, 60-frame windows, ms per Newton iteration

**rigid-wrecking-balls** (kernel total 7.9444 → 7.1748 → **6.4001** ms/it):

| kernel | A | B | C | geometry A → C | B → C |
|---|---|---|---|---|---|
| `ortho_potential` G/H | 0.7740 | 0.4413 | 0.4450 | (3,256) → (18,32) | (s20) |
| `abd_diag_preconditioner` assemble | 0.5885 | 0.3359 | 0.3368 | (3,256) → (18,32) | (s20) |
| `ortho_potential` energy | 0.2121 | 0.0454 | 0.0459 | (1,576) → (18,32) | (s20) |
| contact energy `k2` | 0.1756 | 0.1731 | **0.0652** | (2,512) → (23,32) | **−62.3 %** |
| contact energy `k4` | 0.1730 | 0.1721 | **0.0654** | (3,640) → (46,32) | **−62.0 %** |
| contact energy `k3` | 0.1721 | 0.1709 | **0.1129** | (1,768) → (31,32) | **−33.9 %** |
| contact energy `k1` | 0.1517 | 0.1628 | **0.0413** | (1,512) → (2,32) | **−74.6 %** |
| vertex–half-plane `do_assemble` | 0.1557 | 0.1558 | **0.0667** | (1,640) → (10,32) | **−57.2 %** |
| `assemble_kinetic_shape_k2` | 0.0902 | 0.0902 | **0.0211** | (3,256) → (18,32) | **−76.6 %** |
| vertex–half-plane energy | 0.0841 | 0.0838 | **0.0355** | (1,640) → (10,32) | **−57.6 %** |
| `affine_body_bdf1_kinetic` G/H | 0.0498 | 0.0498 | **0.0104** | (1,896) → (18,32) | **−79.1 %** |
| CCD `feasible_step_PT` | 0.0473 | 0.0476 | **0.0132** | (1,512) → (2,32) | **−72.3 %** |
| CCD `feasible_step_PE` | 0.0294 | 0.0293 | **0.0216** | (2,640) → (31,32) | −26.3 % |
| CCD `feasible_step_EE` | 0.0297 | 0.0283 | **0.0170** | (2,384) → (23,32) | −39.9 % |
| dytopo pair group (k1/k2/k3) | 0.5735 | 0.5481 | 0.5049 | | −7.9 % |
| **the 11 named s21 sites together** | 1.1567 | **1.1637** | **0.4703** | | **−59.6 %** |
| *untargeted controls* | | | | | |
| contact G+H `do_assemble<...>` | 2.3168 | 2.3058 | 2.2828 | (3,256) unchanged | −1.0 % |
| BVH `stacklessSelf` | 0.3889 | 0.3878 | 0.3863 | unchanged | −0.4 % |
| `fused_update_xr` / `fused_dot` | 0.0396 / 0.0338 | 0.0392 / 0.0334 | 0.0394 / 0.0338 | unchanged | 0 |

**cube-wall-cloth** (kernel total 8.7176 → 8.0694 → **7.2348** ms/it):

| kernel | A | B | C | geometry A → C | B → C |
|---|---|---|---|---|---|
| **`DiscreteShellBending` G/H (the hinge)** | 1.5952 | 1.5900 | **1.3446** | **(48,256) → (380,32)** | **−15.4 %** |
| `ortho_potential` G/H | 0.6121 | 0.3671 | 0.3677 | (8,256) → (60,32) | (s20) |
| `abd_diag_preconditioner` assemble | 0.5896 | 0.3735 | 0.3853 | (8,256) → (60,32) | (s20) |
| `ortho_potential` energy | 0.1957 | 0.0411 | 0.0413 | (4,576) → (60,32) | (s20) |
| contact energy `k3` | 0.1490 | 0.1465 | **0.0678** | (1,640) → (6,32) | **−53.7 %** |
| contact energy `k2` | 0.1339 | 0.1312 | **0.0474** | (2,512) → (26,32) | **−63.9 %** |
| contact energy `k1` | 0.1306 | 0.1279 | **0.0316** | (1,512) → (9,32) | **−75.3 %** |
| `affine_body_bdf1_kinetic` G/H | 0.1059 | 0.1056 | **0.0148** | (3,896) → (60,32) | **−86.0 %** |
| `assemble_kinetic_shape_k2` | 0.0916 | 0.0914 | **0.0326** | (8,256) → (60,32) | **−64.3 %** |
| *control*: `Spmv_rbk_sym_spmv_dot_chunked` | 0.4830 | 0.4687 | 0.4670 | unchanged | −0.4 % |
| *control*: `StrainLimitingBaraffWitkinShell2D` G/H | 0.1155 | 0.1149 | 0.1149 | (32,256) unchanged | 0 |

### End-to-end, default frame counts, one build, **3 runs each arm**

| scene | arm | mean ms/frame (3 runs) | mean | ms per Newton it | Newton | PCG |
|---|---|---|---|---|---|---|
| **rigid-wrecking-balls** | A | 31.067 / 31.868 / 32.451 | 31.795 | 8.0209 | 466/479/482 | 12855/12865/13415 |
| | B (s20) | 29.285 / 28.407 / 28.321 | 28.671 | 7.2533 | 482/468/473 | 12720/12785/12960 |
| | **C (s20+s21)** | 25.223 / 25.531 / 25.461 | **25.405** | **6.3470** | 479/479/483 | 13150/12990/13405 |
| **cube-wall-cloth** | A | 60.716 / 61.785 / 62.841 | 61.780 | 12.1454 | 501/508/517 | 19595/19440/20095 |
| | B | 59.789 / 59.143 / 61.007 | 59.980 | 11.7220 | 510/507/518 | 19915/19540/20440 |
| | **C** | 54.202 / 56.765 / 56.848 | **55.938** | **10.8957** | 504/518/518 | 19520/20405/20420 |
| **stiff-gipc-case2** | A | 157.222 / 157.015 / 156.485 | 156.907 | 23.6544 | 1663/1657/1655 | 64830/64930/63525 |
| | B | 157.000 / 155.004 / 156.836 | 156.280 | 23.5549 | 1662/1652/1662 | 63645/62265/63670 |
| | **C** | 144.980 / 146.037 / 146.737 | **145.918** | **21.9933** | 1655/1663/1658 | 63680/64295/65315 |
| **mas-bunny** | A | 45.834 / 45.790 / 45.843 | 45.822 | 9.8543 | 465/465/465 | 35210/35220/35195 |
| | **C** | 45.195 / 45.180 / 45.191 | **45.189** | **9.7181** | 465/465/465 | 35220/35200/35220 |

| scene | s21 alone (C vs B) | whole launch-geometry step (C vs A) |
|---|---|---|
| rigid-wrecking-balls | **−11.4 % mean, −12.5 %/it** | **−20.1 % mean, −20.9 %/it** |
| cube-wall-cloth | **−6.7 % mean, −7.0 %/it** | **−9.5 % mean, −10.3 %/it** |
| stiff-gipc-case2 | **−6.6 % mean, −6.6 %/it** | **−7.0 % mean, −7.0 %/it** |
| mas-bunny | (B ≡ A: no ABD) | **−1.4 % mean, −1.4 %/it** |

**Every A / B / C distribution is disjoint on every scene**, run-to-run, mean and per-iteration
alike. Newton counts are unchanged across arms within their normal scatter (mas-bunny is exactly
465 in all six runs), so these are throughput numbers, not path changes.

`stiff-gipc-case2` has no ABD bodies, so arm B is a measured no-op there (−0.4 %, inside scatter);
its whole −7.0 % is s21, and the hinge is most of it. `mas-bunny` was a pure control for s20 and is
no longer one for s21: it drives the CCD feasible-step kernels and the contact line-search
energies, hence a small but perfectly repeatable −1.4 % with identical Newton and PCG counts.

## 6. Correctness gate

`bash /root/work/run_tests.sh`, compared against this box's own `baseline_tests.txt`.
**All pass, identical counts, no new failures and no flakes:**

| suite | s21 | baseline |
|---|---|---|
| common | 11 assertions / 3 cases | 11 / 3 |
| core | 1 112 / 36 | 1 112 / 36 |
| geometry | 2 730 / 46 | 2 730 / 46 |
| sanity_check | 100 / 3 | 100 / 3 |
| regression | 4 / 1 | 4 / 1 |
| backend_cuda | 352 / 22 | 352 / 22 |
| sim_case | 14 213 / 95 | 14 213 / 95 |
| pytest (`cuda and not example`) | 48 passed, 1 skipped | 48 passed, 1 skipped |

## 7. Cross-architecture expectation

This is the transfer claim the round should be most confident about, because **two workers now hold
it from opposite ends of the fleet** — w0 on cc 7.5 (40 SMs) and w3 on cc 12.0 (36 SMs) — and it is
structural rather than occupancy-based:

- A grid of 1–5 blocks is not an occupancy-tuning problem, it is a launch that never reaches the
  SMs. Resident warps per SM are unchanged by the swap (w0: 8 either way at 255 registers), so the
  usual "occupancy tuning does not transfer" caveat does not apply.
- The deficit scales with **SM count**. A 5090 has ~170 SMs: at base these launches occupy 1/170 to
  5/170 of the device — 0.6–3 % — against 3/36 to 5/36 here. **Expect every grid-1-to-5 site to pay
  more on the 5090 than it does here**, per site: `kinetic_shape_k2`, both vertex–half-plane
  assemblies, the vertex–half-plane energy, `affine_body_bdf1_kinetic` G/H, `feasible_step_PT`, and
  contact energies `k1`/`k4`.
- **Where it will pay *less*: the saturating sites.** s20's caveat now binds for real and it binds
  on the 5090, not here. The spread is capped by the item count: 576 ABD bodies = 18 warps, so at
  most 18 SMs are reachable — 50 % of this box and **11 % of a 5090**. `ortho_potential`,
  `abd_diag_preconditioner` and `kinetic_shape_k2` on rigid-wrecking-balls are all at that ceiling
  already (grid 18 of 36). They cannot get worse on a 5090, but their *relative* gain will not grow
  with SM count the way the unsaturated sites' will, because the extra SMs have nothing to run. The
  remaining lever there is more parallelism per item, which is w0's and w1's territory, not
  geometry.
- **The hinge is the one site with a measured cross-architecture number, and it went the other
  way**: w0 measured −23.3 % on a 2070S (cc 7.5), this box measures **−15.4 %** (cc 12.0), same
  geometry change `(48,256) → (380,32)`. The tail-wave argument explains it: 48 blocks on 40 SMs is
  1.2 waves rounded up to 2 (a 67 % penalty); 48 blocks on 36 SMs is 1.33 waves rounded up to 2 (a
  50 % penalty). The wave-quantisation gain depends on where `n / (blocks-per-wave)` falls relative
  to an integer, which is an **arithmetic accident of the SM count** — so this half of the helper's
  benefit is genuinely architecture-dependent in size (never in sign: rounding up can only cost).
  On a 5090, 12 288 items at 256 threads is 48 blocks over ~170 SMs = 0.28 waves, i.e. the hinge is
  in the *never-reaches-the-SMs* regime there rather than the wave-quantisation regime, and the
  gain should be **much larger than either measurement here**.
- Nothing can regress: with `n` large enough that the grid already covers the device several times,
  the helper returns `best_block_dim`'s answer unchanged, which is why the untargeted kernels move
  by ≤1 % in every table above.

## 8. Found in other workers' areas

- **w2 (D2H / host sync):** `fused_pcg_scalar_kernel` is launched `<<<1, 1>>>` **4 015 times** in a
  60-frame wrecking-balls window and costs **0.0199 ms per Newton iteration** — 0.3 % of the scene
  in pure launch overhead for a scalar update, and it is *inside* the CUDA-graph-captured PCG loop.
  It is not a geometry problem (there is one thread of work); it is a candidate for folding into the
  neighbouring `fused_dot` / `fused_update_p_beta` kernels.
- **w0 (`make_spd`) / contact owner:** `ipc_simplex_normal_contact::do_assemble` — the round's
  largest kernel at 2.28 ms/it on wrecking-balls — runs at grid 26–31 of 36 SMs at 255 registers,
  i.e. it is itself in the wave-quantisation regime (one resident block per SM, so grid 28 is one
  wave with 8 SMs idle, and grid 31 leaves 5 idle). The ramp would take it to grid ~219 at block 32.
  It was left alone here because it contains `__any_sync` and because a throughput change to the
  biggest kernel deserves its own measurement, not a line in a sweep. **This is the single largest
  remaining launch-geometry item in the suite.**
- **`cuda_tool::details::buffer_fill_kernel`** runs at grid 1–5 in every scene (0.0207–0.0310 ms/it).
  It is `cuda_tool` infrastructure shared by every subsystem; applying the helper inside
  `BufferLaunch` would fix it everywhere at once, and would be a one-line change to a file nobody
  owns yet.

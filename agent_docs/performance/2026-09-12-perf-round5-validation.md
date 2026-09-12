# Perf round 5 — validation pass (composed head, local RTX 2070 SUPER, cc 7.5)

> Merge target: append to `2026-09-12-perf-round5.md`. This is the pass `PERF_METHOD.md` §5
> describes — *"per-step gates are regression gates; they do not prove the composition is sound"* —
> and it is the gap round 4 admitted to and skipped for 17 steps.

**Subject.** `perf/round5` head `10fcbabc` (11 accepted steps s19-s32, s21 superseded, 8 rejections,
+4482/−573 source lines), against `perf-round5-base` = `890482c2`.
**Branch.** `perf/round5-validation`, cut from `fork/perf/round5` at `10fcbabc`.
**Box.** Local RTX 2070 SUPER (cc 7.5, 40 SMs, 8 GB, driver 595.84, CUDA 12.8), 32 cores / 60 GB.
**Builds.** `build-perf` (head, Release, sm_75, tests ON) and a **freshly configured, freshly built
`/workspace/deps/libuipc-valbase/build-base`** at `perf-round5-base` with identical flags and its own
venv — so every "pre-existing" claim below is *reproduced on a baseline binary*, not argued.

**Verdict: the round is sound to ship.** Nothing found invalidates the head. Details, and the four
things that are *not* covered, are at the end.

---

## 1. Composed correctness gate — four runs, identical every time

`uipc_test_{common,core,geometry,sanity_check,regression,backend_cuda,sim_case}` plus
`pytest -q -m "cuda and not example" python/tests`, run **four times** on the head (twice at the
start, once mid-pass, once after the tree had been patched and restored for the R7 experiment in §6).

| suite | head, all 4 runs | base build, 1 run |
|---|---|---|
| common | 11 assertions / 3 cases | 11 / 3 |
| core | 1112 / 36 | 1112 / 36 |
| geometry | 2730 / 46 | 2730 / 46 |
| sanity_check | 100 / 3 | 100 / 3 |
| regression | 4 / 1 | 4 / 1 |
| **backend_cuda** | **448 / 23** | **352 / 22** |
| **sim_case** | **14213 / 95** | **14213 / 95** |
| pytest | 48 passed, 1 skipped, 86 deselected | 48 passed, 1 skipped, 86 deselected |

The expected counts. The backend_cuda difference is the one case s23 added
(`apps/tests/backends/cuda/qr_svd.cu`), which is why every box's `baseline_tests.txt` reads 352/22 —
confirmed against the baseline binary rather than waved through. **No flakes**: zero failures in four
head gate runs, one base gate run, and a further six full runs of `backend_cuda`, `sanity_check` and
`regression` under the three sanitizers. Nothing needed statistical characterisation.

## 2. Sanitizers — the gap round 4 admitted to

`compute-sanitizer` 2025.1.0.0, **all three tools over all four scenes on both revisions**, 20 frames
each, plus the focused test binaries. Every non-zero finding was reproduced on the freshly built
baseline.

### Scenes, 20 frames, head vs base

| tool | rigid-wrecking-balls | cube-wall-cloth | stiff-gipc-case2 | mas-bunny |
|---|---|---|---|---|
| **memcheck** | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| **racecheck** (warnings) | 0 / 0 | **1305 / 1307** | **3379 / 3380** | **5996 / 6000** |
| **initcheck** (errors) | 0 / 0 | **25522 / 25522** | 0 / 0 | 0 / 0 |

*(head / base; racecheck reported 0 **errors** everywhere, only warnings.)*

### Test binaries

| binary | memcheck | initcheck | racecheck |
|---|---|---|---|
| backend_cuda | 0 / 0 | 0 / 0 | 0 / 0 warnings |
| sanity_check | 0 / 0 | 0 / 0 | 0 / 0 |
| regression | 0 / 0 | **12 / 12** | **882 / 887** warnings |
| sim_case | head partial (see limits) | not run | not run |

### The three findings, and the proof each is pre-existing

1. **racecheck warnings are all `MASPreconditionerEngine_*`.** With `--print-limit 600` (against the
   20 of the sweep, so this is not a sampling artefact) the *only* kernels carrying hazards on
   cube-wall-cloth are `prefix_sum_Lx`, `build_connect_mask_Lx`, `next_level_cluster` and
   `build_multi_level_R` — on **both** revisions (head 1311 warnings, base 1306 over 600 displayed
   hazards). `src/backends/cuda/finite_element/mas_preconditioner_engine.cu` is **not in the round's
   32-file diff**. This is the check the brief singled out ("this round shipped a great deal of new
   launch-geometry and atomics-adjacent code"): *no round-5-touched kernel appears in any racecheck
   hazard on any scene or binary.* The ±0.1 % warning-count wobble is the solver taking a marginally
   different trajectory, not a difference in kind.
2. **initcheck on cube-wall-cloth: 25 522 errors, the identical count on both revisions**, every one
   a 4-byte uninitialised global read inside CUB's `DeviceScanKernel`, called from
   `GlobalDyTopoEffectManager::Impl::_distribute` (`global_dytopo_effect_manager.cu`, also untouched
   by the round). Same address broadcast to many threads of block 0, consistent with the `N == 0`
   case: `loose_resize(selected_hessian, N+1)` sizes the buffer to 1 and the `if(n > 0)` guard skips
   `distribute_k3_kernel`, so `ExclusiveSum` scans one never-written int. Benign — the value read is
   past the last element that contributes to `h_total_count` — but it is a real latent defect and it
   is **not this round's**. Recorded for whoever owns that subsystem.
3. **initcheck on `uipc_test_regression`: 12 "Host API memory access error", identical on both**,
   from `GlobalVertexManager::Impl::compute_axis_max_displacement_argmax` via
   `MaxTranslationChecker::do_check`. The head's stack additionally shows s22's
   `cuda_tool::details::host_read_pinned` because that readback now goes through the funnel — but the
   count, the size (16 bytes) and the originating site are identical to base, so s22 relocated the
   copy, it did not create the defect.

## 3. Accumulated drift — composed, against each scene's own envelope

Default frame counts (120/120/250/100), **n = 5 per arm**, head vs the freshly built base. Wall
deltas are Mann-Whitney with the statistic stated, as this round's own rule requires.

| scene | head mean ms | base mean ms | Δ | U, p | Newton (head/base) | PCG (head/base) |
|---|---|---|---|---|---|---|
| rigid-wrecking-balls | 28.594 | 39.612 | **−27.8 %** | U=0, p=0.009 | 478.8 / 475.0 (p=0.35) | 13035 / 13000 (p=0.68) |
| cube-wall-cloth | 67.873 | 85.788 | **−20.9 %** | U=0, p=0.009 | 589.2 / 587.8 (p=0.84) | 22160 / 22028 (p=0.60) |
| stiff-gipc-case2 | 186.199 | 223.256 | **−16.6 %** | U=0, p=0.009 | 1660.2 / 1660.2 (p=0.84) | 64737 / 64622 (p=0.60) |
| mas-bunny | 67.293 | 74.002 | **−9.1 %** | U=0, p=0.009 | **465 / 465 exactly** | 35221 / 35225 (p=0.68) |

**Iteration counts do not move on any scene**, so the wall numbers are readable by this round's own
"if iteration counts move more than the effect you are claiming" rule. mas-bunny holds Newton at
exactly 465 in all ten runs, which makes it the reference scene for the whole pass.

**Observables.** Fourteen scalar components across the four scenes. Twelve overlap outright; the two
that separated at n=5 were re-measured at **n = 10 per arm**, with a third arm that turns s31 off:

| observable | head (n=10) | base (n=10) | head, `UIPC_CONTACT_SPD2=0` (n=10) | head vs base |
|---|---|---|---|---|
| cube-wall `cloth_min_y` | [0.22572, 0.24794] mean 0.23666 | [0.23458, 0.24948] mean 0.24257 | [0.22811, 0.25784] mean 0.24153 | U=24, **p=0.049** |
| wrecking-balls `ball_center[2]` | [−0.1762, 0.0909] mean −0.0511 | [−0.0927, 0.0302] mean −0.0070 | [−0.1437, 0.0587] mean −0.0331 | U=26, p=0.070 (was p=0.047 at n=5) |

- `ball_center[2]` **washed out** when n went 5 → 10. That is the round's own n=3 lesson repeating
  one rung higher: at n=5 a 1-in-20 rank accident is still a 1-in-20 rank accident.
- `cloth_min_y` is the one residual signal: head's mean sits **2.4 % lower** than base's, p=0.049 at
  n=10. Three things say it is not a defect. (a) The head distribution **contains** the base one
  ([0.2257, 0.2479] ⊃ [0.2346, 0.2495]) — it is a location shift inside a wider spread, not a
  different resting state. (b) It is **one of fourteen components tested**; 0.049 × 14 ≈ 0.7, so it
  does not survive the multiplicity it was found under. (c) **It is not s31's**: turning s31 off
  moves the mean to 0.2415, indistinguishable from head (p=0.199) *and* from base, and the
  all-switches-off arm of §5 spans 0.2284-0.2465, i.e. the whole range. The cause is the scene's own
  chaotic contact trajectory, which the round-record's own "vacuous controls" note already measured
  at ±4.8 % on this box.
- **s31 specifically** (the brief's special-attention item, the round's widest numerical change):
  nothing in the composed observables, the iteration counts, or the sanitizers distinguishes it. Its
  part-2 kernel switches instantiation cleanly (§4), Newton and PCG counts are unmoved with it on or
  off, and the one observable that separated at all moves the *wrong way* when it is disabled. The
  ~10x-wider p99.9 tail w2 measured against the old path's own tangent-frame arbitrariness does not
  reach any composed observable over 120 frames.

**Determinism, measured and worth recording.** The composed head is **not** run-to-run deterministic
on wrecking-balls, cube-wall or case2 — Newton moves 471-486 on wrecking-balls between identical
reruns — and this is equally true of the base. mas-bunny *is* effectively deterministic (Newton 465
every run, centroid stable to ~1e-7). Anyone designing a drift check on this codebase should use
mas-bunny as the instrument and treat the other three as envelope tests.

## 4. Env-switch audit — 32 new switches

`git grep` over `src`+`apps` at both revisions: **32 added, 0 removed**. Wall-clock A/B at n=2 could
not resolve anything below ~3 % (the reference block and the arm block drifted ~1-3 % apart against a
0.28 ms within-block sd), so the **primary instrument is kernel-level**: an `nsys` trace at fixed
frame count, diffed on `(kernel name incl. template arguments, grid, block, registers, launch
count)`. That is drift-immune and needs one run per arm. `nsysrun.sh` deletes the report and both
CSVs before each run and **raises if no CSV appears**, which is what caught the wrapper being broken
on this box at all (see §8) — the round-5 stale-trace defect could not have produced a silent result
here.

### 4a. The switches that select a different kernel instantiation — proved by the symbol

| switch | scene | default instantiation → `=0` | kernel ms (15 or 12 frames) |
|---|---|---|---|
| `UIPC_MAKE_SPD_JACOBI=0` (s19, hinge) | cube-wall | `DiscreteShellBending_..._kernel<1,1>` → `<1,0>` | 60.44 → 69.24 (**+14.6 %**) |
| `UIPC_MAKE_SPD_JACOBI=0` (s19, ortho) | cube-wall | `ortho_potential_..._kernel<1>` → `<0>` | 8.26 → 11.98 (**+45.0 %**) |
| `UIPC_MAKE_SPD_JACOBI=0` (s19, ortho) | wrecking-balls | `ortho_potential_..._kernel<1>` (reg 192) → `<0>` (reg 254) | 12.45 → 15.14 (**+21.6 %**) |
| `UIPC_QR_SVD_FIXED=0` (s23) | case2 | `SNH_..._kernel_occ<1,1,1>` → `<1,0,1>` | 108.77 → 125.87 (**+15.7 %**) |
| `UIPC_CONTACT_SPD_TQL=0` (s25) | wrecking-balls | `do_assemble_kernel<0,1,1,1,0>` → `<0,1,1,0,0>` | 47.54 → 51.64 (**+8.6 %**) |
| `UIPC_SNK1_STENCIL2=0` (s29) | case2 | `SNH_..._kernel_occ<1,1,1>` → `<1,1,0>` | 108.77 → 141.10 (**+29.7 %**) |
| `UIPC_SNK1_OCC=0` (s30) | case2 | `SNH_..._kernel_occ<1,1,1>` → `SNH_..._kernel<1,1,1>` | 108.77 → 141.06 (**+29.7 %**) |
| `UIPC_CONTACT_SPD2=0` (s31) | wrecking-balls | `do_assemble_kernel<0,2,0,0,1>` → `<0,2,0,0,0>` | 40.23 → 41.40 (**+2.9 %**) |

Two by-products worth keeping:

- **`UIPC_CONTACT_SPD_TQL=0` still does exactly what s25 measured, after s31 landed** — the
  carried-forward item the merge audit could only check statically. Part 1 flips `SpdTql`
  (`<0,1,1,1,0>` → `<0,1,1,0,0>`, +8.6 % on that kernel) while **part 2 stays on `Spd2=1`**
  (`<0,2,0,0,1>`, 28 launches both arms) — the two axes really are independent at runtime, not just
  in the source. The binary confirms the closure: `nm` finds exactly the 17 reachable instantiations (8 fused Part 0, 4 Part 1, 2 Part 2, 3 gradient-only),
  part 1 only ever with `Spd2=0` and part 2 only ever with `SpdTql=0`.
- **s29 and s30 are not additive.** Either one alone costs the SNH G/H kernel ~+29.7 % (141.1 vs
  141.1 ms); together they buy −22.9 %. The launch bound only pays on the stencil-2 body and vice
  versa. Nothing is wrong — but the acceptance run should not expect to read s30's contribution by
  disabling it alone.

### 4b. The switches that change launch geometry or launch counts

| switch | scene | what the trace shows |
|---|---|---|
| `UIPC_GRID_SPREAD=0` (s20+s24+s28 master) | wrecking-balls | ortho G/H 18×32 → 3×256 (12.45 → 36.93 ms), abd_diag assemble 18×32 → 3×256 (15.66 → 28.28), ortho energy 18×32 → 1×1024 (1.93 → 15.32), kinetic_shape_k2 → 3×256, bdf1 kinetic → 1×1024. **Whole-trace kernel time 307.2 → 388.6 ms** |
| `UIPC_GRID_SPREAD=0` | case2 | +6.3 % frame (n=2), geometry reverts at every site |
| `UIPC_GRID_SPREAD_BLOCK=32` | wrecking-balls | every spread site pinned to 32 threads; `pairFilter` 1772×256 → 1024×256 |
| `UIPC_GRID_SPREAD_BPSM=1` | wrecking-balls | ramp target drops, part of the sites keep the occupancy-max block |
| `UIPC_BUFFER_FILL_SPREAD=0` (s28) | case2 | fills revert 1×32 → 1×256, 51×64 → 13×256, 50×64 → 13×256 … |
| `UIPC_BUFFER_FILL_BPSM=8` | case2 | fill grids widen 195×256 → 390×128, 167×256 → 333×128, … (~100 launches at 12 frames) |
| `UIPC_SKIP_DEAD_FILL=0` (s26) | case2 | **adds back 37 `buffer_view_fill_kernel<Matrix3x3>` launches** at grids 7224-7357, **+26.7 ms** over 12 frames |
| `UIPC_SEG_NARROW_FILL=0` (s32) | case2 | **adds back 37 `buffer_fill_kernel<Matrix3x3>` launches** at grid 1119, +4.23 ms |
| `UIPC_SPMV_GRID_STRIDE=1` (R6, opt-in) | case2 | SpMV grid **7977 → 160**, 1245 launches both |
| `UIPC_SPMV_GRID_WAVES=4` | case2 | only reachable with `GRID_STRIDE=1`; changes the resident cap |
| `UIPC_PCG_FOLD=1` (R7, opt-in) | wrecking-balls | `fused_pcg_scalar_kernel<<<1,1>>>` (280 launches, 0.54 ms) **disappears**; `fused_update_p_beta_kernel` → `fused_update_p_scalar_kernel` (0.51 → 0.61 ms) |
| `UIPC_PCG_FOLD=2` (R7, opt-in) | wrecking-balls | scalar node likewise gone; dot-tail variant |
| `UIPC_PCG_FOLD_MAXGRID`, `UIPC_PCG_DOT_FENCE` | wrecking-balls | only reachable inside a fold arm; no effect on the shipped path |

### 4c. The D2H switches, proved on the memcpy rows

15 frames of wrecking-balls, D2H copies counted by kind:

| arm | D2H pinned | D2H pageable | total D2H | D2H time |
|---|---|---|---|---|
| head (default) | 606 | 15 | **621** | 789 us |
| `UIPC_HOST_SYNC_FAST=0` | 0 | 621 | 621 | 790 us |
| `UIPC_BVH_BATCH_COUNTS=0` | 816 | 15 | 831 | 1024 us |
| **`perf-round5-base`** | 0 | 832 | **832** | 1007 us |

A clean decomposition of s22: **the batching removes 211 round trips (−25.4 %)** and the funnel makes
the survivors pinned; each switch reproduces its own half, and the two together reproduce the
baseline exactly (831 against 832).

### 4d. Diagnostic switches

`UIPC_GRID_SPREAD_VERIFY`, `UIPC_BUFFER_FILL_VERIFY`, `UIPC_BVH_BATCH_COUNTS_VERIFY`,
`UIPC_HOST_SYNC_VERIFY`, `UIPC_PCG_FOLD_VERIFY`, `UIPC_PCG_FUSE_DOT_VERIFY`, `UIPC_SEG_FILL_POISON`,
`UIPC_BCOO_HASH`, `UIPC_FILL_PROBE`, `UIPC_D2H_PROFILE` — all default-off; see §5.

### 4e. The whole rollback, exercised at once — the round's strongest composed result

All twelve step switches set to `0` in one run, against the base build, default frame counts:

| scene | base (n=5) | head, all step switches off (n=3) | head default (n=5) |
|---|---|---|---|
| rigid-wrecking-balls | 39.61 [38.79, 40.62] | **39.93 [39.03, 40.97]** | 28.59 |
| cube-wall-cloth | 85.79 [85.11, 86.24] | **86.79 [84.96, 87.84]** | 67.87 |
| stiff-gipc-case2 | 223.26 [222.59, 224.72] | **223.11 [222.25, 224.14]** | 186.20 |
| mas-bunny | 74.00 [73.96, 74.07] | **74.02 [73.93, 74.07]** | 67.29 |

The all-off arm overlaps the baseline on every scene, and on the deterministic scene it matches to
**0.03 %**. Two things follow: **(a)** every accepted step has a working rollback and no step lost its
switch to a later step — the failure round 4 had; **(b)** nothing outside the switched paths
contributes measurably to the round's gain, i.e. the +4482 lines contain no unswitched behaviour
change large enough to see.

## 5. The round's own verification probes, composed and run together

All ~24 `*_VERIFY` / `*_PROBE` / `*_POISON` / `*_HASH` switches on **in a single run**, on all four
scenes (12 frames) and again at default frame counts for the ones that report periodically. This is
the check round 4 wishes it had run: it found one of its probes broken by a later step.

**Every probe that can fire on these scenes still fires, and every one reports zero mismatches.**

| probe | fired on | result |
|---|---|---|
| `UIPC_GRID_SPREAD_VERIFY` (s20/s24) | all | **16 tagged sites on wrecking-balls** (18 distinct tags across the four scenes), 100.9 M 32-bit words on wrecking-balls alone, **0 mismatching** — plus the two untagged ones: `[OrthoSpreadVerify]` 2 507 232 words and `[ABDDiagSpreadVerify]` 2 314 368 words, 0 mismatching |
| `UIPC_BUFFER_FILL_VERIFY` (s28) | all | wb 378 fills / 31.9 M words; cwc 596 / 229 M; **c2 636 fills / 1 995 432 377 words**; mb 438 / 575 M — **0 mismatching**, 5-35 skipped inside a stream capture (by design) |
| `UIPC_SEG_FILL_POISON=2` (s32) | all | 48-74 reports per scene, **`unwritten=0 partial=0` in every one** |
| `UIPC_HOST_SYNC_VERIFY` (s22) | all | `[d2h] verify=1398 bad=0` (wb), 1989/0 (cwc), 2193/0 (c2), 1396/0 (mb) |
| `UIPC_BVH_BATCH_COUNTS_VERIFY` (s22) | wb, mb @120/100 frames | **6000 checks, 0 mismatches** |
| `UIPC_SPMV_VERIFY` | all | c2: 1195 launches, max rel y **1.743e-15**, max rel dot **7.609e-15** |
| `UIPC_CONVERT_VERIFY` | all | shape 0, index 0; bit-mismatch 0 on wb/cwc, non-zero on c2/mb with **max\|diff\| 1.42e-14, rel 1.02e-16** (the converter's own atomic nondeterminism) |
| `UIPC_FILL_PROBE` | all | `unwritten_blocks=0 partial_blocks=0` every call |
| `UIPC_BCOO_HASH` | all | emits (c2 `nnz=286419 xor=0xb1c00ba278ee8287`) |
| `UIPC_TRIPLET_PATTERN_PROBE` | all | emits |
| `UIPC_ABD_DIAG_APPLY_VERIFY` | wb, cwc | 159 / 616 applies, 0 mismatched |
| `UIPC_MAS_{APPLY,INVERT,SCATTER}_VERIFY` | mb @100 frames | fire; scatter reports rel ~3e-13 against a ~1e3 reference |
| `UIPC_BVH_{REFIT,TWO_PHASE}_VERIFY` | wb, mb @120/100 | "N calls, **0 set mismatches**" |
| `UIPC_PCG_AP_ZERO_VERIFY` | wb, mb @120/100 | fires |
| `UIPC_PCG_FOLD_VERIFY` (R7) | wb with `UIPC_PCG_FOLD=1` | `[SpreadVerify] LinearFusedPCG::fold_update_p` **5 924 755 words, 0 mismatching** |
| `UIPC_PCG_FUSE_DOT_VERIFY` (R7) | wb with `UIPC_PCG_FOLD=2` | `[PcgScalarVerify] 28 805 words, 0 mismatching` |

**Three notes a future reader needs**, none of them a defect:

1. **`UIPC_PCG_FUSE_DOT_VERIFY` only does anything when `UIPC_PCG_FOLD=2`.** Its drain is gated on
   `fuse && fold == 2`. On the shipped default it is silent and verifies nothing — a reader could
   easily take it for a verifier of the shipped `fused_dot`. It works; it just is not about the
   default path. The same is true of `UIPC_PCG_FOLD_VERIFY` (needs `fold == 1`).
2. **`UIPC_SPMV_PROBE` has no output by design** — it launches three discarded timing variants and is
   readable only in a profiler; it also silently does nothing unless `UIPC_SPMV_VERIFY` is on too,
   because it borrows `m_verify_y`.
3. **`UIPC_MAS_R_TAIL_VERIFY` and `UIPC_BVH_SELF_CULL_VERIFY` never reached their conditions** on any
   of the four scenes (`m_total_num_clusters > m_total_map_nodes` and the self-cull path
   respectively). Not shown broken, not shown working. See the limits.

## 6. The four items carried forward to this pass

### 6.1 `UIPC_CONTACT_SPD_TQL=0` has never been re-measured since s31 landed — now it has

Done, and it is intact: §4a gives the audit table the SNH family got. The switch flips part 1's
instantiation (`<0,1,1,1,0>` → `<0,1,1,0,0>`, **47.54 → 51.64 ms per 15 frames, +8.6 %**), part 2 is
untouched by it and stays on `Spd2=1`, the end-to-end arm moves **+2.46 %** on wrecking-balls, and the
binary contains exactly the 17 reachable instantiations with no `SpdTql=1, Spd2=1` cross term.

### 6.2 R6's default-off path was only argued byte-equivalent — now it is run

`Spmv_rbk_sym_spmv_dot_chunked_kernel` on stiff-gipc-case2, 12 frames:

| build / arm | launches | grid × block | total | mean per launch |
|---|---|---|---|---|
| **head, default (`UIPC_SPMV_GRID_STRIDE` unset)** | 1245 | **7977 × 256** | 143.17 ms | **115.00 us** |
| **`perf-round5-base` build** | 1245 | **7977 × 256** | 142.92 ms | **114.80 us** |
| head, `UIPC_SPMV_GRID_STRIDE=1` | 1245 | **160 × 256** | 141.94 ms | 114.00 us |

Same launch count, **same capacity grid**, mean per launch within 0.17 % — i.e. inside the
per-launch scatter. The source argument (with the capacity grid `vb += gridDim.x` overshoots on pass
2, so the loop breaks after one pass) is confirmed at runtime: the shipped default really is the
pre-R6 kernel's behaviour, and the opt-in really does something.

### 6.3 R7's kept-in-tree code had never been A/B'd against pre-R7 — now it has

The residue is a `save_rz_prev` parameter and an `if(i == 0 && save_rz_prev)` store in
`fused_update_xr_kernel`, a hot kernel. A **pre-R7 variant of `linear_fused_pcg.cu` was built** (the
parameter and branch removed at the kernel and its one call site), measured, and the tree restored
and re-gated afterwards (§1, run 4).

*Static, sm_75, `-Xptxas -v` + `cuobjdump -sass`:*

| | registers | stack | spills | barriers | `cmem[0]` | SASS instructions |
|---|---|---|---|---|---|---|
| pre-R7 | 42 | 48 B | 0/0 | 0 | 488 B | **376** |
| shipped (with residue) | 42 | 48 B | 0/0 | 0 | **504 B** | **384** |

The eight extra instructions are a prologue `BSSY`/`BSYNC`/`BMOV` reconvergence trio around the
thread-0 store, one extra `ULDC` parameter load, one `ISETP`, and the guarded `LDG`/`STG` pair. No
register, spill or barrier cost.

*Runtime, `nsys`, 2 reps per arm, same grid (500×256 on case2, 27×256 on wrecking-balls), same 42
registers, same launch count:*

| | case2 rep1 / rep2 mean ns | wrecking-balls rep1 / rep2 mean ns |
|---|---|---|
| pre-R7 | 14 681.6 / 14 792.2 | 2 597.5 / 2 385.9 |
| shipped | 14 593.9 / 14 682.6 | 2 592.0 / 2 533.0 |
| untouched control (`fused_dot_kernel`) | 12 771.3 / 12 919.8 | 2 722.5 / 2 514.2 |

The shipped arm is nominally 0.6-0.7 % *faster*, which is nonsense — it is the same ±1.3 % run-to-run
movement the untouched control kernel shows between its own two reps. **The honest statement: the
residue's cost is below this kernel's own scatter at n=2 per arm.** Bounding it crudely, the kernel
is 18.2 ms of case2's 919 ms of kernel time, so ±1.3 % of it is ≲0.03 % of kernel time and ≲0.01 % of
frame time. It is free enough to keep; it is not *proved* free to better than that.

### 6.4 What `UIPC_GRID_SPREAD=0` actually reproduces

It rolls back **s20, s24 and s28 together**, as the merge audit suspected, and the trace says so
directly: turning it off simultaneously reverts the hand-rolled sites (`ortho_potential` ×2,
`abd_diag_preconditioner`), the 15 tagged `launch_spread` sites, and the `buffer_fill` ramp. It is
also an *exact* geometry rollback rather than an approximate one: `spread_grid_dim` with the gate off
computes `ceil(n / best_block_dim)`, which is `best_grid_dim`'s definition verbatim.

What it is **not** is a reproduction of pre-s24 s20. The criterion that ships is w0's blocks-per-SM
ramp (`spread_block_dim_from`, default `bpsm = 8`), not w3's binary SM test, and there is no switch
that restores the binary test — `UIPC_GRID_SPREAD_BPSM=1` is the nearest thing and it is a different
rule, not s20's. **No switch isolates s20 from s24.** On wrecking-balls the family as a whole is
worth **307.2 → 388.6 ms of kernel time (−21.0 %)**, i.e. ~90 % of the round's whole
wrecking-balls kernel-time gain (307.2 vs base 397.9).

## 7. The two instrument gaps the cleanup pass handed over — now quantified

### 7.1 `UIPC_GRID_SPREAD_ONLY` misses 86 % of what it claims to isolate

`SpreadVerifier::spread()` consults `UIPC_GRID_SPREAD_ONLY`; the three hand-rolled sites in
`ortho_potential.cu` (2) and `abd_diag_preconditioner.cu` (1) call `spread_grid_dim` /
`spread_block_dim` directly and never see it. Measured with an arm that matches no tag
(`UIPC_GRID_SPREAD_ONLY=ZZNOSUCHTAG`), 15 frames of wrecking-balls:

| site | tagged? | default | `ONLY=<no match>` | `GRID_SPREAD=0` |
|---|---|---|---|---|
| `abd_linear_subsystem_assemble_kinetic_shape_k2` | yes | 18×32, 1.12 ms | **3×256, 5.36 ms** | 3×256 |
| `affine_body_bdf1_kinetic_compute_gradient_hessian` | yes | 18×32, 0.66 ms | **1×1024, 3.41 ms** | 1×1024 |
| `affine_body_bdf1_kinetic_compute_energy` | yes | 18×32, 0.32 ms | **1×1024, 1.13 ms** | 1×1024 |
| `abd_linear_subsystem_assemble_kinetic_shape_k1` | yes | 18×32, 0.08 ms | **1×1024, 0.43 ms** | 1×1024 |
| `ortho_potential_compute_gradient_hessian` | **no** | 18×32, 12.45 ms | **18×32, 11.35 ms (unchanged)** | 3×256, 36.93 ms |
| `ortho_potential_compute_energy` | **no** | 18×32, 1.93 ms | **18×32, 1.78 ms (unchanged)** | 1×1024, 15.32 ms |
| `abd_diag_preconditioner_do_assemble` | **no** | 18×32, 15.66 ms | **18×32, 14.55 ms (unchanged)** | 3×256, 28.28 ms |

Whole-trace kernel time: default **307.2 ms**, `ONLY=<no match>` **319.0 ms**, `GRID_SPREAD=0`
**388.6 ms**. So of the spread family's 81.4 ms on this scene, the per-site instrument can switch off
**11.8 ms (14.5 %)** and cannot touch **69.6 ms (85.5 %)**.

**How s20's and s24's per-site numbers should therefore be read**: any `UIPC_GRID_SPREAD_ONLY` arm is
an *incremental* measurement on top of an always-on ortho/abd_diag spread, never an isolation of it.
Those two kernels are exactly s20's own subject, so a reader must not take a `GRID_SPREAD_ONLY` arm
as an s20-off arm. The published s20 and s24 verdicts do not depend on this — both were measured with
the master switch — but the instrument does not do what its name says, and fixing it means giving the
three sites their own `SpreadVerifier` tags, which moves launch geometry and so is a measurement, not
a cleanup.

### 7.2 `UIPC_BUFFER_FILL_SPREAD` is not an independent control of s28's ramp

Confirmed in source and at runtime. `buffer_fill_block_dim` → `spread_block_dim_from`, whose **first**
test is `grid_spread_enabled()`, so `UIPC_GRID_SPREAD=0` disables the fill ramp too, and
`UIPC_GRID_SPREAD_BLOCK=<n>` also pins the fill block size. The two arms are therefore nested, not
independent: `BUFFER_FILL_SPREAD=0` ⊂ `GRID_SPREAD=0`. `UIPC_BUFFER_FILL_BPSM` *is* an independent
control of the ramp *target* (it changes fill grids with the master gate on), and both were exercised
in §4b. The practical consequence is small — s28's own effect is 0.3 ms of a 307 ms trace on
wrecking-balls and 0.013 % of case2 by w3's own measurement — but a reader who sets
`UIPC_GRID_SPREAD=0` to "measure s20/s24" is silently also disabling s28.

## 8. Process findings

- **`nsys` on this box needs `/workspace/deps/nsight/nsight-systems/2024.6.2/bin/nsys`**; the
  `bin/nsys` shim fails with `Nsight Systems #VERSION_RSPLIT# hasn't been installed with CUDA
  Toolkit #CUDA_MAJOR#.#CUDA_MINOR#`. 28 profiling runs failed instantly — and were *caught*, because
  the wrapper raises when no trace CSV appears. The round-5 stale-trace defect is the same failure
  mode with the opposite symptom; the fix generalises: **a profiling wrapper must delete its outputs
  first and fail loudly if they do not reappear.** Copy at
  `data/2026-09-12-round5-validation/scripts/nsysrun.sh`.
- **Wall-clock A/B at n=2 is worthless on this box for effects under ~3 %.** The reference block and
  the arm block sat 1-3 % apart while the within-block sd was 0.28 ms (1.2 %). Interleave the
  reference, or use an instrument that does not measure time. Nine of the thirty-two switches would
  have been mis-audited on wall clock alone; all nine are unambiguous on the kernel trace.
- **CUDA MPS is running on this box** (`nvidia-cuda-mps-server`). It affected both arms identically
  and nothing here depends on it, but a "CUDA-capable device(s) is/are busy or unavailable" failure
  appeared once when a killed sanitizer left the device held; that run was redone.

## 9. What this pass did **not** cover

1. **`uipc_test_sim_case` under the sanitizers.** memcheck reached 19 of 95 Catch2 cases in 35
   minutes with **0 errors** and was stopped for time; initcheck and racecheck on it were not run.
   The CUDA-kernel surface is covered instead by `uipc_test_backend_cuda` (complete, all three tools,
   both revisions, 0 findings) and by the four scenes.
2. **Short sanitizer runs do not cover every trajectory.** 20 frames per scene exercises the assembly,
   contact, line-search and PCG paths repeatedly but not rare branches — a contact configuration that
   only appears at frame 200 of case2 is untested under memcheck.
3. **`UIPC_MAS_R_TAIL_VERIFY` and `UIPC_BVH_SELF_CULL_VERIFY` are untested**: their conditions are
   not reached on any of the four scenes. An untested fallback path is untested.
4. **The R7 residue's cost is bounded, not measured** — n=2 per arm, and the effect is under the
   kernel's own scatter (§6.3).
5. **`UIPC_SPMV_GRID_WAVES`, `UIPC_PCG_FOLD_MAXGRID` and `UIPC_PCG_DOT_FENCE`** were exercised only
   inside their own opt-in arms; they are unreachable on the shipped default, which is all the audit
   needs, but their *values* were not swept.
6. **One architecture.** Everything here is cc 7.5. The transfer claims in the round record are not
   tested by this pass; the 5090 acceptance A/B is.
7. **N runs per side is N runs per side**: n=5 for the composed drift, n=10 for the two observables
   that separated, n=3 for the all-off rollback, n=2 for most switch arms, n=1 per nsys arm.

## 10. Verdict

**Sound to ship.** Four identical composed gate runs; no sanitizer finding attributable to the round
(all three findings reproduced, with identical counts, on a freshly built `perf-round5-base`); no
kernel touched by the round appears in any racecheck hazard; iteration counts unmoved on all four
scenes so the −9.1 % to −27.8 % composed wall gain on cc 7.5 is readable; every one of the 32 switches
selects the path it claims, with the whole rollback reproducing the baseline to 0.03 % on the
deterministic scene; and every verification probe that can fire still fires and reports zero
mismatches. The two instrument gaps are real but are *measurement* limitations, not shipped defects —
they change how s20/s24/s28's per-site numbers should be read, not whether those steps work.

Raw evidence, logs and scripts: `data/2026-09-12-round5-validation/`.

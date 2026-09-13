## s06 — the conservative bound pays where the distance is already computed: a compacted candidate array for the DCD narrow phase

Evidence: `agent_docs/performance/data/2026-09-13-round6-s06/`
(`predictions.txt`, `ccd_compact_probe.{cu,txt}`, `ccd_compact_probe_5M.txt`, `insitu_verify_tumbler.txt`,
`sass_identity.txt`, `sass_resusage.txt`, `scope_all.txt`, `accounting.txt`, `cub_accounting.txt`,
`rerank_c1.txt`, `gate.txt`, `gate_c0.txt`, `ab/`, `verify/`, `nsys/`).

**Verdict: accepted, default on.** s05 proved the bound and proved the *placement* wrong. Moving it to
the one kernel that already evaluates the distance — and compacting the candidate array with it —
makes it free: `filter_active` runs **−84 % / −82 %** per launch on the tumbler and the four-kernel
accounting closes to **0.2 pp** against the scene. **`mas-bunny` −0.53 % `meanFrameMs`, disjoint,
p=4.4e-04 at Newton 465 in all ten runs; `stiff-gipc-case2` −1.88 % mean and −2.02 % ms/Newton, both
disjoint (p=2.4e-03 / 8.2e-05) with the iteration guard silent.** The active contact set handed to
contact assembly is **provably unchanged, sequence for sequence**, which is what makes this a
bit-identical change rather than a conservative-with-an-argument one.

### The brief's premise, checked first — and it is right, but two of its details are not

The brief (from s05's own candidate 1) proposed: *append the ~9 % survivors and their `toi` to a
compacted array, resize, and let `filter_active` run over 9 % of the pairs.* Three corrections, all
of which make the step smaller and safer:

1. **Only the DCD half of s05's condition is needed.** The compacted array is read by `filter_active`
   and by nothing else — `filter_toi` has already run over the full array by the time the compaction
   exists. So the CCD half (`(1-eta)(d0-thickness) > toc_prev*m`) is not required at all, and the cull
   is `(d0 - thickness) > d_hat + m` alone. That is strictly *looser* than s05's conjunction, so
   s05's 91.1 % / 91.7 % drop rate is a floor: measured in situ, **90.7 %** of a billion real tumbler
   candidates (the two are not the same population — s05 measured its rate at `detect` time).
2. **The `toi` must NOT be compacted, and neither must the candidate arrays themselves.** s06 writes
   the survivors into *separate* buffers and leaves `candidate_*_pairs` and `tois` exactly as they
   were. That is what keeps every position-indexed consumer on its old contract: the AL-IPC
   `GlobalActiveSetManager` (which reads `candidate_PTs()/EEs()` **and** `toi_PTs()/toi_EEs()` and
   builds `*_max_vertex_min_toi` per candidate), the `dcd_candidate_reuse_verify` snapshots, and the
   `dump_candidates` exporter. s05 had to force its cull off under `contact/constitution == "al-ipc"`;
   **s06 does not need that guard at all**, because AL-IPC never sees the compacted array.
3. **The ceiling arithmetic is right for three scenes and wrong for one.** Recomputed from s05's own
   `scope_*.txt`: tumbler `(529.0-73.0)+(303.0-47.4)` µs x 1816 launches = **−1292.3 ms / −5.69 %**
   (the brief's number, confirmed); `stiff-gipc-case2` **−940.4 ms / −2.16 %** (brief: −975 / −2.2 %,
   confirmed); `cube-wall-cloth` **−93.1 ms / −1.49 %**, *not* the brief's −119 ms / −1.9 %. All of
   this step's predictions were written down before any measurement (`predictions.txt`), including a
   coverage correction the brief does not have: the compaction only exists between a `filter_toi` and
   the next `detect`, and the frame's **first** `filter_active` — the one after
   `detect_dcd_candidates(0)` — has no `filter_toi` in front of it, so 90.1 % (tumbler) / 94.1 %
   (case2) / 83.1 % (cwc) / 98.7 % (mas-bunny) of `filter_active` launches can be covered at all.

### The change

`distance::{edge_edge,point_triangle,point_edge,point_point}_ccd` gain a fourth template parameter
`bool DcdCull` and two trailing defaulted arguments `T d_hat, uint8_t* dcd_keep`. When `DcdCull` is
on, at the point where the function has *already* computed `dist2_cur` and `maxDispMag`:

```
const T dcd_lim = (thickness + d_hat + maxDispMag) * (T(1) + T(1e-9));
*dcd_keep       = (dist2_cur > dcd_lim * dcd_lim) ? 0 : 1;
```

Two adds, two multiplies, a compare and a byte store; no sqrt (the test is done in squares, both
sides non-negative) and **no extra loads at all**. It reads the *first* `dist2_cur` — the one
`edge_edge_ccd`'s degenerate fall-back would otherwise overwrite with a **larger** endpoint distance
— so the fall-back cannot make the verdict unsafe. A zero `maxDispMag` stores 1 = keep.

`filter_toi_k1..k4` gain `bool Compact` and a trailing `uint8_t* keep`; `filter_toi` then runs
`cub::DeviceSelect::Flagged` (**order preserving**) per candidate array into four `compact_*` buffers
and reads the four counts back in one D2H that lands immediately before
`GlobalTrajectoryFilter::filter_toi`'s own blocking read of the per-filter toi, so it costs no extra
pipeline drain. `filter_active` reads the compacted arrays when they exist; `detect` clears the flag
that says they do. `UIPC_CCD_COMPACT=0` restores the old path; `UIPC_CCD_COMPACT_VERIFY=1` turns on
the two-pass in-situ check below.

**`filter_active_k1..k4` are not touched at all** — the same kernels, byte for byte, launched over a
shorter array.

### The register trap, recorded because it was invisible in the source

The obvious shape — have the ACCD return the verdict through a `bool*` and let the kernel do
`keep[i] = ...` after the call — costs **+15 registers on `filter_toi_k4` (218 -> 233)** and +8 on
`k3` (184 -> 192). On cc 7.5 that crosses a register granule: 218 regs allow 9 resident warps per SM,
233 allow 8; 184 allow 11, 192 allow 10. `best_block_dim` calls `cudaOccupancyMaxPotentialBlockSize`
**per kernel function pointer**, so the two arms would have launched at *different block sizes* — the
exact confound s04 went to trouble to avoid. Moving the store **inside** the ACCD, at the point where
the distance is first available, kills the pointer before the loop starts and puts both instantiations
back at **REG 218 / 184, STACK 0, LOCAL 0**, identical to main's. Cost of the compacting
instantiation: **+16 SASS instructions on `k4`, +24 on `k3`**.

### The old arm reproduces main's binary exactly

Standalone compile of the filter's TU with the build's exact flags at `711d2300` and at this head,
SASS-diffed (`sass_identity.txt`). The new template argument renames all 16 `filter_toi`
instantiations, so those are matched by body hash:

| | |
|---|---|
| common SASS streams identical | **88 / 88** — including `filter_active_k1..k4`, `pairFilter` x4, `stacklessSelf/Other`, every `detect_*` leaf predicate |
| base-only (renamed) `filter_toi_k1..k4<EarlyOut, Stats>` found **byte-identical** in head | **16 / 16** |
| registers, `filter_toi_k4/k3`, both arms | **218 / 184**, STACK 0, LOCAL 0 — unchanged from main |
| new in head | the 16 `Compact = true` instantiations, and one `cub::DeviceSelectSweepKernel<Vector2i>` |

### Numerics: the active contact set is provably unchanged, sequence for sequence

This is stronger than "conservative". Three facts compose into bit-identity:

1. **Every dropped pair is inactive over the whole swept step.** `filter_active` keeps a pair iff
   `d < thickness + d_hat`; the relative motion of the two primitives is bounded by `maxDispMag` per
   unit `t` (the ACCD's own bound), the line search only applies `alpha_ls <= alpha_detect`, so
   `d(t) >= d0 - maxDispMag > thickness + d_hat` for every `t` in `[0, 1]`.
2. **`cub::DeviceSelect::Flagged` is order preserving**, so the compacted array is an ordered
   *subsequence* of the raw one — checked element-wise, not assumed (below).
3. Therefore the compacted run's `temp_*` arrays are the subsequence of the raw run's restricted to
   surviving candidates, and since `DeviceSelect::If` is order preserving too, the four selected
   active sets are **equal as sequences iff their four sizes agree**.

**Standalone, against the real shipped device function** (`ccd_compact_probe.cu`,
`ccd_compact_probe_5M.txt`): 5x10^6 samples per pair type, **half adversarial** (near-parallel edges,
near-degenerate triangles, zero-length edges, coincident points, zero relative displacement,
separations placed *exactly* at `thickness + d_hat`, asymmetric per-primitive thickness, `d_hat`
log-uniform over 1e-4..1e-2 m). The verdict comes out of
`distance::*_ccd<Float, EarlyOut=true, Stats=false, DcdCull=true>` itself, and for every dropped
sample the probe walks the swept segment at 221 points:

| | EE | PT | PE | PP |
|---|---:|---:|---:|---:|
| samples | 5 000 000 | 5 000 000 | 5 000 000 | 5 000 000 |
| dropped | 27.13 % | 22.86 % | 29.70 % | 27.51 % |
| active somewhere on [0,1] over the whole population (sanity) | 33.25 % | 30.61 % | 26.26 % | 31.90 % |
| **dropped but ACTIVE somewhere on [0,1]** | **0** | **0** | **0** | **0** |
| **dropped but within `thickness` somewhere on [0,1]** | **0** | **0** | **0** | **0** |
| dropped and active only *past* t=1 (informational) | 19 | 12 | 29 | 143 |

The last row is worth keeping: a handful of samples per five million become active just **outside**
the domain of the claim. The bound is tight, not vacuous — it is not culling pairs that were never
going to matter.

**In situ, on the real population** (`UIPC_CCD_COMPACT_VERIFY=1`, one full 180-frame tumbler run,
`insitu_verify_tumbler.txt`). `filter_active` runs its whole pass **twice** — once over the raw
candidate arrays, once over the compacted ones — and compares the four active-set sizes; `filter_toi`
downloads the raw array, the flags and the compacted array and checks element-wise that the latter is
exactly the ordered subsequence the flags select:

| | |
|---|---|
| candidate pairs seen | **1 056 871 554** |
| dropped by the cull | **958 915 814 (90.73 %)** |
| active pairs produced | 40 765 709 |
| `filter_active` calls with a raw/compacted **active-set size mismatch** | **0 / 1600** |
| `filter_toi` calls whose compacted array is not the exact ordered subsequence | **0 / 1200** |

So over a billion real pairs at the real line-search positions, contact assembly receives the
identical four arrays in the identical order. Newton and PCG counts therefore cannot be moved by this
change, and `ms_per_newton` is legitimate wherever the harness's guard fires (the s04 reading).

### Performance, targeted scope: n=3 full runs per arm, one build, five scenes

`nsys --report cuda_gpu_kern_sum` over **complete** benchmark runs, fresh prefix per run.
(`scope_all.txt`, `accounting.txt`.)

| µs/launch | `filter_active_k4` (DCD EE) | `filter_active_k3` (DCD PT) | `filter_toi_k4` | `filter_toi_k3` |
|---|---|---|---|---|
| **tumbler-garments** | 484.2 -> 75.9 **−84.3 %** | 277.5 -> 49.0 **−82.3 %** | +3.1 % | +2.7 % |
| **stiff-gipc-case2** | 355.9 -> 47.0 **−86.8 %** | 195.9 -> 42.3 **−78.4 %** | +2.4 % | +2.4 % |
| **cube-wall-cloth** | 172.5 -> 64.9 **−62.4 %** | 103.7 -> 55.2 **−46.7 %** | +2.4 % | +2.1 % |
| **mas-bunny** | 43.9 -> 8.9 **−79.7 %** | 40.7 -> 8.2 **−80.0 %** | +1.2 % | +0.9 % |
| **rigid-wrecking-balls** | 107.1 -> 40.6 **−62.1 %** | 71.8 -> 39.4 **−45.1 %** | +3.7 % | +1.8 % |

All ten `filter_active` deltas have disjoint per-run ranges. The `filter_toi` column is the price:
the keep-flag store plus four arithmetic ops, **+1 to +4 %**, exactly the size the +16/+24 SASS
instructions predict. On `mas-bunny` and `cube-wall-cloth` the compacted array is often empty and the
kernel is then **not launched at all** (471 -> 123 / 471 -> 107 launches on mas-bunny, 589 -> 447 on
cube-wall-cloth), which is why those two read a smaller per-launch delta than the others and a bigger
total one.

**The cub stream compaction is a net win, not a cost** (`cub_accounting.txt`). The four new
`DeviceSelect::Flagged` passes are smaller than the shrink of `filter_active`'s own four
`DeviceSelect::If` passes, because the `temp_PP`/`temp_PE` arrays it selects over are sized as the
*sum* of the four candidate arrays:

| scene | `DeviceCompactInit` | `DeviceSelectSweep` | net |
|---|---|---|---|
| tumbler | +3.2 ms | −60.5 ms | **−57.3 ms (−0.26 % of scene)** |
| case2 | +3.7 ms | −52.9 ms | **−49.1 ms** |
| cube-wall-cloth | +1.1 ms | −5.6 ms | −4.6 ms |
| mas-bunny | +0.9 ms | −3.6 ms | −2.6 ms |
| rigid-wrecking-balls | +1.0 ms | −3.9 ms | −2.9 ms |

**The accounting, and how it is normalised.** The tumbler drew +3.2 % more Newton iterations in the
new arm over these three runs (`filter_toi_k4` launches 1425 -> 1470), so its *raw* scene total is
unreadable; dividing scene GPU kernel time by `filter_toi` launches removes it. The other four scenes
move <2 %.

| scene | four targeted kernels (ms) | + cub | = predicted | scene GPU kernel time / `filter_toi` launch |
|---|---:|---:|---:|---|
| tumbler-garments | −1072.1 | −57.3 | **−5.05 %** | **−5.26 %** |
| stiff-gipc-case2 | −893.9 | −49.1 | **−2.16 %** | −1.57 % |
| cube-wall-cloth | −103.3 | −4.6 | **−1.75 %** | −1.57 % |
| mas-bunny | −37.3 | −2.6 | **−0.61 %** | −0.75 % |
| rigid-wrecking-balls | −59.1 | −2.9 | **−1.69 %** | −0.64 % (Newton −2.0 % between arms; unreliable) |

The tumbler closes to **0.2 pp**, which is s05's standard. On `stiff-gipc-case2` the gap is 0.6 pp in
the other direction and it is scatter, not a hidden cost: the untouched families read +1.0 to +1.3 %
in the new arm at n=3, but their per-run ranges overlap (spmv c0 7513.8 / 7623.6 / 7723.8 vs c1
7662.1 / 7739.1 / 7762.8), and the n=5 end-to-end below lands on −2.02 %, i.e. on the accounting
rather than on the n=3 normalised total.

### Performance, end to end: ABBA + one discarded warm-up, one build

| scene | predicted | `meanFrameMs` | `ms_per_newton` | Newton / PCG | n |
|---|---:|---|---|---|---|
| **mas-bunny** | −0.61 to −0.75 % | **−0.53 % (DISJOINT, p=4.4e-04)** | −0.53 % (DISJOINT) | **+0.00 % / −0.01 %**, Newton 465 in all ten runs | 5 |
| **stiff-gipc-case2** | −1.57 to −2.16 % | **−1.88 % (DISJOINT, p=2.4e-03)** | **−2.02 % (DISJOINT, p=8.2e-05)** | +0.14 % / +0.50 % — guard silent | 5 |
| **cube-wall-cloth** | −1.57 to −1.75 % | −1.37 % (overlapping, p=0.41) | −1.86 % (p=0.040) | +0.51 % / −0.17 % | 5 |
| **rigid-wrecking-balls** | −1.69 % | −1.73 % (overlapping, p=0.17) | −1.35 % (p=0.031) | −0.38 % / −0.34 % — guard silent | 6 |
| **tumbler-garments** | −5.05 to −5.26 % | −2.06 % (overlapping) | −2.31 % | +0.30 % / **+3.92 %** — **guard fired** | 5 |

**No scene regresses on any statistic.** The two deterministic-ish instruments carry the result:
`mas-bunny` runs Newton 465 and PCG flat to 0.01 % in all ten runs, so `meanFrameMs` *is* its
throughput statistic, and it reads −0.53 % disjoint against a predicted −0.61 %; `stiff-gipc-case2`
reads −1.88 % / −2.02 % disjoint with Newton +0.14 % and PCG +0.50 %.

**The tumbler is unreadable and that is the honest reading**: PCG moved **+3.92 %** against a 2.06 %
wall change, so the harness's guard fired and the wall number is a different scene state, not
throughput. Its evidence is the scope gate (−5.26 % normalised, −5.05 % accounted) and the fact that
the four other scenes land on their predictions. This is the same conclusion the round record already
records for this scene: it cannot resolve its own biggest wins.

### Gates

- **Correctness**: `bash /workspace/output/round6/gate.sh` against `baseline_tests.txt` in **both**
  arms — identical assertion counts on all eight suites (11/3, 1112/36, 2730/46, 100/3, 4/1, 448/23,
  **14213/95**, pytest 48 passed 1 skipped). The only textual difference in either file is pytest's
  own wall-clock string (1.20 vs 1.21 / 1.22 s). The gate that ran with the compaction **on** is the
  stronger one: `sim_case`'s 14 213 assertions all pass with 91 % of the DCD candidate array removed.
- **Numerics**: the two proofs above (2x10^7 randomised samples with half adversarial, and 1.06x10^9
  real candidate pairs in situ with 0 active-set mismatches over 1600 `filter_active` calls and 0
  contents mismatches over 1200 `filter_toi` calls).
- **Env-switch audit at kernel level**: `UIPC_CCD_COMPACT=0` restores the old per-launch cost exactly
  on all five scenes (a 45-87 % per-launch difference on `filter_active` is not something a broken
  switch could fake), and the old arm's 16 `filter_toi` instantiations plus all 88 other kernels in
  the TU are byte-identical to main's SASS.

### Transfer prediction

**Algorithmic — it transfers.** The step launches `filter_active` over 9 % of the threads, writes 9 %
of the `temp_*` traffic, and selects over 9 % of it, at an unchanged output. Nothing about it depends
on occupancy, register pressure, spill traffic or wave quantisation — and the register trap above was
found and removed precisely so that it does not.

Three caveats a rented run should test:
- the work removed from `filter_active` is a **mix**: an FP64 distance-flag + distance evaluation
  (which this 1/32-rate part over-rewards) and ~36 B/pair of `temp_*` writes (which it does not). The
  per-launch −62 to −87 % should hold, but the FP64 half of it shrinks on a modern part.
- the **share** of `filter_active` in scene GPU kernel time is what sets the end-to-end number, and
  shares move with architecture (§6). On cc 7.5 it is **6.16 %** of the tumbler, 2.95 % of
  rigid-wrecking-balls, 2.64 % of cube-wall-cloth, 2.58 % of stiff-gipc-case2 and 0.61 % of
  mas-bunny.
- the **drop rate is a property of the workload, not the hardware**: 90.7 % on the tumbler over a
  billion pairs, and the per-launch deltas on five very different scenes agree in sign and to within
  40 points, so the mechanism reproduces anywhere.

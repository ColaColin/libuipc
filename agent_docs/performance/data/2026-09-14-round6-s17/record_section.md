## s17 — the converter's segmented reduce ran five warp-tree levels for every warp when its segments average 6.5 elements: stopping at the first level no lane needs is **−22.4 % / −24.8 % of the kernel** on `stiff-gipc-case2` / `mas-bunny`, bit-identical for every slot the kernel stores — **−0.95 % of `mas-bunny` end to end (n=10, disjoint, p=2.5e-10, Newton 465 in every run) and **−1.10 % ms/Newton (n=16 over two independent sweeps, t=8.5, p=2.2e-9; each sweep −1.1 %, the second disjoint)** of `stiff-gipc-case2`**, both predicted from the share; and the probes put a **5.9 % ceiling** on everything a re-summation could still take from this kernel

Evidence: `agent_docs/performance/data/2026-09-14-round6-s17/`.

**Ships default ON as `UIPC_SEG_REDUCE2=1`; `UIPC_SEG_REDUCE2=0` launches the round-4 (s10) kernel, whose
SASS is byte-identical to main's — the A/B arm and the rollback.** Diagnostics (all default off):
`UIPC_SEG_VERIFY=1` (device-side bit comparison against the old kernel, per slot class),
`UIPC_SEG_PROBE=1` (two timing stubs into scratch), `UIPC_SEG_HIST=k` (per-warp tree-level histogram).

### The brief's two premises, checked first — one holds, one is not what the code does

- **"4.8 % of case2, ~3.5 % of `mas-bunny`, untouched"** — holds at this head (`4181f28b`, nsys
  `cuda_gpu_kern_sum`, full runs, fresh prefix): `fast_segmental_reduce_matrix_kernel` is **4.88 % of
  case2** (1960 ms, 1664 launches x 1177.9 µs, 14 699–15 706 blocks x 128 at 46 registers, the
  fifth-largest kernel in the scene) and **3.76 % of `mas-bunny`** (489.2 µs x 465). The converter chain
  around it, re-ranked: the CUB radix sort **2.94 % / 2.68 %** (709 µs per convert on case2 — larger than
  every other converter kernel combined), `fused_compact_keys_k2` 0.61 / 0.46 %, the sorted-key unpack
  0.35 / 0.26 %, `zero_cross_warp` 0.27 / 0.14 %, `fused_flag_upper_k1` 0.25 / 0.18 %, scans + RLE ~0.45 /
  0.3 %; the chain totals **~10.2 % of case2 and ~8.0 % of `mas-bunny`** (`scope.md`, `nsys/head_*.csv`).
- **"s13 found the converter uses the destination arrays as sort scratch (`matrix_converter.inl:485`)"** —
  not what the line does. At s13's head line 485 is `to.resize_triplets_discard(from.triplet_count())` in
  the non-fused `convert()`, and the default fused `convert_sym` does the same at its step 5
  (`to.resize_triplets_discard(m)`): the destination is *resized* to the raw upper-triangular count before
  the RLE shrinks it to `h_count`, and **nothing writes `m` entries into it** — every sort buffer
  (`ij_hash*`, `sort_index*`, `ij_pairs`, `unique_*`, `sorted_partition_*`) is the converter's own. So
  `bcoo_A`'s capacity is coupled to the raw count by a *resize ordering*, not by a scratch dependency, and
  the decoupling is two lines (resize after the RLE; reserve `bcoo_A` from a reduced high-water mark),
  not a new scratch allocation. **Decided by measurement, not built**: `run_benchmark.py` records case2's
  peak at **1968 MiB of 8192** (`baseline-runs/stiff-gipc-case2.json`, `gpuMemory.peakTotalMiB`), so the
  ~180 MB it would free (2.65 M slots x 80 B against 286 k x 1.5) is not binding on this box, and it has
  no kernel-time effect at all now that s13's fitted grid already removes the idle SpMV blocks. It stays a
  footprint candidate (below), and the kernel was the better target.

### The finding, from the SASS and a histogram, before any code

`cuobjdump -sass` of the shipped kernel (`sass/census_gls.txt`): **392 instructions, 50 DADD, 95 SHFL, 13 LDG**,
2 branches. The nine `cub::WarpReduce::HeadSegmentedReduce` calls each run **five fixed shuffle levels**
(offsets 1, 2, 4, 8, 16) and at each level every lane executes `if(lane + offset <= last_lane) v = other + v`
— the add is *predicated per lane*, so a level at which no lane of the warp qualifies still costs the warp
one FP64 instruction per matrix entry, 9 x ~16 clocks on a 1/32-rate part. The predicate is monotone in
the offset (a lane that fails at 2^d fails at 2^(d+1)), so the useful trip count of a warp is
`ceil(log2(longest in-warp run))`, and the sorted input's runs average **6.5 elements** (case2: in = 1.86 M
per launch, out = 286 k; bunny 819 k / 129 k).

`UIPC_SEG_HIST` counts the warps of a launch by that trip count (`verify/new_vs_old_*.txt`):

| scene | in / out per launch | warps | levels 0 / 1 / 2 / 3 / 4 / 5 | mean levels | FP64 removable, bit-identically |
|---|---:|---:|---|---:|---:|
| `stiff-gipc-case2` | 1 856 397 / 286 419 | 58 016 | 0 / 0 / 0 / 18 381 / 23 014 / 16 621 | **3.97** | 20.6 % |
| `mas-bunny` | 818 543 / 128 736 | 25 580 | 0 / 0 / 0 / 8 463 / 10 458 / 6 659 | **3.93** | 21.4 % |
| `tumbler-garments` | 390 652 / 55 350 | 12 208 | 0 / 0 / 0 / 2 629 / 6 247 / 3 332 | 4.06 | 18.8 % |
| `cube-wall-cloth` | 194 177 / 48 001 | 6 072 | 600 / 0 / 0 / 1 612 / 1 622 / 2 238 | 3.71 | 25.8 % |
| `rigid-wrecking-balls` | 5 740 / 5 740 | 180 | **179** / 0 / 0 / 0 / 0 / 1 | 0.03 | 99 % — **no duplicate triplets at all** (the ABD pair pre-reduction of round 4's s03 leaves nothing to sum) |

### The change (`fast_segmental_reduce_matrix_k2_kernel<..., Probe>`)

Same prologue as the old kernel (three key reads, the 72-byte gather through the permuted value functor,
the head / cross-warp flags, the int flag reduce), then `last_lane` computed exactly as cub's
`SegmentedReduce<HEAD_SEGMENTED>` computes it (ballot of the head flags, shifted to tail flags, masked to
lanes >= the caller, the warp's last lane forced in, `__clz(__brev(...))`), and the tree as an unrolled
five-step loop that **breaks at the first step where `__any_sync(lane + offset <= last_lane)` is false**.
The per-lane add is `value = op(other, value)` with `other = __shfl_down_sync(full, value, offset)` — the
generic-op path cub takes for `::cuda::std::plus<double>` (its `add.f64` asm specialisation is for
`cub::Sum` only), so every lane executes the same adds on the same operands in the same order as before and
simply stops issuing the ones cub would have predicated off. Stores and atomics are unchanged. The old
kernel is untouched (not templated over, not moved); the k2 kernel is dispatched beside it, and
`FastSegmentalReduce<64, 32>` / the scalar reduce keep the old path (the tree assumes a full physical warp).

Binary: k2 is 1 359 static instructions (the unrolled loop's five exit paths duplicate code: 94 static
DADD for the same 45 per-lane maximum), **48 registers vs 46 — the same 8-register allocation granule at
128-thread blocks, so the same occupancy**, 0 B stack in both. **All 57 / 63 / 38 pre-existing functions in
the three TUs that instantiate this header (`global_linear_system.cu`, `global_dytopo_effect_manager.cu`,
`external_articulation_constraint.cu`) are SASS-identical to the head object** (`sass_identity.txt`,
normalised per function; 29 head-only functions are the k2 / probe / verify / histogram kernels). The
`=0` arm is main's binary.

### Numerics: bit-identical where the old kernel stores, at the old path's own noise where it accumulates — proved on device

`UIPC_SEG_VERIFY=1` runs, after every production launch, the **old** kernel into a zeroed scratch copy and
compares every 64-bit word, splitting the slots into two classes computed from the keys: a segment that
spans **>= 3 warps** takes >= 3 atomic operands in *both* kernels and its result depends on arrival order
(2-warp segments are `a + b`, commutative, exact); everything else must match bit for bit.

| run (short, all scenes) | calls | words compared | mismatching words | of which in >= 3-warp slots | **elsewhere** | multi-warp slots | max rel |
|---|---:|---:|---:|---:|---:|---:|---:|
| **new vs old, `stiff-gipc-case2`** | 14 | 36 093 312 | 8 443 | 8 443 | **0** | 6 390 | 2.1e-14 |
| **new vs old, `mas-bunny`** | 12 | 13 903 488 | 3 198 | 3 198 | **0** | 2 352 | 3.8e-14 |
| new vs old, `cube-wall-cloth` / rwb / tumbler | 9 / 7 / 8 | 3 888 081 / 361 620 / 3 985 200 | **0 / 0 / 0** | — | **0** | 0 | 0 |
| *control:* **old vs old**, case2 (`UIPC_SEG_REDUCE2=0`) | 14 | 36 093 312 | 8 109 | 8 109 | **0** | 6 390 | 4.1e-13 |
| *control:* old vs old, `mas-bunny` | 12 | 13 903 488 | 2 962 | 2 962 | **0** | 2 352 | 5.1e-14 |

**0 mismatching words outside the arrival-order class over 58 M compared words on five scenes, and inside
it the new-vs-old count (8 443 / 3 198) is the old-vs-old count (8 109 / 2 962)** — the same class, the
same size, the same magnitude (round 4's s10 verify measured the same class at 22 466 words per 12 case2
frames). The kernel computes the same values as main's; the matrix's residual non-determinism is the
pre-existing cross-warp atomic order, present in both arms.

Counts: `mas-bunny` Newton 465 and line search 465 in all 20 A/B runs, PCG 35 185–35 245 (inside the
baseline's 35 185–35 240 envelope in both arms); case2 Newton −0.16 % / PCG −0.18 % (sweep 1),
Newton −0.36 % / PCG −0.19 % (sweep 2), no PCG excursion in either sweep.

### Scope: one build, env switch, full runs, identical launch geometry (`scope.md`)

| scene | old (cub tree) | **k2** | delta | k2 Probe=1, no tree | k2 Probe=2, no gather |
|---|---:|---:|---:|---:|---:|
| **`stiff-gipc-case2`** (250 f, 1665 launches) | 1179.3 µs | **915.0 µs** | **−22.4 %** | 861.8 | 803.7 |
| **`mas-bunny`** (100 f, 465) | 488.4 µs | **367.4 µs** | **−24.8 %** | 361.2 | 318.7 |

Block count, block size and the `(blockIdx, threadIdx) -> element` map are unchanged, so s13's per-block
CUPTI over-credit does not apply; the untouched SpMV / SNH / MAS kernels read within 0.3 % between the
sessions (`scope.md`). Whole-run kernel time −1.24 % (case2) / −1.11 % (bunny) at flat counts.

**The probes bracket what is left, and it is not much.** With the *entire* remaining tree deleted (Probe=1)
the kernel reads 861.8 / 361.2 µs against the production 916.0 / 369.9 — **54 µs = 5.9 % on case2, 9 µs
= 2.4 % on the bunny** — and with the gather deleted instead (Probe=2) 803.7 / 318.7. Read together: the
tree (~700 µs) and the random 72-byte gather (~760 µs) now cost about the same and overlap almost
completely; the old kernel's five-level tree (~880 µs) was the one that stuck out. **That 5.9 % is the
ceiling for any rounding-level re-summation of this kernel** (K elements per thread with in-thread
pre-adds, a serial per-output loop, ...), which is why none was built: the bit-identical stop took ~83 %
of what the tree can give, and the remainder is worth ~0.3 % of case2 at the cost of a changed summation
order on the system matrix. The kernel is now gather-bound.

### Performance, end to end — predicted first (`predictions.txt`), then measured

Predicted from the share before the sweeps: **`mas-bunny` −0.8…−1.0 %** (−0.93 % of kernel time on a
GPU-bound scene), **case2 −1.0…−1.3 % ms/Newton** (−1.10 % of kernel time), cube-wall-cloth flat (the
launch is a tenth of case2's, well under 1 % of that scene), rwb 0 (the kernel is ~5 µs there — but it
*does* execute, tree skipped, so it is a real control), tumbler not run (2.50 % share per s00, −0.55 %
predicted against a ~4 % resolution).

`ab.py`, one build, ABBA, one discarded warm-up per arm, `UIPC_SEG_REDUCE2=0` vs `=1`
(`ab/ab_*.txt`, per-run json in `ab/<scene>/`):

| scene | n/arm | mean ms/frame old → new | delta | ms per Newton | ms per PCG | Newton | PCG |
|---|---:|---|---:|---|---|---|---|
| **`mas-bunny`** | 10 | 62.4494 → **61.8565** | **−0.95 %, t=12.6, p=2.5e-10, DISJOINT** | **−0.95 %, DISJOINT** | −0.96 %, DISJOINT | **465 in all 20 runs** | +0.01 % (35 185–35 245) |
| **`stiff-gipc-case2`, sweep 1** | 8 | 157.1009 → 155.1061 | −1.27 %, t=5.05, p=1.8e-4, overlapping | **−1.11 %, t=5.9, p=4.1e-5** | −1.10 %, p=1.4e-3 | −0.16 % | −0.18 % |
| **`stiff-gipc-case2`, sweep 2** (independent, same protocol) | 8 | 157.2027 → **154.9423** | **−1.44 %, t=6.21, p=2.3e-5, DISJOINT** | **−1.08 %, t=5.67, p=6.3e-5, DISJOINT** | −1.25 %, p=9.7e-3 | −0.36 % | −0.19 % |
| **`stiff-gipc-case2`, pooled** | **16** | 157.1518 → **155.0242** | **−1.35 %, t=8.16, p=4.1e-9** | **−1.10 %, t=8.47, p=2.2e-9** | −1.17 %, p=3.8e-5 | −0.26 % (p=0.002) | −0.19 % (p=0.64) |
| `cube-wall-cloth` (control) | 6 | 55.8396 → 55.7468 | −0.17 %, p=0.92 | **+0.05 %, p=0.93** | −0.63 %, p=0.11 | −0.23 % | +0.46 % |
| `rigid-wrecking-balls` (control) | 6 | 25.4171 → 25.4194 | +0.01 %, p=0.99 | +0.68 %, p=0.18, **guard fired** (Newton −0.67 %, PCG −0.84 %) | +0.85 % | −0.67 % | −0.84 % |

Both targeted scenes land on their predictions, `mas-bunny` disjoint at n=10 with every count identical,
case2 at −1.10 % ms/Newton pooled over 16 runs per arm (t=8.5), with the second sweep disjoint on both the mean and ms/Newton; its pooled Newton count drifts −0.26 % (p=0.002), a quarter of the effect and — since the kernel computes the same values as main's for every slot whose value is defined — the scene's own trajectory scatter, so ms/Newton is the statistic to read (the raw mean, −1.35 %, carries some of that drift). Peak GPU memory is unchanged in both arms (`gpuMemory.peakTotalMiB` case2 1888–1918 old vs 1895–1914 new, `mas-bunny` 1253 in all 20 runs): the default path allocates nothing new The two controls are flat: cube-wall-cloth's ms/Newton is +0.05 % (p=0.93)
inside its ±0.54 % envelope, and rwb's wall is unreadable by the harness's own guard (its counts moved
0.7–0.8 % against a 0.01 % wall change — trajectory scatter on a scene where the kernel is ~5 µs per
launch and 179 of its 180 warps skip the tree), reported as flat, not as a win. **No scene regresses.**

### Gates

- `gate.sh` vs `baseline_tests.txt`, default arm (`=1`): **identical counts** — common 11/3, core 1112/36, geometry 2730/46, sanity_check 100/3, regression 4/1, backend_cuda 448/23, sim_case **14213/95**, pytest 48 passed 1 skipped (`gate_default.txt`, also `/workspace/output/round6/gate_after_s17.txt`); the only textual difference from the baseline file is pytest's wall time. The `=0` arm executes main's device
  functions (SASS identity above) and the untouched host branch, so its gate is s16's.
- `pgrep -af` clean at the end of the step; tree returned to `main`, `build-perf` rebuilt from main.
- Tumbler `--verify` not run: the kernel computes the same values as main's for every slot whose value is
  defined (proved above on the tumbler's own population, 3 985 200 words, 0 mismatches), so a trajectory
  audit has nothing to find; the tumbler was also not benchmarked (predicted −0.55 % against a ~4 %
  resolution). Stated as a limit.

### Transfer prediction

Category (PERF_METHOD §6): **algorithmic — fewer FP64 warp instructions for the same bytes, same launch
geometry, same values.** Not wave quantisation (grid unchanged), not spill (0 B stack), not an occupancy
effect (same register granule; measured 46 → 48). Two things move the *size* on another part: (1) FP64 rate
— a 1/32-rate part over-rewards FP64 removal, and a 1/64-rate consumer Blackwell would reward it *more* per
instruction while its ~4x bandwidth shrinks the gather side, so the kernel there is more tree-bound than
here and the fraction should be at least as large; (2) the level histogram is a property of the mesh, not
the GPU (3.97 / 3.93 mean levels), so the 21 % of FP64 removed is the same everywhere. **Sign transfers;
the fraction is expected to hold or grow.** The number a rented run should compare is the per-launch pair
in `scope.md` plus the Probe=1 floor, which says how far from gather-bound the kernel is on that part.

### Limits, stated plainly

- Scope numbers are single sessions per arm; every cross-arm wall claim is from the `ab.py` sweeps.
- The dytopo manager's non-fused `convert()` and `external_articulation_constraint` reach the same host
  function and therefore the k2 kernel, but neither runs on the fast path of any suite scene (case2 has
  zero `distribute_*` launches, s14); they are covered by the SASS identity and the shared code, not by a
  timing.
- The 48-register k2 kernel is 1 359 static instructions from the unrolled early-exit; a `#pragma unroll 1`
  shape was not tried (it would trade the duplicated tails for a loop-carried offset).
- `clang-format` is still not installed on this box; the style was kept by hand.

## Candidates for the next step

| candidate | measured evidence | where to gate it |
|---|---|---|
| **The CUB radix sort behind the reduce: 2.94 % of case2 / 2.68 % of `mas-bunny`, and its key is twice as wide as it needs to be** | `nsys/head_c2.csv`: the onesweep passes total **1180 ms / 1664 converts = 709 µs per convert** on case2, more than every other converter kernel combined. The key is `uint64 row * cols + col` sorted over `bit_width(rows * cols − 1)` bits (`matrix_converter.inl`, `matrix_converter_key_bits`), with `SortPairs<uint64, int>`; for every suite scene `rows * cols < 2^32` (case2's blocked DoF count is ~40 k), so a **32-bit key** halves the key bytes each pass moves (12 → 8 B per pair per pass) at the same 4 passes. Bit-identical by construction (a permutation). Beyond that, round 4's s10 rejection records the *stable-prefix* cache (case2 92–96 % of the pattern unchanged between iterations) as still open — it would remove the sort, not shrink it | case2 (2.9 %) + `mas-bunny` |
| **The reduce is gather-bound now: 862 µs of its 915 are the random 72-byte gather through `perm`** | Probe=1 vs Probe=2 above. Each lane's 9 x 8 B block at 8-byte alignment always spans **3 sectors (96 B for 72 B, 33 % waste)**, and the source order is the assembly's element order, so locality is the mesh's. Two shapes, both bit-identical: (a) gather in the *sorted* order once per Newton iteration only where the pattern is stable (the same s10 prefix finding), (b) an element/vertex ordering pass on load (RCM-style) so that consecutive sorted keys read nearby source blocks — a scene-preprocessing candidate whose ceiling is the 862 → ~500 µs DRAM floor | case2 per launch; needs a locality probe first |
| **`bcoo_A` reserved from the raw triplet count x 1.5 (~180 MB on case2) by a resize ordering, not a scratch dependency** | this step's premise check: `convert_sym` step 5 resizes `to` to `m` before the RLE; moving that after the RLE and reserving `bcoo_A` from a reduced high-water mark is a footprint cleanup, **not a speed step** (case2 peaks at 1968 MiB of 8192; s13's fitted grid already removes the idle SpMV blocks, and would still be needed at a 1.5x reduced reservation) | memory only; `run_benchmark.py`'s `gpuMemory.peakTotalMiB` |
| **rwb has no duplicate triplets: the whole sort + RLE + reduce there is a permutation of 5 740 entries** | `verify/new_vs_old_rwb.txt`: in = out = 5 740, 179 of 180 warps skip the tree. The chain is ~0.3 % of rwb, so this is a note, not a target — unless a future ABD scene is large | — |
| ~~the reduce's warp tree~~ | **done — this step.** Ceiling for anything further on the tree: 5.9 % of the kernel (Probe=1) | — |

### Found outside this step's area (reported, not fixed)

- **The brief's `matrix_converter.inl:485` reading is a resize, not a scratch use** (above); s13's candidate 4
  should be re-worded before anyone builds "its own sort scratch" — the converter already has one.
- **`FastSegmentalReduce`'s ragged tail does one pointless atomic per launch** whenever `in_size % 32 != 0`
  (the s32 comment already records it); harmless, unchanged here, and the k2 kernel reproduces it exactly
  because it must.
- `clang-format` still not installed on this box (s04–s16 recorded the same).

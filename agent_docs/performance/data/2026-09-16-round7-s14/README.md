# s14 — sort (key, index) in the gradient-doublet convert and fold the segment gather into the 3x1 reduce

Branch `perf/round7-s14-doublet` from main `59174d75`. One engine file changed:
`src/backends/cuda/algorithm/details/matrix_converter.inl` (s13's candidate #2).

## The premise, verified first (fresh capture at main head)

The gradient-doublet chain (`convert(DeviceDoubletVector, DeviceBCOOVector)`, called once
per Newton iteration by the dytopo effect manager — the only doublet call site in the
engine), full 130-frame crease-press nsys (`nsys/main_head_*.csv`, total 34.44 s GPU):

| kernel | launches | per-launch | total | share |
|---|---:|---:|---:|---:|
| CUB onesweep `<int, Matrix<double,3,1>>` (2 passes/convert) | 1148 | 174.7 µs | 200.6 ms | 0.58 % |
| + histogram / exclusive-sum | 574 | 14.4 / 1.6 µs | 9.2 ms | 0.03 % |
| run-length-encode by key | 575 | 17.4 µs | 10.0 ms | 0.03 % |
| 3x1 reduce `fast_segmental_reduce_matrix_k2_kernel<64,32,double,3,1>` (staged read) | 575 | 237.8 µs | 136.7 ms | 0.40 % |
| + zero-cross fill, uniquify k1s | 575 | — | 6.9 ms | 0.02 % |

Bytes: the onesweep moves 56 B/elem/pass (4 B key r+w + 24 B payload r+w) over ~1.25 M
doublets = 409 GB/s effective = **91 % of the 448 GB/s DRAM roof** — the pass is
payload-width-bound, NOT the regime s11's key-narrowing rejection covered (that was a
4 B payload next to a 4 B key). Folding prices at ~−143 ms ≈ −0.42 % of scene GPU
(payload 24→4 B per pass, plus the iota and the reduce's sequential→random read swap).

Other convert()-family staged sorts, enumerated and priced:
`active_set_manager` and `abd_linear_subsystem` SortPairs carry int payloads (s11's wash
class); the BVH morton sorts carry int payloads; the triplet `convert()` and
`convert_sym` sites are already index-only (s13 / round-4 s10); `sym2ge`'s in-place
re-sort already sorts (key, int) inside CUB and its k3 gather IS the output (nothing
follows it that could absorb the gather; in-place semantics require the scratch). **The
doublet sort was the only wide-payload staged sort left in the family.**

Scene coverage: the chain launches on crease-press (575/run) and cube-wall-cloth (361/run,
`nsys/cwc_new_*.csv`), but **NOT on mas-bunny** (s13's capture: no `<int, Matrix3x1>`
onesweep, no `k2<64,32,...,3,1>` — only convert_sym's fused triplet path runs there);
mas-bunny is therefore the null control for this change.

## What shipped

`_radix_sort_indices_and_segments` sorts `(key, iota)` pairs — a new identity k1 writes
the iota, the CUB sort carries 4 B/pair — and `_make_unique_segment_warp_reduction`'s
reduce gathers the segments through the permutation
(`matrix_converter_permuted_segment_value_op`, the s13 fold on the
`FastSegmentalReduce<64,32>` 3x1 instantiation). Both arms run the same stable CUB sort
over the same keys, so the payload at sorted position j is the same source slot the
wide-payload sort would have staged: same bytes in the same slots of the same summation
tree. Knob **`UIPC_DOUBLET_UNSTAGE`** (default on = folded; `=0` = wide-payload sort +
staged read = main's path). Diagnostic **`UIPC_DOUBLET_VERIFY=1`** re-runs the whole
staged chain into scratch after every production launch and counts mismatching words
with the ≥ 3-warp atomic-arrival split (s13's instrument; a dedicated compare kernel
because a doublet segment is N words, not N*N).

## Numerics — bit-identical for every order-determined slot

The numerics class: every output slot whose segment sits inside one warp is
order-determined (same warp tree over the same operands in the same order) and must
match bit for bit; slots whose segment spans ≥ 3 warps accumulate ≥ 3 atomic operands
in arrival order in BOTH arms — the matrix's pre-existing cross-warp atomic
nondeterminism (s17's class). Proof on device, full 130-frame crease-press runs:

| arm | launches | 64-bit words | mismatching | in ≥ 3-warp class | elsewhere |
|---|---:|---:|---:|---:|---:|
| folded vs staged, run 1 | 564 | 18,084,690 | 3,071,051 (16.98 %) | 3,071,051 | **0** |
| folded vs staged, run 2 | 555 | 17,836,842 | 2,989,735 (16.76 %) | 2,989,735 | **0** |
| control: staged vs staged | 558 | 18,043,260 | 3,017,454 (16.72 %) | 3,017,454 | **0** |

The in-class rate is far higher than s13's triplet (2.27 %) because gradient-doublet
segments are long (several reporters write the same vertex index), and the control shows
the OLD path's own arrival-order noise produces the same class at the same rate — the
folded arm adds nothing outside it. maxrel 1.4e-9 (folded) vs 4.4e-9 (control), the same
tail of the same distribution.

## Gates

- Correctness: gate identical to `baseline_tests.txt` at **default, `UIPC_DOUBLET_UNSTAGE=0`,
  and after clang-format** (pytest's own duration string is the only diff).
- Binary identity (canonicalised per-function SASS, s13's tool): main → branch, **1624
  head functions, 1616 identical, 8 differ by ±8 instructions of 16-24 k** (the documented
  TU ripple in TUs transitively including the changed header), **0 missing** (the old
  wide-payload CUB instantiations remain for the rollback arm), **49 new** = the iota k1
  TU clones, the folded 3x1 k2 reduce clones, the verify-compare clones, and the CUB
  `<int,int>` radix-sort policy family (onesweep/histogram/exclusivesum/upsweep/downsweep/
  scanbins/singletile) — the payload-width change moves the sort between template
  instantiations exactly as the s11 lesson warned; kern_sum separates them by full
  template name and no `<int,int>` policy existed at main, so attribution is clean.
  Resource usage: folded 3x1 k2 **32/24 REG, 0 stack** = the staged arm's figures (same
  granule, launch geometry unchanged); iota k1 REG:8; the 3x3 k2 classes unchanged
  (48/35/36). clang-format codegen-neutral (pre/post-format diff: same 7-function ripple,
  0 added/removed). The full ~590 MB SASS dumps stay outside the repo in
  `/workspace/output/round7/s14/` (`sass_branch.txt`, `sass_branch_fmt.txt`, `sass_main.txt`).

## Performance

- **Scope** (full 130-frame nsys, env A/B in one build, ABBA ×2, fresh prefixes,
  `--cuda-graph-trace=node`; per-kernel with in-run controls):
  **onesweep 165.4/172.2 → 74.3/72.7 µs per launch (−55.1 %/−57.8 %)**; the iota k1
  costs 7.4-7.5 µs, the histogram +1.5 µs; **the 3x1 reduce is flat** (234.7/229.8
  folded vs 224.5/233.8 staged — the random gather is free once the staging write stops
  thrashing L2, s13's effect again); RLE/zero-cross/uniquify flat. **Chain total −112.1
  and −111.9 ms per run in the two rounds (−0.35 %/−0.37 % of scene GPU kernel time),
  −29 % of the chain per Newton iteration** (607.8 → 430.8 µs/newton at the means).
  Controls flat: SpMV 64.6→65.0 / 66.2→65.3 µs. Newton 577/569 vs 533/565 across
  captures = the scene's documented chaos; the per-Newton figures are the instrument.
  The old `<int, Matrix3x1>` rows: 0 launches in the new arm.
- **End-to-end crease-press** (n=8/arm ABBA): **meanFrameMs 234.10 → 244.06 ms
  (+4.25 %, p=0.191, OVERLAPPING — below the 5.4 % MDE)**; ms/newton +4.64 % (p=0.087).
  The count guard fired (PCG +10.83 % against the wall, in-arm spread 96-145 k on both
  arms = the documented frame-0 chaos); since no computed quantity can differ outside
  the both-arms-nondeterministic atomic class (above), the drift is the scene's own
  non-determinism and the wall number is unreadable — the claim rests on the scope gate,
  as it did for s13.
- **mas-bunny** (n=5/arm, null — the chain does not launch there): +0.09 % mean
  (p=0.48, overlapping), **Newton exactly 465 in all 10 runs**, PCG −0.06 %.
- **cube-wall-cloth** (chain launches, 361 converts): n=4 read was guard-noisy
  (+1.97 % mean with +1.63 % count drift); extended to **n=5/arm: mean −0.37 %
  (p=0.79), ms/newton −0.20 % (p=0.74), Newton/PCG counts flat** (−0.16 %/+0.08 %) —
  no regression, no resolvable win (the chain is ~1 % of cwc GPU).

## Transfer (recorded as a prediction, no acceptance box this round)

**Algorithmic** — the bytes moved through a bandwidth-bound sort drop 3.5× per pair
(56 → 16 B/elem/pass) at unchanged problem size, launch count and geometry; the
machine-dependent inputs are the sort's effective bandwidth (91 % of roof here) and the
random-vs-sequential read rates of the reduce, which measured flat. Holds a fortiori on
parts with more bandwidth or larger L2. Prediction for a rented re-measure
(5090-class): **onesweep −50..−60 % per launch survives; the chain-level −25..−30 % per
Newton survives; scene-level −0.3..−0.45 % of GPU kernel time on crease-press-class
mixes**. Check per-launch figures by full template name (the instantiation moved from
`<int, Matrix<double,3,1>>` to `<int, int>`), not scene totals.

## Files

| file | what |
|---|---|
| `nsys/main_head_*` | fresh main-head capture behind the premise table |
| `verify_fullrun_r{1,2}.log`, `verify_fullrun_control.log` | the two both-paths full-run probes + the staged-vs-staged control |
| `scope_analysis.txt` + `nsys/scope_{old,new}_r{1,2}_*` | the scope A/B (ABBA ×2) |
| `nsys/cwc_new_*` | proof the chain (folded kernels) launches on cube-wall-cloth |
| `gate_default.txt`, `gate_knob0.txt`, `gate_default_postfmt.txt` | correctness gates, identical counts to `baseline_tests.txt` |
| `ab_cp.txt` + `ab_cp_summary.json` | crease-press end-to-end A/B (n=8/arm) |
| `s14_bunny_*.json`, `s14_cwc_*.json`, `s14_cwc5_*.json` | regression A/Bs |
| `sass_diff_main_vs_branch.txt`, `sass_diff_prefmt_vs_fmt.txt` | per-function SASS identity |
| `so_sha_start.txt` | main-head binary hash at step start |

## Candidates found while working (for s15+)

1. **The FP64 warp tree is now the doublet chain's binding cost too** — after this fold
   the chain's biggest kernel is the 3x1 reduce (229-235 µs, 0.40 % of scene GPU,
   tree-bound like its siblings). The three k2 classes (3x3 folded 945 µs + 3x3 GLS
   367 µs + 3x1 doublet 230 µs ≈ 1.54 ms/newton ≈ 3 % of scene GPU, tree 43-58 %)
   make s13's standing candidate #1 (~1.5 % ceiling) stronger: one re-summation
   mechanism now pays in three places.
2. The `segments_sorted` member (a ~30 MB capacity buffer) is still allocated in the
   folded arm for the rollback/verify paths — a memory-only note, no kernels touch it
   at default.
3. mas-bunny's wall does not track its GPU time (s13's note, unchanged); cwc is now
   also shown to absorb a ~0.4 %-of-GPU change without a readable wall move.

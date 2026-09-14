# round 6 / s09 — the K9 contact stream split, re-measured

Head `53aa9712`, branch `perf/round6-s09-stream-split`, RTX 2070 SUPER (cc 7.5, 40 SMs).
**No engine behaviour changed.** The step measures the pre-existing `UIPC_CONTACT_SPLIT` switch in
all three of its arms and the only source edit is the comment above `m_split`.

| arm | `UIPC_CONTACT_SPLIT` | what it does |
|---|---|---|
| `a2` | 2 (**shipped**) | part 1 (PT+EE) on a side stream, part 2 (PE+PP) on the default stream |
| `a1` | 1 | the same two launches, both on the default stream (serial) |
| `a0` | 0 | one fused launch over all four pair types (the pre-K9 code) |
| `null` | `02` | `std::atoi` reads it as 2 — bit-identical to `a2`, so its delta is the scene's own envelope |

## Claim -> file

| claim | file |
|---|---|
| the predictions, written before any arm ran | `predictions.txt` |
| the critical-path conversion model, fixed on rwb+cwc before case2/tumbler were read | `predictions2.txt` |
| **part 2 is 92–98 % hidden**; union/max; the serial null has 0 % overlap | `overlap_all.txt` (from `overlap.py`) |
| contact-assembly critical path per assemble call, all three arms, four scenes | `critpath.txt` (from `critpath.py`) |
| per-scene kernel sums, wall, sum/wall, per-launch µs, all three arms | `scope_{rwb,cwc,c2,tum,mb}.txt` (from `analyze.py`) |
| the tumbler/rwb normalisation against untouched kernels of other families | `scope_norm_{rwb,tum}.txt` (from `normalize.py`) |
| **8 of 32 warp slots**: block size, registers, blocks/SM, occupancy | `occupancy.txt` |
| launch geometry and pair populations per scene | `pairs_and_grids.txt` |
| what runs next to part 1, and what share of its window is shared | `neighbours_rwb_a2.txt` (from `neighbours.py`) |
| **36.7 % of rwb's kernel time runs on < 28 of 40 SMs** | `grid_census_rwb.txt` |
| the kernel-duration SUM is 1.060x the UNION in a2 and 1.000x in a1/a0 | `busy_union.txt` |
| the fused kernel makes a PE/PP thread carry part 1's frame (2 919 vs 413 local stores) | `sass_parts.txt` |
| registers per instantiation (part 1 255, part 2 154, fused 255) | `res_usage_parts.txt` |
| env-switch audit at kernel level (which instantiation each arm launches) | `default_selects.txt`, and the stream ids in `overlap_all.txt` |
| end-to-end A/B, 4 arms interleaved, n=20/20/7 | `ab/*_summary.json`, `ab/*.log` |
| tumbler `--verify`, 180 frames, n=6 x 3 arms | `verify_stats.txt`, `verify/verify_a*_r*.txt` |
| `gate.sh` in all three arms vs `baseline_tests.txt` | `gate_a2.txt`, `gate_a1.txt`, `gate_a0.txt` |

## Method notes

- **`cuda_gpu_trace` is used only for ratios measured inside a single run** (overlap fraction,
  union/max, sum/union, co-residency, launch geometry). Cross-arm *levels* from trace runs disagree
  with the n=3 kernel sums by up to 6 % on rigid-wrecking-balls; every cross-arm number in the record
  comes from the kernel sums (n=3 full runs per arm) or the end-to-end A/B.
- rigid-wrecking-balls (120 frames, 71 MB csv) and cube-wall-cloth (100 frames, 134 MB) were traced
  in **full**. stiff-gipc-case2 and tumbler-garments were traced over a **60-frame window** because
  their full-run csv is 250–400 MB; both windows contain 329 and 470 contact-assembly launch pairs
  (s03's "a short cube-wall-cloth window has zero contact-assembly launches" trap was checked, not
  assumed) and the window's part-1 per-launch time is within 1.1 % (case2) / 9.5 % (tumbler) of the
  full-run kernel sum.
- The raw `cuda_gpu_trace` csvs (615 MB) stay in `/workspace/output/round6/s09/trace/`; only the
  reductions are committed. `tracerun.sh` regenerates them.
- `mas-bunny` was measured and is **a vacuous control for this subsystem**: 21 contact-assembly
  launches per run for 0.15 % of its GPU kernel time (`scope_mb.txt`).

# round 7 / s18 — fixed-slot (capacity-based) staging of `triplet_A`: PRICED AND REJECTED (nothing ships)

Every file here backs the s18 section of
`agent_docs/performance/2026-09-15-perf-round7.md`. The engine tree on the
branch is **identical to main** (`4d510ac7`, binary sha256 `e708bafe83aab9f0…`,
the s17 head) — this step changed no engine source. It verified s09's recorded
design against the code, enumerated the consumers of the exact counts, and
measured the prize the design could buy on two multi-receiver scenes. The
measurement refutes the pick's magnitude: the ceiling is **5–8x below the
round's 1 % wiring bar** on the target scene, and the code reading found the
exact-count dependency deeper than s09 sketched. The s06/s09/s16 precedent
applies — a priced rejection.

## What the candidate was (s09's design, s10 pick #3)

Give each subsystem a **capacity-based fixed slot** of `triplet_A` (offsets =
prefix sums of capacities, known at arm time), launch the −1 index fill and the
count-independent G/H (FEM kinetic + elastic reporters, ABD body-local) **at arm
time** — before `compute_dytopo_effect()` parks the host in `_distribute`'s
D2H — and let the converter's existing `row >= 0` filter
(`matrix_converter.inl`, round-5 s26) drop the unwritten slack slots. The
classified counts would then gate only the late scatter kernels. Expected by
s10: "realistically −0.5..−1.3 % wall" (from the round-6 case2 s14 deferred-join
precedent ~1.3 % ms/Newton).

## The instruments

1. **GPU-idle gap census** (new, `analyze_idle5*.py`): the busy union over ALL
   streams from the nsys gpu trace; every idle gap ≥ 3 µs attributed by the
   kernels that bracket it and by the host CUDA-API state during the gap (was
   the host in a sync? issuing Free/Malloc? launching?). This is the direct
   measurement of what "overlap the post-count work with the drain" can buy:
   the drain is GPU-busy (s09 proved park GPU-idle = 0.00 ms), so the ONLY
   recoverable time is the idle between phases.
2. **s09's park analyzer** (`analyze_api4.py`, reused verbatim): the readback
   park table at the current head — for context and comparability with s09/s10.
3. Captures (fresh prefixes, `--cuda-graph-trace=node`, same window shapes as
   s09): `cp18_head` = crease-press 12 frames (the press phase, 217 Newton
   iterations, 12.18 s span); `cwc60_head` = cube-wall-cloth 60 frames (the
   contact phase, 2.26 s steady span).

## The pricing (the rejection's evidence)

**Parks at the s18 head (cp window, 4395 readbacks / 12.18 s):**
`matrix_converter` 2763.7 ms = 22.69 % of span (all drain, GPU-idle 0.00 ms),
`other` (PCG + the rest) 3086.7 ms = 25.35 %, `_dist_grad` 45.7 ms = 0.37 %,
`_dist_hess` 4.9 ms = 0.04 % — the s09 structure, minus what s13/s14 (cheaper
sorts) and s17 (PCG poll) took out.

**Total GPU idle (steady span, 3 s skip on cp):**

| scene | steady span | total idle | <10 µs (launch floor) | 10–100 µs | >100 µs |
|---|---|---|---|---|---|
| crease-press | 9.22 s | 259.1 ms (2.81 %) | 205.7 ms | 25.7 ms | ~26 ms |
| cube-wall-cloth | 2.26 s | 258.6 ms (11.43 %) | 139.9 ms | 32.3 ms | 80.0 ms |

**The fixed-slot target region** (idle gaps whose NEXT kernel is the classify /
pairs / fill / assembly-G/H chain — what early enqueue would remove):

| scene | generous (incl. the contact-fork walk) | strict (fixed-slot only) | per Newton |
|---|---|---|---|
| crease-press | 16.7 ms = **0.18 %** of wall | 11.4 ms = **0.12 %** | 77 / 53 µs |
| cube-wall-cloth | 9.0 ms = 0.40 % | ≈1.65 ms = **0.07 %** | — |

(The generous cp bound counts `[CUDA memcpy D2H] -> do_assemble_kernel`
(5.3 ms) — that is the **contact side-stream fork's** host walk after the
trajectory-filter count readback, not the triplet_A region; on cwc it is the
single biggest idle class (7.35 ms in ≥100 µs gaps) and is a *different*
candidate — see the record.) The FEM/ABD G/H kernels themselves are NOT
preceded by idle worth the name: `next==GH_dahl` sums to 0.53 ms over the whole
cp window — they queue behind other GPU work today.

Full-run extrapolation (cp): 77 µs × 566 Newtons ≈ 44 ms ≈ **0.14 % of the
31 s wall**. Against the 1 % wiring bar: rejected by 5–8x. The s10 estimate
(−0.5..−1.3 %) does not survive contact with the timeline: it was extrapolated
from the case2 s14 precedent and the "host parked 24.8 %" number, but that park
is drain (GPU busy sorting — s09's own finding), and s17's doorbell has since
removed the adjacent PCG-readback bubble.

**The partial version (defer only the −1 fill, keep exact sizing): worth
~nothing.** The fill is GPU work that queues behind the host walk either way;
moving its *launch* to arm time removes one launch (~5 µs) from the walk and
lets the fill execute during the drain (which is GPU-busy — no wall change).
The walk itself — extents, resizes, the launches — is unchanged. Priced at
0.02–0.05 %, i.e. not wireable on its own.

## The code-reading findings (why the blast radius is worse than sketched)

1. **ABD's extent is not a host-side read**: `ABDLinearSubsystem::report_extent`
   calls `_prepare_dytopo_pairs()` — a device radix sort + scan + its own
   blocking `host_read(&P)` — so the ABD triplet count P (body-pair-reduced) is
   produced by device work *inside the extent phase*. Its only host-known
   upper bound is the raw contact triplet count C, typically orders of
   magnitude bigger than P. Capacity staging for ABD therefore needs
   last-count-based growth **plus an overflow fallback that redoes the
   assembly** (the early-launched writers wrote the stale layout).
2. The coupling off-diag subsystem (`abd_fem_linear_subsystem.cu`) reads the
   classified counts for both its lr/rl regions — same dependency.
3. `empty_system` (`total_triplet == 0` → skip solve) would need restructuring
   to the converter's `m` readback; the sim_case suite exercises empty-early
   frames (cwc's pre-contact free fall).
4. The whole `subview(offset, count)` extent contract (FEM's
   `hess_offset == info.hessians().triplet_count()` assert, the reporters'
   subviews, the DoF/probe logging) assumes the parts sum exactly to the view;
   capacity staging splits "written count" from "view size" in every subsystem.
5. Capacity is also a **cost**: the fill + flag + scan + compact passes run
   over capacity (~50 B/slot of DRAM traffic). A bound-based capacity (2–3x
   the exact count) would *add* ~0.5–1 % of scene GPU time — more than the
   prize; only a tight last-count capacity keeps it negligible, at the price of
   the fallback machinery in (1).

## The invariant (for any future implementation)

The converter must receive **the same multiset of written (row, col, value)
slots in the same relative order**: subsystem order preserved by capacity-based
prefix offsets in subsystem-index order, within-subsystem order unchanged,
slack slots −1/−1 dropped by the `row >= 0` filter *before* the stable sort.
Then `m`, `h_count`, the segmental reduce and the BCOO output are bit-identical
outside the ≥3-warp atomic-arrival class (s13/s14's verifier split). This was
verified against `convert_sym`'s code path (flag k1 → scan → compact k2 →
stable SortPairs → RLE → reduce); it is documented, not implemented — nothing
shipped to prove on device.

## Side findings (measured, outside this step's scope)

- **The contact-fork host walk on cwc**: `[D2H] → do_assemble_kernel` gaps of
  1–3.7 ms (host in StreamSynchronize + EventRecord/StreamWaitEvent + Free/
  Malloc chains), 7.35 ms in the ≥100 µs class alone of a 2.26 s window. The
  biggest single addressable walk found on that scene; belongs to the contact
  manager (round-6 s10/s11 K9 fork territory), not to the linear system.
- **The realloc cascade**: `reserve_ratio = 1.1` growth makes buffer growth
  frequent (858 cudaFree + 1282 cudaMalloc in the 12-frame cp capture ≈ 14
  frees/frame); each growth event idles the GPU behind cudaFree+cudaMalloc for
  0.1–3 ms. Mid-run steady cost ≈ 4–6 ms per cp run ≈ 0.05 % (the rest is the
  first-contact ramp, where 3–6 reallocs fire in one iteration, and teardown).
  A bigger ratio or a high-water-mark policy is a one-line change — below the
  bar alone, noted so nobody re-derives it.
- mas-bunny is **unaffected by construction**: single diag receiver owning the
  full dynamic range takes the fast path (`global_dytopo_effect_manager.cu`
  `compute_dytopo_effect`, receivers.size() == 1 branch) — no classify, no
  `_distribute` parks, no count dependency before the converter's own.

## Files

| file | claim it backs |
|---|---|
| `analyze_idle5.py` / `analyze_idle5b.py` / `analyze_idle5c.py` | the idle-census instruments (v5 class census / steady-window + size histogram + top gaps with host state / realloc class + target-region totals) |
| `apitrace.sh` | the capture script (s09's shape: `--cuda-graph-trace=node`, fresh prefixes) |
| `cp18_park_analysis.txt` | the park table at head (s09's analyzer on the fresh cp capture) |
| `cp18_idle_analysis{,_b,_c}.txt` | cp idle census: class table, steady histogram + top gaps, realloc + target-region totals |
| `cwc60_idle_analysis{_b,_c}.txt` | the same for the cwc contact window |
| `*_cuda_gpu_kern_sum.csv`, `*_cuda_api_sum.csv` | the capture summaries (launch counts, marker kernel names) |
| `*.run.log` | the captured runs completed (cp: 12/12 frames converged, 217 Newtons; cwc: 60 frames) |
| `gate_head.txt` | gate at head identical to `baseline_tests.txt` (pytest duration string only) — the box health check |

Not archived: the 212 MB + 78 MB per-launch gpu traces and the sqlite exports
(the analysis txts carry every number quoted in the record; the scripts
regenerate them from a fresh capture).

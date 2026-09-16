# Round-7 s19 — the plastic kernels' occupancy restructure (two hinges per thread): PRICED AND REJECTED

The last open lever s16 left for the plastic bending family ("the kernels'
register/occupancy budget ... a `__launch_bounds__`/two-hinges-per-thread
restructure is the one untried in-kernel lever, launch-geometry class,
ceiling ~−15..−25 % if the latency half folds"). **Nothing ships; the tree is
identical to main** (`git diff 65a4eb1a -- src/` empty; gate = baseline). The
implementation was built, proved bit-identical, measured **+21..+23 % per
launch SLOWER**, and reverted — preserved verbatim as `implementation.patch`
(applies cleanly to main).

## The fresh floor read (this step's diagnosis, at head 65a4eb1a)

1. **Launch geometry, measured** (20-frame in-trace capture, `head_geom*`):
   both plastic G/H kernels run `<<<38, 256>>>` at 255 registers — 9 506
   hinges = 37.13 blocks of 256 → 38 blocks on 40 SMs, **one block per SM,
   8 warps = 256 threads per busy SM = the machine's full register-limited
   occupancy** (65 536 regs/SM ÷ 256/thread), 2 SMs idle.
2. **The grid, not the register file, binds concurrency.** The scene supplies
   9 506 hinges ≈ 93 % of the machine's 40 × 256 = 10 240 thread capacity.
   *This kills the `__launch_bounds__` twins by arithmetic before any code is
   written*: a (128, 3)/(128, 4) twin raises the SM's resident-*capacity* to
   384/512 threads, but the grid still supplies ≤ 38–75 blocks and the work
   distributor spreads them one-to-two per SM — no additional warp can become
   resident because there are no more hinges. The register cap would buy only
   spill (round-6 s01's +12 % on the plain hinge is the same conclusion,
   measured). Not built.
3. **SASS at head = s16's binary exactly** (whole-.so function-body multiset,
   `binary_diff.txt`): strain `<1,1,2>` 11 112 instructions, 2 897 DFMA /
   884 DMUL / 302 DADD / 138 MUFU / 802 LDL + 1 042 STL; the QL core runs at
   52 % of its FP64-issue floor — half issue, half serial-chain latency +
   local-memory round trips of the dynamically indexed d/e/V arrays.
4. **Chain accounting for the two-hinges lever**: today 256 independent QL
   chains per SM (256 threads × 1). Two hinges per thread at block 128 →
   38 blocks × 128 threads × 2 = **the same 256 chains per SM** — the lever
   cannot add concurrency either; its only mechanisms are second-order
   (intra-warp interleave of the two chains' long-latency ops without a warp
   switch, vs. the doubled code size, the fused-loop pair divergence, and the
   strictly larger per-thread local-memory footprint). Pre-registered
   expectation: ≈ 0 ± second-order, sign decided only by the measurement —
   which is why it was built: it was the one mechanism with any sign-positive
   pathway left.

## What was built (the rejected arm, in `implementation.patch`)

- `evd_tridiag_ql_pair<T, N>` in `cuda_tool/eigen/evd.h`: both matrices'
  Householder reductions in shared loop nests, and the serial tql2 do-while
  **fused into one loop** that advances whichever chain is still unconverged;
  the rotation loop descends from `max(active mA, active mB) − 1` with each
  chain's step guarded by its activity flag and `i < m`. Every per-matrix
  operation runs with the same expressions in the same order as
  `evd_tridiag_ql`.
- `make_spd_translation_free_4x3_blocked_pair<Solver, SymAsm>` in
  `utils/make_spd.h`: the s08 assembly/back-assembly with both matrices'
  statements in the same unrolled nests + the pair QL (non-default
  Solver/SymAsm fall back to two sequential single calls).
- `..._gradient_hessian_pair_kernel<1, 1, 2>` in both plastic `.cu` files +
  dispatch: knob **`UIPC_PDSB_OCC`** (default on = pair kernel on the default
  projection arm; `=0` = today's single-hinge kernel), probe
  **`UIPC_PDSB_OCC_VERIFY=1`** (SpreadVerifier: single `<1,1,2>` first,
  snapshot G doublets + H triplets, pair kernel over the same inputs, count
  mismatching 32-bit words). Pairing is strided (`i0`, `i0 + ⌈n/2⌉`), block
  dim fixed at 128 so ⌈n/2⌉/128 = ⌈n/256⌉ blocks keep the one-block-per-SM
  fill (the occupancy API would answer 256 and leave half the SMs empty).
- Binary: pair kernels 255 reg / 6 912 B (strain) and 255 / 7 024 (stress)
  stack vs the singles' 255/2 912; 24 104 / 25 192 instructions vs
  11 112 / 11 632. Whole-.so: 2 947 of 2 955 bodies identical, **2 new**
  (the pair kernels), ~8 ripple-changed bodies on non-default Eigen arms
  (±8 instructions, REG/STACK unchanged — the documented TU-ripple class);
  every knob-reachable default path and the `UIPC_PDSB_OCC=0` rollback
  byte-identical (`binary_diff.txt`).

## The measurement (all full 130-frame crease-press runs, ABBA, fresh prefixes)

| arm | strain µs/launch | stress µs/launch | dahl (ctl) | NHS2D (ctl) | SpMV (ctl) |
|---|---|---|---|---|---|
| single (old) | 1 186.1 | 1 205.5 | 458.1 | 1 342.4 | 65.76 |
| pair (new) | **1 457.5** | **1 460.8** | 452.7 | 1 336.9 | 65.89 |
| Δ | **+22.88 %** | **+21.18 %** | −1.2 % | −0.4 % | +0.2 % |

n=2/arm (effect 4× the MDE-class; controls flat inside the ±1.2 % same-binary
drift). Per hinge: 124.8 → 153.3 ns (strain), 126.8 → 153.7 (stress).
`scope_analysis.txt` has the script; `scripts/scope_ab.sh` the capture.

**Why it is slower — the mechanism, not a guess**: the two chains' QL working
sets (V 9×9 + H12 + assembly temporaries) are per-thread *local-memory*
arrays, so pairing doubles the per-thread stack (2 912 → 6 912/7 024 B) while
the thread count halves: the concurrently-live local footprint per SM grows
256 × 2 912 = 745 KB → 128 × 6 912 = 885 KB (+19 %) — the measured +21..23 %
tracks it. The intra-warp interleave bought nothing visible: at 8 warps the
SM already had 256 independent chains to hide the QL's latency behind; the
second chain per thread replaced warp-level hiding capacity one-for-one
(the chain accounting above) and only added cache pressure and code size.
The QL's latency half is bounded by per-thread local-memory footprint, not by
missing concurrency.

## Gates

- Correctness: gate identical to `baseline_tests.txt` at **default, at
  `UIPC_PDSB_OCC=0`, at all-plastic-knobs-off** (on the implementation), and
  at the reverted head — pytest's own duration string only, every time.
- **Numerics: bit-identical, proved on device over full runs**
  (`UIPC_PDSB_OCC_VERIFY=1`, 130 frames): strain G 141 335 208 words,
  H 1 009 537 200 words; stress the same — **0 mismatching** (2.30 G 32-bit
  words combined; values and row/col indices). The rejection is performance,
  not brokenness: the fused-loop pairing reproduced every output bit.
- End-to-end (the rejection's cost, ab.py interleaved, pooled n=10/arm from
  two ABBA batches): **meanFrameMs 232.05 → 236.25 ms (+1.81 %, p=0.47,
  OVERLAPPING — wrong direction, below the MDE as the scope predicts for a
  +1 %-of-GPU cost)**; ms/newton +1.48 % (p=0.44); counts flat (Newton
  +0.32 % p=0.83, PCG +1.28 % p=0.79, LS +0.79 % — the guard does NOT fire at
  n=10; the top-up batch alone showed an n=2 disjoint Newton draw that the
  pooled n resolves, the scene's documented non-determinism). `ab_pooled.txt`.
- Regressions (cwc/tumbler/mas-bunny): vacuous — nothing ships; the reverted
  binary's whole-.so REG/STACK multiset is identical to main's
  (`reverted_so_sha.txt`), so every scene runs the exact main-head code.

## Transfer (recorded as a prediction, no acceptance box this round)

Nothing ships, so there is no speed prediction to falsify. The transferable
content is the arithmetic, which should hold on any part:

1. **Price occupancy levers by min(grid capacity, register capacity), never
   by either alone.** On a part with more SMs than the scene has elements
   (or a bigger register file), the same kernel's occupancy levers change
   sign — this rejection is scene-size-specific in its *arithmetic* (the
   +21 % footprint cost is not). A rented re-measure of any future occupancy
   candidate on a 5090-class part must first check
   `n_elements ≥ threads/SM × SM_count`; if the grid fills the machine,
   launch-geometry levers are dead there too.
2. **ILP-via-multiplicity is not free when the working set lives in local
   memory**: k chains per thread multiply the per-thread stack, and the
   per-SM local footprint grows by the same factor divided by the occupancy
   loss. It pays only where the footprint shrinks or stays flat (register-
   resident working sets).
3. The bit-identity proof strategy (fused two-matrix QL state machine, both-
  paths SpreadVerifier probe) is reusable verbatim for any future pairing
  experiment — 0 mismatching words over full runs at the cost of one extra
  launch per iteration.

## Files

| file | what it is |
|---|---|
| `implementation.patch` | the complete rejected arm (4 files, 1 025 lines), applies cleanly to main |
| `scope_analysis.txt` / `scripts/scope_ab.sh` / `scripts/analyze_scope.py` | the kernel-scope A/B (ABBA ×2 full runs, in-trace per-launch means + controls) |
| `scope_r{1,2}_{new,old}_cuda_gpu_kern_sum.csv` (in the round scratch) + `head_geom*` | raw nsys exports incl. the launch-geometry read |
| `binary_diff.txt` | whole-.so canonicalized SASS diff main vs implementation |
| `res_usage_branch.txt` | res-usage of the implementation (pair kernels 255/6912, 255/7024) |
| `occverify_fullrun.txt` | the full-run bit-identity probe (2.30 G words, 0 mismatching) |
| `gate_branch_{default,occ0,alloff}.txt`, `gate_reverted_head.txt` | the four gate runs |
| `ab_pooled.txt` / `scripts/geometry_capture.sh` | end-to-end pooled stats / the geometry capture script |
| `reverted_so_sha.txt` | the reverted engine binary sha + the REG/STACK-multiset check basis |

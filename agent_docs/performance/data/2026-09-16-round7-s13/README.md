# s13 — close round-6 s17's gather question; fold the staged k3 pass into the segmental reduce

Branch `perf/round7-s13-segred` from main `cf2db3d3`. One engine file changed:
`src/backends/cuda/algorithm/details/matrix_converter.inl`.

## The question this step was assigned

Round-6 s17 left the converter's segmented reduce "gather-bound" with a
"862 -> ~500 µs DRAM floor" gap attributed to the random 72-byte block gathers
through the sort permutation, and listed sorted-order staging / an RCM-like DOF
reorder as candidates. Round-7 s10 re-flagged the family at 2.9 % of crease-press
GPU time with those candidates still open.

## The answer, measured before building anything

1. **The locality gap is gone on this workload.** With `UIPC_SEG_PROBE=1`
   (round-6's shipped stubs) the only random-gather reduce (`convert_sym`'s
   `permuted_value_op`, in ~750-800 k) runs its no-tree floor at **346 GB/s =
   77 % of the 448 GB/s DRAM roof** — the gather is already near-stream
   (crease-press's permutation is local: mesh-ordered assembly vs row-major sort
   keys). The locality prize ceiling is `P0 − P2` = **42 µs/launch = 0.08 % of
   scene GPU kernel time**. Sorted-order staging and RCM reordering are
   **rejected by arithmetic** — the staging copy alone moves more bytes than the
   randomness wastes. Round-6 s17's open question closes.
2. **The family's real waste was the staging itself, at the call sites round-4
   s10 never reached.** The dytopo effect manager's `convert()`
   (triplet -> BCOO, once per Newton iteration, in up to 3.2 M on crease-press)
   still ran the pre-s10 design: `k3` stages *every* block into sorted order
   (random 72 B read + sequential 72 B write) and the reduce then streams the
   staged copy back. k3 alone was **1175.7 µs x 548 = 644 ms = 2.21 % of scene
   GPU kernel time** — the single largest converter kernel, larger than the
   reduce it feeds.

## What shipped

The k2 reduce gathers through the sort permutation directly
(`matrix_converter_permuted_value_op`, exactly `convert_sym`'s round-4 mode-2
design), k3 disappears from the triplet `convert()` chain, and the staged read
remains as the rollback arm. Knob **`UIPC_SEGRED_UNSTAGE`** (default on = folded;
`=0` = k3 + staged read, byte-for-byte main's path). Diagnostic
`UIPC_SEGRED_VERIFY=1` re-runs the whole staged chain into scratch after every
production launch and counts mismatching 64-bit words with the >= 3-warp
atomic-arrival slot split (s17's instrument).

## Numerics — bit-identical for every order-determined slot

The staged copy `dst[i] = src[perm[i]]` read as `in(i)` and the folded
`src[perm[i]]` are the same bytes in the same slot of the same summation tree;
same-order summation preserved (nothing re-associates). Proof on device, full
130-frame crease-press run, `UIPC_SEGRED_VERIFY=1`:

- **566 launches, 531,233,334 64-bit words compared, 0 mismatching outside the
  >= 3-warp atomic class** (`elsewhere=0` on every launch);
- control — the same probe with production also staged (staged-vs-staged, the
  old path's own arrival-order noise), also a full 130-frame run: **545 launches,
  506,477,826 words, 11,474,090 mismatching (2.27 %), `elsewhere=0`, maxrel
  2.3e-10** — the same class at the same rate as folded-vs-staged (12,208,303 of
  531,233,334 = 2.30 %, maxrel 6.7e-9; the maxrel difference is the tail of the
  same arrival-order distribution over tiny-magnitude entries). The in-class
  words are the matrix's pre-existing cross-warp atomic nondeterminism, present
  in both arms.

## Files

| file | what |
|---|---|
| `probe20_decomposition.txt` | the prod/no-tree/no-gather table + the arithmetic that rejected the locality candidates |
| `hist_cp.txt` | launch shapes (`UIPC_SEG_HIST`): in/out/levels per class |
| `verify_smoke.txt`, `verify_control_staged_vs_staged.txt` | 8-frame probe + its staged-vs-staged control |
| `verify_fullrun_aggregate.txt` | last 20 launches of the full-run probe (aggregate: 531.2 M words, elsewhere=0) |
| `verify_fullrun_control.log` (in `/workspace/output/round7/s13/`) | full-run staged-vs-staged control |
| `gate_default.txt`, `gate_knob0.txt` | correctness gates, identical counts to `baseline_tests.txt` |
| `scope_analysis.txt` + `nsys/scope_{old,new}_r{1,2}_cuda_gpu_kern_sum.csv` | full-run env A/B (ABBA x2) |
| `nsys/main_full_*` | main's baseline capture (the family map at cf2db3d3) |
| `nsys/probe20_*` | the probe capture behind `probe20_decomposition.txt` |
| `nsys/bunny_new_*` | mas-bunny on the branch build: k3 = 0 launches, reduce is the PERM variant |
| `ab_cp.txt`, `ab_bunny.txt`, `ab_cwc.txt` | end-to-end A/B sweeps |
| `sass_diff.py` | s17's per-function SASS identity tool (reused) |

SASS identity (main -> branch, canonicalised per function): 1 612 head functions,
**1 605 identical, 7 differ by +/-8 instructions of 16-24 k** (recompile ripples
in TUs that transitively include the changed header via `fem_linear_subsystem.h`;
five of the seven are non-default template arms), **12 new** = the folded k2
instantiations (dytopo TU + scalar twin in the diff-sim TU), the verify/mark/
compare probes, one div helper. New PERM k2: 48 reg / 0 stack — the same
allocation granule as both existing classes; the old k3 and staged-k2 kernels
remain for the rollback arm.

## Performance

- **Scope (full 130-frame nsys, env A/B, ABBA x2, fresh prefixes, per-launch with
  in-run controls)**: k3 **deleted** (1175.7 µs x 548 -> 0 launches); the folded
  dytopo reduce **987.8 -> 944.7 µs/launch (-4.4 %)** — the random gather at 77 %
  of roof beats the staged read once the staging write stops thrashing L2; GLS
  PERM unchanged (364.9 -> 367.0 µs). **Converter family 4.42/4.30 -> 3.15/3.18 ms
  per Newton iteration = -28.7 % / -26.0 %** (=-1.1..-1.3 ms/iter ~ -2.2 % of
  scene GPU kernel time). Controls flat: SpMV 66.2 -> 66.1 µs, do_assemble +/-3 %
  session noise. Run-level GPU totals are unreadable (PCG/Newton 193 -> 221 ->
  311 -> 216 across the four captures — the scene's documented frame-0 chaos).
- **End-to-end crease-press (n=8/arm ABBA)**: **meanFrameMs 244.32 -> 228.83 ms
  (-6.34 %, p=0.338, OVERLAPPING — below the 5.4 % MDE)**; ms/newton 56.38 ->
  54.17 (-3.92 %, p=0.518). The count guard fired (PCG -6.07 % against the wall,
  in-arm spread 93-217 k = the documented chaos); since no computed quantity can
  differ outside the both-arms-nondeterministic atomic class, the drift is the
  scene's own non-determinism and the wall claim rests on the scope gate. Both
  statistics point the scope's way.
- **mas-bunny (n=5/arm)**: -0.23 % mean (p=0.57, overlapping), **Newton exactly
  465 in all 10 runs**, PCG flat — no regression; the capture proves it executes
  the fold (k3 = 0 launches, 377.0 µs x 465 PERM reduce vs s17's 367.4 staged).
- **cube-wall-cloth (n=4/arm)**: a co-beneficiary — **ms/Newton -2.25 % DISJOINT
  (p=0.0049)**, ms/PCG -2.17 % disjoint, Newton/PCG counts flat (-0.69 %/-0.77 %,
  inside envelope).

## Transfer (recorded as a prediction, no acceptance box this round)

**Algorithmic** — the change deletes a full pass over the block array
(in x 144 B of DRAM traffic per convert: the staged copy's read+write) and one
kernel launch per convert; same values, same launch geometry elsewhere, no
occupancy or grid tuning. Nothing machine-specific enters except the random-vs-
sequential read rates: the fold wins wherever a staged copy (write + read) costs
more than a random read, which held here at 346 vs 364 GB/s and holds a fortiori
on parts with larger L2 (better gather locality). Prediction for a rented
re-measure (5090-class): the family-level **-25..-30 % per convert survives**;
end-to-end scales with the converter family's share of that machine's mix
(crease-press-class ~-1.5..-2.5 % GPU time; check per-launch figures, not scene
totals). The rejected-half's prediction: a locality/RCM candidate prices at
< 0.1 % of scene GPU time on any machine whose gather already runs near roof —
measure `P0 - P2` first, as this step did.

## Candidates found while working (for s14+)

1. **The FP64 warp tree is now the family's binding cost** (43-58 % of the three
   k2 classes after s17's early exit; the probes bracket it exactly). A
   re-summation (K elements per thread with in-thread serial adds) is the only
   big lever left and it changes summation order — rounding-level, full verifier
   burden; s17 declined it at a 5.9 % ceiling on case2, but on THIS scene's
   shapes (BUFM33 at in up to 3.2 M, tree ~604 µs/launch at in=2.0 M) the prize
   is ~1.5 % of scene GPU for half the tree. The >= 3-warp atomic slot class is
   already nondeterministic, which lowers the bar for a segment-serial variant.
2. **The gradient doublet sort carries its staging inside CUB**: the int-keyed
   onesweep (1 084-1 190 launches x ~170 µs = ~190 ms/run) sorts a 24 B payload
   (the 3x1 blocks). Sorting (key, index) pairs instead and folding the gather
   into the 3x1 reduce (the same design as this step) would cut the payload
   traffic ~3x; s11's "payload width doesn't matter" finding was about a 4 B
   payload, not 24 B. Needs its own pricing.
3. **mas-bunny's wall does not track its GPU kernel time** (fold removes ~130 ms
   of GPU work there yet the wall moves -0.23 %): the scene is host-latency-bound
   at the margin — relevant to whoever next prices GPU-side wins on it.

## Post-format validation (the committed tree)

clang-format touched the changed file after the measurement build; the
rebuild's SASS diff against the pre-format build shows the same 7-function
±8-instruction TU-ripple class and 0 added/removed functions. Re-validated on
the exact committed build: `gate_default_postfmt.txt` identical to
`baseline_tests.txt` (duration string only), and `nsys/final_check_*` (full
130-frame run) confirms k3 = 0 launches, the PERM k2 row in-family
(1 197 × 669.5 µs merged both-3x3 average), SpMV control 66.4 µs.

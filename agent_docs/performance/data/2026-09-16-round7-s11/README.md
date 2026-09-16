# s11 — 32-bit radix-sort key: measured and REJECTED (2026-09-16)

Implementation by the s11 worker agent (died mid-step); completion, defect repair,
measurement and verdict by the coordinator. Nothing ships — the tree reverted to
main; the rebuilt binary's cuobjdump figure multiset is exactly the main-arm's.

## The mechanism, measured
- Custom key-path kernels DID get faster per launch (full-run nsys, graph-node
  tracing): hash k1 105.5 -> 78.9 us (-25.2%), compact k2 56.6 -> 48.3 (-14.7%),
  the second hash site 56.4 -> 43.8 (-22.3%).
- The CUB onesweep did NOT: the u64 instantiation (4464 launches, 106.8 us avg)
  disappeared into a u32 bucket whose isolated average for our sorts is ~equal
  or slower (bucket 6164 x 85.6 us minus the pre-existing 1620 x ~17 us small
  sorts). Pass count is end_bit-driven (~31 bits either way) and the 4 B int
  payload dominates tile traffic, so halving the key width buys ~nothing at
  these element counts.
- Sort/converter family total: 1512 -> 1497 ms (~ -1.0 % of a 4.6 % family =
  ~ -0.05 % of scene GPU time).
- End-to-end (ab.py, n=8/arm): meanFrameMs 231.38 -> 238.08 (+2.89 %, p=0.51,
  OVERLAPPING, count guard fired - unreadable wall); ms/newton -0.18 % (p=0.96);
  ms/pcg -0.77 % (p=0.80). Flat.
- Bit-identity WAS proven (the implementation carried its own probe,
  UIPC_CONVERTER_SORT32_VERIFY=1): 0 key mismatches, 0 permutation mismatches
  over 632.6 M elements / 505 converter calls (crease-press), 349.2 M / 886
  (cube-wall-cloth), 380.9 M (mas-bunny).

## The defect the gate caught (process finding)
The agent's restructure DROPPED the output-key resize at the BCOO re-sort site
(matrix_converter.inl _radix_sort_indices_and_blocks): SortPairs was told the
count is whatever the stale buffer held from convert_sym's earlier resize (m,
the upper-triangular count) instead of this site's n -> the re-sort silently
dropped entries -> sim_case 18_abd_fem_contact SIGABRT ("terminate called
without an active exception"), in BOTH knob arms. Coordinator fixed both arms
(loose_resize of ij_hash32/ij_hash before SortPairs at that site) before the
measurement pass; both gates then identical to baseline. The agent died before
running its own gates - the gate is what stood between this and a broken
default-on merge. The implementation patch was not preserved: rejected for zero
effect and superseded by the bug fix; the mechanism is fully described above.

## Verdict
REJECTED. The s10 ceiling (-1.3..-1.8 %) assumed the sort chain is key-traffic
bound; it is not (payload + pass count dominate). Premise correction closes the
candidate. The guard (rows*cols <= 2^32 host arithmetic) was sound and is
recorded here for any future narrower-key attempt.

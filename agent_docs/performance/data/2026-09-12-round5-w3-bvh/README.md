# BVH leaf-predicate reordering — written, never compiled

`bvh_pred_reorder_UNBUILT.py` patches
`src/backends/cuda/collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.cu`
to hoist the register-only rejections of the four leaf predicates ahead of the scattered
contact/subscene id loads and the two mask-tabular lookups.

**It has never been compiled.** The box it was written for was destroyed by the fleet's 300-minute
budget cap before the build started (see the round record, "Two process defects this step ran into").
Treat it as a starting point with a known-good design, not as a tested patch.

What it adds:

- `template <int Order> eval()` per predicate, with the free tests under `if constexpr`, so each
  instantiation is exactly the old or the new order and neither carries a runtime branch.
- `UIPC_BVH_PRED_ORDER` — 1 (default) hoisted, 0 pre-change, **2 = evaluate both orders per pair and
  atomically count disagreements on device**. That is the bit-identity proof; the surviving pair set
  should be identical by construction, since the guards are independent side-effect-free
  `return false`s.
- `UIPC_BVH_PRED_STATS=1` — per predicate, how many staged pairs the shared-index test and the
  same-body test each reject.

**Run the stats arm first.** The whole case for the reordering is that a large share of staged broad
pairs are topologically adjacent simplices that the free tests throw away after 8-16 scattered loads
have already been paid. If that share is small, the reordering is dead and round 4's per-vertex record
gather is the only remaining lever on `pairFilter`.

Measured entry point, `7020cd9b`, RTX 5060 Ti (cc 12.0), rigid-wrecking-balls 30 frames / 91 Newton
iterations / 531.4 ms kernel time: `stacklessSelf` 6.68 %, `pairFilter<AllE_AllE>` 4.65 %,
`stacklessOther` 3.19 %, `pairFilter<AllP_AllT>` 2.22 % — **16.74 % of kernel time**, the largest
family round 5 never touched.

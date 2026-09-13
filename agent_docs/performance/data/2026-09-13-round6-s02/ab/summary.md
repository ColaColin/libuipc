# s02 end-to-end A/B: `UIPC_DSB_GAUSS_NEWTON` 0 (exact + PSD projection) -> 1 (Gauss-Newton)

One build, env-switch A/B through `/workspace/output/round6/ab.py` (one discarded
warm-up per arm, ABBA ordering, Welch's t, iteration-count guard). n = 5 per arm on
every scene, manifest default frame counts, full runs.

| scene | bending? | mean ms | median ms | **ms / Newton it** | ms / PCG it | Newton | PCG | line-search |
|---|---|---|---|---|---|---|---|---|
| **cube-wall-cloth** (100 f) | yes | **−12.38 %** t=12.84 p=1.6e-06 **disjoint** | −13.86 % disjoint | **−11.83 %** t=34.1 p=1.0e-09 **disjoint** | −11.75 % disjoint | −0.63 % overlapping | −0.72 % overlapping | −0.60 % overlapping |
| **tumbler-garments** (180 f) | yes | **−18.59 %** t=6.67 p=2.2e-04 **disjoint** | −20.03 % disjoint | **−19.20 %** t=6.99 p=1.3e-04 **disjoint** | −17.02 % disjoint | +0.70 % overlapping | −1.87 % overlapping | −0.32 % overlapping |
| **stiff-gipc-case2** (250 f) | yes | **−6.65 %** t=25.5 p=2.3e-08 **disjoint** | −6.66 % disjoint | **−6.43 %** t=32.2 p=9.8e-10 **disjoint** | −6.40 % disjoint | −0.24 % overlapping | −0.27 % overlapping | +0.34 % overlapping |
| mas-bunny (100 f) | **no** | −0.14 % p=0.46 overlapping | +0.02 % | −0.14 % p=0.46 overlapping | −0.15 % | 465 vs 465, +0.00 % | +0.01 % | +0.00 % |
| rigid-wrecking-balls (120 f) | **no** | −2.27 % p=0.20 overlapping | −1.58 % | −1.19 % p=0.17 overlapping | −2.16 % | −1.08 % overlapping | −0.11 % overlapping | −1.34 % overlapping |

The iteration-count guard did **not** fire on any scene: the largest count movement
anywhere is 1.87 % (tumbler PCG) against an 18.59 % wall change, and the guard
threshold is half the wall change. On the three bending scenes the counts move by
0.24-1.87 % and overlap, in both directions -- i.e. the Gauss-Newton search
direction costs no extra Newton iterations on any measured workload, and the wall
numbers are readable as throughput.

`mas-bunny` and `rigid-wrecking-balls` execute the hinge kernel **zero** times
(`scene_executes_kernel.txt`); they are untargeted-regression checks, and both are
inside their own noise.

Raw per-run json: `/workspace/output/round6/s02/ab_*/`. The per-arm summaries with
every individual run are the `*_summary.json` files beside this note.

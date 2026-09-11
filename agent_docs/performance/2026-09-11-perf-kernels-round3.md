# 2026-09-11 — perf/kernels round 3: contact G+H split, EE reduced SPD, BVH self-cull, MAS memset, ABD side stream

- Status: Accepted (K9, K10, K11, K12, K13); K14 rejected at build time
- Before commit: `28136dc3` (perf/kernels = f4a0b415 + K6 substep fix)
- After commit: `6b1cfac8`
- Benchmark: cloth-dataset drum benchmark (`dataset/bench/drum_bench.py`, specs
  `dataset/bench/specs.json`: towel c002217 468 v / tshirt c002007 3.8 k /
  jacket+shorts c002014 5.9 k), 600 frames single process, metric = ms per
  Newton iteration; 120-frame probes with checkpoints every 5 frames for the
  determinism gate. Full report: cloth-dataset `docs/perf-bench-3080.md` §9,
  data `bench-validation/perfwork/`.

## Question

After K6–K8 the GPU time of the drum scenes is dominated by three kernels
(fused IPC contact G+H 19 %, Dahl hinge G+H 18 %, BVH edge–edge self
traversal 17 %) plus a tail of sub-ms launches. Which of the remaining
exact-arithmetic (or same-projection-to-rounding) changes from
`docs/tasks/perf-next.md` pay, and by how much on the local sm_75 part?

## Environment

| Field | Value |
|---|---|
| GPU / driver | NVIDIA GeForce RTX 2070 SUPER 8 GB (sm_75), driver 595.84, CUDA MPS daemon up |
| CUDA toolkit | 12.8 (`/workspace/deps/cuda-12.8`) |
| OS / compiler | Linux, g++ host compiler, nvcc `-O3 -std=c++20`, sm_75 only |
| Build type and important flags | Release, tests ON, pybind into `uipc-perf-env` (`build-perf`) |
| Worktree state | clean at every measured commit (each build carried exactly one change) |

## Workload and method

One change per commit; for each: build, `uipc_test_sim_case
30_fem_animiator_substep` (the K6 regression test), a 60-frame tshirt run with
`uipc.Timer` trees for the env-switched A/B of the targeted scope (ms per
Newton iteration, since the chaotic trajectory changes the Newton count), the
120-frame probe on the three specs (two runs), then the 600-frame timing.
nsys (`--trace=cuda`, 30 frames tshirt) for kernel-level attribution.
Accept rule (unchanged): max |Δx| vs baseline copy 1 ≤ 3× the worst
baseline-pair noise at every checkpoint ≤ 30 frames. Run-to-run wall noise on
this box: ±2–3 % per Newton iteration on the two large loads, ±8 % on the
towel over 600 frames.

## Correctness and safety

- K9 (contact split), K12 (memset), K13 (side stream): bit-identical by
  construction (same kernels/arithmetic/output slots; a memset of an
  all-zero-bytes struct).
- K10 (EE reduced SPD): standalone nvcc verifier against the real device
  functions, 50 000 random edge pairs inside d_hat (kappa 1e2..1e8, edge scales
  1 mm..0.1 m): max |H t|/|H|_F = 2.3e-15 over rigid translations, max relative
  Frobenius difference to `make_spd<12>` = 6.1e-12 (same projection to
  rounding, as K7); a second run with exactly parallel edges (typed PE-type
  EE flags in 11 k of 50 k samples): 2.3e-15 / 4.0e-12.
- K11 (self cull): candidate set identical by construction (leaf test
  unchanged); `UIPC_BVH_SELF_CULL_VERIFY=1` re-runs every EE self query
  without the cull and compares sets on the host: 0 mismatches in 4 400
  queries (3 specs × 120 frames).
- Probes: every step inside the noise band at every checkpoint ≤ 30 on all
  three specs; `30_fem_animiator_substep` passes after every change.
- Final build: `uipc_test_sim_case` 93/93 (12 611 assertions; the two ABD-joint
  cases 74/80 abort on e1eed4b9 too and were excluded), `uipc_test_regression`
  1/1, `uipc_test_backend_cuda` 21/22 — `collision_filter_registration` is the
  pre-existing failure; `lbvh` is flaky (failed once on each build, passed on
  re-run); `info_stackless_bvh/internal_cull_proof` counted n−1 instead of n
  `node_cull` calls after K11 (the query at the last sorted position has no
  partner and skips the root) — expectation updated in `3cb741fb`.

## Results

600 frames, single process, ms per Newton iteration (towel c002217 / tshirt
c002007 / jacket+shorts c002014); probes accepted at every step.

| Build | Change | 600 frames | Targeted scope (tshirt, ms per Newton it.) |
|---|---|---:|---|
| 28136dc3 (`t_perffix`) | reference | 7.93 / 18.11 / 20.85 | — |
| K9 d54728df | contact G+H two overlapped launches | 8.98 / 18.08 / 20.72 | contact kernel 3.40 → 3.30 ms union wall (nsys), scope 3.32 → 2.99 |
| K10 9d2012d8 | EE reduced 9×9 PSD projection | 8.23 / 17.41 / 20.42 | with K9: contact kernel 3.40 → 2.68 ms (−21 %) |
| K11 aa3c9e09 | BVH self-query subtree cull | 8.28 / 17.57 / 19.99 | BVH Query 3.09 → 2.70 (−12.5 %); self kernel 2.65 → 2.35 ms/launch |
| K12 adda87be | MAS clear via memset | 7.84 / 16.92 / 20.08 | Assemble Preconditioner 1.06 → 0.76 |
| K13 6b1cfac8 | ABD diag inverse on a side stream | 7.41 / 16.45 / 19.42 | Assemble Preconditioner 0.75 → 0.49 |
| **final 6b1cfac8** (`t_fin1/2/3`) | all five | **7.70 / 16.29 / 19.55; 7.39 / 16.30 / 19.75; 7.74 / 16.43 / 19.24 (mean 7.61 / 16.34 / 19.51)** | nsys ms/it 19.4 → 17.5 (−9.7 %) |
| baseline e1eed4b9 (`t_base_n`, `t_base_n2` same night; `t_base` night before) | — | 9.64 / 20.36 / 23.94; 9.52 / 19.94 / 23.25; 8.69 / 19.98 / 23.61 (mean 9.28 / 20.09 / 23.60) | — |
| MPS ×3 aggregate frames/s, 120 frames, two pairs | final vs baseline | 45.8 / 6.40 / 4.94 and 43.7 / 6.77 / 5.39 vs 37.0 / 5.14 / 4.44 and 35.1 / 5.20 / 4.62 (**+24 / +27 / +14 %** on the pair means) | — |

Final build (three-run means) vs 28136dc3: **−4 / −10 / −6 %**; vs the baseline
(three-run mean): **−18 / −19 / −17 %**. The towel's 600-frame numbers scatter
±5–8 % run to run (7.39–8.98 across the perf builds, 8.69–9.64 for the
baseline); only the two large loads resolve single steps.

## Interpretation

Directly measured: the per-launch kernel times (nsys) and the per-scope timer
deltas per Newton iteration; the 600-frame ms/Newton numbers carry ±2–3 %
run-to-run noise on the large loads (the chaotic trajectory changes the
contact-pair population and the PCG iteration count), so single-step 600-frame
deltas of 1–2 % (K9, K11 on the tshirt) are not resolved there and are
attributed by the scope/kernel measurements instead. The contact kernel's cost
is FP64-bound (1/32 rate on consumer parts): the EE branch's full 12×12
eigen-solve, not the PE bulk, dominated it — hence K10 (−21 % on the kernel
together with K9) and the small effect of the split alone (−3 %). K11's gain is
capped by SIMT: a warp traverses the union of its 32 adjacent queries' paths,
so only subtrees left of the whole warp are skipped (−11.5 % on the kernel
instead of the naive −50 %). K12/K13 remove or hide two ~0.3 ms single
launches (−0.3 ms/it each on the preconditioner scope). Remaining ranking on
the tshirt: hinge G+H 3.2 ms (18 %), EE self traversal 2.35 ms (15 %), contact
G+H 2.68 ms overlapped (two launches), PT traversal 0.65 ms; line-search trials
are 91 % GPU-busy compute windows (no idle gaps to fuse away).

## Decision

Accepted and committed on `perf/kernels`: K9 d54728df, K10 9d2012d8, K11
aa3c9e09, K12 adda87be, K13 6b1cfac8 (+ test follow-up 3cb741fb). K14 (`__launch_bounds__(256, 2)` on the
PE+PP contact part) rejected: ptxas refuses the 128-register cap because a
non-inlined Eigen callee (`selfadjoint_matrix_vector_product`) needs 142
registers; never committed. Every step has an env switch back to the previous
path (`UIPC_CONTACT_SPLIT`, `UIPC_EE_REDUCED_SPD`, `UIPC_BVH_SELF_RANGE_CULL`,
`UIPC_MAS_FILL_KERNEL`, `UIPC_ABD_DIAG_SIDE_STREAM`).

## Reproduction and artifacts

cloth-dataset: `python -m dataset.bench.matrix bench-validation/perfwork/plan_*.json
--out bench-validation/perfwork --python $UIPC_PERF_PY`;
`python -m dataset.bench.report bench-validation/perfwork --baseline t_base_n`;
`python bench-validation/tools/ckdx.py bench-validation/perfwork base1 fin_a`.
Verifier source for K10: `bench-validation/perfwork/k10_ee_spd_check.txt`
(output) and `k10_ee_spd_check.cu` (source; compile with the contact
file's nvcc include set, no `-rdc`). nsys reports under `bench-validation/perfwork/scratch/` (not
committed).

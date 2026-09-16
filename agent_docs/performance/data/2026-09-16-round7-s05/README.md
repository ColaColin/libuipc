# Round 7 — s05: mid-round validation + re-profile (the round's V1)

Branch `perf/round7-s05-validation` from main `0545931f` (s04 merged; engine
tree = pre-s04 main). Scene opt-ins on samples main `a47ba7d`. **No engine
source touched** — binary sha256 identical before/after
(`binary_hashes_start.txt`: `c2a104b6…` / `3734c597…`), ninja "no work to do".

## Files

| file | what it is |
|---|---|
| `VERDICT_RULE.md` | **committed before the first arm ran** (commit `ecd42250`) |
| `byte_stability.txt` | default-scene bitwise proof + perturbation footprints |
| `runs/` | 25 arm result JSONs (A = `a01..a10` exact, B = `b01..b10` GN, A'' = `ap1..ap5` perturbed exact); the 1.3 GB of position dumps stay in `/workspace/output/round7/s05/runs/*.npy` (not for the repo) |
| `divergence.txt`, `divergence_summary.json`, `div_*.npz` | per-frame rms divergence, all 300 pairs, families + verdict inputs |
| `observables.txt`, `flagged.json` | safety table, knife-edge counts, MW/Holm statistics, pre-declared bad directions |
| `trajectories.txt` | per-phase Newton/PCG/LS arm medians + solver state (ccd_toi, ls_alpha) |
| `gnverify_fullrun.txt` | `UIPC_DAHL_GN_VERIFY=1` full 130-frame run at head (probe composition) |
| `envaudit_exact_kern_cuda_gpu_kern_sum.csv` | 12-frame exact-arm capture: the launched dahl instantiation is `<1,1>` — the A arm really runs the exact path |
| `nsys2_cp_…csv`, `nsys2_cwc_…csv`, `kernel_ranking_{cp,cwc}.txt` | the fresh full-run rankings (graph-node tracing) |
| `truncated_captures/` | the first ranking attempt WITHOUT `--cuda-graph-trace=node`: CUDA-graph-internal kernels (SpMV/MAS/PCG vectors, ~950k of 1.13M launches) are invisible — kept as the instrument lesson |
| `drift_analysis.txt`, `drift_default_[1-6].json` | Part B drift: head default runs vs `baseline-runs/crease-press.json` |
| `gate_output.txt` (committed from /workspace/output/round7/s05/) | correctness gate at head |
| `scripts/` | everything re-runnable |

## Part A verdict — physics unchanged (all four pre-registered divergence conditions hold)

- **Seed**: exact-vs-GN frame-1 rms **9.36e-6 m** (median, max 1.08e-5) — 4.7e-3
  of `d_hat`, 1.3e-3 of the layer gap; between the natural GPU-reduction-order
  seed (1.45e-6) and the deliberate yaw seed (up to 1.12e-4 after one step).
  Round-6's hinge GN seeded 1.8e-7 m = 1.4e-4 of that scene's `d_hat`; the dahl
  GN seed is ~32x larger in absolute terms and ~34x relative — exactly what the
  larger dropped-term relFro (0.857 vs 0.06-0.67) predicts, and it still sits
  inside the physically-equivalent seed range.
- **Growth**: 1.367 vs 1.500 e-folds/frame (ratio 0.912, window frames 3-15).
- **Saturation**: 4.97 vs 5.73 mm rms over the last 40 frames (ratio 0.869 —
  the GN arm saturates *lower*, as within-B does vs within-A: 3.9 vs 5.2 mm).
- **Frame-wise**: CROSS median ≤ 2x EQUIV p95 at **0 of 123 frames** from
  frame 8; max ratio 0.637. (The widen rule never fired.)
- Safety: every hard item clean in all 25 runs (0 non-converged / 0 newton-limit
  / 0 ls-limit / all finite / speeds ≤ 9.7 of 50 m/s / containment / no
  inversion). The knife-edge regime checks flip inside R at least as often as
  in B (`no_inversion_or_collapse` 2/10 A, 2/5 A'', 1/10 B; `plastic_yielded`
  1/10 A, 0/5 A'', 0/10 B) — s04's marginality, not a GN effect.
- Statistics: **nothing flagged** (no median outside R, no p<0.05 with shift >
  0.5xrange). Smallest Holm p = 0.007 on `meanFrameMs` (B *faster*, 243.8 vs
  266.3 ms — that is the optimization, not physics; effect 0.28xrange(R)).
- **Counts (the honest correction duty)**: newton med 557.5 B vs 567 R
  (p=0.13), pcg 114.7k vs 114.5k (p=0.52), LS 615 vs 636 (p=0.45) — flat;
  s03's "+0.17 % at n=16" stands, sign now slightly negative at n=10, both
  inside R's own spread.
- Solver state: B clips *less* (ccd_toi_min med 0.674 vs 0.470; ls_alpha_min
  0.583 vs 0.421; ~4-5 toi<1 / alpha<1 frames per run in every arm).
- **Probe composition**: `UIPC_DAHL_GN_VERIFY=1` still works at head — gradient
  442,176,588 words **0 mismatching**; Hessian 3,158,404,200 words 89.93 %
  differ (s03: 89.96 %). The exact-arm env audit proves arm A launches
  `<1,1>`, so the A/B is the intended contrast.

## Part B — drift + fresh ranking

- **REQUIRED: meanFrameMs 262.15 → 243.04 ms (−7.29 %)** — baseline-runs value
  (n=1) vs head n=6 mean (cv 4.58 %, s00 floor 4.34 %). Against the s00
  noise-floor mean (277.93, n=5): **−12.56 %**; head max 259.1 < floor min
  262.2 — the distributions are disjoint. ms/newton 61.74 → 55.61 (−9.93 %).
- Counts all inside the scene's own floor envelope: newton 568.2 (floor
  565.2 [552,587]; the +2.9 % is vs the single baseline draw 552 = the floor's
  minimum), pcg 110.9k (floor 117.5k [104.6k,126k] — *below* the floor mean),
  LS 624.7 (floor 623.4 [599,654]). The composition is count-neutral as
  claimed; the wall win is real kernel time.
- Fresh ranking (full-run nsys, graph-node tracing, fresh prefixes):
  see `kernel_ranking_cp.txt`. Bending family 15.4 % → **5.7 %**
  (dahl 0.97 %, strain 2.25 %, stress 2.45 %); PCG family now **30.6 %**,
  contact assemble 16.0 %, BVH+CCD 13.9 %, preconditioners 13.4 %,
  membrane 5.7 % (NeoHookeanShell2D alone 4.1 %, #4 kernel).
  In-run sanity: dahl `<3,1>` **461.9 µs/launch** (16.2 ns/hinge), strain
  `<1,1>` 1219.8 µs (128.3 ns/hinge), stress `<1,1>` 1370.5 µs (144.2
  ns/hinge) — the brief's expected classes, and the plain hinge `<3,1>` on cwc
  177.8 µs in-family with s03's 177.9.

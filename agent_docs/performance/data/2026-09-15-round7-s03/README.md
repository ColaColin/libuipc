# Round 7 — s03: the dahl G/H kernel's Gauss-Newton Hessian (Proj = 3)

Branch `perf/round7-s03-dahl-gn` (cut from main `11dd35af`). Files:
`verify_gn.cu` + `build_and_run.sh` + `verify_gn_seed{7,20260915}.txt` (the math
premise, on device, real functions), `gate_default.txt` / `gate_gn0.txt`
(correctness at default and `UIPC_DAHL_GAUSS_NEWTON=0`), `gnverify_fullrun.txt`
(the in-run `UIPC_DAHL_GN_VERIFY` probe, full 130-frame crease-press),
`verify_gn.json` / `verify_exact.json` (the scene's `--verify` audit, both arms),
`binary_diff.txt` (cuobjdump -res-usage figure summary, main vs branch; the full
dumps stay in `/workspace/output/round7/s03/bindiff/`),
`nsys_{old,new}_cuda_gpu_kern_sum.csv` (scope A/B, full runs, fresh prefixes),
`ab8/` + `ab8b/` + `ab_pooled.txt` (end-to-end, 2x n=8/arm ABBA blocks),
`regr_{cwc,tumbler,mas}_cuda_gpu_kern_sum.csv` + `abnull_*/` + `abnulls.log`
(regression controls; zero dahl launches verified by grep before citing).

## The premise, proved before building (PERF_METHOD §2.2/§2.4)

The dahl response is `P(θ) = κ·w·δ² + W(d)`, so
`ddEddθ = 2κw + ddWdd` with `ddWdd = ((M_e − s·F_commit)/ℓ_e)·e^{−|d|/ℓ_e}`.
Both terms are non-negative on every state the simulator can reach:

- `2κw ≥ 0` for any valid rest mesh and physical stiffness (same dependence the
  plain hinge's accepted GN has);
- `F_commit` is written only by `commit_friction_state()` — a convex combination
  of `F` and `s·M_e`, explicitly clamped to **[−M_e, M_e]** (the brief's
  "[0, M_e]" guess was wrong in sign but immaterial: the box is symmetric) — so
  `|F_commit| ≤ M_e` and `M_e − s·F_commit ≥ M_e − |F_commit| ≥ 0`; at `d = 0`
  (`s = 0`) `ddWdd = M_e/ℓ_e ≥ 0`.

Measured on device over **2×200 000 randomised hinges** against the real
`safe_dihedral_angle` / `friction_response` / `dEdx_ddEddx` /
`dihedral_angle_{gradient,hessian}`, with θ placed exactly by rotating one wing
(d covers ±π) and the committed state drawn four ways:

| class | fraction | how |
|---|---|---|
| R | 45 % | **chaining the real `commit_friction_state`** 1–8 steps from fresh (the simulator's exact mechanism) |
| B | 35 % | uniform in-box, 1/3 at the exact boundary ±M_e |
| Z | 10 % | fresh (F = 0, θ_commit = θ_bar) |
| X | 10 % | out-of-box ±(1.05..3)·M_e — the imported-history path only |

| check (both seeds agree) | value |
|---|---|
| **ddEddθ < 0 on reachable (R+B+Z, 360 039 samples)** | **0** |
| min ddEddθ | +8.0e-6 (the elastic term: argmin sits at saturated friction, e^{−|d|/ℓ} ≈ 0) |
| min margin ddEddθ/(2κw + M_e/ℓ_e) | 2.5e-4 |
| decomposition ‖H_full − (ddEddθ·ggᵀ + dEdθ·hess θ)‖/‖H_full‖ | 2.0e-16 |
| GN fill: asymmetry / min-eig/‖H‖ / ‖H·t‖/‖H‖ | 0 exactly / −7.0e-16 / 6.4e-16 |
| dropped term ‖dEdθ·hess θ‖/‖H_full‖ (med) | 0.954 |
| **A/B Hessian delta relFro(old shipped <1,1>, GN) (med / p1)** | **0.857 / 0.029** |
| out-of-box (X): negatives / min ddEddθ | 243 per 20k (1.2 %) / −2.1e-2 |

The out-of-box class is **not reachable through the commit kernel**; it is
reachable through the imported-history edge attributes
(`dahl_friction_commit`), which are validated for finiteness only. The shipped
GN fill therefore clamps its rank-1 coefficient at 0 — provably inert on the
reachable space (the two summands are non-negative in fp, so their sum cannot
cancel; 0 activations measured), it only keeps the GN Hessian PSD where the
model itself is being fed an unphysical state. Upstream gap (not this step's
area): `do_init` could clamp/validate imported `|F| ≤ M_e`.

## The change

- `dahl_friction_discrete_shell_bending_function.h`: new fused
  `dEdx_ddEddx_gauss_newton(G, H, …, scale)` — G through the identical
  expressions/order of `dEdx_ddEddx` (exact gradient), H as the mirrored rank-1
  fill with `Vdt2` folded into the coefficient and the inert clamp above.
  `dihedral_angle_hessian` is never evaluated.
- `dahl_friction_discrete_shell_bending.cu`: `Proj = 3` arm in the G/H kernel
  template (no projection, no eigen-solve), `m_gauss_newton` knob
  **`UIPC_DAHL_GAUSS_NEWTON`** (default on = `<3,1>`; `=0` restores the exact
  Hessian + the s02 knob tree), and `UIPC_DAHL_GN_VERIFY=1` — the plain hinge's
  `UIPC_DSB_GN_VERIFY` probe shape: run the real `<1,1>`, snapshot gradient
  doublets + Hessian triplets, run `<3,1>` over the same inputs, count
  mismatching 32-bit words on device.

## Binary identity (cuobjdump -res-usage, main 11dd35af vs branch)

Whole `.so` figure diff: **2478 kernels in both, CHANGED = 0, onlyBranch = 1**
(the new `<3,1>`). All six pre-existing dahl instantiations carry exactly
main's = s02's shipped figures (`<1,1>` 255/4064/672+736, `<0,0>` 255/7664,
`<1,0>` 255/3712, `<2,0>` 255/6896, `<2,1>` 255/5472, `<0,1>` 255/3920); every
plain-hinge and plastic instantiation is untouched (hinge `<3,1>` 208/1248,
strain `<1,1>` 255/2912, stress `<1,1>` 255/3600). The GN instantiation:
**210 reg / 1248 B stack / 464+736 cmem** — the plain hinge's GN footprint.

## Numerics

- *Gradient — bit-identical, proved on the binary*: `UIPC_DAHL_GN_VERIFY=1`
  over the full 130-frame crease-press: **457 368 912 words compared, 0
  mismatching**. Hessian census over the same run: 3 266 920 800 words
  compared, 89.96 % mismatching — the controlled perturbation, whose relative
  size the standalone verifier pins at relFro median 0.857 (p1 0.029) against
  the old *shipped* (projected) Hessian, and 0.954 against the raw one.
- *Not claimed bit-identical*: the Hessian — that is the point of the step.

## Correctness

`gate.sh` identical to `baseline_tests.txt` at **default and
`UIPC_DAHL_GAUSS_NEWTON=0`** (only pytest's own duration string differs). No
in-repo test launches dahl (zero callers), so the change is additionally
exercised by the full-run probe and the scene audit below.

## Scene regime (gate 2c): full 130-frame `--verify` runs, both arms

All soundness checks pass in both arms (`verify_ok: true`, 0 non-converged, 0
newton-limit, 0 ls-limit, all finite, containment 0.350/0.358 of 0.38 m and
0.401 of 0.41 m, tri-height above floor, area ratios within the s00 envelope).
Regime observables: dahl `F_commit` active fraction after press 1 = 1.0/0.9999,
crease |F|/M 0.0660 (GN) / 0.0672 (exact) vs s00's 0.0668; plastic yield
fractions within the scene's own run-to-run spread of the s00 baseline
(strain 4.2e-4 → 8.4e-4/1.05e-3, stress 2.8 % → 2.9 %/3.8 % — the exact arm
itself moves as much against s00 as the GN arm does). Single-run Newton totals:
exact 554, GN 592 (+6.9 %) — **not reproduced at n=16** (below); PCG 110.8k →
115.4k (+4.1 %) likewise. The physics-equivalence deep-dive belongs to the
round's validation pass, as it did for the hinge's GN in round 6.

## Scope A/B (full 130-frame crease-press, fresh prefix per arm)

| capture | dahl µs/launch | ns/hinge | launches | stress | strain | SLBW |
|---|---|---|---|---|---|---|
| old (`GN=0`, `<1,1>`) | 4277.6 | 149.8 | 571 | 1367.0 | 1212.0 | 333.0 |
| **new (default, `<3,1>`)** | **458.4** | **16.1** | 594 | 1369.6 | 1215.9 | 331.9 |

**−89.28 % raw per launch, −89.31 % control-normalised** (stress+strain as the
in-run controls, ±0.35 %). Parity with the plain hinge's round-6 −90 %. Below
s02's inert-stub floor (38.3 ns/hinge) because GN also skips
`dihedral_angle_hessian` and the 144-multiply `H *= Vdt2` pass. Dahl falls from
7.2 % to 0.8 % of scene GPU kernel time; single-capture scene totals 33.71 →
33.45 s are n=1 noise (±8 %) and not quoted as an effect.

## End-to-end (ab.py, crease-press, 2 ABBA blocks, n=16/arm pooled)

| statistic | old (`GN=0`) | new (GN) | Δ | p |
|---|---|---|---|---|
| **meanFrameMs** | **265.41** | **248.74** | **−6.28 %** | **0.0151** (Mann-Whitney 0.0275) |
| ms/newton | 61.34 | 57.38 | −6.45 % | 0.0036 |
| ms/pcg (µs) | 297.5 | 277.9 | −6.59 % | 0.0286 |
| medianFrameMs | 85.8 | 87.9 | +2.4 % | worthless (bimodal, s00) |
| newton | 562.3 | 563.2 | +0.17 % | in-arm cv 2.2/2.9 % |
| pcg | 117 576 | 117 420 | −0.13 % | in-arm cv 13-16 % |
| line_search | 617.4 | 628.2 | +1.75 % | in-arm cv 2.8/4.5 % |

The count guard does **not** fire: Newton and PCG are flat at n=16, so the
wall effect is not trajectory drift, and no statistic is materially flattered
(meanFrameMs and both normalised statistics agree to 0.3 %). One single-run
pair (the verify runs) showed Newton +6.9 % — that was the scene's own
non-determinism, resolved to flat by the n=16 pool.

## Regression controls

Zero `DahlFriction*` launches in all three control kern_sums (grep count 0
each) — checked before citing. Binary CHANGED=0 is the primary guard (no
causal path); the nulls measure drift (n=3/arm):

| scene | meanFrameMs | notes |
|---|---|---|
| mas-bunny | 61.98 → 62.09 (**+0.17 %**, p=0.117) | Newton/LS exactly 465 every run, PCG ±0.01 % |
| cube-wall-cloth | 56.13 → 56.00 (−0.22 %, p=0.87) | per-newton −0.03 %; inside the ±0.54 % envelope |
| tumbler-garments | 111.65 → 107.60 (−3.63 %, p=0.338) | overlapping; no causal path (see below); s02's tumbler null was +5.74 % the other way |

Plain hinge `<3,1>` per-launch in this session's control captures: cwc
177.9 µs × 494 (s02's session 178.3), tumbler 352.5 µs × 1438 (s02's 356.9) —
in family. Plastics in the crease-press captures: +0.19 %/+0.33 %.

## Transfer category

**Algorithmic.** The change deletes work (no eigen-solve, no block assembly, no
`dihedral_angle_hessian`, no 144-multiply scale pass) at an unchanged problem
size; nothing grid-, occupancy- or machine-dependent is tuned. The removed work
is FP64-latency-bound serial dense algebra, so on a machine with relatively
weaker FP64 (5090: 1/64) the removed share grows and the per-launch percentage
should hold or grow. The count question (GN search direction on a
friction-dominated hinge) is algorithm-level too: at n=16 the counts are flat on
this scene; a rented re-measure should check them first. Prediction: kernel
−85..−92 % per launch survives; end-to-end −5..−7 % on this scene mix (dahl's
share of it), and any scene where dahl carries a larger fraction gains
proportionally.

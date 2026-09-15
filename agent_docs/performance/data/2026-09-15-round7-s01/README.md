# Round 7 — s01: translation-free PSD projection wired into the two plastic bending G/H kernels

Branch `perf/round7-s01-plastic-proj` (cut from main `fa99082b`). Files:
`binary_diff.txt` (registers/stack/cmem per instantiation, main vs branch),
`verify_proj.cu` + `build_and_run.sh` + `verifier_output.txt` (numerics),
`gate_default.txt` / `gate_alloff.txt` (correctness, default and all-knobs-off),
`nsys_old_/nsys_new__cuda_gpu_kern_sum.csv` (scope A/B, full 130-frame crease-press),
`ab7_summary.json` / `ab15_summary.json` (end-to-end A/B raw runs under `ab7/`, `ab15/`
in `/workspace/output/round7/s01/`), `regr_*.csv` (regression controls).

## Premise verification (before building)

Both plastic energies are functions of the dihedral angle θ alone, given per-hinge
constants — strain: `E = L0·kappa·δ²/h_bar` with `δ = wrap(θ − θ_bar)` (shared
`sym/discrete_shell_bending.inl`); stress: the piecewise `augmented_response_from_angle_delta`
(elastic quadratic / yielded constant-stress). Both `ddEddx` are
`ddEddθ·∇θ∇θᵀ + dEdθ·hess(θ)`. θ, ∇θ and hess(θ) are built exclusively from vertex
**differences** (`dihedral_angle.h`), so the Hessian annihilates rigid translations
exactly and the K16/K7 9×9 translation-free projection applies verbatim — the same
argument as the plain hinge (round-6 s14). Confirmed numerically on device by the
verifier: raw `‖H·t‖/‖H‖_F` ≤ 6.8e-14 (strain) / 1.1e-13 (stress) over 2×200 000
randomised hinges.

## Change

`strain_plastic_discrete_shell_bending.cu`, `stress_plastic_discrete_shell_bending.cu`:
the G/H kernel becomes `template <int Proj, int Solver>` exactly like the plain hinge's
(`Proj` compile-time so each instantiation carries only its own stack frame). Dispatch:
`Proj=1` → `make_spd_translation_free_4x3_blocked<Solver>` (shipped), `Proj=2` →
`make_spd_translation_free_4x3<Solver>`, `Proj=0` → `make_spd<12, Solver>` (old path).
`Solver=1` = tridiagonal QL, `0` = Eigen. Env switches (both kernels, one family):
`UIPC_PDSB_REDUCED_SPD`, `UIPC_PDSB_BLOCKED_PROJ`, `UIPC_PDSB_TQL2`, defaults **on** =
`<1,1>`; all three `=0` = `<0,0>` = the pre-round-7 dense 12×12 Eigen path.
`make_spd.h` and `discrete_shell_bending.cu` untouched.

## Binary identity

`cuobjdump -res-usage` on build-perf's `libuipc_backend_cuda.so`:

| instantiation | main | branch | |
|---|---|---|---|
| strain G/H `<0,0>` (old arm) | 255 reg / 6912 B / cmem 304+640 | 255 / 6912 / 304+640 | identical |
| stress G/H `<0,0>` (old arm) | 255 / 6944 / 304+664 | 255 / 6944 / 304+664 | identical |
| strain G/H `<1,1>` (shipped) | — | 255 / **2912** / 456+640 | new |
| stress G/H `<1,1>` (shipped) | — | 255 / **3600** / 504+664 | new |
| plain hinge, all 7 instantiations | e.g. `<3,1>` 208/1248 | same mangled symbols, same figures | identical |
| dahl G/H (untouched TU) | 255/11704 | 255/11704 | identical |

## Numerics (rounding level, not bit-identical — proved against the old path's own sensitivity)

Standalone verifier against the real device functions (`__device__` kernels including
the real `ddEddx` and both projections), 200 000 randomised hinges per model, parameter
ranges spanning the scene (kappa 1e-6..1, θ_y 1e-3..0.63 rad, 2 % near-degenerate
hinges). Raw H was indefinite in 100 % of inputs (the projection is active work).

| metric (max over inputs) | strain | stress |
|---|---|---|
| min eig / ‖H‖_F, old `<0,0>` | −5.79e-16 | −5.98e-16 |
| min eig / ‖H‖_F, new `<1,1>` | −6.18e-16 | −6.38e-16 |
| ‖H·t‖/‖H‖_F raw / old / new | 6.8e-14 / 5.1e-14 / **4.3e-16** | 1.1e-13 / 1.1e-13 / **4.5e-16** |
| relFro(old, new `<1,1>`) | 3.6e-14 | 8.1e-14 |
| relFro(old, old `<0,1>`) — old path's own solver sensitivity | 7.0e-15 | 7.4e-15 |
| relFro(old, new `<1,0>`) — projection change only | 3.6e-14 | 8.1e-14 |

Both projections are PSD to ~−6e-16 relative (≈ −2.8 ulp; the old path itself sits at
−5.8e-16). The new-vs-old difference is 5–11× the old path's own solver-swap sensitivity
at the extreme tail, equal at the median (~1.1e-15) — rounding-accumulation level for
two different eigensolve arrangements, orders of magnitude below the scene's own
run-to-run nondeterminism. The blocked reconstruction preserves the translation null
space ~100× better than the dense 12×12 round-trip (4e-16 vs 5e-14).

## Scope A/B (full 130-frame crease-press under nsys, fresh prefix per arm)

| kernel | old arm `<0,0>` | new arm `<1,1>` | Δ per launch | ns/hinge |
|---|---|---|---|---|
| StrainPlastic G/H | 2590.8 µs × 567 | 1211.7 µs × 565 | **−53.2 %** | 272.5 → 127.5 |
| StressPlastic G/H | 2554.5 µs × 567 | 1353.0 µs × 565 | **−47.0 %** | 268.7 → 142.3 |

Launch counts differ by 2 (567 vs 565: the scene is nondeterministic from frame 0 —
s00), so per-launch µs is the normalised statistic. In-run control: dahl G/H
4509.5 → 4445.9 µs/launch (−1.4 %, scatter). The two plastic kernels combined:
2.917 s → 1.449 s of GPU kernel time (7.74 % → 4.17 % of the scene total; scene GPU
total −7.8 % between these two single runs, count drift included). Both kernels land
at parity with the round-6-optimised plain hinge (128–145 ns/hinge).

## End-to-end (ab.py, interleaved ABBA, 1 discarded warm-up/arm)

n=7/arm first (`ab7_summary.json`): mean −3.16 % (p=0.237), ms/newton −2.77 %
(p=0.168), ms/pcg −5.05 % (p=0.088) — all overlapping at the scene's MDE (s00:
5.4 % mean / 3.4 % per-newton at n=5), Newton −0.49 %, PCG +2.06 % (in-arm PCG
spread is ±13 %, so the guard's firing is scene nondeterminism).

n=15/arm (`ab15_summary.json`, 32 runs):

| statistic | old | new | Δ | p |
|---|---|---|---|---|
| meanFrameMs | 283.36 | 263.29 | **−7.08 %** | 0.0026 |
| ms/newton | 64.28 | 61.45 | **−4.40 %** | 0.0145 |
| ms/pcg | 0.3094 | 0.2997 | −3.14 % | 0.182 |
| newton | 573.2 | 556.6 | −2.90 % | overlapping |
| pcg | 119827 | 115178 | −3.88 % | overlapping |
| line_search | 636.1 | 618.5 | −2.76 % | overlapping |

The mean_ms figure is flattered by the Newton/PCG count drift (all counts
overlapping, but the guard fired: PCG −3.88 % against −7.08 % wall); the count-
normalised statistics are the honest read: **−3.1 to −4.4 %**, inside the
predicted −2..−5 % band, consistent with the scope gate (the two kernels are
7.7 % of GPU kernel time, halved).

## Regression controls

Zero `*PlasticDiscreteShellBending*` launches in all three control kern_sums
(`reg{cwc,tumbler,mas}_cuda_gpu_kern_sum.csv`, grep count 0) — verified before
citing them, per the round-5 lesson. Null A/B (identical code executes):

| scene | n/arm | mean | ms/newton | notes |
|---|---|---|---|---|
| mas-bunny | 3 | +0.33 % (p=0.81) | +0.33 % | Newton/PCG/LS **exactly** 465/35195/465 in all runs |
| cube-wall-cloth | 3 | −0.59 % | +0.19 % | flat within the ±0.54 % envelope |
| tumbler-garments | 3 | −1.56 % (p=0.82) | +0.41 % (p=0.95) | cannot resolve < ~4 % (round 6) |

Plain hinge (the shared code I must not perturb): same-day cwc nsys, main-build
(stash + rebuild) 181.2 µs/launch vs branch-build 180.9 µs/launch (**−0.17 %**);
launch counts 517/507. The +5 % against round-6 s02's 171.7 µs reference is head
drift (s24's launch-geometry change landed after s02), not this step — the binary
identity table above is the airtight check (identical mangled symbols and figures,
TU not recompiled).

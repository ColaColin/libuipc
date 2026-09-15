# Round 7 — s02: dahl G/H kernel converted from runtime bools + Eigen to `template <Proj, Solver>` with QL

Branch `perf/round7-s02-dahl-proj` (cut from main `58a23165`). Files:
`binary_diff.txt` (registers/stack/cmem, main vs branch), `verify_proj.cu` +
`build_and_run.sh` + `verifier_output.txt` (numerics), `gate_default.txt` /
`gate_alloff.txt` (correctness), `nsys_{mainfresh,old,new,stub}_cuda_gpu_kern_sum.csv`
(scope A/B + inert stage-stub, full 130-frame crease-press), `ab6_summary.json`
(end-to-end), `regr_{cwc,tumbler,mas}_cuda_gpu_kern_sum.csv` + `abnull_*_summary.json`
(regression controls), `nsysrun_scene.sh` (control-scene capture script).

## Premise verification (before building)

The dahl energy is `P(θ) = κ·w·δ² + W(d)`, `δ = wrap(θ − θ_bar)`,
`d = wrap(θ − θ_commit)`, with the committed state `(θ_commit, F_commit)` a
per-edge **constant within a frame** (written once per accepted frame by the
TimeIntegrator's `do_update_state`), so `ddEddx = ddEddθ·∇θ∇θᵀ + dEdθ·hess(θ)`
and θ, ∇θ, hess(θ) are built only from vertex differences. Confirmed on device
against the real `dEdx_ddEddx`: raw `‖H·t‖/‖H‖_F ≤ 7.57e-14` over 200 000
randomised hinges with committed-state variation (`verifier_output.txt`).

## Change

`dahl_friction_discrete_shell_bending.cu`: the G/H kernel becomes
`template <int Proj, int Solver>` like the three siblings. `Proj=1` K16 blocked
translation-free 9×9 (shipped), `2` K7 dense basis, `0` dense `make_spd<12>`;
`Solver=1` QL, `0` Eigen. Knobs `UIPC_DAHL_REDUCED_SPD` / `UIPC_DAHL_BLOCKED_PROJ`
(kept) + new `UIPC_DAHL_TQL2`; defaults on = `<1,1>`, all off = `<0,0>` = the
historical dense 12×12 Eigen path. The G-only path stays a shared runtime-bool
early return (no projection before it). `make_spd.h` untouched.

## Binary identity (`binary_diff.txt`)

Whole-`libuipc_backend_cuda.so` `cuobjdump -res-usage` diff, main vs branch:
**2407 kernels in both, CHANGED = 0** — all 7 plain-hinge and all 12 plastic
instantiations byte-identical to s01's table (e.g. hinge `<3,1>` 208/1248,
`<1,1>` 255/3104; strain `<0,0>` 255/6912, `<1,1>` 255/2912; stress `<0,0>`
255/6944, `<1,1>` 255/3600). The only entries are inside the dahl TU: the 14
non-G/H dahl kernels match pairwise by figures, and main's single runtime-bool
G/H kernel is replaced by 6 instantiations, all 255 registers (launch geometry
unchanged):

| arm | figures | note |
|---|---|---|
| main (runtime bools) | 255 / **11 704** / cmem 728+736 | union of all three Eigen paths |
| `<0,0>` all-off | 255 / 7664 / 472+736 | historical rollback |
| `<1,0>` old default | 255 / 3712 / 624+736 | main's shipped path, dead frame gone |
| `<1,1>` shipped | 255 / **4064** / 672+736 | this step's default |

The brief expected `<0,0>` to reproduce the 11 704 B stack; it cannot: main's
frame is the **union** of the three Eigen paths (dense-12×12 7664 + blocked 3712
+ dense-basis 6896, not overlapped by nvcc) — the union-of-frames cost the
s14/s19 lesson documents, and exactly what templating removes. No instantiation
carries it; the behavioural identity of the old arms is instead proved by the
numerics gate below (same device functions, results equal to the family's own
knob/solver sensitivity).

## Numerics (rounding level, device, real functions)

200 000 randomised dahl hinges (fresh/interior/saturated friction 7/60/28 %,
F_commit fresh/saturated/interior 20/20/60 %, θ_commit offsets ±π, 2 %
near-degenerate). Raw H indefinite on 100 % (projection is active work).

| metric (max over inputs) | value |
|---|---|
| null space ‖H·t‖/‖H‖F raw (premise) | 7.57e-14 |
| null space old `<1,0>` / new `<1,1>` | 4.47e-16 / 4.46e-16 |
| min eig / ‖H‖F old `<1,0>` / new `<1,1>` | −5.95e-16 / −6.34e-16 |
| **relFro(old `<1,0>`, new `<1,1>`)** | **max 6.2e-15, med 1.10e-15** |
| relFro(old, old's own `BLOCKED_PROJ=0` knob `<2,0>`) | max 7.4e-15, med 6.5e-16 |
| relFro(all-off `<0,0>`, its own solver swap `<0,1>`) | max 6.9e-15, med 1.12e-15 |

The change (solver swap inside the blocked 9×9) is **equal at the median and
below the max** of the rounding sensitivity the old binary itself exhibits
across its own knobs — a strictly stronger result than s01's (5–11× tail).

## Correctness

`gate.sh` identical to `baseline_tests.txt` at default **and** all-knobs-off
(only pytest's own duration string differs). No in-repo test launches dahl
(zero callers, round-7 survey), so every knob-reachable instantiation was
additionally smoke-run on crease-press (15 or 5 frames, returnCode 0):
`<0,0>`, `<0,1>`, `<2,0>`, `<2,1>`.

## Scope A/B (full 130-frame crease-press, fresh prefix per arm)

| capture (binary, env) | dahl µs/launch | ns/hinge | strain | stress | SLBW | dahl/ctl |
|---|---|---|---|---|---|---|
| mainfresh (main, default) | 4458.7 × 528 | 156.1 | 1216.3 | 1365.4 | 333.6 | 3.4541 |
| main, killed-attempt capture (corroboration) | 4492.1 × 587 | 157.3 | 1234.8 | 1383.6 | 337.5 | 3.4312 |
| old arm (branch, `TQL2=0` = `<1,0>`) | 4282.1 × 543 | 149.9 | 1205.1 | 1362.4 | 331.0 | 3.3357 |
| new arm (branch, default = `<1,1>`) | 4258.9 × 546 | 149.1 | 1212.7 | 1364.6 | 331.7 | 3.3049 |
| stub (branch, Proj==1 body emptied) | 1092.7 × 555 | 38.3 | 1218.5 | 1370.3 | 333.8 | 0.8441 |

- **main → shipped `<1,1>`: −4.48 % raw, −4.32 % control-normalised** (strain +
  stress G/H as the in-run controls; they hold 1205–1235 µs, ±1.2 %, across all
  five captures). Of that, dead-frame elimination (main → `<1,0>`) is −3.4 %
  normalized and the Eigen→QL swap (`<1,0>` → `<1,1>`) only −0.9 %.
- Launch counts drift 528–587 (scene nondeterminism from frame 0, s00); the
  per-launch µs is the normalised statistic.
- **Inert stage-stub: the projection is 74.3 % of the dahl G/H kernel**
  (4258.9 → 1092.7 µs with the projection call removed; assembly + friction +
  angle work is 38.3 ns/hinge, the blocked 9×9+QL projection 110.9 ns/hinge).
  This is why the QL swap is small here: the eigen-solve core is a thin slice
  of the projection, the block assembly dominates — and the projection itself
  costs about as much as the entire round-6-optimised plain-hinge kernel.

## End-to-end (ab.py, crease-press, n=6/arm, ABBA, 1 warmup discarded/arm)

| statistic | old (`TQL2=0`) | new (default) | Δ | p |
|---|---|---|---|---|
| **meanFrameMs** | **258.49** | **255.74** | **−1.06 %** | 0.657 |
| ms/newton | 60.29 | 59.60 | −1.15 % | 0.553 |
| ms/pcg | 0.3034 | 0.3075 | +1.33 % | 0.672 |
| newton | 557.5 | 557.7 | +0.03 % | — |
| pcg | 110 937 | 108 773 | −1.95 % | — |
| line_search | 617.5 | 625.7 | +1.32 % | — |

**Below the scene's resolution** (MDE 5.4 % mean / 3.4 % per-newton at n=5,
s00) and consistent with the −0.2..−0.4 % predicted from the scope (dahl is
7.0 % of scene GPU kernel time, −4.5 % of the kernel). All statistics
overlapping; the PCG guard fired (−1.95 % count drift against a −1.06 % wall —
in-arm PCG spread ±13 %). The step rests on the scope gate, not the wall
number.

## Regression controls

Zero `DahlFriction*` launches in all three control kern_sums (grep count 0) —
verified before citing them. The binary identity (CHANGED = 0) is the primary
guard; the nulls measure drift:

| scene | n/arm | meanFrameMs | ms/newton | counts |
|---|---|---|---|---|
| mas-bunny | 3 | 62.073 → 62.069 (**−0.01 %**, p=0.975) | −0.01 % | Newton/LS exactly 465 every run; PCG ±0.01 % |
| cube-wall-cloth | 3 | 56.75 → 56.49 (−0.46 %, p=0.596) | −0.59 % (p=0.259) | flat within the ±0.54 % envelope |
| tumbler-garments | 3 | 105.06 → 111.10 (+5.74 %, p=0.23) | +4.58 % (p=0.091, disjoint at n=3) | in-arm PCG spread ±12 % |

The tumbler "+effect" is a **false positive by construction**: dahl launches
zero times there and no other kernel differs between the arms (binary CHANGED=0),
so there is no causal path; it is the n=3-disjointness accident PERF_METHOD §2.5
warns about (s01's tumbler null on the same instrument was −1.56 %, p=0.82).
Plain hinge per-launch in this session's control captures: cwc `<3,1>`
178.3 µs × 501, tumbler 356.9 µs × 1496 — in family with s01's same-day
180.9 µs.

## Transfer category

**Algorithmic + dead-frame elimination.** The dispatch change removes a
statically-reserved stack frame (11 704 B) that no executed path needed — an
occupancy/cache-layout effect, not grid- or machine-dependent, and the 3–4 %
it buys should reproduce on any GPU. The solver swap is a fixed-size dense
algebra rearrangement (same algorithm class, FP64-latency-bound): on a machine
with relatively weaker FP64 the projection share grows and so does both parts
of the win. Prediction for a rented re-measure (5090-class, FP64 1/64): kernel
level −4..−6 % per launch survives; the end-to-end effect stays sub-1 % unless
dahl's share of the scene mix grows.

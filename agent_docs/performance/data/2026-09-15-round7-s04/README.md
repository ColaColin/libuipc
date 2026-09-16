# Round 7 — s04: Gauss-Newton Hessians for the two plastic bending hinges — REJECTED by the regime gate

Branch `perf/round7-s04-plastic-gn` (cut from main `d6f3c0bc`). **The step's
premise was verified true and the implementation was numerically exact — and
the change is rejected anyway**: on the round's own scene both plastic GN
kernels independently explode the linear-solve count (PCG 19-29x) and wreck
the physics containment. Nothing ships; the engine tree on this branch is
byte-identical to main. This directory is the complete record of why.

Files: `verify_gn.cu` + `build_and_run.sh` + `verify_gn_seed{7,20260915}.txt`
(the math premise, on device, real functions, both kernels), `gate_default.txt`
/ `gate_gn0.txt` (correctness of the implementation build at default and both
knobs =0), `gate_reverted_default.txt` (the reverted/shipped state),
`gnverify_fullrun.txt` (the in-run GN_VERIFY probes, full 130-frame run),
`verify_{gn,exact,exact_run2,strain_only}.json` + `verify_{strain,stress}_only.log`
(the regime isolation, full `--verify` runs per arm), `binary_diff.txt`
(cuobjdump -res-usage figure summary; full dumps in
`/workspace/output/round7/s04/bindiff/`), `implementation.patch` (the complete
engine change that was built, measured and reverted).

## 1. The math premise — TRUE for both kernels (2x200 000 hinges each)

Sampled exactly as s03 sampled dahl: θ placed exactly by wing rotation (covers
±π), committed state 45 % by chaining the **real** `update_plastic_state` (the
TimeIntegrator's mechanism; strain: θ̄ += dir·(|δ|−yt), yt += H·excess; stress:
θ̄ += sign·δγ, ys += H·δγ), 35 % direct over the producible range (incl.
yield-stress 0 and the exact elastic boundary), 10 % fresh, 10 % out-of-range
imports (negative κ — the front-end validates finiteness only; negative yield
state). Both seeds agree:

| check | strain-plastic | stress-plastic |
|---|---|---|
| ddEddθ < 0 on reachable (R+B+Z) | **0 of 359 009** | **0 of 351 292** |
| min ddEddθ | +1.36e-6 (= 2L0κ/h̄ at the smallest κ draw) | **0.0 exactly** (the yielded branch) |
| clamp activations reachable / out-of-range | 0 / 199 733 of 201 175 (99.4 %, min −8.4e2) | 0 / 0 (out-of-range is killed by the response's own guards, both arms write 0) |
| split ‖H_full−(ddEddθ·ggᵀ+dEdθ·hessθ)‖/‖H_full‖ | 1.72e-16 | 1.70e-16 |
| GN fill asymmetry / min-eig / ‖H·t‖/‖H‖ | 0 exactly / −6.5e-16 / 8.1e-16 | 0 exactly / −6.7e-16 / 6.7e-16 |
| dropped term ‖dEdθ·hessθ‖/‖H_full‖ (med) | 0.937 (0.024 at \|δ\|<0.05, 0.81 at \|δ\|>1) | 1.000 (0.65-0.84 elastic, 1.0 yielded) |
| relFro(old shipped `<1,1>`, GN) med / p1 | 0.857 / 0.039 | 1.000 / 0.048 |

The two constitutions' curvatures: **strain** `ddEddθ = 2L0κ/h̄` is a *constant*
— no dependence on θ, on the committed θ̄, or on the yield state (the yield law
only decides where θ̄ sits; the GN hinge is the plain hinge's GN with a
state-dependent θ̄). **stress** `ddEddθ = elastic_slope > 0` below yield and
**exactly 0 above** — a yielded hinge contributes zero curvature (the classic
plasticity treatment), and the response's own validity guards
(`elastic_slope <= 0 → false`) prove non-negativity on every state where it
contributes at all. The TODO's "non-negative by inspection" claims were both
correct — and both immaterial (see §3).

## 2. The implementation — built, binary-diffed, probed, then reverted

`implementation.patch`: per-kernel fused `dEdx_ddEddx_gauss_newton` (G through
the identical expressions/order of `dEdx`; H the mirrored rank-1
`max(ddEddθ,0)·Vdt2·ggᵀ`; `dihedral_angle_hessian` never evaluated),
`Proj = 3` arms in both G/H kernels, knobs `UIPC_SPDSB_GAUSS_NEWTON` (strain) /
`UIPC_STPDSB_GAUSS_NEWTON` (stress) — per-kernel, not family, because the two
yield laws carry separately proved PSD arguments and must be independently
revertable — plus `UIPC_{SPDSB,STPDSB}_GN_VERIFY` probes (s03's shape: real
`<1,1>` first, snapshot G doublets + H triplets, `<3,1>` over the same inputs,
count mismatching 32-bit words).

- Binary (whole-.so cuobjdump figure diff vs main): **CHANGED = 0, onlyMain = 0,
  onlyBranch = 3** — the two new `<3,1>` instantiations (strain/stress both
  **202 reg / 1248 B stack**, lighter even than the hinge's GN 208/1248 and
  dahl's 210/1248; vs the old `<1,1>` arms' 255 reg / 2912 and 3600 B) plus
  s03's merged dahl `<3,1>` at exactly its published figures. Every
  pre-existing kernel figure matches main.
- In-run probes (full 130-frame crease-press, 556 G/H launches per kernel):
  gradient **202 553 848 words compared, 0 mismatching, both kernels** — the
  fused G is bit-identical to the shipped arm's. Hessian census 89.3 % (strain)
  / 89.4 % (stress) of 1.45 G words differ — the controlled perturbation, the
  same census size as dahl's 89.96 %.
- Correctness: `gate.sh` identical to `baseline_tests.txt` at default AND both
  knobs =0 (pytest duration string only).

So: exact gradients, PSD-by-construction Hessians, untouched old arms — the
implementation was right. The algorithm is not.

## 3. The regime gate — both kernels independently destroy the scene

Full 130-frame `--verify` runs, one arm at a time (the count guard fires so
hard that no wall number needs a noise model; effects are 7-29x against a
5.4 % MDE):

| arm | newton | pcg_total | pcg_max | ls-limit | meanFrameMs | strain/stress yield | area min/max |
|---|---|---|---|---|---|---|---|
| both GN off (= s03 shipped), run 1 | 574 | 111 435 | 8 180 | 0 | **246.3** | 0.0 %/2.8 % | 0.246/17.0 |
| both GN off, run 2 | 530 | 102 030 | 7 185 | 0 | 226.1 | 0.04 %/2.6 % | 0.252/16.7 |
| **both GN on** | **1164** | **3 176 408** | **345 920** | **2** | **3026.7** | 2.5 %/4.2 % | **0.183/29.4** |
| strain GN only | 784 | 2 066 830 | 99 070 | 0 | 1738.8 | **5.6 %**/1.8 % | 0.212/13.9 (z escapes 0.4132 > 0.41) |
| stress GN only | — | — | — | — | **DNF** | — | did not finish 130 frames in 900 s (>22x the exact arm's wall) |

**meanFrameMs old → new: 246.3 → 3026.7 ms (+1129 %, both on; n=1/arm but the
effect is 12x the arm's own 4.3 % cv); the count guard fires by construction**
(Newton +2x, PCG +29x — the wall number is unreadable as a performance claim
and is reported as the cost of the rejection, not as a measurement of the
kernels' speed). The failure is *not* noise and *not* marginal:

- every exact-arm run sits in the s00 envelope (Newton 530-574, PCG 102-111k,
  0 limit frames);
- the GN arms fail *different, physical* checks: both-on trips
  `no_inversion_or_collapse` (area ratio 0.18/29.4 vs 0.25/17), strain-only
  lets the clamped stack escape the z-containment limit (0.4132 > 0.41 m),
  both-on hits the line-search limit on 2 frames (s00: 0, ever);
- the yield fractions drift 20-100x off baseline (strain 5.6 % vs 0.04-0.13 %)
  — the degraded solve presses deeper creases, so the physics diverges
  progressively over the run (frame-0 PCG is identical across arms: 945-955;
  the explosion grows as yields accumulate).

The exact arm's own `verify_ok` flag flipped marginal checks run-to-run
(`plastic_yielded` on a 0.0 %-strain-yield draw, then
`no_inversion_or_collapse` on a 0.252 area-min draw) — the audit has knife-edge
checks that s03's session happened to pass; the *counts* above are the
non-marginal instrument and they are unambiguous.

### Why the plastic GNs fail where the hinge's and dahl's shipped

Offered as the evidence-backed account, not a proof:

1. **The dropped term is not small where Newton actually iterates.** At the
   committed working state (\|δ\| ≤ yield threshold ≈ 0.006 rad) it is ~2 % of
   ‖H_full‖ — but at \|δ\| > 1 rad, which press-frame iterates visit as the
   fold drives θ past the lagging θ̄, it is med 0.81-1.01 of ‖H_full‖. The
   plain hinge's GN deletes the same term and shipped — on cloth scenes whose
   bending stiffness is not the load-bearing stiffness.
2. **These hinges are the stiff constituent of an 8-decade stiffness mix**
   (s00's own finding: PCG is conditioning-limited here). Degrading the
   stiffest block's Hessian multiplies the conditioning of the global system:
   pcg_max 345 920 in one frame is a stall to the iteration cap — the same
   breakdown signature the round's PCG-stall work documented. Dahl (denim,
   κ = 2e-5, the *softest* sheet) absorbed an identically-sized perturbation
   (relFro med 0.857, dropped-term med 0.954) with flat counts at n=16; the
   plastics cannot.
3. **The stress kernel additionally writes exactly-zero curvature above
   yield**: fold modes lose all bending resistance in H, the global system is
   near-singular along crease lines. Stress-only is the worst arm (DNF),
   consistent with this being the strongest mechanism; strain-only still fails
   on mechanisms 1-2 alone.

## 4. What ships

**Nothing.** The engine tree on this branch is reverted to main; the reverted
build's binary figures match main exactly (onlyMain = 0, onlyBranch = 1 = s03's
merged dahl `<3,1>`), `gate.sh` at default is identical to baseline. The
implementation is preserved verbatim in `implementation.patch` (592 lines) for
any future revisit — e.g. a curvature-floor variant — with all its gates
already built. A default-off knob was considered and rejected: a switch whose
documented effect is a 25x PCG blowup is a trap, not an instrument.

## 5. Candidates redirected by this rejection (for s05+)

1. **The projection block assembly** (s02's candidate #2, now the family's
   main remaining lever): ~110 ns/hinge across all four bending kernels,
   search-direction-neutral (no count risk). Folding Vdt2 into the Helmert
   weights or exploiting the K16 block sparsity pays in dahl+strain+hinge at
   once (~5-7 % of this scene's GPU time combined).
2. **A curvature-floor plastic GN** (e.g. `ddEddθ ← max(ddEddθ, ε·slope)` or
   semi-smooth plasticity): the design this rejection points at; a new
   optimization needing its own premise and gates, not a variant of this step.
   The patch here is its starting point.
3. **The PCG stall on near-singular crease modes** is reproducible at will now
   (any GN arm): a free stress test for the PCG-stall guard's successor.

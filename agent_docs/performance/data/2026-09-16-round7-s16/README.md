# Round-7 s16 — the QL eigen-solve core of the plastic PSD projection: PRICED AND REJECTED

The standing candidate (s08's #1, s10's pick #5): the two plastic hinge kernels run
125.6/128.1 ns/hinge with `evd_tridiag_ql<9>` as the dominant remaining cost.
This step decomposed the QL core, ran the floor arithmetic, and measured every
lever class the diagnosis selected. **Nothing ships; the tree is identical to
main** (`git diff 0811abfd -- src/` empty; gate = baseline). The rejection is
priced, not argued:

| lever class | mechanism | measured | verdict |
|---|---|---|---|
| LDLᵀ / modified-Cholesky projection (the brief's "promising class") | delete the eigen-solve + V entirely; A′ = L D Lᵀ ≻ 0 | isolated 22-68 ns/mat vs QL-clip 121 ns — the prize is real | **numerically dead on 50 % of realistic inputs** (below) |
| transposed-V storage (local-access contiguity) | pair the rotation loop's V accesses | **−1.0/−1.1 %** isolated; static LDL/STL mix identical; no pairing possible in either layout (below) | dead |
| division reduction (reciprocal-mult, sqrt for hypot(p,1)) | cut ~40 % of the FP64 issue count | **+7.9/+8.6 % (SLOWER)** | dead |
| skip-if-PSD fast path | LDLᵀ test, return A when PD | **0 % hit rate** — 100 % of realistic Hr are indefinite | dead |

## The decomposition (ql_probe, realistic inputs = s01's generator verbatim, both models)

Isolated 4x3 blocked helper `<1,2>` (the shipped arm), 65 536 mats, ABBABA ×3:

| variant | strain ns/mat | stress ns/mat |
|---|---|---|
| V0 full helper | 146.6-149.9 | 150.4-150.9 |
| V1 QL stubbed (same I/O) | 60.2 | 60.2 |
| V2 assembly only | 60.1 | 60.1 |

**QL core = 86.5-89.6 ns/mat = 59-60 % of the whole helper.** Reconstruction is
compiler-folded (V1−V2 ≈ 0.1 ns). In-scene anchors (s10, same binary): strain
1194.0 µs/launch = 125.6 ns/hinge, stress 1217.7 = 128.1 — the isolated helper
tracks the in-kernel mix well (the isolated QL-alone kernel at 186-195 ns is a
*different* compilation context; the in-helper marginal 87 ns is the honest
cost, see "context" below).

Dynamic census of tql2 (counting twin, loop structure verbatim from evd.h):
**13.58/14.07 QL iterations and 62.0/64.0 V-column rotation steps per matrix**
(max 96/98); **warp max/mean divergence = 1.28** — real, not dominant.
Negative-eigenvalue count of the input restriction: **4-5 of 9 typically**
(strain hist peak 10 165 mats at 4, 7 444 at 5 of 19 536-sample window) —
partial-spectral methods (bisection + inverse iteration for only the negative
eigenpairs) lose their edge at k = 4-5 and add a worse orthogonality story.

## The floor arithmetic (which bound, which lever it selects)

Dynamic FP64-issue model per matrix: tred2 ≈ 500 instr (incl. ~56 divisions) +
tql2 ≈ 62 rotations × (36 V-FMA + ~15 scalar) + ~214 slow ops (divisions,
hypot) × ~10 instr ≈ **5 800 FP64-class instructions** → at the 128.4 G instr/s
FP64 issue rate of this box (2 lanes × 40 SM × 1.605 GHz) the floor is
**≈ 45 ns/mat**. Measured in-helper QL cost 87 ns ⇒ **the QL runs at ~52 % of
its FP64-issue floor.** It is NOT throughput-bound — half issue, half the serial
iteration latency + spill traffic of the 255-register kernel (8 warps/SM).

That single number selects and kills the levers:
- *Not issue-bound* ⇒ division/hypot-count reduction cannot pay — confirmed by
  measurement (B: +8 %; the reciprocal also lengthens the dependency chain the
  latency half sits on).
- *The spill half has no layout lever*: the shipped rotation loop's V accesses
  are 64-bit local ops throughout (SASS: the two hot loops carry 418 + 295
  LDL/STL.64, zero 128-bit). Transposing V cannot create pairing — in the
  shipped layout the two touched elements V[k][i], V[k][i+1] are adjacent but
  i's runtime parity blocks a forced 16-byte access, and in the transposed
  layout they land in different rows. Measured −1 % isolated with an identical
  static local-op mix. (T is also NOT bit-identical as compiled: eigenvalue
  slots differ on 0.5 % of words, V slots on 88 % — FMA-contraction noise
  through 62 rotations, the "same expressions is not proof" lesson.)
- *The only levers with size delete the eigen-solve* — and those change the
  projection itself (below).

## The modified-Cholesky pricing (mc_probe; three rules × five deltas measured)

The median is deceptively encouraging: with max-|c_jj| symmetric pivoting +
negative-pivot reflection + column-scale-aware floor (δ_rel swept 1e-14..1e-6),
**relFro(MC, clip) med 3.3e-7 (strain) / 1.4e-3 (stress)** — matching the
interlacing bound ‖E‖ ≤ 2√N·|λmin| (the raw matrices' min-eig/‖Hr‖F med is
−1.5e-8 / −1.7e-4) — and min-eig(MC) ≥ 0 everywhere.

**But the distribution's other half is unusable: 49.7 % / 50.0 % of realistic
inputs land at relFro(MC,clip) > 0.1, p99 = 3.8 / 42.8, max = 8.2 / 5.1e4**
(across all five δ choices — δ is not the axis). On those matrices the
factorization's modification is ~1e7× the projection's own size while the clip
itself is a 1e-8-relative perturbation — a qualitatively different (sometimes
garbage-scale) matrix on the *stiff constituent* of the 8-decade mix where s04
measured a 29× PCG explosion for a same-magnitude search-direction change.
Rule-iteration history (all measured, all in the outputs): unpivoted N&W floor
→ min-eig −1e176 (L-blowup); pivoted + plain δ floor → −1e161; + reflection →
correct at the median, tail as above; + θ²/dmax floor → tail 24/1e5 → 8.2/5e4.
A textbook GMW/Schnabel-Eskow two-phase rule might close the arithmetic tail,
but the ~50 %-of-inputs question would remain, and validating that on
9-decade-scale physical Hessians plus the full regime gate is a research step,
not an optimization step.

Also measured and closed: `relFro(clip, raw)` (the projection's own size, after
symmetrizing the raw dump — the Hr upper triangle is zero, comparing against it
is an artifact) = med 1.6e-8 / 3.3e-4, max 0.63/0.71 — the clip is a *tiny*
median perturbation with a ~5-10 % strongly-indefinite tail (min-eig p5
−0.40/−0.69). Any replacement must be tight to |λmin| on the median, which is
exactly what the LDLᵀ class fails to be.

## Files

| file | what it is |
|---|---|
| `ql_probe.cu` / `ql_probe_output.txt` | the decomposition (V0/V1/V2 + QL-alone/stub) and the tql2 census (iters, rotation steps, divergence, indefiniteness, neg-eig histogram) |
| `mc_probe.cu` (+`mc_gen.inc`) / `mc_probe_output.txt` | the modified-Cholesky pricing: pivoted+reflected+θ-floor rule, δ sweep, relFro(MC,clip) vs relFro(clip,raw), min-eig, isolated MC/QL/clip timings. The final function body documents the three earlier rules' failure modes inline |
| `levers_probe.cu` / `levers_output.txt` / `levers_sass.txt` | T (transposed V) and B (division-reduced) variants, own-copy `<0,0>` baseline arm, bit-diff split into eigenvalue vs eigenvector slots, relFro(V), timings; the probe-binary SASS census showing identical static LDL/STL mixes across layouts |
| `strain_112_sass.txt` / `res_usage_head.txt` / `strain_sym.txt` | the shipped strain `<1,1,2>` SASS (11 112 instr: 2 897 DFMA, 884 DMUL, 302 DADD, 117 MUFU.RCP64H, 1 844 static LDL/STL, hot-loop local-op widths) and the head res-usage table (all bending G/H instantiations) |
| `gate_head.txt` | correctness gate at head = `baseline_tests.txt` (counts identical; pytest duration string only) |
| `scripts/build_probes.sh` | builds the three probes with the backend TU's include set |

## Gates

- Correctness: `bash /workspace/output/round7/gate.sh` at head identical to
  `baseline_tests.txt` (11/3, 1112/36, 2730/46, 100/3, 4/1, 448/23, 14213/95,
  pytest 48+1 — duration string only). No knob gates: nothing shipped, no
  engine source changed (`git diff 0811abfd -- src/` empty).
- No A/B, no end-to-end arm: both arms of any comparison would be the same
  binary. The standing head wall reference is s10's meanFrameMs 236.92 ms n=8.
- mas-bunny's role for a bending change is the null by construction (zero
  bending-family launches at default, s00/s10 kern_sums); not re-captured for
  a no-ship step.

## Transfer (recorded as a prediction, no acceptance box)

Nothing ships, so there is no speed prediction to falsify. The transferable
content is the *rejection* arithmetic, which should hold on any part:

1. The FP64-issue floor of a fixed-size eigen-solve is machine-independent
   arithmetic; the "52 % of floor ⇒ not issue-bound ⇒ instruction-count levers
   are dead" argument transfers directly (on a 5090's 1/64 FP64 the floor
   doubles in wall terms, the measured/core ratio would move toward
   issue-bound — re-run the two-line floor check before believing division
   reduction there).
2. The MC-class rejection is about the *input distribution* (50 % of realistic
   plastic Hr restrictions have an LDLᵀ factorization whose bounded-E theory
   does not survive floating point), not about this GPU — it holds everywhere.
3. The isolated-vs-in-context QL inversion (187 ns isolated vs 87 ns marginal
   in the 255-register kernel) is a compilation-context effect worth knowing:
   isolated-helper microbenches UNDERSTATE how well the fused kernel hides the
   QL's latency; price in-kernel levers with in-kernel arms only.

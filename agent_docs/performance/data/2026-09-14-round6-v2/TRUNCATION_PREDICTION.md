# The `cloth_min_y` follow-up: the reading, fixed before the experiment ran

`cloth_min_y` on `cube-wall-cloth` is the only physical observable that moved anywhere in this pass:
p0 0.2862 / p1 0.2910 (+1.66 %, raw p = 0.015) at n = 20, and p0 0.2865 / p1 0.2895 (+1.04 %,
raw p = 0.024) at n = 40, against a null twin (`UIPC_CONTACT_RANK1=00`) at +0.43 % and −0.13 %.

This diagnostic was designed **after** seeing that, and the reading is fixed **before** it runs.

`UIPC_CONTACT_RANK1=1` changes only the Hessian. The energy and the gradient are untouched
(s03/s07), so **the fixed point of Newton's method is identical in both arms** — the two paths solve
the same equations. Any systematic difference in the accepted state can therefore only come from
*where the solver stops*: `cube-wall-cloth` exits on an increment / accumulated-beta rule, and a
different model Hessian gives a different increment at the same state.

**Pre-declared reading.** Divide both Newton stopping tolerances by 10 (`UIPC_CWC_TIGHT=10`,
an opt-in env knob whose default leaves every literal exactly as it was) and run all four arms in one
interleaved sweep:

- if the p1 − p0 gap in `cloth_min_y` **shrinks materially** at the tighter tolerance, the shift is
  **solver truncation**: the two arms are converging to the same answer and differ only in how far
  they get before the increment test fires. That is a convergence-quality statement, not a physics
  one, and it is bounded by the tolerance the user chooses.
- if the gap **does not shrink**, the difference survives convergence and is a change of the
  simulated state. That would be grounds to recommend against shipping, and it would be reported as
  such.

Arms: `p0` (=0), `p1` (=1), `p0t` (=0, tight), `p1t` (=1, tight); n = 25 each, interleaved,
one build, one discarded warm-up per arm.

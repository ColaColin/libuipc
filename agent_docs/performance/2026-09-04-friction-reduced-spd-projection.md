# 2026-09-04 — Friction Contact Reduced SPD Projection

- Status: Accepted
- Workload: samples `88_stiff_gipc_benchmark` (20-frame nsys kernel sums,
  100-frame x 2 end-to-end), `34_cloth_stack` accuracy gate
- Environment: Linux source build, NVIDIA RTX 2070 SUPER (sm_75), CUDA 12.8,
  double precision
- Commits: `1da12a82` (normal-contact rank-(m+1) projection, the template),
  this change

## Question

The friction `do_assemble` kernel projected every assembled 12x12/9x9/6x6
friction Hessian `H = J^T M J` through the generic N x N `make_spd` EVD, at
~2676 us/launch the second-most expensive kernel of scene 88. Can that
projection be made exact and small-dimensional, the way `1da12a82` did for the
normal-contact barrier Hessians?

## Method

`J` stacks the blocks `w_k * basis^T` of an orthonormal tangent basis, so
`J J^T = ||w||^2 I_2` (orthogonal rows of equal norm) and, for any symmetric
`M`, `make_spd(J^T M J) == J^T make_spd(M) J` exactly: with
`Q = J^T/||w||`, `Q^T Q = I_2`, `Q^T H Q = ||w||^2 M`, and clamping
eigenvalues commutes with a positive-scaled isometry. The projection therefore
moves from the assembled N x N Hessian to the 2x2 `M`
(`friction_hessian`'s own `Matrix2x2` overload, closed-form `computeDirect`),
and the four `cuda::make_spd(H)` calls in `ipc_simplex_frictional_contact.cu`
are removed. Rank of the friction Hessian is <= 2 for every pair type, so the
eigenproblem collapses 12x12/9x9/6x6 -> 2x2.

Derivation: `beam-physics/docs/friction-projection-notes.md` (companion of
`beam-physics/docs/makepd-research.md`).

Correctness gates (all pass):

- Standalone device test from this repo's own kernels
  (`/tmp/fric_spd/accept.cu`): 4 seeds x 12000 active cases per pair type,
  knife-edge barycentrics, thin triangles, near-parallel EE, both friction
  regimes plus the exact `|x| == 0` branch, both barrier band edges. Worst
  Frobenius-relative deviation 3.5e-15 (gate 1e-10). A deliberate out-of-band
  slice (negative normal force, unreachable in production because
  `is_active_D` filters those pairs) differs by <= 5.1e-16 relative to the
  pre-projection norm: both sides sit at the EVD round-off floor of the exact
  answer 0.
- Same-state assembled system, scene 88 dump A.1.0: Frobenius-relative
  4.9e-17 vs the pre-change baseline, identical entry structure (2577771 nnz);
  same-binary rerun noise 5.8e-17.

## Results

| Scope | Before | After | Delta |
|---|---|---|---|
| friction `do_assemble` (56 launches / 20 frames) | 2675.5 us/launch, 149.8 ms | 157.1 us/launch, 8.8 ms | **-94.1%** |
| ... no-projection floor (diagnostic build) | — | 149.1 us/launch | projection now +5.2% over floor |
| normal-contact `do_assemble` (control) | 2080.6 us/launch | 2076.6 us/launch | unchanged |
| friction kernel resources | REG 255, STACK 9600 B/thread | REG 250, STACK 2192 B/thread | -77% stack |
| scene 88 wall, 100 frames x 2 | 518.1 / 518.9 ms/frame | 499.9 / 494.2 ms/frame | ~-3.7%, Newton iters 6.09 vs 6.12/6.10 |

The removed N x N `SelfAdjointEigenSolver` was stack-bound (9.6 KB/thread of
local-memory spill), which is why deleting it buys 17x, far beyond the flop
count alone; the remaining 157 us is the distance/Jacobian work shared with the
gradient-only path.

Accuracy gates: `34_cloth_stack` capture validates PASS over 100 frames,
trajectory similarity vs the pristine baseline 1.19e-2 (limit 2.5 x 1.03e-2);
scene 88 100-frame centroid trajectories p50 per-frame max diff 2.0e-4 m
(limit 1e-3, known-good band 1.7e-4..5e-4).

## Notes

- The projection is exact, not approximate: identical operator up to EVD
  round-off, in every branch (`|x| >= eps_vh`, `|x| == 0`, smooth), for either
  sign of the normal force.
- `ipc_vertex_half_plane_frictional_contact.cu` still projects a 3x3 through
  the generic `make_spd`; the same argument may apply there but it is a
  separate stencil and was left untouched.

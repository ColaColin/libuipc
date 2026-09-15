# Round-7 coverage survey (2026-09-15)

Engine-vs-benchmark coverage survey that fixed the round-7 scene choice. Full matrix below
(as produced by the survey agent, condensed only in formatting).

## Benchmark scene constitution usage

| Scene | Constitutions applied |
|---|---|
| rigid-wrecking-balls | AffineBodyConstitution (+ground halfplane) |
| stiff-gipc-case2 | StableNeoHookean, StrainLimitingBaraffWitkinShell, DiscreteShellBending |
| mas-bunny | StableNeoHookean only |
| cube-wall-cloth | AffineBodyConstitution, StrainLimitingBaraffWitkinShell, DiscreteShellBending |
| tumbler-garments | AffineBodyConstitution, SoftTransformConstraint, StrainLimitingBaraffWitkinShell, DiscreteShellBending |

## Unrepresented (no benchmark caller), ranked

1. AL-IPC pipeline (contact/constitution="al-ipc"): engine/advance_al.cu, active_set_system/*,
   al_simplex_* + al_vertex_half_plane_* contact, al filters.
2. All ABD joints (revolute/prismatic/spherical/fixed + limits/driving/ext-force, joint_dof_system).
3. Plastic/frictional shell bending: dahl_friction_discrete_shell_bending.cu (zero repo callers;
   used by cloth-dataset/cloth-machine outside), strain/stress_plastic_discrete_shell_bending.cu
   (sim_case only).
4. Legacy broad-phase filters lbvh/stackless_bvh/info_stackless_bvh_v0 (+atomic_counting_lbvh) —
   runtime-selectable via collision_detection/method (default info_stackless_bvh); example 33
   selects stackless_bvh; no benchmark sets the key. UIPC_WITH_CUDA_LEGACY_COLLISION defaults ON.
5. BDF2 time integration (integrator/type default "bdf1").
6. Remaining FEM constitutions: kirchhoff_rod_bending, hookean_spring_1d, neo_hookean_shell_2d,
   arap_3d, particle_0d, empty_*d.
7. Vertex-stitch family (inter_primitive_effect_system).
8. ABD ARAP (affine_body/constitutions/arap.cu).
9. External-force and articulation constraints (uids 666/667/668/671/23).
10. SoftPositionConstraint (benchmarks pin via is_fixed instead).
11. Plain PCG (linear_system/solver default fused_pcg); use_cuda_graph modes 0 and 2.
12. Diff-sim machinery (diff_sim/enable=0).
13. Adaptive contact parameters (all scenes pass positive kappa; opt-in is negative).
14. FEM/inter-ABD animators (only tumbler inserts one, on the ABD drum).
15. Contact CFL filter (cfl/enable=0). Debug/oracle machinery.

## Kernel-state comparison (the round's target family)

- discrete_shell_bending.cu: template <int Proj, int Solver>; shipped <Proj=3 GN, Solver=1 QL>;
  env knobs UIPC_DSB_{GAUSS_NEWTON,REDUCED_SPD,BLOCKED_PROJ}, UIPC_MAKE_SPD_JACOBI (QL),
  UIPC_DSB_GN_VERIFY probe; s24 launch_spread + SpreadVerifier.
- dahl_friction_discrete_shell_bending.cu: plain __global__ kernel with runtime bools
  reduced_spd/blocked_proj (UIPC_DAHL_{REDUCED_SPD,BLOCKED_PROJ}); make_spd_translation_free_4x3*
  called with default Solver=0 (Eigen SelfAdjointEigenSolver); no GN path; plain best_grid_dim
  launches; owns a per-frame friction-commit TimeIntegrator kernel.
- strain/stress_plastic_discrete_shell_bending.cu: exact Hessian then bare make_spd(H12x12)
  (dense 12x12, Solver=0); no env knobs; plain launches.
- make_spd.h:15-18 states the cold-call-site policy explicitly: s19 changed exactly the two call
  sites it measured; dahl is one of the cold sites left at Solver=0.

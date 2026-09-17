| knob (OFF arm) | old-path kernel evidence | new-path kernel (default arm) | verdict |
|---|---|---|---|
| `UIPC_DAHL_GAUSS_NEWTON=0` | dahl G/H `<1, 1, 2>`: 253 inst (default 0) | dahl G/H `<3, 1, *>`: default 258, OFF-arm 0 | SELECTS OLD |
|   | ↳ 253x `void uipc::backend::cuda::<unnamed>::DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)1, (int)1, (int)2>(u` | | |
| `UIPC_DAHL_REDUCED_SPD=0` | dahl G/H `<0, 1, 2>`: 232 inst (default 0) | dahl G/H `<1, 1, 2>`: default 0, OFF-arm 0 | SELECTS OLD |
|   | ↳ 232x `void uipc::backend::cuda::<unnamed>::DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)0, (int)1, (int)2>(u` | | |
| `UIPC_DAHL_BLOCKED_PROJ=0` | dahl G/H `<2, 1>`: 270 inst (default 0) | dahl G/H `<1, 1, 2>`: default 0, OFF-arm 0 | SELECTS OLD |
|   | ↳ 270x `void uipc::backend::cuda::<unnamed>::DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)2, (int)1, (int)2>(u` | | |
| `UIPC_DAHL_TQL2=0` | dahl G/H `<1, 0, 2>` (Eigen): 252 inst (default 0) | dahl G/H `<1, 1, 2>`: default 0, OFF-arm 0 | SELECTS OLD |
|   | ↳ 252x `void uipc::backend::cuda::<unnamed>::DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)1, (int)0, (int)2>(u` | | |
| `UIPC_PDSB_REDUCED_SPD (strain)=0` | strain G/H `<0, 1, 2>`: 279 inst (default 0) | strain G/H `<1, 1, 2>`: default 258, OFF-arm 0 | SELECTS OLD |
|   | ↳ 279x `void uipc::backend::cuda::<unnamed>::StrainPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)0, (int)1, (int)2>(` | | |
| `UIPC_PDSB_REDUCED_SPD (stress)=0` | stress G/H `<0, 1, 2>`: 279 inst (default 0) | stress G/H `<1, 1, 2>`: default 258, OFF-arm 0 | SELECTS OLD |
|   | ↳ 279x `void uipc::backend::cuda::<unnamed>::StressPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)0, (int)1, (int)2>(` | | |
| `UIPC_PDSB_BLOCKED_PROJ (both)=0` | plastic G/H `<2, 1>`: 488 inst (default 0) | plastic G/H `<1, 1, 2>`: default 516, OFF-arm 0 | SELECTS OLD |
|   | ↳ 244x `void uipc::backend::cuda::<unnamed>::StressPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)2, (int)1, (int)2>(` | | |
|   | ↳ 244x `void uipc::backend::cuda::<unnamed>::StrainPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)2, (int)1, (int)2>(` | | |
| `UIPC_PDSB_TQL2 (both)=0` | plastic G/H `<1, 0, 2>` (Eigen): 440 inst (default 0) | plastic G/H `<1, 1, 2>`: default 516, OFF-arm 0 | SELECTS OLD |
|   | ↳ 220x `void uipc::backend::cuda::<unnamed>::StressPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)1, (int)0, (int)2>(` | | |
|   | ↳ 220x `void uipc::backend::cuda::<unnamed>::StrainPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)1, (int)0, (int)2>(` | | |
| `UIPC_NHS2D_REDUCED_SPD=0` | NHS2D G/H `<0, 1>`: 239 inst (default 0) | NHS2D G/H `<1, 1>`: default 258, OFF-arm 0 | SELECTS OLD |
|   | ↳ 239x `void uipc::backend::cuda::<unnamed>::NeoHookeanShell2D_do_compute_gradient_hessian_kernel<(int)0, (int)1>(uipc::backend::cuda_tool` | | |
| `UIPC_NHS2D_BLOCKED_PROJ=0` | NHS2D G/H `<2, 1>`: 246 inst (default 0) | NHS2D G/H `<1, 1>`: default 258, OFF-arm 0 | SELECTS OLD |
|   | ↳ 246x `void uipc::backend::cuda::<unnamed>::NeoHookeanShell2D_do_compute_gradient_hessian_kernel<(int)2, (int)1>(uipc::backend::cuda_tool` | | |
| `UIPC_NHS2D_TQL2=0` | NHS2D G/H `<1, 0>` (Eigen): 251 inst (default 258) | NHS2D G/H `<1, 1>`: default 258, OFF-arm 0 | SELECTS OLD |
|   | ↳ 251x `void uipc::backend::cuda::<unnamed>::NeoHookeanShell2D_do_compute_gradient_hessian_kernel<(int)1, (int)0>(uipc::backend::cuda_tool` | | |
| `UIPC_MAKE_SPD_BLOCKED_HALF=0` | bending G/H `<1, 1, 0>` (full asm): 490 inst (default 0) | bending G/H `<1, 1, 2>` (half asm): default 516, OFF-arm 0 | SELECTS OLD |
|   | ↳ 245x `void uipc::backend::cuda::<unnamed>::StressPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)1, (int)1, (int)0>(` | | |
|   | ↳ 245x `void uipc::backend::cuda::<unnamed>::StrainPlasticDiscreteShellBending_do_compute_gradient_hessian_kernel<(int)1, (int)1, (int)0>(` | | |
| `UIPC_SEGRED_UNSTAGE=0` | staged k3 sort kernel: 247 inst (default 0) | folded PERM k2 reduce: default 0, OFF-arm 0 | SELECTS OLD |
|   | ↳ 247x `void uipc::backend::cuda::<unnamed>::matrix_converter_radix_sort_indices_and_blocks_k3_kernel<double, (int)3>(uipc::backend::cuda_` | | |
| `UIPC_SEGRED_TREE=0` | k2 warp-tree reduce (wide classes): 708 inst (default 255) | ks K-serial reduce: default 513, OFF-arm 0 | SELECTS OLD |
|   | ↳ 473x `void uipc::backend::cuda_tool::<unnamed>::fast_segmental_reduce_matrix_k2_kernel<(int)128, (int)32, double, (int)3, (int)3, uipc::` | | |
|   | ↳ 235x `void uipc::backend::cuda_tool::<unnamed>::fast_segmental_reduce_matrix_k2_kernel<(int)64, (int)32, double, (int)3, (int)1, uipc::b` | | |
| `UIPC_DOUBLET_UNSTAGE=0` | CUB onesweep `<unsigned int, Matrix3x1>`: 474 inst (default 0) | CUB onesweep `<unsigned int, int>`: default 0, OFF-arm 0 | SELECTS OLD |
|   | ↳ 474x `void cub::CUB_200700_750_NS::DeviceRadixSortOnesweepKernel<cub::CUB_200700_750_NS::DeviceRadixSortPolicy<int, Eigen::Matrix<double` | | |
| `UIPC_PCG_POLL=0` | 8 B D2H memcpyAsync: 8092 (12f) | default arm: 962 (12f) | SELECTS OLD |

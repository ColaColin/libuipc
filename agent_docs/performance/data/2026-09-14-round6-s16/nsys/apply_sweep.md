# s16 nsys kern_sum, full runs, one session per mode (mb = mas-bunny 100 f, c2 = stiff-gipc-case2 250 f); fN = UIPC_MAS_FUSED_R=N

| session | kernel | grid | regs | launches | us/launch | per-apply us | kernel sum ms |
|---|---|---:|---:|---:|---:|---:|---:|
| mb_f0 | `build_multi_level_R_kernel` | 80 | 42 | 35670 | 13.448 | | 6318.3 |
| mb_f0 | `collect_final_Z_kernel` | 81 | 39 | 35670 | 5.835 | | 6318.3 |
| mb_f0 | `schwarz_local_solve_rowdot2_kernel<` | 686 | 72 | 35670 | 24.227 | | 6318.3 |
| mb_f0 | **apply total** | | | | | **43.510** | |
| mb_f1 | `collect_final_Z_kernel` | 81 | 39 | 35700 | 5.950 | | 6146.1 |
| mb_f1 | `schwarz_local_solve_fused_R_kernel<` | 46 | 72 | 35700 | 4.101 | | 6146.1 |
| mb_f1 | `schwarz_local_solve_fused_R_kernel<` | 640 | 118 | 35700 | 28.657 | | 6146.1 |
| mb_f1 | **apply total** | | | | | **38.708** | |
| mb_f2 | `collect_final_Z_kernel` | 81 | 39 | 35680 | 6.147 | | 6084.0 |
| mb_f2 | `schwarz_local_solve_fused_R_kernel<` | 46 | 72 | 35680 | 4.129 | | 6084.0 |
| mb_f2 | `schwarz_local_solve_fused_R_kernel<` | 640 | 72 | 35680 | 27.039 | | 6084.0 |
| mb_f2 | **apply total** | | | | | **37.314** | |
| mb_f2b | `collect_final_Z_kernel` | 81 | 39 | 35685 | 6.015 | | 6051.8 |
| mb_f2b | `schwarz_local_solve_fused_R_kernel<` | 46 | 72 | 35685 | 4.171 | | 6051.8 |
| mb_f2b | `schwarz_local_solve_fused_R_kernel<` | 640 | 78 | 35685 | 26.424 | | 6051.8 |
| mb_f2b | **apply total** | | | | | **36.611** | |
| mb_f3 | `collect_final_Z_kernel` | 81 | 39 | 35665 | 5.848 | | 6244.5 |
| mb_f3 | `schwarz_local_solve_fused_R_kernel<` | 46 | 72 | 35665 | 4.134 | | 6244.5 |
| mb_f3 | `schwarz_local_solve_fused_R_kernel<` | 640 | 92 | 35665 | 31.619 | | 6244.5 |
| mb_f3 | **apply total** | | | | | **41.601** | |
| c2_f0 | `build_multi_level_R_kernel` | 178 | 42 | 66884 | 22.785 | | 41190.6 |
| c2_f0 | `collect_final_Z_kernel` | 180 | 39 | 66884 | 9.926 | | 41190.6 |
| c2_f0 | `schwarz_local_solve_rowdot2_kernel<` | 1522 | 72 | 66884 | 47.386 | | 41190.6 |
| c2_f0 | **apply total** | | | | | **80.098** | |
| c2_f1 | `collect_final_Z_kernel` | 180 | 39 | 66234 | 9.959 | | 40459.9 |
| c2_f1 | `schwarz_local_solve_fused_R_kernel<` | 101 | 72 | 66234 | 5.835 | | 40459.9 |
| c2_f1 | `schwarz_local_solve_fused_R_kernel<` | 1421 | 118 | 66234 | 53.583 | | 40459.9 |
| c2_f1 | **apply total** | | | | | **69.378** | |
| c2_f2 | `collect_final_Z_kernel` | 180 | 39 | 65377 | 9.954 | | 40115.7 |
| c2_f2 | `schwarz_local_solve_fused_R_kernel<` | 101 | 72 | 65377 | 5.835 | | 40115.7 |
| c2_f2 | `schwarz_local_solve_fused_R_kernel<` | 1421 | 72 | 65377 | 51.008 | | 40115.7 |
| c2_f2 | **apply total** | | | | | **66.798** | |
| c2_f2b | `collect_final_Z_kernel` | 180 | 39 | 66123 | 9.965 | | 40190.6 |
| c2_f2b | `schwarz_local_solve_fused_R_kernel<` | 101 | 72 | 66123 | 5.832 | | 40190.6 |
| c2_f2b | `schwarz_local_solve_fused_R_kernel<` | 1421 | 78 | 66123 | 50.910 | | 40190.6 |
| c2_f2b | **apply total** | | | | | **66.708** | |
| c2_f3 | `collect_final_Z_kernel` | 180 | 39 | 64698 | 9.930 | | 40320.3 |
| c2_f3 | `schwarz_local_solve_fused_R_kernel<` | 101 | 72 | 64698 | 5.780 | | 40320.3 |
| c2_f3 | `schwarz_local_solve_fused_R_kernel<` | 1421 | 92 | 64698 | 57.953 | | 40320.3 |
| c2_f3 | **apply total** | | | | | **73.664** | |

# sim_case: sanitizer records attributed to test cases (assign_records.py)

## base__racecheck__sim_case_subsetA
summary: ========= RACECHECK SUMMARY: 4336 hazards displayed (0 errors, 4336 warnings)
tests:   All tests passed (529 assertions in 6 test cases)
   254  53_fem_mas_small_cube                      build_multi_level_R_kernel x250; build_level1_kernel x2; prepare_prefix_sum_L0_kernel x1; build_connect_mask_Lx_kernel x1
  3276  56_fem_mas_bunny_ground                    build_multi_level_R_kernel x3264; prefix_sum_Lx_kernel x4; build_connect_mask_Lx_kernel x3; build_level1_kernel x2; next_level_cluster_kernel x2; prepare_prefix_sum_L0_kernel x1
   806  60_fem_mas_cloth                           build_multi_level_R_kernel x798; build_connect_mask_Lx_kernel x2; build_level1_kernel x2; prefix_sum_Lx_kernel x2; prepare_prefix_sum_L0_kernel x1; next_level_cluster_kernel x1
-- 4336 records in 3 cases; 0 unassigned trailing records

slowest cases (-d yes):
  1005.330 s: 18_abd_fem_contact
  381.097 s: 56_fem_mas_bunny_ground
  277.278 s: 18_abd_fem_contact
  140.703 s: 33_discrete_shell_bending
  111.543 s: 1_abd_contact_pt

## head__memcheck__sim_case_full
summary: ========= ERROR SUMMARY: 0 errors
tests:   All tests passed (14213 assertions in 95 test cases)
-- 0 records in 0 cases; 0 unassigned trailing records

slowest cases (-d yes):
  383.938 s: 17_fem_multi_constituion
  270.032 s: 11_abd_ramp_sliding
  217.957 s: 28_fem_periodically_pressed_tet
  209.352 s: 14_fem_3d_ground_contact
  203.120 s: 18_abd_fem_contact

## head__racecheck__sim_case_full
summary: 
tests:   
-- 0 records in 0 cases; 0 unassigned trailing records

slowest cases (-d yes):
  66.154 s: 10_abd_ground_contact
  17.900 s: 0_abd_gravity
  8.996 s: 0_abd_gravity

## head__racecheck__sim_case_subsetA
summary: ========= RACECHECK SUMMARY: 24 hazards displayed (0 errors, 24 warnings)
tests:   All tests passed (529 assertions in 6 test cases)
     4  53_fem_mas_small_cube                      build_level1_kernel x2; prepare_prefix_sum_L0_kernel x1; build_connect_mask_Lx_kernel x1
    12  56_fem_mas_bunny_ground                    prefix_sum_Lx_kernel x4; build_connect_mask_Lx_kernel x3; build_level1_kernel x2; next_level_cluster_kernel x2; prepare_prefix_sum_L0_kernel x1
     8  60_fem_mas_cloth                           build_connect_mask_Lx_kernel x2; build_level1_kernel x2; prefix_sum_Lx_kernel x2; prepare_prefix_sum_L0_kernel x1; next_level_cluster_kernel x1
-- 24 records in 3 cases; 0 unassigned trailing records

slowest cases (-d yes):
  1004.173 s: 18_abd_fem_contact
  384.683 s: 56_fem_mas_bunny_ground
  280.375 s: 18_abd_fem_contact
  139.056 s: 33_discrete_shell_bending
  112.164 s: 1_abd_contact_pt


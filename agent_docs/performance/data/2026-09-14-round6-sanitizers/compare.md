# Record-level comparisons (compare.py)

## compare.py head base racecheck build_multi_level_R --nokernel
racecheck 6_wrecking_balls_6f              head:    0 records  base:    0  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 88_stiff_gipc_benchmark_6f       head:   19 records  base:   19  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 89_mas_bunny_6f                  head:   16 records  base:   15  DIFFER  (ignoring build_multi_level_R)
     head 1 base 0 ('Warning', '0xa30', '0xb90', 76744)
racecheck 93_cube_wall_cloth_6f            head:   16 records  base:   16  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 95_tumbler_garments_24f          head:   17 records  base:   17  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 95_tumbler_garments_6f           head:   17 records  base:   17  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck backend_cuda                     head:    0 records  base:    0  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck sim_case_subsetA                 head:   24 records  base:   24  IDENTICAL record-for-record  (ignoring build_multi_level_R)

## compare.py head base racecheck build_multi_level_R
racecheck 6_wrecking_balls_6f              head:    0 records  base:    0  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 88_stiff_gipc_benchmark_6f       head:   19 records  base:   19  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 89_mas_bunny_6f                  head:   16 records  base:   15  DIFFER  (ignoring build_multi_level_R)
     head 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     head 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     head 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     head 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     head 1 base 0 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0xa30', 'Read', '0xb90', 76744)
racecheck 93_cube_wall_cloth_6f            head:   16 records  base:   16  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 95_tumbler_garments_24f          head:   17 records  base:   17  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 95_tumbler_garments_6f           head:   17 records  base:   17  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck backend_cuda                     head:    0 records  base:    0  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck sim_case_subsetA                 head:   24 records  base:   24  IDENTICAL record-for-record  (ignoring build_multi_level_R)

## compare.py rb base racecheck -
racecheck 6_wrecking_balls_6f              rb:    0 records  base:    0  IDENTICAL record-for-record
racecheck 88_stiff_gipc_benchmark_6f       rb: 2239 records  base: 2239  IDENTICAL record-for-record
racecheck 89_mas_bunny_6f                  rb: 1176 records  base: 1175  DIFFER
     rb 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     rb 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     rb 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     rb 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     rb 1 base 0 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0xa30', 'Read', '0xb90', 76744)
racecheck 93_cube_wall_cloth_6f            rb:  256 records  base:  256  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           rb:  621 records  base:  792  DIFFER
     rb 604 base 775 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0xc40', 'Read', '0x10b0', 2560)
racecheck backend_cuda                     rb:    0 records  base:    0  IDENTICAL record-for-record

## compare.py rb base racecheck build_multi_level_R --nokernel
racecheck 6_wrecking_balls_6f              rb:    0 records  base:    0  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 88_stiff_gipc_benchmark_6f       rb:   19 records  base:   19  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 89_mas_bunny_6f                  rb:   16 records  base:   15  DIFFER  (ignoring build_multi_level_R)
     rb 1 base 0 ('Warning', '0xa30', '0xb90', 76744)
racecheck 93_cube_wall_cloth_6f            rb:   16 records  base:   16  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 95_tumbler_garments_6f           rb:   17 records  base:   17  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck backend_cuda                     rb:    0 records  base:    0  IDENTICAL record-for-record  (ignoring build_multi_level_R)

## compare.py head base initcheck -
initcheck 6_wrecking_balls_8f              head:    0 records  base:    0  IDENTICAL record-for-record
initcheck 88_stiff_gipc_benchmark_8f       head:    0 records  base:    0  IDENTICAL record-for-record
initcheck 89_mas_bunny_8f                  head:    0 records  base:    0  IDENTICAL record-for-record
initcheck 93_cube_wall_cloth_8f            head: 8127 records  base: 8127  IDENTICAL record-for-record
initcheck 95_tumbler_garments_24f          head: 5418 records  base: 5418  IDENTICAL record-for-record
initcheck 95_tumbler_garments_8f           head: 5418 records  base: 5418  IDENTICAL record-for-record
initcheck backend_cuda                     head:    0 records  base:    0  IDENTICAL record-for-record

## compare.py rb base initcheck -
initcheck 6_wrecking_balls_8f              rb:    0 records  base:    0  IDENTICAL record-for-record
initcheck 88_stiff_gipc_benchmark_8f       rb:    0 records  base:    0  IDENTICAL record-for-record
initcheck 89_mas_bunny_8f                  rb:    0 records  base:    0  IDENTICAL record-for-record
initcheck 93_cube_wall_cloth_8f            rb: 8127 records  base: 8127  IDENTICAL record-for-record
initcheck 95_tumbler_garments_8f           rb: 5418 records  base: 5418  IDENTICAL record-for-record

## compare.py head2 head racecheck -
racecheck 89_mas_bunny_6f                  head2:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           head2:   17 records  head:   17  IDENTICAL record-for-record

## compare.py head3 head racecheck -
racecheck 89_mas_bunny_6f                  head3:   16 records  head:   16  IDENTICAL record-for-record

## compare.py base2 base racecheck -
racecheck 89_mas_bunny_6f                  base2: 1176 records  base: 1175  DIFFER
     base2 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     base2 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     base2 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     base2 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     base2 1 base 0 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0xa30', 'Read', '0xb90', 76744)
racecheck 95_tumbler_garments_6f           base2:  901 records  base:  792  DIFFER
     base2 884 base 775 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0xc40', 'Read', '0x10b0', 2560)

## compare.py base2 head racecheck build_multi_level_R --nokernel
racecheck 89_mas_bunny_6f                  base2:   16 records  head:   16  IDENTICAL record-for-record  (ignoring build_multi_level_R)
racecheck 95_tumbler_garments_6f           base2:   17 records  head:   17  IDENTICAL record-for-record  (ignoring build_multi_level_R)

## compare.py rb2 rb racecheck build_multi_level_R --nokernel
racecheck 95_tumbler_garments_6f           rb2:   17 records  rb:   17  IDENTICAL record-for-record  (ignoring build_multi_level_R)

## compare.py fr0 base racecheck -
racecheck 88_stiff_gipc_benchmark_6f       fr0: 2407 records  base: 2239  DIFFER
     fr0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0x1560', 'Read', '0x1bd0', 72)
     fr0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0x1750', 'Read', '0x1e60', 72)
     fr0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0x1980', 'Read', '0x20f0', 72)
     fr0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0xc40', 'Read', '0x10b0', 11392)
racecheck 89_mas_bunny_6f                  fr0: 1176 records  base: 1175  DIFFER
     fr0 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     fr0 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     fr0 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     fr0 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     fr0 1 base 0 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0xa30', 'Read', '0xb90', 76744)

## compare.py rd0 base racecheck -
racecheck 88_stiff_gipc_benchmark_6f       rd0: 2407 records  base: 2239  DIFFER
     rd0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0x1560', 'Read', '0x1bd0', 72)
     rd0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0x1750', 'Read', '0x1e60', 72)
     rd0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0x1980', 'Read', '0x20f0', 72)
     rd0 597 base 555 ('Warning', 'MASPreconditionerEngine_build_multi_level_R_kernel', 'Write', '0xc40', 'Read', '0x10b0', 11392)
racecheck 89_mas_bunny_6f                  rd0: 1176 records  base: 1175  DIFFER
     rd0 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     rd0 1 base 0 ('Warning', 'MASPreconditionerEngine_build_level1_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     rd0 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x12d0', 'Read', '0x1510', 5124)
     rd0 0 base 1 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0x500', 'Read', '0xff0', 5120)
     rd0 1 base 0 ('Warning', 'MASPreconditionerEngine_prepare_prefix_sum_L0_kernel', 'Write', '0xa30', 'Read', '0xb90', 76744)

## compare.py fr1 head racecheck -
racecheck 88_stiff_gipc_benchmark_6f       fr1:   19 records  head:   19  IDENTICAL record-for-record
racecheck 89_mas_bunny_6f                  fr1:   16 records  head:   16  IDENTICAL record-for-record

## compare.py sr0 head racecheck -
racecheck 88_stiff_gipc_benchmark_6f       sr0:   19 records  head:   19  IDENTICAL record-for-record
racecheck 89_mas_bunny_6f                  sr0:   16 records  head:   16  IDENTICAL record-for-record

## compare.py pp0 head racecheck -
racecheck 6_wrecking_balls_6f              pp0:    0 records  head:    0  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            pp0:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           pp0:   17 records  head:   17  IDENTICAL record-for-record

## compare.py pp1 head racecheck -
racecheck 6_wrecking_balls_6f              pp1:    0 records  head:    0  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            pp1:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           pp1:   17 records  head:   17  IDENTICAL record-for-record

## compare.py pp2 head racecheck -
racecheck 6_wrecking_balls_6f              pp2:    0 records  head:    0  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            pp2:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           pp2:   17 records  head:   17  IDENTICAL record-for-record

## compare.py pp3 head racecheck -
racecheck 6_wrecking_balls_6f              pp3:    0 records  head:    0  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            pp3:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           pp3:   17 records  head:   17  IDENTICAL record-for-record

## compare.py dj0 head racecheck -
racecheck 6_wrecking_balls_6f              dj0:    0 records  head:    0  IDENTICAL record-for-record
racecheck 88_stiff_gipc_benchmark_6f       dj0:   19 records  head:   19  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            dj0:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           dj0:   17 records  head:   17  IDENTICAL record-for-record

## compare.py dj2 head racecheck -
racecheck 6_wrecking_balls_6f              dj2:    0 records  head:    0  IDENTICAL record-for-record
racecheck 88_stiff_gipc_benchmark_6f       dj2:   19 records  head:   19  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            dj2:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           dj2:   17 records  head:   17  IDENTICAL record-for-record

## compare.py sp0 head racecheck -
racecheck 6_wrecking_balls_6f              sp0:    0 records  head:    0  IDENTICAL record-for-record
racecheck 88_stiff_gipc_benchmark_6f       sp0:   19 records  head:   19  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            sp0:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           sp0:   17 records  head:   17  IDENTICAL record-for-record

## compare.py sp1 head racecheck -
racecheck 6_wrecking_balls_6f              sp1:    0 records  head:    0  IDENTICAL record-for-record
racecheck 88_stiff_gipc_benchmark_6f       sp1:   19 records  head:   19  IDENTICAL record-for-record
racecheck 93_cube_wall_cloth_6f            sp1:   16 records  head:   16  IDENTICAL record-for-record
racecheck 95_tumbler_garments_6f           sp1:   17 records  head:   17  IDENTICAL record-for-record

## compare.py head base synccheck -
synccheck 6_wrecking_balls_12f             head:    0 records  base:    0  IDENTICAL record-for-record
synccheck 88_stiff_gipc_benchmark_12f      head:    0 records  base:    0  IDENTICAL record-for-record
synccheck 89_mas_bunny_12f                 head:    0 records  base:    0  IDENTICAL record-for-record
synccheck 93_cube_wall_cloth_12f           head:    0 records  base:    0  IDENTICAL record-for-record
synccheck 95_tumbler_garments_12f          head:    0 records  base:    0  IDENTICAL record-for-record
synccheck backend_cuda                     head:    0 records  base:    0  IDENTICAL record-for-record

## compare.py head base memcheck -
memcheck  6_wrecking_balls_12f             head:    0 records  base:    0  IDENTICAL record-for-record
memcheck  88_stiff_gipc_benchmark_12f      head:    0 records  base:    0  IDENTICAL record-for-record
memcheck  89_mas_bunny_12f                 head:    0 records  base:    0  IDENTICAL record-for-record
memcheck  93_cube_wall_cloth_12f           head:    0 records  base:    0  IDENTICAL record-for-record
memcheck  95_tumbler_garments_12f          head:    0 records  base:    0  IDENTICAL record-for-record
memcheck  backend_cuda                     head:    0 records  base:    0  IDENTICAL record-for-record


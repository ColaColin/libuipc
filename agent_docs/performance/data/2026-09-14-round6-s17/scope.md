# s17 scope: nsys cuda_gpu_kern_sum, full runs, one session per arm, head 4181f28b + s17 (this box, cc 7.5)

| session | kernel (instantiation) | regs | grid x block (first launch) | launches | mean us | min | max | % of scene kernel time | scene kernel sum ms |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| head_c2 | old cub tree | 46 | 18941x128 | 1664 | 1177.9 | 1077.3 | 1383.5 | 4.88 | 40173 |
| c2_old | old cub tree | 46 | 18918x128 | 1665 | 1179.3 | 1076.1 | 1380.8 | 4.90 | 40110 |
| c2_new | k2 (production, Probe=0) | 48 | 18918x128 | 1663 | 915.0 | 818.2 | 1105.9 | 3.84 | 39612 |
| c2_probe | k2 (production, Probe=0) | 48 | 18923x128 | 1667 | 916.0 | 819.1 | 1099.3 | 3.58 | 42694 |
| c2_probe | k2 Probe=1 (no tree) | 36 | 18903x128 | 1667 | 861.8 | 781.9 | 1037.1 | 3.36 | 42694 |
| c2_probe | k2 Probe=2 (no gather) | 48 | 18910x128 | 1667 | 803.7 | 718.0 | 964.9 | 3.14 | 42694 |
| head_mb | old cub tree | 46 | 6395x128 | 465 | 489.2 | 482.5 | 543.9 | 3.76 | 6054 |
| mb_old | old cub tree | 46 | 6395x128 | 465 | 488.4 | 482.4 | 544.8 | 3.76 | 6047 |
| mb_new | k2 (production, Probe=0) | 48 | 6395x128 | 465 | 367.4 | 360.0 | 439.0 | 2.86 | 5980 |
| mb_probe | k2 (production, Probe=0) | 48 | 6395x128 | 465 | 369.9 | 362.8 | 437.9 | 2.71 | 6351 |
| mb_probe | k2 Probe=1 (no tree) | 36 | 6401x128 | 465 | 361.2 | 355.7 | 366.3 | 2.64 | 6351 |
| mb_probe | k2 Probe=2 (no gather) | 48 | 6395x128 | 465 | 318.7 | 312.0 | 382.0 | 2.33 | 6351 |

Other kernels in the same sessions (unchanged by this step), per launch:

| session | SpMV chunked | SNH G/H | MAS fused_R (fine) |
|---|---:|---:|---:|
| c2_old | 108.28 us x 63710 | 2847.96 us x 1665 | 50.86 us x 65375 |
| c2_new | 108.02 us x 63880 | 2846.55 us x 1663 | 50.84 us x 65543 |
| mb_old | 51.72 us x 35225 | 1454.68 us x 465 | 26.42 us x 35690 |
| mb_new | 51.62 us x 35210 | 1455.71 us x 465 | 26.22 us x 35675 |

# s01 scope measurements — `DiscreteShellBending_do_compute_gradient_hessian_kernel`

All numbers are **nsys over the complete benchmark run** (`nsys profile -t cuda
--cuda-graph-trace=node`, `cuda_gpu_kern_sum` only — the trace report is 400 MB on a tumbler
run and the ranking does not need it), fresh output prefix per run, wrapper
`nsys/nsysrun.sh` (fails loudly if no csv appears). Raw csvs in `nsys/`.

`cube-wall-cloth` = 100 frames, 12 160 hinges, launch geometry `<<<380, 32>>>`.
`tumbler-garments` = 180 frames, 27 158 hinges, launch geometry `<<<425, 64>>>`.
**The geometry is identical in every arm** (read out of `cuda_gpu_trace` on a 15-frame run,
`geo_a/geo_b/geo_c`): the `__launch_bounds__` caps `best_block_dim` at 128 instead of 256, but
the s24 ramp then halves it to the same final block size. Only the register budget moves.

## 1. The launch bound, per launch

| scene | arm | ms/launch | runs | mean | vs unbounded |
|---|---|---|---|---|---|
| cube-wall-cloth | unbounded, 255 reg, 8 warps/SM | 1.7497 / 1.7639 / 1.7600 / 1.7537 / 1.7504 | 5 | **1.7555** (sd 0.0064, cv 0.36 %) | — |
| cube-wall-cloth | `__launch_bounds__(128,3)`, 168 reg, 12 warps/SM | 1.6617 / 1.6724 / 1.6622 / 1.6661 | 4 | **1.6656** (sd 0.0050) | **−5.12 %**, disjoint |
| cube-wall-cloth | `__launch_bounds__(128,4)`, 128 reg, 16 warps/SM | 1.7923 | 1 | 1.7923 | +2.1 % |
| tumbler-garments | unbounded | 3.4213 / 3.4252 / 3.4998 | 3 | **3.4488** (sd 0.0443) | — |
| tumbler-garments | `(128,3)` | 3.8838 / 3.8372 / 3.8973 | 3 | **3.8728** (sd 0.0315) | **+12.29 %**, disjoint |
| tumbler-garments | `(128,4)` | 4.9620 | 1 | 4.9620 | +43.9 % |

Untouched kernels in the same cube-wall runs move +0.2 to +1.0 % (`do_assemble`,
`StrainLimiting*` G/H, `abd_diag_*`, `Spmv`), which is this box's run-to-run drift; the
signal is 5–12x that. On the tumbler the CCD kernels move ±20 % between runs because the
contact set is trajectory-dependent — the hinge kernel does **not**: it processes exactly
27 158 hinges on every launch regardless of the trajectory, which is why its per-launch
number is comparable across runs and the CCD kernels' are not.

### Why the sign flips — it is wave quantisation, not occupancy

sm_75 has 40 SMs, 65 536 registers and 32 warp slots per SM. 255 registers/thread admits
**8 warps per SM whatever the block size**; 168 admits 12.

| scene | grid | block | blocks/SM at 255 | rounds | blocks/SM at 168 | rounds |
|---|---|---|---|---|---|---|
| cube-wall-cloth | 380 | 32 | 8 → 320 resident | **2** | 12 → 480 resident | **1** |
| tumbler-garments | 425 | 64 | 4 → 160 resident | **3** | 6 → 240 resident | **2** |

cube-wall buys a whole round (2 → 1) and wins 5 %. The tumbler's rounds×warps product is
unchanged (3×8 = 2×12), so no round is saved and the only thing the bound delivers is
**+4.8 KB per thread of extra spill traffic** (1 168/1 168 B → 3 472/3 680 B), which costs
12.3 %. Same GPU, same kernel, same day, opposite sign.

That also settles what the kernel is bound by: **+50 % resident warps made it 12.3 % slower
where no round was saved**, so it is *not* latency-bound and s30's `StableNeoHookean3D`
result (−21 % for exactly this bound) does not carry over. Its local-memory traffic is
expensive: +4.8 KB/thread of spill costs ~70 % more time per warp.

## 2. Stage stubbing — where the 3.45 ms actually goes

An extra `Proj = 3` instantiation that writes the **unprojected** Hessian (probe build only,
not committed), 20-frame windows, same arm for both sides:

| scene | full | no PSD projection | **projection share** |
|---|---|---|---|
| cube-wall-cloth | 1.6748 | 0.3263 | **80.5 %** |
| tumbler-garments | 3.4014 | 0.5665 | **83.3 %** |

(The 20-frame window reads ~5 % low against the full run — cube-wall full-run mean is 1.7555
against this window's 1.6748 — so treat the shares as ±1 pp. This reproduces round 4's
pre-s19 measurement of 82 %: two rounds of work on the projection have not changed its
*share*, only its absolute cost.)

## 3. Arithmetic bracket at constant occupancy

All four arms are 255 registers / 8 warps / `<<<380, 32>>>` on cube-wall-cloth, full 100-frame
runs — only the projection code changes:

| arm | env | ms/launch | vs shipped |
|---|---|---|---|
| shipped: K16 blocked 9x9 + s19 tridiagonal QL | (default) | **1.7555** | — |
| K16 blocked 9x9 + Eigen `SelfAdjointEigenSolver` | `UIPC_MAKE_SPD_JACOBI=0` | 2.0955 | **+19.4 %** |
| K7 dense 12x9 basis + QL | `UIPC_DSB_BLOCKED_PROJ=0` | 1.9590 | +11.6 % |
| dense 12x12 eigen-solve + QL | `UIPC_DSB_REDUCED_SPD=0` | 3.1773 | **+81.0 %** |

s14 (−40.6 % for the translation-free 9x9) and s19 (−16.2 % for the QL solver) are both
re-confirmed live on this box, in the shipped build, at their recorded sizes.

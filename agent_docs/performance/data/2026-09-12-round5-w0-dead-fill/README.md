# Round 5 s25 (w0-turing) — raw records for "the triplet value zero-fill is dead work"

Box: local RTX 2070 SUPER (cc 7.5), `build-perf` at `bf3457ba`, one build for both arms.
Arm switch: `UIPC_SKIP_DEAD_FILL` — `df1` = new (fill skipped, default), `df0` = old (fill on).

| file | what |
|---|---|
| `fbench_<scene>_df{0,1}_r{1,2,3}.json` | end-to-end A/B on the committed source, default frame counts, 3 runs each way, all four scenes. `reportedFrameTiming` + per-frame `newton_iterations` / `linear_solver_iterations` |
| `bench_cube-wall-cloth_df{0,1}_r{4,5,6}.json` | the extra cube-wall reps pooled into the control number (9 runs each way). Taken on the pre-gating build, which differs only in whether the converter's `row >= 0` term is a compile-time constant or a `bool` kernel argument |
| `stats_f_{c2,mb}_{on,off}_cuda_gpu_kern_sum.csv` | nsys `cuda_gpu_kern_sum`, case2 60 frames / mas-bunny 40 frames, both arms. `buffer_view_fill_kernel<Eigen::Matrix<double,3,3>>` is 245.91 ms / 330 launches in `off` and absent in `on` |
| `gate_new.log`, `gate_old_path.log`, `gate_poison_fill.log` | three of the five correctness-gate arms. The poison arm fills the values with a signalling NaN (`UIPC_FILL_PROBE=1`) instead of zero and returns the same counts, which is the sharp test that nothing reads the fill |

Probes that produced the deadness evidence (all in-tree, default off):

- `UIPC_FILL_PROBE=1` — poison fill + per-slot count of unwritten / partially written / negative-index triplets after assembly.
- `UIPC_FILL_PROBE=2` — the same check **before** assembly as well: the rig validation, which must report every slot unwritten.
- `UIPC_BCOO_HASH=<n>` — order-independent 64-bit hash of the assembled BCOO (nnz, indices, raw value bits) for the first `<n>` conversions.

The analysis is in `../../2026-09-12-perf-round5.md`, section "Step s25 (w0-turing)".

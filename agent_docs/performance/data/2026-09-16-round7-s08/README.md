# Round-7 s08 — the K16 block assembly's dead triangle (4x3) + the mirrored back-assembly

One mechanism: `make_spd_translation_free_4x3_blocked` (and a rejected 3x3 twin)
assembles its restriction only in the triangle the eigen-solvers read, and
back-assembles the symmetric result by mirroring. `UIPC_MAKE_SPD_BLOCKED_HALF=0`
is the rollback (helper-level shared name, `UIPC_MAKE_SPD_JACOBI` precedent).

## Files

| file | what it is |
|---|---|
| `poison_probe.cu/.sh`, `poison_probe_output.txt` | THE premise proof: both eigen-solvers (`evd` Eigen, `evd_tridiag_ql`) read only the LOWER triangle of their input — 2e5 NaN-poisoned random matrices per N in {6,9} per solver, 0 output words differ with the upper triangle poisoned; the lower-triangle poison control mismatches 100 % |
| `verify_asm.cu/.sh`, `verifier_output.txt` | 2x200k randomised plastic hinges, real `ddEddx` on device: cut-only arm vs old = 0/28.8M words differ (bit-identical); shipped arm's upper block-triangle vs old = 0/18M words; mirrored lower blocks relFro med 4.0-4.2e-17 max 2.1e-16 vs the old path's own solver-swap med 9.9e-16-1.1e-15 max 6.7-7.5e-15 (24-30x below); min-eig rel -6.0/-6.4e-16, null space 4.4-4.5e-16 (s01 classes) |
| `bench_helper.cu/.sh`, `bench_summary.txt` | isolated-helper microbench: 4x3 helper -9.6 % (207.9 -> 187.9 ns/mat); the 3x3 cut REJECTED at +4 % in three formulations incl. cut-only (68.5 -> 71.2-72.3 ns/mat) — documented in make_spd.h |
| `asm_probe.cu` | static SASS attribution probe (assembly-only vs +solver variants) |
| `sass_counts.txt` | SASS static FP64 counts before/after (strain 3344 -> 2897 DFMA static; stress static count UP but runtime -11 % — static is a ceiling, not a forecast) |
| `instantiation_figures.txt` | per-instantiation REG/STACK/CMEM table: every non-blocked arm identical to main under rename; every `<1,S,0>` rollback arm identical to main's `<1,S>`; `<1,S,2>` new arms (stress/dahl/hinge stacks shrink 688/688/112 B, strain flat) |
| `res_usage_{main,branch,branch_fmt}.txt` | the cuobjdump dumps; `_fmt` proves the clang-format pass is codegen-neutral (2776-function figure multiset identical) |
| `binary_diff_raw.txt` | whole-.so normalized diff: 2450 identical, 8 changed (ALL in `ipc_simplex_normal_contact` — the EE arm embeds the new helper at default; stack +80..184 B), 26/34 removed/added = the renamed FEM instantiations |
| `gate_{default,knob0,pdsb0}.txt` | correctness gate = `baseline_tests.txt` at default, at `UIPC_MAKE_SPD_BLOCKED_HALF=0`, and at `UIPC_PDSB_*=0` (pytest duration string only) |
| `nsys_{new_r1,old_r1,old_r2,new_r2}_cuda_gpu_kern_sum.csv` + `analyze_scope.py` + `scope_ab.log` | the scope A/B (ABBA x2, full 130-frame runs, graph-node tracing) |
| `nsys_cwc*` | contact-context check on cube-wall-cloth: `do_assemble<...,8>` 1922.7 us/launch vs s05's main reference 1978.0 (-2.8 %, cross-session envelope 1.2-2 %) — no contact regression; plain hinge `<3,1>` 180.9 us in-family |
| `ab_endtoend.log` + `s08n8_*.json` | crease-press end-to-end A/B n=8/arm |
| `ab_null_mas.log` + `s08null_*.json` | mas-bunny null: -0.12 % p=0.42, Newton exactly 465 x 10 |
| `ab_tumbler.log` + `s08tum_*.json` | tumbler null: +2.60 % p=0.51 at n=3 (no causal path; PCG in-arm chaos +-8 %) |
| `scripts/nsys_ab.sh`, `scripts/nsys_cwc.sh` | capture scripts |

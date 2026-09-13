# round-6 s03 evidence — the contact barrier Hessian's closed-form PSD projection

Box: RTX 2070 SUPER (cc 7.5), `build-perf`, branch `perf/round6-s03-contact-proj`
cut from `main` = `e66953ff`.

## Naming: the A/B labels predate the final switch name

The measurements were taken while the switch was still called
`UIPC_CONTACT_GAUSS_NEWTON`; it was renamed (and renumbered) to
`UIPC_CONTACT_RANK1` before commit, with **no change to any generated code
path**. The confirming runs `ab/s03_final_cwc*` were taken on the final build
with the final name and reproduce the earlier numbers.

| in these files | shipped switch | what it selects |
|---|---|---|
| `GAUSS_NEWTON=0`, nsys prefix `*_p0_*` | `UIPC_CONTACT_RANK1=0` (**the default**) | the exact Hessian + the reduced PSD projection of every round up to 5 |
| `GAUSS_NEWTON=2`, nsys prefix `*_p4_*` | `UIPC_CONTACT_RANK1=1` | the closed-form rank-1 Hessian for **PE + PP** |
| `GAUSS_NEWTON=1`, nsys prefix `*_p3_*` (and, before the rename, `*_p1_*`) | `UIPC_CONTACT_RANK1=2` | the same closed form for **all four** branches |
| `PLAIN_GN=1`, nsys prefix `*_p3_*` (first batch) | `UIPC_CONTACT_RANK1=3` | the plain Gauss-Newton coefficient `c = B''` |
| `PROJ_STUB=1`, nsys prefix `*_p2_*` | `UIPC_CONTACT_RANK1=4` | stage stub: the exact Hessian written **unprojected** (diagnosis only) |

`verify_gn0_*` / `verify_gn1_*` / `verify_gn2_*` follow the *old* numbering:
`gn0` = exact, `gn1` = all four branches, `gn2` = PE + PP.

## Files

| file | what it is |
|---|---|
| `gn_contact_probe.cu`, `gn_contact_probe.txt`, `build_probe.sh` | the numerics probe: 200 000 randomised samples per pair type, run **on the device** against the real `__device__` functions |
| `kernel_scope_rwb.txt`, `kernel_scope_tum.txt` | nsys per-kernel scope, all diagnostic arms |
| `kernel_scope_rwb_p4.txt`, `kernel_scope_tum_p4.txt` | nsys scope of the shipped opt-in mode (PE + PP) |
| `sass_identity.txt` | the old arm's SASS against main's compilation of the same TU |
| `scene_executes_kernel.txt` | which scenes actually launch the kernel (control validity, PERF_METHOD §2.2) |
| `verify_audit_ab.txt`, `verify_gn*_r*.txt` | the tumbler's `--verify` physical-soundness audit, 3 runs per arm |
| `gate.txt`, `gate_rank1_1.txt`, `gate_rank1_2.txt` | the correctness gate in three arms |
| `ab/*.json` | every A/B run's raw per-run records and statistics |
| `nsys/*_cuda_gpu_kern_sum.csv` | the raw per-kernel summaries every scope number is computed from |
| `nsysrun.sh`, `scope.py` | the instrument (fresh prefix per run, fails loudly if no csv appears) |

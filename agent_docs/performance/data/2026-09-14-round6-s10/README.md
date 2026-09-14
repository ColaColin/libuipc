# round 6 / s10 — evidence index

Step: **the ABD body-local gradient/Hessian prepass** — run `ortho_potential` (and the BDF1 kinetic)
gradient+Hessian on a side stream inside the shadow contact G+H part 1 leaves on the SMs.
Switch: `UIPC_ABD_GH_PREPASS` — `0` off (pre-s10 order), `1` shipped (kernels enqueued *after* the
contact launches), `2` the placement that does not work (enqueued *before* them).
Verify mode: `UIPC_ABD_GH_PREPASS_VERIFY=1`.

| file | what it backs |
|---|---|
| `predictions.txt` | everything predicted before any arm was built or run, plus one revision made before any s10 arm was run |
| `shadow_note.txt` | why the first placement (`=2`) fails, from the rwb timeline |
| `iter_census_rwb.txt` | the shadow census on s09's shipped-arm trace that motivated the step |
| `shadow_cwc_c2.txt` | the same census on cube-wall-cloth and stiff-gipc-case2; **case2 launches zero ABD G/H kernels** |
| `unspread_rwb.txt` | the §4 audit: every rwb kernel whose grid is below the SM count |
| `overlap_rwb.txt`, `overlap_abd.py`, `tl.py` | per-launch timeline: hidden fraction, part-1 window, empty shadow, assembly critical path |
| `scope_*.txt`, `scope.py` | nsys `cuda_gpu_kern_sum`, n=3 per arm, per-kernel totals |
| `ab/`, `abn.py` | interleaved end-to-end sweeps (rotated order, one discarded warm-up per arm) |
| `sass_identity.txt`, `sass_fn_diff.py`, `sass_dump*.sh` | per-function SASS comparison against the branch base |
| `verify_*.txt` | `UIPC_ABD_GH_PREPASS_VERIFY=1` device-side bit comparison |
| `gate_*.txt` | `gate.sh` against `baseline_tests.txt`, per arm |

Extra files: `ab/<scene>_runs.json` (one condensed record per benchmark run: arm, mean/median ms,
Newton, PCG, the env overrides that identify the arm), `ab/<scene>_summary.json` (abn.py's own
summary), `nsys/*_cuda_gpu_kern_sum.csv.gz` (the per-kernel totals behind `scope_*.txt`; the
400 MB `cuda_gpu_trace` csvs behind `overlap_rwb_final.txt` are NOT archived -- regenerate with
`tracerun.sh`), `timeline_rwb_a2*.txt` (the shipped-arm timeline at the branch base that motivated
the step). `overlap_rwb.txt` is the FIRST implementation (arm `=2` only, build2) and is kept
because `shadow_note.txt` reads from it; `overlap_rwb_final.txt` is the three-arm measurement in
the shipped build.

# round 6 / s04 — the ACCD first-pass early exit (`UIPC_CCD_EARLY_OUT`)

What each file is. Every number here was produced on the local RTX 2070 SUPER (cc 7.5) from the
single build of `perf/round6-s04-ccd`, with the two arms selected by the env switch only.

| file | what |
|---|---|
| `kernel_ranking.txt` | the family and BVH/CCD sub-family split of `tumbler-garments` GPU kernel time at the step's base (`d686bc38`), one full 180-frame run |
| `filter_toi_share.txt` | `filter_toi_k3+k4` as a share of GPU kernel time on all five scenes |
| `control_mas_bunny.txt` | control validity (PERF_METHOD §2.2): every `InfoStacklessBVH*` kernel `mas-bunny` launches, with counts — it **does** execute the code under test, 465 launches per run |
| `ccd_stats_eo0.txt`, `ccd_stats_eo1.txt` | the `UIPC_CCD_STATS=1` diagnosis counters with the switch off and on: candidate pairs per launch, ACCD loop passes per pair, hit rate, early-exit rate |
| `ccd_earlyout_probe.cu`, `.txt`, `build_probe.sh` | the numerics proof: both instantiations of all four `*_ccd` functions, same thread, same inputs, 10^6 randomised samples per pair type |
| `sass_identity.txt` | per-instantiation SASS diff of the filter's translation unit, base vs head |
| `scope_tum.txt`, `scope_cwc.txt`, `scope_rwb.txt`, `scope_c2.txt` | the targeted-scope A/B: per-launch µs and family totals, 3 full runs per arm, plus the same-run ratio normalisation |
| `ab/` | the end-to-end A/B: `ab.py` raw per-run json + summary per scene, n=5 per arm, ABBA + one discarded warm-up |
| `verify/` | the tumbler `--verify` audit, 180 frames, 3 runs per arm |
| `gate.txt`, `gate_eo0.txt` | the correctness gate in both arms |
| `nsysrun.sh`, `scope.py`, `run_scope.sh`, `run_scope2.sh`, `run_ab.sh`, `verify_cmp.py` | the harness; `nsysrun.sh` asks for `cuda_gpu_kern_sum` only (the tumbler's `cuda_gpu_trace` csv is 400 MB) and fails loudly if no csv appears |

Reproduce the two arms on any scene:

```sh
source env_perf.sh
UIPC_CCD_EARLY_OUT=0 $UIPC_PERF_PY scripts/run_benchmark.py run <scene> --python $UIPC_PERF_PY
UIPC_CCD_EARLY_OUT=1 $UIPC_PERF_PY scripts/run_benchmark.py run <scene> --python $UIPC_PERF_PY
```

and the diagnosis counters:

```sh
cd libuipc-samples/examples/95_tumbler_garments
WB_LOG=Warn UIPC_CCD_STATS=1 UIPC_CCD_EARLY_OUT=0 $UIPC_PERF_PY main.py --headless 60
```

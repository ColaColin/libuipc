# round 6 / s05 — where does the conservative candidate cull belong? (`UIPC_CCD_CULL`)

Every number here was produced on the local RTX 2070 SUPER (cc 7.5) from the single build of
`perf/round6-s05-pairfilter`, with the two arms selected by the env switch only
(`UIPC_CCD_CULL=0` = the shipped path, `=1` = the cull in the BVH leaf predicate).

**Verdict: measured and rejected.** The bound *is* computable in the leaf predicate and it *is*
conservative — it removes 91 % of the candidate array and takes 85 % off both narrow phases — but
the leaf predicate is a ~2x larger, ~48 %-divergent population than the candidate array, and
evaluating the same bound there costs more than the staging and re-reading it saves.

| file | what |
|---|---|
| `cull_stats_tumbler.txt` | `UIPC_CCD_STATS=1` cull-rate probe on the tumbler: how much of the candidate array the bound can drop |
| `cull_verify_inscene.txt` | the same probe with the two conservativeness conditions **verified in situ** over 239 million real candidate pairs |
| `broad_stats.txt` | (query, leaf) pairs the traversal **stages** against the pairs the leaf predicate **keeps** — the ratio that decides the placement |
| `ccd_cull_probe.cu`, `ccd_cull_probe.txt`, `ccd_cull_probe_5M.txt`, `build_probe.sh` | the standalone conservativeness proof against the real device functions, 5x10^6 randomised samples per pair type, half of them adversarial |
| `sass_identity.txt`, `sass_cull_twins.txt`, `sass_resusage.txt`, `sass_diff.sh`, `sass_cmp.py` | the old arm reproduces main's binary: all 13 leaf-predicate kernels byte-identical, the shipped `filter_toi` instantiations byte-identical |
| `scope_tum.txt`, `scope_c2.txt`, `scope_cwc.txt` | the targeted-scope A/B, 3 full runs per arm, per-launch µs for both kernels of the trade plus the family totals |
| `nsys/` | the per-run `cuda_gpu_kern_sum` csvs the scope tables are computed from |
| `ab/` | the end-to-end A/B (`ab.py`, ABBA + one discarded warm-up) |
| `verify/`, `verify_compare.txt` | the tumbler `--verify` audit, 180 frames, 6 runs per arm |
| `gate.txt`, `gate_c1.txt` | the correctness gate in both arms |
| `nsysrun.sh`, `scope.py`, `run_scope.sh`, `run_scope2.sh`, `run_verify.sh`, `verify_cmp.py` | the harness |

Reproduce:

```sh
source env_perf.sh
UIPC_CCD_CULL=0 $UIPC_PERF_PY scripts/run_benchmark.py run tumbler-garments --python $UIPC_PERF_PY
UIPC_CCD_CULL=1 $UIPC_PERF_PY scripts/run_benchmark.py run tumbler-garments --python $UIPC_PERF_PY
```

and the diagnosis / in-situ verification (cull **off**, so the probe sees the whole candidate
population and checks every pair the cull would have dropped):

```sh
cd libuipc-samples/examples/95_tumbler_garments
WB_LOG=Warn UIPC_CCD_STATS=1 UIPC_CCD_CULL=0 $UIPC_PERF_PY main.py --headless 60
```

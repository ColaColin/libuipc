# round-6 s06 — the compacted CCD candidate array for the DCD narrow phase

Raw evidence for the `s06` row of `agent_docs/performance/2026-09-13-perf-round6.md`.
`record_section.md` is a copy of that step's section of the round record.

Branch `perf/round6-s06-compact`, cut from `711d2300`. Box: RTX 2070 SUPER (cc 7.5).
Switch: `UIPC_CCD_COMPACT` (default 1; `=0` restores the old path).
Diagnostic: `UIPC_CCD_COMPACT_VERIFY=1`.

| file | what it is |
|---|---|
| `predictions.txt` | every prediction of this step, **written before any measurement of its own arms**, from s05's scope tables, plus the coverage correction |
| `ccd_compact_probe.{cu,txt}`, `ccd_compact_probe_5M.txt`, `build_probe.sh` | standalone conservativeness probe: 5x10^6 randomised samples per pair type, half adversarial, run against the real shipped `distance::*_ccd<..., DcdCull=true>` on the device, checking the swept segment at 221 points. 0 violations |
| `insitu_verify_tumbler.txt`, `insitu_verify_other.txt`, `run_insitu_verify.sh` | `UIPC_CCD_COMPACT_VERIFY=1` on all five scenes: `filter_active` runs twice per call (raw vs compacted) and the four active-set sizes are compared; `filter_toi` checks the compacted array element-wise against the flagged subsequence. 1.94x10^9 pairs, 0 mismatches |
| `sass_identity.txt`, `sass_resusage.txt`, `sass_cmp.py`, `sass_diff.sh` | standalone compile of the filter TU at `711d2300` and at this head with the build's exact flags: 88/88 common kernels byte-identical, 16/16 renamed `filter_toi` instantiations matched by body hash, REG 218/184 unchanged |
| `scope_all.txt`, `accounting.txt`, `cub_accounting.txt`, `scope.py`, `accounting.py` | the targeted-scope A/B: per-launch deltas, the four-kernel accounting normalised by Newton count, and the cub stream-compaction accounting |
| `rerank_c1.txt` | the family and top-kernel ranking **at this head with the switch on** — the input to the next step's pick |
| `nsys/` | the per-run `cuda_gpu_kern_sum` csv behind every scope number (fresh prefix per run, 28 runs) |
| `ab/` | end-to-end A/B: raw per-run benchmark json + the summaries (ABBA + one discarded warm-up, `/workspace/output/round6/ab.py`) |
| `verify/`, `verify_compare.txt`, `verify_cmp.py` | the tumbler `--verify` audit **and** the n=40-per-arm re-characterisation of s05's preconditioner-NaN abort. 80 runs, 0 aborts |
| `gate.txt`, `gate_c0.txt` | the correctness gate in both arms |
| `nsysrun.sh`, `run_scope.sh`, `run_scope_rwb.sh`, `run_ab.sh`, `run_verify.sh` | the harnesses, as run |

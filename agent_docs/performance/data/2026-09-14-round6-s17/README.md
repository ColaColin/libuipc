# round 6, s17 — evidence

- `predictions.txt` — written before the ab.py sweeps.
- `scope.md` — nsys per-launch numbers (old / k2 / probes) on case2 and mas-bunny; `nsys/*_cuda_gpu_kern_sum.csv` the sessions.
- `sass_identity.txt` — per-function normalised SASS diff of the three TUs against the head object (all pre-existing identical); `sass/census_gls.txt` the instruction census of the old and k2 kernels.
- `verify/new_vs_old_<scene>.txt` — `UIPC_SEG_VERIFY=1` (+ `UIPC_SEG_HIST`) on five scenes; `verify/old_vs_old_<scene>.txt` the control with the old kernel in production.
- `ab/` — ab.py outputs (`ab_*.txt`), per-run json per scene, `pooled_c2.txt` (pool.py over both case2 sweeps).
- `gate_default.txt` — gate.sh on the default arm, identical to `/workspace/output/round6/baseline_tests.txt`.
- scripts: `nsysrun.sh`, `run.sh`, `ab_chain.sh`, `scope.py`, `sass_diff.py`, `pool.py`.

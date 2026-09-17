# Round-7 s21 — the final evaluation and round close (evidence map)

Branch `perf/round7-s21-final` from main `0c1de8ef`. **Instrument step, no engine
source changed** — `git diff 0c1de8ef -- src/` empty; the head build is byte-identical
to s20's (`libuipc_backend_cuda.so` sha256 `c5470221…`, ninja no-op) and the base
worktree/build/venv is s20's kept `/workspace/output/round7/base/` (`ac16597d…`).

Arms: **base** = tag `perf-round7-base` = `fce57589` (its own venv), **head** = main
`0c1de8ef` (build-perf / `uipc-perf-env`). Both arms run the same scene code from the
same checkout (`libuipc-samples` `a47ba7d`, = s20's drift runs). Harness: `ab2.py`
(s20's two-build ab.py, default out moved to s21/), one discarded warm-up per arm,
ABBA blocks, manifest default frames, n/arm: crease-press 20, tumbler 10, cwc 10,
case2 / mas-bunny / rwb 5. 110 measured runs + 12 warm-ups in one sitting.

| file/dir | what it is |
|---|---|
| `ab/` | per-run raw jsons + `s21_<scene>_summary.json` for all six scenes |
| `ab2_cp.log` / `run_rest.log` | the two sweep drivers' full logs (per-run lines + statistics) |
| `final_table.txt` | the headline table + per-run means + per-run peak MiB (`scripts/final_table.py`) |
| `tumbler_divergence.txt` | the s20 tumbler flag resolved: per-frame first-diff onset, all pair classes + per-window profiles (`scripts/tumbler_divergence.py`) |
| `verify_{base,head}.{json,log}` | one full 130-frame `--verify` run per arm at the final binaries |
| `kernelsnap_cuda_gpu_kern_sum.csv` + `kernel_ranking_cp.txt` | the closing kernel-ranking snapshot: one full-run nsys at head (graph-node tracing; `kernelsnap.nsys-rep`/`.sqlite` stay in `/workspace/output/round7/s21/`, 145 MB) |
| `gate_head.txt` | the correctness gate at head before any benchmark ran |
| `scripts/` | `ab2.py`, `run_rest.sh` (five scenes + verify + nsys), the three analyzers |

## The verify arms and the knife-edge check

Base draw: `verify_ok` true, all 13 checks. Head draw: `verify_ok` **false** on
`no_inversion_or_collapse` — the failing sub-predicate is
`max(sheet_area_ratio_max_final) < 5.0` (head 7.60 on sheet 7 vs base 4.50; the
transient max is 16.7-17.3 in BOTH arms). This is s05's documented knife-edge pair
(it flipped in 2/10 exact-arm and 3/5 perturbed-exact-arm runs there, vs 1/10 GN),
and s20's head draw flipped the same check. Every non-marginal instrument is in band
in both arms: 0 non-converged / 0 newton-limit / 0 ls-limit frames, Newton 590/578,
PCG 124.5k/105.6k (s00 envelope 102-126k), dahl F_commit 0.9998/0.9997, crease
|F|/M 0.0635/0.0648, stress yield 2.95/2.97 %, residual creases 35.1-48.4 /
35.9-48.2 mm. Note the base draw's own strain yield 0.21 % is OUTSIDE the s00 band
(0.04-0.13 %) — the observables wander past the n=5 bands on their own, which is
why s05's controlled paired design, not single draws, is the physics evidence.

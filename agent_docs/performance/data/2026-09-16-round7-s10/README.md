# round 7 / s10 — the second re-profile + drift audit (instrument step, nothing ships)

Every file here backs the s10 section of
`agent_docs/performance/2026-09-15-perf-round7.md`. Branch
`perf/round7-s10-reprofile` from main `18d9a41e` (the s09 merge). **No engine
source touched** — the step re-profiles the head after s07 (NHS2D −44 %/launch)
and s08 (K16 dead-triangle cut) shifted the GPU mix, re-prices s09's two
host-stall sites, measures the accumulated drift since the round base, and
produces the ranked pick list for s11–s19.

## Files

| file | what it is |
|---|---|
| `binary_identity.txt` | the .so sha at head + the relink note + the per-instantiation REG/STACK checks against s08's table |
| `kernel_ranking_cp.txt`, `s10_cp_rank_cuda_gpu_kern_sum.csv` | **the fresh full-run ranking** (130 frames, graph-node tracing, fresh prefix, s05's family predicates verbatim) |
| `s10_cp_rank.run.log` | the capture's run log (frames/counts of that specific run) |
| `drift_analysis.txt`, `drift_head_[1-8].json` | accumulated drift: n=8 default runs vs `baseline-runs/crease-press.json` (the REQUIRED meanFrameMs line) |
| `verify_head.json`, `verify_head.log` | one full 130-frame `--verify` run at head: `verify_ok` true, all 13 checks, regime observables vs s00 |
| `stall/d2h_sites_crease-press.txt`, `stall/cp_d2hprof.log` | stall map part 1: `UIPC_D2H_PROFILE=2` full run, per-call-site funnel stalls symbolized with addr2line |
| `stall/s10_cp_api_analysis4.txt`, `stall/s10_cp_api_{cuda_gpu_kern_sum,cuda_api_sum}.csv` | stall map part 2: 12-frame nsys api+gpu window (s09's `analyze_api4.py` verbatim) — the converter park split |
| `gate_head.txt` (copy of `/workspace/output/round7/s10/gate_head.txt`) | correctness gate at head: identical to `baseline_tests.txt` (pytest duration string only) |
| `scripts/` | everything re-runnable (`run_drift.sh`, `run_verify.sh`, `run_d2hprof.sh`, `symbolize_d2h.py`, `nsysrun.sh`, `analyze_ranking_s10.py`, `apitrace.sh`, `analyze_api4.py`) |

Not archived: the `.nsys-rep`/sqlite exports and the 226 MB per-launch gpu
trace csv (the txts above carry every number quoted in the record) — they stay
under `/workspace/output/round7/s10/`.

## Headline numbers

- Ranking sanity anchors all inside ±2 %: dahl 469.9 µs/launch (anchor 461.9,
  +1.7 % — inside the documented ±1.2 % cross-session per-launch drift plus a
  566-vs-542-launch count draw), NHS2D 1335.7 (1334.1, +0.1 %), strain 1194.0
  (1193.6, +0.03 %), stress 1217.7 (1216.8, +0.07 %). Per-element: strain
  125.6 ns/hinge (= s08's 125.6), stress 128.1 (128.0), dahl 16.5 (16.2),
  NHS2D 103.6 ns/tri (103.5).
- **REQUIRED drift line: meanFrameMs 262.15 → 236.92 ms (−9.62 %)** — baseline
  n=1 vs head n=8 (cv 6.52 % against the s00 floor's 4.34 %); vs the s00
  floor mean 277.93: **−14.75 %**. `ms_per_newton` 61.74 → 54.62 (−11.53 %).
- Stall map: PCG convergence readback 24,689 × 5.96 µs = **147.1 ms/run ≈
  0.45 % of wall** (s09: 26,824 / 158 ms / 0.50 %); MatrixConverter count
  readbacks park the host **24.76 %** of a 13.68 s window (s09: 19–25 %),
  99.8 % drain, GPU-idle inside the park **0.00 ms**. Nothing new above 0.3 %.
- Regime: `verify_ok` true, all 13 checks pass; dahl F fraction 0.9999
  (s00 0.9996), crease |F|/M 0.0661 (0.064–0.067), stress yield 2.90 %
  (2.2–3.4 %), strain yield 0.063 % (0.04–0.13 %), residual creases 34–49 mm
  on every one of the 7 sheets (s00: 37–48). Physics-unchanged stands.

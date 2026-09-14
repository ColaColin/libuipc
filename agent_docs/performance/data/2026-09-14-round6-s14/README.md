# round 6 / s14 — the contact assembly's join, deferred to the first reader

Every file here backs a claim in `agent_docs/performance/2026-09-13-perf-round6.md`, section **s14**.

| file | claim it backs |
|---|---|
| `predictions.txt` | P1–P7, written before the changed binary was measured. P1, P2, P5 (cwc, rwb flat) held; P3 half held (SNH early in 76 % not 90 %; part 1 +18 % not 0–5 %); **P4 held** (mode 1 −1.26 % inside −0.8 to −1.6; mode 2 at the edge of its range and a tie); P5's `mas-bunny` "vacuous" was wrong — it is the mechanism 21 times; P6's `UIPC_BCOO_HASH` half was unusable (instrument not deterministic); **P7 was wrong** — the reader's GPU position is never ahead of the kernels |
| `read_set.txt` | the code-derived list of every consumer of the collected contact gradient/Hessian, with file:line, and where the join sits in front of each |
| `census_c2_head.txt` | the premise at this head: zero `distribute_*` launches on case2, part 1 8 blocks × 879 µs, 541 µs/iteration idle inside its window (2.2 % ceiling), SNH 378 µs after part 1 ends, 0 % coverage |
| `census_c2_m1.txt`, `census_c2_m2.txt` | the mechanism with the join deferred: SNH 248 µs *before* part 1 ends, 76 % of iterations, window 61 % covered |
| `kmeans_c2.txt` | per-kernel µs/launch in the three case2 traces (within-run only): part 1 +18 %, SNH +0.4 %, strain limiting +8 %, half-plane +15 % |
| `census_rwb.txt` | rwb full-run traces both arms: part 1's window 54.2 → 55.7 % covered, all of it the ABD prepass; no friction kernels on the scene |
| `verify_runs.log` | `UIPC_CONTACT_DEFERRED_JOIN_VERIFY=1` on six scene/mode runs: 1.475 × 10⁹ words, 0 mismatching after the join, 0 stale before it, 86–100 % pending at host time |
| `sass_identity.txt` | eight touched TUs compiled at head and at `8ab46206`: every pre-existing function byte-identical; the contact TU gains one 32-instruction verify kernel in one added hunk |
| `results_c2.txt`, `ab/s14c2_*_summary.json` | the four-arm case2 sweep, n=12/arm (`abn.py`): mode 1 −1.26 % ms/Newton p=4.1e-06, mode 2 −1.27 %, null +2.0 % from one solver-stall run (`null_r11`, PCG 101 830) |
| `results_mb.txt` | `mas-bunny` n=8: −0.25 %, p=0.0025, Newton 465 / LS 465 in all 16 runs |
| `results_cwc.txt` | `cube-wall-cloth` n=8: +0.02 % ms/Newton (p=0.97) — the fallback path |
| `results_rwb8.txt`, `results_rwb16.txt` | rwb: −1.54 % (p=0.02, guard firing) at n=8 → −0.67 % (p=0.088) at n=16 |
| `results_tum.txt` | tumbler n=5: unresolved (+0.9 %, p=0.76, PCG +4 %) |
| `gate_m1.txt`, `gate_off.txt`, `gate_m2.txt` | `gate.sh` vs `baseline_tests.txt`, identical counts in all three arms |
| `tverify_stats.txt` | tumbler `--verify` n=6/arm, `verify_ok` 12/12, everything overlapping |
| `bcoo_hash_*.txt`, `bcoo_hash_repeat.txt` | the `UIPC_BCOO_HASH` instrument is non-deterministic run-to-run at this head (two `=0` runs differ) — the "found outside" item |

Scripts: `tracerun.sh` (per-launch timeline, fresh prefix, fails loudly), `census.py` /
`census_rwb.py` / `kmeans.py` (trace analysis, mechanism only), `verify_runs.sh`, `sweep_c2.sh`
(uses s11's `abn.py`, copied here), `sweep_rest.sh`, `post_gate.sh`, `run_verify_tumbler.sh`,
`verify_cmp.py`, `sass_check.sh`. Raw per-run json not archived (per-arm summaries are).

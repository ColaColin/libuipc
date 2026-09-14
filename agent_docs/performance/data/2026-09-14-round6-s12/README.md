# round 6 / s12 — the FEM gradient/Hessian prepass, refuted

Every file here backs a specific claim in
`agent_docs/performance/2026-09-13-perf-round6.md`, section **s12**.

| file | claim it backs |
|---|---|
| `predictions.txt` | the predictions, written before any A/B arm ran (P1 ceiling, P2 why the ABD mechanism cannot repeat, P3 the copy cost, P4 counts, P5 the alternative) |
| `kernsum_c2_head.txt` | the `stiff-gipc-case2` per-kernel ranking at head 48f1c6e3 (full 250-frame run). `StableNeoHookean3D` G/H is **4729.2 ms, 11.33 %, 2845 us per launch**; contact part 1 is **1423.0 ms, 856 us**; part 2 **709.2 ms, 427 us** |
| `underfill_c2.txt` | the occupancy under-fill census — the whole case2 GPU timeline is **94.5 %** filled on first-wave occupancy; the total idle-machine-equivalent is **451.8 ms of 8170.9 ms = 5.5 %**, and contact assembly is **202 ms** of it. This is the ceiling on any overlap step on this scene |
| `timeline_c2_head.txt` | the launch geometry and co-residency: SNH is **1249 blocks of 128 at 168 reg** (3 blocks/SM on cc 7.5 -> 10.4 waves, saturating) and runs with **zero** co-resident kernels; contact part 1 is **8 blocks median** and is co-resident only with part 2, for 35.3 % of its window |
| `gap_c2.txt` | what runs between contact part 1's end and SNH's start, and between SNH's end and the first read of the contact Hessian — the shadow a *contact postpass* (the candidate) could use |
| `results_c2.txt`, `ab/` | the probe A/B on `stiff-gipc-case2`, n=8/arm, ABBA, one discarded warm-up per arm |
| `results_mb.txt` | the probe A/B on `mas-bunny` (deterministic FEM, Newton 465 every run) |
| `null_c2.txt` | the null arm — two bit-identical arms in the same sweep, the instrument check |
| `sass_identity.txt` | the touched TU compiled at head and at 48f1c6e3 with the identical command line and diffed per function: **0 changed, 0 removed**, two head-only kernels that only the probe can reach |
| `gate_off.txt`, `gate_probe.txt` | `gate.sh` vs `baseline_tests.txt` in both arms |

Scripts: `nsysrun.sh` (kern_sum), `tracerun.sh` (per-launch timeline), `sass_check.sh`.

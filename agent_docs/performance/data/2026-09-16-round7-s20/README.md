# Round-7 s20 — the validation pass (evidence map)

Branch `perf/round7-s20-validation` from main `69c2af51`. **Instrument step, no
engine source changed** (gate at head byte-identical to `baseline_tests.txt`
modulo pytest's duration string; `head_so_sha256.txt` = the head build's
fingerprints, `base_so_sha256.txt` = the base build's).

Arms:
- **head** = main `69c2af51`, build-perf / `uipc-perf-env` (`libuipc_backend_cuda.so`
  sha256 `c5470221…`).
- **base** = tag `perf-round7-base` = `fce57589`, separate worktree + build dir +
  venv at `/workspace/output/round7/base/` (`libuipc_backend_cuda.so` `ac16597d…`,
  round-6 V3's two-build pattern). The base arm's warm-up runs reproduce the
  round's baseline references: crease-press 262.78 vs `baseline-runs` 262.15 ms
  (+0.24 %), mas-bunny 61.46 vs 62.39 (−1.5 %, cross-session box drift; all
  drift comparisons below are interleaved same-session, so internally valid).

| file/dir | what it is |
|---|---|
| `driver.log` | every sanitizer run's one-line census (rc, duration, summary) |
| `summary.md` | the full sanitizer matrix table + per-kernel breakdowns (`summarize.py`) |
| `logs_truncated/` | first 200 records of every log that reported anything (full logs: `/workspace/output/round7/s20/logs/`, 506 MB) |
| `scripts/` | `san.sh` (the wrapper), `matrix_{head,base,repair}.sh` (the three phases + the redo/repair arms), `chain_*.sh`, `summarize.py`, `collect.sh` |
| `drift/` | Part B: per-run raw jsons + `ab2.py` summaries for all six scenes (`drift_table.md` is the digest) |
| `verify_{base,head}.{json,log}` | the full 130-frame `--verify` regime runs, one per arm |
| `envaudit/` | Part C: one kern_sum csv per knob-OFF arm (+ default), `k_pcg_poll{def,0}.sqlite` for the doorbell's api-trace evidence; `audit_table.md` is the digest |
| `gate_head.txt` | the correctness gate at head |

## The three process incidents (all recovered, all documented in `driver.log`)

1. **The cp racecheck processes die mid-frame-1** (head 21:47:34 rc=9, base
   00:35:20; "process didn't terminate successfully", no application traceback —
   consistent with an external kill, cause unidentified, host OOM the leading
   suspect). Each death left the GPU in `cudaErrorDevicesUnavailable` for the
   next ~1-60 s and poisoned exactly one following cell each time (0-1 s runs,
   2 x error 46): head rwb memcheck, base cp memcheck, base rwb memcheck — all
   three redone cleanly in `matrix_repair.sh` phase A4a-c (0 errors each).
2. **memcheck cannot fully track crease-press on this 8 GB card**: every
   non-probe cp memcheck arm reports 100 % "Internal Sanitizer Error … (Unable
   to allocate enough memory)" records (head 30f: 101 177; base 30f: 106 101;
   probe arms 77 806-167 137; `--force-synchronization-limit 1` repair arms
   74 346/83 799 — the option does not cure it). These are instrument failures,
   not application defects; **zero application-level memcheck findings in every
   arm**, and the exhaustion reproduces on the base build (pre-existing).
3. The one probe arm that tracked fully (`UIPC_DAHL_GN_VERIFY=1` memcheck,
   0 internal errors, 0 errors) is arm-specific, not a graphs-off effect (the
   segred/doublet/poll verify arms all exhaust).

## Coverage statement (Part A's limits)

- crease-press sanitizer windows are 30 frames; the scene's phase schedule
  scales with the frame count (`_phase_len = int(f*N)`), so a 30-frame run
  executes the ENTIRE two-cycle press/hold/lift/shift schedule compressed
  (press1=0-2, hold1=3, lift1=4-6, shift=7, press2=8-10, hold2=11, lift2=12-14,
  settle=15-29) — both yield paths and the dahl hysteresis loops execute, at 5x
  the design press speed. memcheck/initcheck covered all 30 frames (initcheck
  fully tracked: 0 internal errors).
- racecheck on crease-press completed frame 0 only (both arms; the process dies
  in frame 1). The round's changed kernels are all `SHARED:0` in `cuobjdump
  -res-usage` (dahl `<3,1,2>` 210 reg, plastics `<1,1,2>` 255, NHS2D `<1,1>` 255,
  the TimeIntegrator commits, `fused_pcg_scalar` 32, every converter k2/ks) —
  racecheck is vacuous for them by construction; the shared-memory families
  (MAS setup, contact, BVH) are unchanged code covered by the five regression
  scenes' full 6f windows and match the base count-for-count.
- The five regression scenes use round-6's counts: memcheck 12f, initcheck 8f,
  racecheck 6f — all completed.

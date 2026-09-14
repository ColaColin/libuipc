# Round-6 V2 validation pass — evidence index

Decides whether `UIPC_CONTACT_RANK1=1` (contact part 2, the closed-form rank-1 Hessian on the PE
**and** PP branches) may become the default. Merged and default-off since s03/s07; worth −3.23 %
ms/Newton on `cube-wall-cloth` (s07), re-measured here at −3.81 %; blocked on `verify_area_ratio_max`
on the tumbler, a statistic sighted in five steps without resolving.

Build: `83ea553c` (= round-6 `main` after s08), one RTX 2070 SUPER, cc 7.5.
Scene changes: `libuipc-samples` branch `perf/round6-v2` (`fa5c3b9`, `6764ad3`).

## The two documents that were written before the measurements they govern

| file | what it fixes, and when |
|---|---|
| **`VERDICT_RULE.md`** | the arms, the statistic, the test, the thresholds and what each outcome means — **committed at `877d0cf9`, before the first arm ran** |
| **`TRUNCATION_PREDICTION.md`** | the reading of the `cloth_min_y` follow-up — written **after** that observable was seen to move but **before** the diagnostic that explains it ran, and it says so |

## Results

| file | what it backs |
|---|---|
| `analysis_n20.txt` | the pre-registered analysis, four arms × 20 runs: §3 safety, §4 the membrane family, §5 the run maximum three ways, §6 the 36-observable screen and the joint signature, §7 cost |
| `membrane_ci.txt` | 20 000-resample bootstrap CIs on the membrane effect size — the envelope that proves the verdict |
| `pe_severity.txt`, `pe_severity_probe.cu` | the device probe: 1.27 M PE samples, where the rank-1 form loses accuracy and **which direction of curvature it drops** |
| `micro.txt`, `contact_micro.py`, `micro/` | the contact-severity micro-test built for this change (PE 100 % / PP 0 % of part-2 pairs), five configurations × 5 runs × 2 arms |
| `crease.txt`, `crease_micro.py`, `crease/` | V1's crease micro-test run against this change as a null (contact is disabled there) |
| `divergence.txt` | per-frame trajectory divergence, exact-vs-rank-1 against the physically-equivalent envelope |
| `cost_cwc.txt` | the cost claim re-measured at this head, `cube-wall-cloth` n = 20, four arms |
| `cost_cminy.txt` | the `cloth_min_y` follow-up at n = 40 |
| `cost_trunc.txt`, `cost_trunc100.txt`, `truncation.txt` | the truncation diagnostic: both arms at the default tolerance, at 10× tighter and at 100× tighter, and the trend that decides it |
| `envaudit.txt` | kernel-level proof that the three arms run the three code paths |
| `gate_rank1_0.txt`, `gate_rank1_1.txt`, `gate_default.txt` | `gate.sh` against `baseline_tests.txt` in all three arms |
| `record_section.md` | the round-record section this directory backs |

## Raw data

| file | what it is |
|---|---|
| `runs/<arm>_<rep>.json` | the `VERIFY_RESULT` summary of each of the 80 tumbler runs |
| `traces.json.gz` | the per-frame `VERIFY_TRACE` of all 80 runs (14 480 frames), gzipped — this is what §5's per-frame distribution analysis reads, and it is the instrument this pass added |
| `sweep.log`, `micro_sweep.log`, `crease_sweep.log` | the run logs of the three sweeps |
| `scripts/` | everything that produced the numbers |

Position dumps (18 runs × 40 MB) stayed in scratch; `divergence.txt` is what they produced.

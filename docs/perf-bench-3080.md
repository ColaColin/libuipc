
### 8.3 Instance change

Instance 50470541 (Argentina host) went `offline` at 12:5x local during the
mps5 two-garment job and did not return within 30 min; destroyed at 13:27
(`absent: true`, estimated $0.86 spent). Lost with it: the queued fine-timer
run, the K6/K7 A/Bs and the thread-percentage / pinning rows (re-run on the
replacement, section 8.4). Replacement: offer 49767341 (RTX 3080 10 GB,
Malaysia, 12 cores, reliability 0.991, $0.10/h — no US/EU 3080 with ≥ 8
cores and ≥ 0.99 was on offer at the time). Noise floor and baseline
references are re-established on it (one baseline pair of 120-frame probes
plus 600-frame baselines and MPS ×3), so every comparison stays within one
GPU.
| K8 | radix sorts of the triplet/doublet keys over `bit_width(rows·cols)` bits instead of 64 (key = `row·cols + col`, CUB `begin_bit/end_bit`); global system, symmetric compaction, dytopo matrix and doublet sorts | bit-identical (same order of distinct keys, stable for equal keys) | built as perf11; A/B queued (phase E) |
| K6 600 frames / MPS ×3 | — | 8.04 / 15.03 / 17.45 (**−10 / −5.9 / −5.0 %**) | 26.8 / 7.98 / 6.09 (+7 / +2 / +12 %) | — | accepted |
| K6+K7 (perf10) | 7.39 / 15.66 / 16.77 | 7.72 / 14.56 / 16.64 (**−14 / −8.8 / −9.4 %**) | (mps3 running) | identical to the noise pair (towel 3.7e-6 / 1.5e-5 / 0.35; tshirt 0.28 / 1.4 / 3.5; jacket+shorts 0.63 / 6.2 / 10.2) | **accepted** — same wheel, K7 off → on: tshirt 15.84 → 15.66, jacket+shorts 17.93 → 16.77; hinge G+H scope 35.8 → 28.5 (tshirt), 35.6 → 26.8 ms/frame (jacket+shorts) |

Host note: the replacement host rebooted at 14:18 UTC (uptime shows it),
killing the queue mid-phase C; 4 h of idle instance time ($0.40) until the
watchdog-less monitor was noticed; MPS restarted and the remaining phases
(K7 tail, thread-percentage / pinning, K8) relaunched at 18:12 UTC.

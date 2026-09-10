
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

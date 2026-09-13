# round-6 s07 — the PE:PP split, and the Newton drift that held s03's mode 1 back

What this directory holds, and which claim each file backs.

| file | claim |
|---|---|
| `predictions.txt` | the end-to-end effect of every arm, predicted from the measured scope **before** any end-to-end run, plus the decision rule for the drift question, fixed in advance |
| `sass_identity.txt` | the old arm reproduces main's binary: **67 / 67** common SASS streams in the TU byte-identical to `a3061ef5`, including both instantiations the scenes launch; the only new functions are the four `Proj = 5 / 6` ones |
| `sass_resusage.txt` | registers per instantiation: part 2 `Proj=0` **255**, `Proj=5` **252** (the shipped default — no occupancy change), `Proj=1`/`=6` **154** (a large occupancy jump, which is why mode 1 is fast and why its transfer is not purely algorithmic) |
| `part2_static.txt` | static SASS instruction + FP64 counts per `Proj` mode for part 1 and part 2 |
| `gn_contact_probe.{cu,txt}` | s03's device probe re-run at this head: the PP closed form reproduces the shipped 6x6 projection to **4.754e-16 mean / 1.451e-15 max** relative Frobenius error over 140 470 samples, `lambda_max` ratio 1.000000 exactly; PE is 9.78e-05 mean / **1.900e-02 max** |
| `scope_cwc.txt`, `scope_c2.txt`, `scope_rwb.txt`, `scope_tum.txt` | full-run nsys, 3 runs per arm, four arms (`p0`/`p1`/`p5`/`p6`), four scenes — the PE:PP attribution |
| `ab/counts_cwc.log`, `ab/counts_c2.log`, `ab/counts_rwb.log` | the interleaved 5-arm / 4-arm count sweeps (n=10 / 10 / 15) |
| `ab/tail_cwc.log` | the n=40-per-arm cube-wall-cloth sweep that characterises the Newton tail |
| `tail_pooled.txt` | the pooled n=50-per-arm tail table and the Fisher test: **the tail event occurs on the exact path too** |
| `drift_ci.txt` | Welch 95 % CI on the Newton-count difference per arm — on `stiff-gipc-case2` the `p1` drift is bounded to **[-0.135 %, +0.207 %]** |
| `verify/`, `verify_compare.txt`, `verify_stats.txt` | tumbler `--verify`, 180 frames, **8 runs per arm**, three arms; every safety observable passes in all 24, and `verify_area_ratio_max` is the one statistic that moves under `p1` |
| `gate.txt`, `gate_p0.txt`, `gate_p1.txt` | the correctness gate in the shipped default (`=5`), in the rollback (`=0`) and in the opt-in (`=1`) — identical assertion counts to `baseline_tests.txt` in all three |
| `default_selects_mode5.txt` | kernel-level env-switch audit: with no env var the part-2 launch is `<..., (int)5>`, with `UIPC_CONTACT_RANK1=0` it is `<..., (int)0>`, and part 1 is `(int)0` in both |
| `abn.py` | the N-arm interleaved harness (rotated + reversed arm order, one discarded warm-up per arm, `ab.py`'s hard checks) |
| `nsys/` | the raw `cuda_gpu_kern_sum` csv of every profiled run, fresh prefix per run |

Arms throughout: `p0` = `UIPC_CONTACT_RANK1=0` (exact), `pnull` = `=00` (bit-identical to
`p0`; `std::atoi("00") == 0`), `p1` = `=1` (PE+PP), `p5` = `=5` (PP only, the new default),
`p6` = `=6` (PE only).

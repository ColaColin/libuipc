# round 6 — s11 evidence: the ABD G/H prepass placement, and the instrument that could not rank it

Each file below is named against the claim it backs in
`agent_docs/performance/2026-09-13-perf-round6.md` § s11.

## The claims and their files

| claim | file |
|---|---|
| Predictions recorded **before** any s11 measurement, including the P2 headline ("the gain replaces s10's, it does not stack") and the P3 call on the `=2` instrument | `predictions.txt` |
| Every end-to-end number, all five arms, five scenes, with the null envelope beside each | `results.txt` |
| `rigid-wrecking-balls` n=30, 5 arms, all 10 pairwise Welch tests — this is what says modes 1, 2 and 4 are indistinguishable there | `arm_vs_arm.txt` |
| `cube-wall-cloth` n=10, 5 arms, all 10 pairwise tests — this is what says mode 1 is worth nothing there and mode 4 is worth −2.1 % | `arm_vs_arm_cwc.txt` |
| **The instrument resolution**: the nsys GPU busy union, normalised per Newton iteration, n=4 per arm, beside the end-to-end column. Spearman rho = +0.10 | `union_noise.txt` |
| The mechanism: which kernel ran when, per arm — the prepass goes from 0 % inside contact part 1's window (modes 0/1/2) to 99.0 % (mode 3) and 91.5 % (mode 4), and part 1's empty shadow falls 430 → 101 ms | `overlap_rwb.txt` |
| The repeat traces behind the union table | `overlap_reps.txt` |
| Bit-identity, 379 696 512 output words compared on device, 0 mismatching, over modes 3, 4 and the shipped default | `verify_prepass.txt` |
| Compile-level identity: the three touched TUs at this head vs the branch base, SASS per function. The only four differing instructions are `__LINE__` immediates of `UIPC_KERNEL_ASSERT`, shifted by two added `#include` lines | `sass_identity.txt` |
| Correctness gate at the new default, at `=0` and at `=3`, vs `baseline_tests.txt` | `gate_default4.txt`, `gate_off.txt`, `gate_m3.txt` |
| Raw sweep output and per-run summaries | `ab/` |

## The sweeps in `ab/`

| file | scene | n | arms |
|---|---|---|---|
| `rwb.txt` | rigid-wrecking-balls | 10 | 0,1,2,3,4 + null — the first pass; this is where mode 3 first read as a null |
| `rwb30.txt` | rigid-wrecking-balls | 30 | 0,1,2,4 + null — **the decisive one** |
| `rwb_f3.txt` | rigid-wrecking-balls | 20 | 0,1,3 + null — mode 3 confirmed at −1.97 % |
| `cwc.txt` | cube-wall-cloth | 10 | 0,1,2,3,4 — **the ranking inverts here** |
| `cwc2.txt` | cube-wall-cloth | 12 | 0,1,4 + null — confirmation with a null arm |
| `mb.txt` | mas-bunny | 6 | 0,1,4 — control (Newton 465 in all 18 runs) |
| `c2.txt` | stiff-gipc-case2 | 4 | 0,1,4 — control; mode 4 read −0.46 % "DISJOINT" here |
| `c2b.txt` | stiff-gipc-case2 | 6 | 0,4 + null — the same control at n=6: **+0.12 %, p=0.58**. The n=4 reading reversed sign |
| `tum.txt` | tumbler-garments | 5 | 0,1,4 — unresolved (sd is 8 % of the mean); no claim |
| `rwb_final.txt`, `cwc_final.txt` | both | 12 | 0,1,4 + the true default arm — re-measured on the **shipped** binary after the default flip and the shared-header refactor |
| `rwb_null.txt` | rigid-wrecking-balls | 20 | `=4` vs the unset default vs `=04` — the env-switch audit: +0.21 % / +0.08 %, both unresolved |

## Scripts

`abn.py` (s07's N-arm interleaved harness, rotated + reversed order, one discarded
warm-up per arm), `run_ab*.sh` (the sweeps in the order they were run),
`tracerun.sh` + `run_trace*.sh` (nsys, fresh prefix per run, fails loudly on a
missing csv), `overlap_abd.py` (the per-launch timeline analysis, extended by s11
to normalise the union per Newton iteration and to report how much of contact
part 2 hides inside part 1), `run_verify*.sh`, `sass_check.sh`.

Not archived: the raw `nsys-rep`/`sqlite`/`cuda_gpu_trace.csv` files (1.9 GB) and
the SASS dumps (860 MB). Both are reproducible from the scripts above; the
scratch copies are in `/workspace/output/round6/s11/`.

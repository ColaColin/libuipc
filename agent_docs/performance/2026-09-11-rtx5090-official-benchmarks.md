# libuipc end-to-end benchmark suite on a rented RTX 5090 — main `df57bfb8` vs upstream base `4d1f3f34` (+ intermediate `e1eed4b9`)

Date: 2026-09-11. Scratch/raw data: `agent_docs/performance/data/2026-09-11-rtx5090/` (this file, `raw/full/<tree>/` = the 60 archived `output/benchmark-runs/` JSON+log records, `raw/timers/`, `raw/timers-records/`, `raw/profile/`, `raw/tests/`, remote build/setup logs, all helper scripts).

## Summary

- **HEAD `df57bfb8` (current main) is 17–34 % faster per frame than the upstream merge-base `4d1f3f34` on all four canonical benchmarks on an RTX 5090**, at unchanged Newton / line-search / PCG counts per frame and with final-state observables inside the same run-to-run envelope. Peak device memory is 11–23 % lower.
- Attribution to the three work items (median of 5 run means, ms/frame): rigid 118.2 → 98.6 (`4d1f3f34`→`e1eed4b9`, perf round 1 + Dahl feature) → 92.8 (`e1eed4b9`→`df57bfb8`, perf rounds 2/3); case2 186.6 → 138.5 → 123.7; MAS bunny 51.8 → 45.3 → 42.9; wall+cloth 109.5 → 90.6 → 81.8.
- All 60 throughput runs (3 trees × 4 scenes × 5 runs) returned 0, every frame `completed=true` and `converged=true`, no Newton or line-search limit hit anywhere.
- The `uipc.profile` baseline created from the `4d1f3f34` runs (10 % allowance) is passed by HEAD on all four benchmarks and all diagnostic counters.
- Against the recorded 2026-09-01 5090 reference (Windows/WDDM, driver 595.79, CUDA 13.2, commit `3e08e005`), HEAD is 28–39 % faster and even `4d1f3f34` is 7–14 % faster — the platform differs (Linux, headless, driver 580.173, CUDA 12.8 runtime, different host), so that comparison is indicative only; the A/B numbers within this run are the controlled result.
- Instance `50602520` was destroyed at 15:51:47Z (API: instance list empty, `show instance` returns null). Rental 14:41:00Z–15:51:47Z = 1.18 h at $0.4222/h ≈ **$0.50** (+ ≈$0.01 bandwidth); account credit went 11.07 → 10.57 (≈ $0.50 charged).

## Trees compared

| label | commit | what | backend `libuipc_backend_cuda.so` SHA-256 |
|---|---|---|---|
| `base` | `4d1f3f3446631b283c14b2500b14b0eeb93ac6f4` | upstream `spiriMirror/libuipc` merge-base of main ("Merge pull request #492 from spiriMirror/refactor-main"); primary baseline per owner | `b3a954ec…7387b67` |
| `dahl` | `e1eed4b929c2d8361f7266e4239cf7417494876e` | end of perf round 1 + Dahl bending feature (17 commits after base; the previous "baseline" of the perf docs); optional intermediate point | `4d3737b4…d87273cb` |
| `head` | `df57bfb8a5bc0e22b2de81aa44ae203301b5b0bb` | current `main` (31 commits after `e1eed4b9`: perf/kernels rounds 2/3, K1–K19). Kernel code identical to `2f47bf50`; the two later commits are docs + a CUDA unit-test expectation only | `aeb61c1e…e25d45e0` |

`libuipc-samples` at `4fb26b7171e53858d6980cfa8b6debbdb61c83fa` (`benchmark-baseline` branch, the gitlink recorded by all three commits) for all trees; `benchmarks/manifest.json` and `scripts/run_benchmark.py` are byte-identical between `4d1f3f34` and `df57bfb8`, so the same runner/manifest was used everywhere. All three source trees were clean (`dirty=false`, no untracked paths in the records). Note that the `4d1f3f34..e1eed4b9` range is not pure kernel work: it also contains `9efc3bcd feat(collision_detection): enable dcd_candidate_reuse by default` and the reduced-SPD projection commits, i.e. default-behaviour changes; the iteration counters below show they did not change Newton/PCG counts on these scenes.

## Environment (instance)

| Field | Value |
|---|---|
| vast.ai instance / offer | `50602520` / offer `50550567`, label `bench5090-libuipc`, on-demand (not interruptible), host 623333 machine 145598, California US |
| Price | $0.400/h GPU + $0.0222/h disk = **$0.4222/h**; bandwidth $5.33/TB (≈50 GB down billed) |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, sm_120, PCIe gen4 x16 (26.5 GB/s reported), no display attached (`gpu_display_active=false`), idle before each run (2 MiB used) |
| Driver / CUDA | driver **580.173.02** (CUDA 13.0 capable); toolkit **nvcc 12.8.93** (`nvidia/cuda:12.8.1-devel-ubuntu24.04` image); backends are self-contained CUDA 12.8 runtimes (`uipc doctor`: "no system CUDA Toolkit is required") |
| Host | AMD EPYC 7K62 48-core (192 hw threads visible, cgroup quota 23.04 CPUs, 183 GB memory limit), Ubuntu 24.04.1, GCC 13.3.0, CMake 3.28.3, Ninja 1.11.1, CPython 3.12.3 |
| Build (identical for all trees) | `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/work/vcpkg/scripts/buildsystems/vcpkg.cmake -DUIPC_CUDA_ARCHITECTURES=native -DUIPC_BUILD_PYBIND=ON -DUIPC_PYTHON_EXECUTABLE_PATH=/work/venv-src-<tree>/bin/python -DUIPC_BUILD_EXAMPLES=OFF -DUIPC_BUILD_TESTS=OFF -DUIPC_BUILD_BENCHMARKS=OFF -DUIPC_DEV_MODE=ON`; `cuobjdump --list-elf` shows exactly one `sm_120` cubin and no PTX per backend; vcpkg (microsoft/vcpkg master, 4.5 min install, shared binary cache) per tree; each tree's `pyuipc 0.9.0` installed into its own venv by the post-build step; installed `.so` hash == `build/Release/bin` hash for every tree |
| Build times | head 14:58–15:11 (incl. vcpkg), base 15:11–15:17, dahl 15:17–15:23 (`-j24`) |
| Runner environment | `UIPC_BENCHMARK_TIMERS=0`, `WB_LOG=Warn`, `NO_MAS/NO_GRAPH/WB_TIMER` unset (manifest canonical env); nothing else running on the GPU or host during the throughput runs |

## Method

1. Three source trees checked out from a git bundle of the local repo (no push anywhere), samples cloned from `spiriMirror/libuipc-samples` at the recorded gitlink; built as above.
2. Verification per tree: `uipc doctor` (all OK, sm_120 native), `python/uipc_info.py` (OK after installing X11 libs for polyscope), and `run_benchmark.py run <each scene> --quick` (4 × 3-frame smoke runs per tree, all rc=0).
3. Throughput: `python scripts/run_benchmark.py run <scene> --python /work/venv-src-<tree>/bin/python` at default frames (120 / 250 / 100 / 100), **5 rounds**, each round runs all four scenes for one tree then the next tree, with the tree order rotated every round (`head,base,dahl` → `base,dahl,head` → …), 15:25:01Z–15:44:56Z. Per-frame times are the scene-reported `world.advance(); world.retrieve();` wall times; "process wall" is the runner's whole-process duration (includes Python/CUDA start-up and scene build).
4. One `UIPC_BENCHMARK_TIMERS=1` diagnostic per scene per tree with the frame counts of the 2026-09-01 doc (100 / 60 / 60 / 70), 15:45–15:48Z.
5. `uipc.profile` gate: records converted into result dirs (`to_profile_dirs.py`, median-mean run per scene, `wall_time = Σ frame_ms`), `python -m uipc benchmark baseline <base dirs> --max-regression-percent 10`, then `check` for head and dahl (`--allow-environment-mismatch`, since `uipc_version` is 0.9.0 for all but the commit differs).
6. Add-on (after all benchmarks): tests enabled by incremental reconfigure (`-DUIPC_BUILD_TESTS=ON`, targets `sim_case backend_cuda`, ~50 s each), ran the requested cases.
7. Instance destroyed; destruction verified through the API.

## Throughput results (5 runs per tree per benchmark)

"mean ms/f" cells are mean / median / min / max over the five run means. Peak-memory delta = peak total device memory during the run minus pre-launch total (`nvidia-smi` sampled every 0.2 s; GPU otherwise idle, so this is close to process-private usage here, unlike the WDDM reference).

| bench | tree | n | mean ms/f (mean/median/min/max of run means) | sum-frame total s (mean) | process wall s (mean) | frame median (median) | p95 (median) | Newton/f | LS/f | PCG/f | peak mem delta MiB (min-max) | vs 2026-09-01 ref ms/f |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| rigid-wrecking-balls | head (df57bfb8) | 5 | 92.97 / 92.78 / 92.12 / 93.93 | 11.16 | 12.3 | 93.06 | 150.1 | 3.98 | 4.10 | 107.8 | 5546-5624 | 129.5 (-28.4% median-of-means) |
| rigid-wrecking-balls | base (4d1f3f34) | 5 | 118.55 / 118.23 / 115.69 / 122.67 | 14.23 | 15.3 | 116.04 | 197.5 | 3.95 | 4.05 | 108.1 | 6660-6708 | 129.5 (-8.7% median-of-means) |
| rigid-wrecking-balls | dahl (e1eed4b9) | 5 | 97.88 / 98.56 / 95.22 / 99.51 | 11.75 | 12.9 | 96.75 | 164.4 | 3.98 | 4.07 | 107.1 | 5928-5936 | 129.5 (-23.9% median-of-means) |
| stiff-gipc-case2 | head (df57bfb8) | 5 | 124.07 / 123.68 / 122.73 / 125.19 | 31.02 | 36.1 | 127.55 | 162.6 | 6.63 | 7.15 | 259.0 | 3772-3806 | 201.1 (-38.5% median-of-means) |
| stiff-gipc-case2 | base (4d1f3f34) | 5 | 186.83 / 186.56 / 186.25 / 187.52 | 46.71 | 51.8 | 193.14 | 238.4 | 6.64 | 7.12 | 257.1 | 4870-4888 | 201.1 (-7.2% median-of-means) |
| stiff-gipc-case2 | dahl (e1eed4b9) | 5 | 138.21 / 138.50 / 137.71 / 138.59 | 34.55 | 39.6 | 143.56 | 177.8 | 6.64 | 7.13 | 258.5 | 4106-4152 | 201.1 (-31.1% median-of-means) |
| mas-bunny | head (df57bfb8) | 5 | 42.77 / 42.85 / 42.31 / 43.00 | 4.28 | 7.1 | 46.54 | 69.7 | 4.65 | 4.65 | 352.2 | 3568-3568 | 60.2 (-28.8% median-of-means) |
| mas-bunny | base (4d1f3f34) | 5 | 52.01 / 51.79 / 51.77 / 52.37 | 5.20 | 8.0 | 55.64 | 82.5 | 4.65 | 4.65 | 352.0 | 4022-4022 | 60.2 (-14.0% median-of-means) |
| mas-bunny | dahl (e1eed4b9) | 5 | 45.31 / 45.31 / 45.05 / 45.71 | 4.53 | 7.3 | 49.30 | 72.9 | 4.65 | 4.65 | 352.1 | 3246-3246 | 60.2 (-24.7% median-of-means) |
| cube-wall-cloth | head (df57bfb8) | 5 | 83.43 / 81.80 / 79.62 / 90.98 | 8.34 | 10.3 | 73.64 | 186.1 | 5.21 | 5.43 | 201.1 | 3568-3662 | 125.6 (-34.9% median-of-means) |
| cube-wall-cloth | base (4d1f3f34) | 5 | 110.37 / 109.53 / 107.34 / 113.15 | 11.04 | 12.9 | 106.43 | 243.7 | 5.07 | 5.27 | 198.1 | 4658-4768 | 125.6 (-12.8% median-of-means) |
| cube-wall-cloth | dahl (e1eed4b9) | 5 | 90.04 / 90.56 / 88.16 / 92.08 | 9.00 | 10.9 | 82.84 | 200.1 | 5.07 | 5.27 | 196.6 | 3882-3992 | 125.6 (-27.9% median-of-means) |


### Change, median of the five run means

| bench | head ms/f | base ms/f | head vs base | dahl ms/f | head vs dahl |
|---|---:|---:|---:|---:|---:|
| rigid-wrecking-balls | 92.78 | 118.23 | -21.5% | 98.56 | -5.9% |
| stiff-gipc-case2 | 123.68 | 186.56 | -33.7% | 138.50 | -10.7% |
| mas-bunny | 42.85 | 51.79 | -17.3% | 45.31 | -5.4% |
| cube-wall-cloth | 81.80 | 109.53 | -25.3% | 90.56 | -9.7% |


Equivalent per-item attribution (median of run means, ms/frame):

| bench | base `4d1f3f34` | dahl `e1eed4b9` | head `df57bfb8` | base→dahl | dahl→head | base→head |
|---|---:|---:|---:|---:|---:|---:|
| rigid-wrecking-balls | 118.23 | 98.56 | 92.78 | −16.6 % | −5.9 % | **−21.5 %** |
| stiff-gipc-case2 | 186.56 | 138.50 | 123.68 | −25.8 % | −10.7 % | **−33.7 %** |
| mas-bunny | 51.79 | 45.31 | 42.85 | −12.5 % | −5.4 % | **−17.3 %** |
| cube-wall-cloth | 109.53 | 90.56 | 81.80 | −17.3 % | −9.7 % | **−25.3 %** |

Run-to-run spread of the run means is small on this headless Linux box: ≤ 2 % for rigid/case2/MAS on every tree, ≤ 6 % on wall+cloth (one head run at 90.98 vs 79.6–83.4 for the other four — the same dynamic-contact envelope the 2026-09-01 doc describes); the head-vs-base gaps (17–34 %) are an order of magnitude larger than the spread.

### Iteration counters and correctness

- Newton / line-search / PCG per frame (5-run means) are equal within noise between trees: rigid 3.98/4.10/107.8 (head) vs 3.95/4.05/108.1 (base); case2 6.63/7.15/259.0 vs 6.64/7.12/257.1; MAS 4.65/4.65/352.2 vs 4.65/4.65/352.0; wall+cloth 5.21/5.43/201.1 vs 5.07/5.27/198.1. The speed-up is therefore per-iteration cost, not fewer iterations.
- 0 of 12 000+ recorded frames hit `hit_newton_limit` or `hit_line_search_limit`; all frames `completed=true`, `converged=true`; all 60 runs `returnCode=0`.
- `uipc benchmark check` (baseline = `4d1f3f34`, 10 % allowance): **PASS** for head on all four benchmarks (`wall_time_ms_per_frame` 92.78 ≤ 130.06, 123.68 ≤ 205.21, 42.85 ≤ 56.96, 81.80 ≤ 120.48; Newton/LS/PCG medians all ≤ baseline·1.1) and PASS for dahl. Files: `raw/profile/baseline-4d1f3f34.json`, `check-head.json`, `check-dahl.json`.

### Final-state observables (must match to tolerance)

| bench | observable | tree | min over 5 runs | max over 5 runs | max per-axis spread |
|---|---|---|---|---|---|
| cube-wall-cloth | cloth_min_y | head | 0.2790 | 0.2982 | 1.92e-02 |
| cube-wall-cloth | cloth_min_y | base | 0.2798 | 0.2935 | 1.37e-02 |
| cube-wall-cloth | cloth_min_y | dahl | 0.2841 | 0.2972 | 1.31e-02 |
| mas-bunny | bunny_centroid | head | -0.1459 -0.7863 0.0552 | -0.1459 -0.7863 0.0552 | 1.12e-06 |
| mas-bunny | bunny_centroid | base | -0.1459 -0.7863 0.0552 | -0.1459 -0.7863 0.0552 | 7.80e-07 |
| mas-bunny | bunny_centroid | dahl | -0.1459 -0.7863 0.0552 | -0.1459 -0.7863 0.0552 | 1.66e-06 |
| rigid-wrecking-balls | ball_center | head | 7.9790 2.4555 -0.1685 | 8.2280 2.7889 0.0951 | 3.33e-01 |
| rigid-wrecking-balls | ball_center | base | 7.9941 2.4054 -0.1362 | 8.2759 2.6195 -0.0427 | 2.82e-01 |
| rigid-wrecking-balls | ball_center | dahl | 7.9792 2.4946 -0.1100 | 8.2027 2.8623 0.0517 | 3.68e-01 |
| stiff-gipc-case2 | lower_centroid | head | -0.0148 -0.8866 0.0509 | 0.0055 -0.8806 0.1247 | 7.38e-02 |
| stiff-gipc-case2 | lower_centroid | base | -0.0066 -0.8871 0.0521 | -0.0013 -0.8828 0.1091 | 5.70e-02 |
| stiff-gipc-case2 | lower_centroid | dahl | -0.0070 -0.8854 0.0497 | 0.0005 -0.8814 0.0830 | 3.33e-02 |
| stiff-gipc-case2 | upper_centroid | head | -0.7579 -0.8671 -0.4520 | -0.6278 -0.8586 -0.3993 | 1.30e-01 |
| stiff-gipc-case2 | upper_centroid | base | -0.7437 -0.8687 -0.4295 | -0.5954 -0.8561 -0.4122 | 1.48e-01 |
| stiff-gipc-case2 | upper_centroid | dahl | -0.8408 -0.8694 -0.4489 | -0.6711 -0.8539 -0.3927 | 1.70e-01 |

Interpretation: MAS bunny is the deterministic sentinel — the three trees' final centroids agree to **≤ 1.7e-6 m** (25 runs total; head-vs-base difference is inside each tree's own 1e-6 spread). The collision-rich scenes are not run-to-run deterministic on any tree (the 2026-09-01 doc records the same behaviour); the head envelopes overlap the base envelopes on every axis (rigid ball centre x 7.98–8.23 vs 7.99–8.28, y 2.46–2.79 vs 2.41–2.62, z −0.17–0.10 vs −0.14–−0.04; case2 upper centroid x −0.76–−0.63 vs −0.74–−0.60; cloth min-y 0.279–0.298 vs 0.280–0.294; the 2026-09-01 reference envelopes were 8.009–8.213 / 2.459–2.811 / −0.083–0.052 and 0.2776–0.2926). **No divergence flagged.**

## Comparison with the recorded 2026-09-01 RTX 5090 reference

Reference: `agent_docs/performance/2026-09-01-cross-domain-baseline.md`, commit `3e08e005` (refactor-main, before `4d1f3f34`), Windows 11 + WDDM display GPU, driver 595.79, CUDA 13.2 toolkit, CPython 3.14, median of three run means.

| bench | ref (Win/WDDM, 595.79, CUDA 13.2) | base `4d1f3f34` here | head `df57bfb8` here | head vs ref | ref Newton/PCG per frame | here (head) |
|---|---:|---:|---:|---:|---|---|
| rigid-wrecking-balls | 129.5 | 118.2 (−8.7 %) | 92.8 | **−28.4 %** | 3.95 / 107.3 | 3.98 / 107.8 |
| stiff-gipc-case2 | 201.1 | 186.6 (−7.2 %) | 123.7 | **−38.5 %** | 6.64 / 246.8 | 6.63 / 259.0 |
| mas-bunny | 60.2 | 51.8 (−14.0 %) | 42.9 | **−28.8 %** | 4.67 / 358.8 | 4.65 / 352.2 |
| cube-wall-cloth | 125.6 | 109.5 (−12.8 %) | 81.8 | **−34.9 %** | 5.13 / 200.1 | 5.21 / 201.1 |

The 7–14 % that `4d1f3f34` already gains over the reference is a platform effect (Linux headless vs WDDM desktop sharing the GPU, driver 580 vs 595, CUDA 12.8 vs 13.2 codegen, different CPU) plus whatever changed between `3e08e005` and `4d1f3f34`; it cannot be separated here. Only the within-run A/B is controlled. Iteration counts match the reference closely, confirming the same scene contract (the case2 PCG/frame of 259 vs 247 is within the doc's own run-to-run variation of that counter). Reference peak-memory deltas (5450–5930 / 4181–7516 / 3243–3245 / 3909–7282 MiB) include desktop activity; here base needed 6660–6708 / 4870–4888 / 4022 / 4658–4768 MiB and head 5546–5624 / 3772–3806 / 3568 / 3568–3662 MiB (head is 11–23 % below base; MAS bunny is the exception where `dahl` at 3246 MiB is lower than both — K18/K19 changed growth policy after `e1eed4b9`, but the MAS scene's 3568 MiB on head is +322 MiB vs dahl; worth a look if memory matters there).

## Synchronized stage diagnostics (`UIPC_BENCHMARK_TIMERS=1`, one run each; ms per Newton call, parent/child rows inclusive — do not add)

| Benchmark (frames / Newton calls) | tree | timer-run mean ms/frame | Newton | Global solve | FusedPCG | Subsystem assembly | Line search | Trajectory detect | DyTopo | inner DCD detect |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rigid-wrecking-balls (100 / 360) | head | 92.1 | 24.34 | 17.93 | 8.37 | 6.41 | 4.82 | 3.17 | 1.57 | 1.09 |
| rigid-wrecking-balls (100 / 346) | dahl | 95.1 | 26.13 | 18.47 | 8.48 | 6.34 | 5.74 | 3.61 | 1.91 | 1.21 |
| rigid-wrecking-balls (100 / 352) | base | 113.8 | 31.01 | 18.42 | 8.34 | 6.36 | 5.77 | 3.64 | 3.89 | 2.92 |
| stiff-gipc-case2 (60 / 331) | head | 100.4 | 17.45 | 9.58 | 4.38 | 2.54 | 6.07 | 4.12 | 1.78 | 0.66 |
| stiff-gipc-case2 (60 / 330) | dahl | 108.5 | 18.94 | 9.85 | 4.44 | 2.49 | 7.02 | 4.50 | 2.06 | 0.69 |
| stiff-gipc-case2 (60 / 328) | base | 145.9 | 25.89 | 10.93 | 4.52 | 2.67 | 7.10 | 4.53 | 4.53 | 3.32 |
| mas-bunny (60 / 283) | head | 43.1 | 8.84 | 6.78 | 4.60 | 0.58 | 1.76 | 1.19 | 0.28 | 0.23 |
| mas-bunny (60 / 283) | dahl | 45.7 | 9.40 | 7.05 | 4.58 | 0.53 | 2.07 | 1.45 | 0.27 | 0.22 |
| mas-bunny (60 / 283) | base | 52.3 | 10.81 | 7.50 | 4.57 | 0.63 | 2.04 | 1.44 | 0.45 | 0.80 |
| cube-wall-cloth (70 / 391) | head | 83.9 | 14.50 | 9.77 | 5.84 | 2.75 | 3.13 | 1.57 | 1.58 | 0.32 |
| cube-wall-cloth (70 / 391) | dahl | 92.4 | 16.01 | 10.13 | 5.79 | 2.71 | 4.07 | 1.94 | 1.78 | 0.35 |
| cube-wall-cloth (70 / 391) | base | 111.4 | 19.39 | 10.21 | 5.78 | 2.71 | 4.05 | 1.91 | 3.29 | 1.81 |

Reading: relative to base, head saves most in **DyTopo** (contact G/H assembly + distribution: 3.9→1.6 rigid, 4.5→1.8 case2, 3.3→1.6 wall) and **inner DCD detect** (2.9→1.1, 3.3→0.7, 0.8→0.2, 1.8→0.3 — the `dcd_candidate_reuse` default and BVH refit/self-cull work), then in **Line search / trajectory detect** (rigid 5.8→4.8, case2 7.1→6.1, wall 4.1→3.1) and Newton overhead outside the listed scopes; the **global solve / FusedPCG / subsystem assembly** are essentially unchanged (FusedPCG 8.3–8.5 rigid, 4.4–4.5 case2, 4.6 MAS, 5.8 wall on all trees). Timer scopes synchronize, so these locate cost and are not a substitute for the throughput table (timer-run means: 92.1/100.4/43.1/83.9 head vs 113.8/145.9/52.3/111.4 base at 100/60/60/70 frames). The reference doc's rows (Newton 33.6/27.9/12.7/21.8 for rigid/case2/MAS/wall at `3e08e005` on WDDM) sit above base here (31.0/25.9/10.8/19.4).

## Add-on: requested correctness tests on both builds (nvcc 12.8.93, sm_120, driver 580.173.02)

| test | head `df57bfb8` | base `4d1f3f34` |
|---|---|---|
| `uipc_test_sim_case "74_abd_revolute_joint_external_force"` | **PASS** (401 assertions) | **PASS** (401 assertions) |
| `uipc_test_sim_case "80_abd_revolute_joint_driving_and_external_torque"` | **PASS** (1201 assertions) | **PASS** (1201 assertions) |
| `uipc_test_backend_cuda "lbvh"` ×3 | pass, **FAIL**, pass | **FAIL, FAIL, FAIL** |

- The local sm_75 device trap in `torque_to_F` (cases 74/80, CUDA error 719) does **not** reproduce on sm_120 with the same nvcc 12.8 on either commit — it is not a regression introduced by our commits, and appears architecture/host-specific.
- The `lbvh` flake reproduces on **both** trees on the 5090 (so it is pre-existing upstream behaviour, not ours): always the same assertion `apps/tests/backends/cuda/lbvh.cu:255 CHECK(diff.empty())` in the `lbvh_query_point`-vs-brute-force section (`test.size()=101802` vs `ground_truth.size()=94728`, extra pairs such as `0 7925`, `0 7932`, …). Logs: `raw/tests/{head,base}_lbvh_{1,2,3}.log`, `raw/tests/*_simcase_*.log`, `raw/tests/tests.log`.

## Deviations, problems, and things to know

- **Baseline change mid-run**: the brief's baseline was `e1eed4b9`; per the owner's later message the primary baseline is `4d1f3f34` and `e1eed4b9` is the optional intermediate. Both were built and run; the deliverable comparison is head vs `4d1f3f34`.
- **Prebuilt wheels not used.** `/workspace/deps/wheels/pyuipc-0.9.0+dahle1eed4b9` and `+perf2f47bf50` do contain sm_120 SASS (cp312), but no wheel exists for `4d1f3f34`, and my upstream link to the instance was ~60 KB/s (a 130 MB wheel upload would have taken > 30 min; the first `scp` of the ship directory dropped after 7 min with only the 19 MB source bundle complete). All three trees were therefore built from source on the instance with one identical configuration, which is also the cleaner A/B. Samples came from GitHub (`spiriMirror/libuipc-samples`, read-only clone); nothing was pushed to any remote.
- **`scripts/run_benchmark.py` issues found (unchanged since `4d1f3f34`, not fixed here — tracked source was not modified):** (a) `resolve_python()` calls `Path(value).resolve()`, which follows a venv's `bin/python` symlink to `/usr/bin/python3.12` and silently runs the scene outside the venv (`ModuleNotFoundError: numpy`) — worked around by replacing the venv symlinks with copies of the interpreter (`fix_venv.sh`); the records show the venv path as `command[0]`. (b) The raw log is written with `log_path.write_text()` before `write_metadata()` creates `output/benchmark-runs/`, so the first run in a fresh checkout dies with `FileNotFoundError` — worked around with `mkdir -p output/benchmark-runs` (ignored path; records still show `untracked: []`).
- `python/uipc_info.py` imports `polyscope`, which needs `libX11.so.6`/`libGL` even headless; installed `libx11-6 libgl1 libxrender1 libxext6 libxi6 libxrandr2 libxcursor1 libxinerama1 libxxf86vm1 libglu1-mesa` on the instance, after which it ran on all three trees. `uipc doctor` had passed before that.
- `uipc doctor` reports `code path unknown` / `[WARN] sm_120 against [native]` for native builds (expected per its hint; the cubin list confirms sm_120 SASS).
- Driver/CUDA on the instance (580.173, CUDA 12.8 toolkit) differ from the 2026-09-01 doc (595.79, CUDA 13.2); a CUDA-13.2-driver 5090 at ≤ $0.60/h existed (Thailand, pcie 13.9 GB/s, reliability 0.973) but the chosen US host had better PCIe/reliability/price; the toolkit is the same 12.8.93 the local production wheels are built with.
- `frame_stats` schema has `linear_solver_iterations` (total PCG iterations per frame, all Newton solves) — used as "PCG/frame" above, matching the doc's definition.
- The `uipc.profile` check compares one run per scene (the run with the median mean) and required `--allow-environment-mismatch` because the baseline/check builds are different commits with the same `0.9.0` version string; environment facts otherwise match.
- Time budget: 1.18 h of rental against the ~4 h allowance; nothing planned was skipped.

## Exact commands (instance)

```
# setup: apt deps, source trees from bundle (df57bfb8 / 4d1f3f34 / e1eed4b9), samples 4fb26b7   -> remote_setup.sh
# build (per tree, identical):                                                                      -> remote_build.sh
python3 -m venv /work/venv-src-$t && pip install numpy pybind11 pybind11-stubgen
python scripts/gen_vcpkg_json.py build --dev_mode=OFF --with_usd_support=OFF --with_vdb_support=OFF --with_cuda_backend=ON
/work/vcpkg/vcpkg install --x-manifest-root=build --x-install-root=build/vcpkg_installed --triplet x64-linux
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/work/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DUIPC_CUDA_ARCHITECTURES=native -DUIPC_BUILD_PYBIND=ON -DUIPC_PYTHON_EXECUTABLE_PATH=/work/venv-src-$t/bin/python \
  -DUIPC_BUILD_EXAMPLES=OFF -DUIPC_BUILD_TESTS=OFF -DUIPC_BUILD_BENCHMARKS=OFF -DUIPC_DEV_MODE=ON
cmake --build build -j24
# verify
/work/venv-src-$t/bin/python -m uipc doctor ; python python/uipc_info.py ; python scripts/run_benchmark.py run <scene> --quick --python ...
# throughput (remote_bench3.sh full 5 "head base dahl"): rotating tree order, per scene
UIPC_BENCHMARK_TIMERS=0 /work/venv-src-$t/bin/python scripts/run_benchmark.py run <scene> --python /work/venv-src-$t/bin/python
# stage diagnostics (remote_timers.sh)
... run <scene> --frames {100|60|60|70} --env UIPC_BENCHMARK_TIMERS=1
# uipc.profile gate
python -m uipc benchmark baseline /work/results/profile/base/* -o baseline-4d1f3f34.json --max-regression-percent 10
python -m uipc benchmark check /work/results/profile/head/* --baseline baseline-4d1f3f34.json --allow-environment-mismatch
# tests add-on (remote_tests.sh): cmake -S . -B build -DUIPC_BUILD_TESTS=ON && cmake --build build -j24 --target sim_case backend_cuda
./uipc_test_sim_case "74_abd_revolute_joint_external_force" ; ./uipc_test_sim_case "80_abd_revolute_joint_driving_and_external_torque" ; ./uipc_test_backend_cuda "lbvh"  (x3)
```
Local: `vastai create instance 50550567 --image nvidia/cuda:12.8.1-devel-ubuntu24.04 --disk 80 --ssh --direct --label bench5090-libuipc`; `vastai destroy instance 50602520 --yes`; analysis `analyze.py raw/full head,base,dahl`, `parse_timers.py`.

## Rental accounting and destruction

| | |
|---|---|
| created | 2026-09-11T14:41:00Z (contract 50602520) |
| destroyed | 2026-09-11T15:51:47Z (`vastai destroy instance 50602520 --yes`) |
| duration | 1.18 h |
| rate | $0.4222/h (GPU $0.40 + disk $0.022) |
| cost | ≈ $0.50 compute + ≈ $0.01 bandwidth (account credit 11.07 → 10.57 after destruction) |
| verification | `vastai show instances --raw` → `[]`; `vastai show instance 50602520 --raw` → `{"instances": null}` |

## Appendix A — per-run table (all 60 throughput runs)

| bench | tree | run | frames | sum frame_ms (s) | process wall (s) | mean ms/f | median | p95 | Newton/f | LS/f | PCG/f | peak mem delta MiB | converged |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rigid-wrecking-balls | head (df57bfb8) | 20260911T152501Z | 120 | 11.07 | 12.3 | 92.25 | 93.10 | 168.5 | 3.96 | 4.05 | 107.5 | 5546 | True |
| rigid-wrecking-balls | head (df57bfb8) | 20260911T153131Z | 120 | 11.27 | 12.4 | 93.93 | 93.71 | 150.1 | 3.99 | 4.07 | 108.0 | 5624 | True |
| rigid-wrecking-balls | head (df57bfb8) | 20260911T153351Z | 120 | 11.05 | 12.2 | 92.12 | 91.89 | 155.3 | 3.94 | 4.05 | 106.7 | 5564 | True |
| rigid-wrecking-balls | head (df57bfb8) | 20260911T153628Z | 120 | 11.13 | 12.2 | 92.78 | 91.85 | 142.9 | 4.00 | 4.18 | 107.3 | 5594 | True |
| rigid-wrecking-balls | head (df57bfb8) | 20260911T154302Z | 120 | 11.25 | 12.3 | 93.78 | 93.06 | 147.1 | 3.98 | 4.13 | 109.4 | 5574 | True |
| rigid-wrecking-balls | base (4d1f3f34) | 20260911T152608Z | 120 | 14.08 | 15.2 | 117.30 | 116.04 | 194.2 | 3.90 | 3.98 | 108.0 | 6688 | True |
| rigid-wrecking-balls | base (4d1f3f34) | 20260911T152850Z | 120 | 14.19 | 15.3 | 118.23 | 115.64 | 205.4 | 3.95 | 4.07 | 108.2 | 6660 | True |
| rigid-wrecking-balls | base (4d1f3f34) | 20260911T153459Z | 120 | 13.88 | 15.0 | 115.69 | 113.50 | 192.0 | 3.90 | 4.02 | 107.0 | 6682 | True |
| rigid-wrecking-balls | base (4d1f3f34) | 20260911T153736Z | 120 | 14.72 | 15.8 | 122.67 | 116.88 | 207.1 | 4.05 | 4.14 | 109.6 | 6708 | True |
| rigid-wrecking-balls | base (4d1f3f34) | 20260911T154019Z | 120 | 14.26 | 15.3 | 118.85 | 117.18 | 197.5 | 3.97 | 4.06 | 107.6 | 6694 | True |
| rigid-wrecking-balls | dahl (e1eed4b9) | 20260911T152738Z | 120 | 11.87 | 13.0 | 98.94 | 96.64 | 167.6 | 4.02 | 4.13 | 107.5 | 5932 | True |
| rigid-wrecking-balls | dahl (e1eed4b9) | 20260911T153019Z | 120 | 11.83 | 13.0 | 98.56 | 96.84 | 169.4 | 3.97 | 4.03 | 109.4 | 5928 | True |
| rigid-wrecking-balls | dahl (e1eed4b9) | 20260911T153239Z | 120 | 11.43 | 12.6 | 95.22 | 93.55 | 133.5 | 3.90 | 4.01 | 103.6 | 5930 | True |
| rigid-wrecking-balls | dahl (e1eed4b9) | 20260911T153906Z | 120 | 11.66 | 12.8 | 97.17 | 96.75 | 152.7 | 3.95 | 4.03 | 105.7 | 5932 | True |
| rigid-wrecking-balls | dahl (e1eed4b9) | 20260911T154149Z | 120 | 11.94 | 13.0 | 99.51 | 98.40 | 164.4 | 4.05 | 4.14 | 109.5 | 5936 | True |
| stiff-gipc-case2 | head (df57bfb8) | 20260911T152514Z | 250 | 30.68 | 35.6 | 122.73 | 127.55 | 159.0 | 6.62 | 7.14 | 255.0 | 3774 | True |
| stiff-gipc-case2 | head (df57bfb8) | 20260911T153144Z | 250 | 31.30 | 36.4 | 125.19 | 130.12 | 165.0 | 6.63 | 7.17 | 263.8 | 3776 | True |
| stiff-gipc-case2 | head (df57bfb8) | 20260911T153404Z | 250 | 30.90 | 36.0 | 123.61 | 127.32 | 162.6 | 6.64 | 7.09 | 254.1 | 3772 | True |
| stiff-gipc-case2 | head (df57bfb8) | 20260911T153641Z | 250 | 31.29 | 36.4 | 125.15 | 128.17 | 162.5 | 6.64 | 7.17 | 261.3 | 3806 | True |
| stiff-gipc-case2 | head (df57bfb8) | 20260911T154315Z | 250 | 30.92 | 35.9 | 123.68 | 124.58 | 166.9 | 6.63 | 7.15 | 260.6 | 3784 | True |
| stiff-gipc-case2 | base (4d1f3f34) | 20260911T152624Z | 250 | 46.86 | 51.9 | 187.43 | 197.81 | 238.4 | 6.64 | 7.14 | 257.9 | 4886 | True |
| stiff-gipc-case2 | base (4d1f3f34) | 20260911T152906Z | 250 | 46.56 | 51.6 | 186.25 | 193.14 | 236.9 | 6.63 | 7.06 | 252.7 | 4888 | True |
| stiff-gipc-case2 | base (4d1f3f34) | 20260911T153514Z | 250 | 46.64 | 51.9 | 186.56 | 192.62 | 238.8 | 6.64 | 7.10 | 256.5 | 4872 | True |
| stiff-gipc-case2 | base (4d1f3f34) | 20260911T153753Z | 250 | 46.60 | 51.6 | 186.41 | 191.61 | 234.4 | 6.64 | 7.15 | 258.5 | 4870 | True |
| stiff-gipc-case2 | base (4d1f3f34) | 20260911T154035Z | 250 | 46.88 | 51.9 | 187.52 | 194.45 | 242.0 | 6.65 | 7.14 | 260.3 | 4882 | True |
| stiff-gipc-case2 | dahl (e1eed4b9) | 20260911T152751Z | 250 | 34.43 | 39.5 | 137.73 | 143.09 | 174.5 | 6.64 | 7.11 | 256.9 | 4106 | True |
| stiff-gipc-case2 | dahl (e1eed4b9) | 20260911T153032Z | 250 | 34.63 | 39.6 | 138.52 | 143.56 | 175.6 | 6.66 | 7.18 | 256.7 | 4152 | True |
| stiff-gipc-case2 | dahl (e1eed4b9) | 20260911T153252Z | 250 | 34.43 | 39.5 | 137.71 | 140.58 | 185.2 | 6.65 | 7.12 | 257.4 | 4140 | True |
| stiff-gipc-case2 | dahl (e1eed4b9) | 20260911T153920Z | 250 | 34.65 | 39.8 | 138.59 | 144.77 | 178.9 | 6.64 | 7.11 | 261.8 | 4126 | True |
| stiff-gipc-case2 | dahl (e1eed4b9) | 20260911T154202Z | 250 | 34.63 | 39.7 | 138.50 | 145.00 | 177.8 | 6.62 | 7.14 | 259.7 | 4116 | True |
| mas-bunny | head (df57bfb8) | 20260911T152550Z | 100 | 4.23 | 7.1 | 42.31 | 46.08 | 69.0 | 4.65 | 4.65 | 352.2 | 3568 | True |
| mas-bunny | head (df57bfb8) | 20260911T153221Z | 100 | 4.29 | 7.1 | 42.89 | 46.59 | 69.6 | 4.65 | 4.65 | 352.2 | 3568 | True |
| mas-bunny | head (df57bfb8) | 20260911T153440Z | 100 | 4.28 | 7.2 | 42.85 | 46.54 | 70.0 | 4.65 | 4.65 | 351.9 | 3568 | True |
| mas-bunny | head (df57bfb8) | 20260911T153717Z | 100 | 4.28 | 7.1 | 42.82 | 46.45 | 69.7 | 4.65 | 4.65 | 352.4 | 3568 | True |
| mas-bunny | head (df57bfb8) | 20260911T154351Z | 100 | 4.30 | 7.1 | 43.00 | 46.65 | 69.7 | 4.65 | 4.65 | 352.4 | 3568 | True |
| mas-bunny | base (4d1f3f34) | 20260911T152716Z | 100 | 5.18 | 8.0 | 51.77 | 55.64 | 82.5 | 4.65 | 4.65 | 352.1 | 4022 | True |
| mas-bunny | base (4d1f3f34) | 20260911T152958Z | 100 | 5.18 | 7.9 | 51.79 | 55.63 | 82.6 | 4.65 | 4.65 | 352.1 | 4022 | True |
| mas-bunny | base (4d1f3f34) | 20260911T153606Z | 100 | 5.24 | 8.0 | 52.37 | 56.28 | 83.1 | 4.65 | 4.65 | 351.9 | 4022 | True |
| mas-bunny | base (4d1f3f34) | 20260911T153845Z | 100 | 5.18 | 7.9 | 51.77 | 55.55 | 82.5 | 4.65 | 4.65 | 351.9 | 4022 | True |
| mas-bunny | base (4d1f3f34) | 20260911T154127Z | 100 | 5.24 | 8.0 | 52.37 | 56.15 | 82.5 | 4.65 | 4.65 | 352.1 | 4022 | True |
| mas-bunny | dahl (e1eed4b9) | 20260911T152831Z | 100 | 4.51 | 7.2 | 45.05 | 48.88 | 72.6 | 4.65 | 4.65 | 352.1 | 3246 | True |
| mas-bunny | dahl (e1eed4b9) | 20260911T153112Z | 100 | 4.51 | 7.2 | 45.07 | 48.95 | 72.9 | 4.65 | 4.65 | 351.9 | 3246 | True |
| mas-bunny | dahl (e1eed4b9) | 20260911T153332Z | 100 | 4.57 | 7.4 | 45.71 | 49.55 | 73.5 | 4.65 | 4.65 | 352.2 | 3246 | True |
| mas-bunny | dahl (e1eed4b9) | 20260911T154000Z | 100 | 4.54 | 7.3 | 45.43 | 49.34 | 73.4 | 4.65 | 4.65 | 352.1 | 3246 | True |
| mas-bunny | dahl (e1eed4b9) | 20260911T154243Z | 100 | 4.53 | 7.3 | 45.31 | 49.30 | 72.7 | 4.65 | 4.65 | 352.1 | 3246 | True |
| cube-wall-cloth | head (df57bfb8) | 20260911T152558Z | 100 | 8.14 | 10.0 | 81.45 | 73.62 | 186.1 | 5.10 | 5.29 | 200.7 | 3662 | True |
| cube-wall-cloth | head (df57bfb8) | 20260911T153229Z | 100 | 8.18 | 10.1 | 81.80 | 78.24 | 185.1 | 5.13 | 5.33 | 197.9 | 3626 | True |
| cube-wall-cloth | head (df57bfb8) | 20260911T153448Z | 100 | 8.33 | 10.4 | 83.31 | 72.66 | 188.0 | 5.14 | 5.33 | 199.7 | 3650 | True |
| cube-wall-cloth | head (df57bfb8) | 20260911T153725Z | 100 | 9.10 | 11.0 | 90.98 | 96.67 | 183.7 | 5.71 | 5.99 | 215.7 | 3628 | True |
| cube-wall-cloth | head (df57bfb8) | 20260911T154359Z | 100 | 7.96 | 10.0 | 79.62 | 73.64 | 191.0 | 4.96 | 5.19 | 191.4 | 3568 | True |
| cube-wall-cloth | base (4d1f3f34) | 20260911T152724Z | 100 | 10.90 | 12.8 | 109.01 | 105.14 | 242.2 | 5.04 | 5.25 | 193.3 | 4768 | True |
| cube-wall-cloth | base (4d1f3f34) | 20260911T153006Z | 100 | 10.73 | 12.7 | 107.34 | 106.43 | 242.3 | 4.99 | 5.18 | 193.6 | 4718 | True |
| cube-wall-cloth | base (4d1f3f34) | 20260911T153615Z | 100 | 10.95 | 12.8 | 109.53 | 104.78 | 247.7 | 5.05 | 5.23 | 196.4 | 4658 | True |
| cube-wall-cloth | base (4d1f3f34) | 20260911T153853Z | 100 | 11.28 | 13.1 | 112.84 | 108.46 | 254.2 | 5.11 | 5.29 | 203.2 | 4760 | True |
| cube-wall-cloth | base (4d1f3f34) | 20260911T154135Z | 100 | 11.31 | 13.2 | 113.15 | 107.39 | 243.7 | 5.18 | 5.40 | 203.8 | 4726 | True |
| cube-wall-cloth | dahl (e1eed4b9) | 20260911T152839Z | 100 | 8.82 | 10.8 | 88.16 | 80.62 | 200.1 | 5.01 | 5.18 | 193.4 | 3958 | True |
| cube-wall-cloth | dahl (e1eed4b9) | 20260911T153120Z | 100 | 8.87 | 10.8 | 88.67 | 81.70 | 200.6 | 5.02 | 5.21 | 195.3 | 3950 | True |
| cube-wall-cloth | dahl (e1eed4b9) | 20260911T153340Z | 100 | 9.07 | 11.0 | 90.71 | 82.84 | 201.0 | 5.12 | 5.33 | 197.3 | 3992 | True |
| cube-wall-cloth | dahl (e1eed4b9) | 20260911T154008Z | 100 | 9.06 | 10.8 | 90.56 | 82.99 | 196.9 | 5.06 | 5.27 | 195.2 | 3908 | True |
| cube-wall-cloth | dahl (e1eed4b9) | 20260911T154250Z | 100 | 9.21 | 11.1 | 92.08 | 83.95 | 199.0 | 5.14 | 5.34 | 201.9 | 3882 | True |



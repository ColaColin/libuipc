# Multi-process-per-GPU throughput of the official suite on the local RTX 2070 SUPER (post round 7)

**Question.** Rounds 4–7 cut per-process VRAM massively and made each process much
faster. Does that change the multi-process-per-GPU calculus (CUDA MPS) on the one
local GPU? More VRAM → more concurrent processes; but a faster, better-filled
process leaves less idle SM time for the others to soak up.

**Answer in one line.** The VRAM effect is real and large — every scene now fits
3–8 copies on the 8 GB card (the round-4 baseline fit 1–3) — but the throughput
ceiling per GPU is now **1.11–1.77×** at N=2–8, because single-process GPU
utilisation is already 53–92 % and per-process slowdown grows ≈ linearly beyond
N=2. Multi-process still pays, much less than it used to on lighter scenes.

*(Same day, rented RTX 3090: see "Rented RTX 3090" below — scaling there is
**1.29–2.92×**, better than the 2070S on every scene, and per-machine throughput
at best-N is a uniform ~2.0–2.2× the 2070S.)*

## Method

- Box: RTX 2070 SUPER (cc 7.5, 8 GB, 215 W cap), 32-core host, user-level CUDA MPS
  daemon (`nvidia-cuda-mps-control`, pipe `/tmp/nvidia-mps`) — the same production
  setup as the cloth-machine / cloth-dataset studies. Verified active during runs:
  100 % util with all client memory attributed to the MPS server.
- Tree: `main` @ `114190f9` (round-7 final), `uipc-perf-env`, `build-perf` current.
- Tool: `/workspace/tools/local/mp_sweep.py` launches N simultaneous
  `scripts/run_benchmark.py run <scene>` invocations (canonical env, default frames,
  0.15 s stagger), samples `nvidia-smi` at 0.5 s, parses each process's own
  `TOTAL frames=… mean=…ms` line (archive JSONs collide at second resolution and are
  not relied on). VRAM guard refuses an arm unless `free ≥ N×peak+400 MiB`.
- Metrics (2–3 arms per (scene, N), interleaved):
  - `steady f/s = N / slowest copy's mean frame time` — the old cloth-machine /
    cloth-dataset protocol ("sum frames / slowest loop"); **headline**.
  - `makespan f/s = N×frames / (last exit − first launch)` — includes ~1.2 s fixed
    harness startup per process; conservative lower bound.
- Raw data: `agent_docs/performance/data/2026-09-17-2070s-mp/` (per-scene JSONs,
  per-process logs, SPLIT A/B), plus the run_benchmark archive records.

## Main sweep (2026-09-17, medians over arms)

| scene | N=1 ms/frame, util | best N | steady f/s N=1 → best | speedup | per-proc MiB (const in N) | Nmax by VRAM |
|---|---|---|---|---|---|---|
| rigid-wrecking-balls | 25.6, 53 % | 8 (flat ≥4) | 39.0 → 69.1 | **1.77×** | 690 | 8+ |
| cube-wall-cloth | 55.2, 66 % | 5 (flat ≥4) | 18.1 → 26.4 | **1.46×** | ~1080 | 5 (6 guard-skipped) |
| tumbler-garments | 103.6, 88 % | 3 (flat ≥3) | 9.7 → 12.7–13.1 | **1.31–1.35×** | ~1100 | 6 |
| mas-bunny | 60.9, 62 % | 4 | 16.4 → 20.5 | **1.25×** | 838 | 6 (8 guard-skipped) |
| crease-press | 216.7, 91 % | 2 | 4.6 → 5.2 | **1.13×** | ~2100 | 3 |
| stiff-gipc-case2 | 154.6, 88 % | 3 | 6.5 → 7.2 | **1.11×** | ~1490 | 4 |

Shape: per-process slowdown is ≈1.4–1.8× at N=2 and then grows ≈ linearly with N
(e.g. rwb 1.42/2.39/3.51/4.45× at N=2/4/6/8; mas-bunny 1.69/3.20/4.93×) — MPS is
time-slicing a nearly-busy GPU, not finding free SMs. Utilisation climbs to
92–98 % at the best N and mean power stays ≤ ~200 W (below the 215 W cap; SM clocks
*rise* with N because the GPU stays busy — the 217 W spot reading during the sweep
was a peak, not a limit).

## Then vs now

- **The official suite had never been measured multi-process** (rounds 4–7 were
  single-process; round 6 listed it as open work, `2026-09-13-perf-round6.md`).
- Last multi-process data on this box (round 3, 2026-09-11, drum scenes, MPS ×3):
  2.3–2.5× aggregate; cloth-machine (2026-09-06, `cloth-machine/results/gpu_throughput.md`):
  tiny `towel1_f2` scaled **5.75×** at N=8 (17.7 → 101.6 f/s, per-proc 56.9 → ~77 ms,
  680 MiB/proc, SM 90 %), realistic 4.6 k-vertex scene only **1.83×** at N=4.
- Round-4 baseline VRAM (2026-09-12): rwb **4022** MiB, case2 2230, mas-bunny 1462,
  cube-wall 2048 → on this 8 GB card that allowed **N ≤ 1–3**; today's 690–2136 MiB
  allows **N = 3–8**. The VRAM side of the user's question is unambiguous.
- The scenes that scale worst now are the ones that were optimised hardest:
  crease-press (round-7 subject) starts at 91 % util single-process and gains 13 %,
  tumbler (round-6 subject) starts at 88 % and gains ~35 %. The light-contact rwb
  (53 % util at N=1) still gains 77 %. **"Less VRAM" and "loads the GPU harder" are
  both true; on this suite the second effect caps the aggregate at well under the
  old 5× class of wins.**
- No apples-to-apples old-vs-new MPS measurement on the same scene exists (the old
  MPS scenes predate the official suite); the comparison above is across scenes.

## Round-6 open item closed: `UIPC_CONTACT_SPLIT` under MPS

Round 6 predicted (`2026-09-13-perf-round6.md`): under MPS the SM slack the
2-stream split exploits is filled by other processes, so `=0` (fused) should win.
Measured at N=4, 2 interleaved pairs per arm, slowest-proc aggregate:

| scene | `=2` (default) | `=0` | `=0`/`=2` |
|---|---|---|---|
| rigid-wrecking-balls | 63.3 f/s | 63.7 f/s | **1.006** (dead heat) |
| cube-wall-cloth | 25.9 f/s | 25.2 f/s | **0.975** |

The hypothesis is **not confirmed**: `=2` remains equal-or-better under MPS
(cube-wall's −2.5 % for `=0` matches its single-process sign from round-6 s09).
Keep the default; n=2 pairs, so treat <±3 % as noise.

## Rented RTX 3090 (same protocol, 2026-09-17)

Owner-requested repeat on a rented 24 GB card ("rent a 3090 or similar and repeat").
Driver `tools/vast/vast_run_mp_sweep.sh` + on-box `vast_remote_mp_sweep.sh` (single
tree at fork branch `mp-sweep-20260917` = main `8382c951`, native sm_86, MPS daemon
on the box, stagger 0.05 s, reps=2, adaptive per-proc VRAM guard). Same scenes,
same default frames, same slowest-proc aggregate metric — directly comparable to
the 2070S table above.

| scene | 3090 N=1 (ms, util) | best N | aggregate | scaling | MiB/proc on 3090 (2070S) |
|---|---|---|---|---|---|
| rigid-wrecking-balls | 20.2, 61 % | 12 | **144.3 f/s** | **2.92×** | 1578 (692) |
| cube-wall-cloth | 36.5, 56 % | 8 | **57.6 f/s** | **2.10×** | 1984 (1086) |
| tumbler-garments | 68.0, 82 % | 8 | **28.2 f/s** | **1.92×** | 2033 (1100) |
| mas-bunny | 37.8, 52 % | 8 | **40.5 f/s** | **1.53×** | 1783 (838) |
| crease-press | 130.6, 87 % | 4 | **10.3 f/s** | **1.35×** | 3005 (2136) |
| stiff-gipc-case2 | 84.6, 80 % | 4 | **15.2 f/s** | **1.29×** | 2378 (1490) |

Findings beyond the 2070S picture:

- **Scaling is better on the bigger card for every scene** (e.g. mas-bunny
  1.53× vs 1.25×, cube-wall 2.10× vs 1.46×, tumbler 1.92× vs 1.35×). 82 SMs vs 36
  leave more room for MPS to overlap kernels before time-slicing takes over;
  utilisation at the best N reaches 80–97 %, same saturation signature as locally.
- **Per-process VRAM grows with the card** (~1.6–2.9× the 2070S per-proc figure),
  confirming the old cloth-machine observation that allocator pools/context scale
  with device size. On 24 GB the practical ceilings were N≈12–14 (rwb), ~11–12
  (cube/tumbler — the guard skipped their N=12 arms at ~1.97–2.0 GB/proc),
  ~9–10 (case2), ~7 (crease). Throughput had saturated before all of these.
- **Per-machine throughput at best N is a uniform ~2.0–2.2× the 2070S** on every
  scene, although single-process speed is only 1.3–1.8× — the bigger card gains
  more from concurrency than the small one, so the *machine* ratio exceeds the
  *process* ratio.
- Single-process util is *lower* on the 3090 than the 2070S on the same scenes
  (e.g. cube-wall 56 % vs 66 %; mas-bunny 52 % vs 62 %): the same code leaves a
  larger fraction of a bigger GPU idle, which is exactly what MPS converts into
  the better scaling above. The round-4/5 "grid-size against machine-size"
  transfer table predicted this direction.

Ledger (VAST_AI.md rule): instance **51295839**, offer 51016746, RTX 3090,
$0.3622/h, 64 effective cores, Michigan US, image nvidia/cuda:12.8.1-devel-
ubuntu24.04; created 2026-09-17T10:05:44Z, destroyed 10:47:03Z (41 min, incl.
~9 min build); charged **$0.28** (credit 3.7625 → 3.4959). One earlier failed
attempt (instance 51295416, same offer, $0.02) died at cmake configure: my
remote script dropped the old script's `mkdir -p build`, and the shell opens
`> build/configure.log` before `cmake -B build` can create the directory —
fixed in `vast_remote_mp_sweep.sh`. Raw data:
`data/2026-09-17-rtx3090-mp/` (per-scene sweep JSONs, timeline, instance facts).

## Caveats

- 2–3 arms per (scene, N); day-to-day scatter on these scenes is 2–8 %, so the
  speedups are ± a few points; the *shape* (saturation by N=3–4 on heavy scenes) is
  unambiguous across all reps.
- cube-wall N=6 and mas-bunny N=8 were skipped by the static VRAM guard
  (need > free 6985 MiB); both had already saturated at lower N, and observed
  per-proc memory was constant in N, so no signal was lost.
- One transient failure: first arm of the sweep (rwb rep0 N=1) exited rc=250 after
  1.6 s, never reproduced (15+ later N=1 arms clean). Treated as a startup flake.
- WDDM-style caveat does not apply (Linux), but the ~800 MiB display+MPS baseline
  is included in every peakTotalMiB; per-proc MiB above already subtracts it.
- 3090 arms: reps=2 per (scene, N), stagger 0.05 s (short scenes on a fast card);
  rwb at ~2 s/run is the timing-sensitive end — trust its per-process ms over the
  makespan figure. The N=12 arms the guard skipped (cube-wall, tumbler) had
  already-flat throughput at N=8.

# round 7 / s09 — the `_distribute` blocking D2H: measured and REJECTED (nothing ships)

Every file here backs the s09 section of
`agent_docs/performance/2026-09-15-perf-round7.md`. The engine tree on the
branch is identical to main (`57dc6dfd`) — this step changed no engine source;
it priced the standing TODO item ("Remove the blocking D2H in `_distribute`")
with two instruments and closed it as a documented rejection.

## The two instruments

1. **`UIPC_D2H_PROFILE=2`** (round 4's `host_read` funnel, `cuda_tool/host_sync.h`):
   full default-frame runs of the three multi-receiver scenes + the site
   attribution (per-call-stack n / stall / bytes), symbolized with `addr2line`
   against the installed `libuipc_backend_cuda.so`.
   `d2hprof/d2h_sites_{crease-press,cube-wall-cloth,tumbler-garments}.txt`.
2. **nsys cuda API + GPU traces** (`d2hprof/apitrace.sh`, fresh prefixes,
   `--cuda-graph-trace=node`): short windows — crease-press 12 frames (the
   press phase, 12.46 s of readback span), cube-wall-cloth 60 frames (2.89 s;
   the first 20 frames are pre-contact — see `cwc_full` below).
   `analyze_api4.py` attributes every `cudaMemcpyAsync`+`cudaStreamSynchronize`
   pair to its launch context, splits each park into **drain** (default-stream
   kernels still queued at sync start) and **stall** (round trip after they
   finished), and checks whether ANY kernel (any stream — the contact K9 side
   stream, the ABD prepass stream) executed during the park.

## The site identification (why these are the `_distribute` reads)

`GlobalDyTopoEffectManager::Impl::_distribute` has exactly two blocking reads
per receiver per dytopo call — the 8-byte `Vector2i gradient_range`
(`DeviceVar::operator T()`, `global_dytopo_effect_manager.cu:364`) and the
4-byte `IndexT h_total_count` (`VarView::copy_to`, `:419`). Both go through
the round-4 `host_read` pinned funnel, so the profiler sees them. Counts
cross-check three ways on crease-press: the 12-frame window has k1×442 / k3×663
kernel launches and exactly 442+663 k-chain readbacks; the full profiled run
has 1 116 gradient + 1 674 hessian reads; on cube-wall-cloth the
reads exceed the kernel counts by exactly the empty calls
(992-728 / 1488-1092) because the reads are unconditional while the launches
are `n > 0`-guarded — the pre-contact free-fall frames of cwc read zeros
(that is why `cwc_api` (20 frames) shows zero distribute *kernels* and
`cwc_full` (100 frames, kern_sum only) shows k1×728 / k3×1092).

## Headline numbers

| scene | wall | `_distribute` D2Hs / run | stall (round trips) | % wall | park (window) | GPU idle inside the park |
|---|---|---|---|---|---|---|
| crease-press | 31.7 s (s08 head) | 2 790 (1 116 grad + 1 674 hess) | 16.4 ms | **0.05 %** | 51.8 ms / 12.46 s = **0.42 %** | **0.00 ms** |
| cube-wall-cloth | 5.7 s | 2 480 (992 + 1 488) | 14.6 ms | **0.26 %** | 13.2 ms / 2.89 s = **0.46 %** | **0.00 ms** |
| tumbler-garments | 18.9 s | 7 340 (2 936 + 4 404) | 43.1 ms | **0.23 %** | (not captured; stall is the bound) | — |

* The park is 96 %+ drain — the host waiting for work whose output the host
  needs anyway; during **every** `_distribute` round trip on both traced
  scenes some kernel was executing on another stream (contact part 1 on the K9
  side stream, the ABD prepass, the sorts), so the D2H latency costs no GPU
  idle: `stall(GPU-idle) = 0.00 ms` in `*_api_analysis4.txt`.
* The per-readback stall is 5.9–6.6 µs — the round-4 pinned-staging funnel
  already removed the 14–22 µs pageable penalty this TODO was written against.

## Why the s14 wait-move does not transfer (the code proof)

The read value is not a join flag; it **gates the downstream layout**:
`GlobalLinearSystem::build_linear_system` opens with `_update_subsystem_extent`
(`global_linear_system.cu:787`), where every subsystem's
`report_extent` reads `gradient_count()` / `hessian_count()` (the s14 host
accessors over the classified views) to compute its block offset — which sizes
`triplet_A`, `bcoo_A` and every subsystem's subview, and (FEM) the launch
geometry of `_assemble_dytopo_effect` and (ABD) `_prepare_dytopo_pairs`.
Deferring the *wait* into those accessors moves the park by only the
`receive()`-stash + `solve()`-entry host walk (µs, single digits) — there is no
GPU-launchable count-independent work in between (FEM G/H writes into the
layout-gated `triplet_A`, round-6 s12's finding; the ABD G/H prepass is already
forked inside the contact fork by s11 mode 4). Predicting the count does not
help either: a wrong prediction is a wrong matrix layout, with no fallback
after the fact.

## Files

| file | claim it backs |
|---|---|
| `d2hprof/d2h_sites_*.txt` | per-site stalls on all three scenes (the two `_distribute` sites, the converter sites, the PCG convergence site that actually dominates) |
| `d2hprof/{cp,cwc,tum}_prof2.log`, `cp_prof1.log` | the raw `[d2h]`/`[d2h-site]` dumps |
| `d2hprof/cp_api_analysis4.txt` | crease-press window: park/drain/stall split + GPU-idle = 0.00 ms for `_dist_grad` (n=221) and `_dist_hess` (n=884) |
| `d2hprof/cwc60_api_analysis4.txt` | cube-wall-cloth contact-phase window: same (n=181+724), park 0.46 % of span, GPU-idle 0.00 ms |
| `d2hprof/cwc_api_analysis4.txt` | the pre-contact 20-frame window (no distribute kernels — the empty-call explanation) |
| `d2hprof/cp_api_analysis3.txt` | the same window before the caller split; shows the `matrix_converter` parks |
| `d2hprof/*_cuda_api_sum.csv` | `cudaStreamSynchronize` totals (11 991 calls / 10.93 s in the crease-press window) |
| `d2hprof/*_cuda_gpu_kern_sum.csv` | distribute k1..k4 launch counts (the read/kernel cross-check); `cwc_full` is the 100-frame kern_sum |
| `d2hprof/*.run.log` | frame counts / timings of every captured run |
| `d2hprof/apitrace.sh`, `kernsum.sh`, `analyze_api*.py` | the capture + analysis scripts (v4 is the final analysis) |
| `gate_head.txt` (in `/workspace/output/round7/s09/`) | gate at head identical to `baseline_tests.txt` |

Not archived: the 200 MB per-launch gpu traces and the sqlite exports (the
analysis txts above carry every number quoted in the record).

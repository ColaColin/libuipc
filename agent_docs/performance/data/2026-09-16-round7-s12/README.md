# round 7 / s12 — MAS cluster-matrix L2 pinning: PRICED AND REJECTED at the platform boundary; nothing ships, tree = main

The s10 pick list's #6: keep the MAS cluster matrices L2-resident across the PCG
iteration stream via a `cudaAccessPolicyWindow` over `cluster_inverses` (~20
host-side lines, no kernel changes, bit-identical by construction). The step
was to verify the premise first — and the premise fails on this box at the
first API call: **cc 7.5 does not implement the L2 residency-control feature
family at all.** No knob exists to wire (there is nothing for
`UIPC_MAS_L2_PIN` to toggle), so there is no A/B and nothing to gate beyond
the head health check. The rejection carries three measured legs instead: the
capability probe, the working-set arithmetic, and the residency-floor prize.

## 1. The platform probe: every entry point of the mechanism is closed

Probes (source + output archived here, compiled with
`/workspace/deps/cuda-12.8/bin/nvcc`, run on the RTX 2070 SUPER, driver
595.84):

| probe | result |
|---|---|
| `cudaDeviceProp::l2CacheSize` | 4,194,304 B (4 MiB) |
| `cudaDeviceProp::persistingL2CacheMaxSize` | **0** |
| `cudaDeviceProp::accessPolicyMaxWindowSize` | **0** |
| `cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)` — real device pointer, 1 MiB and 4 KiB windows, `hitProp` = persisting **and** normal | **`invalid argument`** (every variant) |
| `cudaStreamSetAttribute(...)` called **while the stream is capturing** (the exact code point `MASPreconditionerEngine::apply` would use inside the PCG graph capture) | **`invalid argument`** (capture itself survives) |
| `cudaGraphKernelNodeSetAttribute(cudaKernelNodeAttributeAccessPolicyWindow)` on a correctly captured 1-node graph, 1 MiB and 4 KiB | **`invalid argument`** |
| `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1 MiB)` | **"limit is not supported on this architecture"** |

Files: `l2_capability_probe.{cu,out}` (properties + first attribute call),
`l2_capability_probe2.{cu,out}` (the exhaustive stream/window matrix),
`l2_capture_attr_probe.{cu,out}` (attribute-during-capture),
`l2_graph_probe.{cu,out}` (graph-node route; note `cudaGraphGetNodes`'s
`numNodes` argument must be pre-set to the array capacity or it returns
`invalid argument` — the probe initialises it).

The brief's premise "cc 7.5 has 4 MB of setaside-capable L2" conflates the L2
*size* (4 MiB, true) with set-aside *capability* (requires cc 8.0+; the
runtime says so by name in the `setLimit` error).

## 2. The graph-replay question — unanswerable here, answered-by-probe elsewhere

"Does a stream access-policy window survive CUDA-graph capture/replay?" cannot
be measured on this GPU: the attribute cannot be set at all, in any form, so
there is nothing to replay. `l2_graph_probe.cu` is the one-run answer for any
future cc 8.0+ box: set the window on the capture stream (or before capture),
capture, instantiate, then `cudaGraphKernelNodeGetAttribute(
cudaKernelNodeAttributeAccessPolicyWindow)` on the captured kernel node tells
whether the policy was baked into the node. Run it before re-attempting this
candidate on rented hardware.

## 3. Working-set arithmetic (measured from launch grids at this head)

`grid_extract.txt` (from 12-frame nsys captures, `--cuda-graph-trace=node`;
`cluster_inverses` = total cluster nodes / 16 blocks x 4,896 B — 136
`Matrix3f` blocks, the round-4 "already minimal" representation):

| scene | fine grid | coarse grid | cluster blocks | `cluster_inverses` | vs 4 MiB L2 |
|---|---|---|---|---|---|
| crease-press | 561 | 42 | 1,206 | **5,904,576 B = 5.63 MiB** | **141 %** |
| mas-bunny | 640 | 46 | 1,372 | **6,717,312 B = 6.41 MiB** | **160 %** |

Even on a part that *did* support the window, the full working set cannot be
resident; partial pinning (what the set-aside would have allowed, minus the
PCG vectors' own L2 share) bounds the win at a fraction of leg 4's floor.

## 4. The prize, measured with the shipped round-6 probe (no code changes)

`UIPC_MAS_ROWDOT2=1002` = the s15 timing-only L2-residency probe (every block
reads `cluster_id & 63` = 313 KB, resident after first touch). 12-frame
crease-press captures, one binary (main `0e603f07`), fresh prefixes:

| arm | kernel | per-launch |
|---|---|---|
| default | `fused_R<(bool)1,(int)1>` (fine) | 23.51 us x 34,851 |
| default | `fused_R<(bool)0,(int)0>` (coarse) | 4.05 us x 34,851 |
| `UIPC_MAS_FUSED_R=0 UIPC_MAS_ROWDOT2=2` | `rowdot2` u2 (real) | 23.80 us x 39,592 |
| `UIPC_MAS_FUSED_R=0 UIPC_MAS_ROWDOT2=1002` | `rowdot2` u2 **L2-resident probe** | **17.18 us x 10,666** |

**Full residency is worth −27.8 % of the local solve (−6.6 us/launch)** —
if the whole 5.63 MiB could sit in L2, which it cannot (leg 3). At full-run
scale (s10: 110,081 launches, 31.26 s GPU kernel time) that unattainable
upper bound is ≈ 726 ms ≈ −2.3 % of scene GPU time; the pick list's
"ceiling −1.8 %" was already the realistic (partial) version of this.

Side readings (in-family sanity): the two-kernel path per apply costs
23.80 + 7.79 (build_R) = 31.6 us vs the fused path's 23.51 + 4.05 = 27.6 us
(s16's fusion still paying on this scene); mas-bunny fused fine 25.95 us
matches round-6 s16's 26.4. The probe capture's SpMV reads 43.9 us vs ~67 in
the real arms — an artifact of the garbage-preconditioner arm (PCG pinned at
the iteration cap, vector values diverged), **not** an L2 observation; it is
not usable as the SpMV control and is not cited as one. There is no ON arm,
so the SpMV-cost question (would it lose L2 to the pin?) has no measurement
on this box — leg 1 makes it unanswerable here.

`cp_rowdot_probe_run_log_excerpt.txt` documents the probe arm's behaviour
(line-search cap at frame 1, teardown SIGSEGV rc=139 after the profile was
complete — timing-only arm, results invalid by design).

## 5. What would be needed instead (recorded, not built)

The only reachable emulation on cc 7.5 is kernel-side eviction hints:
`__ldcs` (evict-first) on the SpMV's matrix loads plus selectively normal
loads for a pinned prefix of `cluster_inverses` in the local solve. Leg 3's
arithmetic caps it (pin ≤ ~3 MiB of 5.63 → ≈ 3-4 us → ≈ −1.2 % of scene GPU
time) and it inherits every property the round distrusts in the
occupancy-tuning class: two kernel edits, a tuning cutoff, sign instability
across architectures, and a 5090 (where the real window may exist and the
L2 is large enough to hold the whole buffer) would want the API version
instead. Not attempted.

## 6. Gates

Nothing ships (no engine source changed; `git diff 0e603f07 -- src/` empty),
so the only gate is the head health check: `bash
/workspace/output/round7/gate.sh` at main `0e603f07`, identical to
`baseline_tests.txt` (see `gate_head.txt`, copied from
`/workspace/output/round7/s12/gate/`). No end-to-end A/B: both arms of any
comparison would be the same binary. The round's drift reference at this
head stands (s10: meanFrameMs 236.92 ms, n=8).

## Files

| file | what it is |
|---|---|
| `l2_capability_probe.{cu,out}` | device properties + first window attempt |
| `l2_capability_probe2.{cu,out}` | exhaustive stream-window matrix + setLimit |
| `l2_capture_attr_probe.{cu,out}` | attribute-set during capture |
| `l2_graph_probe.{cu,out}` | graph-node attribute route; the cc 8.0+ replay test |
| `grid_extract.txt` | launch-grid -> cluster-block -> byte arithmetic (both scenes) |
| `{cp_default,cp_rowdot_u2,cp_rowdot_probe,bunny_default}_cuda_gpu_kern_sum.csv` | the four captures behind legs 3-4 |
| `analyze_caps_s12.py`, `nsysrun_scene.sh` | re-runnable analysis + capture scripts |
| `cp_rowdot_probe_run_log_excerpt.txt` | probe-arm caveat |
| `gate_head.txt` | head gate output |

# round 7 / s17 — the PCG convergence readback as a pinned zero-copy poll

Every file here backs the s17 section of
`agent_docs/performance/2026-09-15-perf-round7.md`. Branch
`perf/round7-s17-pcgpoll` from main `da9bd2d0`; two engine files changed
(`src/backends/cuda/linear_system/linear_fused_pcg.{cu,h}`).

## The premise, re-priced at head (da9bd2d0) before building

Two instruments, one capture each:

1. **`UIPC_D2H_PROFILE=2` funnel, full 130-frame crease-press**
   (`head_d2hprof.log`, totals in `head_d2h_funnel_total.txt`, top sites
   symbolized in `head_d2h_sites_top.txt`): the PCG site
   (`LinearFusedPCG::fused_pcg`) is **n = 19,797 readbacks, 116.8 ms stall,
   5.90 µs/readback** — the s10 numbers (24,689 × 5.96 µs = 147.1 ms) within
   this run's chaotic PCG draw (99k iterations vs s10's 110k). The funnel's
   "stall" is the drained-stream round trip of the 8-byte `d_rz_new` read.
2. **nsys api+gpu trace, 12-frame window** (`head_cp12_*`, analysis in
   `premise_and_scope_analysis.txt`): 7,336 readbacks in 13.88 s. The honest
   ceiling is BIGGER than the funnel number: the GPU-idle bubble around each
   readback (from the block's last GPU row to the next block's first kernel)
   is **med 14.69 µs** (2.33 µs sync-return latency + 12.34 µs post-resume
   host work incl. the next `cudaGraphLaunch` enqueue; nsys-inflated),
   **111.2 ms per 12-frame window**. The park itself (~587 µs med) is drain —
   overlapped, physics, not the target.

So the brief's "ceiling ~0.2 %" framing (which priced only the memcpy round
trip) undersold the mechanism: the whole between-block bubble is host-side
and attackable by a poll that also returns early.

## What shipped

`PcgPollWord { Float rz; unsigned long long seq; }` — one 64-byte pinned
mapped (zero-copy) allocation per solver instance. The graph's existing
`fused_pcg_scalar_kernel` `<<<1,1>>>` node (the node whose job is already to
read the completed `d_rz_new`) additionally publishes `{rz, seq=++d_seq}` to
the pinned word; the host spins on `seq >= m_poll_expected` (iterations
launched) and reads `rz` from the cacheline — no `cudaMemcpyAsync`, no
`cudaStreamSynchronize`. The publish sits one node before the block's last
kernel, so the predicate + next graph launch overlap the block's tail on the
GPU. Fold-1 publishes from `fused_update_p_scalar` (thread 0 of block 0),
fold-2 from the dot tail's live call.

- Knob **`UIPC_PCG_POLL`** (default 1): `=0` passes null pointers — kernels
  skip the publish, the host uses the old blocking funnel read. Behavioural
  rollback in one build.
- Probe **`UIPC_PCG_POLL_VERIFY=1`**: after every poll, also read `d_rz_new`
  through the blocking funnel and bit-compare (pays both costs; probe only).
- Ordering without a fence: both fields share one cacheline, `seq` is stored
  strictly after `rz` by one thread (PTX volatile order), PCIe posted writes
  from one requester complete in order, x86 DMA is coherent — see the comment
  on `pcg_poll_publish`. Empirically guarded by POLL_VERIFY.
- Safety: a doorbell frozen for 100 ms falls back to the blocking read (warns
  once); correctness never depends on the poll.

## The graph-interaction finding (measured on this box)

Device writes to host-mapped pinned memory inside a captured CUDA graph DO
update per replay — no API is called at publish time, so nothing for capture
to forbid (contrast s12, where *setting a stream attribute* during capture
was rejected by the runtime). Evidence: full 130-frame crease-press with
`UIPC_PCG_POLL_VERIFY=1`: **22,424 polled values, 0 mismatching, 0 poll
fallbacks**; mas-bunny: **7,042 / 0 / 0**; the 12-frame smoke: 8,002 / 0 / 0.
A stale or reordered doorbell would have shown as bit mismatches or fallbacks.

## Gates

- Correctness: `bash /workspace/output/round7/gate.sh` identical to
  `baseline_tests.txt` at **default and `UIPC_PCG_POLL=0`** (11/3, 1112/36,
  2730/46, 100/3, 4/1, 448/23, 14213/95, pytest 48+1 — pytest's own duration
  string only). Gate outputs: `gate_default.txt`, `gate_poll0.txt`.
- Numerics — **the exit iteration cannot move by construction**: the host
  predicate consumes the same bits (the doorbell copy is written from the
  same register the scalar node read; POLL_VERIFY proved 0/37,468 mismatches
  over two full runs), the check cadence is unchanged, and no device-side
  computed value differs (the publish is a store no device code reads;
  registers/grids identical: scalar 32 reg `<<<1,1>>>`, p_beta 32 reg
  `<<<197,256>>>`, xr 42 reg). Trajectory check: mas-bunny **Newton exactly
  465 in all 10 A/B runs**, line_search exactly 465 both arms, PCG 35,200 to
  35,230 overlapping across arms (the old arm's own in-arm spread is the same
  ±25-count class). crease-press counts chaotic within arms as documented.

## Performance

- **Scope** (12-frame nsys windows, env A/B in one build, fresh prefixes,
  `--cuda-graph-trace=node`; `premise_and_scope_analysis.txt` +
  `scope_{old,new}_cuda_{api,gpu}_kern_sum.csv`):
  - PCG readbacks (memcpyAsync+sync after a graph launch): **7,645 → 0**.
  - Between-block GPU-idle gaps (>5 µs ending at a PCG kernel): **7,664 /
    124.5 ms / med 14.72 µs → 7,484 / 67.3 ms / med 7.20 µs (−51 % per
    block)**.
  - No overshoot: per-launch kernel times flat — SpMV 67.47→67.57 µs,
    fused_dot 7.02→7.02, fused_update_xr 8.19→8.19, MAS local solve
    4.06→4.07, collect 4.84→4.79; the publisher itself costs **+0.31 µs
    (fused_pcg_scalar 1.92→2.23 µs)** and its successor p_beta +0.18 µs
    (3.64→3.82) — ≈ +0.49 µs/block against −7.5 µs/block recovered, ~15:1.
- **End-to-end** (ab.py, interleaved ABBA, one build, `ab_all_summary.log` +
  raw json dirs):
  - **crease-press n=10/arm: meanFrameMs 228.28 → 227.01 ms (−0.56 %,
    p=0.843, OVERLAPPING — below the ~3.8 % MDE at n=10**, direction agrees
    with the scope); ms/newton −1.73 % (p=0.44); count guard fired (Newton
    +1.19 %, PCG −1.19 %, in-arm 96-154 k both arms = the documented chaos) —
    wall unreadable, rests on scope + mas-bunny.
  - **mas-bunny n=5/arm: meanFrameMs 61.43 → 60.91 ms (−0.85 %, DISJOINT,
    p=4.8e-05); ms/newton −0.85 % disjoint; ms/pcg −0.82 % disjoint; Newton
    exactly 465 in all 10 runs, line_search exactly 465 both arms**, PCG
    35,185-35,235 overlapping (the old arm's own ±25-count spread). A first
    mas-bunny A/B in the transient-slow window read −0.89 % disjoint
    (p=0.0016) — same effect, both runs kept (`ab_bunny`, `ab_bunny2`).
  - **cube-wall-cloth n=4/arm (regression): mean +0.24 % (p=0.72,
    overlapping), ms/newton −0.97 % (p=0.049, at n=4 — not claimed)**; no
    regression (cwc's wall does not resolve sub-% effects, s13/s14 finding).
- Non-default arms on the final (post-format) binary, POLL_VERIFY: fold=1
  (publisher inside `fused_update_p_scalar`) 7,516 polls / 0 mismatches;
  forced preconditioner-bypass every 37th solve (plain path, no graph)
  19,093 polls / 0 mismatches; default 8,549 / 0.

## Files

| file | claim it backs |
|---|---|
| `premise_and_scope_analysis.txt` | the re-priced stall at head + both scope arms (readback count, bubble, split) |
| `head_d2hprof.log` (not archived — see funnel txts) | full-run funnel profile at head |
| `head_d2h_funnel_total.txt`, `head_d2h_sites_top.txt` | 19,797 × 5.90 µs = 116.8 ms at the PCG site |
| `head_cp12_cuda_{api,gpu}_kern_sum.csv` | head window kernel/api sums |
| `scope_{old,new}_cuda_{api,gpu}_kern_sum.csv` | the A/B scope arms |
| `gate_default.txt`, `gate_poll0.txt` | correctness gates |
| `ab_bunny/`, `ab_cp2/`, `ab_cwc/` + `ab_*.log` | end-to-end A/B raw runs |
| `box_anomalynote.md` | the transient-slow-box incident: first crease-press A/B (both arms +65 %) invalidated and rerun after the head binary reproduced normal speed |
| `scripts/apitrace.sh`, `scripts/analyze_pcggap.py`, `scripts/analyze_pcggap2.py` | capture + analysis instruments |

Not archived: the ~200 MB per-window gpu/api trace csvs (the analysis txts
carry every quoted number; s09 precedent).

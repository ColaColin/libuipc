# Round 5, s27 + s28 (w3-blackwell) — what a graph node costs, and what removing it costs

Step ids **s27/s28 are provisional**: while this step ran the coordinator assigned
s25 to w2-ada and s26 to w0-turing (both merged after `d3f3a553`), so these are simply the next
free ids, not a central assignment. Renumber freely at merge.

RTX 5060 Ti (cc 12.0, **36 SMs**), instance 50737979, branch `perf/round5-w3-blackwell` on top of
`perf/round5` = `d3f3a553` (s19–s24 merged). Two steps, two commits:

- **s27** `1f82cfd8` — fold the `<<<1, 1>>>` PCG scalar kernel into a neighbouring kernel.
  **Rejected**, both designs, default off, code kept as measurement arms.
- **s28** `e6ead26d` — spread `cuda_tool`'s own elementwise fill launches over the SMs.
  **Accepted on the scope gate**; the end-to-end gate does not resolve it and the step says so.

Baseline for both, measured fresh on `d3f3a553` (60-frame windows, 40 for mas-bunny):

| scene | kernel total ms/Newton-it | `fused_pcg_scalar` | fill family | PCG vector grid |
|---|---|---|---|---|
| rigid-wrecking-balls | 6.4107 | 0.0195 (0.30 %) | 0.0437 (0.68 %) | 27 |
| cube-wall-cloth | 7.2244 | 0.0296 (0.41 %) | 0.0939 (1.30 %) | 140 |
| mas-bunny | 8.2329 | 0.0579 (0.70 %) | 0.2435 (2.96 %) | 225 |
| stiff-gipc-case2 | 21.0779 | 0.0430 (0.20 %) | 0.5736 (2.72 %) | 500 |

---

## 0. An instrument defect found first, and fixed in `audit.py`

The first baseline audit of this step reported the s24 sites at their **pre-s24 geometry** — grid 3 at
block 256 for `assemble_kinetic_shape_k2`, which s24 had moved to (18, 32) and proved. Nothing had
regressed. `nsys stats` **refuses to overwrite an existing `<prefix>_cuda_gpu_trace.csv` (and
`<prefix>.sqlite`) and exits quietly**, so a re-used output prefix silently re-parses the *previous*
session's trace while the `BENCHMARK_RESULT` line beside it comes from the fresh run. The two halves
of the report disagree and nothing says so.

`audit.py` now deletes both files before calling `nsys stats` and raises if no csv appears. **Any
worker re-using an `audit.py` prefix before this fix has a trace that may not be its own.** This is
the same class of defect as the vacuous correctness gate recorded at 14:00: a gate that cannot fail.

---

## 1. s27 — the `<<<1, 1>>>` PCG scalar node

### What it costs

`fused_pcg_scalar_kernel` computes five scalars (`conv`, `beta`, the guarded `rz <- rz_new`, the
`d_pAp` reset) once per PCG iteration, inside the graph-captured loop. Cost per launch:

| scene | launches / window | ms/Newton-it | **us per launch** |
|---|---|---|---|
| rigid-wrecking-balls | 3 975 | 0.0195 | **0.93** |
| cube-wall-cloth | 10 305 | 0.0296 | **0.92** |
| stiff-gipc-case2 | 12 745 | 0.0430 | **1.12** |
| mas-bunny | 10 355 | 0.0579 | **0.93** |

**0.93 us per node on three of four scenes**, independent of problem size — the kernel does five
scalar accesses, so this is node overhead plus one dependent memory round trip, not work. That is
the answer to the round's standing question about small nodes in a captured region: **inside a CUDA
graph a tiny node costs about its own reported kernel duration**, roughly 1 us, not the 5–10 us a
stream launch would cost. The graph has already eaten the launch gap; what is left is the node.

### Design 1 — fold into the tail of the dot (last-block-done). Rejected, with the cost priced.

The natural fold: the last block of the `r^T z` reduction has the completed accumulator, so it can
run the scalar body. Standard `threadFenceReduction` handshake — `__threadfence()`, a ticket
`atomicAdd`, the block drawing the last ticket does the work and resets the ticket for the next
replay.

Correct, and **slower**. On mas-bunny the dot goes 0.2916 -> 0.3916 ms/Newton-it. A three-way
compile-time fence probe (`UIPC_PCG_DOT_FENCE`, one build) prices it:

| arm | dot ms/Newton-it | delta over the plain dot |
|---|---|---|
| plain `fused_dot_kernel` | 0.2916 | — |
| tail, **no fence** (incorrect — attribution only) | 0.3139 | +0.0224 = the ticket atomic |
| tail + PTX `fence.acq_rel.gpu` | 0.3911 | +0.0996 |
| tail + `__threadfence()` (`membar.gl`) | 0.3916 | +0.1000 |

**The fence is 77 % of the added cost, and the weakest legal ordering is no cheaper than the
strongest on cc 12.0.** 0.077 ms/it of fence against a 0.058 ms/it node is a straight loss. Per
launch the fence costs 0.8 us at grid 27 and 1.6 us at grid 225 — it is a fixed drain plus a
per-block component, and it is always in the same range as the node it removes.

### Design 2 — fold into `fused_update_p_beta`, with no added synchronisation

The obstruction to the obvious fold is one write: `*d_rz = rz_new` races with every thread's read of
`d_rz` for `beta = rz_new / rz`. The resolution needs no barrier:

- every thread recomputes `conv` and `beta` itself, from the same operands;
- the three values the node *published* — `d_converged`, `d_rz`, `d_pAp` — are each read by **no
  thread of this kernel** (their consumers are the next iteration's `fused_update_xr`, the
  preconditioner, and the next SpMV), so block 0 / thread 0 can write them with no ordering at all;
- the divisor is taken from a new `d_rz_prev`, carried aside by `fused_update_xr` earlier in the
  same iteration. That kernel only *reads* `d_rz` (every thread does, for alpha) and writes a
  location nothing in it reads. No seeding is needed: `fused_update_xr` always precedes
  `fused_update_p` in the same iteration, and the very first one reads the init dot's `rz0`.

Zero fences, zero atomics, zero barriers, one extra predicated 8-byte store. And still not worth it:

| scene | grid | A = node, ms/Newton-it | B = folded | delta | runs | distributions |
|---|---|---|---|---|---|---|
| rigid-wrecking-balls | 27 | 6.4159 | 6.3253 | **−1.41 %** | 3 + 3 | disjoint |
| rigid-wrecking-balls | 27 | 6.3965 | 6.3433 | −0.83 % | 5 + 5 | **overlapping**, t = 1.39, p ≈ 0.2 |
| cube-wall-cloth | 140 | 10.7658 | 10.7547 | −0.10 % | 3 + 3 | overlapping |
| mas-bunny | 225 | 9.4322 | 9.5011 | **+0.73 %** | 3 + 3 | **disjoint — a regression** |
| stiff-gipc-case2 | 500 | 21.4580 | 21.5837 | **+0.59 %** | 3 + 3 | **disjoint — a regression** |

**The three-run result on wrecking-balls was luck.** Two more runs per arm turned a disjoint −1.41 %
into an overlapping −0.83 %, and the scope measurement says why the truth is smaller still: the node
is worth 0.0200 ms/it, the fold hands 0.0053 back, net **−0.0146 ms/it = −0.23 % of kernel time** —
five times below wrecking-balls' own ±1.3 % per-iteration scatter. *Three runs are not enough for a
sub-1 % effect on a scene that scatters like this; the round's ">= 3 runs" floor is a floor.*

### Why the sign flips with grid — and the law that generalises

**The node's cost is fixed; the fold's cost is per block, hence per wave.** The folded kernel adds
three scalar loads and one predicated store per block; measured, that is 10–13 ns per block:

| scene | grid | waves on 36 SMs | added kernel cost | node removed | net |
|---|---|---|---|---|---|
| rigid-wrecking-balls | 27 | 0.75 | +0.0053 | −0.0200 | −0.0146 |
| mas-bunny | 225 | 6.25 | +0.1421 | −0.0579 | **+0.0842** |

The crossover is bracketed between grid 27 and grid 140 and was not resolved further.
`UIPC_PCG_FOLD_MAXGRID` (default = the device SM count) gates mode 1 on exactly this; with the gate
on, only wrecking-balls folds and the other three scenes run the pre-s27 chain, confirmed in the
trace. It is still shipped **off**, because even the gated case is not demonstrable.

### Numerics — bit-identical, proved on device

`UIPC_PCG_FOLD_VERIFY=1` runs the pre-s27 chain as the reference, snapshots every output, **restores
the read-modify-write ones** (`p` and `Ap` are `p = z + beta*p` and a zeroing store; `d_rz`,
`d_converged`, `d_pAp` are scalars the reference overwrites) using s21's `SpreadVerifier`
save/restore phase unchanged, then runs the folded kernel and compares 32-bit words on device. Run
with the grid gate disabled so all four grids are exercised:

| scene | frames | output words (32-bit) | mismatching |
|---|---|---|---|
| stiff-gipc-case2 | 15 | 807 912 460 | **0** |
| cube-wall-cloth | 30 | 378 592 250 | **0** |
| mas-bunny | 15 | 172 740 750 | **0** |
| rigid-wrecking-balls | 60 | 109 401 290 | **0** |
| **total** | | **1 468 646 750** | **0** |

This was the step where bit-identity was *not* automatic — the scalars are computed in a different
thread context, by every thread instead of one. They come out bit-identical because the operands are
the same values and the expressions are compiled from one shared `pcg_scalar_body`; the proof is the
table, not that argument.

The verifier snapshots on the default stream, which cannot happen inside a stream capture, so it
forces the plain launch path. The captured and plain paths launch the same kernels with the same
arguments in the same order (the property s11/s13 already rely on), so this does not weaken the
claim — but it is a stated limit: **the graph-replayed path is not what the verifier ran.**

---

## 2. s28 — `cuda_tool`'s own fill launches

### The population

Two kernels, identical bodies (`if(i >= n) return; dst[i] = value;`), behind `DeviceVector::fill`,
`BufferLaunch::fill` and `BufferView::fill`. On wrecking-balls **811 of 1 768 fill launches per
60-frame window run at gridDim <= 36** on a 36-SM device, for 0.0298 ms/Newton-it.

### The heuristic had to move house

`spread_launch.h` includes `launch.h`, which includes `view.h` — so `view.h` and `buffer.h`, which
host the fill kernels, cannot include it. The dependency-free half (`device_sm_count`, the
`UIPC_GRID_SPREAD*` switches, and the warp-halving ramp, now `spread_block_dim_from`) moved to a new
`cuda_tool/spread_block.h`; `spread_launch.h` includes it and `spread_block_dim` is now one line.
Still one heuristic in one place — s21's consolidation rule, extended to the shared infrastructure.

### One block per SM, not eight — and why that matters more than it should

The kernel call sites ramp to 8 blocks/SM. **For a fill that is wrong**: it is bandwidth-bound, not
occupancy-bound, so once the grid covers the device there is nothing left to win, and the wider ramp
perturbs far more launches for nothing. Measured on stiff-gipc-case2, where the entire sub-SM fill
population is 0.003 ms/it (**0.013 %** of kernel time), bpsm = 8 moved ~700 of 3 300 fill launches
and produced:

| arm | ms/Newton-it | PCG iterations per run |
|---|---|---|
| A (off), 6 runs | 21.383 | 63 635 / 63 535 / 62 635 / 64 020 / 63 980 / 62 580 |
| B (bpsm 8), 6 runs | 21.533 | 64 245 / 64 960 / 64 745 / 64 735 / 64 240 / 64 875 |

**+1.95 % PCG iterations per Newton step, consistently, distributions disjoint over six runs per
arm** — for a geometry change worth 0.003 ms/it. The fill is bit-identical (proved below), so this is
not a numerical change in the fill: atomic accumulation order elsewhere in the solver is
timing-dependent, and shifting *any* kernel's block schedule shifts it. Per PCG iteration arm B was
1.1 % *faster*; per Newton iteration it was 0.70 % slower. Neither number means anything.

With bpsm = 1 the shift is gone — case2 4 runs per arm: 21.5256 -> 21.4844 (−0.19 %, overlapping),
PCG 64 320 -> 64 193 (−0.2 %, overlapping) — and wrecking-balls keeps essentially the whole gain
(−0.0202 vs −0.0210 ms/it). **bpsm = 1 ships.** The lesson is general: *a geometry change that buys
nothing still costs something, because it moves the trajectory of a solver whose reductions are
order-dependent. Only change geometry where it pays.*

### Scope — nsys, ms per Newton iteration, whole fill family

| scene | A (off) | B (shipped) | delta | of kernel time | sub-SM population |
|---|---|---|---|---|---|
| **rigid-wrecking-balls** | 0.04365 | **0.02347** | **−46 %** | **−0.32 %** | 0.02980 -> 0.00309 |
| cube-wall-cloth | 0.09370 | 0.08482 | −9.5 % | −0.12 % | 0.01518 -> 0.00342 |
| mas-bunny | 0.24351 | 0.24150 | −0.8 % | −0.02 % | 0.00412 -> 0.00079 |
| stiff-gipc-case2 | 0.57356 | 0.57306 | −0.1 % | −0.00 % | 0.00284 -> 0.00078 |

**This corrects the expectation in the step's own brief.** "One line inside `BufferLaunch` fixes
every scene at once" is not what the measurement says: case2's and mas-bunny's fill time — 2.7 % and
2.8 % of kernel time, far more than wrecking-balls' — sits almost entirely in grids of 500 to 7 348
blocks. That is bandwidth, and no geometry change touches it. The sub-SM population that *can* be
fixed is 0.013 % and 0.050 % of those two scenes.

### End-to-end — and an honest statement that it does not resolve

| scene | runs/arm | A | B | delta | note |
|---|---|---|---|---|---|
| rigid-wrecking-balls | 5 | 6.4003 | 6.3780 | −0.35 % | overlapping; matches the −0.32 % scope prediction |
| cube-wall-cloth | 3 | 10.6858 | 10.7530 | +0.63 % | overlapping; PCG +1.7 %, −0.96 %/PCG-it |
| **mas-bunny** | 3 | 9.4236 | 9.4242 | **+0.006 %** | **the tightest scene in the suite (±0.1 %), exactly neutral, PCG identical** |
| stiff-gipc-case2 | 4 | 21.5256 | 21.4844 | −0.19 % | overlapping |

**The end-to-end gate does not resolve this change.** A −0.32 % effect on one scene cannot be seen
against ±1.2 % scatter in five runs (it would take ~76 runs per arm). What the table *does*
establish is the thing that matters for a change to shared infrastructure: **no scene regresses**,
and on mas-bunny — Newton 465 in every run, PCG matched — the two arms agree to 0.006 %.

It ships on: a disjoint −46 % on the targeted kernel family, zero measured regression anywhere,
bit-identical output, no added work, and a pathology that gets *worse* on a wider GPU (a grid-5 fill
uses 5/36 of this device and 5/170 of a 5090).

### Numerics — coverage, because a bit-comparison would be vacuous

A fill writes a **constant**. Comparing two geometries' outputs word for word — the s21/s24
instrument — proves nothing here: any geometry that covers `[0, n)` produces identical bytes. What a
geometry change could actually break is **coverage**. So `UIPC_BUFFER_FILL_VERIFY=1` poisons the
destination with `0xA5` bytes, runs the shipped fill, and compares every 32-bit word of every element
against the corresponding word of `value`; a missed element keeps `0xA5A5A5A5` and is counted.

| scene | frames | fills checked | output words (32-bit) | mismatching |
|---|---|---|---|---|
| stiff-gipc-case2 | 20 | 688 | 2 803 166 070 | **0** |
| mas-bunny | 20 | 525 | 908 546 426 | **0** |
| cube-wall-cloth | 20 | 755 | 367 064 996 | **0** |
| rigid-wrecking-balls | 20 | 562 | 56 477 145 | **0** |
| **total** | | **2 530** | **4 135 254 637** | **0** |

0 fills skipped for being inside a stream capture and 0 for an element size that is not a multiple
of 4. Limits: a fill whose `value` is literally `0xA5A5A5A5…` in every word would hide a miss (none
exists in this backend), and fills issued during a stream capture are skipped by construction —
there were none in these runs.

**Cooperation audit** (s21's precondition): both kernel bodies are elementwise on
`blockIdx.x * blockDim.x + threadIdx.x` with an `i >= n` guard; `buffer.h` and `view.h` contain no
`__shared__`, no `__syncthreads`, no `__shfl`, no `*_sync`, no `__activemask` and no atomics outside
the verifier itself. Block sizes stay whole warps. No caller was excluded.

---

## 3. Correctness gate

`bash /root/work/run_tests.sh`, on the exact committed source:

| suite | s27+s28 | this box's baseline |
|---|---|---|
| common | 11 / 3 | 11 / 3 |
| core | 1 112 / 36 | 1 112 / 36 |
| geometry | 2 730 / 46 | 2 730 / 46 |
| sanity_check | 100 / 3 | 100 / 3 |
| regression | 4 / 1 | 4 / 1 |
| backend_cuda | **448 / 23** | 352 / 22 |
| sim_case | 14 213 / 95 | 14 213 / 95 |
| pytest (`cuda and not example`) | 48 passed, 1 skipped | 48 passed, 1 skipped |

The one count difference is **pre-existing at this step's base**: the extra case is
`fixed-sweep 3x3 SVD matches the iterative QR-SVD on GPU`, from w1's s23 (`d8a2e91c`,
`apps/tests/backends/cuda/qr_svd.cu`), already in `perf/round5`. Confirmed by listing the binary's
cases, and this step's diff touches no test source. **This box's `baseline_tests.txt` is now one
test case stale for every worker that branches off `perf/round5`.**

`sim_case` and `backend_cuda` also pass with each retained arm forced on — `UIPC_PCG_FOLD=1`,
`UIPC_PCG_FOLD=1 UIPC_PCG_FOLD_MAXGRID=0`, `UIPC_PCG_FOLD=2` — and `sim_case` passes with the s28
rollback `UIPC_BUFFER_FILL_SPREAD=0`.

---

## 4. Cross-architecture expectation

**s28 transfers, and grows.** A sub-SM launch is not an occupancy problem; it is a launch that never
reaches the machine. A grid-5 fill occupies 5/36 of this device and **5/170 of a 5090** — the deficit
scales with SM count, so every one of the 811 sub-SM fill launches per wrecking-balls window costs
proportionally ~4.7x more of the target GPU than it does here. The absolute numbers are small on any
architecture; the *share* is larger there.

**s27 is the interesting one, and it should be re-tested on the acceptance box.** The two costs move
in opposite directions with SM count:

- the node's ~0.93 us is **fixed** — it is dispatch and one dependent memory round trip, and if
  anything it becomes a larger share of a faster frame, which is the standing cross-architecture
  argument for launch overhead;
- the fold's cost is **per block, paid once per wave**. Grid 225 is 6.25 waves on 36 SMs and **1.3
  waves on ~170**, so mas-bunny's +0.142 ms/it should fall by roughly 5x, to ~0.03 — below the
  0.058 node it removes. Grid 500 on case2 goes 13.9 waves -> 2.9.

So the prediction is that **mode 1 flips from a regression to a win on a 5090 on every scene**, and
the arm to test it is one environment variable: `UIPC_PCG_FOLD=1 UIPC_PCG_FOLD_MAXGRID=0`. That is
why the code ships in tree rather than being deleted. It is a prediction, not a measurement, and it
is the coordinator's to take or leave.

A caveat in the other direction: nothing here says the *node* costs 0.93 us on a 5090. If graph node
overhead is largely driver- and dispatch-bound it will be similar; if it is partly memory-latency
bound it will be smaller, and the trade moves back.

---

## 5. Candidates for the next step

1. **`ipc_simplex_normal_contact::do_assemble` launch geometry** — still the single largest
   remaining launch-geometry item in the suite, and still unclaimed. Measured here on `d3f3a553`:
   the `do_assemble<...>` template family is **2.301 ms/Newton-it = 35.9 % of wrecking-balls
   kernel time** over 539 launches, spread across grids 8-167 at 255 registers, the biggest
   single row being grid 28 at 0.4343 ms/it (6.8 %). At 255 registers exactly one 256-thread
   block is resident per SM, so grid 28 is one wave with 8 SMs idle and grid 31 leaves 5 idle.
   s24 left it alone deliberately (it contains
   `__any_sync(__activemask(), …)`; warp-quantised block sizes preserve warp composition, so the
   change is safe, but it is a *throughput* change to the round's largest kernel and needs its own
   measurement). **s28's `spread_block.h` is exactly the helper to do it with.** Caveat: these
   numbers are from `d3f3a553` and **predate w2's contact part-1 merge (`29aef03c`)**, which
   touches this very kernel — re-profile before picking it up.
2. ~~**The large-grid fills.**~~ **Closed while this step ran** — w0's `bf3457ba` removed
   `triplet_A.values().fill(Matrix3x3::Zero())` outright as dead work (745 us/Newton-it on
   case2). That is exactly the `buffer_view_fill_kernel<Matrix3x3>` at grids of 500–7 348
   measured above, so the fill-family baselines in section 2 (0.574 ms/it on case2, 0.244 on
   mas-bunny) are **superseded on `perf/round5`**. The sub-SM population s28 targets is
   untouched by it, and s28's A/B was taken in one build, so that measurement stands — but a
   re-profile should confirm that what is left of the fill family is almost entirely the
   sub-SM part s28 fixes.

3. **`fused_update_xr` / `fused_dot`** are 0.33 and 0.29 ms/it (4.0 % and 3.5 %) on mas-bunny and
   0.37 / 0.30 on case2 — 8 % of those two scenes in three trivially memory-bound vector kernels
   over the same vectors. They are already at 256 threads and grids of 225–500, so this is a fusion
   question (fewer passes over `x`, `r`, `p`, `z`, `Ap`), not a geometry one. It is inside the
   linear solver, which is now w3's.
4. **`InfoStacklessBVH_pairFilter`** (grids 1 024-1 772) and **`stacklessSelf`** (grid 110) are
   0.354 and 0.379 ms/it on wrecking-balls (**5.5 % and 5.9 %**) -- untouched all round, already
   covering the device several times over, so not a geometry item.

## 6. Found in other workers' areas

- **w2 (D2H):** nothing new; s22 is in the measured build and the PCG loop's host interaction was
  not touched here beyond the scalar node.
- **Everyone:** the `nsys stats` overwrite defect in section 0 affects any `audit.py` user who
  re-used a prefix. The fix is in `/root/work/audit.py` on this box only — **the other boxes' copies
  still have it.**
- **Coordinator:** `baseline_tests.txt` on every box is now one Catch2 case behind `perf/round5`
  (w1's s23). Every later worker will see `backend_cuda 448/23` against a baseline of `352/22` and
  has to re-derive that it is benign, or will wave it through. Worth refreshing centrally.

#!/usr/bin/env python3
"""s17 v2: the recoverable GPU-idle bubble around each PCG convergence readback.

Method: merge ALL gpu-trace rows (every stream, kernels + memcpys) into a
global busy timeline. For each PCG readback (cudaGraphLaunch -> memcpyAsync ->
cudaStreamSynchronize), the sync-return moment Se_k sits inside a global idle
gap (the GPU finished the block's last row + the 8-byte D2H before the host
resumed). The part of that gap that ends at the NEXT PCG-family kernel is the
host bubble the poll could recover: [gap_start, next_pcg_kernel_start].
Reports the count, the total, and the split at Se_k (sync-return latency vs
post-resume host work). Gaps whose following kernel is not PCG-family are the
solve-exit boundaries (attributed to the Newton loop, not this readback).
"""
import csv, sys, statistics, bisect
from collections import defaultdict

api_csv, gpu_csv = sys.argv[1], sys.argv[2]

PCG_MARKERS = ("Spmv_rbk_sym", "fused_update_xr", "fused_dot", "fused_pcg_scalar",
               "fused_update_p", "MASPreconditionerEngine", "abd_diag", "buffer_copy",
               "fused_swap_rz", "fused_update_converged", "Memcpy")

rows = []
with open(gpu_csv) as f:
    for row in csv.DictReader(f):
        try:
            s = int(row["Start (ns)"]); d = int(row["Duration (ns)"])
        except ValueError:
            continue
        nm = row["Name"].split("(")[0].split("::")[-1]
        rows.append((s, s + d, nm, row["Strm"]))
rows.sort()
# merge into busy intervals + remember the name that starts each merged block
merged = []
for s, e, nm, st in rows:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e, nm])
gaps = []  # (gap_start, gap_end, kernel_after_name)
for a, b in zip(merged, merged[1:]):
    if b[0] > a[1]:
        gaps.append((a[1], b[0], b[2]))
gap_starts = [g[0] for g in gaps]

def gap_containing(t):
    i = bisect.bisect_right(gap_starts, t) - 1
    if i >= 0 and gaps[i][0] <= t < gaps[i][1]:
        return gaps[i]
    return None

api = []
with open(api_csv) as f:
    for row in csv.DictReader(f):
        api.append((int(row["Start (ns)"]), int(row["Duration (ns)"]), row["Name"]))
api.sort()

pcg_sync_ends = []
i = 0
while i < len(api):
    s, d, nm = api[i]
    if nm == "cudaMemcpyAsync" and i + 1 < len(api) and api[i + 1][2] == "cudaStreamSynchronize":
        j = i - 1
        steps = 0
        while j >= 0 and steps < 8:
            if api[j][2] == "cudaGraphLaunch_v10000":
                pcg_sync_ends.append(api[i + 1][0] + api[i + 1][1])
                break
            j -= 1
            steps += 1
    i += 1

span = (merged[-1][1] - merged[0][0]) / 1e9 if merged else 0
tot_busy = sum(e - s for s, e, _ in merged)
print(f"window={span:.2f}s busy={tot_busy/1e9:.2f}s idle={(span*1e9-tot_busy)/1e9:.2f}s "
      f"pcg_readbacks={len(pcg_sync_ends)}")

recov, pre_sync, post_sync, exits = [], [], [], 0
for se in pcg_sync_ends:
    g = gap_containing(se)
    if g is None:  # gpu already busy at sync return (other stream) -> nothing recoverable
        continue
    gs, ge, nm_after = g
    if not any(m in nm_after for m in PCG_MARKERS):
        exits += 1
        continue
    recov.append(ge - gs)
    pre_sync.append(se - gs)
    post_sync.append(ge - se)

if recov:
    rs = sorted(recov)
    print(f"recoverable bubble (gap ends at next PCG kernel): n={len(recov)} exits={exits} "
          f"total={sum(recov)/1e6:.1f} ms med={rs[len(rs)//2]/1e3:.2f} us "
          f"p90={rs[int(len(rs)*0.9)]/1e3:.2f} us")
    print(f"  split at sync-return: sync-return latency med={statistics.median(pre_sync)/1e3:.2f} us "
          f"(total {sum(pre_sync)/1e6:.1f} ms); post-resume host work med={statistics.median(post_sync)/1e3:.2f} us "
          f"(total {sum(post_sync)/1e6:.1f} ms)")
else:
    print(f"no recoverable bubbles; exits={exits}")

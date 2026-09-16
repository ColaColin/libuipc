#!/usr/bin/env python3
"""s09: attribute the cudaStreamSynchronize host parks around the
GlobalDyTopoEffectManager::_distribute D2H reads.

Reads the nsys cuda_api_trace + cuda_gpu_trace CSVs (12-frame crease-press
window) and, for every blocking readback (cudaMemcpyAsync DtoH immediately
followed by cudaStreamSynchronize, same tid):
  * names the last <=3 kernels launched before the memcpy (via CorrId),
  * splits the sync's wall duration into drain (GPU kernels still running at
    sync start) and stall (GPU empty),
  * tags readbacks whose launch context contains distribute_k1/k3 or the
    cub scan -> the _distribute sites;
and prints: per-tag counts/durations, the top parks, and the GPU idle gap
after each iteration's last distribute_k4 (phase-entry bubble) with the name
of the next kernel that starts after it.
"""
import csv, sys
from collections import defaultdict

api_csv, gpu_csv = sys.argv[1], sys.argv[2]

# ---- gpu trace: corrId -> (start, end, name); timeline of kernel intervals
kern_by_corr = {}
intervals = []  # (start, end, name)
with open(gpu_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            corr = int(row["CorrId"])
        except (KeyError, ValueError, TypeError):
            corr = None
        name = row["Name"]
        if row.get("GrdX") not in (None, ""):  # a kernel launch row
            s = int(row["Start (ns)"]); d = int(row["Duration (ns)"])
            intervals.append((s, s + d, name))
            if corr is not None:
                kern_by_corr[corr] = name
intervals.sort()
print(f"gpu kernels: {len(intervals)}")

# ---- api trace, per thread: sequence of (start, dur, name, corrId)
rows = []
with open(api_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append((int(row["Start (ns)"]), int(row["Duration (ns)"]),
                     row["Name"], row.get("CorrID") or row.get("CorrId"),
                     int(row["Tid"])))
rows.sort()
print(f"api rows: {len(rows)}")

# index kernel ends for drain computation
ends = [e for _, e, _ in intervals]
import bisect
def pending_end(t0):
    """end of the last kernel still running at t0 (>= t0), else None"""
    i = bisect.bisect_right(ends, t0)
    # walk back to find a kernel with start < t0 <= end
    j = i - 1
    while j >= 0 and i - j < 4000:
        if intervals[j][0] < t0 <= intervals[j][1]:
            return intervals[j][1]
        if intervals[j][1] <= t0 and intervals[j][0] < t0:
            break
        j -= 1
    return None

readbacks = []  # (sync_start, sync_end, dur, tag, ctx_names)
launch_hist = []  # (api_start, corr, name_is_launch)
i = 0
n = len(rows)
by_tid = defaultdict(list)
for row in rows:
    by_tid[row[4]].append(row)

for tid, trs in by_tid.items():
    last_launches = []  # (api_start, corr)
    for k, (s, d, name, corr, _) in enumerate(trs):
        if name == "cudaLaunchKernel":
            try:
                last_launches.append((s, int(corr)))
            except (TypeError, ValueError):
                pass
            if len(last_launches) > 8:
                last_launches.pop(0)
        elif name == "cudaMemcpyAsync":
            # look ahead: immediate sync on same tid?
            if k + 1 < len(trs) and trs[k + 1][2] == "cudaStreamSynchronize":
                ss, sd = trs[k + 1][0], trs[k + 1][1]
                ctx = []
                for ls, lc in last_launches[-3:]:
                    nm = kern_by_corr.get(lc)
                    if nm:
                        # shorten
                        nm = nm.split("::")[-1].split("(")[0]
                        ctx.append(nm)
                tag = "other"
                joined = " ".join(ctx)
                if "distribute_k1" in joined or "distribute_k3" in joined:
                    tag = "_distribute"
                elif "distribute" in joined:
                    tag = "distribute-other"
                elif "DeviceScan" in joined or "cub" in joined.lower() or "scan" in joined.lower():
                    tag = "scan/convert"
                readbacks.append((ss, ss + sd, sd, tag, ctx))

print(f"blocking readbacks (memcpyAsync+sync pairs): {len(readbacks)}")

stats = defaultdict(lambda: [0, 0, 0, 0])  # n, drain_ns, stall_ns, park_ns
top = []
for s0, s1, dur, tag, ctx in readbacks:
    pe = pending_end(s0)
    drain = max(0, pe - s0) if pe else 0
    stall = max(0, s1 - max(pe, s0)) if pe else dur
    stall = min(stall, dur)
    st = stats[tag]
    st[0] += 1; st[2] += stall; st[3] += dur
    st[1] += min(drain, dur)
    top.append((dur, drain, stall, tag, ctx))
print(f"{'tag':16s} {'n':>6s} {'park_ms':>9s} {'drain_ms':>9s} {'stall_ms':>9s} {'stall/rb_us':>11s}")
for tag, st in sorted(stats.items(), key=lambda kv: -kv[1][3]):
    print(f"{tag:16s} {st[0]:6d} {st[3]/1e6:9.1f} {st[1]/1e6:9.1f} {st[2]/1e6:9.1f} {st[2]/1e3/max(1,st[0]):11.2f}")

top.sort(reverse=True)
print("\ntop 12 parks overall (dur_us drain_us stall_us tag ctx):")
for dur, drain, stall, tag, ctx in top[:12]:
    print(f"  {dur/1000:9.1f} {drain/1000:9.1f} {stall/1000:9.1f} {tag:14s} {ctx}")

print("\ntop 8 _distribute parks:")
for dur, drain, stall, tag, ctx in [t for t in top if t[3] == "_distribute"][:8]:
    print(f"  {dur/1000:9.1f} {drain/1000:9.1f} {stall/1000:9.1f} {ctx}")

# ---- phase-entry bubble: GPU idle after the last distribute_k4 before the
# next kernel; report the gap and the next kernel's name.
print("\nphase-entry bubbles (idle after last distribute_k4 of a burst):")
gaps = []
for idx, (s, e, name) in enumerate(intervals):
    if "distribute_k4_kernel" in name:
        # last of a burst: next kernel is not k4
        if idx + 1 < len(intervals) and "distribute_k4_kernel" not in intervals[idx + 1][2]:
            ns, _, nname = intervals[idx + 1]
            gaps.append((ns - e, nname.split('::')[-1].split('(')[0]))
gaps.sort(reverse=True)
import statistics
if gaps:
    vals = [g for g, _ in gaps]
    print(f"  n={len(gaps)} median={statistics.median(vals)/1000:.1f}us mean={statistics.mean(vals)/1000:.1f}us "
          f"p90={sorted(vals)[int(0.9*len(vals))]/1000:.1f}us max={max(vals)/1000:.1f}us sum={sum(vals)/1e6:.2f}ms")
    for g, nm in gaps[:10]:
        print(f"    {g/1000:9.1f}us -> {nm}")

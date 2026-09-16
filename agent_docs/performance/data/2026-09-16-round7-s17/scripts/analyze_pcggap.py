#!/usr/bin/env python3
"""s17: price the PCG convergence-readback stall from an nsys api+gpu capture.

Instruments (per capture):
 1. The PCG readback: cudaMemcpyAsync + cudaStreamSynchronize right after a
    cudaGraphLaunch (the PCG block replay) -> count and park.
 2. The GPU-idle bubble between consecutive PCG graph blocks: on the stream
    carrying the graph-internal kernels, gaps between consecutive kernels; a
    gap is a BETWEEN-BLOCK boundary iff a cudaGraphLaunch api call overlaps it
    (the enqueue of the next block). Splits the bubble by whether any other
    stream was executing (recoverable = GPU otherwise idle).
 3. cudaGraphLaunch api durations (nsys-inflated; both arms equally).
"""
import csv, sys, statistics, bisect
from collections import defaultdict

api_csv, gpu_csv = sys.argv[1], sys.argv[2]

strm_iv = defaultdict(list)
nrows = 0
with open(gpu_csv) as f:
    for row in csv.DictReader(f):
        try:
            s = int(row["Start (ns)"]); d = int(row["Duration (ns)"])
        except ValueError:
            continue
        nrows += 1
        strm_iv[row["Strm"]].append((s, s + d, row["Name"].split("(")[0].split("::")[-1][:48]))
all_iv = sorted([(s, e, n, st) for st, iv in strm_iv.items() for (s, e, n) in iv])
starts = [x[0] for x in all_iv]

def gpu_busy(t0, t1):
    i = bisect.bisect_right(starts, t1)
    j = max(0, i - 400)
    while j < len(all_iv) and all_iv[j][0] < t1:
        s, e = all_iv[j][0], all_iv[j][1]
        if e > t0 and s < t1:
            return True
        j += 1
    return False

pcg_stream = None
for st, iv in strm_iv.items():
    if any("fused_update_xr" in n for _, _, n in iv):
        pcg_stream = st
        break

api = []
with open(api_csv) as f:
    for row in csv.DictReader(f):
        api.append((int(row["Start (ns)"]), int(row["Duration (ns)"]), row["Name"]))
api.sort()

glaunches = [(s, s + d, d) for s, d, nm in api if nm == "cudaGraphLaunch_v10000"]
gl_starts = [g[0] for g in glaunches]

# PCG readbacks: memcpyAsync followed by sync, with a graph launch just before
pcg_rb = []
i = 0
while i < len(api):
    s, d, nm = api[i]
    if nm == "cudaMemcpyAsync" and i + 1 < len(api) and api[i + 1][2] == "cudaStreamSynchronize":
        j = i - 1
        gl = None
        steps = 0
        while j >= 0 and steps < 8:
            if api[j][2] == "cudaGraphLaunch_v10000":
                gl = api[j]
                break
            j -= 1
            steps += 1
        if gl is not None:
            pcg_rb.append((s, api[i + 1][0], api[i + 1][0] + api[i + 1][1]))
    i += 1

span = (all_iv[-1][1] - all_iv[0][0]) / 1e9 if all_iv else 0
print(f"gpu rows={nrows} pcg_stream={pcg_stream} window={span:.2f}s")
print(f"graph_launches={len(glaunches)} pcg_readbacks={len(pcg_rb)}")
if pcg_rb:
    parks = [e - s for s, m, e in pcg_rb]
    print(f"pcg_readback park: total={sum(parks)/1e6:.1f} ms med={statistics.median(parks)/1e3:.2f} us")
if glaunches:
    ds = sorted(g[2] for g in glaunches)
    print(f"cudaGraphLaunch api us: med={ds[len(ds)//2]/1e3:.2f} p90={ds[int(len(ds)*0.9)]/1e3:.2f} mean={sum(ds)/len(ds)/1e3:.2f}")

if pcg_stream:
    iv = sorted(strm_iv[pcg_stream])
    gaps = []
    for a, b in zip(iv, iv[1:]):
        g = b[0] - a[1]
        if g > 2000:
            gaps.append((a[1], b[0], g))
    # classify: block boundary iff a graph-launch api call overlaps the gap
    blk, within = [], []
    for s, e, g in gaps:
        k = bisect.bisect_right(gl_starts, e)
        is_blk = False
        for q in range(max(0, k - 3), k):
            gs, ge, _ = glaunches[q]
            if ge > s and gs < e:
                is_blk = True
                break
        (blk if is_blk else within).append((s, e, g))
    for tag, lst in (("block-boundary", blk), ("within/other", within)):
        if not lst:
            print(f"{tag}: none")
            continue
        gs = sorted(g for _, _, g in lst)
        idle = sum(g for s, e, g in lst if not gpu_busy(s, e))
        tot = sum(gs)
        print(f"{tag}: n={len(lst)} total={tot/1e6:.1f} ms GPU-idle={idle/1e6:.1f} ms "
              f"med={gs[len(gs)//2]/1e3:.2f} us p90={gs[int(len(gs)*0.9)]/1e3:.2f} us")

#!/usr/bin/env python3
"""s09 v3: attribute cudaStreamSynchronize parks via the API trace alone.

For every cudaMemcpyAsync (DtoH) immediately followed by cudaStreamSynchronize
(same tid):
  tag    = names of the last 2 default-stream kernels launched before it,
  drain  = max(0, latest END of those default-stream kernels - sync_start),
  stall  = park - drain,
  elsewhere = kernels on other streams executing during the park.
"""
import csv, sys, bisect, statistics
from collections import defaultdict

api_csv, gpu_csv = sys.argv[1], sys.argv[2]

# gpu kernels: corr -> (end, strm, shortname); all intervals for busy-elsewhere
kern = {}
iv = []
with open(gpu_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            s = int(row["Start (ns)"]); d = int(row["Duration (ns)"])
        except ValueError:
            continue
        try:
            c = int(row["CorrId"])
        except (TypeError, ValueError):
            c = -1
        strm = row["Strm"]
        name = row["Name"].split("(")[0].split("::")[-1][:60]
        kern[c] = (s + d, strm, name)
        iv.append((s, s + d, name))
iv.sort()
ends = [e for _, e, _ in iv]

api = []
with open(api_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            c = int(row.get("CorrID") or -1)
        except ValueError:
            c = -1
        api.append((int(row["Start (ns)"]), int(row["Duration (ns)"]), row["Name"], c, int(row["Tid"])))
api.sort()
print(f"api rows={len(api)} gpu kernels={len(iv)}")

by_tid = defaultdict(list)
for row in api:
    by_tid[row[4]].append(row)

def busy_elsewhere(t0, t1):
    out = []
    i = bisect.bisect_left(ends, t0)
    j = i
    while j < len(iv):
        s, e, n = iv[j]
        if s >= t1:
            break
        if e > t0:
            out.append(n)
        j += 1
        if len(out) >= 4 or j - i > 50000:
            break
    return out

results = []
for tid, trs in by_tid.items():
    pend = []  # (corr) launches on this tid, in order
    for k, (s, d, nm, c, _) in enumerate(trs):
        if nm == "cudaLaunchKernel":
            pend.append(c)
            if len(pend) > 64:
                pend.pop(0)
        elif nm == "cudaMemcpyAsync":
            if k + 1 < len(trs) and trs[k + 1][2] == "cudaStreamSynchronize":
                s0 = trs[k + 1][0]; sd = trs[k + 1][1]
                s1 = s0 + sd
                # default-stream kernels among the pend list
                dfl = [kern[c] for c in pend if c in kern]
                ctx = [n for _, _, n in dfl[-2:]]
                maxend = max((e for e, _, _ in dfl), default=s0)
                drain = max(0, min(maxend, s1) - s0)
                stall = (s1 - s0) - drain
                cj = " ".join(ctx)
                if "distribute_k1" in cj or "distribute_k3" in cj:
                    tag = "_distribute"
                elif "distribute" in cj:
                    tag = "distribute-adj"
                elif any(("Radix" in x or "DeviceScan" in x or "Scan" in x) for x in ctx):
                    tag = "scan/sort"
                else:
                    tag = "other"
                results.append((s0, s1, drain, stall, tag, ctx))

print(f"readbacks: {len(results)}")
t0w = min(r[0] for r in results); t1w = max(r[1] for r in results)
span = t1w - t0w
print(f"window span {span/1e9:.2f}s (from first to last readback)")

stats = defaultdict(lambda: [0, 0, 0, 0])
for s0, s1, drain, stall, tag, ctx in results:
    st = stats[tag]; st[0] += 1; st[1] += drain; st[2] += stall; st[3] += s1 - s0

print(f"{'tag':16s} {'n':>6s} {'park_ms':>9s} {'drain_ms':>9s} {'stall_ms':>9s} {'stall/rb_us':>11s} {'park%span':>9s} {'drain%span':>10s}")
for tag, st in sorted(stats.items(), key=lambda kv: -kv[1][3]):
    print(f"{tag:16s} {st[0]:6d} {st[3]/1e6:9.1f} {st[1]/1e6:9.1f} {st[2]/1e6:9.1f} {st[2]/1e3/max(1,st[0]):11.2f} {100*st[3]/span:9.2f} {100*st[1]/span:10.2f}")

for want in ["_distribute", "distribute-adj"]:
    sel = sorted([r for r in results if r[4] == want], key=lambda r: -(r[1] - r[0]))
    if not sel:
        continue
    print(f"\n{want}: n={len(sel)} park med={statistics.median([r[1]-r[0] for r in sel])/1e3:.1f}us "
          f"drain med={statistics.median([r[2] for r in sel])/1e3:.1f}us stall med={statistics.median([r[3] for r in sel])/1e3:.1f}us")
    print(f"top 10 parks:")
    for s0, s1, drain, stall, tag, ctx in sel[:10]:
        be = busy_elsewhere(s0, s1)
        print(f"  park={(s1-s0)/1e3:9.1f}us drain={drain/1e3:9.1f}us stall={stall/1e3:7.1f}us ctx={ctx} elsewhere={be[:2]}")
    # stall sum with GPU idle (no elsewhere)
    idle_stall = 0; busy_stall = 0
    for s0, s1, drain, stall, tag, ctx in sel:
        be = busy_elsewhere(s0, s1)
        if be: busy_stall += stall
        else:  idle_stall += stall
    print(f"  stall with GPU idle elsewhere: {idle_stall/1e6:.2f}ms; stall while other-stream kernels ran: {busy_stall/1e6:.2f}ms")

# phase-entry bubble: gap after the last distribute_k4 before the next gpu kernel
kk = [(s, e, n) for s, e, n in iv if "distribute_k4" in n]
kk.sort()
gaps = []
for i, (s, e, n) in enumerate(kk):
    j = bisect.bisect_right(iv, (e, float("inf"), ""))
    if j < len(iv):
        # skip consecutive k4s
        if i + 1 < len(kk) and iv[j][2] == kk[i+1][2] and kk[i+1][0] == iv[j][0]:
            continue
        gaps.append((iv[j][0] - e, iv[j][2]))
gaps.sort(reverse=True)
print(f"\nphase-entry gaps after distribute_k4 bursts: n={len(gaps)}")
if gaps:
    vals = [g for g, _ in gaps]
    print(f"  med={statistics.median(vals)/1e3:.1f}us p90={sorted(vals)[int(0.9*len(vals))]/1e3:.1f}us max={max(vals)/1e3:.1f}us sum={sum(vals)/1e6:.2f}ms")
    for g, n in gaps[:6]:
        print(f"    {g/1e3:9.1f}us -> {n}")

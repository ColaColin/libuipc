#!/usr/bin/env python3
"""s09 v4: split the scan/sort readbacks into _distribute-hessian vs converter
vs GLS, by scanning the last 6 default-stream launches before each readback."""
import csv, sys, bisect, statistics
from collections import defaultdict
api_csv, gpu_csv = sys.argv[1], sys.argv[2]
kern = {}; iv = []
with open(gpu_csv) as f:
    for row in csv.DictReader(f):
        try: s=int(row["Start (ns)"]); d=int(row["Duration (ns)"])
        except ValueError: continue
        try: c=int(row["CorrId"])
        except (TypeError,ValueError): c=-1
        nm = row["Name"].split("(")[0].split("::")[-1][:60]
        kern[c]=(s+d,row["Strm"],nm); iv.append((s,s+d,nm))
iv.sort(); ends=[e for _,e,_ in iv]
api=[]
with open(api_csv) as f:
    for row in csv.DictReader(f):
        try: c=int(row.get("CorrID") or -1)
        except ValueError: c=-1
        api.append((int(row["Start (ns)"]),int(row["Duration (ns)"]),row["Name"],c,int(row["Tid"])))
api.sort()
by_tid=defaultdict(list)
for row in api: by_tid[row[4]].append(row)
def busy_elsewhere(t0,t1):
    out=[]; i=bisect.bisect_left(ends,t0); j=i
    while j<len(iv):
        s,e,n=iv[j]
        if s>=t1: break
        if e>t0: out.append(n)
        j+=1
        if len(out)>=3 or j-i>50000: break
    return out
results=[]
for tid,trs in by_tid.items():
    pend=[]
    for k,(s,d,nm,c,_) in enumerate(trs):
        if nm=="cudaLaunchKernel":
            pend.append(c)
            if len(pend)>80: pend.pop(0)
        elif nm=="cudaMemcpyAsync" and k+1<len(trs) and trs[k+1][2]=="cudaStreamSynchronize":
            s0=trs[k+1][0]; sd=trs[k+1][1]; s1=s0+sd
            dfl=[kern[c] for c in pend if c in kern]
            ctx=[n for _,_,n in dfl[-6:]]
            maxend=max((e for e,_,_ in dfl),default=s0)
            drain=max(0,min(maxend,s1)-s0); stall=(s1-s0)-drain
            cj=" ".join(ctx[-4:])
            if "distribute_k3" in cj or "distribute_k4" in cj: tag="_dist_hess"
            elif "distribute_k1" in cj or "distribute_k2" in cj: tag="_dist_grad"
            elif "matrix_converter" in cj: tag="matrix_converter"
            elif any(("Radix" in x or "DeviceScan" in x) for x in ctx[-2:]): tag="cub/scan"
            else: tag="other"
            results.append((s0,s1,drain,stall,tag,ctx[-4:]))
t0w=min(r[0] for r in results); span=max(r[1] for r in results)-t0w
print(f"readbacks={len(results)} span={span/1e9:.2f}s")
stats=defaultdict(lambda:[0,0,0,0])
for s0,s1,dr,st,tag,ctx in results:
    x=stats[tag]; x[0]+=1; x[1]+=dr; x[2]+=st; x[3]+=s1-s0
print(f"{'tag':17s} {'n':>6s} {'park_ms':>9s} {'drain_ms':>9s} {'stall_ms':>9s} {'med_park_us':>11s} {'park%span':>9s}")
for tag,x in sorted(stats.items(),key=lambda kv:-kv[1][3]):
    sel=[r for r in results if r[4]==tag]
    print(f"{tag:17s} {x[0]:6d} {x[3]/1e6:9.1f} {x[1]/1e6:9.1f} {x[2]/1e6:9.1f} {statistics.median([r[1]-r[0] for r in sel])/1e3:11.1f} {100*x[3]/span:9.2f}")
for want in ["_dist_hess","_dist_grad","matrix_converter"]:
    sel=sorted([r for r in results if r[4]==want],key=lambda r:-(r[1]-r[0]))
    if not sel: continue
    idle=sum(r[3] for r in sel if not busy_elsewhere(r[0],r[1]))
    busy=sum(r[3] for r in sel if busy_elsewhere(r[0],r[1]))
    print(f"\n{want}: n={len(sel)} park={sum(r[1]-r[0] for r in sel)/1e6:.2f}ms stall(GPU-idle)={idle/1e6:.2f}ms stall(GPU-busy-elsewhere)={busy/1e6:.2f}ms")
    for s0,s1,dr,st,tag,ctx in sel[:5]:
        print(f"   park={(s1-s0)/1e3:9.1f}us drain={dr/1e3:9.1f}us ctx={ctx}")

#!/usr/bin/env python3
"""s18 v5c: the reallocation-idle class + target-region totals.

1. All steady-window idle gaps where the host issued cudaFree/cudaMalloc
   during the gap (buffer growth -> GPU starves behind the realloc).
2. The full target-region idle: gaps whose bracket is inside the
   [classify -> assembly] span, by kernel-name bracket.
"""
import csv
import sys
import bisect
import statistics
from collections import defaultdict

gpu_csv, api_csv = sys.argv[1], sys.argv[2]
T_SKIP = 3.0e9
MIN_GAP = 3000

iv = []
with open(gpu_csv) as f:
    r = csv.reader(f)
    header = next(r)
    ix_s, ix_d, ix_n = header.index("Start (ns)"), header.index("Duration (ns)"), header.index("Name")
    for row in r:
        if len(row) <= ix_n:
            continue
        try:
            s = int(row[ix_s]); d = int(row[ix_d])
        except ValueError:
            continue
        nm = row[ix_n].split("(")[0].split("::")[-1].split("<")[0]
        iv.append((s, s + d, nm))
iv.sort()
t0 = iv[0][0]

gaps = []
cur_end = iv[0][1]
for s, e, nm in iv[1:]:
    if s > cur_end:
        gaps.append((cur_end, s - cur_end))
    cur_end = max(cur_end, e)

api = []
with open(api_csv) as f:
    r = csv.reader(f)
    header = next(r)
    ix_s, ix_d, ix_n = header.index("Start (ns)"), header.index("Duration (ns)"), header.index("Name")
    for row in r:
        if len(row) <= ix_n:
            continue
        try:
            s = int(row[ix_s]); d = int(row[ix_d])
        except ValueError:
            continue
        api.append((s, s + d, row[ix_n]))
api.sort()
api_starts = [a[0] for a in api]

def host_api(gs, ge):
    i = max(0, bisect.bisect_left(api_starts, gs) - 30)
    out = []
    while i < len(api) and api[i][0] < ge:
        s, e, nm = api[i]
        if e > gs:
            out.append((max(s, gs), min(e, ge), nm))
        i += 1
    return out

def prev_next_name(t):
    pn = None
    for s, e, nm in iv:
        if e <= t:
            if pn is None or e > pn[0]:
                pn = (e, nm)
        if s > t + 5:
            break
    nn = None
    for s, e, nm in iv:
        if s >= t:
            nn = nm
            break
    return (pn[1] if pn else "?"), (nn or "?")

# 1) realloc-idle gaps
ss = [(g, d) for g, d in gaps if g - t0 > T_SKIP and d >= MIN_GAP]
realloc_gaps = []
for gs, gd in ss:
    hs = host_api(gs, gs + gd)
    if any(("Free" in n or "Malloc" in n) for _, _, n in hs):
        realloc_gaps.append((gs, gd, hs))
tot = sum(g for _, g, _ in realloc_gaps)
n_fm = sum(1 for _, _, hs in realloc_gaps for _, _, n in hs if "Free" in n or "Malloc" in n)
print(f"realloc-idle gaps (steady, >=3us): n={len(realloc_gaps)} total_idle={tot/1e6:.1f}ms "
      f"(Free/Malloc api calls inside: {n_fm})")
for gs, gd, hs in sorted(realloc_gaps, key=lambda x: -x[1])[:15]:
    pn, nn = prev_next_name(gs)
    apis = ",".join(n.replace("cuda", "") for _, _, n in hs[:6])
    print(f"  t={(gs-t0)/1e9:6.2f}s idle={gd/1e3:8.1f}us | {pn[:30]:30s}->{nn[:30]:30s} | {apis[:70]}")

# also: total cudaFree/cudaMalloc count in the window (any host state)
n_free = sum(1 for _, _, n in api if "cudaFree" in n)
n_malloc = sum(1 for _, _, n in api if "cudaMalloc" in n and "Async" not in n)
n_ma = sum(1 for _, _, n in api if "cudaMallocAsync" in n)
t_free = sum(d for _, d, n in api if "cudaFree" in n)
t_malloc = sum(d for _, d, n in api if "cudaMalloc" in n)
print(f"\napi totals (whole capture): cudaFree n={n_free} sum={t_free/1e6:.1f}ms; "
      f"cudaMalloc n={n_malloc} sum={t_malloc/1e6:.1f}ms; mallocAsync n={n_ma}")

# 2) target-region brackets by name
TARGET_NEXT = ("distribute_k", "dytopo_pair", "buffer_fill", "buffer_view_fill",
               "Dahl", "Strain", "Stress", "NeoHookean", "StrainLimiting",
               "assemble", "FEMLinear")
cats = defaultdict(lambda: [0, 0])
for gs, gd in ss:
    pn, nn = prev_next_name(gs)
    key = None
    for t in TARGET_NEXT:
        if t in nn:
            key = f"{pn.split('(')[0][:28]}->{nn.split('(')[0][:28]}"
            break
    if key:
        cats[key][0] += 1
        cats[key][1] += gd
print(f"\ntarget-region idle (gaps whose NEXT kernel is classify/assembly):")
tot2 = 0
for key, (n, t) in sorted(cats.items(), key=lambda kv: -kv[1][1])[:15]:
    print(f"  {key:60s} n={n:4d} total={t/1e6:7.2f}ms")
    tot2 += t
print(f"  TOTAL {tot2/1e6:.1f}ms over steady span "
      f"= {100*tot2/(sum(e for _,e,_ in iv)-t0-T_SKIP):.3f}% of wall")

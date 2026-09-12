#!/usr/bin/env python3
"""Summarise an nsys cuda_gpu_trace csv by (kernel, grid, block) and diff two arms."""
import csv, sys, collections, re
def load(p):
    agg = collections.defaultdict(lambda: [0, 0.0])
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("Name") or ""
            if not name: continue
            dur = row.get("Duration (ns)") or row.get("Duration") or "0"
            try: dur = float(str(dur).replace(",", ""))
            except ValueError: dur = 0.0
            gx = row.get("GrdX"); bx = row.get("BlkX")
            if gx is None:   # memcpy/memset rows
                key = (name.split("[")[0].strip(), "", "")
            else:
                key = (name.split("(")[0][:120], gx, bx)
            a = agg[key]; a[0] += 1; a[1] += dur
    return agg
def fmt(k): return f"{k[0]}|g={k[1]}|b={k[2]}"
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) | set(b), key=lambda k: -(a.get(k,[0,0])[1] + b.get(k,[0,0])[1]))
print(f"{'key':<100} {'nA':>7} {'nB':>7} {'msA':>10} {'msB':>10}")
for k in keys:
    x = a.get(k, [0,0.0]); y = b.get(k, [0,0.0])
    if x[0] == y[0] and abs(x[1]-y[1]) < 1e-9: continue
    print(f"{fmt(k):<100} {x[0]:>7} {y[0]:>7} {x[1]/1e6:>10.3f} {y[1]/1e6:>10.3f}")

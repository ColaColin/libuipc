#!/usr/bin/env python3
"""s18 v5b: idle census restricted to the steady-state window, with real
kernel names for the top gaps and a phase-context attribution."""
import csv
import sys
import bisect
import statistics
from collections import defaultdict

gpu_csv, api_csv = sys.argv[1], sys.argv[2]
T_SKIP = float(sys.argv[3]) * 1e9 if len(sys.argv) > 3 else 0.0
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
span = cur_end - t0

# steady-state: skip the first T_SKIP seconds of the trace
ss_gaps = [(g, d) for g, d in gaps if g - t0 > T_SKIP]
ss_span = span - T_SKIP
ss_idle = sum(d for _, d in ss_gaps)
print(f"steady window: span={ss_span/1e9:.2f}s idle={ss_idle/1e6:.1f}ms "
      f"({100*ss_idle/ss_span:.2f}%)  gaps>=3us: "
      f"{sum(d for g, d in ss_gaps if d >= MIN_GAP)/1e6:.1f}ms")

def realname(t, before=True):
    if before:
        best = None
        for s, e, nm in iv:
            if e <= t and (best is None or e > best[0]):
                best = (e, nm)
            if s > t + 5:
                break
        return best[1] if best else "?"
    else:
        for s, e, nm in iv:
            if s >= t:
                return nm
        return "?"

# histogram of gap sizes in steady state
buckets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 500),
           (500, 2000), (2000, 10000), (10000, 1 << 60)]
print(f"{'gap_us':16s} {'n':>7s} {'total_ms':>9s}")
for lo, hi in buckets:
    sel = [d for _, d in ss_gaps if lo * 1000 <= d < hi * 1000]
    hi_s = f"{hi/1000:.0f}ms" if hi >= 10000 else f"{hi}"
    print(f"{lo}-{hi_s:>8s} {'':>{max(0, 6-len(str(lo)))}s} {len(sel):7d} {sum(sel)/1e6:9.1f}")

# top-25 gaps with names
print("\ntop gaps (steady):")
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

def host_state(gs, ge):
    i = bisect.bisect_left(api_starts, gs) - 20
    if i < 0:
        i = 0
    names = defaultdict(int)
    tot = 0
    while i < len(api) and api[i][0] < ge:
        s, e, nm = api[i]
        if e > gs:
            tot += min(e, ge) - max(s, gs)
            names[nm] += 1
        i += 1
    return dict(names), tot

for gs, gd in sorted(ss_gaps, key=lambda x: -x[1])[:25]:
    if gd < 20000:
        break
    pn = realname(gs, True)
    nn = realname(gs + gd, False)
    hs, tot = host_state(gs, gs + min(gd, 200000))
    hs_s = ",".join(f"{k.split('cuda')[-1]}x{v}" for k, v in sorted(hs.items(), key=lambda kv: -kv[1])[:3])
    print(f"  t={(gs-t0)/1e9:7.3f}s gap={gd/1e3:9.1f}us | {pn[:34]:34s} -> {nn[:34]:34s} | {hs_s} api={tot/1e3:.0f}us")

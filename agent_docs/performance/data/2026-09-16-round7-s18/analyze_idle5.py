#!/usr/bin/env python3
"""s18 v5: GPU-idle gap census over a crease-press api+gpu window.

The fixed-slot triplet_A candidate's prize is the GPU idle between phases:
the classify/pairs drain ends, the host walks (extents, resizes, launches),
and the GPU starves until the FEM/ABD G/H kernels arrive. This script
computes the busy union over ALL streams and attributes every idle gap by
the kernels that bracket it and by what the host thread was doing.

Usage: analyze_idle5.py GPU_TRACE.csv API_TRACE.csv [min_gap_us]
"""
import csv
import sys
import bisect
import statistics
from collections import defaultdict

gpu_csv, api_csv = sys.argv[1], sys.argv[2]
MIN_GAP = int(float(sys.argv[3]) * 1000) if len(sys.argv) > 3 else 3000  # ns

# ---------- load gpu trace ----------
iv = []
with open(gpu_csv) as f:
    r = csv.reader(f)
    header = next(r)
    ix_s = header.index("Start (ns)")
    ix_d = header.index("Duration (ns)")
    ix_n = header.index("Name")
    for row in r:
        if len(row) <= ix_n:
            continue
        try:
            s = int(row[ix_s]); d = int(row[ix_d])
        except ValueError:
            continue
        nm = row[ix_n]
        # short name
        nm = nm.split("(")[0].split("::")[-1].split("<")[0]
        iv.append((s, s + d, nm))
iv.sort()
print(f"gpu rows={len(iv)}")

# ---------- busy union -> idle gaps ----------
gaps = []
cur_end = iv[0][1]
t0 = iv[0][0]
for s, e, nm in iv[1:]:
    if s > cur_end:
        gaps.append((cur_end, s - cur_end))
    cur_end = max(cur_end, e)
span = cur_end - t0
total_idle = sum(g for _, g in gaps)
print(f"span={span/1e9:.2f}s busy-union={span-total_idle/1e9:.2f}s "
      f"idle={total_idle/1e9:.3f}s ({100*total_idle/span:.2f}% of span)")

# ---------- classify kernel names ----------
def klass(nm):
    if "distribute_k" in nm: return "dist_k"
    if "dytopo_pair" in nm: return "abd_pairs"
    if "matrix_converter" in nm or "Radix" in nm or "ScanTileState" in nm \
       or "ReduceByKey" in nm or "segmental_reduce" in nm or "RunLengthEncode" in nm \
       or "onesweep" in nm or "DeviceRadixSort" in nm or "histogram" in nm \
       or "zero_cross" in nm: return "convert_sort"
    if "buffer_fill_kernel" in nm or "buffer_view_fill" in nm: return "fill"
    if "Dahl" in nm or "dahl" in nm: return "GH_dahl"
    if "StrainPlastic" in nm or "StressPlastic" in nm or "strain_plastic" in nm or "stress_plastic" in nm: return "GH_plastic"
    if "NeoHookeanShell2D" in nm or "NHS2D" in nm: return "GH_nhs2d"
    if "StrainLimiting" in nm or "SLBW" in nm or "BaraffWitkin" in nm: return "GH_slbw"
    if "Spmv" in nm or "spmv" in nm: return "spmv"
    if "fused_pcg" in nm or "fused_update" in nm or "fused_dot" in nm: return "pcg_vec"
    if "MAS" in nm or "mas_" in nm or "cluster" in nm: return "MAS"
    if "assemble" in nm.lower(): return "assemble_other"
    if "contact" in nm.lower() or "Contact" in nm: return "contact"
    if "do_assemble" in nm: return "contact_asm"
    if "stackless" in nm or "BVH" in nm or "pairFilter" in nm or "filter" in nm.lower(): return "bvh_filter"
    if "energy" in nm.lower(): return "energy"
    return "other"

# bracket-class for each gap: (class of last kernel ending at gap start,
# class of first kernel starting at gap end)
ends_at = defaultdict(list)
for s, e, nm in iv:
    ends_at[e].append(nm)
klasses = [(s, e, klass(nm)) for s, e, nm in iv]

def prev_class(t):
    # latest interval with end <= t
    best = None
    i = bisect.bisect_right([e for _, e, _ in klasses], t)
    # linear scan back a bit (ends can tie)
    for j in range(min(i, len(klasses)) - 1, max(-1, i - 40), -1):
        s, e, k = klasses[j]
        if e <= t:
            return k
    return best or "?"

def next_class(t):
    i = bisect.bisect_right([s for s, _, _ in klasses], t)
    if i < len(klasses):
        return klasses[i][2]
    return "?"

# ---------- api trace: what the host did during each gap ----------
# collect intervals of the main-ish thread api rows
api = []
with open(api_csv) as f:
    r = csv.reader(f)
    header = next(r)
    ix_s = header.index("Start (ns)")
    ix_d = header.index("Duration (ns)")
    ix_n = header.index("Name")
    ix_t = header.index("Tid")
    for row in r:
        if len(row) <= ix_t:
            continue
        try:
            s = int(row[ix_s]); d = int(row[ix_d])
        except ValueError:
            continue
        api.append((s, s + d, row[ix_n]))
api.sort()
api_starts = [a[0] for a in api]

def host_state(gs, ge):
    """return (n_sync, n_launch, n_memcpy, api_ns) overlapping [gs,ge]"""
    n_sync = n_launch = n_mem = 0
    tot = 0
    i = bisect.bisect_left(api_starts, gs) - 1
    if i < 0:
        i = 0
    while i < len(api) and api[i][0] < ge:
        s, e, nm = api[i]
        if e > gs:
            ov = min(e, ge) - max(s, gs)
            if ov > 0:
                tot += ov
                if "Synchronize" in nm or "synchronize" in nm:
                    n_sync += 1
                    # sync that ENDS inside the gap = the wake
                elif "Launch" in nm:
                    n_launch += 1
                elif "Memcpy" in nm or "memset" in nm.lower():
                    n_mem += 1
        i += 1
        if i - bisect.bisect_left(api_starts, gs) > 20000:
            break
    return n_sync, n_launch, n_mem, tot

# ---------- gap census ----------
cats = defaultdict(lambda: [0, 0])  # (prev->next) -> [n, total_ns]
rows = []
for gs, gd in gaps:
    if gd < MIN_GAP:
        continue
    pc = prev_class(gs)
    nc = next_class(gs + gd)
    key = f"{pc}->{nc}"
    cats[key][0] += 1
    cats[key][1] += gd
    rows.append((gs, gd, pc, nc))

print(f"\ngaps>={MIN_GAP/1000:.0f}us: n={len(rows)} total={sum(r[1] for r in rows)/1e6:.1f}ms "
      f"({100*sum(r[1] for r in rows)/span:.2f}% of span)")
print(f"{'bracket':38s} {'n':>5s} {'total_ms':>9s} {'med_us':>8s} {'max_ms':>8s}")
for key, (n, tot) in sorted(cats.items(), key=lambda kv: -kv[1][1])[:22]:
    sel = [r[1] for r in rows if f"{r[2]}->{r[3]}" == key]
    print(f"{key:38s} {n:5d} {tot/1e6:9.1f} {statistics.median(sel)/1e3:8.1f} {max(sel)/1e6:8.2f}")

# ---------- the target region: classify/pairs -> assembly G/H ----------
# idle whose NEXT kernel is an assembly G/H kernel or the -1 fill, i.e. the
# post-park host walk; and idle whose next is FEM/ABD G/H specifically
for want_next in ["GH_dahl", "GH_plastic", "GH_nhs2d", "fill", "abd_pairs", "dist_k"]:
    sel = [r for r in rows if r[3] == want_next]
    if not sel:
        continue
    print(f"\nnext=={want_next}: n={len(sel)} total={sum(r[1] for r in sel)/1e6:.2f}ms "
          f"med={statistics.median([r[1] for r in sel])/1e3:.1f}us")
    from collections import Counter
    c = Counter(f"{r[2]}->{r[3]}" for r in sel)
    for k, n in c.most_common(6):
        tt = sum(r[1] for r in sel if f"{r[2]}->{r[3]}" == k)
        print(f"   {k:34s} n={n:4d} total={tt/1e6:8.2f}ms")

# ---------- biggest single gaps ----------
print("\nbiggest gaps:")
for gs, gd, pc, nc in sorted(rows, key=lambda r: -r[1])[:12]:
    ns, nl, nm_, tot = host_state(gs, gs + gd)
    print(f"   gap={gd/1e3:9.1f}us {pc:14s}->{nc:14s} host: sync={ns} launch={nl} "
          f"memcpy={nm_} api_overlap={tot/1e3:8.1f}us")

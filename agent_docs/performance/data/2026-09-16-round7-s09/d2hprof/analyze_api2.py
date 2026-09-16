#!/usr/bin/env python3
"""s09 v2: attribute cudaStreamSynchronize parks to the _distribute D2H reads.

Anchors on GPU-side small DtoH memcpy rows; the API-side sync that follows the
matching memcpyAsync (same CorrId) gives the park duration. Split:
  drain = memcpy_gpu_start - sync_api_start   (stream backlog the host waited)
  stall = sync_api_end    - memcpy_gpu_start  (copy + host wakeup round trip)
and whether any OTHER-stream kernel was executing during the park (GPU busy
elsewhere, e.g. the ABD prepass side stream).
"""
import csv, sys, bisect, statistics
from collections import defaultdict

api_csv, gpu_csv = sys.argv[1], sys.argv[2]

# ---------- gpu rows ----------
mem_rows = []      # (start, end, corr, stream, bytes_mb, kind)
kern_rows = []     # (start, end, corr, stream, name)
with open(gpu_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            s = int(row["Start (ns)"]); d = int(row["Duration (ns)"])
        except ValueError:
            continue
        corr = row["CorrId"]
        try:
            corr = int(corr)
        except (TypeError, ValueError):
            corr = -1
        strm = row["Strm"]
        name = row["Name"]
        if "Memcpy" in name or "memcpy" in name:
            try:
                b = float(row["Bytes (MB)"] or 0)
            except ValueError:
                b = 0
            kind = "DtoH" if "DtoH" in name else ("HtoD" if "HtoD" in name else "?")
            mem_rows.append((s, s + d, corr, strm, b, kind))
        else:
            kern_rows.append((s, s + d, corr, strm, name))
kern_rows.sort()
mem_rows.sort()
print(f"gpu kernels={len(kern_rows)} gpu memcpys={len(mem_rows)}")

small_d2h = [m for m in mem_rows if m[4] <= 0.001 and m[5] == "DtoH"]  # <= 1KB
print(f"small DtoH gpu rows: {len(small_d2h)}")

# kernel name lookup by corr + per-stream kernel timeline
name_by_corr = {c: n for _, _, c, _, n in kern_rows if c > 0}

# ---------- api rows ----------
api = []
with open(api_csv) as f:
    r = csv.DictReader(f)
    for row in r:
        corr = row.get("CorrID") or row.get("CorrId")
        try:
            corr = int(corr)
        except (TypeError, ValueError):
            corr = -1
        api.append((int(row["Start (ns)"]), int(row["Duration (ns)"]), row["Name"], corr, int(row["Tid"])))
api.sort()

api_by_corr = {c: (s, d, nm) for s, d, nm, c, _ in api if c > 0}

# per-stream kernel starts/ends for context and busy-elsewhere checks
strm_kerns = defaultdict(list)
for s, e, c, st_, n in kern_rows:
    strm_kerns[st_].append((s, e, n))
for st_ in strm_kerns:
    strm_kerns[st_].sort()
all_ends = sorted(e for _, e, _, _, _ in kern_rows)
all_iv = [(s, e, n) for s, e, c, st_, n in kern_rows]

def busy_elsewhere(t0, t1, exclude_strm):
    """any kernel on another stream overlapping [t0,t1]?"""
    i = bisect.bisect_left(all_ends, t0)
    # scan forward while start < t1
    out = []
    j = i
    while j < len(all_iv) and len(out) < 3:
        s, e, n = all_iv[j]
        if s >= t1:
            break
        if e > t0:
            out.append(n.split("::")[-1].split("(")[0][:40])
        j += 1
        if j - i > 20000:
            break
    return out

results = []
for ms, me, mcorr, mstrm, mb, mkind in small_d2h:
    apirow = api_by_corr.get(mcorr)
    if not apirow:
        continue
    a0, ad, anm = apirow
    # the sync: next cudaStreamSynchronize API row on the same tid after this memcpyAsync
    # (find via api sequence -- build per-corr index of following row by tid)
    # cheap: scan a window in api around a0
    i = bisect.bisect_left(api, (a0, -1, "", -1, -1))
    # api rows are tuples; find exact position of this memcpyAsync then look at i+1
    # (assume same tid)
    if i < len(api) and api[i][3] == mcorr:
        if i + 1 < len(api) and api[i + 1][2] == "cudaStreamSynchronize":
            s0, sd = api[i + 1][0], api[i + 1][1]
            s1 = s0 + sd
            # context: last 2 kernels on the memcpy's stream before the memcpy
            kl = strm_kerns.get(mstrm, [])
            j = bisect.bisect_left(kl, (me, 0, ""))  # first kernel starting at/after me
            ctx = [n.split("::")[-1].split("(")[0][:34] for _, _, n in kl[max(0, j - 2):j]]
            drain = max(0, ms - s0)
            stall = max(0, s1 - ms)
            tag = "other"
            cj = " ".join(ctx)
            if "distribute_k1" in cj or "distribute_k3" in cj:
                tag = "_distribute"
            elif "distribute" in cj:
                tag = "distribute-adjacent"
            elif any(("Scan" in c or "scan" in c or "Radix" in c or "DeviceRadix" in c) for c in ctx):
                tag = "scan/sort"
            results.append((s0, s1, drain, stall, tag, ctx, mstrm, mb))

print(f"readbacks matched (gpu memcpy -> api sync): {len(results)}")

stats = defaultdict(lambda: [0, 0, 0, 0])
for s0, s1, drain, stall, tag, ctx, mstrm, mb in results:
    st = stats[tag]
    st[0] += 1; st[1] += drain; st[2] += stall; st[3] += (s1 - s0)

tot_span = max(r[1] for r in results) - min(r[0] for r in results)
print(f"window span: {tot_span/1e9:.2f} s")
print(f"{'tag':18s} {'n':>6s} {'park_ms':>9s} {'drain_ms':>9s} {'stall_ms':>9s} {'stall/rb_us':>11s} {'park%win':>8s}")
for tag, st in sorted(stats.items(), key=lambda kv: -kv[1][3]):
    print(f"{tag:18s} {st[0]:6d} {st[3]/1e6:9.1f} {st[1]/1e6:9.1f} {st[2]/1e6:9.1f} {st[2]/1e3/max(1,st[0]):11.2f} {100*st[3]/tot_span:8.2f}")

# top parks per tag
for want in ["_distribute", "scan/sort"]:
    sel = sorted([r for r in results if r[4] == want], key=lambda r: -(r[1]-r[0]))
    print(f"\ntop 10 {want} parks (park_us drain_us stall_us bytes ctx):")
    for s0, s1, drain, stall, tag, ctx, mstrm, mb in sel[:10]:
        be = busy_elsewhere(s0, s1, mstrm)
        print(f"  {(s1-s0)/1e3:9.1f} {drain/1e3:9.1f} {stall/1e3:9.1f} {int(mb*1e6):6d}B {ctx} elsewhere={be[:2]}")

# per-iteration structure for _distribute: burst pattern
sel = sorted([r for r in results if r[4] == "_distribute"], key=lambda r: r[0])
if sel:
    print(f"\n_distribute readbacks: n={len(sel)}, median park {statistics.median([r[1]-r[0] for r in sel])/1e3:.1f}us, "
          f"median drain {statistics.median([r[2] for r in sel])/1e3:.1f}us, median stall {statistics.median([r[3] for r in sel])/1e3:.1f}us")
    big = [r for r in sel if r[2] > 1000]  # drains > 1ms
    print(f"  with drain>1ms: n={len(big)} sum={sum(r[2] for r in big)/1e6:.2f}ms")
    for s0, s1, drain, stall, tag, ctx, mstrm, mb in sorted(big, key=lambda r: -r[2])[:6]:
        be = busy_elsewhere(s0, s1, mstrm)
        print(f"    park={(s1-s0)/1e3:8.1f}us drain={drain/1e3:8.1f}us stall={stall/1e3:6.1f}us B={int(mb*1e6)} ctx={ctx} elsewhere={be[:2]}")

#!/usr/bin/env python3
"""Rare-event analysis: the drum-tracking transient and the area-ratio excursion.

Both worrying statistics in s02's audit are *extreme values* of a per-frame
series, so the right comparison is the event rate and the per-frame
distribution, not the run maximum of three runs.
"""
import json, math
from pathlib import Path
import numpy as np
from scipy import stats

P = json.loads(Path("/workspace/output/round6/v1/parsed.json").read_text())
OMEGA, DT = 2.0 * math.pi * 40.0 / 60.0, 1.0 / 60.0
arms = {}
for tag in sorted(P):
    arms.setdefault(tag.split("_")[0], []).append(tag)

def err_series(tag):
    tr = P[tag]["trace"]
    ang = np.unwrap([r["drum_angle"] for r in tr])
    cmd = OMEGA * DT * np.array([r["frame"] for r in tr])
    return np.degrees(ang - ang[0] - cmd), tr

print("=== drum tracking: per-run event counts (frames with |err| over a threshold) ===")
print(f"{'run':6s} {'max|err| f>=2':>13s} {'argmax':>7s} {'>0.1deg':>8s} {'>0.3deg':>8s} {'>1.0deg':>8s}")
ev = {}
for a in ("A", "Ap", "B"):
    for tag in arms[a]:
        e, tr = err_series(tag)
        e2 = np.abs(e[2:])                      # drop the frame-1 start-up transient
        ev.setdefault(a, []).append((int((e2 > 0.1).sum()), int((e2 > 0.3).sum()),
                                     int((e2 > 1.0).sum()), float(e2.max())))
        print(f"{tag:6s} {e2.max():13.4f} {int(np.argmax(e2))+2:7d} "
              f"{int((e2>0.1).sum()):8d} {int((e2>0.3).sum()):8d} {int((e2>1.0).sum()):8d}")
print()
for a in ("A", "Ap", "B"):
    v = np.array(ev[a], dtype=float)
    print(f"  {a:3s} n={len(v)}  frames >0.1deg: total {v[:,0].sum():.0f} "
          f"({v[:,0].sum()/ (len(v)*179) * 1e3:.2f} per 1000 frames);  runs with a >1deg event: "
          f"{(v[:,2]>0).sum()};  max|err| median {np.median(v[:,3]):.4f} "
          f"[{v[:,3].min():.4f}, {v[:,3].max():.4f}]")

print("\n=== what the solver did in the >0.3 deg frames (all arms pooled) ===")
print(f"{'run':6s} {'f':>4s} {'err':>9s} {'newton':>7s} {'run-mean N':>11s} {'pcg':>6s} "
      f"{'ls':>4s} {'toi':>8s} {'alpha':>8s} {'cfl':>8s} {'pairs/it':>9s} {'run-mean p':>11s}")
big = []
for a in ("A", "Ap", "B"):
    for tag in arms[a]:
        e, tr = err_series(tag)
        pairs = {p["frame"]: p["pairs_mean"] for p in P[tag]["pairs"]}
        mN = np.mean([r["newton"] for r in tr if "newton" in r])
        mP = np.mean(list(pairs.values()))
        for i in range(2, len(tr)):
            if abs(e[i]) > 0.3:
                r = tr[i]
                big.append((a, r.get("newton", -1), mN, pairs.get(r["frame"], np.nan), mP))
                print(f"{tag:6s} {r['frame']:4d} {e[i]:9.4f} {r.get('newton',-1):7d} {mN:11.2f} "
                      f"{r.get('pcg',-1):6d} {r.get('line_search',-1):4d} {r.get('last_ccd_toi',1):8.5f} "
                      f"{r.get('last_ls_alpha',1):8.5f} {r.get('last_cfl_alpha',1):8.5f} "
                      f"{pairs.get(r['frame'], float('nan')):9.0f} {mP:11.0f}")
if big:
    n = np.array([b[1] for b in big], float); m = np.array([b[2] for b in big], float)
    print(f"\n  Newton in the event frames: mean {n.mean():.2f} against the runs' own mean {m.mean():.2f}")

print("\n=== newton count vs |tracking error|, pooled per-frame, all runs ===")
E, N = [], []
for a in arms:
    for tag in arms[a]:
        e, tr = err_series(tag)
        for i in range(2, len(tr)):
            if "newton" in tr[i]:
                E.append(abs(e[i])); N.append(tr[i]["newton"])
E, N = np.array(E), np.array(N)
for lo, hi in [(0, .01), (.01, .03), (.03, .1), (.1, .3), (.3, 10)]:
    m = (E >= lo) & (E < hi)
    if m.sum():
        print(f"  |err| in [{lo:5.2f},{hi:5.2f}): n={m.sum():5d}  newton mean {N[m].mean():5.2f} "
              f"median {np.median(N[m]):4.1f}")

print("\n=== area_ratio_max: the per-frame series, not the run maximum ===")
print(f"{'arm':4s} {'n_frames':>9s} {'median':>8s} {'p95':>8s} {'p99.5':>8s} {'max':>8s} "
      f"{'frames>1.4':>11s} {'frames>1.8':>11s}")
series = {}
for a in ("A", "Ap", "B"):
    v = np.concatenate([[r["area_ratio_max"] for r in P[t]["trace"]] for t in arms[a]])
    series[a] = v
    print(f"{a:4s} {len(v):9d} {np.median(v):8.4f} {np.percentile(v,95):8.4f} "
          f"{np.percentile(v,99.5):8.4f} {v.max():8.4f} {int((v>1.4).sum()):11d} {int((v>1.8).sum()):11d}")
R = np.concatenate([series["A"], series["Ap"]])
u = stats.mannwhitneyu(series["B"], R, alternative="two-sided")
print(f"  per-frame B vs A+Ap: median {np.median(series['B']):.4f} vs {np.median(R):.4f}, "
      f"p={u.pvalue:.3g} (n is huge, so read the medians not the p)")
print(f"  frames above 1.4: B {(series['B']>1.4).mean()*1e2:.3f} % vs R {(R>1.4).mean()*1e2:.3f} %")

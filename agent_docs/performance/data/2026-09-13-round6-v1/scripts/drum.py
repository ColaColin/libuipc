#!/usr/bin/env python3
"""Where does the drum tracking error come from, and what did the solver do there?"""
import json, math, sys
from pathlib import Path
import numpy as np

P = json.loads(Path("/workspace/output/round6/v1/parsed.json").read_text())
OMEGA = 2.0 * math.pi * 40.0 / 60.0
DT = 1.0 / 60.0
print(f"{'run':6s} {'errmax':>8s} {'frame':>6s} {'newton':>7s} {'pcg':>6s} {'ls':>4s} "
      f"{'toi':>8s} {'alpha':>8s} {'cfl':>8s} {'pairs':>8s} {'d_err/frame':>11s}")
rows = []
for tag in sorted(P):
    tr = P[tag]["trace"]
    ang = np.unwrap([r["drum_angle"] for r in tr])
    cmd = OMEGA * DT * np.array([r["frame"] for r in tr])
    err = np.degrees(ang - ang[0] - cmd)
    k = int(np.argmax(np.abs(err)))
    r = tr[k]
    pairs = {p["frame"]: p for p in P[tag]["pairs"]}
    pk = pairs.get(r["frame"], {})
    rows.append((tag, float(np.abs(err).max()), r["frame"]))
    print(f"{tag:6s} {np.abs(err).max():8.4f} {r['frame']:6d} {r.get('newton',-1):7d} "
          f"{r.get('pcg',-1):6d} {r.get('line_search',-1):4d} {r.get('last_ccd_toi',1):8.5f} "
          f"{r.get('last_ls_alpha',1):8.5f} {r.get('last_cfl_alpha',1):8.5f} "
          f"{pk.get('pairs_mean',float('nan')):8.0f} "
          f"{err[k]-err[k-1] if k else 0:11.4f}")

# the worst run: dump its neighbourhood
worst = max(rows, key=lambda x: x[1])
tag, _, fr = worst
print(f"\n--- {tag}: frames around the peak ({fr}) ---")
tr = P[tag]["trace"]
ang = np.unwrap([r["drum_angle"] for r in tr]); cmd = OMEGA * DT * np.array([r["frame"] for r in tr])
err = np.degrees(ang - ang[0] - cmd)
pairs = {p["frame"]: p for p in P[tag]["pairs"]}
print(f"{'f':>4s} {'err_deg':>9s} {'newton':>7s} {'pcg':>6s} {'ls':>4s} {'toi':>8s} "
      f"{'alpha':>8s} {'cfl':>8s} {'pairs/it':>9s} {'ke':>9s} {'maxspeed':>9s}")
for i in range(max(0, fr - 8), min(len(tr), fr + 9)):
    r = tr[i]; pk = pairs.get(r["frame"], {})
    print(f"{r['frame']:4d} {err[i]:9.4f} {r.get('newton',-1):7d} {r.get('pcg',-1):6d} "
          f"{r.get('line_search',-1):4d} {r.get('last_ccd_toi',1):8.5f} {r.get('last_ls_alpha',1):8.5f} "
          f"{r.get('last_cfl_alpha',1):8.5f} {pk.get('pairs_mean',float('nan')):9.0f} "
          f"{r.get('ke',0):9.4f} {r.get('max_speed',0):9.4f}")

# correlation of |err| with the candidate mechanisms, pooled over every run
import itertools
E, N, T, A, PR, KE = [], [], [], [], [], []
for tag in sorted(P):
    tr = P[tag]["trace"]
    ang = np.unwrap([r["drum_angle"] for r in tr]); cmd = OMEGA*DT*np.array([r["frame"] for r in tr])
    e = np.abs(np.degrees(ang - ang[0] - cmd))
    pairs = {p["frame"]: p for p in P[tag]["pairs"]}
    for i, r in enumerate(tr):
        if "newton" not in r: continue
        E.append(e[i]); N.append(r["newton"]); T.append(r.get("last_ccd_toi",1.0))
        A.append(r.get("last_ls_alpha",1.0)); KE.append(r.get("ke",0.0))
        PR.append(pairs.get(r["frame"], {}).get("pairs_mean", np.nan))
E=np.array(E); 
print("\npooled per-frame correlation with |tracking error| (all 15 runs):")
for name, v in [("newton", N), ("last_ccd_toi", T), ("last_ls_alpha", A),
                ("pairs/it", PR), ("ke", KE)]:
    v = np.array(v, dtype=float)
    m = np.isfinite(v) & np.isfinite(E)
    print(f"  {name:14s} pearson r = {np.corrcoef(E[m], v[m])[0,1]:+.3f}")

#!/usr/bin/env python3
"""Round-6 V1: apply the pre-registered verdict rule to the three arms."""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy import stats

P = json.loads(Path("/workspace/output/round6/v1/parsed.json").read_text())
ARMS = {"A": [], "Ap": [], "B": []}
for tag, rec in sorted(P.items()):
    ARMS[tag.split("_")[0]].append(rec)

OBS = [
    ("verify_area_ratio_max", "up"), ("verify_area_ratio_min", None),
    ("verify_tri_height_min", "down"), ("verify_cc_min_dist", "down"),
    ("verify_bore_gap_min", "down"), ("verify_r_max", None),
    ("verify_abs_z_max", None), ("verify_lifter_depth_max", None),
    ("verify_max_speed", None), ("verify_ke_mean", "up"),
    ("verify_e_tot_last", None), ("verify_mean_disp_mm", None),
    ("verify_mean_disp_mm_last_quarter", None),
    ("verify_drum_track_err_deg_max", "up"),
    ("verify_drum_track_err_deg_final", None),
    ("verify_newton_total", None), ("verify_pcg_total", None),
    ("verify_line_search_total", None), ("verify_ccd_toi_min", None),
    ("verify_ccd_toi_clamped_frames", None), ("verify_ls_alpha_cut_frames", None),
    ("pairs_mean_per_it", None), ("pairs_last_quarter", None),
    ("pairs_converged_mean", None), ("pairs_converged_last_quarter", None),
    ("pairs_converged_max", None),
]

def vals(arm, key):
    out = []
    for rec in ARMS[arm]:
        s = rec["summary"]
        if key.startswith("centroid"):
            out.append(s["verify_centroid_turn_deg"][int(key[-1])])
        else:
            out.append(float(s[key]))
    return np.array(out)

for g in range(4):
    OBS.append((f"centroid_turn_{g}", None))

rows, flagged = [], []
for key, bad_dir in OBS:
    a, ap, b = vals("A", key), vals("Ap", key), vals("B", key)
    R = np.concatenate([a, ap])
    rng = R.max() - R.min()
    shift = abs(np.median(b) - np.median(R))
    u, p = stats.mannwhitneyu(b, R, alternative="two-sided")
    envelope_ok = (b.max() >= R.min() and b.min() <= R.max()
                   and R.min() <= np.median(b) <= R.max())
    eff = shift / rng if rng > 0 else float("inf") if shift > 0 else 0.0
    systematic = bool(p < 0.05 and eff > 0.5)
    bf = stats.levene(b, R, center="median")
    p1 = None
    if bad_dir:
        alt = "greater" if bad_dir == "up" else "less"
        p1 = stats.mannwhitneyu(b, R, alternative=alt).pvalue
    rows.append(dict(obs=key, A=list(a), Ap=list(ap), B=list(b),
                     R_min=R.min(), R_max=R.max(), R_med=float(np.median(R)),
                     B_min=b.min(), B_max=b.max(), B_med=float(np.median(b)),
                     range_ratio=float((b.max()-b.min())/rng) if rng > 0 else float("nan"),
                     p_two=float(p), effect=float(eff), envelope_ok=bool(envelope_ok),
                     systematic=systematic, p_levene=float(bf.pvalue),
                     p_one_bad_dir=(float(p1) if p1 is not None else None)))
    if systematic or not envelope_ok:
        flagged.append(key)

# Holm across the family
ps = sorted((r["p_two"], r["obs"]) for r in rows)
m = len(ps); holm = {}
run_max = 0.0
for i, (p, k) in enumerate(ps):
    adj = min(1.0, (m - i) * p); run_max = max(run_max, adj); holm[k] = run_max
for r in rows:
    r["p_holm"] = holm[r["obs"]]

hdr = (f"{'observable':34s} {'R (A+Ap)':>34s} {'B':>30s} {'rr':>5s} "
       f"{'p':>8s} {'holm':>6s} {'eff':>6s}  verdict")
print(hdr); print("-" * len(hdr))
for r in rows:
    v = "SYSTEMATIC" if r["systematic"] else ("outside-envelope" if not r["envelope_ok"] else "ok")
    R = "[%.5g, %.5g] m=%.5g" % (r['R_min'], r['R_max'], r['R_med'])
    B = "[%.5g, %.5g] m=%.5g" % (r['B_min'], r['B_max'], r['B_med'])
    print(f"{r['obs']:34s} {R:>34s} {B:>30s} {r['range_ratio']:5.2f} "
          f"{r['p_two']:8.4f} {r['p_holm']:6.3f} {r['effect']:6.2f}  {v}")
print("\n-- per-run values of the flagged observables --")
for r in rows:
    if r["systematic"] or not r["envelope_ok"]:
        print(f"  {r['obs']}")
        print(f"     A : {['%.6g' % x for x in r['A']]}")
        print(f"     Ap: {['%.6g' % x for x in r['Ap']]}")
        print(f"     B : {['%.6g' % x for x in r['B']]}")
        if r['p_one_bad_dir'] is not None:
            print(f"     one-sided p (bad direction) = {r['p_one_bad_dir']:.4f}")

print("\nflagged:", flagged or "none")
Path("/workspace/output/round6/v1/analysis.json").write_text(json.dumps(rows, indent=1))

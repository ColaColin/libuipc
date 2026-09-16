#!/usr/bin/env python3
"""Round-7 s05 observable statistics: R = A u A'' (n=15) vs T = B (n=10).

Safety table (worst anywhere per arm) + per-observable envelope / Mann-Whitney
shift / Holm, per VERDICT_RULE.md.  Reads the run result JSONs.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

RUNS = Path("/workspace/output/round7/s05/runs")
A = [f"a{k:02d}" for k in range(1, 11)]
B = [f"b{k:02d}" for k in range(1, 11)]
AP = [f"ap{k}" for k in range(1, 6)]
R = A + AP


def load_obs(name):
    d = json.load(open(RUNS / f"{name}.json"))
    obs = d["observables"]
    fs = d["frame_stats"]
    obs = dict(obs)
    obs["newton_total"] = sum(s["newton_iterations"] for s in fs)
    obs["pcg_total"] = sum(s["linear_solver_iterations"] for s in fs)
    obs["line_search_total"] = sum(s["line_search_trials"] for s in fs)
    obs["ccd_toi_min"] = min(s["last_ccd_toi"] for s in fs)
    obs["ls_alpha_cut_frames"] = sum(1 for s in fs if s["last_line_search_alpha"] < 1.0)
    obs["ccd_clamped_frames"] = sum(1 for s in fs if s["last_ccd_toi"] < 1.0)
    obs["meanFrameMs"] = d["timing"]["mean_ms"] if "timing" in d else float(np.mean(d["frame_ms"]))
    return obs


OBS = [o for o in load_obs("a01").keys()
        if o.startswith(("verify_", "newton_", "pcg_", "line_search_", "ccd_", "ls_", "mean"))]
# keep scalars only, drop lists (sheet vectors) except via derived mins
runs = {n: load_obs(n) for n in R + B}


def vec_min(obs, key):
    v = obs.get(key)
    return float(np.min(v)) if isinstance(v, list) else None


lines = []
W = lines.append

# ---- safety table ----------------------------------------------------------
W("SAFETY (worst value per arm; R split into A and A'')")
safety = [
    ("verify_all_finite", "all", True),
    ("verify_not_converged_frames", "max", 0),
    ("verify_hit_newton_limit_frames", "max", 0),
    ("verify_hit_ls_limit_frames", "max", 0),
    ("verify_max_speed", "max", 50.0),
    ("verify_area_ratio_min", "min", 0.0),
    ("verify_area_ratio_max", "max", 40.0),
    ("verify_abs_x_max", "max", None),
    ("verify_abs_z_max", "max", None),
    ("verify_y_min", "min", None),
    ("verify_tri_height_min", "min", None),
    ("verify_ke_last_quarter", "max", None),
]
for key, agg, bound in safety:
    row = [key]
    for grp in (A, AP, B):
        vals = [runs[n][key] for n in grp]
        row.append(f"{min(vals) if agg=='min' else max(vals):.4g}")
    W(f"  {key:42s} A={row[1]:>10s} A''={row[2]:>10s} B={row[3]:>10s} bound={bound}")
# knife-edge regime checks: count of runs failing each
W("REGIME (knife-edge) check-false counts per arm")
for chk in ["finite", "no_inversion_or_collapse", "tri_height_no_sustained_collapse",
            "bounded_speed", "contained_x", "contained_z", "above_ground",
            "no_newton_limit", "no_newton_divergence", "all_converged",
            "dahl_state_evolved", "plastic_yielded", "residual_creases"]:
    counts = []
    for grp in (A, AP, B):
        c = sum(0 if runs[n].get("checks", {}).get(chk, True) else 1 for n in grp)
        counts.append(c)
    W(f"  {chk:38s} A={counts[0]:2d}/10  A''={counts[1]:2d}/5  B={counts[2]:2d}/10")
W(f"  verify_ok false: A={sum(0 if runs[n]['verify_ok'] else 1 for n in A)}/10"
  f"  A''={sum(0 if runs[n]['verify_ok'] else 1 for n in AP)}/5"
  f"  B={sum(0 if runs[n]['verify_ok'] else 1 for n in B)}/10")

# ---- statistical table -----------------------------------------------------
W("")
W("STATISTICS (R = A u A'', n=15; T = B, n=10; rule: envelope + MW p<0.05 AND "
  "|median shift| > 0.5*range(R))")
# add derived per-sheet minima as observables
for n in R + B:
    v = runs[n].get("verify_sheet_crease_defl_final_mm")
    if isinstance(v, list):
        runs[n]["crease_min_mm"] = float(np.min(v))
        runs[n]["crease_mean_mm"] = float(np.mean(v))

obs_list = [k for k in OBS if not isinstance(runs["a01"].get(k), (list, dict))]
obs_list += ["crease_min_mm", "crease_mean_mm"]
flagged = []
rows = []
for key in obs_list:
    try:
        rv = np.array([float(runs[n][key]) for n in R])
        tv = np.array([float(runs[n][key]) for n in B])
    except (KeyError, TypeError, ValueError):
        continue
    if np.allclose(rv.std(), 0) and np.allclose(tv.std(), 0):
        rows.append((key, rv[0], tv[0], 0.0, 1.0, 1.0, 0.0, "const"))
        continue
    rmin, rmax = rv.min(), rv.max()
    rr = rmax - rmin
    med_r, med_b = float(np.median(rv)), float(np.median(tv))
    try:
        p = float(mannwhitneyu(tv, rv, alternative="two-sided").pvalue)
    except ValueError:
        p = 1.0
    shift = abs(med_b - med_r)
    env_ok = (tv.min() <= rmax) and (tv.max() >= rmin) and (rmin <= med_b <= rmax)
    systematic = (p < 0.05) and rr > 0 and (shift > 0.5 * rr)
    tag = "SYSTEMATIC" if systematic else ("env-ok" if env_ok else "ENVELOPE-FAIL")
    if systematic or not env_ok:
        flagged.append(key)
    rows.append((key, med_r, med_b, shift / rr if rr > 0 else 0.0, p, 0.0,
                 float(rv.std()), tag))

# Holm over the p-values of the non-constant rows (indexing keyed on row id)
nonconst = [i for i, r in enumerate(rows) if r[7] != "const"]
ps = np.array([rows[i][4] for i in nonconst])
order = np.argsort(ps)
holm = {}
running = 0.0
m = len(ps)
for rank, idx in enumerate(order):
    adj = min(1.0, (m - rank) * ps[idx])
    running = max(running, adj)
    holm[nonconst[idx]] = running
holmed = [holm.get(i, None) for i in range(len(rows))]

W(f"  {'observable':44s} {'med(R)':>12s} {'med(B)':>12s} {'shift/rr':>8s} "
  f"{'p':>9s} {'Holm':>6s} {'sd(R)':>10s}  verdict")
for i, (key, mr, mb, sr, p, _, sdr, tag) in enumerate(rows):
    h = f"{holmed[i]:6.3f}" if holmed[i] is not None else "     -"
    W(f"  {key:44s} {mr:12.4g} {mb:12.4g} {sr:8.3f} {p:9.4g} {h} "
      f"{sdr:10.3g}  {tag}")
W("")
W(f"FLAGGED (rule-2 systematic or envelope fail): {flagged if flagged else 'NONE'}")

# pre-declared one-sided bad directions
W("")
W("PRE-DECLARED BAD DIRECTION (one-sided MW, worse sign)")
bad = {"verify_area_ratio_max": "greater", "verify_tri_height_min": "less",
       "verify_max_speed": "greater", "verify_ke_last_quarter": "greater",
       "verify_strain_yield_frac_final": "greater",
       "verify_stress_yield_frac_final": "greater", "newton_total": "greater"}
for key, alt in bad.items():
    rv = np.array([float(runs[n][key]) for n in R])
    tv = np.array([float(runs[n][key]) for n in B])
    p = float(mannwhitneyu(tv, rv, alternative=alt).pvalue)
    W(f"  {key:44s} med R={np.median(rv):12.4g} B={np.median(tv):12.4g} p(1-sided)={p:.4g}")

report = "\n".join(lines)
print(report)
(Path(RUNS) / "observables.txt").write_text(report + "\n")
json.dump(flagged, open(RUNS / "flagged.json", "w"))

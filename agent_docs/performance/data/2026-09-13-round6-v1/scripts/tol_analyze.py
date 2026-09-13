#!/usr/bin/env python3
"""Truncation-bias test: does the arm-to-arm gap shrink when both arms converge harder?

A gap caused by the Newton stopping criterion terminating at a slightly different
point must shrink as velocity_tol -> 0.  A gap that is chaotic divergence must not.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, "/workspace/output/round6/v1")
from parse_run import run_record

base = json.loads(Path("/workspace/output/round6/v1/parsed.json").read_text())
tight = {}
for p in sorted(Path("/workspace/output/round6/v1/toltest").glob("[AB]_*.log")):
    tight[p.stem] = run_record(str(p))
Path("/workspace/output/round6/v1/parsed_tol.json").write_text(json.dumps(tight))

def get(src, arm, key):
    if key == "newton_per_frame":
        return np.array([src[t]["summary"]["verify_newton_mean"] for t in sorted(src)
                         if t.split("_")[0] == arm])
    return np.array([float(src[t]["summary"][key]) for t in sorted(src)
                     if t.split("_")[0] == arm])

KEYS = ["pairs_converged_mean", "pairs_converged_last_quarter", "pairs_last_quarter",
        "verify_newton_total", "verify_pcg_total", "verify_ccd_toi_min",
        "verify_max_speed", "verify_area_ratio_max", "verify_tri_height_min"]
print("probe cross-checks in the tight-tolerance runs: "
      f"{sum(t['summary']['probe_checks'] for t in tight.values())} checks, "
      f"{sum(t['summary']['probe_violations'] for t in tight.values())} violations")
print(f"\n{'observable':32s} | {'vel_tol=0.05 (default, n=10/10)':>40s} | {'vel_tol=0.005 (n=10/10)':>40s}")
print(f"{'':32s} | {'A median':>12s} {'B median':>12s} {'B/A':>7s} {'p':>6s} | "
      f"{'A median':>12s} {'B median':>12s} {'B/A':>7s} {'p':>6s}")
for k in KEYS:
    a0 = np.concatenate([get(base, "A", k), get(base, "Ap", k)])
    b0 = get(base, "B", k)
    a1, b1 = get(tight, "A", k), get(tight, "B", k)
    r0 = np.median(b0) / np.median(a0); r1 = np.median(b1) / np.median(a1)
    p0 = stats.mannwhitneyu(b0, a0).pvalue; p1 = stats.mannwhitneyu(b1, a1).pvalue
    print(f"{k:32s} | {np.median(a0):12.5g} {np.median(b0):12.5g} {r0:7.3f} {p0:6.3f} | "
          f"{np.median(a1):12.5g} {np.median(b1):12.5g} {r1:7.3f} {p1:6.3f}")

print("\nNewton iterations per frame (how much harder the tight arm works):")
for lbl, src in (("vel_tol=0.05", base), ("vel_tol=0.005", tight)):
    for arm in ("A", "B"):
        v = get(src, arm, "verify_newton_mean")
        if len(v):
            print(f"  {lbl:14s} {arm:3s} {np.median(v):.2f} newton/frame  (n={len(v)})")
print("\nsafety in the tight-tolerance runs:")
for t in sorted(tight):
    s = tight[t]["summary"]
    print(f"  {t:6s} ok={s['verify_ok']} notconv={s['verify_not_converged_frames']} "
          f"hitN={s['verify_hit_newton_limit_frames']} hitLS={s['verify_hit_ls_limit_frames']} "
          f"r_max={s['verify_r_max']:.5f} tri_h={s['verify_tri_height_min']:.5f} "
          f"cc={s['verify_cc_min_dist']:.5f} bore={s['verify_bore_gap_min']:.5f}")

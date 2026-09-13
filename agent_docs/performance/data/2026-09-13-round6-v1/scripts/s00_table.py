#!/usr/bin/env python3
"""Re-establish s00's soundness table on the *shipped default* (Gauss-Newton),
at s00's own statistics, beside the exact path measured the same way."""
import json
from pathlib import Path
import numpy as np

P = json.loads(Path("/workspace/output/round6/v1/parsed.json").read_text())
arms = {}
for t in sorted(P):
    arms.setdefault(t.split("_")[0], []).append(P[t]["summary"])

KEYS = ["verify_all_finite", "verify_r_max", "verify_r_limit", "verify_contained_radial",
        "verify_abs_z_max", "verify_z_limit", "verify_contained_axial",
        "verify_lifter_depth_max", "verify_area_ratio_min", "verify_area_ratio_max",
        "verify_tri_height_min", "verify_cc_min_dist", "verify_bore_gap_min",
        "verify_max_speed", "verify_ke_mean", "verify_drum_track_err_deg_max",
        "verify_drum_track_err_deg_final", "verify_drum_turn_deg",
        "verify_mean_disp_mm", "verify_mean_disp_mm_last_quarter",
        "verify_newton_mean", "verify_newton_min", "verify_newton_max",
        "verify_newton_first_quarter", "verify_newton_last_quarter",
        "verify_pcg_mean", "verify_pcg_first_quarter", "verify_pcg_last_quarter",
        "verify_not_converged_frames", "verify_hit_newton_limit_frames",
        "verify_hit_ls_limit_frames", "verify_ccd_toi_min",
        "verify_gpu_mem_proc_max_mib", "verify_ok"]

def cell(vals):
    if isinstance(vals[0], bool):
        s = set(vals); return str(s.pop()) if len(s) == 1 else "MIXED " + str(sorted(s))
    v = np.array(vals, dtype=float)
    return f"{np.median(v):.6g} [{v.min():.5g}, {v.max():.5g}]"

print(f"{'check':38s} {'A exact (n=%d)' % len(arms['A']):>34s} "
      f"{'Ap perturbed-exact (n=%d)' % len(arms['Ap']):>34s} "
      f"{'B Gauss-Newton = SHIPPED (n=%d)' % len(arms['B']):>34s}")
for k in KEYS:
    print(f"{k:38s} " + " ".join(f"{cell([s[k] for s in arms[a]]):>34s}" for a in ("A", "Ap", "B")))
print()
for a in ("A", "Ap", "B"):
    ct = [s["verify_centroid_turn_deg"] for s in arms[a]]
    med = np.median(np.array(ct), axis=0)
    print(f"  {a:3s} garment centroid angular travel (deg, median of runs): "
          + " / ".join(f"{x:.0f}" for x in med) + "   against 720 of drum")

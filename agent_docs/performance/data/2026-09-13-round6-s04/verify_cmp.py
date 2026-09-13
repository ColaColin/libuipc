import json,glob,statistics,sys
KEYS=["verify_ok","verify_all_finite","verify_contained_radial","verify_contained_axial",
      "verify_r_max","verify_z_abs_max","verify_lifter_depth_max","verify_tri_height_min",
      "verify_area_ratio_min","verify_area_ratio_max","verify_max_speed",
      "verify_cc_min_dist","verify_bore_gap_min","verify_drum_track_err_deg_max",
      "verify_ccd_toi_min","verify_ccd_toi_clamped_frames","verify_ls_alpha_min",
      "verify_hit_newton_limit_frames","verify_hit_ls_limit_frames",
      "verify_not_converged_frames","verify_newton_total","verify_pcg_total",
      "verify_mean_disp_mm_last_quarter"]
arms={}
for v in ("0","1"):
    rows=[]
    for f in sorted(glob.glob(f'/workspace/output/round6/s04/verify_eo{v}_r*.txt')):
        for line in open(f):
            if line.startswith("VERIFY_RESULT "):
                rows.append(json.loads(line[len("VERIFY_RESULT "):])); break
    arms[v]=rows
    print(f"arm UIPC_CCD_EARLY_OUT={v}: {len(rows)} runs")
print(f"\n{'key':40} {'eo0 (old)':>34} {'eo1 (new)':>34}")
for k in KEYS:
    def fmt(rows):
        vals=[r.get(k) for r in rows if k in r]
        if not vals: return "n/a"
        if isinstance(vals[0],bool): return " ".join(str(int(v)) for v in vals)
        if all(isinstance(v,int) for v in vals): return f"{statistics.mean(vals):.1f} [{min(vals)},{max(vals)}]"
        return f"{statistics.mean(vals):.5g} [{min(vals):.5g},{max(vals):.5g}]"
    print(f"{k:40} {fmt(arms['0']):>34} {fmt(arms['1']):>34}")

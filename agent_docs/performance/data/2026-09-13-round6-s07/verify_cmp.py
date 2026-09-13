import json,glob,statistics,sys
KEYS=["verify_ok","verify_all_finite","verify_contained_radial","verify_contained_axial",
      "verify_r_max","verify_z_abs_max","verify_lifter_depth_max","verify_tri_height_min",
      "verify_area_ratio_min","verify_area_ratio_max","verify_max_speed",
      "verify_cc_min_dist","verify_bore_gap_min","verify_drum_track_err_deg_max",
      "verify_ccd_toi_min","verify_ccd_toi_clamped_frames","verify_ls_alpha_min",
      "verify_hit_newton_limit_frames","verify_hit_ls_limit_frames",
      "verify_not_converged_frames","verify_newton_total","verify_pcg_total",
      "verify_mean_disp_mm_last_quarter"]
ARMS=["0","1","5"]
arms={}
for v in ARMS:
    rows=[]
    for f in sorted(glob.glob(f'/workspace/output/round6/s07/verify/verify_p{v}_r*.txt')):
        got=False
        for line in open(f):
            if line.startswith("VERIFY_RESULT "):
                rows.append(json.loads(line[len("VERIFY_RESULT "):])); got=True; break
        if not got: print(f"  !! no VERIFY_RESULT in {f}")
    arms[v]=rows
    print(f"arm UIPC_CONTACT_RANK1={v}: {len(rows)} completed runs")
def fmt(rows,k):
    vals=[r.get(k) for r in rows if k in r]
    if not vals: return "n/a"
    if isinstance(vals[0],bool): return f"{sum(vals)}/{len(vals)} true"
    if all(isinstance(v,int) for v in vals): return f"{statistics.mean(vals):.1f} [{min(vals)},{max(vals)}]"
    return f"{statistics.mean(vals):.5g} [{min(vals):.5g},{max(vals):.5g}]"
print(f"\n{'key':38} {'p0 exact':>28} {'p1 PE+PP':>28} {'p5 PP only':>28}")
for k in KEYS:
    print(f"{k:38} {fmt(arms['0'],k):>28} {fmt(arms['1'],k):>28} {fmt(arms['5'],k):>28}")

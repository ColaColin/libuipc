import json, glob, statistics, sys
from scipy import stats
D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/output/round6/s14/tverify'
M = sys.argv[2] if len(sys.argv) > 2 else '1'
KEYS = ["verify_ok","verify_all_finite","verify_contained_radial","verify_contained_axial",
        "verify_cc_min_dist","verify_bore_gap_min","verify_tri_height_min","verify_lifter_depth_max",
        "verify_area_ratio_max","verify_area_ratio_min","verify_max_speed","verify_r_max",
        "verify_mean_disp_mm_last_quarter","verify_newton_total","verify_pcg_total",
        "verify_line_search_total","verify_not_converged_frames","verify_hit_newton_limit_frames",
        "verify_hit_ls_limit_frames","verify_ccd_toi_min","verify_ls_alpha_min"]
arms = {}
for v in ("0", M):
    rows = []
    for f in sorted(glob.glob(f'{D}/verify_m{v}_r*.txt')):
        for line in open(f):
            if line.startswith("VERIFY_RESULT "):
                rows.append(json.loads(line[len("VERIFY_RESULT "):])); break
    arms[v] = rows
print(f"tumbler --verify, 180 frames, off n={len(arms['0'])} / mode {M} n={len(arms[M])}, one build")
for k in KEYS:
    a = [r.get(k) for r in arms['0'] if k in r]; b = [r.get(k) for r in arms[M] if k in r]
    if not a or not b: continue
    if isinstance(a[0], bool):
        print(f"  {k:34} off {sum(a)}/{len(a)} true   m{M} {sum(b)}/{len(b)} true"); continue
    fa = f"{statistics.mean(a):.6g} [{min(a):.6g}, {max(a):.6g}]"; fb = f"{statistics.mean(b):.6g} [{min(b):.6g}, {max(b):.6g}]"
    extra = ""
    if len(set(a)) > 1 or len(set(b)) > 1:
        try:
            extra = f"  Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.3f}  MW p={stats.mannwhitneyu(a, b).pvalue:.3f}"
        except Exception: pass
    print(f"  {k:34} off {fa:>36}   m{M} {fb:>36}{extra}")

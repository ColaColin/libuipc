import json, glob, re, statistics, sys
KEYS=["verify_ok","verify_all_finite","verify_contained_radial","verify_contained_axial",
      "verify_cc_min_dist","verify_bore_gap_min","verify_tri_height_min","verify_lifter_depth_max",
      "verify_area_ratio_max","verify_area_ratio_min","verify_max_speed","verify_r_max",
      "verify_mean_disp_mm_last_quarter","verify_newton_total","verify_pcg_total",
      "verify_line_search_total","verify_not_converged_frames","verify_hit_newton_limit_frames",
      "verify_hit_ls_limit_frames","verify_e_tot_max","verify_ccd_toi_min"]
arms={}
for arm in ['a2','a1','a0']:
    rows=[]
    for f in sorted(glob.glob(f'verify/verify_{arm}_r*.txt')):
        t=open(f, errors='replace').read()
        m=re.search(r'VERIFY_RESULT\s*(\{.*?\})\s*$', t, re.M|re.S)
        if not m:
            i=t.find('VERIFY_RESULT')
            if i<0: print("no VERIFY_RESULT in",f); continue
            j=t.find('{',i); k=t.find('\n',j); m=None
            rows.append(json.loads(t[j:k])); continue
        rows.append(json.loads(m.group(1)))
    arms[arm]=rows
print('tumbler --verify, 180 frames, n=%s per arm; a2 = shipped split, a1 = two launches one stream, a0 = fused'%[len(arms[a]) for a in ['a2','a1','a0']])
try:
    from scipy import stats as st
except Exception:
    st=None
for k in KEYS:
    cols={a:[r.get(k) for r in arms[a]] for a in ['a2','a1','a0']}
    if any(x is None for v in cols.values() for x in v): continue
    a=cols['a2']; b=cols['a1']; c=cols['a0']
    if isinstance(a[0],bool):
        print(f"  {k:34s} a2 {sum(a)}/{len(a)} true   a1 {sum(b)}/{len(b)} true   a0 {sum(c)}/{len(c)} true")
        continue
    def pv(x,y):
        if st and len(set(x))>1 and len(set(y))>1:
            return f"p={st.ttest_ind(x,y,equal_var=False).pvalue:.3f}"
        return ""
    print(f"  {k:34s} a2 {statistics.mean(a):12.6g} [{min(a):.6g}, {max(a):.6g}]   "
          f"a1 {statistics.mean(b):12.6g} [{min(b):.6g}, {max(b):.6g}] {pv(a,b)}   "
          f"a0 {statistics.mean(c):12.6g} [{min(c):.6g}, {max(c):.6g}] {pv(a,c)}")
print("\n  sorted verify_area_ratio_max:")
for arm in ['a2','a1','a0']:
    print(f"    {arm}: {' '.join(f'{x:.3f}' for x in sorted(r['verify_area_ratio_max'] for r in arms[arm]))}")

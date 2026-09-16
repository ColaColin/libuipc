import csv, sys, glob
pats = ['schwarz_local_solve_fused_R_kernel<(bool)1','schwarz_local_solve_fused_R_kernel<(bool)0',
        'schwarz_local_solve_rowdot2_kernel','build_multi_level_R','collect_final_Z',
        'Spmv_rbk_sym','invert_cluster_matrices_sweep']
for f in sys.argv[1:]:
    rows = list(csv.reader(open(f)))
    hdr = rows[0]
    ti, ii, ai, med = hdr.index('Total Time (ns)'), hdr.index('Instances'), hdr.index('Avg (ns)'), hdr.index('Med (ns)')
    ni = hdr.index('Name')
    tot = 0.0; best = {}
    for r in rows[1:]:
        if len(r) <= max(ti, ii, ni): continue
        name = r[ni]
        try: t = float(r[ti]); n = int(r[ii]); avg = float(r[ai]); md = float(r[med])
        except Exception: continue
        tot += t
        for p in pats:
            if p in name:
                a, b, c, d = best.get(p, (0, 0, 0.0, 0.0))
                best[p] = (a + t, b + n, avg, md)
    print(f"=== {f.split('_')[0]} ({f}) ===")
    for k, (t, n, avg, md) in sorted(best.items()):
        print(f"  {k[:52]:54s} {n:7d} x avg {t/n/1000.0:7.2f} us (csv-avg {avg/1000.0:7.2f}, med {md/1000.0:7.2f})  tot {t/1e6:9.1f} ms")
    print(f"  TOTAL GPU kernel time {tot/1e6:.1f} ms")

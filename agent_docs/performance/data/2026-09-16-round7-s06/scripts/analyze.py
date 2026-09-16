#!/usr/bin/env python3
"""s06 analysis: regenerate every number quoted in the round-record section.

Inputs (produced by sweep_blockdim.sh / repeat_def128.sh / the geom traces):
  {sweep_bd64,sweep_bd128,sweep_def,sweep_bd512,sweep_bd1024}_cuda_gpu_kern_sum.csv
  {rep_d1,rep_d2,rep_d3,rep_1a,rep_1b,rep_1c}_cuda_gpu_kern_sum.csv
  {geom_def,geom_bd128}_cuda_gpu_trace.csv
"""
import csv, statistics as st, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

KS = [('fused_update_xr_kernel','xr'),('fused_update_p_beta_kernel','p_beta'),
      ('fused_update_p_kernel','p_old'),('fused_update_p_scalar_kernel','p_scalar'),
      ('fused_dot_kernel','dot'),('fused_pcg_scalar_kernel','scalar'),
      ('Spmv_rbk_sym_spmv_dot_chunked','spmv'),
      ('rowdot','rowdot'),('collect_final_Z','collectZ')]

def grab(pfx):
    with open(os.path.join(HERE, f'{pfx}_cuda_gpu_kern_sum.csv')) as f:
        rows = list(csv.reader(f))
    out = {}
    for r in rows[1:]:
        for k, short in KS:
            if k in r[8] and short not in out:
                out[short] = (float(r[3])/1000, float(r[4])/1000, int(r[2]))
    return out

def sweep_table(out):
    arms = [('bd64','sweep_bd64'),('bd128 (spread pick)','sweep_bd128'),('def=256','sweep_def'),
            ('bd512','sweep_bd512'),('bd1024 (pre-R4s11)','sweep_bd1024')]
    data = {t: grab(p) for t, p in arms}
    out.write("== block-dim sweep, full 130-frame runs, per-launch figures (avg us / med us / instances)\n")
    out.write(f"{'arm':<20}{'xr':>22}{'p_beta':>22}{'dot(ctl)':>22}{'scalar(ctl)':>22}{'spmv(ctl)':>22}\n")
    for t, _ in arms:
        d = data[t]
        out.write(f"{t:<20}")
        for k in ['xr','p_beta','dot','scalar','spmv']:
            v = d.get(k)
            out.write(f"{v[0]:>8.2f}/{v[1]:<6.2f}/{v[2]:<7}" if v else f"{'--':>22}")
        out.write("\n")
    out.write("\n== repeats, def vs bd128, n=4 full runs per arm (sweep arm + 3 reps), MEDIANS\n")
    defs = [grab(p) for p in ['sweep_def','rep_d1','rep_d2','rep_d3']]
    b128 = [grab(p) for p in ['sweep_bd128','rep_1a','rep_1b','rep_1c']]
    out.write(f"{'kernel':<10} {'def(256)':>24} {'bd128(spread)':>24} {'delta':>9} {'ctl-norm':>9}\n")
    for k in ['xr','p_beta','dot','scalar','spmv','rowdot','collectZ']:
        d = [x[k][1] for x in defs if k in x]; b = [x[k][1] for x in b128 if k in x]
        if not d or not b: continue
        dn = [x[k][1]/x['dot'][1] for x in defs]; bn = [x[k][1]/x['dot'][1] for x in b128]
        out.write(f"{k:<10} {st.mean(d):7.2f} [{min(d):.2f},{max(d):.2f}]"
                  f" {st.mean(b):7.2f} [{min(b):.2f},{max(b):.2f}]"
                  f" {(st.mean(b)/st.mean(d)-1)*100:+8.2f}% {(st.mean(bn)/st.mean(dn)-1)*100:+8.2f}%\n")
    out.write(f"\nPCG instances/run: def {[x['spmv'][2] for x in defs]}  bd128 {[x['spmv'][2] for x in b128]}\n")
    out.write("prize: xr 2.90% + p_beta 1.30% of scene GPU kernel time (s05 ranking) x per-launch delta\n")

def trace_table(out):
    for tag in ['geom_def','geom_bd128']:
        with open(os.path.join(HERE, f'{tag}_cuda_gpu_trace.csv')) as f:
            rows = list(csv.reader(f))
        hdr = rows[0]; gi = {h: i for i, h in enumerate(hdr)}
        ev = []
        for r in rows[1:]:
            if len(r) < len(hdr): continue
            ev.append((int(r[gi['Start (ns)']]), int(r[gi['Duration (ns)']]), r[gi['Strm']],
                       r[gi['Name']].split('(')[0].replace('void uipc::backend::cuda::<unnamed>::','')
                       .replace('uipc::backend::cuda::<unnamed>::','')[:44]))
        ev.sort()
        out.write(f"\n== trace {tag} (3 frames)\n")
        ks = ['fused_update_xr_kernel','fused_update_p_beta_kernel','fused_dot_kernel',
              'fused_pcg_scalar_kernel','Spmv_rbk_sym_spmv_dot_chunked']
        for k in ks:
            a = [(int.__class__ and e[0], e[1]) for e in ev if k in e[3]]
            if not a: continue
            durs = sorted(d for _, d in a)
            n = len(durs)
            grids = sorted(set(e for e in [None] )) # placeholder
            out.write(f"  {k:<38} n={n} med={st.median(durs)/1000:.2f}us "
                      f"p10={durs[n//10]/1000:.2f} p90={durs[9*n//10]/1000:.2f} "
                      f"<4us:{100*len([d for d in durs if d<4000])/n:.2f}%\n")
        xrs = [e for e in ev if e[3] == 'fused_update_xr_kernel']
        out.write(f"  xr stream ids: {sorted(set(e[2] for e in xrs))}\n")
        fam = ['fused_','Spmv_rbk','MASPreconditionerEngine']
        gaps = []
        fev = [e for e in ev if any(k in e[3] for k in fam)]
        for i in range(1, len(fev)):
            g = fev[i][0] - (fev[i-1][0] + fev[i-1][1])
            if 0 <= g < 100000: gaps.append(g)
        out.write(f"  PCG-family node-to-node gaps: med={st.median(gaps):.0f}ns "
                  f"p90={sorted(gaps)[9*len(gaps)//10]:.0f}ns\n")

if __name__ == '__main__':
    with open(os.path.join(HERE, 'sweep_and_trace_summary.txt'), 'w') as out:
        sweep_table(out)
        trace_table(out)
    print(open(os.path.join(HERE, 'sweep_and_trace_summary.txt')).read())

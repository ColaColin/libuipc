import csv,glob,re,statistics,sys
KEYS = {
 'filter_toi_k4' : 'CCD narrow EE (filter_toi_k4)',
 'filter_toi_k3' : 'CCD narrow PT (filter_toi_k3)',
 'filter_active_k4': 'DCD narrow EE (filter_active_k4)',
 'filter_active_k3': 'DCD narrow PT (filter_active_k3)',
 'stacklessSelf' : 'BVH traversal self',
 'stacklessOther': 'BVH traversal other',
 'pairFilter_kernel<(bool)1': 'BVH pairFilter self',
 'pairFilter_kernel<(bool)0': 'BVH pairFilter other',
}
def fam(n):
    if 'InfoStacklessBVH' in n: return 'BVH/CCD'
    if 'do_assemble_kernel' in n and 'abd' not in n: return 'contact assemble'
    if 'DiscreteShellBending' in n: return 'bending'
    if 'StrainLimiting' in n: return 'membrane'
    if 'Spmv' in n: return 'spmv'
    if 'MASPrecond' in n or 'diag_preconditioner' in n or 'multi_level' in n: return 'precond'
    if 'do_compute_energy' in n: return 'energy'
    if 'Sort' in n or 'Scan' in n or 'Reduce' in n or 'cub' in n or 'thrust' in n: return 'cub/thrust'
    return 'other'
def load(f):
    rows=list(csv.DictReader(open(f)))
    per={}; fm={}
    for r in rows:
        n=r['Name']; t=float(r['Total Time (ns)'])/1e6
        fm[fam(n)]=fm.get(fam(n),0)+t
        for k in KEYS:
            if k in n:
                a=per.setdefault(k,[0.0,0,0.0])
                a[0]+=t; a[1]+=int(r['Instances'])
    for k,a in per.items(): a[2]=a[0]*1000.0/a[1]
    fm['TOTAL']=sum(fm.values())
    return per,fm
pre=sys.argv[1]; arms=sys.argv[2:]
res={a:[load(f) for f in sorted(glob.glob(f'{pre}_{a}_r*_cuda_gpu_kern_sum.csv'))] for a in arms}
for a in arms: print(f"{a}: {len(res[a])} full runs")
base=arms[0]
def m(rs,k,i): 
    v=[r[0][k][i] for r in rs if k in r[0]]
    return statistics.mean(v) if v else float('nan'), (min(v),max(v)) if v else (0,0)
print(f"\n{'kernel':34} {'us/launch old':>14} {'us/launch new':>14} {'delta':>9}   launches old/new")
for k,label in KEYS.items():
    A,ra=m(res[base],k,2); 
    for a in arms[1:]:
        B,rb=m(res[a],k,2)
        LA,_=m(res[base],k,1); LB,_=m(res[a],k,1)
        print(f"{label:34} {A:14.1f} {B:14.1f} {100*(B-A)/A:+8.2f} %   {LA:.0f}/{LB:.0f}   [{ra[0]:.1f},{ra[1]:.1f}] vs [{rb[0]:.1f},{rb[1]:.1f}]")
print(f"\n{'family (total ms)':34} {'old':>14} {'new':>14} {'delta':>9}")
fams=sorted({k for r in res[base] for k in r[1]}, key=lambda k:-statistics.mean([r[1].get(k,0) for r in res[base]]))
for k in fams:
    A=statistics.mean([r[1].get(k,0) for r in res[base]])
    for a in arms[1:]:
        B=statistics.mean([r[1].get(k,0) for r in res[a]])
        print(f"{k:34} {A:14.1f} {B:14.1f} {100*(B-A)/A:+8.2f} %")

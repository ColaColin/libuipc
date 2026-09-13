import csv,re,glob,statistics,sys
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
    d={}; fm={}
    for r in rows:
        n=r['Name']; t=float(r['Total Time (ns)'])/1e6
        fm[fam(n)]=fm.get(fam(n),0)+t
        if 'do_assemble_kernel' in n and 'abd' not in n:
            m=re.search(r'do_assemble_kernel<([^>]*)>',n)
            if not m: continue
            a=m.group(1)
            k='part1' if a.startswith('(bool)0, (int)1') else ('part2' if a.startswith('(bool)0, (int)2') else ('friction' if a=='(bool)0' else None))
            if k: d[k]=(float(r['Avg (ns)'])/1e3, t, int(r['Instances']))
    fm['TOTAL']=sum(v for k,v in fm.items())
    return d,fm
scene=sys.argv[1]; arms=sys.argv[2:]
res={}
for a in arms:
    fs=sorted(glob.glob(f'{scene}_{a}_r*_cuda_gpu_kern_sum.csv'))
    res[a]=[load(f) for f in fs]
    print(f"== {scene} {a}  ({len(fs)} full runs)")
    for k in ['part1','part2','friction']:
        us=[d[0][k][0] for d in res[a] if k in d[0]]
        nl=[d[0][k][2] for d in res[a] if k in d[0]]
        if not us: continue
        print(f"   {k:9}: {statistics.mean(us):9.1f} us/launch [{min(us):.1f},{max(us):.1f}]  launches {statistics.mean(nl):.0f}")
    print(f"   {'scene':9}: {statistics.mean([d[1]['TOTAL'] for d in res[a]]):9.1f} ms GPU kernel time")
base=res[arms[0]]
for a in arms[1:]:
    print(f"\n-- {a} vs {arms[0]}  (per launch, and per family total)")
    for k in ['part1','part2','friction']:
        A=statistics.mean([d[0][k][0] for d in base]); B=statistics.mean([d[0][k][0] for d in res[a]])
        print(f"   {k:18}: {A:9.1f} -> {B:9.1f}  {100*(B-A)/A:+7.2f} %")
    fams=sorted({k for d in base for k in d[1]}, key=lambda k:-statistics.mean([d[1].get(k,0) for d in base]))
    for k in fams:
        A=statistics.mean([d[1].get(k,0) for d in base]); B=statistics.mean([d[1].get(k,0) for d in res[a]])
        print(f"   {k:18}: {A:9.1f} -> {B:9.1f}  {100*(B-A)/A:+7.2f} %   (family total ms)")

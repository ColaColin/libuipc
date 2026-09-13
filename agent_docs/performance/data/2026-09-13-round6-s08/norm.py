import csv, glob, re, statistics, sys
SC=sys.argv[1]
NORM=['do_compute_energy_k2','do_compute_energy_k1','filter_toi','pairFilter','stacklessSelf']
def per_launch(f):
    out={}
    for r in csv.DictReader(open(f)):
        n=r['Name']; t=int(r['Total Time (ns)']); i=int(r['Instances'])
        m=re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([12]),', n)
        k='part'+m.group(1) if m else None
        if k is None:
            for nm in NORM:
                if nm in n: k=nm; break
        if k:
            a,b=out.get(k,(0,0)); out[k]=(a+t,b+i)
    return {k:(t/i/1000.0) for k,(t,i) in out.items()}
arms={}
for arm in ['b0','b1','b2']:
    arms[arm]=[per_launch(f) for f in sorted(glob.glob(f'nsys/{SC}_{arm}_r*_cuda_gpu_kern_sum.csv'))]
keys=['part1','part2']+[k for k in arms['b0'][0] if k in NORM]
print(f"== {SC}: per-launch us, and part1 normalised by untouched kernels of other families")
for arm in ['b0','b1','b2']:
    row=[f"{arm}"]
    for k in keys:
        row.append(f"{k}={statistics.mean([d.get(k,float('nan')) for d in arms[arm]]):8.1f}")
    print("  ".join(row))
b0=arms['b0']
for arm in ['b1','b2']:
    print(f"-- {arm} vs b0")
    for k in keys:
        m=statistics.mean([d[k] for d in arms[arm]]); b=statistics.mean([d[k] for d in b0])
        print(f"     {k:24s} {100*(m/b-1):+7.2f} %")
    for nk in keys[2:]:
        r1=[d['part1']/d[nk] for d in arms[arm]]; r0=[d['part1']/d[nk] for d in b0]
        print(f"     part1 / {nk:16s} {100*(statistics.mean(r1)/statistics.mean(r0)-1):+7.2f} %   "
              f"arm {[round(x,4) for x in r1]}  base {[round(x,4) for x in r0]}")

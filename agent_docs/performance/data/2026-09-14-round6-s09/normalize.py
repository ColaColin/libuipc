"""s09: per-launch us of the contact-assembly parts normalised by untouched kernels
measured in the same run (the round record's tumbler lesson)."""
import csv, glob, re, statistics, sys
SC=sys.argv[1]
NORM=['do_compute_energy_k2','do_compute_energy_k1','filter_toi','InfoStacklessBVH_pairFilter','stacklessSelf','abd_diag_preconditioner']
def per_launch(f):
    out={}
    for r in csv.DictReader(open(f)):
        n=r['Name']; t=int(r['Total Time (ns)']); i=int(r['Instances'])
        m=re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([012]),', n)
        k='p'+m.group(1) if m else None
        if k is None:
            for nm in NORM:
                if nm in n: k=nm; break
        if k:
            a,b=out.get(k,(0,0)); out[k]=(a+t,b+i)
    return {k:(t/i/1000.0) for k,(t,i) in out.items()}
arms={a:[per_launch(f) for f in sorted(glob.glob(f'nsys/{SC}_{a}_r*_cuda_gpu_kern_sum.csv'))] for a in ['a2','a1','a0']}
allk=set()
for v in arms.values():
    for d in v: allk|=set(d)
nk=[k for k in NORM if k in allk]
print(f"== {SC}: per-launch us; untouched kernels of other families measured in the same runs")
hdr=['p0','p1','p2']+nk
print("      "+"  ".join(f"{k[:22]:>22s}" for k in hdr))
for a in ['a2','a1','a0']:
    print(f"  {a}  "+"  ".join(f"{statistics.mean([d[k] for d in arms[a] if k in d]) if any(k in d for d in arms[a]) else float('nan'):22.1f}" for k in hdr))
print("  -- vs a2, and the same delta normalised by each untouched kernel")
for a in ['a1','a0']:
    for k in ['p0','p1','p2']:
        if not any(k in d for d in arms[a]) or not any(k in d for d in arms['a2']): continue
        m=statistics.mean([d[k] for d in arms[a]]); b=statistics.mean([d[k] for d in arms['a2']])
        print(f"    {a} {k}: {100*(m/b-1):+7.2f} %")
    for k in nk:
        m=statistics.mean([d[k] for d in arms[a] if k in d]); b=statistics.mean([d[k] for d in arms['a2'] if k in d])
        print(f"       control {k[:28]:28s} {100*(m/b-1):+7.2f} %")

#!/usr/bin/env python3
# s19 scope analysis: plastic pair vs single per-launch, in-run controls
import csv, collections, sys, statistics
def stats(pfx):
    rows = list(csv.reader(open(f'{pfx}_cuda_gpu_trace.csv')))
    hdr = rows[0]; gi = {c:i for i,c in enumerate(hdr)}
    agg = collections.defaultdict(list)
    for r in rows[1:]:
        nm = r[gi['Name']]
        if 'StrainPlasticDiscreteShellBending_do_compute_gradient_hessian_pair' in nm: tag='strain_pair'
        elif 'StrainPlasticDiscreteShellBending_do_compute_gradient_hessian' in nm: tag='strain_single'
        elif 'StressPlasticDiscreteShellBending_do_compute_gradient_hessian_pair' in nm: tag='stress_pair'
        elif 'StressPlasticDiscreteShellBending_do_compute_gradient_hessian' in nm: tag='stress_single'
        elif 'DahlFrictionDiscreteShellBending_do_compute_gradient_hessian' in nm: tag='dahl'
        elif 'NeoHookeanShell2D' in nm and 'gradient_hessian' in nm: tag='nhs2d'
        elif 'Spmv_rbk_sym' in nm: tag='spmv'
        else: continue
        agg[tag].append(int(r[gi['Duration (ns)']]))
    return {k: statistics.mean(v) for k,v in agg.items()}, {k: len(v) for k,v in agg.items()}
rounds = sys.argv[1:] if len(sys.argv)>1 else ['1','2']
arms = {'new': [], 'old': []}
ctl = {'new': [], 'old': []}
for r in rounds:
    for arm in ('new','old'):
        m, n = stats(f'scope_r{r}_{arm}')
        arms[arm].append(m); ctl[arm].append(n)
for arm in ('new','old'):
    print(f"== {arm} ==")
    keys = set().union(*[set(m) for m in arms[arm]])
    for k in sorted(keys):
        vals = [m[k] for m in arms[arm] if k in m]
        print(f"  {k:14s} mean-launch={statistics.mean(vals)/1000:8.1f}us  (launches {[n[k] for n in ctl[arm] if k in n]})")
old, new = arms['old'], arms['new']
for k in ('strain_single','stress_single'):
    pass
so = statistics.mean([m['strain_single'] for m in old]); sn = statistics.mean([m['strain_pair'] for m in new])
to = statistics.mean([m['stress_single'] for m in old]); tn = statistics.mean([m['stress_pair'] for m in new])
print(f"\nstrain: {so/1000:.1f} -> {sn/1000:.1f} us/launch ({(sn-so)/so*100:+.2f} %)")
print(f"stress: {to/1000:.1f} -> {tn/1000:.1f} us/launch ({(tn-to)/to*100:+.2f} %)")
print(f"controls: dahl {statistics.mean([m['dahl'] for m in old])/1000:.1f} -> {statistics.mean([m['dahl'] for m in new])/1000:.1f}; nhs2d {statistics.mean([m['nhs2d'] for m in old])/1000:.1f} -> {statistics.mean([m['nhs2d'] for m in new])/1000:.1f}; spmv {statistics.mean([m['spmv'] for m in old])/1000:.2f} -> {statistics.mean([m['spmv'] for m in new])/1000:.2f}")

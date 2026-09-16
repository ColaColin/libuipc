import csv, re
runs = {}
for tag in ['new_r1','old_r1','old_r2','new_r2']:
    runs[tag] = {}
    with open(f'nsys_{tag}_cuda_gpu_kern_sum.csv') as f:
        for r in csv.DictReader(f):
            runs[tag][r['Name']] = dict(avg=float(r['Avg (ns)'])/1000.0,
                                        cnt=int(r['Instances']),
                                        tot=float(r['Total Time (ns)'])/1e6,
                                        pct=float(r['Time (%)']))
def show(label, pat, per=9506, unit='hinge'):
    print(f'== {label}')
    for tag in ['new_r1','old_r1','old_r2','new_r2']:
        for nm, v in runs[tag].items():
            if re.search(pat, nm):
                print(f"  {tag}: {v['avg']:8.1f} us x {v['cnt']:6d} ({v['avg']*1000.0/per:6.1f} ns/{unit}) tot {v['tot']:8.1f} ms  {v['pct']:4.2f}%")
show('strain GH (both template arms land here?)', r'StrainPlasticDiscreteShellBending_do_compute_gradient_hessian')
show('stress GH', r'StressPlasticDiscreteShellBending_do_compute_gradient_hessian')
show('dahl GH (control, GN default)', r'DahlFrictionDiscreteShellBending_do_compute_gradient_hessian')
show('NHS2D GH (control)', r'NeoHookeanShell2D_do_compute_gradient_hessian')
show('do_assemble k8 (contact ctx)', r'do_assemble_kernel<\(bool\)0, \(int\)1, \(bool\)1, \(bool\)1, \(bool\)0, \(int\)8>', per=1, unit='launch')
show('fused_update_xr (control)', r'fused_update_xr', per=1, unit='launch')
print('== total GPU kernel time')
for tag in ['new_r1','old_r1','old_r2','new_r2']:
    t = sum(v['tot'] for v in runs[tag].values())
    print(f'  {tag}: {t/1000.0:.2f} s')

import csv, sys, statistics as st
from collections import defaultdict
keys = ['do_assemble_kernel<(bool)0, (int)1', 'do_assemble_kernel<(bool)0, (int)2', 'StableNeoHookean3D_do_compute_gra',
        'StrainLimitingBaraffWitkinShell2D_do_c', 'DiscreteShellBending', 'FiniteElementBDF1Kinetic_do_co',
        'FEMLinearSubsystem_assemble_dy', 'FEMLinearSubsystem_assemble_re', 'buffer_view_fill_kerne', 'do_assemble_kernel(bool',
        'fast_segmental_reduce_matrix', 'rbk_sym_spmv_dot']
out = {}
for f in sys.argv[1:]:
    d = defaultdict(list)
    for r in csv.DictReader(open(f)):
        n = r['Name']
        for k in keys:
            if k in n:
                d[k].append(int(r['Duration (ns)'])); break
    out[f] = d
print(f"{'kernel (us/launch, n)':42}" + "".join(f"{f.split('/')[-1][:12]:>22}" for f in out))
for k in keys:
    print(f"{k:42}" + "".join(f"{st.mean(out[f][k])/1e3:12.1f} n={len(out[f][k]):5d}" if out[f][k] else f"{'-':>22}" for f in out))

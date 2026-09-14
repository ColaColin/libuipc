"""s10 scope: per-kernel totals from nsys cuda_gpu_kern_sum, arms of UIPC_ABD_GH_PREPASS."""
import csv, glob, re, statistics, sys
SC = sys.argv[1]
ARMS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['c1','c0','c2']
KEYS = [
    ('ortho G/H',   lambda n: 'ortho_potential_compute_gradient_hessian' in n),
    ('abd kin G/H', lambda n: 'affine_body_bdf1_kinetic_compute_gradient_hessian' in n),
    ('contact p1',  lambda n: re.search(r'do_assemble_kernel<\(bool\)0, \(int\)1,', n) is not None),
    ('contact p2',  lambda n: re.search(r'do_assemble_kernel<\(bool\)0, \(int\)2,', n) is not None),
    ('abd_diag',    lambda n: 'abd_diag_preconditioner_do_assemble' in n),
    ('ks k1+k2',    lambda n: 'assemble_kinetic_shape_k' in n),
    ('dytopo warp', lambda n: 'dytopo_effect_pair_warp' in n),
    ('SpMV',        lambda n: 'spmv_dot_chunked' in n),
]
rows = {}
for arm in ARMS:
    files = sorted(glob.glob(f'nsys/{SC}_{arm}_r*_cuda_gpu_kern_sum.csv'))
    if not files:
        continue
    per = {k: [] for k, _ in KEYS}
    inst = {k: [] for k, _ in KEYS}
    gpu = []
    for f in files:
        tot = 0; acc = {k: [0, 0] for k, _ in KEYS}
        for r in csv.DictReader(open(f)):
            n = r['Name']; t = int(r['Total Time (ns)']); i = int(r['Instances'])
            tot += t
            for k, pred in KEYS:
                if pred(n):
                    acc[k][0] += t; acc[k][1] += i
        gpu.append(tot / 1e6)
        for k, _ in KEYS:
            per[k].append(acc[k][0] / 1e6)
            inst[k].append(acc[k][1])
    rows[arm] = (per, inst, gpu)
base = ARMS[0]
print(f"== {SC}: nsys kern_sum, full runs, n={len(rows[base][2])} per arm; totals in ms per run")
hdr = f"{'kernel':14s}" + ''.join(f"{a:>12s}{'d%':>9s}" for a in ARMS if a in rows)
print(hdr)
for k, _ in KEYS:
    line = f"{k:14s}"
    b = statistics.mean(rows[base][0][k]) if rows[base][0][k] else 0
    for a in ARMS:
        if a not in rows: continue
        m = statistics.mean(rows[a][0][k])
        d = (100 * (m / b - 1)) if b else 0
        line += f"{m:12.1f}{d:+9.2f}"
    print(line)
line = f"{'GPU total':14s}"
bg = statistics.mean(rows[base][2])
for a in ARMS:
    if a not in rows: continue
    g = statistics.mean(rows[a][2]); line += f"{g:12.1f}{100*(g/bg-1):+9.2f}"
print(line)
print("launch counts (mean):")
for k, _ in KEYS:
    print(f"  {k:14s} " + ' '.join(f"{a}={statistics.mean(rows[a][1][k]):.0f}" for a in ARMS if a in rows))
for a in ARMS:
    if a in rows:
        print(f"  raw GPU total ms {a}: {[round(x,1) for x in rows[a][2]]}")

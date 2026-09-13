import csv, glob, re, statistics, sys, json
SC = sys.argv[1]; ARMS = sys.argv[2].split(',') if len(sys.argv)>2 else ['b0','b1','b2']
def load(f):
    out = {}
    tot = 0
    for r in csv.DictReader(open(f)):
        n = r['Name']; t = int(r['Total Time (ns)']); inst = int(r['Instances'])
        out[n] = (t, inst); tot += t
    return out, tot
def key(n):
    m = re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([12]),', n)
    if m: return 'part'+m.group(1)
    return None
rows = {}
for arm in ARMS:
    per = {'part1': [], 'part2': [], 'gpu': [], 'p1inst': [], 'p2inst': []}
    for f in sorted(glob.glob(f'nsys/{SC}_{arm}_r*_cuda_gpu_kern_sum.csv')):
        d, tot = load(f)
        p = {}
        for n,(t,i) in d.items():
            k = key(n)
            if k: p[k] = (t,i)
        per['part1'].append(p['part1'][0]/p['part1'][1]/1000.0)
        per['part2'].append(p['part2'][0]/p['part2'][1]/1000.0)
        per['p1inst'].append(p['part1'][1]); per['p2inst'].append(p['part2'][1])
        per['gpu'].append(tot/1e6)
    rows[arm] = per
base = rows[ARMS[0]]
print(f"== {SC}: contact part 1 / part 2 per launch (us), full runs, n={len(base['part1'])} per arm")
print(f"{'arm':4s} {'part1 us/launch':>18s} {'d%':>8s} {'part2 us/launch':>18s} {'d%':>8s} {'sceneGPU ms':>12s} {'d%':>8s} {'p1 launches':>12s}")
for arm in ARMS:
    p = rows[arm]
    m1 = statistics.mean(p['part1']); m2 = statistics.mean(p['part2']); g = statistics.mean(p['gpu'])
    b1 = statistics.mean(base['part1']); b2 = statistics.mean(base['part2']); bg = statistics.mean(base['gpu'])
    print(f"{arm:4s} {m1:18.1f} {100*(m1/b1-1):+8.2f} {m2:18.1f} {100*(m2/b2-1):+8.2f} {g:12.1f} {100*(g/bg-1):+8.2f} {statistics.mean(p['p1inst']):12.0f}")
    print(f"     raw part1: {[round(x,1) for x in p['part1']]}")

import csv, glob, re, statistics, sys
SC = sys.argv[1]
ARMS = ['a2','a1','a0']
def load(f):
    rows=[]
    for r in csv.DictReader(open(f)):
        rows.append((r['Name'], int(r['Total Time (ns)']), int(r['Instances'])))
    return rows
def parts(rows):
    out={'p0':(0,0),'p1':(0,0),'p2':(0,0)}
    tot=0
    for n,t,i in rows:
        tot+=t
        m=re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([012]),', n)
        if m:
            k='p'+m.group(1)
            a,b=out[k]; out[k]=(a+t,b+i)
    return out, tot
res={}
for arm in ARMS:
    per=[]
    for f in sorted(glob.glob(f'nsys/{SC}_{arm}_r*_cuda_gpu_kern_sum.csv')):
        p,tot=parts(load(f))
        log=f.replace('_cuda_gpu_kern_sum.csv','.run.log')
        mm=re.search(r'TOTAL frames=(\d+) mean=([0-9.]+)ms median=([0-9.]+)ms', open(log).read())
        frames=int(mm.group(1)); wall=float(mm.group(2))*frames
        contact=(p['p0'][0]+p['p1'][0]+p['p2'][0])/1e6
        per.append(dict(p0=p['p0'][0]/1e6,p1=p['p1'][0]/1e6,p2=p['p2'][0]/1e6,
                        n0=p['p0'][1],n1=p['p1'][1],n2=p['p2'][1],
                        contact=contact, gpu=tot/1e6, wall=wall,
                        meanms=float(mm.group(2)), medms=float(mm.group(3))))
    if per: res[arm]=per
def m(arm,k): return statistics.mean([d[k] for d in res[arm]])
print(f"== {SC}  n={len(res.get('a2',[]))} nsys full runs per arm")
print(f"{'arm':4s} {'p0 ms':>8s} {'p1 ms':>8s} {'p2 ms':>8s} {'contactSum':>11s} {'d%':>7s} {'gpuSum ms':>10s} {'d%':>7s} {'wall ms':>9s} {'d%':>7s} {'sum/wall':>9s} {'launches':>9s}")
for arm in ARMS:
    if arm not in res: continue
    b='a2'
    line=(f"{arm:4s} {m(arm,'p0'):8.1f} {m(arm,'p1'):8.1f} {m(arm,'p2'):8.1f} "
          f"{m(arm,'contact'):11.1f} {100*(m(arm,'contact')/m(b,'contact')-1):+7.2f} "
          f"{m(arm,'gpu'):10.1f} {100*(m(arm,'gpu')/m(b,'gpu')-1):+7.2f} "
          f"{m(arm,'wall'):9.1f} {100*(m(arm,'wall')/m(b,'wall')-1):+7.2f} "
          f"{m(arm,'gpu')/m(arm,'wall'):9.3f} "
          f"{max(m(arm,'n0'),m(arm,'n1')):9.0f}")
    print(line)
    print(f"     raw wall {[round(d['wall'],1) for d in res[arm]]}  contact {[round(d['contact'],1) for d in res[arm]]}")
# per launch
print("  per-launch us:")
for arm in ARMS:
    if arm not in res: continue
    d=res[arm][0]
    s=[]
    for k,n in (('p0','n0'),('p1','n1'),('p2','n2')):
        inst=statistics.mean([x[n] for x in res[arm]])
        if inst: s.append(f"{k}={statistics.mean([x[k] for x in res[arm]])*1000/inst:8.1f} ({inst:.0f} launches)")
    print(f"   {arm}: "+"  ".join(s))

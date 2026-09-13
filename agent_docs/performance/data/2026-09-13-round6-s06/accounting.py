import csv,glob,statistics,sys
def load(f):
    per={}; tot=0.0
    for r in csv.DictReader(open(f)):
        n=r['Name']; t=float(r['Total Time (ns)'])/1e6; inst=int(r['Instances'])
        tot+=t
        for k in ('filter_toi_k4','filter_toi_k3','filter_active_k4','filter_active_k3',
                  'stacklessSelf','stacklessOther','pairFilter_kernel<(bool)1','pairFilter_kernel<(bool)0'):
            if k in n:
                a=per.setdefault(k,[0.0,0]); a[0]+=t; a[1]+=inst
    per['TOTAL']=[tot,0]
    return per
pre=sys.argv[1]
res={a:[load(f) for f in sorted(glob.glob(f'{pre}_{a}_r*_cuda_gpu_kern_sum.csv'))] for a in ('c0','c1')}
def M(a,k,i): return statistics.mean([r[k][i] for r in res[a] if k in r])
print(f"runs c0={len(res['c0'])} c1={len(res['c1'])}")
nt0=M('c0','filter_toi_k4',1); nt1=M('c1','filter_toi_k4',1)
print(f"filter_toi_k4 launches (~Newton iterations): {nt0:.0f} -> {nt1:.0f}  ({100*(nt1-nt0)/nt0:+.2f} %)")
print(f"\n{'kernel':26} {'total ms old':>13} {'total ms new':>13} {'delta ms':>10} {'delta %':>9}")
acc=0.0
for k in ('filter_active_k4','filter_active_k3','filter_toi_k4','filter_toi_k3',
          'stacklessSelf','stacklessOther','pairFilter_kernel<(bool)1','pairFilter_kernel<(bool)0'):
    A=M('c0',k,0); B=M('c1',k,0); d=B-A
    if k.startswith('filter'): acc+=d
    print(f"{k:26} {A:13.1f} {B:13.1f} {d:+10.1f} {100*d/A:+8.2f} %")
T0=M('c0','TOTAL',0); T1=M('c1','TOTAL',0)
print(f"\nfour targeted kernels        NET {acc:+.1f} ms   = {100*acc/T0:+.2f} % of the old arm's scene GPU kernel time")
print(f"scene GPU kernel time       {T0:.1f} -> {T1:.1f}  ({100*(T1-T0)/T0:+.2f} %) RAW")
print(f"scene GPU kernel time per filter_toi launch: {T0/nt0:.4f} -> {T1/nt1:.4f} ms  ({100*((T1/nt1)-(T0/nt0))/(T0/nt0):+.2f} %) NORMALISED")

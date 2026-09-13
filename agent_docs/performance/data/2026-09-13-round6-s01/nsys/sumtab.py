import csv,sys,re
def short(name):
    n=name.split('(')[0]
    n=n.replace('void ','').replace('<unnamed>::','')
    n=re.sub(r'<[^<>]*>','',n)
    return n.split('::')[-1]
def load(p):
    d={}; tot=0.0
    for row in csv.DictReader(open(p)):
        t=float(row['Total Time (ns)']); n=int(row['Instances']); k=short(row['Name'])
        a,b=d.get(k,(0.0,0)); d[k]=(a+t,b+n); tot+=t
    return d,tot
files=sys.argv[1:]
alld=[load(f) for f in files]
keys=set()
for d,_ in alld: keys|=set(d)
pat=sys.argv[0]
sel=[k for k in keys if any(s in k for s in ('DiscreteShellBending','StrainLimiting','do_assemble_kernel','abd_diag','filter_toi','Spmv'))]
hdr='%-58s '%'kernel (ms/launch / launches)' + ' '.join('%20s'%f.split('/')[-1].replace('_cuda_gpu_kern_sum.csv','') for f in files)
print(hdr)
for k in sorted(sel):
    cells=[]
    for d,_ in alld:
        t,n=d.get(k,(0,0)); cells.append('%9.4f/%6d'%((t/n/1e6) if n else 0,n))
    print('%-58s '%k[:58] + ' '.join('%20s'%c for c in cells))
print('%-58s '%'TOTAL kernel ms' + ' '.join('%20.1f'%(t/1e6) for _,t in alld))

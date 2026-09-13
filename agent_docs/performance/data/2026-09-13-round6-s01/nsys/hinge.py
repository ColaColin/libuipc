import csv,sys,re
for p in sys.argv[1:]:
    tot=0.0; hits=[]
    for row in csv.DictReader(open(p)):
        n=row['Name']
        t=float(row['Total Time (ns)']); c=int(row['Instances']); tot+=t
        if 'DiscreteShellBending_do_compute_gradient_hessian' in n:
            var='occ3' if '_occ3' in n else ('occ4' if '_occ4' in n else 'plain')
            hits.append((var,t/c/1e6,c,t/1e6))
    for v,ms,c,tt in hits:
        print('%-22s %-6s %8.4f ms/launch  n=%5d  total=%8.1f ms   scene kernel total=%8.1f  share=%5.2f%%'%(
            p.split('/')[-1].replace('_cuda_gpu_kern_sum.csv',''),v,ms,c,tt,tot/1e6,100*tt/tot))

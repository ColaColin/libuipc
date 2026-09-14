#!/usr/bin/env python3
"""s15 scope: per-kernel cost + launch geometry per nsys session (sqlite), MAS/PCG family."""
import sqlite3, sys, os
FAMILY=['%MASPreconditionerEngine_%','%Spmv_rbk_sym_spmv_dot_chunked%','%fused_update_xr%','%fused_dot%','%fused_update_p%','%fused_pcg_scalar%','%StableNeoHookean3D%gradient_hessian%']
top = '--top' in sys.argv
for path in [a for a in sys.argv[1:] if a.endswith('.sqlite')]:
    tag=os.path.basename(path).replace('.sqlite','%')
    c=sqlite3.connect(path); cur=c.cursor()
    allt=list(cur.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    print('=== %s  kernel sum %.1f ms'%(tag, allt/1e6))
    if top:
        q="""select s.value, count(*), sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k
             join StringIds s on k.demangledName=s.id group by s.value order by 3 desc limit 16"""
        for name,n,t in cur.execute(q):
            nm=name.split('(')[0].split('::')[-1]
            print('  %5.2f%%  %9.2f ms  %6d x %8.2f us  %s'%(100*t/allt,t/1e6,n,t/1e3/n,nm[:70]))
        continue
    for pat in FAMILY:
        q="""select s.value, k.gridX, k.blockX, k.registersPerThread, count(*), sum(k.end-k.start)
             from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
             where s.value like ? group by s.value, k.gridX, k.blockX order by 6 desc"""
        rows=list(cur.execute(q,(pat,)))
        seen={}
        for name,gx,bx,reg,n,t in rows:
            nm=name.split('(')[0].split('::')[-1].replace('MASPreconditionerEngine_','%MAS_')
            seen.setdefault(nm,[]).append((gx,bx,reg,n,t))
        for nm,lst in seen.items():
            n=sum(x[3] for x in lst); t=sum(x[4] for x in lst)
            geo=' '.join('%dx%d@%dr(%d)'%(gx,bx,reg,nn) for gx,bx,reg,nn,tt in lst[:4])
            print('  %5.2f%%  %9.2f ms  %6d x %8.3f us  %-42s %s'%(100*t/allt,t/1e6,n,t/1e3/n,nm[:42],geo))

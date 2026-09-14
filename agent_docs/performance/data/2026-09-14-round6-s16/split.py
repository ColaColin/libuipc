#!/usr/bin/env python3
"""Per-launch us of each MAS apply kernel by grid size (fine vs coarse launch), plus PCG launches."""
import sqlite3, sys, os
print('| session | kernel | grid | regs | launches | us/launch | per-apply us | kernel sum ms |')
print('|---|---|---:|---:|---:|---:|---:|---:|')
for path in sys.argv[1:]:
    tag=os.path.basename(path)[:-7]
    c=sqlite3.connect(path); cur=c.cursor()
    allt=list(cur.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    q="""select s.value, k.gridX, k.registersPerThread, count(*), sum(k.end-k.start)
         from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
         where s.value like '%MASPreconditionerEngine_%' and (s.value like '%rowdot%' or s.value like '%fused_R%' or s.value like '%build_multi_level_R%' or s.value like '%collect_final_Z%')
         group by s.value, k.gridX order by s.value, k.gridX desc"""
    tot=0.0
    for name,gx,reg,n,t in cur.execute(q):
        nm=name.split('(')[0].split('MASPreconditionerEngine_')[-1][:44]
        tot+=t/1e3/n
        print('| %s | `%s` | %d | %d | %d | %.3f | | %.1f |'%(tag,nm,gx,reg,n,t/1e3/n,allt/1e6))
    print('| %s | **apply total** | | | | | **%.3f** | |'%(tag,tot))

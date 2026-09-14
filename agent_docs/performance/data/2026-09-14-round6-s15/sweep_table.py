#!/usr/bin/env python3
"""s15 sweep table: rowdot per-launch us, grid, regs, and PCG count per nsys session."""
import sqlite3, sys, os, glob
rows=[]
for path in sorted(glob.glob('nsys/*.sqlite')):
    tag=os.path.basename(path)[:-7]
    c=sqlite3.connect(path); cur=c.cursor()
    allt=list(cur.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL"))[0][0]
    q="""select s.value, k.gridX, k.blockX, k.registersPerThread, count(*), sum(k.end-k.start)
         from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
         where s.value like '%schwarz_local_solve_rowdot%' group by s.value"""
    for name,gx,bx,reg,n,t in cur.execute(q):
        nm='rowdot2'+name.split('rowdot2_kernel')[1].split('(')[0] if 'rowdot2' in name else 'rowdot(s12)'
        rows.append((tag,nm,gx,bx,reg,n,t/1e3/n,t/1e6,100*t/allt,allt/1e6))
print('| session | kernel | grid | block | regs | launches | us/launch | ms total | % of kernel sum | kernel sum ms |')
print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    print('| %s | `%s` | %d | %d | %d | %d | %.3f | %.1f | %.2f | %.1f |'%r)

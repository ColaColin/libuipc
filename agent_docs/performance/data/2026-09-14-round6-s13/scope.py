#!/usr/bin/env python3
"""s13 scope report: per-launch SpMV+dot cost and grid shape, per nsys session."""
import sqlite3, sys, os
KERN='%spmv_dot_chunked%'
for path in sys.argv[1:]:
    tag=os.path.basename(path).replace('.sqlite','')
    c=sqlite3.connect(path); cur=c.cursor()
    q="""select k.gridX, count(*), sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k
    join StringIds s on k.demangledName=s.id where s.value like ? group by k.gridX order by 2 desc"""
    rows=list(cur.execute(q,(KERN,)))
    if not rows:
        print('%-14s NO SpMV LAUNCHES'%tag); continue
    tot=sum(r[2] for r in rows); n=sum(r[1] for r in rows)
    qa="""select sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k"""
    allt=list(cur.execute(qa))[0][0]
    print('%-14s n=%6d  %8.3f us/launch  %8.1f ms = %5.2f%% of kernel sum  grids=%s'
          %(tag,n,tot/1e3/n,tot/1e6,100*tot/allt,[(r[0],r[1]) for r in rows[:5]]))

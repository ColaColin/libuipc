import csv, sys, statistics as st, bisect
from collections import defaultdict
rows=[]
for r in csv.DictReader(open(sys.argv[1])):
    try: s=int(r['Start (ns)']); d=int(r['Duration (ns)'])
    except ValueError: continue
    n=r['Name']
    if n.startswith('[CUDA mem'): continue
    rows.append(dict(s=s,e=s+d,dur=d,strm=r['Strm'],name=n))
rows.sort(key=lambda x:x['s'])
p1=[r for r in rows if 'do_assemble_kernel<(bool)0, (int)1' in r['name']]
def short(n): return n.split('::')[-1][:52] if '::' in n else n[:52]
cover=defaultdict(float); tot=0.0; wins=0
for w in p1:
    wins+=1
    for r in rows:
        if r['s']>=w['e']: break
        if r['e']<=w['s'] or r is w or 'do_assemble_kernel<(bool)0, (int)2' in r['name']: continue
        o=min(r['e'],w['e'])-max(r['s'],w['s'])
        if o>0: cover[short(r['name'])]+=o; tot+=o
p1dur=sum(w['dur'] for w in p1)
print(f"part 1: n={len(p1)} mean {p1dur/len(p1)/1e3:.1f} us; window covered by non-part-2 kernels: {100*tot/p1dur:.1f}%")
for k,v in sorted(cover.items(), key=lambda kv:-kv[1])[:8]: print(f"   {100*v/p1dur:5.1f}%  {k}")
keys=['do_assemble_kernel<(bool)0, (int)1','do_assemble_kernel<(bool)0, (int)2','SimplexFrictionalContact','do_assemble_kernel(bool','ortho_potential','abd_linear_subsystem_dytopo_pair_key','abd_linear_subsystem_assemble_dytopo','abd_diag_preconditioner','DeviceRadixSort']
d=defaultdict(list)
for r in rows:
    for k in keys:
        if k in r['name']: d[k].append(r['dur']); break
for k in keys:
    if d[k]: print(f"   {k:44} {st.mean(d[k])/1e3:8.1f} us  n={len(d[k])}")

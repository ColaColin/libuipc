"""s09: what runs next to contact part 1 on the timeline, and how much of part 1's
window the GPU spends with nothing else resident.  Input: a cuda_gpu_trace csv."""
import csv, re, sys, statistics
from collections import Counter, defaultdict
f=sys.argv[1]
ev=[]
for r in csv.DictReader(open(f)):
    n=r.get('Name') or ''
    if not r['Start (ns)'] or not r['Duration (ns)']: continue
    s=float(r['Start (ns)']); d=float(r['Duration (ns)'])
    ev.append((s,s+d,n,r.get('Strm'),r.get('GrdX'),r.get('BlkX'),r.get('Reg/Trd')))
ev.sort()
def short(n):
    n=n.split('(')[0]
    m=re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([012]),.*\(int\)(\d)>',
                n if 'do_assemble_kernel<' in n else '')
    if m: return f'contact_part{m.group(1)}(Proj{m.group(2)})'
    return n.split('::')[-1][:52]
p1=[e for e in ev if 'do_assemble_kernel<(bool)0, (int)1,' in e[2]]
print(f"part1 launches: {len(p1)}; grid {Counter(e[4] for e in p1).most_common(3)}; "
      f"block {Counter(e[5] for e in p1).most_common(3)}; reg {Counter(e[6] for e in p1).most_common(3)}")
# co-residency during part 1 windows
cov=0.0; tot=0.0; comp=Counter(); solo=0.0
for a,b,_,_,_,_,_ in p1:
    tot+=b-a
    iv=[]
    for s,e,n,_,_,_,_ in ev:
        if e<=a or s>=b: continue
        if 'do_assemble_kernel<(bool)0, (int)1,' in n: continue
        iv.append((max(s,a),min(e,b))); comp[short(n)]+=min(e,b)-max(s,a)
    iv.sort(); cur=None; c=0.0
    for s,e in iv:
        if cur is None: cur=[s,e]
        elif s<=cur[1]: cur[1]=max(cur[1],e)
        else: c+=cur[1]-cur[0]; cur=[s,e]
    if cur: c+=cur[1]-cur[0]
    cov+=c
print(f"part1 total window {tot/1e6:8.1f} ms; something else on the GPU for {100*cov/tot:5.1f} % of it")
print("  top co-resident kernels (ms inside part1 windows):")
for k,v in comp.most_common(8): print(f"    {v/1e6:8.1f} ms  {k}")
# what is adjacent on the timeline
before=Counter(); after=Counter()
idx={id(e):i for i,e in enumerate(ev)}
for e in p1:
    i=ev.index(e)
    j=i-1
    while j>=0 and 'do_assemble_kernel<(bool)0, (int)1,' in ev[j][2]: j-=1
    if j>=0: before[short(ev[j][2])]+=1
    j=i+1
    while j<len(ev) and 'do_assemble_kernel<(bool)0, (int)1,' in ev[j][2]: j+=1
    if j<len(ev): after[short(ev[j][2])]+=1
print("  starts immediately before part1:", before.most_common(4))
print("  starts immediately after  part1:", after.most_common(4))
# GPU busy union over the whole trace
iv=sorted((s,e) for s,e,*_ in ev)
busy=0.0; cur=None
for s,e in iv:
    if cur is None: cur=[s,e]
    elif s<=cur[1]: cur[1]=max(cur[1],e)
    else: busy+=cur[1]-cur[0]; cur=[s,e]
busy+=cur[1]-cur[0]
span=iv[-1][1]-iv[0][0]
ksum=sum(e-s for s,e in iv)
print(f"trace span {span/1e6:8.1f} ms; GPU busy (union) {busy/1e6:8.1f} ms = {100*busy/span:5.1f} %; "
      f"kernel-duration sum {ksum/1e6:8.1f} ms (sum/union {ksum/busy:5.3f})")

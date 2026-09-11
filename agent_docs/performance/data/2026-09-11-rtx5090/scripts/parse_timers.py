import re,glob,sys,json
NAMES=[('Newton Iteration','Newton'),('Solve Global Linear System','Global solve'),('FusedPCG','FusedPCG'),('Assemble Subsystems','Subsystem assembly'),('Line Search','Line search'),('Detect Trajectory Candidates','Trajectory detect'),('Compute DyTopo Effect','DyTopo'),('Detect DCD Candidates','inner DCD detect')]
rows={}
for f in sorted(glob.glob('raw/timers/*_timers.stdout')):
    tree,scene=re.match(r'.*/(\w+)_(.+)_timers\.stdout',f).groups()
    txt=open(f).read()
    frames=re.search(r'TOTAL frames=(\d+) mean=([0-9.]+)ms',txt)
    t={}
    for m in re.finditer(r'\*([^*|\n]+?)\s*\|\s*([0-9.]+) ms \|\s*(\d+)',txt):
        name=m.group(1).strip(); 
        if name not in t: t[name]=(float(m.group(2)),int(m.group(3)))
    rows[(scene,tree)]=(frames.groups() if frames else None,t)
order=['rigid-wrecking-balls','stiff-gipc-case2','mas-bunny','cube-wall-cloth']
print('| Benchmark (frames / Newton calls) | tree | timer-run mean ms/frame | '+' | '.join(n[1] for n in NAMES)+' |')
print('|---|---|---:|'+'---:|'*len(NAMES))
for s in order:
    for tree in ['head','dahl','base']:
        fr,t=rows.get((s,tree),(None,{}))
        if not t: continue
        newton=t.get('Newton Iteration',(0,1))[1]
        cells=[]
        for key,_ in NAMES:
            v=t.get(key); cells.append(f'{v[0]/newton:.2f}' if v else 'n/a')
        print(f'| {s} ({fr[0]} / {newton}) | {tree} | {float(fr[1]):.1f} | '+' | '.join(cells)+' |')
# missing-scope report
allnames=set(); [allnames.update(t.keys()) for _,t in rows.values()]
for tree in ['head','base','dahl']:
    have=set(); [have.update(t.keys()) for (s,tr),(_,t) in rows.items() if tr==tree]
    miss=[n for n in [k for k,_ in NAMES] if n not in have]
    if miss: print(f'\n{tree}: scopes missing: {miss}', file=sys.stderr)

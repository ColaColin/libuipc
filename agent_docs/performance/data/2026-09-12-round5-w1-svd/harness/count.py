import re,sys,subprocess
o=subprocess.run(['/workspace/deps/cuda-12.8/bin/cuobjdump','-sass',sys.argv[1]],capture_output=True,text=True).stdout
# split into functions
funcs={}; cur=None
for line in o.splitlines():
    m=re.match(r'\s*Function : (.+)',line)
    if m: cur=m.group(1); funcs[cur]=[]; continue
    if cur is not None: funcs[cur].append(line)
tag=sys.argv[2] if len(sys.argv)>2 else ''
for name,lines in funcs.items():
    if 'gradient_hessian' not in name: continue
    hoist = 'ILb1E' in name
    ops={}
    nbr=0
    for l in lines:
        m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z0-9_.]+)',l)
        if not m: continue
        op=m.group(1).split('.')[0]
        ops[op]=ops.get(op,0)+1
    fp64=sum(v for k,v in ops.items() if k in ('DADD','DMUL','DFMA','DSETP','DMNMX','DDIV'))
    total=sum(ops.values())
    print(f"{tag} hoist={hoist}: FP64={fp64} (DFMA={ops.get('DFMA',0)} DMUL={ops.get('DMUL',0)} DADD={ops.get('DADD',0)} DSETP={ops.get('DSETP',0)} DMNMX={ops.get('DMNMX',0)}) MUFU={ops.get('MUFU',0)} total={total} LDL={ops.get('LDL',0)} STL={ops.get('STL',0)} BRA={ops.get('BRA',0)}")

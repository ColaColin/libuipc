import re,sys,subprocess
o=subprocess.run(['/usr/local/cuda/bin/cuobjdump','-sass',sys.argv[1]],capture_output=True,text=True).stdout
funcs={}; cur=None
for line in o.splitlines():
    m=re.match(r'\s*Function : (.+)',line)
    if m: cur=m.group(1); funcs[cur]=[]; continue
    if cur is not None: funcs[cur].append(line)
tag=sys.argv[2] if len(sys.argv)>2 else ''
for name,lines in sorted(funcs.items()):
    if 'gradient_hessian' not in name: continue
    # template args appear as ILb0E / ILb1E in order
    bools=re.findall(r"kernelI(?:Lb([01])E)(?:Lb([01])E)(?:Lb([01])E)?",name)[0] if re.search(r"kernelILb",name) else []
    ops={}
    for l in lines:
        m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z0-9_.]+)',l)
        if not m: continue
        op=m.group(1).split('.')[0]
        ops[op]=ops.get(op,0)+1
    fp64=sum(v for k,v in ops.items() if k in ('DADD','DMUL','DFMA','DSETP','DMNMX','DDIV'))
    total=sum(ops.values())
    print(f"{tag} tmpl={','.join(bools)}: FP64={fp64} (DFMA={ops.get('DFMA',0)} DMUL={ops.get('DMUL',0)} DADD={ops.get('DADD',0)} DSETP={ops.get('DSETP',0)} DMNMX={ops.get('DMNMX',0)}) MUFU={ops.get('MUFU',0)} total={total} LDL={ops.get('LDL',0)} STL={ops.get('STL',0)} LDG={ops.get('LDG',0)} STG={ops.get('STG',0)} SEL={ops.get('SEL',0)} IMAD={ops.get('IMAD',0)} MOV={ops.get('MOV',0)} BRA={ops.get('BRA',0)}")

import re,sys,subprocess
o=subprocess.run(['/workspace/deps/cuda-12.8/bin/cuobjdump','-sass',sys.argv[1]],capture_output=True,text=True).stdout
cur=None; ins=None; saved=None
for line in o.splitlines():
    m=re.match(r'\s*Function : (.+)',line)
    if m:
        cur=m.group(1)
        ins=[] if ('gradient_hessian' in cur and 'ILb1E' in cur) else None
        if ins is not None: saved=ins
        continue
    if ins is None: continue
    m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?\w+\s+)?([A-Z0-9_.]+)(.*)',line)
    if m: ins.append((int(m.group(1),16),m.group(2),m.group(3)))
ins=saved
addr2i={a:i for i,(a,op,rest) in enumerate(ins)}
FP={'DADD','DMUL','DFMA','DSETP','DMNMX','DDIV'}
print(f"kernel <true>: {len(ins)} instructions, {sum(1 for a,o_,r in ins if o_.split('.')[0] in FP)} FP64")
for i,(a,op,rest) in enumerate(ins):
    if not op.startswith('BRA'): continue
    m=re.search(r'0x([0-9a-f]+)',rest)
    if not m: continue
    tgt=int(m.group(1),16)
    if tgt < a and tgt in addr2i:   # backward branch = loop
        j=addr2i[tgt]
        body=ins[j:i+1]
        fp=sum(1 for _,o_,_ in body if o_.split('.')[0] in FP)
        mufu=sum(1 for _,o_,_ in body if o_.startswith('MUFU'))
        print(f"  LOOP 0x{tgt:x}..0x{a:x}: {len(body)} instr, FP64={fp}, MUFU={mufu}")

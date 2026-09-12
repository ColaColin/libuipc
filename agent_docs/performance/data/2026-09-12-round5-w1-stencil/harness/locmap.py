import re,sys,subprocess
o=subprocess.run(['/usr/local/cuda/bin/cuobjdump','-sass',sys.argv[1]],capture_output=True,text=True).stdout
want=sys.argv[2]  # e.g. Lb1ELb1ELb1E
funcs={}; cur=None
for line in o.splitlines():
    m=re.match(r'\s*Function : (.+)',line)
    if m: cur=m.group(1); funcs[cur]=[]; continue
    if cur is not None: funcs[cur].append(line)
for name,lines in funcs.items():
    if 'gradient_hessian_kernelI'+want not in name: continue
    ev=[]
    for l in lines:
        m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?\w+\s+)?([A-Z0-9_.]+)',l)
        if not m: continue
        addr=int(m.group(1),16); op=m.group(2).split('.')[0]
        if op in ('LDL','STL'): ev.append((addr,op,l.strip()[:110]))
    # bucket per 0x400
    from collections import Counter
    cl=Counter(); cs=Counter()
    for a,op,_ in ev:
        (cl if op=='LDL' else cs)[a//0x800*0x800]+=1
    print("bucket   LDL  STL")
    for b in sorted(set(cl)|set(cs)):
        print(f"0x{b:05x} {cl[b]:5d} {cs[b]:5d}")
    print("total LDL",sum(cl.values()),"STL",sum(cs.values()))
    # print offsets used
    offs=Counter()
    for a,op,txt in ev:
        m=re.search(r'\[(R\d+(\.64|\.128)?)?(\+?-?0x[0-9a-f]+)?\]',txt)
        offs[m.group(0) if m else '?']+=1
    print("addr forms:", offs.most_common(20))

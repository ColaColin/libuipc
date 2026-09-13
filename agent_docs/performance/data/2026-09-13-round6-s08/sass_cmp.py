import re, sys, hashlib
def norm(n):
    n=re.sub(r'__nv_static_\d+__[0-9a-f]+_\d+_[A-Za-z0-9_]+?_cu_[0-9a-f]+_\d+__','ANON',n)
    n=re.sub(r'_GLOBAL__N__[0-9a-f]+_\d+_[A-Za-z0-9_]+?_cu_[0-9a-f]+_\d+','GLOBALN',n)
    return n
def load(f):
    fns={}; cur=None; body=[]
    for line in open(f):
        m=re.match(r'\s*Function : (\S+)', line)
        if m:
            if cur: fns[cur]=body
            cur=norm(m.group(1)); body=[]
            continue
        if cur is not None:
            t=re.sub(r'/\*[0-9a-f]{4}\*/','',line)
            t=re.sub(r'/\* 0x[0-9a-f]+ \*/','',t).strip()
            if t: body.append(t)
    if cur: fns[cur]=body
    return fns
a=load(sys.argv[1]); b=load(sys.argv[2])
ka,kb=set(a),set(b); common=ka&kb
print(f"base functions {len(ka)}, head functions {len(kb)}, common {len(common)}")
diff=0
for k in sorted(common):
    ha=hashlib.md5('\n'.join(a[k]).encode()).hexdigest()[:12]
    hb=hashlib.md5('\n'.join(b[k]).encode()).hexdigest()[:12]
    if ha!=hb:
        diff+=1
        nd=sum(1 for x,y in zip(a[k],b[k]) if x!=y)
        fp=lambda v:sum(1 for l in v if re.match(r'\s*(DADD|DMUL|DFMA|DSETP|DMNMX|MUFU)',l))
        print(f"  DIFFERS {ha}->{hb} instr {len(a[k])}->{len(b[k])} lines-differing {nd} FP64 {fp(a[k])}->{fp(b[k])}")
        print(f"    {k[:120]}")
print(f"identical SASS streams: {len(common)-diff} / {len(common)}   differing: {diff}")
onlyb=sorted(kb-ka)
print(f"functions only in head (new instantiations): {len(onlyb)}")
for k in onlyb:
    print(f"    {len(b[k]):6d} instr  {k[:110]}")
onlya=sorted(ka-kb)
if onlya:
    print(f"functions only in base (REMOVED): {len(onlya)}")
    for k in onlya:
        print(f"    {len(a[k]):6d} instr  {k[:110]}")

# s06: the new template argument renames every filter_toi instantiation, so
# also match base-only against head-only functions by BODY HASH.
ha={hashlib.md5('\n'.join(v).encode()).hexdigest():k for k,v in a.items()}
matched=[]; unmatched=[]
for k in onlya:
    h=hashlib.md5('\n'.join(a[k]).encode()).hexdigest()
    hit=[kk for kk in onlyb if hashlib.md5('\n'.join(b[kk]).encode()).hexdigest()==h]
    (matched if hit else unmatched).append((k,hit[0] if hit else None))
print(f"\nbody-hash rename match: {len(matched)}/{len(onlya)} base-only functions found byte-identical in head")
for k,kk in matched:
    print(f"    OK  {len(a[k]):6d} instr  {k[:90]}\n          ->  {kk[:90]}")
for k,_ in unmatched:
    print(f"    NO MATCH {len(a[k]):6d} instr  {k[:110]}")

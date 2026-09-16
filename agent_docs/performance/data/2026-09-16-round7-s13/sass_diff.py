#!/usr/bin/env python3
"""s17: per-function normalised SASS identity between two cuobjdump dumps + census of named kernels."""
import re,sys,collections
def split(path):
    funcs={}; cur=None
    for line in open(path,errors='replace'):
        m=re.match(r'\s*Function : (.*)',line)
        if m: cur=m.group(1).strip(); funcs[cur]=[]; continue
        if cur is not None: funcs[cur].append(line.rstrip('\n'))
    return funcs
def norm(lines):
    out=[]
    for l in lines:
        if re.match(r'^\s*/\* 0x[0-9a-f]+ \*/\s*$',l): continue
        l=re.sub(r'/\*[0-9a-f]{4,}\*/','',l); l=re.sub(r'/\* 0x[0-9a-f]+ \*/','',l).strip()
        if l and not l.startswith('.') and not l.startswith('//'): out.append(l)
    return out
def census(lines):
    ops=collections.Counter()
    for l in norm(lines):
        m=re.match(r'(?:@!?P\d+\s+)?([A-Z0-9_.]+)',l)
        if m: ops[m.group(1).split('.')[0]]+=1
    return ops
def key(name):  # strip the per-TU anonymous namespace hash
    name=re.sub(r'__nv_static_\d+__[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]+_\d+','ANON',name)
    return re.sub(r'_GLOBAL__N__[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]+_\d+','ANONNS',name)
cmd=sys.argv[1]
if cmd=='diff':
    a=split(sys.argv[2]); b=split(sys.argv[3])
    ka={key(k):k for k in a}; kb={key(k):k for k in b}
    same=diff=missing=0; tot=0
    for k in ka:
        if k not in kb: missing+=1; print('MISSING in new:',k[:100]); continue
        na=norm(a[ka[k]]); nb=norm(b[kb[k]]); tot+=len(na)
        if na==nb: same+=1
        else: diff+=1; print('DIFFERENT:',k[:120],len(na),len(nb))
    new=[k for k in kb if k not in ka]
    print('functions in head: %d, identical: %d, different: %d, missing: %d, head instrs: %d, head-only-in-new: %d'%(len(ka),same,diff,missing,tot,len(new)))
    for k in new: print('  NEW:',k[:140])
elif cmd=='census':
    f=split(sys.argv[2]); pat=sys.argv[3]
    for k in f:
        if pat in k:
            c=census(f[k]); n=sum(c.values())
            print(k[:150]); print('   instrs=%d '%n+' '.join('%s=%d'%(o,c[o]) for o in ['DADD','SHFL','LDG','STG','ATOMG','BRA','VOTE','FLO','BREV','EXIT','BSSY','BSYNC','WARPSYNC']))

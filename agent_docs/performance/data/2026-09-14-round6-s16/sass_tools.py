#!/usr/bin/env python3
"""s16 SASS census: split a `cuobjdump -sass` dump into MAS kernels, normalise, count, diff.

  sass_tools.py census <dump.sass>              -> per-kernel instruction census + regs from a matching ptxas-free method
  sass_tools.py diff <head.sass> <new.sass>     -> normalised body identity for the untouched kernels
"""
import re, sys, collections

KEYS = {
    'build_multi_level_R_kernel': 'build_multi_level_R_kernel',
    'rowdot2<16,0,0>': 'schwarz_local_solve_rowdot2_kernelILi16ELb0ELb0',
    'rowdot(s12)': 'schwarz_local_solve_rowdot_kernel',
    'collect_final_Z_kernel': 'collect_final_Z_kernel',
    'schwarz_local_solve_kernel(old atomic)': 'schwarz_local_solve_kernelE',
    'fused_R<true,0 push-first>': 'fused_R_kernelILb1ELi0',
    'fused_R<true,1 push-after>': 'fused_R_kernelILb1ELi1',
    'fused_R<true,2 prefetch>': 'fused_R_kernelILb1ELi2',
    'fused_R<false,0 coarse>': 'fused_R_kernelILb0ELi0',
}

def split(path):
    funcs = {}
    cur = None
    for line in open(path, errors='replace'):
        m = re.match(r'\s*Function : (.*)', line)
        if m:
            cur = m.group(1).strip()
            funcs[cur] = []
            continue
        if cur is not None:
            funcs[cur].append(line.rstrip('\n'))
    return funcs

def find(funcs, key):
    hits = [k for k in funcs if key in k]
    return hits

def norm(lines):
    out = []
    for l in lines:
        if re.match(r'^\s*/\* 0x[0-9a-f]+ \*/\s*$', l):
            continue  # encoding-only line
        l = re.sub(r'/\*[0-9a-f]{4,}\*/', '', l)   # address
        l = re.sub(r'/\* 0x[0-9a-f]+ \*/', '', l)   # encoding
        l = l.strip()
        if l and not l.startswith('.') and not l.startswith('//'):
            out.append(l)
    return out

def census(lines):
    n = norm(lines)
    ops = collections.Counter()
    for l in n:
        m = re.match(r'(?:@!?P\d+\s+)?([A-Z0-9_.]+)', l)
        if m:
            ops[m.group(1).split('.')[0]] += 1
    tot = len(n)
    def c(*names):
        return sum(v for k, v in ops.items() if k in names)
    return dict(instrs=tot, LDG=c('LDG'), STG=c('STG'), LDS=c('LDS'), STS=c('STS'), FMUL=c('FMUL'), FFMA=c('FFMA'), FADD=c('FADD'),
                BRA=c('BRA'), BAR=c('BAR'), SHFL=c('SHFL'), RED=c('RED'), ATOM=c('ATOM', 'ATOMS', 'ATOMG'), F2F=c('F2F'), EXIT=c('EXIT'))

def main():
    cmd = sys.argv[1]
    if cmd == 'census':
        funcs = split(sys.argv[2])
        for label, key in KEYS.items():
            for k in find(funcs, key):
                print('%-40s %s' % (label, ' '.join('%s=%d' % kv for kv in census(funcs[k]).items())))
    elif cmd == 'diff':
        a = split(sys.argv[2]); b = split(sys.argv[3])
        for label, key in KEYS.items():
            ka = find(a, key); kb = find(b, key)
            if not ka or not kb:
                print('%-40s %s' % (label, 'MISSING in ' + ('head' if not ka else 'new')))
                continue
            na = norm(a[ka[0]]); nb = norm(b[kb[0]])
            same = na == nb
            print('%-40s %s (%d vs %d instrs)' % (label, 'IDENTICAL' if same else 'DIFFERENT', len(na), len(nb)))
            if not same:
                import difflib
                for l in list(difflib.unified_diff(na, nb, lineterm=''))[:20]:
                    print('   ', l)

main()

#!/usr/bin/env python3
"""Record-level comparison of racecheck / initcheck logs between arms.
A racecheck record = (kind, kernel, write PC offset, read PC offset, hazard count).
An initcheck record = (kernel, call site).  Prints, per subject, whether arm A's
multiset of records equals arm B's, optionally ignoring one kernel."""
import re, sys, os, glob, collections
LOGS='/workspace/output/round6/sanitizers/logs'
def kname(s):
    s = re.sub(r'^(void|int|bool)\s+', '', s.strip())
    m = re.match(r'([\w:<>, ]+?)(?:<[^(]*>)?\(', s); n = (m.group(1) if m else s[:60]).split('::')[-1]
    return n
def records(path, tool):
    L = open(path, errors='replace').read().splitlines(); out = collections.Counter()
    for i, l in enumerate(L):
        if tool == 'racecheck':
            m = re.match(r'========= (Warning|Error): Race reported between (\w+) access at (.*?)\+(0x[0-9a-f]+)$', l)
            if m:
                n = L[i+1]; m2 = re.search(r'and (\w+) access at (.*?)\+(0x[0-9a-f]+) \[(\d+) hazards?\]', n)
                out[(m.group(1), kname(m.group(3)), m.group(2), m.group(4), m2.group(1) if m2 else '?', m2.group(3) if m2 else '?', int(m2.group(4)) if m2 else -1)] += 1
        elif tool == 'initcheck':
            if l.startswith('========= Uninitialized'):
                at = kname(re.sub(r'^=+\s*at\s*', '', L[i+1])); site=''
                for j in range(i+2, min(i+40, len(L))):
                    if 'Host Frame' in L[j] and 'libuipc' in L[j] and not re.search(r'cub::|thrust::|cudaLaunch', L[j]):
                        site = kname(re.sub(r'^=+\s*Host Frame:\s*', '', L[j])); break
                out[(at, site)] += 1
    return out
def main(a, b, tool, ignore=None, nokernel=False):
    A = {os.path.basename(f).split('__')[2][:-4]: f for f in glob.glob(f'{LOGS}/{a}__{tool}__*.txt')}
    B = {os.path.basename(f).split('__')[2][:-4]: f for f in glob.glob(f'{LOGS}/{b}__{tool}__*.txt')}
    for subj in sorted(set(A) & set(B)):
        ra, rb = records(A[subj], tool), records(B[subj], tool)
        if ignore:
            ra = collections.Counter({k: v for k, v in ra.items() if ignore not in k[1]})
            rb = collections.Counter({k: v for k, v in rb.items() if ignore not in k[1]})
        if nokernel:  # racecheck's kernel-name attribution can lag a record (seen on base mas-bunny): key on offsets+hazards only
            ra = collections.Counter({(k[0], k[3], k[5], k[6]): v for k, v in ra.items()}) if tool == 'racecheck' else ra
            rb = collections.Counter({(k[0], k[3], k[5], k[6]): v for k, v in rb.items()}) if tool == 'racecheck' else rb
        same = ra == rb
        print(f'{tool:9} {subj:32} {a}:{sum(ra.values()):5d} records  {b}:{sum(rb.values()):5d}  ' + ('IDENTICAL record-for-record' if same else 'DIFFER') + (f'  (ignoring {ignore})' if ignore else ''))
        if not same:
            for k in sorted(set(ra) | set(rb)):
                if ra[k] != rb[k]: print('    ', a, ra[k], b, rb[k], k)
if __name__ == '__main__':
    args=[x for x in sys.argv[1:] if x!='--nokernel']
    main(args[0], args[1], args[2], (args[3] if len(args) > 3 and args[3] != '-' else None), '--nokernel' in sys.argv)

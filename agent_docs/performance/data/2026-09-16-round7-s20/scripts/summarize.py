#!/usr/bin/env python3
"""Digest every compute-sanitizer log in logs/ into one table + per-kernel breakdowns."""
import re, sys, os, glob, collections
LOGS = sys.argv[1] if len(sys.argv) > 1 else '/workspace/output/round7/s20/logs'
KSHORT = re.compile(r'(?:[\w<>:,]*::)?([A-Za-z_][\w]*?)(?:_kernel)?(?:<[^(]*>)?\(')
def kshort(s):
    s = s.strip()
    # drop leading 'void ' and namespaces, keep the function name incl. template args trimmed
    s = re.sub(r'^(void|int|bool)\s+', '', s)
    m = re.match(r'([\w:<>, ]+?)(?:<[^(]*>)?\(', s)
    name = m.group(1) if m else s[:60]
    name = name.split('::')[-1]
    return name[:70]
rows = []
for f in sorted(glob.glob(os.path.join(LOGS, '*.txt'))):
    tag = os.path.basename(f)[:-4]
    parts = tag.split('__')
    if len(parts) != 3: continue
    arm, tool, subj = parts
    txt = open(f, errors='replace').read()
    lines = txt.splitlines()
    summ = [l for l in lines if re.search(r'(ERROR|RACECHECK|SYNCCHECK) SUMMARY', l) and 'not printed' not in l]
    summ = summ[-1].replace('=========', '').strip() if summ else 'NO SUMMARY'
    tests = [l for l in lines if 'All tests passed' in l or 'test cases:' in l or re.search(r'\bFAILED\b', l)]
    per = collections.Counter()
    if tool == 'racecheck':
        for i, l in enumerate(lines):
            m = re.match(r'========= (Warning|Error): Race reported between (\w+) access at (.*)', l)
            if m:
                nxt = lines[i+1] if i+1 < len(lines) else ''
                hz = re.search(r'\[(\d+) hazards?\]', nxt)
                per[(m.group(1), kshort(m.group(3)))] += 1
    elif tool == 'initcheck':
        for i, l in enumerate(lines):
            if l.startswith('========= Uninitialized'):
                at = lines[i+1] if i+1 < len(lines) else ''
                # first non-cub/thrust host frame gives the call site
                site = ''
                for j in range(i+2, min(i+40, len(lines))):
                    if 'Host Frame' in lines[j] and 'libuipc' in lines[j] and not re.search(r'cub::|thrust::|cudaLaunch', lines[j]):
                        site = kshort(re.sub(r'^=+\s*Host Frame:\s*', '', lines[j])); break
                    if lines[j].startswith('========= ') and lines[j].strip() == '=========': break
                per[(kshort(re.sub(r'^=+\s*at\s*', '', at)), site)] += 1
    elif tool == 'memcheck':
        for i, l in enumerate(lines):
            m = re.match(r'========= (Invalid \S+ of size \d+ bytes|Program hit .*|.*[Ll]eak.*)', l)
            if m:
                at = lines[i+1] if i+1 < len(lines) else ''
                per[(m.group(1)[:60], kshort(re.sub(r'^=+\s*at\s*', '', at)))] += 1
    elif tool == 'synccheck':
        for i, l in enumerate(lines):
            m = re.match(r'========= (Barrier error|.*error.*)', l)
            if m and 'SUMMARY' not in l:
                at = lines[i+1] if i+1 < len(lines) else ''
                per[(m.group(1)[:60], kshort(re.sub(r'^=+\s*at\s*', '', at)))] += 1
    fr = re.search(r'TOTAL frames=(\d+)', txt)
    if fr: tests.append(f'frames={fr.group(1)}')
    elif not tests: tests.append('!! NO frames/tests line')
    rows.append((arm, tool, subj, summ, tests[-1].strip() if tests else '', per))
# table
print('| arm | tool | subject | summary | tests |')
print('|---|---|---|---|---|')
for arm, tool, subj, summ, t, per in rows:
    print(f'| {arm} | {tool} | {subj} | {summ} | {t} |')
print()
for arm, tool, subj, summ, t, per in rows:
    if per:
        print(f'## {arm} / {tool} / {subj}  ({sum(per.values())} displayed)')
        for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f'  {v:6d}  {" | ".join(k)}')

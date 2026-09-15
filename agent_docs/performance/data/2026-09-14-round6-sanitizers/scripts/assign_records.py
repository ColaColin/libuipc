#!/usr/bin/env python3
"""Walk a sanitized uipc_test_sim_case log (-d yes) and assign each sanitizer record to the test case
whose duration line follows it. Prints per-case record counts and, with --selector, a Catch2 selector
of the cases that reported anything (or 'NONE')."""
import re, sys, collections
path = sys.argv[1]; sel = '--selector' in sys.argv
per = collections.Counter(); perk = collections.defaultdict(collections.Counter); pending = []
for l in open(path, errors='replace'):
    m = re.match(r'========= (?:Warning|Error): Race reported between \w+ access at (.*?)(?:\+0x[0-9a-f]+)?\s*$', l)
    if m: pending.append(('race', re.sub(r'\(.*', '', m.group(1)).split('::')[-1])); continue
    if l.startswith('========= Uninitialized') or re.match(r'========= (Invalid|Program hit|Barrier error)', l):
        pending.append(('other', l.strip()[10:70])); continue
    m = re.match(r'([0-9.]+) s: (\d+_[A-Za-z]\S*)\s*$', l)
    if m:
        c = m.group(2)
        for kind, k in pending: per[c] += 1; perk[c][k] += 1
        pending = []
if sel:
    print(','.join(sorted(per)) if per else 'NONE')
else:
    for c in sorted(per, key=lambda x: (int(x.split('_')[0]), x)):
        print(f'{per[c]:6d}  {c:42} ' + '; '.join(f'{k} x{v}' for k, v in perk[c].most_common()))
    print(f'-- {sum(per.values())} records in {len(per)} cases; {len(pending)} unassigned trailing records')

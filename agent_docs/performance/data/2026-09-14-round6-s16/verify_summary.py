#!/usr/bin/env python3
"""Summarise [MAS apply verify] lines: per arr, count, max of max|diff|, max rel, max of max|ref|."""
import re, sys, collections
for path in sys.argv[1:]:
    d = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    mode = None
    for l in open(path, errors='replace'):
        m = re.search(r'mode=(\d+) arr=(\w+) n=(\d+) max\|diff\|=([0-9.e+-]+) max\|ref\|=([0-9.e+-]+) rel=([0-9.e+-]+)', l)
        if not m: continue
        mode = m.group(1); arr = m.group(2)
        e = d[arr]; e[0] += 1; e[1] = max(e[1], float(m.group(4))); e[2] = max(e[2], float(m.group(6))); e[3] = max(e[3], float(m.group(5)))
    print('%s mode=%s' % (path.split('/')[-1], mode))
    for arr, (n, dmax, rmax, refmax) in d.items():
        print('   %-9s applies=%4d  max|diff|=%.6e  max rel=%.6e  max|ref|=%.6e' % (arr, n, dmax, rmax, refmax))

import re, sys
from collections import defaultdict

def parse(path):
    funs = {}
    sym = None
    for line in open(path):
        m = re.match(r'\s*Function (\S+?):', line)
        if m:
            name = m.group(1)
            # normalize: drop runs that are pure hex >=8 chars and long digit counters
            n = re.sub(r'[0-9a-f]{8,}', 'H', name)
            n = re.sub(r'_2\d{6,}', '_C', n)
            sym = n
            funs[n] = None
        elif sym and re.match(r'\s+REG:', line):
            fm = re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', line)
            cm = re.search(r'CONSTANT\[2\]:(\d+)', line)
            c0 = re.search(r'CONSTANT\[0\]:(\d+)', line)
            if fm and funs.get(sym, 'x') is None:
                funs[sym] = (int(fm.group(1)), int(fm.group(2)), int(fm.group(3)), int(fm.group(4)),
                             int(cm.group(1)) if cm else -1, int(c0.group(1)) if c0 else -1)
    return funs

main = parse(sys.argv[1]); branch = parse(sys.argv[2])
changed, onlym, onlyb = [], [], []
same = 0
for k in sorted(set(main) | set(branch)):
    if k in main and k in branch:
        if main[k] == branch[k]:
            same += 1
        else:
            changed.append((k, main[k], branch[k]))
    elif k in main:
        onlym.append(k)
    else:
        onlyb.append(k)
print(f"normalized symbols: identical={same} changed={len(changed)} only-main={len(onlym)} only-branch={len(onlyb)}")
print("\n== CHANGED (same name, different figures) ==")
for k, a, b in changed:
    short = re.search(r'(\w+_cu)', k)
    print(f"  [{short.group(1) if short else k[:40]}] {a} -> {b}")
print("\n== only-branch (new symbols) ==")
for k in onlyb[:40]:
    short = re.search(r'(\w+_cu)', k)
    print(f"  [{short.group(1) if short else '?'}] ...{k[-70:]}")
print("\n== only-main (removed symbols) ==")
for k in onlym[:40]:
    short = re.search(r'(\w+_cu)', k)
    print(f"  [{short.group(1) if short else '?'}] ...{k[-70:]}")

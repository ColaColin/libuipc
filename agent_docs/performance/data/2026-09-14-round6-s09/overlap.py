"""s09: measure the overlap actually achieved between contact part 1 and part 2,
from the nsys per-launch timeline (cuda_gpu_trace), not from inference."""
import csv, re, sys, statistics
f = sys.argv[1]
rows = []
hdr = None
for r in csv.DictReader(open(f)):
    n = r.get('Name') or ''
    m = re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([012]),', n)
    if not m:
        continue
    start = float(r['Start (ns)']); dur = float(r['Duration (ns)'])
    strm = r.get('Strm') or r.get('Stream')
    rows.append((start, start+dur, 'p'+m.group(1), strm))
rows.sort()
print(f"{len(rows)} contact-assembly launches; streams seen: "
      f"{sorted({(k,s) for _,_,k,s in rows})}")
# pair consecutive p1/p2 launches
pairs = []
i = 0
while i < len(rows)-1:
    a, b = rows[i], rows[i+1]
    if {a[2], b[2]} == {'p1','p2'}:
        p1 = a if a[2]=='p1' else b
        p2 = b if a[2]=='p1' else a
        ov = max(0.0, min(p1[1],p2[1]) - max(p1[0],p2[0]))
        union = max(p1[1],p2[1]) - min(p1[0],p2[0])
        pairs.append((p1[1]-p1[0], p2[1]-p2[0], ov, union))
        i += 2
    else:
        i += 1
if not pairs:
    solo = [b-a for a,b,k,s in rows]
    print(f"single fused launches: n={len(solo)} mean {statistics.mean(solo)/1000:.1f} us")
    sys.exit()
d1 = statistics.mean(p[0] for p in pairs)/1000
d2 = statistics.mean(p[1] for p in pairs)/1000
ov = statistics.mean(p[2] for p in pairs)/1000
un = statistics.mean(p[3] for p in pairs)/1000
print(f"n pairs = {len(pairs)}")
print(f"  part1 {d1:8.1f} us   part2 {d2:8.1f} us   sum {d1+d2:8.1f} us")
print(f"  overlap {ov:8.1f} us  = {100*ov/d2:5.1f} % of part 2, {100*ov/d1:5.1f} % of part 1")
print(f"  UNION (critical path of the contact assembly) {un:8.1f} us")
print(f"  serial equivalent would be {d1+d2:8.1f} us -> the split saves {100*(1-un/(d1+d2)):5.2f} %"
      f" of the two-launch cost")
print(f"  union / max(part1,part2) = {un/max(d1,d2):5.3f}  (1.000 = perfectly hidden)")

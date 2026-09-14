"""s10: how much of the ABD body-local G/H actually runs inside the contact
assembly's shadow, from the nsys per-launch timeline (cuda_gpu_trace).

Also reports the assembly critical path: the span from the first contact
part-1 launch of a Newton iteration to the start of assemble_kinetic_shape_k1
(the first kernel that consumes the prepass output)."""
import csv, re, sys, statistics

f = sys.argv[1]
ev = []
for r in csv.DictReader(open(f)):
    n = r.get('Name') or ''
    if not r.get('Start (ns)'):
        continue
    ev.append((int(r['Start (ns)']), int(r['Duration (ns)']), r.get('Strm'), n))
ev.sort()

def tag(n):
    if re.search(r'do_assemble_kernel<\(bool\)0, \(int\)1,', n): return 'p1'
    if re.search(r'do_assemble_kernel<\(bool\)0, \(int\)2,', n): return 'p2'
    if re.search(r'do_assemble_kernel<\(bool\)0,\s*\(int\)0,', n): return 'p0'
    if 'ortho_potential_compute_gradient_hessian' in n: return 'ortho'
    if 'arap_compute_gradient_hessian' in n: return 'arap'
    if 'kinetic_compute_gradient_hessian' in n: return 'kin'
    if 'assemble_kinetic_shape_k1' in n: return 'ks1'
    if 'abd_diag_preconditioner_do_assemble' in n: return 'diag'
    return None

idx = {}
for i,(s,d,st,n) in enumerate(ev):
    t = tag(n)
    if t: idx.setdefault(t, []).append(i)

print(f"# {f}")
for t in ('p1','p2','p0','ortho','arap','kin','ks1','diag'):
    if t in idx:
        tot = sum(ev[i][1] for i in idx[t])
        strm = sorted({ev[i][2] for i in idx[t]})
        print(f"  {t:6s} n={len(idx[t]):5d} total {tot/1e6:8.1f} ms  per-launch {tot/len(idx[t])/1000:8.1f} us  streams {strm}")

# busy union / duration sum
iv = sorted((e[0], e[0]+e[1]) for e in ev)
u = 0; cs, ce = iv[0]
for s,e in iv[1:]:
    if s > ce: u += ce-cs; cs, ce = s, e
    else: ce = max(ce, e)
u += ce-cs
span = iv[-1][1]-iv[0][0]
dsum = sum(e[1] for e in ev)
print(f"  trace span {span/1e6:8.1f} ms   busy union {u/1e6:8.1f} ms   duration sum {dsum/1e6:8.1f} ms"
      f"   sum/union {dsum/u:5.3f}")

# hidden fraction of the ABD prepass kernels inside part-1 windows
p1iv = [(ev[i][0], ev[i][0]+ev[i][1]) for i in idx.get('p1', [])]
p1iv.sort()
def hidden(tagname):
    if tagname not in idx: return None
    tot = 0; hid = 0
    import bisect
    starts = [a for a,_ in p1iv]
    for i in idx[tagname]:
        s, d = ev[i][0], ev[i][1]; e = s+d
        tot += d
        j = bisect.bisect_right(starts, e) - 1
        for k in range(max(0, j-2), min(len(p1iv), j+3)):
            a,b = p1iv[k]
            ov = min(e,b) - max(s,a)
            if ov > 0: hid += ov
    return tot, hid
for t in ('ortho','arap','kin'):
    r = hidden(t)
    if r:
        tot, hid = r
        print(f"  {t:6s} hidden inside contact part 1: {hid/1e6:8.1f} ms of {tot/1e6:8.1f} ms = {100*hid/tot:5.1f} %")

# empty shadow of part 1
sh_tot = 0; sh_cov = 0
for i in idx.get('p1', []):
    s, d = ev[i][0], ev[i][1]; e = s+d
    cov = []
    for j in range(max(0,i-60), min(len(ev), i+300)):
        if j == i: continue
        s2,d2 = ev[j][0], ev[j][1]; e2 = s2+d2
        a=max(s,s2); b=min(e,e2)
        if b>a: cov.append((a,b))
    cov.sort(); tot=0
    if cov:
        cs,ce = cov[0]
        for a,b in cov[1:]:
            if a>ce: tot += ce-cs; cs,ce=a,b
            else: ce=max(ce,b)
        tot += ce-cs
    sh_tot += d; sh_cov += tot
if sh_tot:
    print(f"  part 1 window {sh_tot/1e6:8.1f} ms, covered by other kernels {sh_cov/1e6:8.1f} ms,"
          f" EMPTY SHADOW {(sh_tot-sh_cov)/1e6:8.1f} ms")

# assembly critical path: p1 start -> ks1 start
if 'ks1' in idx and 'p1' in idx:
    ks1s = sorted(ev[i][0] for i in idx['ks1'])
    p1s  = sorted(ev[i][0] for i in idx['p1'])
    import bisect
    gaps = []
    for a in p1s:
        j = bisect.bisect_right(ks1s, a)
        if j < len(ks1s): gaps.append(ks1s[j]-a)
    if gaps:
        print(f"  contact part1 start -> next kinetic_shape_k1 start: median {statistics.median(gaps)/1000:8.1f} us"
              f"  mean {statistics.mean(gaps)/1000:8.1f} us  (n={len(gaps)})")

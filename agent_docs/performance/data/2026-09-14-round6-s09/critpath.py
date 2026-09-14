"""s09: the contact-assembly critical path per assemble call, per arm.
 a2's union is measured directly from the cuda_gpu_trace timeline (union/max ratio)
 and applied to the full-run kern_sum per-launch time of part 1, because the trace runs
 are separate (and, for c2/tum, windowed)."""
import csv,glob,re,statistics
UNION_OVER_MAX={'rwb':1.006,'cwc':1.035,'c2':1.012,'tum':1.005}   # measured, overlap.py
HID={'rwb':98.3,'cwc':91.7,'c2':96.8,'tum':98.1}
def perlaunch(sc,arm):
    out={}
    for f in sorted(glob.glob(f'nsys/{sc}_{arm}_r*_cuda_gpu_kern_sum.csv')):
        for r in csv.DictReader(open(f)):
            m=re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([012]),',r['Name'])
            if m:
                k='p'+m.group(1)
                out.setdefault(k,[]).append(int(r['Total Time (ns)'])/int(r['Instances'])/1000)
    return {k:statistics.mean(v) for k,v in out.items()}
print("contact assembly, us per assemble call (nsys full runs, n=3/arm)")
print(f"{'scene':6s} {'a2 part1':>9s} {'a2 part2':>9s} {'hidden%':>8s} {'a2 UNION':>9s} "
      f"{'a0 fused':>9s} {'vs a2':>8s} {'a1 serial':>10s} {'vs a2':>8s}")
for sc in ['rwb','cwc','c2','tum']:
    A2=perlaunch(sc,'a2'); A1=perlaunch(sc,'a1'); A0=perlaunch(sc,'a0')
    union=A2['p1']*UNION_OVER_MAX[sc]
    ser=A1['p1']+A1['p2']
    print(f"{sc:6s} {A2['p1']:9.1f} {A2['p2']:9.1f} {HID[sc]:8.1f} {union:9.1f} "
          f"{A0['p0']:9.1f} {100*(A0['p0']/union-1):+8.2f} {ser:10.1f} {100*(ser/union-1):+8.2f}")
    print(f"       (serial part1 {A1['p1']:.1f} + part2 {A1['p2']:.1f}; part1 concurrency cost "
          f"{100*(A2['p1']/A1['p1']-1):+.2f} % while hiding {A1['p2']:.0f} us of part 2)")

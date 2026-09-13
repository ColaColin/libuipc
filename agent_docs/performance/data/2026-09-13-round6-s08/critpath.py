import csv, glob, re, statistics, sys
for SC,F in [('cwc',100),('rwb',120),('c2',250),('tum',180)]:
    for arm in ['b0','b1']:
        ks=[];wl=[];p1=[];p2=[]
        for f in sorted(glob.glob(f'nsys/{SC}_{arm}_r*_cuda_gpu_kern_sum.csv')):
            tot=0; a=b=0
            for r in csv.DictReader(open(f)):
                t=int(r['Total Time (ns)']); tot+=t
                m=re.search(r'do_assemble_kernel<\(bool\)0, \(int\)([12]),', r['Name'])
                if m:
                    if m.group(1)=='1': a=t
                    else: b=t
            log=f.replace('_cuda_gpu_kern_sum.csv','.run.log')
            mm=re.search(r'TOTAL frames=(\d+) mean=([0-9.]+)ms', open(log).read())
            if not mm: continue
            ks.append(tot/1e6); wl.append(float(mm.group(2))*int(mm.group(1)))
            p1.append(a/1e6); p2.append(b/1e6)
        if not ks: continue
        k=statistics.mean(ks); w=statistics.mean(wl)
        print(f"{SC:4s} {arm} kernel-sum {k:8.0f} ms  wall(nsys) {w:8.0f} ms  sum/wall {k/w:6.3f} | "
              f"part1 {statistics.mean(p1):7.0f} ms ({100*statistics.mean(p1)/k:5.2f}% of kernel sum)  "
              f"part2 {statistics.mean(p2):7.0f} ms ({100*statistics.mean(p2)/k:5.2f}%)  "
              f"critical stream = part {'1' if statistics.mean(p1)>statistics.mean(p2) else '2'}")

#!/usr/bin/env python3
"""s14: mechanism-only analysis of a cuda_gpu_trace csv (s11 doctrine: never a cross-arm level).
 - duration-weighted first-wave occupancy census (s12's instrument), whole trace + contact window
 - per Newton iteration: contact part 1 / part 2 geometry, what overlaps part 1, SNH start offset
"""
import csv, sys, statistics as st
from collections import defaultdict
SMS, REGFILE, MAXTHR, MAXBLK, MAXWARP = 40, 65536, 2048, 32, 32

def resident_blocks(reg, thr):
    if thr <= 0: return 1
    warps = (thr + 31) // 32
    reg_alloc = ((reg * 32 + 255) // 256) * 256 * warps  # 256-reg allocation granularity per warp
    if reg_alloc <= 0: reg_alloc = 1
    return max(0, min(REGFILE // reg_alloc, MAXTHR // thr, MAXBLK, MAXWARP // max(1, warps)))

rows = []
with open(sys.argv[1]) as f:
    r = csv.DictReader(f)
    for d in r:
        try:
            s = int(d['Start (ns)']); dur = int(d['Duration (ns)'])
            gx, gy, gz = int(d['GrdX'] or 1), int(d['GrdY'] or 1), int(d['GrdZ'] or 1)
            bx, by, bz = int(d['BlkX'] or 1), int(d['BlkY'] or 1), int(d['BlkZ'] or 1)
            reg = int(d['Reg/Trd'] or 0)
        except ValueError:
            continue
        name = d['Name']
        if name.startswith('[CUDA mem'): continue
        rows.append(dict(s=s, e=s+dur, dur=dur, grid=gx*gy*gz, thr=bx*by*bz, reg=reg,
                         strm=d['Strm'], name=name))
rows.sort(key=lambda x: x['s'])
def short(n): return n[:70]
def is_part1(n): return 'do_assemble_kernel<(bool)0, (int)1' in n
def is_part2(n): return 'do_assemble_kernel<(bool)0, (int)2' in n
def is_snh(n): return 'StableNeoHookean3D' in n and 'gradient_hessian' in n.lower() or ('StableNeoHookean3D' in n and 'do_compute' in n and 'energy' not in n)

# ---- census
tot = sum(r['dur'] for r in rows); wsum = 0.0; idle_by = defaultdict(float); dur_by = defaultdict(float); n_by = defaultdict(int)
for r in rows:
    rb = resident_blocks(r['reg'], r['thr']); cap = SMS * max(rb, 1)
    fill = min(1.0, r['grid'] / cap) if cap else 1.0
    r['fill'] = fill; wsum += r['dur'] * fill
    idle_by[short(r['name'])] += r['dur'] * (1 - fill); dur_by[short(r['name'])] += r['dur']; n_by[short(r['name'])] += 1
print(f"trace: kernel duration sum {tot/1e6:.1f} ms, occupancy-weighted {wsum/1e6:.1f} ms => mean fill {100*wsum/tot:.1f}%")
print(f"under-filled time = {(tot-wsum)/1e6:.1f} ms = {100*(tot-wsum)/tot:.1f}% of kernel time")
print("biggest under-filled kernels:")
for k, v in sorted(idle_by.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  idle {v/1e6:8.1f}ms  tot {dur_by[k]/1e6:8.1f}ms  fill {100*(1-v/dur_by[k]):5.1f}%  n={n_by[k]:6d}  {k}")

# ---- contact windows
p1 = [r for r in rows if is_part1(r['name'])]
p2 = [r for r in rows if is_part2(r['name'])]
snh = [r for r in rows if 'StableNeoHookean3D' in r['name'] and 'energy' not in r['name']]
print(f"\ncontact part 1: n={len(p1)} mean {st.mean(r['dur'] for r in p1)/1e3:.1f} us, median grid {st.median(r['grid'] for r in p1)} x {p1[0]['thr']} thr @ {p1[0]['reg']} reg, stream(s) {sorted(set(r['strm'] for r in p1))}" if p1 else "no part 1")
print(f"contact part 2: n={len(p2)} mean {st.mean(r['dur'] for r in p2)/1e3:.1f} us, median grid {st.median(r['grid'] for r in p2)} x {p2[0]['thr']} thr @ {p2[0]['reg']} reg, stream(s) {sorted(set(r['strm'] for r in p2))}" if p2 else "no part 2")
snh_names = sorted(set(short(r['name']) for r in snh))
print("SNH kernels:", snh_names)
if p1 and snh:
    # for each part1: first SNH launch that starts after part1's start
    import bisect
    snh_starts = [r['s'] for r in snh]
    offs = []; hidden = []; inside_snh = []; idle_win = 0.0; win_dur = 0.0; p1_slow = []
    for w in p1:
        i = bisect.bisect_left(snh_starts, w['s'])
        if i < len(snh):
            offs.append((snh[i]['s'] - w['e']) / 1e3)
        # overlap of part1's window with any kernel on another stream that is not part2
        cover = 0.0; cov_snh = 0.0
        for r in rows:
            if r['s'] >= w['e']: break
            if r['e'] <= w['s'] or r is w or is_part2(r['name']): continue
            o = min(r['e'], w['e']) - max(r['s'], w['s'])
            if o > 0:
                cover += o
                if 'StableNeoHookean3D' in r['name']: cov_snh += o
        hidden.append(cover / w['dur']); inside_snh.append(cov_snh / w['dur'])
        # idle-machine inside the window: sum over launches inside of dur*(1-fill), clipped
        for r in rows:
            if r['s'] >= w['e']: break
            if r['e'] <= w['s']: continue
            o = min(r['e'], w['e']) - max(r['s'], w['s'])
            if o > 0: idle_win += o * (1 - r['fill']); win_dur += o
    print(f"SNH start - part1 end: mean {st.mean(offs):.1f} us, median {st.median(offs):.1f} us, min {min(offs):.1f}, frac<0 (SNH starts before part 1 ends) = {sum(1 for o in offs if o < 0)/len(offs):.3f}")
    print(f"part 1 window covered by non-part-2 kernels: mean {100*st.mean(hidden):.1f}% ; by SNH: {100*st.mean(inside_snh):.1f}%")
    print(f"idle-machine inside part-1 windows (dur x (1-fill), naive per-launch): {idle_win/1e6:.1f} ms over {len(p1)} windows = {idle_win/len(p1)/1e3:.0f} us/iter")
    print(f"SNH per launch: mean {st.mean(r['dur'] for r in snh)/1e3:.1f} us, n={len(snh)}, grid {st.median(r['grid'] for r in snh)} x {snh[0]['thr']} @ {snh[0]['reg']} reg")

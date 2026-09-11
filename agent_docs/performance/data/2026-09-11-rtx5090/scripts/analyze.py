#!/usr/bin/env python3
"""Summarise output/benchmark-runs records from several trees. Usage: analyze.py raw/<tree>/... -> markdown to stdout."""
import json, sys, glob, statistics as st, math, os
from collections import defaultdict
ROOT = sys.argv[1] if len(sys.argv) > 1 else 'raw'
TREES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['head', 'base', 'dahl']
REF = {  # 2026-09-01 doc: reference = median of three run means (ms/frame), Newton/frame, PCG/frame
    'rigid-wrecking-balls': (129.5, 3.95, 107.3), 'stiff-gipc-case2': (201.1, 6.64, 246.8),
    'mas-bunny': (60.2, 4.67, 358.8), 'cube-wall-cloth': (125.6, 5.13, 200.1)}
ORDER = ['rigid-wrecking-balls', 'stiff-gipc-case2', 'mas-bunny', 'cube-wall-cloth']
runs = defaultdict(list)  # (tree, bench) -> list of record dicts
for tree in TREES:
    for f in sorted(glob.glob(f'{ROOT}/{tree}/*-*Z.json')):
        d = json.load(open(f))
        rb = d.get('reportedBenchmark')
        if d.get('returnCode') != 0 or not rb:
            print(f'SKIP {f} rc={d.get("returnCode")}', file=sys.stderr); continue
        fm = rb['frame_ms']; fs = rb['frame_stats']
        if d['frames'] != len(fm):
            continue  # quick runs etc.
        r = dict(file=os.path.basename(f), frames=len(fm), total_s=sum(fm)/1000, wall_s=d['durationSeconds'],
                 mean=st.mean(fm), median=st.median(fm), p95=sorted(fm)[min(len(fm)-1, int(round(0.95*(len(fm)-1))))],
                 newton=sum(x.get('newton_iterations', 0) for x in fs)/len(fs),
                 ls=sum(x.get('line_search_trials', 0) for x in fs)/len(fs),
                 pcg=sum(x.get('linear_solver_iterations', 0) for x in fs)/len(fs),
                 converged=all(x.get('converged', True) for x in fs), completed=all(x.get('completed', True) for x in fs),
                 obs=rb.get('observables', {}), mem=(d.get('gpuMemory') or {}).get('peakDeltaMiB', [None])[0],
                 memabs=(d.get('gpuMemory') or {}).get('peakTotalMiB', [None])[0],
                 commit=d['revisions']['libuipc']['commit'][:8], runId=d['runId'])
        runs[(tree, d['benchmark'])].append(r)
def fmt(v, p=1): return '-' if v is None else f'{v:.{p}f}'
def agg(rs, k): 
    v = [r[k] for r in rs if r[k] is not None]; return (st.mean(v), st.median(v), min(v), max(v)) if v else (None,)*4
print('## Per-run table\n')
print('| bench | tree | run | frames | sum frame_ms (s) | process wall (s) | mean ms/f | median | p95 | Newton/f | LS/f | PCG/f | peak mem delta MiB | converged |')
print('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for b in ORDER:
    for tree in TREES:
        for r in runs.get((tree, b), []):
            print(f"| {b} | {tree} ({r['commit']}) | {r['runId']} | {r['frames']} | {fmt(r['total_s'],2)} | {fmt(r['wall_s'],1)} | {fmt(r['mean'],2)} | {fmt(r['median'],2)} | {fmt(r['p95'],1)} | {fmt(r['newton'],2)} | {fmt(r['ls'],2)} | {fmt(r['pcg'],1)} | {r['mem']} | {r['converged'] and r['completed']} |")
print('\n## Aggregate (per benchmark, per tree)\n')
print('| bench | tree | n | mean ms/f (mean/median/min/max of run means) | sum-frame total s (mean) | process wall s (mean) | frame median (median) | p95 (median) | Newton/f | LS/f | PCG/f | peak mem delta MiB (min-max) | vs 2026-09-01 ref ms/f |')
print('|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|')
summary = {}
for b in ORDER:
    for tree in TREES:
        rs = runs.get((tree, b), [])
        if not rs: continue
        m = agg(rs, 'mean'); summary[(tree, b)] = m
        mem = agg(rs, 'mem'); ref = REF[b][0]
        print(f"| {b} | {tree} ({rs[0]['commit']}) | {len(rs)} | {fmt(m[0],2)} / {fmt(m[1],2)} / {fmt(m[2],2)} / {fmt(m[3],2)} | {fmt(agg(rs,'total_s')[0],2)} | {fmt(agg(rs,'wall_s')[0],1)} | {fmt(agg(rs,'median')[1],2)} | {fmt(agg(rs,'p95')[1],1)} | {fmt(agg(rs,'newton')[0],2)} | {fmt(agg(rs,'ls')[0],2)} | {fmt(agg(rs,'pcg')[0],1)} | {fmt(mem[2],0)}-{fmt(mem[3],0)} | {ref} ({(m[1]/ref-1)*100:+.1f}% median-of-means) |")
print('\n## Change (head vs others), median of run means\n')
print('| bench | head ms/f | ' + ' | '.join(f'{t} ms/f | head vs {t}' for t in TREES if t != 'head') + ' |')
print('|---|---:|' + '---:|---:|' * (len(TREES)-1))
for b in ORDER:
    h = summary.get(('head', b))
    if not h: continue
    cells = []
    for t in TREES:
        if t == 'head': continue
        o = summary.get((t, b))
        cells.append(f"{fmt(o[1],2)} | {(h[1]/o[1]-1)*100:+.1f}%" if o else '- | -')
    print(f"| {b} | {fmt(h[1],2)} | " + ' | '.join(cells) + ' |')
print('\n## Final-state observables\n')
for b in ORDER:
    for tree in TREES:
        for r in runs.get((tree, b), []):
            print(f"- {b} {tree} {r['runId']}: {json.dumps(r['obs'])}")

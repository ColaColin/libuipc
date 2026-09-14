#!/usr/bin/env python3
"""s17: pool two ab.py sweeps (raw per-run json) per arm; Welch t, disjointness, counts."""
import json,glob,sys,statistics as st
from scipy import stats
def load(d):
    arms={}
    for f in sorted(glob.glob(d+'/*.json')):
        import re,os
        mm=re.search(r'_(old|new)_r(\d+)\.json$',os.path.basename(f))
        if not mm: continue
        arms.setdefault(mm.group(1),[]).append(json.load(open(f)))
    return arms
def summ(m):
    fs=m['reportedBenchmark']['frame_stats']; t=m['reportedFrameTiming']; fr=m['frames']
    nw=sum(s['newton_iterations'] for s in fs); pc=sum(s['linear_solver_iterations'] for s in fs)
    return dict(mean=t['meanFrameMs'],mpn=t['meanFrameMs']*fr/nw,mpp=t['meanFrameMs']*fr/pc,newton=nw,pcg=pc)
dirs=sys.argv[1:]
pool={}
for d in dirs:
    for arm,runs in load(d).items(): pool.setdefault(arm,[]).extend(summ(m) for m in runs)
arms=list(pool); print('arms:',{a:len(pool[a]) for a in arms})
a,b=('old','new') if 'old' in pool else (arms[0],arms[1])
for k in ['mean','mpn','mpp','newton','pcg']:
    x=[r[k] for r in pool[a]]; y=[r[k] for r in pool[b]]
    t,p=stats.ttest_ind(x,y,equal_var=False)
    dis='DISJOINT' if max(y)<min(x) or max(x)<min(y) else 'overlapping'
    print('%8s: %s %.4f [%.4f,%.4f]  %s %.4f [%.4f,%.4f]  %+.2f%%  t=%.2f p=%.2e %s'%(k,a,st.mean(x),min(x),max(x),b,st.mean(y),min(y),max(y),100*(st.mean(y)/st.mean(x)-1),t,p,dis))

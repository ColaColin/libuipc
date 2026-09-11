#!/usr/bin/env python3
"""Adapt run_benchmark.py records into uipc.profile result dirs (benchmark.json + frame_stats.json).
Usage: to_profile_dirs.py <records_dir> <out_root>  -> one result dir per benchmark, using the run with the MEDIAN mean ms/frame."""
import json, sys, glob, os, statistics as st
from collections import defaultdict
src, out = sys.argv[1], sys.argv[2]
by = defaultdict(list)
for f in sorted(glob.glob(os.path.join(src, '*-*Z.json'))):
    d = json.load(open(f)); rb = d.get('reportedBenchmark')
    if d.get('returnCode') != 0 or not rb or d['frames'] != len(rb['frame_ms']): continue
    by[d['benchmark']].append((st.mean(rb['frame_ms']), d, rb))
for name, rs in by.items():
    rs.sort(key=lambda t: t[0]); mean, d, rb = rs[len(rs)//2]
    rd = os.path.join(out, name); os.makedirs(rd, exist_ok=True)
    env = {'backend': 'cuda', 'uipc_version': d['runtime'].get('uipcVersion'), 'platform_system': 'Linux',
           'platform_machine': 'x86_64', 'nvidia_gpus': [d['runtime'].get('gpu')], 'python_version': d['runtime'].get('pythonVersion'),
           'build_type': 'Release', 'cuda_architectures': 'native', 'commit': d['revisions']['libuipc']['commit']}
    json.dump({'name': name, 'num_frames': len(rb['frame_ms']), 'wall_time': sum(rb['frame_ms'])/1000.0,
               'environment': env, 'source_run': d['runId']}, open(os.path.join(rd, 'benchmark.json'), 'w'), indent=1)
    json.dump(rb['frame_stats'], open(os.path.join(rd, 'frame_stats.json'), 'w'))
    print(name, 'runs', len(rs), 'median-mean run', d['runId'], f'{mean:.2f} ms/f')

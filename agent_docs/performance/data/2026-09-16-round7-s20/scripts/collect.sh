#!/bin/bash
# Copy the s20 evidence into the repo data directory: driver log, scripts, the
# summary table with per-kernel breakdowns, and truncated logs (first 200
# records each; the full logs stay in /workspace/output/round7/s20/logs/).
set -u
OUT=/workspace/output/round7/s20
D=/workspace/deps/libuipc-src/agent_docs/performance/data/2026-09-16-round7-s20
mkdir -p $D/scripts $D/logs_truncated
cp $OUT/driver.log $D/driver.log
cp $OUT/{san.sh,matrix_head.sh,matrix_base.sh,matrix_repair.sh,chain_base.sh,chain_repair.sh,chain_drift.sh,summarize.py,drift.sh,drift_table.py,ab2.py,envaudit_s20.sh,audit_table.py,collect.sh} $D/scripts/ 2>/dev/null
cp $OUT/head_so_sha256.txt $OUT/base_so_sha256.txt $D/
python3 $OUT/summarize.py > $D/summary.md
python3 - <<'PY'
import glob, os, re
OUT='/workspace/output/round7/s20/logs'; D='/workspace/deps/libuipc-src/agent_docs/performance/data/2026-09-16-round7-s20/logs_truncated'
for f in sorted(glob.glob(OUT+'/*.txt')):
    L=open(f,errors='replace').read().splitlines()
    has=any(re.match(r'========= (Warning|Error|Uninitialized|Invalid|Program hit|Barrier|Internal Sanitizer)', l) for l in L)
    if not has: continue
    out=[]; n=0; keep=True
    for l in L:
        if re.match(r'========= (Warning|Error|Uninitialized|Invalid|Program hit|Barrier|Internal Sanitizer)', l):
            n+=1
            if n==201: out.append('========= [... truncated after 200 records; full log in /workspace/output/round7/s20/logs/ ...]'); keep=False
        if keep and (l.startswith('=========') or re.match(r'frame +[0-9]+ ', l) or 'TOTAL frames' in l or 'PcgPollVerify' in l or 'SpreadVerify' in l or 'BENCHMARK_RESULT' in l):
            out.append(l[:400])
        elif not keep and (re.match(r'frame +[0-9]+ ', l) or 'SUMMARY' in l or 'TOTAL frames' in l or 'PcgPollVerify' in l or 'SpreadVerify' in l):
            out.append(l[:400])
    open(os.path.join(D,os.path.basename(f)),'w').write('\n'.join(out)+'\n')
    print(os.path.basename(f), n)
PY
# drift + envaudit artifacts
mkdir -p $D/drift $D/envaudit
cp -r $OUT/drift/* $D/drift/ 2>/dev/null
cp $OUT/verify_base.json $OUT/verify_head.json $OUT/verify_base.log $OUT/verify_head.log $D/ 2>/dev/null
cp -r $OUT/envaudit $D/envaudit/ 2>/dev/null
du -sh $D

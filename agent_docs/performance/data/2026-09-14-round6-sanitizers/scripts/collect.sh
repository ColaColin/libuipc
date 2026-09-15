#!/bin/bash
# Copy the sweep's evidence into the repo data directory: driver log, scripts, the summary table with
# per-kernel breakdowns, the record-level comparisons, per-case attributions for the sim_case runs,
# and every log that reported anything, truncated to its first 200 records (the full logs stay in
# /workspace/output/round6/sanitizers/logs/).
set -u
OUT=/workspace/output/round6/sanitizers
D=/workspace/deps/libuipc-src/agent_docs/performance/data/2026-09-14-round6-sanitizers
mkdir -p $D/scripts $D/logs_truncated
cp $OUT/driver.log $D/driver.log
cp $OUT/{san.sh,calib.sh,matrix.sh,matrix2.sh,matrix3.sh,matrix4.sh,matrix5.sh,summarize.py,compare.py,assign_records.py,collect.sh,killsan.sh,ps_san.sh} $D/scripts/
python3 $OUT/summarize.py > $D/summary.md
{
  echo "# Record-level comparisons (compare.py)"; echo
  for spec in "head base racecheck build_multi_level_R --nokernel" "head base racecheck build_multi_level_R" "rb base racecheck -" "rb base racecheck build_multi_level_R --nokernel" "head base initcheck -" "rb base initcheck -" "head2 head racecheck -" "head3 head racecheck -" "base2 base racecheck -" "base2 head racecheck build_multi_level_R --nokernel" "rb2 rb racecheck build_multi_level_R --nokernel" "fr0 base racecheck -" "rd0 base racecheck -" "fr1 head racecheck -" "sr0 head racecheck -" "pp0 head racecheck -" "pp1 head racecheck -" "pp2 head racecheck -" "pp3 head racecheck -" "dj0 head racecheck -" "dj2 head racecheck -" "sp0 head racecheck -" "sp1 head racecheck -" "head base synccheck -" "head base memcheck -"; do
    echo "## compare.py $spec"; python3 $OUT/compare.py $spec; echo
  done
} > $D/compare.md
{
  echo "# sim_case: sanitizer records attributed to test cases (assign_records.py)"; echo
  for f in $OUT/logs/*__*__sim_case_*.txt; do
    b=$(basename $f .txt); case $b in aborted*|failed*) continue;; esac
    echo "## $b"; echo "summary: $(grep -E 'SUMMARY' $f | grep -v printed | tail -1)"; echo "tests:   $(grep -E 'All tests passed|test cases:' $f | tail -1)"
    python3 $OUT/assign_records.py $f | sed -E 's/MASPreconditionerEngine_//g'; echo
    echo "slowest cases (-d yes):"; grep -E '^[0-9.]+ s: [0-9]+_[A-Za-z]' $f | sort -rn | head -5 | sed 's/^/  /'; echo
  done
} > $D/sim_case_records.md
# truncated logs: every run that reported at least one record, plus the head/base memcheck of the full suite (head only)
python3 - <<'PY'
import glob, os, re
OUT='/workspace/output/round6/sanitizers/logs'; D='/workspace/deps/libuipc-src/agent_docs/performance/data/2026-09-14-round6-sanitizers/logs_truncated'
for f in sorted(glob.glob(OUT+'/*.txt')):
    b=os.path.basename(f)
    if b.startswith(('aborted','failed','calib')): continue
    L=open(f,errors='replace').read().splitlines()
    summ=[l for l in L if 'SUMMARY' in l]
    has=any(re.match(r'========= (Warning|Error|Uninitialized|Invalid|Program hit|Barrier)', l) for l in L)
    if not has and 'sim_case_full' not in b: continue
    out=[]; n=0; keep=True
    for l in L:
        if re.match(r'========= (Warning|Error|Uninitialized|Invalid|Program hit|Barrier)', l):
            n+=1
            if n==201: out.append('========= [... truncated after 200 records; full log in /workspace/output/round6/sanitizers/logs/ ...]'); keep=False
        if keep and (l.startswith('=========') or re.match(r'[0-9.]+ s: ', l) or 'TOTAL frames' in l or 'All tests passed' in l or 'test cases:' in l or 'BENCHMARK_RESULT' in l):
            out.append(l[:400])
        elif not keep and (re.match(r'[0-9.]+ s: ', l) or 'SUMMARY' in l or 'TOTAL frames' in l or 'All tests passed' in l or 'test cases:' in l or 'BENCHMARK_RESULT' in l):
            out.append(l[:400])
    open(os.path.join(D,b),'w').write('\n'.join(out)+'\n')
PY
du -sh $D; ls $D $D/logs_truncated | head -80

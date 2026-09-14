#!/bin/bash
# Round-6 V2: copy the evidence that backs each claim into the repo.
# Raw position dumps (~40 MB per run) stay in scratch; everything a reader needs
# to re-derive a number goes in.
set -eu
SRC=/workspace/output/round6/v2
DST=/workspace/deps/libuipc-src/agent_docs/performance/data/2026-09-14-round6-v2
mkdir -p "$DST/runs" "$DST/scripts" "$DST/micro" "$DST/crease"

# per-run observables: the VERIFY_RESULT / VERIFY_TRACE lines only (the logs also
# carry the benchmark blob, which is not evidence for this pass)
for f in "$SRC"/runs/*.log; do
  b=$(basename "$f")
  grep -h -E '^(tumbler:|PERTURBED|VERIFY_RESULT|VERIFY_TRACE)' "$f" > "$DST/runs/$b"
done

cp "$SRC"/analyze.py "$SRC"/divergence.py "$SRC"/micro_analyze.py \
   "$SRC"/crease_analyze.py "$SRC"/run_arms.sh "$SRC"/run_micro.sh \
   "$SRC"/run_crease.sh "$SRC"/envaudit.sh "$SRC"/build_probe.sh \
   "$SRC"/abn.py "$SRC"/collect_evidence.sh "$DST/scripts/"
cp "$SRC"/pe_severity_probe.cu "$SRC"/contact_micro.py "$SRC"/crease_micro.py "$DST/"

for f in analysis_n20.txt analysis_final.txt pe_severity.txt micro.txt crease.txt \
         divergence.txt cost_cwc.txt cost_cminy.txt envaudit.txt gate_rank1_0.txt \
         gate_rank1_1.txt gate_default.txt membrane_ci.txt record_section.md \
         sweep.log micro_sweep.log crease_sweep.log; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"
done
[ -d "$SRC/micro" ] && find "$SRC/micro" -name '*.log' -exec cp {} "$DST/micro/" \;
[ -d "$SRC/crease" ] && find "$SRC/crease" -name '*.txt' -exec cp {} "$DST/crease/" \;
du -sh "$DST"

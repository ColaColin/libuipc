#!/usr/bin/env python3
import json, statistics as st
from pathlib import Path
D = Path("/workspace/output/round7/s10/drift")
runs = []
for p in sorted(D.glob("drift_head_*.json")):
    d = json.load(open(p))
    ft = d["reportedFrameTiming"]; fs = d["reportedBenchmark"]["frame_stats"]
    nt = sum(f["newton_iterations"] for f in fs)
    pcg = sum(f["linear_solver_iterations"] for f in fs)
    ls = sum(f["line_search_trials"] for f in fs)
    wall = d.get("durationSeconds")
    runs.append(dict(file=p.name, mean=ft["meanFrameMs"], med=ft["medianFrameMs"],
                     nt=nt, pcg=pcg, ls=ls, wall=wall,
                     mpn=ft["meanFrameMs"]*130/nt))
base = json.load(open("/workspace/output/round7/baseline-runs/crease-press.json"))
bft = base["reportedFrameTiming"]
print("baseline-runs/crease-press.json (n=1): meanFrameMs=%.2f median=%.2f" % (bft["meanFrameMs"], bft["medianFrameMs"]))
bfs = base["reportedBenchmark"]["frame_stats"]
bnt = sum(f["newton_iterations"] for f in bfs); bpcg = sum(f["linear_solver_iterations"] for f in bfs)
bls = sum(f["line_search_trials"] for f in bfs)
print("baseline newton=%d pcg=%d ls=%d" % (bnt, bpcg, bls))
print()
print(f"{'run':>18s} {'meanFrameMs':>11s} {'ms/newton':>10s} {'newton':>7s} {'pcg':>7s} {'ls':>5s}")
for r in runs:
    print(f"{r['file']:>18s} {r['mean']:11.2f} {r['mpn']:10.2f} {r['nt']:7d} {r['pcg']:7d} {r['ls']:5d}")
means = [r["mean"] for r in runs]; mpns = [r["mpn"] for r in runs]
nts = [r["nt"] for r in runs]; pcgs = [r["pcg"] for r in runs]; lss = [r["ls"] for r in runs]
m, sd = st.mean(means), st.stdev(means)
print()
print(f"head n={len(runs)}: meanFrameMs {m:.2f} +- {sd:.2f} (cv {sd/m*100:.2f}%)  min {min(means):.2f} max {max(means):.2f}")
print(f"  vs baseline {bft['meanFrameMs']:.2f} (n=1): {(m-bft['meanFrameMs'])/bft['meanFrameMs']*100:+.2f}%")
print(f"  vs s00 floor mean 277.93 (n=5, cv 4.34%): {(m-277.93)/277.93*100:+.2f}%")
print(f"  baseline min draw sits {'ABOVE' if bft['meanFrameMs'] > max(means) else 'inside'} head range [%.2f, %.2f]" % (min(means), max(means)) if False else "")
print(f"  head max {max(means):.2f} vs baseline single draw {bft['meanFrameMs']:.2f}")
print(f"ms_per_newton: head {st.mean(mpns):.2f} (cv {st.stdev(mpns)/st.mean(mpns)*100:.2f}%) vs baseline {bft['meanFrameMs']*130/bnt:.2f} -> {(st.mean(mpns)-bft['meanFrameMs']*130/bnt)/(bft['meanFrameMs']*130/bnt)*100:+.2f}%")
print()
print("counts vs s00 noise-floor envelope (floor: newton 565.2 [552,587] cv2.56%; pcg 117.5k [104.6k,126k] cv7.64%; ls 623.4 [599,654]):")
print(f"  newton {st.mean(nts):.1f} range [{min(nts)},{max(nts)}]  -> {'INSIDE' if min(nts)>=552 and max(nts)<=587 else 'OUTSIDE'} floor range")
print(f"  pcg    {st.mean(pcgs)/1000:.1f}k range [{min(pcgs)/1000:.1f}k,{max(pcgs)/1000:.1f}k] -> {'INSIDE' if min(pcgs)>=104600 and max(pcgs)<=126000 else 'OUTSIDE'} floor range")
print(f"  ls     {st.mean(lss):.1f} range [{min(lss)},{max(lss)}] -> {'INSIDE' if min(lss)>=599 and max(lss)<=654 else 'OUTSIDE'} floor range")

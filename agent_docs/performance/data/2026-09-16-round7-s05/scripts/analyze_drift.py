#!/usr/bin/env python3
"""Round-7 s05 Part B drift: current-head default runs (n=6) vs the round
baseline (baseline-runs/crease-press.json, n=1 at perf-round7-base) and the
scene's own s00 noise floor.  The REQUIRED line: meanFrameMs old -> new.
"""
import json
from pathlib import Path

import numpy as np

S = Path("/workspace/output/round7/s05")

base = json.load(open("/workspace/output/round7/baseline-runs/crease-press.json"))
bt = base["reportedFrameTiming"]
bfs = base["reportedBenchmark"]["frame_stats"]
b_newton = sum(s["newton_iterations"] for s in bfs)
b_pcg = sum(s["linear_solver_iterations"] for s in bfs)
b_ls = sum(s["line_search_trials"] for s in bfs)

rows = []
for k in range(1, 7):
    d = json.load(open(S / f"drift_default_{k}.json"))
    t = d["reportedFrameTiming"]
    fs = d["reportedBenchmark"]["frame_stats"]
    rows.append(dict(
        mean=t["meanFrameMs"], median=t["medianFrameMs"], p95=t["p95FrameMs"],
        newton=sum(s["newton_iterations"] for s in fs),
        pcg=sum(s["linear_solver_iterations"] for s in fs),
        ls=sum(s["line_search_trials"] for s in fs),
        dur=d["durationSeconds"],
        ret=d["returnCode"], frames=t["frames"]))

print(f"baseline (perf-round7-base, n=1): meanFrameMs {bt['meanFrameMs']:.2f}  "
      f"median {bt['medianFrameMs']:.2f}  newton {b_newton}  pcg {b_pcg}  ls {b_ls}")
print(f"{'run':>3} {'meanFrameMs':>11} {'newton':>7} {'pcg':>8} {'ls':>5} {'wall_s':>7} {'rc':>3} {'frames':>6}")
for i, r in enumerate(rows, 1):
    print(f"{i:3d} {r['mean']:11.2f} {r['newton']:7d} {r['pcg']:8d} {r['ls']:5d} "
          f"{r['dur']:7.1f} {r['ret']:3d} {r['frames']:6d}")

for key in ("mean", "newton", "pcg", "ls"):
    v = np.array([r[key] for r in rows], dtype=float)
    print(f"{key:6s} mean={v.mean():10.3f} sd={v.std(ddof=1):9.3f} cv={v.std(ddof=1)/v.mean()*100:5.2f}%")

mean_new = float(np.mean([r["mean"] for r in rows]))
mean_old = bt["meanFrameMs"]
newton_new = float(np.mean([r["newton"] for r in rows]))
pcg_new = float(np.mean([r["pcg"] for r in rows]))
ls_new = float(np.mean([r["ls"] for r in rows]))
msn_new = mean_new * 130 / newton_new
msn_old = mean_old * 130 / b_newton
print()
print(f"REQUIRED: meanFrameMs old -> new = {mean_old:.2f} -> {mean_new:.2f} ms "
      f"({(mean_new/mean_old-1)*100:+.2f} %), head n=6 (cv 4.34 % s00 noise floor; "
      f"n=6 MDE ~4.4 %)")
print(f"ms_per_newton: {msn_old:.3f} -> {msn_new:.3f} ({(msn_new/msn_old-1)*100:+.2f} %)")
print(f"newton_total: {b_newton} -> {newton_new:.1f} ({(newton_new/b_newton-1)*100:+.2f} %)  "
      f"s00 noise floor cv 2.56 %, range 552-587")
print(f"pcg_total:    {b_pcg} -> {pcg_new:.0f} ({(pcg_new/b_pcg-1)*100:+.2f} %)  "
      f"s00 cv 7.64 %, range 104.6k-126.0k")
print(f"ls_total:     {b_ls} -> {ls_new:.1f} ({(ls_new/b_ls-1)*100:+.2f} %)  "
      f"s00 cv 4.09 %, range 599-654")

#!/usr/bin/env python3
"""Round-7 s20 Part B: drift table from the ab2.py summary jsons."""
import glob
import json
import sys

BASE = "/workspace/output/round7/s20/drift"

order = ["drift_cp", "drift_cwc", "drift_tumb", "drift_case2", "drift_bunny", "drift_rwb"]
disp = {"drift_cp": "crease-press", "drift_cwc": "cube-wall-cloth", "drift_tumb": "tumbler-garments",
        "drift_case2": "stiff-gipc-case2", "drift_bunny": "mas-bunny", "drift_rwb": "rigid-wrecking-balls"}

print("| scene | n/arm | frames | meanFrameMs base → head (Δ%) | medianFrameMs | ms/newton | Newton tot | PCG tot | LS tot | disjoint(mean) | Welch p (mean) |")
print("|---|---:|---:|---|---|---|---|---|---|---|---|")
for lbl in order:
    fs = sorted(glob.glob(f"{BASE}/{lbl}/{lbl}_*_summary.json"))
    if not fs:
        continue
    d = json.load(open(fs[-1]))
    v = d["verdict"]
    def g(k, f="{:.2f}"):
        a, b = v[k]["base_mean"], v[k]["head_mean"]
        return f"{a:.2f} → {b:.2f} ({v[k]['delta_pct']:+.2f}%)"
    n = d["n"]
    print(f"| {disp[lbl]} | {n} | {d['frames']} | {g('mean_ms')} | {g('median_ms')} | "
          f"{g('ms_per_newton')} | {v['newton']['base_mean']:.0f} → {v['newton']['head_mean']:.0f} "
          f"({v['newton']['delta_pct']:+.2f}%) | {v['pcg']['base_mean']/1000:.1f}k → {v['pcg']['head_mean']/1000:.1f}k "
          f"({v['pcg']['delta_pct']:+.2f}%) | {v['line_search']['base_mean']:.0f} → {v['line_search']['head_mean']:.0f} | "
          f"{'yes' if v['mean_ms']['disjoint'] else 'no'} | {v['mean_ms'].get('p', float('nan')):.2g} |")
print()
print("per-run mean ms/frame:")
for lbl in order:
    fs = sorted(glob.glob(f"{BASE}/{lbl}/{lbl}_*_summary.json"))
    if not fs:
        continue
    d = json.load(open(fs[-1]))
    for arm in ("base", "head"):
        runs = [f"{r['mean_ms']:.1f}" for r in d["arms"][arm]]
        warm = [f"{r['mean_ms']:.1f}" for r in d["warmup_discarded"] if r["arm"] == arm]
        print(f"  {disp[lbl]:20s} {arm:5s} {' '.join(runs)}   warm-up(discarded): {' '.join(warm)}")

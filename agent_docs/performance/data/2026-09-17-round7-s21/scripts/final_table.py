#!/usr/bin/env python3
"""Round-7 s21: final-evaluation headline table from the ab2.py summary jsons."""
import glob
import json

BASE = "/workspace/output/round7/s21/s21"

order = ["s21", "s21", "s21", "s21", "s21", "s21"]
scenes = ["crease-press", "tumbler-garments", "cube-wall-cloth", "stiff-gipc-case2",
          "mas-bunny", "rigid-wrecking-balls"]

print("| scene | n/arm | frames | meanFrameMs base → head (Δ%) | ms/newton (Δ%) | Newton tot (Δ%) | PCG tot (Δ%) | LS | peak MiB base → head | verdict(mean) | p(mean) |")
print("|---|---:|---:|---|---|---|---|---|---|---|---|")
for sc in scenes:
    fs = sorted(glob.glob(f"{BASE}/s21_{sc}_summary.json"))
    if not fs:
        continue
    d = json.load(open(fs[-1]))
    v = d["verdict"]
    def g(k):
        a, b = v[k]["base_mean"], v[k]["head_mean"]
        return f"{a:.2f} → {b:.2f} ({v[k]['delta_pct']:+.2f}%)"
    print(f"| {sc} | {d['n']} | {d['frames']} | {g('mean_ms')} | {g('ms_per_newton')} | "
          f"{v['newton']['base_mean']:.0f} → {v['newton']['head_mean']:.0f} ({v['newton']['delta_pct']:+.2f}%) | "
          f"{v['pcg']['base_mean']/1000:.1f}k → {v['pcg']['head_mean']/1000:.1f}k ({v['pcg']['delta_pct']:+.2f}%) | "
          f"{v['line_search']['base_mean']:.0f} → {v['line_search']['head_mean']:.0f} | "
          f"{v['peak_mib']['base_mean']:.0f} → {v['peak_mib']['head_mean']:.0f} ({v['peak_mib']['delta_pct']:+.1f}%) | "
          f"{'DISJOINT' if v['mean_ms']['disjoint'] else 'overlapping'} | {v['mean_ms'].get('p', float('nan')):.2g} |")
print()
print("per-run mean ms/frame:")
for sc in scenes:
    fs = sorted(glob.glob(f"{BASE}/s21_{sc}_summary.json"))
    if not fs:
        continue
    d = json.load(open(fs[-1]))
    for arm in ("base", "head"):
        runs = [f"{r['mean_ms']:.1f}" for r in d["arms"][arm]]
        warm = [f"{r['mean_ms']:.1f}" for r in d["warmup_discarded"] if r["arm"] == arm]
        print(f"  {sc:20s} {arm:5s} {' '.join(runs)}   warm-up(discarded): {' '.join(warm)}")
print()
print("per-run peak MiB:")
for sc in scenes:
    fs = sorted(glob.glob(f"{BASE}/s21_{sc}_summary.json"))
    if not fs:
        continue
    d = json.load(open(fs[-1]))
    for arm in ("base", "head"):
        runs = [str(r["peak_mib"]) for r in d["arms"][arm]]
        print(f"  {sc:20s} {arm:5s} {' '.join(runs)}")

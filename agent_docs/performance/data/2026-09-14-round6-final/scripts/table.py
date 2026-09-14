#!/usr/bin/env python3
"""Round-6 V3: the base-vs-head table from ab2.py's per-scene summaries (markdown + json)."""
import json, glob, os, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else "/workspace/output/round6/v3/ab"
order = ["tumbler-garments", "rigid-wrecking-balls", "cube-wall-cloth", "stiff-gipc-case2", "mas-bunny"]
S = {}
for f in glob.glob(f"{D}/*_summary.json"):
    j = json.load(open(f)); S[j["scene"]] = j

def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"

print("| scene | frames | n/arm | base mean | head mean | Δ mean | base median | head median | Δ median | disjoint (mean) | Welch t / p (mean) | MWU p |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
for sc in order:
    if sc not in S: continue
    j = S[sc]; v = j["verdict"]; m = v["mean_ms"]; md = v["median_ms"]
    print(f"| `{sc}` | {j['frames']} | {j['n']} | {m['base_mean']:.2f} | {m['head_mean']:.2f} | **{m['delta_pct']:+.2f} %** | "
          f"{md['base_mean']:.2f} | {md['head_mean']:.2f} | {md['delta_pct']:+.2f} % | "
          f"{'**yes**' if m['disjoint'] else 'no'} [{m['base_min']:.1f}–{m['base_max']:.1f}] vs [{m['head_min']:.1f}–{m['head_max']:.1f}] | "
          f"{m['t']:+.1f} / {fmt_p(m['p'])} | {fmt_p(m['mwu_p'])} |")

print("\n| scene | Newton/frame base | Newton/frame head | Δ Newton | PCG/frame base | PCG/frame head | Δ PCG | line search base → head | ms/Newton Δ (p) | ms/PCG Δ (p) | peak MiB base → head (Δ) |")
print("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
for sc in order:
    if sc not in S: continue
    j = S[sc]; v = j["verdict"]
    nw, pc, ls, pk = v["newton_per_frame"], v["pcg_per_frame"], v["line_search"], v["peak_mib"]
    mn, mp = v["ms_per_newton"], v["ms_per_pcg"]
    print(f"| `{sc}` | {nw['base_mean']:.2f} [{nw['base_min']:.2f}–{nw['base_max']:.2f}] | {nw['head_mean']:.2f} [{nw['head_min']:.2f}–{nw['head_max']:.2f}] | {nw['delta_pct']:+.2f} % | "
          f"{pc['base_mean']:.1f} [{pc['base_min']:.1f}–{pc['base_max']:.1f}] | {pc['head_mean']:.1f} [{pc['head_min']:.1f}–{pc['head_max']:.1f}] | {pc['delta_pct']:+.2f} % | "
          f"{ls['base_mean']:.0f} → {ls['head_mean']:.0f} ({ls['delta_pct']:+.2f} %) | "
          f"{mn['delta_pct']:+.2f} % ({fmt_p(mn['p'])}) | {mp['delta_pct']:+.2f} % ({fmt_p(mp['p'])}) | "
          f"{pk['base_mean']:.0f} [{pk['base_min']:.0f}–{pk['base_max']:.0f}] → {pk['head_mean']:.0f} [{pk['head_min']:.0f}–{pk['head_max']:.0f}] ({pk['delta_pct']:+.1f} %) |")

# the per-run lists, for the appendix
print("\nper-run mean ms/frame (rep order):")
for sc in order:
    if sc not in S: continue
    j = S[sc]
    for arm in j["arms"]:
        runs = j["arms"][arm]
        print(f"  {sc:22s} {arm:5s} " + " ".join(f"{r['mean_ms']:.2f}" for r in runs))
    w = j.get("warmup_discarded", [])
    print(f"  {sc:22s} warm-up (discarded): " + ", ".join(f"{r['arm']} {r['mean_ms']:.2f}" for r in w))

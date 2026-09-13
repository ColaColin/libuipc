#!/usr/bin/env python3
"""Crease-severity micro-test: Newton cost vs |theta - theta_bar|, A (exact) vs B (GN)."""
import re, sys
from pathlib import Path
import numpy as np
from scipy import stats

D = Path("/workspace/output/round6/v1/crease")
prefix = sys.argv[1] if len(sys.argv) > 1 else ""

def load(tag):
    rows = []
    for f in sorted(D.glob(f"{prefix}{tag}_*.txt")):
        for line in f.read_text().splitlines():
            if not line.startswith("CREASE "):
                continue
            d = dict(kv.split("=", 1) for kv in line[7:].split())
            rows.append({k: float(v) for k, v in d.items()} | {"run": f.stem})
    return rows

A, B = load("A"), load("B")
if not A or not B:
    raise SystemExit(f"no data for prefix {prefix!r}")
print(f"[{prefix or 'base'}] runs A={len({r['run'] for r in A})} B={len({r['run'] for r in B})}, "
      f"frames A={len(A)} B={len(B)}")
for n, S in (("A(exact)", A), ("B(gauss-newton)", B)):
    print(f"  {n:16s} converged all={all(r['converged']==1 for r in S)} "
          f"hit_newton={sum(r['hit_newton'] for r in S):.0f} hit_ls={sum(r['hit_ls'] for r in S):.0f} "
          f"finite all={all(r['finite']==1 for r in S)} "
          f"theta_max reached {max(r['theta_max'] for r in S):.3f} rad")

BINS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.5]
print(f"\n{'theta_p95 bin (rad)':>22s} {'nA':>5s} {'nB':>5s} {'newton A':>16s} {'newton B':>16s} "
      f"{'dN %':>7s} {'p':>8s} | {'pcg A':>9s} {'pcg B':>9s} {'ls A':>7s} {'ls B':>7s}")
for lo, hi in zip(BINS, BINS[1:]):
    a = [r for r in A if lo <= r["theta_p95"] < hi]
    b = [r for r in B if lo <= r["theta_p95"] < hi]
    if len(a) < 5 or len(b) < 5:
        continue
    na = np.array([r["newton"] for r in a]); nb = np.array([r["newton"] for r in b])
    pa = np.array([r["pcg"] for r in a]); pb = np.array([r["pcg"] for r in b])
    la = np.array([r["ls"] for r in a]); lb = np.array([r["ls"] for r in b])
    p = stats.mannwhitneyu(nb, na, alternative="two-sided").pvalue
    print(f"  [{lo:5.2f}, {hi:5.2f}) {len(a):11d} {len(b):5d} "
          f"{na.mean():8.3f}+-{na.std(ddof=1)/np.sqrt(len(na)):5.3f} "
          f"{nb.mean():8.3f}+-{nb.std(ddof=1)/np.sqrt(len(nb)):5.3f} "
          f"{(nb.mean()/na.mean()-1)*100:+7.2f} {p:8.4f} | "
          f"{pa.mean():9.1f} {pb.mean():9.1f} {la.mean():7.2f} {lb.mean():7.2f}")

na = np.array([r["newton"] for r in A]); nb = np.array([r["newton"] for r in B])
print(f"\n  whole sweep: newton A {na.mean():.3f}  B {nb.mean():.3f}  "
      f"({(nb.mean()/na.mean()-1)*100:+.2f} %, p={stats.mannwhitneyu(nb,na).pvalue:.4f})")
# the trend that matters: does the B-A gap grow with severity?
xs, ys = [], []
for lo, hi in zip(BINS, BINS[1:]):
    a = [r["newton"] for r in A if lo <= r["theta_p95"] < hi]
    b = [r["newton"] for r in B if lo <= r["theta_p95"] < hi]
    if len(a) >= 5 and len(b) >= 5:
        xs.append((lo + hi) / 2); ys.append(np.mean(b) / np.mean(a) - 1)
if len(xs) >= 3:
    sl = stats.linregress(xs, ys)
    print(f"  trend of (newton_B/newton_A - 1) against crease severity: "
          f"slope {sl.slope:+.4f} /rad, r={sl.rvalue:+.3f}, p={sl.pvalue:.3f}")

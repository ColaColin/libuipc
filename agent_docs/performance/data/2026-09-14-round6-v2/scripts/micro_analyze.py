#!/usr/bin/env python3
"""Round-6 V2: analysis of the contact-severity micro-test.

Pre-declared reading (VERDICT_RULE.md §8): if the rank-1 PE Hessian degrades,
Newton counts must rise in arm 1 relative to arm 0 **as contact severity
rises**, with the gap growing monotonically.  Severity is measured, not
assumed: `g = (clearance - 2r)/d_hat`, the inter-layer surface separation in units
of the activation distance, recomputed from the retrieved positions every
frame.
"""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/output/round6/v2/micro"
CFGS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base", "deep", "slide", "fastslide", "tight"]


def parse(path):
    rows = []
    setup = {}
    for line in open(path):
        if line.startswith("CONTACT_SETUP "):
            setup = dict(kv.split("=", 1) for kv in line.split()[1:])
        elif line.startswith("CONTACT "):
            d = {}
            for kv in line.split()[1:]:
                k, v = kv.split("=", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
            rows.append(d)
    return setup, rows


def main():
    for cfg in CFGS:
        files = sorted(glob.glob(os.path.join(OUT, f"{cfg}_*_*.log")))
        if not files:
            continue
        arms = defaultdict(list)
        setup = {}
        for f in files:
            m = re.match(rf"{cfg}_(\d)_(\d+)\.log$", os.path.basename(f))
            if not m:
                continue
            s, rows = parse(f)
            if not rows:
                print(f"!! {f}: no frames")
                continue
            setup = s or setup
            arms[m.group(1)].append(rows)
        if not arms:
            continue
        print("=" * 96)
        print(f"## config {cfg}   runs per arm: "
              f"{ {k: len(v) for k, v in sorted(arms.items())} }")
        print(f"   setup: {setup}")
        print("=" * 96)

        # --- determinism / safety
        for a in sorted(arms):
            R = arms[a]
            nf = [len(r) for r in R]
            nconv = sum(sum(1 for x in r if x["converged"] == 0) for r in R)
            nlim = sum(sum(1 for x in r if x["hit_newton"] or x["hit_ls"]) for r in R)
            nfin = sum(sum(1 for x in r if x["finite"] == 0) for r in R)
            tot = [sum(x["newton"] for x in r) for r in R]
            gmin = min(min(x["g"] for x in r) for r in R)
            dmin = min(min(x["d_min"] for x in r) for r in R)
            thick = [r[-1]["thickness_mm"] for r in R]
            print(f"   arm rank1={a}: frames {set(nf)}  non-converged {nconv}  "
                  f"limit-hits {nlim}  non-finite {nfin}")
            print(f"      Newton total per run {sorted(tot)}")
            print(f"      severity reached: min g = {gmin:.4f}  (d_min {dmin*1e6:.2f} um)"
                  f"   final stack thickness mm {['%.4f' % t for t in thick]}")
        print()

        # --- severity binned Newton comparison
        # severity bins on the measured clearance g = (clearance - 2r)/d_hat:
        # 1.0 = the activation distance, 0 = touching, negative = the flat-sheet
        # proxy has broken down under wrinkling (the true point-triangle gap
        # cannot be negative).  Deeper = further down the table.
        bins = [(1.00, 1e9), (0.75, 1.00), (0.50, 0.75), (0.35, 0.50),
                (0.20, 0.35), (0.10, 0.20), (0.00, 0.10), (-1e9, 0.00)]
        print(f"   {'g bin':>14s} {'frames 0/1':>12s} {'Newton 0':>9s} {'Newton 1':>9s} "
              f"{'delta':>8s} {'p':>10s} {'PCG 0':>9s} {'PCG 1':>9s} {'dPCG':>8s}")
        xs, ys = [], []
        for lo, hi in bins:
            v0 = [x["newton"] for r in arms.get("0", []) for x in r if lo <= x["g"] < hi]
            v1 = [x["newton"] for r in arms.get("1", []) for x in r if lo <= x["g"] < hi]
            p0 = [x["pcg"] for r in arms.get("0", []) for x in r if lo <= x["g"] < hi]
            p1 = [x["pcg"] for r in arms.get("1", []) for x in r if lo <= x["g"] < hi]
            if len(v0) < 10 or len(v1) < 10:
                continue
            m0, m1 = float(np.mean(v0)), float(np.mean(v1))
            q0, q1 = float(np.mean(p0)), float(np.mean(p1))
            p = float(stats.mannwhitneyu(v1, v0, alternative="two-sided").pvalue)
            d = (m1 / m0 - 1) * 100 if m0 else float("nan")
            dq = (q1 / q0 - 1) * 100 if q0 else float("nan")
            print(f"   [{max(lo,-9.99):5.2f},{min(hi,9.99):5.2f}) "
                  f"{len(v0):5d}/{len(v1):<6d} {m0:9.3f} {m1:9.3f} {d:+7.1f}% "
                  f"{p:10.3g} {q0:9.1f} {q1:9.1f} {dq:+7.1f}%")
            xs.append(float(len(xs)))   # bin index: larger = deeper contact
            ys.append(m1 / m0 - 1 if m0 else np.nan)
        if len(xs) >= 3:
            sl, ic, r, pv, se = stats.linregress(xs, ys)
            print(f"\n   regression of (Newton_1/Newton_0 - 1) on severity "
                  f"(bin index, deeper = larger): slope {sl:+.4f} per bin "
                  f"(r={r:+.2f}, p={pv:.3g})")
            print("   pre-declared degradation signature: slope > 0 and growing. "
                  f"-> {'PRESENT' if (sl > 0 and pv < 0.05) else 'absent'}")

        # --- whole sweep
        t0 = [sum(x["newton"] for x in r) for r in arms.get("0", [])]
        t1 = [sum(x["newton"] for x in r) for r in arms.get("1", [])]
        if t0 and t1:
            print(f"\n   whole sweep Newton totals: arm0 mean {np.mean(t0):.1f} "
                  f"{sorted(t0)}   arm1 mean {np.mean(t1):.1f} {sorted(t1)}"
                  f"   {100*(np.mean(t1)/np.mean(t0)-1):+.2f} %")
        # --- the physically important one: does the change let layers get closer?
        g0 = [min(x["g"] for x in r) for r in arms.get("0", [])]
        g1 = [min(x["g"] for x in r) for r in arms.get("1", [])]
        if g0 and g1:
            p = float(stats.mannwhitneyu(g1, g0, alternative="two-sided").pvalue)
            print(f"   minimum g reached per run: arm0 {sorted(np.round(g0, 5))}")
            print(f"                              arm1 {sorted(np.round(g1, 5))}"
                  f"   MW p={p:.3g}")
            print("   (a *lower* g in arm 1 would mean the softened Hessian lets the "
                  "layers approach further -- a physics failure, not a cost one)")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""s21: tumbler per-frame divergence onset — scene chaos vs trajectory change.

s20 flagged tumbler PCG −9.65 % (p=0.004, nearly disjoint) / Newton −3.75 % at
drift time. Before quoting tumbler's final wall number, decide:
  (a) base and head diverge from frame ~0 in per-frame PCG counts  -> scene chaos
      (the round-6 class: first PCG difference at frame 4 within the SAME binary);
  (b) head tracks base frame-for-frame for N frames, then shifts    -> a trajectory
      change caused by the round's rounding-level steps.

Method: for within-arm pairs (the chaos baseline) and cross-arm pairs, the first
frame whose PCG (or Newton) count differs, plus the signed per-frame delta
profile over the run. If cross-arm first-diff frames sit in the same range as
within-arm ones and the per-frame deltas are two-sided noise that accumulates,
it is chaos; a one-sided shift starting at a late frame would be (b).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

BASE = Path("/workspace/output/round7/s21/s21")


def load(arm: str, rep: int):
    p = BASE / f"s21_tumbler-garments_{arm}_r{rep}.json"
    if not p.exists():
        sys.exit(f"missing {p}")
    meta = json.loads(p.read_text())
    stats = meta["reportedBenchmark"]["frame_stats"]
    return ([s["linear_solver_iterations"] for s in stats],
            [s["newton_iterations"] for s in stats], meta)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def profile(a, b, lo, hi):
    sa = sum(a[lo:hi]); sb = sum(b[lo:hi])
    return sa, sb, (sb - sa) / sa * 100 if sa else 0.0


def main():
    runs = {}
    for arm in ("base", "head"):
        reps = sorted(int(p.stem.split("_r")[1])
                      for p in BASE.glob(f"s21_tumbler-garments_{arm}_r*.json"))
        runs[arm] = {r: load(arm, r) for r in reps}
        print(f"{arm}: {len(reps)} reps {reps}")

    print("\n=== first differing frame (PCG counts; Newton in parens)")
    pairs = []
    for a, b in itertools.combinations(sorted(runs["base"]), 2):
        pairs.append(("within-base", "base", a, "base", b))
    for a, b in itertools.combinations(sorted(runs["head"]), 2):
        pairs.append(("within-head", "head", a, "head", b))
    for a in sorted(runs["base"]):
        for b in sorted(runs["head"]):
            pairs.append(("CROSS", "base", a, "head", b))
    for tag, arm_a, a, arm_b, b in pairs:
        pa, na, _ = runs[arm_a][a]
        qb, nb, _ = runs[arm_b][b]
        fd = first_diff(pa, qb)
        fn = first_diff(na, nb)
        print(f"{tag:>11} {arm_a} r{a} vs {arm_b} r{b}: first PCG diff frame {fd}  (Newton {fn})")

    print("\n=== per-window PCG totals (frames [lo,hi)): signed delta")
    windows = [(0, 10), (10, 30), (30, 60), (60, 100), (100, 140), (140, 180)]
    for tag, arm_a, a, arm_b, b in [p for p in pairs if p[0] == "CROSS"][:6]:
        pa, _, _ = runs[arm_a][a]; qb, _, _ = runs[arm_b][b]
        cells = []
        for lo, hi in windows:
            sa, sb, d = profile(pa, qb, lo, hi)
            cells.append(f"[{lo:3d},{hi:3d}) {sa:6d}->{sb:6d} ({d:+.1f}%)")
        print(f"CROSS r{a} vs r{b}: " + "  ".join(cells))
    # within-arm control profiles for 2 pairs
    for tag, arm_a, a, arm_b, b in [p for p in pairs if p[0] != "CROSS"][:2]:
        pa, _, _ = runs[arm_a][a]; qb, _, _ = runs[arm_b][b]
        cells = []
        for lo, hi in windows:
            sa, sb, d = profile(pa, qb, lo, hi)
            cells.append(f"[{lo:3d},{hi:3d}) {sa:6d}->{sb:6d} ({d:+.1f}%)")
        print(f"{tag:>11} r{a} vs r{b}: " + "  ".join(cells))

    print("\n=== totals")
    for arm in ("base", "head"):
        pcgs = [sum(r[0]) for r in runs[arm].values()]
        news = [sum(r[1]) for r in runs[arm].values()]
        print(f"{arm}: PCG {min(pcgs)}-{max(pcgs)} (mean {sum(pcgs)/len(pcgs):.0f})  "
              f"Newton {min(news)}-{max(news)} (mean {sum(news)/len(news):.0f})")


if __name__ == "__main__":
    main()

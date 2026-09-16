#!/usr/bin/env python3
"""Round-7 s05 divergence analysis (round-6 V1 instrument, re-applied).

Per frame f = 0..130: rms(P_i - P_j) over all 16774 sheet vertices for every
run pair, grouped into families:
  wA   within-A (exact, 10 runs -> 45 pairs)
  wAp  within-A'' (perturbed exact, 5 -> 10)
  xAp  A vs A'' (50)
  EQUIV = wA + wAp + xAp (105)   -- physically equivalent by construction
  CROSS = A vs B (100) + A'' vs B (50) = 150
  wB   within-B (10 -> 45)
Verdict inputs: seed (frame 1), growth rate (log-slope over frames 3-15 of the
median curve), saturation (mean over the last 40 frames), frame-wise
containment (frame 8..130, CROSS median <= 2 x EQUIV p95).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

RUNS = Path("/workspace/output/round7/s05/runs")
A = [f"a{k:02d}" for k in range(1, 11)]
B = [f"b{k:02d}" for k in range(1, 11)]
AP = [f"ap{k}" for k in range(1, 6)]
NF = 131  # frame 0 + 130


def load(names):
    return {n: np.load(RUNS / f"{n}.npy").astype(np.float64) for n in names}


def pair_rms(P, Q):
    d = P - Q
    n = P[0].size
    return np.sqrt(np.einsum("fnc,fnc->f", d, d) / n)


def family_curves(runs, pairs):
    curves = np.empty((len(pairs), NF))
    for i, (u, v) in enumerate(pairs):
        curves[i] = pair_rms(runs[u], runs[v])
    return curves


def efold(curve, lo=3, hi=16):
    """e-folds per frame from a log-linear fit over frames lo..hi."""
    f = np.arange(lo, hi)
    y = np.log(np.maximum(curve[lo:hi], 1e-300))
    slope = np.polyfit(f, y, 1)[0]
    return float(np.exp(slope))


def main():
    runs = {**load(A), **load(B), **load(AP)}
    for n, P in runs.items():
        assert P.shape == (NF, 16774, 3), (n, P.shape)

    fams = {
        "wA": [(u, v) for u, v in combinations(A, 2)],
        "wAp": [(u, v) for u, v in combinations(AP, 2)],
        "xAp": [(u, v) for u in A for v in AP],
        "xAB": [(u, v) for u in A for v in B],
        "xApB": [(u, v) for u in AP for v in B],
        "wB": [(u, v) for u, v in combinations(B, 2)],
    }
    out = {k: family_curves(runs, v) for k, v in fams.items()}
    equiv = np.vstack([out["wA"], out["wAp"], out["xAp"]])
    cross = np.vstack([out["xAB"], out["xApB"]])

    stats = {}
    for name, curves in [("EQUIV", equiv), ("CROSS", cross), ("wA", out["wA"]),
                         ("wAp", out["wAp"]), ("xAp", out["xAp"]),
                         ("xAB", out["xAB"]), ("xApB", out["xApB"]), ("wB", out["wB"])]:
        med = np.median(curves, axis=0)
        p95 = np.percentile(curves, 95, axis=0)
        p05 = np.percentile(curves, 5, axis=0)
        stats[name] = dict(
            n_pairs=len(curves),
            seed_med=float(med[1]), seed_max=float(curves[:, 1].max()),
            efolds=efold(med),
            sat_mean_last40=float(curves[:, -40:].mean()),
            sat_med_last40=float(np.median(curves[:, -40:].mean(axis=1))),
            sat_range_last40=[float(curves[:, -40:].mean(axis=1).min()),
                              float(curves[:, -40:].mean(axis=1).max())],
        )
        np.savez(RUNS / f"div_{name}.npz", med=med, p05=p05, p95=p95,
                 all_last40=curves[:, -40:].mean(axis=1))

    # verdict inputs
    equi_max_seed = float(equiv[:, 1].max())
    cross_seed_med = float(np.median(cross[:, 1]))
    ratio_growth = stats["CROSS"]["efolds"] / stats["EQUIV"]["efolds"]
    ratio_sat = stats["CROSS"]["sat_mean_last40"] / stats["EQUIV"]["sat_mean_last40"]
    med_cross = np.median(cross, axis=0)
    p95_equiv = np.percentile(equiv, 95, axis=0)
    viol = np.flatnonzero(med_cross[8:] > 2.0 * p95_equiv[8:]) + 8
    viol_ratio = (med_cross[8:] / p95_equiv[8:])

    lines = []
    for name, s in stats.items():
        lines.append(f"{name:6s} n={s['n_pairs']:3d} seed(f1) med={s['seed_med']:.3e} "
                     f"max={s['seed_max']:.3e} efolds/frame={s['efolds']:.3f} "
                     f"sat(last40) mean={s['sat_mean_last40']:.4f} "
                     f"[{s['sat_range_last40'][0]:.4f}, {s['sat_range_last40'][1]:.4f}]")
    lines += [
        "",
        f"VERDICT INPUTS",
        f"  (i)   seed: CROSS med {cross_seed_med:.3e} vs 3x EQUIV max {3*equi_max_seed:.3e} "
        f"-> {'PASS' if cross_seed_med <= 3*equi_max_seed else 'FAIL (widen rule fires)'}",
        f"  (ii)  growth ratio CROSS/EQUIV = {ratio_growth:.3f} in [0.5, 2.0] "
        f"-> {'PASS' if 0.5 <= ratio_growth <= 2.0 else 'FAIL'}",
        f"  (iii) saturation ratio CROSS/EQUIV = {ratio_sat:.3f} in [0.5, 2.0] "
        f"-> {'PASS' if 0.5 <= ratio_sat <= 2.0 else 'FAIL'}",
        f"  (iv)  frame-wise: CROSS med <= 2x EQUIV p95 from frame 8: "
        f"{len(viol)} violating frames of 123, max ratio {viol_ratio.max():.3f} "
        f"-> {'PASS' if len(viol) == 0 else 'FAIL at frames ' + str(viol[:10].tolist())}",
        "",
        "per-frame table (median [p05,p95] per family, every 10th frame + first 8):",
        " frame        EQUIV                 CROSS                wB",
    ]
    for f in list(range(0, 8)) + list(range(8, NF, 10)) + [NF - 1]:
        e = np.median(equiv[:, f]); ep = np.percentile(equiv[:, f], 95)
        c = np.median(cross[:, f]); cp = np.percentile(cross[:, f], 95)
        w = np.median(out["wB"][:, f])
        lines.append(f"{f:5d}  {e:.3e} [{ep:.1e}]  {c:.3e} [{cp:.1e}]  {w:.3e}")
    report = "\n".join(lines)
    print(report)
    (RUNS / "divergence.txt").write_text(report + "\n")

    # seed table for the widen rule decision
    seeds = {f"{u}-{v}": float(pair_rms(runs[u], runs[v])[1]) for u, v in fams["xAB"]}
    json.dump({"equiv_max_seed": equi_max_seed, "cross_seed_med": cross_seed_med,
               "growth_ratio": ratio_growth, "sat_ratio": ratio_sat,
               "framewise_violations": viol.tolist(),
               "framewise_max_ratio": float(viol_ratio.max()),
               "stats": stats}, open(RUNS / "divergence_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

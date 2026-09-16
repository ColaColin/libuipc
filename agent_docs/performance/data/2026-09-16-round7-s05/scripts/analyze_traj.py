#!/usr/bin/env python3
"""Round-7 s05 solver-trajectory instrument: per-phase Newton/PCG/LS arm
medians + solver state (ccd_toi, ls_alpha), from the arm result JSONs."""
import json
from pathlib import Path

import numpy as np

RUNS = Path("/workspace/output/round7/s05/runs")
A = [f"a{k:02d}" for k in range(1, 11)]
B = [f"b{k:02d}" for k in range(1, 11)]
AP = [f"ap{k}" for k in range(1, 6)]

# phase lengths for 130 frames: F = (.13,.05,.11,.06,.13,.05,.11) -> sums to 0.64
F = (0.13, 0.05, 0.11, 0.06, 0.13, 0.05, 0.11)
L = [max(1, int(f * 130)) for f in F]
names = ["press1", "hold1", "lift1", "shift", "press2", "hold2", "lift2"]


def phases(frame):  # frame is 1-based frame index
    f = frame - 1
    for nm, ln in zip(names, L):
        if f < ln:
            return nm
        f -= ln
    return "settle"


def traj(name):
    d = json.load(open(RUNS / f"{name}.json"))
    fs = d["frame_stats"]
    return {p: ([s["newton_iterations"] for s in fs if phases(s["frame"]) == p],
                [s["linear_solver_iterations"] for s in fs if phases(s["frame"]) == p],
                [s["line_search_trials"] for s in fs if phases(s["frame"]) == p]) for p in names + ["settle"]}


tr = {n: traj(n) for n in A + B + AP}
print(f"{'phase':8s} {'frames':>6s} | {'Newton A':>9s} {'A2':>4s} {'B':>9s} | {'PCG A':>9s} {'B':>9s} | {'LS A':>6s} {'B':>6s}")
for p in names + ["settle"]:
    na = np.concatenate([tr[n][p][0] for n in A]) if any(tr[n][p][0] for n in A) else []
    nb = np.concatenate([tr[n][p][0] for n in B]) if any(tr[n][p][0] for n in B) else []
    npa = np.concatenate([tr[n][p][1] for n in A]) if any(tr[n][p][1] for n in A) else []
    npb = np.concatenate([tr[n][p][1] for n in B]) if any(tr[n][p][1] for n in B) else []
    la = np.concatenate([tr[n][p][2] for n in A]) if any(tr[n][p][2] for n in A) else []
    lb = np.concatenate([tr[n][p][2] for n in B]) if any(tr[n][p][2] for n in B) else []
    nf = len(tr["a01"][p][0])
    print(f"{p:8s} {nf:6d} | {np.mean(na) if len(na) else 0:9.3f} {'':4s} {np.mean(nb) if len(nb) else 0:9.3f} | "
          f"{np.mean(npa) if len(npa) else 0:9.1f} {np.mean(npb) if len(npb) else 0:9.1f} | "
          f"{np.mean(la) if len(la) else 0:6.2f} {np.mean(lb) if len(lb) else 0:6.2f}")

# totals ranges per arm
print()
for label, grp in (("A (exact)", A), ("A'' (pert-exact)", AP), ("B (GN)", B)):
    nt = [sum(s["newton_iterations"] for s in json.load(open(RUNS / f"{n}.json"))["frame_stats"]) for n in grp]
    pt = [sum(s["linear_solver_iterations"] for s in json.load(open(RUNS / f"{n}.json"))["frame_stats"]) for n in grp]
    lt = [sum(s["line_search_trials"] for s in json.load(open(RUNS / f"{n}.json"))["frame_stats"]) for n in grp]
    print(f"{label:20s} newton [{min(nt)}, {max(nt)}] med {int(np.median(nt))} | "
          f"pcg [{min(pt)}, {max(pt)}] med {int(np.median(pt))} | ls [{min(lt)}, {max(lt)}] med {int(np.median(lt))}")

# solver state
print()
for label, grp in (("A", A), ("A''", AP), ("B", B)):
    toi, alpha, toi_lt1, alpha_lt1 = [], [], 0, 0
    for n in grp:
        fs = json.load(open(RUNS / f"{n}.json"))["frame_stats"]
        t = [s["last_ccd_toi"] for s in fs]
        al = [s["last_line_search_alpha"] for s in fs]
        toi.append(min(t)); alpha.append(min(al))
        toi_lt1 += sum(1 for x in t if x < 1.0)
        alpha_lt1 += sum(1 for x in al if x < 1.0)
    print(f"{label:4s} ccd_toi_min med {np.median(toi):.3f} [{min(toi):.3f},{max(toi):.3f}]  "
          f"frames toi<1: {toi_lt1 / len(grp):.1f}/run | ls_alpha_min med {np.median(alpha):.3f}  "
          f"frames alpha<1: {alpha_lt1 / len(grp):.1f}/run")

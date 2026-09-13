#!/usr/bin/env python3
"""Per-frame trajectory divergence between every pair of V1 runs.

d(i,j,f) = rms over cloth vertices of |P_i(f) - P_j(f)|.

Families:
  EQUIV  physically equivalent pairs: A-A, Ap-Ap, A-Ap  (45 pairs)
  CROSS  A-B and Ap-B                                    (50 pairs)
  BB     B-B                                             (10 pairs)
The pre-registered test is: the CROSS curves must lie inside the EQUIV envelope.
"""
import itertools, json, math
from pathlib import Path
import numpy as np

R = Path("/workspace/output/round6/v1/runs")
tags = sorted(p.stem for p in R.glob("*.npy"))
X = {t: np.load(R / f"{t}.npy") for t in tags}
nf = min(v.shape[0] for v in X.values())
print("runs:", tags, "frames:", nf)

def rms(a, b):
    d = a[:nf] - b[:nf]
    return np.sqrt((d * d).sum(axis=2).mean(axis=1))

curves = {}
for i, j in itertools.combinations(tags, 2):
    curves[f"{i}|{j}"] = rms(X[i], X[j])

def fam(k):
    a, b = (s.split("_")[0] for s in k.split("|"))
    s = {a, b}
    if s == {"A"}:
        return "AA"
    if s == {"Ap"}:
        return "ApAp"
    if s == {"App"}:
        return "AppApp"
    if s <= {"A", "Ap"}:
        return "AAp"
    if s <= {"A", "Ap", "App"}:
        return "AApp"
    if s == {"B"}:
        return "BB"
    return "CROSS"

F = {"AA": [], "ApAp": [], "AppApp": [], "AAp": [], "AApp": [], "CROSS": [], "BB": []}
for k, c in curves.items():
    F[fam(k)].append(c)
F = {k: np.array(v) for k, v in F.items()}

out = {"frames": nf, "tags": tags}
for k, v in F.items():
    out[k] = {"n_pairs": len(v), "min": v.min(axis=0).tolist(),
              "max": v.max(axis=0).tolist(), "median": np.median(v, axis=0).tolist()}

EQ = np.concatenate([F["AA"], F["ApAp"], F["AppApp"], F["AAp"], F["AApp"]])
F["EQUIV"] = EQ
out["EQUIV"] = {"n_pairs": len(EQ), "min": EQ.min(axis=0).tolist(),
                "max": EQ.max(axis=0).tolist(), "median": np.median(EQ, axis=0).tolist()}
lo, hi = EQ.min(axis=0), EQ.max(axis=0)
viol = []
for k, c in curves.items():
    if fam(k) != "CROSS":
        continue
    above = int((c > hi * 1.0).sum()); below = int((c < lo).sum())
    viol.append((k, above, below, float(c[-1])))
out["cross_outside"] = viol

print(f"{'frame':>6s} {'EQUIV min':>11s} {'EQUIV med':>11s} {'EQUIV max':>11s} "
      f"{'CROSS min':>11s} {'CROSS med':>11s} {'CROSS max':>11s} {'BB med':>11s}")
for f in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 90, 120, 150, nf - 1]:
    print(f"{f:6d} {lo[f]:11.3e} {np.median(F['EQUIV'],axis=0)[f]:11.3e} {hi[f]:11.3e} "
          f"{F['CROSS'].min(axis=0)[f]:11.3e} {np.median(F['CROSS'],axis=0)[f]:11.3e} "
          f"{F['CROSS'].max(axis=0)[f]:11.3e} {np.median(F['BB'],axis=0)[f]:11.3e}")

# --- seed size and growth rate: the quantitative form of "chaotic, not biased"
print("\nseed (frame 1 rms displacement, m) and growth")
for k in ("AA", "ApAp", "AppApp", "AAp", "AApp", "CROSS", "BB"):
    v = F[k]
    print(f"  {k:6s} n={len(v):3d}  f1 {np.median(v[:,1]):.3e} [{v[:,1].min():.3e}, {v[:,1].max():.3e}]"
          f"   f3 {np.median(v[:,3]):.3e}   f8 {np.median(v[:,8]):.3e}   f45 {np.median(v[:,45]):.3e}")
lam = {}
for k in ("EQUIV", "CROSS"):
    m = np.median(F[k], axis=0)
    lam[k] = float(np.log(m[8] / m[3]) / 5.0)
print(f"  e-folding rate frames 3->8: EQUIV {lam['EQUIV']:.3f}/frame, CROSS {lam['CROSS']:.3f}/frame")
r = float(np.median(F["CROSS"][:,3]) / np.median(F["EQUIV"][:,3]))
print(f"  CROSS/EQUIV seed ratio at frame 3: {r:.2f}  ->  equivalent head start "
      f"{math.log(r)/lam['EQUIV']:.2f} frames of a {nf-1}-frame run")
n_out = sum(1 for _, a, b, _ in viol if a or b)
cm = np.median(F["CROSS"], axis=0)
outside_med = int(((cm > hi) | (cm < lo)).sum())
print(f"\nframes where the CROSS *median* curve leaves the EQUIV [min,max] envelope: "
      f"{outside_med} of {nf} -> {list(np.nonzero((cm>hi)|(cm<lo))[0][:20])}")
print(f"CROSS pairs with any frame outside the EQUIV envelope: {n_out} of {len(viol)}")
for k, a, b, last in viol:
    if a or b:
        print(f"  {k}: {a} frames above, {b} below, final {last:.4e}")
tail = slice(nf - 45, nf)
for k, v in F.items():
    print(f"saturation (mean over last 45 frames) {k}: "
          f"{v[:, tail].mean():.4e}  [{v[:, tail].mean(axis=1).min():.4e}, "
          f"{v[:, tail].mean(axis=1).max():.4e}]")
Path("/workspace/output/round6/v1/divergence.json").write_text(json.dumps(out))

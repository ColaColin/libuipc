#!/usr/bin/env python3
"""Round-6 V2: the analysis the pre-registered rule in VERDICT_RULE.md asks for.

Reads every `<arm>_<rep>.log` under runs/, pulls VERIFY_RESULT (per-run
observables) and VERIFY_TRACE (per-frame rows), and applies, in order:

  §3 safety items, pass/fail
  §4 the membrane family (the verdict turns on this)
  §5 the run maximum, three ways
  §6 the confirmatory screen, V1's rule
  §7 the cost side
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/workspace/output/round6/v2/runs"
ARMS = ["A", "Ap", "B", "S"]


def load(path):
    res, trace = None, None
    with open(path) as f:
        for line in f:
            if line.startswith("VERIFY_RESULT "):
                res = json.loads(line[len("VERIFY_RESULT "):])
            elif line.startswith("VERIFY_TRACE "):
                trace = json.loads(line[len("VERIFY_TRACE "):])
    return res, trace


def collect():
    runs = defaultdict(list)
    for fn in sorted(os.listdir(RUNS)):
        if not fn.endswith(".log"):
            continue
        arm, rep = fn[:-4].rsplit("_", 1)
        if arm not in ARMS:
            continue
        res, trace = load(os.path.join(RUNS, fn))
        if res is None:
            print(f"!! {fn}: no VERIFY_RESULT")
            continue
        runs[arm].append((int(rep), res, trace))
    for a in runs:
        runs[a].sort()
    return runs


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(idx):
        v = (m - k) * pvals[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj


def desc(v):
    v = np.asarray(v, dtype=float)
    return (f"n={len(v):2d} median={np.median(v):.6g} "
            f"[{v.min():.6g}, {v.max():.6g}] mean={v.mean():.6g} sd={v.std(ddof=1):.3g}")


def mw(b, r):
    try:
        return float(stats.mannwhitneyu(b, r, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def mw1(b, r, greater=True):
    try:
        return float(stats.mannwhitneyu(
            b, r, alternative="greater" if greater else "less").pvalue)
    except ValueError:
        return 1.0


def main():
    runs = collect()
    out = []
    P = out.append
    P("=" * 100)
    P("ROUND-6 V2 -- UIPC_CONTACT_RANK1=1 (contact part 2, PE+PP rank-1 Hessian)")
    P("Rule: agent_docs/performance/data/2026-09-14-round6-v2/VERDICT_RULE.md (committed 877d0cf9)")
    P("=" * 100)
    for a in ARMS:
        P(f"arm {a:3s}: {len(runs.get(a, []))} runs")
    P("")

    # ---------------------------------------------------------------- §3 safety
    P("## §3 SAFETY -- pass/fail, not statistical")
    P("")
    safety_rows = []
    for a in ARMS:
        nf = nrad = nax = nlift = ninv = npen = nconv = nlim = nspd = nok = 0
        worst = defaultdict(lambda: None)
        nframes = 0
        for rep, r, tr in runs.get(a, []):
            nframes += len(tr) if tr else 0
            nf += 0 if r["verify_all_finite"] else 1
            nrad += 0 if r["verify_contained_radial"] else 1
            nax += 0 if r["verify_contained_axial"] else 1
            nlift += 1 if r["verify_lifter_depth_max"] > 0 else 0
            ninv += 1 if r["verify_area_ratio_min"] <= 0 else 0
            npen += 1 if (r["verify_cc_min_dist"] <= 0
                          or r["verify_bore_gap_min"] <= 0) else 0
            nconv += r["verify_not_converged_frames"]
            nlim += (r["verify_hit_newton_limit_frames"]
                     + r["verify_hit_ls_limit_frames"])
            nspd += 1 if r["verify_max_speed"] >= 50.0 else 0
            nok += 0 if r["verify_ok"] else 1
            for k, sign in [("verify_r_max", +1), ("verify_abs_z_max", +1),
                            ("verify_lifter_depth_max", +1),
                            ("verify_area_ratio_min", -1),
                            ("verify_tri_height_min", -1),
                            ("verify_cc_min_dist", -1),
                            ("verify_bore_gap_min", -1),
                            ("verify_max_speed", +1)]:
                v = r[k]
                cur = worst[k]
                if cur is None or (sign > 0 and v > cur) or (sign < 0 and v < cur):
                    worst[k] = v
        safety_rows.append((a, nframes, nf, nrad, nax, nlift, ninv, npen,
                            nconv, nlim, nspd, nok, dict(worst)))
    P(f"{'arm':4s} {'frames':>7s} {'nonfin':>6s} {'radial':>6s} {'axial':>6s} "
      f"{'lifter':>6s} {'invert':>6s} {'penet':>6s} {'nonconv':>7s} {'limits':>6s} "
      f"{'speed':>5s} {'!ok':>4s}")
    for (a, nfr, nf, nrad, nax, nlift, ninv, npen, nconv, nlim, nspd, nok, _) in safety_rows:
        P(f"{a:4s} {nfr:7d} {nf:6d} {nrad:6d} {nax:6d} {nlift:6d} {ninv:6d} "
          f"{npen:6d} {nconv:7d} {nlim:6d} {nspd:5d} {nok:4d}")
    P("")
    P("worst value anywhere in each arm:")
    keys = ["verify_r_max", "verify_abs_z_max", "verify_lifter_depth_max",
            "verify_area_ratio_min", "verify_tri_height_min",
            "verify_cc_min_dist", "verify_bore_gap_min", "verify_max_speed"]
    P(f"{'statistic':28s} " + " ".join(f"{a:>13s}" for a in ARMS))
    for k in keys:
        row = []
        for (a, *_rest, w) in safety_rows:
            row.append(f"{w.get(k, float('nan')):13.6g}")
        P(f"{k:28s} " + " ".join(row))
    P("")

    # --------------------------------------------------- reference / test sets
    def vals(arm, key):
        return np.array([r[key] for _rep, r, _t in runs.get(arm, [])], float)

    def R(key):
        return np.concatenate([vals("A", key), vals("Ap", key)])

    # ---------------------------------------------------------- §4 the membrane
    P("## §4 PRIMARY -- the membrane family (the verdict turns on this)")
    P("   systematic := Holm-p < 0.05 within this family of 4  AND")
    P("                 median(eps_B)/median(eps_R) - 1 > +0.10   (eps = X - 1, areal strain)")
    P("")
    FAM = ["verify_ar_q999_mean", "verify_ar_q99_mean",
           "verify_ar_mean_mean", "verify_ar_total_mean"]
    P("   NULL OF THE NULL -- A vs A' on the same family.  A' carries a perturbation seed")
    P("   larger than the one the rank-1 Hessian injects, so whatever A-vs-A' produces here is")
    P("   what seed size alone can do to these statistics.")
    for k in FAM:
        a, ap = vals("A", k), vals("Ap", k)
        if len(a) and len(ap):
            ea, ep = np.median(a) - 1, np.median(ap) - 1
            P(f"   {k:24s} A median {np.median(a):.6f}  A' median {np.median(ap):.6f}"
              f"  p={mw(ap, a):.4g}"
              f"  relative strain shift {100*(ep/ea-1) if ea else float('nan'):+.2f} %")
    P("")
    praw, rows = [], []
    for k in FAM:
        r, b, s = R(k), vals("B", k), vals("S", k)
        p = mw(b, r)
        praw.append(p)
        er, eb, es = np.median(r) - 1, np.median(b) - 1, np.median(s) - 1
        rel_b = (eb / er - 1) if er != 0 else float("nan")
        rel_s = (es / er - 1) if er != 0 else float("nan")
        rows.append((k, r, b, s, p, mw1(b, r, True), rel_b, rel_s, mw(s, r)))
    padj = holm(np.array(praw))
    for (k, r, b, s, p, p1, rel_b, rel_s, ps), pa in zip(rows, padj):
        P(f"{k}")
        P(f"   R (A+Ap) {desc(r)}")
        P(f"   B        {desc(b)}")
        P(f"   S (ship) {desc(s)}")
        P(f"   strain eps=X-1:  median R {np.median(r)-1:.6g}  B {np.median(b)-1:.6g}"
          f"  S {np.median(s)-1:.6g}")
        P(f"   B vs R: p={p:.4g}  Holm={pa:.4g}  one-sided(up) p={p1:.4g}"
          f"  relative strain shift {rel_b*100:+.2f} %")
        P(f"   S vs R: p={ps:.4g}  relative strain shift {rel_s*100:+.2f} %   [positive control]")
        P(f"   absolute median shift: B-R {np.median(b)-np.median(r):+.3e}"
          f"   S-R {np.median(s)-np.median(r):+.3e}")
        if np.median(r) - 1 <= 0:
            P("   NOTE: median(eps_R) <= 0 (the membrane is, on average, very slightly "
              "*compressed*),")
            P("         so the rule's relative-strain effect size is degenerate for this "
              "statistic.")
            P("         The rule is not changed: the significance half still applies and the "
              "absolute shift is reported above.")
        flag = (pa < 0.05) and (rel_b > 0.10)
        P(f"   => {'SYSTEMATIC' if flag else 'ok'}")
        P("")

    # ------------------------------------------------ §5 the run maximum, 3 ways
    P("## §5 SECONDARY -- verify_area_ratio_max, three ways (confirmatory only)")
    P("")
    k = "verify_area_ratio_max"
    r, b, s = R(k), vals("B", k), vals("S", k)
    P("(1) run maxima")
    P(f"   R {desc(r)}")
    P(f"   B {desc(b)}")
    P(f"   S {desc(s)}")
    P(f"   R sorted: {np.sort(r)}")
    P(f"   B sorted: {np.sort(b)}")
    P(f"   S sorted: {np.sort(s)}")
    P(f"   B vs R MW p={mw(b, r):.4g}  one-sided(up) p={mw1(b, r):.4g}"
      f"   Welch p={stats.ttest_ind(b, r, equal_var=False).pvalue:.4g}")
    P(f"   S vs R MW p={mw(s, r):.4g}")
    P(f"   envelope: median(B)={np.median(b):.4f} inside R [{r.min():.4f},{r.max():.4f}] "
      f"-> {'YES' if r.min() <= np.median(b) <= r.max() else 'NO'}")
    P(f"   range ratio B/R = {(b.max()-b.min())/(r.max()-r.min()):.3f}")
    P("")

    P("(2) pooled per-frame frame-max distribution")
    pooled = {}
    for a in ARMS:
        v = []
        for _rep, _r, tr in runs.get(a, []):
            if tr:
                v += [x["area_ratio_max"] for x in tr]
        pooled[a] = np.array(v, float)
    pooled["R"] = np.concatenate([pooled["A"], pooled["Ap"]])
    P(f"{'arm':4s} {'n':>7s} {'q50':>9s} {'q95':>9s} {'q99':>9s} {'q99.9':>9s} "
      f"{'max':>9s} {'%>1.4':>8s}")
    for a in ["A", "Ap", "R", "B", "S"]:
        v = pooled[a]
        if not len(v):
            continue
        q = np.quantile(v, [0.5, 0.95, 0.99, 0.999])
        P(f"{a:4s} {len(v):7d} {q[0]:9.4f} {q[1]:9.4f} {q[2]:9.4f} {q[3]:9.4f} "
          f"{v.max():9.4f} {100*(v > 1.4).mean():8.3f}")
    if len(pooled["B"]) and len(pooled["R"]):
        P(f"   B vs R pooled-frame MW p={mw(pooled['B'], pooled['R']):.4g}")
        P(f"   S vs R pooled-frame MW p={mw(pooled['S'], pooled['R']):.4g}")
    P("   CAVEAT, and it matters: frames within a run are strongly autocorrelated, so the")
    P("   pooled-frame Mann-Whitney treats ~180 dependent frames as 180 independent draws and")
    P("   its p-value is not valid.  The rule pre-registered this test; the honest version is")
    P("   the run-level one below, where each run contributes ONE value per quantile.")
    P("")
    P("(2b) run-level: each run's own frame-max quantiles, compared across runs")
    for qq, lab in [(0.5, "q50"), (0.95, "q95"), (0.99, "q99"), (1.0, "max")]:
        per = {}
        for a in ARMS:
            v = []
            for _rep, _r, tr in runs.get(a, []):
                if tr:
                    fm = np.array([x["area_ratio_max"] for x in tr])
                    v.append(fm.max() if qq == 1.0 else float(np.quantile(fm, qq)))
            per[a] = np.array(v)
        rr = np.concatenate([per["A"], per["Ap"]])
        if len(per["B"]) and len(rr):
            P(f"   frame-max {lab}: R median {np.median(rr):.5f} "
              f"[{rr.min():.5f},{rr.max():.5f}]  B median {np.median(per['B']):.5f} "
              f"[{per['B'].min():.5f},{per['B'].max():.5f}]  S median "
              f"{np.median(per['S']):.5f}   B vs R p={mw(per['B'], rr):.4g}"
              f"   S vs R p={mw(per['S'], rr):.4g}")
    P("")

    P("(3) tail rate: frames with frame-max > 1.4")
    for thr in (1.2, 1.4, 1.6, 1.8):
        nb = int((pooled["B"] > thr).sum()); NB = len(pooled["B"])
        nr = int((pooled["R"] > thr).sum()); NR = len(pooled["R"])
        ns = int((pooled["S"] > thr).sum()); NS = len(pooled["S"])
        if NB and NR:
            odds, pf = stats.fisher_exact([[nb, NB - nb], [nr, NR - nr]])
            _o2, pf2 = stats.fisher_exact([[ns, NS - ns], [nr, NR - nr]])
            rate_b = nb / NB if NB else float('nan')
            rate_r = nr / NR if NR else float('nan')
            P(f"   >{thr}: B {nb}/{NB} ({100*rate_b:.3f} %)  R {nr}/{NR} ({100*rate_r:.3f} %)"
              f"  S {ns}/{NS}   rate ratio B/R = "
              f"{(rate_b/rate_r if rate_r else float('nan')):.3f}"
              f"  Fisher B:R p={pf:.4g}  S:R p={pf2:.4g}")
    P("")

    P("(4) is the extreme a stretched membrane or one triangle in one frame?")
    P("   for every frame whose frame-max exceeds 1.4: how long the excursion lasts, and")
    P("   how many triangles are above 1.4 at the same time.")
    for a in ARMS:
        runs_a = runs.get(a, [])
        if not runs_a:
            continue
        excursions, lens, widths = 0, [], []
        for _rep, _r, tr in runs_a:
            if not tr:
                continue
            over = [x["area_ratio_max"] > 1.4 for x in tr]
            i = 0
            while i < len(over):
                if over[i]:
                    j = i
                    while j < len(over) and over[j]:
                        j += 1
                    excursions += 1
                    lens.append(j - i)
                    widths += [tr[k]["ar_n_gt_1p4"] for k in range(i, j)]
                    i = j
                else:
                    i += 1
        if excursions:
            P(f"   {a:3s} {excursions} excursions over 1.4 in {len(runs_a)} runs; "
              f"length in frames {sorted(lens)}; triangles simultaneously above 1.4 "
              f"median {int(np.median(widths))} max {max(widths)}")
        else:
            P(f"   {a:3s} no frame above 1.4 in {len(runs_a)} runs")
    P("")
    P("   where the run maximum happens (frame, garment):")
    for a in ARMS:
        fr = [r["verify_ar_argmax_frame"] for _x, r, _t in runs.get(a, [])]
        gm = [r["verify_ar_argmax_garment"] for _x, r, _t in runs.get(a, [])]
        if fr:
            P(f"   {a:3s} frames {sorted(fr)}")
            P(f"   {a:3s} garment histogram {np.bincount(gm, minlength=4)}"
              f"  (0 towel 1 pillowcase 2 shorts 3 washcloth)")
    P("")

    # ------------------------------------------------------- §6 the wide screen
    P("## §6 CONFIRMATORY SCREEN -- V1's rule (raw p<0.05 AND |dmedian| > 0.5*range(R))")
    P("")
    SCREEN = ["verify_mean_disp_mm_last_quarter", "verify_area_ratio_max",
              "verify_area_ratio_min", "verify_tri_height_min",
              "verify_cc_min_dist", "verify_bore_gap_min", "verify_r_max",
              "verify_abs_z_max", "verify_lifter_depth_max", "verify_max_speed",
              "verify_ke_mean", "verify_ke_max", "verify_e_tot_last",
              "verify_e_tot_max", "verify_mean_disp_mm",
              "verify_drum_track_err_deg_max", "verify_newton_total",
              "verify_pcg_total", "verify_line_search_total",
              "verify_ccd_toi_min", "verify_ccd_toi_clamped_frames",
              "verify_ls_alpha_cut_frames", "verify_newton_mean",
              "verify_pcg_mean", "verify_ar_q999_max", "verify_ar_q99_max",
              "verify_ar_total_max", "verify_ar_n_gt_1p2_total",
              "verify_ar_n_gt_1p4_total", "verify_ar_n_gt_1p6_total",
              "verify_ar_frames_gt_1p4", "verify_gpu_mem_proc_max_mib"]
    for g in range(4):
        SCREEN.append(f"centroid_turn_{g}")

    def getv(arm, key):
        if key.startswith("centroid_turn_"):
            g = int(key.rsplit("_", 1)[1])
            return np.array([r["verify_centroid_turn_deg"][g]
                             for _x, r, _t in runs.get(arm, [])], float)
        return vals(arm, key)

    rows, praw = [], []
    for k in SCREEN:
        r = np.concatenate([getv("A", k), getv("Ap", k)])
        b, s = getv("B", k), getv("S", k)
        if not len(b) or not len(r):
            continue
        p = mw(b, r)
        praw.append(p)
        rng = r.max() - r.min()
        shift = abs(np.median(b) - np.median(r))
        rows.append((k, r, b, s, p, shift / rng if rng else 0.0))
    padj = holm(np.array(praw))
    P(f"{'observable':34s} {'median R':>11s} {'median B':>11s} {'median S':>11s} "
      f"{'shift/rng':>9s} {'p':>9s} {'Holm':>8s} {'inR?':>5s} flag")
    for (k, r, b, s, p, sr), pa in zip(rows, padj):
        inR = r.min() <= np.median(b) <= r.max()
        flag = "SYSTEMATIC" if (p < 0.05 and sr > 0.5) else ""
        P(f"{k:34s} {np.median(r):11.6g} {np.median(b):11.6g} "
          f"{(np.median(s) if len(s) else float('nan')):11.6g} {sr:9.3f} "
          f"{p:9.4g} {pa:8.3g} {'yes' if inR else 'NO':>5s} {flag}")
    P("")

    # -------------------------------------------------------- joint signature
    P("## §6 JOINT SIGNATURE (mean_disp_last_quarter up AND a membrane statistic up,")
    P("   both raw p<0.05) -- the combination s07 flagged")
    k = "verify_mean_disp_mm_last_quarter"
    r, b = R(k), vals("B", k)
    pd_ = mw(b, r)
    up_disp = pd_ < 0.05 and np.median(b) > np.median(r)
    P(f"   {k}: median R {np.median(r):.4f} B {np.median(b):.4f} p={pd_:.4g}"
      f"  -> {'UP' if up_disp else 'not up'}")
    mem_up = []
    for kk in FAM:
        rr, bb = R(kk), vals("B", kk)
        pp = mw(bb, rr)
        if pp < 0.05 and np.median(bb) > np.median(rr):
            mem_up.append(kk)
    P(f"   membrane statistics up with raw p<0.05: {mem_up or 'none'}")
    P(f"   => joint signature {'PRESENT' if (up_disp and mem_up) else 'ABSENT'}")
    P("")

    # ---------------------------------------------------------------- §7 cost
    P("## §7 COST -- iteration counts on the tumbler")
    for k in ["verify_newton_total", "verify_pcg_total", "verify_line_search_total"]:
        rr = R(k)
        P(f"{k}")
        for a in ARMS:
            v = vals(a, k)
            if len(v):
                P(f"   {a:3s} {desc(v)}")
        b = vals("B", k)
        if len(b) and len(rr):
            P(f"   B vs R: {100*(np.mean(b)/np.mean(rr)-1):+.2f} % of the mean, "
              f"MW p={mw(b, rr):.4g}")
    P("")
    print("\n".join(out))


if __name__ == "__main__":
    main()

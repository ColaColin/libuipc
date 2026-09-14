#!/usr/bin/env python
"""s07 multi-arm interleaved benchmark sweep.

Same hard checks and the same statistics as round 6's shared `ab.py` (which it is copied
from), but with N arms instead of 2, because this step's question -- "is the +1-2 % Newton
drift real, and which branch causes it?" -- needs four or five arms measured against the
SAME scene state, and running 2-arm A/Bs pairwise would cost 2.5x the GPU time and compare
arms that never ran interleaved.

Ordering: the arm order is ROTATED each rep (rep k starts at arm k mod A) and REVERSED on
odd reps, so no arm keeps a fixed position in the sequence -- the N-arm generalisation of
ab.py's ABBA. One discarded warm-up per arm, as ab.py does.

The null arm is a second copy of the default (`UIPC_CONTACT_RANK1=00`, which `std::atoi`
reads as 0): it executes bit-identical code to the `=0` arm, so its delta IS the scene's
own null envelope, measured in the same sweep as the effects.
"""
from __future__ import annotations
import argparse, itertools, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

REPO = Path("/workspace/deps/libuipc-src")
PY_ = os.environ.get("UIPC_PERF_PY", "/workspace/deps/uipc-perf-env/bin/python")


def run_once(scene, env_pairs, frames):
    cmd = [PY_, "scripts/run_benchmark.py", "run", scene, "--python", PY_]
    for p in env_pairs:
        cmd += ["--env", p]
    if frames:
        cmd += ["--frames", str(frames)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"FAILED: run_benchmark.py exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    latest = REPO / "output" / "benchmark-runs" / f"{scene}.json"
    if not latest.exists():
        sys.exit(f"FAILED: no metadata at {latest}")
    return json.loads(latest.read_text())


def summarise(meta):
    if meta.get("returnCode") != 0:
        sys.exit(f"FAILED: benchmark returnCode {meta.get('returnCode')}")
    stats = meta["reportedBenchmark"]["frame_stats"]
    newton = sum(s["newton_iterations"] for s in stats)
    pcg = sum(s["linear_solver_iterations"] for s in stats)
    ls = sum(s.get("line_search_trials", 0) for s in stats)
    t = meta["reportedFrameTiming"]
    frames = meta["frames"]
    return {"mean_ms": t["meanFrameMs"], "median_ms": t["medianFrameMs"], "p95_ms": t["p95FrameMs"],
            "frames": frames, "newton": newton, "pcg": pcg, "line_search": ls,
            "ms_per_newton": t["meanFrameMs"] * frames / newton if newton else float("nan"),
            "ms_per_pcg": t["meanFrameMs"] * frames / pcg if pcg else float("nan"),
            "converged": all(s.get("converged", True) for s in stats),
            "hit_newton_limit": any(s.get("hit_newton_limit", False) for s in stats),
            "commit": meta["revisions"]["libuipc"]["commit"],
            "samples_commit": meta["revisions"]["libuipc-samples"]["commit"],
            "dirty": meta["revisions"]["libuipc"]["dirty"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--label", default="abn")
    ap.add_argument("--out", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--baseline", default=None, help="arm name used as the reference in the table")
    ap.add_argument("--arm", nargs="+", action="append", required=True)
    args = ap.parse_args()

    arms = [(a[0], a[1:]) for a in args.arm]
    for _, ps in arms:
        for p in ps:
            if "=" not in p:
                sys.exit(f"--arm switches must be KEY=VALUE: {p}")
    base = args.baseline or arms[0][0]
    out = Path(args.out or f"/workspace/output/round6/s07/{args.label}")
    out.mkdir(parents=True, exist_ok=True)

    results = {n: [] for n, _ in arms}
    warm = []
    t0 = time.time()
    for w in range(args.warmup):
        for name, ep in arms:
            s = summarise(run_once(args.scene, ep, args.frames))
            s["arm"], s["rep"], s["warmup"] = name, -1 - w, True
            warm.append(s)
            print(f"[{time.time()-t0:7.1f}s] warmup {name:>6}: {s['mean_ms']:8.3f} ms  Newton {s['newton']}", flush=True)

    A = len(arms)
    for rep in range(1, args.n + 1):
        order = arms[(rep - 1) % A:] + arms[:(rep - 1) % A]
        if rep % 2 == 0:
            order = list(reversed(order))
        for name, ep in order:
            meta = run_once(args.scene, ep, args.frames)
            shutil.copy(REPO / "output" / "benchmark-runs" / f"{args.scene}.json",
                        out / f"{args.label}_{args.scene}_{name}_r{rep}.json")
            s = summarise(meta)
            recorded = meta["environment"]["overrides"]
            asked = dict(p.split("=", 1) for p in ep)
            if recorded != asked:
                sys.exit(f"FAILED: metadata says overrides {recorded}, arm asked for {asked}")
            s["rep"], s["arm"], s["env"] = rep, name, ep
            results[name].append(s)
            print(f"[{time.time()-t0:7.1f}s] rep {rep:2d} {name:>6}: {s['mean_ms']:8.3f} ms  "
                  f"Newton {s['newton']}  PCG {s['pcg']}  LS {s['line_search']}", flush=True)

    every = [s for v in results.values() for s in v]
    if len({s["commit"] for s in every}) != 1 or len({s["samples_commit"] for s in every}) != 1:
        sys.exit("FAILED: the tree moved during the sweep")
    if len({s["frames"] for s in every}) != 1:
        sys.exit("FAILED: frame counts differ between runs")
    if len({s["dirty"] for s in every}) != 1:
        print("WARNING: `dirty` changed between runs")
    if any(s["hit_newton_limit"] for s in every):
        print("WARNING: some frames hit the Newton limit")

    from scipy import stats as st
    report = {"scene": args.scene, "label": args.label, "n": args.n, "baseline": base,
              "commit": every[0]["commit"], "frames": every[0]["frames"], "arms": results,
              "warmup_discarded": warm, "table": {}}
    print(f"\n=== {args.label}  {args.scene}  n={args.n} per arm, {len(arms)} arms, baseline={base} ===")
    for key in ("newton", "pcg", "line_search", "mean_ms", "median_ms", "ms_per_newton", "ms_per_pcg"):
        print(f"\n-- {key}")
        a = [r[key] for r in results[base]]
        for name, _ in arms:
            b = [r[key] for r in results[name]]
            disjoint = max(a) < min(b) or max(b) < min(a)
            t, p = st.ttest_ind(a, b, equal_var=False) if name != base else (0.0, 1.0)
            d = (statistics.mean(b) - statistics.mean(a)) / statistics.mean(a) * 100.0
            print(f"   {name:>6}: mean {statistics.mean(b):11.4f} sd {statistics.pstdev(b):8.4f} "
                  f"[{min(b):.4f}, {max(b):.4f}]  {d:+7.2f}%  "
                  f"{'DISJOINT' if disjoint and name != base else 'overlapping'}  t={t:+.2f} p={p:.3g}")
            report["table"].setdefault(key, {})[name] = {
                "mean": statistics.mean(b), "sd": statistics.pstdev(b), "min": min(b), "max": max(b),
                "delta_pct": d, "disjoint": bool(disjoint and name != base), "t": float(t), "p": float(p)}
    (out / f"{args.label}_{args.scene}_summary.json").write_text(json.dumps(report, indent=1))
    print(f"\nraw runs + summary: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Round-6 V3: interleaved base-vs-head benchmark runs across TWO builds + the statistics.

ab.py (the round's shared harness) drives one build through env switches; this is its
two-build equivalent. Each arm is a *python interpreter* -- a venv holding one build's
pyuipc -- and both arms run the same scene code from the same checkout through
scripts/run_benchmark.py, so the workload is identical and only the engine differs.

    $UIPC_PERF_PY ab2.py --scene tumbler-garments --n 20 --label v3 \
        --arm base /workspace/output/round6/base/venv/bin/python \
        --arm head /workspace/deps/uipc-perf-env/bin/python

Instrument checks (all fail loudly, PERF_METHOD §2.7 / §2.8):
  * returnCode 0 and the manifest's frame count on every run;
  * the metadata's `python` is the arm's interpreter (the arm really ran that build);
  * every run of an arm loaded the same native library (md5 of pyuipc's `_native`
    directory, taken once per arm before the sweep and again after it);
  * the driver checkout did not move (libuipc + libuipc-samples commit identical in
    every run) and its `dirty` flag did not change;
  * the two arms' native libraries differ from each other (otherwise it is a null).
Ordering: one discarded warm-up per arm, then ABBA blocks, exactly as ab.py does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspace/deps/libuipc-src")
DRIVER_PY = os.environ.get("UIPC_PERF_PY", "/workspace/deps/uipc-perf-env/bin/python")


def native_fingerprint(py: str) -> dict:
    """Where this interpreter's uipc package lives, and md5 of its native libraries."""
    code = ("import uipc, os, json; d=os.path.join(os.path.dirname(uipc.__file__), '_native'); "
            "print(json.dumps({'uipc_file': uipc.__file__, 'native_dir': d, "
            "'files': sorted(os.listdir(d))}))")
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, cwd="/")
    if out.returncode != 0:
        sys.exit(f"FAILED: {py} cannot import uipc:\n{out.stderr}")
    info = json.loads(out.stdout.strip().splitlines()[-1])
    md5 = {}
    for f in info["files"]:
        p = Path(info["native_dir"]) / f
        if p.suffix == ".so" or ".so." in f:
            md5[f] = hashlib.md5(p.read_bytes()).hexdigest()
    info["md5"] = md5
    return info


def run_once(scene: str, py: str, env_pairs: list[str], frames: int | None) -> dict:
    cmd = [DRIVER_PY, "scripts/run_benchmark.py", "run", scene, "--python", py]
    for pair in env_pairs:
        cmd += ["--env", pair]
    if frames:
        cmd += ["--frames", str(frames)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"FAILED: run_benchmark.py exited {proc.returncode}\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    latest = REPO / "output" / "benchmark-runs" / f"{scene}.json"
    if not latest.exists():
        sys.exit(f"FAILED: no metadata at {latest}")
    return json.loads(latest.read_text())


def summarise(meta: dict) -> dict:
    if meta.get("returnCode") != 0:
        sys.exit(f"FAILED: benchmark returnCode {meta.get('returnCode')}")
    stats = meta["reportedBenchmark"]["frame_stats"]
    frames = meta["frames"]
    if len(stats) != frames:
        sys.exit(f"FAILED: {len(stats)} frame_stats for {frames} frames")
    newton = sum(s["newton_iterations"] for s in stats)
    pcg = sum(s["linear_solver_iterations"] for s in stats)
    ls = sum(s.get("line_search_trials", 0) for s in stats)
    t = meta["reportedFrameTiming"]
    return {
        "mean_ms": t["meanFrameMs"],
        "median_ms": t["medianFrameMs"],
        "p95_ms": t["p95FrameMs"],
        "frames": frames,
        "newton": newton,
        "pcg": pcg,
        "line_search": ls,
        "newton_per_frame": newton / frames,
        "pcg_per_frame": pcg / frames,
        "ms_per_newton": t["meanFrameMs"] * frames / newton if newton else float("nan"),
        "ms_per_pcg": t["meanFrameMs"] * frames / pcg if pcg else float("nan"),
        "converged": all(s.get("converged", True) for s in stats),
        "hit_newton_limit": any(s.get("hit_newton_limit", False) for s in stats),
        "commit": meta["revisions"]["libuipc"]["commit"],
        "samples_commit": meta["revisions"]["libuipc-samples"]["commit"],
        "dirty": meta["revisions"]["libuipc"]["dirty"],
        "python": meta["python"],
        "peak_mib": (meta.get("gpuMemory", {}).get("peakDeltaMiB") or [None])[0],
        "duration_s": meta["durationSeconds"],
        "run_id": meta["runId"],
    }


def pct(a: float, b: float) -> float:
    return (b - a) / a * 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--label", default="v3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE applied to BOTH arms")
    ap.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "PYTHON"),
                    help="repeat twice: the base arm first, then the head arm")
    args = ap.parse_args()
    if len(args.arm) != 2:
        sys.exit("give exactly two --arm NAME PYTHON groups: base first, then head")
    arms = [(name, str(Path(py).absolute())) for name, py in args.arm]
    out = Path(args.out or f"/workspace/output/round6/v3/{args.label}")
    out.mkdir(parents=True, exist_ok=True)

    # --- fingerprint both builds before anything runs -------------------------
    fp_before = {name: native_fingerprint(py) for name, py in arms}
    for name, py in arms:
        print(f"arm {name:>5}: {py}\n           uipc at {fp_before[name]['uipc_file']}")
        for f, h in sorted(fp_before[name]["md5"].items()):
            print(f"           {h}  {f}")
    (a, _), (b, _) = arms
    if fp_before[a]["md5"] == fp_before[b]["md5"]:
        sys.exit("FAILED: both arms load byte-identical native libraries -- this would be a null A/B")
    same = [f for f in fp_before[a]["md5"] if fp_before[a]["md5"].get(f) == fp_before[b]["md5"].get(f)]
    print(f"libraries identical between arms: {same}")

    results: dict[str, list[dict]] = {name: [] for name, _ in arms}
    warm: list[dict] = []
    t0 = time.time()
    for w in range(args.warmup):
        for name, py in arms:
            s = summarise(run_once(args.scene, py, args.env, args.frames))
            s["rep"], s["arm"], s["warmup"] = -1 - w, name, True
            warm.append(s)
            print(f"[{time.time()-t0:7.1f}s] warmup  {name:>5}: {s['mean_ms']:.3f} ms/frame (discarded)", flush=True)

    for rep in range(1, args.n + 1):
        order = arms if rep % 2 else list(reversed(arms))
        for name, py in order:
            meta = run_once(args.scene, py, args.env, args.frames)
            raw = out / f"{args.label}_{args.scene}_{name}_r{rep}.json"
            shutil.copy(REPO / "output" / "benchmark-runs" / f"{args.scene}.json", raw)
            s = summarise(meta)
            if s["python"] != py:
                sys.exit(f"FAILED: metadata says python {s['python']}, arm {name} asked for {py}")
            recorded = meta["environment"]["overrides"]
            asked = dict(p.split("=", 1) for p in args.env)
            if recorded != asked:
                sys.exit(f"FAILED: metadata overrides {recorded} != asked {asked}")
            s["rep"], s["arm"] = rep, name
            results[name].append(s)
            print(f"[{time.time()-t0:7.1f}s] rep {rep:2d} {name:>5}: {s['mean_ms']:8.3f} ms/frame  "
                  f"median {s['median_ms']:8.3f}  Newton {s['newton']:5d}  PCG {s['pcg']:6d}  "
                  f"peak {s['peak_mib']} MiB", flush=True)

    # --- instrument checks -------------------------------------------------
    fp_after = {name: native_fingerprint(py) for name, py in arms}
    for name, _ in arms:
        if fp_after[name]["md5"] != fp_before[name]["md5"]:
            sys.exit(f"FAILED: arm {name}'s native libraries changed during the sweep")
    every = [s for v in results.values() for s in v]
    commits = {s["commit"] for s in every}
    scommits = {s["samples_commit"] for s in every}
    frames = {s["frames"] for s in every}
    if len(commits) != 1 or len(scommits) != 1:
        sys.exit(f"FAILED: the driver checkout moved -- libuipc {commits}, samples {scommits}")
    if len(frames) != 1:
        sys.exit(f"FAILED: frame counts differ between runs: {frames}")
    if len({s["dirty"] for s in every}) != 1:
        print("WARNING: `dirty` changed between runs")
    if any(s["hit_newton_limit"] for s in every):
        print("WARNING: some frames hit the Newton limit")

    from scipy import stats as st

    report = {"scene": args.scene, "label": args.label, "n": args.n, "frames": frames.pop(),
              "driver_commit": commits.pop(), "samples_commit": scommits.pop(),
              "env": args.env, "fingerprints": fp_before, "arms": {}, "verdict": {}}
    print(f"\n=== {args.label}  {args.scene}  n={args.n} per arm  ({a} -> {b}) ===")
    keys = ("mean_ms", "median_ms", "ms_per_newton", "ms_per_pcg", "newton", "pcg",
            "newton_per_frame", "pcg_per_frame", "line_search", "peak_mib")
    for key in keys:
        xa = [r[key] for r in results[a]]
        xb = [r[key] for r in results[b]]
        if any(v is None for v in xa + xb):
            continue
        disjoint = max(xa) < min(xb) or max(xb) < min(xa)
        ma, mb = statistics.mean(xa), statistics.mean(xb)
        line = (f"{key:>16}: {a} {ma:10.4f} [{min(xa):.4f}, {max(xa):.4f}] med {statistics.median(xa):.4f}   "
                f"{b} {mb:10.4f} [{min(xb):.4f}, {max(xb):.4f}] med {statistics.median(xb):.4f}   "
                f"{pct(ma, mb):+7.2f}%  {'DISJOINT' if disjoint else 'overlapping'}")
        v = {"base_mean": ma, "head_mean": mb, "base_median": statistics.median(xa),
             "head_median": statistics.median(xb), "base_min": min(xa), "base_max": max(xa),
             "head_min": min(xb), "head_max": max(xb), "delta_pct": pct(ma, mb) if ma else 0.0,
             "disjoint": disjoint}
        if args.n > 1 and (len(set(xa)) > 1 or len(set(xb)) > 1):
            t, p = st.ttest_ind(xa, xb, equal_var=False)
            line += f"  t={t:+.2f} p={p:.3g}"
            v["t"], v["p"] = float(t), float(p)
            if key in ("mean_ms", "median_ms"):
                u = st.mannwhitneyu(xa, xb, alternative="two-sided")
                v["mwu_p"] = float(u.pvalue)
                line += f" MWU p={u.pvalue:.3g}"
        report["verdict"][key] = v
        print(line)

    wall = abs(report["verdict"]["mean_ms"]["delta_pct"])
    for key in ("newton", "pcg"):
        moved = abs(report["verdict"][key]["delta_pct"])
        if moved > wall / 2 and moved > 0.5:
            print(f"\n!! {key} moved {moved:.2f}% against a {wall:.2f}% wall change -- iteration "
                  f"counts are part of the effect; report them with the wall number.")
    report["arms"] = results
    report["warmup_discarded"] = warm
    (out / f"{args.label}_{args.scene}_summary.json").write_text(json.dumps(report, indent=1))
    print(f"\nraw runs + summary: {out}")


if __name__ == "__main__":
    main()

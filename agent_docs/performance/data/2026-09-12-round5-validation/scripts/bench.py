#!/usr/bin/env python3
"""Run a scene N times under a given tree/python/env and summarise observables."""
import argparse, json, os, statistics, subprocess, sys, time

SCENES = {
    "wb":  ("6_wrecking_balls", 120),
    "cwc": ("93_cube_wall_cloth", 120),
    "c2":  ("88_stiff_gipc_benchmark", 250),
    "mb":  ("89_mas_bunny", 100),
}
TREES = {
    "head": ("/workspace/deps/libuipc-src", "/workspace/deps/uipc-perf-env/bin/python"),
    "base": ("/workspace/deps/libuipc-valbase", "/workspace/deps/uipc-valbase-env/bin/python"),
}

def run_once(tree, py, scene_dir, frames, extra_env):
    env = dict(os.environ)
    env["PYTHONPATH"] = "/workspace/archive/libuipc-samples/shim" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_HOME"] = "/workspace/deps/cuda-12.8"
    env["LD_LIBRARY_PATH"] = "/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:/workspace/deps/cuda-12.8/lib64"
    env["WB_LOG"] = "Warn"; env["UIPC_BENCHMARK_TIMERS"] = "0"
    for k in ("WB_TIMER", "NO_MAS", "NO_GRAPH"):
        env.pop(k, None)
    env.update(extra_env)
    cwd = os.path.join(tree, "libuipc-samples/examples", scene_dir)
    t0 = time.time()
    p = subprocess.run([py, "main.py", "--headless", str(frames)], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=5400)
    wall = time.time() - t0
    line = None
    for l in p.stdout.splitlines():
        if l.startswith("BENCHMARK_RESULT "):
            line = l[len("BENCHMARK_RESULT "):]
    if line is None:
        sys.stderr.write(p.stdout[-3000:] + "\n" + p.stderr[-3000:] + "\n")
        raise SystemExit("no BENCHMARK_RESULT (rc=%d)" % p.returncode)
    d = json.loads(line)
    fm = d["frame_ms"]
    fs = d["frame_stats"]
    return {
        "wall_s": wall,
        "mean": statistics.mean(fm), "median": statistics.median(fm),
        "p95": sorted(fm)[max(0, int(0.95*len(fm))-1)],
        "newton": sum(f["newton_iterations"] for f in fs),
        "pcg": sum(f["linear_solver_iterations"] for f in fs),
        "ls": sum(f.get("line_search_trials", 0) for f in fs),
        "converged": all(f["converged"] for f in fs),
        "newton_limit": any(f.get("hit_newton_limit") for f in fs),
        "obs": d.get("observables", {}),
        "stderr_tail": p.stderr[-4000:] if p.stderr else "",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--tree", default="head")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="/workspace/output/round5/validation/runs")
    ap.add_argument("--keep-stderr", action="store_true")
    a = ap.parse_args()
    sd, df = SCENES[a.scene]
    tree, py = TREES[a.tree]
    frames = a.frames or df
    extra = dict(kv.split("=", 1) for kv in a.env)
    os.makedirs(a.out, exist_ok=True)
    res = []
    for i in range(a.reps):
        r = run_once(tree, py, sd, frames, extra)
        if not a.keep_stderr:
            r.pop("stderr_tail")
        res.append(r)
        print(f"{a.tag} {a.scene} rep{i+1}: mean={r['mean']:.4f} median={r['median']:.4f} "
              f"newton={r['newton']} pcg={r['pcg']} ls={r['ls']} obs={r['obs']}", flush=True)
    p = os.path.join(a.out, f"{a.tag}_{a.scene}.json")
    json.dump({"tag": a.tag, "scene": a.scene, "tree": a.tree, "frames": frames,
               "env": extra, "reps": res}, open(p, "w"), indent=1)
    means = [r["mean"] for r in res]
    print(f"{a.tag} {a.scene} SUMMARY mean_of_mean={statistics.mean(means):.4f} "
          f"sd={statistics.pstdev(means):.4f} n={len(means)} -> {p}", flush=True)

main()

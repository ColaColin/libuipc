#!/usr/bin/env python3
"""One run of a scene; print a compact fingerprint line (observables + iteration counts)."""
import json, os, subprocess, sys, statistics, time
SCENES={"wb":"6_wrecking_balls","cwc":"93_cube_wall_cloth","c2":"88_stiff_gipc_benchmark","mb":"89_mas_bunny"}
TREES={"head":("/workspace/deps/libuipc-src","/workspace/deps/uipc-perf-env/bin/python"),
       "base":("/workspace/deps/libuipc-valbase","/workspace/deps/uipc-valbase-env/bin/python")}
def one(tree,py,sd,frames,extra):
    env=dict(os.environ)
    env["PYTHONPATH"]="/workspace/archive/libuipc-samples/shim"
    env["CUDA_HOME"]="/workspace/deps/cuda-12.8"
    env["LD_LIBRARY_PATH"]="/workspace/deps/libuipc-src/build-dahl/vcpkg_installed/x64-linux/lib:/workspace/deps/cuda-12.8/lib64"
    env["WB_LOG"]="Warn"; env["UIPC_BENCHMARK_TIMERS"]="0"
    for k in ("WB_TIMER","NO_MAS","NO_GRAPH"): env.pop(k,None)
    env.update(extra)
    t0=time.time()
    p=subprocess.run([py,"main.py","--headless",str(frames)],
        cwd=os.path.join(tree,"libuipc-samples/examples",sd),env=env,
        capture_output=True,text=True,timeout=5400)
    wall=time.time()-t0
    line=[l for l in p.stdout.splitlines() if l.startswith("BENCHMARK_RESULT ")]
    if not line:
        return {"err":True,"rc":p.returncode,"tail":(p.stdout[-1500:]+p.stderr[-1500:])}
    d=json.loads(line[-1][len("BENCHMARK_RESULT "):])
    fs=d["frame_stats"]; fm=d["frame_ms"]
    return {"err":False,"wall":wall,"mean":statistics.mean(fm),"median":statistics.median(fm),
            "newton":sum(f["newton_iterations"] for f in fs),
            "pcg":sum(f["linear_solver_iterations"] for f in fs),
            "ls":sum(f.get("line_search_trials",0) for f in fs),
            "conv":all(f["converged"] for f in fs),
            "obs":d.get("observables",{}),
            "stderr":p.stderr[-6000:]}
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--tree",default="head")
    ap.add_argument("--frames",type=int,default=60); ap.add_argument("--reps",type=int,default=1)
    ap.add_argument("--env",action="append",default=[]); ap.add_argument("--tag",default="t")
    ap.add_argument("--show-stderr",action="store_true")
    a=ap.parse_args()
    tree,py=TREES[a.tree]; extra=dict(kv.split("=",1) for kv in a.env)
    for i in range(a.reps):
        r=one(tree,py,SCENES[a.scene],a.frames,extra)
        if r["err"]:
            print(f"{a.tag} {a.scene} rep{i+1} FAILED rc={r['rc']}"); print(r["tail"]); continue
        obs=json.dumps(r["obs"],sort_keys=True)
        print(f"{a.tag} {a.scene} rep{i+1} mean={r['mean']:.5f} med={r['median']:.5f} "
              f"N={r['newton']} P={r['pcg']} LS={r['ls']} conv={r['conv']} wall={r['wall']:.1f} obs={obs}",flush=True)
        if a.show_stderr and r["stderr"].strip():
            print("STDERR>>>"); print(r["stderr"]); print("<<<",flush=True)

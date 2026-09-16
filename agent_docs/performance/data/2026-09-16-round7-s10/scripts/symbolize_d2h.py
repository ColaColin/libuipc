#!/usr/bin/env python3
"""s10: aggregate [d2h-site] stacks by symbolized caller (first backend_cuda frame after host_read)."""
import re, subprocess, sys
from collections import defaultdict
log = open(sys.argv[1] if len(sys.argv)>1 else 'cp_d2hprof.log').read()
SO = "/workspace/deps/libuipc-src/build-perf/Release/bin/libuipc_backend_cuda.so"
sites = re.findall(r"\[d2h-site\] n=(\d+) stall_us=(\d+) bytes=(\d+) \| (.*)", log)
# also the global line
g = re.search(r"\[d2h\] readbacks=(\d+) bytes=(\d+) drain=([\d.]+) ms stall=([\d.]+) ms \(stall/readback=([\d.]+) us\)", log)
print(f"# crease-press at head 18d9a41e -- UIPC_D2H_PROFILE=2 full 130-frame run, per-call-site readback stalls")
if g:
    print(f"[d2h] readbacks={g.group(1)} bytes={g.group(2)} drain={g.group(3)} ms stall={g.group(4)} ms (stall/readback={g.group(5)} us)")
# symbolize: first .so+0xOFFSET in the backend_cuda .so, frame index >=1 (0 is host_read itself)
cache = {}
def sym(off):
    if off not in cache:
        r = subprocess.run(["addr2line","-f","-C","-e",SO,off],capture_output=True,text=True)
        cache[off] = r.stdout.split("\n")[0].split("(")[0][:80]
    return cache[off]
agg = defaultdict(lambda: [0,0,0])
for n, stall, byts, stack in sites:
    frames = stack.split()
    # find the SECOND backend_cuda frame (first = host_read funnel); fall back to first
    be = [f for f in frames if "backend_cuda.so" in f]
    if not be: continue
    off = be[1].split("+")[-1] if len(be)>1 else be[0].split("+")[-1]
    fn = sym(off)
    a = agg[(fn,off)]; a[0]+=int(n); a[1]+=int(stall); a[2]+=int(byts)
tot_stall = sum(a[1] for a in agg.values())
print(f"# {'stall_us':>9s} {'n':>7s} {'bytes':>9s} fn (first stack frame after host_read)")
for (fn,off),a in sorted(agg.items(), key=lambda kv:-kv[1][1]):
    print(f"{a[1]:9d} {a[0]:7d} {a[2]:9d} {fn}  [{off}]")
print(f"# total site stall {tot_stall/1000:.1f} ms")

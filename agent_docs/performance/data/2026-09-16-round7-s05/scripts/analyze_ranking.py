#!/usr/bin/env python3
"""Round-7 s05 fresh kernel rankings from the full-run nsys kern_sum csvs."""
import csv
import re
import sys
from pathlib import Path

S = Path("/workspace/output/round7/s05")

FAMS = [
    ("pcg kernels (spmv/dot/update)", re.compile(r"Spmv_rbk|fused_dot_kernel|fused_update_xr|fused_axpy|fused_.*(dot|update|spmv)", re.I)),
    ("contact assemble (do_assemble)", re.compile(r"do_assemble_kernel")),
    ("BVH + CCD (InfoStacklessBVH*)", re.compile(r"InfoStacklessBVH|TrajectoryFilter|pairFilter|stacklessSelf|filter_toi")),
    ("preconditioners (MAS + abd_diag)", re.compile(r"MASPreconditioner|abd_diag|DiagonalPreconditioner")),
    ("converter/sort/compact", re.compile(r"converter|radix_sort|compact|scan|DeviceRadixSort|DeviceScan|cub::", re.I)),
    ("dahl bending (G/H + energy + commit)", re.compile(r"DahlFriction")),
    ("strain-plastic bending", re.compile(r"StrainPlastic")),
    ("stress-plastic bending", re.compile(r"StressPlastic")),
    ("membrane (SLBW + NeoHookean2D)", re.compile(r"NeoHookeanShell2D|StrainLimitingBaraffWitkin|BaraffWitkin")),
    ("plain hinge bending", re.compile(r"DiscreteShellBending(?!.*Dahl|.*Plastic)")),
    ("energy", re.compile(r"_do_compute_energy")),
    ("abd ortho", re.compile(r"AffineBodyOrtho|affine_body_ortho")),
]


def load(path):
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith(("Time", '"')):
                continue
            try:
                share = float(row[0])
                total_ns = float(row[1])
                instances = int(row[2])
            except (ValueError, IndexError):
                continue
            name = row[-1].strip()
            rows.append(dict(share=share, total_ms=total_ns / 1e6, n=instances,
                             us=total_ns / 1e6 / instances * 1e6 / 1e6 * 1e6, name=name))
    # recompute us properly: total_ms/n * 1000
    for r in rows:
        r["us"] = r["total_ms"] / r["n"] * 1000.0
    return rows


HINGES = {"Dahl": 28557, "Strain": 9506, "Stress": 9506}


def report(tag, path, hinges=None):
    rows = load(path)
    tot_ms = sum(r["total_ms"] for r in rows)
    tot_n = sum(r["n"] for r in rows)
    out = [f"=== {tag}: fresh full-run nsys cuda_gpu_kern_sum ===",
           f"total GPU kernel time {tot_ms:.1f} ms over {tot_n} launches", ""]
    out.append(f" {'#':>2}  {'share':>6}  {'total ms':>9}  {'launches':>8}  {'us/launch':>9}  kernel")
    for i, r in enumerate(sorted(rows, key=lambda r: -r["total_ms"])[:15], 1):
        nm = r["name"]
        nm = nm.replace("void uipc::backend::cuda::<unnamed>::", "")
        nm = nm.replace("uipc::backend::cuda::<unnamed>::", "")
        nm = nm.replace("uipc::backend::cuda_tool::", "")
        nm = nm.split("(")[0] + ("(" + nm.split("(", 1)[1][:52] if "(" in nm else "")
        out.append(f"{i:3d}  {r['share']:6.2f}%  {r['total_ms']:9.1f}  {r['n']:8d}  "
                   f"{r['us']:9.1f}  {nm[:110]}")
    out.append("")
    out.append("family shares:")
    famtot = {f[0]: 0.0 for f in FAMS}
    for r in rows:
        for fname, pat in FAMS:
            if pat.search(r["name"]):
                famtot[fname] += r["total_ms"]
                break
    for fname, ms in sorted(famtot.items(), key=lambda kv: -kv[1]):
        if ms > 0:
            out.append(f"  {fname:44s} {ms/tot_ms*100:6.2f}%  {ms:9.1f} ms")
    # target kernel checks
    out.append("")
    out.append("bending-family + instantiation checks:")
    for r in sorted(rows, key=lambda r: -r["total_ms"]):
        nm = r["name"]
        if re.search(r"(Dahl|Plastic)DiscreteShellBending_do_compute_gradient_hessian", nm) \
           or re.search(r"DiscreteShellBending_do_compute_gradient_hessian", nm):
            m = re.search(r"kernel<[^>]*>", nm)
            short = re.sub(r"(void )?uipc::backend::cuda::<unnamed>::", "", nm).split("(")[0]
            per = ""
            for k, h in HINGES.items():
                if k in r["name"]:
                    per = f"  {r['us'] / h * 1000:.1f} ns/hinge"
            out.append(f"  {r['n']:6d} launches  {r['us']:8.1f} us/launch  {r['share']:5.2f}%  "
                       f"{m.group(0) if m else '<?>'}  {short}{per}")
    txt = "\n".join(out)
    print(txt)
    return txt


cp = report("crease-press at head (default = dahl GN <3,1>), graph-node tracing",
            S / "nsys2_cp_cuda_gpu_kern_sum.csv", hinges=None)
(S / "kernel_ranking_cp.txt").write_text(cp + "\n")
print()
cwc = report("cube-wall-cloth at head (cross-scene reference), graph-node tracing",
             S / "nsys2_cwc_cuda_gpu_kern_sum.csv")
(S / "kernel_ranking_cwc.txt").write_text(cwc + "\n")

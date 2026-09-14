#!/usr/bin/env python3
"""Round-6 V3: read the env-switch audit sessions and print, per switch, what was launched."""
import csv, re, sqlite3, sys, os
OUT = "/workspace/output/round6/v3/envaudit"

def kern(tag, pat):
    rows = list(csv.DictReader(open(f"{OUT}/{tag}_cuda_gpu_kern_sum.csv")))
    p = re.compile(pat)
    out = []
    for r in rows:
        n = r.get("Name") or r.get("Kernel Name") or ""
        if p.search(n):
            out.append((int(float(r["Instances"])), float(r["Avg (ns)"]) / 1e3, n))
    return sorted(out, key=lambda x: -x[0])

def show(title, tag, pat, trim=None):
    print(f"--- {title}  [{tag}]")
    rows = kern(tag, pat)
    if not rows:
        print("      (no kernel matches)")
    for inst, avg, n in rows:
        if trim:
            n = trim(n)
        print(f"   {inst:7d} x {avg:9.2f} us  {n}")

def short(n):
    n = re.sub(r"uipc::backend::cuda::(\(anonymous namespace\)::)?", "", n)
    n = re.sub(r"InfoStacklessBVHSimplexTrajectoryFilter_", "", n)
    n = re.sub(r"MASPreconditionerEngine_", "", n)
    # keep the template arguments, drop the argument list: cut at the "(" that follows the
    # template's closing ">" (or at the first "(" when there is no template)
    i = n.find("kernel<")
    if i >= 0:
        depth = 0
        for j in range(i + 6, len(n)):
            if n[j] == "<": depth += 1
            elif n[j] == ">":
                depth -= 1
                if depth == 0:
                    n = n[: j + 1]; break
    else:
        n = re.sub(r"\(.*$", "", n)
    n = n.replace("(bool)", "").replace("(int)", "")
    return n[:150]

def sql(tag, q, args=()):
    c = sqlite3.connect(f"{OUT}/{tag}.sqlite")
    return list(c.execute(q, args))

def streams(tag, pat):
    print(f"--- streams of kernels matching {pat!r}  [{tag}]")
    q = """select k.streamId, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
           where s.value like ? group by k.streamId order by 2 desc"""
    for sid, n in sql(tag, q, (pat,)):
        print(f"   stream {sid:3d}: {n:6d} launches")
    q2 = """select k.streamId, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k group by k.streamId order by 2 desc"""
    print("   (all kernels by stream: " + ", ".join(f"{sid}:{n}" for sid, n in sql(tag, q2)) + ")")

def grids(tag, pat):
    print(f"--- gridX of kernels matching {pat!r}  [{tag}]")
    q = """select k.gridX, count(*), avg(k.end-k.start)/1e3 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
           where s.value like ? group by k.gridX order by 2 desc"""
    rows = sql(tag, q, (pat,))
    for g, n, avg in rows[:8]:
        print(f"   gridX {g:6d}: {n:6d} launches, {avg:8.2f} us avg")
    return rows

def join_overlap(tag):
    """case2: kernels on the default stream (other than contact part 2 itself) that START inside a
    contact part 1 window. With the join taken at do_assemble (=0) the host waits on part 1 before
    it can issue anything after part 2; with the join deferred (=1) the default stream runs ahead."""
    q = """select k.start, k.end, k.streamId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
           order by k.start"""
    rows = sql(tag, q)
    p1 = [(a, b, sid) for a, b, sid, n in rows if "do_assemble_kernel<(bool)0, (int)1," in n]
    if not p1:
        print(f"--- [{tag}] no contact part-1 launches (split not taken)"); return
    p1_stream = {sid for _, _, sid in p1}
    from collections import Counter
    default_sid = Counter(sid for _, _, sid, _ in rows).most_common(1)[0][0]
    inside = Counter()
    for a, b, sid, n in rows:
        if sid in p1_stream or "do_assemble_kernel" in n:
            continue
        for s0, e0, _ in p1:
            if s0 < a < e0:
                inside[short(n)] += 1
                break
    tot = sum(inside.values())
    print(f"--- [{tag}] part-1 launches {len(p1)} on stream(s) {sorted(p1_stream)}; busiest stream {default_sid}; "
          f"non-contact kernels starting INSIDE a part-1 window: {tot}")
    for n, c in inside.most_common(6):
        print(f"      {c:5d}  {n}")

if __name__ == "__main__":
    print("=== s02  UIPC_DSB_GAUSS_NEWTON  (default: Gauss-Newton <Proj=3,Solver=1>; =0: exact <1,1>)")
    show("default", "t_default", "DiscreteShellBending_do_compute_gradient_hessian", short)
    show("UIPC_DSB_GAUSS_NEWTON=0", "t_gn0", "DiscreteShellBending_do_compute_gradient_hessian", short)

    print("\n=== s07/V2  UIPC_CONTACT_RANK1  (contact part 2 Proj: default 5 = PP closed form; =0 exact) and")
    print("=== s08     UIPC_CONTACT_SPD1_BASIS (contact part 1 Proj: default 8 = basis-free; =0 dense-Q eigen path)")
    for tag in ("t_default", "t_rank1_0", "t_spd1b0"):
        show(tag, tag, "do_assemble_kernel", short)

    print("\n=== s04  UIPC_CCD_EARLY_OUT and s06 UIPC_CCD_COMPACT  (filter_toi_k*<EarlyOut, Stats, DcdCull>)")
    for tag in ("t_default", "t_ccdeo0", "t_ccdcp0"):
        show(tag, tag, "filter_toi_k|DeviceSelect|filter_active_k", short)

    print("\n=== s17  UIPC_SEG_REDUCE2  (fast_segmental_reduce_matrix_k2_kernel vs the old fast_segmental_reduce_matrix_kernel)")
    for tag in ("t_default", "t_seg0"):
        show(tag, tag, "fast_segmental_reduce_matrix", short)

    print("\n=== s09/K9  UIPC_CONTACT_SPLIT with UIPC_ABD_GH_PREPASS=1 pinned  (=2: part 1 + part 2; =1: same two launches one stream; =0: fused Part 0)")
    for tag in ("r_pre1", "r_split1_pre1", "r_split0_pre1"):
        show(tag, tag, "do_assemble_kernel", short)
        streams(tag, "%do_assemble_kernel%")

    print("\n=== s10/s11  UIPC_ABD_GH_PREPASS  (default 4: the body-local G/H reporters -- affine_body_bdf1_kinetic + ortho_potential gradient/hessian -- on the prepass side stream; =0: on the default stream, in place)")
    for tag in ("r_default", "r_pre1", "r_pre0"):
        show(tag, tag, "kinetic_compute_gradient_hessian|ortho_potential_compute_gradient_hessian|assemble_kinetic_shape", short)
        streams(tag, "%kinetic_compute_gradient_hessian%")
        streams(tag, "%ortho_potential_compute_gradient_hessian%")

    print("\n=== s13  UIPC_SPMV_GRID_FIT  (grid of Spmv_rbk_sym_spmv_dot_chunked_kernel: fitted vs capacity)")
    for tag in ("m_default", "m_fit0"):
        grids(tag, "%spmv_dot_chunked%")

    print("\n=== s15  UIPC_MAS_ROWDOT2  (schwarz_local_solve_rowdot2_kernel<16,..> vs the round-4 rowdot kernel)")
    print("=== s16  UIPC_MAS_FUSED_R  (fused_R_kernel, no build_multi_level_R; =0: build_multi_level_R + rowdot2)")
    for tag in ("m_default", "m_rd0", "m_fr0"):
        show(tag, tag, "schwarz_local_solve|build_multi_level_R", short)

    print("\n=== s14  UIPC_CONTACT_DEFERRED_JOIN  (host run-ahead past contact part 1: default 1 deferred, =0 join at do_assemble)")
    for tag in ("c_default", "c_dj0"):
        join_overlap(tag)

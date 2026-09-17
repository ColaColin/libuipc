#!/usr/bin/env python3
"""Round-7 s20 Part C: switch-audit table from the envaudit kern_sum captures.

For every knob: compare the knob-OFF capture against the default capture and
report instance counts of the NEW-path kernel(s) (must fall to 0) and the
OLD-path kernel(s) (must appear), by demangled template arguments.
PCG_POLL uses the sqlite api trace instead (8-byte D2H memcpyAsync count).
"""
import csv
import glob
import os
import sqlite3
import sys

OUT = "/workspace/output/round7/s20/envaudit"

def kern_instances(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["Name"]] = int(r["Instances"])
    return rows

def count(rows, *needles):
    n = 0
    hits = []
    for name, inst in rows.items():
        if all(nd in name for nd in needles):
            n += inst
            hits.append((inst, name[:130]))
    return n, hits

def d2h8(sqlite_path):
    db = sqlite3.connect(sqlite_path)
    try:
        cur = db.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE bytes=8 AND copyKind=2")
        return cur.fetchone()[0]
    finally:
        db.close()

A = sys.argv[1] if len(sys.argv) > 1 else OUT
D = kern_instances(f"{A}/k_default_cuda_gpu_kern_sum.csv")

def arm(tag):
    return kern_instances(f"{A}/{tag}_cuda_gpu_kern_sum.csv")

print("| knob (OFF arm) | old-path kernel evidence | new-path kernel (default arm) | verdict |")
print("|---|---|---|---|")

def row(knob, old_desc, old_needles, new_desc, new_needles, armrows=None):
    R = armrows
    on_old, oh = count(R, *old_needles)
    def_new, dh = count(D, *new_needles)
    arm_new, _ = count(R, *new_needles)
    def_old, _ = count(D, *old_needles)
    ok = (arm_new == 0 and on_old > 0)
    print(f"| `{knob}=0` | {old_desc}: {on_old} inst (default {def_old}) | "
          f"{new_desc}: default {def_new}, OFF-arm {arm_new} | "
          f"{'SELECTS OLD' if ok else 'CHECK'} |")
    for inst, name in sorted(oh, reverse=True)[:2]:
        print(f"|   | ↳ {inst}x `{name}` | | |")

P = "DiscreteShellBending_do_compute_gradient_hessian_kernel"
row("UIPC_DAHL_GAUSS_NEWTON", "dahl G/H `<1, 1, 2>`",
    ("DahlFriction"+P+"<", "(int)1, (int)1, (int)2"), "dahl G/H `<3, 1, *>`",
    (P+"<(int)3, (int)1, ",), arm("k_dahl_gn0"))
row("UIPC_DAHL_REDUCED_SPD", "dahl G/H `<0, 1, 2>`",
    ("DahlFriction"+P+"<", "(int)0, (int)1, (int)2"), "dahl G/H `<1, 1, 2>`",
    ("DahlFriction"+P+"<", "(int)1, (int)1, (int)2"), arm("k_dahl_rspd0"))
row("UIPC_DAHL_BLOCKED_PROJ", "dahl G/H `<2, 1>`",
    ("DahlFriction"+P+"<", "(int)2, (int)1"), "dahl G/H `<1, 1, 2>`",
    ("DahlFriction"+P+"<", "(int)1, (int)1, (int)2"), arm("k_dahl_blk0"))
row("UIPC_DAHL_TQL2", "dahl G/H `<1, 0, 2>` (Eigen)",
    ("DahlFriction"+P+"<", "(int)1, (int)0, (int)2"), "dahl G/H `<1, 1, 2>`",
    ("DahlFriction"+P+"<", "(int)1, (int)1, (int)2"), arm("k_dahl_tql0"))

for fam, disp in (("Strain", "strain"), ("Stress", "stress")):
    K = fam + "Plastic" + P
    row(f"UIPC_PDSB_REDUCED_SPD ({disp})", f"{disp} G/H `<0, 1, 2>`",
        (K+"<", "(int)0, (int)1, (int)2"), f"{disp} G/H `<1, 1, 2>`",
        (K+"<", "(int)1, (int)1, (int)2"), arm("k_pdsb_rspd0"))
row("UIPC_PDSB_BLOCKED_PROJ (both)", "plastic G/H `<2, 1>`",
    ("Plastic"+P+"<", "(int)2, (int)1"), "plastic G/H `<1, 1, 2>`",
    ("Plastic"+P+"<", "(int)1, (int)1, (int)2"), arm("k_pdsb_blk0"))
row("UIPC_PDSB_TQL2 (both)", "plastic G/H `<1, 0, 2>` (Eigen)",
    ("Plastic"+P+"<", "(int)1, (int)0, (int)2"), "plastic G/H `<1, 1, 2>`",
    ("Plastic"+P+"<", "(int)1, (int)1, (int)2"), arm("k_pdsb_tql0"))

K = "NeoHookeanShell2D_do_compute_gradient_hessian_kernel"
row("UIPC_NHS2D_REDUCED_SPD", "NHS2D G/H `<0, 1>`",
    (K+"<", "(int)0, (int)1"), "NHS2D G/H `<1, 1>`",
    (K+"<", "(int)1, (int)1"), arm("k_nhs2d_rspd0"))
row("UIPC_NHS2D_BLOCKED_PROJ", "NHS2D G/H `<2, 1>`",
    (K+"<", "(int)2, (int)1"), "NHS2D G/H `<1, 1>`",
    (K+"<", "(int)1, (int)1"), arm("k_nhs2d_blk0"))
row("UIPC_NHS2D_TQL2", "NHS2D G/H `<1, 0>` (Eigen)",
    (K+"<", "(int)1, (int)0"), "NHS2D G/H `<1, 1>`",
    (K+"<", "(int)1, (int)1"), arm("k_nhs2d_tql0"))

# half-assembly: <1,S,0> full-assembly arms reappear, <1,S,2> vanish (all four families)
row("UIPC_MAKE_SPD_BLOCKED_HALF", "bending G/H `<1, 1, 0>` (full asm)",
    (P+"<", "(int)1, (int)1, (int)0"), "bending G/H `<1, 1, 2>` (half asm)",
    (P+"<", "(int)1, (int)1, (int)2"), arm("k_half0"))

# s13 fold: staged k3 reappears; folded PERM k2 disappears
row("UIPC_SEGRED_UNSTAGE", "staged k3 sort kernel",
    ("matrix_converter_radix_sort_indices_and_blocks_k3_kernel",), "folded PERM k2 reduce",
    ("fast_segmental_reduce_matrix_k2_kernel", "matrix_converter_permuted_value_op"), arm("k_segred_uns0"))
# s15 tree: ks (K-serial) disappears, k2 warp-tree reappears on M*N>=9 classes
row("UIPC_SEGRED_TREE", "k2 warp-tree reduce (wide classes)",
    ("fast_segmental_reduce_matrix_k2_kernel",), "ks K-serial reduce",
    ("fast_segmental_reduce_matrix_ks_kernel",), arm("k_segred_tree0"))
# s14 doublet fold: wide-payload CUB onesweep reappears, <int,int> one vanishes
row("UIPC_DOUBLET_UNSTAGE", "CUB onesweep `<unsigned int, Matrix3x1>`",
    ("DeviceRadixSortOnesweepKernel", "Eigen::Matrix<double, (int)3, (int)1"), "CUB onesweep `<unsigned int, int>`",
    ("DeviceRadixSortOnesweepKernel", "(unsigned int), (int)",), arm("k_doublet_uns0"))

# s17 doorbell: host-path switch -- api-trace evidence
try:
    n_def = d2h8(f"{A}/k_pcg_polldef.sqlite")
    n_off = d2h8(f"{A}/k_pcg_poll0.sqlite")
    print(f"| `UIPC_PCG_POLL=0` | 8 B D2H memcpyAsync: {n_off} (12f) | "
          f"default arm: {n_def} (12f) | "
          f"{'SELECTS OLD' if n_off > 3 * max(n_def, 1) else 'CHECK'} |")
except Exception as e:
    print(f"| `UIPC_PCG_POLL=0` | sqlite read failed: {e} | | CHECK |")

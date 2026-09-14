"""Round-6 V1: the crease-severity micro-test.

The Gauss-Newton hinge Hessian (s02) drops `E'(theta) * hess(theta)`, and
`E'(theta) = 2*L0*kappa*(theta - theta_bar)/h_bar` is **linear in the crease
angle** while the term that is kept is constant.  So if Gauss-Newton degrades
anywhere, it degrades where `|theta - theta_bar|` is large -- and neither the
tumbler nor any of the four official benchmarks goes there.  This builds a case
that does.

A flat cloth strip (rest angle theta_bar = 0 at every hinge), *no gravity and
no contact*, so the hinge energy is the only thing that can make the solve
hard.  The two end rows are driven by a soft position constraint onto the
rigid rotation of their rest positions by +/- phi about the fold axis; phi
ramps linearly to `--phi-max`.  The strip is short along the fold direction and
wide across it, so the imposed turn 2*phi is carried by only a handful of hinge
rows and the per-hinge angle gets genuinely large.

Every frame it prints one CREASE line: the measured dihedral-angle distribution
(that is the theory's severity variable, measured rather than assumed) and the
solver's own counters.

    python crease_micro.py [--frames=N] [--phi-max=deg] [--cells=N] [--edge=m]
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
from uipc import Animation, Logger, builtin, view
from uipc.core import Engine, World, Scene
from uipc.geometry import label_surface, trimesh
from uipc.constitution import (DiscreteShellBending, ElasticModuli2D,
                               SoftPositionConstraint,
                               StrainLimitingBaraffWitkinShell)


def _opt(name, default, cast=float):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return cast(a.split("=", 1)[1])
    return default


FRAMES = _opt("frames", 150, int)
PHI_MAX = math.radians(_opt("phi-max", 170.0))
N_LONG = _opt("cells", 6, int)        # cells along the fold direction (x)
N_WIDE = _opt("wide", 24, int)        # cells across (z)
EDGE = _opt("edge", 0.010)
DT = 1.0 / 60.0
STRENGTH = _opt("strength", 1.0e3)
TAG = _opt("tag", "run", str)

Logger.set_level(Logger.Level.Warn)
workspace = os.environ.get("UIPC_CREASE_WS", "/workspace/output/round6/v2/crease/ws/")
Path(workspace).mkdir(parents=True, exist_ok=True)
engine = Engine("cuda", workspace)
world = World(engine)

# ---- flat strip: length N_LONG*EDGE along x, width N_WIDE*EDGE along z ------
xs = (np.arange(N_LONG + 1) - N_LONG / 2.0) * EDGE
zs = (np.arange(N_WIDE + 1) - N_WIDE / 2.0) * EDGE
V = np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)
nx = N_LONG + 1
F = []
for j in range(N_WIDE):
    for i in range(N_LONG):
        a, b = j * nx + i, j * nx + i + 1
        c, d = (j + 1) * nx + i, (j + 1) * nx + i + 1
        F += [[a, b, d], [a, d, c]]
F = np.array(F, dtype=np.int32)

# interior edges = hinges, with the two opposite vertices
from collections import defaultdict
edge_tri = defaultdict(list)
for t, tri in enumerate(F):
    for k in range(3):
        e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
        edge_tri[e].append(t)
HINGE = []
for e, ts in edge_tri.items():
    if len(ts) == 2:
        opp = []
        for t in ts:
            opp.append([int(v) for v in F[t] if int(v) not in e][0])
        HINGE.append([e[0], e[1], opp[0], opp[1]])
HINGE = np.array(HINGE, dtype=np.int64)


def dihedrals(P):
    """Signed dihedral angle at every hinge, flat = 0 (the constitution's
    theta_bar for a flat rest shape)."""
    e0, e1, o0, o1 = (P[HINGE[:, k]] for k in range(4))
    e = e1 - e0
    n0 = np.cross(o0 - e0, e)
    n1 = np.cross(e, o1 - e0)
    el = np.linalg.norm(e, axis=1)
    s = (np.einsum("ij,ij->i", np.cross(n0, n1), e) / np.maximum(el, 1e-30))
    c = np.einsum("ij,ij->i", n0, n1)
    return np.arctan2(s, c)


config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [0.0], [0.0]]        # bending only
config["contact"]["enable"] = False              # bending only
config["newton"]["velocity_tol"] = _opt("vel-tol", 0.05)
config["linear_system"]["tol_rate"] = _opt("tol-rate", 1e-3)
scene = Scene(config)

slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()
spc = SoftPositionConstraint()
mesh = trimesh(np.ascontiguousarray(V), F)
label_surface(mesh)
slbws.apply_to(mesh,
               stretch_moduli=ElasticModuli2D.youngs_poisson(5.0e5, 0.4),
               shear_moduli=ElasticModuli2D.youngs_poisson(5.0e3, 0.4),
               mass_density=200.0, thickness=1.0e-3, strain_rate=100.0)
dsb.apply_to(mesh, 1.0e5, 0.4)
spc.apply_to(mesh, STRENGTH)
obj = scene.objects().create("strip")
slot, _ = obj.geometries().create(mesh)

REST = V.copy()
EPS = 1e-9
END = np.abs(REST[:, 0]) > (N_LONG / 2.0) * EDGE - EPS
SIGN = np.where(REST[:, 0] < 0, +1.0, -1.0)


def fold(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    ic = view(geo.vertices().find(builtin.is_constrained))
    ap = view(geo.vertices().find(builtin.aim_position))
    phi = PHI_MAX * min(1.0, info.frame() / float(FRAMES))
    th = SIGN * phi
    c, s = np.cos(th), np.sin(th)
    aim = REST.copy()
    aim[:, 0] = REST[:, 0] * c - REST[:, 1] * s
    aim[:, 1] = REST[:, 0] * s + REST[:, 1] * c
    ap[:] = aim.reshape(-1, 3, 1)
    ic[:] = END.astype(np.int32)


scene.animator().insert(obj, fold)
world.init(scene)
if not world.is_valid():
    raise SystemExit("world invalid")

print(f"CREASE_SETUP tag={TAG} verts={len(V)} tris={len(F)} hinges={len(HINGE)} "
      f"cells={N_LONG}x{N_WIDE} edge={EDGE} phi_max_deg={math.degrees(PHI_MAX):.1f} "
      f"frames={FRAMES} vel_tol={_opt('vel-tol', 0.05)} tol_rate={_opt('tol-rate', 1e-3)} gn={os.environ.get('UIPC_DSB_GAUSS_NEWTON', 'default')}", flush=True)

import time
for _ in range(FRAMES):
    t0 = time.perf_counter()
    world.advance()
    world.retrieve()
    ms = (time.perf_counter() - t0) * 1e3
    P = np.asarray(slot.geometry().positions().view()).reshape(-1, 3)
    th = np.abs(dihedrals(P))
    st = dict(engine.frame_stats())
    print("CREASE " + " ".join(f"{k}={v}" for k, v in dict(
        frame=world.frame(),
        phi_deg=round(math.degrees(PHI_MAX * min(1.0, world.frame() / float(FRAMES))), 4),
        theta_max=round(float(th.max()), 6),
        theta_mean=round(float(th.mean()), 6),
        theta_p95=round(float(np.percentile(th, 95)), 6),
        newton=st.get("newton_iterations"),
        pcg=st.get("linear_solver_iterations"),
        ls=st.get("line_search_trials"),
        converged=int(st.get("converged", -1)),
        hit_newton=int(bool(st.get("hit_newton_limit", 0))),
        hit_ls=int(bool(st.get("hit_line_search_limit", 0))),
        alpha=round(float(st.get("last_line_search_alpha", 1.0)), 6),
        finite=int(bool(np.all(np.isfinite(P)))),
        ms=round(ms, 3)).items()), flush=True)

"""Round-6 V2: the contact-severity micro-test.

V1's `crease_micro.py` swept *bending* severity because s02's approximation
dropped a term linear in the crease angle.  This change is a **contact**
Hessian, so the analogous severity variable is contact severity: how deep into
the barrier the active pairs sit, and how much of the pair population is PE --
the branch `UIPC_CONTACT_RANK1=1` approximates (PP is exact, s07).

The case: three flat cloth patches stacked with a lateral offset, gravity and
all other loads **off**, so the contact barrier is the only thing that can make
the solve hard.  The top and bottom patches are fully soft-position-constrained
and act as cloth platens; the middle patch is free.  The top platen is driven
down by a ramp, so the stack is compressed monotonically and the active pairs
are pushed further into the barrier frame after frame.  The lateral offset
(half an element in x, a third in z) puts vertices over *edges* rather than
over vertices, which is what makes the population PE-dominated.

With `--drag=<m>` the top platen also translates in x, so the contact slides:
that is the axis to sweep if the eigendirection the rank-1 form drops turns out
to be tangential rather than normal.

Every frame it prints one CONTACT line: the measured severity (the minimum
inter-patch vertex distance in units of d_hat, the pair counts by type read
from the engine's own log, the compression achieved) and the solver's counters.

    python contact_micro.py [--frames=N] [--press=m] [--drag=m] [--cells=N]
                            [--edge=m] [--layers=N] [--tag=name]
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
CELLS = _opt("cells", 10, int)
EDGE = _opt("edge", 0.008)
LAYERS = _opt("layers", 3, int)          # 2 platens + (LAYERS-2) free sheets
PRESS = _opt("press", 0.0024)            # total downward travel of the top platen
DRAG = _opt("drag", 0.0)                 # total lateral travel of the top platen
DT = 1.0 / 60.0
STRENGTH = _opt("strength", 1.0e4)
TAG = _opt("tag", "run", str)
CLOTH_R = _opt("cloth-r", 2.0e-4)
D_HAT = _opt("d-hat", 8.0e-4)
GAP0 = _opt("gap0", 0.0020)              # initial vertical spacing between layers

Logger.set_level(getattr(Logger.Level, os.environ.get("WB_LOG", "Warn")))
workspace = os.environ.get("UIPC_MICRO_WS", "/workspace/output/round6/v2/micro/ws/")
Path(workspace).mkdir(parents=True, exist_ok=True)
engine = Engine("cuda", workspace)
world = World(engine)


def patch(off_x, off_z, y):
    xs = (np.arange(CELLS + 1) - CELLS / 2.0) * EDGE + off_x
    zs = (np.arange(CELLS + 1) - CELLS / 2.0) * EDGE + off_z
    V = np.array([[x, y, z] for z in zs for x in xs], dtype=np.float64)
    nx = CELLS + 1
    F = []
    for j in range(CELLS):
        for i in range(CELLS):
            a, b = j * nx + i, j * nx + i + 1
            c, d = (j + 1) * nx + i, (j + 1) * nx + i + 1
            F += [[a, b, d], [a, d, c]]
    return V, np.array(F, dtype=np.int32)


config = Scene.default_config()
config["dt"] = DT
config["gravity"] = [[0.0], [0.0], [0.0]]        # contact only
config["contact"]["enable"] = True
config["contact"]["friction"]["enable"] = True
config["contact"]["d_hat"] = D_HAT
config["newton"]["velocity_tol"] = _opt("vel-tol", 0.05)
config["linear_system"]["tol_rate"] = _opt("tol-rate", 1e-3)
config["linear_system"]["fem_preconditioner"] = "mas"
scene = Scene(config)

ct = scene.contact_tabular()
ct.default_model(0.3, 1.0e8)

slbws = StrainLimitingBaraffWitkinShell()
dsb = DiscreteShellBending()
spc = SoftPositionConstraint()

layers = []
for k in range(LAYERS):
    # the lateral offset is what makes the population PE-dominated: a vertex of
    # one layer lands over an *edge* of the next, not over a vertex or the
    # middle of a face.
    V, F = patch(0.5 * EDGE * k, 0.37 * EDGE * k, GAP0 * k)
    mesh = trimesh(np.ascontiguousarray(V), F)
    label_surface(mesh)
    slbws.apply_to(mesh,
                   stretch_moduli=ElasticModuli2D.youngs_poisson(5.0e5, 0.4),
                   shear_moduli=ElasticModuli2D.youngs_poisson(5.0e3, 0.4),
                   mass_density=200.0, thickness=CLOTH_R, strain_rate=100.0)
    dsb.apply_to(mesh, 1.0e5, 0.4)
    if k == 0 or k == LAYERS - 1:
        spc.apply_to(mesh, STRENGTH)
    obj = scene.objects().create(f"layer{k}")
    slot, _ = obj.geometries().create(mesh)
    layers.append(dict(k=k, V=V, F=F, obj=obj, slot=slot))

TOP = LAYERS - 1


def hold(info: Animation.UpdateInfo):
    """The bottom platen stays on its rest positions."""
    geo = info.geo_slots()[0].geometry()
    view(geo.vertices().find(builtin.is_constrained))[:] = 1
    view(geo.vertices().find(builtin.aim_position))[:] = \
        layers[0]["V"].reshape(-1, 3, 1)


def drive(info: Animation.UpdateInfo):
    """The top platen descends (and optionally slides) on a linear ramp."""
    geo = info.geo_slots()[0].geometry()
    s = min(1.0, info.frame() / float(FRAMES))
    aim = layers[TOP]["V"].copy()
    aim[:, 1] -= PRESS * s
    aim[:, 0] += DRAG * s
    view(geo.vertices().find(builtin.is_constrained))[:] = 1
    view(geo.vertices().find(builtin.aim_position))[:] = aim.reshape(-1, 3, 1)


scene.animator().insert(layers[0]["obj"], hold)
scene.animator().insert(layers[TOP]["obj"], drive)

world.init(scene)
if not world.is_valid():
    raise SystemExit("world invalid")


def positions():
    return [np.asarray(g["slot"].geometry().positions().view()).reshape(-1, 3)
            for g in layers]


def min_inter_layer_distance(P):
    """Smallest *vertical clearance* between two adjacent layers: the layers are
    flat sheets stacked along y with a lateral offset, so the point-triangle gap
    the barrier sees is the vertical separation, not the vertex-vertex distance
    (which is dominated by the lateral offset and would read ~5 mm while the
    sheets are 0.05 mm apart).  Severity is the *surface* separation in units of
    d_hat, g = (clearance - 2r) / d_hat: 1 = the activation distance, 0 =
    touching.  The same measure in both arms."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return float("inf")
    best = float("inf")
    for a in range(len(P) - 1):
        lo, hi = P[a], P[a + 1]
        # the sheets are offset laterally, so "the layer below me" has to be
        # evaluated at my own (x, z): take the 3 nearest vertices of the lower
        # sheet in the xz plane and interpolate its height there.
        t = cKDTree(lo[:, [0, 2]])
        d, j = t.query(hi[:, [0, 2]], k=3, workers=-1)
        w = 1.0 / np.maximum(d, 1e-12)
        y_lo = (lo[j, 1] * w).sum(axis=1) / w.sum(axis=1)
        best = min(best, float((hi[:, 1] - y_lo).min()))
    return best


print(f"CONTACT_SETUP tag={TAG} layers={LAYERS} cells={CELLS} edge={EDGE} "
      f"d_hat={D_HAT} cloth_r={CLOTH_R} gap0={GAP0} press={PRESS} drag={DRAG} "
      f"frames={FRAMES} strength={STRENGTH} "
      f"rank1={os.environ.get('UIPC_CONTACT_RANK1', 'default')}", flush=True)

import time

for _ in range(FRAMES):
    t0 = time.perf_counter()
    world.advance()
    world.retrieve()
    ms = (time.perf_counter() - t0) * 1e3
    P = positions()
    st = dict(engine.frame_stats())
    dmin = min_inter_layer_distance(P)
    allP = np.vstack(P)
    print("CONTACT " + " ".join(f"{k}={v}" for k, v in dict(
        frame=world.frame(),
        press_mm=round(PRESS * min(1.0, world.frame() / float(FRAMES)) * 1e3, 5),
        drag_mm=round(DRAG * min(1.0, world.frame() / float(FRAMES)) * 1e3, 5),
        d_min=round(dmin, 9),
        g=round((dmin - 2.0 * CLOTH_R) / D_HAT, 6),
        thickness_mm=round(float(allP[:, 1].max() - allP[:, 1].min()) * 1e3, 5),
        newton=st.get("newton_iterations"),
        pcg=st.get("linear_solver_iterations"),
        ls=st.get("line_search_trials"),
        converged=int(st.get("converged", -1)),
        hit_newton=int(bool(st.get("hit_newton_limit", 0))),
        hit_ls=int(bool(st.get("hit_line_search_limit", 0))),
        alpha=round(float(st.get("last_line_search_alpha", 1.0)), 6),
        toi=round(float(st.get("last_ccd_toi", 1.0)), 6),
        finite=int(bool(np.all(np.isfinite(allP)))),
        ms=round(ms, 3)).items()), flush=True)

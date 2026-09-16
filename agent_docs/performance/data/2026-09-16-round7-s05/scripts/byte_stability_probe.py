#!/usr/bin/env python3
"""s05 byte-stability probe (round-6 V1 protocol).

Reconstructs the UN-EDITED 103_crease_press rest construction (the pure-numpy
part of main.py at samples main d004635, transcribed below) and compares it
bitwise against the frame-0 position dump of the EDITED main.py run under
default flags.  If the engine's device round-trip is lossless and the edit is
truly inert, the two must be bit-identical (uint64 view).
"""
import sys
import numpy as np

# --- transcribed from git show d004635:examples/103_crease_press/main.py ---
SHEET_X = 0.48
SHEET_Z = 0.80
STACK_Y0 = 0.50
R_NOMINAL_DENIM = 0.0008
R_NOMINAL_CARTON = 0.0025
D_HAT_NOMINAL = 2.0e-3
LAYER_MARGIN = 2.0e-3
EDGE = 0.011        # --edge-len default (r6.add_r6_args(ap, 0.011))
CARTON_EDGE = 0.0155  # --carton-edge default
STACK = [("denim", "dahl"), ("cardboard", "strain"), ("denim", "dahl"),
         ("metal", "stress"), ("denim", "dahl"), ("cardboard", "strain"),
         ("metal", "stress")]


def make_sheet_grid(edge):
    rx = max(3, int(round(SHEET_X / edge)) + 1)
    rz = max(3, int(round(SHEET_Z / edge)) + 1)
    xs = np.linspace(-0.5 * SHEET_X, 0.5 * SHEET_X, rx)
    zs = np.linspace(-0.5 * SHEET_Z, 0.5 * SHEET_Z, rz)
    V = np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)
    return rx, rz, V


def tri_min_height_from_V(V, rx, rz):
    # triangles of the 2-per-cell grid, as in make_sheet_grid's F
    F = []
    for j in range(rz - 1):
        for i in range(rx - 1):
            a = j * rx + i
            F += [[a, a + 1, a + rx + 1], [a, a + rx + 1, a + rx]]
    F = np.asarray(F, dtype=np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    e = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1)], axis=1)
    return float((2.0 * area[:, None] / np.maximum(e, 1e-30)).min())


denim_rx, denim_rz, denim_V = make_sheet_grid(EDGE)
carton_rx, carton_rz, carton_V = make_sheet_grid(CARTON_EDGE)
MIN_TRI_HEIGHT = min(tri_min_height_from_V(denim_V, denim_rx, denim_rz),
                     tri_min_height_from_V(carton_V, carton_rx, carton_rz))
D_HAT = min(D_HAT_NOMINAL, 0.3 * MIN_TRI_HEIGHT)
GAP = R_NOMINAL_DENIM + R_NOMINAL_CARTON + D_HAT + LAYER_MARGIN

sheets = []
for k, (name, kind) in enumerate(STACK):
    V = (denim_V if kind == "dahl" else carton_V).copy()
    V[:, 1] = STACK_Y0 + k * GAP
    sheets.append(V)
constructed = np.vstack(sheets)
# ---------------------------------------------------------------------------

dump = np.load(sys.argv[1])[0]
print(f"constructed {constructed.shape}  dump0 {dump.shape}")
assert constructed.shape == dump.shape, "shape mismatch"
eq = np.array_equal(constructed.view(np.uint64), dump.view(np.uint64))
diff_mask = ~(constructed.view(np.uint64) == dump.view(np.uint64))
ndiff = int(diff_mask.sum())
maxabs = float(np.abs((constructed - dump)[diff_mask.any(axis=1)]).max()) if ndiff else 0.0
print(f"bitwise identical: {eq}   mismatching uint64 words: {ndiff}"
      f"   max|delta| on differing vertices: {maxabs:.3e}")
sys.exit(0 if eq else 1)

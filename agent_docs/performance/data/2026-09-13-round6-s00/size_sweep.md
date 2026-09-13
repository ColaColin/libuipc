# tumbler-garments: how the shipped size was chosen

RTX 2070 SUPER (cc 7.5, 8 GB), `build-perf` at `perf-round6-base`, `$UIPC_PERF_PY`.
The knob is `--edge-len=` (target garment element size); everything else fixed.

## 1. element size vs wall time (60-frame probes, legacy soft cloth)

| `--edge-len` | cloth verts | total tris | mean ms/frame (60 f) |
|---|---|---|---|
| 14.0 mm | 2816 | 5384 | 66.1 |
| 11.0 mm | 3954 | 7600 | 77.0 |
| 8.5 mm | 6078 | 11756 | 116.3 |
| 7.0 mm | 8296 | 16118 | 161.0 |

Roughly linear in triangle count above ~8k triangles.

## 2. membrane stiffness (60 f at 7.0 mm) — picked on *soundness*, and it was also free

`E_stretch / E_shear / E_bend`, with `k_stretch = E * 2r`:

| moduli | area ratio min/max | min triangle height | Newton mean | PCG mean | mean ms |
|---|---|---|---|---|---|
| 1e4 / 1e1 / 1e4 (cloth-machine "legacy") | 0.349 / 1.70 | 2.33 mm | 10.45 | 262 | 157.9 |
| 2e5 / 1e3 / 5e4 | 0.456 / 1.32 | 2.88 mm | 7.93 | 229 | 154.1 |
| **5e5 / 5e3 / 1e5** (shipped, `k_stretch` = 1000 N/m) | **0.813 / 1.20** | **4.00 mm** | 8.87 | 344 | **138.2** |

The legacy moduli crush triangles to a third of their rest area, which pushes the
*dynamic* minimum triangle height below `2r + d_hat` — permanent self-contact.
`k_stretch = 1000 N/m` is the terry-towel entry of the cloth-machine calibrated
material table; it keeps the cloth within ±20 % of its rest area and is 12 % faster.

## 3. drum density — a conditioning parameter, not a physical one (60 f at 6.2 mm)

The drum is kinematic, so its mass only enters the linear system's conditioning and
the soft-constraint penalty. A heavy drum makes the *relative* PCG stopping criterion
the drum's, not the cloth's.

| density | motor strength | Newton mean | PCG mean | PCG max | mean ms | max tracking error |
|---|---|---|---|---|---|---|
| 2000 kg/m³ | 100 | 8.80 / 8.43 | 283 / 267 | 1680 / 735 | 161.0 / 168.8 | 0.040° |
| 200 kg/m³ | 100 | 7.63 / 7.62 | 227 / 225 | 635 / 725 | 146.9 / 141.3 | 0.040° / **0.512°** |
| 200 kg/m³ | 1000 | 8.37 / 8.28 | 233 / 233 | 1425 / 725 | 140.1 / 180.2 | 0.036° |
| **500 kg/m³** | **100** | **7.82 / 7.73** | **220 / 210** | **615 / 625** | **139.6 / 133.0** | **0.040°** |

(two runs per row). 500 kg/m³ gives the lowest PCG count *and* keeps the tracking
error at the 2000 kg/m³ value.

## 4. final size (180 f, shipped configuration)

| | |
|---|---|
| `--edge-len` | 6.2 mm |
| garments | towel 3640 tris, pillowcase 5760, shorts 6828, washcloth 2048 = 18,276 |
| drum | 1,644 tris (828 verts), 3 lifters, 48 circumferential segments |
| total | 10,224 vertices, 19,920 triangles, 27,158 bending hinges |
| `r` / `d_hat` | 1.00 mm / 1.27 mm (capped at 0.25 / 0.3 × min triangle height 4.24 mm) |
| wall time | 26.8 – 30.0 s over n=5 (mean 27.8 s) |
| peak GPU | 1.6 – 2.3 GB device total (process peak 2.0 GB) of the 8 GB card |

Both size targets are met: ~30 s per run and ~20k triangles, at n=5 per arm ≈ 4.6 min
per A/B arm.

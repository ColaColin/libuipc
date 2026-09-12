# round 5, w1-ampere — fixed-sweep branch-free 3x3 SVD (probe p02)

Everything here was produced **without a GPU**: both boxes rented for the w1 slot failed to
provision (see the coordinator log). `nvcc` compiles headless, and `math::qr_svd` is `UIPC_GENERIC`,
so the numerics and the static instruction counts are fully measurable host-side. What is **not**
here, and what the step needs before it can be accepted, is the correctness gate and the two-level
performance gate.

- `r5-s01-fixed-sweep-svd.patch` — the complete change against `perf-round5-base` (890482c2).
  `git apply` it, or run `harness/apply_patch.py <src_root> harness/svd_fixed_body.hpp
  harness/devtest.cu`. Touches three files and nothing else:
  `algorithm/qr_svd.hpp` (adds `qr_svd_fixed`), `constitutions/stable_neo_hookean_3d.cu`
  (second template parameter + `UIPC_QR_SVD_FIXED` switch), `apps/tests/backends/cuda/qr_svd.cu`
  (a 1.57e6-sample device-side verifier as a Catch2 test).

## harness/

| file | what it does |
|---|---|
| `svd_fixed_body.hpp` | the new code, as a standalone body so host tools and the patch share one copy |
| `apply_patch.py` | applies the change to a tree (idempotence-checked, asserts every anchor) |
| `devtest.cu` | the device-side randomised verifier appended to the Catch2 test file |
| `verify.cpp` | host-side verifier, new vs **old path**, 18 families x N samples, 2-6 sweeps |
| `sweeps.cpp` | how many sweeps each sample actually needs (p01-style histogram) |
| `cond.cpp` | accuracy of both paths as a function of `cond(F)` |
| `build.sh`, `count.py`, `loops.py` | p01's SASS harness, reused unchanged |

Build the host tools with:

```sh
S=<libuipc root>
g++ -O2 -std=c++20 -I. -I$S/src -I$S/src/backends/cuda -I$S/include \
  -isystem $S/build-perf/vcpkg_installed/x64-linux/include/eigen3 \
  -isystem $S/build-perf/vcpkg_installed/x64-linux/include verify.cpp -o verify
```

## raw/

| file | what it is |
|---|---|
| `sweeps_to_converge.txt` | sweeps needed per sample, 65 536 per distribution — the basis for choosing 4 |
| `verify_960k_ns4.txt` | 9.6e5 host samples at 4 sweeps, new vs old, 18 families |
| `accuracy_vs_cond.txt` | both paths vs `cond(F)` from 1 to 1e10 |
| `sass_counts.txt` | p01's probe rebuilt with the new SVD at 3 / 4 / 5 sweeps |
| `loop_regions.txt` | the backward-branch scan: 1 loop in base, **0** with the fixed sweeps |
| `sass_counts_real_patched.txt` | the same counts from the *real* patched `stable_neo_hookean_3d.cu` |
| `ptxas_real_patched.txt` | registers / stack / spills of the real patched kernel |

The rig was validated against p01 before anything was concluded: the unpatched baseline reproduces
p01's `base hoist=True: FP64=3729 ... total=8984` exactly, and the `UIPC_QR_SVD_FIXED=0`
instantiation of the patched kernel is instruction-identical to it.

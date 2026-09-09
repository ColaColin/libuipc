# libuipc from-source build notes (no-root Linux box)

Machine: Debian 12 (gcc/g++ 12.2.0), 32 cores, 60 GB RAM, RTX 2070 SUPER (sm_75),
NVIDIA driver 595.84 (CUDA 13.2 capable), no system nvcc, no sudo.
Date: 2026-09-03. All heavy commands were run under `nice -n 19` because a GPU
benchmark was running concurrently.

Layout:
- `/workspace/deps/libuipc-src`      this source checkout (git, with submodules)
- `/workspace/deps/uipc-src-env`     new venv (python 3.11) that receives the built `pyuipc`
- `/workspace/deps/vcpkg`            vcpkg (bootstrapped without root)
- `/workspace/deps/cuda-12.8`        assembled CUDA 12.8.1 toolkit (from NVIDIA redist tarballs)
- `/workspace/deps/cuda-redist-dl`   downloaded redist `.tar.xz` archives (kept for reproducibility)
- `/workspace/deps/localbin`         tiny python shims for `zip`/`unzip` (missing on this box; vcpkg bootstrap requires them)

## 1. Source checkout
```
cd /workspace/deps
git clone --recursive https://github.com/spiriMirror/libuipc libuipc-src
# HEAD = 4d1f3f3446631b283c14b2500b14b0eeb93ac6f4 (2026-09-04, "Merge pull request #492 from spiriMirror/refactor-main")
# submodules: libuipc-samples @ 4fb26b7, scripts/SymEigen @ a12c56e
```
Build system facts (from CMakeLists.txt / docs/build_install/linux.md / pyproject.toml):
- CMake >= 3.26, Python >= 3.11, CUDA >= 12.4, vcpkg (CMAKE_TOOLCHAIN_FILE must point to vcpkg.cmake).
- `scripts/gen_vcpkg_json.py` writes `build/vcpkg.json` + `build/vcpkg-configuration.json`
  (baseline dd3097e3 = vcpkg tag 2025.7.25; extra registry github.com/spiriMirror/vcpkg for `octree`;
  overlay port `ports/tinygltf`). Deps: eigen3, catch2, libigl, spdlog 1.12.0, fmt 10.2.1,
  cppitertools, dylib, nlohmann-json, magic-enum, tinygltf, tbb, urdfdom, cpptrace, octree.
- Official wheel arch list (pyproject.toml): `75-real;80-real;86-real;89-real;120-real;89-virtual`.
- CUDA backend: `enable_language(CUDA)`, RDC (separable compilation) ON, C++20, `--extended-lambda --expt-relaxed-constexpr`.

## 2. New venv
```
/usr/bin/python3.11 -m venv /workspace/deps/uipc-src-env
uipc-src-env/bin/pip install --upgrade pip
uipc-src-env/bin/pip install numpy warp-lang polyscope imageio imageio-ffmpeg pillow matplotlib scipy trimesh cmake ninja
#   -> cmake 4.4.3, ninja 1.13.2, numpy 2.4.6, warp-lang 1.17.0, polyscope 2.6.1, ...
uipc-src-env/bin/pip install "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-runtime-cu12==12.8.*" \
   "nvidia-cuda-cccl-cu12==12.8.*" "nvidia-cublas-cu12==12.8.*" "nvidia-cusparse-cu12==12.5.*" \
   "nvidia-cusolver-cu12==11.7.*" "nvidia-nvjitlink-cu12==12.8.*" "nvidia-cuda-nvrtc-cu12==12.8.*" \
   "nvidia-cuda-profiler-api-cu12==12.8.*" "nvidia-curand-cu12==10.3.9.*"
```
PROBLEM: the PyPI `nvidia-cuda-nvcc-cu12` wheels (checked 12.6.85 and 12.8.93) contain ONLY
`bin/ptxas`, `nvvm/` and `include/crt`; there is NO `nvcc` driver binary, so they cannot be used
as a compiler. (They exist to support JIT users like JAX.) The runtime-library wheels
(cublas/cusparse/cusolver/nvjitlink/nvrtc) are still useful at run time (LD_LIBRARY_PATH).

FIX: build a CUDA_HOME from NVIDIA's official redistributable tarballs (no root needed):
```
curl -sL -o redistrib_12.8.1.json https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json
# components downloaded (linux-x86_64, sha256 verified against the json):
#   cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90, cuda_nvrtc 12.8.93, cuda_profiler_api,
#   cuda_nvtx, cuda_cuobjdump, cuda_cuxxfilt, cuda_nvdisasm, cuda_nvprune, cuda_nvml_dev, cuda_cupti,
#   cuda_sanitizer_api, libcublas 12.8.4.1, libcusparse 12.5.8.93, libcusolver 11.7.3.90,
#   libnvjitlink 12.8.93, libcurand 10.3.9.90, libnvfatbin, nsight_compute 2025.1.1.2 (for later profiling)
# each archive is `tar -xJf <a>.tar.xz --strip-components=1 -C /workspace/deps/cuda-12.8`
# then `ln -s lib lib64` so FindCUDAToolkit's lib64 lookup works.
```

## 3. vcpkg
```
cd /workspace/deps && git clone https://github.com/microsoft/vcpkg   # 7ff71c68 (2026-09-03)
```
PROBLEM: `bootstrap-vcpkg.sh` aborts if `zip`/`unzip` are not on PATH (they are not installed and
there is no sudo). FIX: `/workspace/deps/localbin/{zip,unzip}` are ~20-line python `zipfile`
shims; put `/workspace/deps/localbin` first on PATH.
```
export PATH=/workspace/deps/localbin:/workspace/deps/uipc-src-env/bin:$PATH
cd /workspace/deps/vcpkg && ./bootstrap-vcpkg.sh -disableMetrics   # -> vcpkg 2026-07-27-98d7cb0c
# generate manifest exactly as CMake would, then pre-install deps (same paths CMake uses later)
cd /workspace/deps/libuipc-src && mkdir -p build
/workspace/deps/uipc-src-env/bin/python scripts/gen_vcpkg_json.py build --dev_mode=OFF \
    --with_usd_support=OFF --with_vdb_support=OFF --with_cuda_backend=ON
nice -n 19 /workspace/deps/vcpkg/vcpkg install --x-manifest-root=build \
    --x-install-root=build/vcpkg_installed --triplet x64-linux --disable-metrics  # log: build/vcpkg_install.log
```
Result: "All requested installations completed successfully in: 2.3 min" (34 ports, incl. transitive
boost-*, libdwarf, zstd, zlib, tinyxml, console_bridge, catch2). Static libs except urdfdom (.so).
Note: the first version of the `zip` shim did not understand vcpkg's `--exclude .DS_Store` argument,
which produced 22 non-fatal "error: zip ... failed" lines (binary-cache export only, packages still
installed fine); the shim was fixed (handles --exclude/-x, -y symlinks) before the build proceeded.
Log: build/vcpkg_install.log.

## 4. Helper scripts (added to the checkout)
- `build_env.sh`   exports CUDA_HOME/CUDACXX/CUDAToolkit_ROOT/CMAKE_TOOLCHAIN_FILE/PATH/LD_LIBRARY_PATH
- `configure.sh`   the exact cmake configure line (Release, Ninja, sm_75, pybind ON, examples/tests/benchmarks OFF)

## 5. nvcc sanity test (assembled toolkit)
```
source build_env.sh
nvcc -std=c++20 -arch=sm_75 -rdc=true --extended-lambda --expt-relaxed-constexpr -ccbin g++ -o t t.cu
./t   # -> sum=2016 runtime=12080 driver=13020   (cuda::atomic_ref + cub include, ran on the RTX 2070 SUPER)
```
nvcc 12.8.93 accepts the system g++ 12.2.0 as host compiler (nvcc 12.8 supports gcc <= 14).

## 6. Configure
```
./configure.sh            # log: build/configure.log
```
Key output: "The CUDA compiler identification is NVIDIA 12.8.93 with host compiler GNU 12.2.0",
"CMAKE_CUDA_ARCHITECTURES: 75", pybind11 3.1.0 + pybind11_stubgen 2.5.5 found in
/workspace/deps/uipc-src-env. Choices made:
- `UIPC_CUDA_ARCHITECTURES=75` (sm_75 SASS + compute_75 PTX) instead of the wheel's
  `75-real;80-real;86-real;89-real;120-real;89-virtual`: only an RTX 2070 SUPER is present and each
  extra arch multiplies device-compile time. Change in configure.sh if other GPUs are needed.
- `UIPC_BUILD_EXAMPLES/TESTS/BENCHMARKS=OFF` for a lean first build (flip to ON to get the Catch2
  tests in build/Release/bin); `UIPC_DEV_MODE=ON` so re-configures skip the vcpkg manifest re-install.
- `UIPC_BUILD_PYBIND=ON`, `UIPC_PYTHON_EXECUTABLE_PATH=/workspace/deps/uipc-src-env/bin/python`:
  the pyuipc POST_BUILD step (scripts/after_build_pyuipc.py) copies all .so from build/Release/bin
  into build/python/src/uipc/_native, generates .pyi stubs, and `pip install`s build/python into that venv.
- Harmless warnings: CMake policy warnings from install() in uipc_utils.cmake, "unused-cli CUDAToolkit_ROOT".

## 7. Build
```
nice -n 19 cmake --build build -j24      # log: build/build.log
```

## 8. Runtime / smoke test helpers (added to the checkout)
- `env-src.sh`     runtime env: UIPC_PY, LD_LIBRARY_PATH -> /workspace/deps/cuda-12.8/lib64 (cublas/cusparse/cusolver),
                   PYTHONPATH -> /workspace/libuipc-samples/shim (headless polyscope), PS_SHIM_FRAMES
- `smoke_test.sh [frames]`  runs libuipc-samples/examples/34_cloth_stack/main.py (default 20 frames)

## 9. Where the solver hot paths live (src/backends/cuda/, for later kernel work)
- Newton/IPC pipeline orchestration: `engine/advance_ipc.cu` (`SimEngine::advance()`; stage order and `Timer` scopes),
  `engine/advance_al.cu` (AL variant), `engine/sim_engine_do_advance.cu` (dispatch).
- Global linear system + PCG: `linear_system/global_linear_system.cu` (assembly to BCOO, `solve()`),
  `linear_system/linear_fused_pcg.cu` (DEFAULT solver `linear_system/solver="fused_pcg"`, CUDA-graph replay
  via `linear_system/use_cuda_graph`), `linear_system/linear_pcg.cu` (reference PCG), `linear_system/spmv.cu`
  (`rbk_sym_spmv`, `rbk_sym_spmv_dot` = the hot SpMV kernels), `linear_system/{diag,off_diag}_linear_subsystem.cu`.
- MAS preconditioner: `finite_element/mas_preconditioner_engine.cu` (hierarchy build + `scatter_hessian_to_clusters`,
  `invert_cluster_matrices`, `build_multi_level_R`/`schwarz_local_solve`/`collect_final_Z`),
  `finite_element/fem_mas_preconditioner.cu` (enable with config `linear_system/fem_preconditioner="mas"`, default "diag");
  diag variants `finite_element/fem_diag_preconditioner.cu`, `affine_body/abd_diag_preconditioner.cu`.
- Collision detection: `collision_detection/global_trajectory_filter.cu` (detect/filter_active/filter_toi fan-out),
  `collision_detection/simplex_trajectory_filter.cu`, DEFAULT broad phase
  `collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.cu` (+ `info_stackless_bvh.h`/`details/*.inl`;
  config `collision_detection/method="info_stackless_bvh"`), legacy lbvh/stackless variants in `filters/`.
- Contact energy/Hessian: `contact_system/global_contact_manager.cu` (adaptive kappa, d_hat, CFL, feasible step),
  `contact_system/contact_models/ipc_simplex_normal_contact.cu` + `ipc_simplex_frictional_contact.cu`
  (fused PT/EE/PE/PP `do_assemble_kernel`), half-plane variants alongside.
- Line search + CCD: `line_search/line_searcher.cu` (energy eval; `line_search/max_iter`), reporters
  `finite_element/fem_line_search_reporter.cu`, `affine_body/abd_line_search_reporter.cu`; CCD = additive CCD (ACCD,
  Codim-IPC style) in `utils/distance/ccd.h` + `details/ccd.inl`, launched from the `*_filter_toi_k1..k4_kernel`s in
  `filters/info_stackless_bvh_simplex_trajectory_filter.cu`.
- Kernel idiom: plain `__global__` kernels launched with `cuda_tool::best_grid_dim/best_block_dim` (`cuda_tool/launch.h`),
  CUB wrappers in `cuda_tool/cub.h`; timing via `uipc.Timer.enable_all()` (see `include/uipc/common/timer.h`),
  per-frame counters via `Engine.frame_stats()`, profiling helpers in `python/src/uipc/profile/` (ncu integration).
Build result: 540/540 steps, exit 0, no compile errors (log build/build.log). Outputs in
build/Release/bin: libuipc_{core,geometry,constitution,io,sanity_check}.so, libuipc_backend_{cuda,none}.so,
pyuipc.cpython-311-x86_64-linux-gnu.so. The POST_BUILD step then ran `pip install build/python`
-> "Successfully installed pyuipc-0.9.0" in /workspace/deps/uipc-src-env (version comes from UIPC_VERSION
in CMakeLists.txt, not from git tags, so it is 0.9.0 rather than 0.0.x like the PyPI wheel).
NOTE: libuipc_backend_cuda.so / libuipc_io.so link the vcpkg-built liburdfdom_*.so.3.0 through an absolute
RUNPATH into build/vcpkg_installed/x64-linux/lib (they are not copied into site-packages/uipc/_native),
so do not delete the build tree. Rebuild after editing kernels with:
    source build_env.sh && nice -n 19 cmake --build build -j24    # re-runs the pip install automatically

## 10. Smoke test
```
./smoke_test.sh 20        # == cd /workspace/libuipc-samples/examples/34_cloth_stack &&
                          #    PYTHONPATH=/workspace/libuipc-samples/shim PS_SHIM_FRAMES=20 /workspace/deps/uipc-src-env/bin/python main.py
```
Result (build/smoke_test.log): "[cuda] Device: [0] NVIDIA GeForce RTX 2070 SUPER, Compute Capability: 7.5",
"SanityCheck Summary: 0 errors", frames 1..20 simulated (config: fem_preconditioner=mas, solver=fused_pcg,
collision_detection/method=info_stackless_bvh), "SHIM PERF frames=20 mean=138.5ms median=100.5ms",
"Cuda Backend Shutdown Success", exit 0. `python/uipc_info.py` also runs fine.
Runtime-library finding: unlike the PyPI 0.0.27 wheel (which needs libcublas/cusparse/cusolver on
LD_LIBRARY_PATH), this build's libuipc_backend_cuda.so has NO dynamic CUDA dependencies (cudart is linked
statically, the solver is the in-house fused PCG). The prescribed command therefore also passes with
LD_LIBRARY_PATH unset (build/smoke_test_bare_env.log). env-src.sh still adds /workspace/deps/cuda-12.8/lib64
for convenience (nvrtc/cublas would be found there if a future revision needs them).

## 11. Quick reference
Activate:   source /workspace/deps/libuipc-src/env-src.sh      (runtime: $UIPC_PY, PYTHONPATH shim, PS_SHIM_FRAMES)
Rebuild:    source /workspace/deps/libuipc-src/build_env.sh && cd /workspace/deps/libuipc-src && nice -n 19 cmake --build build -j24
Reconfigure:/workspace/deps/libuipc-src/configure.sh [-DUIPC_BUILD_TESTS=ON ...]
Versions:   libuipc 4d1f3f3446631b283c14b2500b14b0eeb93ac6f4 (pyuipc 0.9.0), CUDA 12.8.93 (nvcc) / 12.8.90 (cudart),
            host g++ 12.2.0, arch sm_75 (+compute_75 PTX), vcpkg 2026-07-27 w/ baseline 2025.7.25,
            cmake 4.4.3, ninja 1.13.2, pybind11 3.1.0, python 3.11.
Timing:     configure 18:01, build 18:01:39 -> 18:08:04 (~6.5 min wall, -j24 under nice 19, single arch).
Profiling:  Nsight Compute 2025.1.1 launcher is /workspace/deps/cuda-12.8/ncu (top level, not bin/); compute-sanitizer is /workspace/deps/cuda-12.8/bin/compute-sanitizer. Nsight Systems was not downloaded (955 MB; add nsight_systems from the redist json if needed).

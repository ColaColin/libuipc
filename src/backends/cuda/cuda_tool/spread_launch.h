#pragma once
#include <cuda_tool/launch.h>
#include <cstdlib>

namespace uipc::backend::cuda_tool
{
// perf round 5 (w3): `cudaOccupancyMaxPotentialBlockSize` (what best_block_dim
// uses) maximises resident warps *per SM*, which is the wrong objective when
// the whole launch is smaller than the GPU. The ABD per-body kernels are one
// thread per body: on rigid-wrecking-balls that is 576 threads, and at 254
// registers the occupancy-max block size is 256, so the launch is **gridDim 3**
// — 3 of the 36 SMs of an RTX 5060 Ti, with 92 % of the FP64 units idle
// (measured per Newton iteration: `ortho_potential_compute_gradient_hessian`
// 777 us, `abd_diag_preconditioner_do_assemble` 587 us, together 18 % of the
// scene's kernel time).
//
// For such launches, narrow blocks spread the grid over distinct SMs instead.
// The kernel binary is untouched and every thread still computes
// `blockIdx.x * blockDim.x + threadIdx.x` and guards on `i >= n`, so each item
// is handled by exactly the same instruction sequence on exactly the same data
// and writes exactly the same output slot: bit-identical, for any elementwise
// kernel with no block- or warp-level cooperation.
//
// UIPC_GRID_SPREAD=0 restores the occupancy-max geometry (the old path).
// UIPC_GRID_SPREAD_BLOCK=<threads> overrides the spread block size (default 32).
inline int device_sm_count()
{
    static const int sms = []
    {
        int device = 0;
        int value  = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device);
        return value > 0 ? value : 1;
    }();
    return sms;
}

inline bool grid_spread_enabled()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_GRID_SPREAD");
        return !(e && e[0] == '0');
    }();
    return on;
}

inline int grid_spread_block()
{
    static const int bs = []
    {
        const char* e = std::getenv("UIPC_GRID_SPREAD_BLOCK");
        int         v  = e ? std::atoi(e) : 32;
        if(v < 32)
            v = 32;
        return (v / 32) * 32;
    }();
    return bs;
}

// Block dim for launching `kernel` over n items: the occupancy-max block size,
// except for launches too small to reach every SM, which get narrow blocks.
template <typename Kernel>
int spread_block_dim(int n, Kernel kernel, size_t shared_mem_size = 0)
{
    int bd = best_block_dim(kernel, shared_mem_size);
    if(!grid_spread_enabled() || n <= 0)
        return bd;
    int sms = device_sm_count();
    if((n + bd - 1) / bd >= sms)  // the natural grid already covers the device
        return bd;
    int bs = grid_spread_block();
    // never below one warp per SM's worth of items
    int want = (n + sms - 1) / sms;
    want     = ((want + 31) / 32) * 32;
    if(want > bs)
        bs = want;
    if(bs > bd)
        bs = bd;
    return bs;
}

template <typename Kernel>
int spread_grid_dim(int n, Kernel kernel, size_t shared_mem_size = 0)
{
    int bs = spread_block_dim(n, kernel, shared_mem_size);
    return (n + bs - 1) / bs;
}
}  // namespace uipc::backend::cuda_tool

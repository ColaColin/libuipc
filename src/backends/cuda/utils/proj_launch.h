#pragma once
#include <cuda_tool/launch.h>

namespace uipc::backend::cuda
{
// perf/round5 (w0, s20): block size chosen for *grid* coverage, not only for
// per-SM occupancy.
//
// `cuda_tool::best_block_dim` is `cudaOccupancyMaxPotentialBlockSize`, which
// maximises resident warps per SM and is blind to how many SMs the resulting
// grid actually reaches. The two PSD-projection call sites are the worst case
// for that: one dense eigen-solve per thread, 192-255 registers, no shared
// memory, no `__syncthreads`, and a small `n`. Measured on the 2070 SUPER
// (40 SMs, nsys `cuda_gpu_trace`):
//
//   ortho_potential G/H   rigid-wrecking-balls  <<< 3, 256>>>  ->  3 of 40 SMs
//   ortho_potential G/H   cube-wall-cloth       <<< 8, 256>>>  ->  8 of 40 SMs
//   DiscreteShellBending  cube-wall / case2     <<<48, 256>>>  ->  one resident
//                         block per SM at 255 registers, so two waves are run
//                         for 1.2 waves of work.
//
// A register-bound kernel with no shared memory and no block-wide cooperation
// pays nothing for a smaller block: the same number of warps is resident per
// SM, they are simply spread over more SMs. So take the occupancy-optimal
// block size and halve it (down to one warp) while the resulting grid is
// smaller than `blocks_per_sm` blocks per SM. When `n` is large enough that
// the grid already covers the device, the occupancy-optimal answer is returned
// unchanged, so this only ever acts in the small-grid regime.
//
// The launched `__global__` function is bit-for-bit the same one either way:
// both kernels derive everything from the global thread index and write to
// index-addressed destinations (`DoubletVectorAssembler::write`,
// `TripletMatrixAssembler::write`, `body_hessian(i)`), with no atomics and no
// cross-thread communication, so the output is a function of the thread index
// alone and does not depend on how the indices are grouped into blocks.
template <typename Kernel>
int fitted_block_dim(Kernel kernel, int n, int blocks_per_sm = 8, size_t shared_mem_size = 0)
{
    static const int sm_count = []
    {
        int dev = 0, c = 1;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, dev);
        return c > 0 ? c : 1;
    }();

    int       b    = cuda_tool::best_block_dim(kernel, shared_mem_size);
    const int want = blocks_per_sm * sm_count;
    while(b > 32 && (n + b - 1) / b < want)
        b >>= 1;
    return b < 32 ? 32 : b;
}
}  // namespace uipc::backend::cuda

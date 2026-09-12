#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

// ---------------------------------------------------------------------------
// perf round 5, launch geometry -- the part of `cuda_tool/spread_launch.h`
// that depends on nothing but the CUDA runtime.
//
// `spread_launch.h` includes `launch.h`, which includes `view.h`; so `view.h`
// and `buffer.h` -- which host cuda_tool's own elementwise fill kernels -- can
// not include it. The heuristic itself has no such dependency, so it lives
// here and `spread_launch.h` includes this file. One heuristic, one place
// (s21's consolidation rule), now including the shared infrastructure.
// ---------------------------------------------------------------------------
namespace uipc::backend::cuda_tool
{
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
        int         v  = e ? std::atoi(e) : 0;  // 0 = no hard override
        return (v / 32) * 32;
    }();
    return bs;
}

// The ramp target: how many blocks per SM the grid should reach before the
// block size is left alone. 8 caps the tail-wave imbalance at 1/8.
inline int grid_spread_blocks_per_sm()
{
    static const int b = []
    {
        const char* e = std::getenv("UIPC_GRID_SPREAD_BPSM");
        int         v = e ? std::atoi(e) : 8;
        return v < 1 ? 1 : v;
    }();
    return b;
}

// The ramp itself, over a block size the caller already has: halve it **in
// whole warps**, down to one warp, while the resulting grid is smaller than
// `blocks_per_sm` blocks per SM. Whole warps are what keeps warp composition
// -- and therefore `__activemask()` -- independent of the geometry.
inline int spread_block_dim_from(int n, int bd, int blocks_per_sm = 0)
{
    if(!grid_spread_enabled() || n <= 0 || bd <= 32 || (bd % 32) != 0)
        return bd;
    if(int forced = grid_spread_block())
        return forced < bd ? forced : bd;
    int       warps = bd / 32;
    const int bpsm  = blocks_per_sm > 0 ? blocks_per_sm : grid_spread_blocks_per_sm();
    const int want  = bpsm * device_sm_count();
    while(warps > 1 && (n + warps * 32 - 1) / (warps * 32) < want)
        warps >>= 1;
    return warps * 32;
}

// ---------------------------------------------------------------------------
// s28: the same ramp for cuda_tool's own elementwise fill kernels
// (`buffer_fill_kernel` in buffer.h, `buffer_view_fill_kernel` in view.h).
// Both bodies are `if(i >= n) return; dst[i] = value;` -- no shared memory, no
// __syncthreads, no warp primitives, no atomics, no accumulation -- so every
// element is written by the same instruction on the same data into the same
// slot whatever the geometry.
//
// The ramp target here is **1 block per SM**, not the 8 the kernel call sites
// use. A fill is bandwidth-bound, not occupancy-bound: once the grid covers
// the device there is nothing left to win, and the wider ramp perturbs far
// more launches for nothing. Measured on stiff-gipc-case2, where the whole
// sub-SM fill population is 0.003 ms per Newton iteration (0.013 % of kernel
// time), the bpsm=8 ramp moved ~700 of 3 300 fill launches and shifted the
// solver on to a different (equally valid) trajectory: +1.95 % PCG iterations
// per Newton step, consistently over 6 runs per arm. Atomic accumulation
// order is timing-dependent in this solver, so a geometry change anywhere can
// do that; the answer is to change geometry only where it pays.
//
//   UIPC_BUFFER_FILL_SPREAD=0   the pre-s28 geometry (the rollback)
//   UIPC_BUFFER_FILL_BPSM=<n>   ramp target in blocks/SM (default 1)
//   UIPC_BUFFER_FILL_VERIFY=1   device-side coverage proof (see below)
// ---------------------------------------------------------------------------
inline bool buffer_fill_spread_enabled()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_BUFFER_FILL_SPREAD");
        return !(e && e[0] == '0');
    }();
    return on;
}

inline int buffer_fill_blocks_per_sm()
{
    static const int b = []
    {
        const char* e = std::getenv("UIPC_BUFFER_FILL_BPSM");
        int         v = e ? std::atoi(e) : 1;
        return v < 1 ? 1 : v;
    }();
    return b;
}

inline int buffer_fill_block_dim(int n, int bd)
{
    if(!buffer_fill_spread_enabled())
        return bd;
    return spread_block_dim_from(n, bd, buffer_fill_blocks_per_sm());
}

// --- the coverage proof -----------------------------------------------------
// A fill writes a CONSTANT, so comparing two geometries' outputs is vacuous:
// any geometry that covers [0, n) produces the same bytes. What a geometry
// change could break is COVERAGE, so that is what is checked: the destination
// is poisoned with 0xA5 bytes, the spread fill runs, and every 32-bit word of
// every element is compared against the corresponding word of `value`. A miss
// leaves 0xA5A5A5A5 and is counted. (A fill whose value is literally
// 0xA5A5A5A5... in every word would hide a miss; none exists in this backend.)
//
// Fills issued inside a stream capture are skipped and counted separately --
// the readback below cannot happen during capture -- and they are the same
// kernel with the same geometry rule.
inline bool buffer_fill_verify_enabled()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_BUFFER_FILL_VERIFY");
        return e && e[0] != '0';
    }();
    return on;
}

struct BufferFillVerifyState
{
    unsigned long long* d_acc     = nullptr;
    unsigned long long  words     = 0;
    unsigned long long  mismatch  = 0;
    unsigned long long  fills     = 0;
    unsigned long long  captured  = 0;
    unsigned long long  odd_type  = 0;
    ~BufferFillVerifyState()
    {
        if(fills || captured)
            std::fprintf(stderr,
                         "[BufferFillVerify] %llu fills checked, %llu words (32-bit) compared, "
                         "%llu mismatching; %llu skipped (inside a stream capture), "
                         "%llu skipped (element size not a multiple of 4)\n",
                         fills,
                         words,
                         mismatch,
                         captured,
                         odd_type);
    }
};

inline BufferFillVerifyState& buffer_fill_verify_state()
{
    static BufferFillVerifyState s;
    return s;
}

template <typename T>
__global__ void buffer_fill_verify_kernel(const T* p, T value, int n, unsigned long long* acc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    constexpr int words = (int)(sizeof(T) / sizeof(unsigned int));
    const unsigned int* a = reinterpret_cast<const unsigned int*>(p + i);
    const unsigned int* b = reinterpret_cast<const unsigned int*>(&value);
    unsigned int        bad = 0;
    for(int w = 0; w < words; ++w)
        if(a[w] != b[w])
            ++bad;
    if(i == 0)
        atomicAdd(acc, (unsigned long long)n * (unsigned long long)words);
    if(bad)
        atomicAdd(acc + 1, (unsigned long long)bad);
}

// returns true when the region was poisoned and the check should run
template <typename T>
inline bool buffer_fill_verify_poison(const T* p, int n, cudaStream_t s)
{
    if(!buffer_fill_verify_enabled() || n <= 0)
        return false;
    if constexpr (sizeof(T) % sizeof(unsigned int) != 0)
    {
        ++buffer_fill_verify_state().odd_type;
        return false;
    }
    else
    {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if(cudaStreamIsCapturing(s, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone)
        {
            ++buffer_fill_verify_state().captured;
            return false;
        }
        cudaMemsetAsync(const_cast<T*>(p), 0xA5, (size_t)n * sizeof(T), s);
        return true;
    }
}

template <typename T>
inline void buffer_fill_verify_check(const T* p, const T& value, int n, cudaStream_t s)
{
    auto& st = buffer_fill_verify_state();
    if(!st.d_acc)
    {
        cudaMalloc(&st.d_acc, 2 * sizeof(unsigned long long));
        cudaMemset(st.d_acc, 0, 2 * sizeof(unsigned long long));
    }
    buffer_fill_verify_kernel<T><<<(n + 255) / 256, 256, 0, s>>>(p, value, n, st.d_acc);
    unsigned long long h[2] = {0, 0};
    cudaMemcpy(h, st.d_acc, sizeof(h), cudaMemcpyDeviceToHost);
    cudaMemset(st.d_acc, 0, sizeof(h));
    st.words += h[0];
    st.mismatch += h[1];
    ++st.fills;
}
}  // namespace uipc::backend::cuda_tool

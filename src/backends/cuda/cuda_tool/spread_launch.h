#pragma once
#include <cuda_tool/launch.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <utility>
#include <type_traits>
#include <cstring>

namespace uipc::backend::cuda_tool
{
// perf round 5, launch geometry (w3 s20 + w0 s20, consolidated here by w3 in
// s21). `cudaOccupancyMaxPotentialBlockSize` -- which is what
// `cuda_tool::best_block_dim` calls -- maximises resident warps *per SM*. That
// is the wrong objective whenever the launch is small relative to the machine,
// and the backend is full of such launches:
//
//   ortho_potential G/H          rigid-wrecking-balls  <<<  3, 256>>>   3 of 36 SMs
//   abd_diag_preconditioner      rigid-wrecking-balls  <<<  3, 256>>>   3 of 36 SMs
//   assemble_kinetic_shape_k2    rigid-wrecking-balls  <<<  3, 256>>>   3 of 36 SMs
//   vertex-half-plane assemble   rigid-wrecking-balls  <<<  1, 640>>>   1 of 36 SMs
//   DiscreteShellBending G/H     cube-wall / case2     <<< 48, 256>>>   two waves
//                                                                      for 1.2
//
// Narrow blocks spread the same warps over more SMs. The kernel binary is
// untouched and every thread still computes `blockIdx.x * blockDim.x +
// threadIdx.x` and guards on `i >= n`, so each item is handled by the same
// instruction sequence on the same data and writes the same output slot:
// **bit-identical**, for any kernel that is elementwise in that index, has no
// block-wide cooperation, and does not accumulate through atomics. Block sizes
// stay whole numbers of warps, so warp composition -- and therefore
// `__activemask()` -- is unchanged as well.
//
// Env switches:
//   UIPC_GRID_SPREAD=0             the pre-round-5 geometry everywhere (rollback)
//   UIPC_GRID_SPREAD_BPSM=<n>      the ramp target in blocks per SM (default 8)
//   UIPC_GRID_SPREAD_BLOCK=<n>     hard override of the block size
//   UIPC_GRID_SPREAD_ONLY=<substr> spread only at call sites whose tag matches
//   UIPC_GRID_SPREAD_VERIFY=1      run both geometries and compare the outputs
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

// Block dim for launching `kernel` over n items.
//
// round-5 consolidation (w3 s21, merging w0's `utils/proj_launch.h`
// `fitted_block_dim` with w3's s20 `spread_block_dim`): start from the
// occupancy-max block size and halve it **in whole warps**, down to one warp,
// while the resulting grid is smaller than `blocks_per_sm` blocks per SM.
// Once n is big enough that the grid already covers the device several times
// over, the occupancy-max answer is returned unchanged, so this only ever acts
// in the small-grid regime.
//
// Two criteria were on the table and this is the union of them:
//  * w3's binary test ("is the natural grid smaller than the SM count?")
//    catches only the extreme case, gridDim 1-3 on a 36-SM device;
//  * w0's ramp also catches **wave quantisation**: at 255 registers exactly one
//    256-thread block is resident per SM, so a grid of 48 on 40 SMs runs two
//    waves for 1.2 waves of work. The ramp is the better-justified criterion
//    and it subsumes the binary one, so the ramp is what ships.
//
// The one correction to w0's version: repeated halving of a block size that is
// a multiple of 32 but not a power of two leaves warps straddling block
// boundaries (`best_block_dim` really does return 640, 768 and 896 at sites in
// this backend; 640 -> 320 -> 160 -> 80 -> **40**). Halving the *warp count*
// instead keeps every block a whole number of warps, which is what makes the
// warp composition -- and therefore `__activemask()`, and therefore the
// bit-identity argument -- independent of the geometry.
template <typename Kernel>
int spread_block_dim(int n, Kernel kernel, size_t shared_mem_size = 0)
{
    int bd = best_block_dim(kernel, shared_mem_size);
    if(!grid_spread_enabled() || n <= 0 || bd <= 32 || (bd % 32) != 0)
        return bd;
    if(int forced = grid_spread_block())
        return forced < bd ? forced : bd;
    int       warps = bd / 32;
    const int want  = grid_spread_blocks_per_sm() * device_sm_count();
    while(warps > 1 && (n + warps * 32 - 1) / (warps * 32) < want)
        warps >>= 1;
    return warps * 32;
}

template <typename Kernel>
int spread_grid_dim(int n, Kernel kernel, size_t shared_mem_size = 0)
{
    int bs = spread_block_dim(n, kernel, shared_mem_size);
    return (n + bs - 1) / bs;
}

// ---------------------------------------------------------------------------
// Device-side bit-identity proof for a geometry-only change (perf round 5, w3).
//
// UIPC_GRID_SPREAD_VERIFY=1 makes an instrumented call site run the SAME kernel
// twice per invocation: first with the occupancy-max geometry `best_*_dim`
// would have picked (the reference), then — after a D2D snapshot of every
// output region — with the spread geometry that ships. Every 64-bit word of
// every output is then compared on device and the mismatch count is
// accumulated. The live state after the pair is the spread launch's, so the
// simulation still runs the shipped path while it is being checked.
//
// This is generic: the call site declares its output regions, nothing about
// the kernel signature is baked in here.
// ---------------------------------------------------------------------------
inline bool grid_spread_verify_enabled()
{
    static const bool on = []
    {
        const char* e = std::getenv("UIPC_GRID_SPREAD_VERIFY");
        return e && e[0] != '0';
    }();
    return on;
}

// 32-bit words, not 64: a TripletMatrixView's row/col index arrays are `int*`
// at an arbitrary triplet offset, so an 8-byte load off them can be misaligned.
template <int Dummy>
__global__ void spread_verify_cmp_kernel(const unsigned int* a,
                                         const unsigned int* b,
                                         unsigned long long* counter,
                                         int                 nwords)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nwords)
        return;
    if(a[i] != b[i])
        atomicAdd(counter, 1ull);
}

class SpreadVerifier
{
  public:
    explicit SpreadVerifier(const char* tag)
        : m_tag(tag)
    {
    }

    ~SpreadVerifier()
    {
        // host-side counts only: no CUDA call at static-destruction time
        if(m_words)
            std::fprintf(stderr,
                         "[SpreadVerify] %-46s %llu output words (32-bit) compared, %llu mismatching\n",
                         m_tag,
                         m_words,
                         m_mismatch);
    }

    static bool on() { return grid_spread_verify_enabled(); }

    // UIPC_GRID_SPREAD_ONLY=<substring> restricts the spread geometry to the
    // call sites whose tag contains <substring>; every other site keeps the
    // occupancy-max geometry. This is the per-site A/B instrument: it is how
    // the s21 sites are measured against the s20 ones in one build.
    bool spread() const
    {
        static const char* only = std::getenv("UIPC_GRID_SPREAD_ONLY");
        if(!only || !only[0])
            return true;
        return std::strstr(m_tag, only) != nullptr;
    }

    // phase 1: declare the output regions written by the reference launch
    void begin() { m_regions.clear(); m_bytes = 0; }

    void add(const void* p, size_t bytes)
    {
        if(!p || bytes == 0)
            return;
        m_regions.emplace_back(p, bytes);
        m_bytes += (bytes + 7) & ~size_t(7);  // keep each region 8-byte aligned
    }

    template <typename View>
    void add_buffer(const View& v)
    {
        using Elem = std::remove_cv_t<std::remove_reference_t<decltype(*v.data())>>;
        add(v.data(), (size_t)v.size() * sizeof(Elem));
    }

    template <typename V>
    void add_triplet(const V& v)
    {
        add_buffer(v.values());
        add_buffer(v.row_indices());
        add_buffer(v.col_indices());
    }

    template <typename V>
    void add_doublet(const V& v)
    {
        add_buffer(v.values());
        add_buffer(v.indices());
    }

    // phase 2: copy those regions aside (call after the reference launch)
    void snapshot()
    {
        if(m_bytes == 0)
            return;
        if(m_bytes > m_cap)
        {
            if(m_shadow)
                cudaFree(m_shadow);
            cudaMalloc(&m_shadow, m_bytes);
            m_cap = m_bytes;
        }
        size_t off = 0;
        for(auto&& [p, n] : m_regions)
        {
            cudaMemcpyAsync(m_shadow + off, p, n, cudaMemcpyDeviceToDevice, nullptr);
            off += (n + 7) & ~size_t(7);
        }
    }

    // phase 3: compare the live outputs (written by the spread launch) with them
    void compare()
    {
        if(m_bytes == 0)
            return;
        if(!m_counter)
        {
            cudaMalloc(&m_counter, sizeof(unsigned long long));
            cudaMemset(m_counter, 0, sizeof(unsigned long long));
        }
        size_t off = 0;
        for(auto&& [p, n] : m_regions)
        {
            int nw = (int)(n / sizeof(unsigned int));
            if(nw > 0)
            {
                auto k = spread_verify_cmp_kernel<0>;
                k<<<(nw + 255) / 256, 256, 0, nullptr>>>(
                    reinterpret_cast<const unsigned int*>(p),
                    reinterpret_cast<const unsigned int*>(m_shadow + off),
                    m_counter,
                    nw);
                m_words += (unsigned long long)nw;
            }
            off += (n + 7) & ~size_t(7);
        }
        // drain to the host now, so the report needs no CUDA context at exit
        unsigned long long h = 0;
        cudaMemcpy(&h, m_counter, sizeof(h), cudaMemcpyDeviceToHost);
        m_mismatch += h;
        cudaMemset(m_counter, 0, sizeof(unsigned long long));
    }

    // Kernels that *accumulate* into one of their outputs (k3 adds into
    // diag_hessian) are not idempotent, so the reference launch has to be
    // undone before the shipped one runs. Declare such buffers here: they are
    // saved before the reference launch and restored before the live one.
    void begin_inout()
    {
        m_in.clear();
        m_in_bytes = 0;
    }

    void add_in(const void* p, size_t bytes)
    {
        if(!p || bytes == 0)
            return;
        m_in.emplace_back(p, bytes);
        m_in_bytes += (bytes + 7) & ~size_t(7);
    }

    template <typename View>
    void add_in_buffer(const View& v)
    {
        using Elem = std::remove_cv_t<std::remove_reference_t<decltype(*v.data())>>;
        add_in(v.data(), (size_t)v.size() * sizeof(Elem));
    }

    void save_inputs()
    {
        if(m_in_bytes == 0)
            return;
        if(m_in_bytes > m_in_cap)
        {
            if(m_in_shadow)
                cudaFree(m_in_shadow);
            cudaMalloc(&m_in_shadow, m_in_bytes);
            m_in_cap = m_in_bytes;
        }
        size_t off = 0;
        for(auto&& [p, n] : m_in)
        {
            cudaMemcpyAsync(m_in_shadow + off, p, n, cudaMemcpyDeviceToDevice, nullptr);
            off += (n + 7) & ~size_t(7);
        }
    }

    void restore_inputs()
    {
        size_t off = 0;
        for(auto&& [p, n] : m_in)
        {
            cudaMemcpyAsync(const_cast<void*>(p), m_in_shadow + off, n, cudaMemcpyDeviceToDevice, nullptr);
            off += (n + 7) & ~size_t(7);
        }
    }

  private:
    const char*                                 m_tag;
    char*                                       m_in_shadow = nullptr;
    size_t                                      m_in_cap    = 0;
    size_t                                      m_in_bytes  = 0;
    std::vector<std::pair<const void*, size_t>> m_in;
    char*                                       m_shadow  = nullptr;
    size_t                                      m_cap     = 0;
    size_t                                      m_bytes   = 0;
    unsigned long long*                         m_counter = nullptr;
    unsigned long long                          m_words    = 0;
    unsigned long long                          m_mismatch = 0;
    std::vector<std::pair<const void*, size_t>> m_regions;
};

// The shipped launch wrapper. `launch(grid, block)` performs the `<<<>>>`;
// `declare(v)` names the kernel's output regions. With UIPC_GRID_SPREAD_VERIFY
// unset (the shipped configuration) this is exactly one launch with the spread
// geometry and the verifier costs a single static bool test.
template <typename Kernel, typename Launch, typename Declare, typename Inout>
inline void launch_spread_io(
    SpreadVerifier& sv, int n, Kernel k, Launch&& launch, Declare&& declare, Inout&& inout)
{
    if(n <= 0)
        return;
    const bool use_spread = sv.spread();
    if(!use_spread)
    {
        launch(best_grid_dim(n, k), best_block_dim(k));
        return;
    }
    const bool verify = SpreadVerifier::on();
    if(verify)
    {
        sv.begin_inout();
        inout(sv);
        sv.save_inputs();
        launch(best_grid_dim(n, k), best_block_dim(k));
        sv.begin();
        declare(sv);
        sv.snapshot();
        sv.restore_inputs();
    }
    launch(spread_grid_dim(n, k), spread_block_dim(n, k));
    if(verify)
        sv.compare();
}

template <typename Kernel, typename Launch, typename Declare>
inline void launch_spread(SpreadVerifier& sv, int n, Kernel k, Launch&& launch, Declare&& declare)
{
    if(n <= 0)
        return;
    const bool use_spread = sv.spread();
    if(!use_spread)
    {
        launch(best_grid_dim(n, k), best_block_dim(k));
        return;
    }
    const bool verify = SpreadVerifier::on();
    if(verify)
    {
        launch(best_grid_dim(n, k), best_block_dim(k));
        sv.begin();
        declare(sv);
        sv.snapshot();
    }
    launch(spread_grid_dim(n, k), spread_block_dim(n, k));
    if(verify)
        sv.compare();
}
}  // namespace uipc::backend::cuda_tool

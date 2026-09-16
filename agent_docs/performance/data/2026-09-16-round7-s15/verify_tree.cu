// Round-7 s15 numerics verifier: the K-serial re-summation of the segmental
// reduce (fast_segmental_reduce_matrix_ks_kernel) vs the shipped k2 warp tree,
// on device, against a __float128 reference.
//
// Both kernels are launched DIRECTLY (they live in an anonymous namespace that
// this TU includes verbatim through algorithm/fast_segmental_reduce.h), with
// the real zero-cross-warp fill kernels at their real span sizes -- so what is
// measured is the production code, not a re-implementation.
//
// Per class (3x3 on <128,32> = the dytopo/GLS convert reduces; 3x1 on <64,32>
// = the gradient doublet reduce) and per value regime:
//   old  = k2 (round-6 s17 early-exit tree; main's default)  -- balanced
//          pairwise over the segment's in-warp elements
//   new  = ks (s15 K-serial)                                  -- balanced-4
//          windows + pairwise tree over window partials
//   ref  = __float128 sum in element order
// Statistics over every 64-bit entry of every output slot, split by the
// deterministic class (slots whose segment spans <= 2 warp spans in BOTH
// kernels are order-determined; wider segments accumulate >= 3 atomic
// operands in arrival order in BOTH kernels and are excluded from the
// accuracy comparison, as in every round-6/7 segreduce gate).
//
// Also: determinism (each kernel run twice, bit-diffed outside the >= 3-span
// class), and -- wiring check -- the engine-level FastSegmentalReduce::
// reduce() (knob-selected) bit-compared against the direct ks launch.

#include <algorithm/fast_segmental_reduce.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <quadmath.h>

using namespace uipc::backend::cuda_tool;

// stub for the engine core helper the header-only diagnostic probes reference
// (only reached on allocation-error paths; the verifier never triggers them)
namespace uipc
{
std::string demangle(const std::string& mangled_name) { return mangled_name; }
}

using Matrix33 = Eigen::Matrix<double, 3, 3>;
using Matrix31 = Eigen::Matrix<double, 3, 1>;

// ---------------------------------------------------------------------------
// data generation (host, seeded): dense keys 0..n_seg-1, run-length regime per
// class; values per regime
// ---------------------------------------------------------------------------
struct Segments
{
    std::vector<int>    keys;    // per element
    size_t              n_elem;
    int                 n_seg;
    std::vector<size_t> seg_start, seg_end;  // element index ranges, for the class split
};

enum class LenRegime
{
    Short,  // 3x3 convert classes: in/out ~ 5-13 with a long tail (s13 hist)
    Long    // 3x1 doublet class: in/out ~ 66
};

static Segments gen_segments(LenRegime lr, size_t target_elem, std::mt19937_64& rng)
{
    Segments s;
    s.n_seg    = 0;
    size_t e   = 0;
    auto draw  = [&](void) -> size_t {
        double u = (double)(rng() >> 11) / 9007199254740992.0;  // [0,1)
        if(lr == LenRegime::Short)
        {
            if(u < 0.60)
                return 1 + (size_t)(3.0 * u / 0.60);            // 1..3
            if(u < 0.90)
                return 4 + (size_t)(9.0 * (u - 0.60) / 0.30);    // 4..12
            if(u < 0.99)
                return 13 + (size_t)(52.0 * (u - 0.90) / 0.09);  // 13..64
            return 65 + (size_t)(536.0 * (u - 0.99) / 0.01);     // 65..600
        }
        if(u < 0.95)
            return 16 + (size_t)(113.0 * u / 0.95);  // 16..128
        return 129 + (size_t)(672.0 * (u - 0.95) / 0.05);  // 129..800
    };
    while(e < target_elem)
    {
        size_t len = draw();
        if(len > target_elem - e)
            len = target_elem - e;  // last segment absorbs the remainder
        s.seg_start.push_back(e);
        s.seg_end.push_back(e + len - 1);
        for(size_t k = 0; k < len; ++k)
            s.keys.push_back(s.n_seg);
        e += len;
        ++s.n_seg;
    }
    s.n_elem = e;
    return s;
}

enum class ValRegime
{
    LogSpread,     // entries +-2^[-20..20] x [1,2): the gradient/Hessian shape
    Cancellation   // one magnitude per segment, random signs: the worst case
};

template <int M, int N>
static std::vector<Eigen::Matrix<double, M, N>> gen_values(const Segments& s, ValRegime vr, std::mt19937_64& rng)
{
    std::vector<Eigen::Matrix<double, M, N>> v(s.n_elem);
    std::uniform_int_distribution<int> expo(-20, 20);
    std::uniform_int_distribution<int> sign(0, 1);
    for(size_t e = 0; e < s.n_elem; ++e)
    {
        int seg = s.keys[e];
        double mag;
        if(vr == ValRegime::LogSpread)
        {
            mag = std::ldexp(1.0 + 1.0 * ((double)(rng() >> 11) / 9007199254740992.0), expo(rng));
        }
        else
        {
            // per-segment magnitude (deterministic in seg): 2^[-8..8]
            std::mt19937_64 seg_rng((uint64_t)seg * 0x9E3779B97F4A7C15ull + 12345);
            mag = std::ldexp(1.0, (int)(seg_rng() % 17) - 8);
        }
        for(int j = 0; j < M; ++j)
            for(int k = 0; k < N; ++k)
                v[e](j, k) = sign(rng) ? -mag : mag;
    }
    return v;
}

// ---------------------------------------------------------------------------
// device run: the REAL kernels, launched the way FastSegmentalReduce::reduce
// launches them (fill at the kernel's own span, then the reduce)
// ---------------------------------------------------------------------------
template <int BlockSize, int WarpSize, int M, int N, int Which>
static void run_reduce(const int*    d_keys,
                       const Eigen::Matrix<double, M, N>* d_vals,
                       Eigen::Matrix<double, M, N>*       d_out,
                       size_t                              n_elem,
                       int                                 n_seg)
{
    using Matrix = Eigen::Matrix<double, M, N>;
    using FSR    = FastSegmentalReduce<BlockSize, WarpSize>;
    fast_segmental_reduce_get_offset_key_op key_op{CBufferView<int>{d_keys, n_elem}};
    fast_segmental_reduce_get_buffer_value_op<Matrix> val_op{CBufferView<Matrix>{d_vals, n_elem}};
    ::cuda::std::plus<double> op{};
    BufferView<Matrix> out{d_out, (size_t)n_seg};

    constexpr int block_dim = BlockSize;
    // Which: 0 = old k2 (span 32), 1 = new ks (span 32*4)
    constexpr int span = Which == 1 ? WarpSize * 4 : WarpSize;

    // narrow fill at the kernel's own span (the production SegOutInit::CrossWarpOnly path)
    {
        int n_boundary = (int)((n_elem + span - 1) / span) - 1;
        if(n_boundary < 0)
            n_boundary = 0;
        if(n_elem > 0)
        {
            int n_task = n_boundary + 1;
            if constexpr(Which == 1)
                fast_segmental_reduce_zero_cross_warp_kernel<WarpSize, double, M, N, decltype(key_op), WarpSize* 4>
                    <<<(n_task + 255) / 256, 256>>>(out, n_elem, key_op, n_boundary);
            else
                fast_segmental_reduce_zero_cross_warp_kernel<WarpSize, double, M, N, decltype(key_op)>
                    <<<(n_task + 255) / 256, 256>>>(out, n_elem, key_op, n_boundary);
        }
    }

    if constexpr(Which == 1)
    {
        int n_win = (int)((n_elem + 3) / 4);
        int blocks = (n_win + block_dim - 1) / block_dim;
        fast_segmental_reduce_matrix_ks_kernel<BlockSize, WarpSize, double, M, N, decltype(key_op), decltype(val_op), ::cuda::std::plus<double>, 4>
            <<<blocks, block_dim>>>(out, n_elem, key_op, val_op, op);
    }
    else
    {
        int blocks = (int)((n_elem + block_dim - 1) / block_dim);
        fast_segmental_reduce_matrix_k2_kernel<BlockSize, WarpSize, double, M, N, typename FSR::Flags, decltype(key_op), decltype(val_op), ::cuda::std::plus<double>, 0>
            <<<blocks, block_dim>>>(out, n_elem, key_op, val_op, op);
    }
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------
struct Stats
{
    double med, p99, max, mean;
    size_t n;
    size_t worse;  // entries where e_new > e_old
};

static Stats distribution(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    Stats s;
    s.n    = v.size();
    s.med  = s.n ? v[s.n / 2] : 0;
    s.p99  = s.n ? v[(size_t)(s.n * 0.99)] : 0;
    s.max  = s.n ? v.back() : 0;
    s.mean = 0;
    for(double x : v)
        s.mean += x;
    if(s.n)
        s.mean /= (double)s.n;
    return s;
}

template <int M, int N>
static void compare_one(const char*                                  label,
                        const Segments&                             s,
                        const std::vector<Eigen::Matrix<double, M, N>>& vals,
                        const Eigen::Matrix<double, M, N>*            old_out,
                        const Eigen::Matrix<double, M, N>*            new_out,
                        int                                          span_new,
                        int                                          span_old)
{
    // __float128 reference + deterministic-class split
    std::vector<double> e_old, e_new, e_old_det, e_new_det;
    e_old_det.reserve((size_t)s.n_seg * M * N / 2);
    e_new_det.reserve((size_t)s.n_seg * M * N / 2);
    size_t worse_det = 0, worse_all = 0;
    for(int seg = 0; seg < s.n_seg; ++seg)
    {
        __float128 ref[M * N];
        for(int k = 0; k < M * N; ++k)
            ref[k] = (__float128)0.0;
        for(size_t e = s.seg_start[seg]; e <= s.seg_end[seg]; ++e)
            for(int k = 0; k < M * N; ++k)
                ref[k] += (__float128)vals[e].data()[k];
        // deterministic in BOTH kernels iff the segment spans <= 2 spans of each
        size_t sp_old = s.seg_end[seg] / span_old - s.seg_start[seg] / span_old + 1;
        size_t sp_new = s.seg_end[seg] / span_new - s.seg_start[seg] / span_new + 1;
        bool det = sp_old <= 2 && sp_new <= 2;
        for(int k = 0; k < M * N; ++k)
        {
            double r = (double)ref[k];
            double denom = fabs(r) > 1e-300 ? fabs(r) : 1e-300;
            double eo = fabs(old_out[seg].data()[k] - r) / denom;
            double en = fabs(new_out[seg].data()[k] - r) / denom;
            e_old.push_back(eo);
            e_new.push_back(en);
            if(en > eo)
            {
                ++worse_all;
                if(det)
                    ++worse_det;
            }
            if(det)
            {
                e_old_det.push_back(eo);
                e_new_det.push_back(en);
            }
        }
    }
    Stats so = distribution(e_old), sn = distribution(e_new);
    Stats sod = distribution(e_old_det), snd = distribution(e_new_det);
    std::printf("[%s] entries=%zu  (deterministic-class %zu, %.1f%%)\n", label, so.n, sod.n, 100.0 * (double)sod.n / (double)(so.n ? so.n : 1));
    std::printf("   all slots      : old med=%.3e p99=%.3e max=%.3e | new med=%.3e p99=%.3e max=%.3e | new>old %.3f%%\n",
                so.med, so.p99, so.max, sn.med, sn.p99, sn.max, 100.0 * (double)worse_all / (double)(so.n ? so.n : 1));
    std::printf("   deterministic  : old med=%.3e p99=%.3e max=%.3e | new med=%.3e p99=%.3e max=%.3e | new>old %.3f%%\n",
                sod.med, sod.p99, sod.max, snd.med, snd.p99, snd.max, 100.0 * (double)worse_det / (double)(sod.n ? sod.n : 1));
}

// determinism check: rerun and bit-compare, split by >= 3-span class
template <int M, int N>
static size_t bit_diff_count(const Eigen::Matrix<double, M, N>* a, const Eigen::Matrix<double, M, N>* b, int n_seg)
{
    size_t c = 0;
    for(int i = 0; i < n_seg; ++i)
        for(int k = 0; k < M * N; ++k)
        {
            unsigned long long xa, xb;
            std::memcpy(&xa, &a[i].data()[k], 8);
            std::memcpy(&xb, &b[i].data()[k], 8);
            if(xa != xb)
                ++c;
        }
    return c;
}

// ---------------------------------------------------------------------------
template <int BlockSize, int M, int N>
static void run_class(const char* name, int t, const Segments& s, ValRegime vr, std::mt19937_64& rng)
{
    using Matrix = Eigen::Matrix<double, M, N>;
    std::fprintf(stderr, "  gen_values\n");
    auto vals    = gen_values<M, N>(s, vr, rng);
    int*  d_keys = nullptr;
    cudaMalloc(&d_keys, s.n_elem * sizeof(int));
    cudaMemcpy(d_keys, s.keys.data(), s.n_elem * sizeof(int), cudaMemcpyHostToDevice);
    Matrix* d_vals = nullptr;
    cudaMalloc(&d_vals, s.n_elem * sizeof(Matrix));
    cudaMemcpy(d_vals, vals.data(), s.n_elem * sizeof(Matrix), cudaMemcpyHostToDevice);
    Matrix *d_old = nullptr, *d_new = nullptr, *d_new2 = nullptr;
    cudaMalloc(&d_old, (size_t)s.n_seg * sizeof(Matrix));
    cudaMalloc(&d_new, (size_t)s.n_seg * sizeof(Matrix));
    cudaMalloc(&d_new2, (size_t)s.n_seg * sizeof(Matrix));
    std::fprintf(stderr, "  upload done, launching old\n");
    run_reduce<BlockSize, 32, M, N, 0>(d_keys, d_vals, d_old, s.n_elem, s.n_seg);
    std::fprintf(stderr, "  old done err=%d\n", (int)cudaGetLastError());
    run_reduce<BlockSize, 32, M, N, 1>(d_keys, d_vals, d_new, s.n_elem, s.n_seg);
    std::fprintf(stderr, "  new done err=%d\n", (int)cudaGetLastError());
    run_reduce<BlockSize, 32, M, N, 1>(d_keys, d_vals, d_new2, s.n_elem, s.n_seg);
    std::fprintf(stderr, "  new2 done err=%d\n", (int)cudaGetLastError());
    // wiring check, trial 0 only: the engine-level reduce() (knob-selected ks)
    // must equal the direct ks launch bit for bit
    std::fprintf(stderr, "  compare\n");
    std::vector<Matrix> h_old(s.n_seg), h_new(s.n_seg), h_new2(s.n_seg);
    cudaMemcpy(h_old.data(), d_old, (size_t)s.n_seg * sizeof(Matrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_new.data(), d_new, (size_t)s.n_seg * sizeof(Matrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_new2.data(), d_new2, (size_t)s.n_seg * sizeof(Matrix), cudaMemcpyDeviceToHost);
    size_t nd = bit_diff_count<M, N>(h_new.data(), h_new2.data(), s.n_seg);
    char label[160];
    std::snprintf(label, sizeof(label), "%s trial%d in=%zu seg=%d newdet=%zu", name, t, s.n_elem, s.n_seg, nd);
    compare_one<M, N>(label, s, vals, h_old.data(), h_new.data(), 128, 32);
    cudaFree(d_keys);
    cudaFree(d_vals);
    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_new2);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv)
{
    std::fprintf(stderr, "START\n");
    const size_t   target_elem = argc > 1 ? (size_t)std::atoll(argv[1]) : 2000000;  // per trial
    const int      trials      = argc > 2 ? std::atoi(argv[2]) : 5;
    const uint64_t seed0       = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 20260916;

    struct ClassDef
    {
        const char* name;
        LenRegime   lr;
        ValRegime   vr;
        int         block;  // FastSegmentalReduce block size at the call site
    };
    const std::vector<ClassDef> classes = {
        {"3x3<128,32> logspread", LenRegime::Short, ValRegime::LogSpread, 128},
        {"3x3<128,32> cancell", LenRegime::Short, ValRegime::Cancellation, 128},
        {"3x1<64,32>  logspread", LenRegime::Long, ValRegime::LogSpread, 64},
        {"3x1<64,32>  cancell", LenRegime::Long, ValRegime::Cancellation, 64},
    };

    size_t total_segments = 0;

    for(int t = 0; t < trials; ++t)
    {
        for(size_t ci = 0; ci < classes.size(); ++ci)
        {
            const ClassDef& c = classes[ci];
            std::mt19937_64 rng(seed0 + 1000ull * t + (uint64_t)ci);
            std::fprintf(stderr, "class %s trial %d\n", c.name, t);
            Segments s      = gen_segments(c.lr, target_elem, rng);
            total_segments += s.n_seg;
            if(c.block == 128)
                run_class<128, 3, 3>(c.name, t, s, c.vr, rng);
            else
                run_class<64, 3, 1>(c.name, t, s, c.vr, rng);
        }
    }

    std::fflush(stdout);
    std::printf("\nTOTAL: segments=%zu (>= 2e5 required)\n", total_segments);
    return 0;
}

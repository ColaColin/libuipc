// s24 numerics verifier: the shipped part-1 projection (s19 tridiagonal QL +
// K16 blocked translation-free basis) against the old one (Eigen
// SelfAdjointEigenSolver + dense 12x9 basis), fed by the repo's own device
// functions over randomised EE and PT pairs.  s16/s14 verifier pattern.
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/make_spd.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance/edge_edge_mollifier.h>
#include <curand_kernel.h>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;
using namespace uipc::backend::cuda::sym::codim_ipc_simplex_contact;

__device__ __forceinline__ double u01(curandState* s) { return curand_uniform_double(s); }
__device__ __forceinline__ Vector3 rnd3(curandState* s, double a)
{
    return Vector3(a * (2 * u01(s) - 1), a * (2 * u01(s) - 1), a * (2 * u01(s) - 1));
}

struct Res
{
    double maxrel;
    int    nan;
    int    negdiag;
    int    moll;
    int    dim2, dim3, dim4;
};

__global__ void verify_ee(unsigned long long seed, int n, double* rel, int* flags, int* dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    curandState st;
    curand_init(seed, i, 0, &st);

    // rest configuration: two edges, one in eight built nearly parallel so the
    // mollifier fires; the whole pair translated up to 10 length units away
    Vector3 o = rnd3(&st, 10.0);
    Vector3 ea_dir = rnd3(&st, 1.0).normalized();
    Vector3 eb_dir;
    bool    parallel = (i % 8) == 0;
    if(parallel)
        eb_dir = (ea_dir + rnd3(&st, 2e-2)).normalized();
    else
        eb_dir = rnd3(&st, 1.0).normalized();
    double la = 0.2 + 2.0 * u01(&st);
    double lb = 0.2 + 2.0 * u01(&st);
    double gap = 1e-4 + 5e-2 * u01(&st);
    Vector3 sep = ea_dir.cross(eb_dir);
    if(sep.norm() < 1e-8) sep = Vector3(0, 0, 1);
    sep.normalize();

    Vector3 Ea0 = o;
    Vector3 Ea1 = o + la * ea_dir;
    Vector3 Eb0 = o + gap * sep + (u01(&st) - 0.5) * la * ea_dir + rnd3(&st, 0.1);
    Vector3 Eb1 = Eb0 + lb * eb_dir;
    // rest positions: perturbed copies, so eps_x is a realistic rest threshold
    Vector3 t0a0 = Ea0 + rnd3(&st, 1e-2), t0a1 = Ea1 + rnd3(&st, 1e-2);
    Vector3 t0b0 = Eb0 + rnd3(&st, 1e-2), t0b1 = Eb1 + rnd3(&st, 1e-2);

    double thickness = (i % 3 == 0) ? 0.0 : 0.1 * gap;
    double d_hat     = 0.01 + 0.09 * u01(&st);
    double kt2       = 1e6 * powf(10.0f, (float)(2.0 * u01(&st)));  // kappa dt^2 up to 1e8

    Vector4i flag = distance::edge_edge_distance_flag(Ea0, Ea1, Eb0, Eb1);
    Vector12    G;
    Matrix12x12 H0, H1;
    bool        mollified = true;
    mollified_EE_barrier_gradient_hessian<true>(
        G, H0, mollified, flag, kt2, d_hat, thickness, t0a0, t0a1, t0b0, t0b1, Ea0, Ea1, Eb0, Eb1);
    H1 = H0;

    if(!mollified)
    {
        EE_barrier_make_spd<0>(H0, flag, Ea0, Ea1, Eb0, Eb1);
        EE_barrier_make_spd<1>(H1, flag, Ea0, Ea1, Eb0, Eb1);
    }
    else
    {
        make_spd_translation_free_4x3<0>(H0);
        make_spd_translation_free_4x3_blocked<1>(H1);
    }

    double scale = 0.0, diff = 0.0;
    int    nan = 0, neg = 0;
    for(int a = 0; a < 12; ++a)
        for(int b = 0; b < 12; ++b)
        {
            double x = H0(a, b), y = H1(a, b);
            if(isnan(y) || isinf(y)) nan |= 1;
            if(isnan(x) || isinf(x)) nan |= 2;
            scale = fmax(scale, fabs(x));
            diff  = fmax(diff, fabs(x - y));
        }
    for(int a = 0; a < 12; ++a)
        if(H1(a, a) < -1e-9 * scale) neg = 1;
    rel[i]   = (scale > 0.0 && nan == 0) ? diff / scale : 0.0;
    flags[i] = (nan & 1) | (neg ? 2 : 0) | (mollified ? 4 : 0) | ((nan & 2) ? 8 : 0);
    dims[i]  = flag[0] + flag[1] + flag[2] + flag[3] > 0 ? 0 : 0;
    int d = 0;
    for(int k = 0; k < 4; ++k) d += (flag[k] != 0) ? 1 : 0;
    dims[i] = d;
}

__global__ void verify_pt(unsigned long long seed, int n, double* rel, int* flags, int* dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    curandState st;
    curand_init(seed + 777, i, 0, &st);
    Vector3 o  = rnd3(&st, 10.0);
    Vector3 T0 = o;
    Vector3 T1 = o + rnd3(&st, 1.0);
    Vector3 T2 = o + rnd3(&st, 1.0);
    Vector3 nrm = (T1 - T0).cross(T2 - T0);
    if(nrm.norm() < 1e-10) nrm = Vector3(0, 0, 1);
    nrm.normalize();
    double  gap = 1e-4 + 5e-2 * u01(&st);
    // sample the point over and beyond the triangle so all three dims occur
    double  u = 1.6 * u01(&st) - 0.3, v = 1.6 * u01(&st) - 0.3;
    Vector3 P = T0 + u * (T1 - T0) + v * (T2 - T0) + gap * nrm;

    double thickness = (i % 3 == 0) ? 0.0 : 0.1 * gap;
    double d_hat     = 0.01 + 0.09 * u01(&st);
    double kt2       = 1e6 * powf(10.0f, (float)(2.0 * u01(&st)));

    Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);
    Vector12    G;
    Matrix12x12 H0, H1;
    PT_barrier_gradient_hessian(G, H0, flag, kt2, d_hat, thickness, P, T0, T1, T2);
    H1 = H0;
    PT_barrier_make_spd<0>(H0, flag, P, T0, T1, T2);
    PT_barrier_make_spd<1>(H1, flag, P, T0, T1, T2);

    double scale = 0.0, diff = 0.0;
    int    nan = 0, neg = 0;
    for(int a = 0; a < 12; ++a)
        for(int b = 0; b < 12; ++b)
        {
            double x = H0(a, b), y = H1(a, b);
            if(isnan(y) || isinf(y)) nan |= 1;
            if(isnan(x) || isinf(x)) nan |= 2;
            scale = fmax(scale, fabs(x));
            diff  = fmax(diff, fabs(x - y));
        }
    for(int a = 0; a < 12; ++a)
        if(H1(a, a) < -1e-9 * scale) neg = 1;
    rel[i]   = (scale > 0.0 && nan == 0) ? diff / scale : 0.0;
    flags[i] = (nan & 1) | (neg ? 2 : 0) | ((nan & 2) ? 8 : 0);
    int d = 0;
    for(int k = 0; k < 4; ++k) d += (flag[k] != 0) ? 1 : 0;
    dims[i] = d;
}

static void report(const char* tag, std::vector<double>& r, std::vector<int>& f, std::vector<int>& d)
{
    int nan = 0, neg = 0, moll = 0, nan_old = 0, dc[6] = {0, 0, 0, 0, 0, 0};
    for(size_t i = 0; i < f.size(); ++i)
    {
        if(f[i] & 1) ++nan;
        if(f[i] & 2) ++neg;
        if(f[i] & 4) ++moll;
        if(f[i] & 8) ++nan_old;
        if(d[i] >= 0 && d[i] < 6) ++dc[d[i]];
    }
    std::sort(r.begin(), r.end());
    auto q = [&](double p) { return r[std::min(r.size() - 1, (size_t)(p * r.size()))]; };
    printf("%-6s n=%zu  p50 %.3e  p90 %.3e  p99 %.3e  p99.9 %.3e  max %.3e | NaN %d  negdiag %d  mollified %d | dim2/3/4 %d/%d/%d\n",
           tag, r.size(), q(0.5), q(0.9), q(0.99), q(0.999), r.back(), nan, neg, moll, dc[2], dc[3], dc[4]);
    printf("       NaN new-path-only %d, old path NaN %d (degenerate samples, both paths)\n", nan, nan_old);
}

int main(int argc, char** argv)
{
    int  n     = argc > 1 ? atoi(argv[1]) : 300000;
    int  seeds = argc > 2 ? atoi(argv[2]) : 3;
    double *rel;
    int    *flg, *dim;
    cudaMallocManaged(&rel, sizeof(double) * n);
    cudaMallocManaged(&flg, sizeof(int) * n);
    cudaMallocManaged(&dim, sizeof(int) * n);
    for(int s = 0; s < seeds; ++s)
    {
        verify_ee<<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        char tag[16];
        snprintf(tag, sizeof(tag), "EE s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pt<<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        char tag[16];
        snprintf(tag, sizeof(tag), "PT s%d", s);
        report(tag, r, f, d);
    }
    printf("cuda err: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}

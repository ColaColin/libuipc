// s26 numerics verifier: the shipped contact part-2 projection (s19
// tridiagonal QL at N=4 + s26's basis-free range reduction) against the old
// one (Eigen SelfAdjointEigenSolver + the explicit basis Q of
// barrier_range_basis), fed by the repo's own device functions over
// randomised PE and PP pairs.  s14/s16/s24 verifier pattern.
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/make_spd.h>
#include <utils/distance/distance_flagged.h>
#include <curand_kernel.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#ifndef KLO
#define KLO 1e6
#endif
using namespace uipc;
using namespace uipc::backend::cuda;
using namespace uipc::backend::cuda::sym::codim_ipc_simplex_contact;

__device__ __forceinline__ double u01(curandState* s) { return curand_uniform_double(s); }
__device__ __forceinline__ Vector3 rnd3(curandState* s, double a)
{
    return Vector3(a * (2 * u01(s) - 1), a * (2 * u01(s) - 1), a * (2 * u01(s) - 1));
}


// A second, equally valid instance of the OLD path: byte-for-byte the dense-Q
// reduction of `make_spd_contact`, except that the arbitrary tangent seed is
// (0,0,1)-based instead of (1,0,0)-based.  The tangent frame (t1, t2) is an
// arbitrary orthonormal completion of n^ -- the projection does not depend on
// it mathematically -- so old-vs-alt measures the OLD path's own sensitivity
// to a choice it already makes arbitrarily.  Reference noise floor, per
// PERF_METHOD 2.3.
template <int N, int M, bool Clamp = true>
__device__ void make_spd_contact_alt(Matrix<Float, N, N>&     H,
                                     const Vector<IndexT, M>& act,
                                     const Vector<Float, M>&  s,
                                     const Vector3&           gap)
{
    constexpr int         NM = 3 * M;
    Matrix<Float, NM, NM> Hs;
    for(int a = 0; a < M; ++a)
        for(int b = 0; b < M; ++b)
            Hs.template block<3, 3>(3 * a, 3 * b) =
                H.template block<3, 3>(3 * act[a], 3 * act[b]);

    Matrix<Float, NM, M + 1> Q = Matrix<Float, NM, M + 1>::Zero();
    {
        Vector3 nh = gap.normalized();
        Vector3 a0 = Vector3(0.0, 0.0, 1.0);
        if(std::abs(nh[2]) >= Float(0.9))
            a0 = Vector3(0.0, 1.0, 0.0);
        Vector3          t1 = (a0 - nh.dot(a0) * nh).normalized();
        Vector3          t2 = nh.cross(t1);
        Vector<Float, M> sh = s.normalized();
        Matrix<Float, M, M - 1> U;
        for(int j = 0; j < M - 1; ++j)
        {
            Vector<Float, M> u;
            u.setZero();
            u[j]     = 1.0;
            u[M - 1] = -1.0;
            for(int pp = 0; pp < j; ++pp)
                u -= U.col(pp) * u.dot(U.col(pp));
            U.col(j) = u.normalized();
        }
        for(int k = 0; k < M; ++k)
        {
            Q.template block<3, 1>(3 * k, 0) = sh[k] * t1;
            Q.template block<3, 1>(3 * k, 1) = sh[k] * t2;
            for(int j = 0; j < M - 1; ++j)
                Q.template block<3, 1>(3 * k, 2 + j) = U(k, j) * nh;
        }
    }
    Matrix<Float, M + 1, M + 1> Hred = Q.transpose() * Hs * Q;
    if constexpr(Clamp)
        make_spd<M + 1, 0>(Hred);
    Matrix<Float, NM, NM> Hspd = Q * Hred * Q.transpose();
    for(int a = 0; a < M; ++a)
        for(int b = 0; b < M; ++b)
            H.template block<3, 3>(3 * act[a], 3 * act[b]) =
                Hspd.template block<3, 3>(3 * a, 3 * b);
}

__device__ void PE_make_spd_alt(Matrix9x9& H, const Vector3i& flag, const Vector3& P, const Vector3& E0, const Vector3& E1)
{
    const Vector3 X[3] = {P, E0, E1};
    IndexT        dim  = distance::detail::active_count(flag);
    if(dim == 2)
    {
        Vector2i act = distance::detail::pp_from_pe(flag);
        Vector2  s{1.0, -1.0};
        make_spd_contact_alt<9, 2>(H, act, s, X[act[0]] - X[act[1]]);
    }
    else
    {
        Vector3  e = E1 - E0;
        Float    t = (P - E0).dot(e) / e.squaredNorm();
        Vector3  s{1.0, t - 1.0, -t};
        Vector3i act{0, 1, 2};
        make_spd_contact_alt<9, 3>(H, act, s, P - (E0 + t * e));
    }
}

__device__ void PP_make_spd_alt(Matrix6x6& H, const Vector2i& flag, const Vector3& P0, const Vector3& P1)
{
    Vector2  s{1.0, -1.0};
    Vector2i act{0, 1};
    make_spd_contact_alt<6, 2>(H, act, s, P0 - P1);
}

// Clamp-free copies of the two reductions: Q (Q^T H Q) Q^T, i.e. the orthogonal
// projector onto range(Q) applied to H, with no eigenvalue clamping. If the two
// bases span the same subspace the results must agree to rounding; any larger
// difference in the clamped comparison is then the clamp boundary, not the
// algebra. (Old path = make_spd_contact_alt<.,.,false> with the shipping
// tangent seed is not needed: the projector is basis-independent by
// construction, so the alt seed is used and the only variable is my basis.)
template <int N, int M>
__device__ void reduce_noclamp_new(Matrix<Float, N, N>&     H,
                                   const Vector<IndexT, M>& act,
                                   const Vector<Float, M>&  s,
                                   const Vector3&           gap)
{
    const Vector<Float, M> sh = s.normalized();
    if constexpr(M == 2)
    {
        Matrix3x3 Hss = Matrix3x3::Zero();
        for(int a = 0; a < M; ++a)
            for(int b = 0; b < M; ++b)
                Hss += (sh[a] * sh[b]) * H.template block<3, 3>(3 * act[a], 3 * act[b]);
        for(int a = 0; a < M; ++a)
            for(int b = 0; b < M; ++b)
                H.template block<3, 3>(3 * act[a], 3 * act[b]) = (sh[a] * sh[b]) * Hss;
    }
    else
    {
        const Vector3 nh = gap.normalized();
        Vector3       v{sh[2] - sh[1], sh[0] - sh[2], sh[1] - sh[0]};
        v.normalize();
        Matrix3x3 Hss = Matrix3x3::Zero();
        Vector3   m   = Vector3::Zero();
        Float     c   = 0.0;
        for(int b = 0; b < 3; ++b)
        {
            Matrix3x3 Tb = Matrix3x3::Zero();
            Vector3   tb = Vector3::Zero();
            Vector3   qb = Vector3::Zero();
            for(int a = 0; a < 3; ++a)
            {
                auto          Aab = H.template block<3, 3>(3 * act[a], 3 * act[b]);
                const Vector3 w   = Aab * nh;
                Tb += sh[a] * Aab;
                tb += sh[a] * w;
                qb += v[a] * w;
            }
            Hss += sh[b] * Tb;
            m += v[b] * tb;
            c += v[b] * nh.dot(qb);
        }
        const Matrix3x3 A = Hss;
        const Matrix3x3 B = m * nh.transpose();
        const Matrix3x3 C = (c * nh) * nh.transpose();
        for(int b = 0; b < 3; ++b)
        {
            const Matrix3x3 Pb = sh[b] * A + v[b] * B;
            const Matrix3x3 Rb = sh[b] * B.transpose() + v[b] * C;
            for(int a = 0; a < 3; ++a)
                H.template block<3, 3>(3 * act[a], 3 * act[b]) = sh[a] * Pb + v[a] * Rb;
        }
    }
}

__device__ void PE_noclamp_old(Matrix9x9& H, const Vector3i& flag, const Vector3& P, const Vector3& E0, const Vector3& E1)
{
    const Vector3 X[3] = {P, E0, E1};
    if(distance::detail::active_count(flag) == 2)
    {
        Vector2i act = distance::detail::pp_from_pe(flag);
        Vector2  s{1.0, -1.0};
        make_spd_contact_alt<9, 2, false>(H, act, s, X[act[0]] - X[act[1]]);
    }
    else
    {
        Vector3  e = E1 - E0;
        Float    t = (P - E0).dot(e) / e.squaredNorm();
        Vector3  s{1.0, t - 1.0, -t};
        Vector3i act{0, 1, 2};
        make_spd_contact_alt<9, 3, false>(H, act, s, P - (E0 + t * e));
    }
}
__device__ void PE_noclamp_new(Matrix9x9& H, const Vector3i& flag, const Vector3& P, const Vector3& E0, const Vector3& E1)
{
    const Vector3 X[3] = {P, E0, E1};
    if(distance::detail::active_count(flag) == 2)
    {
        Vector2i act = distance::detail::pp_from_pe(flag);
        Vector2  s{1.0, -1.0};
        reduce_noclamp_new<9, 2>(H, act, s, X[act[0]] - X[act[1]]);
    }
    else
    {
        Vector3  e = E1 - E0;
        Float    t = (P - E0).dot(e) / e.squaredNorm();
        Vector3  s{1.0, t - 1.0, -t};
        Vector3i act{0, 1, 2};
        reduce_noclamp_new<9, 3>(H, act, s, P - (E0 + t * e));
    }
}

// arm: 0 = shipped (solver 1 + basis 1), 1 = basis change alone (solver 0)
template <int Arm, int N>
__device__ __forceinline__ void cmp(const Matrix<Float, N, N>& H0,
                                    const Matrix<Float, N, N>& H1,
                                    double&                    rel,
                                    int&                       flags)
{
    double scale = 0.0, diff = 0.0;
    int    nan = 0, neg = 0;
    for(int a = 0; a < N; ++a)
        for(int b = 0; b < N; ++b)
        {
            double x = H0(a, b), y = H1(a, b);
            if(isnan(y) || isinf(y)) nan |= 1;
            if(isnan(x) || isinf(x)) nan |= 2;
            scale = fmax(scale, fabs(x));
            diff  = fmax(diff, fabs(x - y));
        }
    for(int a = 0; a < N; ++a)
        if(H1(a, a) < -1e-9 * scale) neg = 1;
    rel   = (scale > 0.0 && nan == 0) ? diff / scale : 0.0;
    flags = (nan & 1) | (neg ? 2 : 0) | ((nan & 2) ? 8 : 0);
}

template <int Arm>
__global__ void verify_pe(unsigned long long seed, int n, double* rel, int* flags, int* dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    curandState st;
    curand_init(seed, i, 0, &st);

    // the whole pair translated up to 10 length units from the origin
    Vector3 o  = rnd3(&st, 10.0);
    Vector3 e  = rnd3(&st, 1.0).normalized();
    double  le = 0.2 + 2.0 * u01(&st);
    Vector3 E0 = o;
    Vector3 E1 = o + le * e;
    // t over [-0.3, 1.3] so both the interior (dim 3) and the endpoint (dim 2)
    // branches of point_edge_distance_flag occur
    double  t   = 1.6 * u01(&st) - 0.3;
    double  gap = 1e-4 + 5e-2 * u01(&st);
    Vector3 perp = e.cross(rnd3(&st, 1.0));
    if(perp.norm() < 1e-9) perp = Vector3(0, 0, 1);
    perp.normalize();
    Vector3 P = E0 + t * (E1 - E0) + gap * perp;

    double thickness = (i % 3 == 0) ? 0.0 : 0.1 * gap;
    double d_hat     = 0.01 + 0.09 * u01(&st);
    double kt2       = KLO * powf(10.0f, (float)(2.0 * u01(&st)));  // kappa dt^2 up to 1e8

    Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);
    Vector9   G;
    Matrix9x9 H0, H1;
    PE_barrier_gradient_hessian(G, H0, flag, kt2, d_hat, thickness, P, E0, E1);
    H1 = H0;
    // Arm 0: old reduced vs the shipped new reduced path
    // Arm 1: old reduced vs the basis change alone (old solver)
    // Arm 2: old reduced vs the *exact* full 9x9 projection (s16's standard)
    // Arm 3: shipped new reduced vs the exact full 9x9 projection
    if constexpr(Arm == 3)
        PE_barrier_make_spd<1, 1>(H0, flag, P, E0, E1);
    else
        PE_barrier_make_spd<0, 0>(H0, flag, P, E0, E1);
    if constexpr(Arm == 0)
        PE_barrier_make_spd<1, 1>(H1, flag, P, E0, E1);
    else if constexpr(Arm == 1)
        PE_barrier_make_spd<0, 1>(H1, flag, P, E0, E1);
    else if constexpr(Arm == 4)
        PE_make_spd_alt(H1, flag, P, E0, E1);
    else
        make_spd<9, 1>(H1);
    if constexpr(Arm == 5)
    {
        PE_barrier_gradient_hessian(G, H0, flag, kt2, d_hat, thickness, P, E0, E1);
        H1 = H0;
        PE_noclamp_old(H0, flag, P, E0, E1);
        PE_noclamp_new(H1, flag, P, E0, E1);
    }

    cmp<Arm, 9>(H0, H1, rel[i], flags[i]);
    int d = 0;
    for(int k = 0; k < 3; ++k) d += (flag[k] != 0) ? 1 : 0;
    dims[i] = d;
}

template <int Arm>
__global__ void verify_pp(unsigned long long seed, int n, double* rel, int* flags, int* dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    curandState st;
    curand_init(seed + 777, i, 0, &st);

    Vector3 o   = rnd3(&st, 10.0);
    double  gap = 1e-4 + 5e-2 * u01(&st);
    Vector3 dir = rnd3(&st, 1.0).normalized();
    Vector3 P0  = o;
    Vector3 P1  = o + gap * dir;

    double thickness = (i % 3 == 0) ? 0.0 : 0.1 * gap;
    double d_hat     = 0.01 + 0.09 * u01(&st);
    double kt2       = KLO * powf(10.0f, (float)(2.0 * u01(&st)));

    Vector2i  flag = distance::point_point_distance_flag(P0, P1);
    Vector6   G;
    Matrix6x6 H0, H1;
    PP_barrier_gradient_hessian(G, H0, flag, kt2, d_hat, thickness, P0, P1);
    H1 = H0;
    if constexpr(Arm == 3)
        PP_barrier_make_spd<1, 1>(H0, flag, P0, P1);
    else
        PP_barrier_make_spd<0, 0>(H0, flag, P0, P1);
    if constexpr(Arm == 0)
        PP_barrier_make_spd<1, 1>(H1, flag, P0, P1);
    else if constexpr(Arm == 1)
        PP_barrier_make_spd<0, 1>(H1, flag, P0, P1);
    else if constexpr(Arm == 4)
        PP_make_spd_alt(H1, flag, P0, P1);
    else
        make_spd<6, 1>(H1);

    cmp<Arm, 6>(H0, H1, rel[i], flags[i]);
    int d = 0;
    for(int k = 0; k < 2; ++k) d += (flag[k] != 0) ? 1 : 0;
    dims[i] = d;
}

static void report(const char* tag, std::vector<double>& r, std::vector<int>& f, std::vector<int>& d)
{
    int nan = 0, neg = 0, nan_old = 0, dc[6] = {0, 0, 0, 0, 0, 0};
    for(size_t i = 0; i < f.size(); ++i)
    {
        if(f[i] & 1) ++nan;
        if(f[i] & 2) ++neg;
        if(f[i] & 8) ++nan_old;
        if(d[i] >= 0 && d[i] < 6) ++dc[d[i]];
    }
    std::sort(r.begin(), r.end());
    auto q = [&](double p) { return r[std::min(r.size() - 1, (size_t)(p * r.size()))]; };
    printf("%-14s n=%zu  p50 %.3e  p90 %.3e  p99 %.3e  p99.9 %.3e  max %.3e | NaN new %d old %d  negdiag %d | dim2/dim3 %d/%d\n",
           tag, r.size(), q(0.5), q(0.9), q(0.99), q(0.999), r.back(), nan, nan_old, neg, dc[2], dc[3]);
}

int main(int argc, char** argv)
{
    int n     = argc > 1 ? atoi(argv[1]) : 300000;
    int seeds = argc > 2 ? atoi(argv[2]) : 3;
    double* rel;
    int *   flg, *dim;
    cudaMallocManaged(&rel, sizeof(double) * n);
    cudaMallocManaged(&flg, sizeof(int) * n);
    cudaMallocManaged(&dim, sizeof(int) * n);
    char tag[64];
    for(int s = 0; s < seeds; ++s)
    {
        verify_pe<0><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PE shipped s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pe<1><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PE basis s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pe<2><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PE old-vs-exact s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pe<3><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PE new-vs-exact s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pp<2><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PP old-vs-exact s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pp<3><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PP new-vs-exact s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pe<5><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PE noclamp s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pe<4><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PE old-vs-alt s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pp<4><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PP old-vs-alt s%d", s);
        report(tag, r, f, d);
    }
    for(int s = 0; s < seeds; ++s)
    {
        verify_pp<0><<<(n + 127) / 128, 128>>>(1234ull + 991ull * s, n, rel, flg, dim);
        cudaDeviceSynchronize();
        std::vector<double> r(rel, rel + n);
        std::vector<int>    f(flg, flg + n), d(dim, dim + n);
        snprintf(tag, sizeof(tag), "PP shipped s%d", s);
        report(tag, r, f, d);
    }
    printf("cuda err: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}

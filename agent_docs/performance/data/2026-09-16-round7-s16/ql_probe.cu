// Round-7 s16 diagnosis: where do the plastic hinge kernels' QL-core cycles go?
//
// Instruments, all on REALISTIC hinge inputs (s01's generator verbatim: both
// plastic models, crease-press parameter ranges):
//   1. Runtime decomposition of the 4x3 blocked helper <1,2> (the shipped arm):
//      V0 full | V1 QL stubbed (same I/O pattern) | V2 assembly-only.
//      -> QL core = V0 - V1, reconstruction = V1 - V2.
//   2. QL alone on the real restricted matrices Hr (precomputed by a phase-1
//      kernel so the solve sees exactly the shipped input distribution):
//      A real solve (eigen dump) | B stub | C load/store only.
//   3. Dynamic census of tql2: per-matrix QL iterations + rotation steps, and
//      per-warp max vs mean -> warp-divergence factor. Also: negative-eigenvalue
//      count of Hr (is the projection active?) and an LDL^T PSD test (would a
//      skip-if-PSD fast path ever fire?).
#include <utils/make_spd.h>
#include <finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h>
#include <cuda_tool/cuda_tool.h>
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;

struct HingeInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar, yield_p, Vdt2;
};

// ---------------- phase 1: realistic hinge -> H12 -> Hr (9x9) ----------------
__global__ void restrict_kernel(const HingeInput* __restrict__ in,
                                double* __restrict__ outH,   // 144
                                double* __restrict__ outHr,  // 81
                                int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    const HingeInput& h = in[I];
    Vector3 x0(h.x[0][0], h.x[0][1], h.x[0][2]);
    Vector3 x1(h.x[1][0], h.x[1][1], h.x[1][2]);
    Vector3 x2(h.x[2][0], h.x[2][1], h.x[2][2]);
    Vector3 x3(h.x[3][0], h.x[3][1], h.x[3][2]);
    Matrix12x12 H;
    if(h.yield_p < 0.0)  // model tag packed in sign
        sym::strain_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa);
    else
        sym::stress_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa, h.yield_p);
    H *= h.Vdt2;
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            outH[((size_t)I) * 144 + r * 12 + c] = H(r, c);
    // the shipped K16 forward restriction (SymAsm=2 arm: lower triangle)
    constexpr Float r2 = 0.70710678118654752440;
    constexpr Float r6 = 0.40824829046386301637;
    constexpr Float r12 = 0.28867513459481288225;
    constexpr Float hwb[3][4] = {{r2, -r2, 0.0, 0.0}, {r6, r6, -2.0 * r6, 0.0}, {r12, r12, r12, -3.0 * r12}};
    constexpr int nz[3] = {2, 3, 4};
    Eigen::Matrix<Float, 9, 9> Hr;
    Hr.setZero();
#pragma unroll
    for(int j = 0; j < 3; ++j)
#pragma unroll
        for(int k = 0; k <= j; ++k)
        {
            if(j == k)
            {
#pragma unroll
                for(int p = 0; p < 3; ++p)
#pragma unroll
                    for(int q = 0; q <= p; ++q)
                    {
                        Float s = 0.0;
#pragma unroll
                        for(int a = 0; a < nz[j]; ++a)
#pragma unroll
                            for(int b = 0; b < nz[j]; ++b)
                                s += (hwb[j][a] * hwb[j][b]) * H(3 * a + p, 3 * b + q);
                        Hr(3 * j + p, 3 * j + q) = s;
                    }
            }
            else
            {
                Matrix3x3 B = Matrix3x3::Zero();
                for(int a = 0; a < nz[j]; ++a)
                    for(int b = 0; b < nz[k]; ++b)
                        B += (hwb[j][a] * hwb[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
                Hr.template block<3, 3>(3 * j, 3 * k) = B;
            }
        }
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            outHr[((size_t)I) * 81 + r * 9 + c] = Hr(r, c);
}

// ---------------- phase 2 variants ----------------
__device__ double g_opaque = 0.70710678118654752440;  // defeats const-prop in stubs

// V_A: the real QL on Hr
__global__ void ql_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    Eigen::Matrix<Float, 9, 9> M;
#pragma unroll
    for(int r = 0; r < 9; ++r)
#pragma unroll
        for(int c = 0; c < 9; ++c)
            M(r, c) = in[((size_t)I) * 81 + r * 9 + c];
    Eigen::Vector<Float, 9>    ev;
    Eigen::Matrix<Float, 9, 9> V;
    uipc::backend::cuda_tool::eigen::evd_tridiag_ql<Float, 9>(M, ev, V);
    Float acc = 0;
#pragma unroll
    for(int r = 0; r < 9; ++r)
    {
        acc += ev(r);
        out[((size_t)I) * 82 + r] = ev(r);
    }
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 82 + 9 + r * 9 + c] = V(r, c);
    out[((size_t)I) * 82 + 81] = acc;
}

// V_B: the stub -- same input read, same output write pattern, no solve
__global__ void stub_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    Eigen::Matrix<Float, 9, 9> M;
#pragma unroll
    for(int r = 0; r < 9; ++r)
#pragma unroll
        for(int c = 0; c < 9; ++c)
            M(r, c) = in[((size_t)I) * 81 + r * 9 + c];
    Eigen::Vector<Float, 9>    ev;
    Eigen::Matrix<Float, 9, 9> V;
    const Float o = g_opaque;
#pragma unroll
    for(int r = 0; r < 9; ++r)
        ev(r) = M(r, r) * o;
#pragma unroll
    for(int r = 0; r < 9; ++r)
#pragma unroll
        for(int c = 0; c < 9; ++c)
            V(r, c) = (r == c ? M(r, c) : o * M(r, c));
    Float acc = 0;
#pragma unroll
    for(int r = 0; r < 9; ++r)
    {
        acc += ev(r);
        out[((size_t)I) * 82 + r] = ev(r);
    }
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 82 + 9 + r * 9 + c] = V(r, c);
    out[((size_t)I) * 82 + 81] = acc;
}

// ---------------- dynamic census of tql2 ----------------
// A local copy of tred2+tql2 (loop structure verbatim from evd.h) with counters
// at the two data-dependent loops: QL iterations (do-while trips) and V-column
// rotations (the O(N)-work inner step). `neg` counts negative eigenvalues of
// the input restriction -- the fraction of hinges where the projection is
// active work, and the upper bound on a skip-if-PSD fast path's hit rate.
template <typename T, int N>
__global__ void census2_kernel(const double* __restrict__ in,
                               int* __restrict__ iters_out,
                               int* __restrict__ steps_out,
                               int* __restrict__ neg_out,
                               int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    T Vv[N][N];
    T d[N];
    T e[N];
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            Vv[i][j] = in[((size_t)I) * N * N + i * N + j];
    // --- tred2 (verbatim) ---
    for(int j = 0; j < N; ++j)
        d[j] = Vv[N - 1][j];
    for(int i = N - 1; i > 0; --i)
    {
        T scale = 0.0;
        T h = 0.0;
        for(int k = 0; k < i; ++k)
            scale += abs(d[k]);
        if(scale == 0.0)
        {
            e[i] = d[i - 1];
            for(int j = 0; j < i; ++j)
            {
                d[j] = Vv[i - 1][j];
                Vv[i][j] = 0.0;
                Vv[j][i] = 0.0;
            }
        }
        else
        {
            for(int k = 0; k < i; ++k)
            {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            T f = d[i - 1];
            T g = sqrt(h);
            if(f > 0)
                g = -g;
            e[i] = scale * g;
            h = h - f * g;
            d[i - 1] = f - g;
            for(int j = 0; j < i; ++j)
                e[j] = 0.0;
            for(int j = 0; j < i; ++j)
            {
                f = d[j];
                Vv[j][i] = f;
                g = e[j] + Vv[j][j] * f;
                for(int k = j + 1; k <= i - 1; ++k)
                {
                    g += Vv[k][j] * d[k];
                    e[k] += Vv[k][j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for(int j = 0; j < i; ++j)
            {
                e[j] /= h;
                f += e[j] * d[j];
            }
            T hh = f / (h + h);
            for(int j = 0; j < i; ++j)
                e[j] -= hh * d[j];
            for(int j = 0; j < i; ++j)
            {
                f = d[j];
                g = e[j];
                for(int k = j; k <= i - 1; ++k)
                    Vv[k][j] -= (f * e[k] + g * d[k]);
                d[j] = Vv[i - 1][j];
                Vv[i][j] = 0.0;
            }
        }
        d[i] = h;
    }
    for(int i = 0; i < N - 1; ++i)
    {
        Vv[N - 1][i] = Vv[i][i];
        Vv[i][i] = 1.0;
        T h = d[i + 1];
        if(h != 0.0)
        {
            for(int k = 0; k <= i; ++k)
                d[k] = Vv[k][i + 1] / h;
            for(int j = 0; j <= i; ++j)
            {
                T g = 0.0;
                for(int k = 0; k <= i; ++k)
                    g += Vv[k][i + 1] * Vv[k][j];
                for(int k = 0; k <= i; ++k)
                    Vv[k][j] -= g * d[k];
            }
        }
        for(int k = 0; k <= i; ++k)
            Vv[k][i + 1] = 0.0;
    }
    for(int j = 0; j < N; ++j)
    {
        d[j] = Vv[N - 1][j];
        Vv[N - 1][j] = 0.0;
    }
    Vv[N - 1][N - 1] = 1.0;
    e[0] = 0.0;
    // --- tql2 with counters ---
    for(int i = 1; i < N; ++i)
        e[i - 1] = e[i];
    e[N - 1] = 0.0;
    T f = 0.0;
    T tst1 = 0.0;
    constexpr T eps = 2.220446049250313e-16;
    int iters = 0, steps = 0;
    for(int l = 0; l < N; ++l)
    {
        T tl = abs(d[l]) + abs(e[l]);
        if(tl > tst1)
            tst1 = tl;
        int m = l;
        while(m < N)
        {
            if(abs(e[m]) <= eps * tst1)
                break;
            ++m;
        }
        if(m > l)
        {
            int iter = 0;
            do
            {
                ++iter;
                ++iters;
                T g = d[l];
                T p = (d[l + 1] - g) / (2.0 * e[l]);
                T r = hypot(p, T(1.0));
                if(p < 0)
                    r = -r;
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                T dl1 = d[l + 1];
                T h = g - d[l];
                for(int i = l + 2; i < N; ++i)
                    d[i] -= h;
                f += h;
                p = d[m];
                T c = 1.0;
                T c2 = c;
                T c3 = c;
                T el1 = e[l + 1];
                T s = 0.0;
                T s2 = 0.0;
                for(int i = m - 1; i >= l; --i)
                {
                    ++steps;  // one V-column rotation
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for(int k = 0; k < N; ++k)
                    {
                        h = Vv[k][i + 1];
                        Vv[k][i + 1] = s * Vv[k][i] + c * h;
                        Vv[k][i] = c * Vv[k][i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
            } while(abs(e[l]) > eps * tst1 && iter < 40);
        }
        d[l] += f;
        e[l] = 0.0;
    }
    int neg = 0;
    for(int r = 0; r < N; ++r)
        if(d[r] < 0.0)
            ++neg;
    iters_out[I] = iters;
    steps_out[I] = steps;
    neg_out[I] = neg;
}

// ---------------- helper variants on H12 (realistic inputs) ----------------
__global__ void helper_full_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    Matrix12x12 H;
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            H(i, j) = in[((size_t)I) * 144 + i * 12 + j];
    make_spd_translation_free_4x3_blocked<1, 2>(H);
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            out[((size_t)I) * 144 + i * 12 + j] = H(i, j);
}

// stub of make_spd<9,1>: identical surrounding code, the solve replaced by an
// opaque near-identity system with the same output-buffer write pattern
template <typename T, int N>
UIPC_GENERIC void evd_stub(const Eigen::Matrix<T, N, N>& M,
                           Eigen::Vector<T, N>& ev,
                           Eigen::Matrix<T, N, N>& V)
{
    const T o = g_opaque;
#pragma unroll
    for(int i = 0; i < N; ++i)
        ev(i) = M(i, i) * o;
#pragma unroll
    for(int i = 0; i < N; ++i)
#pragma unroll
        for(int j = 0; j < N; ++j)
            V(i, j) = (i == j ? M(i, i) : o * M(i, j));
}

template <int N, int Solver = 0>
UIPC_GENERIC void make_spd_stub(Matrix<Float, N, N>& H)
{
    Vector<Float, N>    eigen_values;
    Matrix<Float, N, N> eigen_vectors;
    evd_stub<Float, N>(H, eigen_values, eigen_vectors);
    for(int i = 0; i < N; ++i)
    {
        auto& v = eigen_values(i);
        v = v < 0.0 ? 0.0 : v;
    }
#pragma unroll
    for(int i = 0; i < N; ++i)
#pragma unroll
        for(int j = i; j < N; ++j)
        {
            Float sum = 0.0;
#pragma unroll
            for(int k = 0; k < N; ++k)
                sum += eigen_vectors(i, k) * eigen_values(k) * eigen_vectors(j, k);
            H(i, j) = sum;
            H(j, i) = sum;
        }
}

__global__ void helper_stub_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    Matrix12x12 H;
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            H(i, j) = in[((size_t)I) * 144 + i * 12 + j];
    // the shipped assembly, stubbed solve, shipped back-assembly
    constexpr Float r2 = 0.70710678118654752440;
    constexpr Float r6 = 0.40824829046386301637;
    constexpr Float r12 = 0.28867513459481288225;
    constexpr Float h[3][4] = {{r2, -r2, 0.0, 0.0}, {r6, r6, -2.0 * r6, 0.0}, {r12, r12, r12, -3.0 * r12}};
    constexpr int nz[3] = {2, 3, 4};
    Eigen::Matrix<Float, 9, 9> Hr;
    Hr.setZero();
#pragma unroll
    for(int j = 0; j < 3; ++j)
#pragma unroll
        for(int k = 0; k <= j; ++k)
        {
            if(j == k)
            {
#pragma unroll
                for(int p = 0; p < 3; ++p)
#pragma unroll
                    for(int q = 0; q <= p; ++q)
                    {
                        Float s = 0.0;
#pragma unroll
                        for(int a = 0; a < nz[j]; ++a)
#pragma unroll
                            for(int b = 0; b < nz[j]; ++b)
                                s += (h[j][a] * h[j][b]) * H(3 * a + p, 3 * b + q);
                        Hr(3 * j + p, 3 * j + q) = s;
                    }
                continue;
            }
            Matrix3x3 B = Matrix3x3::Zero();
            for(int a = 0; a < nz[j]; ++a)
                for(int b = 0; b < nz[k]; ++b)
                    B += (h[j][a] * h[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
            Hr.template block<3, 3>(3 * j, 3 * k) = B;
        }
    make_spd_stub<9>(Hr);
#pragma unroll
    for(int a = 0; a < 4; ++a)
#pragma unroll
        for(int b = a; b < 4; ++b)
        {
            Matrix3x3 B = Matrix3x3::Zero();
            for(int j = 0; j < 3; ++j)
            {
                if(a >= nz[j])
                    continue;
                for(int k = 0; k < 3; ++k)
                {
                    if(b >= nz[k])
                        continue;
                    B += (h[j][a] * h[k][b]) * Hr.template block<3, 3>(3 * j, 3 * k);
                }
            }
            H.template block<3, 3>(3 * a, 3 * b) = B;
            if(a != b)
                H.template block<3, 3>(3 * b, 3 * a) = B.transpose();
        }
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            out[((size_t)I) * 144 + i * 12 + j] = H(i, j);
}

__global__ void helper_asm_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    Matrix12x12 H;
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            H(i, j) = in[((size_t)I) * 144 + i * 12 + j];
    constexpr Float r2 = 0.70710678118654752440;
    constexpr Float r6 = 0.40824829046386301637;
    constexpr Float r12 = 0.28867513459481288225;
    constexpr Float h[3][4] = {{r2, -r2, 0.0, 0.0}, {r6, r6, -2.0 * r6, 0.0}, {r12, r12, r12, -3.0 * r12}};
    constexpr int nz[3] = {2, 3, 4};
    Eigen::Matrix<Float, 9, 9> Hr;
    Hr.setZero();
#pragma unroll
    for(int j = 0; j < 3; ++j)
#pragma unroll
        for(int k = 0; k <= j; ++k)
        {
            if(j == k)
            {
#pragma unroll
                for(int p = 0; p < 3; ++p)
#pragma unroll
                    for(int q = 0; q <= p; ++q)
                    {
                        Float s = 0.0;
#pragma unroll
                        for(int a = 0; a < nz[j]; ++a)
#pragma unroll
                            for(int b = 0; b < nz[j]; ++b)
                                s += (h[j][a] * h[j][b]) * H(3 * a + p, 3 * b + q);
                        Hr(3 * j + p, 3 * j + q) = s;
                    }
                continue;
            }
            Matrix3x3 B = Matrix3x3::Zero();
            for(int a = 0; a < nz[j]; ++a)
                for(int b = 0; b < nz[k]; ++b)
                    B += (h[j][a] * h[k][b]) * H.template block<3, 3>(3 * a, 3 * b);
            Hr.template block<3, 3>(3 * j, 3 * k) = B;
        }
#pragma unroll
    for(int a = 0; a < 4; ++a)
#pragma unroll
        for(int b = a; b < 4; ++b)
        {
            Matrix3x3 B = Matrix3x3::Zero();
            for(int j = 0; j < 3; ++j)
            {
                if(a >= nz[j])
                    continue;
                for(int k = 0; k < 3; ++k)
                {
                    if(b >= nz[k])
                        continue;
                    B += (h[j][a] * h[k][b]) * Hr.template block<3, 3>(3 * j, 3 * k);
                }
            }
            H.template block<3, 3>(3 * a, 3 * b) = B;
            if(a != b)
                H.template block<3, 3>(3 * b, 3 * a) = B.transpose();
        }
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            out[((size_t)I) * 144 + i * 12 + j] = H(i, j);
}

// ---------------- timing harness ----------------
template <typename Launch>
static void timeit(const double* d_in, double* d_out, int n, int reps, const char* tag, Launch launch)
{
    launch(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int r = 0; r < reps; ++r)
        launch(d_in, d_out, n);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    printf("%-28s: %8.2f us/launch of %d mats = %7.2f ns/mat\n", tag, ms * 1000.f / reps, n, ms * 1e6f / reps / n);
}

static std::vector<HingeInput> gen_inputs(int n, unsigned seed, int model)
{
    std::mt19937_64              rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<HingeInput>      h(n);
    for(int i = 0; i < n; ++i)
    {
        double L0 = 0.014 + U(rng) * 0.036;
        double th = U(rng) * 2 * M_PI, ph = std::acos(2 * U(rng) - 1);
        Eigen::Vector3d dir(std::sin(ph) * std::cos(th), std::sin(ph) * std::sin(th), std::cos(ph));
        Eigen::Vector3d any(0, 0, 1);
        if(std::abs(dir.z()) > 0.9)
            any = Eigen::Vector3d(1, 0, 0);
        Eigen::Vector3d u = dir.cross(any).normalized();
        Eigen::Vector3d mid_edge = 0.5 * L0 * dir;
        double h0 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
        double h3 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
        double lat0 = (U(rng) - 0.5) * 0.02, lat3 = (U(rng) - 0.5) * 0.02;
        Eigen::Vector3d x1 = mid_edge, x2 = -mid_edge;
        Eigen::Vector3d x0 = lat0 * dir + h0 * u;
        Eigen::Vector3d x3 = lat3 * dir - h3 * u;
        for(int k = 0; k < 3; ++k)
        {
            h[i].x[0][k] = x0(k);
            h[i].x[1][k] = x1(k);
            h[i].x[2][k] = x2(k);
            h[i].x[3][k] = x3(k);
        }
        h[i].kappa = std::pow(10.0, -6.0 + 6.0 * U(rng));
        h[i].L0 = L0;
        h[i].h_bar = 0.0008 + U(rng) * 0.003;
        h[i].theta_bar = U(rng) < 0.5 ? 0.0 : (U(rng) - 0.5) * 3.0;
        double theta_y = std::pow(10.0, -3.0 + 2.8 * U(rng));
        double slope = 2.0 * h[i].kappa * L0 / h[i].h_bar;
        h[i].yield_p = model == 0 ? -1.0 : theta_y * slope;  // negative tags strain
        h[i].Vdt2 = std::pow(10.0, -9.0 + 3.0 * U(rng));
    }
    return h;
}

int main(int argc, char** argv)
{
    const int n = argc > 1 ? atoi(argv[1]) : 65536;
    const int reps = argc > 2 ? atoi(argv[2]) : 100;

    for(int model = 0; model < 2; ++model)
    {
        printf("=== model %s ===\n", model == 0 ? "strain-plastic" : "stress-plastic");
        auto h = gen_inputs(n, 7 + model, model);
        HingeInput* d_in;
        double *d_H, *d_Hr, *d_out;
        cudaMalloc(&d_in, n * sizeof(HingeInput));
        cudaMalloc(&d_H, (size_t)n * 144 * 8);
        cudaMalloc(&d_Hr, (size_t)n * 81 * 8);
        cudaMalloc(&d_out, (size_t)n * 144 * 8);
        cudaMemcpy(d_in, h.data(), n * sizeof(HingeInput), cudaMemcpyHostToDevice);
        restrict_kernel<<<(n + 255) / 256, 256>>>(d_in, d_H, d_Hr, n);
        cudaDeviceSynchronize();

        // ABBABA interleave of the three helper variants
        for(int rep = 0; rep < 3; ++rep)
        {
            timeit(d_H, d_out, n, reps, "V0 full helper <1,2>", [&](const double* i, double* o, int m) {
                helper_full_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
            timeit(d_H, d_out, n, reps, "V1 stubbed solve", [&](const double* i, double* o, int m) {
                helper_stub_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
            timeit(d_H, d_out, n, reps, "V2 assembly only", [&](const double* i, double* o, int m) {
                helper_asm_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
        }
        double* d_qlo;
        cudaMalloc(&d_qlo, (size_t)n * 82 * 8);
        for(int rep = 0; rep < 3; ++rep)
        {
            timeit(d_Hr, d_qlo, n, reps, "A QL alone (9x9)", [&](const double* i, double* o, int m) {
                ql_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
            timeit(d_Hr, d_qlo, n, reps, "B stub alone", [&](const double* i, double* o, int m) {
                stub_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
        }
        // census
        int *d_it, *d_st, *d_neg;
        cudaMalloc(&d_it, n * 4);
        cudaMalloc(&d_st, n * 4);
        cudaMalloc(&d_neg, n * 4);
        census2_kernel<double, 9><<<(n + 255) / 256, 256>>>(d_Hr, d_it, d_st, d_neg, n);
        cudaDeviceSynchronize();
        std::vector<int> it(n), st(n), neg(n);
        cudaMemcpy(it.data(), d_it, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(st.data(), d_st, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(neg.data(), d_neg, n * 4, cudaMemcpyDeviceToHost);
        double it_mean = 0, st_mean = 0;
        long neg_mats = 0;
        int st_max = 0;
        for(int i = 0; i < n; ++i)
        {
            it_mean += it[i];
            st_mean += st[i];
            st_max = std::max(st_max, st[i]);
            neg_mats += neg[i] > 0;
        }
        // per-warp divergence: warp = 32 consecutive indices (same block dim)
        double warp_waste_num = 0;
        long nwarps = 0;
        for(int w = 0; w + 32 <= n; w += 32)
        {
            int mx = 0;
            double mn = 0;
            for(int k = 0; k < 32; ++k)
            {
                mx = std::max(mx, st[w + k]);
                mn += st[w + k];
            }
            warp_waste_num += mx;
            nwarps++;
        }
        printf("census: QL iters/mat %.2f, rotation steps/mat %.2f (max %d), warp max/mean %.3f, indefinite mats %.2f %%\n",
               it_mean / n, st_mean / n, st_max, warp_waste_num / st_mean, 100.0 * neg_mats / n);
        cudaFree(d_in);
        cudaFree(d_H);
        cudaFree(d_Hr);
        cudaFree(d_out);
        cudaFree(d_qlo);
        cudaFree(d_it);
        cudaFree(d_st);
        cudaFree(d_neg);
    }
    return 0;
}

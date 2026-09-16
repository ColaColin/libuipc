// Round-7 s16 part 3: the two semantics-preserving QL levers, measured on the
// realistic restricted matrices (same generator/feed as ql_probe/mc_probe):
//   T = evd_tridiag_ql with V stored TRANSPOSED -- the tql2 rotation loop and
//       tred2's accumulation access V column-wise (strided local memory);
//       transposing makes those rows contiguous (LDL.128/STL.128 pairing).
//       Same arithmetic, same order -> must be BIT-IDENTICAL (verified here
//       word-for-word on device).
//   B = division reduction -- reciprocal-multiply for the rotation's two
//       divisions, sqrt(p*p+1) for the shift hypot. Rounding-level (<= ulp per
//       use), the s01 solver-swap class.
//   TB = both.
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Eigenvalues>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;
using uipc::Float;

// ---- local copies of evd_tridiag_ql (structure verbatim from evd.h) ----
// Layout: 0 = as shipped, 1 = V stored transposed (Vt[a][b] = V[b][a])
// DivRed: 0 = as shipped, 1 = reciprocal-mult + sqrt(p^2+1) shift
template <typename T, int N, int Layout, int DivRed>
UIPC_GENERIC void evd_ql_var(const Eigen::Matrix<T, N, N>& M,
                             Eigen::Vector<T, N>&          eigen_values,
                             Eigen::Matrix<T, N, N>&       eigen_vectors)
{
    static_assert(N >= 2);
    T V[N][N];
    T d[N];
    T e[N];
#pragma unroll
    for(int i = 0; i < N; ++i)
#pragma unroll
        for(int j = 0; j < N; ++j)
            V[i][j] = M(i, j);

    // --- tred2 ---
#pragma unroll
    for(int j = 0; j < N; ++j)
        d[j] = V[N - 1][j];
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
                d[j] = V[i - 1][j];
                V[i][j] = 0.0;
                V[j][i] = 0.0;
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
                V[j][i] = f;
                g = e[j] + V[j][j] * f;
                for(int k = j + 1; k <= i - 1; ++k)
                {
                    g += V[k][j] * d[k];
                    e[k] += V[k][j] * f;
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
                    V[k][j] -= (f * e[k] + g * d[k]);
                d[j] = V[i - 1][j];
                V[i][j] = 0.0;
            }
        }
        d[i] = h;
    }
    for(int i = 0; i < N - 1; ++i)
    {
        V[N - 1][i] = V[i][i];
        V[i][i] = 1.0;
        T h = d[i + 1];
        if(h != 0.0)
        {
            for(int k = 0; k <= i; ++k)
                d[k] = V[k][i + 1] / h;
            for(int j = 0; j <= i; ++j)
            {
                T g = 0.0;
                for(int k = 0; k <= i; ++k)
                    g += V[k][i + 1] * V[k][j];
                for(int k = 0; k <= i; ++k)
                    V[k][j] -= g * d[k];
            }
        }
        for(int k = 0; k <= i; ++k)
            V[k][i + 1] = 0.0;
    }
#pragma unroll
    for(int j = 0; j < N; ++j)
    {
        d[j] = V[N - 1][j];
        V[N - 1][j] = 0.0;
    }
    V[N - 1][N - 1] = 1.0;
    e[0] = 0.0;

    // --- tql2 ---
    for(int i = 1; i < N; ++i)
        e[i - 1] = e[i];
    e[N - 1] = 0.0;
    T f = 0.0;
    T tst1 = 0.0;
    constexpr T eps = 2.220446049250313e-16;
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
                T g = d[l];
                T p = (d[l + 1] - g) / (2.0 * e[l]);
                T r;
                if(DivRed)
                    r = sqrt(p * p + T(1.0));
                else
                    r = hypot(p, T(1.0));
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
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot(p, e[i]);
                    e[i + 1] = s * r;
                    if(DivRed)
                    {
                        T ri = T(1.0) / r;
                        s = e[i] * ri;
                        c = p * ri;
                    }
                    else
                    {
                        s = e[i] / r;
                        c = p / r;
                    }
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    if(Layout == 0)
                    {
#pragma unroll
                        for(int k = 0; k < N; ++k)
                        {
                            h = V[k][i + 1];
                            V[k][i + 1] = s * V[k][i] + c * h;
                            V[k][i] = c * V[k][i] - s * h;
                        }
                    }
                    else
                    {
                        // Vt[a][b] = V[b][a]: the two touched columns of V are
                        // two contiguous rows of Vt
#pragma unroll
                        for(int k = 0; k < N; ++k)
                        {
                            h = V[i + 1][k];
                            V[i + 1][k] = s * V[i][k] + c * h;
                            V[i][k] = c * V[i][k] - s * h;
                        }
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

#pragma unroll
    for(int i = 0; i < N; ++i)
        eigen_values(i) = d[i];
    if(Layout == 0)
    {
#pragma unroll
        for(int i = 0; i < N; ++i)
#pragma unroll
            for(int j = 0; j < N; ++j)
                eigen_vectors(i, j) = V[i][j];
    }
    else
    {
#pragma unroll
        for(int i = 0; i < N; ++i)
#pragma unroll
            for(int j = 0; j < N; ++j)
                eigen_vectors(i, j) = V[j][i];
    }
}

__device__ double g_dummy = 0.0;

template <int Layout, int DivRed>
__global__ void var_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
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
    evd_ql_var<Float, 9, Layout, DivRed>(M, ev, V);
    for(int r = 0; r < 9; ++r)
        out[((size_t)I) * 82 + r] = ev(r);
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 82 + 9 + r * 9 + c] = V(r, c);
}

// the shipped original, for reference timing and bit-comparison
__global__ void orig_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
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
    for(int r = 0; r < 9; ++r)
        out[((size_t)I) * 82 + r] = ev(r);
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 82 + 9 + r * 9 + c] = V(r, c);
}

// same generators as the other probes (regenerates the identical matrices)
struct HingeInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar, yield_p, Vdt2;
};
#include "mc_gen.inc"
#include <finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h>

__global__ void restrict_kernel(const HingeInput* __restrict__ in, double* __restrict__ outHr, int n)
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
    if(h.yield_p < 0.0)
        sym::strain_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa);
    else
        sym::stress_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa, h.yield_p);
    H *= h.Vdt2;
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
    printf("%-24s: %8.2f us = %7.2f ns/mat\n", tag, ms * 1000.f / reps, ms * 1e6f / reps / n);
}

static long count_diff(const double* a, const double* b, size_t words)
{
    long d = 0;
    for(size_t i = 0; i < words; ++i)
    {
        uint64_t xa, xb;
        memcpy(&xa, a + i, 8);
        memcpy(&xb, b + i, 8);
        if(xa != xb)
            ++d;
    }
    return d;
}
static long count_diff_ev(const double* a, const double* b, size_t words)
{
    long d = 0;  // eigenvalue slots only: word (i*82 + r), r < 9
    for(size_t i = 0; i * 82 + 9 <= words; ++i)
        for(int r = 0; r < 9; ++r)
        {
            uint64_t xa, xb;
            memcpy(&xa, a + i * 82 + r, 8);
            memcpy(&xb, b + i * 82 + r, 8);
            if(xa != xb)
                ++d;
        }
    return d;
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
        double *d_Hr, *d_o, *d_a, *d_b, *d_c;
        cudaMalloc(&d_in, n * sizeof(HingeInput));
        cudaMalloc(&d_Hr, (size_t)n * 81 * 8);
        cudaMalloc(&d_o, (size_t)n * 82 * 8);
        cudaMalloc(&d_a, (size_t)n * 82 * 8);
        cudaMalloc(&d_b, (size_t)n * 82 * 8);
        cudaMalloc(&d_c, (size_t)n * 82 * 8);
        cudaMemcpy(d_in, h.data(), n * sizeof(HingeInput), cudaMemcpyHostToDevice);
        restrict_kernel<<<(n + 255) / 256, 256>>>(d_in, d_Hr, n);
        cudaDeviceSynchronize();

        const int ns = std::min(n, 100000);
        // numerics: bit-identity of T; rounding class of B / TB
        orig_kernel<<<(ns + 255) / 256, 256>>>(d_Hr, d_o, ns);
        var_kernel<1, 0><<<(ns + 255) / 256, 256>>>(d_Hr, d_a, ns);
        var_kernel<0, 1><<<(ns + 255) / 256, 256>>>(d_Hr, d_b, ns);
        var_kernel<1, 1><<<(ns + 255) / 256, 256>>>(d_Hr, d_c, ns);
        cudaDeviceSynchronize();
        std::vector<double> o((size_t)ns * 82), a((size_t)ns * 82), b((size_t)ns * 82), c((size_t)ns * 82);
        cudaMemcpy(o.data(), d_o, o.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(a.data(), d_a, a.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, b.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(c.data(), d_c, c.size() * 8, cudaMemcpyDeviceToHost);
        size_t words = o.size();
        printf("bit-diff vs shipped: T %ld / %zu words (eigenvalue slots: %ld), B %ld (ev: %ld), TB %ld (ev: %ld)\n",
               count_diff(o.data(), a.data(), words), words,
               count_diff_ev(o.data(), a.data(), words),
               count_diff(o.data(), b.data(), words),
               count_diff_ev(o.data(), b.data(), words),
               count_diff(o.data(), c.data(), words),
               count_diff_ev(o.data(), c.data(), words));
        // relFro of the projected matrices for B/TB (the s01-class check)
        auto relfro = [&](const std::vector<double>& x) {
            std::vector<double> v;
            for(int i = 0; i < ns; ++i)
            {
                double fn = 0, dn = 0;
                for(int r = 0; r < 9; ++r)
                    for(int cc = 0; cc < 9; ++cc)
                    {
                        double A = o[(size_t)i * 82 + 9 + r * 9 + cc];
                        double B = x[(size_t)i * 82 + 9 + r * 9 + cc];
                        fn += (A - B) * (A - B);
                        dn += A * A;
                    }
                if(dn > 0)
                    v.push_back(sqrt(fn / dn));
            }
            std::sort(v.begin(), v.end());
            return std::pair<double, double>(v[v.size() / 2], v.back());
        };
        auto rb = relfro(b), rc = relfro(c);
        printf("relFro(V): B med %.2e max %.2e | TB med %.2e max %.2e\n", rb.first, rb.second, rc.first, rc.second);

        for(int rep = 0; rep < 3; ++rep)
        {
            timeit(d_Hr, d_o, n, reps, "copy <0,0> baseline", [&](const double* i, double* oo, int m) {
                var_kernel<0, 0><<<(m + 255) / 256, 256>>>(i, oo, m);
            });
            timeit(d_Hr, d_o, n, reps, "orig (shipped)", [&](const double* i, double* oo, int m) {
                orig_kernel<<<(m + 255) / 256, 256>>>(i, oo, m);
            });
            timeit(d_Hr, d_o, n, reps, "T transposed V", [&](const double* i, double* oo, int m) {
                var_kernel<1, 0><<<(m + 255) / 256, 256>>>(i, oo, m);
            });
            timeit(d_Hr, d_o, n, reps, "B div-reduced", [&](const double* i, double* oo, int m) {
                var_kernel<0, 1><<<(m + 255) / 256, 256>>>(i, oo, m);
            });
            timeit(d_Hr, d_o, n, reps, "TB both", [&](const double* i, double* oo, int m) {
                var_kernel<1, 1><<<(m + 255) / 256, 256>>>(i, oo, m);
            });
        }
        cudaFree(d_in);
        cudaFree(d_Hr);
        cudaFree(d_o);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    return 0;
}

// Round-7 s16 part 2: price the modified-Cholesky (LDL^T) PSD projection
// against the shipped QL eigen-clip on the same realistic restricted matrices.
//   - census: negative-eigenvalue COUNT distribution + min-eig/||Hr||_F
//   - MC prototype (Nocedal-Wright unpivoted modified LDL^T, delta sweep):
//     relFro(MC, clip) vs relFro(clip, raw) vs the old path's own solver-swap
//     sensitivity; min-eig(MC result); ||E|| vs |lambda_min|.
//   - timing: MC alone vs QL alone (A/B interleave) on identical inputs.
#include <utils/make_spd.h>
#include <finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h>
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Eigenvalues>
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;

// ---------------- inputs: read the Hr matrices dumped by ql_probe ----------
// (regenerated here to keep this probe self-contained)
struct HingeInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar, yield_p, Vdt2;
};
#include "mc_gen.inc"

__global__ void restrict_kernel(const HingeInput* __restrict__ in,
                                double* __restrict__ outH,
                                double* __restrict__ outHr,
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
    if(h.yield_p < 0.0)
        sym::strain_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa);
    else
        sym::stress_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa, h.yield_p);
    H *= h.Vdt2;
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            outH[((size_t)I) * 144 + r * 12 + c] = H(r, c);
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

// ---------------- the MC projection prototype ----------------
// Modified LDL^T with symmetric diagonal pivoting (max-|c_jj| selection) and a
// delta floor: whenever a (permuted) pivot would drop below delta it is held
// at delta, so D >= delta > 0 and A' = L D L^T is strictly PD. With
// max-diagonal symmetric pivoting the negative pivot magnitudes are bounded by
// |lambda_min| through Cauchy interlacing (lambda_min(A) <= c_jj), so the
// implicit modification E stays |lambda_min|-scale rather than blowing up (the
// unpivoted N&W variant measured min-eig -1e176 on these inputs -- pivoting is
// not optional). Rebuild A' = L D L^T in the ORIGINAL ordering via the
// permutation. Reads only the lower triangle (the s08 solver contract).
template <typename T, int N>
UIPC_GENERIC void psd_ldlt_modified(Eigen::Matrix<T, N, N>& A, T delta_rel)
{
    T W[N][N];  // working copy, FULL symmetric (the pivot interchange permutes
                // rows+cols and must see mirrored entries, not zeros); the
                // lower triangle becomes L, D on the diagonal
    T dmax = 0.0;
#pragma unroll
    for(int i = 0; i < N; ++i)
    {
        dmax = fmax(dmax, fabs(A(i, i)));
#pragma unroll
        for(int j = 0; j < N; ++j)
            W[i][j] = (i >= j) ? A(i, j) : A(j, i);
    }
    T delta = delta_rel * dmax;
    T D[N];
    int p[N];
#pragma unroll
    for(int i = 0; i < N; ++i)
        p[i] = i;
#pragma unroll
    for(int k = 0; k < N; ++k)
    {
        // pivot: r = argmax_{i>=k} |W[i][i]| (symmetric interchange)
        int r = k;
        T   best = fabs(W[k][k]);
#pragma unroll
        for(int i = k + 1; i < N; ++i)
        {
            T v = fabs(W[i][i]);
            if(v > best)
            {
                best = v;
                r = i;
            }
        }
        if(r != k)
        {
            for(int j = 0; j < N; ++j)
            {
                T t = W[k][j];
                W[k][j] = W[r][j];
                W[r][j] = t;
            }
            for(int i = 0; i < N; ++i)
            {
                T t = W[i][k];
                W[i][k] = W[i][r];
                W[i][r] = t;
            }
            int t = p[k];
            p[k] = p[r];
            p[r] = t;
        }
        T c = W[k][k];
        // column-scale-aware floor: a pivot small next to its own column
        // (theta_k = max_{i>k} |W[i][k]|) must not be divided into L -- floor
        // at theta_k^2/dmax (dimension of A). Without it the tail explodes
        // (measured max relFro(MC,clip) 24 / 1e5).
        T theta = 0.0;
#pragma unroll
        for(int q = k + 1; q < N; ++q)
            theta = fmax(theta, fabs(W[q][k]));
        T adelta = fmax(delta, (theta * theta) / dmax);
        // reflect strongly negative pivots (E_kk = 2|c| <= 2|lambda_min| by
        // interlacing; keeps L = W/D moderate), floor small-magnitude pivots.
        D[k] = (c >= adelta) ? c : ((c <= -adelta) ? -c : adelta);
        W[k][k] = 1.0;  // L's unit diagonal (the rebuild reads W[q][q] as 1)
        if(k + 1 < N)
        {
#pragma unroll
            for(int i = k + 1; i < N; ++i)
                W[i][k] = W[i][k] / D[k];
            // trailing update, FULL symmetric block (the pivot interchange
            // permutes rows+cols and must never mix stale upper entries):
            // W[i][j] -= L_ik L_jk D_k for all i, j > k
#pragma unroll
            for(int i = k + 1; i < N; ++i)
#pragma unroll
                for(int j = k + 1; j < N; ++j)
                {
                    T t = W[i][k] * W[j][k];
                    W[i][j] -= t * D[k];
                }
        }
    }
    // rebuild A' = L D L^T mapped back through p: A'(p_i, p_j), j <= i, mirror
#pragma unroll
    for(int i = 0; i < N; ++i)
#pragma unroll
        for(int j = 0; j <= i; ++j)
        {
            T s = 0.0;
#pragma unroll
            for(int q = 0; q <= j; ++q)
                s += W[i][q] * D[q] * W[j][q];
            A(p[i], p[j]) = s;
            A(p[j], p[i]) = s;
        }
}

__device__ double g_delta_rel = 1e-12;  // host-tunable

__global__ void mc_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
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
    psd_ldlt_modified<Float, 9>(M, g_delta_rel);
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 81 + r * 9 + c] = M(r, c);
}

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
    for(int r = 0; r < 9; ++r)
        out[((size_t)I) * 81 + r] = ev(r);
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 81 + 9 + r * 9 + c] = V(r, c);
}

// ---------------- the full shipped helper (clip through QL) for relFro ref --
__global__ void clip_kernel(const double* __restrict__ in, double* __restrict__ out, int n)
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
    make_spd<9, 1>(M);
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 81 + r * 9 + c] = M(r, c);
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
    printf("%-28s: %8.2f us/launch of %d mats = %7.2f ns/mat\n", tag, ms * 1000.f / reps, n, ms * 1e6f / reps / n);
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
        cudaMalloc(&d_out, (size_t)n * 81 * 8);
        cudaMemcpy(d_in, h.data(), n * sizeof(HingeInput), cudaMemcpyHostToDevice);
        restrict_kernel<<<(n + 255) / 256, 256>>>(d_in, d_H, d_Hr, n);
        cudaDeviceSynchronize();

        const int ns = std::min(n, 20000);  // numerics sample
        std::vector<double> hr((size_t)ns * 81), clip((size_t)ns * 81), mc((size_t)ns * 81);
        cudaMemcpy(hr.data(), d_Hr, hr.size() * 8, cudaMemcpyDeviceToHost);
        clip_kernel<<<(ns + 255) / 256, 256>>>(d_Hr, d_out, ns);
        cudaDeviceSynchronize();
        cudaMemcpy(clip.data(), d_out, clip.size() * 8, cudaMemcpyDeviceToHost);

        for(double dr : {1e-14, 1e-12, 1e-10, 1e-8, 1e-6})
        {
            cudaMemcpyToSymbol(g_delta_rel, &dr, 8);
            mc_kernel<<<(ns + 255) / 256, 256>>>(d_Hr, d_out, ns);
            cudaDeviceSynchronize();
            cudaMemcpy(mc.data(), d_out, mc.size() * 8, cudaMemcpyDeviceToHost);
            // host comparison
            double rel_mc_clip_max = 0, rel_mc_clip_med = 0, rel_clip_raw_max = 0, rel_clip_raw_med = 0;
            double min_eig_mc_min = 1e300, min_eig_clip_min = 1e300, eig_scale_med = 0;
            std::vector<double> a1, a2;
            long n_pos_after = 0;
            for(int i = 0; i < ns; ++i)
            {
                Eigen::Matrix<double, 9, 9> R, C, M;
                for(int r = 0; r < 9; ++r)
                    for(int c = 0; c < 9; ++c)
                    {
                        // the dumped Hr's upper triangle is ZERO (the shipped
                        // assembly only fills the lower) -- symmetrize or every
                        // raw-relative norm is an artifact
                        R(r, c) = r >= c ? hr[(size_t)i * 81 + r * 9 + c] : hr[(size_t)i * 81 + c * 9 + r];
                        C(r, c) = clip[(size_t)i * 81 + r * 9 + c];
                        M(r, c) = mc[(size_t)i * 81 + r * 9 + c];
                    }
                double fr = R.norm();
                if(fr < 1e-300)
                    continue;
                a1.push_back((M - C).norm() / fr);
                a2.push_back((C - R).norm() / fr);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(M), esc(C);
                min_eig_mc_min = std::min(min_eig_mc_min, es.eigenvalues().minCoeff() / fr);
                min_eig_clip_min = std::min(min_eig_clip_min, esc.eigenvalues().minCoeff() / fr);
            }
            std::sort(a1.begin(), a1.end());
            std::sort(a2.begin(), a2.end());
            long n_big = std::count_if(a1.begin(), a1.end(), [](double v) { return v > 0.1; });
            long n_mid = std::count_if(a1.begin(), a1.end(), [](double v) { return v > 1e-3; });
            printf("delta_rel=%.0e : relFro(MC,clip) med %.3e p99 %.3e p999 %.3e max %.3e (>1e-3: %.2f%%, >0.1: %.2f%%) | relFro(clip,raw) med %.3e max %.3e | min-eig MC %.2e clip %.2e\n",
                   dr,
                   a1[a1.size() / 2],
                   a1[(size_t)(0.99 * (a1.size() - 1))],
                   a1[(size_t)(0.999 * (a1.size() - 1))],
                   a1.back(),
                   100.0 * n_mid / a1.size(),
                   100.0 * n_big / a1.size(),
                   a2[a2.size() / 2],
                   a2.back(),
                   min_eig_mc_min,
                   min_eig_clip_min);
        }

        // negative-eigenvalue count distribution + min-eig/||.|| on raw
        {
            double mineig_rel_min = 1e300, mineig_rel_med = 0;
            std::vector<double> me;
            std::vector<int>    neghist(10, 0);
            for(int i = 0; i < ns; ++i)
            {
                Eigen::Matrix<double, 9, 9> R;
                for(int r = 0; r < 9; ++r)
                    for(int c = 0; c < 9; ++c)
                        R(r, c) = r >= c ? hr[(size_t)i * 81 + r * 9 + c] : hr[(size_t)i * 81 + c * 9 + r];
                double fr = R.norm();
                if(fr < 1e-300)
                    continue;
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(R);
                auto ev = es.eigenvalues();
                int neg = 0;
                for(int k = 0; k < 9; ++k)
                    neg += ev(k) < 0;
                neghist[neg]++;
                me.push_back(ev.minCoeff() / fr);
            }
            std::sort(me.begin(), me.end());
            printf("neg-eig count hist [0..9]: ");
            for(int k = 0; k < 10; ++k)
                printf("%d ", neghist[k]);
            printf("\nmin-eig/||Hr||F: med %.3e p5 %.3e min %.3e\n", me[me.size() / 2], me[me.size() / 20], me.front());
        }

        // timing interleave
        for(int rep = 0; rep < 3; ++rep)
        {
            double dr = 1e-12;
            cudaMemcpyToSymbol(g_delta_rel, &dr, 8);
            timeit(d_Hr, d_out, n, reps, "MC alone (9x9)", [&](const double* i, double* o, int m) {
                mc_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
            timeit(d_Hr, d_out, n, reps, "QL alone (9x9)", [&](const double* i, double* o, int m) {
                ql_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
            timeit(d_Hr, d_out, n, reps, "clip alone (make_spd)", [&](const double* i, double* o, int m) {
                clip_kernel<<<(m + 255) / 256, 256>>>(i, o, m);
            });
        }
        cudaFree(d_in);
        cudaFree(d_H);
        cudaFree(d_Hr);
        cudaFree(d_out);
    }
    return 0;
}

// s08 pre-build probe: which triangle of the input matrix do the two
// eigen-solvers actually read?
//
//   evd<Float, N>            (Eigen SelfAdjointEigenSolver::compute, Solver=0)
//   evd_tridiag_ql<Float, N> (fixed-size tred2+tql2, Solver=1)
//
// Arm A solves M as-is. Arm B solves M with the UPPER triangle overwritten by
// NaN. Arm C (positive control) poisons the LOWER triangle. Outputs compared
// word-by-word: 0 mismatching words in B = the upper triangle is dead input
// (any arithmetic use propagates NaN; a comparison-only use flips the branch
// and changes iteration counts / output bits). The positive control must
// mismatch massively, or the probe itself is broken.
//
// N in {6, 9} -- the two sizes the translation-free blocked projections feed.
#include <cuda_tool/eigen/evd.h>
#include <random>
#include <vector>
#include <cstdio>
#include <cstring>

using namespace uipc;
using uipc::backend::cuda_tool::eigen::evd;
using uipc::backend::cuda_tool::eigen::evd_tridiag_ql;

template <int N, int Solver, int Poison>  // Poison: 0 none, 1 upper, 2 lower
__global__ void solve_kernel(const double* in, double* vals, double* vecs, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    Eigen::Matrix<double, N, N> M;
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            M(i, j) = in[((size_t)I) * N * N + i * N + j];
    if constexpr(Poison == 1)
    {
        for(int i = 0; i < N; ++i)
            for(int j = i + 1; j < N; ++j)
                M(i, j) = std::numeric_limits<double>::quiet_NaN();
    }
    else if constexpr(Poison == 2)
    {
        for(int i = 0; i < N; ++i)
            for(int j = 0; j < i; ++j)
                M(i, j) = std::numeric_limits<double>::quiet_NaN();
    }
    Eigen::Vector<double, N>    w;
    Eigen::Matrix<double, N, N> V;
    if constexpr(Solver == 0)
        evd<double, N>(M, w, V);
    else
        evd_tridiag_ql<double, N>(M, w, V);
    for(int i = 0; i < N; ++i)
        vals[((size_t)I) * N + i] = w(i);
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            vecs[((size_t)I) * N * N + i * N + j] = V(i, j);
}

// returns mismatching word counts (values, vectors) between poison arm P and 0
template <int N, int Solver, int Poison>
void run_arm(const double* d_in, double* d_v0, double* d_x0, double* d_vP, double* d_xP, int n, long& bad_v, long& bad_x)
{
    std::vector<double> v0((size_t)n * N), x0((size_t)n * N * N);
    std::vector<double> vP((size_t)n * N), xP((size_t)n * N * N);
    solve_kernel<N, Solver, 0><<<64, 128>>>(d_in, d_v0, d_x0, n);
    solve_kernel<N, Solver, Poison><<<64, 128>>>(d_in, d_vP, d_xP, n);
    cudaMemcpy(v0.data(), d_v0, v0.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(x0.data(), d_x0, x0.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(vP.data(), d_vP, vP.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(xP.data(), d_xP, xP.size() * 8, cudaMemcpyDeviceToHost);
    bad_v = bad_x = 0;
    for(size_t i = 0; i < v0.size(); ++i)
        if(std::memcmp(&v0[i], &vP[i], 8) != 0)
            bad_v++;
    for(size_t i = 0; i < x0.size(); ++i)
        if(std::memcmp(&x0[i], &xP[i], 8) != 0)
            bad_x++;
}

template <int N>
static void run_n(int total, unsigned seed)
{
    const int          batch = 8192;
    std::mt19937_64     rng(seed);
    std::vector<double> h((size_t)batch * N * N);
    double*             d_in;
    double*             d_v[3];
    double*             d_x[3];
    cudaMalloc(&d_in, h.size() * sizeof(double));
    for(int p = 0; p < 3; ++p)
    {
        cudaMalloc(&d_v[p], (size_t)batch * N * sizeof(double));
        cudaMalloc(&d_x[p], (size_t)batch * N * N * sizeof(double));
    }
    long bad_v_up[2] = { 0, 0 }, bad_x_up[2] = { 0, 0 };
    long bad_v_lo[2] = { 0, 0 }, bad_x_lo[2] = { 0, 0 };
    long solved = 0;
    std::normal_distribution<double> G(0.0, 1.0);
    for(int done = 0; done < total; done += batch)
    {
        int n = std::min(batch, total - done);
        // random symmetric matrices with a negative-eigenvalue tail, the
        // class the projection actually meets (raw Hessians are indefinite)
        for(int i = 0; i < n; ++i)
        {
            Eigen::MatrixXd A = Eigen::MatrixXd::NullaryExpr(N, N, [&] { return G(rng); });
            Eigen::MatrixXd S = A.transpose() * A;  // PSD-ish core
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
            Eigen::VectorXd w = es.eigenvalues();
            for(int k = 0; k < N; ++k)
                if(k % 3 == 0)
                    w(k) = -w(k) * (0.1 + 0.8 * std::abs(G(rng)));
            Eigen::MatrixXd Mx =
                es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
            for(int r = 0; r < N; ++r)
                for(int c = 0; c < N; ++c)
                    h[((size_t)i) * N * N + r * N + c] = Mx(r, c);
        }
        cudaMemcpy(d_in, h.data(), (size_t)n * N * N * sizeof(double), cudaMemcpyHostToDevice);
        for(int solver = 0; solver < 2; ++solver)
        {
            long bv, bx;
            if(solver == 0)
            {
                run_arm<N, 0, 1>(d_in, d_v[0], d_x[0], d_v[1], d_x[1], n, bv, bx);
                bad_v_up[0] += bv;
                bad_x_up[0] += bx;
                run_arm<N, 0, 2>(d_in, d_v[0], d_x[0], d_v[2], d_x[2], n, bv, bx);
                bad_v_lo[0] += bv;
                bad_x_lo[0] += bx;
            }
            else
            {
                run_arm<N, 1, 1>(d_in, d_v[0], d_x[0], d_v[1], d_x[1], n, bv, bx);
                bad_v_up[1] += bv;
                bad_x_up[1] += bx;
                run_arm<N, 1, 2>(d_in, d_v[0], d_x[0], d_v[2], d_x[2], n, bv, bx);
                bad_v_lo[1] += bv;
                bad_x_lo[1] += bx;
            }
        }
        solved += n;
    }
    printf("N=%d, %ld matrices\n", N, solved);
    printf("  Solver=0 (Eigen evd):      upper-NaN -> %ld/%ld value + %ld/%ld vector words differ; lower-NaN -> %ld + %ld\n",
           bad_v_up[0], solved * N, bad_x_up[0], solved * N * N, bad_v_lo[0], bad_x_lo[0]);
    printf("  Solver=1 (evd_tridiag_ql): upper-NaN -> %ld/%ld value + %ld/%ld vector words differ; lower-NaN -> %ld + %ld\n",
           bad_v_up[1], solved * N, bad_x_up[1], solved * N * N, bad_v_lo[1], bad_x_lo[1]);
}

int main(int argc, char** argv)
{
    int total = argc > 1 ? atoi(argv[1]) : 200000;
    run_n<9>(total, 11);
    run_n<6>(total, 13);
    return 0;
}

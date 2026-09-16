// Round-7 s07 numerics verifier: old vs new PSD projection of the
// NeoHookeanShell2D (triangle membrane) Hessian, on device, against the real
// device functions.
//
// Compares, over randomised triangle inputs drawn from the crease-press
// scene's parameter and deformation ranges:
//   old      = make_spd<9,0>                          (dense 9x9, Eigen) -- the path main ships
//   new11    = make_spd_translation_free_3x3_blocked<1> (K16-style + QL)  -- shipped
//   new21    = dense-basis 9x6 + QL                   (isolates the reduction change)
//   new10    = blocked + Eigen                        (isolates the solver change)
//   old01    = dense 9x9 + QL                         (the OLD path's own solver
//                                                        sensitivity -- the bound)
//   raw      = unprojected ddEddX                     (audits the projection is
//                                                        active work + the null space)
//
// Checks: (premise) H t = 0 for the three rigid translations, raw; min
// eigenvalue of every projected result >= 0 (to rounding); relFro(old,new)
// bounded by the old path's own solver-swap sensitivity relFro(old00,old01).
//
// Build: see build_and_run.sh next to this file.

#include <utils/make_spd.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>
#include <Eigen/Eigenvalues>
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;

struct TriInput
{
    double X[9];
    double IB[4];
    double lambda, mu, Vdt2;
};

// Body replicates the shipped G/H kernel's Hessian path verbatim
// (ddEddX, project, *=Vdt2).
template <int Proj, int Solver>
__global__ void project_kernel(const TriInput* __restrict__ in,
                               double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    const TriInput& t = in[I];
    Vector9         X;
    for(int k = 0; k < 9; ++k)
        X(k) = t.X[k];
    Matrix2x2 IB;
    IB(0, 0) = t.IB[0];
    IB(0, 1) = t.IB[1];
    IB(1, 0) = t.IB[2];
    IB(1, 1) = t.IB[3];

    Matrix9x9 H;
    sym::neo_hookean_shell_2d::ddEddX(H, t.lambda, t.mu, X, IB);

    if constexpr(Proj == 1)
        make_spd_translation_free_3x3_blocked<Solver>(H);
    else if constexpr(Proj == 2)
        make_spd_translation_free_3x3<Solver>(H);
    else if constexpr(Proj == 3)
        ;  // raw, unprojected
    else
        make_spd<9, Solver>(H);

    H *= t.Vdt2;

    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            out[((size_t)I) * 81 + r * 9 + c] = H(r, c);
}

static Eigen::MatrixXd as_mat(const double* p)
{
    Eigen::MatrixXd M(9, 9);
    for(int r = 0; r < 9; ++r)
        for(int c = 0; c < 9; ++c)
            M(r, c) = p[r * 9 + c];
    return M;
}

struct RelStats
{
    double max = 0, p999 = 0, med = 0;
    long   n = 0;
    std::vector<double> all;
    void add(double v) { all.push_back(v); }
    void finish()
    {
        n = (long)all.size();
        if(!n)
            return;
        std::sort(all.begin(), all.end());
        max  = all.back();
        p999 = all[(size_t)(0.999 * (n - 1))];
        med  = all[n / 2];
    }
};

int main(int argc, char** argv)
{
    const int total    = argc > 1 ? atoi(argv[1]) : 200000;
    const int batch    = 16384;
    const unsigned seed0 = argc > 2 ? atoi(argv[2]) : 11;

    RelStats rel_new, rel_new21, rel_new10, rel_old01;
    double   min_eig_rel[5] = { 1e300, 1e300, 1e300, 1e300, 1e300 };
    double   max_null[3]    = { 0, 0, 0 };  // raw, old00, new11
    long     n_neg_raw = 0, n_zero_H = 0, n_used = 0;

    TriInput* d_in;
    double*   d_out;
    cudaMalloc(&d_in, batch * sizeof(TriInput));
    cudaMalloc(&d_out, (size_t)batch * 81 * sizeof(double));

    for(int done = 0; done < total; done += batch)
    {
        int n = std::min(batch, total - done);
        std::mt19937_64 rng(seed0 + done);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<TriInput> h(n);
        for(int i = 0; i < n; ++i)
        {
            // rest triangle: two edges with scene-scale lengths and an angle;
            // the 2x2 rest metric Bbar follows, IB = inv(Bbar).
            double l1 = 0.014 + U(rng) * 0.036;
            double l2 = 0.014 + U(rng) * 0.036;
            double ang = (25.0 + U(rng) * 65.0) * M_PI / 180.0;
            Eigen::Vector3d e1(l1, 0, 0);
            Eigen::Vector3d e2(l2 * std::cos(ang), l2 * std::sin(ang), 0);
            Eigen::Matrix2d Bbar;
            Bbar << e1.squaredNorm(), e1.dot(e2), e1.dot(e2), e2.squaredNorm();
            Eigen::Matrix2d IB = Bbar.inverse();

            // deformed triangle: random rotation R, then a random symmetric
            // stretch S with singular-value-like spectrum spanning the
            // scene's area-ratio range 0.26..17 (2% near-degenerate).
            double th = U(rng) * 2 * M_PI, ph = std::acos(2 * U(rng) - 1);
            Eigen::Matrix3d R;
            {
                Eigen::Vector3d a(std::sin(ph) * std::cos(th), std::sin(ph) * std::sin(th), std::cos(ph));
                Eigen::Vector3d b = a.cross(Eigen::Vector3d::UnitZ());
                if(b.norm() < 1e-3)
                    b = a.cross(Eigen::Vector3d::UnitX());
                b.normalize();
                Eigen::Vector3d c = a.cross(b);
                R.col(0) = a; R.col(1) = b; R.col(2) = c;
            }
            double smin = (U(rng) < 0.02) ? 0.02 + U(rng) * 0.03 : 0.2 + U(rng) * 3.8;
            double smaj = smin + 0.1 + U(rng) * 1.5;
            Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
            S(0, 0) = smin;
            S(1, 1) = smin + U(rng) * (smaj - smin);
            S(2, 2) = smaj;
            Eigen::Matrix3d A = R * S * R.transpose();
            Eigen::Vector3d x0 = Eigen::Vector3d::Zero();
            Eigen::Vector3d x1 = A * e1;
            Eigen::Vector3d x2 = A * e2;
            // small random translation (the energy is translation invariant;
            // keeping it non-zero exercises exactly that)
            Eigen::Vector3d tr((U(rng) - 0.5) * 0.4, (U(rng) - 0.5) * 0.4, (U(rng) - 0.5) * 0.4);
            x0 += tr; x1 += tr; x2 += tr;
            for(int k = 0; k < 3; ++k)
            {
                h[i].X[0 * 3 + k] = x0(k);
                h[i].X[1 * 3 + k] = x1(k);
                h[i].X[2 * 3 + k] = x2(k);
            }
            h[i].IB[0] = IB(0, 0); h[i].IB[1] = IB(0, 1);
            h[i].IB[2] = IB(1, 0); h[i].IB[3] = IB(1, 1);
            // scene: E=8e4, nu=0.35 -> lambda~3.2e4, mu~3.0e4; span 5 decades
            h[i].lambda = std::pow(10.0, 2.0 + 5.0 * U(rng));
            h[i].mu     = std::pow(10.0, 2.0 + 5.0 * U(rng));
            // Vdt2 = rest_area*2*thickness*dt^2: dt=1/60, area 1e-4..2.4e-3, 2t 1.6e-3..5e-3
            h[i].Vdt2 = std::pow(10.0, -11.0 + 2.0 * U(rng));
        }
        cudaMemcpy(d_in, h.data(), n * sizeof(TriInput), cudaMemcpyHostToDevice);

        std::vector<std::vector<double>> outs(6);
        for(int v = 0; v < 6; ++v)
        {
            // v: 0=old00 1=new11 2=new21 3=new10 4=old01 5=raw
            if(v == 0) project_kernel<0, 0><<<64, 256>>>(d_in, d_out, n);
            if(v == 1) project_kernel<1, 1><<<64, 256>>>(d_in, d_out, n);
            if(v == 2) project_kernel<2, 1><<<64, 256>>>(d_in, d_out, n);
            if(v == 3) project_kernel<1, 0><<<64, 256>>>(d_in, d_out, n);
            if(v == 4) project_kernel<0, 1><<<64, 256>>>(d_in, d_out, n);
            if(v == 5) project_kernel<3, 0><<<64, 256>>>(d_in, d_out, n);
            cudaDeviceSynchronize();
            outs[v].resize((size_t)n * 81);
            cudaMemcpy(outs[v].data(), d_out, (size_t)n * 81 * sizeof(double), cudaMemcpyDeviceToHost);
        }

        for(int i = 0; i < n; ++i)
        {
            Eigen::MatrixXd H00  = as_mat(&outs[0][(size_t)i * 81]);
            Eigen::MatrixXd H11  = as_mat(&outs[1][(size_t)i * 81]);
            Eigen::MatrixXd H21  = as_mat(&outs[2][(size_t)i * 81]);
            Eigen::MatrixXd H10  = as_mat(&outs[3][(size_t)i * 81]);
            Eigen::MatrixXd H01  = as_mat(&outs[4][(size_t)i * 81]);
            Eigen::MatrixXd Hraw = as_mat(&outs[5][(size_t)i * 81]);

            double f00 = H00.norm();
            double fraw = Hraw.norm();
            if(f00 < 1e-300 || fraw < 1e-300)
            {
                n_zero_H++;
                continue;
            }
            n_used++;

            {   // premise: raw H annihilates rigid translations
                double m = 0;
                for(int k = 0; k < 3; ++k)
                {
                    Eigen::VectorXd t = Eigen::VectorXd::Zero(9);
                    for(int a = 0; a < 3; ++a)
                        t(3 * a + k) = 1.0;
                    m = std::max(m, (Hraw * t).norm() / fraw);
                }
                max_null[0] = std::max(max_null[0], m);
            }
            for(int w = 0; w < 2; ++w)  // projected arms keep it
            {
                Eigen::MatrixXd& Hw = w == 0 ? H00 : H11;
                double m = 0;
                for(int k = 0; k < 3; ++k)
                {
                    Eigen::VectorXd t = Eigen::VectorXd::Zero(9);
                    for(int a = 0; a < 3; ++a)
                        t(3 * a + k) = 1.0;
                    m = std::max(m, (Hw * t).norm() / Hw.norm());
                }
                max_null[1 + w] = std::max(max_null[1 + w], m);
            }

            {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esr(Hraw);
                if(esr.eigenvalues().minCoeff() < 0.0)
                    n_neg_raw++;
                min_eig_rel[0] = std::min(min_eig_rel[0], esr.eigenvalues().minCoeff() / fraw);
            }
            {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es0(H00);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es1(H11);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(H21);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es3(H10);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es4(H01);
                min_eig_rel[1] = std::min(min_eig_rel[1], es0.eigenvalues().minCoeff() / f00);
                min_eig_rel[2] = std::min(min_eig_rel[2], es1.eigenvalues().minCoeff() / H11.norm());
                min_eig_rel[3] = std::min(min_eig_rel[3], es2.eigenvalues().minCoeff() / H21.norm());
                min_eig_rel[4] = std::min(min_eig_rel[4], es3.eigenvalues().minCoeff() / H10.norm());
            }

            rel_new.add((H00 - H11).norm() / f00);
            rel_new21.add((H00 - H21).norm() / f00);
            rel_new10.add((H00 - H10).norm() / f00);
            rel_old01.add((H00 - H01).norm() / f00);
        }
        printf("batch done %d/%d\n", done + n, total);
        fflush(stdout);
    }
    rel_new.finish(); rel_new21.finish(); rel_new10.finish(); rel_old01.finish();

    printf("\n== NeoHookeanShell2D projection verifier: %ld used, %ld zero-H, %ld raw-indefinite (%.1f%%)\n",
           n_used, n_zero_H, n_neg_raw, 100.0 * n_neg_raw / std::max(1L, n_used));
    printf("premise  raw ||H t||/||H||F  max = %.3e   (old %.3e, new11 %.3e)\n",
           max_null[0], max_null[1], max_null[2]);
    printf("min-eig/||H||F   raw=%.3e old00=%.3e new11=%.3e new21=%.3e new10=%.3e\n",
           min_eig_rel[0], min_eig_rel[1], min_eig_rel[2], min_eig_rel[3], min_eig_rel[4]);
    printf("relFro(old00,new11) med=%.3e p999=%.3e max=%.3e\n", rel_new.med, rel_new.p999, rel_new.max);
    printf("relFro(old00,new21) med=%.3e p999=%.3e max=%.3e\n", rel_new21.med, rel_new21.p999, rel_new21.max);
    printf("relFro(old00,new10) med=%.3e p999=%.3e max=%.3e\n", rel_new10.med, rel_new10.p999, rel_new10.max);
    printf("relFro(old00,old01) med=%.3e p999=%.3e max=%.3e   (old's own solver sensitivity)\n",
           rel_old01.med, rel_old01.p999, rel_old01.max);
    return 0;
}

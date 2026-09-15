// Round-7 s01 numerics verifier: old vs new PSD projection of the two plastic
// discrete-shell bending Hessians, on device, against the real device functions.
//
// Compares, over randomised hinge inputs drawn from the crease-press scene's
// parameter ranges:
//   old      = make_spd<12,0>            (dense 12x12, Eigen) -- today's path
//   new      = make_spd_translation_free_4x3_blocked<1> (K16 + QL) -- shipped
//   new10    = blocked + Eigen           (isolates the projection change)
//   new21    = dense-basis K7 + QL       (isolates the solver change)
//   old01    = dense 12x12 + QL          (the OLD path's own solver sensitivity
//                                          -- the bound the diff must respect)
//
// Checks: min eigenvalue of every projected result >= 0 (to rounding), the
// translation null space H t = 0 is preserved, and relFro(old,new) is bounded
// by the old path's own solver-swap sensitivity relFro(old00, old01).
//
// Build (same include set/flags as the backend TU, minus -rdc):
//   see build_and_run.sh next to this file.

#include <utils/make_spd.h>
#include <finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h>
#include <Eigen/Eigenvalues>
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

// Model 0 = strain-plastic, 1 = stress-plastic. Body replicates the shipped
// G/H kernel's Hessian path verbatim (ddEddx, *=Vdt2, project).
template <int Model, int Proj, int Solver>
__global__ void project_kernel(const HingeInput* __restrict__ in,
                               double* __restrict__ out, int n)
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
    if constexpr(Model == 0)
        sym::strain_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa);
    else
        sym::stress_plastic_discrete_shell_bending::ddEddx(
            H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa, h.yield_p);
    H *= h.Vdt2;

    if constexpr(Proj == 1)
        make_spd_translation_free_4x3_blocked<Solver>(H);
    else if constexpr(Proj == 2)
        make_spd_translation_free_4x3<Solver>(H);
    else if constexpr(Proj == 3)
        ;  // raw, unprojected -- audits that the projection is active work
    else
        make_spd<12, Solver>(H);

    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            out[((size_t)I) * 144 + r * 12 + c] = H(r, c);
}

static Eigen::MatrixXd as_mat(const double* p)
{
    Eigen::MatrixXd M(12, 12);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            M(r, c) = p[r * 12 + c];
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

static const char* kModelName[2] = { "strain-plastic", "stress-plastic" };

int main(int argc, char** argv)
{
    const int  total = argc > 1 ? atoi(argv[1]) : 200000;
    const int  batch = 16384;
    const unsigned seed0 = argc > 2 ? atoi(argv[2]) : 7;

    for(int model = 0; model < 2; ++model)
    {
        RelStats rel_new, rel_new10, rel_new21, rel_old01;
        double   min_eig[5] = { 1e300, 1e300, 1e300, 1e300, 1e300 };
        // scale-normalised min eigenvalue (min_eig / ||H||_F): the inputs span
        // six decades of kappa and nine of Vdt2, so the absolute value alone
        // would mix scales.
        double   min_eig_rel[5] = { 1e300, 1e300, 1e300, 1e300, 1e300 };
        double   max_null[3] = { 0, 0, 0 };  // raw, old00, new11
        long     n_neg_raw = 0, n_zero_H = 0, n_used = 0;

        HingeInput* d_in;
        double*     d_out;
        cudaMalloc(&d_in, batch * sizeof(HingeInput));
        cudaMalloc(&d_out, (size_t)batch * 144 * sizeof(double));

        for(int done = 0; done < total; done += batch)
        {
            int n = std::min(batch, total - done);
            std::mt19937_64 rng(seed0 + 1000 * model + done);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            std::vector<HingeInput> h(n);
            for(int i = 0; i < n; ++i)
            {
                // middle edge of length L0 in a random direction
                double L0 = 0.014 + U(rng) * 0.036;
                double th = U(rng) * 2 * M_PI, ph = std::acos(2 * U(rng) - 1);
                Eigen::Vector3d dir(std::sin(ph) * std::cos(th),
                                    std::sin(ph) * std::sin(th), std::cos(ph));
                Eigen::Vector3d any(0, 0, 1);
                if(std::abs(dir.z()) > 0.9)
                    any = Eigen::Vector3d(1, 0, 0);
                Eigen::Vector3d u = dir.cross(any).normalized();
                Eigen::Vector3d mid_edge = 0.5 * L0 * dir;
                // heights: mostly sane, with a 2% tail of near-degenerate flats
                double h0 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
                double h3 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
                double lat0 = (U(rng) - 0.5) * 0.02, lat3 = (U(rng) - 0.5) * 0.02;
                Eigen::Vector3d x1 = mid_edge, x2 = -mid_edge;
                Eigen::Vector3d x0 = lat0 * dir + h0 * u;
                Eigen::Vector3d x3 = lat3 * dir - h3 * u;  // opposite side
                for(int k = 0; k < 3; ++k)
                {
                    h[i].x[0][k] = x0(k);
                    h[i].x[1][k] = x1(k);
                    h[i].x[2][k] = x2(k);
                    h[i].x[3][k] = x3(k);
                }
                h[i].kappa = std::pow(10.0, -6.0 + 6.0 * U(rng));
                h[i].L0    = L0;
                h[i].h_bar = 0.0008 + U(rng) * 0.003;
                h[i].theta_bar = U(rng) < 0.5 ? 0.0 : (U(rng) - 0.5) * 3.0;
                // yield parameter: pick the yield ANGLE first so both the
                // elastic and the plastic branch of the stress model are
                // exercised; the strain model reads it as a threshold angle.
                double theta_y = std::pow(10.0, -3.0 + 2.8 * U(rng));
                double slope = 2.0 * h[i].kappa * L0 / h[i].h_bar;
                h[i].yield_p = theta_y * slope;
                h[i].Vdt2 = std::pow(10.0, -9.0 + 3.0 * U(rng));
            }
            cudaMemcpy(d_in, h.data(), n * sizeof(HingeInput), cudaMemcpyHostToDevice);

            std::vector<std::vector<double>> outs(6);
            for(int v = 0; v < 6; ++v)
            {
                // v: 0=old00 1=new11 2=new21 3=new10 4=old01 5=raw
                if(model == 0)
                {
                    if(v == 0) project_kernel<0, 0, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 1) project_kernel<0, 1, 1><<<64, 256>>>(d_in, d_out, n);
                    if(v == 2) project_kernel<0, 2, 1><<<64, 256>>>(d_in, d_out, n);
                    if(v == 3) project_kernel<0, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 4) project_kernel<0, 0, 1><<<64, 256>>>(d_in, d_out, n);
                    if(v == 5) project_kernel<0, 3, 0><<<64, 256>>>(d_in, d_out, n);
                }
                else
                {
                    if(v == 0) project_kernel<1, 0, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 1) project_kernel<1, 1, 1><<<64, 256>>>(d_in, d_out, n);
                    if(v == 2) project_kernel<1, 2, 1><<<64, 256>>>(d_in, d_out, n);
                    if(v == 3) project_kernel<1, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 4) project_kernel<1, 0, 1><<<64, 256>>>(d_in, d_out, n);
                    if(v == 5) project_kernel<1, 3, 0><<<64, 256>>>(d_in, d_out, n);
                }
                outs[v].resize((size_t)n * 144);
                cudaMemcpy(outs[v].data(), d_out,
                           (size_t)n * 144 * sizeof(double), cudaMemcpyDeviceToHost);
            }

            for(int i = 0; i < n; ++i)
            {
                Eigen::MatrixXd H00 = as_mat(&outs[0][(size_t)i * 144]);
                Eigen::MatrixXd H11 = as_mat(&outs[1][(size_t)i * 144]);
                Eigen::MatrixXd H21 = as_mat(&outs[2][(size_t)i * 144]);
                Eigen::MatrixXd H10 = as_mat(&outs[3][(size_t)i * 144]);
                Eigen::MatrixXd H01 = as_mat(&outs[4][(size_t)i * 144]);
                Eigen::MatrixXd Hraw = as_mat(&outs[5][(size_t)i * 144]);

                double f00 = H00.norm(), f11 = H11.norm();
                double fraw = Hraw.norm();
                if(f00 < 1e-300)
                {
                    n_zero_H++;
                    continue;
                }
                n_used++;

                {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esr(Hraw);
                    if(esr.eigenvalues().minCoeff() < 0.0)
                        n_neg_raw++;
                    max_null[0] = std::max(max_null[0], [&]
                    {
                        double m = 0;
                        for(int k = 0; k < 3; ++k)
                        {
                            Eigen::VectorXd t = Eigen::VectorXd::Zero(12);
                            for(int a = 0; a < 4; ++a)
                                t(3 * a + k) = 1.0;
                            m = std::max(m, (Hraw * t).norm() / fraw);
                        }
                        return m;
                    }());
                }

                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es0(H00);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es1(H11);
                min_eig[0] = std::min(min_eig[0], es0.eigenvalues().minCoeff());
                min_eig[1] = std::min(min_eig[1], es1.eigenvalues().minCoeff());
                min_eig_rel[0] = std::min(min_eig_rel[0], es0.eigenvalues().minCoeff() / f00);
                min_eig_rel[1] = std::min(min_eig_rel[1], es1.eigenvalues().minCoeff() / f11);
                {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H21);
                    min_eig[2] = std::min(min_eig[2], es.eigenvalues().minCoeff());
                    min_eig_rel[2] = std::min(min_eig_rel[2], es.eigenvalues().minCoeff() / H21.norm());
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(H10);
                    min_eig[3] = std::min(min_eig[3], es2.eigenvalues().minCoeff());
                    min_eig_rel[3] = std::min(min_eig_rel[3], es2.eigenvalues().minCoeff() / H10.norm());
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es3(H01);
                    min_eig[4] = std::min(min_eig[4], es3.eigenvalues().minCoeff());
                    min_eig_rel[4] = std::min(min_eig_rel[4], es3.eigenvalues().minCoeff() / H01.norm());
                }

                rel_new.add((H00 - H11).norm() / f00);
                rel_new21.add((H00 - H21).norm() / f00);
                rel_new10.add((H00 - H10).norm() / f00);
                rel_old01.add((H00 - H01).norm() / f00);

                // translation null space of the projected results
                for(int k = 0; k < 3; ++k)
                {
                    Eigen::VectorXd t = Eigen::VectorXd::Zero(12);
                    for(int a = 0; a < 4; ++a)
                        t(3 * a + k) = 1.0;
                    max_null[1] = std::max(max_null[1], (H00 * t).norm() / f00);
                    max_null[2] = std::max(max_null[2], (H11 * t).norm() / f11);
                }
            }
        }
        rel_new.finish();
        rel_new10.finish();
        rel_new21.finish();
        rel_old01.finish();

        printf("=== %s   N=%d used=%ld zeroH=%ld  rawH-negative %ld (%.2f%%)\n",
               kModelName[model], total, n_used, n_zero_H, n_neg_raw,
               100.0 * n_neg_raw / std::max(1L, n_used));
        printf("min eigenvalue  old00 %.3e  new11 %.3e  new21 %.3e  new10 %.3e  old01 %.3e\n",
               min_eig[0], min_eig[1], min_eig[2], min_eig[3], min_eig[4]);
        printf("min eig / ||H||F  old00 %.3e  new11 %.3e  new21 %.3e  new10 %.3e  old01 %.3e\n",
               min_eig_rel[0], min_eig_rel[1], min_eig_rel[2], min_eig_rel[3], min_eig_rel[4]);
        printf("null-space ||Ht||/||H||F  max: raw %.3e  old %.3e  new %.3e\n",
               max_null[0], max_null[1], max_null[2]);
        printf("relFro(old,new11)  max %.3e  p99.9 %.3e  med %.3e\n",
               rel_new.max, rel_new.p999, rel_new.med);
        printf("relFro(old,new10)  max %.3e  p99.9 %.3e  med %.3e   [projection change only]\n",
               rel_new10.max, rel_new10.p999, rel_new10.med);
        printf("relFro(old,new21)  max %.3e  p99.9 %.3e  med %.3e   [K7 + QL]\n",
               rel_new21.max, rel_new21.p999, rel_new21.med);
        printf("relFro(old,old01)  max %.3e  p99.9 %.3e  med %.3e   [old path own solver sensitivity]\n",
               rel_old01.max, rel_old01.p999, rel_old01.med);
        cudaFree(d_in);
        cudaFree(d_out);
    }
    return 0;
}

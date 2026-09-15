// Round-7 s02 numerics verifier: old vs new PSD projection of the dahl-friction
// discrete-shell bending Hessian, on device, against the real device functions.
//
// Premise check first: the dahl energy is P(theta) = kappa*w*del^2 + W(d) with
// del = wrap(theta - theta_bar), d = wrap(theta - theta_commit), and the
// committed state (theta_commit, F_commit) is a per-edge constant within a
// frame (updated once per accepted frame in the TimeIntegrator). So ddEddx is
// ddEddtheta*grad(theta)grad(theta)^T + dEdtheta*hess(theta) with everything
// through vertex differences -> H t = 0 exactly. The raw variant audits this.
//
// Compares, over randomised hinge inputs drawn from the crease-press scene's
// parameter ranges (kappa 1e-6..1, ell_e 0.03..3 rad, M_e spanning
// friction-dominated and elastic-dominated hinges, theta_commit offsets +-pi,
// F_commit fresh / interior / saturated):
//   old      = blocked + Eigen  <1,0>   -- main's shipped default (both runtime
//                                          bools on, Solver=0)
//   new      = blocked + QL     <1,1>   -- this step's shipped default
//   knob20   = dense basis + Eigen <2,0> -- main's own UIPC_DAHL_BLOCKED_PROJ=0
//                                          arm: the old binary's own knob
//                                          sensitivity (same solver, different
//                                          assembly order)
//   alloff   = dense 12x12 + Eigen <0,0> -- main's all-knobs-off arm (this
//                                          step's rollback), and
//   alloff01  = dense 12x12 + QL <0,1>   -- its own solver-swap sensitivity
//   raw      = unprojected               -- audits that the projection is
//                                          active work + the H t = 0 premise
//
// Checks: min eigenvalue of every projected result >= 0 (to rounding), the
// translation null space H t = 0 (raw: premise; projected: preserved), and
// relFro(old,new) against the old path's own knob/solver sensitivities.
//
// Build: see build_and_run.sh next to this file (s01's include set/flags).

#include <utils/make_spd.h>
#include <finite_element/constitutions/dahl_friction_discrete_shell_bending_function.h>
#include <Eigen/Eigenvalues>
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;

namespace DFDSB = sym::dahl_friction_discrete_shell_bending;

struct HingeInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar, M_e, ell_e, theta_commit, F_commit, Vdt2;
};

// Body replicates the shipped G/H kernel's Hessian path verbatim
// (dEdx_ddEddx, H *= Vdt2, project). Proj = 3 leaves H raw; the raw variant
// additionally reports theta and the friction increment d for regime binning.
template <int Proj, int Solver>
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

    Vector12    G;
    Matrix12x12 H;
    DFDSB::dEdx_ddEddx(G, H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa,
                       h.M_e, h.ell_e, h.theta_commit, h.F_commit);
    H *= h.Vdt2;

    if constexpr(Proj == 1)
        make_spd_translation_free_4x3_blocked<Solver>(H);
    else if constexpr(Proj == 2)
        make_spd_translation_free_4x3<Solver>(H);
    else if constexpr(Proj == 3)
        ;  // raw, unprojected
    else
        make_spd<12, Solver>(H);

    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            out[((size_t)I) * 146 + r * 12 + c] = H(r, c);

    if constexpr(Proj == 3)
    {
        Float theta = 0.0;
        if(DFDSB::safe_dihedral_angle(x0, x1, x2, x3, theta))
        {
            out[((size_t)I) * 146 + 144] = theta;
            out[((size_t)I) * 146 + 145] =
                DFDSB::angle_delta(theta, h.theta_commit);
        }
        else
        {
            out[((size_t)I) * 146 + 144] = 999.0;  // guard failed marker
            out[((size_t)I) * 146 + 145] = 999.0;
        }
    }
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

int main(int argc, char** argv)
{
    const int  total  = argc > 1 ? atoi(argv[1]) : 200000;
    const int  batch  = 16384;
    const unsigned seed0 = argc > 2 ? atoi(argv[2]) : 7;

    RelStats rel_new, rel_knob, rel_alloff, rel_altoff01, rel_old_knob20;
    double   min_eig_rel[5] = { 1e300, 1e300, 1e300, 1e300, 1e300 };
    double   max_null[4] = { 0, 0, 0, 0 };  // raw, old <1,0>, new <1,1>, alloff <0,0>
    long     n_neg_raw = 0, n_zero_H = 0, n_used = 0;
    long     n_guard = 0, n_fresh = 0, n_saturated = 0, n_mid = 0;
    long     n_F_fresh = 0, n_F_sat = 0;

    HingeInput* d_in;
    double*     d_out;
    cudaMalloc(&d_in, batch * sizeof(HingeInput));
    cudaMalloc(&d_out, (size_t)batch * 146 * sizeof(double));

    for(int done = 0; done < total; done += batch)
    {
        int n = std::min(batch, total - done);
        std::mt19937_64 rng(seed0 + done);
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
            // friction parameters: ell_e spans the scene's 0.3 rad by +-1.5
            // decades; M_e = m_hat * L0 with m_hat spanning weak..strong
            // friction against the elastic slope 2*kappa*L0/h_bar.
            h[i].ell_e = std::pow(10.0, -1.5 + 2.0 * U(rng));
            double m_hat = std::pow(10.0, -5.0 + 4.0 * U(rng));
            h[i].M_e     = m_hat * L0;
            // committed state: theta_commit = theta_bar +- pi (the frame
            // constant), F_commit fresh (0) / saturated (+-M_e) / interior
            h[i].theta_commit =
                h[i].theta_bar + (U(rng) < 0.1 ? 0.0 : (U(rng) - 0.5) * 2 * M_PI);
            double fr = U(rng);
            if(fr < 0.2)
            {
                h[i].F_commit = 0.0;
                n_F_fresh++;
            }
            else if(fr < 0.4)
            {
                h[i].F_commit = (U(rng) < 0.5 ? -1.0 : 1.0) * h[i].M_e;
                n_F_sat++;
            }
            else
            {
                h[i].F_commit = (2 * U(rng) - 1) * h[i].M_e;
            }
            h[i].Vdt2 = std::pow(10.0, -9.0 + 3.0 * U(rng));
        }
        cudaMemcpy(d_in, h.data(), n * sizeof(HingeInput), cudaMemcpyHostToDevice);

        std::vector<std::vector<double>> outs(6);
        for(int v = 0; v < 6; ++v)
        {
            // v: 0=old<1,0> 1=new<1,1> 2=knob<2,0> 3=alloff<0,0> 4=alloff01<0,1> 5=raw
            if(v == 0) project_kernel<1, 0><<<64, 256>>>(d_in, d_out, n);
            if(v == 1) project_kernel<1, 1><<<64, 256>>>(d_in, d_out, n);
            if(v == 2) project_kernel<2, 0><<<64, 256>>>(d_in, d_out, n);
            if(v == 3) project_kernel<0, 0><<<64, 256>>>(d_in, d_out, n);
            if(v == 4) project_kernel<0, 1><<<64, 256>>>(d_in, d_out, n);
            if(v == 5) project_kernel<3, 0><<<64, 256>>>(d_in, d_out, n);
            outs[v].resize((size_t)n * 146);
            cudaMemcpy(outs[v].data(), d_out,
                       (size_t)n * 146 * sizeof(double), cudaMemcpyDeviceToHost);
        }

        for(int i = 0; i < n; ++i)
        {
            Eigen::MatrixXd Hold = as_mat(&outs[0][(size_t)i * 146]);
            Eigen::MatrixXd Hnew = as_mat(&outs[1][(size_t)i * 146]);
            Eigen::MatrixXd Hk20 = as_mat(&outs[2][(size_t)i * 146]);
            Eigen::MatrixXd Ha00 = as_mat(&outs[3][(size_t)i * 146]);
            Eigen::MatrixXd Ha01 = as_mat(&outs[4][(size_t)i * 146]);
            Eigen::MatrixXd Hraw = as_mat(&outs[5][(size_t)i * 146]);
            double theta = outs[5][(size_t)i * 146 + 144];
            double d     = outs[5][(size_t)i * 146 + 145];

            double fold = Hold.norm();
            if(fold < 1e-300)
            {
                n_zero_H++;
                continue;
            }
            n_used++;

            if(theta > 100.0)
                n_guard++;  // dihedral guard fired (H = 0 handled above)
            else
            {
                double x = std::abs(d) / h[i].ell_e;
                if(x < 0.1)
                    n_fresh++;
                else if(x > 10.0)
                    n_saturated++;
                else
                    n_mid++;
            }

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esr(Hraw);
            if(esr.eigenvalues().minCoeff() < 0.0)
                n_neg_raw++;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es1(Hnew);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(Hk20);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es3(Ha00);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es4(Ha01);
            min_eig_rel[1] = std::min(min_eig_rel[1], es1.eigenvalues().minCoeff() / Hnew.norm());
            min_eig_rel[2] = std::min(min_eig_rel[2], es2.eigenvalues().minCoeff() / Hk20.norm());
            min_eig_rel[3] = std::min(min_eig_rel[3], es3.eigenvalues().minCoeff() / Ha00.norm());
            min_eig_rel[4] = std::min(min_eig_rel[4], es4.eigenvalues().minCoeff() / Ha01.norm());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es0(Hold);
            min_eig_rel[0] = std::min(min_eig_rel[0], es0.eigenvalues().minCoeff() / fold);

            rel_new.add((Hold - Hnew).norm() / fold);
            rel_knob.add((Hold - Hk20).norm() / fold);      // old's own BLOCKED_PROJ knob
            rel_alloff.add((Hold - Ha00).norm() / fold);    // old default vs all-off arm
            rel_altoff01.add((Ha00 - Ha01).norm() / fold);  // all-off arm's solver swap
            rel_old_knob20.add((Ha00 - Hk20).norm() / fold);

            // translation null space
            for(int k = 0; k < 3; ++k)
            {
                Eigen::VectorXd t = Eigen::VectorXd::Zero(12);
                for(int a = 0; a < 4; ++a)
                    t(3 * a + k) = 1.0;
                max_null[0] = std::max(max_null[0], (Hraw * t).norm() / Hraw.norm());
                max_null[1] = std::max(max_null[1], (Hold * t).norm() / fold);
                max_null[2] = std::max(max_null[2], (Hnew * t).norm() / Hnew.norm());
                max_null[3] = std::max(max_null[3], (Ha00 * t).norm() / Ha00.norm());
            }
        }
    }
    rel_new.finish();
    rel_knob.finish();
    rel_alloff.finish();
    rel_altoff01.finish();
    rel_old_knob20.finish();

    printf("=== dahl-friction   N=%d used=%ld zeroH=%ld guardKilled=%ld\n",
           total, n_used, n_zero_H, n_guard);
    printf("regime coverage: |d|/ell_e <0.1 (fresh) %ld  0.1..10 %ld  >10 (saturated) %ld\n",
           n_fresh, n_mid, n_saturated);
    printf("F_commit: fresh(=0) %ld  saturated(=+-M_e) %ld  interior %ld\n",
           n_F_fresh, n_F_sat, total - n_F_fresh - n_F_sat);
    printf("raw H indefinite (projection is active work): %ld (%.2f%%)\n",
           n_neg_raw, 100.0 * n_neg_raw / std::max(1L, n_used));
    printf("null-space ||Ht||/||H||F max: raw(premise) %.3e  old<1,0> %.3e  new<1,1> %.3e  alloff<0,0> %.3e\n",
           max_null[0], max_null[1], max_null[2], max_null[3]);
    printf("min eig / ||H||F:  old<1,0> %.3e  new<1,1> %.3e  knob<2,0> %.3e  alloff<0,0> %.3e  alloff<0,1> %.3e\n",
           min_eig_rel[0], min_eig_rel[1], min_eig_rel[2], min_eig_rel[3], min_eig_rel[4]);
    printf("relFro(old<1,0>, new<1,1>)   max %.3e  p99.9 %.3e  med %.3e   [THE CHANGE: solver swap]\n",
           rel_new.max, rel_new.p999, rel_new.med);
    printf("relFro(old<1,0>, knob<2,0>)  max %.3e  p99.9 %.3e  med %.3e   [old binary's own BLOCKED_PROJ knob]\n",
           rel_knob.max, rel_knob.p999, rel_knob.med);
    printf("relFro(alloff<0,0>,<0,1>)    max %.3e  p99.9 %.3e  med %.3e   [all-off arm's own solver swap]\n",
           rel_altoff01.max, rel_altoff01.p999, rel_altoff01.med);
    printf("relFro(old<1,0>, alloff<0,0>) max %.3e  p99.9 %.3e  med %.3e  [old default vs old all-off]\n",
           rel_alloff.max, rel_alloff.p999, rel_alloff.med);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

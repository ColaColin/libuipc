// s02 numerics probe (host, double precision, real UIPC_GENERIC device functions):
//  1) the exact hinge Hessian really is  E'' * g g^T + E' * hess(theta)
//  2) the Gauss-Newton Hessian is exactly symmetric and PSD
//  3) how much of the exact Hessian the approximation discards, and how often
//     the exact one is actually indefinite (i.e. how often the PSD projection
//     had anything to do).
#include <finite_element/constitutions/discrete_shell_bending_function.h>
#include <utils/make_spd.h>
#include <Eigen/Eigenvalues>
#include <cstdio>
#include <random>
#include <cmath>

using namespace uipc;
using namespace uipc::backend::cuda;
namespace DSB = sym::discrete_shell_bending;

int main(int argc, char** argv)
{
    const int N = argc > 1 ? atoi(argv[1]) : 200000;
    const double AMP = argc > 2 ? atof(argv[2]) : 0.6;  // deformation amplitude
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    double max_term_err = 0, max_asym = 0, min_gn_eig_rel = 0;
    double max_drop_ratio = 0, sum_drop_ratio = 0;
    long   indefinite = 0, ok = 0, gn_neg = 0, rank_sum = 0;
    double dtheta_sum = 0, dtheta_max = 0;
    double sum_negfrac = 0;

    for(int s = 0; s < N; ++s)
    {
        Vector3 X[4], x[4];
        for(int i = 0; i < 4; ++i)
            for(int k = 0; k < 3; ++k) X[i][k] = U(rng);
        // rest shape: a real hinge (two triangles sharing edge X1-X2)
        for(int i = 0; i < 4; ++i)
            for(int k = 0; k < 3; ++k) x[i][k] = X[i][k] + AMP * U(rng);

        Float L0, h_bar, theta_bar, V_bar;
        DSB::compute_constants(L0, h_bar, theta_bar, V_bar, X[0], X[1], X[2], X[3], 0, 0, 0, 0);
        Float kappa = 1e-3 * std::fabs(U(rng)) + 1e-6;
        if(!std::isfinite(L0) || !std::isfinite(h_bar) || !std::isfinite(theta_bar) || h_bar <= 0)
            continue;

        Matrix12x12 Hfull, Hgn;
        DSB::ddEddx(Hfull, x[0], x[1], x[2], x[3], L0, h_bar, theta_bar, kappa);
        DSB::ddEddx_gauss_newton(Hgn, x[0], x[1], x[2], x[3], L0, h_bar, theta_bar, kappa, 1.0);
        if(!Hfull.allFinite() || !Hgn.allFinite()) continue;
        ++ok;

        // (1) the dropped term equals E' * hess(theta)
        Float theta, dEdtheta;
        dihedral_angle(x[0], x[1], x[2], x[3], theta);
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);
        Matrix12x12 Hth;
        dihedral_angle_hessian(x[0], x[1], x[2], x[3], Hth);
        Matrix12x12 dropped = Hfull - Hgn;
        Matrix12x12 pred    = dEdtheta * Hth;
        double scale = std::max(1e-300, Hfull.cwiseAbs().maxCoeff());
        max_term_err = std::max(max_term_err, (dropped - pred).cwiseAbs().maxCoeff() / scale);

        // (2) exact symmetry + PSD of the Gauss-Newton Hessian
        max_asym = std::max(max_asym, (Hgn - Hgn.transpose()).cwiseAbs().maxCoeff());
        Eigen::SelfAdjointEigenSolver<Matrix12x12> es_gn(Hgn);
        double gnmax = es_gn.eigenvalues().cwiseAbs().maxCoeff();
        double gnmin = es_gn.eigenvalues().minCoeff();
        if(gnmax > 0)
        {
            min_gn_eig_rel = std::min(min_gn_eig_rel, gnmin / gnmax);
            if(gnmin < -1e-12 * gnmax) ++gn_neg;
        }

        // (3) how indefinite is the exact Hessian, and how big is the dropped term
        Eigen::SelfAdjointEigenSolver<Matrix12x12> es(Hfull);
        auto ev = es.eigenvalues();
        double amax = ev.cwiseAbs().maxCoeff();
        double negsum = 0, possum = 0;
        int nneg = 0;
        for(int i = 0; i < 12; ++i)
        {
            if(ev[i] < -1e-12 * amax) { negsum += -ev[i]; ++nneg; }
            else if(ev[i] > 0) possum += ev[i];
        }
        if(nneg) ++indefinite;
        for(int i = 0; i < 12; ++i) if(ev[i] > 1e-12 * amax) ++rank_sum;
        double dth = std::fabs(theta - theta_bar); dtheta_sum += dth; dtheta_max = std::max(dtheta_max, dth);
        sum_negfrac += (possum + negsum) > 0 ? negsum / (possum + negsum) : 0.0;
        double r = dropped.norm() / std::max(1e-300, Hfull.norm());
        max_drop_ratio = std::max(max_drop_ratio, r);
        sum_drop_ratio += r;
    }

    std::printf("samples usable                              : %ld / %d\n", ok, N);
    std::printf("(1) max |(H_full - H_gn) - E'*hess(theta)| / max|H_full| : %.3e\n", max_term_err);
    std::printf("(2) max |H_gn - H_gn^T|                     : %.3e (exact symmetry)\n", max_asym);
    std::printf("    min eig(H_gn) / max|eig(H_gn)|          : %.3e\n", min_gn_eig_rel);
    std::printf("    samples with eig(H_gn) < -1e-12*max     : %ld\n", gn_neg);
    std::printf("(3) exact H indefinite in                   : %ld / %ld = %.1f%% of samples\n",
                indefinite, ok, 100.0 * indefinite / std::max(1L, ok));
    std::printf("    mean |negative| share of sum|eig(H_full)| : %.4f\n", sum_negfrac / std::max(1L, ok));
    std::printf("    ||H_full - H_gn|| / ||H_full||  mean %.4f  max %.4f\n",
                sum_drop_ratio / std::max(1L, ok), max_drop_ratio);
    std::printf("    mean rank of the PSD-projected exact Hessian : %.2f  (Gauss-Newton rank is 1)\n",
                double(rank_sum) / std::max(1L, ok));
    std::printf("    |theta - theta_bar| (rad)  mean %.4f  max %.4f   [deformation amplitude %.2f]\n",
                dtheta_sum / std::max(1L, ok), dtheta_max, AMP);
    return 0;
}

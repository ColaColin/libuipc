// Round-7 s04 math verifier: is ddEddtheta >= 0 on every REACHABLE state of
// BOTH plastic bending hinges, so that the Gauss-Newton Hessian
//   H_gn = ddEddtheta * grad(theta) grad(theta)^T
// is PSD by construction? Two constitutions with different yield laws, each
// swept separately against its own real device functions:
//
//  STRAIN-plastic (cardboard):  E = (L0*kappa/h_bar) * del^2,
//    del = wrap(theta - theta_bar), theta_bar = COMMITTED plastic angle.
//    ddEddtheta = 2*L0*kappa/h_bar  (a CONSTANT -- no dependence on theta, on
//    the committed theta_bar, or on the yield threshold; the commit only
//    moves theta_bar). The claim to check: non-negative for every reachable
//    (kappa, L0, h_bar); the yield law enters only through where theta_bar
//    can sit, which the commit chain must cover.
//  STRESS-plastic (sheet metal): piecewise in del = wrap(theta - theta_bar):
//    |del| <= theta_y:  ddEddtheta = elastic_slope = 2*kappa*L0/h_bar
//    |del| >  theta_y:  ddEddtheta = 0 exactly,  dEdtheta = sign(del)*ys
//    theta_y = yield_stress / elastic_slope, yield_stress = COMMITTED state
//    (grows by hardening*delta_gamma under the commit).
//    Above yield the GN hinge contributes ZERO curvature -- that is the
//    controlled perturbation this kernel adds (the old path keeps the
//    indefinite dEdtheta*hess(theta) term there, PSD-projected).
//
// Reachable-state sampling of the committed constants (s03's methodology):
//   R (45%): (theta_bar, yield state) produced by CHAINING the real
//            update_plastic_state() -- the exact mechanism each
//            TimeIntegrator's do_update_state kernel runs once per accepted
//            frame (strain: theta_bar += dir*(|del|-yt), yt += H*excess;
//             stress: theta_bar += sign*delta_gamma, ys += H*delta_gamma).
//   B (35%): committed state drawn directly over its producible range
//            (theta_bar anywhere in +-pi -- a chain of commits can walk it
//            anywhere; yield state >= 0 up to decades of hardening growth,
//            including the exact boundaries 0 and the elastic limit).
//   Z (10%): fresh hinge (theta_bar = rest angle, initial yield state).
//   X (10%): OUT-OF-RANGE imports the engine cannot produce through the
//            commit: negative kappa (the front-end validates bending
//            stiffness for finiteness only, not sign), negative yield
//            state. Documents what the shipped max(ddEddtheta, 0) clamp (or
//            the stress kernel's own guards) does with them.
//
// theta itself is placed exactly (rotate one wing around the middle edge), so
// del covers +-pi by construction.
//
// Checks (all against the REAL device functions):
//   1. ddEddtheta >= 0 on every R/B/Z sample; margin statistics.
//   2. The split is what the shipped exact path assembles:
//      H_full == ddEddtheta*g*g^T + dEdtheta*hess(theta) to rounding.
//   3. The planned GN fill (mirrored rank-1, clamp, scale folded in):
//      exactly symmetric, min-eig/norm >= 0, H_gn t = 0.
//   4. Size of the dropped dEdtheta*hess(theta) term vs ||H_full||F -- the
//      controlled perturbation GN introduces -- binned by yield regime.
//   5. relFro(old SHIPPED Hessian <1,1> projected, GN) -- the A/B delta.
//   6. Stress only: how often ddEddtheta == 0 (the yielded fraction) and how
//      often the response's own guards kill the hinge entirely.
//
// Build: see build_and_run.sh (s03's include set/flags).

#include <finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h>
#include <utils/make_spd.h>
#include <utils/dihedral_angle.h>
#include <random>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <Eigen/Eigenvalues>

using namespace uipc;
using namespace uipc::backend::cuda;

namespace PDSB  = sym::strain_plastic_discrete_shell_bending;
namespace SPDSB = sym::stress_plastic_discrete_shell_bending;

struct StrainInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar;
    int    cls;  // 0=R chained, 1=B direct, 2=Z fresh, 3=X out-of-range
};

struct StressInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar, yield_stress;
    int    cls;
};

constexpr int STRIDE = 600;  // 144 H_full + 144 hess + 12 g + 144 H_gn + 8 scalars + 144 H_proj

template <typename In>
__device__ void place_geometry(const In& h, Vector3& x0, Vector3& x1, Vector3& x2, Vector3& x3)
{
    x0 = Vector3(h.x[0][0], h.x[0][1], h.x[0][2]);
    x1 = Vector3(h.x[1][0], h.x[1][1], h.x[1][2]);
    x2 = Vector3(h.x[2][0], h.x[2][1], h.x[2][2]);
    x3 = Vector3(h.x[3][0], h.x[3][1], h.x[3][2]);
}

// The exact GN fill loop the shipped kernels will use (mirrored, clamp, scale
// folded into the rank-1 coefficient).
__device__ void gn_fill(Matrix12x12& Hgn, const Vector12& g, Float ddEddtheta, Float scale)
{
    const Float cc = (ddEddtheta < Float(0) ? Float(0) : ddEddtheta) * scale;
#pragma unroll
    for(int i = 0; i < 12; ++i)
    {
        const Float ci = cc * g(i);
        Hgn(i, i)      = ci * g(i);
#pragma unroll
        for(int j = i + 1; j < 12; ++j)
        {
            const Float v = ci * g(j);
            Hgn(i, j)     = v;
            Hgn(j, i)     = v;
        }
    }
}

__global__ void strain_extract(const StrainInput* __restrict__ in,
                               double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    const StrainInput& h = in[I];
    Vector3 x0, x1, x2, x3;
    place_geometry(h, x0, x1, x2, x3);
    double* o = out + (size_t)I * STRIDE;

    // the real exact path
    Matrix12x12 H;
    PDSB::ddEddx(H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[r * 12 + c] = H(r, c);

    // the old arm's SHIPPED Hessian: same H through the blocked 9x9 projection
    Matrix12x12 Hp = H;
    make_spd_translation_free_4x3_blocked<1>(Hp);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[452 + r * 12 + c] = Hp(r, c);

    Matrix12x12 hess;
    dihedral_angle_hessian(x0, x1, x2, x3, hess);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[144 + r * 12 + c] = hess(r, c);

    Vector12 g;
    dihedral_angle_gradient(x0, x1, x2, x3, g);
    for(int i = 0; i < 12; ++i)
        o[288 + i] = g(i);

    // scalars through the real generated functions
    Float theta = 0.0, delta = 0.0;
    bool  okang = PDSB::try_angle_delta(x0, x1, x2, x3, h.theta_bar, theta, delta);
    Float dEdt = 0.0, ddEdd = 0.0;
    if(okang)
    {
        PDSB::dEdtheta(dEdt, h.kappa, h.theta_bar + delta, h.theta_bar, h.L0, h.h_bar);
        PDSB::ddEddtheta(ddEdd, h.kappa, h.theta_bar + delta, h.theta_bar, h.L0, h.h_bar);
    }

    Matrix12x12 Hgn;
    gn_fill(Hgn, g, okang ? ddEdd : Float(0), Float(1));
    if(!okang)
        Hgn.setZero();
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[300 + r * 12 + c] = Hgn(r, c);

    o[444] = okang ? theta : 999.0;
    o[445] = okang ? delta : 999.0;
    o[446] = okang ? ddEdd : 999.0;
    o[447] = okang ? dEdt : 999.0;
    o[448] = (okang && ddEdd < Float(0)) ? 1.0 : 0.0;  // clamp activation marker
    o[449] = h.cls;
    o[450] = okang ? 1.0 : 0.0;
    o[451] = 0.0;
}

__global__ void stress_extract(const StressInput* __restrict__ in,
                               double* __restrict__ out, int n)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= n)
        return;
    const StressInput& h = in[I];
    Vector3 x0, x1, x2, x3;
    place_geometry(h, x0, x1, x2, x3);
    double* o = out + (size_t)I * STRIDE;

    Matrix12x12 H;
    SPDSB::ddEddx(H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa, h.yield_stress);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[r * 12 + c] = H(r, c);

    Matrix12x12 Hp = H;
    make_spd_translation_free_4x3_blocked<1>(Hp);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[452 + r * 12 + c] = Hp(r, c);

    Matrix12x12 hess;
    dihedral_angle_hessian(x0, x1, x2, x3, hess);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[144 + r * 12 + c] = hess(r, c);

    Vector12 g;
    dihedral_angle_gradient(x0, x1, x2, x3, g);
    for(int i = 0; i < 12; ++i)
        o[288 + i] = g(i);

    // scalars through the real response
    Float theta = 0.0, delta = 0.0;
    bool  okang = SPDSB::try_angle_delta(x0, x1, x2, x3, h.theta_bar, theta, delta);
    Float energy = 0.0, dEdt = 0.0, ddEdd = 0.0;
    bool  okresp = false;
    if(okang)
        okresp = SPDSB::augmented_response_from_angle_delta(
            delta, h.kappa, h.L0, h.h_bar, h.yield_stress, energy, dEdt, ddEdd);

    Matrix12x12 Hgn;
    if(okresp)
        gn_fill(Hgn, g, ddEdd, Float(1));
    else
        Hgn.setZero();
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[300 + r * 12 + c] = Hgn(r, c);

    o[444] = okang ? theta : 999.0;
    o[445] = okang ? delta : 999.0;
    o[446] = okresp ? ddEdd : 999.0;
    o[447] = okresp ? dEdt : 999.0;
    o[448] = (okresp && ddEdd < Float(0)) ? 1.0 : 0.0;
    o[449] = h.cls;
    o[450] = okresp ? 1.0 : 0.0;  // response ok (else the hinge contributes nothing)
    o[451] = (okresp && ddEdd == Float(0)) ? 1.0 : 0.0;  // yielded (zero curvature)
}

static Eigen::MatrixXd as_mat(const double* p, int off)
{
    Eigen::MatrixXd M(12, 12);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            M(r, c) = p[off + r * 12 + c];
    return M;
}

// Rodrigues rotation of v around unit axis a by angle t.
static Eigen::Vector3d rot(const Eigen::Vector3d& v, const Eigen::Vector3d& a, double t)
{
    return v * std::cos(t) + a.cross(v) * std::sin(t) + a * (a.dot(v) * (1 - std::cos(t)));
}

// Fills x[4][3] with a hinge whose dihedral angle is theta_t exactly (s03's
// construction: middle edge along dir, wings rotated by theta_t).
static void make_hinge(double x[4][3], std::mt19937_64& rng,
                       std::uniform_real_distribution<double>& U,
                       double L0, double theta_t)
{
    double th = U(rng) * 2 * M_PI, ph = std::acos(2 * U(rng) - 1);
    Eigen::Vector3d dir(std::sin(ph) * std::cos(th), std::sin(ph) * std::cos(th), std::cos(ph));
    dir.normalize();
    Eigen::Vector3d any(0, 0, 1);
    if(std::abs(dir.z()) > 0.9)
        any = Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d u = dir.cross(any).normalized();
    Eigen::Vector3d mid = 0.5 * L0 * dir;

    double h0 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
    double h3 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
    double lat0 = (U(rng) - 0.5) * 0.02, lat3 = (U(rng) - 0.5) * 0.02;
    Eigen::Vector3d x0 = lat0 * dir + h0 * u;
    Eigen::Vector3d x3f = lat3 * dir - h3 * u;  // theta = 0 side
    Eigen::Vector3d x3 = rot(x3f, dir, theta_t);
    Eigen::Vector3d x1 = mid, x2 = -mid;
    for(int k = 0; k < 3; ++k)
    {
        x[0][k] = x0(k);
        x[1][k] = x1(k);
        x[2][k] = x2(k);
        x[3][k] = x3(k);
    }
}

struct Stats
{
    long   n_cls[4] = { 0, 0, 0, 0 };
    long   n_neg_reach = 0, n_neg_out = 0, n_clamp_reach = 0, n_clamp_out = 0;
    double min_dd_reach = 1e300, min_dd_out = 1e300;
    double min_margin_rel = 1e300;
    double max_split_resid = 0, max_asym = 0, min_eig_gn = 1e300, max_null_gn = 0;
    std::vector<double> drop_all, drop_fresh, drop_mid, drop_far;
    std::vector<double> rel_ab;
    long   n_used = 0, n_zeroH = 0, n_guard = 0, n_yield0 = 0, n_resp_ok = 0;
    long   n_reg[3] = { 0, 0, 0 };
    double argmin_ctx[4] = { 0, 0, 0, 0 };
};

int main(int argc, char** argv)
{
    const int      total = argc > 1 ? atoi(argv[1]) : 200000;
    const int      batch = 16384;
    const unsigned seed0 = argc > 2 ? atoi(argv[2]) : 7;

    // ---------------- STRAIN ----------------
    {
        Stats S;
        StrainInput* d_in;
        double*      d_out;
        cudaMalloc(&d_in, batch * sizeof(StrainInput));
        cudaMalloc(&d_out, (size_t)batch * STRIDE * sizeof(double));

        for(int done = 0; done < total; done += batch)
        {
            int n = std::min(batch, total - done);
            std::mt19937_64             rng(seed0 + 101 * done);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            std::vector<StrainInput> h(n);
            for(int i = 0; i < n; ++i)
            {
                double L0 = 0.005 + U(rng) * 0.045;
                double theta_t = (U(rng) < 0.02) ? (U(rng) < 0.5 ? -M_PI + 1e-3 : M_PI - 1e-3)
                                                 : (2 * U(rng) - 1) * M_PI;
                make_hinge(h[i].x, rng, U, L0, theta_t);

                h[i].kappa = std::pow(10.0, -6.0 + 6.0 * U(rng));
                h[i].L0    = L0;
                h[i].h_bar = std::pow(10.0, -4.0 + 2.0 * U(rng));
                double rest_tb = U(rng) < 0.5 ? 0.0 : (U(rng) - 0.5) * 1.0;  // rest dihedral
                double yt0 = std::pow(10.0, -4.0 + 3.5 * U(rng));           // yield threshold [rad]
                double Hmod = (U(rng) < 0.5) ? 0.0 : std::pow(10.0, -2.0 + 3.0 * U(rng));

                int cls;
                double cr = U(rng);
                double theta_bar = rest_tb;
                if(cr < 0.45)
                {
                    cls = 0;  // R: chain the REAL strain commit
                    double tb = rest_tb, yt = yt0;
                    int K = 1 + (int)(U(rng) * 8);
                    for(int s = 0; s < K; ++s)
                    {
                        double step = (U(rng) < 0.1 ? 2.0 : std::pow(10.0, -3.0 + 2.5 * U(rng)))
                                      * (U(rng) < 0.5 ? -1.0 : 1.0);
                        double tgt = tb + step;  // the frame's final angle
                        if(!PDSB::update_plastic_state<double>(tgt, tb, yt, Hmod))
                            break;
                    }
                    theta_bar = tb;
                }
                else if(cr < 0.80)
                {
                    cls = 1;  // B: committed theta_bar anywhere a chain can walk it
                    theta_bar = (2 * U(rng) - 1) * M_PI;
                }
                else if(cr < 0.90)
                {
                    cls = 2;  // Z: fresh
                    theta_bar = rest_tb;
                }
                else
                {
                    cls = 3;  // X: out-of-range import (negative stiffness)
                    h[i].kappa = -std::pow(10.0, -6.0 + 6.0 * U(rng));
                }
                h[i].cls      = cls;
                h[i].theta_bar = theta_bar;
            }
            cudaMemcpy(d_in, h.data(), n * sizeof(StrainInput), cudaMemcpyHostToDevice);

            std::vector<double> out((size_t)n * STRIDE);
            strain_extract<<<64, 256>>>(d_in, d_out, n);
            cudaMemcpy(out.data(), d_out, (size_t)n * STRIDE * sizeof(double),
                       cudaMemcpyDeviceToHost);

            for(int i = 0; i < n; ++i)
            {
                const double* o = &out[(size_t)i * STRIDE];
                int cls = (int)o[449];
                S.n_cls[cls]++;
                if(o[450] < 0.5)
                {
                    S.n_guard++;
                    continue;
                }
                double theta = o[444], delta = o[445];
                double ddEdd = o[446], dEdt = o[447];

                Eigen::MatrixXd Hfull = as_mat(o, 0);
                Eigen::MatrixXd hess = as_mat(o, 144);
                Eigen::VectorXd g(12);
                for(int k = 0; k < 12; ++k)
                    g(k) = o[288 + k];
                Eigen::MatrixXd Hgn = as_mat(o, 300);
                Eigen::MatrixXd Hproj = as_mat(o, 452);

                double ffull = Hfull.norm();
                if(ffull < 1e-300)
                {
                    S.n_zeroH++;
                    continue;
                }
                S.n_used++;

                double slope = 2.0 * h[i].kappa * h[i].L0 / h[i].h_bar;
                if(cls == 3)
                {
                    if(ddEdd < 0)
                        S.n_neg_out++;
                    S.min_dd_out = std::min(S.min_dd_out, ddEdd);
                    if(ddEdd < 0)
                        S.n_clamp_out++;
                }
                else
                {
                    if(ddEdd < 0)
                        S.n_neg_reach++;
                    double mrel = ddEdd / slope;  // slope is the constant itself
                    if(mrel < S.min_margin_rel)
                    {
                        S.min_margin_rel = mrel;
                        S.argmin_ctx[0]  = std::abs(delta);
                        S.argmin_ctx[1]  = ddEdd;
                        S.argmin_ctx[2]  = mrel;
                        S.argmin_ctx[3]  = theta;
                    }
                    S.min_dd_reach = std::min(S.min_dd_reach, ddEdd);
                    if(ddEdd < 0)
                        S.n_clamp_reach++;
                }

                Eigen::MatrixXd split = ddEdd * g * g.transpose() + dEdt * hess;
                S.max_split_resid = std::max(S.max_split_resid, (Hfull - split).norm() / ffull);

                S.max_asym = std::max(S.max_asym, (Hgn - Hgn.transpose()).norm() / Hgn.norm());
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hgn);
                double min_e = es.eigenvalues().minCoeff();
                if(Hgn.norm() > 1e-300)
                    S.min_eig_gn = std::min(S.min_eig_gn, min_e / Hgn.norm());
                for(int k = 0; k < 3; ++k)
                {
                    Eigen::VectorXd t = Eigen::VectorXd::Zero(12);
                    for(int a = 0; a < 4; ++a)
                        t(3 * a + k) = 1.0;
                    if(Hgn.norm() > 1e-300)
                        S.max_null_gn = std::max(S.max_null_gn, (Hgn * t).norm() / Hgn.norm());
                }

                double rel_drop = (dEdt * hess).norm() / ffull;
                S.drop_all.push_back(rel_drop);
                if(Hproj.norm() > 1e-300)
                    S.rel_ab.push_back((Hproj - Hgn).norm() / Hproj.norm());
                double ad = std::abs(delta);
                if(ad < 0.05) { S.drop_fresh.push_back(rel_drop); S.n_reg[0]++; }
                else if(ad < 1.0) { S.drop_mid.push_back(rel_drop); S.n_reg[1]++; }
                else { S.drop_far.push_back(rel_drop); S.n_reg[2]++; }
            }
        }

        auto q = [](std::vector<double>& v, double p)
        {
            if(v.empty()) return -1.0;
            std::sort(v.begin(), v.end());
            return v[(size_t)(p * (v.size() - 1))];
        };
        auto md = [](std::vector<double>& v) { return v.empty() ? -1.0 : v[v.size() / 2]; };

        printf("=== STRAIN-plastic GN premise N=%d used=%ld zeroH=%ld guardKilled=%ld\n",
               total, S.n_used, S.n_zeroH, S.n_guard);
        printf("classes: R(chained-commit) %ld  B(direct) %ld  Z(fresh) %ld  X(out-of-range) %ld\n",
               S.n_cls[0], S.n_cls[1], S.n_cls[2], S.n_cls[3]);
        printf("|del| regime: <0.05 %ld  0.05..1 %ld  >1 %ld\n", S.n_reg[0], S.n_reg[1], S.n_reg[2]);
        printf("[1] ddEddtheta<0 on REACHABLE (R+B+Z): %ld   min ddEddtheta %.6e   min margin ddEdd/(2*k*L0/h) %.6e\n",
               S.n_neg_reach, S.min_dd_reach, S.min_margin_rel);
        printf("[1] argmin sample: |del| %.3f  ddEdd %.3e  margin %.3e  theta %.3f\n",
               S.argmin_ctx[0], S.argmin_ctx[1], S.argmin_ctx[2], S.argmin_ctx[3]);
        printf("[1] clamp activations reachable %ld   out-of-range %ld / %ld (min ddEddtheta %.6e)\n",
               S.n_clamp_reach, S.n_clamp_out, S.n_cls[3], S.min_dd_out);
        printf("[2] ||H_full-(ddEdd*g*g'+dEdt*hess)||/||H_full|| max %.3e\n", S.max_split_resid);
        printf("[3] GN fill: max asym %.3e   min eig/||H|| %.3e   max ||Hgn*t||/||Hgn|| %.3e\n",
               S.max_asym, S.min_eig_gn, S.max_null_gn);
        printf("[4] ||dEdt*hess||/||H_full||: med %.4f p99 %.4f max %.4f | del<0.05 %.4f mid %.4f far %.4f\n",
               md(S.drop_all), q(S.drop_all, 0.99), q(S.drop_all, 1.0),
               md(S.drop_fresh), md(S.drop_mid), md(S.drop_far));
        printf("[4] relFro(old shipped projected <1,1>, GN): med %.4f p1 %.4f p99 %.4f max %.4f  [THE A/B HESSIAN DELTA]\n",
               md(S.rel_ab), q(S.rel_ab, 0.01), q(S.rel_ab, 0.99), q(S.rel_ab, 1.0));
        cudaFree(d_in);
        cudaFree(d_out);
    }

    // ---------------- STRESS ----------------
    {
        Stats S;
        StressInput* d_in;
        double*      d_out;
        cudaMalloc(&d_in, batch * sizeof(StressInput));
        cudaMalloc(&d_out, (size_t)batch * STRIDE * sizeof(double));

        for(int done = 0; done < total; done += batch)
        {
            int n = std::min(batch, total - done);
            std::mt19937_64             rng(seed0 + 202 * done);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            std::vector<StressInput> h(n);
            for(int i = 0; i < n; ++i)
            {
                double L0 = 0.005 + U(rng) * 0.045;
                double theta_t = (U(rng) < 0.02) ? (U(rng) < 0.5 ? -M_PI + 1e-3 : M_PI - 1e-3)
                                                 : (2 * U(rng) - 1) * M_PI;
                make_hinge(h[i].x, rng, U, L0, theta_t);

                h[i].kappa = std::pow(10.0, -6.0 + 6.0 * U(rng));
                h[i].L0    = L0;
                h[i].h_bar = std::pow(10.0, -4.0 + 2.0 * U(rng));
                double rest_tb = U(rng) < 0.5 ? 0.0 : (U(rng) - 0.5) * 1.0;
                double slope = 2.0 * h[i].kappa * h[i].L0 / h[i].h_bar;
                // sample the yield stress through its observable, theta_y, so
                // both branches get dense coverage for every parameter draw
                double theta_y0 = std::pow(10.0, -3.0 + 3.3 * U(rng));
                double ys0 = theta_y0 * slope;
                if(U(rng) < 0.05)
                    ys0 = 0.0;  // exact zero: perfect plasticity, zero stress
                double Hmod = (U(rng) < 0.5) ? 0.0 : std::pow(10.0, -2.0 + 3.0 * U(rng));

                int cls;
                double cr = U(rng);
                double theta_bar = rest_tb, ys = ys0;
                if(cr < 0.45)
                {
                    cls = 0;  // R: chain the REAL stress commit
                    double tb = rest_tb;
                    int K = 1 + (int)(U(rng) * 8);
                    for(int s = 0; s < K; ++s)
                    {
                        double step = (U(rng) < 0.1 ? 2.0 : std::pow(10.0, -3.0 + 2.5 * U(rng)))
                                      * (U(rng) < 0.5 ? -1.0 : 1.0);
                        double tgt = tb + step;
                        if(!SPDSB::update_plastic_state<double>(
                               tgt, tb, ys, Hmod, h[i].kappa, h[i].L0, h[i].h_bar))
                            break;
                    }
                    theta_bar = tb;
                }
                else if(cr < 0.80)
                {
                    cls = 1;  // B: committed state anywhere a chain can walk it
                    theta_bar = (2 * U(rng) - 1) * M_PI;
                    ys = std::pow(10.0, -3.0 + 4.0 * U(rng)) * slope;  // theta_y up to ~10 rad
                    if(U(rng) < 0.05)
                        ys = 0.0;
                }
                else if(cr < 0.90)
                {
                    cls = 2;  // Z: fresh
                }
                else
                {
                    cls = 3;  // X: out-of-range imports
                    double pick = U(rng);
                    if(pick < 0.5)
                        h[i].kappa = -std::pow(10.0, -6.0 + 6.0 * U(rng));  // negative stiffness
                    else
                        ys = -std::pow(10.0, -3.0 + 3.0 * U(rng)) * slope;  // negative yield
                }
                h[i].cls        = cls;
                h[i].theta_bar  = theta_bar;
                h[i].yield_stress = ys;
            }
            cudaMemcpy(d_in, h.data(), n * sizeof(StressInput), cudaMemcpyHostToDevice);

            std::vector<double> out((size_t)n * STRIDE);
            stress_extract<<<64, 256>>>(d_in, d_out, n);
            cudaMemcpy(out.data(), d_out, (size_t)n * STRIDE * sizeof(double),
                       cudaMemcpyDeviceToHost);

            for(int i = 0; i < n; ++i)
            {
                const double* o = &out[(size_t)i * STRIDE];
                int cls = (int)o[449];
                S.n_cls[cls]++;
                if(o[450] < 0.5)
                {
                    S.n_guard++;  // response's own guards killed the hinge
                    continue;
                }
                S.n_resp_ok++;
                double theta = o[444], delta = o[445];
                double ddEdd = o[446], dEdt = o[447];
                if(o[451] > 0.5)
                    S.n_yield0++;

                Eigen::MatrixXd Hfull = as_mat(o, 0);
                Eigen::MatrixXd hess = as_mat(o, 144);
                Eigen::VectorXd g(12);
                for(int k = 0; k < 12; ++k)
                    g(k) = o[288 + k];
                Eigen::MatrixXd Hgn = as_mat(o, 300);
                Eigen::MatrixXd Hproj = as_mat(o, 452);

                double ffull = Hfull.norm();
                if(ffull < 1e-300)
                {
                    S.n_zeroH++;
                    continue;
                }
                S.n_used++;

                double slope = 2.0 * std::abs(h[i].kappa) * h[i].L0 / h[i].h_bar;
                double theta_y = h[i].yield_stress / slope;
                if(cls == 3)
                {
                    if(ddEdd < 0)
                        S.n_neg_out++;
                    S.min_dd_out = std::min(S.min_dd_out, ddEdd);
                    if(ddEdd < 0)
                        S.n_clamp_out++;
                }
                else
                {
                    if(ddEdd < 0)
                        S.n_neg_reach++;
                    double mrel = ddEdd / slope;
                    if(mrel < S.min_margin_rel)
                    {
                        S.min_margin_rel = mrel;
                        S.argmin_ctx[0]  = std::abs(delta) / std::max(theta_y, 1e-300);
                        S.argmin_ctx[1]  = ddEdd;
                        S.argmin_ctx[2]  = mrel;
                        S.argmin_ctx[3]  = theta;
                    }
                    S.min_dd_reach = std::min(S.min_dd_reach, ddEdd);
                    if(ddEdd < 0)
                        S.n_clamp_reach++;
                }

                Eigen::MatrixXd split = ddEdd * g * g.transpose() + dEdt * hess;
                S.max_split_resid = std::max(S.max_split_resid, (Hfull - split).norm() / ffull);

                if(Hgn.norm() > 1e-300)
                {
                    S.max_asym = std::max(S.max_asym, (Hgn - Hgn.transpose()).norm() / Hgn.norm());
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hgn);
                    double min_e = es.eigenvalues().minCoeff();
                    S.min_eig_gn = std::min(S.min_eig_gn, min_e / Hgn.norm());
                    for(int k = 0; k < 3; ++k)
                    {
                        Eigen::VectorXd t = Eigen::VectorXd::Zero(12);
                        for(int a = 0; a < 4; ++a)
                            t(3 * a + k) = 1.0;
                        S.max_null_gn = std::max(S.max_null_gn, (Hgn * t).norm() / Hgn.norm());
                    }
                }

                double rel_drop = (dEdt * hess).norm() / ffull;
                S.drop_all.push_back(rel_drop);
                if(Hproj.norm() > 1e-300)
                    S.rel_ab.push_back((Hproj - Hgn).norm() / Hproj.norm());
                double r = std::abs(delta) / std::max(theta_y, 1e-300);
                if(r <= 1.0) { S.drop_fresh.push_back(rel_drop); S.n_reg[0]++; }      // elastic
                else if(r < 2.0) { S.drop_mid.push_back(rel_drop); S.n_reg[1]++; }    // near yield
                else { S.drop_far.push_back(rel_drop); S.n_reg[2]++; }                // deep yield
            }
        }

        auto q = [](std::vector<double>& v, double p)
        {
            if(v.empty()) return -1.0;
            std::sort(v.begin(), v.end());
            return v[(size_t)(p * (v.size() - 1))];
        };
        auto md = [](std::vector<double>& v) { return v.empty() ? -1.0 : v[v.size() / 2]; };

        printf("=== STRESS-plastic GN premise N=%d used=%ld zeroH=%ld respGuardKilled=%ld\n",
               total, S.n_used, S.n_zeroH, S.n_guard);
        printf("classes: R(chained-commit) %ld  B(direct) %ld  Z(fresh) %ld  X(out-of-range) %ld\n",
               S.n_cls[0], S.n_cls[1], S.n_cls[2], S.n_cls[3]);
        printf("branch regime: elastic(|del|<=th_y) %ld  near-yield %ld  deep-yield %ld   yielded(ddEdd==0) %ld / resp-ok %ld\n",
               S.n_reg[0], S.n_reg[1], S.n_reg[2], S.n_yield0, S.n_resp_ok);
        printf("[1] ddEddtheta<0 on REACHABLE (R+B+Z): %ld   min ddEddtheta %.6e   min margin ddEdd/(2*k*L0/h) %.6e\n",
               S.n_neg_reach, S.min_dd_reach, S.min_margin_rel);
        printf("[1] argmin sample: |del|/th_y %.3f  ddEdd %.3e  margin %.3e  theta %.3f\n",
               S.argmin_ctx[0], S.argmin_ctx[1], S.argmin_ctx[2], S.argmin_ctx[3]);
        printf("[1] clamp activations reachable %ld   out-of-range %ld / %ld (min ddEddtheta %.6e)\n",
               S.n_clamp_reach, S.n_clamp_out, S.n_cls[3], S.min_dd_out);
        printf("[2] ||H_full-(ddEdd*g*g'+dEdt*hess)||/||H_full|| max %.3e\n", S.max_split_resid);
        printf("[3] GN fill: max asym %.3e   min eig/||H|| %.3e   max ||Hgn*t||/||Hgn|| %.3e\n",
               S.max_asym, S.min_eig_gn, S.max_null_gn);
        printf("[4] ||dEdt*hess||/||H_full||: med %.4f p99 %.4f max %.4f | elastic %.4f near-yield %.4f deep-yield %.4f\n",
               md(S.drop_all), q(S.drop_all, 0.99), q(S.drop_all, 1.0),
               md(S.drop_fresh), md(S.drop_mid), md(S.drop_far));
        printf("[4] relFro(old shipped projected <1,1>, GN): med %.4f p1 %.4f p99 %.4f max %.4f  [THE A/B HESSIAN DELTA]\n",
               md(S.rel_ab), q(S.rel_ab, 0.01), q(S.rel_ab, 0.99), q(S.rel_ab, 1.0));
        cudaFree(d_in);
        cudaFree(d_out);
    }
    return 0;
}

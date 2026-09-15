// Round-7 s03 math verifier: is the dahl-friction hinge's ddEddtheta >= 0 on
// every REACHABLE state, so that the Gauss-Newton Hessian
// H_gn = ddEddtheta * grad(theta) grad(theta)^T is PSD by construction?
//
// The claim to prove on device (not argue):
//   P(theta)     = kappa*w*del^2 + W(d),  w = L0/h_bar
//   del          = wrap(theta - theta_bar),  d = wrap(theta - theta_commit)
//   ddEddtheta   = 2*kappa*w + ddWdd,
//   ddWdd        = ((M_e - s*F_commit)/ell_e) * exp(-|d|/ell_e)   [friction_response]
//
// Reachable-state sampling of the committed frame constants:
//   R (45%): F_commit/theta_commit produced by CHAINING the real
//            commit_friction_state() -- the exact mechanism the
//            TimeIntegrator's do_update_state kernel runs once per accepted
//            frame (convex combination of F and s*M_e, then the [-M_e, M_e]
//            clamp). This is the ground truth for "what can actually be
//            produced", not an assumed box.
//   B (35%): F_commit uniform in [-M_e, M_e] directly (1/3 of those at the
//            exact boundary +-M_e), theta_commit offsets +-pi.
//   Z (10%): fresh hinge, F_commit = 0, theta_commit = theta_bar.
//   X (10%): OUT-OF-BOX F_commit = +-(1.05..3)*M_e. Not reachable through the
//            commit kernel; reachable through the imported-history path
//            (dahl_friction_commit edge attribute), which asserts only
//            isfinite. Documented separately -- this sizes the hazard the
//            max(ddEddtheta, 0) clamp in the shipped GN fill guards.
//
// theta itself is placed exactly (rotate one wing around the middle edge), so
// d = wrap(theta - theta_commit) covers +-pi by construction.
//
// Checks (all against the REAL device functions):
//   1. ddEddtheta >= 0 on every R/B/Z sample; margin statistics.
//   2. The split is what the shipped exact path assembles:
//      H_full == ddEddtheta*g*g^T + dEdtheta*hess(theta) to rounding.
//   3. The planned GN fill (mirrored rank-1, clamp, scale folded in):
//      exactly symmetric, min-eig/norm >= 0, H_gn t = 0.
//   4. Size of the dropped dEdtheta*hess(theta) term vs ||H_full||F -- the
//      controlled perturbation GN introduces -- binned by friction regime.
//   5. Out-of-box class X: how negative can ddEddtheta get, and how often.
//
// Build: see build_and_run.sh (s02's include set/flags).

#include <finite_element/constitutions/dahl_friction_discrete_shell_bending_function.h>
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

namespace DFDSB = sym::dahl_friction_discrete_shell_bending;

struct HingeInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar, M_e, ell_e, theta_commit, F_commit;
    int    cls;  // 0=R chained, 1=B in-box, 2=Z fresh, 3=X out-of-box
};

constexpr int STRIDE = 596;  // 144 H_full + 144 hess + 12 g + 144 H_gn + 144 H_proj + 8 scalars

// Extracts everything through the real device functions and fills the planned
// GN matrix with the exact loop the shipped dEdx_ddEddx_gauss_newton will use.
__global__ void extract_kernel(const HingeInput* __restrict__ in,
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

    double* o = out + (size_t)I * STRIDE;

    Vector12    G;   // real fused G+H, the shipped exact path
    Matrix12x12 H;
    DFDSB::dEdx_ddEddx(G, H, x0, x1, x2, x3, h.L0, h.h_bar, h.theta_bar, h.kappa,
                       h.M_e, h.ell_e, h.theta_commit, h.F_commit);
    for(int r = 0; r < 12; ++r)
        for(int c = 0; c < 12; ++c)
            o[r * 12 + c] = H(r, c);

    // the shipped OLD arm's Hessian: same H through the blocked 9x9 projection
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

    // ---- scalars: theta, d, friction response, the extracted coefficients
    Float theta = 0.0;
    bool  okang = DFDSB::safe_dihedral_angle(x0, x1, x2, x3, theta);
    Float W = 0, dWdd = 0, ddWdd = 0;
    Float d = okang ? DFDSB::angle_delta(theta, h.theta_commit) : 0.0;
    bool  okfr = okang && DFDSB::friction_response(d, h.F_commit, h.M_e, h.ell_e, W, dWdd, ddWdd);

    const Float w          = h.L0 / h.h_bar;
    const Float del        = okang ? DFDSB::angle_delta(theta, h.theta_bar) : 0.0;
    const Float dEdtheta   = 2.0 * h.kappa * w * del + dWdd;
    const Float ddEddtheta = 2.0 * h.kappa * w + ddWdd;

    // ---- the planned shipped GN fill (mirrored, scale folded, clamped)
    const Float scale = 1.0;
    Float       c     = DFDSB::max_value(ddEddtheta, Float(0)) * scale;
    Matrix12x12 Hgn;
#pragma unroll
    for(int i = 0; i < 12; ++i)
    {
        const Float ci = c * g(i);
        Hgn(i, i)      = ci * g(i);
#pragma unroll
        for(int j = i + 1; j < 12; ++j)
        {
            const Float v = ci * g(j);
            Hgn(i, j)     = v;
            Hgn(j, i)     = v;
        }
    }
    for(int r = 0; r < 12; ++r)
        for(int cc = 0; cc < 12; ++cc)
            o[300 + r * 12 + cc] = Hgn(r, cc);

    o[444] = okang ? theta : 999.0;
    o[445] = okfr ? d : 999.0;
    o[446] = okfr ? ddEddtheta : 999.0;
    o[447] = okfr ? dEdtheta : 999.0;
    o[448] = (okfr && ddEddtheta < Float(0)) ? 1.0 : 0.0;  // clamp activation marker
    o[449] = h.cls;
    o[450] = okfr ? ddWdd : 999.0;
    o[451] = (okang && okfr) ? 1.0 : 0.0;
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

int main(int argc, char** argv)
{
    const int      total  = argc > 1 ? atoi(argv[1]) : 200000;
    const int      batch  = 16384;
    const unsigned seed0  = argc > 2 ? atoi(argv[2]) : 7;

    // reachability classes
    long n_cls[4] = { 0, 0, 0, 0 };
    // the premise
    long   n_neg_reach = 0, n_neg_outbox = 0, n_clamp_reach = 0, n_clamp_outbox = 0;
    double min_dd_reach = 1e300, min_dd_outbox = 1e300;
    double min_margin_rel = 1e300;  // ddEddtheta / (2*kappa*w + M_e/ell_e)
    // decomposition + GN fill
    double max_split_resid = 0, max_asym = 0, min_eig_gn = 1e300, max_null_gn = 0;
    // dropped-term size
    std::vector<double> drop_all, drop_fresh, drop_mid, drop_sat, drop_bigdef;
    std::vector<double> rel_ab;  // relFro(old projected <1,1>, GN) -- the A/B delta
    long                n_used = 0, n_zeroH = 0, n_guard = 0;
    long                n_reg[3] = { 0, 0, 0 };
    double              argmin_ctx[5] = { 0, 0, 0, 0, 0 };  // x, |F|/Me, 2kw/(2kw+Me/ell), ddEdd, relmargin

    HingeInput* d_in;
    double*     d_out;
    cudaMalloc(&d_in, batch * sizeof(HingeInput));
    cudaMalloc(&d_out, (size_t)batch * STRIDE * sizeof(double));

    for(int done = 0; done < total; done += batch)
    {
        int n = std::min(batch, total - done);
        std::mt19937_64             rng(seed0 + done);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<HingeInput> h(n);
        for(int i = 0; i < n; ++i)
        {
            // ---- geometry with exact placement of theta
            double L0 = 0.014 + U(rng) * 0.036;
            double th = U(rng) * 2 * M_PI, ph = std::acos(2 * U(rng) - 1);
            Eigen::Vector3d dir(std::sin(ph) * std::cos(th), std::sin(ph) * std::cos(th),
                                std::cos(ph));
            dir.normalize();
            Eigen::Vector3d any(0, 0, 1);
            if(std::abs(dir.z()) > 0.9)
                any = Eigen::Vector3d(1, 0, 0);
            Eigen::Vector3d u = dir.cross(any).normalized();
            Eigen::Vector3d w = dir.cross(u);
            Eigen::Vector3d mid = 0.5 * L0 * dir;

            // target dihedral across +-pi (tiny tails at the fold endpoints)
            double theta_t = (U(rng) < 0.02) ? (U(rng) < 0.5 ? -M_PI + 1e-3 : M_PI - 1e-3)
                                             : (2 * U(rng) - 1) * M_PI;
            double h0 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
            double h3 = (U(rng) < 0.02 ? 1e-4 : 0.004 + U(rng) * 0.026);
            double lat0 = (U(rng) - 0.5) * 0.02, lat3 = (U(rng) - 0.5) * 0.02;
            Eigen::Vector3d x0 = lat0 * dir + h0 * u;
            Eigen::Vector3d x3f = lat3 * dir - h3 * u;  // theta = 0 side
            Eigen::Vector3d x3 = rot(x3f, dir, theta_t);
            Eigen::Vector3d x1 = mid, x2 = -mid;
            for(int k = 0; k < 3; ++k)
            {
                h[i].x[0][k] = x0(k);
                h[i].x[1][k] = x1(k);
                h[i].x[2][k] = x2(k);
                h[i].x[3][k] = x3(k);
            }

            // ---- material parameters (the scene's ranges and +-decades)
            h[i].kappa     = std::pow(10.0, -6.0 + 6.0 * U(rng));
            h[i].L0        = L0;
            h[i].h_bar     = 0.0008 + U(rng) * 0.003;
            h[i].theta_bar = U(rng) < 0.5 ? 0.0 : (U(rng) - 0.5) * 3.0;
            h[i].ell_e     = std::pow(10.0, -1.5 + 2.0 * U(rng));
            double m_hat   = std::pow(10.0, -5.0 + 4.0 * U(rng));
            h[i].M_e       = m_hat * L0;

            // ---- committed state: pick the class, then theta_commit from a
            // target friction increment d_t so d covers +-pi by construction
            double d_t = (2 * U(rng) - 1) * M_PI;
            double theta_commit = theta_t - d_t;  // wrapped by angle_delta later
            double F            = 0.0;
            int    cls;
            double cr = U(rng);
            if(cr < 0.45)
            {
                cls = 0;  // R: chain the REAL commit function, fresh -> K steps
                int K = 1 + (int)(U(rng) * 8);
                F = 0.0;
                double tc = h[i].theta_bar;  // fresh cloth commits the rest angle
                for(int s = 0; s < K; ++s)
                {
                    double step = (U(rng) < 0.1 ? 2.0 : std::pow(10.0, -3.0 + 2.5 * U(rng)))
                                  * (U(rng) < 0.5 ? -1.0 : 1.0);
                    double tgt = tc + step;  // the frame's final angle
                    double dd  = DFDSB::angle_delta(tgt, tc);
                    double Fm;
                    if(!DFDSB::commit_friction_state(dd, F, h[i].M_e, h[i].ell_e, Fm))
                        break;
                    F  = Fm;
                    tc = tgt;
                }
                theta_commit = theta_t - d_t;  // current-frame offset independent
            }
            else if(cr < 0.80)
            {
                cls = 1;  // B: in-box direct, 1/3 at the exact boundary
                if(U(rng) < (1.0 / 3.0))
                    F = (U(rng) < 0.5 ? -1.0 : 1.0) * h[i].M_e;
                else
                    F = (2 * U(rng) - 1) * h[i].M_e;
            }
            else if(cr < 0.90)
            {
                cls = 2;  // Z: fresh
                F = 0.0;
                theta_commit = h[i].theta_bar;
            }
            else
            {
                cls = 3;  // X: out-of-box import
                F = (1.05 + 1.95 * U(rng)) * (U(rng) < 0.5 ? -1.0 : 1.0) * h[i].M_e;
            }
            h[i].cls         = cls;
            h[i].theta_commit = theta_commit;
            h[i].F_commit     = F;
        }
        cudaMemcpy(d_in, h.data(), n * sizeof(HingeInput), cudaMemcpyHostToDevice);

        std::vector<double> out((size_t)n * STRIDE);
        extract_kernel<<<64, 256>>>(d_in, d_out, n);
        cudaMemcpy(out.data(), d_out, (size_t)n * STRIDE * sizeof(double),
                   cudaMemcpyDeviceToHost);

        for(int i = 0; i < n; ++i)
        {
            const double* o = &out[(size_t)i * STRIDE];
            int           cls = (int)o[449];
            n_cls[cls]++;
            if(o[451] < 0.5)
            {
                n_guard++;
                continue;
            }
            double theta = o[444], d = o[445];
            double ddEdd = o[446], dEdt = o[447];

            Eigen::MatrixXd Hfull = as_mat(o, 0);
            Eigen::MatrixXd hess  = as_mat(o, 144);
            Eigen::VectorXd g(12);
            for(int k = 0; k < 12; ++k)
                g(k) = o[288 + k];
            Eigen::MatrixXd Hgn = as_mat(o, 300);
            Eigen::MatrixXd Hproj = as_mat(o, 452);  // the old arm's shipped Hessian

            double ffull = Hfull.norm();
            if(ffull < 1e-300)
            {
                n_zeroH++;
                continue;
            }
            n_used++;

            // ---- 1. the premise: ddEddtheta >= 0
            double scale_ref = 2.0 * h[i].kappa * (h[i].L0 / h[i].h_bar) + h[i].M_e / h[i].ell_e;
            if(cls == 3)
            {
                if(ddEdd < 0)
                    n_neg_outbox++;
                min_dd_outbox = std::min(min_dd_outbox, ddEdd);
                if(ddEdd < 0)
                    n_clamp_outbox++;
            }
            else
            {
                if(ddEdd < 0)
                    n_neg_reach++;
                double mrel = ddEdd / scale_ref;
                if(mrel < min_margin_rel)
                {
                    min_margin_rel = mrel;
                    argmin_ctx[0]  = std::abs(d) / h[i].ell_e;
                    argmin_ctx[1]  = std::abs(h[i].F_commit) / h[i].M_e;
                    argmin_ctx[2]  = 2.0 * h[i].kappa * (h[i].L0 / h[i].h_bar) / scale_ref;
                    argmin_ctx[3]  = ddEdd;
                    argmin_ctx[4]  = mrel;
                }
                min_dd_reach = std::min(min_dd_reach, ddEdd);
                if(ddEdd < 0)
                    n_clamp_reach++;
            }

            // ---- 2. the split the exact path assembles
            Eigen::MatrixXd split = ddEdd * g * g.transpose() + dEdt * hess;
            max_split_resid = std::max(max_split_resid, (Hfull - split).norm() / ffull);

            // ---- 3. the planned GN fill
            max_asym = std::max(max_asym, (Hgn - Hgn.transpose()).norm() / Hgn.norm());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hgn);
            double                                          min_e = es.eigenvalues().minCoeff();
            if(Hgn.norm() > 1e-300)
                min_eig_gn = std::min(min_eig_gn, min_e / Hgn.norm());
            for(int k = 0; k < 3; ++k)
            {
                Eigen::VectorXd t = Eigen::VectorXd::Zero(12);
                for(int a = 0; a < 4; ++a)
                    t(3 * a + k) = 1.0;
                if(Hgn.norm() > 1e-300)
                    max_null_gn = std::max(max_null_gn, (Hgn * t).norm() / Hgn.norm());
            }

            // ---- 4. dropped-term size, binned by friction regime
            double rel_drop = (dEdt * hess).norm() / ffull;
            drop_all.push_back(rel_drop);
            if(Hproj.norm() > 1e-300)
                rel_ab.push_back((Hproj - Hgn).norm() / Hproj.norm());
            double x = std::abs(d) / h[i].ell_e;
            if(x < 0.1) { drop_fresh.push_back(rel_drop); n_reg[0]++; }
            else if(x < 10.0) { drop_mid.push_back(rel_drop); n_reg[1]++; }
            else { drop_sat.push_back(rel_drop); n_reg[2]++; }
            double bigdef = std::abs(theta - h[i].theta_bar);
            if(bigdef > 0.5)
                drop_bigdef.push_back(rel_drop);
        }
    }

    auto q = [](std::vector<double>& v, double p)
    {
        if(v.empty())
            return -1.0;
        std::sort(v.begin(), v.end());
        return v[(size_t)(p * (v.size() - 1))];
    };
    auto md = [](std::vector<double>& v) { return v.empty() ? -1.0 : v[v.size() / 2]; };

    printf("=== dahl GN premise N=%d used=%ld zeroH=%ld guardKilled=%ld\n", total, n_used,
           n_zeroH, n_guard);
    printf("classes: R(chained-commit) %ld  B(in-box) %ld  Z(fresh) %ld  X(out-of-box) %ld\n",
           n_cls[0], n_cls[1], n_cls[2], n_cls[3]);
    printf("friction regime: |d|/ell<0.1 %ld  0.1..10 %ld  >10 %ld\n", n_reg[0], n_reg[1],
           n_reg[2]);
    printf("[1] ddEddtheta<0 on REACHABLE (R+B+Z): %ld   min ddEddtheta %.6e   min margin ddEdd/(2kw+Me/ell) %.6e\n",
           n_neg_reach, min_dd_reach, min_margin_rel);
    printf("[1] argmin sample: |d|/ell %.3f  |F|/M_e %.3f  elastic share of scale %.3f  ddEdd %.3e  margin %.3e\n",
           argmin_ctx[0], argmin_ctx[1], argmin_ctx[2], argmin_ctx[3], argmin_ctx[4]);
    printf("[1] clamp activations reachable %ld   out-of-box %ld / %ld (min ddEddtheta %.6e)\n",
           n_clamp_reach, n_clamp_outbox, n_cls[3], min_dd_outbox);
    printf("[2] ||H_full-(ddEdd*g*g'+dEdt*hess)||/||H_full|| max %.3e\n", max_split_resid);
    printf("[3] GN fill: max asym %.3e   min eig/||H|| %.3e   max ||Hgn*t||/||Hgn|| %.3e\n",
           max_asym, min_eig_gn, max_null_gn);
    printf("[4] ||dEdt*hess||/||H_full||: med %.4f p99 %.4f max %.4f | fresh %.4f mid %.4f sat %.4f | |del|>0.5rad %.4f\n",
           md(drop_all), q(drop_all, 0.99), q(drop_all, 1.0), md(drop_fresh), md(drop_mid),
           md(drop_sat), md(drop_bigdef));
    printf("[4] relFro(old shipped projected <1,1>, GN): med %.4f p1 %.4f p99 %.4f max %.4f  [THE A/B HESSIAN DELTA]\n",
           md(rel_ab), q(rel_ab, 0.01), q(rel_ab, 0.99), q(rel_ab, 1.0));
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

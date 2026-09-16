// Round-7 s08 numerics verifier: old vs new K16 block assembly inside the
// translation-free 9x9 PSD projection of the two plastic discrete-shell
// bending Hessians, on device, against the real device functions.
//
// Arms (helper = make_spd_translation_free_4x3_blocked<Solver, SymAsm>):
//   v0 old      = <1, 0>  full-triangle assembly + QL   -- main's shipped path
//   v1 new      = <1, 2>  dead-triangle cut + mirrored back + QL -- shipped
//   v2 cutonly  = <1, 1>  forward cut only, old back    -- the bit-identity arm
//   v3 newE     = <0, 2>  cut+mirror + Eigen
//   v4 oldE     = <0, 0>  full assembly + Eigen         -- the old path's own
//                                                            solver sensitivity
//   v5 raw      = unprojected ddEddx
//
// Checks over randomised hinge inputs (s01's generator verbatim):
//   1. v2 == v0 bitwise, all 144 words, every hinge (the forward cut deletes
//      only solver-dead entries -- the NaN-poison proof says the solvers read
//      the lower triangle only).
//   2. v1 vs v0: the upper 10 blocks bit-identical (same expressions, same
//      order); the lower 6 mirrored blocks at reassociation level, bounded by
//      the old path's own solver-swap sensitivity relFro(v0, v4).
//   3. min eigenvalue of v1 >= 0 (to rounding); translation null space kept.
#include <utils/make_spd.h>
#include <finite_element/constitutions/strain_plastic_discrete_shell_bending_function.h>
#include <finite_element/constitutions/stress_plastic_discrete_shell_bending_function.h>
#include <Eigen/Eigenvalues>
#include <random>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;

struct HingeInput
{
    double x[4][3];
    double kappa, L0, h_bar, theta_bar, yield_p, Vdt2;
};

// Model 0 = strain-plastic, 1 = stress-plastic. Proj selects the helper arm.
template <int Model, int Proj, int Solver, int SymAsm>
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
        make_spd_translation_free_4x3_blocked<Solver, 0>(H);   // old assembly
    else if constexpr(Proj == 2)
        make_spd_translation_free_4x3_blocked<Solver, 1>(H);   // cut only
    else if constexpr(Proj == 3)
        make_spd_translation_free_4x3_blocked<Solver, 2>(H);   // cut + mirror
    else if constexpr(Proj == 9)
        ;                                                       // raw
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
    const int       total  = argc > 1 ? atoi(argv[1]) : 200000;
    const int       batch  = 16384;
    const unsigned  seed0  = argc > 2 ? atoi(argv[2]) : 7;
    const int       models = argc > 3 ? atoi(argv[3]) : 2;

    for(int model = 0; model < models; ++model)
    {
        RelStats rel_new_lower, rel_old_solver, rel_newE_lower;
        double   min_eig_new = 1e300, min_eig_new_rel = 1e300;
        double   max_null_new = 0;
        long     n_used = 0;
        long     bit_cutonly_bad = 0;   // v2 vs v0, all 144 words
        long     bit_upper_bad = 0;     // v1 vs v0, upper 10 blocks (90 words)
        long     words_checked_upper = 0, words_checked_full = 0;

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
                // s01's generator, verbatim
                double L0 = 0.014 + U(rng) * 0.036;
                double th = U(rng) * 2 * M_PI, ph = std::acos(2 * U(rng) - 1);
                Eigen::Vector3d dir(std::sin(ph) * std::cos(th),
                                    std::sin(ph) * std::sin(th), std::cos(ph));
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
                h[i].L0    = L0;
                h[i].h_bar = 0.0008 + U(rng) * 0.003;
                h[i].theta_bar = U(rng) < 0.5 ? 0.0 : (U(rng) - 0.5) * 3.0;
                double theta_y = std::pow(10.0, -3.0 + 2.8 * U(rng));
                double slope   = 2.0 * h[i].kappa * L0 / h[i].h_bar;
                h[i].yield_p   = theta_y * slope;
                h[i].Vdt2      = std::pow(10.0, -9.0 + 3.0 * U(rng));
            }
            cudaMemcpy(d_in, h.data(), n * sizeof(HingeInput), cudaMemcpyHostToDevice);

            // v: 0=old 1=new 2=cutonly 3=newE 4=oldE
            std::vector<std::vector<double>> outs(5);
            for(int v = 0; v < 5; ++v)
            {
                if(model == 0)
                {
                    if(v == 0) project_kernel<0, 1, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 1) project_kernel<0, 3, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 2) project_kernel<0, 2, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 3) project_kernel<0, 3, 0, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 4) project_kernel<0, 1, 0, 0><<<64, 256>>>(d_in, d_out, n);
                }
                else
                {
                    if(v == 0) project_kernel<1, 1, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 1) project_kernel<1, 3, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 2) project_kernel<1, 2, 1, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 3) project_kernel<1, 3, 0, 0><<<64, 256>>>(d_in, d_out, n);
                    if(v == 4) project_kernel<1, 1, 0, 0><<<64, 256>>>(d_in, d_out, n);
                }
                outs[v].resize((size_t)n * 144);
                cudaMemcpy(outs[v].data(), d_out,
                           (size_t)n * 144 * sizeof(double), cudaMemcpyDeviceToHost);
            }

            for(int i = 0; i < n; ++i)
            {
                const double* p0  = &outs[0][(size_t)i * 144];
                const double* p1  = &outs[1][(size_t)i * 144];
                const double* p2  = &outs[2][(size_t)i * 144];
                const double* p3  = &outs[3][(size_t)i * 144];
                const double* p4  = &outs[4][(size_t)i * 144];
                Eigen::MatrixXd H0 = as_mat(p0), H1 = as_mat(p1), H4 = as_mat(p4), H3 = as_mat(p3);
                double f0 = H0.norm();
                if(f0 < 1e-300)
                    continue;
                n_used++;

                // 1. cut-only arm vs old: full bit-identity
                for(int w = 0; w < 144; ++w)
                {
                    words_checked_full++;
                    if(std::memcmp(&p2[w], &p0[w], 8) != 0)
                        bit_cutonly_bad++;
                }
                // 2. new vs old: upper block-triangle bit-identity
                for(int a = 0; a < 4; ++a)
                    for(int b = a; b < 4; ++b)
                        for(int r = 0; r < 3; ++r)
                            for(int c = 0; c < 3; ++c)
                            {
                                int w = (3 * a + r) * 12 + (3 * b + c);
                                words_checked_upper++;
                                if(std::memcmp(&p1[w], &p0[w], 8) != 0)
                                    bit_upper_bad++;
                            }
                // lower blocks: relFro of the 6 mirrored blocks, vs the old
                // path's own solver swap and the Eigen-solver variant
                {
                    Eigen::MatrixXd D1 = H1 - H0, D4 = H4 - H0, D3 = H3 - H4;
                    rel_new_lower.add(D1.norm() / f0);
                    rel_old_solver.add(D4.norm() / f0);
                    rel_newE_lower.add(D3.norm() / H4.norm());
                }
                // 3. min eig + null space of the new arm
                {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H1);
                    min_eig_new     = std::min(min_eig_new, es.eigenvalues().minCoeff());
                    min_eig_new_rel = std::min(min_eig_new_rel, es.eigenvalues().minCoeff() / f0);
                    for(int k = 0; k < 3; ++k)
                    {
                        Eigen::VectorXd t = Eigen::VectorXd::Zero(12);
                        for(int a = 0; a < 4; ++a)
                            t(3 * a + k) = 1.0;
                        max_null_new = std::max(max_null_new, (H1 * t).norm() / f0);
                    }
                }
            }
        }
        rel_new_lower.finish();
        rel_old_solver.finish();
        rel_newE_lower.finish();
        printf("== %s: %ld hinges used ==\n", kModelName[model], n_used);
        printf("  [bit-identity] cut-only vs old: %ld / %ld words differ (expect 0)\n",
               bit_cutonly_bad, words_checked_full);
        printf("  [bit-identity] new upper-triangle vs old: %ld / %ld words differ (expect 0)\n",
               bit_upper_bad, words_checked_upper);
        printf("  [rounding] mirrored lower blocks relFro(new,old): med %.3e p999 %.3e max %.3e\n",
               rel_new_lower.med, rel_new_lower.p999, rel_new_lower.max);
        printf("  [bound] old path own solver-swap relFro(QL,Eigen): med %.3e p999 %.3e max %.3e\n",
               rel_old_solver.med, rel_old_solver.p999, rel_old_solver.max);
        printf("  [rounding] under Eigen solver relFro(newE,oldE): med %.3e p999 %.3e max %.3e\n",
               rel_newE_lower.med, rel_newE_lower.p999, rel_newE_lower.max);
        printf("  [projection] min eig(new) %.3e (rel %.3e); null space max %.3e\n",
               min_eig_new, min_eig_new_rel, max_null_new);
    }
    return 0;
}

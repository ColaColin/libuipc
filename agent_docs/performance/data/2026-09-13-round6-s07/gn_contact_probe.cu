// s03 numerics probe -- runs the REAL device functions on the device (they are
// `__device__`-only, so a host build is not possible), one thread per sample,
// and reduces per-sample statistics on the host.
//
// What it establishes:
//  (1) the shipped Stiff-GIPC log^2 barrier has B''(D) >= 0 on (and beyond)
//      its whole active domain, and B'(D) <= 0 on it;
//  (2) the exact contact Hessian really is  B'' gradD gradD^T + B' hessD --
//      i.e. the Gauss-Newton split is exactly the first term;
//  (3) the Gauss-Newton Hessian is exactly symmetric and PSD;
//  (4) the rank the SHIPPED exact projection actually keeps, and how far the
//      rank-1 Gauss-Newton Hessian is from it.
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <Eigen/Eigenvalues>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace uipc;
using namespace uipc::backend::cuda;
namespace CN = sym::codim_ipc_simplex_contact;
namespace CC = sym::codim_ipc_contact;


template <typename M>
__device__ inline bool all_finite(const M& m)
{
    for(int i = 0; i < m.rows(); ++i)
        for(int j = 0; j < m.cols(); ++j)
            if(!isfinite(m(i, j))) return false;
    return true;
}

struct Sample
{
    double term_err;     // |(H_exact - H_gn) - B' hessD| / max|H_exact|
    double asym;         // |H_gn - H_gn^T|
    double gn_min_rel;   // min eig(H_gn) / max|eig(H_gn)|
    double full_min_rel; // min eig(H_exact) / max|eig(H_exact)|
    double proj_rank;    // rank of the shipped projected exact Hessian
    double rel;          // ||H_proj - H_gn|| / ||H_proj||
    double top_ratio;    // lambda_max(H_gn) / lambda_max(H_proj)
    double b2;           // B''  (must be >= 0)
    double b1;           // B'   (must be <= 0)
    double drop_scale;   // |B'| / (B'' * D)
    double rel2;         // ||H_proj - H_gn2|| / ||H_proj||   (corrected coefficient)
    double top_ratio2;   // lambda_max(H_gn2) / lambda_max(H_proj)
    double grad_rel;     // max|G_exact - G_gn| / max|G_exact|
    double asym2;
    double gn2_min_rel;
    int    ok;
};

__device__ inline double rnd(unsigned long long& s)
{
    s += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
__device__ inline double U(unsigned long long& s, double a, double b) { return a + (b - a) * rnd(s); }

template <int N>
__device__ void fill_stats(Sample&                              o,
                           const Eigen::Matrix<Float, N, N>&    Hfull,
                           const Eigen::Matrix<Float, N, N>&    Hgn,
                           const Eigen::Matrix<Float, N, N>&    Hproj,
                           const Eigen::Matrix<Float, N, N>&    HessD,
                           Float                                dBdD,
                           const Eigen::Matrix<Float, N, N>&    Hgn2)
{
    using M = Eigen::Matrix<Float, N, N>;
    double scale = Hfull.cwiseAbs().maxCoeff();
    if(!(scale > 0)) scale = 1.0;
    M dropped = Hfull - Hgn;
    M pred    = dBdD * HessD;
    o.term_err = (dropped - pred).cwiseAbs().maxCoeff() / scale;
    o.asym     = (Hgn - Hgn.transpose()).cwiseAbs().maxCoeff();

    Eigen::SelfAdjointEigenSolver<M> g(Hgn);
    double gmax_abs = g.eigenvalues().cwiseAbs().maxCoeff();
    o.gn_min_rel = gmax_abs > 0 ? g.eigenvalues().minCoeff() / gmax_abs : 0.0;

    Eigen::SelfAdjointEigenSolver<M> f(Hfull);
    double fmax_abs = f.eigenvalues().cwiseAbs().maxCoeff();
    o.full_min_rel = fmax_abs > 0 ? f.eigenvalues().minCoeff() / fmax_abs : 0.0;

    Eigen::SelfAdjointEigenSolver<M> p(Hproj);
    double pmax_abs = p.eigenvalues().cwiseAbs().maxCoeff();
    int    r        = 0;
    for(int i = 0; i < N; ++i)
        if(p.eigenvalues()[i] > 1e-10 * pmax_abs) ++r;
    o.proj_rank = r;

    double pn = Hproj.norm();
    o.rel     = pn > 0 ? (Hproj - Hgn).norm() / pn : 0.0;
    double pt = p.eigenvalues().maxCoeff();
    o.top_ratio = pt > 0 ? g.eigenvalues().maxCoeff() / pt : 0.0;

    o.rel2 = pn > 0 ? (Hproj - Hgn2).norm() / pn : 0.0;
    o.asym2 = (Hgn2 - Hgn2.transpose()).cwiseAbs().maxCoeff();
    Eigen::SelfAdjointEigenSolver<M> gg2(Hgn2);
    double g2max = gg2.eigenvalues().cwiseAbs().maxCoeff();
    o.gn2_min_rel = g2max > 0 ? gg2.eigenvalues().minCoeff() / g2max : 0.0;
    o.top_ratio2 = pt > 0 ? gg2.eigenvalues().maxCoeff() / pt : 0.0;
}

// ---------------------------------------------------------------- barrier
__global__ void barrier_kernel(Sample* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    unsigned long long s = 0x1234567ULL + 7919ULL * (unsigned long long)i;
    Sample o{}; o.ok = 0;
    double dhat  = pow(10.0, U(s, -4.0, -1.0));
    double kappa = pow(10.0, U(s, 0.0, 9.0));
    double xi    = (i % 2) ? dhat * U(s, 0.01, 2.0) : 0.0;
    double lo = xi * xi, hi = (dhat + xi) * (dhat + xi);
    double D;
    if(i % 10 == 0)
        D = hi * U(s, 1.0, 4.0);                 // deliberately INACTIVE
    else
        D = lo + (hi - lo) * pow(10.0, U(s, -12.0, 0.0));
    Float b1, b2;
    CC::dKappaBarrierdD(b1, (Float)kappa, (Float)D, (Float)dhat, (Float)xi);
    CC::ddKappaBarrierddD(b2, (Float)kappa, (Float)D, (Float)dhat, (Float)xi);
    if(!isfinite(b1) || !isfinite(b2)) { out[i] = o; return; }
    o.ok = (D < hi) ? 1 : 2;                      // 1 = active, 2 = inactive
    o.b1 = b1 / kappa;
    o.b2 = b2 / kappa;
    o.drop_scale = (b2 > 0) ? fabs(b1) / (b2 * D) : -1.0;
    out[i] = o;
}

// ---------------------------------------------------------------- PT / EE / PE / PP
__global__ void simplex_kernel(Sample* pt, Sample* ee, Sample* pe, Sample* pp, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    unsigned long long s = 0xABCDEF01ULL + 104729ULL * (unsigned long long)i;
    Sample opt{}, oee{}, ope{}, opp{};
    opt.ok = oee.ok = ope.ok = opp.ok = 0;

    double dhat  = pow(10.0, U(s, -4.0, -2.0));
    double kappa = pow(10.0, U(s, 2.0, 8.0));
    double xi    = (i % 3 == 0) ? dhat * U(s, 0.05, 1.0) : 0.0;
    double lo = xi * xi, hi = (dhat + xi) * (dhat + xi);

    // ---- PT
    {
        Vector3 T0, T1, T2;
        for(int k = 0; k < 3; ++k) { T0[k] = U(s,-1,1); T1[k] = U(s,-1,1); T2[k] = U(s,-1,1); }
        Vector3 nb = (T1 - T0).cross(T2 - T0);
        if(nb.norm() > 1e-6)
        {
            nb.normalize();
            double a = U(s,0.0,1.0), b = U(s,0.0,1.0);
            if(a + b > 1.0) { a = 1.0 - a; b = 1.0 - b; }
            if(i % 10 >= 7) { a = U(s,-0.6,1.4); b = U(s,-0.6,1.4); }   // outside -> other flags
            Vector3 base = T0 + a * (T1 - T0) + b * (T2 - T0);
            double  gap  = (xi + dhat) * pow(10.0, U(s,-5.0,0.0));
            Vector3 P    = base + gap * nb;
            Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);
            Float D; distance::point_triangle_distance2(flag, P, T0, T1, T2, D);
            if(D > lo && D < hi)
            {
                Vector12 G, Ggn; Matrix12x12 H, Hgn, Hproj;
                CN::PT_barrier_gradient_hessian(G, H, flag, (Float)kappa, (Float)dhat, (Float)xi, P, T0, T1, T2);
                CN::PT_barrier_gradient_hessian_gn<3>(Ggn, Hgn, flag, (Float)kappa, (Float)dhat, (Float)xi, P, T0, T1, T2);
                Matrix12x12 Hgn1; Vector12 Ggn1;
                CN::PT_barrier_gradient_hessian_gn<1>(Ggn1, Hgn1, flag, (Float)kappa, (Float)dhat, (Float)xi, P, T0, T1, T2);
                if(all_finite(H) && all_finite(Hgn))
                {
                    Hproj = H;
                    CN::PT_barrier_make_spd<1>(Hproj, flag, P, T0, T1, T2);
                    if(all_finite(Hproj))
                    {
                        Matrix12x12 HessD;
                        distance::point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);
                        Float dBdD; CC::dKappaBarrierdD(dBdD, (Float)kappa, D, (Float)dhat, (Float)xi);
                        Float ddBddD; CC::ddKappaBarrierddD(ddBddD, (Float)kappa, D, (Float)dhat, (Float)xi);
                        (void)ddBddD;
                        fill_stats<12>(opt, H, Hgn, Hproj, HessD, dBdD, Hgn1);
                        opt.b1 = (G - Ggn1).cwiseAbs().maxCoeff();   // gradient difference (must be 0)
                        opt.grad_rel = opt.b1 / (G.cwiseAbs().maxCoeff() + 1e-300);
                        opt.ok = 1;
                    }
                }
            }
        }
    }
    // ---- EE
    {
        Vector3 A0, ea, eb;
        for(int k = 0; k < 3; ++k) { A0[k] = U(s,-1,1); ea[k] = U(s,-1,1); eb[k] = U(s,-1,1); }
        Vector3 nb = ea.cross(eb);
        if(ea.norm() > 1e-6 && eb.norm() > 1e-6 && nb.norm() > 1e-9)
        {
            nb.normalize();
            Vector3 E0 = A0, E1 = A0 + ea;
            double  gap = (xi + dhat) * pow(10.0, U(s,-5.0,0.0));
            Vector3 mid = A0 + U(s,0.1,0.9) * ea + gap * nb;
            Vector3 E2 = mid - U(s,0.1,0.9) * eb, E3 = E2 + eb;
            Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);
            Float D; distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);
            if(D > lo && D < hi)
            {
                Vector12 G, Ggn; Matrix12x12 H, Hgn, Hproj;
                bool m1 = true, m2 = true;
                CN::mollified_EE_barrier_gradient_hessian<true>(
                    G, H, m1, flag, (Float)kappa, (Float)dhat, (Float)xi, E0, E1, E2, E3, E0, E1, E2, E3);
                CN::mollified_EE_barrier_gradient_hessian_gn<3>(
                    Ggn, Hgn, m2, flag, (Float)kappa, (Float)dhat, (Float)xi, E0, E1, E2, E3, E0, E1, E2, E3);
                Matrix12x12 Hgn1; Vector12 Ggn1; bool m3 = true;
                CN::mollified_EE_barrier_gradient_hessian_gn<1>(
                    Ggn1, Hgn1, m3, flag, (Float)kappa, (Float)dhat, (Float)xi, E0, E1, E2, E3, E0, E1, E2, E3);
                oee.drop_scale = (m1 == m2) ? 0.0 : 1.0;    // flags must agree
                if(m1)
                    oee.ok = 3;                              // mollified: exact path kept
                else if(all_finite(H) && all_finite(Hgn))
                {
                    Hproj = H;
                    CN::EE_barrier_make_spd<1>(Hproj, flag, E0, E1, E2, E3);
                    if(all_finite(Hproj))
                    {
                        Matrix12x12 HessD;
                        distance::edge_edge_distance2_hessian(flag, E0, E1, E2, E3, HessD);
                        Float dBdD; CC::dKappaBarrierdD(dBdD, (Float)kappa, D, (Float)dhat, (Float)xi);
                        Float ddBddD; CC::ddKappaBarrierddD(ddBddD, (Float)kappa, D, (Float)dhat, (Float)xi);
                        (void)ddBddD;
                        fill_stats<12>(oee, H, Hgn, Hproj, HessD, dBdD, Hgn1);
                        oee.b1 = (G - Ggn1).cwiseAbs().maxCoeff();
                        oee.grad_rel = oee.b1 / (G.cwiseAbs().maxCoeff() + 1e-300);
                        oee.ok = 1;
                    }
                }
            }
        }
    }
    // ---- PE
    {
        Vector3 E0, e, nb;
        for(int k = 0; k < 3; ++k) { E0[k] = U(s,-1,1); e[k] = U(s,-1,1); nb[k] = U(s,-1,1); }
        if(e.norm() > 1e-6)
        {
            Vector3 eh = e.normalized();
            nb = nb - nb.dot(eh) * eh;
            if(nb.norm() > 1e-6)
            {
                nb.normalize();
                double gap = (xi + dhat) * pow(10.0, U(s,-5.0,0.0));
                double t   = (i % 5 == 0) ? U(s,-0.5,1.5) : U(s,0.05,0.95);
                Vector3 E1 = E0 + e, P = E0 + t * e + gap * nb;
                Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);
                Float D; distance::point_edge_distance2(flag, P, E0, E1, D);
                if(D > lo && D < hi)
                {
                    Vector9 G, Ggn; Matrix9x9 H, Hgn, Hproj;
                    CN::PE_barrier_gradient_hessian(G, H, flag, (Float)kappa, (Float)dhat, (Float)xi, P, E0, E1);
                    CN::PE_barrier_gradient_hessian_gn<3>(Ggn, Hgn, flag, (Float)kappa, (Float)dhat, (Float)xi, P, E0, E1);
                    Matrix9x9 Hgn1; Vector9 Ggn1;
                    CN::PE_barrier_gradient_hessian_gn<1>(Ggn1, Hgn1, flag, (Float)kappa, (Float)dhat, (Float)xi, P, E0, E1);
                    if(all_finite(H) && all_finite(Hgn))
                    {
                        Hproj = H;
                        CN::PE_barrier_make_spd<1, 1>(Hproj, flag, P, E0, E1);
                        if(all_finite(Hproj))
                        {
                            Matrix9x9 HessD;
                            distance::point_edge_distance2_hessian(flag, P, E0, E1, HessD);
                            Float dBdD; CC::dKappaBarrierdD(dBdD, (Float)kappa, D, (Float)dhat, (Float)xi);
                            Float ddBddD; CC::ddKappaBarrierddD(ddBddD, (Float)kappa, D, (Float)dhat, (Float)xi);
                            (void)ddBddD;
                            fill_stats<9>(ope, H, Hgn, Hproj, HessD, dBdD, Hgn1);
                            ope.b1 = (G - Ggn1).cwiseAbs().maxCoeff();
                            ope.grad_rel = ope.b1 / (G.cwiseAbs().maxCoeff() + 1e-300);
                            ope.ok = 1;
                        }
                    }
                }
            }
        }
    }
    // ---- PP
    {
        Vector3 P0, d;
        for(int k = 0; k < 3; ++k) { P0[k] = U(s,-1,1); d[k] = U(s,-1,1); }
        if(d.norm() > 1e-9)
        {
            d.normalize();
            double  gap = (xi + dhat) * pow(10.0, U(s,-5.0,0.0));
            Vector3 P1  = P0 + gap * d;
            Vector2i flag = distance::point_point_distance_flag(P0, P1);
            Float D; distance::point_point_distance2(flag, P0, P1, D);
            if(D > lo && D < hi)
            {
                Vector6 G, Ggn; Matrix6x6 H, Hgn, Hproj;
                CN::PP_barrier_gradient_hessian(G, H, flag, (Float)kappa, (Float)dhat, (Float)xi, P0, P1);
                CN::PP_barrier_gradient_hessian_gn<3>(Ggn, Hgn, flag, (Float)kappa, (Float)dhat, (Float)xi, P0, P1);
                Matrix6x6 Hgn1; Vector6 Ggn1;
                CN::PP_barrier_gradient_hessian_gn<1>(Ggn1, Hgn1, flag, (Float)kappa, (Float)dhat, (Float)xi, P0, P1);
                if(all_finite(H) && all_finite(Hgn))
                {
                    Hproj = H;
                    CN::PP_barrier_make_spd<1, 1>(Hproj, flag, P0, P1);
                    if(all_finite(Hproj))
                    {
                        Matrix6x6 HessD;
                        distance::point_point_distance2_hessian(flag, P0, P1, HessD);
                        Float dBdD; CC::dKappaBarrierdD(dBdD, (Float)kappa, D, (Float)dhat, (Float)xi);
                        Float ddBddD; CC::ddKappaBarrierddD(ddBddD, (Float)kappa, D, (Float)dhat, (Float)xi);
                        (void)ddBddD;
                        fill_stats<6>(opp, H, Hgn, Hproj, HessD, dBdD, Hgn1);
                        opp.b1 = (G - Ggn1).cwiseAbs().maxCoeff();
                        opp.grad_rel = opp.b1 / (G.cwiseAbs().maxCoeff() + 1e-300);
                        opp.ok = 1;
                    }
                }
            }
        }
    }
    pt[i] = opt; ee[i] = oee; pe[i] = ope; pp[i] = opp;
}

static void report(const char* tag, int dim, const std::vector<Sample>& v)
{
    long ok = 0, indef = 0, gn_neg = 0, rank_sum = 0, grad_diff = 0, molli = 0;
    double te = 0, as = 0, mg = 0, srel = 0, mrel = 0, stop = 0, srel2 = 0, stop2 = 0, mgrad = 0;
    double mrel2 = 0, tmin2 = 1e300, tmax2 = 0, as2 = 0; long gn2_neg = 0;
    for(const auto& o : v)
    {
        if(o.ok == 3) { ++molli; continue; }
        if(o.ok != 1) continue;
        ++ok;
        te = std::max(te, o.term_err);
        as = std::max(as, o.asym);
        mg = std::min(mg, o.gn_min_rel);
        if(o.gn_min_rel < -1e-12) ++gn_neg;
        if(o.full_min_rel < -1e-12) ++indef;
        rank_sum += (long)o.proj_rank;
        srel += o.rel; mrel = std::max(mrel, o.rel);
        stop += o.top_ratio;
        srel2 += o.rel2; stop2 += o.top_ratio2;
        mrel2 = std::max(mrel2, o.rel2);
        tmin2 = std::min(tmin2, o.top_ratio2); tmax2 = std::max(tmax2, o.top_ratio2);
        as2 = std::max(as2, o.asym2);
        if(o.gn2_min_rel < -1e-12) ++gn2_neg;
        mgrad = std::max(mgrad, o.grad_rel);
        if(o.b1 != 0.0) ++grad_diff;
    }
    if(!ok) { printf("## %-18s : no usable samples (mollified %ld)\n", tag, molli); return; }
    printf("## %s : %ld usable samples\n", tag, ok);
    printf("    gradient differs (exact vs Gauss-Newton)        : %ld samples, max relative %.3e\n", grad_diff, mgrad);
    printf("    max |(H_exact - H_gn) - B' hessD| / max|H_exact|: %.3e   <- the split is exact\n", te);
    printf("    max |H_gn - H_gn^T|                             : %.3e   <- exact symmetry\n", as);
    printf("    min eig(H_gn)/max|eig(H_gn)|                    : %.3e   (samples < -1e-12: %ld)\n", mg, gn_neg);
    printf("    exact H indefinite in                           : %ld / %ld = %.1f%%\n", indef, ok, 100.0 * indef / ok);
    printf("    mean rank of the SHIPPED projected exact H       : %.2f of %d   (Gauss-Newton rank is 1)\n",
           (double)rank_sum / ok, dim);
    printf("    ||H_proj - H_gn|| / ||H_proj||   mean %.4f  max %.4f\n", srel / ok, mrel);
    printf("    mean lambda_max(H_gn)/lambda_max(H_proj)        : %.4f\n", stop / ok);
    printf("    --- SHIPPED rank-1, c = B'' + B'/(2D) (Proj=1) ---\n");
    printf("    ||H_proj - H_rank1||/||H_proj||  mean %.3e  max %.3e\n", srel2 / ok, mrel2);
    printf("    lambda_max(H_rank1)/lambda_max(H_proj)  mean %.6f  min %.6f  max %.6f\n",
           stop2 / ok, tmin2, tmax2);
    printf("    max |H_rank1 - H_rank1^T| %.3e   samples with min eig < -1e-12: %ld\n", as2, gn2_neg);
    if(molli) printf("    samples that took the MOLLIFIED branch (exact path kept): %ld\n", molli);
}

int main(int argc, char** argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 200000;
    Sample *dpt, *dee, *dpe, *dpp, *dba;
    cudaMalloc(&dpt, n * sizeof(Sample)); cudaMalloc(&dee, n * sizeof(Sample));
    cudaMalloc(&dpe, n * sizeof(Sample)); cudaMalloc(&dpp, n * sizeof(Sample));
    cudaMalloc(&dba, n * sizeof(Sample));

    barrier_kernel<<<(n + 63) / 64, 64>>>(dba, n);
    simplex_kernel<<<(n + 31) / 32, 32>>>(dpt, dee, dpe, dpp, n);
    cudaError_t e = cudaDeviceSynchronize();
    if(e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

    std::vector<Sample> hba(n), hpt(n), hee(n), hpe(n), hpp(n);
    cudaMemcpy(hba.data(), dba, n * sizeof(Sample), cudaMemcpyDeviceToHost);
    cudaMemcpy(hpt.data(), dpt, n * sizeof(Sample), cudaMemcpyDeviceToHost);
    cudaMemcpy(hee.data(), dee, n * sizeof(Sample), cudaMemcpyDeviceToHost);
    cudaMemcpy(hpe.data(), dpe, n * sizeof(Sample), cudaMemcpyDeviceToHost);
    cudaMemcpy(hpp.data(), dpp, n * sizeof(Sample), cudaMemcpyDeviceToHost);

    // ---- (1) barrier
    long act = 0, inact = 0, neg_b2_act = 0, neg_b2_inact = 0, pos_b1_act = 0;
    double min_b2_act = 1e300, min_b2_inact = 1e300, max_b1_act = -1e300, max_drop = 0;
    for(const auto& o : hba)
    {
        if(o.ok == 1) { ++act;
            min_b2_act = std::min(min_b2_act, o.b2); if(o.b2 < 0) ++neg_b2_act;
            max_b1_act = std::max(max_b1_act, o.b1); if(o.b1 > 0) ++pos_b1_act;
            if(o.drop_scale >= 0) max_drop = std::max(max_drop, o.drop_scale); }
        else if(o.ok == 2) { ++inact;
            min_b2_inact = std::min(min_b2_inact, o.b2); if(o.b2 < 0) ++neg_b2_inact; }
    }
    printf("## (1) barrier scan: %ld ACTIVE + %ld inactive samples "
           "(log-uniform dHat 1e-4..1e-1, kappa 1..1e9, D across 12 decades, half with thickness)\n", act, inact);
    printf("    ACTIVE   : min B''/kappa %.6e   (B'' < 0 in %ld samples)\n", min_b2_act, neg_b2_act);
    printf("               max B' /kappa %.6e   (B' > 0 in %ld samples)\n", max_b1_act, pos_b1_act);
    printf("               max |B'| / (B'' D)  %.4f   <- scale of the term Gauss-Newton drops\n", max_drop);
    printf("    INACTIVE : min B''/kappa %.6e   (B'' < 0 in %ld samples)\n", min_b2_inact, neg_b2_inact);

    report("(2-4) PT              ", 12, hpt);
    report("(2-4) EE (un-mollified)", 12, hee);
    report("(2-4) PE              ", 9, hpe);
    report("(2-4) PP              ", 6, hpp);
    return 0;
}

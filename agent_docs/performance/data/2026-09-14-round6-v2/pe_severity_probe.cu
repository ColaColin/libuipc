// Round-6 V2: where does the PE rank-1 contact Hessian actually lose something,
// and *what* does it lose?
//
// `UIPC_CONTACT_RANK1=1` replaces the PE branch's exact PSD projection (which
// keeps rank 2 of 9) with the closed-form rank-1 matrix c gradD gradD^T.  s07
// measured the relative Frobenius error at 9.78e-05 mean / 1.900e-02 max over a
// uniform random sweep, but never said *which* configurations sit at the max --
// which is what a targeted micro-test has to drive the scene into.
//
// This probe runs the real __device__ functions and reports, binned by contact
// severity:
//   * ||H_proj - H_rank1|| / ||H_proj||                (the approximation error)
//   * lambda_2 / lambda_1 of the SHIPPED projection    (the stiffness dropped)
//   * the direction of the dropped eigenvector, resolved into the contact
//     normal and the two tangential directions of the closest-point pair --
//     i.e. is the model going soft along the normal (penetration) or along the
//     slide (friction / membrane drag)?
//
// Severity axes swept: the normalised gap g = sqrt(D)/(xi + dHat) across five
// decades, the foot position t along the edge (interior vs the two endpoint
// flags), and the edge length against the gap.
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

__device__ inline double rnd(unsigned long long& s)
{
    s += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
__device__ inline double U(unsigned long long& s, double a, double b)
{
    return a + (b - a) * rnd(s);
}

struct S
{
    double g;        // sqrt(D) / (xi + dHat), the normalised gap
    double t;        // foot position along the edge
    double rel;      // ||H_proj - H_rank1|| / ||H_proj||
    double lam_ratio;// lambda_2 / lambda_1 of the shipped projection
    double v2_n;     // |component of the dropped mode along the contact normal|
    double v2_t;     // |...along the edge direction| (the slide)
    double v2_b;     // |...along the third direction|
    double drop;     // |B'| / (B'' D)
    double lam1s;    // lambda_1(H_proj) * dHat^2 / kappa   (dimensionless)
    double lam2s;    // lambda_2(H_proj) * dHat^2 / kappa   -- the stiffness DROPPED
    double b2;       // B'' * dHat^4 / kappa
    int    flag_kind;// 0 = interior foot, 1 = endpoint
    int    ok;
};

__global__ void pe_kernel(S* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    S o{};
    o.ok = 0;
    out[i] = o;

    unsigned long long s = 0x51ED270BULL + 104729ULL * (unsigned long long)i;

    // the tumbler's own scales, plus two decades either side so the answer is
    // not a property of one scene: d_hat 1.27 mm, edge 6.2 mm, kappa 1e8.
    double dhat  = pow(10.0, U(s, -4.0, -2.0));
    double kappa = pow(10.0, U(s, 4.0, 9.0));
    double xi    = (i % 3 == 0) ? dhat * U(s, 0.05, 1.0) : 0.0;
    double lo = xi * xi, hi = (dhat + xi) * (dhat + xi);

    // structured severity sweep: 40 gap decades-bins, spread over the samples
    double ge = -5.0 + 5.0 * (double)((i / 4) % 40) / 39.0;   // 1e-5 .. 1
    double gap = (xi + dhat) * pow(10.0, ge);
    // edge length: 1x to 1000x the activation distance
    double L = (xi + dhat) * pow(10.0, U(s, 0.0, 3.0));
    // foot position: 3/4 interior, 1/4 beyond an endpoint (the other flags)
    double t = (i % 4 == 0) ? U(s, -0.5, 1.5) : U(s, 0.05, 0.95);

    Vector3 E0, e, nb;
    for(int k = 0; k < 3; ++k) { E0[k] = U(s, -1, 1); e[k] = U(s, -1, 1); nb[k] = U(s, -1, 1); }
    if(e.norm() < 1e-9) return;
    Vector3 eh = e.normalized();
    nb = nb - nb.dot(eh) * eh;
    if(nb.norm() < 1e-9) return;
    nb.normalize();
    Vector3 bh = eh.cross(nb);          // the third direction

    Vector3 E1 = E0 + L * eh;
    Vector3 P  = E0 + t * L * eh + gap * nb;

    Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);
    Float D;
    distance::point_edge_distance2(flag, P, E0, E1, D);
    if(!(D > lo && D < hi)) return;

    Vector9   G, G1;
    Matrix9x9 H, H1, Hproj;
    CN::PE_barrier_gradient_hessian(G, H, flag, (Float)kappa, (Float)dhat, (Float)xi, P, E0, E1);
    CN::PE_barrier_gradient_hessian_gn<1>(G1, H1, flag, (Float)kappa, (Float)dhat, (Float)xi, P, E0, E1);
    for(int a = 0; a < 9; ++a)
        for(int b = 0; b < 9; ++b)
            if(!isfinite(H(a, b)) || !isfinite(H1(a, b))) return;

    Hproj = H;
    CN::PE_barrier_make_spd<1, 1>(Hproj, flag, P, E0, E1);
    for(int a = 0; a < 9; ++a)
        for(int b = 0; b < 9; ++b)
            if(!isfinite(Hproj(a, b))) return;

    double nrm = Hproj.norm();
    if(!(nrm > 0)) return;
    o.rel = (Hproj - H1).norm() / nrm;

    Eigen::SelfAdjointEigenSolver<Matrix9x9> es(Hproj);
    Eigen::Matrix<Float, 9, 1> ev = es.eigenvalues();
    double l1 = ev(8), l2 = ev(7);
    if(!(l1 > 0)) return;
    o.lam_ratio = (l2 > 0) ? l2 / l1 : 0.0;
    // absolute scale: a 100 % *relative* error on a Hessian whose magnitude is
    // ~0 is not a physics problem, so the two eigenvalues are also reported in
    // the dimensionless unit kappa / dHat^2.
    double dh2 = (dhat + xi) * (dhat + xi);
    o.lam1s = l1 * dh2 / kappa;
    o.lam2s = ((l2 > 0) ? l2 : 0.0) * dh2 / kappa;

    // the dropped mode: the second eigenvector of the shipped projection.
    // Resolve the relative motion it induces between the point and its foot on
    // the edge into (normal, edge, binormal).
    Eigen::Matrix<Float, 9, 1> v2 = es.eigenvectors().col(7);
    double tc = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    Vector3 dP(v2(0), v2(1), v2(2));
    Vector3 d0(v2(3), v2(4), v2(5));
    Vector3 d1(v2(6), v2(7), v2(8));
    Vector3 rel = dP - ((1.0 - tc) * d0 + tc * d1);
    double rn = rel.norm();
    if(rn > 1e-300)
    {
        o.v2_n = fabs(rel.dot(nb)) / rn;
        o.v2_t = fabs(rel.dot(eh)) / rn;
        o.v2_b = fabs(rel.dot(bh)) / rn;
    }

    Float dBdD, ddBddD;
    CC::dKappaBarrierdD(dBdD, (Float)kappa, D, (Float)dhat, (Float)xi);
    CC::ddKappaBarrierddD(ddBddD, (Float)kappa, D, (Float)dhat, (Float)xi);
    o.drop = (ddBddD * D != 0.0) ? fabs((double)dBdD) / ((double)ddBddD * (double)D) : 0.0;
    o.b2 = (double)ddBddD * dh2 * dh2 / kappa;

    o.g = sqrt((double)D) / (xi + dhat);
    o.t = t;
    o.flag_kind = (flag[0] >= 0 && flag[1] >= 0 && flag[2] >= 0) ? 0 : 1;
    // the flag encoding differs per branch; use the geometric test instead
    o.flag_kind = (t > 0.0 && t < 1.0) ? 0 : 1;
    o.ok = 1;
    out[i] = o;
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 4000000;
    S* d = nullptr;
    cudaMalloc(&d, sizeof(S) * n);
    pe_kernel<<<(n + 127) / 128, 128>>>(d, n);
    cudaDeviceSynchronize();
    auto e = cudaGetLastError();
    if(e != cudaSuccess) { printf("CUDA: %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<S> h(n);
    cudaMemcpy(h.data(), d, sizeof(S) * n, cudaMemcpyDeviceToHost);

    std::vector<S> ok;
    for(auto& x : h) if(x.ok) ok.push_back(x);
    printf("## PE severity probe: %zu usable samples of %d\n", ok.size(), n);
    if(ok.empty()) return 1;

    // --- binned by the normalised gap g = sqrt(D)/(xi+dHat)
    printf("\n## relative error of the rank-1 PE Hessian, binned by contact severity\n");
    printf("   g = sqrt(D)/(xi+dHat): 1 = just touching the activation distance,\n");
    printf("   1e-5 = five decades deeper into the barrier.\n\n");
    printf("%-14s %8s %12s %12s %12s %12s %14s %14s\n", "g bin", "n",
           "rel mean", "rel p99", "rel max", "lam2/lam1 mn",
           "lam1 med [k/d2]", "lam2 max [k/d2]");
    for(int b = 0; b < 10; ++b)
    {
        double lo = pow(10.0, -5.0 + 0.5 * b), hi = pow(10.0, -5.0 + 0.5 * (b + 1));
        std::vector<double> r, l, a1, a2;
        for(auto& x : ok) if(x.g >= lo && x.g < hi)
        { r.push_back(x.rel); l.push_back(x.lam_ratio); a1.push_back(x.lam1s); a2.push_back(x.lam2s); }
        if(r.size() < 20) continue;
        std::sort(r.begin(), r.end());
        std::sort(a1.begin(), a1.end());
        double mean = 0; for(double v : r) mean += v; mean /= r.size();
        double lmean = 0; for(double v : l) lmean += v; lmean /= l.size();
        printf("%.1e-%.1e %8zu %12.4e %12.4e %12.4e %12.4e %14.4e %14.4e\n", lo, hi, r.size(),
               mean, r[(size_t)(0.99 * (r.size() - 1))], r.back(),
               lmean, a1[a1.size()/2], *std::max_element(a2.begin(), a2.end()));
    }
    printf("   (lam1/lam2 are in the dimensionless unit kappa/dHat^2: the ABSOLUTE stiffness.\n"
           "    A large relative error where lam2 is orders of magnitude below the deep-contact\n"
           "    lam1 is not a physics problem -- it is a near-zero matrix being set to zero.)\n");

    // the sub-population that would actually matter: a large relative error AND
    // an absolute dropped stiffness comparable to the deepest contacts' leading one
    {
        double ref = 0;  // the median lam1 of the deepest decade = "a real contact"
        std::vector<double> deep;
        for(auto& x : ok) if(x.g < 1e-4) deep.push_back(x.lam1s);
        std::sort(deep.begin(), deep.end());
        if(!deep.empty()) ref = deep[deep.size() / 2];
        size_t big = 0, bigrel = 0;
        for(auto& x : ok) { if(x.rel > 0.01) ++bigrel; if(x.rel > 0.01 && x.lam2s > 0.01 * ref) ++big; }
        printf("\n## does a large relative error ever coincide with a large ABSOLUTE dropped stiffness?\n");
        printf("   reference lam1 (median over g < 1e-4, i.e. a real contact) = %.4e kappa/dHat^2\n", ref);
        printf("   samples with rel > 1e-2                                    : %zu of %zu (%.3f %%)\n",
               bigrel, ok.size(), 100.0 * bigrel / ok.size());
        printf("   ... AND dropped stiffness lam2 > 1 %% of that reference     : %zu (%.4f %%)\n",
               big, 100.0 * big / ok.size());
        // the direct form of the same question: restrict to LOAD-BEARING pairs
        // (leading stiffness within two decades of a real contact) and ask how
        // wrong the rank-1 form is there.
        for(double frac : {1e-2, 1e-4})
        {
            double relmax = 0, ratmax = 0, relmean = 0; size_t cnt = 0;
            for(auto& x : ok)
                if(x.lam1s > frac * ref)
                { relmax = std::max(relmax, x.rel); ratmax = std::max(ratmax, x.lam_ratio);
                  relmean += x.rel; ++cnt; }
            if(cnt)
                printf("   pairs with lam1 > %.0e x reference (%zu, %.1f %%): "
                       "rel mean %.4e max %.4e, lam2/lam1 max %.4e\n",
                       frac, cnt, 100.0 * cnt / ok.size(), relmean / cnt, relmax, ratmax);
        }
    }

    // --- binned by foot position kind
    printf("\n## by where the point's foot falls on the edge\n");
    printf("%-24s %8s %12s %12s %12s\n", "case", "n", "rel mean", "rel max", "lam2/lam1 max");
    for(int k = 0; k < 2; ++k)
    {
        std::vector<double> r, l;
        for(auto& x : ok) if(x.flag_kind == k) { r.push_back(x.rel); l.push_back(x.lam_ratio); }
        if(r.empty()) continue;
        double mean = 0; for(double v : r) mean += v; mean /= r.size();
        printf("%-24s %8zu %12.4e %12.4e %12.4e\n",
               k == 0 ? "interior foot (dim 3)" : "beyond an endpoint",
               r.size(), mean, *std::max_element(r.begin(), r.end()),
               *std::max_element(l.begin(), l.end()));
    }

    // --- the worst 0.1 %: what do they look like?
    printf("\n## the worst 0.1 %% of samples by relative error -- what configuration are they?\n");
    std::sort(ok.begin(), ok.end(), [](const S& a, const S& b) { return a.rel > b.rel; });
    size_t m = std::max<size_t>(1, ok.size() / 1000);
    double g_mean = 0, t_mean = 0, lam = 0, vn = 0, vt = 0, vb = 0, drop = 0;
    int interior = 0;
    for(size_t i = 0; i < m; ++i)
    {
        g_mean += log10(ok[i].g); t_mean += ok[i].t; lam += ok[i].lam_ratio;
        vn += ok[i].v2_n; vt += ok[i].v2_t; vb += ok[i].v2_b; drop += ok[i].drop;
        interior += (ok[i].flag_kind == 0);
    }
    printf("   n=%zu   rel in [%.4e, %.4e]\n", m, ok[m - 1].rel, ok[0].rel);
    printf("   mean log10(g) = %.2f  (g = %.3e)   mean t = %.3f   interior foot %d/%zu\n",
           g_mean / m, pow(10.0, g_mean / m), t_mean / m, interior, m);
    printf("   mean lambda_2/lambda_1 of the shipped projection = %.4e\n", lam / m);
    printf("   mean |B'|/(B'' D) = %.4f\n", drop / m);
    {
        double l1s = 0, l2s = 0, b2s = 0;
        for(size_t i = 0; i < m; ++i) { l1s += ok[i].lam1s; l2s += ok[i].lam2s; b2s += ok[i].b2; }
        printf("   mean lambda_1 = %.4e, lambda_2 = %.4e  [kappa/dHat^2]   mean B'' = %.4e [kappa/dHat^4]\n",
               l1s / m, l2s / m, b2s / m);
    }
    printf("   the dropped eigenvector's relative motion, resolved:\n");
    printf("      along the contact normal  %.4f\n", vn / m);
    printf("      along the edge (the slide) %.4f\n", vt / m);
    printf("      along the binormal         %.4f\n", vb / m);

    // --- the whole population, for contrast
    double vn2 = 0, vt2 = 0, vb2 = 0, lam2 = 0;
    for(auto& x : ok) { vn2 += x.v2_n; vt2 += x.v2_t; vb2 += x.v2_b; lam2 += x.lam_ratio; }
    printf("\n## the whole population, for contrast\n");
    printf("   mean lambda_2/lambda_1 = %.4e\n", lam2 / ok.size());
    printf("   dropped eigenvector: normal %.4f  edge %.4f  binormal %.4f\n",
           vn2 / ok.size(), vt2 / ok.size(), vb2 / ok.size());
    double rmean = 0, rmax = 0;
    for(auto& x : ok) { rmean += x.rel; rmax = std::max(rmax, x.rel); }
    printf("   ||H_proj - H_rank1||/||H_proj||: mean %.4e  max %.4e\n",
           rmean / ok.size(), rmax);
    return 0;
}

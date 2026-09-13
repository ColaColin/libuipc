// s08 numerics probe -- the M = 4 basis-free reduced PSD projection against
// the shipped dense-Q one, on the device, against the REAL `__device__`
// functions (they are __device__-only, so a host build is not possible).
//
// Claim under test: for contact part 1's PT and un-mollified-EE branches,
//   make_spd_contact<12, 4, Solver, Basis = 1>   (round-6 s08, basis-free)
// is the SAME projection as
//   make_spd_contact<12, 4, Solver, Basis = 0>   (the shipped dense-Q form)
// up to rounding, and both equal the full 12x12 make_spd of the same Hessian.
//
// The third comparison is the one that proves the subspace claim itself
// (range(H) is inside range(Q)), not merely that two implementations of the
// same formula agree.
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

struct S
{
    double rel;        // ||H1 - H0|| / ||H0||          (new vs shipped)
    double rel_full0;  // ||H0 - Hfull|| / ||Hfull||    (shipped vs full 12x12)
    double rel_full1;  // ||H1 - Hfull|| / ||Hfull||    (new vs full 12x12)
    double asym1;      // ||H1 - H1^T|| / ||H1||
    double minev1;     // min eig(H1) / max|eig(H1)|
    double topratio;   // lmax(H1)/lmax(H0)
    int    ok;         // 1 = a usable dim-4 sample
    int    adv;        // which adversarial family
    double dratio;     // D / dHat_eff^2  (1 = exactly at d_hat, 0 = coincident)
    double n0, n1, nf; // Frobenius norms of the three projections
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
__device__ inline double U(unsigned long long& s, double a, double b)
{
    return a + (b - a) * rnd(s);
}

__device__ void cmp(S& o, const Matrix12x12& H, const Matrix12x12& H0, const Matrix12x12& H1)
{
    Matrix12x12 Hfull = H;
    uipc::backend::cuda::make_spd<12, 1>(Hfull);

    double n0 = H0.norm(), n1 = H1.norm(), nf = Hfull.norm();
    o.rel       = (n0 > 0) ? (H1 - H0).norm() / n0 : (H1 - H0).norm();
    o.rel_full0 = (nf > 0) ? (H0 - Hfull).norm() / nf : (H0 - Hfull).norm();
    o.rel_full1 = (nf > 0) ? (H1 - Hfull).norm() / nf : (H1 - Hfull).norm();
    o.asym1     = (n1 > 0) ? (H1 - H1.transpose()).norm() / n1 : 0.0;

    Eigen::SelfAdjointEigenSolver<Matrix12x12> e1(H1), e0(H0);
    double m1 = e1.eigenvalues().cwiseAbs().maxCoeff();
    o.minev1  = (m1 > 0) ? e1.eigenvalues().minCoeff() / m1 : 0.0;
    double l0 = e0.eigenvalues().maxCoeff();
    o.topratio = (l0 != 0) ? e1.eigenvalues().maxCoeff() / l0 : 1.0;
    o.n0 = n0; o.n1 = n1; o.nf = nf;
    o.ok = 1;
}

// adv families: 0 generic, 1 near-parallel edges / near-degenerate barycentric,
// 2 exactly at d_hat, 3 gap -> 0 (deep), 4 near-zero-length primitive
__global__ void k_pt(S* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    unsigned long long s = 0x51ED270BULL + 104729ULL * (unsigned long long)i;
    S o{}; o.ok = 0; o.adv = i % 5;

    double dhat  = pow(10.0, U(s, -4.0, -2.0));
    double kappa = pow(10.0, U(s, 2.0, 8.0));
    double xi    = (i % 3 == 0) ? dhat * U(s, 0.05, 1.0) : 0.0;
    double lo = xi * xi, hi = (dhat + xi) * (dhat + xi);

    Vector3 T0, T1, T2;
    double  scale = (o.adv == 4) ? pow(10.0, U(s, -7.0, -4.0)) : 1.0;
    for(int k = 0; k < 3; ++k)
    { T0[k] = U(s,-1,1); T1[k] = T0[k] + scale * U(s,-1,1); T2[k] = T0[k] + scale * U(s,-1,1); }
    Vector3 nb = (T1 - T0).cross(T2 - T0);
    if(!(nb.norm() > 0)) { out[i] = o; return; }
    nb.normalize();

    double a = U(s, 0.0, 1.0), b = U(s, 0.0, 1.0);
    if(a + b > 1.0) { a = 1.0 - a; b = 1.0 - b; }
    if(o.adv == 1)  // push the closest point onto an edge / a vertex: s^ degenerates
    { a = (i % 2) ? 1.0 - pow(10.0, U(s,-9.0,-3.0)) : pow(10.0, U(s,-9.0,-3.0));
      b = pow(10.0, U(s,-9.0,-3.0)); }

    Vector3 base = T0 + a * (T1 - T0) + b * (T2 - T0);
    double  gap;
    if(o.adv == 2)      gap = (xi + dhat) * (1.0 - 1e-15);   // exactly at d_hat
    else if(o.adv == 3) gap = (xi + dhat) * pow(10.0, U(s, -14.0, -8.0));
    else                gap = (xi + dhat) * pow(10.0, U(s, -5.0, 0.0));
    Vector3 P = base + gap * nb;

    Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);
    if(distance::detail::active_count(flag) != 4) { out[i] = o; return; }
    Float D; distance::point_triangle_distance2(flag, P, T0, T1, T2, D);
    if(!(D > lo && D < hi)) { out[i] = o; return; }

    Vector12 G; Matrix12x12 H;
    CN::PT_barrier_gradient_hessian(G, H, flag, (Float)kappa, (Float)dhat, (Float)xi, P, T0, T1, T2);
    if(!all_finite(H)) { out[i] = o; return; }
    Matrix12x12 H0 = H, H1 = H;
    CN::PT_barrier_make_spd<1, 0>(H0, flag, P, T0, T1, T2);
    CN::PT_barrier_make_spd<1, 1>(H1, flag, P, T0, T1, T2);
    if(!all_finite(H0) || !all_finite(H1)) { out[i] = o; return; }
    cmp(o, H, H0, H1);
    o.dratio = D / hi;
    out[i] = o;
}

__global__ void k_ee(S* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    unsigned long long s = 0x7E10B3C5ULL + 104729ULL * (unsigned long long)i;
    S o{}; o.ok = 0; o.adv = i % 5;

    double dhat  = pow(10.0, U(s, -4.0, -2.0));
    double kappa = pow(10.0, U(s, 2.0, 8.0));
    double xi    = (i % 3 == 0) ? dhat * U(s, 0.05, 1.0) : 0.0;
    double lo = xi * xi, hi = (dhat + xi) * (dhat + xi);

    Vector3 A0, ea, eb;
    double  el = (o.adv == 4) ? pow(10.0, U(s, -7.0, -4.0)) : 1.0;
    for(int k = 0; k < 3; ++k) { A0[k] = U(s,-1,1); ea[k] = el * U(s,-1,1); eb[k] = el * U(s,-1,1); }
    if(o.adv == 1)  // near-parallel: |ea x eb|^2 -> 0, the dim-4 `den` the mollifier guards
    {
        double eps = pow(10.0, U(s, -8.0, -2.0));
        Vector3 perp; for(int k = 0; k < 3; ++k) perp[k] = U(s,-1,1);
        eb = ea * U(s, 0.5, 2.0) + eps * perp;
    }
    Vector3 nb = ea.cross(eb);
    if(!(ea.norm() > 0 && eb.norm() > 0 && nb.norm() > 0)) { out[i] = o; return; }
    nb.normalize();
    Vector3 E0 = A0, E1 = A0 + ea;
    double  gap;
    if(o.adv == 2)      gap = (xi + dhat) * (1.0 - 1e-15);
    else if(o.adv == 3) gap = (xi + dhat) * pow(10.0, U(s, -14.0, -8.0));
    else                gap = (xi + dhat) * pow(10.0, U(s, -5.0, 0.0));
    Vector3 mid = A0 + U(s, 0.1, 0.9) * ea + gap * nb;
    Vector3 E2 = mid - U(s, 0.1, 0.9) * eb, E3 = E2 + eb;

    Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);
    if(distance::detail::active_count(flag) != 4) { out[i] = o; return; }
    Float D; distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);
    if(!(D > lo && D < hi)) { out[i] = o; return; }

    // the un-mollified EE Hessian is the plain flagged-distance barrier Hessian
    Vector12 G; Matrix12x12 H; bool moll = true;
    CN::mollified_EE_barrier_gradient_hessian<true>(
        G, H, moll, flag, (Float)kappa, (Float)dhat, (Float)xi,
        E0, E1, E2, E3, E0, E1, E2, E3);
    if(moll) { o.ok = 2; out[i] = o; return; }   // mollified pairs never reach this projection
    if(!all_finite(H)) { out[i] = o; return; }
    Matrix12x12 H0 = H, H1 = H;
    CN::EE_barrier_make_spd<1, 0>(H0, flag, E0, E1, E2, E3);
    CN::EE_barrier_make_spd<1, 1>(H1, flag, E0, E1, E2, E3);
    if(!all_finite(H0) || !all_finite(H1)) { out[i] = o; return; }
    cmp(o, H, H0, H1);
    o.dratio = D / hi;
    out[i] = o;
}

static void report(const char* name, const std::vector<S>& v)
{
    static const char* fam[5] = {"generic          ",
                                 "near-parallel / s^ degenerate",
                                 "exactly at d_hat ",
                                 "gap -> 0 (D/dhat^2 = 1e-28..1e-16)",
                                 "tiny primitive (1e-7..1e-4 edge)"};
    long   n = 0, moll = 0, nonpsd = 0;
    double masym = 0, minev = 1e300;
    printf("%s\n", name);
    printf("   %-34s %8s | %-21s | %-21s | %-21s\n", "adversarial family", "n",
           "new vs shipped", "shipped vs make_spd<12>", "new vs make_spd<12>");
    for(int f = 0; f < 5; ++f)
    {
        long   nf = 0;
        double sr = 0, mr = 0, s0 = 0, m0 = 0, s1 = 0, m1 = 0, mind = 1e300;
        for(const auto& o : v)
        {
            if(o.ok == 2) { if(f == 0) ++moll; continue; }
            if(o.ok != 1 || o.adv != f) continue;
            ++nf;
            sr += o.rel; mr = std::max(mr, o.rel);
            s0 += o.rel_full0; m0 = std::max(m0, o.rel_full0);
            s1 += o.rel_full1; m1 = std::max(m1, o.rel_full1);
            mind = std::min(mind, o.dratio);
            masym = std::max(masym, o.asym1);
            minev = std::min(minev, o.minev1);
            if(o.minev1 < -1e-12) ++nonpsd;
        }
        n += nf;
        if(!nf) { printf("   %-34s %8s | -\n", fam[f], "0"); continue; }
        printf("   %-34s %8ld | %.2e / %.2e | %.2e / %.2e | %.2e / %.2e   (min D/dhat^2 %.1e)\n",
               fam[f], nf, sr/nf, mr, s0/nf, m0, s1/nf, m1, mind);
    }
    printf("   totals: n = %ld%s;  max asymmetry of H_new %.3e;  "
           "min eig(H_new)/max|eig| %.3e;  non-PSD samples %ld\n\n",
           n, moll ? " (mollified samples skipped)" : "", masym, minev, nonpsd);
}

int main(int argc, char** argv)
{
    long total = argc > 1 ? atol(argv[1]) : 1000000;
    const int chunk = 250000;
    S* d; cudaMalloc(&d, chunk * sizeof(S));
    std::vector<S> apt, aee, h(chunk);
    for(long off = 0; off < total; off += chunk)
    {
        int n = (int)std::min<long>(chunk, total - off);
        k_pt<<<(n + 31) / 32, 32>>>(d, n);
        cudaError_t e = cudaDeviceSynchronize();
        if(e != cudaSuccess) { printf("CUDA error (pt): %s\n", cudaGetErrorString(e)); return 2; }
        cudaMemcpy(h.data(), d, n * sizeof(S), cudaMemcpyDeviceToHost);
        apt.insert(apt.end(), h.begin(), h.begin() + n);
        k_ee<<<(n + 31) / 32, 32>>>(d, n);
        e = cudaDeviceSynchronize();
        if(e != cudaSuccess) { printf("CUDA error (ee): %s\n", cudaGetErrorString(e)); return 2; }
        cudaMemcpy(h.data(), d, n * sizeof(S), cudaMemcpyDeviceToHost);
        aee.insert(aee.end(), h.begin(), h.begin() + n);
    }
    printf("## s08: M=4 basis-free vs dense-Q reduced PSD projection, %ld draws per pair type\n", total);
    report("PT (dim 4)", apt);
    report("EE (dim 4, un-mollified)", aee);
    return 0;
}

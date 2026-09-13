// perf/round6 s06 -- conservativeness proof for the candidate compaction
// (UIPC_CCD_COMPACT).
//
// The claim is NOT bit-identity of the candidate array: the compaction drops
// pairs from it.  The claim is that every dropped pair is one `filter_active`
// would have rejected at EVERY position the line search can evaluate, so the
// four ACTIVE sets it produces are unchanged.  `filter_active` keeps a pair
// iff  D <= thickness^2  or  thickness^2 < D < (thickness + d_hat)^2,
// i.e. iff  d < thickness + d_hat.  The line search only ever applies steps
// alpha_ls <= alpha_detect, so the domain of the claim is t in [0, 1] of the
// detected displacement.
//
// The verdict under test is produced by the REAL shipped device function --
// `distance::*_ccd<Float, EarlyOut=true, Stats=false, DcdCull=true>` writing
// through `dcd_inactive` -- not by a re-implementation of it.
#include <utils/distance/distance_flagged.h>
#include <utils/distance.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <curand_kernel.h>

using namespace uipc;
using namespace uipc::backend::cuda;

__device__ __forceinline__ Vector3 rnd3(curandState& s, double scale)
{
    return Vector3{(curand_uniform_double(&s) - 0.5) * scale,
                   (curand_uniform_double(&s) - 0.5) * scale,
                   (curand_uniform_double(&s) - 0.5) * scale};
}

// counters:
//  0 samples
//  1 dropped (dcd_inactive)
//  2 VIOLATION: dropped but ACTIVE (d < thickness + d_hat) somewhere on [0,1]
//  3 VIOLATION: dropped but within thickness somewhere on [0,1]
//  4 active-somewhere on [0,1] over the whole population (sanity)
//  5 adversarial samples drawn
//  6 dropped but active somewhere on (1, 1.1]  -- OUTSIDE the claim, informational
//  7 hits reported by the ACCD on the whole population (sanity)
__global__ void probe(unsigned long long seed, int n, unsigned long long* c, int kind)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    curandState s;
    curand_init(seed, i, 0, &s);

    const double pos_scale  = 0.02;
    const double sep        = curand_uniform_double(&s) * 0.02;
    const double disp_scale = pow(10.0, -4.0 + 4.0 * curand_uniform_double(&s));
    const unsigned adv      = curand(&s) & 7u;  // 1 in 8 of each adversarial kind
    const bool   parallel   = (adv == 0u);
    const bool   degenerate = (adv == 1u);   // zero-length edge / coincident points
    const bool   zero_disp  = (adv == 2u);   // no relative motion at all
    const bool   at_d_hat   = (adv == 3u);   // separation placed exactly at thickness+d_hat

    double eta       = 0.1 + 0.8 * curand_uniform_double(&s);
    double th_a      = (curand(&s) & 1u) ? 0.0 : curand_uniform_double(&s) * 1e-3;
    double th_b      = (curand(&s) & 1u) ? 0.0 : curand_uniform_double(&s) * 1e-3;
    double thickness = th_a + th_b;
    double d_hat     = pow(10.0, -4.0 + 2.0 * curand_uniform_double(&s));  // 1e-4 .. 1e-2

    Vector3 P[4], D[4];
    int     np = 0;

    if(kind == 0)  // edge-edge
    {
        Vector3 a0  = rnd3(s, pos_scale);
        Vector3 dir = rnd3(s, 1.0).normalized();
        Vector3 a1  = a0 + dir * (degenerate ? 0.0 : (0.002 + 0.01 * curand_uniform_double(&s)));
        Vector3 off = rnd3(s, 1.0).normalized() * (at_d_hat ? (thickness + d_hat) : sep);
        Vector3 b0   = a0 + off;
        Vector3 dir2 = parallel ? dir : rnd3(s, 1.0).normalized();
        Vector3 b1 = b0 + dir2 * (degenerate ? 0.0 : (0.002 + 0.01 * curand_uniform_double(&s)));
        P[0] = a0; P[1] = a1; P[2] = b0; P[3] = b1;
        np   = 4;
    }
    else if(kind == 1)  // point-triangle
    {
        Vector3 t0 = rnd3(s, pos_scale);
        Vector3 t1 = t0 + rnd3(s, 1.0).normalized() * (0.002 + 0.01 * curand_uniform_double(&s));
        Vector3 t2 = t0 + rnd3(s, 1.0).normalized() * (0.002 + 0.01 * curand_uniform_double(&s));
        if(parallel)
            t2 = t0 + (t1 - t0) * 1.0001;  // near-degenerate triangle
        if(degenerate)
        {
            t1 = t0;
            t2 = t0;
        }
        Vector3 p = t0 + rnd3(s, 1.0).normalized() * (at_d_hat ? (thickness + d_hat) : sep);
        P[0] = p; P[1] = t0; P[2] = t1; P[3] = t2;
        np   = 4;
    }
    else if(kind == 2)  // point-edge
    {
        Vector3 e0 = rnd3(s, pos_scale);
        Vector3 e1 = e0 + rnd3(s, 1.0).normalized()
                              * (degenerate ? 0.0 : (0.002 + 0.01 * curand_uniform_double(&s)));
        Vector3 p = e0 + rnd3(s, 1.0).normalized() * (at_d_hat ? (thickness + d_hat) : sep);
        P[0] = p; P[1] = e0; P[2] = e1;
        np   = 3;
    }
    else  // point-point
    {
        Vector3 p0 = rnd3(s, pos_scale);
        Vector3 p1 = p0 + rnd3(s, 1.0).normalized()
                              * (at_d_hat ? (thickness + d_hat) : (degenerate ? 0.0 : sep));
        P[0] = p0; P[1] = p1;
        np   = 2;
    }
    for(int k = 0; k < np; ++k)
        D[k] = zero_disp ? Vector3::Zero().eval() : rnd3(s, disp_scale);

    uint8_t dcd_keep = 1;
    bool    hit      = false;
    Float toc  = 1.1;  // large_enough_toi, exactly as filter_toi passes it
    if(kind == 0)
        hit = distance::edge_edge_ccd<Float, true, false, true>(
            P[0], P[1], P[2], P[3], D[0], D[1], D[2], D[3], eta, thickness, 1000, toc, nullptr, d_hat, &dcd_keep);
    else if(kind == 1)
        hit = distance::point_triangle_ccd<Float, true, false, true>(
            P[0], P[1], P[2], P[3], D[0], D[1], D[2], D[3], eta, thickness, 1000, toc, nullptr, d_hat, &dcd_keep);
    else if(kind == 2)
        hit = distance::point_edge_ccd<Float, true, false, true>(
            P[0], P[1], P[2], D[0], D[1], D[2], eta, thickness, 1000, toc, nullptr, d_hat, &dcd_keep);
    else
        hit = distance::point_point_ccd<Float, true, false, true>(
            P[0], P[1], D[0], D[1], eta, thickness, 1000, toc, nullptr, d_hat, &dcd_keep);

    atomicAdd(c + 0, 1ull);
    if(adv < 4u)
        atomicAdd(c + 5, 1ull);
    if(hit)
        atomicAdd(c + 7, 1ull);
    const bool drop = (dcd_keep == 0);
    if(drop)
        atomicAdd(c + 1, 1ull);

    // sample the swept segment densely
    const double act = (thickness + d_hat) * (thickness + d_hat);
    const double pen = thickness * thickness;
    bool active_in01 = false, active_past1 = false, pen_in01 = false;
    for(int q = 0; q <= 220; ++q)
    {
        const double u = q / 200.0;  // covers [0, 1.1]
        Vector3      A[4];
        for(int k = 0; k < np; ++k)
            A[k] = P[k] + u * D[k];
        Float D2 = 0;
        if(kind == 0)
            distance::edge_edge_distance2(
                distance::edge_edge_distance_flag(A[0], A[1], A[2], A[3]), A[0], A[1], A[2], A[3], D2);
        else if(kind == 1)
            distance::point_triangle_distance2(
                distance::point_triangle_distance_flag(A[0], A[1], A[2], A[3]), A[0], A[1], A[2], A[3], D2);
        else if(kind == 2)
            distance::point_edge_distance2(
                distance::point_edge_distance_flag(A[0], A[1], A[2]), A[0], A[1], A[2], D2);
        else
            distance::point_point_distance2(
                distance::point_point_distance_flag(A[0], A[1]), A[0], A[1], D2);
        if(D2 < act)
        {
            if(u <= 1.0) active_in01 = true;
            else         active_past1 = true;
        }
        if(u <= 1.0 && D2 <= pen)
            pen_in01 = true;
    }
    if(active_in01)
        atomicAdd(c + 4, 1ull);
    if(drop && active_in01)
        atomicAdd(c + 2, 1ull);
    if(drop && pen_in01)
        atomicAdd(c + 3, 1ull);
    if(drop && active_past1 && !active_in01)
        atomicAdd(c + 6, 1ull);
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    unsigned long long* c;
    cudaMallocManaged(&c, 8 * sizeof(unsigned long long));
    const char* names[4] = {"EE", "PT", "PE", "PP"};
    int         fail     = 0;
    for(int kind = 0; kind < 4; ++kind)
    {
        for(int k = 0; k < 8; ++k)
            c[k] = 0;
        cudaDeviceSynchronize();
        probe<<<(n + 255) / 256, 256>>>(0x9e3779b97f4a7c15ull + kind, n, c, kind);
        cudaError_t e = cudaDeviceSynchronize();
        if(e != cudaSuccess)
        {
            printf("CUDA ERROR %s\n", cudaGetErrorString(e));
            return 2;
        }
        printf("%s: samples=%llu adversarial=%llu dropped=%llu (%.2f %%) "
               "active_somewhere_on_[0,1]=%llu (%.2f %%) accd_hits=%llu (%.2f %%) "
               "| VIOLATIONS active=%llu penetrating=%llu | (info) dropped+active_only_past_t=1: %llu\n",
               names[kind], c[0], c[5], c[1], 100.0 * c[1] / c[0], c[4],
               100.0 * c[4] / c[0], c[7], 100.0 * c[7] / c[0], c[2], c[3], c[6]);
        if(c[2] || c[3])
            fail = 1;
    }
    printf(fail ? "RESULT: THE COMPACTION IS NOT CONSERVATIVE\n" :
                  "RESULT: every dropped pair is inactive over the whole swept step (0 violations)\n");
    return fail;
}

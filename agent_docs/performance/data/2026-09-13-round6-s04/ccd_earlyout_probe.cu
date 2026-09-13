// perf/round6 s04 — numerics/conservativeness proof for the ACCD first-pass
// early exit (UIPC_CCD_EARLY_OUT).
//
// The claim is that `EarlyOut = true` is an EXACT transformation of the
// shipped `EarlyOut = false` path, not a looser filter: for every input the
// two must return the same bool AND the same bit pattern in `toc`.  Anything
// else would mean the filter can report a different time of impact, and a
// larger one would be an interpenetration.
//
// This runs the REAL __device__ functions on the device (they are
// __host__ __device__, but FMA contraction differs between the two passes, so
// the device is the only place the claim matters).  Both instantiations are
// evaluated on the same inputs in the same thread.
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

// counters: 0 samples, 1 bool mismatches, 2 toc bit mismatches,
//           3 hits(old), 4 early-exits taken(new), 5 old-toc-less-than-new
__global__ void probe(unsigned long long seed, int n, unsigned long long* c, int kind)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    curandState s;
    curand_init(seed, i, 0, &s);

    // Mimic the candidate population: two primitives a few mm apart inside a
    // ~1 cm box, displacements of a comparable or much larger magnitude, and
    // (one sample in four) a deliberately near-degenerate / near-parallel
    // configuration, which is where ACCD iterates most.
    const double pos_scale  = 0.02;
    const double sep        = curand_uniform_double(&s) * 0.02;
    const double disp_scale = pow(10.0, -4.0 + 4.0 * curand_uniform_double(&s));
    const bool   parallel   = (curand(&s) & 3u) == 0u;

    double  eta       = 0.1 + 0.8 * curand_uniform_double(&s);
    double  thickness = (curand(&s) & 1u) ? 0.0 : curand_uniform_double(&s) * 1e-3;
    double  toc_a = 1.1, toc_b = 1.1;
    bool    ha, hb;

    if(kind == 0)  // edge-edge
    {
        Vector3 a0 = rnd3(s, pos_scale);
        Vector3 dir = rnd3(s, 1.0).normalized();
        Vector3 a1 = a0 + dir * (0.002 + 0.01 * curand_uniform_double(&s));
        Vector3 off = rnd3(s, 1.0).normalized() * sep;
        Vector3 b0 = a0 + off;
        Vector3 dir2 = parallel ? dir : rnd3(s, 1.0).normalized();
        Vector3 b1 = b0 + dir2 * (0.002 + 0.01 * curand_uniform_double(&s));
        Vector3 da0 = rnd3(s, disp_scale), da1 = rnd3(s, disp_scale);
        Vector3 db0 = rnd3(s, disp_scale), db1 = rnd3(s, disp_scale);
        ha = distance::edge_edge_ccd<Float, false, false>(
            a0, a1, b0, b1, da0, da1, db0, db1, eta, thickness, 1000, toc_a);
        hb = distance::edge_edge_ccd<Float, true, false>(
            a0, a1, b0, b1, da0, da1, db0, db1, eta, thickness, 1000, toc_b);
    }
    else if(kind == 1)  // point-triangle
    {
        Vector3 t0 = rnd3(s, pos_scale);
        Vector3 t1 = t0 + rnd3(s, 1.0).normalized() * (0.002 + 0.01 * curand_uniform_double(&s));
        Vector3 t2 = t0 + rnd3(s, 1.0).normalized() * (0.002 + 0.01 * curand_uniform_double(&s));
        if(parallel)
            t2 = t0 + (t1 - t0) * 1.0001;  // near-degenerate triangle
        Vector3 p  = t0 + rnd3(s, 1.0).normalized() * sep;
        Vector3 dp = rnd3(s, disp_scale);
        Vector3 d0 = rnd3(s, disp_scale), d1 = rnd3(s, disp_scale), d2 = rnd3(s, disp_scale);
        ha = distance::point_triangle_ccd<Float, false, false>(
            p, t0, t1, t2, dp, d0, d1, d2, eta, thickness, 1000, toc_a);
        hb = distance::point_triangle_ccd<Float, true, false>(
            p, t0, t1, t2, dp, d0, d1, d2, eta, thickness, 1000, toc_b);
    }
    else if(kind == 2)  // point-edge
    {
        Vector3 e0 = rnd3(s, pos_scale);
        Vector3 e1 = e0 + rnd3(s, 1.0).normalized() * (0.002 + 0.01 * curand_uniform_double(&s));
        Vector3 p  = e0 + rnd3(s, 1.0).normalized() * sep;
        Vector3 dp = rnd3(s, disp_scale);
        Vector3 d0 = rnd3(s, disp_scale), d1 = rnd3(s, disp_scale);
        ha = distance::point_edge_ccd<Float, false, false>(
            p, e0, e1, dp, d0, d1, eta, thickness, 1000, toc_a);
        hb = distance::point_edge_ccd<Float, true, false>(
            p, e0, e1, dp, d0, d1, eta, thickness, 1000, toc_b);
    }
    else  // point-point
    {
        Vector3 p0 = rnd3(s, pos_scale);
        Vector3 p1 = p0 + rnd3(s, 1.0).normalized() * sep;
        Vector3 d0 = rnd3(s, disp_scale), d1 = rnd3(s, disp_scale);
        ha = distance::point_point_ccd<Float, false, false>(
            p0, p1, d0, d1, eta, thickness, 1000, toc_a);
        hb = distance::point_point_ccd<Float, true, false>(
            p0, p1, d0, d1, eta, thickness, 1000, toc_b);
    }

    atomicAdd(c + 0, 1ull);
    if(ha != hb)
        atomicAdd(c + 1, 1ull);
    uint64_t ba, bb;
    memcpy(&ba, &toc_a, 8);
    memcpy(&bb, &toc_b, 8);
    if(ba != bb)
        atomicAdd(c + 2, 1ull);
    if(ha)
        atomicAdd(c + 3, 1ull);
    // the early exit is taken exactly when the old path also returned false
    // on its first pass; count how often the new path returned false with a
    // toc that equals its own first lower bound (a proxy, reported only)
    if(!hb)
        atomicAdd(c + 4, 1ull);
    // the safety-critical direction: a LARGER reported toc on the new path
    // would let the solver take a bigger step than the old filter allowed
    if(ha && hb && toc_b > toc_a)
        atomicAdd(c + 5, 1ull);
}

int main(int argc, char** argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    unsigned long long* c;
    cudaMallocManaged(&c, 6 * sizeof(unsigned long long));
    const char* names[4] = {"EE", "PT", "PE", "PP"};
    int fail = 0;
    for(int kind = 0; kind < 4; ++kind)
    {
        for(int k = 0; k < 6; ++k)
            c[k] = 0;
        cudaDeviceSynchronize();
        probe<<<(n + 255) / 256, 256>>>(0x9e3779b97f4a7c15ull + kind, n, c, kind);
        cudaError_t e = cudaDeviceSynchronize();
        if(e != cudaSuccess)
        {
            printf("CUDA ERROR %s\n", cudaGetErrorString(e));
            return 2;
        }
        printf("%s: samples=%llu  bool_mismatch=%llu  toc_bit_mismatch=%llu  hits(old)=%llu (%.2f %%)  no-hit(new)=%llu  larger_toc_on_new=%llu\n",
               names[kind], c[0], c[1], c[2], c[3], 100.0 * c[3] / c[0], c[4], c[5]);
        if(c[1] || c[2] || c[5])
            fail = 1;
    }
    printf(fail ? "RESULT: MISMATCH\n" : "RESULT: BIT-IDENTICAL on every sample\n");
    return fail;
}

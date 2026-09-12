#include <cuda_tool/spread_launch.h>
#include <contact_system/simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance/edge_edge_mollifier.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <utils/matrix_assembler.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>
#include <pipeline/ipc_pipeline_flag.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
namespace
{
    __global__ void do_compute_energy_k1_kernel(cuda_tool::CDense2D<ContactCoeff> table,
                                                cuda_tool::CBufferView<IndexT> contact_ids,
                                                cuda_tool::CBufferView<Vector4i> PTs,
                                                cuda_tool::BufferView<Float> Es,
                                                cuda_tool::CBufferView<Vector3> Ps,
                                                cuda_tool::CBufferView<Float> thicknesses,
                                                cuda_tool::CBufferView<Float> d_hats,
                                                Float dt,
                                                int   n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;

        using namespace sym::codim_ipc_simplex_contact;

        Vector4i PT = PTs(i);

        Vector4i cids = {contact_ids(PT[0]),
                         contact_ids(PT[1]),
                         contact_ids(PT[2]),
                         contact_ids(PT[3])};
        Float    kt2  = PT_kappa(table, cids) * dt * dt;

        const auto& P  = Ps(PT[0]);
        const auto& T0 = Ps(PT[1]);
        const auto& T1 = Ps(PT[2]);
        const auto& T2 = Ps(PT[3]);


        Float thickness = PT_thickness(thicknesses(PT(0)),
                                       thicknesses(PT(1)),
                                       thicknesses(PT(2)),
                                       thicknesses(PT(3)));

        Float d_hat =
            PT_d_hat(d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));

        Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);

        if constexpr(RUNTIME_CHECK)
        {
            Float D;
            distance::point_triangle_distance2(flag, P, T0, T1, T2, D);

            Vector2 range = D_range(thickness, d_hat);

            UIPC_KERNEL_ASSERT(is_active_D(range, D),
                               "PT[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                               PT(0),
                               PT(1),
                               PT(2),
                               PT(3),
                               D,
                               range(0),
                               range(1));
        }

        Es(i) = PT_barrier_energy(flag, kt2, d_hat, thickness, P, T0, T1, T2);
    }

    __global__ void do_compute_energy_k2_kernel(cuda_tool::CDense2D<ContactCoeff> table,
                                                cuda_tool::CBufferView<IndexT> contact_ids,
                                                cuda_tool::CBufferView<Vector4i> EEs,
                                                cuda_tool::BufferView<Float> Es,
                                                cuda_tool::CBufferView<Vector3> Ps,
                                                cuda_tool::CBufferView<Float> thicknesses,
                                                cuda_tool::CBufferView<Vector3> rest_Ps,
                                                cuda_tool::CBufferView<Float> d_hats,
                                                Float dt,
                                                int   n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;

        using namespace sym::codim_ipc_simplex_contact;

        Vector4i EE = EEs(i);

        Vector4i cids = {contact_ids(EE[0]),
                         contact_ids(EE[1]),
                         contact_ids(EE[2]),
                         contact_ids(EE[3])};
        Float    kt2  = EE_kappa(table, cids) * dt * dt;

        const auto& E0 = Ps(EE[0]);
        const auto& E1 = Ps(EE[1]);
        const auto& E2 = Ps(EE[2]);
        const auto& E3 = Ps(EE[3]);

        const auto& t0_Ea0 = rest_Ps(EE[0]);
        const auto& t0_Ea1 = rest_Ps(EE[1]);
        const auto& t0_Eb0 = rest_Ps(EE[2]);
        const auto& t0_Eb1 = rest_Ps(EE[3]);

        Float thickness = EE_thickness(thicknesses(EE(0)),
                                       thicknesses(EE(1)),
                                       thicknesses(EE(2)),
                                       thicknesses(EE(3)));

        Float d_hat =
            EE_d_hat(d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3)));

        Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

        if constexpr(RUNTIME_CHECK)
        {
            Float D;
            distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);
            Vector2 range = D_range(thickness, d_hat);
            UIPC_KERNEL_ASSERT(is_active_D(range, D),
                               "EE[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                               EE(0),
                               EE(1),
                               EE(2),
                               EE(3),
                               D,
                               range(0),
                               range(1));
        }


        Es(i) = mollified_EE_barrier_energy(flag,
                                            // coefficients
                                            kt2,
                                            d_hat,
                                            thickness,
                                            // positions
                                            t0_Ea0,
                                            t0_Ea1,
                                            t0_Eb0,
                                            t0_Eb1,
                                            E0,
                                            E1,
                                            E2,
                                            E3);
    }

    __global__ void do_compute_energy_k3_kernel(cuda_tool::CDense2D<ContactCoeff> table,
                                                cuda_tool::CBufferView<IndexT> contact_ids,
                                                cuda_tool::CBufferView<Vector3i> PEs,
                                                cuda_tool::BufferView<Float> Es,
                                                cuda_tool::CBufferView<Vector3> Ps,
                                                cuda_tool::CBufferView<Float> thicknesses,
                                                cuda_tool::CBufferView<Float> d_hats,
                                                Float dt,
                                                int   n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;

        using namespace sym::codim_ipc_simplex_contact;

        Vector3i PE = PEs(i);

        Vector3i cids = {contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
        Float kt2 = PE_kappa(table, cids) * dt * dt;

        const auto& P  = Ps(PE[0]);
        const auto& E0 = Ps(PE[1]);
        const auto& E1 = Ps(PE[2]);

        Float thickness =
            PE_thickness(thicknesses(PE(0)), thicknesses(PE(1)), thicknesses(PE(2)));

        Float d_hat = PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));

        Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

        if constexpr(RUNTIME_CHECK)
        {
            Float D;
            distance::point_edge_distance2(flag, P, E0, E1, D);

            Vector2 range = D_range(thickness, d_hat);

            UIPC_KERNEL_ASSERT(is_active_D(range, D),
                               "PE[%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                               PE(0),
                               PE(1),
                               PE(2),
                               D,
                               range(0),
                               range(1));
        }

        Es(i) = PE_barrier_energy(flag, kt2, d_hat, thickness, P, E0, E1);
    }

    __global__ void do_compute_energy_k4_kernel(cuda_tool::CDense2D<ContactCoeff> table,
                                                cuda_tool::CBufferView<IndexT> contact_ids,
                                                cuda_tool::CBufferView<Vector2i> PPs,
                                                cuda_tool::BufferView<Float> Es,
                                                cuda_tool::CBufferView<Vector3> Ps,
                                                cuda_tool::CBufferView<Float> thicknesses,
                                                cuda_tool::CBufferView<Float> d_hats,
                                                Float dt,
                                                int   n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;

        using namespace sym::codim_ipc_simplex_contact;

        Vector2i PP = PPs(i);

        Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
        Float    kt2  = PP_kappa(table, cids) * dt * dt;

        const auto& Pa = Ps(PP[0]);
        const auto& Pb = Ps(PP[1]);

        Float thickness = PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));

        Float d_hat = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));

        Vector2i flag = distance::point_point_distance_flag(Pa, Pb);

        if constexpr(RUNTIME_CHECK)
        {
            Float D;
            distance::point_point_distance2(flag, Pa, Pb, D);

            Vector2 range = D_range(thickness, d_hat);

            UIPC_KERNEL_ASSERT(is_active_D(range, D),
                               "PP[%d,%d] d^2(%f) out of range, (%f,%f)",
                               PP(0),
                               PP(1),
                               D,
                               range(0),
                               range(1));
        }

        Es(i) = PP_barrier_energy(flag, kt2, d_hat, thickness, Pa, Pb);
    }

    // round-4 (s17): partition the EE pair list by mollifier activity.
    // s16 picks the reduced rank-(m+1) projection per *warp* (`__any_sync`),
    // so one nearly-parallel pair drags its whole warp onto the 9x9 path. The
    // predicate is `|ea x eb|^2 < eps_x` on the *current* positions with
    // `eps_x` from the rest positions -- exactly what
    // `mollified_EE_barrier_gradient_hessian<true>` recomputes per pair -- so
    // reordering the pair list by it makes every warp but the boundary one
    // uniform. Non-mollified pairs fill `perm` from the front, mollified ones
    // from the back; the order inside each part is irrelevant because every
    // pair writes its own gradient/Hessian slot (indexed by the *original*
    // pair index), so the output buffers are unchanged except for which
    // projection a formerly-mixed warp takes (rounding level, see s16).
    __global__ void ee_mollifier_partition_kernel(cuda_tool::CBufferView<Vector4i> EEs,
                                                  cuda_tool::CBufferView<Vector3> Ps,
                                                  cuda_tool::CBufferView<Vector3> rest_Ps,
                                                  cuda_tool::BufferView<IndexT> perm,
                                                  cuda_tool::BufferView<IndexT> counters,
                                                  int                           n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;

        Vector4i EE = EEs(i);

        Float eps_x;
        distance::edge_edge_mollifier_threshold(rest_Ps(EE[0]),
                                                rest_Ps(EE[1]),
                                                rest_Ps(EE[2]),
                                                rest_Ps(EE[3]),
                                                static_cast<Float>(1e-3),
                                                eps_x);
        Float cross_norm2;
        distance::edge_edge_cross_norm2(
            Ps(EE[0]), Ps(EE[1]), Ps(EE[2]), Ps(EE[3]), cross_norm2);

        bool   mollified = cross_norm2 < eps_x;
        IndexT slot = mollified ? n - 1 - atomicAdd(&counters(1), 1) :
                                  atomicAdd(&counters(0), 1);
        perm(slot)  = i;
    }

    // round-5 (s25): `SpdTql` picks the PSD projection of contact part 1
    // (PT + EE). false = the pre-round-5 path (Eigen's SelfAdjointEigenSolver
    // inside `make_spd`, and the dense 12x9 Helmert basis for the mollified EE
    // Hessian); true = s19's fixed-size tridiagonal QL plus K16's blocked form
    // of the same translation-free 9x9 projection, which drops the 12x9 basis
    // and its two 12x9 temporaries. Same projection up to rounding; a template
    // parameter, not a runtime flag, so one stack frame per instantiation
    // (s14). Part 2 (PE + PP) is deliberately left on the old path.
    // round-5 (s31): `Spd2` picks the PSD projection of contact part 2
    // (PE + PP). false = the pre-s31 path (Eigen's SelfAdjointEigenSolver and
    // the explicit range basis Q of `barrier_range_basis`); true = s19's
    // fixed-size tridiagonal QL for the 4x4 of the PE dim-3 branch plus s31's
    // basis-free form of the same reduced projection (no Q, no 9x9 Hs / Hspd
    // temporary, no tangent frame). Same projection up to rounding; a template
    // parameter, not a runtime flag, so one stack frame per instantiation
    // (s14). Part 1 (PT + EE) is on the `SpdTql` axis above and passes
    // `Spd2 = false`.
    template <bool GradientOnly, int Part, bool EEReducedRange, bool SpdTql = false, bool Spd2 = false>
    __global__ void do_assemble_kernel(cuda_tool::CDense2D<ContactCoeff> table,
                                       cuda_tool::CBufferView<IndexT> contact_ids,
                                       cuda_tool::CBufferView<Vector3> Ps,
                                       cuda_tool::CBufferView<Vector3> rest_Ps,
                                       cuda_tool::CBufferView<Float> thicknesses,
                                       cuda_tool::CBufferView<Float>    d_hats,
                                       Float                            dt,
                                       cuda_tool::CBufferView<Vector4i> PTs,
                                       cuda_tool::DoubletVectorView<Float, 3> PT_Gs,
                                       cuda_tool::TripletMatrixView<Float, 3> PT_Hs,
                                       cuda_tool::CBufferView<Vector4i> EEs,
                                       cuda_tool::CBufferView<IndexT>   ee_perm,
                                       cuda_tool::DoubletVectorView<Float, 3> EE_Gs,
                                       cuda_tool::TripletMatrixView<Float, 3> EE_Hs,
                                       cuda_tool::CBufferView<Vector3i> PEs,
                                       cuda_tool::DoubletVectorView<Float, 3> PE_Gs,
                                       cuda_tool::TripletMatrixView<Float, 3> PE_Hs,
                                       cuda_tool::CBufferView<Vector2i> PPs,
                                       cuda_tool::DoubletVectorView<Float, 3> PP_Gs,
                                       cuda_tool::TripletMatrixView<Float, 3> PP_Hs,
                                       IndexT ee_offset,
                                       IndexT pe_offset,
                                       IndexT pp_offset,
                                       bool   ee_reduced_spd,
                                       int    n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n)
            return;

        using namespace sym::codim_ipc_simplex_contact;

        constexpr int SpdSolver = SpdTql ? 1 : 0;
        constexpr int Spd2Solver = Spd2 ? 1 : 0;
        constexpr int Spd2Basis  = Spd2 ? 1 : 0;

        // perf/kernels (K9): Part 1 = PT+EE (12x12 branches), Part 2 = PE+PP
        // (the bulk, small reduced projections), Part 0 = the fused kernel.
        if constexpr(Part != 2)
        {
            if(idx < ee_offset)  // PT
            {
                int      i    = idx;
                Vector4i PT   = PTs(i);
                Vector4i cids = {contact_ids(PT[0]),
                                 contact_ids(PT[1]),
                                 contact_ids(PT[2]),
                                 contact_ids(PT[3])};
                Float    kt2  = PT_kappa(table, cids) * dt * dt;

                const auto& P  = Ps(PT[0]);
                const auto& T0 = Ps(PT[1]);
                const auto& T1 = Ps(PT[2]);
                const auto& T2 = Ps(PT[3]);

                Float thickness = PT_thickness(thicknesses(PT(0)),
                                               thicknesses(PT(1)),
                                               thicknesses(PT(2)),
                                               thicknesses(PT(3)));
                Float d_hat =
                    PT_d_hat(d_hats(PT(0)), d_hats(PT(1)), d_hats(PT(2)), d_hats(PT(3)));
                Vector4i flag = distance::point_triangle_distance_flag(P, T0, T1, T2);

                Vector12 G;
                if constexpr(GradientOnly)
                {
                    PT_barrier_gradient(G, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                    DoubletVectorAssembler DVA{PT_Gs};
                    DVA.segment<4>(i * 4).write(PT, G);
                }
                else
                {
                    Matrix12x12 H;
                    PT_barrier_gradient_hessian(
                        G, H, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                    PT_barrier_make_spd<SpdSolver>(H, flag, P, T0, T1, T2);
                    DoubletVectorAssembler DVA{PT_Gs};
                    DVA.segment<4>(i * 4).write(PT, G);
                    TripletMatrixAssembler TMA{PT_Hs};
                    TMA.half_block<4>(i * SimplexNormalContact::PTHalfHessianSize)
                        .write(PT, H);
                }
                return;
            }
            if(idx < pe_offset)  // EE
            {
                // round-4 (s17): with the mollifier partition on, the pair
                // handled by this lane is `ee_perm[k]` instead of `k`; an
                // empty view means the natural order.
                int k = idx - ee_offset;
                int i = ee_perm.size() ? (int)ee_perm(k) : k;
                Vector4i EE = EEs(i);
                Vector4i cids = {contact_ids(EE[0]),
                                 contact_ids(EE[1]),
                                 contact_ids(EE[2]),
                                 contact_ids(EE[3])};
                Float    kt2  = EE_kappa(table, cids) * dt * dt;

                const auto& E0     = Ps(EE[0]);
                const auto& E1     = Ps(EE[1]);
                const auto& E2     = Ps(EE[2]);
                const auto& E3     = Ps(EE[3]);
                const auto& t0_Ea0 = rest_Ps(EE[0]);
                const auto& t0_Ea1 = rest_Ps(EE[1]);
                const auto& t0_Eb0 = rest_Ps(EE[2]);
                const auto& t0_Eb1 = rest_Ps(EE[3]);

                Float thickness = EE_thickness(thicknesses(EE(0)),
                                               thicknesses(EE(1)),
                                               thicknesses(EE(2)),
                                               thicknesses(EE(3)));
                Float d_hat =
                    EE_d_hat(d_hats(EE(0)), d_hats(EE(1)), d_hats(EE(2)), d_hats(EE(3)));
                Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                Vector12 G;
                if constexpr(GradientOnly)
                {
                    mollified_EE_barrier_gradient(
                        G, flag, kt2, d_hat, thickness, t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, E0, E1, E2, E3);
                    DoubletVectorAssembler DVA{EE_Gs};
                    DVA.segment<4>(i * 4).write(EE, G);
                }
                else
                {
                    Matrix12x12 H;
                    bool        mollified = true;
                    mollified_EE_barrier_gradient_hessian<EEReducedRange>(
                        G, H, mollified, flag, kt2, d_hat, thickness, t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, E0, E1, E2, E3);
                    // round-4 (s16): when the mollifier is inactive the EE
                    // Hessian is a plain flagged-distance barrier Hessian, so
                    // the exact rank-(m+1) reduced projection of 1da12a82
                    // applies (5x5 / 4x4 / 3x3 eigen-solve instead of 9x9).
                    // The choice is made per *warp*, not per thread: the K10
                    // 9x9 projection is valid for every EE pair, so a warp
                    // that holds even one mollified pair runs the 9x9 path for
                    // all of its lanes. Both branches are expensive, and a
                    // divergent warp would execute both of them in full --
                    // measured at +20 to +25 % on the wrecking balls and
                    // cube-wall against a warp-uniform -20 %.
                    bool warp_reduced = false;
                    if constexpr(EEReducedRange)
                        warp_reduced = !__any_sync(__activemask(), mollified);
                    if(warp_reduced)
                    {
                        EE_barrier_make_spd<SpdSolver>(H, flag, E0, E1, E2, E3);
                    }
                    // perf/kernels (K10): the mollified EE barrier depends on
                    // relative positions only, so H annihilates rigid
                    // translations and the PSD projection reduces to the
                    // 9x9 translation-free subspace (same projection up to
                    // rounding, as K7 for the hinge)
                    else if(ee_reduced_spd)
                    {
                        // round-5 (s25): K16's blocked assembly of the same
                        // translation-free 9x9 projection -- constant Helmert
                        // weights on 3x3 blocks instead of a 12x9 basis matrix
                        // and its Q^T H Q / Q Hr Q^T temporaries
                        if constexpr(SpdTql)
                            make_spd_translation_free_4x3_blocked<SpdSolver>(H);
                        else
                            make_spd_translation_free_4x3<SpdSolver>(H);
                    }
                    else
                        make_spd<12, SpdSolver>(H);
                    DoubletVectorAssembler DVA{EE_Gs};
                    DVA.segment<4>(i * 4).write(EE, G);
                    TripletMatrixAssembler TMA{EE_Hs};
                    TMA.half_block<4>(i * SimplexNormalContact::EEHalfHessianSize)
                        .write(EE, H);
                }
                return;
            }
        }
        if constexpr(Part != 1)
        {
            if(idx < pp_offset)  // PE
            {
                int      i  = idx - pe_offset;
                Vector3i PE = PEs(i);
                Vector3i cids = {contact_ids(PE[0]), contact_ids(PE[1]), contact_ids(PE[2])};
                Float kt2 = PE_kappa(table, cids) * dt * dt;

                const auto& P  = Ps(PE[0]);
                const auto& E0 = Ps(PE[1]);
                const auto& E1 = Ps(PE[2]);

                Float thickness = PE_thickness(
                    thicknesses(PE(0)), thicknesses(PE(1)), thicknesses(PE(2)));
                Float d_hat = PE_d_hat(d_hats(PE(0)), d_hats(PE(1)), d_hats(PE(2)));
                Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                Vector9 G;
                if constexpr(GradientOnly)
                {
                    PE_barrier_gradient(G, flag, kt2, d_hat, thickness, P, E0, E1);
                    DoubletVectorAssembler DVA{PE_Gs};
                    DVA.segment<3>(i * 3).write(PE, G);
                }
                else
                {
                    Matrix9x9 H;
                    PE_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P, E0, E1);
                    PE_barrier_make_spd<Spd2Solver, Spd2Basis>(H, flag, P, E0, E1);
                    DoubletVectorAssembler DVA{PE_Gs};
                    DVA.segment<3>(i * 3).write(PE, G);
                    TripletMatrixAssembler TMA{PE_Hs};
                    TMA.half_block<3>(i * SimplexNormalContact::PEHalfHessianSize)
                        .write(PE, H);
                }
                return;
            }
            // PP
            {
                int         i    = idx - pp_offset;
                const auto& PP   = PPs(i);
                Vector2i    cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                Float       kt2  = PP_kappa(table, cids) * dt * dt;

                const auto& P0 = Ps(PP[0]);
                const auto& P1 = Ps(PP[1]);

                Float thickness = PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));
                Float    d_hat = PP_d_hat(d_hats(PP(0)), d_hats(PP(1)));
                Vector2i flag  = distance::point_point_distance_flag(P0, P1);

                Vector6 G;
                if constexpr(GradientOnly)
                {
                    PP_barrier_gradient(G, flag, kt2, d_hat, thickness, P0, P1);
                    DoubletVectorAssembler DVA{PP_Gs};
                    DVA.segment<2>(i * 2).write(PP, G);
                }
                else
                {
                    Matrix6x6 H;
                    PP_barrier_gradient_hessian(G, H, flag, kt2, d_hat, thickness, P0, P1);
                    PP_barrier_make_spd<Spd2Solver, Spd2Basis>(H, flag, P0, P1);
                    DoubletVectorAssembler DVA{PP_Gs};
                    DVA.segment<2>(i * 2).write(PP, G);
                    TripletMatrixAssembler TMA{PP_Hs};
                    TMA.half_block<2>(i * SimplexNormalContact::PPHalfHessianSize)
                        .write(PP, H);
                }
            }
        }
    }

}  // namespace

class IPCSimplexNormalContact final : public SimplexNormalContact
{
  public:
    using SimplexNormalContact::SimplexNormalContact;

    // perf/kernels (K9): the fused G+H kernel is compiled for its worst
    // branch (EE: mollified 12x12 Hessian + full make_spd; 255 registers,
    // 13.5 KB stack per thread), which the PE/PP bulk then pays for too.
    // Split into two launches with unchanged per-pair arithmetic and
    // output slots (bit-identical): PT+EE on a side stream forked from and
    // joined back into the default stream, PE+PP (lean) on the default
    // stream, so the rare expensive pairs overlap the bulk instead of
    // serialising behind it. UIPC_CONTACT_SPLIT=0 restores the fused
    // launch, =1 runs the two launches back to back on the default stream.
    int m_split = 2;
    // perf/kernels (K10): EE Hessian PSD projection on the translation-free
    // 9x9 subspace (UIPC_EE_REDUCED_SPD=0 restores the 12x12 eigen-solve)
    bool         m_ee_reduced_spd = true;
    // round-4 (s16): exact rank-(m+1) reduced PSD projection for the EE branch
    // whenever the edge-edge mollifier is inactive (UIPC_EE_REDUCED_RANGE=0
    // restores the unconditional K10 9x9 projection)
    bool m_ee_reduced_range = true;
    // round-4 (s17): reorder the EE pairs so that the mollified ones are
    // contiguous and the per-warp choice above is uniform
    // (UIPC_EE_PARTITION=0 restores the natural order)
    bool                            m_ee_partition     = true;
    // round-5 (s25): part 1's PSD projection on s19's tridiagonal-QL solver and
    // K16's blocked translation-free basis (UIPC_CONTACT_SPD_TQL=0 restores the
    // Eigen SelfAdjointEigenSolver and the dense 12x9 basis)
    bool                            m_spd_tql          = true;

    // round-5 (s31): contact part 2's (PE + PP) PSD projection on s19's
    // tridiagonal-QL solver and s31's basis-free range reduction
    // (UIPC_CONTACT_SPD2=0 restores Eigen + the explicit basis Q)
    bool                            m_spd2             = true;
    IndexT                          m_ee_partition_min = 64;
    cuda_tool::DeviceBuffer<IndexT> m_ee_perm;
    cuda_tool::DeviceBuffer<IndexT> m_ee_counters;
    cudaStream_t m_side_stream    = nullptr;
    cudaEvent_t  m_fork           = nullptr;
    cudaEvent_t  m_join           = nullptr;

    virtual void do_build(BuildInfo& info) override
    {
        require<IPCPipelineFlag>();

        if(const char* e = std::getenv("UIPC_CONTACT_SPLIT"))
            m_split = std::atoi(e);
        if(const char* e = std::getenv("UIPC_EE_REDUCED_SPD"))
            m_ee_reduced_spd = !(e[0] == '0');
        if(const char* e = std::getenv("UIPC_EE_REDUCED_RANGE"))
            m_ee_reduced_range = !(e[0] == '0');
        if(const char* e = std::getenv("UIPC_EE_PARTITION"))
            m_ee_partition = !(e[0] == '0');
        if(const char* e = std::getenv("UIPC_CONTACT_SPD_TQL"))
            m_spd_tql = !(e[0] == '0');
        if(const char* e = std::getenv("UIPC_CONTACT_SPD2"))
            m_spd2 = !(e[0] == '0');
        if(m_split == 2)
        {
            CUDA_TOOL_CHECK(cudaStreamCreateWithFlags(&m_side_stream, cudaStreamNonBlocking));
            CUDA_TOOL_CHECK(cudaEventCreateWithFlags(&m_fork, cudaEventDisableTiming));
            CUDA_TOOL_CHECK(cudaEventCreateWithFlags(&m_join, cudaEventDisableTiming));
        }
    }

    ~IPCSimplexNormalContact() override
    {
        // best effort: the CUDA context may already be gone at exit
        if(m_join)
            cudaEventDestroy(m_join);
        if(m_fork)
            cudaEventDestroy(m_fork);
        if(m_side_stream)
            cudaStreamDestroy(m_side_stream);
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace cuda_tool;
        static cuda_tool::SpreadVerifier sv_energy{"IPCSimplexNormalContact::energy"};
        using namespace sym::codim_ipc_simplex_contact;

        // Compute Point-Triangle energy
        auto PT_count = info.PTs().size();
        cuda_tool::launch_spread(
            sv_energy,
            (int)(PT_count),
            do_compute_energy_k1_kernel,
            [&](int grid, int block)
            {
                do_compute_energy_k1_kernel<<<grid, block, 0, nullptr>>>(
                    info.contact_tabular().viewer(),
                    info.contact_element_ids().viewer(),
                    info.PTs().viewer(),
                    info.PT_energies().viewer(),
                    info.positions().viewer(),
                    info.thicknesses().viewer(),
                    info.d_hats().viewer(),
                    info.dt(),
                    (int)PT_count);
            },
            [&](cuda_tool::SpreadVerifier& v)
            {
                v.add_buffer(info.PT_energies());
            });


        // Compute Edge-Edge energy
        auto EE_count = info.EEs().size();
        cuda_tool::launch_spread(
            sv_energy,
            (int)(EE_count),
            do_compute_energy_k2_kernel,
            [&](int grid, int block)
            {
                do_compute_energy_k2_kernel<<<grid, block, 0, nullptr>>>(
                    info.contact_tabular().viewer(),
                    info.contact_element_ids().viewer(),
                    info.EEs().viewer(),
                    info.EE_energies().viewer(),
                    info.positions().viewer(),
                    info.thicknesses().viewer(),
                    info.rest_positions().viewer(),
                    info.d_hats().viewer(),
                    info.dt(),
                    (int)EE_count);
            },
            [&](cuda_tool::SpreadVerifier& v)
            {
                v.add_buffer(info.EE_energies());
            });


        // Compute Point-Edge energy
        auto PE_count = info.PEs().size();
        cuda_tool::launch_spread(
            sv_energy,
            (int)(PE_count),
            do_compute_energy_k3_kernel,
            [&](int grid, int block)
            {
                do_compute_energy_k3_kernel<<<grid, block, 0, nullptr>>>(
                    info.contact_tabular().viewer(),
                    info.contact_element_ids().viewer(),
                    info.PEs().viewer(),
                    info.PE_energies().viewer(),
                    info.positions().viewer(),
                    info.thicknesses().viewer(),
                    info.d_hats().viewer(),
                    info.dt(),
                    (int)PE_count);
            },
            [&](cuda_tool::SpreadVerifier& v)
            {
                v.add_buffer(info.PE_energies());
            });


        // Compute Point-Point energy
        auto PP_count = info.PPs().size();
        cuda_tool::launch_spread(
            sv_energy,
            (int)(PP_count),
            do_compute_energy_k4_kernel,
            [&](int grid, int block)
            {
                do_compute_energy_k4_kernel<<<grid, block, 0, nullptr>>>(
                    info.contact_tabular().viewer(),
                    info.contact_element_ids().viewer(),
                    info.PPs().viewer(),
                    info.PP_energies().viewer(),
                    info.positions().viewer(),
                    info.thicknesses().viewer(),
                    info.d_hats().viewer(),
                    info.dt(),
                    (int)PP_count);
            },
            [&](cuda_tool::SpreadVerifier& v)
            {
                v.add_buffer(info.PP_energies());
            });

    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace cuda_tool;
        using namespace sym::codim_ipc_simplex_contact;

        auto pt_count = (IndexT)info.PTs().size();
        auto ee_count = (IndexT)info.EEs().size();
        auto pe_count = (IndexT)info.PEs().size();
        auto pp_count = (IndexT)info.PPs().size();
        auto total    = pt_count + ee_count + pe_count + pp_count;

        if(total == 0)
            return;

        // round-4 (s17): empty = the natural pair order (no partition)
        cuda_tool::CBufferView<IndexT> ee_perm{};

        IndexT ee_offset = pt_count;
        IndexT pe_offset = ee_offset + ee_count;
        IndexT pp_offset = pe_offset + pe_count;

        // One launch per Part (see m_split): the fused kernel (Part 0) or
        // PT+EE (Part 1, [0, pt+ee)) and PE+PP (Part 2, [0, pe+pp)); the
        // offsets are re-based so every pair keeps its per-type index and
        // output slot. Two launches on the same stream would serialise the
        // rare, individually expensive PT/EE Hessians behind the bulk, so
        // Part 1 runs on a side stream (fork/join with events).
        auto launch_t = [&]<bool GradientOnly, int Part, bool EEReducedRange, bool SpdTql, bool Spd2>(
                            IndexT       ee_offset,
                            IndexT       pe_offset,
                            IndexT       pp_offset,
                            IndexT       n,
                            cudaStream_t s)
        {
            auto k = do_assemble_kernel<GradientOnly, Part, EEReducedRange, SpdTql, Spd2>;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, s>>>(
                info.contact_tabular().viewer(),
                info.contact_element_ids().viewer(),
                info.positions().viewer(),
                info.rest_positions().viewer(),
                info.thicknesses().viewer(),
                info.d_hats().viewer(),
                info.dt(),
                info.PTs().viewer(),
                info.PT_gradients().viewer(),
                info.PT_hessians().viewer(),
                info.EEs().viewer(),
                ee_perm,
                info.EE_gradients().viewer(),
                info.EE_hessians().viewer(),
                info.PEs().viewer(),
                info.PE_gradients().viewer(),
                info.PE_hessians().viewer(),
                info.PPs().viewer(),
                info.PP_gradients().viewer(),
                info.PP_hessians().viewer(),
                ee_offset,
                pe_offset,
                pp_offset,
                m_ee_reduced_spd,
                n);
        };

        // round-4 (s16): the reduced-range EE projection is a template
        // parameter, not a runtime flag, so the instantiation that takes it
        // does not carry the 9x9 path's stack frame (cf. s14).
        // round-5 (s31): the part-2 projection is the third template axis. Part 1
        // has no PE/PP branch, so it is only ever instantiated with Spd2 = false.
        auto launch_2 = [&]<bool GradientOnly, int Part, bool EEReducedRange, bool SpdTql>(
                            IndexT       ee_offset,
                            IndexT       pe_offset,
                            IndexT       pp_offset,
                            IndexT       n,
                            cudaStream_t s)
        {
            if constexpr(!GradientOnly && Part != 1)
            {
                if(m_spd2)
                {
                    launch_t.template operator()<GradientOnly, Part, EEReducedRange, SpdTql, true>(
                        ee_offset, pe_offset, pp_offset, n, s);
                    return;
                }
            }
            launch_t.template operator()<GradientOnly, Part, EEReducedRange, SpdTql, false>(
                ee_offset, pe_offset, pp_offset, n, s);
        };

        auto launch = [&]<bool GradientOnly, int Part>(IndexT       ee_offset,
                                                      IndexT       pe_offset,
                                                      IndexT       pp_offset,
                                                      IndexT       n,
                                                      cudaStream_t s)
        {
            if constexpr(!GradientOnly && Part != 2)
            {
                // round-5 (s25): the part-1 PSD projection solver is the second
                // template axis; gradient-only launches and part 2 never reach
                // it, so they are only instantiated with SpdTql = false.
                if(m_ee_reduced_range)
                {
                    if(m_spd_tql)
                        launch_2.template operator()<GradientOnly, Part, true, true>(
                            ee_offset, pe_offset, pp_offset, n, s);
                    else
                        launch_2.template operator()<GradientOnly, Part, true, false>(
                            ee_offset, pe_offset, pp_offset, n, s);
                    return;
                }
                if(m_spd_tql)
                {
                    launch_2.template operator()<GradientOnly, Part, false, true>(
                        ee_offset, pe_offset, pp_offset, n, s);
                    return;
                }
            }
            launch_2.template operator()<GradientOnly, Part, false, false>(
                ee_offset, pe_offset, pp_offset, n, s);
        };

        // round-4 (s17): one extra launch (plus a 2-int memset) that sorts
        // the EE pairs into (not mollified | mollified). 8-17 us per Newton
        // iteration on the benchmarks, against 150-290 us saved in the EE
        // branch; see the `run` lambda for why it goes on the default stream.
        auto partition_ee = [&](cudaStream_t s)
        {
            if(!m_ee_partition || !m_ee_reduced_range || ee_count < m_ee_partition_min)
                return;
            if(m_ee_counters.size() != 2)
                m_ee_counters.resize(2);
            m_ee_perm.resize_discard(ee_count);
            CUDA_TOOL_CHECK(cudaMemsetAsync(
                m_ee_counters.view().data(), 0, 2 * sizeof(IndexT), s));
            auto k = ee_mollifier_partition_kernel;
            k<<<cuda_tool::best_grid_dim((int)ee_count, k), cuda_tool::best_block_dim(k), 0, s>>>(
                info.EEs().viewer(),
                info.positions().viewer(),
                info.rest_positions().viewer(),
                m_ee_perm.view(),
                m_ee_counters.view(),
                (int)ee_count);
            ee_perm = m_ee_perm.view();
        };

        auto run = [&]<bool GradientOnly>()
        {
            IndexT n_ptee = pt_count + ee_count;
            IndexT n_pepp = pe_count + pp_count;
            if(GradientOnly || m_split == 0 || n_ptee == 0 || n_pepp == 0)
            {
                // gradient-only is lean already (144 registers); a single
                // non-empty part needs no split either
                if constexpr(!GradientOnly)
                    partition_ee(nullptr);
                launch.operator()<GradientOnly, 0>(ee_offset, pe_offset, pp_offset, total, nullptr);
                return;
            }
            // the partition goes on the *default* stream, ahead of the fork:
            // put on the side stream it delays part 1 by its own ~20 us, and
            // part 2 (the bigger grid, 167 blocks on the wrecking balls)
            // then wins the race for the SMs and part 1 serialises behind it
            // (+1 ms per iteration, measured).
            if constexpr(!GradientOnly)
                partition_ee(nullptr);
            cudaStream_t side = nullptr;
            if(m_split == 2)
            {
                side = m_side_stream;
                CUDA_TOOL_CHECK(cudaEventRecord(m_fork, nullptr));
                CUDA_TOOL_CHECK(cudaStreamWaitEvent(side, m_fork, 0));
            }
            launch.operator()<GradientOnly, 1>(pt_count, n_ptee, n_ptee, n_ptee, side);
            launch.operator()<GradientOnly, 2>(0, 0, pe_count, n_pepp, nullptr);
            if(m_split == 2)
            {
                CUDA_TOOL_CHECK(cudaEventRecord(m_join, side));
                CUDA_TOOL_CHECK(cudaStreamWaitEvent(nullptr, m_join, 0));
            }
        };

        if(info.gradient_only())
            run.operator()<true>();
        else
            run.operator()<false>();
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContact);
}  // namespace uipc::backend::cuda

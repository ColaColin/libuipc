#include <uipc/common/timer.h>
#include <collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.h>
#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <iterator>
#include <utils/distance/distance_flagged.h>
#include <utils/distance.h>
#include <utils/codim_thickness.h>
#include <utils/simplex_contact_mask_utils.h>
#include <uipc/common/zip.h>
#include <utils/primitive_d_hat.h>
#include <array>
#include <cstdio>

namespace uipc::backend::cuda
{
constexpr bool PrintDebugInfo          = false;
constexpr bool PrintKernelZeroDistance = false;

namespace
{
    constexpr SizeT max_iter = 1000;

    constexpr Float large_enough_toi = 1.1;

}  // namespace

// perf/round6 (s04): slots per pair type in the ACCD diagnosis buffer:
// 0 calls, 1 first-pass early exits, 2 loop passes, 3 converged hits.
constexpr int CCD_STAT_SLOTS = 4;

namespace
{

    /****************************************************
    *                   Broad Phase
    ****************************************************/

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_detect_k1_kernel(
        cuda_tool::CBufferView<IndexT>  codimVs,
        cuda_tool::CBufferView<Vector3> Ps,
        cuda_tool::CBufferView<Vector3> dxs,
        cuda_tool::BufferView<AABB>     aabbs,
        cuda_tool::CBufferView<IndexT>  v2bs,
        cuda_tool::CBufferView<IndexT>  contact_ids,
        cuda_tool::BufferView<IndexT>   bids,
        cuda_tool::BufferView<IndexT>   cids,
        cuda_tool::CBufferView<Float>   thicknesses,
        cuda_tool::CBufferView<Float>   d_hats,
        Float                           alpha,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto vI = codimVs(i);

        Float thickness       = thicknesses(vI);
        Float d_hat_expansion = point_dcd_expansion(d_hats(vI));

        const auto& pos   = Ps(vI);
        Vector3     pos_t = pos + dxs(vI) * alpha;

        AABB aabb;
        aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>());

        float expand = d_hat_expansion + thickness;

        aabb.min().array() -= expand;
        aabb.max().array() += expand;
        aabbs(i) = aabb;
        bids(i)  = v2bs(vI);
        cids(i)  = contact_ids(vI);
    }

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_detect_k2_kernel(
        cuda_tool::CBufferView<IndexT>  Vs,
        cuda_tool::CBufferView<Vector3> dxs,
        cuda_tool::CBufferView<Vector3> Ps,
        cuda_tool::BufferView<AABB>     aabbs,
        cuda_tool::CBufferView<IndexT>  v2bs,
        cuda_tool::CBufferView<IndexT>  contact_ids,
        cuda_tool::BufferView<IndexT>   bids,
        cuda_tool::BufferView<IndexT>   cids,
        cuda_tool::CBufferView<Float>   thicknesses,
        cuda_tool::CBufferView<Float>   d_hats,
        Float                           alpha,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto vI = Vs(i);

        Float thickness       = thicknesses(vI);
        Float d_hat_expansion = point_dcd_expansion(d_hats(vI));

        const auto& pos   = Ps(vI);
        Vector3     pos_t = pos + dxs(vI) * alpha;

        AABB aabb;
        aabb.extend(pos.cast<float>()).extend(pos_t.cast<float>());

        float expand = d_hat_expansion + thickness;

        aabb.min().array() -= expand;
        aabb.max().array() += expand;
        aabbs(i) = aabb;
        bids(i)  = v2bs(vI);
        cids(i)  = contact_ids(vI);
    }

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_detect_k3_kernel(
        cuda_tool::CBufferView<Vector2i> Es,
        cuda_tool::CBufferView<Vector3>  Ps,
        cuda_tool::BufferView<AABB>      aabbs,
        cuda_tool::CBufferView<IndexT>   v2bs,
        cuda_tool::CBufferView<IndexT>   contact_ids,
        cuda_tool::BufferView<IndexT>    bids,
        cuda_tool::BufferView<IndexT>    cids,
        cuda_tool::CBufferView<Vector3>  dxs,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::CBufferView<Float>    d_hats,
        Float                            alpha,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto eI = Es(i);

        Float thickness = edge_thickness(thicknesses(eI[0]), thicknesses(eI[1]));
        Float d_hat_expansion = edge_dcd_expansion(d_hats(eI[0]), d_hats(eI[1]));

        const auto& pos0   = Ps(eI[0]);
        const auto& pos1   = Ps(eI[1]);
        Vector3     pos0_t = pos0 + dxs(eI[0]) * alpha;
        Vector3     pos1_t = pos1 + dxs(eI[1]) * alpha;

        AABB aabb;

        aabb.extend(pos0.cast<float>())
            .extend(pos1.cast<float>())
            .extend(pos0_t.cast<float>())
            .extend(pos1_t.cast<float>());

        float expand = d_hat_expansion + thickness;

        aabb.min().array() -= expand;
        aabb.max().array() += expand;
        aabbs(i) = aabb;
        bids(i)  = v2bs(eI[0]);
        cids(i)  = contact_ids(eI[0]);
    }

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_detect_k4_kernel(
        cuda_tool::CBufferView<Vector3i> Fs,
        cuda_tool::CBufferView<Vector3>  Ps,
        cuda_tool::BufferView<AABB>      aabbs,
        cuda_tool::CBufferView<IndexT>   v2bs,
        cuda_tool::CBufferView<IndexT>   contact_ids,
        cuda_tool::BufferView<IndexT>    bids,
        cuda_tool::BufferView<IndexT>    cids,
        cuda_tool::CBufferView<Vector3>  dxs,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::CBufferView<Float>    d_hats,
        Float                            alpha,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto fI = Fs(i);

        Float thickness = triangle_thickness(
            thicknesses(fI[0]), thicknesses(fI[1]), thicknesses(fI[2]));
        Float d_hat_expansion =
            triangle_dcd_expansion(d_hats(fI[0]), d_hats(fI[1]), d_hats(fI[2]));

        const auto& pos0   = Ps(fI[0]);
        const auto& pos1   = Ps(fI[1]);
        const auto& pos2   = Ps(fI[2]);
        Vector3     pos0_t = pos0 + dxs(fI[0]) * alpha;
        Vector3     pos1_t = pos1 + dxs(fI[1]) * alpha;
        Vector3     pos2_t = pos2 + dxs(fI[2]) * alpha;

        AABB aabb;

        aabb.extend(pos0.cast<float>())
            .extend(pos1.cast<float>())
            .extend(pos2.cast<float>())
            .extend(pos0_t.cast<float>())
            .extend(pos1_t.cast<float>())
            .extend(pos2_t.cast<float>());

        float expand = d_hat_expansion + thickness;

        aabb.min().array() -= expand;
        aabb.max().array() += expand;
        aabbs(i) = aabb;
        bids(i)  = v2bs(fI[0]);
        cids(i)  = contact_ids(fI[0]);
    }

    struct InfoStacklessBVHSimplexTrajectoryFilter_detect_node_pred
    {
        cuda_tool::CBufferView<IndexT> body_self_collision;
        cuda_tool::CDense2D<IndexT>    cmts;

        __device__ bool operator()(InfoStacklessBVH::NodePredInfo info) const
        {
            constexpr IndexT invalid = static_cast<IndexT>(-1);
            auto             qbid    = info.query_bid;
            auto             qcid    = info.query_cid;
            bool bid_cull = info.node_bid != invalid && qbid != invalid
                            && qbid == info.node_bid && !body_self_collision(qbid);
            bool cid_cull = info.node_cid != invalid && qcid != invalid
                            && !cmts(qcid, info.node_cid);
            return !(bid_cull || cid_cull);
        }
    };

    struct InfoStacklessBVHSimplexTrajectoryFilter_detect_AllP_CodimP_pred
    {
        cuda_tool::CBufferView<IndexT>  Vs;
        cuda_tool::CBufferView<IndexT>  codimVs;
        cuda_tool::CBufferView<Vector3> Ps;
        cuda_tool::CBufferView<Vector3> dxs;
        cuda_tool::CBufferView<Float>   thicknesses;
        cuda_tool::CBufferView<IndexT>  dimensions;
        cuda_tool::CBufferView<IndexT>  contact_element_ids;
        cuda_tool::CDense2D<IndexT>     contact_mask_tabular;
        cuda_tool::CBufferView<IndexT>  subscene_element_ids;
        cuda_tool::CDense2D<IndexT>     subscene_mask_tabular;
        cuda_tool::CBufferView<IndexT>  body_self_collision;
        cuda_tool::CBufferView<Float>   d_hats;
        Float                           alpha;

        __device__ bool operator()(InfoStacklessBVH::LeafPredInfo info) const
        {
            auto        i      = info.i;
            auto        j      = info.j;
            const auto& V      = Vs(i);
            const auto& codimV = codimVs(j);

            Vector2i cids = {contact_element_ids(V), contact_element_ids(codimV)};
            Vector2i scids = {subscene_element_ids(V), subscene_element_ids(codimV)};

            if(!allow_PP_contact(subscene_mask_tabular, scids))
                return false;
            if(!allow_PP_contact(contact_mask_tabular, cids))
                return false;

            bool V_is_codim = dimensions(V) <= 2;

            if(V_is_codim && V >= codimV)
                return false;

            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 P0 = Ps(V);
            Vector3 P1 = Ps(codimV);

            Float thickness = PP_thickness(thicknesses(V), thicknesses(codimV));
            Float d_hat     = PP_d_hat(d_hats(V), d_hats(codimV));

            Float expand = d_hat + thickness;

            if(alpha == 0.0)
            {
                // DCD detect pass: exact distance test. The conservative
                // ccd_broadphase degenerates to a per-axis box test at
                // alpha==0 and passes ~50x more candidates than the exact
                // activation filter downstream re-keeps anyway. Pairs with
                // D2 <= thickness^2 are kept so the thickness-violation
                // handling in filter_active still sees them.
                Float D2;
                distance::point_point_distance2(P0, P1, D2);
                return D2 < expand * expand;
            }

            Vector3 dP0 = alpha * dxs(V);
            Vector3 dP1 = alpha * dxs(codimV);

            if(!distance::point_point_ccd_broadphase(P0, P1, dP0, dP1, expand))
                return false;

            return true;
        }
    };

    struct InfoStacklessBVHSimplexTrajectoryFilter_detect_CodimP_AllE_pred
    {
        cuda_tool::CBufferView<IndexT>   codimVs;
        cuda_tool::CBufferView<Vector2i> Es;
        cuda_tool::CBufferView<Vector3>  Ps;
        cuda_tool::CBufferView<Vector3>  dxs;
        cuda_tool::CBufferView<Float>    thicknesses;
        cuda_tool::CBufferView<IndexT>   contact_element_ids;
        cuda_tool::CDense2D<IndexT>      contact_mask_tabular;
        cuda_tool::CBufferView<IndexT>   subscene_element_ids;
        cuda_tool::CDense2D<IndexT>      subscene_mask_tabular;
        cuda_tool::CBufferView<IndexT>   body_self_collision;
        cuda_tool::CBufferView<Float>    d_hats;
        Float                            alpha;

        __device__ bool operator()(InfoStacklessBVH::LeafPredInfo info) const
        {
            auto        i      = info.i;
            auto        j      = info.j;
            const auto& codimV = codimVs(i);
            const auto& E      = Es(j);

            Vector3i cids = {contact_element_ids(codimV),
                             contact_element_ids(E[0]),
                             contact_element_ids(E[1])};

            Vector3i scids = {subscene_element_ids(codimV),
                              subscene_element_ids(E[0]),
                              subscene_element_ids(E[1])};

            if(!allow_PE_contact(subscene_mask_tabular, scids))
                return false;
            if(!allow_PE_contact(contact_mask_tabular, cids))
                return false;

            if(E[0] == codimV || E[1] == codimV)
                return false;

            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 E0 = Ps(E[0]);
            Vector3 E1 = Ps(E[1]);

            Vector3 P = Ps(codimV);

            Float thickness =
                PE_thickness(thicknesses(codimV), thicknesses(E[0]), thicknesses(E[1]));
            Float d_hat = PE_d_hat(d_hats(codimV), d_hats(E[0]), d_hats(E[1]));

            Float expand = d_hat + thickness;

            if(alpha == 0.0)
            {
                // DCD detect pass: exact distance test (see PP pred)
                Float    D2;
                Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);
                distance::point_edge_distance2(flag, P, E0, E1, D2);
                return D2 < expand * expand;
            }

            Vector3 dE0 = alpha * dxs(E[0]);
            Vector3 dE1 = alpha * dxs(E[1]);
            Vector3 dP  = alpha * dxs(codimV);

            if(!distance::point_edge_ccd_broadphase(P, E0, E1, dP, dE0, dE1, expand))
                return false;

            return true;
        }
    };

    struct InfoStacklessBVHSimplexTrajectoryFilter_detect_AllE_AllE_pred
    {
        cuda_tool::CBufferView<Vector2i> Es;
        cuda_tool::CBufferView<Vector3>  Ps;
        cuda_tool::CBufferView<Vector3>  dxs;
        cuda_tool::CBufferView<Float>    thicknesses;
        cuda_tool::CBufferView<IndexT>   contact_element_ids;
        cuda_tool::CDense2D<IndexT>      contact_mask_tabular;
        cuda_tool::CBufferView<IndexT>   subscene_element_ids;
        cuda_tool::CDense2D<IndexT>      subscene_mask_tabular;
        cuda_tool::CBufferView<IndexT>   body_self_collision;
        cuda_tool::CBufferView<Float>    d_hats;
        Float                            alpha;

        __device__ bool operator()(InfoStacklessBVH::LeafPredInfo info) const
        {
            auto        i  = info.i;
            auto        j  = info.j;
            const auto& E0 = Es(i);
            const auto& E1 = Es(j);

            Vector4i cids = {contact_element_ids(E0[0]),
                             contact_element_ids(E0[1]),
                             contact_element_ids(E1[0]),
                             contact_element_ids(E1[1])};

            Vector4i scids = {subscene_element_ids(E0[0]),
                              subscene_element_ids(E0[1]),
                              subscene_element_ids(E1[0]),
                              subscene_element_ids(E1[1])};

            if(!allow_EE_contact(subscene_mask_tabular, scids))
                return false;
            if(!allow_EE_contact(contact_mask_tabular, cids))
                return false;

            if(E0[0] == E1[0] || E0[0] == E1[1] || E0[1] == E1[0] || E0[1] == E1[1])
                return false;

            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 E0_0 = Ps(E0[0]);
            Vector3 E0_1 = Ps(E0[1]);

            Vector3 E1_0 = Ps(E1[0]);
            Vector3 E1_1 = Ps(E1[1]);

            Float thickness = EE_thickness(thicknesses(E0[0]),
                                           thicknesses(E0[1]),
                                           thicknesses(E1[0]),
                                           thicknesses(E1[1]));

            Float d_hat =
                EE_d_hat(d_hats(E0[0]), d_hats(E0[1]), d_hats(E1[0]), d_hats(E1[1]));

            Float expand = d_hat + thickness;

            if(alpha == 0.0)
            {
                // DCD detect pass: exact distance test (see PP pred); the
                // plain EE distance is a correct superset of the mollified
                // degenerate handling in filter_active
                Float D2;
                Vector4i flag = distance::edge_edge_distance_flag(E0_0, E0_1, E1_0, E1_1);
                distance::edge_edge_distance2(flag, E0_0, E0_1, E1_0, E1_1, D2);
                return D2 < expand * expand;
            }

            Vector3 dE0_0 = alpha * dxs(E0[0]);
            Vector3 dE0_1 = alpha * dxs(E0[1]);
            Vector3 dE1_0 = alpha * dxs(E1[0]);
            Vector3 dE1_1 = alpha * dxs(E1[1]);

            if(!distance::edge_edge_ccd_broadphase(
                   E0_0, E0_1, E1_0, E1_1, dE0_0, dE0_1, dE1_0, dE1_1, expand))
                return false;

            return true;
        }
    };

    struct InfoStacklessBVHSimplexTrajectoryFilter_detect_AllP_AllT_pred
    {
        cuda_tool::CBufferView<IndexT>   Vs;
        cuda_tool::CBufferView<Vector3i> Fs;
        cuda_tool::CBufferView<Vector3>  Ps;
        cuda_tool::CBufferView<Vector3>  dxs;
        cuda_tool::CBufferView<Float>    thicknesses;
        cuda_tool::CBufferView<IndexT>   contact_element_ids;
        cuda_tool::CDense2D<IndexT>      contact_mask_tabular;
        cuda_tool::CBufferView<IndexT>   subscene_element_ids;
        cuda_tool::CDense2D<IndexT>      subscene_mask_tabular;
        cuda_tool::CBufferView<IndexT>   body_self_collision;
        cuda_tool::CBufferView<Float>    d_hats;
        Float                            alpha;

        __device__ bool operator()(InfoStacklessBVH::LeafPredInfo info) const
        {
            auto i = info.i;
            auto j = info.j;
            auto V = Vs(i);
            auto F = Fs(j);

            Vector4i cids = {contact_element_ids(V),
                             contact_element_ids(F[0]),
                             contact_element_ids(F[1]),
                             contact_element_ids(F[2])};

            Vector4i scids = {subscene_element_ids(V),
                              subscene_element_ids(F[0]),
                              subscene_element_ids(F[1]),
                              subscene_element_ids(F[2])};

            if(!allow_PT_contact(subscene_mask_tabular, scids))
                return false;
            if(!allow_PT_contact(contact_mask_tabular, cids))
                return false;

            if(F[0] == V || F[1] == V || F[2] == V)
                return false;

            if(info.bid_i == info.bid_j && info.bid_i != static_cast<IndexT>(-1)
               && !body_self_collision(info.bid_i))
                return false;

            Vector3 P = Ps(V);

            Vector3 F0 = Ps(F[0]);
            Vector3 F1 = Ps(F[1]);
            Vector3 F2 = Ps(F[2]);

            Float thickness = PT_thickness(thicknesses(V),
                                           thicknesses(F[0]),
                                           thicknesses(F[1]),
                                           thicknesses(F[2]));

            Float d_hat = PT_d_hat(d_hats(V), d_hats(F[0]), d_hats(F[1]), d_hats(F[2]));

            Float expand = d_hat + thickness;

            if(alpha == 0.0)
            {
                // DCD detect pass: exact distance test (see PP pred)
                Float D2;
                Vector4i flag = distance::point_triangle_distance_flag(P, F0, F1, F2);
                distance::point_triangle_distance2(flag, P, F0, F1, F2, D2);
                return D2 < expand * expand;
            }

            Vector3 dP = alpha * dxs(V);

            Vector3 dF0 = alpha * dxs(F[0]);
            Vector3 dF1 = alpha * dxs(F[1]);
            Vector3 dF2 = alpha * dxs(F[2]);

            if(!distance::point_triangle_ccd_broadphase(P, F0, F1, F2, dP, dF0, dF1, dF2, expand))
                return false;

            return true;
        }
    };

    /****************************************************
    *                   Filter Active
    ****************************************************/

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k1_kernel(
        cuda_tool::CBufferView<Vector3>  positions,
        cuda_tool::CBufferView<Vector2i> PCodimP_pairs,
        cuda_tool::CBufferView<IndexT>   surf_vertices,
        cuda_tool::CBufferView<IndexT>   codim_vertices,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::BufferView<Vector2i>  temp_PPs,
        cuda_tool::CBufferView<Float>    d_hats,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto& PP = temp_PPs(i);
        PP.setConstant(-1);

        Vector2i indices = PCodimP_pairs(i);

        IndexT P0 = surf_vertices(indices(0));
        IndexT P1 = codim_vertices(indices(1));

        const auto& V0 = positions(P0);
        const auto& V1 = positions(P1);

        Float thickness = PP_thickness(thicknesses(P0), thicknesses(P1));
        Float d_hat     = PP_d_hat(d_hats(P0), d_hats(P1));

        Vector2 range = D_range(thickness, d_hat);

        Float D;
        distance::point_point_distance2(V0, V1, D);

        if constexpr(PrintKernelZeroDistance)
        {
            if(D <= range.x())
            {
                printf(
                    "[ISBVH][PP][low-dist] i=%d P=(%d,%d) D=%e range=(%e,%e) "
                    "thickness=%e d_hat=%e\n",
                    i,
                    P0,
                    P1,
                    D,
                    range.x(),
                    range.y(),
                    thickness,
                    d_hat);
            }
        }

        UIPC_KERNEL_ASSERT(D > range.x(),
                           "Thickness Violated! D(%f) should be > D_range.x(%f), "
                           "P=(%d,%d), thickness=%f, d_hat=%f",
                           D,
                           range.x(),
                           P0,
                           P1,
                           thickness,
                           d_hat);
        if(!is_active_D(range, D))
            return;

        PP = {P0, P1};
    }

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k2_kernel(
        cuda_tool::CBufferView<Vector3>  positions,
        cuda_tool::CBufferView<Vector2i> CodimP_AllE_pairs,
        cuda_tool::CBufferView<IndexT>   codim_veritces,
        cuda_tool::CBufferView<Vector2i> surf_edges,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::BufferView<Vector2i>  temp_PPs,
        cuda_tool::BufferView<Vector3i>  temp_PEs,
        cuda_tool::CBufferView<Float>    d_hats,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto& PP = temp_PPs(i);
        PP.setConstant(-1);
        auto& PE = temp_PEs(i);
        PE.setConstant(-1);

        Vector2i indices = CodimP_AllE_pairs(i);
        IndexT   V       = codim_veritces(indices(0));
        Vector2i E       = surf_edges(indices(1));

        Vector3i vIs = {V, E(0), E(1)};
        Vector3 Ps[] = {positions(vIs(0)), positions(vIs(1)), positions(vIs(2))};

        Float thickness =
            PE_thickness(thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));

        Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));

        Vector3i flag = distance::point_edge_distance_flag(Ps[0], Ps[1], Ps[2]);

        Vector2 range = D_range(thickness, d_hat);

        Float D;
        distance::point_edge_distance2(flag, Ps[0], Ps[1], Ps[2], D);

        if constexpr(PrintKernelZeroDistance)
        {
            if(D <= range.x())
            {
                printf(
                    "[ISBVH][PE][low-dist] i=%d V-E=(%d,%d,%d) flag=(%d,%d,%d) "
                    "D=%e range=(%e,%e) thickness=%e d_hat=%e\n",
                    i,
                    vIs(0),
                    vIs(1),
                    vIs(2),
                    flag(0),
                    flag(1),
                    flag(2),
                    D,
                    range.x(),
                    range.y(),
                    thickness,
                    d_hat);
            }
        }

        UIPC_KERNEL_ASSERT(D > range.x(),
                           "Thickness Violated! D(%f) should be > D_range.x(%f), "
                           "V-E=(%d,%d,%d), flag=(%d,%d,%d), thickness=%f, d_hat=%f",
                           D,
                           range.x(),
                           vIs(0),
                           vIs(1),
                           vIs(2),
                           flag(0),
                           flag(1),
                           flag(2),
                           thickness,
                           d_hat);
        if(!is_active_D(range, D))
            return;

        Vector3i offsets;
        auto     dim = distance::degenerate_point_edge(flag, offsets);

        switch(dim)
        {
            case 2: {
                IndexT V0 = vIs(offsets(0));
                IndexT V1 = vIs(offsets(1));
                PP        = {V0, V1};
            }
            break;
            case 3: {
                PE = vIs;
            }
            break;
            default: {
                UIPC_KERNEL_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
            }
            break;
        }
    }

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k3_kernel(
        cuda_tool::CBufferView<Vector3>  positions,
        cuda_tool::CBufferView<Vector2i> PT_pairs,
        cuda_tool::CBufferView<IndexT>   surf_vertices,
        cuda_tool::CBufferView<Vector3i> surf_triangles,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::BufferView<Vector2i>  temp_PPs,
        cuda_tool::BufferView<Vector3i>  temp_PEs,
        cuda_tool::BufferView<Vector4i>  temp_PTs,
        cuda_tool::CBufferView<Float>    d_hats,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto& PP = temp_PPs(i);
        PP.setConstant(-1);
        auto& PE = temp_PEs(i);
        PE.setConstant(-1);
        auto& PT = temp_PTs(i);
        PT.setConstant(-1);

        Vector2i indices = PT_pairs(i);
        IndexT   V       = surf_vertices(indices(0));
        Vector3i F       = surf_triangles(indices(1));

        Vector4i vIs  = {V, F(0), F(1), F(2)};
        Vector3  Ps[] = {
            positions(vIs(0)), positions(vIs(1)), positions(vIs(2)), positions(vIs(3))};

        Float thickness = PT_thickness(
            thicknesses(V), thicknesses(F(0)), thicknesses(F(1)), thicknesses(F(2)));

        Float d_hat = PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

        Vector4i flag =
            distance::point_triangle_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

        Vector2 range = D_range(thickness, d_hat);

        Float D;
        distance::point_triangle_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

        if constexpr(PrintKernelZeroDistance)
        {
            if(D <= range.x())
            {
                printf(
                    "[ISBVH][PT][low-dist] i=%d V-F=(%d,%d,%d,%d) "
                    "flag=(%d,%d,%d,%d) D=%e range=(%e,%e) thickness=%e d_hat=%e\n",
                    i,
                    vIs(0),
                    vIs(1),
                    vIs(2),
                    vIs(3),
                    flag(0),
                    flag(1),
                    flag(2),
                    flag(3),
                    D,
                    range.x(),
                    range.y(),
                    thickness,
                    d_hat);
            }
        }

        UIPC_KERNEL_ASSERT(
            D > 0.0, "D=%f, V F = (%d,%d,%d,%d)", D, vIs(0), vIs(1), vIs(2), vIs(3));

        UIPC_KERNEL_ASSERT(D > range.x(),
                           "Thickness Violated! D(%f) should be > D_range.x(%f), "
                           "V-F=(%d,%d,%d,%d), flag=(%d,%d,%d,%d), thickness=%f, d_hat=%f",
                           D,
                           range.x(),
                           vIs(0),
                           vIs(1),
                           vIs(2),
                           vIs(3),
                           flag(0),
                           flag(1),
                           flag(2),
                           flag(3),
                           thickness,
                           d_hat);
        if(!is_active_D(range, D))
            return;

        Vector4i offsets;
        auto     dim = distance::degenerate_point_triangle(flag, offsets);

        switch(dim)
        {
            case 2: {
                IndexT V0 = vIs(offsets(0));
                IndexT V1 = vIs(offsets(1));
                PP        = {V0, V1};
            }
            break;
            case 3: {
                IndexT V0 = vIs(offsets(0));
                IndexT V1 = vIs(offsets(1));
                IndexT V2 = vIs(offsets(2));
                PE        = {V0, V1, V2};
            }
            break;
            case 4: {
                PT = vIs;
            }
            break;
            default: {
                UIPC_KERNEL_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
            }
            break;
        }
    }

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k4_kernel(
        cuda_tool::CBufferView<Vector3>  positions,
        cuda_tool::CBufferView<Vector3>  rest_positions,
        cuda_tool::CBufferView<Vector2i> EE_pairs,
        cuda_tool::CBufferView<Vector2i> surf_edges,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::BufferView<Vector2i>  temp_PPs,
        cuda_tool::BufferView<Vector3i>  temp_PEs,
        cuda_tool::BufferView<Vector4i>  temp_EEs,
        cuda_tool::CBufferView<Float>    d_hats,
        int                              n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto& PP = temp_PPs(i);
        PP.setConstant(-1);
        auto& PE = temp_PEs(i);
        PE.setConstant(-1);
        auto& EE = temp_EEs(i);
        EE.setConstant(-1);

        Vector2i indices = EE_pairs(i);
        Vector2i E0      = surf_edges(indices(0));
        Vector2i E1      = surf_edges(indices(1));

        Vector4i vIs  = {E0(0), E0(1), E1(0), E1(1)};
        Vector3  Ps[] = {
            positions(vIs(0)), positions(vIs(1)), positions(vIs(2)), positions(vIs(3))};

        Float thickness = EE_thickness(thicknesses(E0(0)),
                                       thicknesses(E0(1)),
                                       thicknesses(E1(0)),
                                       thicknesses(E1(1)));

        Float d_hat =
            EE_d_hat(d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));

        Vector2 range = D_range(thickness, d_hat);

        Vector4i flag = distance::edge_edge_distance_flag(Ps[0], Ps[1], Ps[2], Ps[3]);

        Float D;
        distance::edge_edge_distance2(flag, Ps[0], Ps[1], Ps[2], Ps[3], D);

        if constexpr(PrintKernelZeroDistance)
        {
            if(D <= range.x())
            {
                printf(
                    "[ISBVH][EE][low-dist] i=%d E-E=(%d,%d,%d,%d) "
                    "flag=(%d,%d,%d,%d) D=%e range=(%e,%e) thickness=%e d_hat=%e\n",
                    i,
                    vIs(0),
                    vIs(1),
                    vIs(2),
                    vIs(3),
                    flag(0),
                    flag(1),
                    flag(2),
                    flag(3),
                    D,
                    range.x(),
                    range.y(),
                    thickness,
                    d_hat);
            }
        }
        if(D <= range.x())
        {
            EE = vIs;
            return;
        }
        if(!is_active_D(range, D))
            return;

        Float eps_x;
        distance::edge_edge_mollifier_threshold(rest_positions(vIs(0)),
                                                rest_positions(vIs(1)),
                                                rest_positions(vIs(2)),
                                                rest_positions(vIs(3)),
                                                static_cast<Float>(1e-3),
                                                eps_x);

        if(distance::need_mollify(Ps[0], Ps[1], Ps[2], Ps[3], eps_x))
        {
            EE = vIs;
            return;
        }
        else
        {
            Vector4i offsets;
            auto     dim = distance::degenerate_edge_edge(flag, offsets);

            switch(dim)
            {
                case 2: {
                    IndexT V0 = vIs(offsets(0));
                    IndexT V1 = vIs(offsets(1));
                    PP        = {V0, V1};
                }
                break;
                case 3: {
                    IndexT V0 = vIs(offsets(0));
                    IndexT V1 = vIs(offsets(1));
                    IndexT V2 = vIs(offsets(2));
                    PE        = {V0, V1, V2};
                }
                break;
                case 4: {
                    EE = vIs;
                }
                break;
                default: {
                    UIPC_KERNEL_ERROR_WITH_LOCATION("unexpected degenerate case dim=%d", dim);
                }
                break;
            }
        }
    }

    struct InfoStacklessBVHSimplexTrajectoryFilter_filter_active_PP_pred
    {
        CUB_RUNTIME_FUNCTION bool operator()(const Vector2i& PP) const
        {
            return PP(0) != -1;
        }
    };

    struct InfoStacklessBVHSimplexTrajectoryFilter_filter_active_PE_pred
    {
        CUB_RUNTIME_FUNCTION bool operator()(const Vector3i& PE) const
        {
            return PE(0) != -1;
        }
    };

    struct InfoStacklessBVHSimplexTrajectoryFilter_filter_active_PT_pred
    {
        CUB_RUNTIME_FUNCTION bool operator()(const Vector4i& PT) const
        {
            return PT(0) != -1;
        }
    };

    struct InfoStacklessBVHSimplexTrajectoryFilter_filter_active_EE_pred
    {
        CUB_RUNTIME_FUNCTION bool operator()(const Vector4i& EE) const
        {
            return EE(0) != -1;
        }
    };

    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_collect_query_counts_kernel(
        cuda_tool::CVarView<IndexT>   allp_codimp_count,
        cuda_tool::CVarView<IndexT>   codimp_alle_count,
        cuda_tool::CVarView<IndexT>   alle_alle_count,
        cuda_tool::CVarView<IndexT>   allp_allt_count,
        cuda_tool::CVarView<int>      allp_codimp_broad,
        cuda_tool::CVarView<int>      codimp_alle_broad,
        cuda_tool::CVarView<int>      alle_alle_broad,
        cuda_tool::CVarView<int>      allp_allt_broad,
        cuda_tool::BufferView<IndexT> counts)
    {
        if(blockIdx.x != 0 || threadIdx.x != 0)
            return;
        counts(0) = *allp_codimp_count;
        counts(1) = *codimp_alle_count;
        counts(2) = *alle_alle_count;
        counts(3) = *allp_allt_count;
        // perf/round5 (w2): the two-phase broad-stage capacity counters travel
        // in the same transfer; prepare_query_result used to fetch each of
        // them with its own blocking D2H.
        counts(4) = (IndexT)*allp_codimp_broad;
        counts(5) = (IndexT)*codimp_alle_broad;
        counts(6) = (IndexT)*alle_alle_broad;
        counts(7) = (IndexT)*allp_allt_broad;
    }

    /****************************************************
    *                   Filter TOI
    ****************************************************/

    // perf/round6 (s04): EarlyOut selects the ACCD first-pass early exit
    // (UIPC_CCD_EARLY_OUT=0 restores the old path); Stats accumulates the
    // ACCD diagnosis counters (UIPC_CCD_STATS=1). Both are template
    // parameters so the shipped instantiation carries neither a branch nor
    // an extra kernel argument register for them.
    // perf/round6 (s06): Compact additionally writes, per candidate, whether
    // the pair is provably OUT of the contact activation window over the
    // WHOLE swept step -- the verdict the ACCD can hand out for free because
    // it has already computed the distance and the motion bound. The
    // `keep` pointer is the LAST kernel parameter, so the `Compact = false`
    // instantiation keeps every other parameter at its old constant-bank
    // offset -- its SASS is byte-identical to main's. The verdict is stored
    // by the ACCD itself, at the point where it already holds the distance,
    // so nothing new is live across its loop (registers stay 218 / 184 and
    // the launch geometry with them). UIPC_CCD_COMPACT=0 = the old path.
    template <bool EarlyOut, bool Stats, bool Compact>
    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k1_kernel(
        cuda_tool::BufferView<Float>     PP_tois,
        cuda_tool::CBufferView<Vector2i> PCodimP_pairs,
        cuda_tool::CBufferView<IndexT>   codim_vertices,
        cuda_tool::CBufferView<IndexT>   surf_vertices,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::CBufferView<Vector3>  positions,
        cuda_tool::CBufferView<Vector3>  dxs,
        cuda_tool::CBufferView<Float>    d_hats,
        Float                            eta,
        Float                            alpha,
        int                              n,
        distance::CCDStatCounter*        stats,
        uint8_t*                         keep)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto   indices = PCodimP_pairs(i);
        IndexT V0      = surf_vertices(indices(0));
        IndexT V1      = codim_vertices(indices(1));

        Float thickness = PP_thickness(thicknesses(V0), thicknesses(V1));
        Float d_hat     = PP_d_hat(d_hats(V0), d_hats(V1));

        Vector3 VP0  = positions(V0);
        Vector3 VP1  = positions(V1);
        Vector3 dVP0 = alpha * dxs(V0);
        Vector3 dVP1 = alpha * dxs(V1);

        Float toi = large_enough_toi;

        bool faraway =
            !distance::point_point_ccd_broadphase(VP0, VP1, dVP0, dVP1, d_hat + thickness);

        if(faraway)
        {
            // the swept AABBs are more than `d_hat + thickness` apart on some
            // axis over the WHOLE step, so the pair is inactive everywhere
            // on it -- and `large_enough_toi` is already its toi.
            if constexpr(Compact)
                keep[i] = 0;
            PP_tois(i) = toi;
            return;
        }

        bool hit =
            distance::point_point_ccd<Float, EarlyOut, Stats, Compact>(
                VP0, VP1, dVP0, dVP1, eta, thickness, max_iter, toi, stats, d_hat, Compact ? keep + i : nullptr);

        if(!hit)
            toi = large_enough_toi;

        PP_tois(i) = toi;
    }

    // perf/round6 (s04): EarlyOut selects the ACCD first-pass early exit
    // (UIPC_CCD_EARLY_OUT=0 restores the old path); Stats accumulates the
    // ACCD diagnosis counters (UIPC_CCD_STATS=1). Both are template
    // parameters so the shipped instantiation carries neither a branch nor
    // an extra kernel argument register for them.
    // perf/round6 (s06): Compact additionally writes, per candidate, whether
    // the pair is provably OUT of the contact activation window over the
    // WHOLE swept step -- the verdict the ACCD can hand out for free because
    // it has already computed the distance and the motion bound. The
    // `keep` pointer is the LAST kernel parameter, so the `Compact = false`
    // instantiation keeps every other parameter at its old constant-bank
    // offset -- its SASS is byte-identical to main's. The verdict is stored
    // by the ACCD itself, at the point where it already holds the distance,
    // so nothing new is live across its loop (registers stay 218 / 184 and
    // the launch geometry with them). UIPC_CCD_COMPACT=0 = the old path.
    template <bool EarlyOut, bool Stats, bool Compact>
    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k2_kernel(
        cuda_tool::BufferView<Float>     PE_tois,
        cuda_tool::CBufferView<Vector2i> CodimP_AllE_pairs,
        cuda_tool::CBufferView<IndexT>   codim_vertices,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::CBufferView<Vector2i> surf_edges,
        cuda_tool::CBufferView<Vector3>  Ps,
        cuda_tool::CBufferView<Vector3>  dxs,
        cuda_tool::CBufferView<Float>    d_hats,
        Float                            eta,
        Float                            alpha,
        int                              n,
        distance::CCDStatCounter*        stats,
        uint8_t*                         keep)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto     indices = CodimP_AllE_pairs(i);
        IndexT   V       = codim_vertices(indices(0));
        Vector2i E       = surf_edges(indices(1));

        Float thickness =
            PE_thickness(thicknesses(V), thicknesses(E(0)), thicknesses(E(1)));
        Float d_hat = PE_d_hat(d_hats(V), d_hats(E(0)), d_hats(E(1)));

        Vector3 VP  = Ps(V);
        Vector3 dVP = alpha * dxs(V);

        Vector3 EP0  = Ps(E[0]);
        Vector3 EP1  = Ps(E[1]);
        Vector3 dEP0 = alpha * dxs(E[0]);
        Vector3 dEP1 = alpha * dxs(E[1]);

        Float toi = large_enough_toi;

        bool faraway = !distance::point_edge_ccd_broadphase(
            VP, EP0, EP1, dVP, dEP0, dEP1, d_hat + thickness);

        if(faraway)
        {
            if constexpr(Compact)
                keep[i] = 0;
            PE_tois(i) = toi;
            return;
        }

        bool hit = distance::point_edge_ccd<Float, EarlyOut, Stats, Compact>(
            VP, EP0, EP1, dVP, dEP0, dEP1, eta, thickness, max_iter, toi, stats, d_hat, Compact ? keep + i : nullptr);

        if(!hit)
            toi = large_enough_toi;

        PE_tois(i) = toi;
    }

    // perf/round6 (s04): EarlyOut selects the ACCD first-pass early exit
    // (UIPC_CCD_EARLY_OUT=0 restores the old path); Stats accumulates the
    // ACCD diagnosis counters (UIPC_CCD_STATS=1). Both are template
    // parameters so the shipped instantiation carries neither a branch nor
    // an extra kernel argument register for them.
    // perf/round6 (s06): Compact additionally writes, per candidate, whether
    // the pair is provably OUT of the contact activation window over the
    // WHOLE swept step -- the verdict the ACCD can hand out for free because
    // it has already computed the distance and the motion bound. The
    // `keep` pointer is the LAST kernel parameter, so the `Compact = false`
    // instantiation keeps every other parameter at its old constant-bank
    // offset -- its SASS is byte-identical to main's. The verdict is stored
    // by the ACCD itself, at the point where it already holds the distance,
    // so nothing new is live across its loop (registers stay 218 / 184 and
    // the launch geometry with them). UIPC_CCD_COMPACT=0 = the old path.
    template <bool EarlyOut, bool Stats, bool Compact>
    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k3_kernel(
        cuda_tool::BufferView<Float>     PT_tois,
        cuda_tool::CBufferView<Vector2i> PT_pairs,
        cuda_tool::CBufferView<IndexT>   surf_vertices,
        cuda_tool::CBufferView<Vector3i> surf_triangles,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::CBufferView<Vector3>  Ps,
        cuda_tool::CBufferView<Vector3>  dxs,
        cuda_tool::CBufferView<Float>    d_hats,
        Float                            eta,
        Float                            alpha,
        int                              n,
        distance::CCDStatCounter*        stats,
        uint8_t*                         keep)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto     indices = PT_pairs(i);
        IndexT   V       = surf_vertices(indices(0));
        Vector3i F       = surf_triangles(indices(1));

        Float thickness = PT_thickness(
            thicknesses(V), thicknesses(F(0)), thicknesses(F(1)), thicknesses(F(2)));
        Float d_hat = PT_d_hat(d_hats(V), d_hats(F(0)), d_hats(F(1)), d_hats(F(2)));

        Vector3 VP  = Ps(V);
        Vector3 dVP = alpha * dxs(V);

        Vector3 FP0 = Ps(F[0]);
        Vector3 FP1 = Ps(F[1]);
        Vector3 FP2 = Ps(F[2]);

        Vector3 dFP0 = alpha * dxs(F[0]);
        Vector3 dFP1 = alpha * dxs(F[1]);
        Vector3 dFP2 = alpha * dxs(F[2]);

        Float toi = large_enough_toi;

        bool faraway = !distance::point_triangle_ccd_broadphase(
            VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, d_hat + thickness);

        if(faraway)
        {
            if constexpr(Compact)
                keep[i] = 0;
            PT_tois(i) = toi;
            return;
        }

        bool hit = distance::point_triangle_ccd<Float, EarlyOut, Stats, Compact>(
            VP, FP0, FP1, FP2, dVP, dFP0, dFP1, dFP2, eta, thickness, max_iter, toi, stats, d_hat, Compact ? keep + i : nullptr);

        if(!hit)
            toi = large_enough_toi;

        PT_tois(i) = toi;
    }

    // perf/round6 (s04): EarlyOut selects the ACCD first-pass early exit
    // (UIPC_CCD_EARLY_OUT=0 restores the old path); Stats accumulates the
    // ACCD diagnosis counters (UIPC_CCD_STATS=1). Both are template
    // parameters so the shipped instantiation carries neither a branch nor
    // an extra kernel argument register for them.
    // perf/round6 (s06): Compact additionally writes, per candidate, whether
    // the pair is provably OUT of the contact activation window over the
    // WHOLE swept step -- the verdict the ACCD can hand out for free because
    // it has already computed the distance and the motion bound. The
    // `keep` pointer is the LAST kernel parameter, so the `Compact = false`
    // instantiation keeps every other parameter at its old constant-bank
    // offset -- its SASS is byte-identical to main's. The verdict is stored
    // by the ACCD itself, at the point where it already holds the distance,
    // so nothing new is live across its loop (registers stay 218 / 184 and
    // the launch geometry with them). UIPC_CCD_COMPACT=0 = the old path.
    template <bool EarlyOut, bool Stats, bool Compact>
    __global__ void InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k4_kernel(
        cuda_tool::BufferView<Float>     EE_tois,
        cuda_tool::CBufferView<Vector2i> EE_pairs,
        cuda_tool::CBufferView<Vector2i> surf_edges,
        cuda_tool::CBufferView<Float>    thicknesses,
        cuda_tool::CBufferView<Vector3>  Ps,
        cuda_tool::CBufferView<Vector3>  dxs,
        cuda_tool::CBufferView<Float>    d_hats,
        Float                            eta,
        Float                            alpha,
        int                              n,
        distance::CCDStatCounter*        stats,
        uint8_t*                         keep)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        auto     indices = EE_pairs(i);
        Vector2i E0      = surf_edges(indices(0));
        Vector2i E1      = surf_edges(indices(1));

        Float thickness = EE_thickness(thicknesses(E0(0)),
                                       thicknesses(E0(1)),
                                       thicknesses(E1(0)),
                                       thicknesses(E1(1)));

        Float d_hat =
            EE_d_hat(d_hats(E0(0)), d_hats(E0(1)), d_hats(E1(0)), d_hats(E1(1)));

        Vector3 EP0  = Ps(E0[0]);
        Vector3 EP1  = Ps(E0[1]);
        Vector3 dEP0 = alpha * dxs(E0[0]);
        Vector3 dEP1 = alpha * dxs(E0[1]);

        Vector3 EP2  = Ps(E1[0]);
        Vector3 EP3  = Ps(E1[1]);
        Vector3 dEP2 = alpha * dxs(E1[0]);
        Vector3 dEP3 = alpha * dxs(E1[1]);

        Float toi = large_enough_toi;

        bool faraway = !distance::edge_edge_ccd_broadphase(
            EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2, dEP3, d_hat + thickness);

        if(faraway)
        {
            if constexpr(Compact)
                keep[i] = 0;
            EE_tois(i) = toi;
            return;
        }

        bool hit = distance::edge_edge_ccd<Float, EarlyOut, Stats, Compact>(
            EP0, EP1, EP2, EP3, dEP0, dEP1, dEP2, dEP3, eta, thickness, max_iter, toi, stats, d_hat, Compact ? keep + i : nullptr);

        if(!hit)
            toi = large_enough_toi;

        EE_tois(i) = toi;
    }
}  // namespace

REGISTER_SIM_SYSTEM(InfoStacklessBVHSimplexTrajectoryFilter);

void InfoStacklessBVHSimplexTrajectoryFilter::do_build(BuildInfo&)
{
    auto& config = world().scene().config();
    auto  method = config.find<std::string>("collision_detection/method");
    if(method->view()[0] != "info_stackless_bvh")
    {
        throw SimSystemException("Info stackless BVH unused");
    }

    m_impl.query_counts.resize(8);
    m_impl.selected_counts.resize(4);

    // perf/kernels: BVH refit for the per-iteration trajectory detects
    const char* refit_env       = std::getenv("UIPC_BVH_REFIT");
    m_impl.bvh_refit_enabled    = !(refit_env && refit_env[0] == '0');
    const char* verify_env      = std::getenv("UIPC_BVH_REFIT_VERIFY");
    m_impl.bvh_refit_verify     = verify_env && verify_env[0] == '1';
    const char* cull_verify_env = std::getenv("UIPC_BVH_SELF_CULL_VERIFY");
    m_impl.bvh_self_cull_verify = cull_verify_env && cull_verify_env[0] == '1';
    const char* two_phase_verify_env = std::getenv("UIPC_BVH_TWO_PHASE_VERIFY");
    m_impl.bvh_two_phase_verify = two_phase_verify_env && two_phase_verify_env[0] == '1';

    // perf/round6 (s04): ACCD first-pass early exit + its diagnosis counters
    const char* ccd_early_env  = std::getenv("UIPC_CCD_EARLY_OUT");
    m_impl.ccd_early_out       = !(ccd_early_env && ccd_early_env[0] == '0');
    const char* ccd_stats_env  = std::getenv("UIPC_CCD_STATS");
    m_impl.ccd_stats           = ccd_stats_env && ccd_stats_env[0] == '1';
    m_impl.ccd_stat_buffer.resize(4 * CCD_STAT_SLOTS);
    m_impl.ccd_stat_buffer.fill(0);

    // perf/round6 (s06): the candidate compaction for filter_active
    const char* compact_env = std::getenv("UIPC_CCD_COMPACT");
    m_impl.ccd_compact      = !(compact_env && compact_env[0] == '0');
    const char* compact_verify_env = std::getenv("UIPC_CCD_COMPACT_VERIFY");
    m_impl.ccd_compact_verify = compact_verify_env && compact_verify_env[0] == '1';
    m_impl.compact_counts.resize(4);
}

// DIAGNOSTIC (UIPC_CCD_COMPACT_VERIFY, s06): the compacted array must be the
// ordered subsequence of the raw array picked out by `keep_flags` -- exactly,
// including its length and the order of its elements. Blocking readbacks;
// diagnostic path only.
void InfoStacklessBVHSimplexTrajectoryFilter::Impl::compact_verify_contents()
{
    const ThisBVH::QueryBuffer* srcs[4] = {&candidate_AllP_CodimP_pairs,
                                           &candidate_CodimP_AllE_pairs,
                                           &candidate_AllP_AllT_pairs,
                                           &candidate_AllE_AllE_pairs};
    const cuda_tool::DeviceBuffer<Vector2i>* dsts[4] = {&compact_AllP_CodimP_pairs,
                                                        &compact_CodimP_AllE_pairs,
                                                        &compact_AllP_AllT_pairs,
                                                        &compact_AllE_AllE_pairs};

    std::vector<uint8_t> flags;
    keep_flags.copy_to(flags);

    SizeT flag_offset = 0;
    ++compact_contents_calls;
    for(int k = 0; k < 4; ++k)
    {
        SizeT n = srcs[k]->size();
        if(n > 0)
        {
            std::vector<Vector2i> raw(n);
            cudaMemcpy(raw.data(), srcs[k]->view().data(), n * sizeof(Vector2i), cudaMemcpyDeviceToHost);
            std::vector<Vector2i> got;
            dsts[k]->copy_to(got);

            std::vector<Vector2i> want;
            want.reserve(got.size());
            for(SizeT i = 0; i < n; ++i)
                if(flags[flag_offset + i] != 0)
                    want.push_back(raw[i]);

            bool ok = want.size() == got.size();
            for(SizeT i = 0; ok && i < want.size(); ++i)
                ok = (want[i] == got[i]);
            if(!ok)
            {
                ++compact_contents_mismatches;
                logger::warn(
                    "[ccd_compact_verify] contents mismatch on array {}: raw={} flagged={} compacted={}",
                    k,
                    n,
                    want.size(),
                    got.size());
            }
        }
        flag_offset += n;
    }
    if(compact_contents_calls % 200 == 0)
    {
        logger::warn("[ccd_compact_verify] contents checks={} MISMATCHES={}",
                     compact_contents_calls,
                     compact_contents_mismatches);
    }
}

void InfoStacklessBVHSimplexTrajectoryFilter::do_detect(DetectInfo& info)
{
    m_impl.detect(info);
}

void InfoStacklessBVHSimplexTrajectoryFilter::do_filter_active(FilterActiveInfo& info)
{
    m_impl.filter_active(info);
}

void InfoStacklessBVHSimplexTrajectoryFilter::do_filter_toi(FilterTOIInfo& info)
{
    m_impl.filter_toi(info);
}

cuda_tool::CBufferView<Vector2i> InfoStacklessBVHSimplexTrajectoryFilter::candidate_PTs() const noexcept
{
    return m_impl.candidate_AllP_AllT_pairs.view();
}

cuda_tool::CBufferView<Vector2i> InfoStacklessBVHSimplexTrajectoryFilter::candidate_EEs() const noexcept
{
    return m_impl.candidate_AllE_AllE_pairs.view();
}

cuda_tool::CBufferView<Float> InfoStacklessBVHSimplexTrajectoryFilter::toi_PTs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    return m_impl.tois.view(pp_size + pe_size, pt_size);
}

cuda_tool::CBufferView<Float> InfoStacklessBVHSimplexTrajectoryFilter::toi_EEs() const noexcept
{
    auto pp_size = m_impl.candidate_AllP_CodimP_pairs.size();
    auto pe_size = m_impl.candidate_CodimP_AllE_pairs.size();
    auto pt_size = m_impl.candidate_AllP_AllT_pairs.size();
    auto ee_size = m_impl.candidate_AllE_AllE_pairs.size();
    return m_impl.tois.view(pp_size + pe_size + pt_size, ee_size);
}

void InfoStacklessBVHSimplexTrajectoryFilter::Impl::detect(DetectInfo& info)
{
    // perf/round6 (s06): a new detection replaces the candidate arrays, so
    // any compacted view of the previous ones is stale from here until the
    // filter_toi that follows.
    compact_ready = false;

    auto alpha                = info.alpha();
    auto Ps                   = info.positions();
    auto dxs                  = info.displacements();
    auto codimVs              = info.codim_vertices();
    auto Vs                   = info.surf_vertices();
    auto Es                   = info.surf_edges();
    auto Fs                   = info.surf_triangles();
    auto v2bs                 = info.v2b();
    auto body_self_collisions = info.body_self_collision();
    auto contact_element_ids  = info.contact_element_ids();
    auto cmts                 = info.contact_mask_tabular();

    point_aabbs.resize(Vs.size());
    triangle_aabbs.resize(Fs.size());
    edge_aabbs.resize(Es.size());
    point_bids.resize(Vs.size());
    triangle_bids.resize(Fs.size());
    edge_bids.resize(Es.size());
    point_cids.resize(Vs.size());
    triangle_cids.resize(Fs.size());
    edge_cids.resize(Es.size());

    // build AABBs for codim vertices
    if(codimVs.size() > 0)
    {
        codim_point_aabbs.resize(codimVs.size());
        codim_point_bids.resize(codimVs.size());
        codim_point_cids.resize(codimVs.size());

        int  n = static_cast<int>(codimVs.size());
        auto k = InfoStacklessBVHSimplexTrajectoryFilter_detect_k1_kernel;
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            codimVs,
            Ps,
            dxs,
            codim_point_aabbs.view(),
            v2bs,
            contact_element_ids,
            codim_point_bids.view(),
            codim_point_cids.view(),
            info.thicknesses(),
            info.d_hats(),
            alpha,
            n);
    }

    // build AABBs for surf vertices (including codim vertices)
    {
        int n = static_cast<int>(Vs.size());
        if(n > 0)
        {
            auto k = InfoStacklessBVHSimplexTrajectoryFilter_detect_k2_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                Vs,
                dxs,
                Ps,
                point_aabbs.view(),
                v2bs,
                contact_element_ids,
                point_bids.view(),
                point_cids.view(),
                info.thicknesses(),
                info.d_hats(),
                alpha,
                n);
        }
    }

    // build AABBs for edges
    {
        int n = static_cast<int>(Es.size());
        if(n > 0)
        {
            auto k = InfoStacklessBVHSimplexTrajectoryFilter_detect_k3_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                Es,
                Ps,
                edge_aabbs.view(),
                v2bs,
                contact_element_ids,
                edge_bids.view(),
                edge_cids.view(),
                dxs,
                info.thicknesses(),
                info.d_hats(),
                alpha,
                n);
        }
    }

    // build AABBs for triangles
    {
        int n = static_cast<int>(Fs.size());
        if(n > 0)
        {
            auto k = InfoStacklessBVHSimplexTrajectoryFilter_detect_k4_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                Fs,
                Ps,
                triangle_aabbs.view(),
                v2bs,
                contact_element_ids,
                triangle_bids.view(),
                triangle_cids.view(),
                dxs,
                info.thicknesses(),
                info.d_hats(),
                alpha,
                n);
        }
    }

    // perf/kernels: the hierarchy is rebuilt once per frame (the DCD call,
    // alpha == 0) and refitted for the per-iteration trajectory calls; the
    // candidate set is the same (every pair of overlapping boxes is reported
    // by any valid tree), only the pair order may differ. A rebuild is forced
    // every `refit_rebuild_every` refits to bound the quality loss.
    Timer      build_timer{"BVH Build/Refit"};  // perf/kernels diag
    const bool topo_ok =
        lbvh_E.can_refit(Es.size()) && lbvh_T.can_refit(Fs.size())
        && (codimVs.size() == 0 || lbvh_CodimP.can_refit(codimVs.size()));
    const bool do_refit = bvh_refit_enabled && topo_ok && alpha != 0.0
                          && refits_since_build < refit_rebuild_every;
    if(do_refit)
    {
        lbvh_E.refit(edge_aabbs, edge_bids, edge_cids);
        lbvh_T.refit(triangle_aabbs, triangle_bids, triangle_cids);
        ++refits_since_build;
    }
    else
    {
        lbvh_E.build(edge_aabbs, edge_bids, edge_cids);
        lbvh_T.build(triangle_aabbs, triangle_bids, triangle_cids);
        refits_since_build = 0;
        candidate_AllP_CodimP_pairs.invalidate();
        candidate_CodimP_AllE_pairs.invalidate();
        candidate_AllE_AllE_pairs.invalidate();
        candidate_AllP_AllT_pairs.invalidate();
    }
    auto node_pred = InfoStacklessBVHSimplexTrajectoryFilter_detect_node_pred{
        body_self_collisions, cmts.viewer()};
    auto allp_codimp_pred = InfoStacklessBVHSimplexTrajectoryFilter_detect_AllP_CodimP_pred{
        Vs,
        codimVs,
        Ps,
        dxs,
        info.thicknesses(),
        info.dimensions(),
        info.contact_element_ids(),
        info.contact_mask_tabular().viewer(),
        info.subscene_element_ids(),
        info.subscene_mask_tabular().viewer(),
        info.body_self_collision(),
        info.d_hats(),
        alpha};
    auto codimp_alle_pred = InfoStacklessBVHSimplexTrajectoryFilter_detect_CodimP_AllE_pred{
        codimVs,
        Es,
        Ps,
        dxs,
        info.thicknesses(),
        info.contact_element_ids(),
        info.contact_mask_tabular().viewer(),
        info.subscene_element_ids(),
        info.subscene_mask_tabular().viewer(),
        info.body_self_collision(),
        info.d_hats(),
        alpha};
    auto alle_alle_pred = InfoStacklessBVHSimplexTrajectoryFilter_detect_AllE_AllE_pred{
        Es,
        Ps,
        dxs,
        info.thicknesses(),
        info.contact_element_ids(),
        info.contact_mask_tabular().viewer(),
        info.subscene_element_ids(),
        info.subscene_mask_tabular().viewer(),
        info.body_self_collision(),
        info.d_hats(),
        alpha};
    auto allp_allt_pred = InfoStacklessBVHSimplexTrajectoryFilter_detect_AllP_AllT_pred{
        Vs,
        Fs,
        Ps,
        dxs,
        info.thicknesses(),
        info.contact_element_ids(),
        info.contact_mask_tabular().viewer(),
        info.subscene_element_ids(),
        info.subscene_mask_tabular().viewer(),
        info.body_self_collision(),
        info.d_hats(),
        alpha};

    auto launch_allp_codimp = [&](bool rebuild_query)
    {
        lbvh_CodimP.launch_query(point_aabbs,
                                 point_bids,
                                 point_cids,
                                 cmts,
                                 node_pred,
                                 allp_codimp_pred,
                                 candidate_AllP_CodimP_pairs,
                                 rebuild_query);
    };
    auto launch_codimp_alle = [&](bool rebuild_query)
    {
        lbvh_E.launch_query(codim_point_aabbs,
                            codim_point_bids,
                            codim_point_cids,
                            cmts,
                            node_pred,
                            codimp_alle_pred,
                            candidate_CodimP_AllE_pairs,
                            rebuild_query);
    };
    auto launch_alle_alle = [&]
    {
        lbvh_E.launch_detect(cmts, node_pred, alle_alle_pred, candidate_AllE_AllE_pairs);
    };
    auto launch_allp_allt = [&](bool rebuild_query)
    {
        lbvh_T.launch_query(
            point_aabbs, point_bids, point_cids, cmts, node_pred, allp_allt_pred, candidate_AllP_AllT_pairs, rebuild_query);
    };

    if(codimVs.size() > 0)
    {
        if(do_refit)
            lbvh_CodimP.refit(codim_point_aabbs, codim_point_bids, codim_point_cids);
        else
            lbvh_CodimP.build(codim_point_aabbs, codim_point_bids, codim_point_cids);
        launch_allp_codimp(true);
    }
    else
    {
        candidate_AllP_CodimP_pairs.m_cpNum.fill(0);
    }
    build_timer.~Timer();
    new(&build_timer) Timer{"BVH Query"};  // perf/kernels diag
    launch_codimp_alle(true);
    launch_alle_alle();
    launch_allp_allt(true);

    InfoStacklessBVHSimplexTrajectoryFilter_collect_query_counts_kernel<<<1, 1>>>(
        candidate_AllP_CodimP_pairs.m_cpNum.cview(),
        candidate_CodimP_AllE_pairs.m_cpNum.cview(),
        candidate_AllE_AllE_pairs.m_cpNum.cview(),
        candidate_AllP_AllT_pairs.m_cpNum.cview(),
        candidate_AllP_CodimP_pairs.m_broadNum.cview(),
        candidate_CodimP_AllE_pairs.m_broadNum.cview(),
        candidate_AllE_AllE_pairs.m_broadNum.cview(),
        candidate_AllP_AllT_pairs.m_broadNum.cview(),
        query_counts.view());

    std::array<IndexT, 8> host_counts{};
    query_counts.copy_to(host_counts.data());

    if(lbvh_CodimP.prepare_query_result(candidate_AllP_CodimP_pairs, host_counts[0], bvh_batch_counts_enabled() ? (int)host_counts[0+4] : -1))
        launch_allp_codimp(false);
    if(lbvh_E.prepare_query_result(candidate_CodimP_AllE_pairs, host_counts[1], bvh_batch_counts_enabled() ? (int)host_counts[1+4] : -1))
        launch_codimp_alle(false);
    if(lbvh_E.prepare_query_result(candidate_AllE_AllE_pairs, host_counts[2], bvh_batch_counts_enabled() ? (int)host_counts[2+4] : -1))
        launch_alle_alle();
    if(lbvh_T.prepare_query_result(candidate_AllP_AllT_pairs, host_counts[3], bvh_batch_counts_enabled() ? (int)host_counts[3+4] : -1))
        launch_allp_allt(false);

    // DIAGNOSTIC (env UIPC_BVH_SELF_CULL_VERIFY=1, K11): redo the edge-edge
    // self query without the sorted-range subtree cull and compare the
    // candidate sets on the host (order-agnostic). Logs a warning on any
    // difference; the culled result is restored afterwards.
    // (env UIPC_BVH_TWO_PHASE_VERIFY=1, s04): the same with the two-phase
    // query switch flipped, for the EE self query and the
    // point-triangle query (both traversal kernels).
    if(bvh_two_phase_verify)
    {
        auto snapshot = [&](InfoStacklessBVH::QueryBuffer& qb, std::vector<Vector2i>& out)
        {
            auto v = qb.view();
            out.resize(v.size());
            if(v.size() > 0)
                v.copy_to(out.data());
            std::sort(out.begin(),
                      out.end(),
                      [](const Vector2i& a, const Vector2i& b)
                      { return a.x() != b.x() ? a.x() < b.x() : a.y() < b.y(); });
        };
        auto rerun = [&]
        {
            launch_alle_alle();
            launch_allp_allt(false);
            InfoStacklessBVHSimplexTrajectoryFilter_collect_query_counts_kernel<<<1, 1>>>(
                candidate_AllP_CodimP_pairs.m_cpNum.cview(),
                candidate_CodimP_AllE_pairs.m_cpNum.cview(),
                candidate_AllE_AllE_pairs.m_cpNum.cview(),
                candidate_AllP_AllT_pairs.m_cpNum.cview(),
                candidate_AllP_CodimP_pairs.m_broadNum.cview(),
                candidate_CodimP_AllE_pairs.m_broadNum.cview(),
                candidate_AllE_AllE_pairs.m_broadNum.cview(),
                candidate_AllP_AllT_pairs.m_broadNum.cview(),
                query_counts.view());
            std::array<IndexT, 8> c{};
            query_counts.copy_to(c.data());
            if(lbvh_E.prepare_query_result(candidate_AllE_AllE_pairs, c[2], bvh_batch_counts_enabled() ? (int)c[2+4] : -1))
                launch_alle_alle();
            if(lbvh_T.prepare_query_result(candidate_AllP_AllT_pairs, c[3], bvh_batch_counts_enabled() ? (int)c[3+4] : -1))
                launch_allp_allt(false);
        };
        std::vector<Vector2i> ee_a, ee_b, pt_a, pt_b;
        snapshot(candidate_AllE_AllE_pairs, ee_a);
        snapshot(candidate_AllP_AllT_pairs, pt_a);
        const bool two_phase_was = lbvh_E.two_phase();
        lbvh_E.set_two_phase(!two_phase_was);
        lbvh_T.set_two_phase(!two_phase_was);
        rerun();
        snapshot(candidate_AllE_AllE_pairs, ee_b);
        snapshot(candidate_AllP_AllT_pairs, pt_b);
        lbvh_E.set_two_phase(two_phase_was);
        lbvh_T.set_two_phase(two_phase_was);
        rerun();
        if(ee_a != ee_b || pt_a != pt_b)
        {
            logger::warn("BVH two-phase verify alpha={} : EE {} vs {} pairs, PT {} vs {} pairs (two_phase={} vs {})",
                         alpha,
                         ee_a.size(),
                         ee_b.size(),
                         pt_a.size(),
                         pt_b.size(),
                         two_phase_was,
                         !two_phase_was);
            ++bvh_two_phase_verify_mismatches;
        }
        ++bvh_two_phase_verify_calls;
        if(bvh_two_phase_verify_calls % 200 == 0)
            logger::warn("BVH two-phase verify: {} calls, {} set mismatches so far",
                         bvh_two_phase_verify_calls,
                         bvh_two_phase_verify_mismatches);
    }
    if(bvh_self_cull_verify)
    {
        auto snapshot_ee = [&](std::vector<Vector2i>& out)
        {
            auto v = candidate_AllE_AllE_pairs.view();
            out.resize(v.size());
            if(v.size() > 0)
                v.copy_to(out.data());
            std::sort(out.begin(),
                      out.end(),
                      [](const Vector2i& a, const Vector2i& b)
                      { return a.x() != b.x() ? a.x() < b.x() : a.y() < b.y(); });
            out.erase(std::unique(out.begin(), out.end()), out.end());
        };
        auto rerun_ee = [&](bool cull)
        {
            lbvh_E.set_self_range_cull(cull);
            launch_alle_alle();
            InfoStacklessBVHSimplexTrajectoryFilter_collect_query_counts_kernel<<<1, 1>>>(
                candidate_AllP_CodimP_pairs.m_cpNum.cview(),
                candidate_CodimP_AllE_pairs.m_cpNum.cview(),
                candidate_AllE_AllE_pairs.m_cpNum.cview(),
                candidate_AllP_AllT_pairs.m_cpNum.cview(),
                candidate_AllP_CodimP_pairs.m_broadNum.cview(),
                candidate_CodimP_AllE_pairs.m_broadNum.cview(),
                candidate_AllE_AllE_pairs.m_broadNum.cview(),
                candidate_AllP_AllT_pairs.m_broadNum.cview(),
                query_counts.view());
            std::array<IndexT, 8> c{};
            query_counts.copy_to(c.data());
            if(lbvh_E.prepare_query_result(candidate_AllE_AllE_pairs, c[2], bvh_batch_counts_enabled() ? (int)c[2+4] : -1))
                launch_alle_alle();
        };
        std::vector<Vector2i> cull_set, full_set;
        snapshot_ee(cull_set);
        const bool cull_was = lbvh_E.self_range_cull();
        rerun_ee(!cull_was);
        snapshot_ee(full_set);
        rerun_ee(cull_was);
        if(cull_set != full_set)
        {
            logger::warn("BVH self cull verify [AllE-AllE] alpha={} : cull {} pairs, no-cull {} pairs",
                         alpha,
                         cull_set.size(),
                         full_set.size());
            ++bvh_self_cull_verify_mismatches;
        }
        ++bvh_self_cull_verify_calls;
        if(bvh_self_cull_verify_calls % 200 == 0)
            logger::warn("BVH self cull verify: {} calls, {} set mismatches so far",
                         bvh_self_cull_verify_calls,
                         bvh_self_cull_verify_mismatches);
    }

    // DIAGNOSTIC (env UIPC_BVH_REFIT_VERIFY=1): after a refit, rebuild the
    // trees from scratch, redo the queries and compare the candidate pair
    // sets on the host (order-agnostic). Logs a warning on any difference.
    if(do_refit && bvh_refit_verify)
    {
        auto snapshot = [&](std::array<std::vector<Vector2i>, 4>& out)
        {
            const InfoStacklessBVH::QueryBuffer* qs[4] = {&candidate_AllP_CodimP_pairs,
                                                          &candidate_CodimP_AllE_pairs,
                                                          &candidate_AllE_AllE_pairs,
                                                          &candidate_AllP_AllT_pairs};
            for(int k = 0; k < 4; ++k)
            {
                auto v = qs[k]->view();
                out[k].resize(v.size());
                if(v.size() > 0)
                    v.copy_to(out[k].data());
                std::sort(out[k].begin(),
                          out[k].end(),
                          [](const Vector2i& a, const Vector2i& b) {
                              return a.x() != b.x() ? a.x() < b.x() : a.y() < b.y();
                          });
                out[k].erase(std::unique(out[k].begin(), out[k].end()), out[k].end());
            }
        };
        std::array<std::vector<Vector2i>, 4> refit_sets, build_sets;
        snapshot(refit_sets);

        lbvh_E.build(edge_aabbs, edge_bids, edge_cids);
        lbvh_T.build(triangle_aabbs, triangle_bids, triangle_cids);
        if(codimVs.size() > 0)
            lbvh_CodimP.build(codim_point_aabbs, codim_point_bids, codim_point_cids);
        refits_since_build = 0;
        candidate_AllP_CodimP_pairs.invalidate();
        candidate_CodimP_AllE_pairs.invalidate();
        candidate_AllE_AllE_pairs.invalidate();
        candidate_AllP_AllT_pairs.invalidate();
        if(codimVs.size() > 0)
            launch_allp_codimp(true);
        else
            candidate_AllP_CodimP_pairs.m_cpNum.fill(0);
        launch_codimp_alle(true);
        launch_alle_alle();
        launch_allp_allt(true);
        InfoStacklessBVHSimplexTrajectoryFilter_collect_query_counts_kernel<<<1, 1>>>(
            candidate_AllP_CodimP_pairs.m_cpNum.cview(),
            candidate_CodimP_AllE_pairs.m_cpNum.cview(),
            candidate_AllE_AllE_pairs.m_cpNum.cview(),
            candidate_AllP_AllT_pairs.m_cpNum.cview(),
            candidate_AllP_CodimP_pairs.m_broadNum.cview(),
            candidate_CodimP_AllE_pairs.m_broadNum.cview(),
            candidate_AllE_AllE_pairs.m_broadNum.cview(),
            candidate_AllP_AllT_pairs.m_broadNum.cview(),
            query_counts.view());
        std::array<IndexT, 8> counts2{};
        query_counts.copy_to(counts2.data());
        if(lbvh_CodimP.prepare_query_result(candidate_AllP_CodimP_pairs, counts2[0], bvh_batch_counts_enabled() ? (int)counts2[0+4] : -1))
            launch_allp_codimp(false);
        if(lbvh_E.prepare_query_result(candidate_CodimP_AllE_pairs, counts2[1], bvh_batch_counts_enabled() ? (int)counts2[1+4] : -1))
            launch_codimp_alle(false);
        if(lbvh_E.prepare_query_result(candidate_AllE_AllE_pairs, counts2[2], bvh_batch_counts_enabled() ? (int)counts2[2+4] : -1))
            launch_alle_alle();
        if(lbvh_T.prepare_query_result(candidate_AllP_AllT_pairs, counts2[3], bvh_batch_counts_enabled() ? (int)counts2[3+4] : -1))
            launch_allp_allt(false);
        snapshot(build_sets);

        static const char* names[4] = {"AllP-CodimP", "CodimP-AllE", "AllE-AllE", "AllP-AllT"};
        for(int k = 0; k < 4; ++k)
        {
            if(refit_sets[k] != build_sets[k])
            {
                std::vector<Vector2i> only_refit, only_build;
                auto cmp = [](const Vector2i& a, const Vector2i& b)
                { return a.x() != b.x() ? a.x() < b.x() : a.y() < b.y(); };
                std::set_difference(refit_sets[k].begin(),
                                    refit_sets[k].end(),
                                    build_sets[k].begin(),
                                    build_sets[k].end(),
                                    std::back_inserter(only_refit),
                                    cmp);
                std::set_difference(build_sets[k].begin(),
                                    build_sets[k].end(),
                                    refit_sets[k].begin(),
                                    refit_sets[k].end(),
                                    std::back_inserter(only_build),
                                    cmp);
                logger::warn("BVH refit verify [{}] alpha={} : refit {} pairs, build {} pairs, only-refit {}, only-build {}",
                             names[k],
                             alpha,
                             refit_sets[k].size(),
                             build_sets[k].size(),
                             only_refit.size(),
                             only_build.size());
                ++bvh_refit_verify_mismatches;
            }
        }
        ++bvh_refit_verify_calls;
        if(bvh_refit_verify_calls % 200 == 0)
            logger::warn("BVH refit verify: {} calls, {} set mismatches so far",
                         bvh_refit_verify_calls,
                         bvh_refit_verify_mismatches);
    }
}

void InfoStacklessBVHSimplexTrajectoryFilter::Impl::filter_active(FilterActiveInfo& info)
{
    using namespace cuda_tool;

    auto positions = info.positions();

    // perf/round6 (s06): when the preceding filter_toi compacted the
    // candidate arrays, run over the survivors. The dropped pairs are
    // provably outside the activation window at EVERY point of the swept
    // step the line search samples, so each of them would have written only
    // invalid markers here; dropping them leaves the four selected active
    // sets as the identical sequences the old path produced.
    // UIPC_CCD_COMPACT=0 (or a filter_active with no filter_toi before it,
    // i.e. the frame's first DCD detection) takes the raw arrays.
    const bool use_compact = ccd_compact && compact_ready;

    const auto& c_PCoimP = compact_AllP_CodimP_pairs;
    const auto& c_CodimPE = compact_CodimP_AllE_pairs;
    const auto& c_PTs     = compact_AllP_AllT_pairs;
    const auto& c_EEs     = compact_AllE_AllE_pairs;

    CBufferView<Vector2i> PCoimP_pairs =
        use_compact ? c_PCoimP.view() : candidate_AllP_CodimP_pairs.view();
    CBufferView<Vector2i> CodimPE_pairs =
        use_compact ? c_CodimPE.view() : candidate_CodimP_AllE_pairs.view();
    CBufferView<Vector2i> PT_pairs =
        use_compact ? c_PTs.view() : candidate_AllP_AllT_pairs.view();
    CBufferView<Vector2i> EE_pairs =
        use_compact ? c_EEs.view() : candidate_AllE_AllE_pairs.view();

    // The whole pass, as a callable over the four candidate views, so
    // UIPC_CCD_COMPACT_VERIFY can run it twice -- once over the raw arrays,
    // once over the compacted ones -- and compare.
    std::array<IndexT, 4> last_counts{};
    auto run = [&](cuda_tool::CBufferView<Vector2i> PCoimP_pairs,
                   cuda_tool::CBufferView<Vector2i> CodimPE_pairs,
                   cuda_tool::CBufferView<Vector2i> PT_pairs,
                   cuda_tool::CBufferView<Vector2i> EE_pairs)
    {
    SizeT N_PCoimP  = PCoimP_pairs.size();
    SizeT N_CodimPE = CodimPE_pairs.size();
    SizeT N_PTs     = PT_pairs.size();
    SizeT N_EEs     = EE_pairs.size();

    temp_PPs.resize_discard(N_PCoimP + N_CodimPE + N_PTs + N_EEs);
    temp_PEs.resize_discard(N_CodimPE + N_PTs + N_EEs);

    temp_PTs.resize_discard(N_PTs);
    temp_EEs.resize_discard(N_EEs);

    SizeT temp_PP_offset = 0;
    SizeT temp_PE_offset = 0;

    // AllP and CodimP
    if(N_PCoimP > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PCoimP);

        int n = static_cast<int>(N_PCoimP);
        auto k = InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k1_kernel;
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            positions,
            PCoimP_pairs,
            info.surf_vertices(),
            info.codim_vertices(),
            info.thicknesses(),
            PP_view,
            info.d_hats(),
            n);

        temp_PP_offset += N_PCoimP;
    }
    // CodimP and AllE
    if(N_CodimPE > 0)
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_CodimPE);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_CodimPE);

        int n = static_cast<int>(N_CodimPE);
        auto k = InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k2_kernel;
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            positions,
            CodimPE_pairs,
            info.codim_vertices(),
            info.surf_edges(),
            info.thicknesses(),
            PP_view,
            PE_view,
            info.d_hats(),
            n);

        temp_PP_offset += N_CodimPE;
        temp_PE_offset += N_CodimPE;
    }

    // AllP and AllT
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_PTs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_PTs);

        int n = static_cast<int>(N_PTs);
        if(n > 0)
        {
            auto k = InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k3_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                positions,
                PT_pairs,
                info.surf_vertices(),
                info.surf_triangles(),
                info.thicknesses(),
                PP_view,
                PE_view,
                temp_PTs.view(),
                info.d_hats(),
                n);
        }

        temp_PP_offset += N_PTs;
        temp_PE_offset += N_PTs;
    }
    // AllE and AllE
    {
        auto PP_view = temp_PPs.view(temp_PP_offset, N_EEs);
        auto PE_view = temp_PEs.view(temp_PE_offset, N_EEs);

        int n = static_cast<int>(N_EEs);
        if(n > 0)
        {
            auto k = InfoStacklessBVHSimplexTrajectoryFilter_filter_active_k4_kernel;
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                positions,
                info.rest_positions(),
                EE_pairs,
                info.surf_edges(),
                info.thicknesses(),
                PP_view,
                PE_view,
                temp_EEs.view(),
                info.d_hats(),
                n);
        }

        temp_PP_offset += N_EEs;
        temp_PE_offset += N_EEs;
    }

    UIPC_ASSERT(temp_PP_offset == temp_PPs.size(), "size mismatch");
    UIPC_ASSERT(temp_PE_offset == temp_PEs.size(), "size mismatch");

    {
        PPs.resize_discard(temp_PPs.size());
        PEs.resize_discard(temp_PEs.size());
        PTs.resize_discard(temp_PTs.size());
        EEs.resize_discard(temp_EEs.size());

        DeviceSelect().If(temp_PPs.data(),
                          PPs.data(),
                          selected_counts.data(),
                          temp_PPs.size(),
                          InfoStacklessBVHSimplexTrajectoryFilter_filter_active_PP_pred{});

        DeviceSelect().If(temp_PEs.data(),
                          PEs.data(),
                          selected_counts.data() + 1,
                          temp_PEs.size(),
                          InfoStacklessBVHSimplexTrajectoryFilter_filter_active_PE_pred{});

        DeviceSelect().If(temp_PTs.data(),
                          PTs.data(),
                          selected_counts.data() + 2,
                          temp_PTs.size(),
                          InfoStacklessBVHSimplexTrajectoryFilter_filter_active_PT_pred{});

        DeviceSelect().If(temp_EEs.data(),
                          EEs.data(),
                          selected_counts.data() + 3,
                          temp_EEs.size(),
                          InfoStacklessBVHSimplexTrajectoryFilter_filter_active_EE_pred{});

        // `selected_counts` is resize(4) (do_build), and copy_to transfers
        // size() elements, so slots 4..7 were never written or read. s22
        // widened this array as collateral of widening `query_counts`.
        std::array<IndexT, 4> host_counts{};
        selected_counts.copy_to(host_counts.data());

        IndexT PP_count = host_counts[0];
        IndexT PE_count = host_counts[1];
        IndexT PT_count = host_counts[2];
        IndexT EE_count = host_counts[3];

        PPs.resize_discard(PP_count);
        PEs.resize_discard(PE_count);
        PTs.resize_discard(PT_count);
        EEs.resize_discard(EE_count);

        last_counts = {PP_count, PE_count, PT_count, EE_count};
    }
    };  // end of `run`

    // DIAGNOSTIC (env UIPC_CCD_COMPACT_VERIFY=1, s06): run the pass over the
    // RAW candidate arrays first, then over the compacted ones, and compare
    // the four active-set sizes. The compacted array is an order-preserving
    // SUBSEQUENCE of the raw one (cub::DeviceSelect::Flagged, checked
    // element-wise by compact_verify_contents()), so the compacted active set
    // is the subsequence of the raw active set restricted to surviving
    // candidates: the two are equal as sequences IF AND ONLY IF the four
    // counts agree. Equal counts on every launch is therefore a complete
    // proof that no active pair was dropped -- at the real line-search
    // positions, on the real population.
    if(use_compact && ccd_compact_verify)
    {
        run(candidate_AllP_CodimP_pairs.view(),
            candidate_CodimP_AllE_pairs.view(),
            candidate_AllP_AllT_pairs.view(),
            candidate_AllE_AllE_pairs.view());
        auto ref = last_counts;

        run(PCoimP_pairs, CodimPE_pairs, PT_pairs, EE_pairs);

        ++compact_verify_calls;
        compact_verify_pairs += candidate_AllP_CodimP_pairs.size()
                                + candidate_CodimP_AllE_pairs.size()
                                + candidate_AllP_AllT_pairs.size()
                                + candidate_AllE_AllE_pairs.size();
        compact_verify_dropped +=
            (candidate_AllP_CodimP_pairs.size() - PCoimP_pairs.size())
            + (candidate_CodimP_AllE_pairs.size() - CodimPE_pairs.size())
            + (candidate_AllP_AllT_pairs.size() - PT_pairs.size())
            + (candidate_AllE_AllE_pairs.size() - EE_pairs.size());
        for(int k = 0; k < 4; ++k)
        {
            compact_verify_active += ref[k];
            if(ref[k] != last_counts[k])
            {
                ++compact_verify_mismatches;
                logger::warn(
                    "[ccd_compact_verify] active-set size {} differs: raw={} compacted={} (call {})",
                    k,
                    ref[k],
                    last_counts[k],
                    compact_verify_calls);
            }
        }
        if(compact_verify_calls % 200 == 0)
        {
            logger::warn(
                "[ccd_compact_verify] calls={} candidates={} dropped={} ({:.2f} %) active_pairs={} MISMATCHES={}",
                compact_verify_calls,
                compact_verify_pairs,
                compact_verify_dropped,
                100.0 * (double)compact_verify_dropped / (double)std::max<SizeT>(compact_verify_pairs, 1),
                compact_verify_active,
                compact_verify_mismatches);
        }
    }
    else
    {
        run(PCoimP_pairs, CodimPE_pairs, PT_pairs, EE_pairs);
    }

    info.PPs(PPs);
    info.PEs(PEs);
    info.PTs(PTs);
    info.EEs(EEs);

    if constexpr(PrintDebugInfo)
    {
        std::vector<Vector2i> PPs_host;
        std::vector<Float>    PP_thicknesses_host;

        std::vector<Vector3i> PEs_host;
        std::vector<Float>    PE_thicknesses_host;

        std::vector<Vector4i> PTs_host;
        std::vector<Float>    PT_thicknesses_host;

        std::vector<Vector4i> EEs_host;
        std::vector<Float>    EE_thicknesses_host;

        PPs.copy_to(PPs_host);
        PEs.copy_to(PEs_host);
        PTs.copy_to(PTs_host);
        EEs.copy_to(EEs_host);

        std::cout << "filter result:" << std::endl;

        for(auto&& [PP, thickness] : zip(PPs_host, PP_thicknesses_host))
        {
            std::cout << "PP: " << PP.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PE, thickness] : zip(PEs_host, PE_thicknesses_host))
        {
            std::cout << "PE: " << PE.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [PT, thickness] : zip(PTs_host, PT_thicknesses_host))
        {
            std::cout << "PT: " << PT.transpose() << " thickness: " << thickness << "\n";
        }

        for(auto&& [EE, thickness] : zip(EEs_host, EE_thicknesses_host))
        {
            std::cout << "EE: " << EE.transpose() << " thickness: " << thickness << "\n";
        }

        std::cout << std::flush;
    }
}

// perf/round6 (s06): the three compile-time flags of the CCD narrow phase
// (s04's EarlyOut, s04's Stats, s06's Compact) select an instantiation each,
// so the shipped kernel carries no branch and no extra register for any of
// them. Spelled as a macro because a function template cannot take a
// function template as a template argument.
#define UIPC_S06_TOI_DISPATCH(kname)                                            \
    do                                                                          \
    {                                                                           \
        if(cp)                                                                  \
        {                                                                       \
            if(eo && st)                                                        \
                launch(kname<true, true, true>);                                \
            else if(eo)                                                         \
                launch(kname<true, false, true>);                               \
            else if(st)                                                         \
                launch(kname<false, true, true>);                               \
            else                                                                \
                launch(kname<false, false, true>);                              \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            if(eo && st)                                                        \
                launch(kname<true, true, false>);                               \
            else if(eo)                                                         \
                launch(kname<true, false, false>);                              \
            else if(st)                                                         \
                launch(kname<false, true, false>);                              \
            else                                                                \
                launch(kname<false, false, false>);                             \
        }                                                                       \
    } while(0)

void InfoStacklessBVHSimplexTrajectoryFilter::Impl::filter_toi(FilterTOIInfo& info)
{
    using namespace cuda_tool;

    auto toi_size =
        candidate_AllP_CodimP_pairs.size() + candidate_CodimP_AllE_pairs.size()
        + candidate_AllP_AllT_pairs.size() + candidate_AllE_AllE_pairs.size();

    tois.resize_discard(toi_size);

    auto offset  = 0;
    auto PP_tois = tois.view(offset, candidate_AllP_CodimP_pairs.size());
    offset += candidate_AllP_CodimP_pairs.size();
    auto PE_tois = tois.view(offset, candidate_CodimP_AllE_pairs.size());
    offset += candidate_CodimP_AllE_pairs.size();
    auto PT_tois = tois.view(offset, candidate_AllP_AllT_pairs.size());
    offset += candidate_AllP_AllT_pairs.size();
    auto EE_tois = tois.view(offset, candidate_AllE_AllE_pairs.size());
    offset += candidate_AllE_AllE_pairs.size();

    UIPC_ASSERT(offset == toi_size, "size mismatch");

    // perf/round6 (s06): one keep flag per candidate, laid out exactly like
    // `tois`. Written by every thread of every filter_toi kernel on both of
    // its exits, so the array is fully defined for the compaction below.
    const bool cp = ccd_compact;
    compact_ready = false;
    uint8_t *PP_keep = nullptr, *PE_keep = nullptr, *PT_keep = nullptr, *EE_keep = nullptr;
    if(cp)
    {
        keep_flags.resize_discard(toi_size);
        auto koff = 0;
        PP_keep   = keep_flags.data() + koff;
        koff += candidate_AllP_CodimP_pairs.size();
        PE_keep = keep_flags.data() + koff;
        koff += candidate_CodimP_AllE_pairs.size();
        PT_keep = keep_flags.data() + koff;
        koff += candidate_AllP_AllT_pairs.size();
        EE_keep = keep_flags.data() + koff;
    }

    // AllP and CodimP
    {
        int n = static_cast<int>(candidate_AllP_CodimP_pairs.size());
        if(n > 0)
        {
            const bool eo = ccd_early_out;
            const bool st = ccd_stats;
            distance::CCDStatCounter* stats =
                st ? ccd_stat_buffer.data() + 0 * CCD_STAT_SLOTS : nullptr;
            auto launch = [&](auto k)
            {
                k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                PP_tois,
                candidate_AllP_CodimP_pairs.view(),
                info.codim_vertices(),
                info.surf_vertices(),
                info.thicknesses(),
                info.positions(),
                info.displacements(),
                info.d_hats(),
                info.toi_safety_margin(),
                info.alpha(),
                n,
                stats,
                PP_keep);
            };
            UIPC_S06_TOI_DISPATCH(InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k1_kernel);
        }
    }

    // CodimP and AllE
    {
        int n = static_cast<int>(candidate_CodimP_AllE_pairs.size());
        if(n > 0)
        {
            const bool eo = ccd_early_out;
            const bool st = ccd_stats;
            distance::CCDStatCounter* stats =
                st ? ccd_stat_buffer.data() + 1 * CCD_STAT_SLOTS : nullptr;
            auto launch = [&](auto k)
            {
                k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                PE_tois,
                candidate_CodimP_AllE_pairs.view(),
                info.codim_vertices(),
                info.thicknesses(),
                info.surf_edges(),
                info.positions(),
                info.displacements(),
                info.d_hats(),
                info.toi_safety_margin(),
                info.alpha(),
                n,
                stats,
                PE_keep);
            };
            UIPC_S06_TOI_DISPATCH(InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k2_kernel);
        }
    }

    // AllP and AllT
    {
        int n = static_cast<int>(candidate_AllP_AllT_pairs.size());
        if(n > 0)
        {
            const bool eo = ccd_early_out;
            const bool st = ccd_stats;
            distance::CCDStatCounter* stats =
                st ? ccd_stat_buffer.data() + 2 * CCD_STAT_SLOTS : nullptr;
            auto launch = [&](auto k)
            {
                k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                PT_tois,
                candidate_AllP_AllT_pairs.view(),
                info.surf_vertices(),
                info.surf_triangles(),
                info.thicknesses(),
                info.positions(),
                info.displacements(),
                info.d_hats(),
                info.toi_safety_margin(),
                info.alpha(),
                n,
                stats,
                PT_keep);
            };
            UIPC_S06_TOI_DISPATCH(InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k3_kernel);
        }
    }

    // AllE and AllE
    {
        int n = static_cast<int>(candidate_AllE_AllE_pairs.size());
        if(n > 0)
        {
            const bool eo = ccd_early_out;
            const bool st = ccd_stats;
            distance::CCDStatCounter* stats =
                st ? ccd_stat_buffer.data() + 3 * CCD_STAT_SLOTS : nullptr;
            auto launch = [&](auto k)
            {
                k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                EE_tois,
                candidate_AllE_AllE_pairs.view(),
                info.surf_edges(),
                info.thicknesses(),
                info.positions(),
                info.displacements(),
                info.d_hats(),
                info.toi_safety_margin(),
                info.alpha(),
                n,
                stats,
                EE_keep);
            };
            UIPC_S06_TOI_DISPATCH(InfoStacklessBVHSimplexTrajectoryFilter_filter_toi_k4_kernel);
        }
    }

    // perf/round6 (s06): order-preserving stream compaction of the four
    // candidate arrays, keyed on the flags the kernels above just wrote.
    // cub::DeviceSelect::Flagged preserves the relative order of the selected
    // items, and every dropped pair is one `filter_active` would have written
    // its invalid marker for, so the active sets its own DeviceSelect::If
    // produces downstream are the SAME SEQUENCES as on the old path.
    // The raw candidate arrays and `tois` are untouched.
    if(cp)
    {
        bool selected[4] = {false, false, false, false};
        auto compact_one = [&](const ThisBVH::QueryBuffer&       src,
                               cuda_tool::DeviceBuffer<Vector2i>& dst,
                               const uint8_t*                     flags,
                               int                                slot)
        {
            int m = static_cast<int>(src.size());
            dst.resize_discard(m);
            if(m == 0)
                return;
            DeviceSelect().Flagged(
                src.view().data(), flags, dst.data(), compact_counts.data() + slot, m);
            selected[slot] = true;
        };

        compact_one(candidate_AllP_CodimP_pairs, compact_AllP_CodimP_pairs, PP_keep, 0);
        compact_one(candidate_CodimP_AllE_pairs, compact_CodimP_AllE_pairs, PE_keep, 1);
        compact_one(candidate_AllP_AllT_pairs, compact_AllP_AllT_pairs, PT_keep, 2);
        compact_one(candidate_AllE_AllE_pairs, compact_AllE_AllE_pairs, EE_keep, 3);

        // One batched D2H for the four counts. It lands immediately before
        // GlobalTrajectoryFilter::filter_toi's own blocking read of the per
        // filter toi, so it costs no additional pipeline drain.
        std::array<IndexT, 4> h{};
        compact_counts.copy_to(h.data());
        for(int k = 0; k < 4; ++k)
            compact_sizes[k] = selected[k] ? h[k] : 0;

        compact_AllP_CodimP_pairs.resize_discard(compact_sizes[0]);
        compact_CodimP_AllE_pairs.resize_discard(compact_sizes[1]);
        compact_AllP_AllT_pairs.resize_discard(compact_sizes[2]);
        compact_AllE_AllE_pairs.resize_discard(compact_sizes[3]);
        compact_ready = true;

        if(ccd_compact_verify)
            compact_verify_contents();
    }

    if(tois.size())
    {
        DeviceReduce().Min(tois.data(), info.toi().data(), tois.size());
    }
    else
    {
        info.toi().fill(large_enough_toi);
    }

    // DIAGNOSTIC (env UIPC_CCD_STATS=1, s04): how much of the ACCD narrow
    // phase is spent on pairs that cannot collide within the step. Blocking
    // readback, diagnostic path only.
    if(ccd_stats)
    {
        ++ccd_stats_calls;
        if(ccd_stats_calls % 200 == 0)
        {
            std::array<distance::CCDStatCounter, 4 * CCD_STAT_SLOTS> h{};
            ccd_stat_buffer.view().copy_to(h.data());
            const char* names[4] = {"PP", "PE", "PT", "EE"};
            for(int t = 0; t < 4; ++t)
            {
                auto* r = h.data() + t * CCD_STAT_SLOTS;
                if(r[0] == 0)
                    continue;
                logger::warn(
                    "CCD stats [{} calls] {}: accd_calls={} first_pass_exits={} ({:.2f} %) loop_passes={} ({:.3f}/call) converged_hits={} ({:.4f} %)",
                    ccd_stats_calls,
                    names[t],
                    r[0],
                    r[1],
                    100.0 * (double)r[1] / (double)r[0],
                    r[2],
                    (double)r[2] / (double)r[0],
                    r[3],
                    100.0 * (double)r[3] / (double)r[0]);
            }
        }
    }
}

#undef UIPC_S06_TOI_DISPATCH
}  // namespace uipc::backend::cuda

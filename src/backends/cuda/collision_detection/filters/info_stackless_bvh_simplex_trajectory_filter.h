#pragma once
#include <sim_system.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/info_stackless_bvh.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <array>
#include <vector>

namespace uipc::backend::cuda
{
class InfoStacklessBVHSimplexTrajectoryFilter final : public SimplexTrajectoryFilter
{
  public:
    using SimplexTrajectoryFilter::SimplexTrajectoryFilter;

    class Impl
    {
      public:
        void detect(DetectInfo& info);
        void filter_active(FilterActiveInfo& info);
        void filter_toi(FilterTOIInfo& info);

        /****************************************************
        *                   Broad Phase
        ****************************************************/

        cuda_tool::DeviceBuffer<AABB>   codim_point_aabbs;
        cuda_tool::DeviceBuffer<AABB>   point_aabbs;
        cuda_tool::DeviceBuffer<AABB>   edge_aabbs;
        cuda_tool::DeviceBuffer<AABB>   triangle_aabbs;
        cuda_tool::DeviceBuffer<IndexT> codim_point_bids;
        cuda_tool::DeviceBuffer<IndexT> point_bids;
        cuda_tool::DeviceBuffer<IndexT> edge_bids;
        cuda_tool::DeviceBuffer<IndexT> triangle_bids;
        cuda_tool::DeviceBuffer<IndexT> codim_point_cids;
        cuda_tool::DeviceBuffer<IndexT> point_cids;
        cuda_tool::DeviceBuffer<IndexT> edge_cids;
        cuda_tool::DeviceBuffer<IndexT> triangle_cids;

        using ThisBVH = InfoStacklessBVH;

        // CodimP count always less or equal to AllP count.
        ThisBVH              lbvh_CodimP;
        ThisBVH::QueryBuffer candidate_AllP_CodimP_pairs;

        // Used to detect CodimP-AllE, and AllE-AllE pairs.
        ThisBVH              lbvh_E;
        ThisBVH::QueryBuffer candidate_CodimP_AllE_pairs;
        ThisBVH::QueryBuffer candidate_AllE_AllE_pairs;

        // Used to detect AllP-AllT pairs.
        ThisBVH              lbvh_T;
        ThisBVH::QueryBuffer candidate_AllP_AllT_pairs;
        // perf/kernels: refit instead of rebuild for the trajectory detects
        bool bvh_refit_enabled = true;  // env UIPC_BVH_REFIT=0 disables
        bool bvh_refit_verify = false;  // env UIPC_BVH_REFIT_VERIFY=1: rebuild + compare sets
        bool bvh_self_cull_verify = false;  // env UIPC_BVH_SELF_CULL_VERIFY=1 (K11): EE self query with/without the range cull
        bool bvh_two_phase_verify = false;  // env UIPC_BVH_TWO_PHASE_VERIFY=1 (s04): EE/PT queries with and without the two-phase split
        SizeT bvh_two_phase_verify_calls      = 0;
        SizeT bvh_two_phase_verify_mismatches = 0;
        SizeT bvh_self_cull_verify_calls      = 0;
        SizeT bvh_self_cull_verify_mismatches = 0;
        int   refit_rebuild_every             = 64;
        int   refits_since_build              = 0;
        SizeT bvh_refit_verify_calls          = 0;
        SizeT bvh_refit_verify_mismatches     = 0;

        // Four query counts and four compacted PP/PE/PT/EE counts are each
        // downloaded in one contiguous transfer.
        cuda_tool::DeviceBuffer<IndexT> query_counts;
        cuda_tool::DeviceBuffer<IndexT> selected_counts;

        // ---- certified-reuse verification staging
        // (extras/debug/dcd_candidate_reuse_verify, defined in
        // candidate_reuse_verify.cu): host copies of the raw broadphase
        // candidate sets, order PP, PE, EE, PT. ----
        std::array<std::vector<Vector2i>, 4> reuse_snapshots;

        cuda_tool::DeviceBuffer<Vector4i> temp_PTs;
        cuda_tool::DeviceBuffer<Vector4i> temp_EEs;
        cuda_tool::DeviceBuffer<Vector3i> temp_PEs;
        cuda_tool::DeviceBuffer<Vector2i> temp_PPs;

        cuda_tool::DeviceBuffer<Vector4i> PTs;
        cuda_tool::DeviceBuffer<Vector4i> EEs;
        cuda_tool::DeviceBuffer<Vector3i> PEs;
        cuda_tool::DeviceBuffer<Vector2i> PPs;


        /****************************************************
        *                   CCD TOI
        ****************************************************/

        cuda_tool::DeviceBuffer<Float> tois;  // PP, PE, PT, EE

        // perf/round6 (s04): the ACCD first-pass early exit and its
        // diagnosis counters. `ccd_early_out` selects a template
        // instantiation, not a branch inside the kernel.
        bool  ccd_early_out = true;   // env UIPC_CCD_EARLY_OUT=0 = old path
        bool  ccd_stats     = false;  // env UIPC_CCD_STATS=1
        SizeT ccd_stats_calls = 0;
        // 4 pair types x CCD_STAT_SLOTS (calls, early exits, loop passes, hits)
        // (distance::CCDStatCounter; spelled out so this header does not
        //  have to pull in ccd.h ahead of the distance declarations it needs)
        cuda_tool::DeviceBuffer<unsigned long long> ccd_stat_buffer;
    };

    virtual cuda_tool::CBufferView<Vector2i> candidate_PTs() const noexcept override;
    virtual cuda_tool::CBufferView<Vector2i> candidate_EEs() const noexcept override;
    virtual cuda_tool::CBufferView<Float> toi_PTs() const noexcept override;
    virtual cuda_tool::CBufferView<Float> toi_EEs() const noexcept override;

    // DIAGNOSTIC (extras/debug/dcd_candidate_reuse_verify): snapshot the raw
    // broadphase candidate sets / verify + restore them against a fresh DCD
    // detection. Defined in candidate_reuse_verify.cu; no-op cost when the
    // flag is off (never called).
    void  reuse_candidates_snapshot() noexcept;
    SizeT reuse_candidates_verify(SizeT frame, SizeT newton_iter) noexcept;

  private:
    Impl m_impl;

    virtual void do_build(BuildInfo& info) override final;
    virtual void do_detect(DetectInfo& info) override final;
    virtual void do_filter_active(FilterActiveInfo& info) override final;
    virtual void do_filter_toi(FilterTOIInfo& info) override final;
};
}  // namespace uipc::backend::cuda

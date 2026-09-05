// DIAGNOSTIC-ONLY certification verifier for the certified DCD candidate
// reuse (default off, extras/debug/dcd_candidate_reuse_verify). See
// scene_default_config.cpp and the certification comment in
// src/backends/cuda/engine/advance_ipc.cu.
//
// With collision_detection/dcd_candidate_reuse active, the engine keeps the
// raw broadphase candidate set produced by the previous line-search
// trajectory detection for every Newton iteration past the first, instead of
// re-running the DCD detection. The certification invariant is that this
// reused set contains every pair a fresh DCD detection at the current
// positions would report (then filter_active reports the identical active
// set either way). This file checks that invariant at runtime:
//
//   1. reuse_candidates_snapshot(): download the four raw candidate sets
//      (PP, PE, EE, PT broadphase index pairs) BEFORE the fresh detection.
//   2. the engine then runs the ordinary fresh DCD detection, which
//      overwrites the raw sets.
//   3. reuse_candidates_verify(): download the fresh sets, check on the host
//      that every fresh pair is contained in the snapshotted reused set,
//      log one line per iteration with the per-type sizes
//      (reused/fresh/missing), restore the reused sets on the device so the
//      run continues exactly as the plain reuse path would, and return the
//      number of violations (0 = invariant holds).
//
// All work is host-side after device downloads and only runs when the flag
// is set; it is never part of the hot path.
#include <collision_detection/filters/info_stackless_bvh_simplex_trajectory_filter.h>
#include <cuda_tool/cuda_tool.h>
#include <uipc/common/log.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace uipc::backend::cuda
{
namespace
{
    // The broadphase pair buffers hold index pairs into the surface
    // vertex/edge/triangle lists (Vector2i, both components < 2^31), so a
    // 64-bit packing is collision-free.
    uint64_t pack_pair(const Vector2i& p)
    {
        return (static_cast<uint64_t>(static_cast<uint32_t>(p(0))) << 32)
               | static_cast<uint64_t>(static_cast<uint32_t>(p(1)));
    }

    void download(InfoStacklessBVH::QueryBuffer& q, std::vector<Vector2i>& out)
    {
        out.resize(q.size());
        if(q.size() > 0)
            q.view().copy_to(out.data());
    }
}  // namespace

void InfoStacklessBVHSimplexTrajectoryFilter::reuse_candidates_snapshot() noexcept
{
    auto& I = m_impl;
    download(I.candidate_AllP_CodimP_pairs, I.reuse_snapshots[0]);
    download(I.candidate_CodimP_AllE_pairs, I.reuse_snapshots[1]);
    download(I.candidate_AllE_AllE_pairs, I.reuse_snapshots[2]);
    download(I.candidate_AllP_AllT_pairs, I.reuse_snapshots[3]);
}

SizeT InfoStacklessBVHSimplexTrajectoryFilter::reuse_candidates_verify(
    SizeT frame, SizeT newton_iter) noexcept
{
    auto& I = m_impl;

    InfoStacklessBVH::QueryBuffer* qbs[4] = {&I.candidate_AllP_CodimP_pairs,
                                             &I.candidate_CodimP_AllE_pairs,
                                             &I.candidate_AllE_AllE_pairs,
                                             &I.candidate_AllP_AllT_pairs};
    const char* names[4]                  = {"PP", "PE", "EE", "PT"};

    SizeT     total_reused  = 0;
    SizeT     total_fresh   = 0;
    SizeT     total_missing = 0;
    std::string per_type;

    for(int t = 0; t < 4; ++t)
    {
        auto&                     snap = I.reuse_snapshots[t];
        InfoStacklessBVH::QueryBuffer* q   = qbs[t];

        std::vector<Vector2i> fresh;
        download(*q, fresh);

        std::unordered_set<uint64_t> reused_keys;
        reused_keys.reserve(2 * snap.size() + 1);
        for(const auto& p : snap)
            reused_keys.insert(pack_pair(p));

        SizeT    missing = 0;
        Vector2i first_missing{-1, -1};
        for(const auto& p : fresh)
        {
            if(reused_keys.find(pack_pair(p)) == reused_keys.end())
            {
                if(missing == 0)
                    first_missing = p;
                ++missing;
            }
        }

        per_type += std::string{" "} + names[t] + ": " + std::to_string(snap.size())
                    + "/" + std::to_string(fresh.size()) + " (-"
                    + std::to_string(missing) + ")";
        total_reused += snap.size();
        total_fresh += fresh.size();
        total_missing += missing;

        if(missing > 0)
        {
            logger::error("[DCDReuseVerify] VIOLATION f={} k={} type={} pair=({},{}) "
                          "is in the fresh DCD set but not in the reused one",
                          frame,
                          newton_iter,
                          names[t],
                          first_missing(0),
                          first_missing(1));
        }

        // Restore the reused set so the verification pass leaves the pipeline
        // in the exact state the plain reuse path would have. (The next
        // writer of these buffers is the line-search trajectory detection,
        // but restoring keeps the diagnostic side-effect-free by
        // construction: QueryBuffer capacity only ever grows, so the
        // snapshot always fits.)
        q->m_pairs.resize_discard(snap.size());
        if(snap.size() > 0)
            q->m_pairs.copy_from(snap.data(), snap.size());
        q->m_size = snap.size();
    }

    logger::info("[DCDReuseVerify] f={} k={} reused={} fresh={} missing={} |{}",
                 frame,
                 newton_iter,
                 total_reused,
                 total_fresh,
                 total_missing,
                 per_type);

    return total_missing;
}
}  // namespace uipc::backend::cuda

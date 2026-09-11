// DIAGNOSTIC-ONLY per-pair candidate dump for the NN-acceleration research
// (default off, extras/debug/dump_candidates). See scene_default_config.cpp.
//
// After every DCD candidate detection (frame start + every Newton iteration
// > 0), appends one record group to $UIPC_ORACLE_DIR/candidates.bin:
//
//   file header (once):  int32 magic=0x43414E44, int32 version=1,
//                        int32 record_bytes=44, int32 vertex_slots=4
//   per detection:       int32 frame, iter, nPT, nEE, nPE, nPP
//                        followed by nPT+nEE+nPE+nPP records, PT records
//                        first, then EE, PE, PP (the type is implicit in the
//                        position within the group).
//
// Record (44 bytes, host endianess, all floats are IEEE binary32):
//   int32  v[4]   vertex ids of the pair, -1-padded (PT: p,F0,F1,F2;
//                 EE: e0a,e0b,e1a,e1b; PE: p,e0,e1,-1; PP: p0,p1,-1,-1)
//   float  dist2  squared distance of the pair (same flagged-distance
//                 functions the narrowphase uses, evaluated on the host)
//   float  disp_a max |dx| since the previous Newton iteration over the
//                 vertices of the FIRST primitive
//   float  disp_b same for the SECOND primitive
//   float  d_hat  the pair's d_hat
//   float  thick  the pair's thickness
//   int32  body_a vertex->body id of the first vertex (same for body_b)
//
// The intent is offline training of a candidate-persistence predictor; it is
// never part of the hot path (all work happens on the host after a device
// download, and only when the flag is on).
#include <collision_detection/simplex_trajectory_filter.h>
#include <cuda_tool/cuda_tool.h>
#include <utils/distance.h>
#include <utils/distance/distance_flagged.h>
#include <utils/primitive_d_hat.h>
#include <utils/codim_thickness.h>
#include <uipc/common/log.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace uipc::backend::cuda
{
namespace
{
    constexpr int32_t CandidateDumpMagic       = 0x43414E44;  // 'CAND'
    constexpr int32_t CandidateDumpVersion     = 1;
    constexpr int32_t CandidateDumpRecordBytes = 44;

    // 44 bytes, deliberately field-order-stable for the numpy reader.
    struct CandidateRecord
    {
        int32_t v[4];
        float   dist2;
        float   disp_a;
        float   disp_b;
        float   d_hat;
        float   thick;
        int32_t body_a;
        int32_t body_b;
    };
    static_assert(sizeof(CandidateRecord) == CandidateDumpRecordBytes);

    Float max_disp(const Vector3* dxs, IndexT v)
    {
        return dxs[v].norm();
    }

    template <typename PairT>
    void fill_common(CandidateRecord& r,
                     const PairT&     ids,
                     const Vector3*   dxs,
                     const Float*     d_hats,
                     const Float*     thicknesses,
                     const IndexT*    body_ids,
                     SizeT            n_prim_a,
                     SizeT            n_prim_b)
    {
        for(SizeT j = 0; j < 4; ++j)
            r.v[j] = j < SizeT(ids.size()) ? int32_t(ids[j]) : -1;

        Float da = 0.0, db = 0.0;
        for(SizeT j = 0; j < n_prim_a; ++j)
            da = std::max<Float>(da, max_disp(dxs, ids[j]));
        for(SizeT j = n_prim_a; j < n_prim_a + n_prim_b; ++j)
            db = std::max<Float>(db, max_disp(dxs, ids[j]));

        // The engine's pair d_hat/thickness convention (PT_d_hat, PT_thickness,
        // ... in utils/primitive_d_hat.h / codim_thickness.h): mean of the two
        // primitives' representative-vertex d_hats, sum of their thicknesses.
        // Each primitive's vertices share one d_hat/thickness (runtime-asserted
        // upstream), so the first vertex of each primitive is representative.
        const IndexT ra = ids[0];
        const IndexT rb = ids[n_prim_a];
        const Float  dh = Float(0.5) * (d_hats[ra] + d_hats[rb]);
        const Float  th = thicknesses[ra] + thicknesses[rb];

        r.disp_a = float(da);
        r.disp_b = float(db);
        r.d_hat  = float(dh);
        r.thick  = float(th);
        r.body_a = int32_t(body_ids[ra]);
        r.body_b = int32_t(body_ids[rb]);
    }
}  // namespace

void SimplexTrajectoryFilter::Impl::dump_candidate_pairs(SizeT frame, SizeT newton_iter)
{
    // ---- lazy open ----
    if(candidate_dump_file == nullptr)
    {
        const char* dir = std::getenv("UIPC_ORACLE_DIR");
        if(dir == nullptr || *dir == '\0')
        {
            logger::error(
                "[dump_candidates] extras/debug/dump_candidates is set "
                "but UIPC_ORACLE_DIR is empty; the dump is disabled.");
            candidate_dump_on = false;
            return;
        }
        std::filesystem::create_directories(dir);
        candidate_dump_path = std::string(dir) + "/candidates.bin";
        candidate_dump_file = std::fopen(candidate_dump_path.c_str(), "wb");
        if(candidate_dump_file == nullptr)
        {
            logger::error("[dump_candidates] cannot create {}; the dump is disabled.",
                          candidate_dump_path);
            candidate_dump_on = false;
            return;
        }
        const int32_t header[4] = {
            CandidateDumpMagic, CandidateDumpVersion, CandidateDumpRecordBytes, 4};
        std::fwrite(header, sizeof(int32_t), 4, candidate_dump_file);
        logger::warn(
            "[dump_candidates] ACTIVE: writing per-pair candidates of every "
            "DCD detection to {} (diagnostic only, expect large files)",
            candidate_dump_path);
    }

    auto gvm = global_vertex_manager;

    // CBufferView only offers the raw-pointer copy_to; resize + download.
    auto download = []<typename T>(cuda_tool::CBufferView<T> view, std::vector<T>& host)
    {
        host.resize(view.size());
        view.copy_to(host.data());
    };

    std::vector<Vector3> positions, dxs;
    std::vector<Float>   thicknesses, d_hats;
    std::vector<IndexT>  body_ids;

    download(gvm->positions(), positions);
    download(gvm->displacements(), dxs);
    download(gvm->thicknesses(), thicknesses);
    download(gvm->d_hats(), d_hats);
    download(gvm->body_ids(), body_ids);

    std::vector<Vector4i> h_PTs, h_EEs;
    std::vector<Vector3i> h_PEs;
    std::vector<Vector2i> h_PPs;
    download(PTs, h_PTs);
    download(EEs, h_EEs);
    download(PEs, h_PEs);
    download(PPs, h_PPs);

    std::vector<CandidateRecord> records;
    records.reserve(h_PTs.size() + h_EEs.size() + h_PEs.size() + h_PPs.size());

    const Vector3* P  = positions.data();
    const Vector3* DX = dxs.data();

    // ---- PT ----
    for(const auto& PT : h_PTs)
    {
        CandidateRecord r{};
        fill_common(r, PT, DX, d_hats.data(), thicknesses.data(), body_ids.data(), 1, 3);
        Vector4i flag = distance::point_triangle_distance_flag(
            P[PT[0]], P[PT[1]], P[PT[2]], P[PT[3]]);
        Float D = 0.0;
        distance::point_triangle_distance2(flag, P[PT[0]], P[PT[1]], P[PT[2]], P[PT[3]], D);
        r.dist2 = float(D);
        records.push_back(r);
    }
    // ---- EE ----
    for(const auto& EE : h_EEs)
    {
        CandidateRecord r{};
        fill_common(r, EE, DX, d_hats.data(), thicknesses.data(), body_ids.data(), 2, 2);
        Vector4i flag =
            distance::edge_edge_distance_flag(P[EE[0]], P[EE[1]], P[EE[2]], P[EE[3]]);
        Float D = 0.0;
        distance::edge_edge_distance2(flag, P[EE[0]], P[EE[1]], P[EE[2]], P[EE[3]], D);
        r.dist2 = float(D);
        records.push_back(r);
    }
    // ---- PE ----
    for(const auto& PE : h_PEs)
    {
        CandidateRecord r{};
        fill_common(r, PE, DX, d_hats.data(), thicknesses.data(), body_ids.data(), 1, 2);
        Vector3i flag =
            distance::point_edge_distance_flag(P[PE[0]], P[PE[1]], P[PE[2]]);
        Float D = 0.0;
        distance::point_edge_distance2(flag, P[PE[0]], P[PE[1]], P[PE[2]], D);
        r.dist2 = float(D);
        records.push_back(r);
    }
    // ---- PP ----
    for(const auto& PP : h_PPs)
    {
        CandidateRecord r{};
        fill_common(r, PP, DX, d_hats.data(), thicknesses.data(), body_ids.data(), 1, 1);
        Float D = 0.0;
        distance::point_point_distance2(P[PP[0]], P[PP[1]], D);
        r.dist2 = float(D);
        records.push_back(r);
    }

    const int32_t header[6] = {int32_t(frame),
                               int32_t(newton_iter),
                               int32_t(h_PTs.size()),
                               int32_t(h_EEs.size()),
                               int32_t(h_PEs.size()),
                               int32_t(h_PPs.size())};
    std::fwrite(header, sizeof(int32_t), 6, candidate_dump_file);
    if(!records.empty())
        std::fwrite(records.data(), sizeof(CandidateRecord), records.size(), candidate_dump_file);
    std::fflush(candidate_dump_file);
}
}  // namespace uipc::backend::cuda

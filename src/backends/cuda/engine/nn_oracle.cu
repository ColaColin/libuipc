// DIAGNOSTIC-ONLY oracle hooks for the NN-acceleration research (default off).
// See extras/debug/warm_start_oracle in scene_default_config.cpp.
//
// Mode 1 (capture): at the end of every frame, append the converged global
// vertex positions to <UIPC_ORACLE_DIR>/positions_f64.bin as one raw record
// (N * 3 float64 per frame, frame index = frame - 1).
//
// Mode 2 (replay): right after predict_dof of frame t, overwrite the initial
// Newton iterate (FEM xs + the mirrored global positions buffer) with record
// t-1 of the capture. Everything downstream of the initial iterate -- x_tilde,
// x_prev, the friction anchors, contact detection, the exit policy -- is built
// from the run's own state, so this isolates exactly one variable: the quality
// of the initial iterate. This is the "perfect next-frame predictor" ceiling.
#include <sim_engine.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <uipc/common/log.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace uipc::backend::cuda
{
static std::string oracle_positions_path(const std::string& dir)
{
    return dir + "/positions_f64.bin";
}

// Offset of the FEM vertex block inside the global vertex buffer. The global
// buffer is the concatenation of the vertex reporters sorted by uid
// (AffineBody=0 | FiniteElement=1 | HalfPlane=2), so on scenes like the stiff
// benchmark (FEM bodies + a ground half-plane) the FEM xs buffer maps to a
// contiguous sub-range of the captured global record. The caller (a SimEngine
// member, which is GlobalVertexManager's friend) hands us the Impl.
static SizeT fem_vertex_offset(GlobalVertexManager::Impl& gvm, bool& found)
{
    auto reporters = gvm.vertex_reporters.view();
    auto counts    = gvm.reporter_vertex_offsets_counts.counts();
    SizeT offset   = 0;
    found = false;
    for(SizeT i = 0; i < reporters.size(); ++i)
    {
        if(dynamic_cast<const FiniteElementVertexReporter*>(reporters[i]) != nullptr)
        {
            found = true;
            break;
        }
        offset += counts[i];
    }
    return offset;
}

void SimEngine::init_warm_start_oracle()
{
    if(m_warm_start_oracle->view()[0] == 0)
        return;

    const char* dir = std::getenv("UIPC_ORACLE_DIR");
    if(dir == nullptr || *dir == '\0')
    {
        logger::error(
            "[warm_start_oracle] mode {} is set but UIPC_ORACLE_DIR is empty; "
            "the oracle is disabled.",
            m_warm_start_oracle->view()[0]);
        return;
    }
    m_oracle_dir = dir;

    m_oracle_vertex_count = m_global_vertex_manager->positions().size();

    if(m_warm_start_oracle->view()[0] == 1)  // capture
    {
        m_oracle_frame.resize(m_oracle_vertex_count);
        // a fresh run appends; report what was already on disk
        std::error_code ec;
        auto size = std::filesystem::file_size(oracle_positions_path(m_oracle_dir), ec);
        m_oracle_frames =
            ec ? 0 : size / (sizeof(Vector3) * m_oracle_vertex_count);
        logger::warn("[warm_start_oracle] capture ACTIVE: {} vertices, {} "
                     "existing records in {}",
                     m_oracle_vertex_count,
                     m_oracle_frames,
                     oracle_positions_path(m_oracle_dir));
        return;
    }

    // replay: load the whole capture into host memory
    if(m_finite_element_method == nullptr)
    {
        logger::error("[warm_start_oracle] replay needs an FEM pipeline; disabled.");
        return;
    }

    bool found_fem = false;
    m_oracle_fem_offset =
        fem_vertex_offset(m_global_vertex_manager->m_impl, found_fem);
    const SizeT fem_vertices = m_finite_element_method->xs().size();
    if(!found_fem || m_oracle_fem_offset + fem_vertices > m_oracle_vertex_count)
    {
        logger::error(
            "[warm_start_oracle] cannot locate the FEM vertex block "
            "(fem_offset={}, fem={}, global={}); replay disabled.",
            m_oracle_fem_offset,
            fem_vertices,
            m_oracle_vertex_count);
        return;
    }

    std::ifstream in(oracle_positions_path(m_oracle_dir), std::ios::binary);
    if(!in)
    {
        logger::error("[warm_start_oracle] cannot open {}; disabled.",
                      oracle_positions_path(m_oracle_dir));
        return;
    }
    in.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::streamoff>(in.tellg());
    in.seekg(0, std::ios::beg);

    const SizeT record = sizeof(Vector3) * m_oracle_vertex_count;
    if(bytes < 0 || static_cast<SizeT>(bytes) % record != 0)
    {
        logger::error("[warm_start_oracle] {} has {} bytes, not a multiple of "
                      "the {}-byte record of this scene; disabled.",
                      oracle_positions_path(m_oracle_dir),
                      static_cast<long long>(bytes),
                      static_cast<long long>(record));
        return;
    }

    m_oracle_frames   = static_cast<SizeT>(bytes) / record;
    SizeT to_load     = m_oracle_frames * m_oracle_vertex_count;
    m_oracle_host.resize(to_load);
    in.read(reinterpret_cast<char*>(m_oracle_host.data()),
            static_cast<std::streamsize>(to_load * sizeof(Vector3)));
    if(!in)
    {
        logger::error("[warm_start_oracle] short read on {}; disabled.",
                      oracle_positions_path(m_oracle_dir));
        m_oracle_host.clear();
        return;
    }

    logger::warn("[warm_start_oracle] replay ACTIVE: {} vertices (FEM block at "
                 "{}, {}) x {} frames from {}",
                 m_oracle_vertex_count,
                 m_oracle_fem_offset,
                 fem_vertices,
                 m_oracle_frames,
                 oracle_positions_path(m_oracle_dir));
}

void SimEngine::oracle_capture_frame()
{
    if(m_oracle_dir.empty() || m_oracle_frame.size() != m_oracle_vertex_count)
        return;

    // global positions == FEM xs == the converged iterate at frame end
    m_global_vertex_manager->m_impl.positions.view().copy_to(m_oracle_frame.data());

    FILE* f = std::fopen(oracle_positions_path(m_oracle_dir).c_str(), "ab");
    if(f == nullptr)
    {
        logger::error("[warm_start_oracle] cannot open {} for append.",
                      oracle_positions_path(m_oracle_dir));
        m_oracle_dir.clear();  // give up after the first failure
        return;
    }
    const SizeT n = std::fwrite(m_oracle_frame.data(),
                                sizeof(Vector3),
                                m_oracle_frame.size(),
                                f);
    std::fclose(f);
    if(n != m_oracle_frame.size())
    {
        logger::error("[warm_start_oracle] short write on frame {}.",
                      m_current_frame);
        m_oracle_dir.clear();
        return;
    }
    ++m_oracle_frames;
}

void SimEngine::oracle_inject_frame()
{
    if(m_oracle_host.empty())
        return;

    const SizeT idx = m_current_frame - 1;  // frame is 1-based
    if(idx >= m_oracle_frames)
    {
        logger::warn("[warm_start_oracle] no record for frame {}; keeping the "
                     "normal iterate.",
                     m_current_frame);
        return;
    }

    const Vector3* src =
        m_oracle_host.data() + idx * m_oracle_vertex_count;

    // 1) FEM xs: the buffer the Newton loop actually iterates on (needs the
    //    mutable buffer; the public xs() accessor returns a read-only view).
    //    The FEM block sits at m_oracle_fem_offset inside the global record.
    m_finite_element_method->m_impl.xs.view().copy_from(src + m_oracle_fem_offset);

    // 2) global positions: the buffer collision detection and the line search
    //    mirror xs into; it must not lag the injected iterate
    m_global_vertex_manager->m_impl.positions.view().copy_from(src);
}
}  // namespace uipc::backend::cuda

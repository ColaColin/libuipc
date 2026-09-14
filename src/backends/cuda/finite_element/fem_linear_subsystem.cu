#include <uipc/common/timer.h>
#include <sim_engine.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/fem_linear_subsystem_reporter.h>
#include <finite_element/finite_element_kinetic.h>
#include <kernel_cout.h>
#include <cuda_tool/cub.h>
#include <cuda_tool/cuda_tool.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/fem_dytopo_effect_receiver.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/flag.h>
#include <utils/report_extent_check.h>

namespace uipc::backend::cuda
{
// ============================================================================
// Named kernels (replacements for the former lambda kernel launches)
// ============================================================================
namespace
{
    __global__ void FEMLinearSubsystem_assemble_k1_kernel(cuda_tool::CBufferView<IndexT> is_fixed,
                                                          cuda_tool::DenseVectorView<Float> gradients,
                                                          int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        if(is_fixed(i))
        {
            gradients.segment<3>(i * 3).as_eigen().setZero();
        }
    }

    __global__ void FEMLinearSubsystem_assemble_k2_kernel(
        cuda_tool::CBufferView<IndexT>            is_fixed,
        cuda_tool::TripletMatrixView<Float, 3, 3> hessians,
        int                                       n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        auto&& [i, j, H3] = hessians(I).read();

        if(is_fixed(i) || is_fixed(j))
        {
            if(i != j)
                hessians(I).write(i, j, Matrix3x3::Zero());
            else
                hessians(I).write(i, j, Matrix3x3::Identity());
        }
    }

    __global__ void FEMLinearSubsystem_assemble_kinetic_kernel(
        cuda_tool::DenseVectorView<Float>       dst,
        cuda_tool::CDoubletVectorView<Float, 3> src,
        cuda_tool::CBufferView<IndexT>          is_fixed,
        int                                     n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        auto&& [i, G3] = src(I);
        if(is_fixed(i))
            return;
        dst.segment<3>(i * 3).atomic_add(G3);
    }

    __global__ void FEMLinearSubsystem_assemble_reporters_kernel(
        cuda_tool::DenseVectorView<Float>       dst,
        cuda_tool::CDoubletVectorView<Float, 3> src,
        cuda_tool::CBufferView<IndexT>          is_fixed,
        int                                     n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        auto&& [i, G3] = src(I);
        if(is_fixed(i))
            return;
        dst.segment<3>(i * 3).atomic_add(G3);
    }

    __global__ void FEMLinearSubsystem_assemble_dytopo_effect_k1_kernel(
        cuda_tool::CDoubletVectorView<Float, 3> dytopo_effect_gradient,
        cuda_tool::DenseVectorView<Float>       gradients,
        IndexT                                  vertex_offset,
        cuda_tool::CBufferView<IndexT>          is_fixed,
        int                                     n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const auto& [g_i, G3] = dytopo_effect_gradient(I);
        auto i                = g_i - vertex_offset;  // from global to local

        if(is_fixed(i))
            return;

        gradients.segment<3>(i * 3).atomic_add(G3);
    }

    __global__ void FEMLinearSubsystem_assemble_dytopo_effect_k2_kernel(
        cuda_tool::CTripletMatrixView<Float, 3, 3> dytopo_effect_hessian,
        cuda_tool::TripletMatrixView<Float, 3, 3>  hessians,
        IndexT                                     vertex_offset,
        int                                        n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const auto& [g_i, g_j, H3] = dytopo_effect_hessian(I);
        auto i                     = g_i - vertex_offset;
        auto j                     = g_j - vertex_offset;
        hessians(I).write(i, j, H3);
    }

    __global__ void FEMLinearSubsystem_retrieve_solution_kernel(
        cuda_tool::BufferView<Vector3> dxs, cuda_tool::CDenseVectorView<Float> result, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        dxs(i) = -result.segment<3>(i * 3).as_eigen();
    }

    __global__ void FEMLinearSubsystem_diag_norm_kernel(
        cuda_tool::CTripletMatrixView<Float, 3, 3> triplet,
        cuda_tool::BufferView<Float>               diag_blocks_norm,
        IndexT                                     fem_segment_offset,
        IndexT                                     fem_segment_count,
        cuda_tool::CBufferView<IndexT>             is_fixed,
        int                                        n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        auto&& [g_i, g_j, H3x3] = triplet(I);

        IndexT i = g_i - fem_segment_offset;
        IndexT j = g_j - fem_segment_offset;

        if(i >= fem_segment_count || j >= fem_segment_count)
            return;
        if(i == j)
        {
            auto a              = abs(H3x3(0, 0));
            auto b              = abs(H3x3(1, 1));
            auto c              = abs(H3x3(2, 2));
            diag_blocks_norm(i) = is_fixed(i) ? 0 : max(max(a, b), c);
        }
    }

    __global__ void FEMLinearSubsystem_mass_norm_kernel(cuda_tool::CBufferView<Float> mass,
                                                        cuda_tool::BufferView<Float> diag_blocks_norm,
                                                        cuda_tool::CBufferView<IndexT> is_fixed,
                                                        int n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        diag_blocks_norm(I) = is_fixed(I) ? 0 : mass(I);
    }
}  // namespace

REGISTER_SIM_SYSTEM(FEMLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 FEMLinearSubsystemUID = 1ull;

U64 FEMLinearSubsystem::get_uid() const noexcept
{
    return FEMLinearSubsystemUID;
}

void FEMLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo&)
{
    m_impl.finite_element_method = require<FiniteElementMethod>();
    m_impl.finite_element_vertex_reporter = require<FiniteElementVertexReporter>();
    m_impl.sim_engine = &engine();
    m_impl.dt_attr    = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");

    m_impl.dytopo_effect_receiver = find<FEMDyTopoEffectReceiver>();
}

void FEMLinearSubsystem::do_init(DiagLinearSubsystem::InitInfo& info)
{
    m_impl.init();
}

void FEMLinearSubsystem::Impl::init()
{
    auto reporter_view = reporters.view();
    for(auto&& [i, r] : enumerate(reporter_view))
        r->m_index = i;
    for(auto& r : reporter_view)
        r->init();

    reporter_gradient_offsets_counts.resize(reporter_view.size());
    reporter_hessian_offsets_counts.resize(reporter_view.size());
}

void FEMLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(fem().xs.size() * 3);
}

void FEMLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto& geo_infos = fem().geo_infos;
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    finite_element_method->for_each(
        geo_slots,
        [&](const FiniteElementMethod::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on FEM mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on FEM mesh why can it happen?");

            IndexT this_dof_count = 3 * sc.vertices().size();
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void FEMLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    bool gradient_only = info.gradient_only();

    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    // 1) Hessian Count
    IndexT grad_offset = 0;
    IndexT hess_offset = 0;

    // We assume reporters won't produce contact
    if(has_complement)
    {
        // Kinetic
        auto kinetic_grad_count = fem().xs.size();
        grad_offset += kinetic_grad_count;
        if(!gradient_only)
            hess_offset += fem().xs.size();

        // Reporters
        auto grad_counts = reporter_gradient_offsets_counts.counts();
        auto hess_counts = reporter_hessian_offsets_counts.counts();

        for(auto& reporter : reporters.view())
        {
            ReportExtentInfo this_info;
            this_info.m_gradient_only = gradient_only;
            reporter->report_extent(this_info);
            grad_counts[reporter->m_index] = this_info.m_gradient_count;
            hess_counts[reporter->m_index] = this_info.m_hessian_count;

            UIPC_ASSERT(!(gradient_only && !this_info.m_hessian_count == 0),
                        "When gradient_only is true, hessian_offset must be 0, yours {}.\n"
                        "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                        this_info.m_hessian_count);
        }

        // [KineticG ... | OtherG ... ]
        // [KineticH ... | OtherH ... ]

        reporter_gradient_offsets_counts.scan();
        reporter_hessian_offsets_counts.scan();

        grad_offset += reporter_gradient_offsets_counts.total_count();
        hess_offset += reporter_hessian_offsets_counts.total_count();
    }

    if(dytopo_effect_receiver)  // if dytopo_effect enabled
    {
        // s14: host-side counts only -- the device views (which join a
        // deferred contact assembly) are first taken in _assemble_dytopo_effect
        grad_offset += dytopo_effect_receiver->gradient_count();
        hess_offset += dytopo_effect_receiver->hessian_count();

        UIPC_ASSERT(!(gradient_only && !dytopo_effect_receiver->hessian_count() == 0),
                    "When gradient_only is true, hessian_offset must be 0, yours {}.\n"
                    "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                    dytopo_effect_receiver->hessian_count());
    }

    // 2) Gradient Count
    auto dof_count = fem().xs.size() * 3;

    UIPC_ASSERT(!(gradient_only && !hess_offset == 0),
                "When gradient_only is true, hessian_offset must be 0, yours {}.\n"
                "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                hess_offset);

    info.extent(hess_offset, dof_count);
}

void FEMLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    using namespace cuda_tool;

    // 0) record dof info
    auto frame = sim_engine->frame();
    fem().set_dof_info(frame, info.gradients().offset(), info.gradients().size());

    // 1) Prepare Gradient Buffer
    kinetic_gradients.resize_doublets_discard(fem().xs.size());
    kinetic_gradients.reshape(fem().xs.size());
    loose_resize_entries(reporter_gradients, reporter_gradient_offsets_counts.total_count());
    reporter_gradients.reshape(fem().xs.size());

    info.gradients().buffer_view().fill(0);

    // 2) Assemble Gradient and Hessian
    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    IndexT hess_offset = 0;
    if(has_complement)
    {
        {
            Timer timer{"FEM Kinetic G/H"};
            _assemble_kinetic(hess_offset, info);
        }
        {
            Timer timer{"FEM Reporters G/H"};
            _assemble_reporters(hess_offset, info);
        }
    }

    if(dytopo_effect_receiver)  // if dytopo_effect enabled
    {
        // DyTopo System will decide the `component_flags` itself
        {
            Timer timer{"FEM Dytopo Copy"};
            _assemble_dytopo_effect(hess_offset, info);
        }
    }

    UIPC_ASSERT(hess_offset == info.hessians().triplet_count(),
                "Hessian offset mismatch expected {}, got {}",
                info.hessians().triplet_count(),
                hess_offset);


    // 3) Clear Fixed Vertex gradient (double check)
    {
        auto k = FEMLinearSubsystem_assemble_k1_kernel;
        int  n = (int)fem().xs.size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                fem().is_fixed.cview(), info.gradients(), n);
        }
    }

    if(info.gradient_only())
        return;

    // 4) Clear Fixed Vertex hessian
    {
        auto k = FEMLinearSubsystem_assemble_k2_kernel;
        int  n = (int)info.hessians().triplet_count();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                fem().is_fixed.cview(), info.hessians(), n);
        }
    }
}


void FEMLinearSubsystem::Impl::_assemble_kinetic(IndexT& hess_offset,
                                                 GlobalLinearSystem::DiagInfo& info)
{
    using namespace cuda_tool;

    IndexT hess_count = info.gradient_only() ? 0 : fem().xs.size();
    IndexT grad_count = fem().xs.size();

    auto gradient_view = kinetic_gradients.view();
    auto hessian_view  = info.hessians().subview(hess_offset, hess_count);

    FEMLinearSubsystem::ComputeGradientHessianInfo kinetic_info{
        info.gradient_only(), gradient_view, hessian_view, dt_attr->view()[0]};
    kinetic->compute_gradient_hessian(kinetic_info);

    {
        auto k = FEMLinearSubsystem_assemble_kinetic_kernel;
        int  n = (int)kinetic_gradients.doublet_count();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                info.gradients(), kinetic_gradients.cview(), fem().is_fixed.cview(), n);
        }
    }

    hess_offset += hess_count;
}

// perf round 6 (s12): why the FEM elastic G/H is NOT hoistable onto a side
// stream the way s10/s11 hoisted the ABD body-local kinetic/shape G/H.
//
// The ABD prepass works because its four output buffers
// (`body_id_to_{shape,kinetic}_{gradient,hessian}`) are **private, allocated
// once in Impl::init(), and read by nothing but the assemble kernels that
// follow the join**. The FEM reporters are not shaped like that. Only the
// gradient half is private (`reporter_gradients`); the Hessian half is written
// straight into `info.hessians()`, which is a subview of the **global**
// `triplet_A`, and three separate things make that destination unavailable
// before the contact phase has finished:
//
//   1. its size is not known. `report_extent` above adds
//      `dytopo_effect_receiver->hessians().triplet_count()` to the FEM extent,
//      so `GlobalLinearSystem::Impl::_update_subsystem_extent`
//      (global_linear_system.cu:858) only calls `triplet_A.resize_triplets_discard`
//      -- and, on growth, `reserve_triplets_discard`, which **reallocates** --
//      after the contact count exists;
//   2. its offset is not known. The region starts at
//      `subsystem_triplet_offsets[triplet_i] + kinetic_count`
//      (global_linear_system.cu:920), and that base includes every preceding
//      subsystem's triplet count, contact included, on any scene with more than
//      one diagonal subsystem;
//   3. it is overwritten afterwards anyway. `_assemble_linear_system` opens with
//      `triplet_A.row_indices().fill(-1)` / `col_indices().fill(-1)`
//      (global_linear_system.cu:881-882) over the whole matrix, which runs on
//      the default stream *after* the contact phase and would erase anything a
//      prepass had written.
//
// The *inputs* are hoistable -- the elastic kernels read `xs`, `x_bars`,
// `Dm_invs`, `rest_volumes`, the tet/tri indices and the per-element material
// parameters, none of which the contact phase writes -- so the obstacle is the
// destination, not the dependency. The only design that survives is ABD's:
// stage the Hessian in a private triplet buffer during the prepass and copy it
// into `info.hessians()` once that exists. `UIPC_FEM_GH_COST_PROBE=1` measures
// what that copy would cost, by moving exactly its bytes into scratch nothing
// reads. See agent_docs/performance/2026-09-13-perf-round6.md, step s12.
void FEMLinearSubsystem::Impl::_assemble_reporters(IndexT& hess_offset,
                                                   GlobalLinearSystem::DiagInfo& info)
{
    using namespace cuda_tool;
    auto grad_count = reporter_gradient_offsets_counts.total_count();
    auto hess_count =
        info.gradient_only() ? 0 : reporter_hessian_offsets_counts.total_count();

    // Let reporters assemble their gradient and hessian
    auto reporter_gradient_view = reporter_gradients.view();
    auto reporter_hessian_view = info.hessians().subview(hess_offset, hess_count);

    if(!gh_cost_probe_read) [[unlikely]]
    {
        const char* e      = std::getenv("UIPC_FEM_GH_COST_PROBE");
        gh_cost_probe      = e && e[0] != '0';
        gh_cost_probe_read = true;
    }
    if(gh_cost_probe && hess_count > 0) [[unlikely]]
    {
        // the staging copy the prepass design would have to pay, byte for byte:
        // row indices (4 B), col indices (4 B) and the 3x3 block (72 B) per
        // triplet, read once and written once. Destination is scratch; nothing
        // reads it, so no computed quantity changes.
        if(probe_vals.size() < (SizeT)hess_count)
        {
            probe_rows.resize(hess_count);
            probe_cols.resize(hess_count);
            probe_vals.resize(hess_count);
        }
        CUDA_TOOL_CHECK(cudaMemcpyAsync(probe_rows.data(),
                                        reporter_hessian_view.row_indices().data(),
                                        (size_t)hess_count * sizeof(int),
                                        cudaMemcpyDeviceToDevice,
                                        nullptr));
        CUDA_TOOL_CHECK(cudaMemcpyAsync(probe_cols.data(),
                                        reporter_hessian_view.col_indices().data(),
                                        (size_t)hess_count * sizeof(int),
                                        cudaMemcpyDeviceToDevice,
                                        nullptr));
        CUDA_TOOL_CHECK(cudaMemcpyAsync(probe_vals.data(),
                                        reporter_hessian_view.values().data(),
                                        (size_t)hess_count * sizeof(Matrix3x3),
                                        cudaMemcpyDeviceToDevice,
                                        nullptr));
        if(probe_calls == 0)
            std::fprintf(stderr,
                         "[FEMGHCostProbe] staging copy of %lld reporter triplets "
                         "(%.1f MB moved per Newton iteration, read+write)\n",
                         (long long)hess_count,
                         2.0 * (double)hess_count * (2 * sizeof(int) + sizeof(Matrix3x3)) / 1e6);
        probe_triplets = hess_count;
        ++probe_calls;
    }
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, reporter_hessian_view, info.gradient_only()};
        R->assemble(assemble_info);
    }

    {
        auto k = FEMLinearSubsystem_assemble_reporters_kernel;
        int  n = (int)reporter_gradients.doublet_count();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                info.gradients(), reporter_gradients.cview(), fem().is_fixed.cview(), n);
        }
    }

    // offset update
    hess_offset += hess_count;
}

void FEMLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace cuda_tool;

    // No need to add to grad_offset, the grad buffer is not from reporter_gradients
    auto grad_count = dytopo_effect_receiver->gradients().doublet_count();

    // 1) Assemble DyTopoEffect Gradient to Gradient
    if(grad_count)
    {
        auto k = FEMLinearSubsystem_assemble_dytopo_effect_k1_kernel;
        k<<<cuda_tool::best_grid_dim((int)grad_count, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            dytopo_effect_receiver->gradients(),
            info.gradients(),
            finite_element_vertex_reporter->vertex_offset(),
            fem().is_fixed.cview(),
            (int)grad_count);
    }

    if(info.gradient_only())
        return;

    // Need to update hess_offset, we are assembling to the global hessian buffer
    auto hess_count = dytopo_effect_receiver->hessians().triplet_count();

    // 2) Assemble DyTopoEffect Hessian to Hessian
    if(hess_count)
    {
        auto dst_H3x3s = info.hessians().subview(hess_offset, hess_count);

        // NOTE: We don't consider fixed vertex here,
        // because in final phase we willclear the fixed vertex hessian anyway.
        auto k = FEMLinearSubsystem_assemble_dytopo_effect_k2_kernel;
        k<<<cuda_tool::best_grid_dim(hess_count, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            dytopo_effect_receiver->hessians(),
            dst_H3x3s,
            finite_element_vertex_reporter->vertex_offset(),
            hess_count);
    }

    hess_offset += hess_count;
}

void FEMLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.satisfied(true);
}

void FEMLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    using namespace cuda_tool;

    auto dxs = fem().dxs.view();

    auto k = FEMLinearSubsystem_retrieve_solution_kernel;
    int  n = (int)fem().xs.size();
    if(n > 0)
    {
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            dxs, info.solution(), n);
    }
}

Float FEMLinearSubsystem::Impl::diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    diag_blocks_norm.resize(finite_element_method->xs().size());

    auto k = FEMLinearSubsystem_diag_norm_kernel;
    int  n = (int)info.A().triplet_count();
    if(n > 0)
    {
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            info.A(),
            diag_blocks_norm.view(),
            info.dof_offset() / 3,
            info.dof_count() / 3,
            fem().is_fixed.cview(),
            n);
    }

    cuda_tool::DeviceReduce().Max(diag_blocks_norm.data(),
                                  reduced_diag_norm.data(),
                                  diag_blocks_norm.size());

    return reduced_diag_norm;
}

Float FEMLinearSubsystem::Impl::mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    diag_blocks_norm.resize(fem().xs.size());
    UIPC_ASSERT(fem().xs.size() == fem().masses.size(), "size not matched");

    auto k = FEMLinearSubsystem_mass_norm_kernel;
    int  n = (int)fem().xs.size();
    if(n > 0)
    {
        k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
            fem().masses.cview(), diag_blocks_norm.view(), fem().is_fixed.cview(), n);
    }

    cuda_tool::DeviceReduce().Max(diag_blocks_norm.data(),
                                  reduced_diag_norm.data(),
                                  diag_blocks_norm.size());

    return reduced_diag_norm;
}

void FEMLinearSubsystem::Impl::loose_resize_entries(cuda_tool::DeviceDoubletVector<Float, 3>& v,
                                                    SizeT size)
{
    if(size > v.doublet_capacity())
    {
        v.reserve_doublets_discard(size * reserve_ratio);
    }
    v.resize_doublets_discard(size);
}

void FEMLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void FEMLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void FEMLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void FEMLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

void FEMLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void FEMLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

Float FEMLinearSubsystem::do_diag_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.diag_norm(info);
}

Float FEMLinearSubsystem::do_mass_norm(GlobalLinearSystem::DiagNormInfo& info)
{
    return m_impl.mass_norm(info);
}

cuda_tool::DoubletVectorView<Float, 3> FEMLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = m_impl->reporter_gradient_offsets_counts[m_index];
    return m_impl->reporter_gradients.view().subview(offset, count);
}

cuda_tool::TripletMatrixView<Float, 3, 3> FEMLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = m_impl->reporter_hessian_offsets_counts[m_index];
    return m_hessians.subview(offset, count);
}

Float FEMLinearSubsystem::AssembleInfo::dt() const noexcept
{
    return m_impl->dt_attr->view()[0];
}

bool FEMLinearSubsystem::AssembleInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void FEMLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void FEMLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void FEMLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

void FEMLinearSubsystem::add_reporter(FEMLinearSubsystemReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter cannot be null");
    check_state(SimEngineState::BuildSystems, "add_reporter");
    m_impl.reporters.register_sim_system(*reporter);
}

void FEMLinearSubsystem::add_kinetic(FiniteElementKinetic* kinetic)
{
    UIPC_ASSERT(kinetic, "kinetic cannot be null");
    check_state(SimEngineState::BuildSystems, "add_kinetic");
    m_impl.kinetic.register_sim_system(*kinetic);
}
}  // namespace uipc::backend::cuda

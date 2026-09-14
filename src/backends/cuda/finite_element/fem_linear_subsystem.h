#pragma once
#include <algorithm/matrix_converter.h>
#include <linear_system/diag_linear_subsystem.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::cuda
{
class FiniteElementKinetic;
class FEMDyTopoEffectReceiver;
class FEMLinearSubsystemReporter;
class FEMLinearSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(bool gradient_only,
                                   cuda_tool::DoubletVectorView<Float, 3> gradients,
                                   cuda_tool::TripletMatrixView<Float, 3, 3> hessians,
                                   Float dt) noexcept
            : m_gradient_only(gradient_only)
            , m_gradients(gradients)
            , m_hessians(hessians)
            , m_dt(dt)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        auto gradients() const noexcept { return m_gradients; }
        auto hessians() const noexcept { return m_hessians; }
        auto dt() const noexcept { return m_dt; }

      private:
        bool                                      m_gradient_only = false;
        cuda_tool::DoubletVectorView<Float, 3>    m_gradients;
        cuda_tool::TripletMatrixView<Float, 3, 3> m_hessians;
        Float                                     m_dt = 0.0;
    };

    class ReportExtentInfo
    {
      public:
        // DoubletVector3 count
        void gradient_count(SizeT size);
        // TripletMatrix3x3 count
        void hessian_count(SizeT size);
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class FEMLinearSubsystem;
        friend class FEMLinearSubsystemReporter;
        SizeT        m_gradient_count        = 0;
        SizeT        m_hessian_count         = 0;
        bool         m_gradient_only         = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl;

    class AssembleInfo
    {
      public:
        AssembleInfo(Impl*                                     impl,
                     IndexT                                    index,
                     cuda_tool::TripletMatrixView<Float, 3, 3> hessians,
                     bool gradient_only) noexcept
            : m_impl(impl)
            , m_index(index)
            , m_hessians(hessians)
            , m_gradient_only(gradient_only)
        {
        }

        cuda_tool::DoubletVectorView<Float, 3>    gradients() const;
        cuda_tool::TripletMatrixView<Float, 3, 3> hessians() const;
        Float                                     dt() const noexcept;
        bool gradient_only() const noexcept;

      private:
        friend class FEMLinearSubsystem;

        Impl*                                     m_impl  = nullptr;
        IndexT                                    m_index = ~0;
        cuda_tool::TripletMatrixView<Float, 3, 3> m_hessians;
        bool                                      m_gradient_only = false;
    };

    class Impl
    {
      public:
        void init();
        void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
        void receive_init_dof_info(WorldVisitor& w, GlobalLinearSystem::InitDofInfo& info);

        void report_extent(GlobalLinearSystem::DiagExtentInfo& info);

        void assemble(GlobalLinearSystem::DiagInfo& info);
        void _assemble_kinetic(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_reporters(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_dytopo_effect(IndexT& hess_offset, GlobalLinearSystem::DiagInfo& info);


        void  accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
        void  retrieve_solution(GlobalLinearSystem::SolutionInfo& info);
        Float diag_norm(GlobalLinearSystem::DiagNormInfo& info);
        Float mass_norm(GlobalLinearSystem::DiagNormInfo& info);

        SimEngine* sim_engine = nullptr;

        SimSystemSlot<FiniteElementMethod> finite_element_method;
        FiniteElementMethod::Impl&         fem() noexcept
        {
            return finite_element_method->m_impl;
        }

        SimSystemSlot<FiniteElementVertexReporter> finite_element_vertex_reporter;

        SimSystemSlot<FEMDyTopoEffectReceiver> dytopo_effect_receiver;
        SimSystemSlotCollection<FEMLinearSubsystemReporter> reporters;

        SimSystemSlot<FiniteElementKinetic> kinetic;


        S<const geometry::AttributeSlot<Float>> dt_attr;
        Float                                   reserve_ratio = 1.5;

        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        cuda_tool::DeviceDoubletVector<Float, 3> kinetic_gradients;
        cuda_tool::DeviceDoubletVector<Float, 3> reporter_gradients;

        // perf round 6 (s12): DIAGNOSTIC ONLY, `UIPC_FEM_GH_COST_PROBE=1`.
        // The FEM elastic gradient/Hessian cannot be hoisted onto a side stream
        // the way s10/s11 hoisted the ABD body-local one, because its Hessian
        // destination is the *global* triplet matrix, which does not exist until
        // after the contact phase has produced its triplet count (see the long
        // comment in `_assemble_reporters`). The only shape that could work is
        // ABD's: compute into a private staging buffer, then copy the staged
        // triplets into `info.hessians()` once it exists. This probe moves
        // exactly the bytes that copy would move -- row indices, col indices and
        // the 3x3 values of the whole reporter Hessian region -- into scratch
        // that nothing reads, so the *cost floor* of that design can be measured
        // without building it. It changes no computed quantity.
        bool                                      gh_cost_probe = false;
        bool                                      gh_cost_probe_read = false;
        cuda_tool::DeviceBuffer<int>              probe_rows;
        cuda_tool::DeviceBuffer<int>              probe_cols;
        cuda_tool::DeviceBuffer<Matrix3x3>        probe_vals;
        SizeT                                     probe_triplets = 0;
        SizeT                                     probe_calls    = 0;
        cuda_tool::DeviceBuffer<Float>           diag_blocks_norm;
        cuda_tool::DeviceVar<Float>              reduced_diag_norm;

        void loose_resize_entries(cuda_tool::DeviceDoubletVector<Float, 3>& v, SizeT size);
    };

  private:
    virtual void do_build(DiagLinearSubsystem::BuildInfo& info) override;
    virtual void do_init(DiagLinearSubsystem::InitInfo& info) override;
    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override;
    virtual Float do_diag_norm(GlobalLinearSystem::DiagNormInfo& info) override;
    virtual Float do_mass_norm(GlobalLinearSystem::DiagNormInfo& info) override;
    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override;
    virtual U64 get_uid() const noexcept override;

    friend class FEMLinearSubsystemReporter;
    void add_reporter(FEMLinearSubsystemReporter* reporter);

    friend class FiniteElementKinetic;
    void add_kinetic(FiniteElementKinetic* kinetic);

    Impl m_impl;
};
}  // namespace uipc::backend::cuda

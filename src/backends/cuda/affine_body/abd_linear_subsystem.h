#pragma once
#include <linear_system/diag_linear_subsystem.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_dytopo_effect_receiver.h>
#include <affine_body/affine_body_vertex_reporter.h>
#include <utils/offset_count_collection.h>

namespace uipc::backend::cuda
{
class ABDLinearSubsystemReporter;
class ABDLinearSubsystem final : public DiagLinearSubsystem
{
  public:
    using DiagLinearSubsystem::DiagLinearSubsystem;

    cuda_tool::CBufferView<Matrix12x12> diag_hessian() const noexcept
    {
        return m_impl.diag_hessian.view();
    }

    class ComputeGradientHessianInfo
    {
      public:
        ComputeGradientHessianInfo(bool gradient_only,
                                   cuda_tool::BufferView<Vector12>    gradient,
                                   cuda_tool::BufferView<Matrix12x12> hessians,
                                   Float        dt,
                                   cudaStream_t stream = nullptr) noexcept
            : m_gradient_only(gradient_only)
            , m_gradients(gradient)
            , m_hessians(hessians)
            , m_dt(dt)
            , m_stream(stream)
        {
        }

        auto gradient_only() const noexcept { return m_gradient_only; }
        auto hessians() const noexcept { return m_hessians; }
        auto gradients() const noexcept { return m_gradients; }
        auto dt() const noexcept { return m_dt; }
        // perf round 6 (s10): the launch stream of the body-local kinetic/shape
        // gradient+hessian. nullptr = the legacy default stream; the s10
        // prepass passes its side stream so these launches run inside the
        // contact assembly's shadow.
        auto stream() const noexcept { return m_stream; }

      private:
        bool                               m_gradient_only = false;
        cuda_tool::BufferView<Matrix12x12> m_hessians;
        cuda_tool::BufferView<Vector12>    m_gradients;
        Float                              m_dt     = 0.0;
        cudaStream_t                       m_stream = nullptr;
    };

    class ReportExtentInfo
    {
      public:
        // DoubletVector12 count
        void gradient_count(SizeT size);
        // TripletMatrix12x12 count
        void hessian_count(SizeT size);
        bool gradient_only() const noexcept
        {
            m_gradient_only_checked = true;
            return m_gradient_only;
        }
        void check(std::string_view name) const;

      private:
        friend class ABDLinearSubsystem;
        friend class ABDLinearSubsystemReporter;
        SizeT        m_gradient_count        = 0;
        SizeT        m_hessian_count         = 0;
        bool         m_gradient_only         = false;
        mutable bool m_gradient_only_checked = false;
    };

    class Impl;

    class AssembleInfo
    {
      public:
        AssembleInfo(Impl* impl, IndexT index, bool gradient_only) noexcept;
        cuda_tool::DoubletVectorView<Float, 12>     gradients() const;
        cuda_tool::TripletMatrixView<Float, 12, 12> hessians() const;
        bool gradient_only() const noexcept;

      private:
        friend class ABDLinearSubsystem;

        Impl*  m_impl          = nullptr;
        IndexT m_index         = ~0;
        bool   m_gradient_only = false;
    };

    class Impl
    {
      public:
        void init();
        void report_extent(GlobalLinearSystem::DiagExtentInfo& info);

        void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
        void receive_init_dof_info(WorldVisitor& w, GlobalLinearSystem::InitDofInfo& info);

        void assemble(GlobalLinearSystem::DiagInfo& info);
        void _assemble_kinetic_shape(IndexT& offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_reporters(IndexT& offset, GlobalLinearSystem::DiagInfo& info);
        void _assemble_dytopo_effect(IndexT& offset, GlobalLinearSystem::DiagInfo& info);
        // s03: group the dytopo (contact) 3x3 blocks by (body_i, body_j) pair;
        // returns the number of distinct pairs (each expands to 16 triplets)
        SizeT _prepare_dytopo_pairs();

        void  accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
        void  retrieve_solution(GlobalLinearSystem::SolutionInfo& info);
        Float diag_norm();
        Float mass_norm();

        SimSystemSlot<AffineBodyDynamics>       affine_body_dynamics;
        AffineBodyDynamics::Impl&               abd() const noexcept;
        SimSystemSlot<ABDDyTopoEffectReceiver>  dytopo_effect_receiver;
        SimSystemSlot<AffineBodyVertexReporter> affine_body_vertex_reporter;

        Float reserve_ratio = 1.5;

        SimSystemSlotCollection<ABDLinearSubsystemReporter> reporters;
        OffsetCountCollection<IndexT> reporter_gradient_offsets_counts;
        OffsetCountCollection<IndexT> reporter_hessian_offsets_counts;

        cuda_tool::DeviceTripletMatrix<Float, 12, 12> reporter_hessians;
        cuda_tool::DeviceDoubletVector<Float, 12>     reporter_gradients;

        // intermediate gradient/hessian buffers for kinetic/shape
        cuda_tool::DeviceBuffer<Matrix12x12> body_id_to_shape_hessian;
        cuda_tool::DeviceBuffer<Vector12>    body_id_to_shape_gradient;
        cuda_tool::DeviceBuffer<Matrix12x12> body_id_to_kinetic_hessian;
        cuda_tool::DeviceBuffer<Vector12>    body_id_to_kinetic_gradient;

        // diag hessian for preconditioner
        cuda_tool::DeviceBuffer<Matrix12x12> diag_hessian;

        // s03: per-body-pair pre-reduction of the dytopo effect hessians
        // (UIPC_ABD_PAIR_REDUCE=0 restores the 16-triplets-per-contact path);
        // s05: warp-cooperative accumulation in pair-sorted order
        // (UIPC_ABD_PAIR_WARP=0 restores the one-thread-per-contact kernel)
        bool                                 dytopo_pair_reduce      = true;
        bool                                 dytopo_pair_warp        = true;
        int                                  dytopo_pair_warp_rounds = 1;
        SizeT                                dytopo_pair_count       = 0;
        cuda_tool::DeviceBuffer<uint64_t>    dytopo_pair_key_in;
        cuda_tool::DeviceBuffer<uint64_t>    dytopo_pair_key_sorted;
        cuda_tool::DeviceBuffer<int>         dytopo_pair_idx_in;
        cuda_tool::DeviceBuffer<int>         dytopo_pair_perm;
        cuda_tool::DeviceBuffer<int>         dytopo_pair_head;
        cuda_tool::DeviceBuffer<int>         dytopo_pair_seg;
        cuda_tool::DeviceBuffer<int>         dytopo_contact_to_pair;
        cuda_tool::DeviceBuffer<uint64_t>    dytopo_pair_key;
        cuda_tool::DeviceBuffer<Matrix12x12> dytopo_pair_hessian;
        cuda_tool::DeviceBuffer<Float>       block_norm;
        cuda_tool::DeviceVar<Float>          reduced_norm;

        S<const geometry::AttributeSlot<Float>> dt_attr;

        // perf round 6 (s10): the body-local kinetic + shape gradient/hessian
        // (bdf1 kinetic G/H and every AffineBodyConstitution, e.g.
        // ortho_potential) depends only on qs / q_prevs / q_tildes / masses /
        // material parameters -- all frozen for the whole Newton iteration --
        // and writes only body_id_to_{shape,kinetic}_{gradient,hessian}, which
        // nothing reads before assemble_kinetic_shape_k1/k2. So it can be
        // launched on a side stream *before* the contact (dytopo effect) phase
        // and joined inside _assemble_kinetic_shape, where it lands inside the
        // shadow of contact G+H part 1 (24 blocks of 256 at 255 registers on a
        // 40-SM part: one block per SM, ~990 us per launch with nothing else
        // resident). UIPC_ABD_GH_PREPASS=0 = the old path (no side stream).
        // The five placements, the measurement that picks between them and the
        // default all live in `affine_body/abd_gh_prepass_mode.h`. For this
        // class only three cases exist: 0 = no prepass at all, 2 = arm and
        // launch here, and 1/3/4 = arm here and let the call site that owns
        // that slot do the launch (SimEngine's backstop for 1, the K9 contact
        // fork for 3 and 4).
        int          gh_prepass         = 4;
        cudaStream_t gh_prepass_stream  = nullptr;
        cudaEvent_t  gh_prepass_fork    = nullptr;
        cudaEvent_t  gh_prepass_join    = nullptr;
        bool         gh_prepass_armed   = false;
        bool         gh_prepass_pending = false;

        // UIPC_ABD_GH_PREPASS_VERIFY=1: after joining the prepass, snapshot its
        // four output buffers, recompute them in place on the default stream,
        // and count mismatching 64-bit words on device. The claim being tested
        // is that moving these launches to a side stream changes nothing at
        // all -- same kernels, same arguments, same geometry, one writer per
        // output element -- so the expected count is exactly zero.
        bool                                        gh_verify = false;
        cuda_tool::DeviceBuffer<Vector12>           gh_ref_shape_g;
        cuda_tool::DeviceBuffer<Matrix12x12>        gh_ref_shape_h;
        cuda_tool::DeviceBuffer<Vector12>           gh_ref_kin_g;
        cuda_tool::DeviceBuffer<Matrix12x12>        gh_ref_kin_h;
        cuda_tool::DeviceBuffer<unsigned long long> gh_verify_counters;
        unsigned long long                          gh_verify_words      = 0;
        unsigned long long                          gh_verify_mismatches = 0;

        void arm_kinetic_shape_prepass();
        void launch_kinetic_shape_prepass();
        void _compute_kinetic_shape(bool gradient_only, cudaStream_t stream);
        void _join_prepass();
        void _verify_prepass();
        ~Impl();
    };

  private:
    virtual void do_build(DiagLinearSubsystem::BuildInfo& info) override;
    virtual void do_init(InitInfo& info) override;

    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) override;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) override;

    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) override;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info) override;
    virtual void do_arm_assemble_prepass() override;
    virtual void do_launch_assemble_prepass() override;
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info) override;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) override;
    virtual Float do_diag_norm(GlobalLinearSystem::DiagNormInfo& info) override;
    virtual Float do_mass_norm(GlobalLinearSystem::DiagNormInfo& info) override;

    virtual U64 get_uid() const noexcept override;

    friend class ABDLinearSubsystemReporter;
    void add_reporter(ABDLinearSubsystemReporter* reporter);  // only be called by ABDLinearSubsystemReporter

    Impl m_impl;
};
}  // namespace uipc::backend::cuda

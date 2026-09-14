#pragma once
#include <sim_system.h>
#include <linear_system/global_linear_system.h>
namespace uipc::backend::cuda
{
/**
 * @brief A diag linear subsystem represents a submatrix of the global hessian and a subvector of the global gradient.
 */
class DiagLinearSubsystem : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    class InitInfo
    {
      public:
    };

    U64 uid() const noexcept;

    IndexT dof_offset() const noexcept;
    IndexT dof_count() const noexcept;

  protected:
    virtual void do_build(BuildInfo& info);
    virtual void do_init(InitInfo& info) = 0;

    virtual void do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info) = 0;
    virtual void do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info) = 0;

    virtual void do_report_extent(GlobalLinearSystem::DiagExtentInfo& info) = 0;
    virtual void do_assemble(GlobalLinearSystem::DiagInfo& info)            = 0;
    // perf round 6 (s10): two optional hooks around the dytopo-effect (contact)
    // gradient/hessian phase, called once per Newton iteration. A subsystem may
    // use them to run the part of its assembly that depends only on the state
    // frozen at the start of the iteration on a side stream, inside the shadow
    // the contact assembly leaves on the SMs. Both default to nothing.
    //
    // The split into two phases is the whole point, and it was measured:
    //   * `do_arm_assemble_prepass` runs BEFORE the contact phase and is where
    //     the fork event must be recorded, because that is the last point at
    //     which the default stream is not yet ordered behind contact part 1's
    //     join;
    //   * `do_launch_assemble_prepass` runs AFTER the contact launches have
    //     been issued, and is where the kernels must actually be enqueued.
    // Issuing them in the first hook instead puts the small blocks in front of
    // contact part 1 in the work distributor's queue, and one 32-thread block
    // at 192 registers denies a whole SM to a 256-thread block at 255 -- part 1
    // then waits for them instead of covering them.
    //
    // round 6 (s11) refined the second half of that, and the refinement is the
    // reason `GlobalLinearSystem::launch_assembly_prepass` is public: this hook
    // fires too LATE when SimEngine calls it, because
    // GlobalDyTopoEffectManager::_distribute blocks the host on a D2H until the
    // contact kernels have drained, so the launch reaches the queue ~2 us after
    // contact part 1 has already finished. The useful call site is inside the
    // contact assembly itself, just after both contact parts are queued
    // (`UIPC_ABD_GH_PREPASS=4`, the default; see
    // affine_body/abd_gh_prepass_mode.h for the five arms and the numbers).
    // SimEngine's call stays as the backstop, and the launch is idempotent, so
    // a subsystem that adopts these hooks needs no knowledge of which call site
    // actually fired. **Going one slot earlier -- ahead of contact part 2 --
    // costs 3 pp of the win on rigid-wrecking-balls**: the same
    // submission-order effect, one kernel further down the list.
    virtual void do_arm_assemble_prepass() {}
    virtual void do_launch_assemble_prepass() {}
    virtual void do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)  = 0;
    virtual void do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info) = 0;

    virtual Float do_diag_norm(GlobalLinearSystem::DiagNormInfo& info) = 0;
    virtual Float do_mass_norm(GlobalLinearSystem::DiagNormInfo& info) = 0;
    virtual U64   get_uid() const noexcept                             = 0;

  private:
    friend class GlobalLinearSystem;
    virtual void do_build() final override;

    void init();  // only be called by GlobalLinearSystem

    void report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info);
    void receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info);

    void report_extent(GlobalLinearSystem::DiagExtentInfo& info);
    void arm_assemble_prepass();
    void launch_assemble_prepass();
    void assemble(GlobalLinearSystem::DiagInfo& info);
    void accuracy_check(GlobalLinearSystem::AccuracyInfo& info);
    void retrieve_solution(GlobalLinearSystem::SolutionInfo& info);

    Float diag_norm(GlobalLinearSystem::DiagNormInfo& info);
    Float mass_norm(GlobalLinearSystem::DiagNormInfo& info);

    SizeT m_index = ~0ull;

    SimSystemSlot<GlobalLinearSystem> m_global_linear_system;
};
}  // namespace uipc::backend::cuda

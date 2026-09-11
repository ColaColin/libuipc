#pragma once
#include <sim_system.h>
#include <linear_system/global_linear_system.h>

namespace uipc::backend::cuda
{
class DiagLinearSubsystem;

class LocalPreconditioner : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class InitInfo
    {
      public:
    };

    class BuildInfo
    {
      public:
        void connect(DiagLinearSubsystem* system);

      private:
        friend class LocalPreconditioner;
        DiagLinearSubsystem* m_subsystem = nullptr;
    };

  protected:
    virtual void do_build(BuildInfo& info) = 0;
    virtual void do_init(InitInfo& info)   = 0;
    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) = 0;
    // perf/kernels (K13): called once all local preconditioners have been
    // assembled; a preconditioner that assembles on a side stream joins it
    // back into the default stream here.
    virtual void do_finish_assemble() {}
    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) = 0;

  private:
    friend class GlobalLinearSystem;

    virtual void do_build() final override;
    virtual void init();

    void assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info);
    void finish_assemble();
    void apply(GlobalLinearSystem::ApplyPreconditionerInfo& info);
    DiagLinearSubsystem* m_subsystem = nullptr;
};
}  // namespace uipc::backend::cuda

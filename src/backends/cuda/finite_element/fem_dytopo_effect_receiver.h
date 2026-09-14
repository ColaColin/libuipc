#pragma once
#include <dytopo_effect_system/dytopo_effect_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>

namespace uipc::backend::cuda
{
class FEMDyTopoEffectReceiver final : public DyTopoEffectReceiver
{
  public:
    using DyTopoEffectReceiver::DyTopoEffectReceiver;

    class Impl
    {
      public:
        void receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);

        FiniteElementVertexReporter* finite_element_vertex_reporter = nullptr;

        cuda_tool::CDoubletVectorView<Float, 3> gradients;
        cuda_tool::CTripletMatrixView<Float, 3> hessians;
    };

    // device views: joined before they are handed out (s14, see the base)
    auto gradients() const { join_assemble(); return m_impl.gradients; }
    auto hessians() const { join_assemble(); return m_impl.hessians; }
    // host-side sizes for report_extent(): no join
    SizeT gradient_count() const noexcept { return m_impl.gradients.doublet_count(); }
    SizeT hessian_count() const noexcept { return m_impl.hessians.triplet_count(); }

  protected:
    virtual void do_build(DyTopoEffectReceiver::BuildInfo& info) override;

  private:
    friend class FEMLinearSubsystem;

    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) override;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) override;
    Impl m_impl;
};
}  // namespace uipc::backend::cuda

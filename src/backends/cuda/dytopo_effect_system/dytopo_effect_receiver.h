#pragma once
#include <sim_system.h>
#include <dytopo_effect_system/global_dytopo_effect_manager.h>

namespace uipc::backend::cuda
{
class DyTopoEffectReceiver : public SimSystem
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

  protected:
    // round-6 (s14): every accessor that hands out a device view of the
    // received gradient/Hessian calls this first, so a contact assembly whose
    // join was deferred (IPCSimplexNormalContact, UIPC_CONTACT_DEFERRED_JOIN)
    // is joined before its first reader, not at a point somebody remembered.
    // Host-side counts (doublet/triplet counts) do not need it.
    void         join_assemble() const;
    virtual void do_init(InitInfo&);
    virtual void do_report(GlobalDyTopoEffectManager::ClassifyInfo& info) = 0;
    virtual void do_receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info) = 0;
    virtual void do_build(BuildInfo& info) = 0;

  private:
    friend class GlobalDyTopoEffectManager;
    virtual void do_build() final override;
    void         init();  // only be called by GlobalDyTopoEffectManager
    void         report(GlobalDyTopoEffectManager::ClassifyInfo& info);
    void  receive(GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info);
    SizeT m_index = ~0ull;
    GlobalDyTopoEffectManager* m_manager = nullptr;
};
}  // namespace uipc::backend::cuda

#pragma once
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/common/unit.h>

namespace uipc::constitution
{
class UIPC_CONSTITUTION_API DahlFrictionDiscreteShellBending : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;

  public:
    DahlFrictionDiscreteShellBending(const Json& json = default_config());

    void apply_to(geometry::SimplicialComplex& sc,
                  Float                        bending_stiffness,
                  Float                        friction_moment_per_length,
                  Float                        friction_transition_angle);

    static Json default_config();

  private:
    virtual U64 get_uid() const noexcept final override;
    Json        m_config;
};
}  // namespace uipc::constitution

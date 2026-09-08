#include <uipc/constitution/dahl_friction_discrete_shell_bending.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/common/log.h>
#include <cmath>

namespace uipc::constitution
{
// User-defined range UID (official UIDs live in [0, 2^32-1]); marks this local
// extension until it is potentially upstreamed.
constexpr U64 DahlFrictionDiscreteShellBendingUID = 4294967297;

REGISTER_CONSTITUTION_UIDS()
{
    list<builtin::UIDInfo> uid_infos;
    builtin::UIDInfo       info;
    info.uid  = DahlFrictionDiscreteShellBendingUID;
    info.name = "DahlFrictionDiscreteShellBending";
    info.type = string{builtin::FiniteElement};
    uid_infos.push_back(info);
    return uid_infos;
}

DahlFrictionDiscreteShellBending::DahlFrictionDiscreteShellBending(const Json& json)
    : m_config{json}
{
}

void DahlFrictionDiscreteShellBending::apply_to(geometry::SimplicialComplex& sc,
                                                Float bending_stiffness_v,
                                                Float friction_moment_per_length_v,
                                                Float friction_transition_angle_v)
{
    UIPC_ASSERT_THROW(std::isfinite(bending_stiffness_v) && bending_stiffness_v > 0.0,
                      "DahlFrictionDiscreteShellBending requires a finite bending_stiffness > 0, got {}",
                      bending_stiffness_v);
    UIPC_ASSERT_THROW(std::isfinite(friction_moment_per_length_v)
                          && friction_moment_per_length_v >= 0.0,
                      "DahlFrictionDiscreteShellBending requires a finite friction_moment_per_length >= 0 "
                      "(0 disables friction and degenerates to DiscreteShellBending), got {}",
                      friction_moment_per_length_v);
    UIPC_ASSERT_THROW(std::isfinite(friction_transition_angle_v)
                          && friction_transition_angle_v > 0.0,
                      "DahlFrictionDiscreteShellBending requires a finite friction_transition_angle > 0, got {}",
                      friction_transition_angle_v);

    Base::apply_to(sc);

    auto bs = sc.edges().find<Float>("bending_stiffness");
    if(!bs)
        bs = sc.edges().create<Float>("bending_stiffness");
    std::ranges::fill(geometry::view(*bs), bending_stiffness_v);

    auto friction_moment_per_length = sc.edges().find<Float>("friction_moment_per_length");
    if(!friction_moment_per_length)
        friction_moment_per_length = sc.edges().create<Float>("friction_moment_per_length");
    std::ranges::fill(geometry::view(*friction_moment_per_length), friction_moment_per_length_v);

    auto friction_transition_angle = sc.edges().find<Float>("friction_transition_angle");
    if(!friction_transition_angle)
        friction_transition_angle = sc.edges().create<Float>("friction_transition_angle");
    std::ranges::fill(geometry::view(*friction_transition_angle), friction_transition_angle_v);
}

U64 DahlFrictionDiscreteShellBending::get_uid() const noexcept
{
    return DahlFrictionDiscreteShellBendingUID;
}

Json DahlFrictionDiscreteShellBending::default_config()
{
    return Json::object();
}
}  // namespace uipc::constitution

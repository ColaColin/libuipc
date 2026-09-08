#include <pyuipc/constitution/dahl_friction_discrete_shell_bending.h>
#include <uipc/constitution/finite_element_extra_constitution.h>
#include <uipc/constitution/dahl_friction_discrete_shell_bending.h>

namespace pyuipc::constitution
{
using namespace uipc::constitution;

PyDahlFrictionDiscreteShellBending::PyDahlFrictionDiscreteShellBending(py::module& m)
{
    auto class_DahlFrictionDiscreteShellBending =
        py::class_<DahlFrictionDiscreteShellBending, FiniteElementExtraConstitution>(
            m,
            "DahlFrictionDiscreteShellBending",
            R"(DahlFrictionDiscreteShellBending constitution: discrete-shell bending with
Dahl-style internal-friction hysteresis (cloth deformation memory).

The elastic part is identical to DiscreteShellBending (same bending_stiffness
meaning); the friction part adds a rate-independent, recoverable internal
friction moment per hinge that saturates exponentially, so crumpled cloth
stores the deformation path and does not spring open by itself. Use this
constitution INSTEAD OF DiscreteShellBending (it contains the elastic term;
applying both double-counts bending).

State is history-dependent (committed angle and friction moment per edge) and
is included in engine dump/recover.)");

    class_DahlFrictionDiscreteShellBending.def(
        py::init<const Json&>(),
        py::arg("config") = DahlFrictionDiscreteShellBending::default_config(),
        R"(Create a DahlFrictionDiscreteShellBending constitution.
Args:
    config: Configuration dictionary (optional, uses default if not provided).)");

    class_DahlFrictionDiscreteShellBending.def_static(
        "default_config",
        &DahlFrictionDiscreteShellBending::default_config,
        R"(Get the default DahlFrictionDiscreteShellBending configuration.
Returns:
    dict: Default configuration dictionary.)");

    class_DahlFrictionDiscreteShellBending.def("apply_to",
                                               &DahlFrictionDiscreteShellBending::apply_to,
                                               py::arg("sc"),
                                               py::arg("bending_stiffness"),
                                               py::arg("friction_moment_per_length"),
                                               py::arg("friction_transition_angle"),
                                               R"(Apply DahlFrictionDiscreteShellBending constitution to a simplicial complex.
Args:
    sc: SimplicialComplex to apply to.
    bending_stiffness: Elastic bending stiffness in N*m (same meaning as DiscreteShellBending).
    friction_moment_per_length: Saturated internal-friction moment per unit crease
        length in N*m/m. Multiplied by each edge's rest length to get the per-edge
        saturation moment (refinement consistent). 0 disables friction (pure elastic bending).
    friction_transition_angle: Angle in rad over which the friction moment reaches ~63% of
        its saturation when starting from zero; the initial slope is
        friction_moment_per_length / friction_transition_angle per unit crease length.
        A mesh-independent material constant (small = stiff, cardboard-like hysteresis).)");
}
}  // namespace pyuipc::constitution

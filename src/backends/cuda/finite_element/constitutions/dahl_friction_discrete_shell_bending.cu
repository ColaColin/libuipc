#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/finite_element_method.h>
#include <time_integrator/time_integrator.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/constitutions/dahl_friction_discrete_shell_bending_function.h>
#include <utils/make_spd.h>
#include <cuda_tool/spread_launch.h>
#include <cstdlib>
#include <utils/matrix_assembler.h>
#include <utils/dump_utils.h>
#include <algorithm>
#include <cmath>

namespace uipc::backend::cuda
{
namespace
{
    struct Vector2iHash
    {
        size_t operator()(const Vector2i& v) const
        {
            size_t front = v[0];
            size_t end   = v[1];
            return front << 32 | end;
        }
    };

    struct StencilRecord
    {
        Vector4i stencil;
        Float    bending_stiffness = 0.0;
        Float    moment_per_length = 0.0;
        Float    transition_angle  = 0.0;
        // optional imported history (dataset cases): committed angle / moment
        bool  has_history  = false;
        Float theta_commit = 0.0;
        Float F_commit     = 0.0;
    };

    bool stencil_less(const Vector4i& a, const Vector4i& b)
    {
        for(int i = 0; i < 4; ++i)
        {
            if(a[i] != b[i])
                return a[i] < b[i];
        }

        return false;
    }

    namespace DFDSB = sym::dahl_friction_discrete_shell_bending;

    constexpr SizeT StencilSize     = 4;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    __global__ void DahlFrictionDiscreteShellBending_do_compute_energy_kernel(
        cuda_tool::BufferView<Vector4i> stencils,
        cuda_tool::BufferView<Float>    bending_stiffnesses,
        cuda_tool::BufferView<Float>    saturation_moments,
        cuda_tool::BufferView<Float>    transition_angles,
        cuda_tool::BufferView<Float>    theta_bars,
        cuda_tool::BufferView<Float>    theta_commits,
        cuda_tool::BufferView<Float>    F_commits,
        cuda_tool::BufferView<Float>    h_bars,
        cuda_tool::BufferView<Float>    V_bars,
        cuda_tool::BufferView<Float>    L0s,
        cuda_tool::CBufferView<Vector3> xs,
        cuda_tool::BufferView<Float>    energies,
        Float                           dt,
        int                             n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        Vector4i stencil      = stencils(I);
        Float    kappa        = bending_stiffnesses(I);
        Float    M_e          = saturation_moments(I);
        Float    ell_e        = transition_angles(I);
        Float    L0           = L0s(I);
        Float    h_bar        = h_bars(I);
        Float    theta_bar    = theta_bars(I);
        Float    theta_commit = theta_commits(I);
        Float    F_commit     = F_commits(I);
        Float    V_bar        = V_bars(I);

        Vector3 x0 = xs(stencil[0]);
        Vector3 x1 = xs(stencil[1]);
        Vector3 x2 = xs(stencil[2]);
        Vector3 x3 = xs(stencil[3]);

        Float E = DFDSB::E(x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, M_e, ell_e, theta_commit, F_commit);
        energies(I) = E * V_bar * dt * dt;
    }

    // s02 (round 7): Proj = 0 dense 12x12 eigen-solve (this kernel's
    // historical fallback), 1 = K16 block-assembled translation-free 9x9
    // projection, 2 = K7 dense 12x9 basis products. The projection used to be
    // selected by runtime bools, so the single instantiation carried the union
    // of all three paths' stack frames (11 704 B) and Eigen's solver
    // (Solver = 0); it is a template parameter now, exactly like the plain
    // hinge's (rounds 5/6) and the plastic variants' (round 7 s01), so each
    // instantiation carries only its own frame. The dahl energy is
    // P(theta) = kappa*w*del^2 + W(d) of the dihedral angle alone -- the
    // committed state (theta_commit, F_commit) is a per-edge constant within a
    // frame -- and theta goes through vertex differences only, so H t = 0
    // exactly and the translation-free projection applies verbatim.
    // s03 (round 7): Proj = 3 is the Gauss-Newton dahl Hessian --
    // ddEddtheta * grad(theta) grad(theta)^T with the indefinite
    // dEdtheta * hess(theta) term dropped (see
    // dahl_friction_discrete_shell_bending_function.h for the PSD proof).
    // Unlike Proj 0/1/2 this is an algorithmic approximation, not a
    // rearrangement: same energy, same gradient, different search direction.
    // It runs no eigen-solve and never evaluates dihedral_angle_hessian.
    // UIPC_DAHL_GAUSS_NEWTON=0 restores the exact Hessian + projection path.
    // UIPC_DAHL_REDUCED_SPD=0 restores the 12x12 eigen-solve;
    // UIPC_DAHL_BLOCKED_PROJ=0 uses the dense 12x9 basis products of K7
    // instead of the K16 block assembly; UIPC_DAHL_TQL2=0 restores Eigen's
    // SelfAdjointEigenSolver inside the 9x9 (or 12x12) PSD projection.
    // All three off is the pre-round-7 dense 12x12 Eigen path.
    template <int Proj, int Solver>
    __global__ void DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel(
        cuda_tool::BufferView<Vector4i>        stencils,
        cuda_tool::BufferView<Float>           bending_stiffnesses,
        cuda_tool::BufferView<Float>           saturation_moments,
        cuda_tool::BufferView<Float>           transition_angles,
        cuda_tool::BufferView<Float>           theta_bars,
        cuda_tool::BufferView<Float>           theta_commits,
        cuda_tool::BufferView<Float>           F_commits,
        cuda_tool::BufferView<Float>           h_bars,
        cuda_tool::BufferView<Float>           V_bars,
        cuda_tool::BufferView<Float>           L0s,
        cuda_tool::CBufferView<Vector3>        xs,
        cuda_tool::DoubletVectorView<Float, 3> G3s,
        cuda_tool::TripletMatrixView<Float, 3> H3x3s,
        Float                                  dt,
        bool                                   gradient_only,
        int                                    n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        Vector4i stencil      = stencils(I);
        Float    kappa        = bending_stiffnesses(I);
        Float    M_e          = saturation_moments(I);
        Float    ell_e        = transition_angles(I);
        Float    L0           = L0s(I);
        Float    h_bar        = h_bars(I);
        Float    theta_bar    = theta_bars(I);
        Float    theta_commit = theta_commits(I);
        Float    F_commit     = F_commits(I);
        Float    V_bar        = V_bars(I);

        Vector3 x0 = xs(stencil[0]);
        Vector3 x1 = xs(stencil[1]);
        Vector3 x2 = xs(stencil[2]);
        Vector3 x3 = xs(stencil[3]);

        Float Vdt2 = V_bar * dt * dt;

        Vector12    G12;
        Matrix12x12 H12x12;

        if(gradient_only)
        {
            DFDSB::dEdx(G12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, M_e, ell_e, theta_commit, F_commit);
            G12 *= Vdt2;
            DoubletVectorAssembler DVA{G3s};
            DVA.segment<StencilSize>(I * StencilSize).write(stencil, G12);
            return;
        }

        // one evaluation of the dihedral angle / friction response / angle
        // gradient for both G and H (bit-identical to dEdx + ddEddx)
        if constexpr(Proj == 3)
        {
            // s03: Gauss-Newton -- G through the identical expressions of the
            // fused exact path (bit-identical gradient), H as the PSD rank-1
            // ddEddtheta * grad(theta) grad(theta)^T with Vdt2 folded into
            // the coefficient. No projection, no hess(theta).
            DFDSB::dEdx_ddEddx_gauss_newton(
                G12, H12x12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, M_e, ell_e, theta_commit, F_commit, Vdt2);
        }
        else
        {
            DFDSB::dEdx_ddEddx(
                G12, H12x12, x0, x1, x2, x3, L0, h_bar, theta_bar, kappa, M_e, ell_e, theta_commit, F_commit);
        }
        G12 *= Vdt2;
        DoubletVectorAssembler DVA{G3s};
        DVA.segment<StencilSize>(I * StencilSize).write(stencil, G12);

        if constexpr(Proj == 3)
        {
            // PSD by construction (see the function header comment): no
            // eigen-solve at all.
        }
        else
        {
            H12x12 *= Vdt2;
            if constexpr(Proj == 1)
                make_spd_translation_free_4x3_blocked<Solver>(H12x12);  // K16: block-assembled K7 projection
            else if constexpr(Proj == 2)
                make_spd_translation_free_4x3<Solver>(H12x12);  // K7: 9x9 eigen-solve
            else
                make_spd<12, Solver>(H12x12);
        }

        TripletMatrixAssembler TMA{H3x3s};
        TMA.half_block<StencilSize>(I * HalfHessianSize).write(stencil, H12x12);
    }

    // Commit the friction state from the frame's final positions. Called once
    // per frame by the TimeIntegrator after the Newton/line-search loop, i.e.
    // after the accepted step; trial (line-search) positions never reach this.
    __global__ void DahlFrictionDiscreteShellBendingTimeIntegrator_do_update_state_kernel(
        cuda_tool::CBufferView<Vector4i> stencils,
        cuda_tool::BufferView<Float>     theta_commits,
        cuda_tool::BufferView<Float>     F_commits,
        cuda_tool::CBufferView<Float>    saturation_moments,
        cuda_tool::CBufferView<Float>    transition_angles,
        cuda_tool::CBufferView<Vector3>  xs,
        int                              n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        Vector4i stencil = stencils(I);

        Vector3 x0 = xs(stencil[0]);
        Vector3 x1 = xs(stencil[1]);
        Vector3 x2 = xs(stencil[2]);
        Vector3 x3 = xs(stencil[3]);

        Float theta = 0.0;
        if(!DFDSB::safe_dihedral_angle(x0, x1, x2, x3, theta))
            return;

        Float theta_commit = theta_commits(I);
        Float F_commit     = F_commits(I);
        Float M_e          = saturation_moments(I);
        Float ell_e        = transition_angles(I);

        Float d = DFDSB::angle_delta(theta, theta_commit);

        Float F_new = F_commit;
        if(!DFDSB::commit_friction_state(d, F_commit, M_e, ell_e, F_new))
            return;

        theta_commits(I) = theta;
        F_commits(I)     = F_new;
    }
}  // namespace

class DahlFrictionDiscreteShellBending final : public FiniteElementExtraConstitution
{
    // User-defined UID (official range is [0, 2^32-1]; user range is
    // [2^32, 2^64-1]). 2^32+1 marks this local extension until it is
    // (potentially) upstreamed with an official UID.
    static constexpr U64   DahlFrictionDiscreteShellBendingUID = 4294967297;
    static constexpr SizeT StencilSize                         = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;
    using Base = FiniteElementExtraConstitution;

  public:
    using Base::Base;
    U64 get_uid() const noexcept override
    {
        return DahlFrictionDiscreteShellBendingUID;
    }

    class InitInfo
    {
      public:
        bool        valid_bending() const { return oppo_verts.size() == 2; }
        IndexT      edge_index = -1;
        set<IndexT> oppo_verts;
    };

    vector<Vector4i> h_stencils;
    vector<Float>    h_bending_stiffness;
    vector<Float>    h_moment_per_length;
    vector<Float>    h_transition_angle;
    vector<Float>    h_rest_lengths;
    vector<Float>    h_h_bars;
    vector<Float>    h_theta_bars;
    vector<Float>    h_saturation_moments;
    vector<Float>    h_V_bars;

    cuda_tool::DeviceBuffer<Vector4i> stencils;
    cuda_tool::DeviceBuffer<Float>    bending_stiffnesses;
    cuda_tool::DeviceBuffer<Float>    saturation_moments;
    cuda_tool::DeviceBuffer<Float>    transition_angles;
    cuda_tool::DeviceBuffer<Float>    rest_lengths;
    cuda_tool::DeviceBuffer<Float>    h_bars;
    cuda_tool::DeviceBuffer<Float>    theta_bars;
    cuda_tool::DeviceBuffer<Float>    theta_commits;
    cuda_tool::DeviceBuffer<Float>    F_commits;
    cuda_tool::DeviceBuffer<Float>    V_bars;

    // s02 (round 7): translation-free 9x9 PSD projection of the hinge
    // Hessian, now dispatched at compile time like the plain hinge's and the
    // plastic variants' (the runtime bools cost every instantiation the union
    // of all three projection paths' stack frames). Defaults on =
    // <Proj=1 blocked, Solver=1 QL>; all three knobs =0 is the historical
    // dense 12x12 Eigen path.
    // s03 (round 7): Gauss-Newton dahl Hessian (ddEddtheta * grad(theta)
    // grad(theta)^T, PSD by construction, no eigen-solve, no hess(theta) --
    // the plain hinge's round-6 s02 analogue). UIPC_DAHL_GAUSS_NEWTON=0
    // restores the exact Hessian followed by the 9x9 PSD projection, and then
    // the three switches below select which projection.
    // UIPC_DAHL_REDUCED_SPD=0 restores the 12x12 eigen-solve;
    // UIPC_DAHL_BLOCKED_PROJ=0 = the dense 12x9 basis products of K7;
    // UIPC_DAHL_TQL2=0 restores Eigen's SelfAdjointEigenSolver.
    bool m_reduced_spd  = true;
    bool m_blocked_proj = true;
    bool m_tql2         = true;
    bool m_gauss_newton = true;

    virtual void do_build(BuildInfo& info) override
    {
        const char* e  = std::getenv("UIPC_DAHL_REDUCED_SPD");
        m_reduced_spd  = !(e && e[0] == '0');
        const char* b  = std::getenv("UIPC_DAHL_BLOCKED_PROJ");
        m_blocked_proj = !(b && b[0] == '0');
        const char* t  = std::getenv("UIPC_DAHL_TQL2");
        m_tql2         = !(t && t[0] == '0');
        const char* gn = std::getenv("UIPC_DAHL_GAUSS_NEWTON");
        m_gauss_newton = !(gn && gn[0] == '0');
    }

    virtual void do_init(FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        auto geo_slots    = world().scene().geometries();

        vector<StencilRecord> stencil_records;

        info.for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                unordered_map<Vector2i, InitInfo, Vector2iHash> stencil_map;

                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
                auto vertex_offset_v = vertex_offset->view().front();

                auto edges = sc.edges().topo().view();
                for(auto&& [i, e] : enumerate(edges))
                {
                    Vector2i E = e;
                    std::sort(E.begin(), E.end());
                    stencil_map[E].edge_index = i;
                }

                auto triangles = sc.triangles().topo().view();
                for(auto&& t : triangles)
                {
                    Vector3i T = t;
                    std::sort(T.begin(), T.end());

                    Vector2i E01 = {T[0], T[1]};
                    Vector2i E02 = {T[0], T[2]};
                    Vector2i E12 = {T[1], T[2]};

                    stencil_map[E01].oppo_verts.insert(T[2]);
                    stencil_map[E02].oppo_verts.insert(T[1]);
                    stencil_map[E12].oppo_verts.insert(T[0]);
                }

                auto bending_stiffnesses = sc.edges().find<Float>("bending_stiffness");
                auto moment_per_lengths =
                    sc.edges().find<Float>("friction_moment_per_length");
                auto transition_angles = sc.edges().find<Float>("friction_transition_angle");
                UIPC_ASSERT(bending_stiffnesses, "Bending stiffness not found, why?");
                UIPC_ASSERT(moment_per_lengths, "friction_moment_per_length not found, why?");
                UIPC_ASSERT(transition_angles, "friction_transition_angle not found, why?");

                auto bs_view = bending_stiffnesses->view();
                auto ml_view = moment_per_lengths->view();
                auto ta_view = transition_angles->view();

                // Optional per-edge history import: a stored (crumpled) state
                // carries the committed Dahl state of every hinge. Both
                // attributes must be present; otherwise fresh cloth is assumed.
                auto theta_commit_attr = sc.edges().find<Float>("dahl_theta_commit");
                auto F_commit_attr = sc.edges().find<Float>("dahl_friction_commit");
                bool has_history = theta_commit_attr && F_commit_attr;
                UIPC_ASSERT(!(bool(theta_commit_attr) ^ bool(F_commit_attr)),
                            "dahl_theta_commit / dahl_friction_commit must be given together");

                for(auto&& [E, stencil_info] : stencil_map)
                {
                    if(!stencil_info.valid_bending())
                        continue;

                    Vector4i stencil{*stencil_info.oppo_verts.begin(),
                                     E(0),
                                     E(1),
                                     *stencil_info.oppo_verts.rbegin()};

                    StencilRecord record{
                        .stencil           = stencil.array() + vertex_offset_v,
                        .bending_stiffness = bs_view[stencil_info.edge_index],
                        .moment_per_length = ml_view[stencil_info.edge_index],
                        .transition_angle  = ta_view[stencil_info.edge_index],
                    };
                    if(has_history)
                    {
                        record.has_history = true;
                        record.theta_commit =
                            theta_commit_attr->view()[stencil_info.edge_index];
                        record.F_commit = F_commit_attr->view()[stencil_info.edge_index];
                        UIPC_ASSERT(std::isfinite(record.theta_commit)
                                        && std::isfinite(record.F_commit),
                                    "non-finite imported Dahl history on edge {}",
                                    stencil_info.edge_index);
                    }
                    stencil_records.push_back(record);
                }
            });

        std::ranges::sort(stencil_records,
                          [](const StencilRecord& a, const StencilRecord& b)
                          { return stencil_less(a.stencil, b.stencil); });

        h_stencils.resize(stencil_records.size());
        h_bending_stiffness.resize(stencil_records.size());
        h_moment_per_length.resize(stencil_records.size());
        h_transition_angle.resize(stencil_records.size());
        vector<bool>  h_has_history(stencil_records.size());
        vector<Float> h_theta_commit_init(stencil_records.size());
        vector<Float> h_F_commit_init(stencil_records.size());
        for(auto&& [i, record] : enumerate(stencil_records))
        {
            h_stencils[i]          = record.stencil;
            h_bending_stiffness[i] = record.bending_stiffness;
            h_moment_per_length[i] = record.moment_per_length;
            h_transition_angle[i]  = record.transition_angle;
            h_has_history[i]       = record.has_history;
            h_theta_commit_init[i] = record.theta_commit;
            h_F_commit_init[i]     = record.F_commit;
        }

        auto x_bars      = info.rest_positions();
        auto thicknesses = info.thicknesses();
        h_rest_lengths.resize(h_stencils.size());
        h_h_bars.resize(h_stencils.size());
        h_theta_bars.resize(h_stencils.size());
        h_saturation_moments.resize(h_stencils.size());
        h_V_bars.resize(h_stencils.size());

        // Fresh, never-bent cloth: committed angle = rest angle, F = 0.
        vector<Float> h_theta_commits(h_stencils.size());
        vector<Float> h_F_commits(h_stencils.size(), 0.0);

        for(auto&& [i, stencil] : enumerate(h_stencils))
        {
            Vector3 X0         = x_bars[stencil[0]];
            Vector3 X1         = x_bars[stencil[1]];
            Vector3 X2         = x_bars[stencil[2]];
            Vector3 X3         = x_bars[stencil[3]];
            Float   thickness0 = thicknesses[stencil[0]];
            Float   thickness1 = thicknesses[stencil[1]];
            Float   thickness2 = thicknesses[stencil[2]];
            Float   thickness3 = thicknesses[stencil[3]];

            Float L0, V_bar, h_bar, theta_bar;
            DFDSB::compute_constants(
                L0, h_bar, theta_bar, V_bar, X0, X1, X2, X3, thickness0, thickness1, thickness2, thickness3);

            h_rest_lengths[i]       = L0;
            h_h_bars[i]             = h_bar;
            h_theta_bars[i]         = theta_bar;
            h_saturation_moments[i] = h_moment_per_length[i] * L0;
            // imported history (dataset case) or fresh cloth (rest angle, F=0)
            h_theta_commits[i] = h_has_history[i] ? h_theta_commit_init[i] : theta_bar;
            h_F_commits[i] = h_has_history[i] ? h_F_commit_init[i] : 0.0;
            h_V_bars[i]    = V_bar;
        }

        stencils.resize(h_stencils.size());
        stencils.view().copy_from(h_stencils.data());

        bending_stiffnesses.resize(h_bending_stiffness.size());
        bending_stiffnesses.view().copy_from(h_bending_stiffness.data());

        saturation_moments.resize(h_saturation_moments.size());
        saturation_moments.view().copy_from(h_saturation_moments.data());

        transition_angles.resize(h_transition_angle.size());
        transition_angles.view().copy_from(h_transition_angle.data());

        rest_lengths.resize(h_rest_lengths.size());
        rest_lengths.view().copy_from(h_rest_lengths.data());

        h_bars.resize(h_h_bars.size());
        h_bars.view().copy_from(h_h_bars.data());

        theta_bars.resize(h_theta_bars.size());
        theta_bars.view().copy_from(h_theta_bars.data());

        theta_commits.resize(h_theta_commits.size());
        theta_commits.view().copy_from(h_theta_commits.data());

        F_commits.resize(h_F_commits.size());
        F_commits.view().copy_from(h_F_commits.data());

        V_bars.resize(h_V_bars.size());
        V_bars.view().copy_from(h_V_bars.data());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(stencils.size());
        info.gradient_count(stencils.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(stencils.size() * HalfHessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        auto k = DahlFrictionDiscreteShellBending_do_compute_energy_kernel;
        int  n = (int)info.energies().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                stencils.view(),
                bending_stiffnesses.view(),
                saturation_moments.view(),
                transition_angles.view(),
                theta_bars.view(),
                theta_commits.view(),
                F_commits.view(),
                h_bars.view(),
                V_bars.view(),
                rest_lengths.view(),
                info.xs(),
                info.energies(),
                info.dt(),
                n);
        }
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        int n = (int)stencils.size();
        if(n == 0)
            return;

        auto launch = [&](auto k)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                stencils.view(),
                bending_stiffnesses.view(),
                saturation_moments.view(),
                transition_angles.view(),
                theta_bars.view(),
                theta_commits.view(),
                F_commits.view(),
                h_bars.view(),
                V_bars.view(),
                rest_lengths.view(),
                info.xs(),
                info.gradients(),
                info.hessians(),
                info.dt(),
                info.gradient_only(),
                n);
        };

        // s03 (round 7): UIPC_DAHL_GN_VERIFY=1 is the numerics probe for the
        // Gauss-Newton path, the plain hinge's UIPC_DSB_GN_VERIFY shape: it
        // runs the *real* shipped kernel <Proj=1, Solver=1> first, snapshots
        // the gradient doublets AND the Hessian triplets it wrote, then runs
        // <Proj=3, Solver=1> over the same inputs and counts the mismatching
        // 32-bit words on device. The gradient is supposed to be bit-identical
        // between the two instantiations (only the Hessian is approximated) --
        // that is proved on the binary, not by reading the source; the Hessian
        // word count is the deviation census of the controlled perturbation
        // (its relative size is measured by this step's standalone verifier).
        // The live state after the pair is the Gauss-Newton launch's, so the
        // run continues on the shipped path while it is being checked.
        static const bool gn_verify = []
        {
            const char* v = std::getenv("UIPC_DAHL_GN_VERIFY");
            return v && v[0] != '0';
        }();
        if(m_gauss_newton && gn_verify)
        {
            static cuda_tool::SpreadVerifier sv_gn{
                "DahlFrictionDiscreteShellBending::gauss_newton_gradient"};
            static cuda_tool::SpreadVerifier sv_gnh{
                "DahlFrictionDiscreteShellBending::gauss_newton_hessian"};
            sv_gn.begin();
            sv_gn.add_doublet(info.gradients());
            sv_gnh.begin();
            sv_gnh.add_triplet(info.hessians());
            launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<1, 1>);
            sv_gn.snapshot();
            sv_gnh.snapshot();
            launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<3, 1>);
            sv_gn.compare();
            sv_gnh.compare();
            return;
        }

        // s02 (round 7): the plain hinge's dispatch shape -- only the
        // instantiations the knobs can reach, each launched by name.
        // s03: the Gauss-Newton arm first (Solver is irrelevant there --
        // Proj = 3 runs no eigen-solve).
        if(m_gauss_newton)
        {
            launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<3, 1>);
        }
        else if(m_tql2)
        {
            if(!m_reduced_spd)
                launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<0, 1>);
            else if(m_blocked_proj)
                launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<1, 1>);
            else
                launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<2, 1>);
        }
        else
        {
            if(!m_reduced_spd)
                launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<0, 0>);
            else if(m_blocked_proj)
                launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<1, 0>);
            else
                launch(DahlFrictionDiscreteShellBending_do_compute_gradient_hessian_kernel<2, 0>);
        }
    }
};
REGISTER_SIM_SYSTEM(DahlFrictionDiscreteShellBending);

class DahlFrictionDiscreteShellBendingTimeIntegrator final : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<DahlFrictionDiscreteShellBending> dfdsb;
    SimSystemSlot<FiniteElementMethod>              fem;
    BufferDump                                      dump_theta_commits;
    BufferDump                                      dump_F_commits;

    void do_build(TimeIntegrator::BuildInfo& info) override
    {
        dfdsb = require<DahlFrictionDiscreteShellBending>();
        fem   = require<FiniteElementMethod>();
    }

    void do_init(TimeIntegrator::InitInfo& info) override {}

    void do_predict_dof(TimeIntegrator::PredictDofInfo& info) override {}

    void do_update_state(TimeIntegrator::UpdateVelocityInfo& info) override
    {
        auto k = DahlFrictionDiscreteShellBendingTimeIntegrator_do_update_state_kernel;
        int n = (int)dfdsb->stencils.size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                dfdsb->stencils.cview(),
                dfdsb->theta_commits.view(),
                dfdsb->F_commits.view(),
                dfdsb->saturation_moments.cview(),
                dfdsb->transition_angles.cview(),
                fem->xs().cview(),
                n);
        }
    }

    bool do_dump(DumpInfo& info) override
    {
        auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
        auto frame = info.frame();

        return dump_theta_commits.dump(fmt::format("{}theta_commit.{}", path, frame),
                                       dfdsb->theta_commits)
               && dump_F_commits.dump(fmt::format("{}friction_moment.{}", path, frame),
                                      dfdsb->F_commits);
    }

    bool do_try_recover(RecoverInfo& info) override
    {
        auto path  = info.dump_path(UIPC_RELATIVE_SOURCE_FILE);
        auto frame = info.frame();

        return dump_theta_commits.load(fmt::format("{}theta_commit.{}", path, frame))
               && dump_F_commits.load(fmt::format("{}friction_moment.{}", path, frame));
    }

    void do_apply_recover(RecoverInfo& info) override
    {
        dump_theta_commits.apply_to(dfdsb->theta_commits);
        dump_F_commits.apply_to(dfdsb->F_commits);
    }

    void do_clear_recover(RecoverInfo& info) override
    {
        dump_theta_commits.clean_up();
        dump_F_commits.clean_up();
    }
};
REGISTER_SIM_SYSTEM(DahlFrictionDiscreteShellBendingTimeIntegrator);
}  // namespace uipc::backend::cuda

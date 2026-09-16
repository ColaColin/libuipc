#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>
#include <kernel_cout.h>
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Dense>
#include <utils/codim_thickness.h>
#include <utils/make_spd.h>
#include <utils/matrix_assembler.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
namespace
{
    namespace NH = sym::neo_hookean_shell_2d;

    constexpr SizeT StencilSize     = 3;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    __global__ void NeoHookeanShell2D_do_init_kernel(cuda_tool::CBufferView<Vector3i> prims,
                                                     cuda_tool::CBufferView<Vector3> x_bars,
                                                     cuda_tool::BufferView<Matrix2x2> inv_B_mats,
                                                     int n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        Vector9  X_bar;
        Vector3i idx = prims(I);
        for(int i = 0; i < 3; ++i)
            X_bar.segment<3>(3 * i) = x_bars(idx(i));
        Matrix2x2 B;
        NH::A(B, X_bar);

        inv_B_mats(I) = cuda_tool::eigen::inverse(B);
    }

    __global__ void NeoHookeanShell2D_do_compute_energy_kernel(
        cuda_tool::CBufferView<Float>     lambdas,
        cuda_tool::CBufferView<Float>     mus,
        cuda_tool::CBufferView<Float>     rest_areas,
        cuda_tool::CBufferView<Float>     thicknesses,
        cuda_tool::BufferView<Float>      energies,
        cuda_tool::CBufferView<Vector3i>  indices,
        cuda_tool::CBufferView<Vector3>   xs,
        cuda_tool::CBufferView<Matrix2x2> IBs,
        Float                             dt,
        int                               n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        Vector9          X;
        Vector3i         idx = indices(I);
        const Matrix2x2& IB  = IBs(I);
        for(int i = 0; i < 3; ++i)
            X.segment<3>(3 * i) = xs(idx(i));

        Float lambda = lambdas(I);
        Float mu     = mus(I);

        Float rest_area = rest_areas(I);
        Float thickness = triangle_thickness(
            thicknesses(idx(0)), thicknesses(idx(1)), thicknesses(idx(2)));
        Float E;
        NH::E(E, lambda, mu, X, IB);

        // thickness is one-sided so we multiply by 2
        Float Vdt2  = rest_area * 2 * thickness * dt * dt;
        energies(I) = E * Vdt2;
    }

    // perf/round7 (s07): Proj = 0 dense 9x9 eigen-solve (the path every
    // round before 7 used here), 1 = block-assembled translation-free 6x6
    // projection, 2 = dense 9x6 basis products. The NeoHookeanShell2D energy
    // depends on the vertices only through edge differences (the 2x2 deformed
    // metric B and IB = inv(rest metric)), so H t = 0 exactly for every rigid
    // translation t and the translation-free projection applies with one
    // vertex fewer than the hinge's 4x3 variant. Proj is a template parameter
    // so each instantiation carries only one code path's stack frame (the
    // s14 lesson); Solver = 0 restores Eigen's SelfAdjointEigenSolver inside
    // the 9x9 (or 6x6) PSD projection, 1 = the fixed-size tridiagonal QL.
    template <int Proj, int Solver>
    __global__ void NeoHookeanShell2D_do_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Float>          lambdas,
        cuda_tool::CBufferView<Float>          mus,
        cuda_tool::CBufferView<Vector3i>       indices,
        cuda_tool::CBufferView<Vector3>        xs,
        cuda_tool::CBufferView<Matrix2x2>      IBs,
        cuda_tool::CBufferView<Float>          thicknesses,
        cuda_tool::DoubletVectorView<Float, 3> G3s,
        cuda_tool::TripletMatrixView<Float, 3> H3x3s,
        cuda_tool::CBufferView<Float>          rest_areas,
        Float                                  dt,
        SizeT                                  half_hessian_size,
        bool                                   gradient_only,
        int                                    n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        Vector9  X;
        Vector3i idx = indices(I);
        for(int i = 0; i < 3; ++i)
            X.segment<3>(3 * i) = xs(idx(i));

        Matrix2x2 IB = IBs(I);

        Float lambda    = lambdas(I);
        Float mu        = mus(I);
        Float rest_area = rest_areas(I);
        Float thickness = triangle_thickness(
            thicknesses(idx(0)), thicknesses(idx(1)), thicknesses(idx(2)));

        // thickness is one-sided so we multiply by 2
        Float Vdt2 = rest_area * 2 * thickness * dt * dt;

        Vector9 G;
        NH::dEdX(G, lambda, mu, X, IB);
        G *= Vdt2;
        DoubletVectorAssembler DVA{G3s};
        DVA.segment<StencilSize>(I * StencilSize).write(idx, G);

        if(gradient_only)
            return;

        Matrix9x9 H;
        NH::ddEddX(H, lambda, mu, X, IB);

        if constexpr(Proj == 1)
            make_spd_translation_free_3x3_blocked<Solver>(H);
        else if constexpr(Proj == 2)
            make_spd_translation_free_3x3<Solver>(H);
        else
            make_spd<9, Solver>(H);

        H *= Vdt2;

        TripletMatrixAssembler TMA{H3x3s};
        TMA.half_block<StencilSize>(I * half_hessian_size).write(idx, H);
    }
}  // namespace

class NeoHookeanShell2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 11;
    static constexpr SizeT StencilSize     = 3;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using Codim2DConstitution::Codim2DConstitution;

    vector<Float> h_lambdas;
    vector<Float> h_mus;

    cuda_tool::DeviceBuffer<Float>     lambdas;
    cuda_tool::DeviceBuffer<Float>     mus;
    cuda_tool::DeviceBuffer<Matrix2x2> inv_B_matrices;

    SimSystemSlot<FiniteElementMethod> fem;

    // s07: translation-free 6x6 PSD projection of the membrane Hessian.
    // UIPC_NHS2D_REDUCED_SPD=0 restores the dense 9x9 eigen-solve;
    // UIPC_NHS2D_BLOCKED_PROJ=0 uses the dense 9x6 basis products;
    // UIPC_NHS2D_TQL2=0 -> Eigen's SelfAdjointEigenSolver in the PSD
    // projection (the pre-round-7 path). All three =0 is <0,0>, the
    // historical make_spd<9> + Eigen path byte-identical to main.
    bool m_reduced_spd  = true;
    bool m_blocked_proj = true;
    bool m_tql2         = true;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override
    {
        fem = require<FiniteElementMethod>();

        const char* e  = std::getenv("UIPC_NHS2D_REDUCED_SPD");
        m_reduced_spd  = !(e && e[0] == '0');
        const char* b  = std::getenv("UIPC_NHS2D_BLOCKED_PROJ");
        m_blocked_proj = !(b && b[0] == '0');
        const char* t  = std::getenv("UIPC_NHS2D_TQL2");
        m_tql2         = !(t && t[0] == '0');
    }

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_mus.resize(N);
        h_lambdas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto lambda = sc.triangles().find<Float>("lambda");
                auto mu     = sc.triangles().find<Float>("mu");


                return zip(lambda->view(), mu->view());
            },
            [&](const ForEachInfo& I, auto lambda_mu)
            {
                auto vI = I.global_index();

                auto&& [lambda, mu] = lambda_mu;

                h_lambdas[vI] = lambda;
                h_mus[vI]     = mu;
            });

        lambdas.resize(N);
        lambdas.view().copy_from(h_lambdas.data());

        mus.resize(N);
        mus.view().copy_from(h_mus.data());

        auto& cinfo       = info.constitution_info();
        auto  prim_offset = cinfo.primitive_offset;
        auto  prim_count  = cinfo.primitive_count;

        auto prims  = fem->codim_2ds().subview(prim_offset, prim_count);
        auto x_bars = fem->x_bars();

        inv_B_matrices.resize(N);

        // Precompute inverse of rest shape matrix for each triangle
        auto k = NeoHookeanShell2D_do_init_kernel;
        int  n = (int)N;
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                prims, x_bars, inv_B_matrices.view(), n);
        }
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(mus.size());
        info.gradient_count(mus.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(mus.size() * HalfHessianSize);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        auto k = NeoHookeanShell2D_do_compute_energy_kernel;
        int  n = (int)info.indices().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                lambdas.cview(),
                mus.cview(),
                info.rest_areas(),
                info.thicknesses(),
                info.energies(),
                info.indices(),
                info.xs(),
                inv_B_matrices.cview(),
                info.dt(),
                n);
        }
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using K = decltype(NeoHookeanShell2D_do_compute_gradient_hessian_kernel<1, 1>);
        int n = (int)info.indices().size();
        if(n == 0)
            return;

        auto launch = [&](K ke)
        {
            ke<<<cuda_tool::best_grid_dim(n, ke), cuda_tool::best_block_dim(ke), 0, nullptr>>>(
                lambdas.cview(),
                mus.cview(),
                info.indices(),
                info.xs(),
                inv_B_matrices.cview(),
                info.thicknesses(),
                info.gradients(),
                info.hessians(),
                info.rest_areas(),
                info.dt(),
                HalfHessianSize,
                info.gradient_only(),
                n);
        };

        if(m_tql2)
        {
            if(!m_reduced_spd)
                launch(NeoHookeanShell2D_do_compute_gradient_hessian_kernel<0, 1>);
            else if(m_blocked_proj)
                launch(NeoHookeanShell2D_do_compute_gradient_hessian_kernel<1, 1>);
            else
                launch(NeoHookeanShell2D_do_compute_gradient_hessian_kernel<2, 1>);
        }
        else
        {
            if(!m_reduced_spd)
                launch(NeoHookeanShell2D_do_compute_gradient_hessian_kernel<0, 0>);
            else if(m_blocked_proj)
                launch(NeoHookeanShell2D_do_compute_gradient_hessian_kernel<1, 0>);
            else
                launch(NeoHookeanShell2D_do_compute_gradient_hessian_kernel<2, 0>);
        }
    }
};

REGISTER_SIM_SYSTEM(NeoHookeanShell2D);
}  // namespace uipc::backend::cuda

#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_utils.h>
#include <kernel_cout.h>
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Dense>
#include <utils/matrix_assembler.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
namespace
{
    // Stiff-GIPC SNK1 (energy, gradient and the analytically SPD-projected
    // Hessian) replaces the SymEigen-generated SNH + generic make_spd EVD
    namespace SNH = snk1;

    constexpr SizeT StencilSize     = 4;
    constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    __global__ void StableNeoHookean3D_do_compute_energy_kernel(
        cuda_tool::CBufferView<Float>     mus,
        cuda_tool::CBufferView<Float>     lambdas,
        cuda_tool::BufferView<Float>      energies,
        cuda_tool::CBufferView<Vector4i>  indices,
        cuda_tool::CBufferView<Vector3>   xs,
        cuda_tool::CBufferView<Matrix3x3> Dm_invs,
        cuda_tool::CBufferView<Float>     volumes,
        Float                             dt,
        int                               n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const Vector4i&  tet    = indices(I);
        const Matrix3x3& Dm_inv = Dm_invs(I);
        Float            mu     = mus(I);
        Float            lambda = lambdas(I);

        const Vector3& x0 = xs(tet(0));
        const Vector3& x1 = xs(tet(1));
        const Vector3& x2 = xs(tet(2));
        const Vector3& x3 = xs(tet(3));

        auto F = fem::F(x0, x1, x2, x3, Dm_inv);

        Float E;

        SNH::E(E, mu, lambda, F);
        E *= dt * dt * volumes(I);
        energies(I) = E;
    }

    // s15: HoistStretch = the three stretch-mode matrices q_m = U diag(bv_m) V^T
    // and the twelve contractions q_m * shape_gradients.col(i) depend only on
    // (m, i), never on the (i, j) stencil block, but the fully unrolled block
    // loop below rebuilds them for every one of the 10 blocks. Evaluating them
    // once, with the same expressions in the same order, is bit-identical.
    // A template parameter so that each instantiation keeps its own register /
    // stack profile (UIPC_SNK1_HOIST_STRETCH=0 restores the in-loop form).
    template <bool HoistStretch>
    __global__ void StableNeoHookean3D_do_compute_gradient_hessian_kernel(
        cuda_tool::CBufferView<Float>          mus,
        cuda_tool::CBufferView<Float>          lambdas,
        cuda_tool::CBufferView<Vector4i>       indices,
        cuda_tool::CBufferView<Vector3>        xs,
        cuda_tool::CBufferView<Matrix3x3>      Dm_invs,
        cuda_tool::DoubletVectorView<Float, 3> G3s,
        cuda_tool::TripletMatrixView<Float, 3> H3x3s,
        cuda_tool::CBufferView<Float>          volumes,
        Float                                  dt,
        bool                                   gradient_only,
        int                                    n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        const Vector4i&  tet    = indices(I);
        const Matrix3x3& Dm_inv = Dm_invs(I);
        Float            mu     = mus(I);
        Float            lambda = lambdas(I);

        const Vector3& x0 = xs(tet(0));
        const Vector3& x1 = xs(tet(1));
        const Vector3& x2 = xs(tet(2));
        const Vector3& x3 = xs(tet(3));

        auto F = fem::F(x0, x1, x2, x3, Dm_inv);

        auto Vdt2 = volumes(I) * dt * dt;

        Matrix3x3 dEdF;
        SNH::dEdF(dEdF, mu, lambda, F);
        auto VecdEdF = flatten(dEdF);
        VecdEdF *= Vdt2;

        auto shape_gradients = fem::tetrahedron_shape_gradients(Dm_inv);

        DoubletVectorAssembler DVA{G3s};
#pragma unroll
        for(int i = 0; i < StencilSize; ++i)
        {
            Vector3 G = fem::project_F_gradient(shape_gradients.col(i), VecdEdF);
            DVA(I * StencilSize + i).write(tet(i), G);
        }

        if(gradient_only)
            return;

        // Factored form of the analytically SPD-projected 9x9 energy Hessian
        // (Stiff SNK1). Same eigensystem as SNH::ddEddF_spd, but the 9x9
        // H = Q diag(lam) Q^T is never materialized: each 3x3 stencil block
        // accumulates block_ij = sum_m lam_m * w_m_i * w_m_j^T, where the
        // eigenvectors are contracted with the shape-gradient columns on the
        // fly. Verified against the explicit path to 5.3e-13 relative over
        // 4000 randomized deformation cases (host build). This keeps the
        // kernel's register footprint small enough to avoid the
        // 1.7 KB/thread stack spill of the explicit 9x9 version.
        const Float J = F.determinant();
        Matrix3x3   U, V;
        Vector3     S;
        math::qr_svd(F, S, U, V);

        const Float     evScale = lambda * (J - 1.0) - mu;
        const Matrix3x3 sV      = V * Float(0.70710678118654752440);

        // stretch-block eigensystem (identical to SNH::ddEddF_spd's)
        Vector3   block_values;
        Matrix3x3 block_vectors;
        {
            Matrix3x3 A;
            A(0, 0)              = mu + lambda * S(1) * S(1) * S(2) * S(2);
            A(1, 1)              = mu + lambda * S(0) * S(0) * S(2) * S(2);
            A(2, 2)              = mu + lambda * S(0) * S(0) * S(1) * S(1);
            const Float evScale2 = lambda * (2.0 * J - 1.0) - mu;
            A(0, 1) = A(1, 0) = evScale2 * S(2);
            A(0, 2) = A(2, 0) = evScale2 * S(1);
            A(1, 2) = A(2, 1) = evScale2 * S(0);
            cuda_tool::eigen::evd<Float, 3>(A, block_values, block_vectors);
        }

        // sa[i][k] = sV.col(k) . shape_gradients.col(i)
        Float sa[StencilSize][3];
#pragma unroll
        for(int i = 0; i < StencilSize; ++i)
#pragma unroll
            for(int k = 0; k < 3; ++k)
                sa[i][k] = sV(0, k) * shape_gradients(0, i)
                           + sV(1, k) * shape_gradients(1, i)
                           + sV(2, k) * shape_gradients(2, i);

        // column pairs of the twist/flip eigenvectors (U.col(p2)*sV.col(p1)^T
        // -/+ U.col(p1)*sV.col(p2)^T), matching build_twist_flip_eigenvectors
        constexpr int TwistFlipPairs[3][2] = {{1, 2}, {0, 2}, {0, 1}};

        // s15: sw[m][i] = (U diag(block_vectors.col(m)) V^T) * shape_gradients.col(i)
        Vector3 sw[3][StencilSize];
        if constexpr(HoistStretch)
        {
#pragma unroll
            for(int m = 0; m < 3; ++m)
            {
                const Matrix3x3 q =
                    U * block_vectors.col(m).asDiagonal() * V.transpose();
#pragma unroll
                for(int i = 0; i < StencilSize; ++i)
#pragma unroll
                    for(int ip = 0; ip < 3; ++ip)
                    {
                        Float sl = 0;
#pragma unroll
                        for(int k = 0; k < 3; ++k)
                            sl += shape_gradients(k, i) * q(ip, k);
                        sw[m][i](ip) = sl;
                    }
            }
        }

        IndexT hessian_offset = I * HalfHessianSize;
#pragma unroll
        for(int i = 0; i < StencilSize; ++i)
        {
#pragma unroll
            for(int j = i; j < StencilSize; ++j)
            {
                int left  = i;
                int right = j;
                if(tet(left) > tet(right))
                {
                    left  = j;
                    right = i;
                }

                Matrix3x3 H = Matrix3x3::Zero();
#pragma unroll
                for(int m = 0; m < 3; ++m)
                {
                    const int     p1 = TwistFlipPairs[m][0];
                    const int     p2 = TwistFlipPairs[m][1];
                    const Vector3 Up1{U(0, p1), U(1, p1), U(2, p1)};
                    const Vector3 Up2{U(0, p2), U(1, p2), U(2, p2)};
#pragma unroll
                    for(int sgn = 0; sgn < 2; ++sgn)  // 0: twist, 1: flip
                    {
                        Float l = (mu + (sgn ? -S(m) : S(m)) * evScale) * Vdt2;
                        if(l < 0.0)
                            l = 0.0;
                        const Float s = sgn ? 1.0 : -1.0;
                        const Vector3 w_l = Up2 * sa[left][p1] + s * Up1 * sa[left][p2];
                        const Vector3 w_r =
                            Up2 * sa[right][p1] + s * Up1 * sa[right][p2];
                        H += (l * w_l) * w_r.transpose();
                    }
                }
#pragma unroll
                for(int m = 0; m < 3; ++m)  // stretch modes
                {
                    Float l = block_values(m) * Vdt2;
                    if(l < 0.0)
                        l = 0.0;
                    Vector3 w_l, w_r;
                    if constexpr(HoistStretch)
                    {
                        w_l = sw[m][left];
                        w_r = sw[m][right];
                    }
                    else
                    {
                        const Matrix3x3 q =
                            U * block_vectors.col(m).asDiagonal() * V.transpose();
#pragma unroll
                        for(int ip = 0; ip < 3; ++ip)
                        {
                            Float sl = 0, sr = 0;
#pragma unroll
                            for(int k = 0; k < 3; ++k)
                            {
                                sl += shape_gradients(k, left) * q(ip, k);
                                sr += shape_gradients(k, right) * q(ip, k);
                            }
                            w_l(ip) = sl;
                            w_r(ip) = sr;
                        }
                    }
                    H += (l * w_l) * w_r.transpose();
                }

                H3x3s(hessian_offset++).write(tet(left), tet(right), H);
            }
        }
    }
}  // namespace

class StableNeoHookean3D final : public FEM3DConstitution
{
  public:
    // Constitution UID by libuipc specification
    static constexpr U64   ConstitutionUID = 10;
    static constexpr SizeT StencilSize     = 4;
    static constexpr SizeT HalfHessianSize = StencilSize * (StencilSize + 1) / 2;

    using FEM3DConstitution::FEM3DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;

    cuda_tool::DeviceBuffer<Float> mus;
    cuda_tool::DeviceBuffer<Float> lambdas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    // s15: hoist the stretch-mode contractions out of the stencil-block loop
    // (UIPC_SNK1_HOIST_STRETCH=0 restores the in-loop form). Bit-identical.
    bool m_hoist_stretch = true;

    virtual void do_build(BuildInfo& info) override
    {
        const char* e   = std::getenv("UIPC_SNK1_HOIST_STRETCH");
        m_hoist_stretch = !(e && e[0] == '0');
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(mus.size());
        info.gradient_count(mus.size() * StencilSize);

        if(info.gradient_only())
            return;

        info.hessian_count(mus.size() * HalfHessianSize);
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
                auto mu     = sc.tetrahedra().find<Float>("mu");
                auto lambda = sc.tetrahedra().find<Float>("lambda");

                return zip(mu->view(), lambda->view());
            },
            [&](const ForEachInfo& I, auto mu_and_lambda)
            {
                auto&& [mu, lambda] = mu_and_lambda;

                auto vI = I.global_index();

                h_mus[vI]     = mu;
                h_lambdas[vI] = lambda;
            });

        mus.resize(N);
        mus.view().copy_from(h_mus.data());

        lambdas.resize(N);
        lambdas.view().copy_from(h_lambdas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        auto k = StableNeoHookean3D_do_compute_energy_kernel;
        int  n = (int)info.indices().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                mus.cview(),
                lambdas.cview(),
                info.energies(),
                info.indices(),
                info.xs(),
                info.Dm_invs(),
                info.rest_volumes(),
                info.dt(),
                n);
        }
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        int n = (int)info.indices().size();
        if(n == 0)
            return;

        auto launch = [&](auto k)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                mus.cview(),
                lambdas.cview(),
                info.indices(),
                info.xs(),
                info.Dm_invs(),
                info.gradients(),
                info.hessians(),
                info.rest_volumes(),
                info.dt(),
                info.gradient_only(),
                n);
        };

        if(m_hoist_stretch)
            launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<true>);
        else
            launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<false>);
    }
};

REGISTER_SIM_SYSTEM(StableNeoHookean3D);
}  // namespace uipc::backend::cuda

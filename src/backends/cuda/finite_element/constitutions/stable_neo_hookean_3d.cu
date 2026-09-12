#include <finite_element/fem_3d_constitution.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_utils.h>
#include <kernel_cout.h>
#include <cuda_tool/cuda_tool.h>
#include <Eigen/Dense>
#include <utils/matrix_assembler.h>
#include <cstdlib>
#include <type_traits>

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
    // r5-s01: number of cyclic Jacobi sweeps in the fixed-iteration SVD.
    // 4 is the smallest count that reaches full double precision: over 65 536
    // samples of every deformation regime probed, the worst case needs 4
    // (mean 3.2, median 3, p95 4) - the same shape as the iterative path's
    // trip-count histogram in probe p01, and the same reason a fixed count is
    // affordable here. 3 sweeps leaves 3.5e-6 residual; 5 buys nothing and
    // costs 153 more FP64 instructions.
    constexpr int SnkSvdSweeps = 4;

    // s27: the kernel body lives in a __device__ function so that two
    // __global__ wrappers can compile the *same PTX* under different
    // `__launch_bounds__`. Nothing else differs between them -- ptxas only
    // reallocates registers, it does not touch floating-point semantics
    // (contraction is already fixed in the PTX), so the two are bit-identical.
    template <bool HoistStretch, bool FixedSvd, bool Stencil2>
    __device__ __forceinline__ void StableNeoHookean3D_gradient_hessian_body(
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
        if constexpr(FixedSvd)
            math::qr_svd_fixed<SnkSvdSweeps>(F, S, U, V);
        else
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
        // s26: Stencil2 always uses the hoisted form (it is the only one that
        // keeps every `sw` index a compile-time constant), so
        // UIPC_SNK1_HOIST_STRETCH has no effect when UIPC_SNK1_STENCIL2=1.
        Vector3 sw[3][StencilSize];
        if constexpr(HoistStretch || Stencil2)
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

        // s26: one 3x3 stencil block, always in the *canonical* (a, b)
        // orientation, i.e. sum_m l_m * w_m_a * w_m_b^T. The old path folded
        // the (tet(a) > tet(b)) swap into the accumulation, which made every
        // `sa[..]` / `sw[..]` index data dependent -- and a thread-local array
        // with a data-dependent index cannot live in registers, so ptxas put
        // both arrays in local memory. That is where this kernel's whole
        // LDL/STL traffic came from (it is *not* register spilling: ptxas
        // reports 0 spill bytes). Assembling canonically and transposing at
        // write time makes every index a compile-time constant.
        auto stencil_block = [&](auto A, auto B) -> Matrix3x3
        {
            constexpr int a = decltype(A)::value;
            constexpr int b = decltype(B)::value;

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
                    const Float   s   = sgn ? 1.0 : -1.0;
                    const Vector3 w_a = Up2 * sa[a][p1] + s * Up1 * sa[a][p2];
                    const Vector3 w_b = Up2 * sa[b][p1] + s * Up1 * sa[b][p2];
                    H += (l * w_a) * w_b.transpose();
                }
            }
#pragma unroll
            for(int m = 0; m < 3; ++m)  // stretch modes
            {
                Float l = block_values(m) * Vdt2;
                if(l < 0.0)
                    l = 0.0;
                H += (l * sw[m][a]) * sw[m][b].transpose();
            }
            return H;
        };

        // triangular slot of block (a, b), a <= b, in the same order the old
        // sequential `hessian_offset++` produced
        auto slot = [](int a, int b) { return a * StencilSize - a * (a - 1) / 2 + (b - a); };

        // branch-free: a data-dependent `if` here would put ten divergent
        // branches in the block sequence, which the old loop did not have
        // (it only swapped two index *variables*)
        auto emit = [&](int a, int b, const Matrix3x3& Hab)
        {
            const IndexT off = I * HalfHessianSize + slot(a, b);
            const IndexT ra = tet(a), rb = tet(b);
            const bool   tp = ra > rb;
            Matrix3x3    Hw;
#pragma unroll
            for(int p = 0; p < 3; ++p)
#pragma unroll
                for(int q = 0; q < 3; ++q)
                    Hw(p, q) = tp ? Hab(q, p) : Hab(p, q);
            H3x3s(off).write(tp ? rb : ra, tp ? ra : rb, Hw);
        };

        if constexpr(Stencil2)
        {
            // The four blocks that touch node 0 are not assembled at all.
            // `tetrahedron_shape_gradients` builds g_0 = -(g_1 + g_2 + g_3)
            // and every w is linear in g, so sum_k w_m_k = 0 and therefore
            // H_0b = -(H_1b + H_2b + H_3b) exactly. Accumulating the three
            // already-computed blocks of a column is 9 adds where the direct
            // assembly of that block is ~190 FP64 ops. Rounding-level, not
            // bit-identical: the sum is re-associated.
            using I1 = std::integral_constant<int, 1>;
            using I2 = std::integral_constant<int, 2>;
            using I3 = std::integral_constant<int, 3>;

            const Matrix3x3 H11 = stencil_block(I1{}, I1{});
            const Matrix3x3 H12 = stencil_block(I1{}, I2{});
            const Matrix3x3 H13 = stencil_block(I1{}, I3{});
            const Matrix3x3 H22 = stencil_block(I2{}, I2{});
            const Matrix3x3 H23 = stencil_block(I2{}, I3{});
            const Matrix3x3 H33 = stencil_block(I3{}, I3{});

            const Matrix3x3 H01 = -(H11 + H12.transpose() + H13.transpose());
            const Matrix3x3 H02 = -(H12 + H22 + H23.transpose());
            const Matrix3x3 H03 = -(H13 + H23 + H33);

            // H_00 = -(H_10 + H_20 + H_30) = -(H_01^T + H_02^T + H_03^T).
            // Symmetrised explicitly: the direct form is exactly symmetric
            // (l*w(p)*w(q) == l*w(q)*w(p) term by term) and the derived one is
            // only symmetric up to rounding; the linear system assumes a
            // symmetric diagonal block.
            const Matrix3x3 M   = -(H01.transpose() + H02.transpose() + H03.transpose());
            const Matrix3x3 H00 = 0.5 * (M + M.transpose());

            emit(0, 0, H00);
            emit(0, 1, H01);
            emit(0, 2, H02);
            emit(0, 3, H03);
            emit(1, 1, H11);
            emit(1, 2, H12);
            emit(1, 3, H13);
            emit(2, 2, H22);
            emit(2, 3, H23);
            emit(3, 3, H33);
        }
        else
        {
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
                            const Vector3 w_l =
                                Up2 * sa[left][p1] + s * Up1 * sa[left][p2];
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
    }

#define UIPC_SNK1_GH_ARGS                                                      \
    cuda_tool::CBufferView<Float> mus, cuda_tool::CBufferView<Float> lambdas,  \
        cuda_tool::CBufferView<Vector4i> indices,                              \
        cuda_tool::CBufferView<Vector3> xs,                                    \
        cuda_tool::CBufferView<Matrix3x3> Dm_invs,                             \
        cuda_tool::DoubletVectorView<Float, 3> G3s,                            \
        cuda_tool::TripletMatrixView<Float, 3> H3x3s,                          \
        cuda_tool::CBufferView<Float> volumes, Float dt, bool gradient_only, int n

#define UIPC_SNK1_GH_CALL                                                      \
    mus, lambdas, indices, xs, Dm_invs, G3s, H3x3s, volumes, dt, gradient_only, n

    template <bool HoistStretch, bool FixedSvd, bool Stencil2>
    __global__ void StableNeoHookean3D_do_compute_gradient_hessian_kernel(UIPC_SNK1_GH_ARGS)
    {
        StableNeoHookean3D_gradient_hessian_body<HoistStretch, FixedSvd, Stencil2>(
            UIPC_SNK1_GH_CALL);
    }

    // s27: the same body under an occupancy bound. Without it ptxas takes 254
    // registers and the SM holds one 256-thread block = 8 warps; the kernel is
    // latency bound, not instruction bound, and paying 1.1 KB of frame and
    // ~1.6 KB of genuine spill traffic to reach 12 warps is a large net win
    // (measured: -21 % on case2, -22 % on mas-bunny). 128x4 (16 warps) is not
    // better than 128x3, so this takes the cheaper of the two.
    // UIPC_SNK1_OCC=0 restores the unbounded kernel.
    template <bool HoistStretch, bool FixedSvd, bool Stencil2>
    __global__ __launch_bounds__(128, 3) void StableNeoHookean3D_do_compute_gradient_hessian_kernel_occ(
        UIPC_SNK1_GH_ARGS)
    {
        StableNeoHookean3D_gradient_hessian_body<HoistStretch, FixedSvd, Stencil2>(
            UIPC_SNK1_GH_CALL);
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

    // r5-s01: fixed-sweep branch-free Jacobi SVD instead of the iterative
    // Wilkinson-shift bidiagonal QR (UIPC_QR_SVD_FIXED=0 restores the old
    // path). Rounding-level change, not bit-identical.
    bool m_fixed_svd = true;

    // s26: canonical-orientation stencil assembly (no data-dependent index
    // into the thread-local `sa` / `sw`) plus the four node-0 blocks derived
    // from sum_k w_k = 0 (UIPC_SNK1_STENCIL2=0 restores the old loop).
    // Rounding-level, not bit-identical.
    bool m_stencil2 = true;

    // s27: occupancy bound on the G/H kernel (UIPC_SNK1_OCC=0 = unbounded).
    // Bit-identical: same PTX body, ptxas only reallocates registers.
    bool m_occ = true;

    virtual void do_build(BuildInfo& info) override
    {
        const char* e   = std::getenv("UIPC_SNK1_HOIST_STRETCH");
        m_hoist_stretch = !(e && e[0] == '0');

        const char* f = std::getenv("UIPC_QR_SVD_FIXED");
        m_fixed_svd   = !(f && f[0] == '0');

        const char* g = std::getenv("UIPC_SNK1_STENCIL2");
        m_stencil2    = !(g && g[0] == '0');

        const char* h = std::getenv("UIPC_SNK1_OCC");
        m_occ         = !(h && h[0] == '0');
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

        auto dispatch = [&]<bool HS, bool FS>(std::integral_constant<bool, HS>,
                                              std::integral_constant<bool, FS>)
        {
            if(m_occ)
            {
                if(m_stencil2)
                    launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel_occ<HS, FS, true>);
                else
                    launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel_occ<HS, FS, false>);
            }
            else
            {
                if(m_stencil2)
                    launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<HS, FS, true>);
                else
                    launch(StableNeoHookean3D_do_compute_gradient_hessian_kernel<HS, FS, false>);
            }
        };

        constexpr std::true_type  T{};
        constexpr std::false_type F{};

        if(m_hoist_stretch)
        {
            if(m_fixed_svd)
                dispatch(T, T);
            else
                dispatch(T, F);
        }
        else
        {
            if(m_fixed_svd)
                dispatch(F, T);
            else
                dispatch(F, F);
        }
    }
};

REGISTER_SIM_SYSTEM(StableNeoHookean3D);
}  // namespace uipc::backend::cuda

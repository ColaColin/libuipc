#pragma once
#include <type_define.h>
#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <utils/make_spd.h>

namespace uipc::backend::cuda
{
namespace sym::codim_ipc_simplex_contact
{
    inline __device__ Float PT_kappa(const cuda_tool::CDense2D<ContactCoeff>& table,
                                     const Vector4i& cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 4; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / 3.0;
    }

    inline __device__ Float EE_kappa(const cuda_tool::CDense2D<ContactCoeff>& table,
                                     const Vector4i& cids)
    {
        Float kappa = 0.0;
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 2; k < 4; ++k)
            {
                ContactCoeff coeff = table(cids[j], cids[k]);
                kappa += coeff.kappa;
            }
        }
        return kappa / 4.0;
    }

    inline __device__ Float PE_kappa(const cuda_tool::CDense2D<ContactCoeff>& table,
                                     const Vector3i& cids)
    {
        Float kappa = 0.0;
        for(int j = 1; j < 3; ++j)
        {
            ContactCoeff coeff = table(cids[0], cids[j]);
            kappa += coeff.kappa;
        }
        return kappa / 2.0;
    }

    inline __device__ Float PP_kappa(const cuda_tool::CDense2D<ContactCoeff>& table,
                                     const Vector2i& cids)
    {
        ContactCoeff coeff = table(cids[0], cids[1]);
        return coeff.kappa;
    }


    // round-6 (s03): the IPC barrier's PSD-projected Hessian in closed form.
    //
    // Every un-mollified barrier here is a function of one scalar, the squared
    // distance D, so its exact Hessian is
    //
    //   H = B''(D) gradD gradD^T + B'(D) hessD,                            (*)
    //
    // which is literally what `*_barrier_gradient_hessian` assembles. Two
    // facts about (*) were measured over 200 000 randomised pairs of each type
    // against the real device functions (`gn_contact_probe.cu`):
    //
    //  * The shipped Stiff-GIPC log^2 barrier has **B'' >= 0 and B' <= 0**
    //    everywhere on its active domain (and beyond it). B'' is a sum of four
    //    terms that are each non-negative there, because L = log(D/dHat^2) < 0
    //    and D - dHat^2 < 0 make every term a product of an even power or of
    //    two negatives; the thickness variant has the same structure. So the
    //    first term of (*) is PSD and all of the indefiniteness is in the
    //    second.
    //  * The *whole* PSD projection of (*) -- the K10 / s16 / s25 / s31 chain
    //    of reduced eigen-solves -- is, to a mean relative Frobenius error of
    //    1e-4 or better, the **rank-1** matrix
    //
    //      H_spd ~= c gradD gradD^T,     c = B''(D) + B'(D) / (2 D).        (**)
    //
    //    The reason is that D is a squared distance, so along its own gradient
    //    direction `ghat = gradD/|gradD|` one has `ghat^T hessD ghat =
    //    |gradD|^2 / (2 D)` exactly for PP and to leading order for PE/PT/EE;
    //    (**) therefore reproduces the projected Hessian's leading eigenvalue
    //    *exactly*, and the projection's remaining eigenvalues are ~1e-4 of it.
    //    Measured: mean lambda_max(H_rank1)/lambda_max(H_spd) = 1.0000 for all
    //    four pair types, and the rank the exact projection keeps is 3 of 12
    //    (PT, EE), 2 of 9 (PE) and 1 of 6 (PP).
    //
    // `c` is non-negative wherever the barrier is active: the probe measures
    // |B'| <= 0.9651 B'' D, so 2 B'' D + B' >= 1.03 B'' D > 0. It is clamped
    // anyway -- one instruction, and it keeps the PSD guarantee if a future
    // step substitutes a different barrier.
    //
    // Dropping `B'/(2D)` gives the *plain* Gauss-Newton Hessian `c = B''`,
    // which is also PSD but over-stiffens the contact by a measured factor of
    // 1.74; round 6 measured that variant (`Proj = 3`) and rejected it -- see
    // the round record. This is an approximation, not a rewrite: the energy
    // and the gradient are untouched, the Hessian is not.
    template <int N>
    inline __device__ void barrier_rank1_hessian(Matrix<Float, N, N>&    H,
                                                 const Vector<Float, N>& GradD,
                                                 Float                   c)
    {
        if(c < 0.0)
            c = 0.0;
        // explicit mirrored fill: the result is *exactly* symmetric, which the
        // half-block assembler and the downstream solvers both rely on
#pragma unroll
        for(int i = 0; i < N; ++i)
        {
            const Float ci = c * GradD[i];
#pragma unroll
            for(int j = i; j < N; ++j)
            {
                const Float v = ci * GradD[j];
                H(i, j)       = v;
                H(j, i)       = v;
            }
        }
    }

    //tex: $$ c = B''(D) + \frac{B'(D)}{2D} \quad (Rank1 = 1) \qquad c = B''(D) \quad (Rank1 = 3, \text{plain Gauss-Newton})$$
    template <int Rank1>
    inline __device__ Float barrier_rank1_coeff(Float ddBddD, Float dBdD, Float D)
    {
        if constexpr(Rank1 == 3)
            return ddBddD;
        else
            return ddBddD + dBdD / (2.0 * D);
    }

    inline __device__ Float PT_barrier_energy(Float          kappa,
                                              Float          d_hat,
                                              Float          thickness,
                                              const Vector3& P,
                                              const Vector3& T0,
                                              const Vector3& T1,
                                              const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline __device__ Float PT_barrier_energy(const Vector4i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  T0,
                                              const Vector3&  T1,
                                              const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);
        return B;
    }

    inline __device__ void PT_barrier_gradient_hessian(Vector12&      G,
                                                       Matrix12x12&   H,
                                                       Float          kappa,
                                                       Float          d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(P, T0, T1, T2, HessD);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    inline __device__ void PT_barrier_gradient_hessian(Vector12&       G,
                                                       Matrix12x12&    H,
                                                       const Vector4i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& T0,
                                                       const Vector3& T1,
                                                       const Vector3& T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Matrix12x12 HessD;
        point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    // round-6 (s03): rank-1 PT barrier Hessian -- same gradient, no
    // `point_triangle_distance2_hessian`, no PSD projection.
    template <int Rank1 = 1>
    inline __device__ void PT_barrier_gradient_hessian_gn(Vector12&       G,
                                                          Matrix12x12&    H,
                                                          const Vector4i& flag,
                                                          Float           kappa,
                                                          Float           d_hat,
                                                          Float           thickness,
                                                          const Vector3&  P,
                                                          const Vector3&  T0,
                                                          const Vector3&  T1,
                                                          const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        barrier_rank1_hessian<12>(H, GradD, barrier_rank1_coeff<Rank1>(ddBddD, dBdD, D));
    }

    inline __device__ void PT_barrier_gradient(Vector12&       G,
                                               const Vector4i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P,
                                               const Vector3&  T0,
                                               const Vector3&  T1,
                                               const Vector3&  T2)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        point_triangle_distance2(flag, P, T0, T1, T2, D);

        Vector12 GradD;
        point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }


    inline __device__ Float mollified_EE_barrier_energy(const Vector4i& flag,
                                                        Float           kappa,
                                                        Float           d_hat,
                                                        Float thickness,
                                                        const Vector3& t0_Ea0,
                                                        const Vector3& t0_Ea1,
                                                        const Vector3& t0_Eb0,
                                                        const Vector3& t0_Eb1,
                                                        const Vector3& Ea0,
                                                        const Vector3& Ea1,
                                                        const Vector3& Eb0,
                                                        const Vector3& Eb1)
    {
        // using mollifier to improve the smoothness of the edge-edge barrier
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);
        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        return ek * B;
    }

    // round-4 (s16): with `Reduced`, the (dominant) branch in which the
    // edge-edge mollifier is inactive -- the two edges are far from parallel,
    // |ea x eb|^2 >= eps_x -- is taken separately. There ek == 1,
    // grad(ek) == 0 and hess(ek) == 0 exactly, so
    // `G = grad(ek) B + ek grad(B)` and
    // `H = hess(ek) B + grad(ek) grad(B)^T + grad(B) grad(ek)^T + ek hess(B)`
    // collapse to `G = grad(B)`, `H = hess(B)`: the Hessian is then a *plain*
    // barrier Hessian of a flagged simplex distance, exactly like PT/PE/PP, so
    // `EE_barrier_make_spd` (the rank-(m+1) reduced projection of 1da12a82)
    // applies to it and replaces the 9x9 eigen-solve of K10 by a 5x5 / 4x4 /
    // 3x3 one. `mollified` reports which branch was taken so the caller can
    // pick the projection. Identical arithmetic on the taken values (only a
    // -0.0 can turn into +0.0 where the old expression added 0.0 * B).
    template <bool Reduced = false>
    inline __device__ void mollified_EE_barrier_gradient_hessian(Vector12&    G,
                                                                 Matrix12x12& H,
                                                                 bool&        mollified,
                                                                 const Vector4i& flag,
                                                                 Float kappa,
                                                                 Float d_hat,
                                                                 Float thickness,
                                                                 const Vector3& t0_Ea0,
                                                                 const Vector3& t0_Ea1,
                                                                 const Vector3& t0_Eb0,
                                                                 const Vector3& t0_Eb1,
                                                                 const Vector3& Ea0,
                                                                 const Vector3& Ea1,
                                                                 const Vector3& Eb0,
                                                                 const Vector3& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        //tex: $$ \nabla D$$
        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        //tex: $$ \nabla^2 D$$
        Matrix12x12 HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        //tex: $$ \frac{\partial B}{\partial D} $$
        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial^2 B}{\partial D^2} $$
        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex: $$ \epsilon_x $$
        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        mollified = true;
        if constexpr(Reduced)
        {
            Float cross_norm2;
            edge_edge_cross_norm2(Ea0, Ea1, Eb0, Eb1, cross_norm2);
            if(!(cross_norm2 < eps_x))  // mollifier inactive: ek == 1, its derivatives == 0
            {
                mollified = false;
                G         = dBdD * GradD;
                H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
                return;
            }
        }

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        //tex: $$ \nabla B = \frac{\partial B}{\partial D} \nabla D$$
        Vector12 GradB = dBdD * GradD;

        //tex:
        //$$
        // \nabla^2 B = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T + \frac{\partial B}{\partial D} \nabla^2 D
        //$$
        Matrix12x12 HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        //tex: $$ e_k $$
        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        //tex: $$\nabla e_k$$
        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);


        //tex: $$ \nabla^2 e_k$$
        Matrix12x12 Hessek;
        edge_edge_mollifier_hessian(Ea0, Ea1, Eb0, Eb1, eps_x, Hessek);

        //tex:
        //$$
        // G = \nabla e_k B + e_k \nabla B
        //$$
        G = Gradek * B + ek * GradB;

        //tex: $$ \nabla^2 e_k B + \nabla e_k \nabla B^T + \nabla B \nabla e_k^T + e_k \nabla^2 B$$
        H = Hessek * B + Gradek * GradB.transpose() + GradB * Gradek.transpose() + ek * HessB;
    }

    // round-6 (s03): Gauss-Newton EE barrier Hessian for the branch in which
    // the edge-edge mollifier is inactive -- there `ek == 1` and both its
    // derivatives vanish, so the Hessian is a plain barrier Hessian of a
    // flagged distance and the rank-1 form applies verbatim. The *mollified*
    // branch is a function of two scalars (the mollifier argument and D), so
    // this decomposition does not hold for it; it falls through to the exact
    // Hessian and the K10 9x9 projection, unchanged. `mollified` reports which
    // branch was taken, exactly as in the exact version.
    // Unlike the exact function, `edge_edge_distance2_hessian` is only
    // evaluated on the mollified branch -- the un-mollified branch never needs
    // it at all.
    template <int Rank1 = 1>
    inline __device__ void mollified_EE_barrier_gradient_hessian_gn(Vector12&    G,
                                                                    Matrix12x12& H,
                                                                    bool& mollified,
                                                                    const Vector4i& flag,
                                                                    Float kappa,
                                                                    Float d_hat,
                                                                    Float thickness,
                                                                    const Vector3& t0_Ea0,
                                                                    const Vector3& t0_Ea1,
                                                                    const Vector3& t0_Eb0,
                                                                    const Vector3& t0_Eb1,
                                                                    const Vector3& Ea0,
                                                                    const Vector3& Ea1,
                                                                    const Vector3& Eb0,
                                                                    const Vector3& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float cross_norm2;
        edge_edge_cross_norm2(Ea0, Ea1, Eb0, Eb1, cross_norm2);

        if(!(cross_norm2 < eps_x))  // mollifier inactive: ek == 1, derivatives == 0
        {
            mollified = false;
            G         = dBdD * GradD;
            barrier_rank1_hessian<12>(H, GradD, barrier_rank1_coeff<Rank1>(ddBddD, dBdD, D));
            return;
        }

        mollified = true;

        Matrix12x12 HessD;
        edge_edge_distance2_hessian(flag, Ea0, Ea1, Eb0, Eb1, HessD);

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Vector12    GradB = dBdD * GradD;
        Matrix12x12 HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        Matrix12x12 Hessek;
        edge_edge_mollifier_hessian(Ea0, Ea1, Eb0, Eb1, eps_x, Hessek);

        G = Gradek * B + ek * GradB;
        H = Hessek * B + Gradek * GradB.transpose() + GradB * Gradek.transpose() + ek * HessB;
    }

    inline __device__ void mollified_EE_barrier_gradient(Vector12&       G,
                                                         const Vector4i& flag,
                                                         Float           kappa,
                                                         Float           d_hat,
                                                         Float thickness,
                                                         const Vector3& t0_Ea0,
                                                         const Vector3& t0_Ea1,
                                                         const Vector3& t0_Eb0,
                                                         const Vector3& t0_Eb1,
                                                         const Vector3& Ea0,
                                                         const Vector3& Ea1,
                                                         const Vector3& Eb0,
                                                         const Vector3& Eb1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D;
        edge_edge_distance2(flag, Ea0, Ea1, Eb0, Eb1, D);

        Vector12 GradD;
        edge_edge_distance2_gradient(flag, Ea0, Ea1, Eb0, Eb1, GradD);

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        Vector12 GradB = dBdD * GradD;

        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

        Float ek;
        edge_edge_mollifier(Ea0, Ea1, Eb0, Eb1, eps_x, ek);

        Vector12 Gradek;
        edge_edge_mollifier_gradient(Ea0, Ea1, Eb0, Eb1, eps_x, Gradek);

        G = Gradek * B + ek * GradB;
    }

    inline __device__ Float PE_barrier_energy(const Vector3i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P,
                                              const Vector3&  E0,
                                              const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);
        Float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline __device__ void PE_barrier_gradient_hessian(Vector9&        G,
                                                       Matrix9x9&      H,
                                                       const Vector3i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P,
                                                       const Vector3& E0,
                                                       const Vector3& E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Matrix9x9 HessD;
        point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    // round-6 (s03): rank-1 PE barrier Hessian.
    template <int Rank1 = 1>
    inline __device__ void PE_barrier_gradient_hessian_gn(Vector9&        G,
                                                          Matrix9x9&      H,
                                                          const Vector3i& flag,
                                                          Float           kappa,
                                                          Float           d_hat,
                                                          Float           thickness,
                                                          const Vector3&  P,
                                                          const Vector3&  E0,
                                                          const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        barrier_rank1_hessian<9>(H, GradD, barrier_rank1_coeff<Rank1>(ddBddD, dBdD, D));
    }

    inline __device__ void PE_barrier_gradient(Vector9&        G,
                                               const Vector3i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P,
                                               const Vector3&  E0,
                                               const Vector3&  E1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_edge_distance2(flag, P, E0, E1, D);

        Vector9 GradD;
        point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }

    inline __device__ Float PP_barrier_energy(const Vector2i& flag,
                                              Float           kappa,
                                              Float           d_hat,
                                              Float           thickness,
                                              const Vector3&  P0,
                                              const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;
        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);
        Float E = 0.0;
        KappaBarrier(E, kappa, D, d_hat, thickness);
        return E;
    }

    inline __device__ void PP_barrier_gradient_hessian(Vector6&        G,
                                                       Matrix6x6&      H,
                                                       const Vector2i& flag,
                                                       Float           kappa,
                                                       Float           d_hat,
                                                       Float          thickness,
                                                       const Vector3& P0,
                                                       const Vector3& P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Matrix6x6 HessD;
        point_point_distance2_hessian(flag, P0, P1, HessD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // G = \frac{\partial B}{\partial D} \frac{\partial D}{\partial x}
        //$$
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex:
        //$$
        // H = \frac{\partial^2 B}{\partial D^2} \frac{\partial D}{\partial x} \frac{\partial D}{\partial x}^T + \frac{\partial B}{\partial D} \frac{\partial^2 D}{\partial x^2}
        //$$
        H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
    }

    // round-6 (s03): rank-1 PP barrier Hessian.
    template <int Rank1 = 1>
    inline __device__ void PP_barrier_gradient_hessian_gn(Vector6&        G,
                                                          Matrix6x6&      H,
                                                          const Vector2i& flag,
                                                          Float           kappa,
                                                          Float           d_hat,
                                                          Float           thickness,
                                                          const Vector3&  P0,
                                                          const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
        G = dBdD * GradD;

        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
        barrier_rank1_hessian<6>(H, GradD, barrier_rank1_coeff<Rank1>(ddBddD, dBdD, D));
    }

    inline __device__ void PP_barrier_gradient(Vector6&        G,
                                               const Vector2i& flag,
                                               Float           kappa,
                                               Float           d_hat,
                                               Float           thickness,
                                               const Vector3&  P0,
                                               const Vector3&  P1)
    {
        using namespace codim_ipc_contact;
        using namespace distance;

        Float D = 0.0;
        point_point_distance2(flag, P0, P1, D);

        Vector6 GradD;
        point_point_distance2_gradient(flag, P0, P1, GradD);

        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        G = dBdD * GradD;
    }


    //tex:
    //$$
    // \text{basis of the range of the barrier Hessian } H = B''(D) \nabla D \nabla D^T + B'(D) \nabla^2 D
    //$$
    // for M active vertices: the separation modes $\hat s \otimes t_1$,
    // $\hat s \otimes t_2$ plus the mean-free normal modes $u_j \otimes \hat n$,
    // where $s$ are the signed closest-point weights ($+1$ on the point,
    // $-\alpha$ on the other simplex, $\sum s = 0$), $\hat n$ is the unit gap
    // direction and $(t_1, t_2)$ any orthonormal tangent frame.
    template <int M>
    inline __device__ void barrier_range_basis(Matrix<Float, 3 * M, M + 1>& Q,
                                               const Vector<Float, M>&      s,
                                               const Vector3&               gap)
    {
        static_assert(M >= 2 && M <= 4, "active vertex count out of range");

        Vector3 nh = gap.normalized();

        //tex: $$ \text{any orthonormal tangent frame } (t_1, t_2) \perp \hat n$$
        Vector3 a = Vector3(1.0, 0.0, 0.0);
        if(std::abs(nh[0]) >= Float(0.9))
            a = Vector3(0.0, 1.0, 0.0);
        Vector3 t1 = (a - nh.dot(a) * nh).normalized();
        Vector3 t2 = nh.cross(t1);

        Vector<Float, M> sh = s.normalized();

        //tex: $$ \text{orthonormal basis } u_j \text{ of the mean-free weights } (\sum_j u_j = 0)$$
        Matrix<Float, M, M - 1> U;
        for(int j = 0; j < M - 1; ++j)
        {
            Vector<Float, M> u;
            u.setZero();
            u[j]     = 1.0;
            u[M - 1] = -1.0;
            for(int p = 0; p < j; ++p)
                u -= U.col(p) * u.dot(U.col(p));
            U.col(j) = u.normalized();
        }

        for(int k = 0; k < M; ++k)
        {
            Q.template block<3, 1>(3 * k, 0) = sh[k] * t1;
            Q.template block<3, 1>(3 * k, 1) = sh[k] * t2;
            for(int j = 0; j < M - 1; ++j)
                Q.template block<3, 1>(3 * k, 2 + j) = U(k, j) * nh;
        }
    }

    //tex:
    //$$
    // make\_spd(H) = Q \; make\_spd(Q^T H Q) \; Q^T \text{, exactly if } range(H) \subseteq range(Q)
    //$$
    // Reduce the SPD projection of H (N x N, N = 3 * n vertices, only the M
    // active vertex blocks at slots `act` are non-zero) to the (M+1)x(M+1)
    // eigenproblem carried by the basis Q of `barrier_range_basis`.
    // round-5 (s25): `Solver` selects the eigen-solve behind `make_spd<M+1>`
    // exactly as s19's `make_spd<N, Solver>` does -- 0 = Eigen's
    // SelfAdjointEigenSolver (every round up to 4), 1 = the fixed-size
    // Householder + implicit-QL of `cuda_tool::eigen::evd_tridiag_ql`. It is a
    // template parameter so each instantiation carries one code path's stack
    // frame (the s14 lesson); M + 1 <= 3 takes Eigen's closed form either way.
    // round-5 (s31): `Basis` selects how that same subspace is carried.
    // 0 = the explicit 3M x (M+1) matrix Q of `barrier_range_basis` and its
    // Q^T Hs Q / Q Hred Q^T products (every round up to 5). 1 = the
    // basis-free form below, for M <= 3.
    //
    // Q's columns are (s^ (x) t1, s^ (x) t2, u_1 (x) n^, ..., u_{M-1} (x) n^).
    // The u_j span the mean-free weights of R^M and s^ is itself mean-free, so
    // span{u_j} = span{s^, v_1, ..., v_{M-2}} with v_i perp s^, and
    //
    //   range(Q) = s^ (x) span{t1, t2, n^}  (+)  v_1 (x) n^  (+) ...
    //            = s^ (x) R^3               (+)  v_1 (x) n^  (+) ...
    //
    // because (t1, t2, n^) is an orthonormal basis of R^3. The tangent frame
    // therefore drops out of the *subspace* entirely, and so does Q: for any
    // two orthonormal bases Q' = Q R of the same subspace (R orthogonal),
    // Q' make_spd(Q'^T H Q') Q'^T = Q make_spd(Q^T H Q) Q^T, since make_spd
    // clamps eigenvalues and therefore commutes with orthogonal conjugation.
    // Same projection up to rounding, with no Q, no Hs and no Hspd temporary,
    // and no tangent frame / Gram-Schmidt.
    //   M = 2: range(Q) = s^ (x) R^3, a plain 3x3 eigen-solve on
    //          Hss = sum_ab s^_a s^_b H_ab, mapped back as H_ab = s^_a s^_b Hss.
    //   M = 3: one extra direction v = ((1,1,1) x s^)/|.| (the mean-free unit
    //          orthogonal to s^), giving the bordered 4x4
    //          [[Hss, m], [m^T, c]] with m = (sum_ab s^_a v_b H_ab) n^ and
    //          c = n^^T (sum_ab v_a v_b H_ab) n^.
    //   M = 4: not implemented; falls through to the dense-Q path. In the
    //          shipped dispatch that arm is never instantiated either: M = 4
    //          is reached only from `PT_/EE_barrier_make_spd`, and both call
    //          sites pass only `Solver`, i.e. Basis = 0. The `Basis` parameter
    //          of those two wrappers is therefore inert today.
    //   round-6 (s08): M = 4 IS implemented now -- see below. Contact part 1
    //          (PT, and EE when the mollifier is inactive) is M = 4 in
    //          **100 %** of the pairs the five benchmark scenes produce
    //          (`flagstats_*.txt`: 0 of 3.9M PT and 0 of 13.0M un-mollified EE
    //          pairs on the tumbler take dim 2 or 3), so leaving M = 4 on the
    //          dense-Q path left s31's reduction with no caller at all in the
    //          suite's largest kernel. The construction is the same one the
    //          M = 3 comment describes, with **two** extra directions instead
    //          of one: the mean-free subspace of R^4 is 3-dimensional, s^ is
    //          in it, so range(Q) = s^ (x) R^3 (+) v_1 (x) n^ (+) v_2 (x) n^
    //          with (v_1, v_2) any orthonormal basis of (mean-free) ^ s^.
    //          The 5x5 is bordered by two columns:
    //          [[Hss, m_1, m_2], [m_1^T, C_11, C_12], [m_2^T, C_12, C_22]].
    template <int N, int M, int Solver = 0, int Basis = 0>
    inline __device__ void make_spd_contact(Matrix<Float, N, N>&     H,
                                            const Vector<IndexT, M>& act,
                                            const Vector<Float, M>&  s,
                                            const Vector3&           gap)
    {
        constexpr int NM = 3 * M;

        if constexpr(Basis == 1 && M == 4)
        {
            const Vector3          nh = gap.normalized();
            const Vector<Float, 4> sh = s.normalized();

            //tex:
            //$$
            // \text{Helmert basis of the mean-free subspace of } \mathbb{R}^4,
            // \quad h_1, h_2, h_3 \text{ orthonormal}, \; \sum_k (h_i)_k = 0
            //$$
            // s^ is mean-free (sum s = 0 for both the PT and the EE dim-4
            // weights), so s^ = sum_i alpha_i h_i with |alpha| = 1, and the two
            // directions we need are any orthonormal pair orthogonal to alpha
            // in that 3-dimensional coefficient space -- the same
            // pick-an-axis-and-cross trick `barrier_range_basis` uses for its
            // tangent frame, but on 3 coefficients instead of on R^3, so no
            // Gram-Schmidt over 4-vectors and no tangent frame at all.
            constexpr Float r2 = 0.70710678118654752440;  // 1/sqrt(2)
            constexpr Float r6 = 0.40824829046386301637;  // 1/sqrt(6)
            constexpr Float r12 = 0.28867513459481288225;  // 1/sqrt(12)

            const Vector3 alpha{r2 * (sh[0] - sh[1]),
                                r6 * (sh[0] + sh[1] - 2.0 * sh[2]),
                                r12 * (sh[0] + sh[1] + sh[2] - 3.0 * sh[3])};

            Vector3 ax = Vector3(1.0, 0.0, 0.0);
            if(std::abs(alpha[0]) >= Float(0.9))
                ax = Vector3(0.0, 1.0, 0.0);
            const Vector3 w1 = (ax - alpha.dot(ax) * alpha).normalized();
            const Vector3 w2 = alpha.cross(w1);

            //tex: $$ v_i = \sum_j (w_i)_j h_j $$
            const Vector<Float, 4> v1{r2 * w1[0] + r6 * w1[1] + r12 * w1[2],
                                      -r2 * w1[0] + r6 * w1[1] + r12 * w1[2],
                                      -2.0 * r6 * w1[1] + r12 * w1[2],
                                      -3.0 * r12 * w1[2]};
            const Vector<Float, 4> v2{r2 * w2[0] + r6 * w2[1] + r12 * w2[2],
                                      -r2 * w2[0] + r6 * w2[1] + r12 * w2[2],
                                      -2.0 * r6 * w2[1] + r12 * w2[2],
                                      -3.0 * r12 * w2[2]};

            Matrix3x3 Hss = Matrix3x3::Zero();
            Vector3   m1  = Vector3::Zero();
            Vector3   m2  = Vector3::Zero();
            Float     c11 = 0.0, c12 = 0.0, c22 = 0.0;
#pragma unroll
            for(int b = 0; b < 4; ++b)
            {
                Matrix3x3 Tb = Matrix3x3::Zero();
                Vector3   tb = Vector3::Zero();
                Vector3   q1 = Vector3::Zero();
                Vector3   q2 = Vector3::Zero();
#pragma unroll
                for(int a = 0; a < 4; ++a)
                {
                    auto          Aab = H.template block<3, 3>(3 * act[a], 3 * act[b]);
                    const Vector3 w   = Aab * nh;
                    Tb += sh[a] * Aab;
                    tb += sh[a] * w;
                    q1 += v1[a] * w;
                    q2 += v2[a] * w;
                }
                Hss += sh[b] * Tb;
                m1 += v1[b] * tb;
                m2 += v2[b] * tb;
                c11 += v1[b] * nh.dot(q1);
                c12 += v2[b] * nh.dot(q1);
                c22 += v2[b] * nh.dot(q2);
            }

            Matrix<Float, 5, 5> Hred;
            Hred.template block<3, 3>(0, 0) = Hss;
            Hred.template block<3, 1>(0, 3) = m1;
            Hred.template block<3, 1>(0, 4) = m2;
            Hred.template block<1, 3>(3, 0) = m1.transpose();
            Hred.template block<1, 3>(4, 0) = m2.transpose();
            Hred(3, 3)                      = c11;
            Hred(3, 4)                      = c12;
            Hred(4, 3)                      = c12;
            Hred(4, 4)                      = c22;

            make_spd<5, Solver>(Hred);

            const Matrix3x3 A   = Hred.template block<3, 3>(0, 0);
            const Vector3   m1r = Hred.template block<3, 1>(0, 3);
            const Vector3   m2r = Hred.template block<3, 1>(0, 4);
            const Matrix3x3 B1  = m1r * nh.transpose();
            const Matrix3x3 B2  = m2r * nh.transpose();
            const Matrix3x3 NN  = nh * nh.transpose();
            const Float     d11 = Hred(3, 3);
            const Float     d12 = Hred(3, 4);
            const Float     d22 = Hred(4, 4);
#pragma unroll
            for(int b = 0; b < 4; ++b)
            {
                const Matrix3x3 Pb = sh[b] * A + v1[b] * B1 + v2[b] * B2;
                const Matrix3x3 R1 = sh[b] * B1.transpose()
                                     + (d11 * v1[b] + d12 * v2[b]) * NN;
                const Matrix3x3 R2 = sh[b] * B2.transpose()
                                     + (d12 * v1[b] + d22 * v2[b]) * NN;
#pragma unroll
                for(int a = 0; a < 4; ++a)
                    H.template block<3, 3>(3 * act[a], 3 * act[b]) =
                        sh[a] * Pb + v1[a] * R1 + v2[a] * R2;
            }
            return;
        }

        if constexpr(Basis == 1 && M <= 3)
        {
            const Vector<Float, M> sh = s.normalized();

            if constexpr(M == 2)
            {
                Matrix3x3 Hss = Matrix3x3::Zero();
#pragma unroll
                for(int a = 0; a < M; ++a)
#pragma unroll
                    for(int b = 0; b < M; ++b)
                        Hss += (sh[a] * sh[b])
                               * H.template block<3, 3>(3 * act[a], 3 * act[b]);

                make_spd<3, Solver>(Hss);

#pragma unroll
                for(int a = 0; a < M; ++a)
#pragma unroll
                    for(int b = 0; b < M; ++b)
                        H.template block<3, 3>(3 * act[a], 3 * act[b]) =
                            (sh[a] * sh[b]) * Hss;
            }
            else  // M == 3
            {
                const Vector3 nh = gap.normalized();

                //tex: $$ v = \frac{(1,1,1) \times \hat s}{\|\cdot\|} \perp \hat s, \; \sum_j v_j = 0$$
                Vector3 v{sh[2] - sh[1], sh[0] - sh[2], sh[1] - sh[0]};
                v.normalize();

                Matrix3x3 Hss = Matrix3x3::Zero();
                Vector3   m   = Vector3::Zero();
                Float     c   = 0.0;
#pragma unroll
                for(int b = 0; b < 3; ++b)
                {
                    Matrix3x3 Tb = Matrix3x3::Zero();
                    Vector3   tb = Vector3::Zero();
                    Vector3   qb = Vector3::Zero();
#pragma unroll
                    for(int a = 0; a < 3; ++a)
                    {
                        auto          Aab = H.template block<3, 3>(3 * act[a], 3 * act[b]);
                        const Vector3 w   = Aab * nh;
                        Tb += sh[a] * Aab;
                        tb += sh[a] * w;
                        qb += v[a] * w;
                    }
                    Hss += sh[b] * Tb;
                    m += v[b] * tb;
                    c += v[b] * nh.dot(qb);
                }

                Matrix<Float, 4, 4> Hred;
                Hred.template block<3, 3>(0, 0) = Hss;
                Hred.template block<3, 1>(0, 3) = m;
                Hred.template block<1, 3>(3, 0) = m.transpose();
                Hred(3, 3)                      = c;

                make_spd<4, Solver>(Hred);

                const Matrix3x3 A  = Hred.template block<3, 3>(0, 0);
                const Vector3   mr = Hred.template block<3, 1>(0, 3);
                const Matrix3x3 B  = mr * nh.transpose();
                const Matrix3x3 C  = (Hred(3, 3) * nh) * nh.transpose();
#pragma unroll
                for(int b = 0; b < 3; ++b)
                {
                    const Matrix3x3 Pb = sh[b] * A + v[b] * B;
                    const Matrix3x3 Rb = sh[b] * B.transpose() + v[b] * C;
#pragma unroll
                    for(int a = 0; a < 3; ++a)
                        H.template block<3, 3>(3 * act[a], 3 * act[b]) =
                            sh[a] * Pb + v[a] * Rb;
                }
            }
            return;
        }

        Matrix<Float, NM, NM> Hs;
        for(int a = 0; a < M; ++a)
            for(int b = 0; b < M; ++b)
                Hs.template block<3, 3>(3 * a, 3 * b) =
                    H.template block<3, 3>(3 * act[a], 3 * act[b]);

        Matrix<Float, NM, M + 1> Q;
        barrier_range_basis<M>(Q, s, gap);

        Matrix<Float, M + 1, M + 1> Hred = Q.transpose() * Hs * Q;
        make_spd<M + 1, Solver>(Hred);

        Matrix<Float, NM, NM> Hspd = Q * Hred * Q.transpose();
        for(int a = 0; a < M; ++a)
            for(int b = 0; b < M; ++b)
                H.template block<3, 3>(3 * act[a], 3 * act[b]) =
                    Hspd.template block<3, 3>(3 * a, 3 * b);
    }

    //tex: $$ \text{reduced } make\_spd \text{ of the PT barrier Hessian (12x12)}$$
    // round-5 (s25): `Solver` is forwarded to `make_spd_contact`; the PT
    // branch lives in contact part 1, whose projection this step measures.
    // `Basis` is forwarded too but is inert here: both call sites pass only
    // `Solver`, and s31's basis-free form needs M <= 3 while PT is M = 4.
    template <int Solver = 0, int Basis = 0>
    inline __device__ void PT_barrier_make_spd(Matrix12x12&    H,
                                               const Vector4i& flag,
                                               const Vector3&  P,
                                               const Vector3&  T0,
                                               const Vector3&  T1,
                                               const Vector3&  T2)
    {
        using namespace distance;

        const Vector3 X[4] = {P, T0, T1, T2};

        IndexT dim = detail::active_count(flag);
        if(dim == 2)
        {
            Vector2i act = detail::pp_from_pt(flag);
            Vector2  s   = {1.0, -1.0};
            make_spd_contact<12, 2, Solver, Basis>(H, act, s, X[act[0]] - X[act[1]]);
        }
        else if(dim == 3)
        {
            //tex: $$ \text{closest point } E_0 + t (E_1 - E_0) \text{ on the active edge}$$
            Vector3i act = detail::pe_from_pt(flag);
            Vector3  e   = X[act[2]] - X[act[1]];
            Float    t   = (X[act[0]] - X[act[1]]).dot(e) / e.squaredNorm();
            Vector3  s   = {1.0, t - 1.0, -t};
            Vector3  gap = X[act[0]] - (X[act[1]] + t * e);
            make_spd_contact<12, 3, Solver, Basis>(H, act, s, gap);
        }
        else
        {
            //tex: $$ \text{plane projection } T_0 + u (T_1 - T_0) + v (T_2 - T_0)$$
            Vector3  e1  = T1 - T0;
            Vector3  e2  = T2 - T0;
            Vector3  w   = P - T0;
            Float    a   = e1.dot(e1);
            Float    b   = e1.dot(e2);
            Float    c   = e2.dot(e2);
            Float    u   = (c * e1.dot(w) - b * e2.dot(w)) / (a * c - b * b);
            Float    v   = (a * e2.dot(w) - b * e1.dot(w)) / (a * c - b * b);
            Vector4  s   = {1.0, u + v - 1.0, -u, -v};
            Vector4i act = {0, 1, 2, 3};
            make_spd_contact<12, 4, Solver, Basis>(H, act, s, w - u * e1 - v * e2);
        }
    }

    //tex: $$ \text{reduced } make\_spd \text{ of the PE barrier Hessian (9x9)}$$
    // round-5 (s31): `Solver` and `Basis` are forwarded to `make_spd_contact`;
    // the PE branch lives in contact part 2, whose projection this step
    // measures (dim 3 alone is 40 % of that kernel).
    template <int Solver = 0, int Basis = 0>
    inline __device__ void PE_barrier_make_spd(Matrix9x9&      H,
                                               const Vector3i& flag,
                                               const Vector3&  P,
                                               const Vector3&  E0,
                                               const Vector3&  E1)
    {
        using namespace distance;

        const Vector3 X[3] = {P, E0, E1};

        IndexT dim = detail::active_count(flag);
        if(dim == 2)
        {
            Vector2i act = detail::pp_from_pe(flag);
            Vector2  s   = {1.0, -1.0};
            make_spd_contact<9, 2, Solver, Basis>(H, act, s, X[act[0]] - X[act[1]]);
        }
        else
        {
            //tex: $$ \text{closest point } E_0 + t (E_1 - E_0) \text{ on the edge}$$
            Vector3  e   = E1 - E0;
            Float    t   = (P - E0).dot(e) / e.squaredNorm();
            Vector3  s   = {1.0, t - 1.0, -t};
            Vector3i act = {0, 1, 2};
            make_spd_contact<9, 3, Solver, Basis>(H, act, s, P - (E0 + t * e));
        }
    }

    //tex: $$ \text{reduced } make\_spd \text{ of the un-mollified EE barrier Hessian (12x12)}$$
    // round-4 (s16): only valid when the edge-edge mollifier is inactive, i.e.
    // when `mollified_EE_barrier_gradient_hessian` took its un-mollified
    // branch. H is then B''(D) grad(D) grad(D)^T + B'(D) hess(D) of the
    // flagged edge-edge distance, whose range is the same rank-(m+1) space as
    // for PT/PE/PP. The guard |ea x eb|^2 >= eps_x = 1e-3 |ea|^2 |eb|^2 is
    // exactly `den > 0` of the two-line closest-point solve below, with a
    // condition number bounded by 1e3, so the dim == 4 branch cannot divide by
    // a vanishing determinant.
    // round-5 (s25): `Solver` is forwarded to `make_spd_contact`; the EE
    // branch lives in contact part 1, whose projection this step measures.
    // `Basis` is forwarded too but is inert here: both call sites pass only
    // `Solver`, and s31's basis-free form needs M <= 3 while EE is M = 4.
    template <int Solver = 0, int Basis = 0>
    inline __device__ void EE_barrier_make_spd(Matrix12x12&    H,
                                               const Vector4i& flag,
                                               const Vector3&  Ea0,
                                               const Vector3&  Ea1,
                                               const Vector3&  Eb0,
                                               const Vector3&  Eb1)
    {
        using namespace distance;

        const Vector3 X[4] = {Ea0, Ea1, Eb0, Eb1};

        IndexT dim = detail::active_count(flag);
        if(dim == 2)
        {
            Vector2i act = detail::pp_from_ee(flag);
            Vector2  s   = {1.0, -1.0};
            make_spd_contact<12, 2, Solver, Basis>(H, act, s, X[act[0]] - X[act[1]]);
        }
        else if(dim == 3)
        {
            //tex: $$ \text{closest point } E_0 + t (E_1 - E_0) \text{ on the active edge}$$
            Vector3i act = detail::pe_from_ee(flag);  // [P, E0, E1]
            Vector3  e   = X[act[2]] - X[act[1]];
            Float    t   = (X[act[0]] - X[act[1]]).dot(e) / e.squaredNorm();
            Vector3  s   = {1.0, t - 1.0, -t};
            Vector3  gap = X[act[0]] - (X[act[1]] + t * e);
            make_spd_contact<12, 3, Solver, Basis>(H, act, s, gap);
        }
        else
        {
            //tex: $$ \text{closest points } E_{a0} + a\,e_a \text{ and } E_{b0} + b\,e_b$$
            Vector3  ea  = Ea1 - Ea0;
            Vector3  eb  = Eb1 - Eb0;
            Vector3  w   = Ea0 - Eb0;
            Float    aa  = ea.dot(ea);
            Float    ab  = ea.dot(eb);
            Float    bb  = eb.dot(eb);
            Float    aw  = ea.dot(w);
            Float    bw  = eb.dot(w);
            Float    den = aa * bb - ab * ab;
            Float    a   = (ab * bw - bb * aw) / den;
            Float    b   = (aa * bw - ab * aw) / den;
            Vector4  s   = {1.0 - a, a, b - 1.0, -b};
            Vector4i act = {0, 1, 2, 3};
            // the gap of two non-parallel segments is along ea x eb; taking it
            // there instead of reconstructing `w + a ea - b eb` avoids the
            // cancellation of a gap that is orders smaller than the vertices
            // (the basis only uses the direction, so the sign and scale of the
            // cross product are irrelevant), and `den = |ea x eb|^2` is
            // exactly the quantity the mollifier guard bounds away from 0
            make_spd_contact<12, 4, Solver, Basis>(H, act, s, ea.cross(eb));
        }
    }

    //tex: $$ \text{reduced } make\_spd \text{ of the PP barrier Hessian (6x6)}$$
    // round-5 (s31): see `PE_barrier_make_spd`.
    template <int Solver = 0, int Basis = 0>
    inline __device__ void PP_barrier_make_spd(Matrix6x6&      H,
                                               const Vector2i& flag,
                                               const Vector3&  P0,
                                               const Vector3&  P1)
    {
        Vector2  s   = {1.0, -1.0};
        Vector2i act = {0, 1};
        make_spd_contact<6, 2, Solver, Basis>(H, act, s, P0 - P1);
    }
}  // namespace sym::codim_ipc_simplex_contact
}  // namespace uipc::backend::cuda

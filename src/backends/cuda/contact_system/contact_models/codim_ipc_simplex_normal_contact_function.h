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

    inline __device__ void mollified_EE_barrier_gradient_hessian(Vector12&    G,
                                                                 Matrix12x12& H,
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

        Float B;
        KappaBarrier(B, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial B}{\partial D} $$
        Float dBdD;
        dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);

        //tex: $$ \frac{\partial^2 B}{\partial D^2} $$
        Float ddBddD;
        ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);

        //tex: $$ \nabla B = \frac{\partial B}{\partial D} \nabla D$$
        Vector12 GradB = dBdD * GradD;

        //tex:
        //$$
        // \nabla^2 B = \frac{\partial^2 B}{\partial D^2} \nabla D \nabla D^T + \frac{\partial B}{\partial D} \nabla^2 D
        //$$
        Matrix12x12 HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;

        //tex: $$ \epsilon_x $$
        Float eps_x;
        edge_edge_mollifier_threshold(
            t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, static_cast<Float>(1e-3), eps_x);

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
    template <int N, int M>
    inline __device__ void make_spd_contact(Matrix<Float, N, N>&     H,
                                            const Vector<IndexT, M>& act,
                                            const Vector<Float, M>&  s,
                                            const Vector3&           gap)
    {
        constexpr int NM = 3 * M;

        Matrix<Float, NM, NM> Hs;
        for(int a = 0; a < M; ++a)
            for(int b = 0; b < M; ++b)
                Hs.template block<3, 3>(3 * a, 3 * b) =
                    H.template block<3, 3>(3 * act[a], 3 * act[b]);

        Matrix<Float, NM, M + 1> Q;
        barrier_range_basis<M>(Q, s, gap);

        Matrix<Float, M + 1, M + 1> Hred = Q.transpose() * Hs * Q;
        make_spd(Hred);

        Matrix<Float, NM, NM> Hspd = Q * Hred * Q.transpose();
        for(int a = 0; a < M; ++a)
            for(int b = 0; b < M; ++b)
                H.template block<3, 3>(3 * act[a], 3 * act[b]) =
                    Hspd.template block<3, 3>(3 * a, 3 * b);
    }

    //tex: $$ \text{reduced } make\_spd \text{ of the PT barrier Hessian (12x12)}$$
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
            make_spd_contact<12, 2>(H, act, s, X[act[0]] - X[act[1]]);
        }
        else if(dim == 3)
        {
            //tex: $$ \text{closest point } E_0 + t (E_1 - E_0) \text{ on the active edge}$$
            Vector3i act = detail::pe_from_pt(flag);
            Vector3  e   = X[act[2]] - X[act[1]];
            Float    t   = (X[act[0]] - X[act[1]]).dot(e) / e.squaredNorm();
            Vector3  s   = {1.0, t - 1.0, -t};
            Vector3  gap = X[act[0]] - (X[act[1]] + t * e);
            make_spd_contact<12, 3>(H, act, s, gap);
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
            make_spd_contact<12, 4>(H, act, s, w - u * e1 - v * e2);
        }
    }

    //tex: $$ \text{reduced } make\_spd \text{ of the PE barrier Hessian (9x9)}$$
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
            make_spd_contact<9, 2>(H, act, s, X[act[0]] - X[act[1]]);
        }
        else
        {
            //tex: $$ \text{closest point } E_0 + t (E_1 - E_0) \text{ on the edge}$$
            Vector3  e   = E1 - E0;
            Float    t   = (P - E0).dot(e) / e.squaredNorm();
            Vector3  s   = {1.0, t - 1.0, -t};
            Vector3i act = {0, 1, 2};
            make_spd_contact<9, 3>(H, act, s, P - (E0 + t * e));
        }
    }

    //tex: $$ \text{reduced } make\_spd \text{ of the PP barrier Hessian (6x6)}$$
    inline __device__ void PP_barrier_make_spd(Matrix6x6&      H,
                                               const Vector2i& flag,
                                               const Vector3&  P0,
                                               const Vector3&  P1)
    {
        Vector2  s   = {1.0, -1.0};
        Vector2i act = {0, 1};
        make_spd_contact<6, 2>(H, act, s, P0 - P1);
    }
}  // namespace sym::codim_ipc_simplex_contact
}  // namespace uipc::backend::cuda

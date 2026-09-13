#pragma once
#include <type_define.h>
#include <finite_element/constitutions/discrete_shell_bending_reference.h>
#include <utils/dihedral_angle.h>
//ref: https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf
namespace uipc::backend::cuda
{
namespace sym::discrete_shell_bending
{
#include "sym/discrete_shell_bending.inl"

    inline UIPC_GENERIC void compute_constants(Float&         L0,
                                               Float&         h_bar,
                                               Float&         theta_bar,
                                               Float&         V_bar,
                                               const Vector3& x0_bar,
                                               const Vector3& x1_bar,
                                               const Vector3& x2_bar,
                                               const Vector3& x3_bar,
                                               Float          thickness0,
                                               Float          thickness1,
                                               Float          thickness2,
                                               Float          thickness3)

    {
        detail::compute_discrete_shell_bending_reference(
            L0, h_bar, theta_bar, V_bar, x0_bar, x1_bar, x2_bar, x3_bar);
    }

    inline UIPC_GENERIC Float E(const Vector3& x0,
                                const Vector3& x1,
                                const Vector3& x2,
                                const Vector3& x3,
                                Float          L0,
                                Float          h_bar,
                                Float          theta_bar,
                                Float          kappa)
    {

        namespace DSB = sym::discrete_shell_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float R;
        DSB::E(R, kappa, theta, theta_bar, L0, h_bar);

        return R;
    }

    inline UIPC_GENERIC void dEdx(Vector12&      G,
                                  const Vector3& x0,
                                  const Vector3& x1,
                                  const Vector3& x2,
                                  const Vector3& x3,
                                  Float          L0,
                                  Float          h_bar,
                                  Float          theta_bar,
                                  Float          kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        G = dEdtheta * dthetadx;
    }

    // perf/round6 (s02): the Gauss-Newton hinge Hessian.
    //
    //   E(x)  = L0 * kappa * (theta(x) - theta_bar)^2 / h_bar
    //   dEdx  = E'(theta) * grad(theta)
    //   ddEddx= E''(theta) * grad(theta) grad(theta)^T + E'(theta) * hess(theta)
    //
    // E'' = 2 * L0 * kappa / h_bar is a *constant* (it does not depend on
    // theta at all) and it is non-negative for every reachable input: L0 is a
    // rest edge length (> 0), h_bar = A / (3 L0) a rest height (> 0) and kappa
    // the bending stiffness (>= 0; the physical formula is
    // E_young * h^3 / (12 (1 - nu^2))). So the first term is a PSD rank-1
    // outer product and *all* of the indefiniteness of the hinge Hessian lives
    // in the second, E'(theta) * hess(theta).
    //
    // Dropping that second term therefore yields a Hessian that is PSD by
    // construction -- the Gauss-Newton approximation -- and needs no eigen
    // decomposition at all. It also never evaluates dihedral_angle_hessian().
    // The energy and the gradient are untouched, so the stationary point is
    // unchanged; only the Newton search direction differs.
    //
    // theta is passed to ddEddtheta() rather than elided so that this stays
    // correct if the generated energy ever gains a theta dependence; for the
    // current expression the compiler drops the whole dihedral-angle
    // evaluation as dead (and in the kernel it is a common subexpression with
    // the gradient's anyway).
    // `scale` is folded into the rank-1 coefficient instead of being applied
    // to the assembled matrix afterwards: one multiply per coefficient instead
    // of 144, and the explicit mirrored loop makes H *exactly* symmetric
    // (`(c * g_i) * g_j` is stored on both sides), which the assembler needs
    // because it writes the four diagonal 3x3 blocks in full.
    inline UIPC_GENERIC void ddEddx_gauss_newton(Matrix12x12&   H,
                                                 const Vector3& x0,
                                                 const Vector3& x1,
                                                 const Vector3& x2,
                                                 const Vector3& x3,
                                                 Float          L0,
                                                 Float          h_bar,
                                                 Float          theta_bar,
                                                 Float          kappa,
                                                 Float          scale)
    {
        namespace DSB = sym::discrete_shell_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float ddEddtheta;
        DSB::ddEddtheta(ddEddtheta, kappa, theta, theta_bar, L0, h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        const Float c = ddEddtheta * scale;
#pragma unroll
        for(int i = 0; i < 12; ++i)
        {
            const Float ci = c * dthetadx(i);
            H(i, i)        = ci * dthetadx(i);
#pragma unroll
            for(int j = i + 1; j < 12; ++j)
            {
                const Float v = ci * dthetadx(j);
                H(i, j)       = v;
                H(j, i)       = v;
            }
        }
    }

    inline UIPC_GENERIC void ddEddx(Matrix12x12&   H,
                                    const Vector3& x0,
                                    const Vector3& x1,
                                    const Vector3& x2,
                                    const Vector3& x3,
                                    Float          L0,
                                    Float          h_bar,
                                    Float          theta_bar,
                                    Float          kappa)
    {
        namespace DSB = sym::discrete_shell_bending;
        Float theta;
        dihedral_angle(x0, x1, x2, x3, theta);

        Float dEdtheta;
        DSB::dEdtheta(dEdtheta, kappa, theta, theta_bar, L0, h_bar);

        Float ddEddtheta;
        DSB::ddEddtheta(ddEddtheta, kappa, theta, theta_bar, L0, h_bar);

        Vector12 dthetadx;
        dihedral_angle_gradient(x0, x1, x2, x3, dthetadx);

        Matrix12x12 ddthetaddx;
        dihedral_angle_hessian(x0, x1, x2, x3, ddthetaddx);


        H = dthetadx * ddEddtheta * dthetadx.transpose() + dEdtheta * ddthetaddx;
    }

}  // namespace sym::discrete_shell_bending
}  // namespace uipc::backend::cuda
